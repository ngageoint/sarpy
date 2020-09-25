# -*- coding: utf-8 -*-
"""
Functionality for reading TerraSAR-X data into a SICD model.
"""

import os
import logging
from datetime import datetime
from xml.etree import ElementTree
from typing import List, Tuple, Union
from functools import reduce

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light
from scipy.interpolate import griddata

from sarpy.compliance import string_types, int_func
from sarpy.io.general.base import SubsetReader, BaseReader
from sarpy.io.general.tiff import TiffDetails, TiffReader
from sarpy.io.general.utils import get_seconds, parse_timestring

from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType, TxStepType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType, TxFrequencyProcType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from sarpy.geometry import point_projection
from sarpy.geometry.geocoords import geodetic_to_ecf
from sarpy.io.complex.utils import two_dim_poly_fit, get_im_physical_coords, fit_position_xvalidation


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a TerraSAR-X file SSC package.
    Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    TSXReader|None
        `TSXReader` instance if TerraSAR-X file file, `None` otherwise
    """

    try:
        tsx_details = TSXDetails(file_name)
        print('Path {} is determined to be a TerraSAR-X file package.'.format(file_name))
        return TSXReader(tsx_details)
    except (IOError, AttributeError, SyntaxError, ElementTree.ParseError):
        return None

##########
# helper functions and basic interpreter

def _parse_xml(file_name, without_ns=False):
    root_node = ElementTree.parse(file_name).getroot()
    if without_ns:
        return root_node
    else:
        ns = dict([node for _, node in ElementTree.iterparse(file_name, events=('start-ns', ))])
        return ns, root_node


def _is_level1_product(prospective_file):
    with open(prospective_file, 'r') as fi:
        check = fi.read(30)
    return check.startswith('<level1Product')


class TSXDetails(object):
    """
    Parser and interpreter for the TerraSAR-X file package meta-data.
    """

    __slots__ = (
        '_main_file', '_georef_file', '_main_root', '_georef_root')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            The top-level directory, or the basic package xml file.
        """

        self._main_file = None
        self._georef_file = None
        self._main_root = None
        self._georef_root = None
        self._validate_file(file_name)

    def _validate_file(self, file_name):
        """
        Validate the input file location.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        None
        """

        if not isinstance(file_name, string_types):
            raise IOError('file_name must be of string type.')
        if not os.path.exists(file_name):
            raise IOError('file {} does not exist'.format(file_name))

        found_file = None
        if os.path.isdir(file_name):
            for entry in os.listdir(file_name):
                prop_file = os.path.join(entry)
                if os.path.isfile(prop_file) and os.path.splitext(prop_file)[1] == '.xml' \
                        and _is_level1_product(prop_file):
                    found_file = prop_file

            if found_file is None:
                raise IOError(
                    'The provided argument is a directory, but we found no level1Product xml file at the top level.')
        elif os.path.splitext(file_name)[1] == '.xml':
            if _is_level1_product(file_name):
                found_file = file_name
            else:
                raise IOError(
                    'The provided argument is an xml file, which is not a level1Product xml file.')
        else:
            raise IOError(
                'The provided argument is an file, but does not have .xml extension.')

        if file_name is None:
            raise ValueError('Unspecified error where main_file is not defined.')
        self._main_file = found_file
        self._main_root = _parse_xml(self._main_file, without_ns=True)

        georef_file = os.path.join(os.path.split(found_file)[0], 'ANNOTATION', 'GEOREF.xml')
        if not os.path.isfile(georef_file):
            logging.warning(
                'The input file was determined to be or contain a TerraSAR-X level 1 product file, '
                'but the ANNOTATION/GEOREF.xml is not in the expected relative location.')
        else:
            self._georef_file = georef_file
            self._georef_root = _parse_xml(self._georef_file, without_ns=True)

    @property
    def file_name(self):
        """
        str: the main file name
        """

        return self._main_file

    def _find_main(self, tag):
        """
        Pass through to ElementTree.Element.find(tag).

        Parameters
        ----------
        tag : str

        Returns
        -------
        ElementTree.Element
        """

        return self._main_root.find(tag)

    def _findall_main(self, tag):
        """
        Pass through to ElementTree.Element.findall(tag).

        Parameters
        ----------
        tag : str

        Returns
        -------
        List[ElementTree.Element
        """

        return self._main_root.findall(tag)

    def _find_georef(self, tag):
        """
        Pass through to ElementTree.Element.find(tag).

        Parameters
        ----------
        tag : str

        Returns
        -------
        ElementTree.Element
        """

        return None if self._georef_root is None else self._georef_root.find(tag)

    def _findall_georef(self, tag):
        """
        Pass through to ElementTree.Element.findall(tag).

        Parameters
        ----------
        tag : str

        Returns
        -------
        List[ElementTree.Element
        """

        return None if self._georef_root is None else self._georef_root.findall(tag)

    def _get_state_vector_data(self):
        """
        Gets the state vector data.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """

        state_vecs = self._findall_main('./platform/orbit/stateVec')
        tims = numpy.zeros((len(state_vecs),), dtype='datetime64[us]')
        pos = numpy.zeros((len(state_vecs), 3), dtype='float64')
        vel = numpy.zeros((len(state_vecs), 3), dtype='float64')
        for i, entry in enumerate(state_vecs):
            tims[i] = parse_timestring(entry.find('./timeUTC').text, precision='us')
            pos[i, :] = [float(entry.find('./posX').text), float(entry.find('./posY').text),
                         float(entry.find('./posZ').text)]
            vel[i, :] = [float(entry.find('./velX').text), float(entry.find('./velY').text),
                         float(entry.find('./velZ').text)]
        return tims, pos, vel

    @staticmethod
    def _parse_pol_string(str_in):
        # type: (str) -> (str, str)
        return str_in[0], str_in[1]

    def _get_sicd_tx_rcv_pol(self, str_in):
        # type: (str) -> str
        tx_pol, rcv_pol = self._parse_pol_string(str_in)
        return '{}:{}'.format(tx_pol, rcv_pol)

    def _get_full_pol_list(self):
        """
        Gets the full list of polarization states.

        Returns
        -------
        (list, list, list)
        """

        t_original_pols = []
        t_tx_pols = []
        t_tx_rcv_pols = []
        # TODO: this is particular is probably troubled for ScanSAR mode
        for node in self._findall_main('./productComponents/imageData'):
            orig_pol = node.find('./polLayer').text
            tx_part, rcv_part = self._parse_pol_string(orig_pol)
            t_original_pols.append(orig_pol)
            t_tx_pols.append(tx_part)
            t_tx_rcv_pols.append('{}:{}'.format(tx_part, rcv_part))
        return t_original_pols, t_tx_pols, t_tx_rcv_pols

    def _find_middle_grid_node(self):
        """
        Find and returns the middle geolocationGrid point, if it exists.
        Otherwise, returns None.

        Returns
        -------
        None|ElementTree.Element
        """

        if self._georef_root is None:
            return None

        # determine the middle grid location
        az_grid_pts = int_func(self._find_georef('./geolocationGrid/numberOfGridPoints/azimuth').text)
        rg_grid_pts = int_func(self._find_georef('./geolocationGrid/numberOfGridPoints/range').text)
        mid_az = int_func(round(az_grid_pts/2.0)) + 1
        mid_rg = int_func(round(rg_grid_pts/2.0)) + 1
        return self._find_georef('./geolocationGrid/gridPoint[@iaz="{}" @irg="{}"]'.format(mid_az, mid_rg))

    def _calculate_dop_polys(self, polarization, azimuth_time_scp, range_time_scp, collect_start):
        """
        Calculate the doppler centroid polynomials. This is apparently extracted
        from the paper "TerraSAR-X Deskew Description" by Michael Stewart dated
        December 11, 2008.

        Parameters
        ----------
        polarization : str
            The polarization string, required for extracting correct metadata.
        azimuth_time_scp : float
            This is in seconds relative to the collection start.
        range_time_scp : float
            This is in seconds.
        collect_start : numpy.datetime64
            The collection start time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
        """

        # parse the doppler centroid estimates
        doppler_estimate_nodes = self._findall_main(
            './processing/doppler/dopplerCentroid[@polLayer="{}"]/dopplerEstimate'.format(polarization))
        doppler_count = len(doppler_estimate_nodes)
        center_node = doppler_estimate_nodes[int(doppler_count/2)]
        ref_time = parse_timestring(center_node.find('./timeUTC').text, precision='us')
        rg_ref_time = float(center_node.find('./combinedDoppler/referencePoint').text)
        diff_times_raw = numpy.zeros((doppler_count, ), dtype='float64') # offsets from reference time, in seconds
        doppler_range_min = numpy.zeros((doppler_count, ), dtype='float64') # offsets in seconds
        doppler_range_max = numpy.zeros((doppler_count, ), dtype='float64') # offsets in seconds
        doppler_poly_est = numpy.zeros((doppler_count, 2), dtype='float64')
        for i, node in enumerate(doppler_estimate_nodes):
            diff_times_raw[i] = get_seconds(
                parse_timestring(node.find('./timeUTC').text, precision='us'),
                collect_start, precision='us')
            combined_node = node.find('./combinedDoppler')
            doppler_range_min[i] = float(combined_node.find('./validityRangeMin').text)
            doppler_range_max[i] = float(combined_node.find('./validityRangeMax').text)
            if combined_node.find('./polynomialDegree').text.strip() != '1':
                raise ValueError('Got unexpected doppler polynomial degree estimate')
            doppler_poly_est[i, :] = [
                float(combined_node.find('./coefficient[@exponent="0"]').text),
                float(combined_node.find('./coefficient[@exponent="1"]').text)]
        # parse the doppler rate estimates
        doppler_rate_nodes = self._findall_main('./processing/geometry/dopplerRate')
        center_node = doppler_rate_nodes[int(len(doppler_rate_nodes)/2)]
        fm_dop = float(center_node.find('./dopplerRatePolynomial/coefficient[@exponent="0"]').text)
        ss_zd_s = float(self._find_main('./productInfo/imageDataInfo/imageRaster/columnSpacing').text)
        side_of_track = self._find_main('./productInfo/acquisitionInfo/lookDirection').text[0].upper()
        ss_zd_m = float(self._find_main('./productSpecific/complexImageInfo/projectedSpacingAzimuth').text)

        # create a sampled doppler centroid grid
        range_samples = 49  # this is suggested in the paper
        time_coa = numpy.zeros((doppler_count, range_samples), dtype='float64')
        diff_t_range = numpy.zeros((doppler_count, range_samples), dtype='float64')
        dopp_centroid = numpy.zeros((doppler_count, range_samples), dtype='float64')
        for i, entry in enumerate(diff_times_raw):
            time_coa[i, :] = entry
            diff_t_range[i, :] = numpy.linspace(doppler_range_min[i], doppler_range_max[i], num=range_samples) - rg_ref_time
            dopp_centroid[i, :] = polynomial.polyval(diff_t_range[i, :], doppler_poly_est[i, :])
        diff_t_zd = time_coa - dopp_centroid/fm_dop
        coords_rg = 0.5*(diff_t_range + rg_ref_time - range_time_scp)*speed_of_light
        coords_az = ss_zd_m*(diff_t_zd + get_seconds(ref_time, collect_start) - azimuth_time_scp)/ss_zd_s
        if side_of_track == 'L':
            coords_az *= -1
        # perform our fitting
        poly_order = 3
        dop_centroid_poly, residuals, rank, sing_values = two_dim_poly_fit(
            coords_rg, coords_az, dopp_centroid,
            x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-35)
        logging.info(
            'The dop centroid polynomial fit details:\nroot mean square residuals = {}\nrank = {}\n'
            'singular values = {}'.format(residuals, rank, sing_values))

        time_coa_poly, residuals, rank, sing_values = two_dim_poly_fit(
            coords_rg, coords_az, time_coa,
            x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-35)
        logging.info(
            'The dop centroid polynomial fit details:\nroot mean square residuals = {}\nrank = {}\n'
            'singular values = {}'.format(residuals, rank, sing_values))

        return dop_centroid_poly, time_coa_poly

    def _calculate_drate_sf_poly(self):
        # TODO: continue at calculate DRateSFPoly at line 417...
        pass

    def _get_basic_sicd_shell(self, center_freq, dop_bw, ss_zd_s):
        """
        Define the common sicd elements.

        Parameters
        ----------
        center_freq : float
            The center frequency.
        dop_bw : float
            The doppler bandwidth.
        ss_zd_s : float
            The (positive) zero doppler spacing in the time domain.

        Returns
        -------
        SICDType
        """

        def get_collection_info():
            # type: () -> CollectionInfoType
            collector_name = self._find_main('./productInfo/missionInfo/mission').text
            core_name = self._find_main('./productInfo/sceneInfo/sceneID').text

            mode_id = self._find_main('./productInfo/acquisitionInfo/imagingMode').text
            if mode_id == 'ST':
                # TSX "staring" mode, corresponds to SICD spotlight
                mode_type = 'SPOTLIGHT'
            elif mode_id in ['SL', 'HS']:
                # confusing, but TSX mode "spolight" and "high-resolution spotlight",
                # which actually has a moving beam
                mode_type = 'DYNAMIC STRIPMAP'
            elif mode_id == 'SM':
                # TSX stripmap mode
                mode_type = 'STRIPMAP'
            elif mode_id == 'SC':
                # TSX scansar mode
                mode_type = 'STRIPMAP'
            else:
                raise ValueError('Got unexpected mode id {}'.format(mode_id))

            return CollectionInfoType(
                CollectorName=collector_name,
                CoreName=core_name,
                CollectType='MONOSTATIC',
                RadarMode=RadarModeType(ModeID=mode_id, ModeType=mode_type),
                Classification='UNCLASSIFIED')

        def get_image_creation():
            # type: () -> ImageCreationType
            from sarpy.__about__ import __version__

            create_time = self._find_main('./generalHeader/generationTime').text
            site = self._find_main('./productInfo/level1ProcessingFacility').text
            app_node = self._find_main('./generalHeader/generationSystem')
            application = 'Unknown' if app_node is None else \
                '{} {}'.format(app_node.text, app_node.attrib.get('version', 'version_unknown'))
            return ImageCreationType(Application=application,
                                     DateTime=parse_timestring(create_time, precision='us'),
                                     Site=site,
                                     Profile='sarpy {}'.format(__version__))

        def get_initial_grid():
            # type: () -> GridType
            proj_string = self._find_main('./setup/orderInfo/projection').text
            if proj_string == 'GROUNDRANGE':
                image_plane = 'GROUND'
            elif proj_string == 'SLANTRANGE':
                image_plane = 'SLANT'
            else:
                logging.warning('Got image projection {}'.format(proj_string))
                image_plane = 'OTHER'

            the_type = None
            if self._find_main('./productSpecific/complexImageInfo/imageCoordinateType').text == 'ZERODOPPLER':
                the_type = 'RGZERO'

            row_ss = 0.5*float(self._find_main('./productInfo/imageDataInfo/imageRaster/rowSpacing').text)*speed_of_light
            row_bw = 2*float(self._find_main('./processing/processingParameter/rangeLookBandwidth').text)/speed_of_light
            row_win_name = self._find_main('./processing/processingParameter/rangeWindowID').text
            row_wgt_type = WgtTypeType(WindowName=row_win_name)
            if row_win_name == 'HAMMING':
                row_wgt_type.Parameters = {
                    'COEFFICIENT': self._find_main('./processing/processingParameter/rangeWindowCoefficient').text}

            row = DirParamType(
                SS=row_ss,
                Sgn=-1,
                ImpRespBW=row_bw,
                KCtr=2*center_freq/speed_of_light,
                DeltaK1=-0.5*row_bw,
                DeltaK2=0.5*row_bw,
                DeltaKCOAPoly=[[0,],],
                WgtType=row_wgt_type)

            col_ss = float(self._find_main('./productSpecific/complexImageInfo/projectedSpacingAzimuth').text)
            col_win_name = self._find_main('./processing/processingParameter/azimuthWindowID').text
            col_wgt_type = WgtTypeType(WindowName=col_win_name)
            if col_win_name == 'HAMMING':
                col_wgt_type.Parameters = {
                    'COEFFICIENT': self._find_main('./processing/processingParameter/azimuthWindowCoefficient').text}
            col = DirParamType(
                SS=col_ss,
                Sgn=-1,
                ImpRespBW=dop_bw*ss_zd_s/col_ss,
                KCtr=0,
                WgtType=col_wgt_type)

            return GridType(
                ImagePlane=image_plane,
                Type=the_type,
                Row=row,
                Col=col)

        def get_initial_image_formation():
            # type: () -> ImageFormationType
            return ImageFormationType(
                RcvChanProc=RcvChanProcType(NumChanProc=1, PRFScaleFactor=1),  # ChanIndex set later
                ImageFormAlgo='RMA',
                TStartProc=0,
                TEndProc=0,  # corrected later
                ImageBeamComp='SV',
                AzAutofocus='NO',
                RgAutofocus='NO',
                STBeamComp='SV' if collection_info.RadarMode.ModeID in ['SL','HS'] else 'GLOBAL')
                # NB: SL and HS are the proper spotlight modes

        def get_initial_rma():
            # type: () -> RMAType
            return RMAType(RMAlgoType='OMEGA_K',
                           INCA=INCAType(FreqZero=center_freq))

        collection_info = get_collection_info()
        image_creation = get_image_creation()
        init_grid = get_initial_grid()
        init_image_formation = get_initial_image_formation()
        init_rma = get_initial_rma()

        return SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            Grid=init_grid,
            ImageFormation=init_image_formation,
            RMA=init_rma)

    def _populate_basic_image_data(self, sicd, grid_node):
        """
        Populate the basic ImageData and GeoData. This assumes not ScanSAR mode.
        This modifies the provided sicd in place.

        Parameters
        ----------
        sicd : SICDType
        grid_node : None|ElementTree.Element
            The central geolocationGrid point, if it exists.

        Returns
        -------
        None
        """

        # NB: the role of rows and columns is switched in TSX/SICD convention
        rows = int_func(self._find_main('./productInfo/imageDataInfo/imageRaster/numberOfColumns').text)
        cols = int_func(self._find_main('./productInfo/imageDataInfo/imageRaster/numberOfRows').text)

        if grid_node is not None:
            scp_row = int_func(grid_node.find('./col').text)
            scp_col = int_func(grid_node.find('./row').text)
            scp_llh = [
                float(grid_node.find('./lat').text),
                float(grid_node.find('./lon').text),
                float(grid_node.find('./height').text)]
        else:
            scp_row = int_func(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/refColumn').text) - 1
            scp_col = int_func(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/refRow').text) - 1
            scp_llh = [
                float(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/lat').text),
                float(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/lon').text),
                float(self._find_main('./productInfo/sceneInfo/sceneAverageHeight').text)]

        sicd.ImageData = ImageDataType(
            NumRows=rows, NumCols=cols, FirstRow=0, FirstCol=0, FullImage=(rows, cols),
            PixelType='RE16I_IM16I', SCPPixel=(scp_row, scp_col))
        sicd.GeoData = GeoDataType(SCP=SCPType(LLH=scp_llh))

    @staticmethod
    def _populate_initial_radar_collection(sicd, tx_pols, tx_rcv_pols):
        """
        Populate the initial radar collection information. This modifies the
        provided sicd in place.

        Parameters
        ----------
        sicd : SICDType
        tx_pols : List[str]
        tx_rcv_pols : List[str]

        Returns
        -------
        None
        """

        tx_pol_count = len(set(tx_pols))
        if tx_pol_count == 1:
            the_tx_pol = tx_pols[0]
            tx_sequence = None
        else:
            the_tx_pol = 'SEQUENCE'
            tx_sequence = [TxStepType(TxPolarization=tx_pol) for tx_pol in tx_pols]
        sicd.RadarCollection = RadarCollectionType(
            TxPolarization=the_tx_pol,
            TxSequence=tx_sequence,
            RcvChannels=[ChanParametersType(TxRcvPolarization=tx_rcv_pol) for tx_rcv_pol in tx_rcv_pols])

    def _complete_sicd(self, sicd, orig_pol, layer_index, pol_index, ss_zd_s, center_freq,
                       arp_times, arp_pos, arp_vel, middle_grid):
        """
        Complete the remainder of the sicd information and populate as collection,
        if appropriate. **This assumes that this is not ScanSAR mode.**

        Parameters
        ----------
        sicd : SICDType
        orig_pol : str
            The TSX polarization string.
        layer_index : str
            The layer index entry.
        pol_index : int
            The polarization index (1 based) here.
        ss_zd_s : float
            The zero doppler spacing in the time domain.
        center_freq : float
            The center frequency.
        arp_times : numpy.ndarray
            The array of reference times for the state information.
        arp_pos : numpy.ndarray
        arp_vel : numpy.ndarray
        middle_grid : None|ElementTree.Element
            The central geolocationGrid point, if it exists.

        Returns
        -------
        SICDType
        """

        def set_timeline():
            prf = float(self._find_main('./instrument'
                                  '/settings[@polLayer="{}"]'.format(orig_pol)+
                                  '/settingRecord'
                                  '/PRF').text)
            ipp_poly = Poly1DType(Coefs=[0, prf])
            out_sicd.Timeline = TimelineType(
                CollectStart=collect_start,
                CollectDuration=collect_duration,
                IPP=[IPPSetType(TStart=0,
                                TEnd=collect_duration,
                                IPPPoly=ipp_poly,
                                IPPStart=0,
                                IPPEnd=int_func(ipp_poly(collect_duration)))])

        def set_position():
            times_s = numpy.array(
                [get_seconds(entry, collect_start, precision='us') for entry in arp_times], dtype='float64')
            P_x, P_y, P_z = fit_position_xvalidation(times_s, arp_pos, arp_vel, max_degree=8)
            out_sicd.Position = PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

        def complete_radar_collection():
            tx_pulse_length = float(
                self._find_main('./processing'
                                '/processingParameter'
                                '/rangeCompression'
                                '/chirps'
                                '/referenceChirp'
                                '/pulseLength').text)*32/3.29658384e8
            # NB: the matlab version indicates that this conversion comes via personal
            # communication with Juergen Janoth, Head of Application Development, Infoterra
            # The times of this communication is not indicated

            sample_rate = float(
                self._find_main('./instrument'
                                '/settings[@polLayer="{}"]'.format(orig_pol) +
                                '/RSF').text)
            rcv_window_length = float(
                self._find_main('./instrument'
                                '/settings[@polLayer="{}"]'.format(orig_pol) +
                                '/settingRecord'
                                '/echowindowLength').text)/sample_rate

            out_sicd.RadarCollection.TxFrequency = TxFrequencyType(Min=tx_freq_start, Max=tx_freq_end)
            out_sicd.RadarCollection.Waveform = [
                WaveformParametersType(TxPulseLength=tx_pulse_length,
                                       TxRFBandwidth=band_width,
                                       TxFreqStart=tx_freq_start,
                                       TxFMRate=band_width/tx_pulse_length,
                                       ADCSampleRate=sample_rate,
                                       RcvWindowLength=rcv_window_length,
                                       RcvFMRate=0)]

        def complete_image_formation():
            out_sicd.ImageFormation.RcvChanProc.ChanIndices = [pol_index, ]
            out_sicd.ImageFormation.TEndProc = collect_duration
            out_sicd.ImageFormation.TxFrequencyProc = TxFrequencyProcType(MinProc=tx_freq_start,
                                                                          MaxProc=tx_freq_end)
            out_sicd.ImageFormation.TxRcvPolarizationProc = self._parse_pol_string(orig_pol)

        def complete_rma():
            if self._georef_root is not None:
                if middle_grid is None:
                    raise ValueError('middle_grid should have been provided here')

                ref_time = parse_timestring(
                    self._find_georef('./geolocationGrid'
                                      '/gridReferenceTime'
                                      '/tReferenceTimeUTC').text, precision='us')
                az_offset = get_seconds(ref_time, collect_start, precision='us')
                time_ca_scp = float(middle_grid.find('./t').text)
                # get the sum of all provided azimuth shifts?
                # NB: this is obviously assuming that all entries are constant shifts...should we check?
                azimuths_shifts = [
                    float(entry.find('./coefficient').text) for entry in
                    self._findall_georef('./signalPropagationEffects/azimuthShift')]
                azimuth_shift = reduce(sum, azimuths_shifts)
                out_sicd.RMA.INCA.TimeCAPoly = Poly1DType(Coefs=[time_ca_scp + az_offset - azimuth_shift, ])
                azimuth_time_scp = get_seconds(ref_time, collect_start, precision='us') + time_ca_scp

                range_time_scp = float(self._find_georef('./geolocationGrid/gridReferenceTime/tauReferenceTime').text) + \
                                 float(middle_grid.find('./tau').text)
                # get the sum of all provided range delays?
                # NB: this is obviously assuming that all entries are constant shifts...should we check?
                range_delays = [
                    float(entry.find('./coefficient').text) for entry in
                    self._findall_georef('./signalPropagationEffects/rangeDelay')]
                range_delay = reduce(sum, range_delays)
                out_sicd.RMA.INCA.R_CA_SCP = 0.5*(range_time_scp - range_delay)*speed_of_light
            else:
                azimuth_time_scp = get_seconds(
                    parse_timestring(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/azimuthTimeUTC').text, precision='us'),
                    collect_start, precision='us')
                range_time_scp = float(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/rangeTime').text)
                out_sicd.RMA.INCA.TimeCAPoly = Poly1DType(Coefs=[azimuth_time_scp, ])
                out_sicd.RMA.INCA.R_CA_SCP = 0.5*range_time_scp*speed_of_light

            if out_sicd.CollectionInfo.RadarMode.ModeID == 'ST':
                # proper spotlight mode
                out_sicd.Grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=[[0,],]) # this seems fishy to me
                out_sicd.Grid.TimeCOAPoly = Poly2DType(Coefs=[[out_sicd.RMA.INCA.TimeCAPoly.Coefs[0],], ])
                pass
            else:
                dop_centroid_poly, time_coa_poly = self._calculate_dop_polys(orig_pol, azimuth_time_scp, range_time_scp, collect_start)
                out_sicd.RMA.INCA.DopCentroidPoly = Poly2DType(Coefs=dop_centroid_poly)
                out_sicd.RMA.INCA.DopCentroidCOA = True
                out_sicd.Grid.TimeCOAPoly = Poly2DType(Coefs=time_coa_poly)
                out_sicd.Grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=dop_centroid_poly*ss_zd_s/out_sicd.Grid.Col.SS)

            # TODO: continue at calculate DRateSFPoly at line 417...

        out_sicd = sicd.copy()
        # get some common use parameters
        collect_start = parse_timestring(
            self._find_main('./instrument'
                            '/settings[@polLayer="{}"]'.format(orig_pol)+
                            '/settingRecord'
                            '/dataSegment'
                            '/startTimeUTC').text, precision='us')
        collect_end = parse_timestring(
            self._find_main('./instrument'
                            '/settings[@polLayer="{}"]'.format(orig_pol)+
                            '/settingRecord'
                            '/dataSegment'
                            '/stopTimeUTC').text, precision='us')
        collect_duration = get_seconds(collect_end, collect_start, precision='us')
        band_width = float(
            self._find_main('./instrument'
                            '/settings[@polLayer="{}"]'.format(orig_pol) +
                            '/rxBandwidth').text)
        tx_freq_start = center_freq - 0.5 * band_width
        tx_freq_end = center_freq + 0.5 * band_width

        set_timeline()
        set_position()
        complete_radar_collection()
        complete_image_formation()
        complete_rma()

        # TODO:
        #  x 0.) copy the input sicd shell
        #  x 1.) define timeline
        #  x 2.) define position
        #  x 3.) complete radar collection
        #  x 4.) complete image formation
        #  5.) complete RMA
        #  6.) define radiometric
        #  7.) derive fields
        #  8.) define RNIIRS
        #  9.) return it

        pass

    def get_sicd_collection(self):
        """
        Gets the sicd metadata collection.

        Returns
        -------

        """

        the_files = []
        the_sicds = []

        # get some basic common use parameters
        side_of_track = self._find_main('./productInfo/acquisitionInfo/lookDirection').text[0].upper()
        center_freq = float(self._find_main('./instrument/radarParameters/centerFrequency').text)
        dop_bw = float(self._find_main('./processing/processingParameter/azimuthLookBandwidth').text)
        ss_zd_s = float(self._find_main('./productInfo/imageDataInfo/imageRaster/columnSpacing').text)
        use_ss_zd_s = ss_zd_s if side_of_track == 'R' else -ss_zd_s

        # define the basic SICD shell
        basic_sicd = self._get_basic_sicd_shell(center_freq, dop_bw, ss_zd_s)
        if basic_sicd.CollectionInfo.RadarMode.ModeID == 'SC':
            raise ValueError('ScanSAR mode is currently unsupported')

        # fetch the state vector data
        times, positions, velocities = self._get_state_vector_data()
        # fetch the polarization list(s) (maybe ScanSAR modification required here)
        original_pols, tx_pols, tx_rcv_pols = self._get_full_pol_list()
        if basic_sicd.CollectionInfo.RadarMode.ModeID == 'SC':
            raise ValueError('ScanSAR mode is currently unsupported')
        else:
            middle_grid = self._find_middle_grid_node()
            self._populate_basic_image_data(basic_sicd, middle_grid)
            self._populate_initial_radar_collection(basic_sicd, tx_pols, tx_rcv_pols)

            # TODO: finish this here.
            for i, orig_pol in enumerate(original_pols):
                pass

#########
# the reader implementation

class TSXReader(BaseReader):
    pass
