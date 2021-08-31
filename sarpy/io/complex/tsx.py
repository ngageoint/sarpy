"""
Functionality for reading TerraSAR-X data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import os
import logging
from xml.etree import ElementTree
from typing import Union, List
from functools import reduce
import struct

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from sarpy.compliance import string_types, int_func
from sarpy.io.general.base import BaseReader, SubsetChipper, BIPChipper, SarpyIOError
from sarpy.io.general.utils import get_seconds, parse_timestring, is_file_like

from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    WaveformParametersType, ChanParametersType, TxStepType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, \
    RcvChanProcType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from sarpy.io.complex.utils import two_dim_poly_fit, fit_position_xvalidation

logger = logging.getLogger(__name__)

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

    if is_file_like(file_name):
        return None

    try:
        tsx_details = TSXDetails(file_name)
        logger.info('Path {} is determined to be a TerraSAR-X file package.'.format(tsx_details.file_name))
        return TSXReader(tsx_details)
    except (SarpyIOError, AttributeError, SyntaxError, ElementTree.ParseError):
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


############
# metadata helper class

class TSXDetails(object):
    """
    Parser and interpreter for the TerraSAR-X file package meta-data.
    """

    __slots__ = (
        '_parent_directory', '_main_file', '_georef_file', '_main_root', '_georef_root',
        '_im_format')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            The top-level directory, or the basic package xml file.
        """

        self._parent_directory = None
        self._main_file = None
        self._georef_file = None
        self._main_root = None
        self._georef_root = None
        self._im_format = None
        self._validate_file(file_name)
        self._im_format = self._find_main('./productInfo/imageDataInfo/imageDataFormat').text
        if self._im_format not in ['COSAR', 'GEOTIFF']:
            raise ValueError(
                'The file is determined to be of type TerraSAR-X, but we got '
                'unexpected image format value {}'.format(self.image_format))

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
            raise SarpyIOError('file_name must be of string type.')
        if not os.path.exists(file_name):
            raise SarpyIOError('file {} does not exist'.format(file_name))

        found_file = None
        if os.path.isdir(file_name):
            for entry in os.listdir(file_name):
                prop_file = os.path.join(file_name, entry)
                if os.path.isfile(prop_file) and os.path.splitext(prop_file)[1] == '.xml' \
                        and _is_level1_product(prop_file):
                    found_file = prop_file

            if found_file is None:
                raise SarpyIOError(
                    'The provided argument is a directory, but we found no level1Product xml file at the top level.')
        elif os.path.splitext(file_name)[1] == '.xml':
            if _is_level1_product(file_name):
                found_file = file_name
            else:
                raise SarpyIOError(
                    'The provided argument is an xml file, which is not a level1Product xml file.')
        else:
            raise SarpyIOError(
                'The provided argument is an file, but does not have .xml extension.')

        if file_name is None:
            raise ValueError('Unspecified error where main_file is not defined.')

        parent_directory = os.path.split(found_file)[0]
        if not os.path.isdir(os.path.join(parent_directory, 'IMAGEDATA')):
            raise ValueError(
                'The input file was determined to be or contain a TerraSAR-X level 1 product file, '
                'but the IMAGEDATA directory is not in the expected relative location.')
        self._parent_directory = parent_directory
        self._main_file = found_file
        self._main_root = _parse_xml(self._main_file, without_ns=True)

        georef_file = os.path.join(parent_directory, 'ANNOTATION', 'GEOREF.xml')
        if not os.path.isfile(georef_file):
            logger.warning(
                'The input file was determined to be or contain a TerraSAR-X level 1 product file,\n\t'
                'but the ANNOTATION/GEOREF.xml is not in the expected relative location.')
        else:
            self._georef_file = georef_file
            self._georef_root = _parse_xml(self._georef_file, without_ns=True)

    @property
    def file_name(self):
        """
        str: the package directory location
        """

        return self._parent_directory

    @property
    def image_format(self):
        """
        str: The image file format enum value.
        """

        return self._im_format

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
            pos[i, :] = [
                float(entry.find('./posX').text), float(entry.find('./posY').text),
                float(entry.find('./posZ').text)]
            vel[i, :] = [
                float(entry.find('./velX').text), float(entry.find('./velY').text),
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
        test_nodes = self._findall_georef('./geolocationGrid/gridPoint[@iaz="{}"]'.format(mid_az))
        for entry in test_nodes:
            if entry.attrib['irg'] == '{}'.format(mid_rg):
                return entry
        return test_nodes[int(len(test_nodes)/2)]

    def _calculate_dop_polys(self, layer_index, azimuth_time_scp, range_time_scp, collect_start,
                             doppler_rate_reference_node):
        """
        Calculate the doppler centroid polynomials. This is apparently extracted
        from the paper "TerraSAR-X Deskew Description" by Michael Stewart dated
        December 11, 2008.

        Parameters
        ----------
        layer_index : str
            The layer index string, required for extracting correct metadata.
        azimuth_time_scp : float
            This is in seconds relative to the collection start.
        range_time_scp : float
            This is in seconds.
        collect_start : numpy.datetime64
            The collection start time.
        doppler_rate_reference_node : ElementTree.Element

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, ElementTree.Element)
        """

        # parse the doppler centroid estimates nodes
        doppler_estimate_nodes = self._findall_main(
            './processing/doppler/dopplerCentroid[@layerIndex="{}"]/dopplerEstimate'.format(layer_index))
        # find the center node and extract some reference parameters
        doppler_count = len(doppler_estimate_nodes)
        doppler_estimate_center_node = doppler_estimate_nodes[int(doppler_count/2)]
        rg_ref_time = float(doppler_estimate_center_node.find('./combinedDoppler/referencePoint').text)

        # extract the doppler centroid information from all the nodes
        diff_times_raw = numpy.zeros((doppler_count, ), dtype='float64') # offsets from reference time, in seconds
        doppler_range_min = numpy.zeros((doppler_count, ), dtype='float64') # offsets in seconds
        doppler_range_max = numpy.zeros((doppler_count, ), dtype='float64') # offsets in seconds
        doppler_poly_est = []
        for i, node in enumerate(doppler_estimate_nodes):
            diff_times_raw[i] = get_seconds(
                parse_timestring(node.find('./timeUTC').text, precision='us'),
                collect_start, precision='us')
            combined_node = node.find('./combinedDoppler')
            doppler_range_min[i] = float(combined_node.find('./validityRangeMin').text)
            doppler_range_max[i] = float(combined_node.find('./validityRangeMax').text)
            doppler_poly_est.append(
                numpy.array([float(entry.text) for entry in combined_node.findall('./coefficient')], dtype='float64'))

        # parse the doppler rate estimate from our provided reference node
        fm_dop = float(doppler_rate_reference_node.find('./dopplerRatePolynomial/coefficient[@exponent="0"]').text)
        ss_zd_s = float(self._find_main('./productInfo/imageDataInfo/imageRaster/columnSpacing').text)
        side_of_track = self._find_main('./productInfo/acquisitionInfo/lookDirection').text[0].upper()
        ss_zd_m = float(self._find_main('./productSpecific/complexImageInfo/projectedSpacingAzimuth').text)
        use_ss_zd_s = -ss_zd_s if side_of_track == 'L' else ss_zd_s

        # create a sampled doppler centroid grid
        range_samples = 49  # this is suggested in the paper
        time_coa = numpy.zeros((doppler_count, range_samples), dtype='float64')
        diff_t_range = numpy.zeros((doppler_count, range_samples), dtype='float64')
        dopp_centroid = numpy.zeros((doppler_count, range_samples), dtype='float64')
        for i, entry in enumerate(diff_times_raw):
            time_coa[i, :] = entry
            diff_t_range[i, :] = numpy.linspace(doppler_range_min[i], doppler_range_max[i], num=range_samples) - rg_ref_time
            dopp_centroid[i, :] = polynomial.polyval(diff_t_range[i, :], doppler_poly_est[i])
        diff_t_zd = time_coa - dopp_centroid/fm_dop
        coords_rg = 0.5*(diff_t_range + rg_ref_time - range_time_scp)*speed_of_light
        coords_az = ss_zd_m*(diff_t_zd - azimuth_time_scp)/use_ss_zd_s
        # perform our fitting
        poly_order = 3
        dop_centroid_poly, residuals, rank, sing_values = two_dim_poly_fit(
            coords_rg, coords_az, dopp_centroid,
            x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-35)
        logger.info(
            'The dop centroid polynomial fit details:\n\t'
            'root mean square residuals = {}\n\t'
            'rank = {}\n\t'
            'singular values = {}'.format(residuals, rank, sing_values))

        time_coa_poly, residuals, rank, sing_values = two_dim_poly_fit(
            coords_rg, coords_az, time_coa,
            x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-35)
        logger.info(
            'The dop centroid polynomial fit details:\n\t'
            'root mean square residuals = {}\n\t'
            'rank = {}\n\t'
            'singular values = {}'.format(residuals, rank, sing_values))

        return dop_centroid_poly, time_coa_poly

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
            site = self._find_main('./productInfo/generationInfo/level1ProcessingFacility').text
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
                logger.warning('Got image projection {}'.format(proj_string))
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

    def _complete_sicd(self, sicd, orig_pol, layer_index, pol_index, ss_zd_s, side_of_track,
                       center_freq, arp_times, arp_pos, arp_vel, middle_grid, doppler_rate_reference_node):
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
        side_of_track : str
            One of ['R', 'S']
        center_freq : float
            The center frequency.
        arp_times : numpy.ndarray
            The array of reference times for the state information.
        arp_pos : numpy.ndarray
        arp_vel : numpy.ndarray
        middle_grid : None|ElementTree.Element
            The central geolocationGrid point, if it exists.
        doppler_rate_reference_node : ElementTree.Element

        Returns
        -------
        SICDType
        """

        def get_settings_node():
            # type: () -> Union[None, ElementTree.Element]
            for entry in self._findall_main('./instrument/settings'):
                if entry.find('./polLayer').text == orig_pol:
                    return entry
            return None

        def set_timeline():
            prf = float(settings_node.find('./settingRecord/PRF').text)
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

            sample_rate = float(settings_node.find('./RSF').text)
            rcv_window_length = float(settings_node.find('./settingRecord/echowindowLength').text)/sample_rate

            out_sicd.RadarCollection.TxFrequency = (tx_freq_start, tx_freq_end)
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
            out_sicd.ImageFormation.TxFrequencyProc = (tx_freq_start, tx_freq_end)
            out_sicd.ImageFormation.TxRcvPolarizationProc = self._get_sicd_tx_rcv_pol(orig_pol)

        def complete_rma():
            use_ss_zd_s = -ss_zd_s if side_of_track == 'L' else ss_zd_s
            time_ca_linear = use_ss_zd_s/out_sicd.Grid.Col.SS
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
                azimuth_shift = reduce(lambda x, y: x+y, azimuths_shifts)
                out_sicd.RMA.INCA.TimeCAPoly = Poly1DType(Coefs=[time_ca_scp + az_offset - azimuth_shift, time_ca_linear])
                azimuth_time_scp = get_seconds(ref_time, collect_start, precision='us') + time_ca_scp

                range_time_scp = float(self._find_georef('./geolocationGrid/gridReferenceTime/tauReferenceTime').text) + \
                                 float(middle_grid.find('./tau').text)
                # get the sum of all provided range delays?
                # NB: this is obviously assuming that all entries are constant shifts...should we check?
                range_delays = [
                    float(entry.find('./coefficient').text) for entry in
                    self._findall_georef('./signalPropagationEffects/rangeDelay')]
                range_delay = reduce(lambda x, y: x+y, range_delays)
                out_sicd.RMA.INCA.R_CA_SCP = 0.5*(range_time_scp - range_delay)*speed_of_light
            else:
                azimuth_time_scp = get_seconds(
                    parse_timestring(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/azimuthTimeUTC').text, precision='us'),
                    collect_start, precision='us')
                range_time_scp = float(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/rangeTime').text)
                out_sicd.RMA.INCA.TimeCAPoly = Poly1DType(Coefs=[azimuth_time_scp, time_ca_linear])
                out_sicd.RMA.INCA.R_CA_SCP = 0.5*range_time_scp*speed_of_light

            # populate DopCentroidPoly and TimeCOAPoly
            if out_sicd.CollectionInfo.RadarMode.ModeID == 'ST':
                # proper spotlight mode
                out_sicd.Grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=[[0,],]) # NB: this seems fishy to me
                out_sicd.Grid.TimeCOAPoly = Poly2DType(Coefs=[[out_sicd.RMA.INCA.TimeCAPoly.Coefs[0],], ])
            else:
                dop_centroid_poly, time_coa_poly = self._calculate_dop_polys(
                    layer_index, azimuth_time_scp, range_time_scp, collect_start, doppler_rate_reference_node)
                out_sicd.RMA.INCA.DopCentroidPoly = Poly2DType(Coefs=dop_centroid_poly)
                out_sicd.RMA.INCA.DopCentroidCOA = True
                out_sicd.Grid.TimeCOAPoly = Poly2DType(Coefs=time_coa_poly)
                out_sicd.Grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=dop_centroid_poly*use_ss_zd_s/out_sicd.Grid.Col.SS)
            # calculate DRateSFPoly
            vm_vel_sq = numpy.sum(out_sicd.Position.ARPPoly.derivative_eval(azimuth_time_scp)**2)
            r_ca = numpy.array([out_sicd.RMA.INCA.R_CA_SCP, 1], dtype='float64')
            dop_rate_poly_coefs = [float(entry.text) for entry in doppler_rate_reference_node.findall('./dopplerRatePolynomial/coefficient')]
            # Shift 1D polynomial to account for SCP
            dop_rate_ref_time = float(doppler_rate_reference_node.find('./dopplerRatePolynomial/referencePoint').text)
            dop_rate_poly_rg = Poly1DType(Coefs=dop_rate_poly_coefs).shift(dop_rate_ref_time - range_time_scp,
                                                                           alpha=2/speed_of_light,
                                                                           return_poly=False)

            # NB: assumes a sign of -1
            drate_poly = -polynomial.polymul(dop_rate_poly_rg, r_ca)*speed_of_light/(2*center_freq*vm_vel_sq)
            out_sicd.RMA.INCA.DRateSFPoly = Poly2DType(Coefs=numpy.reshape(drate_poly, (-1, 1)))

        def define_radiometric():
            beta_factor = float(self._find_main('./calibration'
                                                '/calibrationConstant[@layerIndex="{}"]'.format(layer_index) +
                                                '/calFactor').text)
            range_time_scp = float(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/rangeTime').text)
            # now, calculate the radiometric noise polynomial
            # find the noise node
            noise_node = self._find_main('./noise[@layerIndex="{}"]'.format(layer_index))
            # extract the middle image noise node
            noise_data_nodes = noise_node.findall('./imageNoise')
            noise_data_node = noise_data_nodes[int(len(noise_data_nodes)/2)]
            range_min = float(noise_data_node.find('./noiseEstimate/validityRangeMin').text)
            range_max = float(noise_data_node.find('./noiseEstimate/validityRangeMax').text)
            ref_point = float(noise_data_node.find('./noiseEstimate/referencePoint').text)
            poly_coeffs = numpy.array(
                [float(coeff.text) for coeff in noise_data_node.findall('./noiseEstimate/coefficient')], dtype='float64')
            # create a sample grid in range time and evaluate the noise
            range_time = numpy.linspace(range_min, range_max, 100) - ref_point
            # this should be an absolute squared magnitude value
            raw_noise_values = polynomial.polyval(range_time, poly_coeffs)
            # we convert to db
            noise_values = 10*numpy.log10(raw_noise_values)
            coords_range_m = 0.5*(range_time + ref_point - range_time_scp)*speed_of_light
            # fit the polynomial
            scale = 1e-3
            deg = poly_coeffs.size-1
            coeffs = polynomial.polyfit(coords_range_m*scale, noise_values, deg=deg, rcond=1e-30, full=False)
            coeffs *= numpy.power(scale, numpy.arange(deg+1))
            coeffs = numpy.reshape(coeffs, (-1, 1))
            out_sicd.Radiometric = RadiometricType(
                BetaZeroSFPoly=Poly2DType(Coefs=[[beta_factor, ], ]),
                NoiseLevel=NoiseLevelType_(
                    NoiseLevelType='ABSOLUTE', NoisePoly=Poly2DType(Coefs=coeffs)))

        def revise_scp():
            scp_ecf = out_sicd.project_image_to_ground(out_sicd.ImageData.SCPPixel.get_array())
            out_sicd.update_scp(scp_ecf, coord_system='ECF')

        out_sicd = sicd.copy()
        # get some common use parameters
        settings_node = get_settings_node()
        if settings_node is None:
            raise ValueError('Cannot find the settings node for polarization {}'.format(orig_pol))
        collect_start = parse_timestring(
            settings_node.find('./settingRecord'
                               '/dataSegment'
                               '/startTimeUTC').text, precision='us')
        collect_end = parse_timestring(
            settings_node.find('./settingRecord'
                               '/dataSegment'
                               '/stopTimeUTC').text, precision='us')
        collect_duration = get_seconds(collect_end, collect_start, precision='us')
        band_width = float(settings_node.find('./rxBandwidth').text)
        tx_freq_start = center_freq - 0.5 * band_width
        tx_freq_end = center_freq + 0.5 * band_width

        # populate the missing sicd elements
        set_timeline()
        set_position()
        complete_radar_collection()
        complete_image_formation()
        complete_rma()
        define_radiometric()
        out_sicd.derive()
        revise_scp()
        out_sicd.populate_rniirs(override=False)
        return out_sicd

    def get_sicd_collection(self):
        """
        Gets the sicd metadata collection.

        Returns
        -------
        (List[str], List[SICDType])
        """

        def get_file_name(layer_index):
            file_node = self._find_main('./productComponents/imageData[@layerIndex="{}"]/file/location'.format(layer_index))
            path_stem = file_node.find('./path').text
            file_name = file_node.find('./filename').text
            full_file = os.path.join(self._parent_directory, path_stem, file_name)
            if not os.path.isfile(full_file):
                raise ValueError('Expected image file at\n\t{}\n\tbut this path does not exist'.format(full_file))
            return full_file

        the_files = []
        the_sicds = []

        # get some basic common use parameters
        center_freq = float(self._find_main('./instrument/radarParameters/centerFrequency').text)
        dop_bw = float(self._find_main('./processing/processingParameter/azimuthLookBandwidth').text)
        ss_zd_s = float(self._find_main('./productInfo/imageDataInfo/imageRaster/columnSpacing').text)
        side_of_track = self._find_main('./productInfo/acquisitionInfo/lookDirection').text[0].upper()

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
            # get the doppler rate reference node
            doppler_rate_nodes = self._findall_main('./processing/geometry/dopplerRate')
            doppler_rate_center_node = doppler_rate_nodes[int(len(doppler_rate_nodes) / 2)]

            for i, orig_pol in enumerate(original_pols):
                the_layer = '{}'.format(i+1)
                pol_index = i+1
                the_sicds.append(self._complete_sicd(
                    basic_sicd, orig_pol, the_layer, pol_index, ss_zd_s, side_of_track,
                    center_freq, times, positions, velocities, middle_grid, doppler_rate_center_node))
                the_files.append(get_file_name(the_layer))
        return the_files, the_sicds


class COSARDetails(object):
    __slots__ = (
        '_file_name', '_file_size', '_header_offsets', '_data_offsets',
        '_burst_index', '_burst_size', '_data_sizes')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        self._header_offsets = []
        self._data_offsets = []
        self._burst_index = []
        self._burst_size = []
        self._data_sizes = []

        if not os.path.isfile(file_name):
            raise SarpyIOError('path {} is not not a file'.format(file_name))
        self._file_name = file_name
        self._file_size = os.path.getsize(file_name)
        self._parse_details()

    @property
    def burst_count(self):
        """
        int: The discovered burst count
        """

        return len(self._data_offsets)

    def _process_burst_header(self, fi, the_offset):
        """

        Parameters
        ----------
        fi : BinaryIO
        the_offset : int

        Returns
        -------

        """

        if the_offset >= self._file_size - 48:
            raise ValueError(
                'The seek location + basic header size is greater than the file size.')
        # seek to our desired location
        fi.seek(the_offset, os.SEEK_SET)
        # read the desired bytes
        header_bytes = fi.read(48)
        # interpret the data
        burst_in_bytes = struct.unpack('>I', header_bytes[:4])[0]
        rsri = struct.unpack('>I', header_bytes[4:8])[0]
        range_samples = struct.unpack('>I', header_bytes[8:12])[0]
        azimuth_samples = struct.unpack('>I', header_bytes[12:16])[0]
        burst_index = struct.unpack('>I', header_bytes[16:20])[0]
        # these two are only useful in the first record
        rtnb = struct.unpack('>I', header_bytes[20:24])[0]
        tnl = struct.unpack('>I', header_bytes[24:28])[0]
        # basic check bytes
        csar = struct.unpack('>4s', header_bytes[28:32])[0]
        version = struct.unpack('>4s', header_bytes[32:36])[0]
        oversample = struct.unpack('>I', header_bytes[36:40])[0]
        scaling_rate = struct.unpack('>d', header_bytes[40:])[0]
        if csar.upper() != b'CSAR':
            raise ValueError('unexpected csar value {}'.format(csar))
        logger.debug(
            'Parsed COSAR burst:'
            '\n\tburst_in_bytes = {}'
            '\n\trsri = {}'
            '\n\trange samples = {}'
            '\n\tazimuth samples = {}'
            '\n\trtnb = {}'
            '\n\ttnl = {}'
            '\n\tcsar = {}'
            '\n\tversion = {}'
            '\n\toversample = {}'
            '\n\tscaling rate = {}'.format(
                burst_in_bytes, rsri, range_samples, azimuth_samples, rtnb, tnl,
                csar, version, oversample, scaling_rate))

        # now, populate our appropriate details
        data_offset = the_offset + (int_func(range_samples)+2)*4*4
        burst_size = 4*(int_func(range_samples)+2)*(int_func(azimuth_samples) + 4)
        self._header_offsets.append(the_offset)
        self._data_offsets.append(data_offset)
        self._burst_index.append(int_func(burst_index))
        self._burst_size.append(burst_size)
        self._data_sizes.append((range_samples, azimuth_samples))
        if the_offset + burst_size > self._file_size:
            raise ValueError(
                'The file size for {} is given as {} bytes, but '
                'the burst at index {} has size {} and offset {}'.format(
                    self._file_name, self._file_size, self._burst_index[-1],
                    self._burst_size[-1], the_offset))

    def _parse_details(self):
        with open(self._file_name, 'rb') as fi:
            # process the first burst header
            self._process_burst_header(fi, 0)
            cont = True
            while cont:
                next_burst_location = self._header_offsets[-1] + self._burst_size[-1]
                if next_burst_location < self._file_size:
                    self._process_burst_header(fi, next_burst_location)
                else:
                    cont = False

    def construct_chipper(self, index, symmetry, expected_size):
        """
        Construct a chipper for the given burst index.

        Parameters
        ----------
        index : int
        symmetry : tuple
        expected_size : tuple

        Returns
        -------
        SubsetChipper
        """

        index = int_func(index)
        if not (0 <= index < self.burst_count):
            raise KeyError('Provided index {} must be in the range [0, {})'.format(index, self.burst_count))
        # get data_size
        offset = self._data_offsets[index]
        range_samples, azimuth_samples = self._data_sizes[index]
        exp_cols, exp_rows = expected_size
        if not (exp_rows == range_samples and exp_cols == azimuth_samples):
            raise ValueError(
                'Expected raw burst size is {}, while actual raw burst size '
                'is {}'.format(expected_size, (range_samples, azimuth_samples)))

        p_chipper = BIPChipper(
            self._file_name, raw_dtype=numpy.dtype('>i2'), data_size=(azimuth_samples, range_samples + 2), raw_bands=2, output_bands=1,
            output_dtype='complex64', symmetry=symmetry, transform_data='COMPLEX', data_offset=offset)
        return SubsetChipper(p_chipper, (2, exp_rows+2), (0, exp_cols))


#########
# the reader implementation

class TSXReader(BaseReader, SICDTypeReader):
    """
    The TerraSAR-X reader implementation
    """

    __slots__ = ('_tsx_details', )

    def __init__(self, tsx_details):
        """

        Parameters
        ----------
        tsx_details : str|TSXDetails
        """

        if isinstance(tsx_details, string_types):
            tsx_details = TSXDetails(tsx_details)
        if not isinstance(tsx_details, TSXDetails):
            raise TypeError(
                'tsx_details is expected to be the path to the TerraSAR-X package '
                'directory or main xml file, of TSXDetails instance. Got type {}'.format(type(tsx_details)))
        self._tsx_details = tsx_details
        chippers = []
        image_format = tsx_details.image_format
        the_files, the_sicds = tsx_details.get_sicd_collection()
        for the_file, the_sicd in zip(the_files, the_sicds):
            rows = the_sicd.ImageData.NumRows
            cols = the_sicd.ImageData.NumCols
            symmetry = (False, (the_sicd.SCPCOA.SideOfTrack == 'L'), True)
            if image_format != 'COSAR':
                raise ValueError(
                    'Expected complex data for TerraSAR-X to be in COSAR format. '
                    'Got unhandled format {}'.format(image_format))
            cosar_details = COSARDetails(the_file)
            if cosar_details.burst_count != 1:
                raise ValueError(
                    'Expected one burst in the COSAR file {}, but got {} bursts'.format(the_file, cosar_details.burst_count))
            chippers.append(cosar_details.construct_chipper(0, symmetry, (cols, rows)))

        SICDTypeReader.__init__(self, tuple(the_sicds))
        BaseReader.__init__(self, tuple(chippers), reader_type="SICD")

    @property
    def file_name(self):
        # type: () -> str
        return self._tsx_details.file_name
