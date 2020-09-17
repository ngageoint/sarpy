# -*- coding: utf-8 -*-
"""
Functionality for reading Radarsat Constellation Mission ScanSAR data into a SICD model.
"""

import copy
import re
import os
from datetime import datetime
from xml.etree import ElementTree
from typing import Tuple, List, Union
import logging

import numpy
import numpy.linalg
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from sarpy.compliance import string_types
from sarpy.io.general.base import BaseReader
from sarpy.io.general.tiff import TiffDetails, TiffReader
from sarpy.io.general.utils import get_seconds

from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from sarpy.io.complex.utils import fit_time_coa_polynomial
from sarpy.io.complex.radarsat import parse_xml, RSDetails, RSCdp

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Khanh Ho", "Wade Schwartzkopf")


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a RCM ScanSAR file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    RcmScanSarReader|None
        `RcmScanSarReader` instance if RCM ScanSAR file, `None` otherwise
    """

    try:
        rcm_scan_sar_details = RcmScanSarDetails(file_name)
        print('Path {} is determined to be or contain an RCM ScanSAR product.xml file.'.format(file_name))
        return RcmScanSarReader(rcm_scan_sar_details)
    except (IOError) as exc:
        return None


###########
# parser and interpreter for radarsat constellation mission product.xml
class RcmScanSarDetails(RSDetails):
    __slots__ = ('_file_name', '_root_node', '_satellite', '_product_type', '_num_lines')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        if os.path.isdir(file_name):  # it is the directory - point it at the product.xml file
            for t_file_name in [
                    os.path.join(file_name, 'product.xml'),
                    os.path.join(file_name, 'metadata', 'product.xml')]:
                if os.path.exists(t_file_name):
                    file_name = t_file_name
                    break
        if not os.path.exists(file_name):
            raise IOError('path {} does not exist'.format(file_name))
        self._file_name = file_name

        root_node = parse_xml(file_name, without_ns=True)
        self._satellite = root_node.find('./sourceAttributes/satellite').text.upper()
        self._product_type = root_node.find(
            './imageGenerationParameters'
            '/generalProcessingInformation'
            '/productType').text.upper()
        if not (self._satellite.startswith('RCM') and self._product_type == 'SLC'):
            raise IOError('File {} does not appear to be an SLC product for an '
                          'RCM mission.'.format(self._file_name))
        if not root_node.find('./sourceAttributes/beamModeMnemonic').text.startswith('SC'):
            raise IOError("Only ScanSAR Supported")

        max_line = 0
        for ia_node in root_node.findall('./sceneAttributes/imageAttributes'):
            burst_end_line = int(ia_node.findtext('./lineOffset')) + int(ia_node.findtext('./numLines'))
            max_line = max(max_line, burst_end_line)
        self._num_lines = max_line

        self._root_node = root_node

    @property
    def generation(self):
        """
        str: RCM
        """
        return 'RCM'

    def get_cdps(self):
        base_path = os.path.dirname(self.file_name)

        cdps = []
        for ia in self._findall('./sceneAttributes/imageAttributes'):
            for ipdf in ia.findall('./ipdf'):
                cdp = {'filename': os.path.join(base_path, ipdf.text)}
                if 'pole' in ipdf.attrib:
                    cdp['pole'] = ipdf.attrib['pole']

                if 'burst' in ia.attrib:
                    cdp['burst'] = ia.attrib['burst']

                if 'beam' in ia.attrib:
                    cdp['beam'] = ia.attrib['beam']
                cdps.append(cdp)
        return cdps

    def get_data_file_names(self):
        """
        Gets the list of data file names.

        Returns
        -------
        Tuple[str]
        """

        return [cdp['filename'] for cdp in self.get_cdps()]

    def get_sicd_collection(self):
        """
        Gets the list of sicd objects, one per polarimetric entry.

        Returns
        -------
        Tuple[SICDType]
        """
        sicds = tuple([RcmScanSarCdp(self, cdp['pole'], cdp['burst'], cdp['beam']).get_sicd() for cdp in self.get_cdps()])
        return sicds


class RcmScanSarCdp(RSCdp):
    """Compute SICD metadata for a single ScanSAR Coherent Data Period / burst"""
    def __init__(self, details, pole, burst, beam):
        self.details = details
        self.pole = pole
        self.burst = burst
        self.beam = beam
        self.satellite = details.satellite
        self.generation = details.generation

        root = copy.deepcopy(details._root_node)

        def _strip(parent_node, child_name):
            for node in parent_node.findall(child_name):
                if (node.attrib.get('beam', self.beam) != self.beam or
                    node.attrib.get('pole', self.pole) != self.pole or
                    node.attrib.get('burst', self.burst) != self.burst):
                    parent_node.remove(node)

        parent = root.find('./sourceAttributes/radarParameters')
        _strip(parent, './pulsesReceivedPerDwell')
        _strip(parent, './numberOfPulseIntervalsPerDwell')
        _strip(parent, './rank')
        _strip(parent, './settableGain')
        _strip(parent, './pulseLength')
        _strip(parent, './pulseBandwidth')
        _strip(parent, './samplingWindowStartTimeFirstRawLine')
        _strip(parent, './samplingWindowStartTimeLastRawLine')
        _strip(parent, './numberOfSwstChanges')
        _strip(parent, './adcSamplingRate')
        _strip(parent, './samplesPerEchoLine')

        parent = root.find('./sourceAttributes/rawDataAttributes')
        _strip(parent, './numberOfMissingLines')
        _strip(parent, './rawDataAnalysis')

        parent = root.find('./imageGenerationParameters/sarProcessingInformation')
        _strip(parent, './numberOfLinesProcessed')
        _strip(parent, './perBeamNumberOfRangeLooks')
        _strip(parent, './perBeamRangeLookBandwidths')
        _strip(parent, './perBeamTotalProcessedRangeBandwidths')
        _strip(parent, './azimuthLookBandwidth')
        _strip(parent, './totalProcessedAzimuthBandwidth')
        _strip(parent, './azimuthWindow')

        parent = root.find('./imageGenerationParameters')
        _strip(parent, './chirp')

        parent = root.find('./imageReferenceAttributes')
        _strip(parent, './lookupTableFileName')
        _strip(parent, './noiseLevelFileName')

        parent = root.find('./sceneAttributes')
        _strip(parent, './imageAttributes')

        parent = root.find('./sceneAttributes/imageAttributes')
        _strip(parent, 'ipdf')
        _strip(parent, 'mean')
        _strip(parent, 'sigma')

        parent = root.find('./dopplerCentroid')
        _strip(parent, 'dopplerCentroidEstimate')

        parent = root.find('./dopplerRate')
        _strip(parent, 'dopplerRateEstimate')

        keep_node = None
        keep_burst = -1
        parent = root.find('./sourceAttributes/radarParameters')
        for node in parent.findall('./prfInformation'):
            node_burst = int(node.attrib.get('burst'))
            if (node.attrib.get('beam') != self.beam or
                node_burst > int(self.burst) or
                node_burst < keep_burst):
                parent.remove(node)
            else:
                if(keep_node is not None):
                    parent.remove(keep_node)
                keep_node = node
                keep_burst = node_burst

        self._root_node = root

    def _get_image_and_geo_data(self):
        """
        Gets the ImageData and GeoData metadata.

        Returns
        -------
        Tuple[ImageDataType, GeoDataType]
        """

        pixel_type = 'RE16I_IM16I'
        lines = int(self._find('./sceneAttributes/imageAttributes/numLines').text)
        samples = int(self._find('./sceneAttributes/imageAttributes/samplesPerLine').text)
        tie_points = self._findall('./imageReferenceAttributes'
                                   '/geographicInformation'
                                   '/geolocationGrid'
                                   '/imageTiePoint')
        if self._findall('./imageReferenceAttributes/rasterAttributes/bitsPerSample')[0].text == '32':
            pixel_type = 'RE32F_IM32F'
        line_offset = int(self._find('./sceneAttributes/imageAttributes/lineOffset').text)
        pixel_offset = int(self._find('./sceneAttributes/imageAttributes/pixelOffset').text)
        center_pixel = ([pixel_offset, line_offset] + 0.5*numpy.array([samples-1, lines-1], dtype=numpy.float64))
        # let's use the tie point closest to the center as the SCP
        tp_pixels = numpy.zeros((len(tie_points), 2), dtype=numpy.float64)
        tp_llh = numpy.zeros((len(tie_points), 3), dtype=numpy.float64)
        for j, tp in enumerate(tie_points):
            tp_pixels[j, :] = (float(tp.find('imageCoordinate/pixel').text),
                               float(tp.find('imageCoordinate/line').text))
            tp_llh[j, :] = (float(tp.find('geodeticCoordinate/latitude').text),
                            float(tp.find('geodeticCoordinate/longitude').text),
                            float(tp.find('geodeticCoordinate/height').text))

        scp_index = numpy.argmin(numpy.sum((tp_pixels - center_pixel)**2, axis=1))
        rows = samples
        cols = lines
        scp_row, scp_col = self._copg_ls_to_sicd_rc(tp_pixels[scp_index, 1], tp_pixels[scp_index, 0])

        im_data = ImageDataType(NumRows=rows,
                                NumCols=cols,
                                FirstRow=0,
                                FirstCol=0,
                                PixelType=pixel_type,
                                FullImage=(rows, cols),
                                ValidData=((0, 0), (0, cols-1), (rows-1, cols-1), (rows-1, 0)),
                                SCPPixel=numpy.round([scp_row, scp_col]))
        geo_data = GeoDataType(SCP=SCPType(LLH=tp_llh[scp_index, :]))
        return im_data, geo_data

    def _get_timeline(self):
        """
        Gets the Timeline metadata.

        Returns
        -------
        TimelineType
        """
        start_time = self._get_start_time()
        timeline = TimelineType(start_time)

        state_vectors = self._findall('./sourceAttributes'
                                      '/orbitAndAttitude'
                                      '/orbitInformation'
                                      '/stateVector')
        duration = get_seconds(numpy.datetime64(state_vectors[-1].find('timeStamp').text, 'us'),
                               start_time, precision='us')
        timeline.CollectDuration = duration

        # Generate IPP Poly with accurate linear term
        num_of_pulses = int(self._find('./sourceAttributes'
                                       '/radarParameters'
                                       '/numberOfPulseIntervalsPerDwell').text)
        pulse_rep_freq = float(self._find('./sourceAttributes'
                                          '/radarParameters'
                                          '/prfInformation'
                                          '/pulseRepetitionFrequency').text)

        t_proc_center = get_seconds(numpy.datetime64(self._find('./dopplerCentroid/'
                                                                'dopplerCentroidEstimate/'
                                                                'timeOfDopplerCentroidEstimate').text, 'us'),
                                    start_time, precision='us')
        t_proc_span = (num_of_pulses - 1) / pulse_rep_freq

        t_proc_start = t_proc_center - t_proc_span / 2
        t_proc_end = t_proc_center + t_proc_span / 2

        timeline.IPP = [IPPSetType(
            index=0, TStart=t_proc_start, TEnd=t_proc_end, IPPStart=0,
            IPPEnd=num_of_pulses-1,
            IPPPoly=Poly1DType(Coefs=(-t_proc_start * pulse_rep_freq, pulse_rep_freq))), ]

        return timeline


    def _sicd_rc_to_copg_ls(self, row, col):
        """Convert SICD pixel location to Common Output Pixel Grid pixel"""

        line_offset = int(self._find('./sceneAttributes/imageAttributes/lineOffset').text)
        pixel_offset = int(self._find('./sceneAttributes/imageAttributes/pixelOffset').text)
        pass_dir = self._find('sourceAttributes/orbitAndAttitude/orbitInformation/passDirection').text

        # RCM is always Right Look
        if pass_dir == 'Descending':
            # increasing COPG line ==> increasing SICD column
            # increasing COPG sample ==> decreasing SICD row
            samples = int(self._find('./sceneAttributes/imageAttributes/samplesPerLine').text)
            line_copg = line_offset + col
            samp_copg = pixel_offset + samples - 1 - row
        else:
            # increasing COPG line ==> decreasing SICD column
            # increasing COPG sample ==> increasing SICD row
            lines = int(self._find('./sceneAttributes/imageAttributes/numLines').text)
            line_copg = line_offset + lines - 1 - col
            samp_copg = pixel_offset + row

        return line_copg, samp_copg

    def _copg_ls_to_sicd_rc(self, line_copg, samp_copg):
        """Convert Common Output Pixel Grid pixel location to SICD pixel"""

        line_offset = int(self._find('./sceneAttributes/imageAttributes/lineOffset').text)
        pixel_offset = int(self._find('./sceneAttributes/imageAttributes/pixelOffset').text)
        pass_dir = self._find('sourceAttributes/orbitAndAttitude/orbitInformation/passDirection').text

        # RCM is always Right Look
        if pass_dir == 'Descending':
            # increasing COPG line ==> increasing SICD column
            # increasing COPG sample ==> decreasing SICD row
            samples = int(self._find('./sceneAttributes/imageAttributes/samplesPerLine').text)
            col = line_copg - line_offset
            row = pixel_offset + samples - 1 - samp_copg
        else:
            # increasing COPG line ==> decreasing SICD column
            # increasing COPG sample ==> increasing SICD row
            lines = int(self._find('./sceneAttributes/imageAttributes/numLines').text)
            col = line_offset + lines - 1 - line_copg
            row = samp_copg - pixel_offset

        return row, col

    def _copg_samp_and_sicd_row_align(self):
        """Returns +1 if increasing COPG samples align with increasing SICD row, -1 otherwise"""
        pass_dir = self._find('sourceAttributes/orbitAndAttitude/orbitInformation/passDirection').text
        if pass_dir == 'Descending':
            return -1
        else:
            return 1

    def _copg_line_and_sicd_col_align(self):
        """Returns +1 if increasing COPG lines align with increasing SICD columns, -1 otherwise"""
        pass_dir = self._find('sourceAttributes/orbitAndAttitude/orbitInformation/passDirection').text
        if pass_dir == 'Descending':
            return 1
        else:
            return -1

    def _get_rma_adjust_grid(self, scpcoa, grid, image_data, geo_data, position, collection_info):
        """
        Gets the RMA metadata, and adjust the Grid.Col metadata.

        Parameters
        ----------
        scpcoa : SCPCOAType
        grid : GridType
        image_data : ImageDataType
        geo_data : GeoDataType
        position : PositionType
        collection_info : CollectionInfoType

        Returns
        -------
        RMAType
        """

        look = scpcoa.look
        start_time = self._get_start_time()
        center_freq = self._get_center_frequency()
        doppler_bandwidth = float(self._find('./imageGenerationParameters'
                                             '/sarProcessingInformation'
                                             '/totalProcessedAzimuthBandwidth').text)
        zero_dop_last_line = numpy.datetime64(self._find('./imageGenerationParameters'
                                                         '/sarProcessingInformation'
                                                         '/zeroDopplerTimeLastLine').text, 'us')
        zero_dop_first_line = numpy.datetime64(self._find('./imageGenerationParameters'
                                                          '/sarProcessingInformation'
                                                          '/zeroDopplerTimeFirstLine').text, 'us')

        line_spacing_zd = (get_seconds(zero_dop_last_line, zero_dop_first_line, precision='us')
                          / self.details._num_lines) # Will be negative for Ascending
        # zero doppler time of SCP relative to collect start
        scp_copg_line, _ = self._sicd_rc_to_copg_ls(image_data.SCPPixel.Row, image_data.SCPPixel.Col)
        time_scp_zd = get_seconds(zero_dop_first_line, start_time, precision='us') + \
            scp_copg_line*line_spacing_zd

        col_spacing_zd = numpy.abs(line_spacing_zd)

        r_ca_scp = numpy.linalg.norm(position.ARPPoly(time_scp_zd) - geo_data.SCP.ECF.get_array())
        inca = INCAType(
            R_CA_SCP=r_ca_scp,
            FreqZero=center_freq)
        # doppler rate calculations
        velocity = position.ARPPoly.derivative_eval(time_scp_zd, 1)
        vel_ca_squared = numpy.sum(velocity*velocity)
        # polynomial representing range as a function of range distance from SCP
        r_ca = numpy.array([inca.R_CA_SCP, 1], dtype=numpy.float64)
        # doppler rate coefficients
        doppler_rate_coeffs = numpy.array(
            [float(entry) for entry in self._find('./dopplerRate'
                                                  '/dopplerRateEstimate'
                                                  '/dopplerRateCoefficients').text.split()],
            dtype=numpy.float64)
        doppler_rate_ref_time = float(self._find('./dopplerRate'
                                                 '/dopplerRateEstimate'
                                                 '/dopplerRateReferenceTime').text)

        # the doppler_rate_coeffs represents a polynomial in time, relative to
        #   doppler_rate_ref_time.
        # to construct the doppler centroid polynomial, we need to change scales
        #   to a polynomial in space, relative to SCP.
        doppler_rate_poly = Poly1DType(Coefs=doppler_rate_coeffs)
        alpha = 2.0/speed_of_light
        t_0 = doppler_rate_ref_time - alpha*inca.R_CA_SCP
        dop_rate_scaled_coeffs = doppler_rate_poly.shift(t_0, alpha, return_poly=False)
        # DRateSFPoly is then a scaled multiple of this scaled poly and r_ca above
        coeffs = -numpy.convolve(dop_rate_scaled_coeffs, r_ca)/(alpha*center_freq*vel_ca_squared)
        inca.DRateSFPoly = Poly2DType(Coefs=numpy.reshape(coeffs, (coeffs.size, 1)))

        # modify a few of the other fields
        ss_scale = numpy.sqrt(vel_ca_squared)*inca.DRateSFPoly[0, 0]
        grid.Col.SS = col_spacing_zd*ss_scale
        grid.Col.ImpRespBW = -look*doppler_bandwidth/ss_scale
        inca.TimeCAPoly = Poly1DType(Coefs=[time_scp_zd, 1./ss_scale])

        # doppler centroid
        doppler_cent_coeffs = numpy.array(
            [float(entry) for entry in self._find('./dopplerCentroid'
                                                  '/dopplerCentroidEstimate'
                                                  '/dopplerCentroidCoefficients').text.split()],
            dtype=numpy.float64)
        doppler_cent_ref_time = float(self._find('./dopplerCentroid'
                                                 '/dopplerCentroidEstimate'
                                                 '/dopplerCentroidReferenceTime').text)
        doppler_cent_time_est = numpy.datetime64(self._find('./dopplerCentroid'
                                                            '/dopplerCentroidEstimate'
                                                            '/timeOfDopplerCentroidEstimate').text, 'us')

        doppler_cent_poly = Poly1DType(Coefs=doppler_cent_coeffs)
        alpha = 2.0/speed_of_light
        t_0 = doppler_cent_ref_time - alpha*inca.R_CA_SCP
        scaled_coeffs = doppler_cent_poly.shift(t_0, alpha, return_poly=False)
        inca.DopCentroidPoly = Poly2DType(Coefs=numpy.reshape(scaled_coeffs, (scaled_coeffs.size, 1)))
        # adjust doppler centroid for spotlight, we need to add a second
        # dimension to DopCentroidPoly
        if collection_info.RadarMode.ModeType == 'SPOTLIGHT':
            doppler_cent_est = get_seconds(doppler_cent_time_est, start_time, precision='us')
            doppler_cent_col = (doppler_cent_est - time_scp_zd)/col_spacing_zd
            dop_poly = numpy.zeros((scaled_coeffs.shape[0], 2), dtype=numpy.float64)
            dop_poly[:, 0] = scaled_coeffs
            dop_poly[0, 1] = -look*center_freq*alpha*numpy.sqrt(vel_ca_squared)/inca.R_CA_SCP
            # dopplerCentroid in native metadata was defined at specific column,
            # which might not be our SCP column.  Adjust so that SCP column is correct.
            dop_poly[0, 0] = dop_poly[0, 0] - (dop_poly[0, 1]*doppler_cent_col*grid.Col.SS)
            inca.DopCentroidPoly = Poly2DType(Coefs=dop_poly)

        grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=inca.DopCentroidPoly.get_array()*col_spacing_zd/grid.Col.SS)
        # compute grid.Col.DeltaK1/K2 from DeltaKCOAPoly
        coeffs = grid.Col.DeltaKCOAPoly.get_array()[:, 0]
        # get roots
        roots = polynomial.polyroots(coeffs)
        # construct range bounds (in meters)
        range_bounds = (numpy.array([0, image_data.NumRows-1], dtype=numpy.float64)
                        - image_data.SCPPixel.Row)*grid.Row.SS
        possible_ranges = numpy.copy(range_bounds)
        useful_roots = ((roots > numpy.min(range_bounds)) & (roots < numpy.max(range_bounds)))
        if numpy.any(useful_roots):
            possible_ranges = numpy.concatenate((possible_ranges, roots[useful_roots]), axis=0)
        azimuth_bounds = (numpy.array([0, (image_data.NumCols-1)], dtype=numpy.float64)
                          - image_data.SCPPixel.Col) * grid.Col.SS
        coords_az_2d, coords_rg_2d = numpy.meshgrid(azimuth_bounds, possible_ranges)
        possible_bounds_deltak = grid.Col.DeltaKCOAPoly(coords_rg_2d, coords_az_2d)
        grid.Col.DeltaK1 = numpy.min(possible_bounds_deltak) - 0.5*grid.Col.ImpRespBW
        grid.Col.DeltaK2 = numpy.max(possible_bounds_deltak) + 0.5*grid.Col.ImpRespBW
        # Wrapped spectrum
        if (grid.Col.DeltaK1 < -0.5/grid.Col.SS) or (grid.Col.DeltaK2 > 0.5/grid.Col.SS):
            grid.Col.DeltaK1 = -0.5/abs(grid.Col.SS)
            grid.Col.DeltaK2 = -grid.Col.DeltaK1
        time_coa_poly = fit_time_coa_polynomial(inca, image_data, grid, dop_rate_scaled_coeffs, poly_order=2)
        if collection_info.RadarMode.ModeType == 'SPOTLIGHT':
            # using above was convenience, but not really sensible in spotlight mode
            grid.TimeCOAPoly = Poly2DType(Coefs=[[time_coa_poly.Coefs[0, 0], ], ])
            inca.DopCentroidPoly = None
        elif 'STRIPMAP' in collection_info.RadarMode.ModeType:
            # fit TimeCOAPoly for grid
            grid.TimeCOAPoly = time_coa_poly
            inca.DopCentroidCOA = True
        else:
            raise ValueError('unhandled ModeType {}'.format(collection_info.RadarMode.ModeType))
        return RMAType(RMAlgoType='OMEGA_K', INCA=inca)

    def _get_radiometric(self, image_data, grid):
        """
        Gets the Radiometric metadata.

        Parameters
        ----------
        image_data : ImageDataType
        grid : GridType

        Returns
        -------
        RadiometricType
        """

        def perform_radiometric_fit(component_file):
            comp_struct = parse_xml(component_file, without_ns=True)
            comp_values = numpy.array(
                [float(entry) for entry in comp_struct.find('./gains').text.split()], dtype=numpy.float64)
            comp_values = 1. / (comp_values * comp_values)  # adjust for sicd convention
            if numpy.all(comp_values == comp_values[0]):
                return numpy.array([[comp_values[0], ], ], dtype=numpy.float64)
            else:
                # fit a 1-d polynomial in range
                pfv = int(comp_struct.find('./pixelFirstLutValue').text)
                step = int(comp_struct.find('./stepSize').text)
                num_vs = int(comp_struct.find('./numberOfValues').text)
                _, scp_samp = self._sicd_rc_to_copg_ls(image_data.SCPPixel.Row, image_data.SCPPixel.Col)
                range_coords = grid.Row.SS * (numpy.arange(num_vs)*step + pfv - scp_samp)
                range_coords *= self._copg_samp_and_sicd_row_align()
                return numpy.atleast_2d(polynomial.polyfit(range_coords, comp_values, 3))

        base_path = os.path.dirname(self.details.file_name)
        beta_file = os.path.join(base_path, 'calibration', self._find('./imageReferenceAttributes'
                                                                      '/lookupTableFileName'
                                                                      '[@sarCalibrationType="Beta Nought"]').text)
        sigma_file = os.path.join(base_path, 'calibration', self._find('./imageReferenceAttributes'
                                                                       '/lookupTableFileName'
                                                                       '[@sarCalibrationType="Sigma Nought"]').text)
        gamma_file = os.path.join(base_path, 'calibration', self._find('./imageReferenceAttributes'
                                                                       '/lookupTableFileName'
                                                                       '[@sarCalibrationType="Gamma"]').text)
        if not os.path.isfile(beta_file):
            logging.error(
                msg="Beta calibration information should be located in file {}, "
                    "which doesn't exist.".format(beta_file))
            return None

        # perform beta, sigma, gamma fit
        beta_zero_sf_poly = perform_radiometric_fit(beta_file)
        sigma_zero_sf_poly = perform_radiometric_fit(sigma_file)
        gamma_zero_sf_poly = perform_radiometric_fit(gamma_file)

        # construct noise poly
        noise_level = None
        noise_stem = None
        for node in self._findall('./imageReferenceAttributes/noiseLevelFileName'):
            if node.attrib.get('pole') == self.pole:
                noise_stem = node.text

        noise_file = os.path.join(base_path, 'calibration', noise_stem)
        noise_root = parse_xml(noise_file, without_ns=True)

        noise_levels = noise_root.findall('./perBeamReferenceNoiseLevel')
        beta0s = [entry for entry in noise_levels if (entry.find('sarCalibrationType').text.startswith('Beta')
                                                      and entry.find('beam').text == self.beam)]
        beta0_element = beta0s[0] if len(beta0s) > 0 else None

        if beta0_element is not None:
            noise_level = NoiseLevelType_(NoiseLevelType='ABSOLUTE')
            pfv = float(beta0_element.find('pixelFirstNoiseValue').text)
            step = float(beta0_element.find('stepSize').text)
            beta0s = numpy.array(
                [float(x) for x in beta0_element.find('noiseLevelValues').text.split()])
            _, scp_samp = self._sicd_rc_to_copg_ls(image_data.SCPPixel.Row, image_data.SCPPixel.Col)
            range_coords = grid.Row.SS * (numpy.arange(len(beta0s))*step + pfv - scp_samp)
            range_coords *= self._copg_samp_and_sicd_row_align()
            noise_poly = polynomial.polyfit(
                range_coords,
                beta0s - 10*numpy.log10(polynomial.polyval(range_coords, beta_zero_sf_poly[:, 0])), 2)
            noise_level.NoisePoly = Poly2DType(Coefs=numpy.atleast_2d(noise_poly))

        return RadiometricType(BetaZeroSFPoly=beta_zero_sf_poly,
                               SigmaZeroSFPoly=sigma_zero_sf_poly,
                               GammaZeroSFPoly=gamma_zero_sf_poly,
                               NoiseLevel=noise_level)

    def get_sicd(self):
        nitf, collection_info = self._get_collection_info()
        image_creation = self._get_image_creation()
        image_data, geo_data = self._get_image_and_geo_data()
        position = self._get_position()
        grid = self._get_grid()
        radar_collection = self._get_radar_collection()
        timeline = self._get_timeline()
        image_formation = self._get_image_formation(timeline, radar_collection)
        scpcoa = self._get_scpcoa()
        rma = self._get_rma_adjust_grid(scpcoa, grid, image_data, geo_data, position, collection_info)
        radiometric = self._get_radiometric(image_data, grid)
        base_sicd = SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            GeoData=geo_data,
            ImageData=image_data,
            Position=position,
            Grid=grid,
            RadarCollection=radar_collection,
            Timeline=timeline,
            ImageFormation=image_formation,
            SCPCOA=scpcoa,
            RMA=rma,
            Radiometric=radiometric)
        if len(nitf) > 0:
            base_sicd._NITF = nitf
        self._update_geo_data(base_sicd)
        base_sicd.derive()  # derive all the fields

        chan_index = 0
        tx_rcv_pole = self.pole[0] + ':' + self.pole[1]
        for rcv_chan in radar_collection.RcvChannels:
            if rcv_chan.TxRcvPolarization == tx_rcv_pole:
                chan_index = rcv_chan.index
                break
        base_sicd.ImageFormation.RcvChanProc.ChanIndices = [chan_index, ]
        base_sicd.ImageFormation.TxRcvPolarizationProc = \
            base_sicd.RadarCollection.RcvChannels[chan_index].TxRcvPolarization
        base_sicd.populate_rniirs(override=False)

        return base_sicd


class RcmScanSarReader(BaseReader):
    """
    Gets a reader type object for Radarsat Constellation Mission ScanSAR files.
    RCM ScanSAR files correspond to one tiff per burst and polarimetric band, so
    this will result in one tiff reader per burst/pole.
    """
    __slots__ = ('_rcm_scan_sar_details', '_readers')

    def __init__(self, rcm_scan_sar_details):
        """

        Parameters
        ----------
        rcm_scan_sar_details : str|RcmScanSarDetails
            file name or RadarSatDeatils object
        """

        if isinstance(rcm_scan_sar_details, string_types):
            rcm_scan_sar_details = RcmScanSarDetails(rcm_scan_sar_details)
        if not isinstance(rcm_scan_sar_details, RcmScanSarDetails):
            raise TypeError('The input argument for RcmScanSarReader must be a '
                            'filename or RcmScanSarDetails object')
        self._rcm_scan_sar_details = rcm_scan_sar_details
        # determine symmetry
        symmetry = self._rcm_scan_sar_details.get_symmetry()
        # get the datafiles
        data_files = self._rcm_scan_sar_details.get_data_file_names()
        # get the sicd metadata objects
        sicds = self._rcm_scan_sar_details.get_sicd_collection()
        readers = []
        for sicd, file_name in zip(sicds, data_files):
            # create one reader per file/sicd
            # NB: the files are implicitly listed in the same order as polarizations
            tiff_details = TiffDetails(file_name)
            readers.append(TiffReader(tiff_details, sicd_meta=sicd, symmetry=symmetry))
        self._readers = tuple(readers)  # type: Tuple[TiffReader]
        sicd_tuple = tuple(reader.sicd_meta for reader in readers)
        chipper_tuple = tuple(reader._chipper for reader in readers)
        super(RcmScanSarReader, self).__init__(sicd_tuple, chipper_tuple)
