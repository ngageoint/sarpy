"""
Functionality for reading Radarsat (RS2 and RCM) data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Khanh Ho", "Wade Schwartzkopf", "Nathan Bombaci")


import logging
import re
import os
from datetime import datetime
from xml.etree import ElementTree
from typing import Tuple, List, Union

import numpy
from scipy.interpolate import RectBivariateSpline
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from sarpy.compliance import string_types, int_func
from sarpy.io.general.base import BaseReader, SarpyIOError
from sarpy.io.general.tiff import TiffDetails, TiffReader
from sarpy.io.general.utils import get_seconds, parse_timestring, is_file_like
from sarpy.geometry.geocoords import geodetic_to_ecf

from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.other_nitf import ComplexNITFReader
from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, \
    RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    WaveformParametersType, ChanParametersType, TxStepType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from sarpy.io.complex.utils import fit_time_coa_polynomial, fit_position_xvalidation

logger = logging.getLogger(__name__)

########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a RadarSat file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    RadarSatReader|None
        `RadarSatReader` instance if RadarSat file, `None` otherwise
    """

    if is_file_like(file_name):
        return None

    try:
        details = RadarSatDetails(file_name)
        logger.info('Path {} is determined to be or contain a RadarSat or RCM product.xml file.'.format(file_name))
        return RadarSatReader(details)
    except SarpyIOError:
        return None


############
# Helper functions

def _parse_xml(file_name, without_ns=False):
    # type: (str, bool) -> ElementTree.Element
    if without_ns:
        with open(file_name, 'rb') as fi:
            xml_string = fi.read()
        # Remove the (first) default namespace definition (xmlns="http://some/namespace") and parse
        return ElementTree.fromstring(re.sub(b'\\sxmlns="[^"]+"', b'', xml_string, count=1))
    else:
        return ElementTree.parse(file_name).getroot()


def _format_class_str(class_str):
    if 'UNCLASS' in class_str:
        return 'UNCLASSIFIED'
    else:
        return class_str


def _validate_chipper_and_sicd(the_sicd, chipper, name, the_file):
    """
    Check that chipper and sicd are compatible.

    Parameters
    ----------
    the_sicd : SICDType
    chipper : BaseChipper
    name : str
    the_file : str

    Returns
    -------
    None
    """

    rows, cols = the_sicd.ImageData.NumRows, the_sicd.ImageData.NumCols
    data_size = chipper.data_size
    if data_size[0] != rows or data_size[1] != cols:
        raise ValueError(
            'The {} chipper construction for file {}\ngot incompatible sicd size ({}, {}) and '
            'chipper size {}'.format(name, the_file, rows, cols, data_size))


def _construct_tiff_chipper(the_sicd, the_file, symmetry):
    """

    Parameters
    ----------
    the_sicd : SICDType
    the_file : str
    symmetry : tuple

    Returns
    -------
    BaseChipper
    """

    tiff_details = TiffDetails(the_file)
    reader = TiffReader(tiff_details, symmetry=symmetry)
    # noinspection PyProtectedMember
    chipper = reader._chipper
    _validate_chipper_and_sicd(the_sicd, chipper, 'tiff', the_file)
    return chipper


def _construct_single_nitf_chipper(the_sicd, the_file, symmetry):
    """

    Parameters
    ----------
    the_sicd : SICDType
    the_file : str
    symmetry : tuple

    Returns
    -------
    BaseChipper
    """

    reader = ComplexNITFReader(the_file, symmetry=symmetry, split_bands=True)
    # noinspection PyProtectedMember
    chipper = reader._chipper
    if len(chipper) > 1:
        raise ValueError(
            'The SLC data for a single polarmetric band was provided '
            'in a NITF file which has more than a single complex band.')
    _validate_chipper_and_sicd(the_sicd, chipper[0], 'Single NITF', the_file)
    return chipper[0]


def _construct_multiple_nitf_chippers(the_sicds, the_file, symmetry):
    """

    Parameters
    ----------
    the_sicds : List[SICDType]
    the_file : str
    symmetry : tuple

    Returns
    -------
    List[BaseChipper]
    """

    reader = ComplexNITFReader(the_file, symmetry=symmetry, split_bands=True)
    # noinspection PyProtectedMember
    chippers = list(reader._chipper)
    if len(chippers) != len(the_sicds):
        raise ValueError(
            'The SLC data for {} polarmetric bands was provided '
            'in a NITF file which has {} single complex band.'.format(len(the_sicds), len(chippers)))
    for i, (the_sicd, chipper) in enumerate(zip(the_sicds, chippers)):
        _validate_chipper_and_sicd(the_sicd, chipper, 'NITF band {}'.format(i), the_file)
    return chippers


##############
# Class for meta-data interpretation

class RadarSatDetails(object):
    """
    Class for interpreting RadarSat-2 and RadarSat Constellation Mission (RCM)
    metadata files, and creating the corresponding sicd structure(s).
    """

    __slots__ = (
        '_file_name', '_satellite', '_root_node', '_beams', '_bursts',
        '_num_lines_processed', '_polarizations',
        '_x_spline', '_y_spline', '_z_spline',
        '_state_time', '_state_position', '_state_velocity')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        self._beams = None
        self._bursts = None
        self._num_lines_processed = None
        self._polarizations = None
        self._x_spline = None
        self._y_spline = None
        self._z_spline = None
        self._state_time = None
        self._state_position = None
        self._state_velocity = None

        if os.path.isdir(file_name):  # it is the directory - point it at the product.xml file
            for t_file_name in [
                    os.path.join(file_name, 'product.xml'),
                    os.path.join(file_name, 'metadata', 'product.xml')]:
                if os.path.exists(t_file_name):
                    file_name = t_file_name
                    break
        if not os.path.isfile(file_name):
            raise SarpyIOError('path {} does not exist or is not a file'.format(file_name))
        if os.path.split(file_name)[1] != 'product.xml':
            raise SarpyIOError('The radarsat or rcm file is expected to be named product.xml, got path {}'.format(file_name))

        self._file_name = file_name
        root_node = _parse_xml(file_name, without_ns=True)

        sat_node = root_node.find('./sourceAttributes/satellite')
        satellite = 'None' if sat_node is None else sat_node.text.upper()
        product_node = root_node.find(
            './imageGenerationParameters/generalProcessingInformation/productType')
        product_type = 'None' if product_node is None else product_node.text.upper()
        if not ((satellite == 'RADARSAT-2' or satellite.startswith('RCM')) and product_type == 'SLC'):
            raise SarpyIOError('File {} does not appear to be an SLC product for a RADARSAT-2 '
                          'or RCM mission.'.format(file_name))

        self._root_node = root_node
        self._satellite = satellite

        self._build_location_spline()
        self._parse_state_vectors()
        self._extract_beams_and_bursts()

    @property
    def file_name(self):
        """
        str: the file name
        """

        return self._file_name

    @property
    def satellite(self):
        """
        str: the satellite name
        """

        return self._satellite

    @property
    def generation(self):
        """
        str: RS2 or RCM
        """

        if self._satellite == 'RADARSAT-2':
            return 'RS2'
        else:
            return 'RCM'

    @property
    def pass_direction(self):
        """
        str: The pass direction
        """

        return self._find('./sourceAttributes/orbitAndAttitude/orbitInformation/passDirection').text

    def get_symmetry(self):
        """
        Get the symmetry tuple.

        Returns
        -------
        Tuple[bool]
        """

        look_dir = self._find('./sourceAttributes/radarParameters/antennaPointing').text.upper()[0]
        if self.generation == 'RS2':
            line_order = self._find('./imageAttributes/rasterAttributes/lineTimeOrdering').text.upper()
            sample_order = self._find('./imageAttributes/rasterAttributes/pixelTimeOrdering').text.upper()
        else:
            line_order = self._find('./imageReferenceAttributes/rasterAttributes/lineTimeOrdering').text.upper()
            sample_order = self._find('./imageReferenceAttributes/rasterAttributes/pixelTimeOrdering').text.upper()
        reverse_cols = not (
                (line_order == 'DECREASING' and look_dir == 'L') or
                (line_order != 'DECREASING' and look_dir != 'L'))
        return reverse_cols, sample_order == 'DECREASING', True

    def _find(self, tag):
        # type: (str) -> ElementTree.Element
        return self._root_node.find(tag)

    def _findall(self, tag):
        # type: (str) -> List[ElementTree.Element, ...]
        return self._root_node.findall(tag)

    def _get_tiepoint_nodes(self):
        """
        Fetch the tie point nodes.

        Returns
        -------
        List[ElementTree.Element]
        """

        return self._findall('./imageAttributes/geographicInformation/geolocationGrid/imageTiePoint')

    def _build_location_spline(self):
        """
        Populates the three (line, sample) -> location coordinate splines. This
        should be done once for all images.

        Returns
        -------
        None
        """

        if self.generation == 'RS2':
            tie_points = self._findall('./imageAttributes/geographicInformation/geolocationGrid/imageTiePoint')
        elif self.generation == 'RCM':
            tie_points = self._findall('./imageReferenceAttributes/geographicInformation/geolocationGrid/imageTiePoint')
        else:
            raise ValueError('unexpected generation {}'.format(self.generation))

        lines =  []
        samples = []
        llh_coords = numpy.zeros((len(tie_points), 3), dtype='float64')
        grid_row, grid_col = None, None
        for i, entry in enumerate(tie_points):
            img_coords = entry.find('./imageCoordinate')
            geo_coords = entry.find('./geodeticCoordinate')
            # parse lat/lon/hae
            llh_coords[i, :] = [
                float(geo_coords.find('./latitude').text),
                float(geo_coords.find('./longitude').text),
                float(geo_coords.find('./height').text)]

            # parse line/sample
            line = float(img_coords.find('./line').text)
            sample = float(img_coords.find('./pixel').text)
            # verify grid structure
            if i == 0:
                lines.append(line)
                samples.append(sample)
                grid_row = 0
                grid_col = 0
                continue

            if sample == samples[0]:
                # we are starting a new grid column
                grid_row += 1
                grid_col = 0
                lines.append(line)
            else:
                grid_col += 1
                if grid_row == 0:
                    samples.append(sample)

            # verify that the grid assumption is preserved
            if grid_col >= len(samples) or grid_row >= len(lines) or \
                    line != lines[grid_row] or sample != samples[grid_col]:
                logger.error(
                    'Failed parsing grid at\n\t'
                    'grid_col = {}\n\t'
                    'samples = {}\n\t'
                    'grid_row = {}\n\t'
                    'lines = {}\n\t'
                    'line={}, sample={}'.format(grid_col, samples, grid_row, lines, line, sample))
                raise ValueError('The grid assumption is invalid at imageTiePoint entry {}'.format(i))
        lines = numpy.array(lines, dtype='float64')
        samples = numpy.array(samples, dtype='float64')
        ecf_coords = geodetic_to_ecf(llh_coords)
        self._x_spline = RectBivariateSpline(
            lines, samples, numpy.reshape(ecf_coords[:, 0], (lines.size, samples.size)), kx=3, ky=3, s=0)
        self._y_spline = RectBivariateSpline(
            lines, samples, numpy.reshape(ecf_coords[:, 1], (lines.size, samples.size)), kx=3, ky=3, s=0)
        self._z_spline = RectBivariateSpline(
            lines, samples, numpy.reshape(ecf_coords[:, 2], (lines.size, samples.size)), kx=3, ky=3, s=0)

    def _get_image_location(self, line, sample):
        """
        Fetch the image location estimate based on the previously constructed splines.

        Parameters
        ----------
        line : int|float
            The RadarSat line number.
        sample : int|float
            The RadarSat sample number.

        Returns
        -------
        numpy.ndarray
        """

        return numpy.array(
            [float(self._x_spline.ev(line, sample)),
             float(self._y_spline.ev(line, sample)),
             float(self._z_spline.ev(line, sample))], dtype='float64')

    def _parse_state_vectors(self):
        """
        Parses the state vectors.

        Returns
        -------
        None
        """

        state_vectors = self._findall(
            './sourceAttributes/orbitAndAttitude/orbitInformation/stateVector')

        self._state_time = numpy.zeros((len(state_vectors), ), dtype='datetime64[us]')
        self._state_position = numpy.zeros((len(state_vectors), 3), dtype='float64')
        self._state_velocity = numpy.zeros((len(state_vectors), 3), dtype='float64')

        for i, state_vec in enumerate(state_vectors):
            self._state_time[i] =  parse_timestring(state_vec.find('timeStamp').text, precision='us')
            self._state_position[i, :] = [
                float(state_vec.find('xPosition').text),
                float(state_vec.find('yPosition').text),
                float(state_vec.find('zPosition').text)]
            self._state_velocity[i, :] = [
                float(state_vec.find('xVelocity').text),
                float(state_vec.find('yVelocity').text),
                float(state_vec.find('zVelocity').text)]

    def _extract_beams_and_bursts(self):
        """
        Extract the beam and burst and polarization information.

        Returns
        -------
        None
        """

        radar_params = self._find('./sourceAttributes/radarParameters')
        self._beams = radar_params.find('./beams').text.strip().split()
        self._polarizations = radar_params.find('./polarizations').text.strip().split()
        if self.generation == 'RCM':
            image_attributes = self._findall('./sceneAttributes/imageAttributes')
            if 'burst' in image_attributes[0].attrib:
                self._bursts = [(entry.attrib['beam'], entry.attrib['burst']) for entry in image_attributes]
                num_lines_processed = 0
                self._bursts = []
                for entry in image_attributes:
                    self._bursts.append((entry.attrib['beam'], entry.attrib['burst']))
                    nlines = int_func(entry.find('./numLines').text)
                    line_offset = int_func(entry.find('./lineOffset').text)
                    num_lines_processed = max(num_lines_processed, nlines+line_offset)
                self._num_lines_processed = num_lines_processed

    def _get_sicd_radar_mode(self):
        """
        Gets the RadarMode information.

        Returns
        -------
        RadarModeType
        """

        mode_id = self._find('./sourceAttributes/beamModeMnemonic').text
        beam_mode = self._find('./sourceAttributes/beamMode')
        acq_type = self._find('./sourceAttributes/radarParameters/acquisitionType')
        if (beam_mode is not None and beam_mode.text.upper().startswith("SPOTLIGHT")) \
                or (acq_type is not None and acq_type.text.upper().startswith("SPOTLIGHT")) \
                or 'SL' in mode_id:
            mode_type = 'SPOTLIGHT'
        elif mode_id.startswith('SC'):
            # ScanSAR modes
            mode_type = 'SPOTLIGHT'
        else:
            mode_type = 'STRIPMAP'
        return RadarModeType(ModeID=mode_id, ModeType=mode_type)

    def _get_sicd_collection_info(self, start_time):
        """
        Gets the sicd CollectionInfo information.

        Parameters
        ----------
        start_time : numpy.datetime64

        Returns
        -------
        (dict, CollectionInfoType)
        """

        try:
            import sarpy.io.complex.radarsat_addin as radarsat_addin
        except ImportError:
            radarsat_addin = None

        collector_name = self.satellite
        start_time_dt = start_time.astype(datetime)
        date_str = start_time_dt.strftime('%d%b%y').upper()
        nitf = {}
        if self.generation == 'RS2':
            classification = 'UNCLASSIFIED'
            core_name = '{}{}{}'.format(date_str, self.generation, self._find('./sourceAttributes/imageId').text)
        elif self.generation == 'RCM':
            class_str = self._find('./securityAttributes/securityClassification').text.upper()
            classification = _format_class_str(class_str) if radarsat_addin is None else radarsat_addin.extract_radarsat_sec(nitf, class_str)
            core_name = '{}{}{}'.format(date_str, collector_name.replace('-', ''), start_time_dt.strftime('%H%M%S'))
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))

        return nitf, CollectionInfoType(
            Classification=classification,
            CollectorName=collector_name,
            CoreName=core_name,
            RadarMode=self._get_sicd_radar_mode(),
            CollectType='MONOSTATIC')

    def _get_sicd_image_creation(self):
        """
        Gets the ImageCreation metadata.

        Returns
        -------
        ImageCreationType
        """

        from sarpy.__about__ import __version__
        processing_info = self._find('./imageGenerationParameters/generalProcessingInformation')
        return ImageCreationType(
            Application=processing_info.find('softwareVersion').text,
            DateTime=processing_info.find('processingTime').text,
            Site=processing_info.find('processingFacility').text,
            Profile='sarpy {}'.format(__version__))

    def _get_sicd_position(self, start_time):
        """
        Gets the SICD Position definition, based on the given start time.

        Parameters
        ----------
        start_time : numpy.datetime64

        Returns
        -------
        PositionType
        """

        # convert to relative time for polynomial fitting
        T = numpy.array([get_seconds(entry, start_time, precision='us') for entry in self._state_time], dtype='float64')
        P_x, P_y, P_z = fit_position_xvalidation(T, self._state_position, self._state_velocity, max_degree=8)
        return PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

    @staticmethod
    def _parse_polarization(str_in):
        """
        Parses the Radarsat polarization string into it's two SICD components.

        Parameters
        ----------
        str_in : str

        Returns
        -------
        (str, str)
        """

        if len(str_in) != 2:
            raise ValueError('Got input string of unexpected length {}'.format(str_in))

        tx_pol = 'RHC' if str_in[0] == 'C' else str_in[0]
        rcv_pol = 'RHC' if str_in[1] == 'C' else str_in[1]  # probably only H/V
        return tx_pol, rcv_pol

    def _get_sicd_polarizations(self):
        # type: () -> (List[str, ...], List[str, ...])
        tx_pols = []
        tx_rcv_pols = []
        for entry in self._polarizations:
            tx_pol, rcv_pol = self._parse_polarization(entry)
            if tx_pol not in tx_pols:
                tx_pols.append(tx_pol)
            tx_rcv_pols.append('{}:{}'.format(tx_pol, rcv_pol))
        return tx_pols, tx_rcv_pols

    def _get_side_of_track(self):
        """
        Gets the sicd side of track.

        Returns
        -------
        str
        """

        return self._find('./sourceAttributes/radarParameters/antennaPointing').text[0].upper()

    def _get_regular_sicd(self):
        """
        Gets the SICD collection. This will return one SICD per polarimetric
        collection. It will also return the data file(s). This is only applicable
        for non-ScanSAR collects.

        Returns
        -------
        (List[SICDType], List[str])
        """

        def get_image_and_geo_data():
            if self.generation == 'RS2':
                pixel_type = 'RE16I_IM16I'
                cols = int_func(self._find('./imageAttributes/rasterAttributes/numberOfLines').text)
                rows = int_func(self._find('./imageAttributes/rasterAttributes/numberOfSamplesPerLine').text)
            elif self.generation == 'RCM':
                cols = int_func(self._find('./sceneAttributes/imageAttributes/numLines').text)
                rows = int_func(self._find('./sceneAttributes/imageAttributes/samplesPerLine').text)
                bits_per_sample = self._find('./imageReferenceAttributes/rasterAttributes/bitsPerSample').text
                if bits_per_sample == '32':
                    pixel_type = 'RE32F_IM32F'
                elif bits_per_sample == '16':
                    pixel_type = 'RE16I_IM16I'
                else:
                    raise ValueError('Got unhandled bites per sample {}'.format(bits_per_sample))
            else:
                raise ValueError('unhandled generation {}'.format(self.generation))
            scp_rows = int_func(0.5*rows)
            scp_cols = int_func(0.5*cols)
            scp_ecf = self._get_image_location(scp_cols, scp_rows)
            im_data = ImageDataType(
                NumRows=rows, NumCols=cols, FirstRow=0, FirstCol=0, PixelType=pixel_type,
                FullImage=(rows, cols), SCPPixel=(scp_rows, scp_cols))
            t_geo_data = GeoDataType(SCP=SCPType(ECF=scp_ecf))
            return im_data, t_geo_data

        def get_grid_row():
            # type: () -> DirParamType
            if self.generation == 'RS2':
                row_ss = float(self._find('./imageAttributes/rasterAttributes/sampledPixelSpacing').text)
                row_irbw = 2*float(self._find('./imageGenerationParameters'
                                              '/sarProcessingInformation'
                                              '/totalProcessedRangeBandwidth').text)/speed_of_light
            elif self.generation == 'RCM':
                row_ss = float(self._find('./imageReferenceAttributes/rasterAttributes/sampledPixelSpacing').text)
                row_irbw = 2*float(self._find('./sourceAttributes'
                                              '/radarParameters'
                                              '/pulseBandwidth').text)/speed_of_light
            else:
                raise ValueError('unhandled generation {}'.format(self.generation))
            row_wgt_type = WgtTypeType(
                WindowName=self._find('./imageGenerationParameters'
                                      '/sarProcessingInformation'
                                      '/rangeWindow/windowName').text.upper())
            if row_wgt_type.WindowName == 'KAISER':
                row_wgt_type.Parameters = {
                    'BETA': self._find('./imageGenerationParameters'
                                       '/sarProcessingInformation'
                                       '/rangeWindow/windowCoefficient').text}
            return DirParamType(
                SS=row_ss, ImpRespBW=row_irbw, Sgn=-1, KCtr=2*center_frequency/speed_of_light,
                DeltaKCOAPoly=Poly2DType(Coefs=((0,),)), WgtType=row_wgt_type)

        def get_grid_col():
            # type: () -> DirParamType
            az_win = self._find('./imageGenerationParameters/sarProcessingInformation/azimuthWindow')
            col_wgt_type = WgtTypeType(WindowName=az_win.find('./windowName').text.upper())
            if col_wgt_type.WindowName == 'KAISER':
                col_wgt_type.Parameters = {'BETA': az_win.find('./windowCoefficient').text}
            return DirParamType(Sgn=-1, KCtr=0, WgtType=col_wgt_type)

        def get_radar_collection():
            # type: () -> RadarCollectionType
            radar_params = self._find('./sourceAttributes/radarParameters')
            # Ultrafine and spotlight modes have t pulses, otherwise just one.
            bandwidth_elements = sorted(radar_params.findall('pulseBandwidth'), key=lambda x: x.get('pulse'))
            pulse_length_elements = sorted(radar_params.findall('pulseLength'), key=lambda x: x.get('pulse'))
            adc_elements = sorted(radar_params.findall('adcSamplingRate'), key=lambda x: x.get('pulse'))
            samples_per_echo = float(radar_params.find('samplesPerEchoLine').text)
            wf_params = []
            bandwidths = numpy.empty((len(bandwidth_elements),), dtype=numpy.float64)
            for j, (bwe, ple, adce) in enumerate(zip(bandwidth_elements, pulse_length_elements, adc_elements)):
                bandwidths[j] = float(bwe.text)
                samp_rate = float(adce.text)
                wf_params.append(WaveformParametersType(index=j,
                                                        TxRFBandwidth=float(bwe.text),
                                                        TxPulseLength=float(ple.text),
                                                        ADCSampleRate=samp_rate,
                                                        RcvWindowLength=samples_per_echo / samp_rate,
                                                        RcvDemodType='CHIRP',
                                                        RcvFMRate=0))
            tot_bw = numpy.sum(bandwidths)
            tx_freq = (center_frequency-0.5*tot_bw, center_frequency+0.5*tot_bw)
            t_radar_collection = RadarCollectionType(TxFrequency=tx_freq, Waveform=wf_params)
            t_radar_collection.Waveform[0].TxFreqStart = tx_freq[0]
            for j in range(1, len(bandwidth_elements)):
                t_radar_collection.Waveform[j].TxFreqStart = t_radar_collection.Waveform[j - 1].TxFreqStart + \
                                                             t_radar_collection.Waveform[j - 1].TxRFBandwidth
            t_radar_collection.RcvChannels = [
                ChanParametersType(TxRcvPolarization=entry, index=j+1) for j, entry in enumerate(tx_rcv_pols)]
            if len(tx_pols) == 1:
                t_radar_collection.TxPolarization = tx_pols[0]
            else:
                t_radar_collection.TxPolarization = 'SEQUENCE'
                t_radar_collection.TxSequence = [
                    TxStepType(TxPolarization=entry, index=j+1) for j, entry in enumerate(tx_pols)]
            return t_radar_collection

        def get_timeline():
            # type: () -> TimelineType

            pulse_parts = len(self._findall('./sourceAttributes/radarParameters/pulseBandwidth'))
            if self.generation == 'RS2':
                pulse_rep_freq = float(self._find('./sourceAttributes/radarParameters/pulseRepetitionFrequency').text)
            elif self.generation == 'RCM':
                pulse_rep_freq = float(self._find('./sourceAttributes/radarParameters/prfInformation/pulseRepetitionFrequency').text)
            else:
                raise ValueError('unhandled generation {}'.format(self.generation))

            pulse_rep_freq *= pulse_parts
            if pulse_parts == 2 and collection_info.RadarMode.ModeType == 'STRIPMAP':
                # it's not completely clear why we need an additional factor of 2 for strip map
                pulse_rep_freq *= 2
            lines_processed = [
                float(entry.text) for entry in
                self._findall('./imageGenerationParameters/sarProcessingInformation/numberOfLinesProcessed')]
            duration = None
            ipp = None
            # there should be one entry of num_lines_processed for each transmit/receive polarization
            # and they should all be the same. Omit if this is not the case.
            if (len(lines_processed) == len(tx_rcv_pols)) and all(x == lines_processed[0] for x in lines_processed):
                num_lines_processed = lines_processed[0] * len(tx_pols)
                duration = num_lines_processed / pulse_rep_freq
                ipp = IPPSetType(
                    index=0, TStart=0, TEnd=duration, IPPStart=0, IPPEnd=int(num_lines_processed),
                    IPPPoly=Poly1DType(Coefs=(0, pulse_rep_freq)))
            return TimelineType(
                CollectStart=collect_start, CollectDuration=duration, IPP=[ipp, ])

        def get_image_formation():
            #type: () -> ImageFormationType
            pulse_parts = len(self._findall('./sourceAttributes/radarParameters/pulseBandwidth'))
            return ImageFormationType(
                # PRFScaleFactor for either polarimetric or multi-step, but not both.
                RcvChanProc=RcvChanProcType(
                    NumChanProc=1, PRFScaleFactor=1./max(pulse_parts, len(tx_pols))),
                ImageFormAlgo='RMA',
                TStartProc=timeline.IPP[0].TStart,
                TEndProc=timeline.IPP[0].TEnd,
                TxFrequencyProc=(
                    radar_collection.TxFrequency.Min, radar_collection.TxFrequency.Max),
                STBeamComp='GLOBAL',
                ImageBeamComp='SV',
                AzAutofocus='NO',
                RgAutofocus='NO')

        def get_rma_adjust_grid():
            # type: () -> RMAType

            # fetch all the things needed below
            # generation agnostic
            doppler_bandwidth = float(
                self._find('./imageGenerationParameters'
                           '/sarProcessingInformation'
                           '/totalProcessedAzimuthBandwidth').text)
            zero_dop_last_line = \
                parse_timestring(
                    self._find('./imageGenerationParameters'
                               '/sarProcessingInformation'
                               '/zeroDopplerTimeLastLine').text,
                    precision='us')
            zero_dop_first_line = parse_timestring(
                self._find('./imageGenerationParameters'
                           '/sarProcessingInformation'
                           '/zeroDopplerTimeFirstLine').text,
                precision='us')
            if self.generation == 'RS2':
                near_range = float(
                    self._find('./imageGenerationParameters'
                               '/sarProcessingInformation'
                               '/slantRangeNearEdge').text)

                doppler_rate_node = self._find('./imageGenerationParameters'
                                               '/dopplerRateValues')
                doppler_rate_coeffs = numpy.array(
                    [float(entry) for entry in doppler_rate_node.find('./dopplerRateValuesCoefficients').text.split()],
                    dtype='float64')

                doppler_centroid_node = self._find('./imageGenerationParameters'
                                                   '/dopplerCentroid')
            elif self.generation == 'RCM':
                near_range = float(
                    self._find('./sceneAttributes/imageAttributes/slantRangeNearEdge').text)

                doppler_rate_node = self._find('./dopplerRate'
                                               '/dopplerRateEstimate')
                doppler_rate_coeffs = numpy.array(
                    [float(entry) for entry in doppler_rate_node.find('./dopplerRateCoefficients').text.split()],
                    dtype='float64')

                doppler_centroid_node = self._find('./dopplerCentroid'
                                                   '/dopplerCentroidEstimate')
            else:
                raise ValueError('unhandled generation {}'.format(self.generation))

            doppler_rate_ref_time = float(doppler_rate_node.find('./dopplerRateReferenceTime').text)
            doppler_cent_coeffs = numpy.array(
                [float(entry) for entry in
                 doppler_centroid_node.find('./dopplerCentroidCoefficients').text.split()],
                dtype='float64')
            doppler_cent_ref_time = float(
                doppler_centroid_node.find('./dopplerCentroidReferenceTime').text)
            doppler_cent_time_est = parse_timestring(
                doppler_centroid_node.find('./timeOfDopplerCentroidEstimate').text, precision='us')

            look = scpcoa.look
            if look > 1:
                # SideOfTrack == 'L'
                # we explicitly want negative time order
                if zero_dop_first_line < zero_dop_last_line:
                    zero_dop_first_line, zero_dop_last_line = zero_dop_last_line, zero_dop_first_line
            else:
                # we explicitly want positive time order
                if zero_dop_first_line > zero_dop_last_line:
                    zero_dop_first_line, zero_dop_last_line = zero_dop_last_line, zero_dop_first_line
            col_spacing_zd = get_seconds(zero_dop_last_line, zero_dop_first_line, precision='us') / (
                        image_data.NumCols - 1)
            # zero doppler time of SCP relative to collect start
            time_scp_zd = get_seconds(zero_dop_first_line, collect_start, precision='us') + \
                          image_data.SCPPixel.Col * col_spacing_zd

            inca = INCAType(
                R_CA_SCP=near_range + (image_data.SCPPixel.Row * grid.Row.SS),
                FreqZero=center_frequency)
            # doppler rate calculations
            velocity = position.ARPPoly.derivative_eval(time_scp_zd, 1)
            vel_ca_squared = numpy.sum(velocity * velocity)
            # polynomial representing range as a function of range distance from SCP
            r_ca = numpy.array([inca.R_CA_SCP, 1], dtype=numpy.float64)

            # the doppler_rate_coeffs represents a polynomial in time, relative to
            #   doppler_rate_ref_time.
            # to construct the doppler centroid polynomial, we need to change scales
            #   to a polynomial in space, relative to SCP.
            doppler_rate_poly = Poly1DType(Coefs=doppler_rate_coeffs)
            alpha = 2.0 / speed_of_light
            t_0 = doppler_rate_ref_time - alpha * inca.R_CA_SCP
            dop_rate_scaled_coeffs = doppler_rate_poly.shift(t_0, alpha, return_poly=False)
            # DRateSFPoly is then a scaled multiple of this scaled poly and r_ca above
            coeffs = -numpy.convolve(dop_rate_scaled_coeffs, r_ca) / (alpha * center_frequency * vel_ca_squared)
            inca.DRateSFPoly = Poly2DType(Coefs=numpy.reshape(coeffs, (coeffs.size, 1)))

            # modify a few of the other fields
            ss_scale = numpy.sqrt(vel_ca_squared) * inca.DRateSFPoly[0, 0]
            grid.Col.SS = col_spacing_zd * ss_scale
            grid.Col.ImpRespBW = -look * doppler_bandwidth / ss_scale
            inca.TimeCAPoly = Poly1DType(Coefs=[time_scp_zd, 1. / ss_scale])

            # doppler centroid
            doppler_cent_poly = Poly1DType(Coefs=doppler_cent_coeffs)
            alpha = 2.0 / speed_of_light
            t_0 = doppler_cent_ref_time - alpha * inca.R_CA_SCP
            scaled_coeffs = doppler_cent_poly.shift(t_0, alpha, return_poly=False)

            # adjust doppler centroid for spotlight, we need to add a second
            # dimension to DopCentroidPoly
            if collection_info.RadarMode.ModeType == 'SPOTLIGHT':
                doppler_cent_est = get_seconds(doppler_cent_time_est, collect_start, precision='us')
                dop_poly = numpy.zeros((scaled_coeffs.shape[0], 2), dtype=numpy.float64)
                dop_poly[0, 1] = -look * center_frequency * alpha * numpy.sqrt(vel_ca_squared) / inca.R_CA_SCP
                dop_poly[1, 1] = look * center_frequency * alpha * numpy.sqrt(vel_ca_squared) / (inca.R_CA_SCP ** 2)
                one_way_time = inca.R_CA_SCP / speed_of_light
                pos = position.ARPPoly(doppler_cent_est + one_way_time)
                vel = position.ARPPoly.derivative_eval(doppler_cent_est + one_way_time)
                los = geo_data.SCP.ECF.get_array() - pos
                vel_hat = vel / numpy.linalg.norm(vel)
                dop_poly[:, 0] = dop_poly[:, 0] - look * (dop_poly[:, 1] * numpy.dot(los, vel_hat))
                inca.DopCentroidPoly = Poly2DType(Coefs=dop_poly)
            else:
                inca.DopCentroidPoly = Poly2DType(Coefs=numpy.reshape(scaled_coeffs, (scaled_coeffs.size, 1)))

            grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=inca.DopCentroidPoly.get_array() * col_spacing_zd / grid.Col.SS)
            # compute grid.Col.DeltaK1/K2 from DeltaKCOAPoly
            coeffs = grid.Col.DeltaKCOAPoly.get_array()[:, 0]
            # get roots
            roots = polynomial.polyroots(coeffs)
            # construct range bounds (in meters)
            range_bounds = (numpy.array([0, image_data.NumRows - 1], dtype=numpy.float64)
                            - image_data.SCPPixel.Row) * grid.Row.SS
            possible_ranges = numpy.copy(range_bounds)
            useful_roots = ((roots > numpy.min(range_bounds)) & (roots < numpy.max(range_bounds)))
            if numpy.any(useful_roots):
                possible_ranges = numpy.concatenate((possible_ranges, roots[useful_roots]), axis=0)
            azimuth_bounds = (numpy.array([0, (image_data.NumCols - 1)], dtype=numpy.float64)
                              - image_data.SCPPixel.Col) * grid.Col.SS
            coords_az_2d, coords_rg_2d = numpy.meshgrid(azimuth_bounds, possible_ranges)
            possible_bounds_deltak = grid.Col.DeltaKCOAPoly(coords_rg_2d, coords_az_2d)
            grid.Col.DeltaK1 = numpy.min(possible_bounds_deltak) - 0.5 * grid.Col.ImpRespBW
            grid.Col.DeltaK2 = numpy.max(possible_bounds_deltak) + 0.5 * grid.Col.ImpRespBW
            # Wrapped spectrum
            if (grid.Col.DeltaK1 < -0.5 / grid.Col.SS) or (grid.Col.DeltaK2 > 0.5 / grid.Col.SS):
                grid.Col.DeltaK1 = -0.5 / abs(grid.Col.SS)
                grid.Col.DeltaK2 = -grid.Col.DeltaK1
            time_coa_poly = fit_time_coa_polynomial(inca, image_data, grid, dop_rate_scaled_coeffs, poly_order=2)
            if collection_info.RadarMode.ModeType == 'SPOTLIGHT':
                # using above was convenience, but not really sensible in spotlight mode
                grid.TimeCOAPoly = Poly2DType(Coefs=[[time_coa_poly.Coefs[0, 0], ], ])
                inca.DopCentroidPoly = None
            elif collection_info.RadarMode.ModeType == 'STRIPMAP':
                # fit TimeCOAPoly for grid
                grid.TimeCOAPoly = time_coa_poly
                inca.DopCentroidCOA = True
            else:
                raise ValueError('unhandled ModeType {}'.format(collection_info.RadarMode.ModeType))
            return RMAType(RMAlgoType='OMEGA_K', INCA=inca)

        def get_radiometric():
            # type: () -> Union[None, RadiometricType]

            def perform_radiometric_fit(component_file):
                comp_struct = _parse_xml(component_file, without_ns=(self.generation != 'RS2'))
                comp_values = numpy.array(
                    [float(entry) for entry in comp_struct.find('./gains').text.split()], dtype='float64')
                comp_values = 1. / (comp_values * comp_values)  # adjust for sicd convention
                if numpy.all(comp_values == comp_values[0]):
                    return numpy.array([[comp_values[0], ], ], dtype=numpy.float64)
                else:
                    # fit a 1-d polynomial in range
                    if self.generation == 'RS2':
                        coords_rg = (numpy.arange(image_data.NumRows) - image_data.SCPPixel.Row + image_data.FirstRow) * grid.Row.SS
                    elif self.generation == 'RCM':  # the rows are sub-sampled
                        start = int(comp_struct.find('./pixelFirstLutValue').text)
                        num_vs = int(comp_struct.find('./numberOfValues').text)
                        t_step = int(comp_struct.find('./stepSize').text)
                        rng_indices = start + numpy.arange(num_vs)*t_step
                        coords_rg = (rng_indices - image_data.SCPPixel.Row + image_data.FirstRow) * grid.Row.SS
                    else:
                        raise ValueError('Unhandled generation {}'.format(self.generation))

                    return numpy.reshape(polynomial.polyfit(coords_rg, comp_values, 3), (-1, 1))

            base_path = os.path.dirname(self.file_name)
            if self.generation == 'RS2':
                beta_file = os.path.join(
                    base_path,
                    self._find('./imageAttributes/lookupTable[@incidenceAngleCorrection="Beta Nought"]').text)
                sigma_file = os.path.join(
                    base_path,
                    self._find('./imageAttributes/lookupTable[@incidenceAngleCorrection="Sigma Nought"]').text)
                gamma_file = os.path.join(
                    base_path,
                    self._find('./imageAttributes/lookupTable[@incidenceAngleCorrection="Gamma"]').text)
            elif self.generation == 'RCM':
                beta_file = os.path.join(
                    base_path, 'calibration',
                    self._find('./imageReferenceAttributes/lookupTableFileName[@sarCalibrationType="Beta Nought"]').text)
                sigma_file = os.path.join(
                    base_path, 'calibration',
                    self._find('./imageReferenceAttributes/lookupTableFileName[@sarCalibrationType="Sigma Nought"]').text)
                gamma_file = os.path.join(
                    base_path, 'calibration',
                    self._find('./imageReferenceAttributes/lookupTableFileName[@sarCalibrationType="Gamma"]').text)
            else:
                raise ValueError('unhandled generation {}'.format(self.generation))

            if not os.path.isfile(beta_file):
                logger.error(
                    msg="Beta calibration information should be located in file {}, "
                        "which doesn't exist.".format(beta_file))
                return None

            # perform beta, sigma, gamma fit
            beta_zero_sf_poly = perform_radiometric_fit(beta_file)
            sigma_zero_sf_poly = perform_radiometric_fit(sigma_file)
            gamma_zero_sf_poly = perform_radiometric_fit(gamma_file)

            # construct noise poly
            noise_level = None
            if self.generation == 'RS2':
                # noise is in the main product.xml
                beta0_element = self._find('./sourceAttributes/radarParameters'
                                           '/referenceNoiseLevel[@incidenceAngleCorrection="Beta Nought"]')
            elif self.generation == 'RCM':
                noise_file = os.path.join(
                    base_path, 'calibration',
                    self._find('./imageReferenceAttributes/noiseLevelFileName').text)
                noise_root = _parse_xml(noise_file, without_ns=True)
                noise_levels = noise_root.findall('./referenceNoiseLevel')
                beta0s = [entry for entry in noise_levels if entry.find('sarCalibrationType').text.startswith('Beta')]
                beta0_element = beta0s[0] if len(beta0s) > 0 else None
            else:
                raise ValueError('unhandled generation {}'.format(self.generation))

            if beta0_element is not None:
                pfv = float(beta0_element.find('pixelFirstNoiseValue').text)
                step = float(beta0_element.find('stepSize').text)
                beta0s = numpy.array(
                    [float(x) for x in beta0_element.find('noiseLevelValues').text.split()])
                range_coords = grid.Row.SS * (numpy.arange(len(beta0s)) * step + pfv - image_data.SCPPixel.Row)
                noise_poly = polynomial.polyfit(
                    range_coords,
                    beta0s - 10 * numpy.log10(polynomial.polyval(range_coords, beta_zero_sf_poly[:, 0])), 2)
                noise_level = NoiseLevelType_(
                    NoiseLevelType='ABSOLUTE',
                    NoisePoly=Poly2DType(Coefs=numpy.reshape(noise_poly, (-1, 1))))

            return RadiometricType(BetaZeroSFPoly=beta_zero_sf_poly,
                                   SigmaZeroSFPoly=sigma_zero_sf_poly,
                                   GammaZeroSFPoly=gamma_zero_sf_poly,
                                   NoiseLevel=noise_level)

        def correct_scp():
            scp_pixel = base_sicd.ImageData.SCPPixel.get_array()
            scp_ecf = base_sicd.project_image_to_ground(scp_pixel, projection_type='HAE')
            base_sicd.update_scp(scp_ecf, coord_system='ECF')

        def get_data_file_names():
            base_path = os.path.dirname(self.file_name)
            image_files = []
            if self.generation == 'RS2':
                for pol in self._polarizations:
                    fname = self._find('./imageAttributes/fullResolutionImageData[@pole="{}"]'.format(pol))
                    if fname is None:
                        raise ValueError('Got unexpected image file structure.')
                    image_files.append(os.path.join(base_path, fname.text))
            else:
                img_attribute_node = self._find('./sceneAttributes/imageAttributes')
                results = img_attribute_node.findall('./ipdf')

                if len(results) == 1:
                    # there's either a single polarization, or it's one of the NITF files
                    image_files.append(os.path.join(base_path, results[0].text))
                else:
                    for pol in self._polarizations:
                        fname = None
                        for entry in results:
                            if entry.attrib.get('pole', None) == pol:
                                fname = entry.text
                        if fname is None:
                            raise ValueError('Got unexpected image file structure.')
                        image_files.append(os.path.join(base_path, fname))
            return image_files

        center_frequency = float(self._find('./sourceAttributes/radarParameters/radarCenterFrequency').text)
        collect_start = parse_timestring(self._find('./sourceAttributes/rawDataStartTime').text)
        tx_pols, tx_rcv_pols = self._get_sicd_polarizations()

        nitf, collection_info = self._get_sicd_collection_info(collect_start)
        image_creation = self._get_sicd_image_creation()
        position = self._get_sicd_position(collect_start)
        image_data, geo_data = get_image_and_geo_data()
        grid = GridType(ImagePlane='SLANT', Type='RGZERO', Row=get_grid_row(), Col=get_grid_col())
        radar_collection = get_radar_collection()
        timeline = get_timeline()
        image_formation = get_image_formation()
        scpcoa = SCPCOAType(SideOfTrack=self._get_side_of_track())
        rma = get_rma_adjust_grid()
        radiometric = get_radiometric()
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
            Radiometric=radiometric,
            _NITF=nitf)
        correct_scp()
        base_sicd.derive()  # derive all the fields
        base_sicd.populate_rniirs(override=False)
        # now, make one copy per polarimetric entry, as appropriate
        the_files = get_data_file_names()

        the_sicds = []
        for i, (original_pol, sicd_pol) in enumerate(zip(self._polarizations, tx_rcv_pols)):
            this_sicd = base_sicd.copy()
            this_sicd.ImageFormation.RcvChanProc.ChanIndices = [i+1, ]
            this_sicd.ImageFormation.TxRcvPolarizationProc = sicd_pol
            the_sicds.append(this_sicd)
        return the_sicds, the_files

    def _get_scansar_sicd(self, beam, burst):
        """
        Gets the SICD collection for the given burst. This is only applicable
        to ScanSAR collects. This will return one SICD per polarimetric collection.
        It will also return the data file(s) for the given beam/burst.

        Parameters
        ----------
        beam : str
        burst : str

        Returns
        -------
        (List[SICDType], List[str])
        """

        def get_image_and_geo_data():
            img_attributes = self._find('./sceneAttributes/imageAttributes[@burst="{}"]'.format(burst))
            sample_offset = int_func(img_attributes.find('./pixelOffset').text)
            line_offset = int_func(img_attributes.find('./lineOffset').text)
            cols = int_func(img_attributes.find('./numLines').text)
            rows = int_func(img_attributes.find('./samplesPerLine').text)
            bits_per_sample = self._find('./imageReferenceAttributes/rasterAttributes/bitsPerSample').text
            if bits_per_sample == '32':
                pixel_type = 'RE32F_IM32F'
            elif bits_per_sample == '16':
                pixel_type = 'RE16I_IM16I'
            else:
                raise ValueError('Got unhandled bites per sample {}'.format(bits_per_sample))
            scp_rows = int_func(0.5*rows)
            scp_cols = int_func(0.5*cols)
            scp_ecf = self._get_image_location(scp_cols+line_offset, scp_rows+sample_offset)
            im_data = ImageDataType(
                NumRows=rows, NumCols=cols, FirstRow=0, FirstCol=0, PixelType=pixel_type,
                FullImage=(rows, cols), SCPPixel=(scp_rows, scp_cols))
            t_geo_data = GeoDataType(SCP=SCPType(ECF=scp_ecf))
            return im_data, t_geo_data, sample_offset

        def get_grid_row():
            # type: () -> DirParamType
            row_ss = float(self._find('./imageReferenceAttributes/rasterAttributes/sampledPixelSpacing').text)
            row_irbw = 2*float(self._find('./sourceAttributes'
                                          '/radarParameters'
                                          '/pulseBandwidth[@beam="{}"]'.format(beam)).text)/speed_of_light
            row_wgt_type = WgtTypeType(
                WindowName=self._find('./imageGenerationParameters'
                                      '/sarProcessingInformation'
                                      '/rangeWindow'
                                      '/windowName').text.upper())
            if row_wgt_type.WindowName == 'KAISER':
                row_wgt_type.Parameters = {
                    'BETA': self._find('./imageGenerationParameters'
                                       '/sarProcessingInformation'
                                       '/rangeWindow'
                                       '/windowCoefficient').text}
            return DirParamType(
                SS=row_ss, ImpRespBW=row_irbw, Sgn=-1, KCtr=2*center_frequency/speed_of_light,
                DeltaKCOAPoly=Poly2DType(Coefs=((0,),)), WgtType=row_wgt_type)

        def get_grid_col():
            # type: () -> DirParamType
            az_win = self._find('./imageGenerationParameters/sarProcessingInformation/azimuthWindow[@beam="{}"]'.format(beam))
            col_wgt_type = WgtTypeType(WindowName=az_win.find('./windowName').text.upper())
            if col_wgt_type.WindowName == 'KAISER':
                col_wgt_type.Parameters = {'BETA': az_win.find('./windowCoefficient').text}

            return DirParamType(Sgn=-1, KCtr=0, WgtType=col_wgt_type)

        def get_radar_collection():
            # type: () -> RadarCollectionType
            radar_params = self._find('./sourceAttributes/radarParameters')
            # Ultrafine and spotlight modes have t pulses, otherwise just one.
            bandwidth_elements = sorted(
                radar_params.findall('pulseBandwidth[@beam="{}"]'.format(beam)), key=lambda x: x.get('pulse'))
            pulse_length_elements = sorted(
                radar_params.findall('pulseLength[@beam="{}"]'.format(beam)), key=lambda x: x.get('pulse'))
            adc_elements = sorted(
                radar_params.findall('adcSamplingRate[@beam="{}"]'.format(beam)), key=lambda x: x.get('pulse'))
            samples_per_echo = float(
                radar_params.find('samplesPerEchoLine[@beam="{}"]'.format(beam)).text)
            wf_params = []
            bandwidths = numpy.empty((len(bandwidth_elements),), dtype=numpy.float64)
            for j, (bwe, ple, adce) in enumerate(zip(bandwidth_elements, pulse_length_elements, adc_elements)):
                bandwidths[j] = float(bwe.text)
                samp_rate = float(adce.text)
                wf_params.append(WaveformParametersType(index=j,
                                                        TxRFBandwidth=float(bwe.text),
                                                        TxPulseLength=float(ple.text),
                                                        ADCSampleRate=samp_rate,
                                                        RcvWindowLength=samples_per_echo / samp_rate,
                                                        RcvDemodType='CHIRP',
                                                        RcvFMRate=0))
            tot_bw = numpy.sum(bandwidths)
            tx_freq = (center_frequency-0.5*tot_bw, center_frequency+0.5*tot_bw)
            t_radar_collection = RadarCollectionType(TxFrequency=tx_freq, Waveform=wf_params)
            t_radar_collection.Waveform[0].TxFreqStart = tx_freq[0]
            for j in range(1, len(bandwidth_elements)):
                t_radar_collection.Waveform[j].TxFreqStart = t_radar_collection.Waveform[j - 1].TxFreqStart + \
                                                             t_radar_collection.Waveform[j - 1].TxRFBandwidth
            t_radar_collection.RcvChannels = [
                ChanParametersType(TxRcvPolarization=entry, index=j+1) for j, entry in enumerate(tx_rcv_pols)]
            if len(tx_pols) == 1:
                t_radar_collection.TxPolarization = tx_pols[0]
            else:
                t_radar_collection.TxPolarization = 'SEQUENCE'
                t_radar_collection.TxSequence = [
                    TxStepType(TxPolarization=entry, index=j+1) for j, entry in enumerate(tx_pols)]
            return t_radar_collection

        def get_timeline():
            # type: () -> TimelineType
            ipp = IPPSetType(
                index=0, TStart=0, TEnd=processing_time_span,
                IPPStart=0, IPPEnd=num_of_pulses - 1,
                IPPPoly=Poly1DType(Coefs=(0, pulse_rep_freq)))

            return TimelineType(
                CollectStart=collect_start,
                CollectDuration=processing_time_span,
                IPP=[ipp, ])

        def get_image_formation():
            # type: () -> ImageFormationType
            pulse_parts = len(self._findall('./sourceAttributes/radarParameters/pulseBandwidth[@beam="{}"]'.format(beam)))
            return ImageFormationType(
                # PRFScaleFactor for either polarimetric or multi-step, but not both.
                RcvChanProc=RcvChanProcType(
                    NumChanProc=1, PRFScaleFactor=1./max(pulse_parts, len(tx_pols))),
                ImageFormAlgo='RMA',
                TStartProc=timeline.IPP[0].TStart,
                TEndProc=timeline.IPP[0].TEnd,
                TxFrequencyProc=(
                    radar_collection.TxFrequency.Min, radar_collection.TxFrequency.Max),
                STBeamComp='GLOBAL',
                ImageBeamComp='SV',
                AzAutofocus='NO',
                RgAutofocus='NO')

        def get_rma_adjust_grid():
            # type: () -> RMAType

            look = scpcoa.look
            sar_processing_info = self._find('./imageGenerationParameters/sarProcessingInformation')

            doppler_bandwidth = float(sar_processing_info.find('./totalProcessedAzimuthBandwidth').text)
            zero_dop_last_line = parse_timestring(
                sar_processing_info.find('./zeroDopplerTimeLastLine').text, precision='us')
            zero_dop_first_line = parse_timestring(
                sar_processing_info.find('./zeroDopplerTimeFirstLine').text, precision='us')

            # doppler rate coefficients
            doppler_rate_coeffs = numpy.array(
                [float(entry) for entry in dop_rate_estimate_node.find('./dopplerRateCoefficients').text.split()],
                dtype='float64')
            doppler_rate_ref_time = float(dop_rate_estimate_node.find('./dopplerRateReferenceTime').text)

            line_spacing_zd = \
                get_seconds(zero_dop_last_line, zero_dop_first_line, precision='us')/self._num_lines_processed
            # NB: this will be negative for Ascending
            col_spacing_zd = numpy.abs(line_spacing_zd)
            # zero doppler time of SCP relative to collect start
            time_scp_zd = 0.5*processing_time_span

            r_ca_scp = numpy.linalg.norm(position.ARPPoly(time_scp_zd) - geo_data.SCP.ECF.get_array())
            inca = INCAType(R_CA_SCP=r_ca_scp, FreqZero=center_frequency)
            # doppler rate calculations
            velocity = position.ARPPoly.derivative_eval(time_scp_zd, der_order=1)
            vel_ca_squared = numpy.sum(velocity * velocity)
            # polynomial representing range as a function of range distance from SCP
            r_ca = numpy.array([inca.R_CA_SCP, 1], dtype='float64')

            # the doppler_rate_coeffs represents a polynomial in time, relative to
            #   doppler_rate_ref_time.
            # to construct the doppler centroid polynomial, we need to change scales
            #   to a polynomial in space, relative to SCP.
            doppler_rate_poly = Poly1DType(Coefs=doppler_rate_coeffs)
            alpha = 2.0 / speed_of_light
            t_0 = doppler_rate_ref_time - alpha*inca.R_CA_SCP
            dop_rate_scaled_coeffs = doppler_rate_poly.shift(t_0, alpha, return_poly=False)
            # DRateSFPoly is then a scaled multiple of this scaled poly and r_ca above
            coeffs = -numpy.convolve(dop_rate_scaled_coeffs, r_ca) / (alpha * center_frequency * vel_ca_squared)
            inca.DRateSFPoly = Poly2DType(Coefs=numpy.reshape(coeffs, (coeffs.size, 1)))

            # modify a few of the other fields
            ss_scale = numpy.sqrt(vel_ca_squared) * inca.DRateSFPoly[0, 0]
            grid.Col.SS = col_spacing_zd * ss_scale
            grid.Col.ImpRespBW = -look * doppler_bandwidth / ss_scale
            inca.TimeCAPoly = Poly1DType(Coefs=[time_scp_zd, 1. / ss_scale])

            # doppler centroid
            doppler_cent_coeffs = numpy.array(
                [float(entry) for entry in dop_centroid_estimate_node.find('./dopplerCentroidCoefficients').text.split()],
                dtype='float64')
            doppler_cent_ref_time = float(
                dop_centroid_estimate_node.find('./dopplerCentroidReferenceTime').text)

            doppler_cent_poly = Poly1DType(Coefs=doppler_cent_coeffs)
            alpha = 2.0 / speed_of_light
            t_0 = doppler_cent_ref_time - alpha * inca.R_CA_SCP
            scaled_coeffs = doppler_cent_poly.shift(t_0, alpha, return_poly=False)

            # adjust doppler centroid for spotlight, we need to add a second
            # dimension to DopCentroidPoly
            if collection_info.RadarMode.ModeType == 'SPOTLIGHT':
                doppler_cent_est = get_seconds(dop_centroid_est_time, collect_start, precision='us')
                dop_poly = numpy.zeros((scaled_coeffs.shape[0], 2), dtype=numpy.float64)
                dop_poly[0, 1] = -look * center_frequency * alpha * numpy.sqrt(vel_ca_squared) / inca.R_CA_SCP
                dop_poly[1, 1] = look * center_frequency * alpha * numpy.sqrt(vel_ca_squared) / (inca.R_CA_SCP ** 2)
                one_way_time = inca.R_CA_SCP / speed_of_light
                use_time = doppler_cent_est + one_way_time
                pos = position.ARPPoly(use_time)
                vel = position.ARPPoly.derivative_eval(use_time, der_order=1)
                los = geo_data.SCP.ECF.get_array() - pos
                vel_hat = vel / numpy.linalg.norm(vel)
                dop_poly[:, 0] = dop_poly[:, 0] - look * (dop_poly[:, 1] * numpy.dot(los, vel_hat))
                inca.DopCentroidPoly = Poly2DType(Coefs=dop_poly)  # NB: this is set for use in fit-time_coa_poly
            else:
                raise ValueError('ScanSAR mode data should be SPOTLIGHT mode')

            grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=inca.DopCentroidPoly.get_array()*col_spacing_zd/grid.Col.SS)
            # compute grid.Col.DeltaK1/K2 from DeltaKCOAPoly
            coeffs = grid.Col.DeltaKCOAPoly.get_array()[:, 0]
            # get roots
            roots = polynomial.polyroots(coeffs)
            # construct range bounds (in meters)
            range_bounds = (numpy.array([0, image_data.NumRows - 1], dtype='float64')
                            - image_data.SCPPixel.Row) * grid.Row.SS
            possible_ranges = numpy.copy(range_bounds)
            useful_roots = ((roots > numpy.min(range_bounds)) & (roots < numpy.max(range_bounds)))
            if numpy.any(useful_roots):
                possible_ranges = numpy.concatenate((possible_ranges, roots[useful_roots]), axis=0)
            azimuth_bounds = (numpy.array([0, (image_data.NumCols - 1)], dtype='float64')
                              - image_data.SCPPixel.Col) * grid.Col.SS
            coords_az_2d, coords_rg_2d = numpy.meshgrid(azimuth_bounds, possible_ranges)
            possible_bounds_deltak = grid.Col.DeltaKCOAPoly(coords_rg_2d, coords_az_2d)
            grid.Col.DeltaK1 = numpy.min(possible_bounds_deltak) - 0.5 * grid.Col.ImpRespBW
            grid.Col.DeltaK2 = numpy.max(possible_bounds_deltak) + 0.5 * grid.Col.ImpRespBW
            # Wrapped spectrum
            if (grid.Col.DeltaK1 < -0.5 / grid.Col.SS) or (grid.Col.DeltaK2 > 0.5 / grid.Col.SS):
                grid.Col.DeltaK1 = -0.5 / abs(grid.Col.SS)
                grid.Col.DeltaK2 = -grid.Col.DeltaK1
            time_coa_poly = fit_time_coa_polynomial(inca, image_data, grid, dop_rate_scaled_coeffs, poly_order=2)
            if collection_info.RadarMode.ModeType == 'SPOTLIGHT':
                # using above was convenience, but not really sensible in spotlight mode
                grid.TimeCOAPoly = Poly2DType(Coefs=[[time_coa_poly.Coefs[0, 0], ], ])
                inca.DopCentroidPoly = None
            else:
                raise ValueError('ScanSAR mode data should be SPOTLIGHT mode')
            return RMAType(RMAlgoType='OMEGA_K', INCA=inca)

        def get_radiometric():
            # type: () -> Union[RadiometricType, None]

            def perform_radiometric_fit(component_file):
                comp_struct = _parse_xml(component_file, without_ns=True)
                comp_values = numpy.array(
                    [float(entry) for entry in comp_struct.find('./gains').text.split()], dtype=numpy.float64)
                comp_values = 1. / (comp_values * comp_values)  # adjust for sicd convention
                if numpy.all(comp_values == comp_values[0]):
                    return numpy.array([[comp_values[0], ], ], dtype=numpy.float64)
                else:
                    # fit a 1-d polynomial in range
                    t_pfv = int(comp_struct.find('./pixelFirstLutValue').text)
                    t_step = int(comp_struct.find('./stepSize').text)
                    num_vs = int(comp_struct.find('./numberOfValues').text)
                    t_range_inds = t_pfv + numpy.arange(num_vs)*t_step
                    coords_rg = grid.Row.SS * (t_range_inds - (image_data.SCPPixel.Row + row_shift))
                    return numpy.reshape(polynomial.polyfit(coords_rg, comp_values, 3), (-1, 1))

            # NB: I am neglecting the difference in polarization for these
            base_path = os.path.dirname(self.file_name)
            beta_file = os.path.join(
                base_path, 'calibration',
                self._find('./imageReferenceAttributes/lookupTableFileName[@sarCalibrationType="Beta Nought"]').text)
            sigma_file = os.path.join(
                base_path, 'calibration',
                self._find('./imageReferenceAttributes/lookupTableFileName[@sarCalibrationType="Sigma Nought"]').text)
            gamma_file = os.path.join(
                base_path, 'calibration',
                self._find('./imageReferenceAttributes/lookupTableFileName[@sarCalibrationType="Gamma"]').text)
            if not os.path.isfile(beta_file):
                logger.error(
                    msg="Beta calibration information should be located in file {}, "
                        "which doesn't exist.".format(beta_file))
                return None

            # perform beta, sigma, gamma fit
            beta_zero_sf_poly = perform_radiometric_fit(beta_file)
            sigma_zero_sf_poly = perform_radiometric_fit(sigma_file)
            gamma_zero_sf_poly = perform_radiometric_fit(gamma_file)

            # construct noise poly, neglecting the polarization again
            noise_level = None
            noise_stem = self._find('./imageReferenceAttributes/noiseLevelFileName').text
            noise_file = os.path.join(base_path, 'calibration', noise_stem)
            noise_root = _parse_xml(noise_file, without_ns=True)

            noise_levels = noise_root.findall('./perBeamReferenceNoiseLevel')
            beta0s = [entry for entry in noise_levels if (entry.find('./sarCalibrationType').text.startswith('Beta')
                                                          and entry.find('./beam').text == beam)]
            beta0_element = beta0s[0] if len(beta0s) > 0 else None

            if beta0_element is not None:
                pfv = float(beta0_element.find('pixelFirstNoiseValue').text)
                step = float(beta0_element.find('stepSize').text)
                beta0s = numpy.array(
                    [float(x) for x in beta0_element.find('noiseLevelValues').text.split()])
                range_inds = pfv + numpy.arange(len(beta0s)) * step
                range_coords = grid.Row.SS * (range_inds - (image_data.SCPPixel.Row + row_shift))
                noise_value = beta0s - 10*numpy.log10(polynomial.polyval(range_coords, beta_zero_sf_poly[:, 0]))
                noise_poly = polynomial.polyfit(range_coords, noise_value, 2)
                noise_level = NoiseLevelType_(
                    NoiseLevelType='ABSOLUTE',
                    NoisePoly = Poly2DType(Coefs=numpy.reshape(noise_poly, (-1, 1))))

            return RadiometricType(BetaZeroSFPoly=beta_zero_sf_poly,
                                   SigmaZeroSFPoly=sigma_zero_sf_poly,
                                   GammaZeroSFPoly=gamma_zero_sf_poly,
                                   NoiseLevel=noise_level)

        def correct_scp():
            scp_pixel = base_sicd.ImageData.SCPPixel.get_array()
            scp_ecf = base_sicd.project_image_to_ground(scp_pixel, projection_type='HAE')
            base_sicd.GeoData.SCP.ECF = scp_ecf

        def get_data_file_names():
            base_path = os.path.dirname(self.file_name)
            image_files = []
            img_attribute_node = self._find('./sceneAttributes/imageAttributes[@burst="{}"]'.format(burst))
            results = img_attribute_node.findall('./ipdf')

            if len(results) == 1:
                # there's either a single polarization, or it's one of the NITF files
                image_files.append(os.path.join(base_path, results[0].text))
            else:
                for pol in self._polarizations:
                    fname = None
                    for entry in results:
                        if entry.attrib.get('pole', None) == pol:
                            fname = entry.text
                    if fname is None:
                        raise ValueError('Got unexpected image file structure for burst {}'.format(burst))
                    image_files.append(os.path.join(base_path, fname))
            return image_files

        if self.generation != 'RCM':
            raise ValueError('Unhandled generation {}'.format(self.generation))

        center_frequency = float(self._find('./sourceAttributes/radarParameters/radarCenterFrequency').text)
        tx_pols, tx_rcv_pols = self._get_sicd_polarizations()

        # extract some common use doppler information
        num_of_pulses = int_func(
            self._find('./sourceAttributes/radarParameters/numberOfPulseIntervalsPerDwell[@beam="{}"]'.format(beam)).text)
        # NB: I am neglecting that prfInformation is provided separately for pols data because
        #   it appears identical
        pulse_rep_freq = float(
            self._find('./sourceAttributes'
                       '/radarParameters'
                       '/prfInformation[@beam="{}"]'.format(beam)+
                       '/pulseRepetitionFrequency').text)
        dop_rate_estimate_node = self._find('./dopplerRate/dopplerRateEstimate[@burst="{}"]'.format(burst))
        dop_centroid_estimate_node = self._find('./dopplerCentroid/dopplerCentroidEstimate[@burst="{}"]'.format(burst))
        dop_centroid_est_time = parse_timestring(
            dop_centroid_estimate_node.find('./timeOfDopplerCentroidEstimate').text, precision='us')
        processing_time_span = (num_of_pulses - 1)/pulse_rep_freq  # in seconds
        collect_start = dop_centroid_est_time - int_func(0.5*(processing_time_span*1000000))

        nitf, collection_info = self._get_sicd_collection_info(collect_start)
        image_creation = self._get_sicd_image_creation()
        position = self._get_sicd_position(collect_start)
        image_data, geo_data, row_shift = get_image_and_geo_data()
        grid = GridType(ImagePlane='SLANT', Type='RGZERO', Row=get_grid_row(), Col=get_grid_col())
        radar_collection = get_radar_collection()
        timeline = get_timeline()
        image_formation = get_image_formation()
        scpcoa = SCPCOAType(SideOfTrack=self._get_side_of_track())
        rma = get_rma_adjust_grid()
        radiometric = get_radiometric()
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
            Radiometric=radiometric,
            _NITF=nitf)
        correct_scp()
        base_sicd.derive()  # derive all the fields
        base_sicd.populate_rniirs(override=False)
        # now, make one copy per polarimetric entry, as appropriate
        the_files = get_data_file_names()

        the_sicds = []
        for i, (original_pol, sicd_pol) in enumerate(zip(self._polarizations, tx_rcv_pols)):
            this_sicd = base_sicd.copy()
            this_sicd.ImageFormation.RcvChanProc.ChanIndices = [i+1, ]
            this_sicd.ImageFormation.TxRcvPolarizationProc = sicd_pol
            the_sicds.append(this_sicd)
        return the_sicds, the_files

    def get_sicd_collection(self):
        """
        Gets the collection of sicd objects.

        Returns
        -------
        (List[List[SICDType]], List[List[str]])
        """

        sicds = []
        data_files = []
        if self._bursts is None:
            t_sicds, t_files = self._get_regular_sicd()
            sicds.append(t_sicds)
            data_files.append(t_files)
        else:
            for (beam, burst) in self._bursts:
                t_sicds, t_files = self._get_scansar_sicd(beam, burst)
                sicds.append(t_sicds)
                data_files.append(t_files)
        return sicds, data_files


##############
# reader implementation - really just borrows from tiff or NITF reader

class RadarSatReader(BaseReader, SICDTypeReader):
    """
    The reader object for RadarSat SAR file package.
    """

    __slots__ = ('_radar_sat_details', '_readers')

    def __init__(self, radar_sat_details):
        """

        Parameters
        ----------
        radar_sat_details : str|RadarSatDetails
            file name or RadarSatDetails object
        """

        if isinstance(radar_sat_details, string_types):
            radar_sat_details = RadarSatDetails(radar_sat_details)
        if not isinstance(radar_sat_details, RadarSatDetails):
            raise TypeError('The input argument for RadarSatReader must be a '
                            'filename or RadarSatDetails object')
        self._radar_sat_details = radar_sat_details
        # determine symmetry
        symmetry = self._radar_sat_details.get_symmetry()
        # get the sicd collection and data file names
        the_sicds, the_files = self.radarsat_details.get_sicd_collection()
        use_sicds = []
        the_chippers = []
        for sicd_entry, file_entry in zip(the_sicds, the_files):
            the_chippers.extend(self._construct_chippers(sicd_entry, file_entry, symmetry))
            use_sicds.extend(sicd_entry)

        SICDTypeReader.__init__(self, tuple(use_sicds))
        BaseReader.__init__(self, tuple(the_chippers), reader_type="SICD")

    def _construct_chippers(self, sicds, data_files, symmetry):
        """
        Construct the chippers for the provided

        Parameters
        ----------
        sicds : List[SICDType]
        data_files : List[str]

        Returns
        -------
        List[BaseChipper]
        """

        chippers = []
        if len(sicds) == len(data_files):
            for sicd, data_file in zip(sicds, data_files):
                fext = os.path.splitext(data_file)[1]
                if fext in ['.tiff', '.tif']:
                    chippers.append(_construct_tiff_chipper(sicd, data_file, symmetry))
                elif fext in ['.nitf', '.ntf']:
                    chippers.append(_construct_single_nitf_chipper(sicd, data_file, symmetry))
                else:
                    raise ValueError(
                        'The radarsat reader requires image files in tiff or nitf format. '
                        'Uncertain how to interpret file {}'.format(data_file))
        elif len(data_files) == 1:
            data_file = data_files[0]
            fext = os.path.splitext(data_file)[1]
            if fext not in ['.nitf', '.ntf']:
                raise ValueError(
                    'The radarsat has image data for multiple polarizations provided in a '
                    'single image file. This requires an image files in nitf format. '
                    'Uncertain how to interpret file {}'.format(data_file))
            chippers.extend(_construct_multiple_nitf_chippers(sicds, data_file, symmetry))
        else:
            raise ValueError(
                'Unclear how to construct chipper elements for {} sicd elements '
                'from {} image files.'.format(len(sicds), len(data_files)))
        return chippers

    @property
    def radarsat_details(self):
        # type: () -> RadarSatDetails
        """
        RadarSarDetails: The RadarSat/RCM details object.
        """

        return self._radar_sat_details

    @property
    def file_name(self):
        return self.radarsat_details.file_name
