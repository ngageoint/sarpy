# -*- coding: utf-8 -*-
"""
Functionality for reading Radarsat (RS2 and RCM) data into a SICD model.
"""

import re
import os
from datetime import datetime
from xml.etree import ElementTree
from typing import Tuple, List, Union
import logging

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from sarpy.compliance import string_types
from ..general.base import BaseReader
from ..general.tiff import TiffDetails, TiffReader
from ..general.utils import get_seconds, parse_timestring

from .sicd_elements.blocks import Poly1DType, Poly2DType
from .sicd_elements.SICD import SICDType
from .sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from .sicd_elements.ImageCreation import ImageCreationType
from .sicd_elements.ImageData import ImageDataType
from .sicd_elements.GeoData import GeoDataType, SCPType
from .sicd_elements.Position import PositionType, XYZPolyType
from .sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from .sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType, TxStepType
from .sicd_elements.Timeline import TimelineType, IPPSetType
from .sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType, TxFrequencyProcType
from .sicd_elements.RMA import RMAType, INCAType
from .sicd_elements.SCPCOA import SCPCOAType
from .sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from ...geometry import point_projection
from .utils import fit_time_coa_polynomial, fit_position_xvalidation

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Khanh Ho", "Wade Schwartzkopf")


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

    try:
        radar_sat_details = RadarSatDetails(file_name)
        print('Path {} is determined to be or contain a RadarSat or RCM product.xml file.'.format(file_name))
        return RadarSatReader(radar_sat_details)
    except IOError:
        return None


##########
# helper functions

def _parse_xml(file_name, without_ns=False):
    # type: (str, bool) -> ElementTree.Element
    if without_ns:
        with open(file_name, 'rb') as fi:
            xml_string = fi.read()
        # Remove the (first) default namespace definition (xmlns="http://some/namespace") and parse
        return ElementTree.fromstring(re.sub(b'\\sxmlns="[^"]+"', b'', xml_string, count=1))
    else:
        return ElementTree.parse(file_name).getroot()


###########
# parser and interpreter for radarsat product.xml

class RadarSatDetails(object):
    __slots__ = ('_file_name', '_root_node', '_satellite', '_product_type')

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
        if not os.path.isfile(file_name):
            raise IOError('path {} does not exist or is not a file'.format(file_name))
        if os.path.split(file_name)[1] != 'product.xml':
            raise IOError('The radarsat or rcm file is expected to be named product.xml, got path {}'.format(file_name))

        self._file_name = file_name

        ns = dict([node for event, node in ElementTree.iterparse(file_name, events=('start-ns', ))])
        ns['default'] = ns.get('')
        root_node = ElementTree.parse(file_name).getroot()
        self._satellite = root_node.find('./default:sourceAttributes/default:satellite', ns).text.upper()
        self._product_type = root_node.find(
            './default:imageGenerationParameters'
            '/default:generalProcessingInformation'
            '/default:productType', ns).text.upper()
        if not ((self._satellite == 'RADARSAT-2' or self._satellite.startswith('RCM'))
                and self._product_type == 'SLC'):
            raise IOError('File {} does not appear to be an SLC product for a RADARSAT-2 '
                          'or RCM mission.'.format(self._file_name))
        # this is a radarsat file. Let's junk the default namespace.
        self._root_node = _parse_xml(file_name, without_ns=True)

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
    def product_type(self):
        """
        str: the product type
        """

        return self._product_type

    @property
    def generation(self):
        """
        str: RS2 or RCM
        """

        if self._satellite == 'RADARSAT-2':
            return 'RS2'
        else:
            return 'RCM'

    def _find(self, tag):
        # type: (str) -> ElementTree.Element
        return self._root_node.find(tag)

    def _findall(self, tag):
        # type: (str) -> List[ElementTree.Element, ...]
        return self._root_node.findall(tag)

    def get_symmetry(self):
        """
        Get the symmetry tuple.

        Returns
        -------
        Tuple[bool]
        """

        if self.generation == 'RS2':
            ia = 'imageAttributes'
        else:
            ia = 'imageReferenceAttributes'

        line_order = self._find('./{}/rasterAttributes/lineTimeOrdering'.format(ia)).text.upper()
        look_dir = self._find('./sourceAttributes/radarParameters/antennaPointing').text.upper()
        sample_order = self._find('./{}/rasterAttributes/pixelTimeOrdering'.format(ia)).text.upper()
        return ((line_order == 'DECREASING') != (look_dir.startswith('L')),
                sample_order == 'DECREASING',
                True)

    def get_data_file_names(self):
        """
        Gets the list of data file names.

        Returns
        -------
        Tuple[str]
        """

        if self.generation == 'RS2':
            data_file_tag = './imageAttributes/fullResolutionImageData'
        else:
            data_file_tag = './sceneAttributes/imageAttributes/ipdf'
        base_path = os.path.dirname(self.file_name)
        return (os.path.join(base_path, entry.text) for entry in self._findall(data_file_tag))

    def _get_start_time(self, get_datetime=False):
        """
        Gets the Collection start time

        Returns
        -------
        numpy.datetime64|datetime
        """

        if get_datetime:
            return datetime.strptime(
                self._find('./sourceAttributes/rawDataStartTime').text,
                '%Y-%m-%dT%H:%M:%S.%fZ')  # still a naive datetime?
        else:
            return parse_timestring(self._find('./sourceAttributes/rawDataStartTime').text)

    def _get_radar_params(self):
        # type: () -> ElementTree.Element
        return self._find('./sourceAttributes/radarParameters')

    def _get_center_frequency(self):
        """
        Gets the center frequency.

        Returns
        -------
        float
        """
        return float(self._find('./sourceAttributes/radarParameters/radarCenterFrequency').text)

    def _get_polarizations(self, radar_params=None):
        # type: (Union[None, ElementTree.Element]) -> (Tuple[str, ...], Tuple[str, ...])
        if radar_params is None:
            radar_params = self._get_radar_params()
        polarizations = radar_params.find('polarizations').text.split()
        tx_polarizations = ['RHC' if entry[0] == 'C' else entry[0] for entry in polarizations]
        rcv_polarizations = ['RHC' if entry[1] == 'C' else entry[1] for entry in polarizations]
        tx_rcv_polarizations = tuple('{}:{}'.format(*entry) for entry in zip(tx_polarizations, rcv_polarizations))
        # I'm not sure using a set object preserves ordering in all versions, so doing it manually
        tx_pols = []
        for el in tx_polarizations:
            if el not in tx_pols:
                tx_pols.append(el)
        return tuple(tx_pols), tx_rcv_polarizations

    def _get_radar_mode(self):
        """
        Gets the RadarMode.

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
            raise ValueError('ScanSAR mode data is not currently handled.')
        else:
            mode_type = 'STRIPMAP'
        return RadarModeType(ModeID=mode_id, ModeType=mode_type)

    def _get_collection_info(self):
        """
        Gets the CollectionInfo.

        Returns
        -------
        CollectionInfoType
        """

        try:
            import sarpy.io.complex.radarsat_addin as radarsat_addin
        except ImportError:
            radarsat_addin = None

        collector_name = self.satellite
        start_time_dt = self._get_start_time(get_datetime=True)
        date_str = start_time_dt.strftime('%d%b%y').upper()
        nitf = {}
        if self.generation == 'RS2':
            classification = 'UNCLASSIFIED'
            core_name = '{}{}{}'.format(date_str, self.generation, self._find('./sourceAttributes/imageId').text)
        elif self.generation == 'RCM':
            class_str = self._find('./securityAttributes/securityClassification').text.upper()
            classification = class_str if radarsat_addin is None else radarsat_addin.extract_radarsat_sec(nitf, class_str)
            core_name = '{}{}{}'.format(date_str, collector_name.replace('-', ''), start_time_dt.strftime('%H%M%S'))
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))
        return nitf, CollectionInfoType(
            Classification=classification, CollectorName=collector_name,
            CoreName=core_name, RadarMode=self._get_radar_mode(), CollectType='MONOSTATIC')

    def _get_image_creation(self):
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

    def _get_image_and_geo_data(self):
        """
        Gets the ImageData and GeoData metadata.

        Returns
        -------
        Tuple[ImageDataType, GeoDataType]
        """

        pixel_type = 'RE16I_IM16I'
        if self.generation == 'RS2':
            cols = int(self._find('./imageAttributes/rasterAttributes/numberOfLines').text)
            rows = int(self._find('./imageAttributes/rasterAttributes/numberOfSamplesPerLine').text)
            tie_points = self._findall('./imageAttributes'
                                       '/geographicInformation'
                                       '/geolocationGrid'
                                       '/imageTiePoint')
        elif self.generation == 'RCM':
            cols = int(self._find('./sceneAttributes/imageAttributes/numLines').text)
            rows = int(self._find('./sceneAttributes/imageAttributes/samplesPerLine').text)
            tie_points = self._findall('./imageReferenceAttributes'
                                       '/geographicInformation'
                                       '/geolocationGrid'
                                       '/imageTiePoint')
            if self._find('./imageReferenceAttributes/rasterAttributes/bitsPerSample').text == '32':
                pixel_type = 'RE32F_IM32F'
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))
        # let's use the tie point closest to the center as the SCP
        center_pixel = 0.5*numpy.array([rows-1, cols-1], dtype=numpy.float64)
        tp_pixels = numpy.zeros((len(tie_points), 2), dtype=numpy.float64)
        tp_llh = numpy.zeros((len(tie_points), 3), dtype=numpy.float64)
        for j, tp in enumerate(tie_points):
            tp_pixels[j, :] = (float(tp.find('imageCoordinate/pixel').text),
                               float(tp.find('imageCoordinate/line').text))
            tp_llh[j, :] = (float(tp.find('geodeticCoordinate/latitude').text),
                            float(tp.find('geodeticCoordinate/longitude').text),
                            float(tp.find('geodeticCoordinate/height').text))
        scp_index = numpy.argmin(numpy.sum((tp_pixels - center_pixel)**2, axis=1))
        im_data = ImageDataType(NumRows=rows,
                                NumCols=cols,
                                FirstRow=0,
                                FirstCol=0,
                                PixelType=pixel_type,
                                FullImage=(rows, cols),
                                ValidData=((0, 0), (0, cols-1), (rows-1, cols-1), (rows-1, 0)),
                                SCPPixel=numpy.round(tp_pixels[scp_index, :]))
        geo_data = GeoDataType(SCP=SCPType(LLH=tp_llh[scp_index, :]))
        return im_data, geo_data

    def _get_position(self):
        """
        Gets the Position.

        Returns
        -------
        PositionType
        """

        start_time = self._get_start_time()
        # get radar position state information
        state_vectors = self._findall('./sourceAttributes'
                                      '/orbitAndAttitude'
                                      '/orbitInformation'
                                      '/stateVector')
        # convert to relevant numpy arrays for polynomial fitting
        T = numpy.array([get_seconds(parse_timestring(state_vec.find('timeStamp').text),
                                     start_time, precision='us')
                         for state_vec in state_vectors], dtype=numpy.float64)
        Pos = numpy.hstack((
            numpy.array([float(state_vec.find('xPosition').text) for state_vec in state_vectors],
                        dtype=numpy.float64)[:, numpy.newaxis],
            numpy.array([float(state_vec.find('yPosition').text) for state_vec in state_vectors],
                        dtype=numpy.float64)[:, numpy.newaxis],
            numpy.array([float(state_vec.find('zPosition').text) for state_vec in state_vectors],
                        dtype=numpy.float64)[:, numpy.newaxis]))
        Vel = numpy.hstack((
            numpy.array([float(state_vec.find('xVelocity').text) for state_vec in state_vectors],
                        dtype=numpy.float64)[:, numpy.newaxis],
            numpy.array([float(state_vec.find('yVelocity').text) for state_vec in state_vectors],
                        dtype=numpy.float64)[:, numpy.newaxis],
            numpy.array([float(state_vec.find('zVelocity').text) for state_vec in state_vectors],
                        dtype=numpy.float64)[:, numpy.newaxis]))
        P_x, P_y, P_z = fit_position_xvalidation(T, Pos, Vel, max_degree=6)

        return PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

    def _get_grid_row(self):
        """
        Gets the Grid.Row metadata.

        Returns
        -------
        DirParamType
        """

        center_freq = self._get_center_frequency()
        if self.generation == 'RS2':
            row_ss = float(self._find('./imageAttributes'
                                      '/rasterAttributes'
                                      '/sampledPixelSpacing').text)
        elif self.generation == 'RCM':
            row_ss = float(self._find('./imageReferenceAttributes'
                                      '/rasterAttributes'
                                      '/sampledPixelSpacing').text)
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))

        row_irbw = 2*float(self._find('./imageGenerationParameters'
                                      '/sarProcessingInformation'
                                      '/totalProcessedRangeBandwidth').text)/speed_of_light
        row_wgt_type = WgtTypeType(WindowName=self._find('./imageGenerationParameters'
                                                         '/sarProcessingInformation'
                                                         '/rangeWindow/windowName').text.upper())
        if row_wgt_type.WindowName == 'KAISER':
            row_wgt_type.Parameters = {'BETA': self._find('./imageGenerationParameters'
                                                          '/sarProcessingInformation'
                                                          '/rangeWindow/windowCoefficient').text}
        return DirParamType(
            SS=row_ss, ImpRespBW=row_irbw, Sgn=-1, KCtr=2*center_freq/speed_of_light,
            DeltaKCOAPoly=Poly2DType(Coefs=((0,),)), WgtType=row_wgt_type)

    def _get_grid_col(self):
        """
        Gets the Grid.Col metadata.

        Returns
        -------
        DirParamType
        """

        col_wgt_type = WgtTypeType(WindowName=self._find('./imageGenerationParameters'
                                                         '/sarProcessingInformation'
                                                         '/azimuthWindow/windowName').text.upper())
        if col_wgt_type.WindowName == 'KAISER':
            col_wgt_type.Parameters = {'BETA': self._find('./imageGenerationParameters'
                                                          '/sarProcessingInformation'
                                                          '/azimuthWindow/windowCoefficient').text}

        return DirParamType(Sgn=-1, KCtr=0, WgtType=col_wgt_type)

    def _get_grid(self):
        """
        Gets the Grid.

        Returns
        -------
        GridType
        """
        return GridType(ImagePlane='SLANT', Type='RGZERO', Row=self._get_grid_row(), Col=self._get_grid_col())

    def _get_radar_collection(self):
        """
        Gets the RadarCollection.

        Returns
        -------
        RadarCollectionType
        """
        radar_params = self._get_radar_params()
        center_freq = self._get_center_frequency()
        # Ultrafine and spotlight modes have t pulses, otherwise just one.
        bandwidth_elements = sorted(radar_params.findall('pulseBandwidth'), key=lambda x: x.get('pulse'))
        pulse_length_elements = sorted(radar_params.findall('pulseLength'), key=lambda x: x.get('pulse'))
        adc_elements = sorted(radar_params.findall('adcSamplingRate'), key=lambda x: x.get('pulse'))
        samples_per_echo = float(radar_params.find('samplesPerEchoLine').text)
        wf_params = []
        bandwidths = numpy.empty((len(bandwidth_elements), ), dtype=numpy.float64)
        for i, (bwe, ple, adce) in enumerate(zip(bandwidth_elements, pulse_length_elements, adc_elements)):
            bandwidths[i] = float(bwe.text)
            samp_rate = float(adce.text)
            wf_params.append(WaveformParametersType(index=i,
                                                    TxRFBandwidth=float(bwe.text),
                                                    TxPulseLength=float(ple.text),
                                                    ADCSampleRate=samp_rate,
                                                    RcvWindowLength=samples_per_echo/samp_rate,
                                                    RcvDemodType='CHIRP',
                                                    RcvFMRate=0))
        tot_bw = numpy.sum(bandwidths)
        tx_freq = TxFrequencyType(Min=center_freq - 0.5*tot_bw, Max=center_freq + 0.5*tot_bw)
        radar_collection = RadarCollectionType(TxFrequency=tx_freq, Waveform=wf_params)
        radar_collection.Waveform[0].TxFreqStart = tx_freq.Min
        for i in range(1, len(bandwidth_elements)):
            radar_collection.Waveform[i].TxFreqStart = radar_collection.Waveform[i-1].TxFreqStart + \
                                                       radar_collection.Waveform[i-1].TxRFBandwidth
        tx_pols, tx_rcv_polarizations = self._get_polarizations(radar_params=radar_params)
        radar_collection.RcvChannels = [ChanParametersType(TxRcvPolarization=entry, index=i)
                                        for i, entry in enumerate(tx_rcv_polarizations)]
        if len(tx_pols) == 1:
            radar_collection.TxPolarization = tx_pols[0]
        else:
            radar_collection.TxPolarization = 'SEQUENCE'
            radar_collection.TxSequence = [TxStepType(TxPolarization=entry, index=i) for i, entry in enumerate(tx_pols)]
        return radar_collection

    def _get_timeline(self):
        """
        Gets the Timeline metadata.

        Returns
        -------
        TimelineType
        """

        timeline = TimelineType(CollectStart=self._get_start_time())
        pulse_parts = len(self._findall('./sourceAttributes/radarParameters/pulseBandwidth'))
        tx_pols, tx_rcv_polarizations = self._get_polarizations()

        if self.generation == 'RS2':
            pulse_rep_freq = float(self._find('./sourceAttributes'
                                              '/radarParameters'
                                              '/pulseRepetitionFrequency').text)
        elif self.generation == 'RCM':
            pulse_rep_freq = float(self._find('./sourceAttributes'
                                              '/radarParameters'
                                              '/prfInformation'
                                              '/pulseRepetitionFrequency').text)
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))

        pulse_rep_freq *= pulse_parts
        if pulse_parts == 2 and self._get_radar_mode().ModeType == 'STRIPMAP':
            # it's not completely clear why we need an additional factor of 2 for strip map
            pulse_rep_freq *= 2
        lines_processed = [
            float(entry.text) for entry in self._findall('./imageGenerationParameters'
                                                         '/sarProcessingInformation'
                                                         '/numberOfLinesProcessed')]
        # there should be one entry of num_lines_processed for each transmit/receive polarization
        # and they should all be the same. Omit if this is not the case.
        if (len(lines_processed) == len(tx_rcv_polarizations)) and \
                all(x == lines_processed[0] for x in lines_processed):
            num_lines_processed = lines_processed[0]*len(tx_pols)
            duration = num_lines_processed/pulse_rep_freq
            timeline.CollectDuration = duration
            timeline.IPP = [IPPSetType(
                index=0, TStart=0, TEnd=duration, IPPStart=0,
                IPPEnd=int(num_lines_processed),
                IPPPoly=Poly1DType(Coefs=(0, pulse_rep_freq))), ]
        return timeline

    def _get_image_formation(self, timeline, radar_collection):
        """
        Gets the ImageFormation.

        Parameters
        ----------
        timeline : TimelineType
        radar_collection : RadarCollectionType

        Returns
        -------
        ImageFormationType
        """
        pulse_parts = len(self._findall('./sourceAttributes/radarParameters/pulseBandwidth'))
        tx_pols, tx_rcv_polarizations = self._get_polarizations()

        return ImageFormationType(
            # PRFScaleFactor for either polarimetric or multi-step, but not both.
            RcvChanProc=RcvChanProcType(NumChanProc=1,
                                        PRFScaleFactor=1./max(pulse_parts, len(tx_pols))),
            ImageFormAlgo='RMA',
            TStartProc=0,
            TEndProc=timeline.CollectDuration,
            TxFrequencyProc=TxFrequencyProcType(MinProc=radar_collection.TxFrequency.Min,
                                                MaxProc=radar_collection.TxFrequency.Max),
            STBeamComp='NO',
            ImageBeamComp='NO',
            AzAutofocus='NO',
            RgAutofocus='NO')

    def _get_scpcoa(self):
        """
        Gets (minimal) SCPCOA.

        Returns
        -------
        SCPCOAType
        """
        side_of_track = self._find('./sourceAttributes/radarParameters/antennaPointing').text[0].upper()
        return SCPCOAType(SideOfTrack=side_of_track)

    def _get_rma_adjust_grid(self, scpcoa, grid, image_data, position, collection_info):
        """
        Gets the RMA metadata, and adjust the Grid.Col metadata.

        Parameters
        ----------
        scpcoa : SCPCOAType
        grid : GridType
        image_data : ImageDataType
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
        zero_dop_last_line = parse_timestring(self._find('./imageGenerationParameters'
                                                         '/sarProcessingInformation'
                                                         '/zeroDopplerTimeLastLine').text)
        zero_dop_first_line = parse_timestring(self._find('./imageGenerationParameters'
                                                          '/sarProcessingInformation'
                                                          '/zeroDopplerTimeFirstLine').text)
        if look > 1:  # SideOfTrack == 'L'
            # we explicitly want negative time order
            if zero_dop_first_line < zero_dop_last_line:
                zero_dop_first_line, zero_dop_last_line = zero_dop_last_line, zero_dop_first_line
        else:
            # we explicitly want positive time order
            if zero_dop_first_line > zero_dop_last_line:
                zero_dop_first_line, zero_dop_last_line = zero_dop_last_line, zero_dop_first_line
        col_spacing_zd = get_seconds(zero_dop_last_line, zero_dop_first_line, precision='us')/(image_data.NumCols - 1)
        # zero doppler time of SCP relative to collect start
        time_scp_zd = get_seconds(zero_dop_first_line, start_time, precision='us') + \
            image_data.SCPPixel.Col*col_spacing_zd
        if self.generation == 'RS2':
            near_range = float(self._find('./imageGenerationParameters'
                                          '/sarProcessingInformation'
                                          '/slantRangeNearEdge').text)
        elif self.generation == 'RCM':
            near_range = float(self._find('./sceneAttributes'
                                          '/imageAttributes'
                                          '/slantRangeNearEdge').text)
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))

        inca = INCAType(
            R_CA_SCP=near_range + (image_data.SCPPixel.Row * grid.Row.SS),
            FreqZero=center_freq)
        # doppler rate calculations
        velocity = position.ARPPoly.derivative_eval(time_scp_zd, 1)
        vel_ca_squared = numpy.sum(velocity*velocity)
        # polynomial representing range as a function of range distance from SCP
        r_ca = numpy.array([inca.R_CA_SCP, 1], dtype=numpy.float64)
        # doppler rate coefficients
        if self.generation == 'RS2':
            doppler_rate_coeffs = numpy.array(
                [float(entry) for entry in self._find('./imageGenerationParameters'
                                                      '/dopplerRateValues'
                                                      '/dopplerRateValuesCoefficients').text.split()],
                dtype=numpy.float64)
            doppler_rate_ref_time = float(self._find('./imageGenerationParameters'
                                                     '/dopplerRateValues'
                                                     '/dopplerRateReferenceTime').text)
        elif self.generation == 'RCM':
            doppler_rate_coeffs = numpy.array(
                [float(entry) for entry in self._find('./dopplerRate'
                                                      '/dopplerRateEstimate'
                                                      '/dopplerRateCoefficients').text.split()],
                dtype=numpy.float64)
            doppler_rate_ref_time = float(self._find('./dopplerRate'
                                                     '/dopplerRateEstimate'
                                                     '/dopplerRateReferenceTime').text)
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))

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
        if self.generation == 'RS2':
            doppler_cent_coeffs = numpy.array(
                [float(entry) for entry in self._find('./imageGenerationParameters'
                                                      '/dopplerCentroid'
                                                      '/dopplerCentroidCoefficients').text.split()],
                dtype=numpy.float64)
            doppler_cent_ref_time = float(self._find('./imageGenerationParameters'
                                                     '/dopplerCentroid'
                                                     '/dopplerCentroidReferenceTime').text)
            doppler_cent_time_est = parse_timestring(self._find('./imageGenerationParameters'
                                                                '/dopplerCentroid'
                                                                '/timeOfDopplerCentroidEstimate').text)
        elif self.generation == 'RCM':
            doppler_cent_coeffs = numpy.array(
                [float(entry) for entry in self._find('./dopplerCentroid'
                                                      '/dopplerCentroidEstimate'
                                                      '/dopplerCentroidCoefficients').text.split()],
                dtype=numpy.float64)
            doppler_cent_ref_time = float(self._find('./dopplerCentroid'
                                                     '/dopplerCentroidEstimate'
                                                     '/dopplerCentroidReferenceTime').text)
            doppler_cent_time_est = parse_timestring(self._find('./dopplerCentroid'
                                                                '/dopplerCentroidEstimate'
                                                                '/timeOfDopplerCentroidEstimate').text)
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))

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
        elif collection_info.RadarMode.ModeType == 'STRIPMAP':
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
            comp_struct = _parse_xml(component_file, without_ns=(self.generation != 'RS2'))
            comp_values = numpy.array(
                [float(entry) for entry in comp_struct.find('./gains').text.split()], dtype=numpy.float64)
            comp_values = 1. / (comp_values * comp_values)  # adjust for sicd convention
            if numpy.all(comp_values == comp_values[0]):
                return numpy.array([[comp_values[0], ], ], dtype=numpy.float64)
            else:
                # fit a 1-d polynomial in range
                coords_rg = (numpy.arange(image_data.NumRows) - image_data.SCPPixel.Row) * grid.Row.SS
                if self.generation == 'RCM':  # the rows are sub-sampled
                    start = int(comp_struct.find('./pixelFirstLutValue').text)
                    num_vs = int(comp_struct.find('./numberOfValues').text)
                    t_step = int(comp_struct.find('./stepSize').text)
                    if t_step > 0:
                        rng_indices = numpy.arange(start, num_vs, t_step)
                    else:
                        rng_indices = numpy.arange(start, -1, t_step)
                    coords_rg = coords_rg[rng_indices]
                return numpy.atleast_2d(polynomial.polyfit(coords_rg, comp_values, 3))

        base_path = os.path.dirname(self.file_name)
        if self.generation == 'RS2':
            beta_file = os.path.join(base_path, self._find('./imageAttributes'
                                                           '/lookupTable'
                                                           '[@incidenceAngleCorrection="Beta Nought"]').text)
            sigma_file = os.path.join(base_path, self._find('./imageAttributes'
                                                            '/lookupTable'
                                                            '[@incidenceAngleCorrection="Sigma Nought"]').text)
            gamma_file = os.path.join(base_path, self._find('./imageAttributes'
                                                            '/lookupTable'
                                                            '[@incidenceAngleCorrection="Gamma"]').text)
        elif self.generation == 'RCM':
            beta_file = os.path.join(base_path, 'calibration', self._find('./imageReferenceAttributes'
                                                                          '/lookupTableFileName'
                                                                          '[@sarCalibrationType="Beta Nought"]').text)
            sigma_file = os.path.join(base_path, 'calibration', self._find('./imageReferenceAttributes'
                                                                           '/lookupTableFileName'
                                                                           '[@sarCalibrationType="Sigma Nought"]').text)
            gamma_file = os.path.join(base_path, 'calibration', self._find('./imageReferenceAttributes'
                                                                           '/lookupTableFileName'
                                                                           '[@sarCalibrationType="Gamma"]').text)
        else:
            raise ValueError('unhandled generation {}'.format(self.generation))

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
            noise_level = NoiseLevelType_(NoiseLevelType='ABSOLUTE')
            pfv = float(beta0_element.find('pixelFirstNoiseValue').text)
            step = float(beta0_element.find('stepSize').text)
            beta0s = numpy.array(
                [float(x) for x in beta0_element.find('noiseLevelValues').text.split()])
            range_coords = grid.Row.SS*(numpy.arange(len(beta0s))*step + pfv - image_data.SCPPixel.Row)
            noise_poly = polynomial.polyfit(
                range_coords,
                beta0s - 10*numpy.log10(polynomial.polyval(range_coords, beta_zero_sf_poly[:, 0])), 2)
            noise_level.NoisePoly = Poly2DType(Coefs=numpy.atleast_2d(noise_poly))

        return RadiometricType(BetaZeroSFPoly=beta_zero_sf_poly,
                               SigmaZeroSFPoly=sigma_zero_sf_poly,
                               GammaZeroSFPoly=gamma_zero_sf_poly,
                               NoiseLevel=noise_level)

    @staticmethod
    def _update_geo_data(sicd):
        """
        Populates the GeoData.

        Parameters
        ----------
        sicd : SICDType

        Returns
        -------
        None
        """

        ecf = point_projection.image_to_ground(
            [sicd.ImageData.SCPPixel.Row, sicd.ImageData.SCPPixel.Col], sicd)
        sicd.GeoData.SCP = SCPType(ECF=ecf)  # LLH implicitly populated

    def get_sicd_collection(self):
        """
        Gets the list of sicd objects, one per polarimetric entry.

        Returns
        -------
        Tuple[SICDType]
        """

        nitf, collection_info = self._get_collection_info()
        image_creation = self._get_image_creation()
        image_data, geo_data = self._get_image_and_geo_data()
        position = self._get_position()
        grid = self._get_grid()
        radar_collection = self._get_radar_collection()
        timeline = self._get_timeline()
        image_formation = self._get_image_formation(timeline, radar_collection)
        scpcoa = self._get_scpcoa()
        rma = self._get_rma_adjust_grid(scpcoa, grid, image_data, position, collection_info)
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
        # now, make one copy per polarimetric entry, as appropriate
        tx_pols, tx_rcv_pols = self._get_polarizations()
        sicd_list = []
        for i, entry in enumerate(tx_rcv_pols):
            this_sicd = base_sicd.copy()
            this_sicd.ImageFormation.RcvChanProc.ChanIndices = [i+1, ]
            this_sicd.ImageFormation.TxRcvPolarizationProc = \
                this_sicd.RadarCollection.RcvChannels[i].TxRcvPolarization
            this_sicd.populate_rniirs(override=False)
            sicd_list.append(this_sicd)
        return tuple(sicd_list)


class RadarSatReader(BaseReader):
    """
    Gets a reader type object for RadarSat SAR files.
    RadarSat files correspond to one tiff per polarimetric band, so this will result
    in one tiff reader per polarimetric band.
    """
    __slots__ = ('_radar_sat_details', '_readers')

    def __init__(self, radar_sat_details):
        """

        Parameters
        ----------
        radar_sat_details : str|RadarSatDetails
            file name or RadarSatDeatils object
        """

        if isinstance(radar_sat_details, string_types):
            radar_sat_details = RadarSatDetails(radar_sat_details)
        if not isinstance(radar_sat_details, RadarSatDetails):
            raise TypeError('The input argument for RadarSatReader must be a '
                            'filename or RadarSatDetails object')
        self._radar_sat_details = radar_sat_details
        # determine symmetry
        symmetry = self._radar_sat_details.get_symmetry()
        # get the datafiles
        data_files = self._radar_sat_details.get_data_file_names()
        # get the sicd metadata objects
        sicds = self._radar_sat_details.get_sicd_collection()
        readers = []
        for sicd, file_name in zip(sicds, data_files):
            # create one reader per file/sicd
            # NB: the files are implicitly listed in the same order as polarizations
            tiff_details = TiffDetails(file_name)
            readers.append(TiffReader(tiff_details, sicd_meta=sicd, symmetry=symmetry))
        self._readers = tuple(readers)  # type: Tuple[TiffReader]
        sicd_tuple = tuple(reader.sicd_meta for reader in readers)
        chipper_tuple = tuple(reader._chipper for reader in readers)
        super(RadarSatReader, self).__init__(sicd_tuple, chipper_tuple, is_sicd_type=True)

    @property
    def radarsat_details(self):
        # type: () -> RadarSatDetails
        """
        RadarSarDetails: The radarsat/RCM details object.
        """

        return self._radar_sat_details

    @property
    def file_name(self):
        return self.radarsat_details.file_name
