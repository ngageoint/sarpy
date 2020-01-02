# -*- coding: utf-8 -*-
"""
Functionality for reading Radarsat (RS2 and RCM) data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"

import logging
import re
import os
from datetime import datetime
from xml.etree import ElementTree
from typing import Union

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from .base import BaseReader
from .bip import BIPChipper

from ..sicd_elements.SICD import SICDType
from ..sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from ..sicd_elements.ImageCreation import ImageCreationType
from ..sicd_elements.ImageData import ImageDataType
from ..sicd_elements.GeoData import GeoDataType, SCPType
from ..sicd_elements.Position import PositionType, XYZPolyType
from ..sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from ..sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType, TxStepType


class RadarSatDetails(object):
    __slots__ = ('_file_name', '_root_node', '_satellite', '_product_type')

    def __init__(self, file_name):
        self._file_name = file_name
        ns = dict([node for event, node in ElementTree.iterparse(file_name, events=('start-ns', ))])
        ns['default'] = ns.get('')
        # TODO: what the hell is the above doing?
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
        with open(file_name) as fi:
            xmlstring = fi.read()
        # Remove the (first) default namespace definition (xmlns="http://some/namespace") and parse
        self._root_node = ElementTree.fromstring(re.sub('\\sxmlns="[^"]+"', '', xmlstring, count=1))

    @property
    def file_name(self):
        return self._file_name

    @property
    def satellite(self):
        return self._satellite

    @property
    def product_type(self):
        return self._product_type

    def find(self, tag):
        return self._root_node.find(tag)

    def findall(self, tag):
        return self._root_node.findall(tag)


# RADARSAT files correspond to one tiff per polarimetric band.
# As such, we will have one reader per polarimetric band (just a tiff)
class RadarSatReader(object):
    __slots__ = ('_radar_sat_details', '_readers')

    def __init__(self, radar_sat_details):
        """

        Parameters
        ----------
        radar_sat_details : RadarSatDetails
        """

        self._radar_sat_details = radar_sat_details
        # determine symmetry
        symmetry = self._get_symmetry()
        # determine datafiles
        data_files = self._get_data_file_names()
        # let's extract the sicd meta-data(s)
        # basic_sicd = self._get_default_sicd()  # not sure this will hold up
        # self._readers = ''

    def _get_symmetry(self):
        if self._radar_sat_details.satellite == 'RADARSAT-2':
            ia = 'imageAttributes'
        else:
            ia = 'imageReferenceAttributes'

        line_order = self._radar_sat_details.find('./default:' + ia +
                                                  '/default:rasterAttributes'
                                                  '/default:lineTimeOrdering').text.upper()
        look_dir = self._radar_sat_details.find('./default:sourceAttributes'
                                                '/default:radarParameters'
                                                '/default:antennaPointing').text.upper()
        sample_order = self._radar_sat_details.find('./default:' + ia +
                                                    '/default:rasterAttributes'
                                                    '/default:pixelTimeOrdering').text.upper()
        return ((line_order == 'DECREASING') != (look_dir.startswith('L')),
                sample_order == 'DECREASING',
                True)

    def _get_data_file_names(self):
        if self._radar_sat_details.satellite == 'RADARSAT-2':
            data_file_tag = './default:imageAttributes/default:fullResolutionImageData'
        else:
            data_file_tag = './default:sceneAttributes/default:imageAttributes/default:ipdf'
        base_path = os.path.dirname(self._radar_sat_details.file_name)
        return (os.path.join(base_path, entry.text) for entry in self._radar_sat_details.findall(data_file_tag))

    def _get_default_sicd(self):
        def find(tag):  # type: (str) -> Union[None, str]
            val = self._radar_sat_details.find(tag)
            if val is None:
                return val
            return val.text

        def get_seconds(dt1, dt2):  # type: (numpy.datetime64, numpy.datetime64) -> float
            tdt1 = dt1.astype('datetime64[us]')
            tdt2 = dt2.astype('datetime64[us]')  # convert both to microsecond precision
            return (tdt1.astype('int64') - tdt2.astype('int64'))*1e-6

        ISO_FMT = '%Y-%m-%dT%H:%M:%S.%fZ'
        # basic meta data used in a variety of places
        start_time_str = find('./sourceAttributes/rawDataStartTime')
        start_time_dt = datetime.strptime(start_time_str, ISO_FMT)  # still a naive dt?
        start_time = numpy.datetime64(start_time_dt, 'us')
        if self._radar_sat_details.satellite == 'RADARSAT-2':
            GEN = 'RS2'
        else:
            GEN = 'RCM'
        #################
        # CollectionInfo
        classification = 'UNCLASSIFIED'
        if GEN == 'RCM':
            class_str = find('./securityAttributes/securityClassification').upper()
            if 'UNCLASS' not in class_str:
                classification = class_str
        collector_name = self._radar_sat_details.satellite
        date_str = start_time_dt.strftime('%d%B%y').upper()
        if GEN == 'RS2':
            core_name = date_str + GEN + find('./sourceAttributes/imageId')
        else:
            core_name = date_str + collector_name.replace('-', '') + start_time_dt.strftime('%H%M%S')
        # CollectionInfo.RadarMode
        mode_id = find('./sourceAttributes/beamModeMnemonic')
        beam_mode = find('./sourceAttributes/beamMode')
        acq_type = find('./sourceAttributes/radarParameters/acquisitionType')
        if (beam_mode is not None and beam_mode.upper().startswith("SPOTLIGHT")) \
                or (acq_type is not None and acq_type.upper().startswith("SPOTLIGHT")) \
                or 'SL' in mode_id:
            mode_type = 'SPOTLIGHT'
        elif mode_id.startswith('SC'):
            raise ValueError('ScanSAR mode data is not currently handled.')
        else:
            mode_type = 'STRIPMAP'
        radar_mode = RadarModeType(ModeId=mode_id, ModeType=mode_type)
        collection_info = CollectionInfoType(
            Classification=classification, CollectorName=collector_name,
            CoreName=core_name, RadarMode=radar_mode, CollectType='MONOSTATIC')
        ###################
        # ImageCreation - all in one tag of product xml
        processing_info = self._radar_sat_details.find('./imageGenerationParameters/generalProcessingInformation')
        image_creation = ImageCreationType(
            Application=processing_info.find('softwareVersion').text,
            DateTime=processing_info.find('processingTime').text,
            Site=processing_info.find('processingFacility').text,
            Profile='Prototype')
        ###################
        # ImageData
        pixel_type = 'RE16I_IM16I'
        if GEN == 'RS2':
            cols = int(find('./imageAttributes/rasterAttributes/numberOfLines'))
            rows = int(find('./imageAttributes/rasterAttributes/numberOfSamplesPerLine'))
            tie_points = self._radar_sat_details.findall('./imageAttributes'
                                                         '/geographicInformation'
                                                         '/geolocationGrid'
                                                         '/imageTiePoint')
        else:
            cols = int(find('./sceneAttributes/imageAttributes/numLines'))
            rows = int(find('./sceneAttributes/imageAttributes/samplesPerLine'))
            tie_points = self._radar_sat_details.findall('./imageReferenceAttributes'
                                                         '/geographicInformation'
                                                         '/geolocationGrid'
                                                         '/imageTiePoint')
            if find('./imageReferenceAttributes/rasterAttributes/bitsPerSample') == '32':
                pixel_type = 'RE32F_IM32F'
        # let's use the tie point closest to the center as the SCP
        center_pixel = 0.5*numpy.array([rows-1, cols-1], dtype=numpy.float64)
        tp_pixels = numpy.array([
            [float(tp.find('imageCoordinate/pixel').text) for tp in tie_points],
            [float(tp.find('imageCoordinate/line').text) for tp in tie_points]], dtype=numpy.float64)
        scp_index = numpy.argmin(numpy.sum((tp_pixels - center_pixel)**2), axis=1)
        image_data = ImageDataType(NumRows=rows, NumCols=cols, FirstRow=0, FirstCol=0, PixelType=pixel_type,
                                   FullImage=(rows, cols),
                                   ValidData=((0, 0), (0, cols-1), (rows-1, cols-1), (rows-1, 0)),
                                   SCPPixel=numpy.round(tp_pixels[scp_index, :]))
        ###################
        # GeoData
        # TODO: why populate this in order to repopulate it later?
        scp = SCPType(LLH=(
            float(tie_points[scp_index].find('geodeticCoordinate/latitude').text),
            float(tie_points[scp_index].find('geodeticCoordinate/longitude').text),
            float(tie_points[scp_index].find('geodeticCoordinate/height').text),))
        geo_data = GeoDataType(SCP=scp)  # everything else will be derived
        ###################
        # Position
        # get radar position state information
        state_vectors = self._radar_sat_details.findall('./sourceAttributes'
                                                        '/orbitAndAttitude'
                                                        '/orbitInformation'
                                                        '/stateVector')
        # convert to relevant numpy arrays for polynomial fitting
        T = numpy.array([get_seconds(state_vec.find('timeStamp').text, start_time)
                         for state_vec in state_vectors], dtype=numpy.float64)
        X_pos = numpy.array([float(state_vec.find('xPosition').text)
                             for state_vec in state_vectors], dtype=numpy.float64)
        Y_pos = numpy.array([float(state_vec.find('yPosition').text)
                             for state_vec in state_vectors], dtype=numpy.float64)
        Z_pos = numpy.array([float(state_vec.find('zPosition').text)
                             for state_vec in state_vectors], dtype=numpy.float64)
        X_vel = numpy.array([float(state_vec.find('xVelocity').text)
                             for state_vec in state_vectors], dtype=numpy.float64)
        Y_vel = numpy.array([float(state_vec.find('yVelocity').text)
                             for state_vec in state_vectors], dtype=numpy.float64)
        Z_vel = numpy.array([float(state_vec.find('zVelocity').text)
                             for state_vec in state_vectors], dtype=numpy.float64)
        # Let's perform polynomial fitting for the position with cross validation for overfitting checks
        deg = 1
        prev_vel_error = numpy.inf
        P_x, P_y, P_z = None, None, None
        while deg < X_pos.size:
            # fit position
            P_x = polynomial.polyfit(T, X_pos, deg=deg)
            P_y = polynomial.polyfit(T, Y_pos, deg=deg)
            P_z = polynomial.polyfit(T, Z_pos, deg=deg)
            # extract estimated velocities
            X_vel_guess = polynomial.polyval(T, polynomial.polyder(P_x))
            Y_vel_guess = polynomial.polyval(T, polynomial.polyder(P_y))
            Z_vel_guess = polynomial.polyval(T, polynomial.polyder(P_z))
            # check our velocity error
            cur_vel_error = numpy.sum(numpy.array([X_vel-X_vel_guess, Y_vel-Y_vel_guess, Z_vel-Z_vel_guess])**2)
            # stop if the error is not smaller
            # TODO: find objectively bad fit?
            if cur_vel_error >= prev_vel_error:
                break
        position = PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))  # everything else will be derived
        ###################
        # Grid
        # get radar parameters node
        radar_params = self._radar_sat_details.find('./sourceAttributes/radarParameters')
        center_freq = float(radar_params.find('radarCenterFrequency').text)
        grid = GridType(ImagePlane='SLANT', Type='RGZERO')
        if GEN == 'RS2':
            row_ss = float(find('./imageAttributes/geographicInformation/geolocationGrid/imageTiePoint'))
        else:
            row_ss = float(find('./imageReferenceAttributes/geographicInformation/geolocationGrid/imageTiePoint'))
        row_irbw = 2*float(find('./imageGenerationParameters'
                                '/sarProcessingInformation'
                                '/totalProcessedRangeBandwidth'))/speed_of_light
        row_wgt_type = WgtTypeType(WindowName=find('./imageGenerationParameters'
                                                   '/sarProcessingInformation'
                                                   '/rangeWindow/windowName').upper())
        if row_wgt_type.WindowName == 'KAISER':
            row_wgt_type.Parameters['BETA'] = find('./imageGenerationParameters'
                                                   '/sarProcessingInformation'
                                                   '/rangeWindow/windowCoefficient').text
        col_wgt_type = WgtTypeType(WindowName=find('./imageGenerationParameters'
                                                   '/sarProcessingInformation'
                                                   '/azimuthWindow/windowName').upper())
        if col_wgt_type.WindowName == 'KAISER':
            col_wgt_type.Parameters['BETA'] = find('./imageGenerationParameters'
                                                   '/sarProcessingInformation'
                                                   '/azimuthWindow/windowCoefficient').text
        grid.Row = DirParamType(
            SS=row_ss, ImpRespBW=row_irbw, Sgn=-1, KCtr=2*center_freq/speed_of_light,
            DeltaKCOAPoly=((0,),), WgtType=row_wgt_type)
        grid.Col = DirParamType(Sgn=-1, KCtr=0, WgtType=col_wgt_type)
        ###################
        # RadarCollection
        # Ultrafine and spotlight modes have two pulses, otherwise just one.
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
        polarizations = radar_params.find('polarizations').text.split()
        tx_polarizations = ('RHC' if entry[0] == 'C' else entry[0] for entry in polarizations)
        rcv_polarizations = ('RHC' if entry[1] == 'C' else entry[1] for entry in polarizations)
        tx_rcv_polarizations = ('{}:{}'.format(*entry) for entry in zip(tx_polarizations, rcv_polarizations))
        radar_collection.RcvChannels = [ChanParametersType(TxRcvPolarization=entry, index=i)
                                        for i, entry in enumerate(tx_rcv_polarizations)]
        tx_pols = tuple(set(tx_polarizations))
        if len(tx_pols) == 1:
            radar_collection.TxPolarization = tx_pols[0]
        else:
            radar_collection.TxPolarization = 'SEQUENCE'
        radar_collection.TxSequence = [TxStepType(TxPolarization=entry, index=i) for i, entry in enumerate(tx_pols)]
        ###################
        # Timeline - line 414

        ###################
        # basic SICD
        base_sicd = SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            ImageData=image_data,
            GeoData=geo_data,
            Position=position)

        out = ()
        return out
