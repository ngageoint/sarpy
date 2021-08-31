"""
Functionality for reading Sentinel-1 data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Daniel Haverporth")


import os
import logging
from datetime import datetime
from xml.etree import ElementTree
from typing import List, Tuple, Union

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light
from scipy.interpolate import griddata

from sarpy.compliance import string_types
from sarpy.io.general.base import SubsetReader, BaseReader, SarpyIOError
from sarpy.io.general.tiff import TiffDetails, TiffReader
from sarpy.io.general.utils import get_seconds, parse_timestring, is_file_like

from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    WaveformParametersType, ChanParametersType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from sarpy.geometry.geocoords import geodetic_to_ecf
from sarpy.io.complex.utils import two_dim_poly_fit, get_im_physical_coords

logger = logging.getLogger(__name__)


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a Sentinel file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    SentinelReader|None
        `SentinelReader` instance if Sentinel-1 file, `None` otherwise
    """

    if is_file_like(file_name):
        return None

    try:
        sentinel_details = SentinelDetails(file_name)
        logger.info('Path {} is determined to be or contain a Sentinel-1 manifest.safe file.'.format(file_name))
        return SentinelReader(sentinel_details)
    except (SarpyIOError, AttributeError, SyntaxError, ElementTree.ParseError):
        return None


##########
# helper functions

def _parse_xml(file_name, without_ns=False):
    root_node = ElementTree.parse(file_name).getroot()
    if without_ns:
        return root_node
    else:
        ns = dict([node for _, node in ElementTree.iterparse(file_name, events=('start-ns', ))])
        return ns, root_node


###########
# parser and interpreter for sentinel-1 manifest.safe file

class SentinelDetails(object):
    __slots__ = ('_file_name', '_root_node', '_ns', '_satellite', '_product_type', '_base_sicd')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        if os.path.isdir(file_name):  # its' the directory - point it at the manifest.safe file
            t_file_name = os.path.join(file_name, 'manifest.safe')
            if os.path.exists(t_file_name):
                file_name = t_file_name
        if not os.path.exists(file_name) or not os.path.isfile(file_name):
            raise SarpyIOError('path {} does not exist or is not a file'.format(file_name))
        if os.path.split(file_name)[1] != 'manifest.safe':
            raise SarpyIOError('The sentinel file is expected to be named manifest.safe, got path {}'.format(file_name))
        self._file_name = file_name

        self._ns, self._root_node = _parse_xml(file_name)
        # note that the manifest.safe apparently does not have a default namespace,
        # so we have to explicitly enter no prefix in the namespace dictionary
        self._ns[''] = ''
        self._satellite = self._find('./metadataSection'
                                     '/metadataObject[@ID="platform"]'
                                     '/metadataWrap'
                                     '/xmlData'
                                     '/safe:platform'
                                     '/safe:familyName').text
        if self._satellite != 'SENTINEL-1':
            raise ValueError('The platform in the manifest.safe file is required '
                             'to be SENTINEL-1, got {}'.format(self._satellite))
        self._product_type = self._find('./metadataSection'
                                        '/metadataObject[@ID="generalProductInformation"]'
                                        '/metadataWrap'
                                        '/xmlData'
                                        '/s1sarl1:standAloneProductInformation'
                                        '/s1sarl1:productType').text
        if self._product_type != 'SLC':
            raise ValueError('The product type in the manifest.safe file is required '
                             'to be "SLC", got {}'.format(self._product_type))
        self._base_sicd = self._get_base_sicd()

    @property
    def file_name(self):
        """
        str: the file name
        """

        return self._file_name

    @property
    def satellite(self):
        """
        str: the satellite
        """

        return self._satellite

    @property
    def product_type(self):
        """
        str: the product type
        """

        return self._product_type

    def _find(self, tag):
        """
        Pass through to ElementTree.Element.find(tag, ns).

        Parameters
        ----------
        tag : str

        Returns
        -------
        ElementTree.Element
        """

        return self._root_node.find(tag, self._ns)

    def _findall(self, tag):
        """
        Pass through to ElementTree.Element.findall(tag, ns).

        Parameters
        ----------
        tag : str

        Returns
        -------
        List[ElementTree.Element
        """

        return self._root_node.findall(tag, self._ns)

    @staticmethod
    def _parse_pol(str_in):
        # type: (str) -> str
        return '{}:{}'.format(str_in[0], str_in[1])

    def _get_file_sets(self):
        """
        Extracts paths for measurement and metadata files from a Sentinel manifest.safe file.
        These files will be grouped according to "measurement data unit" implicit in the
        Sentinel structure.

        Returns
        -------
        List[dict]
        """

        def get_file_location(schema_type, tids):
            if isinstance(tids, string_types):
                tids = [tids, ]
            for tid in tids:
                do = self._find('dataObjectSection/dataObject[@repID="{}"]/[@ID="{}"]'.format(schema_type, tid))
                if do is None:
                    continue
                return os.path.join(base_path, do.find('./byteStream/fileLocation').attrib['href'])
            return None

        base_path = os.path.dirname(self._file_name)

        files = []
        for mdu in self._findall('./informationPackageMap'
                                 '/xfdu:contentUnit'
                                 '/xfdu:contentUnit/[@repID="s1Level1MeasurementSchema"]'):
            # get the data file for this measurement
            fnames = {'data': get_file_location('s1Level1MeasurementSchema',
                                                mdu.find('dataObjectPointer').attrib['dataObjectID'])}
            # get the ids for product, noise, and calibration associated with this measurement data unit
            ids = mdu.attrib['dmdID'].split()
            # translate these ids to data object ids=file ids for the data files
            fids = [self._find('./metadataSection'
                               '/metadataObject[@ID="{}"]'
                               '/dataObjectPointer'.format(did)).attrib['dataObjectID'] for did in ids]
            # NB: there is (at most) one of these per measurement data unit
            fnames['product'] = get_file_location('s1Level1ProductSchema', fids)
            fnames['noise'] = get_file_location('s1Level1NoiseSchema', fids)
            fnames['calibration'] = get_file_location('s1Level1CalibrationSchema', fids)
            files.append(fnames)
        return files

    def _get_base_sicd(self):
        """
        Gets the base SICD element.

        Returns
        -------
        SICDType
        """

        from sarpy.__about__ import __version__

        # CollectionInfo
        platform = self._find('./metadataSection'
                              '/metadataObject[@ID="platform"]'
                              '/metadataWrap'
                              '/xmlData/safe:platform')
        collector_name = platform.find('safe:familyName', self._ns).text + platform.find('safe:number', self._ns).text
        mode_id = platform.find('./safe:instrument'
                                '/safe:extension'
                                '/s1sarl1:instrumentMode'
                                '/s1sarl1:mode', self._ns).text
        if mode_id == 'SM':
            mode_type = 'STRIPMAP'
        else:
            # TOPSAR - closest SICD analog is Dynamic Stripmap
            mode_type = 'DYNAMIC STRIPMAP'
        collection_info = CollectionInfoType(Classification='UNCLASSIFIED',
                                             CollectorName=collector_name,
                                             CollectType='MONOSTATIC',
                                             RadarMode=RadarModeType(ModeID=mode_id, ModeType=mode_type))
        # ImageCreation
        processing = self._find('./metadataSection'
                                '/metadataObject[@ID="processing"]'
                                '/metadataWrap'
                                '/xmlData'
                                '/safe:processing')
        facility = processing.find('safe:facility', self._ns)
        software = facility.find('safe:software', self._ns)
        image_creation = ImageCreationType(
            Application='{name} {version}'.format(**software.attrib),
            DateTime=processing.attrib['stop'],
            Site='{name}, {site}, {country}'.format(**facility.attrib),
            Profile='sarpy {}'.format(__version__))
        # RadarCollection
        polarizations = self._findall('./metadataSection'
                                      '/metadataObject[@ID="generalProductInformation"]'
                                      '/metadataWrap'
                                      '/xmlData'
                                      '/s1sarl1:standAloneProductInformation'
                                      '/s1sarl1:transmitterReceiverPolarisation')

        radar_collection = RadarCollectionType(RcvChannels=[
            ChanParametersType(TxRcvPolarization=self._parse_pol(pol.text), index=i)
            for i, pol in enumerate(polarizations)])
        return SICDType(CollectionInfo=collection_info, ImageCreation=image_creation, RadarCollection=radar_collection)

    def _parse_product_sicd(self, product_file_name):
        """

        Parameters
        ----------
        product_file_name : str

        Returns
        -------
        SICDType|List[SICDType]
        """

        DT_FMT = '%Y-%m-%dT%H:%M:%S.%f'
        root_node = _parse_xml(product_file_name, without_ns=True)
        burst_list = root_node.findall('./swathTiming/burstList/burst')
        # parse the geolocation information - for SCP calculation
        geo_grid_point_list = root_node.findall('./geolocationGrid/geolocationGridPointList/geolocationGridPoint')
        geo_pixels = numpy.zeros((len(geo_grid_point_list), 2), dtype=numpy.float64)
        geo_coords = numpy.zeros((len(geo_grid_point_list), 3), dtype=numpy.float64)
        for i, grid_point in enumerate(geo_grid_point_list):
            geo_pixels[i, :] = (float(grid_point.find('./pixel').text),
                                float(grid_point.find('./line').text))  # (row, col) order
            geo_coords[i, :] = (float(grid_point.find('./latitude').text),
                                float(grid_point.find('./longitude').text),
                                float(grid_point.find('./height').text))
        geo_coords = geodetic_to_ecf(geo_coords)

        def get_center_frequency():  # type: () -> float
            return float(root_node.find('./generalAnnotation/productInformation/radarFrequency').text)

        def get_image_col_spacing_zdt():  # type: () -> float
            # Image column spacing in zero doppler time (seconds)
            # Sentinel-1 is always right-looking, so this should always be positive
            return float(root_node.find('./imageAnnotation/imageInformation/azimuthTimeInterval').text)

        def get_image_data():  # type: () -> ImageDataType
            _pv = root_node.find('./imageAnnotation/imageInformation/pixelValue').text
            if _pv == 'Complex':
                pixel_type = 'RE16I_IM16I'
            else:
                # NB: we only handle SLC
                raise ValueError('SLC data should be 16-bit complex, got pixelValue = {}.'.format(_pv))
            if len(burst_list) > 0:
                # should be TOPSAR
                num_rows = int(root_node.find('./swathTiming/samplesPerBurst').text)
                num_cols = int(root_node.find('./swathTiming/linesPerBurst').text)
            else:
                # STRIPMAP
                # NB - these fields always contain the number of rows/cols in the entire tiff,
                # even if there are multiple bursts
                num_rows = int(root_node.find('./imageAnnotation/imageInformation/numberOfSamples').text)
                num_cols = int(root_node.find('./imageAnnotation/imageInformation/numberOfLines').text)
            # SCP pixel within single burst image is the same for all burst
            return ImageDataType(PixelType=pixel_type,
                                 NumRows=num_rows,
                                 NumCols=num_cols,
                                 FirstRow=0,
                                 FirstCol=0,
                                 FullImage=(num_rows, num_cols),
                                 SCPPixel=(int((num_rows - 1)/2), int((num_cols - 1)/2)))

        def get_common_grid():  # type: () -> GridType
            center_frequency = get_center_frequency()
            image_plane = 'SLANT' if root_node.find('./generalAnnotation/productInformation/projection').text == \
                'Slant Range' else None
            # get range processing node
            range_proc = root_node.find('./imageAnnotation'
                                        '/processingInformation'
                                        '/swathProcParamsList'
                                        '/swathProcParams'
                                        '/rangeProcessing')
            delta_tau_s = 1. / float(root_node.find('./generalAnnotation/productInformation/rangeSamplingRate').text)
            row_window_name = range_proc.find('./windowType').text.upper()
            row_params = None
            if row_window_name == 'NONE':
                row_window_name = 'UNIFORM'
            elif row_window_name == 'HAMMING':
                row_params = {'COEFFICIENT': range_proc.find('./windowCoefficient').text}
            row = DirParamType(SS=(speed_of_light/2)*delta_tau_s,
                               Sgn=-1,
                               KCtr=2*center_frequency/speed_of_light,
                               ImpRespBW=2. * float(range_proc.find('./processingBandwidth').text) / speed_of_light,
                               DeltaKCOAPoly=Poly2DType(Coefs=[[0, ]]),
                               WgtType=WgtTypeType(WindowName=row_window_name, Parameters=row_params))
            # get azimuth processing node
            az_proc = root_node.find('./imageAnnotation'
                                     '/processingInformation'
                                     '/swathProcParamsList'
                                     '/swathProcParams'
                                     '/azimuthProcessing')
            col_ss = float(root_node.find('./imageAnnotation/imageInformation/azimuthPixelSpacing').text)
            dop_bw = float(az_proc.find('./processingBandwidth').text)  # Doppler bandwidth
            ss_zd_s = get_image_col_spacing_zdt()
            col_window_name = az_proc.find('./windowType').text.upper()
            col_params = None
            if col_window_name == 'NONE':
                col_window_name = 'UNIFORM'
            elif col_window_name == 'HAMMING':
                col_params = {'COEFFICIENT': az_proc.find('./windowCoefficient').text}
            col = DirParamType(SS=col_ss,
                               Sgn=-1,
                               KCtr=0,
                               ImpRespBW=dop_bw*ss_zd_s/col_ss,
                               WgtType=WgtTypeType(WindowName=col_window_name, Parameters=col_params))
            return GridType(ImagePlane=image_plane, Type='RGZERO', Row=row, Col=col)

        def get_common_timeline():  # type: () -> TimelineType
            prf = float(root_node.find('./generalAnnotation'
                                       '/downlinkInformationList'
                                       '/downlinkInformation'
                                       '/prf').text)
            # NB: TEnd and IPPEnd are nonsense values which will be corrected
            return TimelineType(IPP=[IPPSetType(TStart=0, TEnd=0, IPPStart=0, IPPEnd=0, IPPPoly=(0, prf), index=0), ])

        def get_common_radar_collection():  # type: () -> RadarCollectionType
            radar_collection = out_sicd.RadarCollection.copy()
            center_frequency = get_center_frequency()
            min_frequency = center_frequency + \
                float(root_node.find('./generalAnnotation/downlinkInformationList/downlinkInformation'
                                     '/downlinkValues/txPulseStartFrequency').text)
            tx_pulse_length = float(root_node.find('./generalAnnotation'
                                                   '/downlinkInformationList'
                                                   '/downlinkInformation'
                                                   '/downlinkValues'
                                                   '/txPulseLength').text)
            tx_fm_rate = float(root_node.find('./generalAnnotation'
                                              '/downlinkInformationList'
                                              '/downlinkInformation'
                                              '/downlinkValues'
                                              '/txPulseRampRate').text)
            band_width = tx_pulse_length*tx_fm_rate
            pol = root_node.find('./adsHeader/polarisation').text
            radar_collection.TxPolarization = pol[0]
            radar_collection.TxFrequency = (min_frequency, min_frequency+band_width)
            adc_sample_rate = float(root_node.find('./generalAnnotation'
                                                   '/productInformation'
                                                   '/rangeSamplingRate').text)  # Raw not decimated
            swl_list = root_node.findall('./generalAnnotation/downlinkInformationList/' +
                                         'downlinkInformation/downlinkValues/swlList/swl')
            radar_collection.Waveform = [
                WaveformParametersType(index=j,
                                       TxFreqStart=min_frequency,
                                       TxPulseLength=tx_pulse_length,
                                       TxFMRate=tx_fm_rate,
                                       TxRFBandwidth=band_width,
                                       RcvFMRate=0,
                                       ADCSampleRate=adc_sample_rate,
                                       RcvWindowLength=float(swl.find('./value').text))
                for j, swl in enumerate(swl_list)]
            return radar_collection

        def get_image_formation():  # type: () -> ImageFormationType
            st_beam_comp = 'GLOBAL' if out_sicd.CollectionInfo.RadarMode.ModeID[0] == 'S' else 'SV'
            pol = self._parse_pol(root_node.find('./adsHeader/polarisation').text)
            # which channel does this pol correspond to?
            chan_indices = None
            for element in out_sicd.RadarCollection.RcvChannels:
                if element.TxRcvPolarization == pol:
                    chan_indices = [element.index, ]

            return ImageFormationType(RcvChanProc=RcvChanProcType(NumChanProc=1,
                                                                  PRFScaleFactor=1,
                                                                  ChanIndices=chan_indices),
                                      TxRcvPolarizationProc=pol,
                                      TStartProc=0,
                                      TxFrequencyProc=(
                                          out_sicd.RadarCollection.TxFrequency.Min,
                                          out_sicd.RadarCollection.TxFrequency.Max),
                                      ImageFormAlgo='RMA',
                                      ImageBeamComp='SV',
                                      AzAutofocus='NO',
                                      RgAutofocus='NO',
                                      STBeamComp=st_beam_comp)

        def get_rma():  # type: () -> RMAType
            center_frequency = get_center_frequency()
            tau_0 = float(root_node.find('./imageAnnotation/imageInformation/slantRangeTime').text)
            delta_tau_s = 1. / float(root_node.find('./generalAnnotation/productInformation/rangeSamplingRate').text)
            return RMAType(
                RMAlgoType='RG_DOP',
                INCA=INCAType(
                    FreqZero=center_frequency,
                    DopCentroidCOA=True,
                    R_CA_SCP=(0.5*speed_of_light)*(tau_0 + out_sicd.ImageData.SCPPixel.Row*delta_tau_s))
                           )

        def get_slice():  # type: () -> str
            slice_number = root_node.find('./imageAnnotation/imageInformation/sliceNumber')
            if slice_number is None:
                return '0'
            else:
                return slice_number.text

        def get_swath():  # type: () -> str
            return root_node.find('./adsHeader/swath').text

        def get_collection_info():  # type: () -> CollectionInfoType
            collection_info = out_sicd.CollectionInfo.copy()
            collection_info.CollectorName = root_node.find('./adsHeader/missionId').text
            collection_info.RadarMode.ModeID = root_node.find('./adsHeader/mode').text
            t_slice = get_slice()
            swath = get_swath()
            collection_info.Parameters = {
                'SLICE': t_slice, 'BURST': '1', 'SWATH': swath, 'ORBIT_SOURCE': 'SLC_INTERNAL'}
            return collection_info

        def get_state_vectors(start):
            # type: (numpy.datetime64) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            orbit_list = root_node.findall('./generalAnnotation/orbitList/orbit')
            shp = (len(orbit_list), )
            Ts = numpy.empty(shp, dtype=numpy.float64)
            Xs = numpy.empty(shp, dtype=numpy.float64)
            Ys = numpy.empty(shp, dtype=numpy.float64)
            Zs = numpy.empty(shp, dtype=numpy.float64)
            for j, orbit in enumerate(orbit_list):
                Ts[j] = get_seconds(parse_timestring(orbit.find('./time').text), start, precision='us')
                Xs[j] = float(orbit.find('./position/x').text)
                Ys[j] = float(orbit.find('./position/y').text)
                Zs[j] = float(orbit.find('./position/z').text)
            return Ts, Xs, Ys, Zs

        def get_doppler_estimates(start):
            # type: (numpy.datetime64) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray]]
            dc_estimate_list = root_node.findall('./dopplerCentroid/dcEstimateList/dcEstimate')
            shp = (len(dc_estimate_list), )
            dc_az_time = numpy.empty(shp, dtype=numpy.float64)
            dc_t0 = numpy.empty(shp, dtype=numpy.float64)
            data_dc_poly = []
            for j, dc_estimate in enumerate(dc_estimate_list):
                dc_az_time[j] = get_seconds(parse_timestring(dc_estimate.find('./azimuthTime').text),
                                            start, precision='us')
                dc_t0[j] = float(dc_estimate.find('./t0').text)
                data_dc_poly.append(numpy.fromstring(dc_estimate.find('./dataDcPolynomial').text, sep=' '))
            return dc_az_time, dc_t0, data_dc_poly

        def get_azimuth_fm_estimates(start):
            # type: (numpy.datetime64) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray]]
            azimuth_fm_rate_list = root_node.findall('./generalAnnotation/azimuthFmRateList/azimuthFmRate')
            shp = (len(azimuth_fm_rate_list), )
            az_t = numpy.empty(shp, dtype=numpy.float64)
            az_t0 = numpy.empty(shp, dtype=numpy.float64)
            k_a_poly = []
            for j, az_fm_rate in enumerate(azimuth_fm_rate_list):
                az_t[j] = get_seconds(parse_timestring(az_fm_rate.find('./azimuthTime').text),
                                      start, precision='us')
                az_t0[j] = float(az_fm_rate.find('./t0').text)
                if az_fm_rate.find('c0') is not None:
                    # old style annotation xml file
                    k_a_poly.append(numpy.array([float(az_fm_rate.find('./c0').text),
                                                 float(az_fm_rate.find('./c1').text),
                                                 float(az_fm_rate.find('./c2').text)], dtype=numpy.float64))
                else:
                    k_a_poly.append(numpy.fromstring(az_fm_rate.find('./azimuthFmRatePolynomial').text, sep=' '))
            return az_t, az_t0, k_a_poly

        def set_core_name(sicd, start_dt, burst_num):
            # type: (SICDType, datetime, int) -> None
            t_slice = int(get_slice())
            swath = get_swath()
            sicd.CollectionInfo.CoreName = '{0:s}{1:s}{2:s}_{3:02d}_{4:s}_{5:02d}'.format(
                start_dt.strftime('%d%b%y'),
                root_node.find('./adsHeader/missionId').text,
                root_node.find('./adsHeader/missionDataTakeId').text,
                t_slice,
                swath,
                burst_num+1)
            sicd.CollectionInfo.Parameters['BURST'] = '{0:d}'.format(burst_num+1)

        def set_timeline(sicd, start, duration):
            # type: (SICDType, numpy.datetime64, float) -> None
            prf = float(root_node.find('./generalAnnotation/downlinkInformationList/downlinkInformation/prf').text)
            timeline = sicd.Timeline
            timeline.CollectStart = start
            timeline.CollectDuration = duration
            timeline.IPP[0].TEnd = duration
            timeline.IPP[0].IPPEnd = int(timeline.CollectDuration*prf)
            sicd.ImageFormation.TEndProc = duration

        def set_position(sicd, start):
            # type: (SICDType, numpy.datetime64) -> None
            Ts, Xs, Ys, Zs = get_state_vectors(start)
            poly_order = min(5, Ts.size-1)
            P_X = polynomial.polyfit(Ts, Xs, poly_order)
            P_Y = polynomial.polyfit(Ts, Ys, poly_order)
            P_Z = polynomial.polyfit(Ts, Zs, poly_order)
            sicd.Position = PositionType(ARPPoly=XYZPolyType(X=P_X, Y=P_Y, Z=P_Z))

        def update_rma_and_grid(sicd, first_line_relative_start, start, return_time_dets=False):
            # type: (SICDType, Union[float, int], numpy.datetime64, bool) -> Union[None, Tuple[float, float]]

            center_frequency = get_center_frequency()
            # set TimeCAPoly
            ss_zd_s = get_image_col_spacing_zdt()
            eta_mid = ss_zd_s * float(out_sicd.ImageData.SCPPixel.Col)
            sicd.RMA.INCA.TimeCAPoly = Poly1DType(
                Coefs=[first_line_relative_start+eta_mid, ss_zd_s/out_sicd.Grid.Col.SS])
            range_time_scp = sicd.RMA.INCA.R_CA_SCP*2/speed_of_light
            # get velocity polynomial
            vel_poly = sicd.Position.ARPPoly.derivative(1, return_poly=True)
            # We pick a single velocity magnitude at closest approach to represent
            # the entire burst.  This is valid, since the magnitude of the velocity
            # changes very little.
            vm_ca = numpy.linalg.norm(vel_poly(sicd.RMA.INCA.TimeCAPoly[0]))
            az_rate_times, az_rate_t0, k_a_poly = get_azimuth_fm_estimates(start)
            # find the closest fm rate polynomial
            az_rate_poly_ind = int(numpy.argmin(numpy.abs(az_rate_times - sicd.RMA.INCA.TimeCAPoly[0])))
            az_rate_poly = Poly1DType(Coefs=k_a_poly[az_rate_poly_ind])
            dr_ca_poly = az_rate_poly.shift(t_0=az_rate_t0[az_rate_poly_ind] - range_time_scp,
                                            alpha=2/speed_of_light,
                                            return_poly=False)
            r_ca = numpy.array([sicd.RMA.INCA.R_CA_SCP, 1], dtype=numpy.float64)
            sicd.RMA.INCA.DRateSFPoly = numpy.reshape(
                -numpy.convolve(dr_ca_poly, r_ca)*(speed_of_light/(2*center_frequency*vm_ca*vm_ca)),
                (-1, 1))
            # Doppler Centroid
            dc_est_times, dc_t0, data_dc_poly = get_doppler_estimates(start)
            # find the closest doppler centroid polynomial
            dc_poly_ind = int(numpy.argmin(numpy.abs(dc_est_times - sicd.RMA.INCA.TimeCAPoly[0])))
            # we are going to move the respective polynomial from reference point as dc_t0 to
            # reference point at SCP time.
            dc_poly = Poly1DType(Coefs=data_dc_poly[dc_poly_ind])

            # Fit DeltaKCOAPoly, DopCentroidPoly, and TimeCOAPoly from data
            tau_0 = float(root_node.find('./imageAnnotation/imageInformation/slantRangeTime').text)
            delta_tau_s = 1./float(root_node.find('./generalAnnotation/productInformation/rangeSamplingRate').text)
            image_data = sicd.ImageData
            grid = sicd.Grid
            inca = sicd.RMA.INCA
            # common use for the fitting efforts
            poly_order = 2
            grid_samples = poly_order + 4
            cols = numpy.linspace(0, image_data.NumCols - 1, grid_samples, dtype=numpy.int64)
            rows = numpy.linspace(0, image_data.NumRows - 1, grid_samples, dtype=numpy.int64)
            coords_az = get_im_physical_coords(cols, grid, image_data, 'col')
            coords_rg = get_im_physical_coords(rows, grid, image_data, 'row')
            coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)

            # fit DeltaKCOAPoly
            tau = tau_0 + delta_tau_s*rows
            # Azimuth steering rate (constant, not dependent on burst or range)
            k_psi = numpy.deg2rad(float(
                root_node.find('./generalAnnotation/productInformation/azimuthSteeringRate').text))
            k_s = vm_ca*center_frequency*k_psi*2/speed_of_light
            k_a = az_rate_poly(tau - az_rate_t0[az_rate_poly_ind])
            k_t = (k_a*k_s)/(k_a-k_s)
            f_eta_c = dc_poly(tau - dc_t0[dc_poly_ind])
            eta = (cols - image_data.SCPPixel.Col)*ss_zd_s
            eta_c = -f_eta_c/k_a  # Beam center crossing time (TimeCOA)
            eta_ref = eta_c - eta_c[0]
            eta_2d, eta_ref_2d = numpy.meshgrid(eta, eta_ref)
            eta_arg = eta_2d - eta_ref_2d
            deramp_phase = 0.5*k_t[:, numpy.newaxis]*eta_arg*eta_arg
            demod_phase = eta_arg*f_eta_c[:, numpy.newaxis]
            total_phase = deramp_phase + demod_phase
            phase, residuals, rank, sing_values = two_dim_poly_fit(
                coords_rg_2d, coords_az_2d, total_phase,
                x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-35)
            logger.info(
                'The phase polynomial fit details:\n\t'
                'root mean square residuals = {}\n\t'
                'rank = {}\n\t'
                'singular values = {}'.format(residuals, rank, sing_values))

            # DeltaKCOAPoly is derivative of phase in azimuth/Col direction
            delta_kcoa_poly = polynomial.polyder(phase, axis=1)
            grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=delta_kcoa_poly)

            # derive the DopCentroidPoly directly
            dop_centroid_poly = delta_kcoa_poly*grid.Col.SS/ss_zd_s
            inca.DopCentroidPoly = Poly2DType(Coefs=dop_centroid_poly)

            # complete deriving the TimeCOAPoly, which depends on the DOPCentroidPoly
            time_ca_sampled = inca.TimeCAPoly(coords_az_2d)
            doppler_rate_sampled = polynomial.polyval(coords_rg_2d, dr_ca_poly)
            dop_centroid_sampled = inca.DopCentroidPoly(coords_rg_2d, coords_az_2d)
            time_coa_sampled = time_ca_sampled + dop_centroid_sampled / doppler_rate_sampled
            time_coa_poly, residuals, rank, sing_values = two_dim_poly_fit(
                coords_rg_2d, coords_az_2d, time_coa_sampled,
                x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
            logger.info(
                'The TimeCOAPoly fit details:\n\t'
                'root mean square residuals = {}\n\t'
                'rank = {}\n\t'
                'singular values = {}'.format(residuals, rank, sing_values))
            grid.TimeCOAPoly = Poly2DType(Coefs=time_coa_poly)
            if return_time_dets:
                return time_coa_sampled.min(), time_coa_sampled.max()

        def adjust_time(sicd, time_offset):
            # type: (SICDType, float) -> None
            # adjust TimeCOAPoly
            sicd.Grid.TimeCOAPoly.Coefs[0, 0] -= time_offset
            # adjust TimeCAPoly
            sicd.RMA.INCA.TimeCAPoly.Coefs[0] -= time_offset
            # shift ARPPoly
            sicd.Position.ARPPoly = sicd.Position.ARPPoly.shift(-time_offset, return_poly=True)

        def update_geodata(sicd):  # type: (SICDType) -> None
            scp_pixel = [sicd.ImageData.SCPPixel.Row, sicd.ImageData.SCPPixel.Col]
            ecf = sicd.project_image_to_ground(scp_pixel, projection_type='HAE')
            sicd.update_scp(ecf, coord_system='ECF')

        def get_scps(count):
            # SCPPixel - points at which to interpolate geo_pixels & geo_coords data
            scp_pixels = numpy.zeros((count, 2), dtype=numpy.float64)
            scp_pixels[:, 0] = int((out_sicd.ImageData.NumRows - 1)/2.)
            scp_pixels[:, 1] = int((out_sicd.ImageData.NumCols - 1)/2.) + \
                               out_sicd.ImageData.NumCols*(numpy.arange(count, dtype=numpy.float64))
            scps = numpy.zeros((count, 3), dtype=numpy.float64)
            for j in range(3):
                scps[:, j] = griddata(geo_pixels, geo_coords[:, j], scp_pixels)
            return scps

        def finalize_stripmap():  # type: () -> SICDType
            # out_sicd is the one that we return, just complete it
            # set preliminary geodata (required for projection)
            scp = get_scps(1)
            out_sicd.GeoData = GeoDataType(SCP=SCPType(ECF=scp[0, :]))  # EarthModel & LLH are implicitly set
            # NB: SCPPixel is already set to the correct thing
            im_dat = out_sicd.ImageData
            im_dat.ValidData = (
                (0, 0), (0, im_dat.NumCols-1), (im_dat.NumRows-1, im_dat.NumCols-1), (im_dat.NumRows-1, 0))
            start_dt = datetime.strptime(root_node.find('./generalAnnotation'
                                                        '/downlinkInformationList'
                                                        '/downlinkInformation'
                                                        '/firstLineSensingTime').text, DT_FMT)
            start = numpy.datetime64(start_dt, 'us')
            stop = parse_timestring(root_node.find('./generalAnnotation'
                                                   '/downlinkInformationList'
                                                   '/downlinkInformation'
                                                   '/lastLineSensingTime').text)
            set_core_name(out_sicd, start_dt, 0)
            set_timeline(out_sicd, start, get_seconds(stop, start, precision='us'))
            set_position(out_sicd, start)

            azimuth_time_first_line = parse_timestring(
                root_node.find('./imageAnnotation/imageInformation/productFirstLineUtcTime').text)
            first_line_relative_start = get_seconds(azimuth_time_first_line, start, precision='us')
            update_rma_and_grid(out_sicd, first_line_relative_start, start)
            update_geodata(out_sicd)
            return out_sicd

        def finalize_bursts():  # type: () -> List[SICDType]
            # we will have one sicd per burst.
            sicds = []
            scps = get_scps(len(burst_list))

            for j, burst in enumerate(burst_list):
                t_sicd = out_sicd.copy()
                # set preliminary geodata (required for projection)
                t_sicd.GeoData = GeoDataType(SCP=SCPType(ECF=scps[j, :]))  # EarthModel & LLH are implicitly set

                xml_first_cols = numpy.fromstring(burst.find('./firstValidSample').text, sep=' ', dtype=numpy.int64)
                xml_last_cols = numpy.fromstring(burst.find('./lastValidSample').text, sep=' ', dtype=numpy.int64)
                valid = (xml_first_cols >= 0) & (xml_last_cols >= 0)
                valid_cols = numpy.arange(xml_first_cols.size, dtype=numpy.int64)[valid]
                first_row = int(numpy.min(xml_first_cols[valid]))
                last_row = int(numpy.max(xml_last_cols[valid]))
                first_col = valid_cols[0]
                last_col = valid_cols[-1]
                t_sicd.ImageData.ValidData = (
                    (first_row, first_col), (first_row, last_col), (last_row, last_col), (last_row, first_col))

                # This is the first and last zero doppler times of the columns in the burst.
                # Not really CollectStart and CollectDuration in SICD (first last pulse time)
                start_dt = datetime.strptime(burst.find('./azimuthTime').text, DT_FMT)
                start = numpy.datetime64(start_dt, 'us')
                set_core_name(t_sicd, start_dt, j)
                set_position(t_sicd, start)
                early, late = update_rma_and_grid(t_sicd, 0, start, return_time_dets=True)
                new_start = start + numpy.timedelta64(numpy.int64(early * 1e6), 'us')
                duration = late - early
                set_timeline(t_sicd, new_start, duration)
                # adjust my time offset
                adjust_time(t_sicd, early)
                update_geodata(t_sicd)
                sicds.append(t_sicd)
            return sicds

        ######
        # create a common sicd with shared basic information here
        out_sicd = self._base_sicd.copy()
        out_sicd.ImageData = get_image_data()
        out_sicd.Grid = get_common_grid()
        out_sicd.Timeline = get_common_timeline()
        out_sicd.RadarCollection = get_common_radar_collection()
        out_sicd.ImageFormation = get_image_formation()
        out_sicd.RMA = get_rma()
        out_sicd.CollectionInfo = get_collection_info()
        ######
        # consider the burst situation, and split into the appropriate sicd collection
        if len(burst_list) > 0:
            return finalize_bursts()
        else:
            return finalize_stripmap()

    def _refine_using_calibration(self, cal_file_name, sicds):
        """

        Parameters
        ----------
        cal_file_name : str
        sicds : SICDType|List[SICDType]

        Returns
        -------
        None
        """

        # do not use before Sentinel baseline processing calibration update on 25 Nov 2015.
        if self._base_sicd.ImageCreation.DateTime < numpy.datetime64('2015-11-25'):
            return
        root_node = _parse_xml(cal_file_name, without_ns=True)
        if isinstance(sicds, SICDType):
            sicds = [sicds, ]

        def update_sicd(sicd, index):  # type: (SICDType, int) -> None
            # NB: in the deprecated version, there is a check if beta is constant,
            #   in which case constant values are used for beta/sigma/gamma.
            #   This has been removed.

            valid_lines = (line >= index*lines_per_burst) & (line < (index+1)*lines_per_burst)
            valid_count = numpy.sum(valid_lines)
            if valid_count == 0:
                # this burst contained no useful calibration data
                return

            coords_rg = (pixel[valid_lines] + sicd.ImageData.FirstRow - sicd.ImageData.SCPPixel.Row)*sicd.Grid.Row.SS
            coords_az = (line[valid_lines] + sicd.ImageData.FirstCol - sicd.ImageData.SCPPixel.Col)*sicd.Grid.Col.SS
            # NB: coords_rg = (valid_count, M) and coords_az = (valid_count, )
            coords_az = numpy.repeat(coords_az, pixel.shape[1])
            if valid_count > 1:
                coords_az = coords_az.reshape((valid_count, -1))

            def create_poly(arr, poly_order=2):
                rg_poly = polynomial.polyfit(coords_rg.flatten(), arr.flatten(), poly_order)
                az_poly = polynomial.polyfit(coords_az.flatten(), arr.flatten(), poly_order)
                return Poly2DType(Coefs=numpy.outer(az_poly/numpy.max(az_poly), rg_poly))

            if sicd.Radiometric is None:
                sicd.Radiometric = RadiometricType()
            sicd.Radiometric.SigmaZeroSFPoly = create_poly(sigma[valid_lines, :], poly_order=2)
            sicd.Radiometric.BetaZeroSFPoly = create_poly(beta[valid_lines, :], poly_order=2)
            sicd.Radiometric.GammaZeroSFPoly = create_poly(gamma[valid_lines, :], poly_order=2)
            return

        cal_vector_list = root_node.findall('./calibrationVectorList/calibrationVector')
        line = numpy.empty((len(cal_vector_list), ), dtype=numpy.float64)
        pixel, sigma, beta, gamma = [], [], [], []
        for i, cal_vector in enumerate(cal_vector_list):
            line[i] = float(cal_vector.find('./line').text)
            pixel.append(numpy.fromstring(cal_vector.find('./pixel').text, sep=' ', dtype=numpy.float64))
            sigma.append(numpy.fromstring(cal_vector.find('./sigmaNought').text, sep=' ', dtype=numpy.float64))
            beta.append(numpy.fromstring(cal_vector.find('./betaNought').text, sep=' ', dtype=numpy.float64))
            gamma.append(numpy.fromstring(cal_vector.find('./gamma').text, sep=' ', dtype=numpy.float64))
        lines_per_burst = sicds[0].ImageData.NumCols
        pixel = numpy.array(pixel)
        sigma = numpy.array(sigma)
        beta = numpy.array(beta)
        gamma = numpy.array(gamma)
        # adjust sentinel values for sicd convention (square and invert)
        sigma = 1./(sigma*sigma)
        beta = 1./(beta*beta)
        gamma = 1./(gamma*gamma)

        for ind, sic in enumerate(sicds):
            update_sicd(sic, ind)

    def _refine_using_noise(self, noise_file_name, sicds):
        """

        Parameters
        ----------
        noise_file_name : str
        sicds : SICDType|List[SICDType]

        Returns
        -------
        None
        """

        # do not use before Sentinel baseline processing calibration update on 25 Nov 2015.
        if self._base_sicd.ImageCreation.DateTime < numpy.datetime64('2015-11-25'):
            return
        root_node = _parse_xml(noise_file_name, without_ns=True)
        if isinstance(sicds, SICDType):
            sicds = [sicds, ]

        mode_id = sicds[0].CollectionInfo.RadarMode.ModeID
        lines_per_burst = sicds[0].ImageData.NumCols
        range_size_pixels = sicds[0].ImageData.NumRows

        def extract_vector(stem):
            # type: (str) -> Tuple[List[numpy.ndarray], List[Union[None, numpy.ndarray]], List[numpy.ndarray]]
            lines = []
            pixels = []
            noises = []
            noise_vector_list = root_node.findall('./{0:s}VectorList/{0:s}Vector'.format(stem))
            for i, noise_vector in enumerate(noise_vector_list):
                line = numpy.fromstring(noise_vector.find('./line').text, dtype=numpy.int64, sep=' ')
                # some datasets have noise vectors for negative lines - ignore these
                if numpy.all(line < 0):
                    continue
                pixel_node = noise_vector.find('./pixel')  # does not exist for azimuth noise
                if pixel_node is not None:
                    pixel = numpy.fromstring(pixel_node.text, dtype=numpy.int64, sep=' ')
                else:
                    pixel = None
                noise = numpy.fromstring(noise_vector.find('./{}Lut'.format(stem)).text, dtype=numpy.float64, sep=' ')
                # some datasets do not have any noise data (all 0's) - skipping these will throw things into disarray
                if not numpy.all(noise == 0):
                    # convert noise to dB - what about -inf values?
                    noise = 10*numpy.log10(noise)
                assert isinstance(noise, numpy.ndarray)

                # do some validity checks
                if (mode_id == 'IW') and numpy.any((line % lines_per_burst) != 0) and (i != len(noise_vector_list)-1):
                    # NB: the final burst has different timing
                    logger.error(
                        'Noise file should have one lut per burst, but more are present.\n\t'
                        'This may lead to confusion.')
                if (pixel is not None) and (pixel[-1] > range_size_pixels):
                    logger.error(
                        'Noise file has more pixels in LUT than range size.\n\t'
                        'This may lead to confusion.')

                lines.append(line)
                pixels.append(pixel)
                noises.append(noise)
            return lines, pixels, noises

        def populate_noise(sicd, index):  # type: (SICDType, int) -> None
            # NB: the default order was 7 before refactoring...that seems excessive.
            rg_poly_order = min(5, range_pixel[0].size-1)
            if sicd.CollectionInfo.RadarMode.ModeID[0] == 'S':
                # STRIPMAP - all LUTs apply
                # Treat range and azimuth polynomial components as fully independent
                az_poly_order = min(4, len(range_line) - 1)

                # NB: the previous rammed together two one-dimensional polys, but
                # we should do a 2-d fit.
                coords_rg = (range_pixel[0] + sicd.ImageData.FirstRow -
                             sicd.ImageData.SCPPixel.Row)*sicd.Grid.Row.SS
                coords_az = (range_line + sicd.ImageData.FirstCol - sicd.ImageData.SCPPixel.Col)*sicd.Grid.Col.SS

                coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)
                noise_poly, res, rank, sing_vals = two_dim_poly_fit(
                    coords_rg_2d, coords_az_2d, numpy.array(range_noise),
                    x_order=rg_poly_order, y_order=az_poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
                logger.info(
                    'NoisePoly fit details:\n\t'
                    'root mean square residuals = {}\n\t'
                    'rank = {}\n\t'
                    'singular values = {}'.format(res, rank, sing_vals))
            else:
                # TOPSAR has single LUT per burst
                # Treat range and azimuth polynomial components as weakly independent
                if index >= len(range_pixel):
                    logger.error(
                        'We have run out of noise information.\n\t'
                        'Current index = {}, length of noise array = {}.\n\t'
                        'The previous noise information will be used to populate the NoisePoly.'.format(
                            index, len(range_pixel)))
                rp_array = range_pixel[min(index, len(range_pixel)-1)]
                rn_array = range_noise[min(index, len(range_pixel)-1)]
                coords_rg = (rp_array + sicd.ImageData.FirstRow -
                             sicd.ImageData.SCPPixel.Row)*sicd.Grid.Row.SS
                rg_poly = numpy.array(
                    polynomial.polyfit(coords_rg, rn_array, rg_poly_order))
                az_poly = None
                if azimuth_noise is not None:
                    line0 = lines_per_burst*index
                    coords_az = (azimuth_line[0] - line0 -
                                 sicd.ImageData.SCPPixel.Col)*sicd.Grid.Col.SS
                    valid_lines = (azimuth_line[0] >= line0) & (azimuth_line[0] < line0 + lines_per_burst)
                    valid_count = numpy.sum(valid_lines)
                    if valid_count > 1:
                        az_poly_order = min(2, valid_lines.size-1)
                        az_poly = numpy.array(
                            polynomial.polyfit(coords_az[valid_lines], azimuth_noise[valid_lines], az_poly_order))
                if az_poly is not None:
                    noise_poly = numpy.zeros((rg_poly.size, az_poly.size), dtype=numpy.float64)
                    noise_poly[:, 0] += rg_poly
                    noise_poly[0, :] += az_poly
                else:
                    noise_poly = numpy.reshape(rg_poly, (-1, 1))
            if sicd.Radiometric is None:
                sicd.Radiometric = RadiometricType()
            sicd.Radiometric.NoiseLevel = NoiseLevelType_(NoiseLevelType='ABSOLUTE',
                                                          NoisePoly=Poly2DType(Coefs=noise_poly))

        # extract our noise vectors (used in populate_noise through implicit reference)
        if root_node.find('./noiseVectorList') is not None:
            # probably prior to March 2018
            range_line, range_pixel, range_noise = extract_vector('noise')
            azimuth_line, azimuth_pixel, azimuth_noise = None, None, None
        else:
            # noiseRange and noiseAzimuth fields began in March 2018
            range_line, range_pixel, range_noise = extract_vector('noiseRange')
            azimuth_line, azimuth_pixel, azimuth_noise = extract_vector('noiseAzimuth')
            azimuth_line = numpy.concatenate(azimuth_line, axis=0)

        # NB: range_line is actually a list of 1 element arrays - probably should have been parsed better
        range_line = numpy.concatenate(range_line, axis=0)
        for ind, sic in enumerate(sicds):
            populate_noise(sic, ind)

    @staticmethod
    def _derive(sicds):
        # type: (Union[SICDType, List[SICDType]]) -> None
        if isinstance(sicds, SICDType):
            sicds.derive()
            sicds.populate_rniirs(override=False)
        else:
            for sicd in sicds:
                sicd.derive()
                sicd.populate_rniirs(override=False)

    def get_sicd_collection(self):
        """
        Get the data file location(s) and corresponding sicd collection for each file.

        Returns
        -------
        List[Tuple[str, SICDType|List[SICDType]]]
            list of the form `(file, sicds)`. Here `file` is the data filename (tiff).
            `sicds` is either a single `SICDType` (STRIPMAP collect),
            or a list of `SICDType` (TOPSAR with multiple bursts).
        """

        out = []
        for entry in self._get_file_sets():
            # get the sicd collection for each product
            sicds = self._parse_product_sicd(entry['product'])
            # refine our sicds(s) using the calibration data (if sensible)
            self._refine_using_calibration(entry['calibration'], sicds)
            # refine our sicd(s) using the noise data (if sensible)
            self._refine_using_noise(entry['noise'], sicds)
            # populate our derived fields for the sicds
            self._derive(sicds)
            out.append((entry['data'], sicds))
        return out


class SentinelReader(BaseReader, SICDTypeReader):
    """
    Gets a reader type object for Sentinel-1 SAR files.
    """

    __slots__ = ('_sentinel_details', '_readers')

    def __init__(self, sentinel_details):
        """

        Parameters
        ----------
        sentinel_details : str|SentinelDetails
        """

        if isinstance(sentinel_details, string_types):
            sentinel_details = SentinelDetails(sentinel_details)
        if not isinstance(sentinel_details, SentinelDetails):
            raise TypeError('Input argument for SentinelReader must be a file name or SentinelReader object.')

        self._sentinel_details = sentinel_details  # type: SentinelDetails

        symmetry = (False, False, True)  # True for all Sentinel-1 data
        readers = []
        sicd_collection = self._sentinel_details.get_sicd_collection()
        sicd_collection_out = []
        for data_file, sicds in sicd_collection:
            tiff_details = TiffDetails(data_file)
            if isinstance(sicds, SICDType):
                readers.append(TiffReader(tiff_details, symmetry=symmetry))
                sicd_collection_out.append(sicds)
            elif len(sicds) == 1:
                readers.append(TiffReader(tiff_details, symmetry=symmetry))
                sicd_collection_out.append(sicds[0])
            else:
                # no need for SICD here - we're using subreaders
                p_reader = TiffReader(tiff_details, symmetry=symmetry)
                begin_col = 0
                for sicd in sicds:
                    assert isinstance(sicd, SICDType)
                    end_col = begin_col + sicd.ImageData.NumCols
                    dim1bounds = (0, tiff_details.tags['ImageWidth'])
                    dim2bounds = (begin_col, end_col)
                    readers.append(SubsetReader(p_reader, dim1bounds, dim2bounds))
                    begin_col = end_col
                    sicd_collection_out.append(sicd)

        self._readers = tuple(readers)  # type: Tuple[Union[TiffReader, SubsetReader]]
        chipper_tuple = tuple(reader._chipper for reader in readers)

        SICDTypeReader.__init__(self, tuple(sicd_collection_out))
        BaseReader.__init__(self, chipper_tuple, reader_type="SICD")

    @property
    def sentinel_details(self):
        # type: () -> SentinelDetails
        """
        SentinelDetails: The sentinel details object.
        """

        return self._sentinel_details

    @property
    def file_name(self):
        return self.sentinel_details.file_name
