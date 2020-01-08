# -*- coding: utf-8 -*-
"""
Functionality for reading Sentinel-1 data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"

import logging
import re
import os
from datetime import datetime
from xml.etree import ElementTree
from typing import List, Tuple

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from .tiff import TiffDetails, TiffReader

from ..sicd_elements.blocks import Poly1DType, Poly2DType
from ..sicd_elements.SICD import SICDType
from ..sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from ..sicd_elements.ImageCreation import ImageCreationType
from ..sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType, TxStepType
from ..sicd_elements.ImageData import ImageDataType
from ..sicd_elements.GeoData import GeoDataType, SCPType
from ..sicd_elements.Position import PositionType, XYZPolyType
from ..sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from ..sicd_elements.Timeline import TimelineType, IPPSetType
from ..sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType, TxFrequencyProcType
from ..sicd_elements.RMA import RMAType, INCAType
from ..sicd_elements.SCPCOA import SCPCOAType
from ..sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from ..geometry import point_projection
from .radarsat import _2d_poly_fit


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

    try:
        sentinel_details = SentinelDetails(file_name)
        print('File {} is determined to be a Sentinel-1 product.xml file.'.format(file_name))
        return SentinelReader(sentinel_details)
    except IOError:
        # we don't want to catch parsing errors, for now
        return None


##########
# helper functions

def _parse_xml(file_name, without_ns=False):
    root_node = ElementTree.parse(file_name).getroot()
    if without_ns:
        return root_node
    else:
        ns = dict([node for _, node in ElementTree.iterparse(file_name, events=('start-ns'))])
        return ns, root_node


def _get_seconds(dt1, dt2):  # type: (numpy.datetime64, numpy.datetime64) -> float
    tdt1 = dt1.astype('datetime64[us]')
    tdt2 = dt2.astype('datetime64[us]')  # convert both to microsecond precision
    return (tdt1.astype('int64') - tdt2.astype('int64'))*1e-6


###########
# parser and interpreter for sentinel-1 manifest.safe file

class SentinelDetails(object):
    __slots__ = ('_file_name', '_root_node', '_ns', '_satellite', '_product_type')

    def __init__(self, file_name):
        self._file_name = file_name
        self._ns, self._root_node = _parse_xml(file_name)
        self._satellite = self.find('./metadataSection'
                                    '/metadataObject[@ID="platform"]'
                                    '/metadataWrap'
                                    '/xmlData'
                                    '/safe:platform'
                                    '/safe:familyName').text
        if self._satellite != 'SENTINEL-1':
            raise ValueError('The platform in the manifest.safe file is required '
                             'to be SENTINEL-1, got {}'.format(self._satellite))
        self._product_type = self.find('./metadataSection'
                                       '/metadataObject[@ID="generalProductInformation"]'
                                       '/metadataWrap'
                                       '/xmlData'
                                       '/s1sarl1:standAloneProductInformation'
                                       '/s1sarl1:productType').text
        if self._product_type != 'SLC':
            raise ValueError('The product type in the manifest.safe file is required '
                             'to be "SLC", got {}'.format(self._product_type))

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

    def findall(self, tag):
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

        def get_file_location(schema_type, ids):
            if isinstance(ids, str):
                ids = [ids, ]
            for id in ids:
                do = self.find('dataObjectSection/dataObject[@repID="{}"]/[@ID="{}"]'.format(schema_type, id))
                # TODO: missing a *, or is there an extra slash in the above?
                if do is None:
                    continue
                return os.path.join(base_path, do.find('./byteStream/fileLocation').attrib['href'])
            return None

        base_path = os.path.dirname(self._file_name)

        files = []
        for mdu in self.findall('./informationPackageMap'
                                '/xfdu:contentUnit'
                                '/xfdu:contentUnit/[@repID="s1Level1MeasurementSchema"]'):
            # TODO: missing a *, or is there an extra slash in the above?
            # get the data file for this measurement
            fnames = {'data': get_file_location('s1Level1MeasurementSchema',
                                                mdu.find('dataObjectPointer').attrib['dataObjectID'])}
            # get the ids for product, noise, and calibration associated with this measurement data unit
            ids = mdu.attrib['dmdID'].split()
            # translate these ids to data object ids=file ids for the data files
            fids = [self.find('./metadataSection'
                              '/metadataObject[@ID="{}"]'
                              '/dataObjectPointer'.format(did)).attrib['dataObjectID'] for did in ids]
            # NB: there is (at most) one of these per measurement data unit
            fnames['product'] = get_file_location('s1Level1ProductSchema', fids)
            fnames['noise'] = get_file_location('s1Level1NoiseSchema', fids)
            fnames['calibration'] = get_file_location('s1Level1CalibrationSchema', fids)
            files.append(fnames)
        return files

    def _get_base_sicd(self):

        # CollectionInfo
        platform = self.find('./metadataSection'
                             '/metadataObject[@ID="platform"]'
                             '/metadataWrap'
                             '/xmlData/safe:platform')
        collector_name = platform.find('safe:familyName', self._ns).text + \
                         platform.find('safe:number', self._ns).text
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
                                             RadarMode=RadarModeType(ModeId=mode_id, ModeType=mode_type))
        # ImageCreation
        processing = self.find('./metadataSection'
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
            Profile='Prototype')
        # RadarCollection
        polarizations = self.findall('./metadataSection'
                                     '/metadataObject[@ID="generalProductInformation"]'
                                     '/metadataWrap'
                                     '/xmlData'
                                     '/s1sarl1:standAloneProductInformation'
                                     '/s1sarl1:transmitterReceiverPolarisation')

        radar_collection = RadarCollectionType(RcvChannels=[
            ChanParametersType(TxRcvPolarization=self._parse_pol(pol.text), index=i) for i, pol in enumerate(polarizations)])
        return SICDType(CollectionInfo=collection_info, ImageCreation=image_creation, RadarCollection=radar_collection)

    def _parse_product_sicd(self, product_file_name, base_sicd):
        """

        Parameters
        ----------
        product_file_name : str
        base_sicd : SICDType

        Returns
        -------
        SICDType|List[SICDType]
        """

        DT_FMT = '%Y-%m-%dT%H:%M:%S.%f'

        def get_center_frequency():
            return float(root_node.find('./generalAnnotation/productInformation/radarFrequency').text)

        def get_image_col_spacing_zdt():
            # Image column spacing in zero doppler time (seconds)
            # Sentinel-1 is always right-looking, so this should always be positive
            return float(root_node.find('./imageAnnotation/imageInformation/azimuthTimeInterval').text)

        def get_image_data(burst_list):
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
                                 SCPPixel=(int(num_rows / 2), int(num_cols / 2)))

        def get_common_grid():
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
                               ImpRespBW=2.*float(range_proc.find('./processingBandwidth').text)/speed_of_light,
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
                col_params = {'COEFFICIENT': range_proc.find('./windowCoefficient').text}
            col = DirParamType(SS=col_ss,
                               Sgn=-1,
                               KCtr=0,
                               ImpRespBW=dop_bw*ss_zd_s/col_ss,
                               WgtType=WgtTypeType(WindowName=col_window_name, Parameters=col_params))
            return GridType(ImagePlane=image_plane, Type='RGZERO', Row=row, Col=col)

        def get_common_timeline():
            prf = float(root_node.find('./generalAnnotation'
                                       '/downlinkInformationList'
                                       '/downlinkInformation'
                                       '/prf').text)
            return TimelineType(IPP=[IPPSetType(IPPPoly=(0, prf), index=0), ])

        def get_common_radar_collection():
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
            radar_collection.TxFrequency = TxFrequencyType(Min=min_frequency, Max=min_frequency+band_width)
            adc_sample_rate = float(root_node.find('./generalAnnotation'
                                                   '/productInformation'
                                                   '/rangeSamplingRate').text)  # Raw not decimated
            swl_list = root_node.findall('./generalAnnotation/downlinkInformationList/' +
                                         'downlinkInformation/downlinkValues/swlList/swl')
            radar_collection.Waveform = [
                WaveformParametersType(index=i,
                                       TxFreqStart=min_frequency,
                                       TxPulseLength=tx_pulse_length,
                                       TxFMRate=tx_fm_rate,
                                       TxRFBandwidth=band_width,
                                       RcvDemodType='CHIRP',
                                       RcvFMRate=0,
                                       ADCSampleRate=adc_sample_rate,
                                       RcvWindowLength=float(swl.find('./value').text))
                for i, swl in enumerate(swl_list)]
            return radar_collection

        def get_image_formation():
            pol = root_node.find('./adsHeader/polarisation').text
            st_bean_comp = 'NO' if out_sicd.CollectionInfo.RadarMode.ModeId[0] == 'S' else 'SV'
            return ImageFormationType(RcvChanProc=RcvChanProcType(NumChanProc=1,
                                                                  PRFScaleFactor=1),
                                      TxRcvPolarizationProc=self._parse_pol(pol),
                                      TStartProc=0,
                                      TxFrequencyProc=TxFrequencyProcType(
                                          MinProc=out_sicd.RadarCollection.TxFrequency.Min,
                                          MaxProc=out_sicd.RadarCollection.TxFrequency.Max),
                                      ImageFormAlgo='RMA',
                                      ImageBeamComp='NO',
                                      AzAutofocus='NO',
                                      RgAutofocus='NO',
                                      STBeamComp=st_bean_comp)

        def get_rma():
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

        def get_slice():
            slice_number = root_node.find('./imageAnnotation/imageInformation/sliceNumber')
            if slice_number is None:
                return '0'
            else:
                return slice_number.text

        def get_swath():
            return root_node.find('./adsHeader/swath').text

        def get_collection_info():
            collection_info = out_sicd.CollectionInfo.copy()
            # TODO: why previously define the below two in order to redefine here?
            collection_info.CollectorName = root_node.find('./adsHeader/missionId').text
            collection_info.RadarMode.ModeId = root_node.find('./adsHeader/mode').text
            slice = get_slice()
            swath = get_swath()
            collection_info.Parameters = {
                'SLICE': slice, 'BURST': '1', 'SWATH': swath, 'ORBIT_SOURCE': 'SLC_INTERNAL'}
            return collection_info

        def get_state_vectors(start):
            orbit_list = root_node.findall('./generalAnnotation/orbitList/orbit')
            shp = (len(orbit_list), )
            Ts = numpy.empty(shp, dtype=numpy.float64)
            Xs = numpy.empty(shp, dtype=numpy.float64)
            Ys = numpy.empty(shp, dtype=numpy.float64)
            Zs = numpy.empty(shp, dtype=numpy.float64)
            for i, orbit in enumerate(orbit_list):
                Ts[i] = _get_seconds(numpy.datetime64(orbit.find('./time').text, 'us'), start)
                Xs[i] = float(orbit.find('./position/x').text)
                Ys[i] = float(orbit.find('./position/y').text)
                Zs[i] = float(orbit.find('./position/z').text)
            return Ts, Xs, Ys, Zs

        def get_doppler_estimates(start):
            dc_estimate_list = root_node.findall('./dopplerCentroid/dcEstimateList/dcEstimate')
            shp = (len(dc_estimate_list), )
            dc_az_time = numpy.empty(shp, dtype=numpy.float64)
            dc_t0 = numpy.empty(shp, dtype=numpy.float64)
            data_dc_poly = []
            for i, dc_estimate in enumerate(dc_estimate_list):
                dc_az_time[i] = _get_seconds(numpy.datetime64(dc_estimate.find('./azimuthTime').text, 'us'), start)
                dc_t0[i] = float(dc_estimate.find('./t0').text)
                data_dc_poly.append(numpy.fromstring(dc_estimate.find('./dataDcPolynomial').text, sep=' '))
            return dc_az_time, dc_t0, data_dc_poly

        def get_azimuth_fm_estimates(start):
            azimuth_fm_rate_list = root_node.findall('./generalAnnotation/azimuthFmRateList/azimuthFmRate')
            shp = (len(azimuth_fm_rate_list), )
            az_t = numpy.empty(shp, dtype=numpy.float64)
            az_t0 = numpy.empty(shp, dtype=numpy.float64)
            k_a_poly = []
            for i, az_fm_rate in enumerate(azimuth_fm_rate_list):
                az_t[i] = _get_seconds(numpy.datetime64(az_fm_rate.find('./azimuthTime').text, 'us'), start)
                az_t0[i] = float(az_fm_rate.find('./t0').text)
                if az_fm_rate.find('c0') is not None:
                    # old style annotation xml file
                    k_a_poly.append(numpy.array([float(az_fm_rate.find('./c0').text),
                                                 float(az_fm_rate.find('./c1').text),
                                                 float(az_fm_rate.find('./c2').text)], dtype=numpy.float64))
                else:
                    k_a_poly.append(numpy.fromstring(az_fm_rate.find('./azimuthFmRatePolynomial').text, sep=' '))
            return az_t, az_t0, k_a_poly

        def set_core_name(sicd, start_dt, burst_num):
            slice = int(get_slice())
            swath = get_swath()
            sicd.CollectionInfo.CoreName = '{0:s}{1:s}{2:s}_{3:02d}_{4:s}_{5:02d}'.format(
                start_dt.strftime('%d%b%y'),
                or_coll_name,
                root_node.find('./adsHeader/missionDataTakeId').text,
                slice,
                swath,
                burst_num+1)

        def set_timeline(sicd, start, duration):
            prf = float(root_node.find('./generalAnnotation/downlinkInformationList/downlinkInformation/prf').text)
            timeline = sicd.Timeline
            timeline.CollectStart = start
            timeline.CollectDuration = duration
            timeline.IPP = [IPPSetType(index=0,
                                       TStart=0,
                                       TEnd=timeline.CollectDuration,
                                       IPPStart=0,
                                       IPPEnd=int(timeline.CollectDuration*prf)), ]
            sicd.ImageFormation.TEndProc = timeline.CollectDuration

        def set_position(sicd, start):
            Ts, Xs, Ys, Zs = get_state_vectors(start)
            poly_order = min(5, Ts.size-1)
            P_X = polynomial.polyfit(Ts, Xs, poly_order)
            P_Y = polynomial.polyfit(Ts, Ys, poly_order)
            P_Z = polynomial.polyfit(Ts, Zs, poly_order)
            sicd.Position = PositionType(ARPPoly=XYZPolyType(X=P_X, Y=P_Y, Z=P_Z))

        def update_rma_and_grid(sicd, first_line_relative_start, start, return_time_dets=False):
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
            poly_order = 2
            grid_samples = poly_order + 2
            cols = numpy.linspace(0, image_data.NumCols - 1, grid_samples, dtype=numpy.int64)
            rows = numpy.linspace(0, image_data.NumRows - 1, grid_samples, dtype=numpy.int64)
            coords_az = (cols - image_data.SCPPixel.Col)*grid.Col.SS
            coords_rg = (rows - image_data.SCPPixel.Row)*grid.Row.SS
            coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)
            time_ca_sampled = inca.TimeCAPoly(coords_rg_2d)
            doppler_rate_sampled = polynomial.polyval(coords_rg_2d, dr_ca_poly)
            tau = tau_0 + delta_tau_s + rows
            # Azimuth steering rate (constant, not dependent on burst or range)
            k_psi = numpy.deg2rad(float(
                root_node.find('./generalAnnotation/productInformation/azimuthSteeringRate').text))
            k_s = vm_ca*center_frequency*k_psi*2/speed_of_light
            k_a = az_rate_poly(tau - az_rate_t0[az_rate_poly_ind])
            k_t = (k_a*k_s)/(k_a-k_s)
            f_eta_c = dc_poly(tau-dc_t0[dc_poly_ind])
            eta = (cols - image_data.SCPPixel.Col)*ss_zd_s
            eta_c = -f_eta_c/k_a  # Beam center crossing time (TimeCOA)
            eta_ref = eta_c - eta_c[0]
            eta_2d, eta_ref_2d = numpy.meshgrid(eta, eta_ref)
            eta_arg = eta_2d - eta_ref_2d
            deramp_phase = k_t*(eta_arg*eta_arg)/2
            demod_phase = f_eta_c*eta_arg
            total_phase = deramp_phase + demod_phase
            phase = _2d_poly_fit(
                coords_az_2d, coords_rg_2d, total_phase, x_order=poly_order, y_order=poly_order)
            delta_kcoa_poly = polynomial.polyder(phase, axis=1)
            grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=delta_kcoa_poly)
            dop_centroid_poly = delta_kcoa_poly*grid.Col.SS/ss_zd_s
            inca.DopCentroidPoly = Poly2DType(Coefs=dop_centroid_poly)
            dop_centroid_sampled = inca.DopCentroidPoly(coords_rg_2d, coords_az_2d)
            time_coa_sampled = time_ca_sampled + dop_centroid_sampled / doppler_rate_sampled
            time_coa_poly = _2d_poly_fit(
                coords_rg_2d, coords_az_2d, time_coa_sampled, x_order=poly_order, y_order=poly_order)
            grid.TimeCOAPoly = Poly2DType(Coefs=time_coa_poly)
            if return_time_dets:
                return time_coa_sampled.min(), time_coa_sampled.max()

        def adjust_time(sicd, time_offset):
            # adjust TimeCOAPoly
            time_coa_poly = sicd.Grid.TimeCOAPoly.get_array()
            time_coa_poly[0, 0] -= time_offset
            sicd.Grid.TimeCOAPoly = Poly2DType(Coefs=time_coa_poly)
            # adjust TimeCAPoly
            time_ca_poly = sicd.RMA.INCA.TimeCAPoly.get_array()
            time_ca_poly[0] -= time_offset
            # shift ARPPoly
            sicd.Position.ARPPoly = sicd.Position.ARPPoly.shift(-time_offset, return_poly=True)

        def update_geodata(sicd):
            ecf = point_projection.image_to_ground([sicd.ImageData.SCPPixel.Row, sicd.ImageData.SCPPixel.Col], sicd)
            sicd.GeoData = GeoDataType(SCP=SCPType(ECF=ecf))

        def finalize_stripmap():
            # out_sicd is the one, we just complete it
            im_dat = out_sicd.ImageData
            im_dat.ValidData = (
                (0, 0), (0, im_dat.NumCols-1), (im_dat.NumRows-1, im_dat.NumCols-1), (im_dat.NumRows-1, 0))
            start_dt = datetime.strptime(root_node.find('./generalAnnotation'
                                                        '/downlinkInformationList'
                                                        '/downlinkInformation'
                                                        '/firstLineSensingTime').text, DT_FMT)
            start = numpy.datetime64(start_dt, 'us')
            stop = numpy.datetime64(root_node.find('./generalAnnotation'
                                                   '/downlinkInformationList'
                                                   '/downlinkInformation'
                                                   '/lastLineSensingTime').text, 'us')
            set_core_name(out_sicd, start_dt, 0)
            set_timeline(out_sicd, start, _get_seconds(stop, start))
            set_position(out_sicd, start)

            azimuth_time_first_line = numpy.datetime64(
                root_node.find('./imageAnnotation/imageInformation/productFirstLineUtcTime').text, 'us')
            first_line_relative_start = int(_get_seconds(azimuth_time_first_line, start))
            update_rma_and_grid(out_sicd, first_line_relative_start, start)
            update_geodata(out_sicd)
            out_sicd.derive()
            return out_sicd

        def finalize_bursts():
            # we will have one sicd per burst.
            sicds = []
            for i, burst in enumerate(burst_list):
                t_sicd = out_sicd.copy()
                t_sicd.CollectionInfo.Parameters['BURST'] = '{0:d}'.format(i+1)
                xml_first_cols = numpy.fromstring(burst.find('./firstValidSample').text, sep=' ', dtype=numpy.int64)
                xml_last_cols = numpy.fromstring(burst.find('./lastValidSample').text, sep=' ', dtype=numpy.int64)
                valid = (xml_first_cols >= 0) & (xml_last_cols >= 0)
                valid_cols = numpy.arange(xml_first_cols.size, dtype=numpy.int64)
                # TODO: I'm guessing - look at line 569
                first_row = int(numpy.min(xml_first_cols[valid]))
                last_row = int(numpy.max(xml_last_cols[valid]))
                first_col = valid_cols[0]
                last_col = valid_cols[-1]
                out_sicd.ImageData.ValidData = (
                    (first_row, first_col), (first_row, last_col), (last_row, last_col), (last_row, first_col))

                # This is the first and last zero doppler times of the columns in the burst.
                # Not really CollectStart and CollectDuration in SICD (first last pulse time)
                start_dt = datetime.strptime(burst.find('./azimuthTime').text, DT_FMT)
                start = numpy.datetime64(start_dt, 'us')
                set_core_name(t_sicd, start_dt, i)
                set_position(t_sicd, start)
                early, late = update_rma_and_grid(t_sicd, 0, start, return_time_dets=True)
                new_start = start + numpy.timedelta64(numpy.int64(early * 1e6), 'us')
                duration = late - early
                set_timeline(t_sicd, new_start, duration)
                # adjust my time offset
                adjust_time(t_sicd, early)
                update_geodata(t_sicd)
                t_sicd.derive()
                sicds.append(t_sicd)
            return sicds

        root_node = _parse_xml(product_file_name, without_ns=True)
        burst_list = root_node.findall('./swathTiming/burstList/burst')
        or_coll_name = base_sicd.CollectionInfo.CollectorName
        ######
        # create a common sicd with shared basic information here
        out_sicd = base_sicd.copy()
        out_sicd.ImageData = get_image_data(burst_list)
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

    def get_sicd_collection(self):
        base_sicd = self._get_base_sicd()
        # TODO:
        #   1.) get the sicd collection for each product
        #   2.) refine that using the noise data, if sensible
        #   3.) refine that using the calibration data, if sensible


class SentinelReader(object):
    def __init__(self, sentinel_meta):
        """

        Parameters
        ----------
        sentinel_meta : str|SentinelDetails
        """

        if isinstance(sentinel_meta, str):
            sentinel_meta = SentinelDetails(sentinel_meta)
        if not isinstance(sentinel_meta, SentinelDetails):
            raise TypeError('Input argumnet for SentinelReader must be a file name or SentinelReader object.')

