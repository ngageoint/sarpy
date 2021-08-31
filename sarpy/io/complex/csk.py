"""
Functionality for reading Cosmo Skymed data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Jarred Barber", "Wade Schwartzkopf")

import logging
from collections import OrderedDict
import os
import re
from typing import Tuple, Dict
from datetime import datetime

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from sarpy.compliance import string_types, bytes_to_string
from sarpy.io.complex.base import SICDTypeReader, H5Chipper, h5py, is_hdf5
from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType, RowColType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    WaveformParametersType, ChanParametersType, TxStepType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, \
    RcvChanProcType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType
from sarpy.io.general.base import BaseReader, SarpyIOError
from sarpy.io.general.utils import get_seconds, parse_timestring, is_file_like
from sarpy.io.complex.utils import fit_time_coa_polynomial, fit_position_xvalidation

try:
    from sarpy.io.complex import csk_addin
except ImportError:
    csk_addin = None

logger = logging.getLogger(__name__)

########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a Cosmo Skymed file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str|BinaryIO
        the file_name to check

    Returns
    -------
    CSKReader|None
        `CSKReader` instance if Cosmo Skymed file, `None` otherwise
    """

    if is_file_like(file_name):
        return None

    if not is_hdf5(file_name):
        return None

    if h5py is None:
        return None

    try:
        csk_details = CSKDetails(file_name)
        logger.info('File {} is determined to be a Cosmo Skymed file.'.format(file_name))
        return CSKReader(csk_details)
    except SarpyIOError:
        return None


##########
# helper functions

def _extract_attrs(h5_element, out=None):
    if out is None:
        out = OrderedDict()
    for the_key in h5_element.attrs:
        val = h5_element.attrs[the_key]
        out[the_key] = bytes_to_string(val) if isinstance(val, bytes) else val
    return out


###########
# parser and interpreter for hdf5 attributes

class CSKDetails(object):
    """
    Parses and converts the Cosmo Skymed metadata
    """

    __slots__ = ('_file_name', '_mission_id', '_product_type')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        if h5py is None:
            raise ImportError("Can't read Cosmo Skymed files, because the h5py dependency is missing.")

        if not os.path.isfile(file_name):
            raise SarpyIOError('Path {} is not a file'.format(file_name))

        with h5py.File(file_name, 'r') as hf:
            try:
                self._mission_id = hf.attrs['Mission ID'].decode('utf-8')
            except KeyError:
                raise SarpyIOError('The hdf file does not have the top level attribute "Mission ID"')
            try:
                self._product_type = hf.attrs['Product Type'].decode('utf-8')
            except KeyError:
                raise SarpyIOError('The hdf file does not have the top level attribute "Product Type"')

        if self._mission_id not in ['CSK', 'CSG', 'KMPS']:
            raise ValueError('Expected hdf5 attribute `Mission ID` should be one of "CSK", "CSG", or "KMPS"). '
                             'Got Mission ID = {}.'.format(self._mission_id))
        if 'SCS' not in self._product_type:
            raise ValueError('Expected hdf to contain complex products '
                             '(attribute `Product Type` which contains "SCS"). '
                             'Got Product Type = {}'.format(self._product_type))

        self._file_name = file_name

    @property
    def file_name(self):
        """
        str: the file name
        """

        return self._file_name

    @property
    def mission_id(self):
        """
        str: the mission id
        """

        return self._mission_id

    @property
    def product_type(self):
        """
        str: the product type
        """

        return self._product_type

    def _get_hdf_dicts(self):
        with h5py.File(self._file_name, 'r') as hf:
            h5_dict = _extract_attrs(hf)
            band_dict = OrderedDict()
            shape_dict = OrderedDict()
            dtype_dict = OrderedDict()

            for gp_name in sorted(hf.keys()):
                if self._mission_id == 'CSG' and gp_name == 'LRHM':
                    continue  # this is extraneous

                gp = hf[gp_name]
                band_dict[gp_name] = OrderedDict()
                _extract_attrs(gp, out=band_dict[gp_name])

                if self._mission_id in ['CSK', 'KMPS']:
                    the_dataset = gp['SBI']
                elif self._mission_id == 'CSG':
                    the_dataset = gp['IMG']
                else:
                    raise ValueError('Unhandled mission id {}'.format(self._mission_id))
                _extract_attrs(the_dataset, out=band_dict[gp_name])
                shape_dict[gp_name] = the_dataset.shape[:2]
                if the_dataset.dtype.name == 'float32':
                    dtype_dict[gp_name] = 'RE32F_IM32F'
                elif the_dataset.dtype.name == 'int16':
                    dtype_dict[gp_name] = 'RE16I_IM16I'
                else:
                    raise ValueError(
                        'Got unexpected data type {}, name {}'.format(
                            the_dataset.dtype, the_dataset.dtype.name))

        return h5_dict, band_dict, shape_dict, dtype_dict

    @staticmethod
    def _parse_pol(str_in):
        return '{}:{}'.format(str_in[0], str_in[1])

    def _get_polarization(self, h5_dict, band_dict, band_name):
        # type: (dict, dict, str) -> str
        if 'Polarisation' in band_dict[band_name]:
            return band_dict[band_name]['Polarisation']
        elif 'Polarization' in h5_dict:
            return h5_dict['Polarization']
        else:
            raise ValueError(
                'Failed finding polarization for file {}\n\t'
                'mission id {} and band name {}'.format(self.file_name, self.mission_id, band_name))

    def _get_base_sicd(self, h5_dict, band_dict):
        # type: (dict, dict) -> SICDType

        def get_collection_info():  # type: () -> (dict, CollectionInfoType)
            acq_mode = h5_dict['Acquisition Mode'].upper()
            if self.mission_id == 'CSK':
                if acq_mode in ['HIMAGE', 'PINGPONG']:
                    mode_type = 'STRIPMAP'
                elif acq_mode in ['WIDEREGION', 'HUGEREGION']:
                    # scansar, processed as stripmap
                    mode_type = 'STRIPMAP'
                elif acq_mode in ['ENHANCED SPOTLIGHT', 'SMART']:
                    mode_type = 'DYNAMIC STRIPMAP'
                else:
                    logger.warning('Got unexpected acquisition mode {}'.format(acq_mode))
                    mode_type = 'DYNAMIC STRIPMAP'
            elif self.mission_id == 'KMPS':
                if acq_mode in ['STANDARD', 'ENHANCED STANDARD']:
                    mode_type = 'STRIPMAP'
                elif acq_mode in ['WIDE SWATH', 'ENHANCED WIDE SWATH']:
                    # scansar, processed as stripmap
                    mode_type = 'STRIPMAP'
                elif acq_mode in ['HIGH RESOLUTION', 'ENHANCED HIGH RESOLUTION', 'ULTRA HIGH RESOLUTION']:
                    # "spotlight"
                    mode_type = 'DYNAMIC STRIPMAP'
                else:
                    logger.warning('Got unexpected acquisition mode {}'.format(acq_mode))
                    mode_type = 'DYNAMIC STRIPMAP'
            elif self.mission_id == 'CSG':
                if acq_mode.startswith('SPOTLIGHT'):
                    mode_type = 'DYNAMIC STRIPMAP'
                elif acq_mode in ['STRIPMAP', 'QUADPOL']:
                    mode_type = "STRIPMAP"
                else:
                    logger.warning(
                        'Got unhandled acquisition mode {},\n\t'
                        'setting to DYNAMIC STRIPMAP'.format(acq_mode))
                    mode_type = 'DYNAMIC STRIPMAP'
            else:
                raise ValueError('Unhandled mission id {}'.format(self._mission_id))

            start_time_dt = collect_start.astype('datetime64[s]').astype(datetime)
            date_str = start_time_dt.strftime('%d%b%y').upper()
            time_str = start_time_dt.strftime('%H%M%S') + 'Z'
            core_name = '{}_{}_{}'.format(date_str, h5_dict['Satellite ID'], time_str)
            collect_info = CollectionInfoType(
                Classification='UNCLASSIFIED',
                CollectorName=h5_dict['Satellite ID'],
                CoreName=core_name,
                CollectType='MONOSTATIC',
                RadarMode=RadarModeType(ModeID=h5_dict['Multi-Beam ID'],
                                        ModeType=mode_type))
            return collect_info

        def get_image_creation():  # type: () -> ImageCreationType
            from sarpy.__about__ import __version__
            return ImageCreationType(
                DateTime=parse_timestring(h5_dict['Product Generation UTC'], precision='ns'),
                Site=h5_dict['Processing Centre'],
                Application='L0: `{}`, L1: `{}`'.format(
                    h5_dict.get('L0 Software Version', 'NONE'),
                    h5_dict.get('L1A Software Version', 'NONE')),
                Profile='sarpy {}'.format(__version__))

        def get_grid():  # type: () -> GridType
            def get_wgt_type(weight_name, coefficient, direction):
                if weight_name == 'GENERAL_COSINE':
                    # probably only for kompsat?
                    weight_name = 'HAMMING'
                    coefficient = 1-coefficient
                if coefficient is None:
                    params = None
                else:
                    params = {'COEFFICIENT': '{0:0.16G}'.format(coefficient)}
                out = WgtTypeType(WindowName=weight_name, Parameters=params)
                if weight_name != 'HAMMING':
                    logger.warning(
                        'Got unexpected weight scheme {} for {}.\n\t'
                        'The weighting will not be properly populated.'.format(weight_name, direction))
                return out

            if re.sub(' ', '', h5_dict['Projection ID']).upper() == 'SLANTRANGE/AZIMUTH':
                image_plane = 'SLANT'
                gr_type = 'RGZERO'
            else:
                image_plane = 'GROUND'
                gr_type = 'PLANE'
            # Row
            row_window_name = h5_dict['Range Focusing Weighting Function'].rstrip().upper()
            row_coefficient = h5_dict.get('Range Focusing Weighting Coefficient', None)
            row_weight = get_wgt_type(row_window_name, row_coefficient, 'Row')
            row = DirParamType(Sgn=-1,
                               KCtr=2*center_frequency/speed_of_light,
                               DeltaKCOAPoly=Poly2DType(Coefs=[[0, ], ]),
                               WgtType=row_weight)
            # Col
            col_window_name = h5_dict['Azimuth Focusing Weighting Function'].rstrip().upper()
            col_coefficient = h5_dict.get('Azimuth Focusing Weighting Coefficient', None)
            col_weight = get_wgt_type(col_window_name, col_coefficient, 'Col')
            col = DirParamType(Sgn=-1, KCtr=0, WgtType=col_weight)
            return GridType(ImagePlane=image_plane, Type=gr_type, Row=row, Col=col)

        def get_timeline():  # type: () -> TimelineType
            # NB: IPPEnd must be set, but will be replaced
            return TimelineType(CollectStart=collect_start,
                                CollectDuration=duration,
                                IPP=[IPPSetType(index=0, TStart=0, TEnd=0, IPPStart=0, IPPEnd=0), ])

        def get_position():  # type: () -> PositionType
            T = h5_dict['State Vectors Times']  # in seconds relative to ref time
            T += ref_time_offset
            Pos = h5_dict['ECEF Satellite Position']
            Vel = h5_dict['ECEF Satellite Velocity']
            P_x, P_y, P_z = fit_position_xvalidation(T, Pos, Vel, max_degree=8)
            return PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

        def get_radar_collection():
            # type: () -> RadarCollectionType
            tx_pols = []
            chan_params = []

            if self.mission_id == 'CSG' and len(band_dict) == 1 and \
                    h5_dict['Acquisition Mode'].upper() == 'QUADPOL':
                # it seems like 2nd generation files only contain one polarization
                pols = ['HH', 'HV', 'VH', 'VV']
                tx_pols.extend([pol[0] for pol in pols])
                chan_params.extend([
                    ChanParametersType(TxRcvPolarization=self._parse_pol(pol), index=i+1)
                    for i, pol in enumerate(pols)])
            else:
                for i, bdname in enumerate(band_dict):
                    pol = self._get_polarization(h5_dict, band_dict, bdname)
                    tx_pols.append(pol[0])
                    chan_params.append(ChanParametersType(TxRcvPolarization=self._parse_pol(pol), index=i+1))

            if len(tx_pols) == 1:
                return RadarCollectionType(RcvChannels=chan_params, TxPolarization=tx_pols[0])
            else:
                return RadarCollectionType(RcvChannels=chan_params,
                                           TxPolarization='SEQUENCE',
                                           TxSequence=[TxStepType(TxPolarization=pol,
                                                                  index=i+1) for i, pol in enumerate(tx_pols)])

        def get_image_formation():
            # type: () -> ImageFormationType
            return ImageFormationType(ImageFormAlgo='RMA',
                                      TStartProc=0,
                                      TEndProc=duration,
                                      STBeamComp='NO',
                                      ImageBeamComp='SV',
                                      AzAutofocus='NO',
                                      RgAutofocus='NO',
                                      RcvChanProc=RcvChanProcType(NumChanProc=1,
                                                                  PRFScaleFactor=1))

        def get_rma():
            # type: () -> RMAType
            inca = INCAType(FreqZero=center_frequency)
            return RMAType(RMAlgoType='OMEGA_K',
                           INCA=inca)

        def get_scpcoa():
            # type: () -> SCPCOAType
            return SCPCOAType(SideOfTrack=h5_dict['Look Side'][0:1].upper())

        # some common use parameters
        center_frequency = h5_dict['Radar Frequency']
        # relative times in csk are wrt some reference time - for sicd they should be relative to start time
        collect_start = parse_timestring(h5_dict['Scene Sensing Start UTC'], precision='ns')
        collect_end = parse_timestring(h5_dict['Scene Sensing Stop UTC'], precision='ns')
        duration = get_seconds(collect_end, collect_start, precision='ns')
        ref_time = parse_timestring(h5_dict['Reference UTC'], precision='ns')
        ref_time_offset = get_seconds(ref_time, collect_start, precision='ns')

        # assemble our pieces
        collection_info = get_collection_info()
        image_creation = get_image_creation()
        grid = get_grid()
        timeline = get_timeline()
        position = get_position()
        radar_collection = get_radar_collection()
        image_formation = get_image_formation()
        rma = get_rma()
        scpcoa = get_scpcoa()
        sicd = SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            Grid=grid,
            Timeline=timeline,
            Position=position,
            RadarCollection=radar_collection,
            ImageFormation=image_formation,
            RMA=rma,
            SCPCOA=scpcoa)
        return sicd

    def _get_dop_poly_details(self, h5_dict, band_dict, band_name):
        # type: (dict, dict, str) -> Tuple[float, float, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        def strip_poly(arr):
            # strip worthless (all zero) highest order terms
            # find last non-zero index
            last_ind = arr.size
            for i in range(arr.size-1, -1, -1):
                if arr[i] != 0:
                    break
                last_ind = i
            if last_ind == 0:
                return numpy.array([0, ], dtype=arr.dtype)
            return arr[:last_ind]

        dop_rate_poly_rg = h5_dict.get('Doppler Rate vs Range Time Polynomial', None)
        if dop_rate_poly_rg is None:
            dop_rate_poly_rg = band_dict[band_name].get('Doppler Rate vs Range Time Polynomial', None)
        if dop_rate_poly_rg is None:
            raise ValueError('No Doppler Rate Range Time polynomial found')
        dop_rate_poly_rg = strip_poly(dop_rate_poly_rg)

        if self._mission_id in ['CSK', 'KMPS']:
            az_ref_time = h5_dict['Azimuth Polynomial Reference Time']  # seconds
            rg_ref_time = h5_dict['Range Polynomial Reference Time']
            dop_poly_az = strip_poly(h5_dict['Centroid vs Azimuth Time Polynomial'])
            dop_poly_rg = strip_poly(h5_dict['Centroid vs Range Time Polynomial'])
        elif self._mission_id == 'CSG':
            az_ref_time = band_dict[band_name]['Azimuth Polynomial Reference Time']
            rg_ref_time = band_dict[band_name]['Range Polynomial Reference Time']
            dop_poly_az = strip_poly(band_dict[band_name]['Doppler Centroid vs Azimuth Time Polynomial'])
            dop_poly_rg = strip_poly(band_dict[band_name]['Doppler Centroid vs Range Time Polynomial'])
        else:
            raise ValueError('Unhandled mission id {}'.format(self._mission_id))
        return az_ref_time, rg_ref_time, dop_poly_az, dop_poly_rg, dop_rate_poly_rg

    def _get_band_specific_sicds(self, base_sicd, h5_dict, band_dict, shape_dict, dtype_dict):
        # type: (SICDType, dict, dict, dict, dict) -> Dict[str, SICDType]

        def update_scp_prelim(sicd, band_name):
            # type: (SICDType, str) -> None
            if self._mission_id in ['CSK', 'KMPS']:
                LLH = band_dict[band_name]['Centre Geodetic Coordinates']
            elif self._mission_id == 'CSG':
                LLH = h5_dict['Scene Centre Geodetic Coordinates']
            else:
                raise ValueError('Unhandled mission id {}'.format(self._mission_id))
            sicd.GeoData = GeoDataType(SCP=SCPType(LLH=LLH))  # EarthModel & ECF will be populated

        def update_image_data(sicd, band_name):
            # type: (SICDType, str) -> Tuple[float, float, float, float, int]
            cols, rows = shape_dict[band_name]
            # zero doppler time of first/last columns
            t_az_first_time = band_dict[band_name]['Zero Doppler Azimuth First Time']
            t_az_last_time = band_dict[band_name]['Zero Doppler Azimuth Last Time']
            t_ss_az_s = band_dict[band_name]['Line Time Interval']
            t_use_sign2 = 1
            if h5_dict['Look Side'].upper() == 'LEFT':
                t_use_sign2 = -1
                t_az_first_time, t_az_last_time = t_az_last_time, t_az_first_time
            # zero doppler time of first row
            t_rg_first_time = band_dict[band_name]['Zero Doppler Range First Time']
            # row spacing in range time (seconds)
            t_ss_rg_s = band_dict[band_name]['Column Time Interval']

            sicd.ImageData = ImageDataType(NumRows=rows,
                                           NumCols=cols,
                                           FirstRow=0,
                                           FirstCol=0,
                                           FullImage=(rows, cols),
                                           PixelType=dtype_dict[band_name],
                                           SCPPixel=RowColType(Row=int(rows/2),
                                                               Col=int(cols/2)))
            return t_rg_first_time, t_ss_rg_s, t_az_first_time, t_ss_az_s, t_use_sign2

        def check_switch_state():
            # type: () -> Tuple[int, Poly1DType]
            use_sign = 1 if t_dop_rate_poly_rg[0] < 0 else -1
            return use_sign, Poly1DType(Coefs=use_sign*t_dop_rate_poly_rg)

        def update_timeline(sicd, band_name):
            # type: (SICDType, str) -> None
            prf = band_dict[band_name]['PRF']
            duration = sicd.Timeline.CollectDuration
            ipp_el = sicd.Timeline.IPP[0]
            ipp_el.IPPEnd = duration*prf
            ipp_el.TEnd = duration
            ipp_el.IPPPoly = Poly1DType(Coefs=(0, prf))

        def update_radar_collection(sicd, band_name):
            # type: (SICDType, str) -> None
            ind = None
            for the_chan_index, chan in enumerate(sicd.RadarCollection.RcvChannels):
                if chan.TxRcvPolarization == polarization:
                    ind = the_chan_index
                    break
            if ind is None:
                raise ValueError('Failed to find receive channel for polarization {}'.format(polarization))

            chirp_length = band_dict[band_name]['Range Chirp Length']
            chirp_rate = abs(band_dict[band_name]['Range Chirp Rate'])
            sample_rate = band_dict[band_name]['Sampling Rate']
            ref_dechirp_time = band_dict[band_name].get('Reference Dechirping Time', 0)  # TODO: is this right?
            win_length = band_dict[band_name]['Echo Sampling Window Length']
            rcv_fm_rate = 0 if numpy.isnan(ref_dechirp_time) else chirp_rate
            band_width = chirp_length*chirp_rate
            fr_min = center_frequency - 0.5*band_width
            fr_max = center_frequency + 0.5*band_width
            sicd.RadarCollection.TxFrequency = (fr_min, fr_max)
            sicd.RadarCollection.Waveform = [
                WaveformParametersType(index=0,
                                       TxPulseLength=chirp_length,
                                       TxRFBandwidth=band_width,
                                       TxFreqStart=fr_min,
                                       TxFMRate=chirp_rate,
                                       ADCSampleRate=sample_rate,
                                       RcvFMRate=rcv_fm_rate,
                                       RcvWindowLength=win_length/sample_rate), ]
            sicd.ImageFormation.RcvChanProc.ChanIndices = [ind+1, ]
            sicd.ImageFormation.TxFrequencyProc = (fr_min, fr_max)

        def update_rma_and_grid(sicd, band_name):
            # type: (SICDType, str) -> None
            rg_scp_time = rg_first_time + (ss_rg_s*sicd.ImageData.SCPPixel.Row)
            az_scp_time = az_first_time + (use_sign2*ss_az_s*sicd.ImageData.SCPPixel.Col)
            r_ca_scp = rg_scp_time*speed_of_light/2
            sicd.RMA.INCA.R_CA_SCP = r_ca_scp
            # compute DRateSFPoly
            scp_ca_time = az_scp_time + ref_time_offset
            vel_poly = sicd.Position.ARPPoly.derivative(der_order=1, return_poly=True)
            vel_ca_vec = vel_poly(scp_ca_time)
            vel_ca_sq = numpy.sum(vel_ca_vec*vel_ca_vec)
            vel_ca = numpy.sqrt(vel_ca_sq)
            r_ca = numpy.array([r_ca_scp, 1.], dtype=numpy.float64)
            dop_rate_poly_rg_shifted = dop_rate_poly_rg.shift(
                rg_ref_time-rg_scp_time, alpha=ss_rg_s/row_ss, return_poly=False)
            drate_sf_poly = -(polynomial.polymul(dop_rate_poly_rg_shifted, r_ca) *
                              speed_of_light/(2*center_frequency*vel_ca_sq))
            # update grid.row
            sicd.Grid.Row.SS = row_ss
            sicd.Grid.Row.ImpRespBW = row_bw
            sicd.Grid.Row.DeltaK1 = -0.5 * row_bw
            sicd.Grid.Row.DeltaK2 = 0.5 * row_bw
            # update grid.col
            col_ss = abs(vel_ca*ss_az_s*drate_sf_poly[0])
            sicd.Grid.Col.SS = col_ss
            if self.mission_id == 'CSK':
                col_bw = min(band_dict[band_name]['Azimuth Focusing Transition Bandwidth']*ss_az_s, 1) / col_ss
            elif self.mission_id in ['CSG', 'KMPS']:
                col_bw = min(band_dict[band_name]['Azimuth Focusing Bandwidth']*ss_az_s, 1) / col_ss
            else:
                raise ValueError('Got unhandled mission_id {}'.format(self.mission_id))
            sicd.Grid.Col.ImpRespBW = col_bw
            # update inca
            sicd.RMA.INCA.DRateSFPoly = Poly2DType(Coefs=numpy.reshape(drate_sf_poly, (-1, 1)))
            sicd.RMA.INCA.TimeCAPoly = Poly1DType(Coefs=[scp_ca_time, use_sign2*ss_az_s/col_ss])
            # compute DopCentroidPoly & DeltaKCOAPoly
            dop_centroid_poly = numpy.zeros((dop_poly_rg.order1+1, dop_poly_az.order1+1), dtype=numpy.float64)
            dop_centroid_poly[0, 0] = dop_poly_rg(rg_scp_time-rg_ref_time) + \
                dop_poly_az(az_scp_time-az_ref_time) - \
                0.5*(dop_poly_rg[0] + dop_poly_az[0])
            dop_poly_rg_shifted = dop_poly_rg.shift(rg_ref_time-rg_scp_time, alpha=ss_rg_s/row_ss)
            dop_poly_az_shifted = dop_poly_az.shift(az_ref_time-az_scp_time, alpha=ss_az_s/col_ss)
            dop_centroid_poly[1:, 0] = dop_poly_rg_shifted[1:]
            dop_centroid_poly[0, 1:] = dop_poly_az_shifted[1:]
            sicd.RMA.INCA.DopCentroidPoly = Poly2DType(Coefs=dop_centroid_poly)
            sicd.RMA.INCA.DopCentroidCOA = True
            sicd.Grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=use_sign*dop_centroid_poly*ss_az_s/col_ss)
            # fit TimeCOAPoly
            sicd.Grid.TimeCOAPoly = fit_time_coa_polynomial(
                sicd.RMA.INCA, sicd.ImageData, sicd.Grid, dop_rate_poly_rg_shifted, poly_order=2)

            if csk_addin is not None:
                csk_addin.check_sicd(sicd, self.mission_id, h5_dict)

        def update_radiometric(sicd, band_name):
            # type: (SICDType, str) -> None
            if self.mission_id in ['KMPS', 'CSG']:
                # TODO: skipping for now - strange results for flag == 77. Awaiting gidance - see Wade.
                return
            if h5_dict['Range Spreading Loss Compensation Geometry'] != 'NONE':
                slant_range = h5_dict['Reference Slant Range']
                exp = h5_dict['Reference Slant Range Exponent']
                sf = slant_range**(2*exp)
                rsf = h5_dict['Rescaling Factor']
                sf /= rsf * rsf
                if h5_dict.get('Calibration Constant Compensation Flag', None) == 0:
                    cal = band_dict[band_name]['Calibration Constant']
                    sf /= cal
                sicd.Radiometric = RadiometricType(BetaZeroSFPoly=Poly2DType(Coefs=[[sf, ], ]))

        def update_geodata(sicd):  # type: (SICDType) -> None
            scp_pixel = [sicd.ImageData.SCPPixel.Row, sicd.ImageData.SCPPixel.Col]
            ecf = sicd.project_image_to_ground(scp_pixel, projection_type='HAE')
            sicd.update_scp(ecf, coord_system='ECF')

            SCP = sicd.GeoData.SCP.ECF.get_array(dtype='float64')
            scp_time = sicd.RMA.INCA.TimeCAPoly[0]
            ca_pos = sicd.Position.ARPPoly(scp_time)
            RG = SCP - ca_pos
            sicd.RMA.INCA.R_CA_SCP = numpy.linalg.norm(RG)

        out = {}
        center_frequency = h5_dict['Radar Frequency']
        # relative times in csk are wrt some reference time - for sicd they should be relative to start time
        collect_start = parse_timestring(h5_dict['Scene Sensing Start UTC'], precision='ns')
        ref_time = parse_timestring(h5_dict['Reference UTC'], precision='ns')
        ref_time_offset = get_seconds(ref_time, collect_start, precision='ns')

        for bd_name in band_dict:
            polarization = self._parse_pol(self._get_polarization(h5_dict, band_dict, bd_name))
            az_ref_time, rg_ref_time, t_dop_poly_az, t_dop_poly_rg, t_dop_rate_poly_rg = \
                self._get_dop_poly_details(h5_dict, band_dict, bd_name)
            dop_poly_az = Poly1DType(Coefs=t_dop_poly_az)
            dop_poly_rg = Poly1DType(Coefs=t_dop_poly_rg)

            t_sicd = base_sicd.copy()
            t_sicd.ImageFormation.TxRcvPolarizationProc = polarization

            row_bw = band_dict[bd_name]['Range Focusing Bandwidth']*2/speed_of_light
            row_ss = band_dict[bd_name]['Column Spacing']
            rg_first_time, ss_rg_s, az_first_time, ss_az_s, use_sign2 = update_image_data(t_sicd, bd_name)
            use_sign, dop_rate_poly_rg = check_switch_state()
            update_timeline(t_sicd, bd_name)
            update_radar_collection(t_sicd, bd_name)
            update_rma_and_grid(t_sicd, bd_name)
            update_radiometric(t_sicd, bd_name)

            update_scp_prelim(t_sicd, bd_name)  # set preliminary value for SCP (required for projection)
            update_geodata(t_sicd)
            t_sicd.derive()
            # t_sicd.populate_rniirs(override=False)
            out[bd_name] = t_sicd
        return out

    @staticmethod
    def _get_symmetry(h5_dict):
        line_order = h5_dict['Lines Order'].upper()
        look_side = h5_dict['Look Side'].upper()
        symm_0 = ((line_order == 'EARLY-LATE') != (look_side == 'RIGHT'))

        column_order = h5_dict['Columns Order'].upper()
        symm_1 = column_order != 'NEAR-FAR'

        return symm_0, symm_1, True

    def get_sicd_collection(self):
        # type: () -> Tuple[Dict[str, SICDType], Dict[str, str], Tuple[bool, bool, bool]]
        """
        Get the sicd collection for the bands.

        Returns
        -------
        Tuple[Dict[str, SICDType], Dict[str, str], Tuple[bool, bool, bool]]
            the first entry is a dictionary of the form {band_name: sicd}
            the second entry is of the form {band_name: shape}
            the third entry is the symmetry tuple
        """

        h5_dict, band_dict, shape_dict, dtype_dict = self._get_hdf_dicts()
        base_sicd = self._get_base_sicd(h5_dict, band_dict)
        return self._get_band_specific_sicds(base_sicd, h5_dict, band_dict, shape_dict, dtype_dict), \
            shape_dict, self._get_symmetry(h5_dict)


################
# The CSK reader


class CSKReader(BaseReader, SICDTypeReader):
    """
    Gets a reader type object for Cosmo Skymed files
    """

    __slots__ = ('_csk_details', )

    def __init__(self, csk_details):
        """

        Parameters
        ----------
        csk_details : str|CSKDetails
            file name or CSKDetails object
        """

        if isinstance(csk_details, string_types):
            csk_details = CSKDetails(csk_details)
        if not isinstance(csk_details, CSKDetails):
            raise TypeError('The input argument for a CSKReader must be a '
                            'filename or CSKDetails object')
        self._csk_details = csk_details
        sicd_data, shape_dict, symmetry = csk_details.get_sicd_collection()
        chippers = []
        sicds = []
        for band_name in sicd_data:
            if self._csk_details.mission_id in ['CSK', 'KMPS']:
                the_band = '{}/SBI'.format(band_name)
            elif self._csk_details.mission_id == 'CSG':
                the_band = '{}/IMG'.format(band_name)
            else:
                raise ValueError('Unhandled mission id {}'.format(self._csk_details.mission_id))

            sicds.append(sicd_data[band_name])
            chippers.append(H5Chipper(csk_details.file_name, the_band, shape_dict[band_name], symmetry))

        SICDTypeReader.__init__(self, tuple(sicds))
        BaseReader.__init__(self, tuple(chippers), reader_type="SICD")

    @property
    def csk_details(self):
        # type: () -> CSKDetails
        """
        CSKDetails: The details object.
        """

        return self._csk_details

    @property
    def file_name(self):
        return self.csk_details.file_name
