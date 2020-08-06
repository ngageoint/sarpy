# -*- coding: utf-8 -*-
"""
Functionality for reading ICEye data into a SICD model.
"""


import logging
import os
from collections import OrderedDict
from typing import Tuple, Dict
import warnings

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

try:
    import h5py
except ImportError:
    h5py = None

# noinspection PyProtectedMember
from .nisar import _stringify
from sarpy.compliance import string_types, int_func, bytes_to_string
from sarpy.io.complex.sicd_elements.blocks import Poly2DType, Poly1DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    TxFrequencyType, ChanParametersType, TxStepType, WaveformParametersType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, TxFrequencyProcType, RcvChanProcType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from sarpy.geometry import point_projection
from sarpy.io.general.base import BaseReader, BaseChipper
from sarpy.io.general.utils import get_seconds, parse_timestring
from sarpy.io.complex.utils import fit_position_xvalidation, two_dim_poly_fit


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"



########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a ICEye file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    CSKReader|None
        `CSKReader` instance if Cosmo Skymed file, `None` otherwise
    """

    if h5py is None:
        return None

    try:
        iceye_details = ICEyeDetails(file_name)
        print('File {} is determined to be a Cosmo Skymed file.'.format(file_name))
        return ICEyeReader(iceye_details)
    except (ImportError, IOError):
        return None


class ICEyeDetails(object):
    """
    Parses and converts the ICEye metadata.
    """
    __slots__ = ('_file_name', )

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        if h5py is None:
            raise ImportError("Can't read ICEye files, because the h5py dependency is missing.")

        if not os.path.isfile(file_name):
            raise IOError('Path {} is not a file'.format(file_name))

        with h5py.File(file_name, 'r') as hf:
            if 's_i' not in hf or 's_q' not in hf:
                raise IOError(
                    'The hdf file does not have the real (s_q) or imaginary dataset (s_i).')
            if 'satellite_name' not in hf:
                raise IOError(
                    'The hdf file does not have the satellite_name dataset.')
            if 'product_name' not in hf:
                raise IOError(
                    'The hdf file does not have the product_name dataset.')
        self._file_name = file_name

    @property
    def file_name(self):
        """
        str: the file name
        """

        return self._file_name

    def get_sicd(self):
        """
        Gets the SICD structure.

        Returns
        -------
        SICDType
        """

        def get_collection_info():
            # type: () -> CollectionInfoType
            return CollectionInfoType(
                CollectorName=_stringify(hf['satellite_name'][()]),
                CoreName=_stringify(hf['product_name'][()]),
                CollectType='MONOSTATIC',
                Classification='UNCLASSIFIED',
                RadarMode=RadarModeType(
                    ModeType=_stringify(hf['acquisition_mode'][()]).upper(),
                    ModeID=_stringify(hf['product_type'][()])))

        def get_image_creation():
            # type: () -> ImageCreationType
            from sarpy.__about__ import __version__
            return ImageCreationType(
                Application='ICEYE_P_{}'.format(hf['processor_version'][()]),
                DateTime=numpy.datetime64(_stringify(hf['processing_time'][()]), 'us'),
                Site='Unknown',
                Profile='sarpy {}'.format(__version__))

        def get_image_data():
            # type: () -> ImageDataType

            samp_prec = _stringify(hf['sample_precision'][()])
            if samp_prec.upper() == 'INT16':
                pixel_type = 'RE16I_IM16I'
            elif samp_prec.upper() == 'FLOAT32':
                pixel_type = 'RE32F_IM32F'
            else:
                raise ValueError('Got unhandled sample precision {}'.format(samp_prec))

            num_rows = int_func(hf['number_of_range_samples'][()])
            num_cols = int_func(hf['number_of_azimuth_samples'][()])
            scp_row = int_func(coord_center[0]) - 1
            scp_col = int_func(coord_center[1]) - 1
            if 0 < scp_col < num_rows-1:
                if look_side == 'left':
                    scp_col = num_cols - scp_col - 1
            else:
                # early ICEye processing bug led to nonsensical SCP
                scp_col = int_func(num_rows/2.0)

            return ImageDataType(
                PixelType=pixel_type,
                NumRows=num_rows,
                NumCols=num_cols,
                FirstRow=0,
                FirstCol=0,
                FullImage=(num_rows, num_cols),
                SCPPixel=(scp_row, scp_col))

        def get_geo_data():
            # type: () -> GeoDataType
            # NB: the remainder will be derived.
            return GeoDataType(
                SCP=SCPType(
                    LLH=[coord_center[2], coord_center[3], avg_scene_height]))

        def get_timeline():
            # type: () -> TimelineType

            acq_prf = hf['acquisition_prf'][()]
            return TimelineType(
                CollectStart=start_time,
                CollectDuration=duration,
                IPP=[IPPSetType(index=0, TStart=0, TEnd=duration,
                                IPPStart=0, IPPEnd=int_func(round(acq_prf*duration)),
                                IPPPoly=[0, acq_prf]), ])

        def get_position():
            # type: () -> PositionType
            times_str = hf['state_vector_utc'][:]
            times = numpy.zeros((times_str.shape[0], ), dtype='float64')
            positions = numpy.zeros((times.size, 3), dtype='float64')
            velocities = numpy.zeros((times.size, 3), dtype='float64')
            for i, entry in times_str:
                dt_time = numpy.datetime64(_stringify(entry), 'us')
                times[i] = get_seconds(dt_time, start_time, 'us')
            positions[:, 0], positions[:, 1], positions[:, 2] = hf['posX'][:], hf['posY'][:], hf['posZ'][:]
            velocities[:, 0], velocities[:, 1], velocities[:, 2] = hf['velX'][:], hf['velY'][:], hf['velZ'][:]
            # calculate the position data using cross validation
            P_x, P_y, P_z = fit_position_xvalidation(times, positions, velocities, max_degree=6)
            return PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

        def get_radar_collection():
            # type : () -> RadarCollection
            return RadarCollectionType(
                TxPolarization=tx_pol,
                TxFrequency=TxFrequencyType(Min=min_freq,
                                            Max=max_freq),
                Waveform=[WaveformParametersType(TxFreqStart=min_freq,
                                                 TxRFBandwidth=tx_bandwidth,
                                                 TxPulseLength=hf['chirp_duration'][()],
                                                 ADCSampleRate=hf['range_sampling_rate'][()],
                                                 RcvDemodType='CHIRP',
                                                 RcvFMRate=0,
                                                 index=1)],
                RcvChannels=[ChanParametersType(TxRcvPolarization=polarization,
                                                index=1)])

        def get_image_formation():
            # type: () -> ImageFormationType
            return ImageFormationType(
                TxRcvPolarizationProc=polarization,
                ImageFormAlgo='RMA',
                TStartProc=0,
                TEndProc=duration,
                TxFrequencyProc=TxFrequencyProcType(MinProc=min_freq, MaxProc=max_freq),
                STBeamComp='NO',
                ImageBeamComp='SV',
                AzAutofocus='NO',
                RgAutofocus='NO',
                RcvChanProc=RcvChanProcType(NumChanProc=1, PRFScaleFactor=1, ChanIndices=[1, ]),)

        def get_radiometric():
            # type: () -> RadiometricType
            return RadiometricType(BetaZeroSFPoly=[[float(hf['calibration_factor'][()]), ],])

        def do_doppler_calcs():
            vel_scp = position.ARPPoly.derivative_eval(zd_time_scp, der_order=1)
            vm_ca_sq = numpy.sum(vel_scp*vel_scp)
            r_ca_coeffs = numpy.array([r_ca_scp, 1])
            dop_rate_coeffs = hf['doppler_rate_coeffs'][:]
            # Prior to ICEYE 1.14 processor, absolute value of Doppler rate was
            # provided, not true Doppler rate. Doppler rate should always be negative
            if dop_rate_coeffs[0] > 0:
                dop_rate_coeffs *= -1
            dop_rate_poly = Poly1DType(Coefs=dop_rate_coeffs)
            # now adjust to create
            dop_rate_poly_rg_scaled = Poly1DType.shift(
                t_0=zd_ref_time-2*r_ca_scp/speed_of_light,
                alpha=2/speed_of_light, return_poly=False)  # TODO: conventions here?
            t_drate_sf_poly_coefs = -numpy.convolve(dop_rate_poly_rg_scaled, r_ca_coeffs)*\
                                  speed_of_light/(2*center_freq*vm_ca_sq)
            t_col_ss = float(t_drate_sf_poly_coefs[0]*vm_ca_sq*abs(ss_zd_s))
            t_col_imp_res_bw = dop_bw/float(t_drate_sf_poly_coefs[0]*vm_ca_sq)
            t_time_ca_poly_coeffs = [zd_time_scp, ss_zd_s/t_col_ss]
            # stopped at line 256
            return t_drate_sf_poly_coefs, t_col_ss, t_col_imp_res_bw, t_time_ca_poly_coeffs


        def get_rma():
            # type: () -> RMAType

            inca = INCAType(
                R_CA_SCP=r_ca_scp,
                FreqZero=center_freq,
                DRateSFPoly=Poly2DType(Coefs=numpy.reshape(drate_sf_poly_coefs, (-1, 1))),
                TimeCAPoly=Poly1DType(Coefs=time_ca_poly_coeffs))

            return RMAType(
                RMAlgoType='OMEGA_K',

                INCA=inca)


        def get_grid():
            # type: () -> GridType

            row_win = _stringify(hf['window_function_range'][()])
            if row_win == 'NONE':
                row_win = 'UNIFORM'
            row = DirParamType(
                SS=row_ss,
                Sgn=-1,
                KCtr=2*center_freq/speed_of_light,
                ImpRespBW=2*tx_bandwidth/speed_of_light,
                DeltaKCOAPoly=Poly2DType(Coefs=[[0,]]),
                WgtType=WgtTypeType(WindowName=row_win))

            col_win = _stringify(hf['window_function_azimuth'][()])
            if col_win == 'NONE':
                col_win = 'UNIFORM'
            col = DirParamType(
                SS=col_ss,
                Sgn=-1,
                KCtr=0,
                ImpRespBW=col_imp_res_bw,
                WgtType=WgtTypeType(WindowName=col_win))

            return GridType(
                Type='RGZERO',
                ImagePlane='SLANT',

                Row=row,
                Col=col)

        with h5py.File(self._file_name, 'r') as hf:
            # some common use variables
            look_side = _stringify(hf['look_side'][()])
            coord_center = hf['coord_center'][:]
            avg_scene_height = float(hf['avg_scene_height'][()])
            start_time = numpy.datetime64(_stringify(hf['acquisition_start_utc'][()]), 'us')
            end_time = numpy.datetime64(_stringify(hf['acquisition_end_utc'][()]), 'us')
            duration = get_seconds(end_time, start_time, 'us')

            center_freq = float(hf['carrier_frequency'][()])
            tx_bandwidth = float(hf['chirp_bandwidth'][()])
            min_freq = center_freq-0.5*tx_bandwidth
            max_freq = center_freq+0.5*tx_bandwidth

            pol_temp = _stringify(hf['polarization'][()])
            tx_pol = pol_temp[0]
            rcv_pol = pol_temp[1]
            polarization = tx_pol + ':' + rcv_pol

            near_range = float(hf['first_pixel_time'][()])
            range_sampling_rate = float(hf['range_sampling_rate'][()])
            row_ss = speed_of_light/(2*range_sampling_rate)
            ss_zd_s = float(hf['azimuth_time_interval'][()])
            zero_doppler_start = numpy.datetime64(_stringify(hf['zerodoppler_start_utc'][()]), 'us')
            zero_doppler_end = numpy.datetime64(_stringify(hf['zerodoppler_end_utc'][()]), 'us')
            if look_side == 'left':
                ss_zd_s *= -1
                zero_doppler_left = zero_doppler_end
            else:
                zero_doppler_left = zero_doppler_start
            dop_bw = hf['total_processed_bandwidth_azimuth'][()]


            # define the sicd elements
            collect_info = get_collection_info()
            image_creation = get_image_creation()
            image_data = get_image_data()
            geo_data = get_geo_data()
            timeline = get_timeline()
            position = get_position()
            radar_collection = get_radar_collection()
            image_formation = get_image_formation()
            radiometric = get_radiometric()
            # some zero doppler parameters
            zd_time_scp = get_seconds(zero_doppler_left, start_time, 'us') + image_data.SCPPixel.Col*ss_zd_s
            zd_ref_time = near_range + float(hf['number_of_range_samples'][()])/range_sampling_rate
            r_ca_scp = near_range + image_data.SCPPixel.Row*row_ss
            drate_sf_poly_coefs, col_ss, col_imp_res_bw, time_ca_poly_coeffs = do_doppler_calcs()
            rma = get_rma()

            grid = get_grid()

            sicd = SICDType(
                CollectionInfo=collect_info,
                ImageCreation=image_creation,
                ImageData=image_data,
                GeoData=geo_data,
                Timeline=timeline,
                Position=position,
                RadarCollection=radar_collection,
                ImageFormation=image_formation,
                Radiometric=radiometric,
                RMA=rma,

                Grid=grid)
        return sicd




class ICEyeReader(BaseReader):
    """
    Gets a reader type object for Cosmo Skymed files
    """

    __slots__ = ('_iceye_details', )

    def __init__(self, iceye_details):
        """

        Parameters
        ----------
        iceye_details : str|ICEyeDetails
            file name or ICEyeDetails object
        """

        if isinstance(iceye_details, string_types):
            iceye_details = ICEyeDetails(iceye_details)
        if not isinstance(iceye_details, ICEyeDetails):
            raise TypeError('The input argument for a ICEyeReader must be a '
                            'filename or ICEyeDetails object')

        # TODO: finish this
