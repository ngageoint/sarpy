"""
Functionality for reading NISAR data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
from collections import OrderedDict
from typing import Tuple, Dict, Union, List, Sequence, Optional

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from sarpy.compliance import bytes_to_string
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.blocks import Poly2DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    ChanParametersType, TxStepType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, \
    RcvChanProcType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from sarpy.geometry import point_projection
from sarpy.io.complex.utils import fit_position_xvalidation, two_dim_poly_fit

from sarpy.io.general.base import BaseReader, SarpyIOError
from sarpy.io.general.data_segment import HDF5DatasetSegment
from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.utils import get_seconds, parse_timestring, is_file_like, is_hdf5, h5py

if h5py is None:
    h5pyFile = None
    h5pyGroup = None
    h5pyDataset = None
else:
    from h5py import File as h5pyFile, Group as h5pyGroup, Dataset as h5pyDataset

logger = logging.getLogger(__name__)


###########
# parser and interpreter for hdf5 attributes

def _stringify(val: Union[str, bytes]) -> str:
    """
    Decode the value as necessary, for hdf5 string support issues.

    Parameters
    ----------
    val : str|bytes

    Returns
    -------
    str
    """

    return bytes_to_string(val).strip()


def _get_ref_time(str_in: Union[str, bytes]) -> numpy.datetime64:
    """
    Extract the given reference time.

    Parameters
    ----------
    str_in : str|bytes

    Returns
    -------
    numpy.datetime64
    """

    str_in = bytes_to_string(str_in)
    prefix = 'seconds since '
    if not str_in.startswith(prefix):
        raise ValueError('Got unexpected reference time string - {}'.format(str_in))
    return parse_timestring(str_in[len(prefix):], precision='ns')


def _get_string_list(array: Sequence[bytes]) -> List[str]:
    return [bytes_to_string(el) for el in array]


class NISARDetails(object):
    """
    Parses and converts the Cosmo Skymed metadata
    """

    __slots__ = ('_file_name', )

    def __init__(self, file_name: str):
        """

        Parameters
        ----------
        file_name : str
        """

        if h5py is None:
            raise ImportError("Can't read NISAR files, because the h5py dependency is missing.")

        if not os.path.isfile(file_name):
            raise SarpyIOError('Path {} is not a file'.format(file_name))

        with h5py.File(file_name, 'r') as hf:
            # noinspection PyBroadException
            try:
                # noinspection PyUnusedLocal
                gp = hf['/science/LSAR/SLC']
            except Exception as e:
                raise SarpyIOError('Got an error when reading required path /science/LSAR/SLC\n\t{}'.format(e))

        self._file_name = file_name

    @property
    def file_name(self) -> str:
        """
        str: the file name
        """

        return self._file_name

    @staticmethod
    def _get_frequency_list(hf: h5pyFile) -> List[str]:
        """
        Gets the list of frequencies.

        Parameters
        ----------
        hf : h5py.File

        Returns
        -------
        numpy.ndarray
        """

        return _get_string_list(hf['/science/LSAR/identification/listOfFrequencies'][:])

    @staticmethod
    def _get_collection_times(hf: h5pyFile) -> Tuple[numpy.datetime64, numpy.datetime64, float]:
        """
        Gets the collection start and end times, and inferred duration.

        Parameters
        ----------
        hf : h5py.File
            The h5py File object.

        Returns
        -------
        start_time : numpy.datetime64
        end_time : numpy.datetime64
        duration : float
        """

        start_time = parse_timestring(
            _stringify(hf['/science/LSAR/identification/zeroDopplerStartTime'][()]),
            precision='ns')
        end_time = parse_timestring(
            _stringify(hf['/science/LSAR/identification/zeroDopplerEndTime'][()]),
            precision='ns')
        duration = get_seconds(end_time, start_time, precision='ns')
        return start_time, end_time, duration

    @staticmethod
    def _get_zero_doppler_data(
            hf: h5pyFile,
            base_sicd: SICDType) -> Tuple[numpy.ndarray, float, numpy.ndarray, numpy.ndarray]:
        """
        Gets zero-doppler parameters.

        Parameters
        ----------
        hf : h5py.File
        base_sicd : SICDType

        Returns
        -------
        azimuth_zero_doppler_times : numpy.ndarray
        azimuth_zero_doppler_spacing : float
        grid_range_array : numpy.ndarray
        range_zero_doppler_times : numpy.ndarray
        """

        gp = hf['/science/LSAR/SLC/swaths']
        ds = gp['zeroDopplerTime']
        ref_time = _get_ref_time(ds.attrs['units'])
        zd_time = ds[:] + get_seconds(ref_time, base_sicd.Timeline.CollectStart, precision='ns')
        ss_az_s = gp['zeroDopplerTimeSpacing'][()]

        if base_sicd.SCPCOA.SideOfTrack == 'L':
            zd_time = zd_time[::-1]
            ss_az_s *= -1

        gp = hf['/science/LSAR/SLC/metadata/processingInformation/parameters']
        grid_r = gp['slantRange'][:]
        ds = gp['zeroDopplerTime']
        ref_time = _get_ref_time(ds.attrs['units'])
        grid_zd_time = ds[:] + get_seconds(ref_time, base_sicd.Timeline.CollectStart, precision='ns')
        return zd_time, ss_az_s, grid_r, grid_zd_time

    def _get_base_sicd(self, hf: h5pyFile) -> SICDType:
        """
        Defines the base SICD object, to be refined with further details.

        Returns
        -------
        SICDType
        """

        def get_collection_info() -> CollectionInfoType:
            gp = hf['/science/LSAR/identification']
            return CollectionInfoType(
                CollectorName=_stringify(hf.attrs['mission_name']),
                CoreName='{0:07d}_{1:s}'.format(gp['absoluteOrbitNumber'][()],
                                                _stringify(gp['trackNumber'][()])),
                CollectType='MONOSTATIC',
                Classification='UNCLASSIFIED',
                RadarMode=RadarModeType(ModeType='STRIPMAP'))

        def get_image_creation() -> ImageCreationType:
            application = 'ISCE'
            # noinspection PyBroadException
            try:
                application = '{} {}'.format(
                    application,
                    _stringify(hf['/science/LSAR/SLC/metadata/processingInformation/algorithms/ISCEVersion'][()]))
            except Exception as e:
                logger.info('Failed extracting the application details with error\n\t{}'.format(e))
                pass

            from sarpy.__about__ import __version__
            # TODO: DateTime?
            return ImageCreationType(
                Application=application,
                Site='Unknown',
                Profile='sarpy {}'.format(__version__))

        def get_geo_data() -> GeoDataType:
            # seeds a rough SCP for projection usage
            poly_str = _stringify(hf['/science/LSAR/identification/boundingPolygon'][()])
            beg_str = 'POLYGON (('
            if not poly_str.startswith(beg_str):
                raise ValueError('Unexpected polygon string {}'.format(poly_str))
            parts = poly_str[len(beg_str):-2].strip().split(',')
            if len(parts) != 5:
                raise ValueError('Unexpected polygon string parts {}'.format(parts))
            lats_lons = numpy.zeros((4, 2), dtype=numpy.float64)
            for i, part in enumerate(parts[:-1]):
                spart = part.strip().split()
                if len(spart) != 2:
                    raise ValueError('Unexpected polygon string parts {}'.format(parts))
                lats_lons[i, :] = float(spart[1]), float(spart[0])

            llh = numpy.zeros((3, ), dtype=numpy.float64)
            llh[0:2] = numpy.mean(lats_lons, axis=0)
            llh[2] = numpy.mean(
                hf['/science/LSAR/SLC/metadata/processingInformation/parameters/referenceTerrainHeight'][:])
            return GeoDataType(SCP=SCPType(LLH=llh))

        def get_grid() -> GridType:

            # TODO: Future Change Required - JPL states that uniform weighting in data simulated
            #  from UAVSAR is a placeholder, not an accurate description of the data.
            #  At this point, it is not clear what the final weighting description for NISAR
            #  will be.

            gp = hf['/science/LSAR/SLC/metadata/processingInformation/parameters']
            row_wgt = gp['rangeChirpWeighting'][:]
            win_name = 'UNIFORM' if numpy.all(row_wgt == row_wgt[0]) else 'UNKNOWN'
            row = DirParamType(
                Sgn=-1,
                DeltaKCOAPoly=[[0, ], ],
                WgtFunct=numpy.cast[numpy.float64](row_wgt),
                WgtType=WgtTypeType(WindowName=win_name))

            col_wgt = gp['azimuthChirpWeighting'][:]
            win_name = 'UNIFORM' if numpy.all(col_wgt == col_wgt[0]) else 'UNKNOWN'
            col = DirParamType(
                Sgn=-1,
                KCtr=0,
                WgtFunct=numpy.cast[numpy.float64](col_wgt),
                WgtType=WgtTypeType(WindowName=win_name))

            return GridType(ImagePlane='SLANT', Type='RGZERO', Row=row, Col=col)

        def get_timeline() -> TimelineType:
            # NB: IPPEnd must be set, but will be replaced
            return TimelineType(
                CollectStart=collect_start,
                CollectDuration=duration,
                IPP=[IPPSetType(index=0, TStart=0, TEnd=duration, IPPStart=0, IPPEnd=0), ])

        def get_position() -> PositionType:
            gp = hf['/science/LSAR/SLC/metadata/orbit']
            ref_time = _get_ref_time(gp['time'].attrs['units'])
            T = gp['time'][:] + get_seconds(ref_time, collect_start, precision='ns')
            Pos = gp['position'][:]
            Vel = gp['velocity'][:]
            P_x, P_y, P_z = fit_position_xvalidation(T, Pos, Vel, max_degree=8)
            return PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

        def get_scpcoa() -> SCPCOAType:
            # remaining fields set later
            sot = _stringify(hf['/science/LSAR/identification/lookDirection'][()])[0].upper()
            return SCPCOAType(SideOfTrack=sot)

        def get_image_formation() -> ImageFormationType:
            return ImageFormationType(
                ImageFormAlgo='RMA',
                TStartProc=0,
                TEndProc=duration,
                STBeamComp='NO',
                ImageBeamComp='SV',
                AzAutofocus='NO',
                RgAutofocus='NO',
                RcvChanProc=RcvChanProcType(NumChanProc=1, PRFScaleFactor=1))

        def get_rma() -> RMAType:
            return RMAType(RMAlgoType='OMEGA_K', INCA=INCAType(DopCentroidCOA=True))

        collect_start, collect_end, duration = self._get_collection_times(hf)
        collection_info = get_collection_info()
        image_creation = get_image_creation()
        geo_data = get_geo_data()
        grid = get_grid()
        timeline = get_timeline()
        position = get_position()
        scpcoa = get_scpcoa()
        image_formation = get_image_formation()
        rma = get_rma()

        return SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            GeoData=geo_data,
            Grid=grid,
            Timeline=timeline,
            Position=position,
            SCPCOA=scpcoa,
            ImageFormation=image_formation,
            RMA=rma)

    @staticmethod
    def _get_freq_specific_sicd(
            gp: h5pyGroup,
            base_sicd: SICDType) -> Tuple[SICDType, List[str], List[str], float]:
        """
        Gets the frequency specific sicd.

        Parameters
        ----------
        gp : h5py.Group
        base_sicd : SICDType

        Returns
        -------
        sicd: SICDType
        pol_names : numpy.ndarray
        pols : List[str]
        center_frequency : float
        """

        def update_grid() -> None:
            row_imp_resp_bw = 2*gp['processedRangeBandwidth'][()]/speed_of_light
            t_sicd.Grid.Row.SS = gp['slantRangeSpacing'][()]
            t_sicd.Grid.Row.ImpRespBW = row_imp_resp_bw
            t_sicd.Grid.Row.DeltaK1 = -0.5*row_imp_resp_bw
            t_sicd.Grid.Row.DeltaK2 = -t_sicd.Grid.Row.DeltaK1

        def update_timeline() -> None:
            prf = gp['nominalAcquisitionPRF'][()]
            t_sicd.Timeline.IPP[0].IPPEnd = prf*t_sicd.Timeline.CollectDuration
            t_sicd.Timeline.IPP[0].IPPPoly = [0, prf]

        def define_radar_collection() -> List[str]:
            tx_rcv_pol_t = []
            tx_pol = []
            for entry in pols:
                tx_rcv_pol_t.append('{}:{}'.format(entry[0], entry[1]))
                if entry[0] not in tx_pol:
                    tx_pol.append(entry[0])
            center_freq_t = gp['acquiredCenterFrequency'][()]
            bw = gp['acquiredRangeBandwidth'][()]
            tx_freq = (center_freq_t - 0.5*bw, center_freq_t + 0.5*bw)
            rcv_chans = [ChanParametersType(TxRcvPolarization=pol) for pol in tx_rcv_pol_t]
            if len(tx_pol) == 1:
                tx_sequence = None
                tx_pol = tx_pol[0]
            else:
                tx_sequence = [TxStepType(WFIndex=j+1, TxPolarization=pol) for j, pol in enumerate(tx_pol)]
                tx_pol = 'SEQUENCE'

            t_sicd.RadarCollection = RadarCollectionType(
                TxFrequency=tx_freq,
                RcvChannels=rcv_chans,
                TxPolarization=tx_pol,
                TxSequence=tx_sequence)
            return tx_rcv_pol_t

        def update_image_formation() -> float:
            center_freq_t = gp['processedCenterFrequency'][()]
            bw = gp['processedRangeBandwidth'][()]
            t_sicd.ImageFormation.TxFrequencyProc = (center_freq_t - 0.5*bw, center_freq_t + 0.5*bw)
            return center_freq_t

        pols = _get_string_list(gp['listOfPolarizations'][:])
        t_sicd = base_sicd.copy()

        update_grid()
        update_timeline()
        tx_rcv_pol = define_radar_collection()
        center_freq = update_image_formation()
        return t_sicd, pols, tx_rcv_pol, center_freq

    @staticmethod
    def _get_pol_specific_sicd(
            hf: h5pyFile,
            ds: h5pyDataset,
            base_sicd: SICDType,
            pol_name: str,
            freq_name: str,
            j: int,
            pol: str,
            r_ca_sampled: numpy.ndarray,
            zd_time: numpy.ndarray,
            grid_zd_time: numpy.ndarray,
            grid_r: numpy.ndarray,
            doprate_sampled: numpy.ndarray,
            dopcentroid_sampled: numpy.ndarray,
            center_freq: float,
            ss_az_s: float,
            dop_bw: float,
            beta0,
            gamma0,
            sigma0) -> Tuple[SICDType, Tuple[int, ...], numpy.dtype]:
        """
        Gets the frequency/polarization specific sicd.

        Parameters
        ----------
        hf : h5py.File
        ds : h5py.Dataset
        base_sicd : SICDType
        pol_name : str
        freq_name : str
        j : int
        pol : str
        r_ca_sampled : numpy.ndarray
        zd_time : numpy.ndarray
        grid_zd_time : numpy.ndarray
        grid_r : numpy.ndarray
        doprate_sampled : numpy.ndarray
        dopcentroid_sampled : numpy.ndarray
        center_freq : float
        ss_az_s : float
        dop_bw : float

        Returns
        -------
        sicd: SICDType
        shape : Tuple[int, ...]
        numpy.dtype
        """

        def define_image_data() -> None:
            if dtype.name in ('float32', 'complex64'):
                pixel_type = 'RE32F_IM32F'
            elif dtype.name == 'int16':
                pixel_type = 'RE16I_IM16I'
            else:
                raise ValueError('Got unhandled dtype {}'.format(dtype))
            t_sicd.ImageData = ImageDataType(
                PixelType=pixel_type,
                NumRows=shape[1],
                NumCols=shape[0],
                FirstRow=0,
                FirstCol=0,
                SCPPixel=[0.5*shape[0], 0.5*shape[1]],
                FullImage=[shape[1], shape[0]])

        def update_image_formation() -> None:
            t_sicd.ImageFormation.RcvChanProc.ChanIndices = [j, ]
            t_sicd.ImageFormation.TxRcvPolarizationProc = pol

        def update_inca_and_grid() -> Tuple[numpy.ndarray, numpy.ndarray]:
            t_sicd.RMA.INCA.R_CA_SCP = r_ca_sampled[t_sicd.ImageData.SCPPixel.Row]
            scp_ca_time = zd_time[t_sicd.ImageData.SCPPixel.Col]

            # compute DRateSFPoly
            # velocity at scp ca time
            vel_ca = t_sicd.Position.ARPPoly.derivative_eval(scp_ca_time, der_order=1)
            # squared magnitude
            vm_ca_sq = numpy.sum(vel_ca*vel_ca)
            # polynomial coefficient for function representing range as a function of range distance from SCP
            r_ca_poly = numpy.array([t_sicd.RMA.INCA.R_CA_SCP, 1], dtype=numpy.float64)
            # closest Doppler rate polynomial to SCP
            min_ind = numpy.argmin(numpy.absolute(grid_zd_time - scp_ca_time))
            # define range coordinate grid
            coords_rg_m = grid_r - t_sicd.RMA.INCA.R_CA_SCP
            # determine dop_rate_poly coordinates
            dop_rate_poly = polynomial.polyfit(coords_rg_m, -doprate_sampled[min_ind, :], 4)  # why fourth order?
            t_sicd.RMA.INCA.FreqZero = center_freq
            t_sicd.RMA.INCA.DRateSFPoly = Poly2DType(Coefs=numpy.reshape(
                -numpy.convolve(dop_rate_poly, r_ca_poly)*speed_of_light/(2*center_freq*vm_ca_sq), (-1, 1)))

            # update Grid.Col parameters
            t_sicd.Grid.Col.SS = numpy.sqrt(vm_ca_sq)*abs(ss_az_s)*t_sicd.RMA.INCA.DRateSFPoly.Coefs[0, 0]
            t_sicd.Grid.Col.ImpRespBW = min(abs(dop_bw*ss_az_s), 1)/t_sicd.Grid.Col.SS
            t_sicd.RMA.INCA.TimeCAPoly = [scp_ca_time, ss_az_s/t_sicd.Grid.Col.SS]

            # TimeCOAPoly/DopCentroidPoly/DeltaKCOAPoly
            coords_az_m = (grid_zd_time - scp_ca_time)*t_sicd.Grid.Col.SS/ss_az_s

            # cerate the 2d grids
            coords_rg_2d_t, coords_az_2d_t = numpy.meshgrid(coords_rg_m, coords_az_m, indexing='xy')

            coefs, residuals, rank, sing_values = two_dim_poly_fit(
                coords_rg_2d_t, coords_az_2d_t, dopcentroid_sampled,
                x_order=3, y_order=3, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
            logger.info(
                'The dop_centroid_poly fit details:\n\t'
                'root mean square residuals = {}\n\t'
                'rank = {}\n\t'
                'singular values = {}'.format(residuals, rank, sing_values))
            t_sicd.RMA.INCA.DopCentroidPoly = Poly2DType(Coefs=coefs)
            t_sicd.Grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=coefs*ss_az_s/t_sicd.Grid.Col.SS)

            timeca_sampled = numpy.outer(grid_zd_time, numpy.ones((grid_r.size, )))
            time_coa_sampled = timeca_sampled + (dopcentroid_sampled/doprate_sampled)
            coefs, residuals, rank, sing_values = two_dim_poly_fit(
                coords_rg_2d_t, coords_az_2d_t, time_coa_sampled,
                x_order=3, y_order=3, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
            logger.info(
                'The time_coa_poly fit details:\n\t'
                'root mean square residuals = {}\n\t'
                'rank = {}\n\t'
                'singular values = {}'.format(residuals, rank, sing_values))
            t_sicd.Grid.TimeCOAPoly = Poly2DType(Coefs=coefs)

            return coords_rg_2d_t, coords_az_2d_t

        def define_radiometric() -> None:
            def get_poly(ds: h5pyDataset, name: str) -> Optional[Poly2DType]:
                array = ds[:]
                fill = ds.attrs['_FillValue']
                boolc = (array != fill)

                if numpy.any(boolc):
                    array = array[boolc]
                    if numpy.any(array != array[0]):
                        coefs, residuals, rank, sing_values = two_dim_poly_fit(
                            coords_rg_2d[boolc], coords_az_2d[boolc], array,
                            x_order=3, y_order=3, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
                        logger.info(
                            'The {} fit details:\n\t'
                            'root mean square residuals = {}\n\t'
                            'rank = {}\n\t'
                            'singular values = {}'.format(name, residuals, rank, sing_values))
                    else:
                        # it's constant, so just use a constant polynomial
                        coefs = [[array[0], ], ]
                        logger.info('The {} values are constant'.format(name))
                    return Poly2DType(Coefs=coefs)
                else:
                    logger.warning('No non-trivial values for {} provided.'.format(name))
                    return None

            beta0_poly = get_poly(beta0, 'beta0')
            gamma0_poly = get_poly(gamma0, 'gamma0')
            sigma0_poly = get_poly(sigma0, 'sigma0')

            nesz = hf['/science/LSAR/SLC/metadata/calibrationInformation/frequency{}/{}/nes0'.format(freq_name,
                                                                                                     pol_name)][:]
            noise_samples = nesz - (10 * numpy.log10(sigma0_poly.Coefs[0, 0]))

            coefs, residuals, rank, sing_values = two_dim_poly_fit(
                coords_rg_2d, coords_az_2d, noise_samples,
                x_order=3, y_order=3, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
            logger.info(
                'The noise_poly fit details:\n\t'
                'root mean square residuals = {}\n\t'
                'rank = {}\n\t'
                'singular values = {}'.format(
                    residuals, rank, sing_values))
            t_sicd.Radiometric = RadiometricType(
                BetaZeroSFPoly=beta0_poly,
                GammaZeroSFPoly=gamma0_poly,
                SigmaZeroSFPoly=sigma0_poly,
                NoiseLevel=NoiseLevelType_(
                    NoiseLevelType='ABSOLUTE', NoisePoly=Poly2DType(Coefs=coefs)))

        def update_geodata() -> None:
            ecf = point_projection.image_to_ground(
                [t_sicd.ImageData.SCPPixel.Row, t_sicd.ImageData.SCPPixel.Col], t_sicd)
            t_sicd.GeoData.SCP = SCPType(ECF=ecf)  # LLH will be populated

        t_sicd = base_sicd.copy()
        shape = ds.shape
        dtype = ds.dtype

        define_image_data()
        update_image_formation()
        coords_rg_2d, coords_az_2d = update_inca_and_grid()
        define_radiometric()
        update_geodata()
        t_sicd.derive()
        t_sicd.populate_rniirs(override=False)
        return t_sicd, shape, dtype

    def get_sicd_collection(self) -> Tuple[
            Dict[str, SICDType],
            Dict[str, Tuple[Tuple[int, ...], numpy.dtype]],
            Optional[Tuple[int, ...]],
            Optional[Tuple[int, ...]]]:
        """
        Get the sicd collection for the bands.

        Returns
        -------
        sicd_dict : Dict[str, SICDType]
        shape_dict : Dict[str, Tuple[Tuple[int, ...], numpy.dtype]]
        reverse_axes : None|Tuple[int, ...]
        transpose_axes : None|Tuple[int, ...]
        """

        # TODO: check if the hdf already has the sicds defined, and fish them out if so.

        with h5py.File(self.file_name, 'r') as hf:
            # fetch the base shared sicd
            base_sicd = self._get_base_sicd(hf)

            # prepare our output workspace
            out_sicds = OrderedDict()
            shapes = OrderedDict()
            reverse_axes = (0, ) if base_sicd.SCPCOA.SideOfTrack == 'L' else None
            transpose_axes = (1, 0)

            # fetch the common use data for frequency issues
            collect_start, collect_end, duration = self._get_collection_times(hf)
            zd_time, ss_az_s, grid_r, grid_zd_time = self._get_zero_doppler_data(hf, base_sicd)

            gp = hf['/science/LSAR/SLC/metadata/calibrationInformation/geometry']
            beta0 = gp['beta0']
            gamma0 = gp['gamma0']
            sigma0 = gp['sigma0']

            # formulate the frequency specific sicd information
            freqs = self._get_frequency_list(hf)
            for i, freq in enumerate(freqs):
                gp_name = '/science/LSAR/SLC/swaths/frequency{}'.format(freq)
                gp = hf[gp_name]
                freq_sicd, pols, tx_rcv_pol, center_freq = self._get_freq_specific_sicd(gp, base_sicd)

                # formulate the frequency dependent doppler grid
                # TODO: Future Change Required - processedAzimuthBandwidth acknowledged
                #  by JPL to be wrong in simulated datasets.
                dop_bw = gp['processedAzimuthBandwidth'][()]
                gp2 = hf['/science/LSAR/SLC/metadata/processingInformation/parameters/frequency{}'.format(freq)]
                dopcentroid_sampled = gp2['dopplerCentroid'][:]
                doprate_sampled = gp2['azimuthFMRate'][:]
                r_ca_sampled = gp['slantRange'][:]
                # formulate the frequency/polarization specific sicd information
                for j, pol in enumerate(pols):
                    ds_name = '{}/{}'.format(gp_name, pol)
                    ds = gp[pol]
                    pol_sicd, shape, dtype = self._get_pol_specific_sicd(
                        hf, ds, freq_sicd, pol, freq, j, tx_rcv_pol[j],
                        r_ca_sampled, zd_time, grid_zd_time, grid_r,
                        doprate_sampled, dopcentroid_sampled, center_freq,
                        ss_az_s, dop_bw, beta0, gamma0, sigma0)
                    out_sicds[ds_name] = pol_sicd
                    shapes[ds_name] = (shape, dtype)
        return out_sicds, shapes, reverse_axes, transpose_axes


################
# The NISAR reader

class NISARReader(SICDTypeReader):
    """
    An NISAR SLC reader implementation.

    **Changed in version 1.3.0** for reading changes.
    """

    __slots__ = ('_nisar_details', )

    def __init__(self, nisar_details: Union[str, NISARDetails]):
        """

        Parameters
        ----------
        nisar_details : str|NISARDetails
            file name or NISARDetails object
        """

        if isinstance(nisar_details, str):
            nisar_details = NISARDetails(nisar_details)
        if not isinstance(nisar_details, NISARDetails):
            raise TypeError('The input argument for NISARReader must be a '
                            'filename or NISARDetails object')
        self._nisar_details = nisar_details
        sicd_data, shape_dict, reverse_axes, transpose_axes = nisar_details.get_sicd_collection()
        data_segments = []
        sicds = []
        for band_name in sicd_data:
            sicds.append(sicd_data[band_name])
            raw_shape, raw_dtype = shape_dict[band_name]
            formatted_shape = (raw_shape[1], raw_shape[0]) if transpose_axes is not None \
                else raw_shape[:2]
            if raw_dtype.name == 'complex64':
                formatted_dtype = raw_dtype
                format_function = None
            else:
                formatted_dtype = 'complex64'
                format_function = ComplexFormatFunction(raw_dtype=raw_dtype, order='IQ', band_dimension=-1)

            data_segments.append(
                HDF5DatasetSegment(
                    nisar_details.file_name, band_name,
                    formatted_dtype=formatted_dtype, formatted_shape=formatted_shape,
                    reverse_axes=reverse_axes, transpose_axes=transpose_axes,
                    format_function=format_function, close_file=True))

        SICDTypeReader.__init__(self, data_segments, sicds, close_segments=True)
        self._check_sizes()

    @property
    def nisar_details(self) -> NISARDetails:
        """
        NISARDetails: The nisar details object.
        """

        return self._nisar_details

    @property
    def file_name(self) -> str:
        return self.nisar_details.file_name


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name: str) -> Optional[NISARReader]:
    """
    Tests whether a given file_name corresponds to a NISAR file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str|BinaryIO
        the file_name to check

    Returns
    -------
    NISARReader|None
        `NISARReader` instance if NISAR file, `None` otherwise
    """

    if is_file_like(file_name):
        return None

    if not is_hdf5(file_name):
        return None

    if h5py is None:
        return None

    try:
        nisar_details = NISARDetails(file_name)
        logger.info('File {} is determined to be a NISAR file.'.format(file_name))
        return NISARReader(nisar_details)
    except (ImportError, SarpyIOError):
        return None
