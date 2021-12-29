"""
Functionality for reading NISAR data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
from collections import OrderedDict
from typing import Tuple, Dict

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from sarpy.compliance import bytes_to_string
from sarpy.io.complex.base import SICDTypeReader, H5Chipper, h5py, is_hdf5
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
from sarpy.io.general.base import BaseReader, SarpyIOError
from sarpy.io.general.utils import get_seconds, parse_timestring, is_file_like
from sarpy.io.complex.utils import fit_position_xvalidation, two_dim_poly_fit

logger = logging.getLogger(__name__)

########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
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


###########
# parser and interpreter for hdf5 attributes

def _stringify(val):
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


def _get_ref_time(str_in):
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


def _get_string_list(array):
    return [bytes_to_string(el) for el in array]


class NISARDetails(object):
    """
    Parses and converts the Cosmo Skymed metadata
    """

    __slots__ = ('_file_name', )

    def __init__(self, file_name):
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
    def file_name(self):
        """
        str: the file name
        """

        return self._file_name

    @staticmethod
    def _get_frequency_list(hf):
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
    def _get_collection_times(hf):
        """
        Gets the collection start and end times, and inferred duration.

        Parameters
        ----------
        hf : h5py.File
            The h5py File object.

        Returns
        -------
        (numpy.datetime64, numpy.datetime64, float)
            Start and end times and duration
        """

        start = parse_timestring(_stringify(hf['/science/LSAR/identification/zeroDopplerStartTime'][()]), precision='ns')
        end = parse_timestring(_stringify(hf['/science/LSAR/identification/zeroDopplerEndTime'][()]), precision='ns')
        duration = get_seconds(end, start, precision='ns')
        return start, end, duration

    @staticmethod
    def _get_zero_doppler_data(hf, base_sicd):
        """
        Gets zero-doppler parameters.

        Parameters
        ----------
        hf : h5py.File
        base_sicd : SICDType

        Returns
        -------
        (numpy.ndarray, float, numpy.ndarray, numpy.ndarray)
            The azimuth zero-doppler time array, azimuth zero-doppler time spacing,
            grid range array, range zero doppler time array.
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

    def _get_base_sicd(self, hf):
        """
        Defines the base SICD object, to be refined with further details.

        Returns
        -------
        SICDType
        """

        def get_collection_info():
            # type: () -> CollectionInfoType
            gp = hf['/science/LSAR/identification']
            return CollectionInfoType(
                CollectorName=_stringify(hf.attrs['mission_name']),
                CoreName='{0:07d}_{1:s}'.format(gp['absoluteOrbitNumber'][()],
                                                _stringify(gp['trackNumber'][()])),
                CollectType='MONOSTATIC',
                Classification='UNCLASSIFIED',
                RadarMode=RadarModeType(ModeType='STRIPMAP'))

        def get_image_creation():
            # type: () -> ImageCreationType
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

        def get_geo_data():
            # type: () -> GeoDataType
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
            llh[2] = numpy.mean(hf['/science/LSAR/SLC/metadata/processingInformation/parameters/referenceTerrainHeight'][:])
            return GeoDataType(SCP=SCPType(LLH=llh))

        def get_grid():
            # type: () -> GridType

            # TODO: Future Change Required - JPL states that uniform weighting in data simulated
            #  from UAVSAR is a placeholder, not an accurate description of the data.
            #  At this point, it is not clear what the final weighting description for NISAR
            #  will be.

            gp = hf['/science/LSAR/SLC/metadata/processingInformation/parameters']
            row_wgt = gp['rangeChirpWeighting'][:]
            win_name = 'UNIFORM' if numpy.all(row_wgt == row_wgt[0]) else 'UNKNOWN'
            row = DirParamType(
                Sgn=-1,
                DeltaKCOAPoly=[[0,]],
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

        def get_timeline():
            # type: () -> TimelineType
            # NB: IPPEnd must be set, but will be replaced
            return TimelineType(
                CollectStart=collect_start,
                CollectDuration=duration,
                IPP=[IPPSetType(index=0, TStart=0, TEnd=duration, IPPStart=0, IPPEnd=0), ])

        def get_position():
            # type: () -> PositionType
            gp = hf['/science/LSAR/SLC/metadata/orbit']
            ref_time = _get_ref_time(gp['time'].attrs['units'])
            T = gp['time'][:] + get_seconds(ref_time, collect_start, precision='ns')
            Pos = gp['position'][:]
            Vel = gp['velocity'][:]
            P_x, P_y, P_z = fit_position_xvalidation(T, Pos, Vel, max_degree=8)
            return PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

        def get_scpcoa():
            # type: () -> SCPCOAType
            # remaining fields set later
            sot = _stringify(hf['/science/LSAR/identification/lookDirection'][()])[0].upper()
            return SCPCOAType(SideOfTrack=sot)

        def get_image_formation():
            # type: () -> ImageFormationType
            return ImageFormationType(
                ImageFormAlgo='RMA',
                TStartProc=0,
                TEndProc=duration,
                STBeamComp='NO',
                ImageBeamComp='SV',
                AzAutofocus='NO',
                RgAutofocus='NO',
                RcvChanProc=RcvChanProcType(NumChanProc=1, PRFScaleFactor=1))

        def get_rma():
            # type: () -> RMAType
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
    def _get_freq_specific_sicd(gp, base_sicd):
        """
        Gets the frequency specific sicd.

        Parameters
        ----------
        gp : h5py.Group
        base_sicd : SICDType

        Returns
        -------
        (SICDType, numpy.ndarray, list, center_freq)
            frequency dependent sicd, array of polarization names, list of formatted polarizations for sicd,
            the processed center frequency
        """

        def update_grid():
            row_imp_resp_bw = 2*gp['processedRangeBandwidth'][()]/speed_of_light
            t_sicd.Grid.Row.SS = gp['slantRangeSpacing'][()]
            t_sicd.Grid.Row.ImpRespBW = row_imp_resp_bw
            t_sicd.Grid.Row.DeltaK1 = -0.5*row_imp_resp_bw
            t_sicd.Grid.Row.DeltaK2 = -t_sicd.Grid.Row.DeltaK1

        def update_timeline():
            prf = gp['nominalAcquisitionPRF'][()]
            t_sicd.Timeline.IPP[0].IPPEnd = prf*t_sicd.Timeline.CollectDuration
            t_sicd.Timeline.IPP[0].IPPPoly = [0, prf]

        def define_radar_collection():
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

        def update_image_formation():
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
    def _get_pol_specific_sicd(hf, ds, base_sicd, pol_name, freq_name, j, pol,
                               r_ca_sampled, zd_time, grid_zd_time, grid_r, doprate_sampled,
                               dopcentroid_sampled, center_freq, ss_az_s, dop_bw, beta0, gamma0, sigma0):
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
        (SICDType, Tuple[int])
        """

        def define_image_data():
            dtype = ds.dtype.name
            if dtype in ('float32', 'complex64'):
                pixel_type = 'RE32F_IM32F'
            elif dtype == 'int16':
                pixel_type = 'RE16I_IM16I'
            else:
                raise ValueError('Got unhandled dtype {}'.format(dtype))
            t_sicd.ImageData = ImageDataType(
                PixelType=pixel_type,
                NumRows=shape[0],
                NumCols=shape[1],
                FirstRow=0,
                FirstCol=0,
                SCPPixel=[0.5*shape[0], 0.5*shape[1]],
                FullImage=[shape[0], shape[1]])

        def update_image_formation():
            t_sicd.ImageFormation.RcvChanProc.ChanIndices = [j, ]
            t_sicd.ImageFormation.TxRcvPolarizationProc = pol

        def update_inca_and_grid():
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

            #TimeCOAPoly/DopCentroidPoly/DeltaKCOAPoly
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

        def define_radiometric():
            def get_poly(ds, name):
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

        def update_geodata():
            ecf = point_projection.image_to_ground(
                [t_sicd.ImageData.SCPPixel.Row, t_sicd.ImageData.SCPPixel.Col], t_sicd)
            t_sicd.GeoData.SCP = SCPType(ECF=ecf)  # LLH will be populated

        t_sicd = base_sicd.copy()
        shape = (int(ds.shape[1]), int(ds.shape[0]))

        define_image_data()
        update_image_formation()
        coords_rg_2d, coords_az_2d = update_inca_and_grid()
        define_radiometric()
        update_geodata()
        t_sicd.derive()
        t_sicd.populate_rniirs(override=False)
        return t_sicd, shape

    def get_sicd_collection(self):
        """
        Get the sicd collection for the bands.

        Returns
        -------
        Tuple[Dict[str, SICDType], Dict[str, str], Tuple[bool, bool, bool]]
            the first entry is a dictionary of the form {band_name: sicd}
            the second entry is of the form {band_name: shape}
            the third entry is the symmetry tuple
        """

        # TODO: check if the hdf already has the sicds defined, and fish them out if so.

        with h5py.File(self.file_name, 'r') as hf:
            # fetch the base shared sicd
            base_sicd = self._get_base_sicd(hf)

            # prepare our output workspace
            out_sicds = OrderedDict()
            shapes = OrderedDict()
            symmetry = (base_sicd.SCPCOA.SideOfTrack == 'L', False, True)

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
                    pol_sicd, shape = self._get_pol_specific_sicd(
                        hf, ds, freq_sicd, pol, freq, j, tx_rcv_pol[j],
                        r_ca_sampled, zd_time, grid_zd_time, grid_r,
                        doprate_sampled, dopcentroid_sampled, center_freq,
                        ss_az_s, dop_bw, beta0, gamma0, sigma0)
                    out_sicds[ds_name] = pol_sicd
                    shapes[ds_name] = ds.shape[:2]
        return out_sicds, shapes, symmetry


################
# The NISAR reader

class NISARReader(BaseReader, SICDTypeReader):
    """
    Gets a reader type object for NISAR files
    """

    __slots__ = ('_nisar_details', )

    def __init__(self, nisar_details):
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
        sicd_data, shape_dict, symmetry = nisar_details.get_sicd_collection()
        chippers = []
        sicds = []
        for band_name in sicd_data:
            sicds.append(sicd_data[band_name])
            chippers.append(H5Chipper(nisar_details.file_name, band_name, shape_dict[band_name], symmetry))

        SICDTypeReader.__init__(self, tuple(sicds))
        BaseReader.__init__(self, tuple(chippers), reader_type="SICD")
        self._check_sizes()

    @property
    def nisar_details(self):
        # type: () -> NISARDetails
        """
        NISARDetails: The nisar details object.
        """

        return self._nisar_details

    @property
    def file_name(self):
        return self.nisar_details.file_name
