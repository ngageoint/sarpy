# -*- coding: utf-8 -*-
"""
Functionality for reading NISAR data into a SICD model.
"""

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
    warnings.warn('The h5py module is not successfully imported, '
                  'which precludes NISAR reading capability!')

from .sicd_elements.blocks import Poly1DType, Poly2DType, RowColType
from .sicd_elements.SICD import SICDType
from .sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from .sicd_elements.ImageCreation import ImageCreationType
from .sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType, TxStepType
from .sicd_elements.ImageData import ImageDataType
from .sicd_elements.GeoData import GeoDataType, SCPType
from .sicd_elements.SCPCOA import SCPCOAType
from .sicd_elements.Position import PositionType, XYZPolyType
from .sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from .sicd_elements.Timeline import TimelineType, IPPSetType
from .sicd_elements.ImageFormation import ImageFormationType, TxFrequencyProcType, RcvChanProcType
from .sicd_elements.RMA import RMAType, INCAType
from .sicd_elements.Radiometric import RadiometricType
from ...geometry import point_projection
from .base import BaseChipper, BaseReader, string_types
from .utils import get_seconds, fit_time_coa_polynomial, fit_position_xvalidation

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Jarred Barber", "Wade Schwartzkopf")


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a NISAR file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    NISARReader|None
        `NISARReader` instance if NISAR file, `None` otherwise
    """

    if h5py is None:
        return None

    try:
        nisar_details = NISARDetails(file_name)
        print('File {} is determined to be a NISAR file.'.format(file_name))
        return NISARReader(nisar_details)
    except (IOError, KeyError, ValueError, SyntaxError):
        # TODO: what all should we catch?
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

    return val.decode('utf-8').strip() if isinstance(val, bytes) else val.strip()

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

    if isinstance(str_in, bytes):
        str_in = str_in.decode('utf-8')

    raise NotImplementedError('Extract from string - {}'.format(str_in))


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

        with h5py.File(file_name, 'r') as hf:
            try:
                gp = hf['/science/LSAR/SLC']
            except:
                raise ValueError('The hdf5 file does not have required path /science/LSAR/SLC')

        # TODO: finish
        raise NotImplementedError

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

        return hf['/science/LSAR/identification/listOfFrequencies'][:]

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

        start = numpy.datetime64(_stringify(hf['/science/LSAR/identification/zeroDopplerStartTime'][:]))
        end = numpy.datetime64(_stringify(hf['/science/LSAR/identification/zeroDopplerEndTime'][:]))
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
        zd_time = ds[:] + get_seconds(base_sicd.Timeline.CollectStart, ref_time)
        ss_az_s = gp['zeroDopplerTimeSpacing'][:]

        if base_sicd.SCPCOA.SideOfTrack == 'L':
            zd_time = zd_time[::-1]
            ss_az_s *= -1

        gp = hf['/science/LSAR/SLC/metadata/processingInformation/parameters']
        grid_r = gp['slantRange'][:]
        ds = gp['zeroDopplerTime']
        ref_time = _get_ref_time(ds.attrs['units'])
        grid_zd_time = ds[:] + get_seconds(base_sicd.Timeline.CollectStart, ref_time)

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
            # TODO: adjust corename formatting
            return CollectionInfoType(
                CollectorName=_stringify(hf.attrs['mission_name']),
                CoreName='{}{}'.format(gp['absoluteOrbitNumber'][:], gp['trackNumber']),
                CollectType='MONOSTATIC',
                Classification='UNCLASSIFIED',
                RadarMode=RadarModeType(ModeType='STRIPMAP'))  # TODO: ModeID?

        def get_image_creation():
            # type: () -> ImageCreationType
            application = 'ISCE'
            try:
                application = '{} {}'.format(
                    application,
                    _stringify(hf['/science/LSAR/SLC/metadata/processingInformation/algorithms/ISCEVersion'][:]))
            except:
                pass

            from sarpy.__about__ import __version__
            # TODO: Site and DateTime?
            return ImageCreationType(
                Application=application,
                Profile='sarpy {}'.format(__version__))

        def get_geo_data():
            # type: () -> GeoDataType
            # seeds a rough SCP for projection usage
            poly_str = _stringify(hf['/science/LSAR/identification/boundingPolygon'][:])
            raise NotImplementedError('poly_str = {}'.format(poly_str))
            # TODO: what is the format of this string? parse this...

            llh = numpy.zeros((3, ), dtype=numpy.float64)
            # llh[0:2] = <junk from above>
            llh[2] = numpy.mean(hf['/science/LSAR/SLC/metadata/processingInformation/parameters/referenceTerrainHeight'][:])
            return GeoDataType(SCP=SCPType(LLH=llh))

        def get_grid():
            # type: () -> GridType

            # TODO: JPL states that uniform weighting in data simulated from UAVSAR is a
            #  placeholder, not an accurate description of the data.  At this point, it
            #  is not clear what the final weighting description for NISAR will be.

            gp = hf['/science/LSAR/SLC/metadata/processingInformation/parameters']
            row_wgt = gp['rangeChirpWeighting'][:]
            win_name = 'UNIFORM' if numpy.all(row_wgt == row_wgt[0]) else 'UNKNOWN'
            row = DirParamType(
                Sgn=-1,
                DeltaKCOAPoly=[[0,]],
                WgtFunct=row_wgt,
                WgtType=WgtTypeType(WindowName=win_name))

            col_wgt = gp['azimuthChirpWeighting'][:]
            win_name = 'UNIFORM' if numpy.all(col_wgt == col_wgt[0]) else 'UNKNOWN'
            col = DirParamType(
                Sgn=-1,
                KCtr=0,
                WgtFunct=col_wgt,
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
            T = gp['time'][:] + get_seconds(ref_time, collect_start)
            Pos = gp['position'][:]
            Vel = gp['velocity'][:]
            P_x, P_y, P_z = fit_position_xvalidation(T, Pos, Vel, max_degree=6)
            return PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

        def get_scpcoa():
            # type: () -> SCPCOAType
            # remaining fields set later
            sot = _stringify(hf['/science/LSAR/identification/lookDirection'])[0].upper()
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

        def get_radiometric():
            # type: () -> RadiometricType

            def get_poly(ds):
                array = ds[:]
                fill = ds.attrs['_FillValue']
                boolc = (array != fill)
                if numpy.any(boolc):
                    return [[numpy.mean(array[boolc]), ], ]
                else:
                    return None

            # TODO: this is forcing everything to be constant...why do that?
            gp = hf['/science/LSAR/SLC/metadata/calibrationInformation/geometry']
            beta = get_poly(gp['beta0'])
            gamma = get_poly(gp['gamma0'])
            sigma = get_poly(gp['sigma0'])
            return RadiometricType(
                BetaZeroSFPoly=beta,
                GammaZeroSFPoly=gamma,
                SigmaZeroSFPoly=sigma)

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
        radiometric = get_radiometric()

        return SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            GeoData=geo_data,
            Grid=grid,
            Timeline=timeline,
            Position=position,
            SCPCOA=scpcoa,
            ImageFormation=image_formation,
            RMA=rma,
            Radiometric=radiometric)

    @staticmethod
    def _get_freq_specific_sicd(gp, base_sicd):
        """
        Gets the frequency specific sicd.

        Parameters
        ----------
        hf : h5py.File
        gp : h5py.Group
        base_sicd : SICDType

        Returns
        -------
        (SICDType, numpy.ndarray)
            frequency dependent sicd and list of polarizations
        """

        def update_grid():
            row_imp_resp_bw = 2*gp['processedRangeBandwidth'][:]/speed_of_light
            t_sicd.Grid.Row.SS = gp['slantRangeSpacing'][:]
            t_sicd.Grid.Row.ImpRespBW = row_imp_resp_bw
            t_sicd.Grid.Row.DeltaK1 -= 0.5*row_imp_resp_bw
            t_sicd.Grid.Row.DeltaK2 -= t_sicd.Grid.Row.DeltaK1

        def update_timeline():
            prf = gp['nominalAcquisitionPRF'][:]
            t_sicd.Timeline.IPP[0].IPPEnd = prf*t_sicd.Timeline.CollectDuration
            t_sicd.Timeline.IPP[0].IPPPoly = [0, prf]

        def define_radar_collection():
            tx_rcv_pol = []
            tx_pol = []
            for entry in pols:
                tx_rcv_pol.append('{}:{}'.format(entry[0], entry[1]))
                if entry[0] not in tx_pol:
                    tx_pol.append(entry[0])
            fc = gp['acquiredCenterFrequency'][:]
            bw = gp['acquiredRangeBandwidth'][:]
            tx_freq = TxFrequencyType(Min=fc - 0.5*bw, Max=fc + 0.5*bw)
            rcv_chans = [ChanParametersType(TxRcvPolarization=pol) for pol in tx_rcv_pol]
            if len(tx_pol) == 1:
                tx_sequence = None
                tx_pol = tx_pol[0]
            else:
                tx_sequence = [TxStepType(WFIndex=j, TxPolarization=pol) for j, pol in enumerate(tx_pol)]
                tx_pol = 'SEQUENCE'

            t_sicd.RadarCollection = RadarCollectionType(
                TxFrequency=tx_freq,
                RcvChannels=rcv_chans,
                TxPolarization=tx_pol,
                TxSequence=tx_sequence)

        def update_image_formation():
            fc = gp['processedCenterFrequency'][:]
            bw = gp['processedRangeBandwidth'][:]
            t_sicd.ImageFormation.TxFrequencyProc = TxFrequencyProcType(
                MinProc=fc - 0.5*bw,
                MaxProc=fc + 0.5*bw)

        pols = gp['listOfPolarizations'][:]
        t_sicd = base_sicd.copy()
        update_grid()
        update_timeline()
        define_radar_collection()
        update_image_formation()

        return t_sicd, pols

    def _get_pol_specific_sicd(selfgp, base_sicd):
        """
        Gets the frequency/polarization specific sicd.

        Parameters
        ----------
        base_sicd : SICDType

        Returns
        -------
        SICDType
        """

        raise NotImplementedError

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
            symmetry = (False, base_sicd.SCPCOA.SideOfTrack == 'L', True)

            # fetch the common use data for frequency issues
            collect_start, collect_end, duration = self._get_collection_times(hf)
            zd_time, ss_az_s, grid_r, grid_zd_time = self._get_zero_doppler_data(hf, base_sicd)

            # formulate the frequency specific sicd information
            freqs = self._get_frequency_list(hf)
            for i, freq in enumerate(freqs):
                gp_name = '/science/LSAR/SLC/swaths/frequency{}'.format(freq)
                gp = hf[gp_name]
                print('freq {} at hdf5 group'.format(freq, gp_name))  # TODO: this is temp for debugging
                freq_sicd, pols = self._get_freq_specific_sicd(gp, base_sicd)

                # formulate the frequency dependent doppler grid
                # TODO: processedAzimuthBandwidth acknowledged by JPL to be wrong
                #   in simulated datasets.
                dop_bw = gp['processedAzimuthBandwidth'][:]
                dopcentroid_sampled = gp['dopplerCentroid'][:]
                doprate_sampled = gp['azimuthFMRate'][:]
                # formulate the frequency/polarization specific sicd information
                for j, pol in enumerate(pols):
                    gp_name2 = '{}/{}'.format(gp_name, pol)
                    gp2 = gp[pol]
                    # TODO: line 326


################
# The NISAR reader

class NISARReader(BaseReader):
    """
    Gets a reader type object for Cosmo Skymed files
    """

    __slots__ = ('_nisar_details', )

    def __init__(self, nisar_details):
        """

        Parameters
        ----------
        nisar_details : str|NISARDetails
            file name or NISARDetails object
        """

        if isinstance(nisar_details, string_types):
            nisar_details = NISARDetails(nisar_details)
        if not isinstance(nisar_details, NISARDetails):
            raise TypeError('The input argument for NISARReader must be a '
                            'filename or NISARDetails object')

        raise NotImplementedError
