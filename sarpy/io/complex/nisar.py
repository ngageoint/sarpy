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
from .utils import get_seconds, fit_time_coa_polynomial

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

    return val.decode('utf-8') if isinstance(val, bytes) else val

class NISARDetails(object):
    """
    Parses and converts the Cosmo Skymed metadata
    """

    __slots__ = ('_file_name', '_satellite', '_product_type')

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

        # we'll have one sicd per frequency and polarization
        #   - keep a running dictionary of these options

        # TODO: finish
        raise NotImplementedError

    @property
    def file_name(self):
        """
        str: the file name
        """

        return self._file_name

    def _get_base_sicd(self):
        """
        Defines the base SICD object, to be refined with further details.

        Returns
        -------
        SICDType
        """

        def get_collection_start():
            # type: () -> numpy.datetime64
            return numpy.datetime64(_stringify(hf['/science/LSAR/identification/zeroDopplerStartTime'][:]).strip())

        def get_collection_end():
            # type: () -> numpy.datetime64
            return numpy.datetime64(_stringify(hf['/science/LSAR/identification/zeroDopplerEndTime'][:]).strip())

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
                    _stringify(hf['/science/LSAR/SLC/metadata/processingInformation/algorithms/ISCEVersion'][:]).strip())
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
            poly_str = _stringify(hf['/science/LSAR/identification/boundingPolygon'][:]).strip()
            # TODO: what is the format of this string? parse this...
            llh = numpy.zeros((3, ), dtype=numpy.float64)
            # llh[0:2] = <junk from above>
            llh[2] = numpy.mean(hf['/science/LSAR/SLC/metadata/processingInformation/parameters/referenceTerrainHeight'][:])
            return GeoDataType(SCP=SCPType(LLH=llh))

        def get_grid():
            # type: () -> GridType
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
            # TODO: line 129
            pass

        with h5py.File(self.file_name, 'r') as hf:
            collection_start = get_collection_start()
            collection_end = get_collection_end()

            collection_info = get_collection_info()
            image_creation = get_image_creation()
            geo_data = get_geo_data()
            grid = get_grid()
            timeline = get_timeline()

        # TODO: finish
        return SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            GeoData=geo_data,
            Grid=grid,
            Timeline=timeline)


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

        # TODO: finish
        raise NotImplementedError


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
