# -*- coding: utf-8 -*-
"""
Functionality for reading Cosmo Skymed data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"

from collections import OrderedDict

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

try:
    import h5py
except ImportError:
    h5py = None  # TODO: warn

from ..sicd_elements.blocks import Poly1DType, Poly2DType
from ..sicd_elements.SICD import SICDType
from ..sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from ..sicd_elements.ImageCreation import ImageCreationType
from ..sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType
from ..sicd_elements.ImageData import ImageDataType
from ..sicd_elements.GeoData import GeoDataType, SCPType
from ..sicd_elements.Position import PositionType, XYZPolyType
from ..sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from ..sicd_elements.Timeline import TimelineType, IPPSetType
from ..sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType, TxFrequencyProcType
from ..sicd_elements.RMA import RMAType, INCAType
from ..sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from ..geometry import point_projection


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a Cosmo Skymed file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    CSKReader|None
        `CSKReader` instance if Cosmo Skymed file, `None` otherwise
    """

    try:
        csk_details = CSKDetails(file_name)
        print('File {} is determined to be a Cosmo Skymed file.'.format(file_name))
        return CSKReader(csk_details)
    except (IOError, ImportError, KeyError, ValueError):
        # TODO: what all should we catch?
        return None


##########
# helper functions

def _extract_attrs(h5_element):
    out = OrderedDict()
    for key in h5_element.attrs:
        val = h5_element.attrs[key]
        out[key] = val.decode('utf-8') if isinstance(val, bytes) else val
    return out


def _get_seconds(dt1, dt2):  # type: (numpy.datetime64, numpy.datetime64) -> float
    tdt1 = dt1.astype('datetime64[ns]')
    tdt2 = dt2.astype('datetime64[ns]')  # convert both to nanosecond precision
    return (tdt1.astype('int64') - tdt2.astype('int64'))*1e-9


###########
# parser and interpreter for hdf5 attributes

class CSKDetails(object):
    __slots__ = ('_file_name', )

    def __init__(self, file_name):
        if h5py is None:
            raise ImportError("Can't open Cosmo Skymed files, because the h5py dependency is missing.")

        with h5py.File(file_name, 'r') as hf:
            sat_id = hf.attrs['Satellite ID'].decode('utf-8')

        if 'CSK' not in sat_id:
            raise ValueError('Expected hdf5 to have attribute `Satellite ID` which contains "CSK". '
                             'Got {}.'.format(sat_id))

        self._file_name = file_name

    def _get_hdf_dicts(self):
        with h5py.File(self._file_name, 'r') as hf:
            h5_dict = _extract_attrs(hf)
            band_dict = OrderedDict()

            for gp_name in sorted(hf.keys()):
                gp = hf[gp_name]
                ord_dict = OrderedDict()
                ord_dict['shape'] = gp['SBI'].shape[:2]
                ord_dict['attrs'] = _extract_attrs(gp)
                band_dict[gp_name] = ord_dict
        return h5_dict, band_dict

    def _parse_h5_dict(self, h5_dict, band_dict):
        # type: (dict, dict) -> SICDType

        def get_collection_info():  # type: () -> CollectionInfoType
            mode_type = 'STRIPMAP' if h5_dict['Acquisition Mode'] in \
                                      ['HIMAGE', 'PINGPONG', 'WIDEREGION', 'HUGEREGION'] else 'DYNAMIC STRIPMAP'
            return CollectionInfoType(Classification='UNCLASSIFIED',
                                      CollectorName=h5_dict['Satellite ID'],
                                      CoreName=str(h5_dict['Programmed Image ID']),
                                      CollectType='MONOSTATIC',
                                      RadarMode=RadarModeType(ModeId=h5_dict['Multi-Beam ID'],
                                                              ModeType=mode_type))

        def get_image_creation():  # type: () -> ImageCreationType
            return ImageCreationType(DateTime=numpy.datetime64(h5_dict['Product Generation UTC'], 'ns'))

        def get_grid():  # type: () -> GridType
            if h5_dict['Projection ID'] == 'SLANT RANGE/AZIMUTH':
                image_plane = 'SLANT'
                gr_type = 'RGZERO'
            else:
                image_plane = 'GROUND'
                gr_type = None
            center_frequency = h5_dict['Radar Frequency']
            # Row
            row_window_name = h5_dict['Range Focusing Weighting Function'].rstrip().upper()
            row_params = None
            if row_window_name == 'HAMMING':
                row_params = {'COEFFICIENT': '{0:15f}'.format(h5_dict['Range Focusing Weighting Coefficient'])}
            row = DirParamType(Sgn=-1,
                               KCtr=2*center_frequency/speed_of_light,
                               DeltaKCOAPoly=Poly2DType(Coefs=[[0, ], ]),
                               WgtType=WgtTypeType(WindowName=row_window_name, Parameters=row_params))
            # Col
            col_window_name = h5_dict['Azimuth Focusing Weighting Function'].rstrip().upper()
            col_params = None
            if col_window_name == 'HAMMING':
                col_params = {'COEFFICIENT': '{0:15f}'.format(h5_dict['Azimuth Focusing Weighting Coefficient'])}
            col = DirParamType(Sgn=-1,
                               KCtr=0,
                               WgtType=WgtTypeType(WindowName=col_window_name, Parameters=col_params))
            return GridType(ImagePlane=image_plane, Type=gr_type, Row=row, Col=col)

        def get_timeline():  # type: () -> TimelineType
            collect_start = numpy.datetime64(h5_dict['Scene Sensing Start UTC'], 'ns')
            collect_end = numpy.datetime64(h5_dict['Scene Sensing Stop UTC'], 'ns')
            return TimelineType(CollectStart=collect_start,
                                CollectDuration=_get_seconds(collect_end, collect_start),
                                IPP=[IPPSetType(index=0, TStart=0, TEnd=0, IPPStart=0), ])

        def get_position():  # type: () -> PositionType
            # TODO: continue line 281
            pass

        collection_info = get_collection_info()
        image_creation = get_image_creation()
        grid = get_grid()
        timeline = get_timeline()
        
        return SICDType(CollectionInfo=collection_info,
                        ImageCreation=image_creation,
                        Grid=grid,
                        Timeline=timeline)

    def get_sicd_collection(self):
        """
        Get the collection of band name and sicd collection for each file.

        Returns
        -------
        """
        # TODO: flesh out this docstring

        h5_dict, band_dict = self._get_hdf_dicts()


################
# The actual reader -
# note that this is motivated by BaseReader


class CSKReader(object):
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

        if isinstance(csk_details, str):
            csk_details = CSKDetails(csk_details)
        if not isinstance(csk_details, CSKDetails):
            raise TypeError('The input argument for RadarSatCSKReader must be a '
                            'filename or CSKDetails object')
        # TODO: all of the things
