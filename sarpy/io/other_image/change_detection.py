"""
Module for reading and interpreting the standard change detection result files.

**This requires optional dependency `pyproj`, and very likely requires `Pillow`
for dealing with a NITF with compressed image segments.**
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
import json
from typing import Union, Dict, Tuple

import numpy

from sarpy.io.general.base import AggregateReader, SarpyIOError
from sarpy.io.general.nitf import NITFReader
from sarpy.geometry.geometry_elements import FeatureCollection, Feature, Geometry

logger = logging.getLogger(__name__)

try:
    # noinspection PyPackageRequirements
    import pyproj
    # NB: this very likely requires PIL (compressed images),
    # which is handled in sarpy.io.general.nitf
except ImportError:
    logger.error(
        'Optional dependency pyproj is required for interpretation '
        'of standard change detection products.')
    pyproj = None


def _validate_coords(coords):
    """
    Validate any coordinate arrays.

    Parameters
    ----------
    coords : numpy.ndarray|list|tuple

    Returns
    -------
    (numpy.ndarray, tuple)
    """

    if not isinstance(coords, numpy.ndarray):
        coords = numpy.array(coords, dtype='float64')

    orig_shape = coords.shape

    if coords.shape[-1] < 2:
        raise ValueError(
            'The coords array must must have final dimension length at least 2. '
            'We have shape = {}'.format(coords.shape))

    if len(coords.shape) == 1:
        coords = numpy.reshape(coords, (1, -1))
    if coords.shape[-1] > 2:
        return coords[:, :2], orig_shape[:-1] + (2, )
    else:
        return coords, orig_shape


def _get_projection(corner_string, hemisphere, data_size):
    """
    Gets the projection method for [lon, lat] -> [row, col].

    Parameters
    ----------
    corner_string : str
    hemisphere : str
    data_size : tuple

    Returns
    -------
    callable
    """

    if pyproj is None:
        raise ValueError('This requires dependency pyproj.')

    # make sure that hemisphere is sensible
    if hemisphere not in ['S', 'N']:
        raise ValueError('hemisphere must be one of "N" or "S", got {}'.format(hemisphere))

    # split into the respective corner parts
    corners_strings = [corner_string[start:stop] for start, stop in zip(range(0, 59, 15), range(15, 74, 15))]
    # parse the utm zone
    utm_zone = corners_strings[0][:2]
    for entry in corners_strings:
        if entry[:2] != utm_zone:
            raise ValueError('Got incompatible corner UTM zones {} and {}'.format(utm_zone, entry[:2]))
    utm_zone = int(utm_zone)
    # parse the corner strings into utm coordinates
    utms = [(float(frag[2:8].strip()), float(frag[8:].strip())) for frag in corners_strings]
    utms = numpy.array(utms, dtype='float64')

    rows, cols = data_size
    col_vector = (utms[1, :] - utms[0, :])/(cols - 1)
    row_vector = (utms[3, :] - utms[0, :])/(rows - 1)
    if numpy.abs(row_vector.dot(col_vector)) > 1e-6:
        raise ValueError('This does not appear to be an ortho-rectified image.')
    test_corner = utms[0, :] + (rows -1)*row_vector + (cols-1)*col_vector
    if numpy.any(numpy.abs(test_corner - utms[2, :]) > 1):
        raise ValueError('This does not appear to be an ortho-rectified image.')
    # define our projection - this is for projecting lon,lat to our UTM coords (and vice versa)
    the_proj = pyproj.Proj(proj='utm',zone=utm_zone, south=(hemisphere == 'S'), ellps='WGS84')
    # account for row/column spacing for pixel conversion
    row_vector /= numpy.sum(row_vector*row_vector)
    col_vector /= numpy.sum(col_vector*col_vector)

    def projection_method(lon_lat):
        # project lon/lat to utm
        lon_lat, o_shape = _validate_coords(lon_lat)
        lon_lat = numpy.reshape(lon_lat, (-1, 2))
        xy = numpy.zeros(lon_lat.shape, dtype='float64')
        xy[:, 0], xy[:, 1] = the_proj(lon_lat[:, 0], lon_lat[:, 1])
        # recenter based on the first corner
        xy -= utms[0, :]
        # convert to pixel
        pixel_xy = numpy.zeros(lon_lat.shape, dtype='float64')
        pixel_xy[:, 0] = xy.dot(row_vector)
        pixel_xy[:, 1] = xy.dot(col_vector)
        return pixel_xy
    return projection_method


class ChangeDetectionDetails(object):
    """
    This is a helper class for interpreting the standard change detection results.
    """

    __slots__ = (
        '_file_names', '_features', '_head_feature')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        self._file_names = {}
        self._features = None
        self._head_feature = None

        if not isinstance(file_name, str):
            raise SarpyIOError('file_name is required to be a string path name.')
        if not os.path.isfile(file_name):
            raise SarpyIOError('file_name = {} is not a valid existing file'.format(file_name))

        root_dir, fname = os.path.split(file_name)
        fstem, fext = os.path.splitext(fname)
        if fext not in ['.ntf', '.nitf', '.json']:
            raise SarpyIOError('file_name is expected to have extension .ntf, .nitf, .json')
        if not (fstem.endswith('_C') or fstem.endswith('_M') or fstem.endswith('_R') or fstem.endswith('_V')):
            raise SarpyIOError(
                'file_name is expected to follow naming pattern XXX_C.ntf, XXX_M.ntf, XXX_R.ntf, '
                'or XXX_V.json')
        fstem = fstem[:-2]
        for part in ['C', 'M', 'R']:
            for ext in ['.ntf', '.nitf']:
                fil = os.path.join(root_dir, '{}_{}{}'.format(fstem, part, ext))
                if os.path.isfile(fil):
                    self._file_names[part] = fil
            if part not in self._file_names:
                raise SarpyIOError('Change detection file part {} not found'.format(part))
        fil = os.path.join(root_dir, '{}_V.json'.format(fstem))
        if os.path.isfile(fil):
            self._file_names['V'] = fil
        else:
            logger.warning('Change detection file part V (i.e. json metadata) not found.')
        self._set_features()

    @property
    def file_names(self):
        # type: () -> Dict[str, str]
        """
        Dict[str, str] : The file names.
        """

        return self._file_names

    def _set_features(self):
        if self._features is not None:
            return

        if 'V' in self._file_names:
            the_file = self._file_names['V']
            try:
                with open(the_file, 'r') as fi:
                    the_dict = json.load(fi)
                    self._features = FeatureCollection.from_dict(the_dict)
            except Exception as e:
                logger.error(
                    'Failed decoding change detection json file {}\n\twith error {}.\n\t'
                    'Skipping json definition.'.format(the_file, e))
                del self._file_names['V']  # drop this non-functional file to avoid repeating
        if self._features is not None:
            self._head_feature = self._features[0]

    @property
    def features(self):
        # type: () -> Union[FeatureCollection, None]
        """
        None|FeatureCollection: The feature list, if it is defined.
        """
        return self._features

    @property
    def head_feature(self):
        # type: () -> Union[Feature, None]
        """
        None|Feature: The main metadata Feature, which is at the head of the feature list.
        """

        return self._head_feature


class ChangeDetectionReader(AggregateReader):
    """
    An aggregate reader for the standard change detection scenario package files.
    The order of the aggregation is given by `C` (index 0), `M` (index 1), `R` (index 2).
    """

    __slots__ = ('_change_details', '_pixel_geometries')

    def __init__(self, change_details):
        """

        Parameters
        ----------
        change_details : str|ChangeDetectionDetails
        """

        self._pixel_geometries = {}  # type: Dict[str, Geometry]

        if isinstance(change_details, str):
            change_details = ChangeDetectionDetails(change_details)
        if not isinstance(change_details, ChangeDetectionDetails):
            raise TypeError(
                'change_details is required to be the file name of one of the change '
                'detection package files (XXX_C.ntf, XXX_M.ntf, XXX_R.ntf, or XXX_V.json) '
                'or a ChangeDetectionDetails instance. Got type {}'.format(change_details))
        self._change_details = change_details
        readers = (NITFReader(change_details.file_names['C']),
                   NITFReader(change_details.file_names['M']),
                   NITFReader(change_details.file_names['R']),)
        super(ChangeDetectionReader, self).__init__(readers, reader_type="OTHER")
        self._set_pixel_geometries()

    @property
    def readers(self):
        # type: () -> Tuple[NITFReader]
        return self._readers

    @property
    def features(self):
        # type: () -> Union[FeatureCollection, None]
        """
        None|FeatureCollection: The feature list, if it is defined.
        """

        return self._change_details.features

    @property
    def head_feature(self):
        # type: () -> Union[Feature, None]
        """
        None|Feature: The main metadata Feature, which is at the head of the feature list.
        """

        return self._change_details.head_feature

    def _extract_geolocation_details(self):
        """
        Extract the projection method `[lon, lat, hae] -> [pixel row, pixel col]`
        for the reader(s).

        Returns
        -------
        callable
        """

        corner_string = None
        hemisphere = None
        data_size = None
        for reader in self.readers:
            if len(reader.nitf_details.img_headers) != 1:
                raise ValueError(
                    'Each reader is expected to have a single image segment, while reader for file\n'
                    '{}\n has {} segments.'.format(reader.file_name, len(reader.nitf_details.img_headers)))
            img_head = reader.nitf_details.img_headers[0]
            if corner_string is None:
                corner_string = img_head.IGEOLO
                hemisphere = img_head.ICORDS
                data_size = reader.get_data_size_as_tuple()[0]
            else:
                if corner_string != img_head.IGEOLO:
                    raise ValueError(
                        'Got two different IGEOLO entries {} and {}'.format(corner_string, img_head.IGEOLO))
                if hemisphere != img_head.ICORDS:
                    raise ValueError(
                        'Got two different ICORDS entires {} and {}'.format(hemisphere, img_head.ICORDS))
                if data_size != reader.get_data_size_as_tuple()[0]:
                    raise ValueError(
                        'Got two different data sizes {} and {}'.format(data_size, reader.get_data_size_as_tuple()[0]))
        if hemisphere not in ['N', 'S']:
            raise ValueError('Got unexpected ICORDS {}'.format(hemisphere))
        return _get_projection(corner_string, hemisphere, data_size)

    def _set_pixel_geometries(self):
        """
        Sets the pixel geometries.

        Returns
        -------
        None
        """

        feats = self.features
        if feats is None:
            return

        proj_method = self._extract_geolocation_details()
        for feat in feats.features:
            if feat.geometry is None or feat.uid in self._pixel_geometries:
                continue
            self._pixel_geometries[feat.uid] = feat.geometry.apply_projection(proj_method)
