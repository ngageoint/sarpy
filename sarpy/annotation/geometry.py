# -*- coding: utf-8 -*-

from collections import OrderedDict
from uuid import uuid4
from typing import List
import json

import numpy


def _compress_identical(coords):
    """
    Eliminate consecutive points with same first two coordinates.

    Parameters
    ----------
    coords : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        coords array with consecutive identical points supressed (last point retained)
    """

    if coords.shape[0] < 2:
        return coords

    include = numpy.zeros((coords.shape[0], ), dtype=numpy.bool)
    include[-1] = True

    for i, (first, last) in enumerate(zip(coords[:-1, :], coords[1:, :])):
        if not (first[0] == last[0] and first[1] == last[1]):
            include[i] = True
    return coords[include, :]


class _Jsonable(object):
    """
    Abstract class for json serializability.
    """

    @classmethod
    def from_json(cls, the_json):
        raise NotImplementedError

    def to_json(self, parent_dict=None):
        raise NotImplementedError

    def __str__(self):
        return '{}(**{})'.format(self.__class__.__name__, json.dumps(self.to_json(), indent=1))

    def __repr__(self):
        return '{}(**{})'.format(self.__class__.__name__, self.to_json())

    def copy(self):
        typ = self.__class__
        return typ.from_json(self.to_json())


# TODO: FeatureList? Or, should we be custom? Maybe Annotations?
#   We would transform to FeatureList.


class Feature(_Jsonable):
    """
    Generic feature class - mirrors basic geojson functionality
    """

    __slots__ = ('_uid', '_geometry', '_properties')
    _type = 'Feature'

    def __init__(self, id=None, geometry=None, properties=None):
        if id is None:
            self._uid = str(uuid4())
        elif not isinstance(id, str):
            raise TypeError('id must be a string.')
        else:
            self._uid = id

        if not isinstance(geometry, Geometry):
            raise TypeError('geometry must be an instance of Geometry base class')
        self._geometry = geometry

        # TODO: type check properties?
        self._properties = properties

    @property
    def uid(self):
        """
        The feature unique identifier.

        Returns
        -------
        str
        """

        return self._uid

    @property
    def type(self):
        """
        The type identifier - `Feature` in this case.

        Returns
        -------
        str
        """

        return self._type

    @property
    def geometry(self):
        """
        The geometry object.

        Returns
        -------
        Geometry
        """

        return self._geometry

    @property
    def properties(self):  # type: () -> _Jsonable
        """
        The properties.

        Returns
        -------

        """

        return self._properties

    @classmethod
    def from_json(cls, the_json):
        raise NotImplementedError

    def to_json(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['id'] = self.uid
        parent_dict['type'] = self.type
        parent_dict['geometry'] = self.geometry.to_json()
        parent_dict['properties'] = self.properties.to_json()
        return parent_dict


class Geometry(_Jsonable):
    """
    Abstract Geometry base class.
    """

    _type = 'Geometry'

    @property
    def type(self):
        """
        The type string for the geometry object class.

        Returns
        -------
        str
        """

        return self._type

    @classmethod
    def from_json(cls, geometry):
        """
        Deserialize from json.

        Parameters
        ----------
        geometry : Dict

        Returns
        -------

        """

        typ = geometry['type']
        if typ == 'GeometryCollection':
            return GeometryCollection.from_json(geometry)
        else:
            return GeometryObject.from_json(geometry)

    def to_json(self, parent_dict=None):
        raise NotImplementedError

    def transform_to_wgs84(self, sicd):
        """

        Parameters
        ----------
        sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

        Returns
        -------
        Feature
        """

        # TODO: transform to WGS-84 coordinates using sicd?
        pass


class GeometryCollection(Geometry):
    """
    Geometry collection - following the geojson structure
    """

    __slots__ = ('_geometries', )
    _type = 'GeometryCollection'

    def __init__(self, geometries):
        """

        Parameters
        ----------
        geometries : None|List[Geometry]
        """

        self._geometries = None
        if geometries is not None:
            self.geometries = geometries

    @property
    def geometries(self):
        """
        The geometry collection.

        Returns
        -------
        None|List[Geometry]
        """

        return self._geometries

    @geometries.setter
    def geometries(self, geometries):
        if geometries is None:
            self._geometries = None
            return
        elif not isinstance(geometries, list):
            raise TypeError(
                'geometries must be None or a list of Geometry objects. Got type {}'.format(type(geometries)))
        elif len(geometries) < 2:
            raise ValueError('geometries must have length greater than 1.')
        for entry in geometries:
            if not isinstance(entry, Geometry):
                raise TypeError(
                    'geometries must be a list of Geometry objects. Got an element of type {}'.format(type(entry)))

    @classmethod
    def from_json(cls, geometry):  # type: (Union[None, dict]) -> GeometryCollection
        typ = geometry.get('type', None)
        if typ != 'GeometryCollection':
            raise ValueError('GeometryCollection cannot be constructed from {}'.format(geometry))
        geometries = []
        for entry in geometry['geometries']:
            if isinstance(entry, Geometry):
                geometries.append(entry)
            elif isinstance(entry, dict):
                geometries.append(Geometry.from_json(entry))
            else:
                raise TypeError(
                    'The geometries attribute must contain either a Geometry or json serialization of a Geometry. '
                    'Got an entry of type {}'.format(type(entry)))
        return cls(geometries)

    def to_json(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        if self.geometries is not None:
            parent_dict['geometries'] = [entry.to_json() for entry in self.geometries]
        return parent_dict


class GeometryObject(Geometry):
    """
    Abstract geometry object class - mirrors basic geojson functionality
    """
    _type = 'Geometry'

    def get_coordinate_list(self):
        """
        The geojson style coordinate list.

        Returns
        -------
        list
        """

        raise NotImplementedError

    @classmethod
    def from_json(cls, geometry):  # type: (dict) -> GeometryObject
        typ = geometry.get('type', None)
        if typ is None:
            raise ValueError('Poorly formed json for GeometryObject {}'.format(geometry))
        elif typ == 'Point':
            return Point(coordinates=geometry['coordinates'])
        elif typ == 'MultiPoint':
            return MultiPoint(coordinates=geometry['coordinates'])
        elif typ == 'LineString':
            pass
        elif typ == 'MultiLineString':
            pass
        elif typ == 'Polygon':
            pass
        elif typ == 'MultiPolygon':
            pass
        else:
            raise ValueError('Unknown type {} for GeometryObject from json {}'.format(typ, geometry))

    def to_json(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        coords = self.get_coordinate_list()
        if coords is not None:
            parent_dict['coordinates'] = coords
        return parent_dict


class Point(GeometryObject):
    __slots__ = ('_coordinates', )
    _type = 'Point'

    def __init__(self, coordinates=None):
        self._coordinates = None
        if coordinates is not None:
            self.coordinates = coordinates

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):  # type: (Union[None, list, tuple, numpy.ndarray]) -> None
        if coordinates is None:
            self._coordinates = None
            return

        if not isinstance(coordinates, numpy.ndarray):
            coordinates = numpy.array(coordinates, dtype=numpy.float64)

        if coordinates.ndim != 1:
            raise ValueError(
                'coordinates must be a one-dimensional array. Got shape {}'.format(coordinates.shape))
        elif not (2 <= coordinates.size <= 4):
            raise ValueError(
                'coordinates must have between 2 and 4 entries. Got shape {}'.format(coordinates.shape))
        else:
            self._coordinates = coordinates

    def get_coordinate_list(self):
        if self._coordinates is None:
            return None
        else:
            return list(self._coordinates)

    @classmethod
    def from_json(cls, geometry):  # type: (dict) -> Point
        if not geometry.get('type', None) == 'Point':
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])


class MultiPoint(GeometryObject):
    _type = 'MultiPoint'
    __slots__ = ('_points', )

    def __init__(self, coordinates=None):
        self._points = None
        if coordinates is not None:
            self.points = coordinates

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if points is None:
            self._points = None
        elif isinstance(points, (list, tuple, numpy.ndarray)):
            self._points = []
            for entry in points:
                if isinstance(entry, Point):
                    self._points.append(entry)
                else:
                    self._points.append(Point(coordinates=entry))
        else:
            raise TypeError(
                'Multipoint requires that points is None or a list/tuple/array of points. '
                'Got type {}'.format(type(points)))

    def get_coordinate_list(self):
        if self._points is None:
            return None
        return [point.get_coordinate_list() for point in self._points]

    @classmethod
    def from_json(cls, geometry):  # type: (dict) -> MultiPoint
        if not geometry.get('type', None) == 'MultiPoint':
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])


class LineString(GeometryObject):
    __slots__ = ('_coordinates', )
    _type = 'LineString'

    def __init__(self, coordinates=None):
        self._coordinates = None
        if coordinates is not None:
            self.coordinates = coordinates

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):  # type: (Union[None, list, tuple, numpy.ndarray]) -> None
        if coordinates is None:
            self._coordinates = None
            return

        if not isinstance(coordinates, numpy.ndarray):
            coordinates = numpy.array(coordinates, dtype=numpy.float64)

        if coordinates.ndim != 2:
            raise ValueError(
                'coordinates must be a two-dimensional array. '
                'Got shape {}'.format(coordinates.shape))
        if not (2 <= coordinates.shape[1] <= 4):
            raise ValueError(
                'The second dimension of coordinates must have between 2 and 4 entries. '
                'Got shape {}'.format(coordinates.shape))
        if coordinates.shape[0] < 2:
            raise ValueError(
                'coordinates must consist of at least 2 points. '
                'Got shape {}'.format(coordinates.shape))
        coordinates = _compress_identical(coordinates)
        if coordinates.shape[0] < 2:
            raise ValueError(
                'coordinates must consist of at least 2 points after suppressing '
                'consecutive repeated points. Got shape {}'.format(coordinates.shape))
        self._coordinates = coordinates

    def get_coordinate_list(self):
        if self._coordinates is None:
            return None
        else:
            return list(self._coordinates)

    @classmethod
    def from_json(cls, geometry):  # type: (dict) -> LineString
        if not geometry.get('type', None) == 'LineString':
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])


class MultiLineString(GeometryObject):
    __slots__ = ('_lines', )
    _type = 'MultiLineString'

    def __init__(self, coordinates=None):
        self._lines = None
        if coordinates is not None:
            self.lines = coordinates

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines):
        if lines is None:
            self._lines = None
        elif isinstance(lines, (list, tuple, numpy.ndarray)):
            self._lines = []
            for entry in lines:
                if isinstance(entry, LineString):
                    self._lines.append(entry)
                else:
                    self._lines.append(LineString(coordinates=entry))
        else:
            raise TypeError(
                'MultiLineString requires that lines is None or a list/tuple/array of LineStrings. '
                'Got type {}'.format(type(lines)))

    def get_coordinate_list(self):
        if self._lines is None:
            return None
        return [line.get_coordinate_list() for line in self._lines]

    @classmethod
    def from_json(cls, geometry):  # type: (dict) -> MultiLineString
        if not geometry.get('type', None) == 'MultiLineString':
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])


class LinearRing(object):
    """
    This is not directly a valid geojson member, but plays the role of a single
    polygonal element, and is only used as a Polygon constituent.
    """
    __slots__ = ('_coordinates', '_bounding_box', '_segmentation', '_diffs')

    def __init__(self, coordinates=None):
        self._coordinates = None
        self._bounding_box = None
        self._segmentation = None
        self._diffs = None
        if coordinates is not None:
            self.set_coordinates(coordinates)

    @property
    def bounding_box(self):
        """
        The bounding box of the form [[x_min, x_max], [y_min, y_max]].
        *Note that would be extremely misleading for a naively constructed
        lat/lon polygon crossing the boundary of discontinuity and/or surrounding a pole.*

        Returns
        -------
        numpy.ndarray
        """

        return self.bounding_box

    def get_area(self):
        """
        Gets the area of the polygon. If a polygon is self-intersecting, then this
        result may be pathological. If this is positive, then the orientation is
        counter-clockwise. If this is negative, then the orientation is clockwise.
        If zero, then this polygon is degenerate.

        Returns
        -------
        float
        """

        return float(0.5*numpy.sum(self._coordinates[:-1, 0]*self._coordinates[1:, 1] -
                                   self._coordinates[1:, 0]*self._coordinates[:-1, 1]))

    def get_centroid(self):
        """
        Gets the centroid of the polygon - note that this may not actually lie in
        the polygon interior for non-convex polygon. This will result in an undefined value
        if the polygon is degenerate.

        Returns
        -------
        numpy.ndarray
        """

        arr = self._coordinates[:-1, 0]*self._coordinates[1:, 1] - \
            self._coordinates[1:, 0]*self._coordinates[:-1, 1]
        area = 0.5*numpy.sum(arr)  # signed area
        x = numpy.sum(0.5*(self._coordinates[:-1, 0] + self._coordinates[1:, 0])*arr)
        y = numpy.sum(0.5*(self._coordinates[:-1, 1] + self._coordinates[1:, 1])*arr)
        return numpy.array([x, y], dtype=numpy.float64)/(3*area)

    def set_coordinates(self, coordinates):
        if coordinates is None:
            self._coordinates = None
            self._bounding_box = None
            self._segmentation = None
            self._diffs = None
            return

        if not isinstance(coordinates, numpy.ndarray):
            coordinates = numpy.array(coordinates, dtype=numpy.float64)
        if len(coordinates.shape) != 2:
            raise ValueError(
                'coordinates must be two-dimensional. Got shape {}'.format(coordinates.shape))
        if not (2 <= coordinates.shape[1] <= 4):
            raise ValueError('The second dimension of coordinates must have between 2 and 4 entries. '
                             'Got shape {}'.format(coordinates.shape))
        if coordinates.shape[0] < 3:
            raise ValueError('coordinates must consist of at least 3 points. '
                             'Got shape {}'.format(coordinates.shape))
        coordinates = _compress_identical(coordinates)
        if (coordinates[0, 0] != coordinates[-1, 0]) or \
                (coordinates[0, 1] != coordinates[-1, 1]):
            coordinates = numpy.vstack((coordinates, coordinates[0, :]))
        if coordinates.shape[0] < 4:
            raise ValueError(
                'After compressing repeated (in sequence) points and ensuring first and '
                'last point are the same, coordinates must contain at least 4 points. '
                'Got shape {}'.format(coordinates.shape))
        self._coordinates = coordinates
        # construct bounding box
        self._bounding_box = numpy.empty((2, 2), dtype=coordinates.dtype)
        self._bounding_box[0, :] = (numpy.min(coordinates[:, 0]), numpy.max(coordinates[:, 0]))
        self._bounding_box[1, :] = (numpy.min(coordinates[:, 1]), numpy.max(coordinates[:, 1]))
        # construct diffs
        self._diffs = coordinates[1:, :] - coordinates[:-1, :]
        self._segmentation = {
            'x': self._construct_segmentation(coordinates[:, 0], coordinates[:, 1]),
            'y': self._construct_segmentation(coordinates[:, 1], coordinates[:, 0])
        }

    @staticmethod
    def _construct_segmentation(coords, o_coords):
        # helper method
        def overlap(fst, lst, segment):
            if fst == lst and fst == segment['min']:
                return True
            if segment['max'] <= fst or lst <= segment['min']:
                return False
            return True

        def do_min_val_value(segment, val1, val2):
            segment['min_value'] = min(val1, val2, segment['min_value'])
            segment['max_value'] = max(val1, val2, segment['max_value'])

        inds = numpy.argsort(coords[:-1])
        segments = []
        beg_val = coords[inds[0]]
        val = None
        for ind in inds[1:]:
            val = coords[ind]
            if val > beg_val:  # make a new segment
                segments.append({'min': beg_val, 'max': val, 'inds': [], 'min_value': numpy.inf, 'max_value': -numpy.inf})
                beg_val = val
        else:
            # it may have ended without appending the segment
            if val > beg_val:
                segments.append({'min': beg_val, 'max': val, 'inds': [], 'min_value': numpy.inf, 'max_value': -numpy.inf})
        del beg_val, val

        # now, let's populate the inds lists and min/max_values elements
        for i, (beg_value, end_value, ocoord1, ocoord2) in enumerate(zip(coords[:-1], coords[1:], o_coords[:-1], o_coords[1:])):
            first, last = (beg_value, end_value) if beg_value <= end_value else (end_value, beg_value)
            # check all the segments for overlap
            for j, seg in enumerate(segments):
                if overlap(first, last, seg):
                    seg['inds'].append(i)
                    do_min_val_value(seg, ocoord1, ocoord2)

        return tuple(segments)

    def _contained_segment_data(self, x, y):
        """
        This is a helper function for the polygon containment effort.
        This determines whether the x or y segmentation should be utilized, and
        the details for doing so.

        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray

        Returns
        -------
        (int|None, int|None, str)
            the segment index start (inclusive), the segment index end (exclusive),
            and "x" or "y" for which segmentation is better.
        """

        def segment(coord, segments):
            tmin = coord.min()
            tmax = coord.max()

            if tmax < segments[0]['min'] or tmin > segments[-1]['max']:
                return None, None

            if len(segments) == 1:
                return 0, 0

            t_first_ind = None if tmin > segments[0]['max'] else 0
            t_last_ind = None if tmax < segments[-1]['min'] else len(segments)

            for i, seg in enumerate(segments):
                if seg['min'] <= tmin < seg['max']:
                    t_first_ind = i
                if seg['min'] <= tmax <= seg['max']:
                    t_last_ind = i+1
                if t_first_ind is not None and t_last_ind is not None:
                    break
            return t_first_ind, t_last_ind

        # let's determine first/last x & y segments and which is better (fewer)
        x_first_ind, x_last_ind = segment(x, self._segmentation['x'])
        if x_first_ind is None:
            return None, None, 'x'

        y_first_ind, y_last_ind = segment(y, self._segmentation['y'])
        if y_first_ind is None:
            return None, None, 'y'

        if (y_last_ind - y_first_ind) < (x_last_ind - x_first_ind):
            return y_first_ind, y_last_ind, 'y'
        return x_first_ind, x_last_ind, 'x'

    # TODO: remainder of polygon containment methods.
    #   Refactor Polygon. Create MultiPolygon.


class Polygon(object):
    """
    Class for representing a polygon with associated basic functionality.
    It is assumed that the coordinate space is for geographical or pixel coordinates.
    """
    _type = 'Polygon'
    __slots__ = (
        '_x_coords', '_y_coords', '_bounding_box', '_x_segments', '_y_segments',
        '_x_diff', '_y_diff')

    def __init__(self, x_coords, y_coords):
        """

        Parameters
        ----------
        x_coords : numpy.ndarray|list|tuple
        y_coords : numpy.ndarray|list|tuple
        """

        if not isinstance(x_coords, numpy.ndarray):
            x_coords = numpy.array(x_coords, dtype=numpy.float64)
        if not isinstance(y_coords, numpy.ndarray):
            y_coords = numpy.array(y_coords, dtype=numpy.float64)

        if x_coords.dtype.name != y_coords.dtype.name:
            raise ValueError(
                'x_coords and y_coords must have the same data type. Got {} and {}'.format(x_coords.dtype.name, y_coords.dtype.name))

        if x_coords.shape != y_coords.shape:
            raise ValueError(
                'x_coords and y_coords must be the same shape. Got {} and {}'.format(x_coords.shape, y_coords.shape))
        if len(x_coords.shape) != 1:
            raise ValueError(
                'x_coords and y_coords must be one-dimensional. Got shape {}'.format(x_coords.shape))

        x_coords, y_coords = _compress_identical(x_coords, y_coords)

        if (x_coords[0] != x_coords[-1]) or (y_coords[0] != y_coords[-1]):
            x_coords = numpy.hstack((x_coords, x_coords[0]))
            y_coords = numpy.hstack((y_coords, y_coords[0]))

        if x_coords.size < 4:
            raise ValueError(
                'After compressing repeated (in sequence) points and ensuring first and last point are the same, '
                'x_coords and y_coords must have length at least 4 (i.e. a triangle). '
                'Got x_coords={}, y_coords={}.'.format(x_coords, y_coords))

        self._x_coords = x_coords
        self._y_coords = y_coords

        # construct bounding box
        self._bounding_box = numpy.empty((2, 2), dtype=x_coords.dtype)
        self._bounding_box[0, :] = (numpy.min(self._x_coords), numpy.max(self._x_coords))
        self._bounding_box[1, :] = (numpy.min(self._y_coords), numpy.max(self._y_coords))
        # construct a monotonic listing in x of intervals and indices of edges which apply
        self._x_segments = self._construct_segmentation(self._x_coords, self._y_coords)
        # construct a monotonic listing in y of intervals and indices of edges which apply
        self._y_segments = self._construct_segmentation(self._y_coords, self._x_coords)
        # construct diffs
        self._x_diff = self._x_coords[1:] - self._x_coords[:-1]
        self._y_diff = self._y_coords[1:] - self._y_coords[:-1]

    @classmethod
    def correct_orientation(cls, x_coords, y_coords):
        """
        Class method to produce polygon with counter-clockwise orientation

        Parameters
        ----------
        x_coords : numpy.ndarray|list|tuple
        y_coords : numpy.ndarray|list|tuple

        Returns
        -------
        Polygon
        """

        out = cls(x_coords, y_coords)
        area = out.get_area()
        xs = out._x_coords
        ys = out._y_coords
        if area >= 0:  # NB: area 0 is degenerate, it's not clear how orientation is defined
            return out
        else:
            return cls(xs[::-1], ys[::-1])

    @staticmethod
    def _construct_segmentation(coords, o_coords):
        def overlap(fst, lst, segment):
            if fst == lst and fst == segment['min']:
                return True
            if segment['max'] <= fst or lst <= segment['min']:
                return False
            return True

        def do_min_val_value(segment, val1, val2):
            segment['min_value'] = min(val1, val2, segment['min_value'])
            segment['max_value'] = max(val1, val2, segment['max_value'])

        inds = numpy.argsort(coords[:-1])
        segments = []
        beg_val = coords[inds[0]]
        val = None
        for ind in inds[1:]:
            val = coords[ind]
            if val > beg_val:  # make a new segment
                segments.append({'min': beg_val, 'max': val, 'inds': [], 'min_value': numpy.inf, 'max_value': -numpy.inf})
                beg_val = val
        else:
            # it may have ended without appending the segment
            if val > beg_val:
                segments.append({'min': beg_val, 'max': val, 'inds': [], 'min_value': numpy.inf, 'max_value': -numpy.inf})
        del beg_val, val

        # now, let's populate the inds lists and min/max_values elements
        for i, (beg_value, end_value, ocoord1, ocoord2) in enumerate(zip(coords[:-1], coords[1:], o_coords[:-1], o_coords[1:])):
            first, last = (beg_value, end_value) if beg_value <= end_value else (end_value, beg_value)
            # check all the segments for overlap
            for j, seg in enumerate(segments):
                if overlap(first, last, seg):
                    seg['inds'].append(i)
                    do_min_val_value(seg, ocoord1, ocoord2)

        return tuple(segments)

    @property
    def bounding_box(self):  # type: () -> numpy.ndarray
        """
        numpy.ndarray: The bounding box of the form [[x_min, x_max], [y_min, y_max]].
        *Note that could be extremely misleading for a lat/lon polygon crossing
        the boundary of discontinuity and/or surrounding a pole.*
        """

        return self._bounding_box

    def get_area(self):
        """
        Gets the area of the polygon. If a polygon is self-intersecting, then this
        result may be pathological. If this is positive, then the orientation is
        counter-clockwise. If this is negative, then the orientation is clockwise.
        If zero, then this polygon is degenerate.

        Returns
        -------
        float
        """

        return float(0.5*numpy.sum(self._x_coords[:-1]*self._y_coords[1:] - self._x_coords[1:]*self._y_coords[:-1]))

    def get_centroid(self):
        """
        Gets the centroid of the polygon - note that this may not actually lie in
        the polygon interior for non-convex polygon. This will result in an undefined value
        if the polygon is degenerate.

        Returns
        -------
        numpy.ndarray
        """

        arr = self._x_coords[:-1]*self._y_coords[1:] - self._x_coords[1:]*self._y_coords[:-1]
        area = 0.5*numpy.sum(arr)  # signed area
        x = numpy.sum(0.5*(self._x_coords[:-1] + self._x_coords[1:])*arr)
        y = numpy.sum(0.5*(self._y_coords[:-1] + self._y_coords[1:])*arr)
        return numpy.array([x, y], dtype=numpy.float64)/(3*area)

    def _contained_segment_data(self, x, y):
        """
        This is a helper function for the polygon containment effort.
        This determines whether the x or y segmentation should be utilized, and
        the details for doing so.

        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray

        Returns
        -------
        (int|None, int|None, str)
            the segment index start (inclusive), the segment index end (exclusive),
            and "x" or "y" for which segmentation is better.
        """

        def segment(coord, segments):
            tmin = coord.min()
            tmax = coord.max()

            if tmax < segments[0]['min'] or tmin > segments[-1]['max']:
                return None, None

            if len(segments) == 1:
                return 0, 0

            t_first_ind = None if tmin > segments[0]['max'] else 0
            t_last_ind = None if tmax < segments[-1]['min'] else len(segments)

            for i, seg in enumerate(segments):
                if seg['min'] <= tmin < seg['max']:
                    t_first_ind = i
                if seg['min'] <= tmax <= seg['max']:
                    t_last_ind = i+1
                if t_first_ind is not None and t_last_ind is not None:
                    break
            return t_first_ind, t_last_ind

        # let's determine first/last x & y segments and which is better (fewer)
        x_first_ind, x_last_ind = segment(x, self._x_segments)
        if x_first_ind is None:
            return None, None, 'x'

        y_first_ind, y_last_ind = segment(y, self._y_segments)
        if y_first_ind is None:
            return None, None, 'y'

        if (y_last_ind - y_first_ind) < (x_last_ind - x_first_ind):
            return y_first_ind, y_last_ind, 'y'
        return x_first_ind, x_last_ind, 'x'

    def _contained_do_segment(self, x, y, segment, direction):
        """
        Helper function for polygon containment effort.

        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray
        segment : dict
        direction : str

        Returns
        -------
        numpy.ndarray
        """

        # we require that all these points are relevant to this slice
        in_poly = numpy.zeros(x.shape, dtype=numpy.bool)
        crossing_counts = numpy.zeros(x.shape, dtype=numpy.int32)
        indices = segment['inds']

        for i in indices:
            if direction == 'x' and self._x_coords[i] == self._x_coords[i+1]:
                # we are segmented horizontally and processing vertically.
                # This is a vertical line - only consider inclusion.
                y_min = min(self._y_coords[i], self._y_coords[i + 1])
                y_max = max(self._y_coords[i], self._y_coords[i + 1])
                # points on the edge are included
                in_poly[(x == self._x_coords[i]) & (y_min <= y) & (y <= y_max)] = True
            elif direction == 'y' and self._y_coords[i] == self._y_coords[i+1]:
                # we are segmented vertically and processing horizontally.
                # This is a horizontal line - only consider inclusion.
                x_min = min(self._x_coords[i], self._x_coords[i + 1])
                x_max = max(self._x_coords[i], self._x_coords[i + 1])
                # points on the edge are included
                in_poly[(y == self._y_coords[i]) & (x_min <= x) & (x <= x_max)] = True
            else:
                nx, ny = self._y_diff[i],  -self._x_diff[i]
                crossing = (x - self._x_coords[i])*nx + (y - self._y_coords[i])*ny
                # dot product of vector connecting (x, y) to segment vertex with normal vector of segment
                crossing_counts[crossing > 0] += 1  # positive crossing number
                crossing_counts[crossing < 0] -= 1  # negative crossing number
                # points on the edge are included
                in_poly[(crossing == 0)] = True
        in_poly |= (crossing_counts != 0)
        return in_poly

    def _contained(self, x, y):
        """
        Helper method for polygon inclusion.
        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        out = numpy.zeros(x.shape, dtype=numpy.bool)

        ind_beg, ind_end, direction = self._contained_segment_data(x, y)
        if ind_beg is None:
            return out  # it missed the whole bounding box

        for index in range(ind_beg, ind_end):
            if direction == 'x':
                seg = self._x_segments[index]
                mask = ((x >= seg['min']) & (x <= seg['max']) & (y >= seg['min_value']) & (y <= seg['max_value']))
            else:
                seg = self._y_segments[index]
                mask = ((y >= seg['min']) & (y <= seg['max']) & (x >= seg['min_value']) & (x <= seg['max_value']))
            if numpy.any(mask):
                out[mask] = self._contained_do_segment(x[mask], y[mask], seg, direction)
        return out

    def contained(self, pts_x, pts_y, block_size=None):
        """
        Determines inclusion of the given points in the interior of the polygon.
        The methodology here is based on the Jordan curve theorem approach.

        ** Warning: This method may provide erroneous results for a lat/lon polygon
        crossing the bound of discontinuity and/or surrounding a pole.**

        * Note: If the points constitute an x/y grid, then the grid contained method will
        be much more performant.*

        Parameters
        ----------
        pts_x : numpy.ndarray|list|tuple|float|int
        pts_y : numpy.ndarray|list|tuple|float|int
        block_size : None|int
            If provided, processing block size. The minimum value used will be
            50,000.

        Returns
        -------
        numpy.ndarray|bool
            boolean array indicating inclusion.
        """

        if not isinstance(pts_x, numpy.ndarray):
            pts_x = numpy.array(pts_x, dtype=numpy.float64)
        if not isinstance(pts_y, numpy.ndarray):
            pts_y = numpy.array(pts_y, dtype=numpy.float64)

        if pts_x.shape != pts_y.shape:
            raise ValueError(
                'pts_x and pts_y must be the same shape. Got {} and {}'.format(pts_x.shape, pts_y.shape))

        o_shape = pts_x.shape

        if len(o_shape) == 0:
            pts_x = numpy.reshape(pts_x, (1, ))
            pts_y = numpy.reshape(pts_y, (1, ))
        else:
            pts_x = numpy.reshape(pts_x, (-1, ))
            pts_y = numpy.reshape(pts_y, (-1, ))

        if block_size is not None:
            block_size = int(block_size)
            block_size = max(50000, block_size)

        if block_size is None or pts_x.size <= block_size:
            in_poly = self._contained(pts_x, pts_y)
        else:
            in_poly = numpy.zeros(pts_x.shape, dtype=numpy.bool)
            start_block = 0
            while start_block < pts_x.size:
                end_block = min(start_block+block_size, pts_x.size)
                in_poly[start_block:end_block] = self._contained(
                    pts_x[start_block:end_block], pts_y[start_block:end_block])
                start_block = end_block

        if len(o_shape) == 0:
            return in_poly[0]
        else:
            return numpy.reshape(in_poly, o_shape)

    def grid_contained(self, grid_x, grid_y):
        """
        Determines inclusion of a coordinate grid inside the polygon. The coordinate
        grid is defined by the two one-dimensional coordinate arrays `grid_x` and `grid_y`.

        Parameters
        ----------
        grid_x : nuumpy.ndarray
        grid_y : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            boolean mask for point inclusion of the grid. Output is of shape
            `(grid_x.size, grid_y.size)`.
        """

        if not isinstance(grid_x, numpy.ndarray):
            grid_x = numpy.array(grid_x, dtype=numpy.float64)
        if not isinstance(grid_y, numpy.ndarray):
            grid_y = numpy.array(grid_y, dtype=numpy.float64)
        if len(grid_x.shape) != 1 or len(grid_y.shape) != 1:
            raise ValueError('grid_x and grid_y must be one dimensional.')
        if numpy.any((grid_x[1:] - grid_x[:-1]) <= 0):
            raise ValueError('grid_x must be monotonically increasing')
        if numpy.any((grid_y[1:] - grid_y[:-1]) <= 0):
            raise ValueError('grid_y must be monotonically increasing')

        out = numpy.zeros((grid_x.size, grid_y.size), dtype=numpy.bool)
        first_ind, last_ind, direction = self._contained_segment_data(grid_x, grid_y)
        if first_ind is None:
            return out  # it missed the whole bounding box

        first_ind, last_ind, direction = self._contained_segment_data(grid_x, grid_y)
        x_inds = numpy.arange(grid_x.size)
        y_inds = numpy.arange(grid_y.size)
        for index in range(first_ind, last_ind):
            if direction == 'x':
                seg = self._x_segments[index]
                start_x = x_inds[grid_x >= seg['min']].min()
                end_x = x_inds[grid_x <= seg['max']].max() + 1
                start_y = y_inds[grid_y >= seg['min_value']].min() if grid_y[-1] >= seg['min_value'] else None
                end_y = y_inds[grid_y <= seg['max_value']].max() + 1 if start_y is not None else None
            else:
                seg = self._y_segments[index]
                start_x = x_inds[grid_x >= seg['min_value']].min() if grid_x[-1] >= seg['min_value'] else None
                end_x = x_inds[grid_x <= seg['max_value']].max() + 1 if start_x is not None else None
                start_y = y_inds[grid_y >= seg['min']].min()
                end_y = y_inds[grid_y <= seg['max']].max() + 1

            if start_x is not None and end_x is not None and start_y is not None and end_y is not None:
                y_temp, x_temp = numpy.meshgrid(grid_y[start_y:end_y], grid_x[start_x:end_x], indexing='xy')
                out[start_x:end_x, start_y:end_y] = self._contained_do_segment(x_temp, y_temp, seg, direction)
        return out
