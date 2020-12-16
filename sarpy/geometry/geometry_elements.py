# -*- coding: utf-8 -*-
"""
This module provides basic geometry elements generally geared towards (geo)json usage.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from collections import OrderedDict
from uuid import uuid4
from typing import Union, List, Tuple, Dict, Callable, Any
import json
import logging
import sys

import numpy

string_types = str
if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    string_types = basestring

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


def _validate_contain_arguments(pts_x, pts_y):
    # helper method for Polygon functionality
    if not isinstance(pts_x, numpy.ndarray):
        pts_x = numpy.array(pts_x, dtype=numpy.float64)
    if not isinstance(pts_y, numpy.ndarray):
        pts_y = numpy.array(pts_y, dtype=numpy.float64)

    if pts_x.shape != pts_y.shape:
        raise ValueError(
            'pts_x and pts_y must be the same shape. Got {} and {}'.format(pts_x.shape, pts_y.shape))
    return pts_x, pts_y


def _validate_grid_contain_arguments(grid_x, grid_y):
    # helper method for Polygon functionality
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

    return grid_x, grid_y


def _get_kml_coordinate_string(coordinates, transform):
    # type: (numpy.ndarray, Union[None, Callable]) -> str
    def identity(x):
        return x

    if transform is None:
        transform = identity

    if coordinates.ndim == 1:
        return '{0:0.9f},{1:0.9f}'.format(*transform(coordinates)[:2])
    return ' '.join(
        '{0:0.9f},{1:0.9f}'.format(*el[:2]) for el in transform(coordinates))


class _Jsonable(object):
    """
    Abstract class for json serializability.
    """
    _type = '_Jsonable'

    @property
    def type(self):
        """
        The type identifier.

        Returns
        -------
        str
        """

        return self._type

    @classmethod
    def from_dict(cls, the_json):
        """
        Deserialize from json.

        Parameters
        ----------
        the_json : Dict

        Returns
        -------

        """

        raise NotImplementedError

    def to_dict(self, parent_dict=None):
        """
        Deserialize from json.

        Parameters
        ----------
        parent_dict : None|Dict

        Returns
        -------
        Dict
        """

        raise NotImplementedError

    def __str__(self):
        return '{}(**{})'.format(self.__class__.__name__, json.dumps(self.to_dict(), indent=1))

    def __repr__(self):
        return '{}(**{})'.format(self.__class__.__name__, self.to_dict())

    def copy(self):
        """
        Make a deep copy of the item.

        Returns
        -------

        """

        the_type = self.__class__
        return the_type.from_dict(self.to_dict())


class Feature(_Jsonable):
    """
    Generic feature class - basic geojson functionality. Should generally be extended
    to coherently handle properties for specific use case.
    """

    __slots__ = ('_uid', '_geometry', '_properties')
    _type = 'Feature'

    def __init__(self, uid=None, geometry=None, properties=None):
        self._geometry = None
        self._properties = None

        self.geometry = geometry
        self.properties = properties

        if uid is None:
            uid = properties.get('identifier', None)

        if uid is None:
            self._uid = str(uuid4())
        elif not isinstance(uid, string_types):
            raise TypeError('uid must be a string.')
        else:
            self._uid = uid

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
    def geometry(self):
        """
        The geometry object.

        Returns
        -------
        Geometry
        """

        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        if geometry is None:
            self._geometry = None
        elif isinstance(geometry, Geometry):
            self._geometry = geometry
        elif isinstance(geometry, dict):
            self._geometry = Geometry.from_dict(geometry)
        else:
            raise TypeError('geometry must be an instance of Geometry base class')

    @property
    def properties(self):  # type: () -> _Jsonable
        """
        The properties.

        Returns
        -------

        """

        return self._properties

    @properties.setter
    def properties(self, properties):
        self._properties = properties

    @classmethod
    def from_dict(cls, the_json):
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('Feature cannot be constructed from {}'.format(the_json))
        return cls(uid=the_json.get('id', None),
                   geometry=the_json.get('geometry', None),
                   properties=the_json.get('properties', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['id'] = self.uid
        if self.geometry is None:
            parent_dict['geometry'] = None
        else:
            parent_dict['geometry'] = self.geometry.to_dict()
        parent_dict['properties'] = self.properties
        return parent_dict

    def add_to_kml(self, doc, coord_transform, parent=None):
        """
        Add this feature to the kml document. **Note that coordinates or transformed
        coordinates are assumed to be WGS-84 coordinates in longitude, latitude order.**
        Currently only the first two (i.e. longitude and latitude) are used in
        this export.

        Parameters
        ----------
        doc : sarpy.io.kml.Document
        coord_transform : None|callable
            If callable, the the transform will be applied to the coordinates before
            adding to the document.
        parent : None|minidom.Element
            The parent node.

        Returns
        -------
        None
        """

        params = {}
        if self.uid is not None:
            params['id'] = self.uid
        if self.properties is not None:
            params['description'] = str(self.properties)
        placemark = doc.add_container(par=parent, typ='Placemark', **params)
        if self.geometry is not None:
            self.geometry.add_to_kml(doc, placemark, coord_transform)


class FeatureCollection(_Jsonable):
    """
    Generic FeatureCollection class - basic geojson functionality. Should generally be
    extended to coherently handle specific Feature extension.
    """

    __slots__ = ('_features', '_feature_dict')
    _type = 'FeatureCollection'

    def __init__(self, features=None):
        self._features = None
        self._feature_dict = None
        if features is not None:
            self.features = features

    def __len__(self):
        if self._features is None:
            return 0
        return len(self._features)

    def __getitem__(self, item):
        # type: (Any) -> Union[Feature, List[Feature]]
        if isinstance(item, string_types):
            return self._feature_dict[item]
        return self._features[item]

    @property
    def features(self):
        """
        The features list.

        Returns
        -------
        List[Feature]
        """

        return self._features

    @features.setter
    def features(self, features):
        if features is None:
            self._features = None
            self._feature_dict = None
            return

        if not isinstance(features, list):
            raise TypeError('features must be a list of features. Got {}'.format(type(features)))

        for entry in features:
            if isinstance(entry, Feature):
                self.add_feature(entry)
            elif isinstance(entry, dict):
                self.add_feature(Feature.from_dict(entry))
            else:
                raise TypeError(
                    'Entries of features are required to be instances of Feature or '
                    'dictionary to be deserialized. Got {}'.format(type(entry)))

    @classmethod
    def from_dict(cls, the_json):
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('FeatureCollection cannot be constructed from {}'.format(the_json))
        return cls(features=the_json['features'])

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        if self._features is None:
            parent_dict['features'] = None
        else:
            parent_dict['features'] = [entry.to_dict() for entry in self._features]
        return parent_dict

    def add_feature(self, feature):
        """
        Add a feature.

        Parameters
        ----------
        feature : Feature

        Returns
        -------
        None
        """

        if not isinstance(feature, Feature):
            raise TypeError('This requires a Feature instance, got {}'.format(type(feature)))

        if self._features is None:
            self._feature_dict = {feature.uid: 0}
            self._features = [feature, ]
        else:
            self._feature_dict[feature.uid] = len(self._features)
            self._features.append(feature)

    def export_to_kml(self, file_name, coord_transform=None, **params):
        """
        Export to a kml document. **Note that underlying geometry coordinates or
        transformed coordinates are assumed in longitude, latitude order.**
        Currently only the first two (i.e. longitude and latitude) are used in this export.

        Parameters
        ----------
        file_name : str|zipfile.ZipFile|file like
        coord_transform : None|callable
            The coordinate transform function.
        params : dict

        Returns
        -------
        None
        """

        from sarpy.io.kml import Document as KML_Document

        with KML_Document(file_name=file_name, **params) as doc:
            if self.features is not None:
                for feat in self.features:
                    feat.add_to_kml(doc, coord_transform)


class Geometry(_Jsonable):
    """
    Abstract Geometry base class.
    """
    _type = 'Geometry'

    @classmethod
    def from_dict(cls, geometry):
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
            obj = GeometryCollection.from_dict(geometry)
            return obj
        else:
            obj = GeometryObject.from_dict(geometry)
            return obj

    def to_dict(self, parent_dict=None):
        raise NotImplementedError

    def add_to_kml(self, doc, parent, coord_transform):
        """
        Add the geometry to the kml document. **Note that coordinates or transformed
        coordinates are assumed in longitude, latitude order.**

        Parameters
        ----------
        doc : sarpy.io.kml.Document
        parent : xml.dom.minidom.Element
        coord_transform : None|callable

        Returns
        -------
        None
        """

        raise NotImplementedError

    def apply_projection(self, proj_method):
        """
        Gets a new version after applying a transform method.

        Parameters
        ----------
        proj_method : callable

        Returns
        -------
        Geometry
        """

        raise NotImplementedError


class GeometryCollection(Geometry):
    """
    Geometry collection - following the geojson structure
    """

    __slots__ = ('_geometries', )
    _type = 'GeometryCollection'

    def __init__(self, geometries=None):
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
        # type: () -> Union[List[Geometry], None]
        """
        None|List[Geometry]: The geometry collection.
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
            logging.warning('GeometryCollection should contain a list of geometries with length greater than 1.')
        for entry in geometries:
            if not isinstance(entry, Geometry):
                raise TypeError(
                    'geometries must be a list of Geometry objects. Got an element of type {}'.format(type(entry)))

    @classmethod
    def from_dict(cls, geometry):
        # type: (Union[None, Dict]) -> GeometryCollection
        typ = geometry.get('type', None)
        if typ != cls._type:
            raise ValueError('GeometryCollection cannot be constructed from {}'.format(geometry))
        geometries = []
        for entry in geometry['geometries']:
            if isinstance(entry, Geometry):
                geometries.append(entry)
            elif isinstance(entry, dict):
                geometries.append(Geometry.from_dict(entry))
            else:
                raise TypeError(
                    'The geometries attribute must contain either a Geometry or json serialization of a Geometry. '
                    'Got an entry of type {}'.format(type(entry)))
        return cls(geometries)

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        if self.geometries is None:
            parent_dict['geometries'] = None
        else:
            parent_dict['geometries'] = [entry.to_dict() for entry in self.geometries]
        return parent_dict

    def add_to_kml(self, doc, parent, coord_transform):
        if self.geometries is None:
            return
        multigeometry = doc.add_multi_geometry(parent)
        for geometry in self.geometries:
            if geometry is not None:
                geometry.add_to_kml(doc, multigeometry, coord_transform)

    def apply_projection(self, proj_method):
        """
        Gets a new version after applying a transform method.

        Parameters
        ----------
        proj_method : callable

        Returns
        -------
        GeometryObject
        """

        if self.geometries is None:
            return GeometryCollection()
        return GeometryCollection(geometries=[geom.apply_projection(proj_method) for geom in self.geometries])


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
        List
        """

        raise NotImplementedError

    def get_bbox(self):
        """
        Get the bounding box list.

        Returns
        -------
        List
        """

        raise NotImplementedError

    @classmethod
    def from_dict(cls, geometry):
        # type: (Dict) -> GeometryObject
        typ = geometry.get('type', None)
        if typ is None:
            raise ValueError('Poorly formed json for GeometryObject {}'.format(geometry))
        elif typ == 'Point':
            return Point(coordinates=geometry['coordinates'])
        elif typ == 'MultiPoint':
            return MultiPoint(coordinates=geometry['coordinates'])
        elif typ == 'LineString':
            return LineString(coordinates=geometry['coordinates'])
        elif typ == 'MultiLineString':
            return MultiLineString(coordinates=geometry['coordinates'])
        elif typ == 'Polygon':
            return Polygon(coordinates=geometry['coordinates'])
        elif typ == 'MultiPolygon':
            return MultiPolygon(coordinates=geometry['coordinates'])
        else:
            raise ValueError('Unknown type {} for GeometryObject from json {}'.format(typ, geometry))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['coordinates'] = self.get_coordinate_list()
        return parent_dict

    def add_to_kml(self, doc, parent, coord_transform):
        raise NotImplementedError

    def apply_projection(self, proj_method):
        """
        Gets a new version after applying a transform method.

        Parameters
        ----------
        proj_method : callable

        Returns
        -------
        GeometryObject
        """

        raise NotImplementedError


class Point(GeometryObject):
    """
    A geometric point.
    """

    __slots__ = ('_coordinates', )
    _type = 'Point'

    def __init__(self, coordinates=None):
        """

        Parameters
        ----------
        coordinates : None|numpy.ndarray|List[float]|Point
        """

        self._coordinates = None
        if coordinates is not None:
            self.coordinates = coordinates

    @property
    def coordinates(self):
        """
        numpy.ndarray: The coordinate array.
        """

        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        # type: (Union[None, List, Tuple, numpy.ndarray]) -> None
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

    def get_bbox(self):
        if self._coordinates is None:
            return None
        else:
            out = self._coordinates.tolist()
            out.extend(self._coordinates.tolist())
            return out

    def get_coordinate_list(self):
        if self._coordinates is None:
            return None
        else:
            return self._coordinates.tolist()

    @classmethod
    def from_dict(cls, geometry):
        # type: (Dict) -> Point
        if not geometry.get('type', None) == cls._type:
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])

    def add_to_kml(self, doc, parent, coord_transform):
        if self.coordinates is None:
            return
        doc.add_point(_get_kml_coordinate_string(self.coordinates, coord_transform), par=parent)

    def apply_projection(self, proj_method):
        # type: (callable) -> Point
        return Point(coordinates=proj_method(self._coordinates))


class MultiPoint(GeometryObject):
    """
    A collection of geometric points.
    """

    _type = 'MultiPoint'
    __slots__ = ('_points', )

    def __init__(self, coordinates=None):
        """

        Parameters
        ----------
        coordinates : None|numpy.ndarray|List[float]|List[Point]|MultiPoint
        """

        self._points = None
        if isinstance(coordinates, MultiPoint):
            coordinates = coordinates.get_coordinate_list()
        if coordinates is not None:
            self.points = coordinates

    @property
    def points(self):
        # type: () -> List[Point]
        """
        List[Point]: The point collection.
        """

        return self._points

    @points.setter
    def points(self, points):
        if points is None:
            self._points = None
        if isinstance(points, numpy.ndarray):
            points = points.tolist()
        if not isinstance(points, list):
            raise TypeError(
                'Multipoint requires that points is None or a list of points. '
                'Got type {}'.format(type(points)))
        pts = []
        for entry in points:
            if isinstance(entry, Point):
                pts.append(entry)
            else:
                pts.append(Point(coordinates=entry))
        self._points = pts

    def get_bbox(self):
        if self._points is None:
            return None
        else:
            # create our output space
            siz = max(point.coordinates.size for point in self.points)
            mins = [None, ]*siz
            maxs = [None, ]*siz

            for element in self.get_coordinate_list():
                for i, entry in enumerate(element):
                    if mins[i] is None or (entry < mins[i]):
                        mins[i] = entry
                    if maxs[i] is None or (entry > maxs[i]):
                        maxs[i] = entry
            return mins.extend(maxs)

    def get_coordinate_list(self):
        if self._points is None:
            return None
        return [point.get_coordinate_list() for point in self._points]

    @classmethod
    def from_dict(cls, geometry):
        # type: (Dict) -> MultiPoint
        if not geometry.get('type', None) == cls._type:
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])

    def add_to_kml(self, doc, parent, coord_transform):
        if self._points is None:
            return
        multigeometry = doc.add_multi_geometry(parent)
        for geometry in self._points:
            if geometry is not None:
                geometry.add_to_kml(doc, multigeometry, coord_transform)

    def apply_projection(self, proj_method):
        # type: (callable) -> MultiPoint
        return MultiPoint(coordinates=[pt.apply_projection(proj_method) for pt in self.points])


class LineString(GeometryObject):
    """
    A geometric line.
    """

    __slots__ = ('_coordinates', )
    _type = 'LineString'

    def __init__(self, coordinates=None):
        """

        Parameters
        ----------
        coordinates : None|numpy.ndarray|List[float]|LineString|LinearRing
        """

        self._coordinates = None
        if isinstance(coordinates, (LineString, LinearRing)):
            coordinates = coordinates.get_coordinate_list()
        if coordinates is not None:
            self.coordinates = coordinates

    @property
    def coordinates(self):
        # type: () -> numpy.ndarray
        """
        numpy.ndarray: The coordinate array.
        """

        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        # type: (Union[None, List, Tuple, numpy.ndarray]) -> None
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
            logging.error(
                'LineString coordinates should consist of at least 2 points. '
                'Got shape {}'.format(coordinates.shape))
        coordinates = _compress_identical(coordinates)
        if coordinates.shape[0] < 2:
            logging.error(
                'coordinates should consist of at least 2 points after suppressing '
                'consecutive repeated points. Got shape {}'.format(coordinates.shape))
        self._coordinates = coordinates

    def get_bbox(self):
        if self._coordinates is None:
            return None
        else:
            mins = numpy.min(self.coordinates, axis=0)
            maxs = numpy.min(self.coordinates, axis=0)
            return mins.tolist().extend(maxs.tolist())

    def get_coordinate_list(self):
        if self._coordinates is None:
            return None
        else:
            return self._coordinates.tolist()

    @classmethod
    def from_dict(cls, geometry):
        # type: (dict) -> LineString
        if not geometry.get('type', None) == cls._type:
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])

    def get_length(self):
        """
        Gets the length of the line.

        Returns
        -------
        None|float
        """

        if self._coordinates is None:
            return None
        diffs = self._coordinates[1:, :] - self._coordinates[:-1, :]
        return float(numpy.sum(numpy.sqrt(diffs[:, 0]*diffs[:, 0] + diffs[:, 1]*diffs[:, 1])))

    def add_to_kml(self, doc, parent, coord_transform):
        if self.coordinates is None:
            return
        doc.add_line_string(_get_kml_coordinate_string(self.coordinates, coord_transform), par=parent)

    def apply_projection(self, proj_method):
        # type: (callable) -> LineString
        return LineString(coordinates=proj_method(self.coordinates))


class MultiLineString(GeometryObject):
    """
    A collection of geometric lines.
    """

    __slots__ = ('_lines', )
    _type = 'MultiLineString'

    def __init__(self, coordinates=None):
        """

        Parameters
        ----------
        coordinates : None|List[numpy.ndarray]|List[List[int]]|List[LineString]|MultiLineString
        """

        self._lines = None
        if isinstance(coordinates, MultiLineString):
            coordinates = coordinates.get_coordinate_list()
        if coordinates is not None:
            self.lines = coordinates

    @property
    def lines(self):
        # type: () -> List[LineString]
        """
        List[LineString]: The line collection.
        """

        return self._lines

    @lines.setter
    def lines(self, lines):
        if lines is None:
            self._lines = None
            return
        if not isinstance(lines, list):
            raise TypeError(
                'MultiLineString requires that lines is None or a list of LineStrings. '
                'Got type {}'.format(type(lines)))
        lins = []
        for entry in lines:
            if isinstance(entry, LineString):
                lins.append(entry)
            else:
                lins.append(LineString(coordinates=entry))
        self._lines = lins

    def get_bbox(self):
        if self._lines is None:
            return None
        else:
            siz = max(line.coordinates.shape[1] for line in self.lines)
            mins = [None, ]*siz
            maxs = [None, ]*siz
            for line in self.lines:
                t_bbox = line.get_bbox()
                for i, entry in enumerate(t_bbox):
                    if mins[i] is None or entry < mins[i]:
                        mins[i] = entry
                    if maxs[i] is None or entry > maxs[i]:
                        maxs[i] = entry
            return mins.extend(maxs)

    def get_coordinate_list(self):
        if self._lines is None:
            return None
        return [line.get_coordinate_list() for line in self._lines]

    @classmethod
    def from_dict(cls, geometry):
        # type: (Dict) -> MultiLineString
        if not geometry.get('type', None) == cls._type:
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])

    def get_length(self):
        """
        Gets the length of the lines.

        Returns
        -------
        None|float
        """

        if self._lines is None:
            return None
        return sum(entry.get_length() for entry in self._lines)

    def add_to_kml(self, doc, parent, coord_transform):
        if self._lines is None:
            return
        multigeometry = doc.add_multi_geometry(parent)
        for geometry in self._lines:
            if geometry is not None:
                geometry.add_to_kml(doc, multigeometry, coord_transform)

    def apply_projection(self, proj_method):
        # type: (callable) -> MultiLineString
        return MultiLineString(coordinates=[line.apply_projection(proj_method) for line in self.lines])


class LinearRing(LineString):
    """
    This is not directly a valid geojson member, but plays the role of a single
    polygonal element, and is only used as a Polygon constituent.
    """
    __slots__ = ('_coordinates', '_diffs', '_bounding_box', '_segmentation')
    _type = 'LinearRing'

    def __init__(self, coordinates=None):
        """

        Parameters
        ----------
        coordinates : None|numpy.ndarray|List[float]|LinearRing|LineString
        """

        self._coordinates = None
        self._diffs = None
        self._bounding_box = None
        self._segmentation = None
        if isinstance(coordinates, (LineString, LinearRing)):
            coordinates = coordinates.get_coordinate_list()
        super(LinearRing, self).__init__(coordinates)

    def get_coordinate_list(self):
        if self._coordinates is None:
            return None
        else:
            return self._coordinates.tolist()

    def reverse_orientation(self):
        if self._coordinates is None:
            return
        self.coordinates = self._coordinates[::-1, :]

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

    def get_perimeter(self):
        """
        Gets the perimeter of the linear ring.

        Returns
        -------
        float
        """

        return self.get_length()

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

    @property
    def coordinates(self):
        """
        Gets the coordinates array.

        Returns
        -------
        numpy.ndarray
        """

        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self.set_coordinates(coordinates)

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
            logging.error('coordinates must consist of at least 3 points. '
                          'Got shape {}'.format(coordinates.shape))
        coordinates = _compress_identical(coordinates)
        if (coordinates[0, 0] != coordinates[-1, 0]) or \
                (coordinates[0, 1] != coordinates[-1, 1]):
            coordinates = numpy.vstack((coordinates, coordinates[0, :]))
        if coordinates.shape[0] < 4:
            logging.error(
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
                return 0, 1

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

        if (y_last_ind - y_first_ind) <= (x_last_ind - x_first_ind):
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
            if direction == 'x' and self._coordinates[i, 0] == self._coordinates[i+1, 0]:
                # we are segmented horizontally and processing vertically.
                # This is a vertical line - only consider inclusion.
                y_min = min(self._coordinates[i, 1], self._coordinates[i+1, 1])
                y_max = max(self._coordinates[i, 1], self._coordinates[i+1, 1])
                # points on the edge are included
                in_poly[(x == self._coordinates[i, 0]) & (y_min <= y) & (y <= y_max)] = True
            elif direction == 'y' and self._coordinates[i, 1] == self._coordinates[i+1, 1]:
                # we are segmented vertically and processing horizontally.
                # This is a horizontal line - only consider inclusion.
                x_min = min(self._coordinates[i, 0], self._coordinates[i+1, 0])
                x_max = max(self._coordinates[i, 0], self._coordinates[i+1, 0])
                # points on the edge are included
                in_poly[(y == self._coordinates[i, 1]) & (x_min <= x) & (x <= x_max)] = True
            else:
                nx, ny = self._diffs[i, 1],  - self._diffs[i, 0]
                crossing = (x - self._coordinates[i, 0])*nx + (y - self._coordinates[i, 1])*ny
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
                seg = self._segmentation['x'][index]
                mask = ((x >= seg['min']) & (x <= seg['max']) & (y >= seg['min_value']) & (y <= seg['max_value']))
            else:
                seg = self._segmentation['y'][index]
                mask = ((y >= seg['min']) & (y <= seg['max']) & (x >= seg['min_value']) & (x <= seg['max_value']))
            if numpy.any(mask):
                out[mask] = self._contained_do_segment(x[mask], y[mask], seg, direction)
        return out

    def contain_coordinates(self, pts_x, pts_y, block_size=None):
        """
        Determines inclusion of the given points in the interior of the polygon.
        The methodology here is based on the Jordan curve theorem approach.

        ** Warning - This method may provide erroneous results for a lat/lon polygon
        crossing the bound of discontinuity and/or surrounding a pole.**

        Note - If the points constitute an x/y grid, then the grid contained method will
        be much more performant.

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

        pts_x, pts_y = _validate_contain_arguments(pts_x, pts_y)

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
        grid_x : numpy.ndarray
        grid_y : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            boolean mask for point inclusion of the grid. Output is of shape
            `(grid_x.size, grid_y.size)`.
        """

        grid_x, grid_y = _validate_grid_contain_arguments(grid_x, grid_y)

        out = numpy.zeros((grid_x.size, grid_y.size), dtype=numpy.bool)
        first_ind, last_ind, direction = self._contained_segment_data(grid_x, grid_y)
        if first_ind is None:
            return out  # it missed the whole bounding box

        first_ind, last_ind, direction = self._contained_segment_data(grid_x, grid_y)
        x_inds = numpy.arange(grid_x.size)
        y_inds = numpy.arange(grid_y.size)
        for index in range(first_ind, last_ind):
            if direction == 'x':
                seg = self._segmentation['x'][index]
                start_x = x_inds[grid_x >= seg['min']].min()
                end_x = x_inds[grid_x <= seg['max']].max() + 1
                start_y = y_inds[grid_y >= seg['min_value']].min() if grid_y[-1] >= seg['min_value'] else None
                end_y = y_inds[grid_y <= seg['max_value']].max() + 1 if start_y is not None else None
            else:
                seg = self._segmentation['y'][index]
                start_x = x_inds[grid_x >= seg['min_value']].min() if grid_x[-1] >= seg['min_value'] else None
                end_x = x_inds[grid_x <= seg['max_value']].max() + 1 if start_x is not None else None
                start_y = y_inds[grid_y >= seg['min']].min()
                end_y = y_inds[grid_y <= seg['max']].max() + 1

            if start_x is not None and end_x is not None and start_y is not None and end_y is not None:
                y_temp, x_temp = numpy.meshgrid(grid_y[start_y:end_y], grid_x[start_x:end_x], indexing='xy')
                out[start_x:end_x, start_y:end_y] = self._contained_do_segment(x_temp, y_temp, seg, direction)
        return out

    def apply_projection(self, proj_method):
        # type: (callable) -> LinearRing
        return LinearRing(coordinates=proj_method(self.coordinates))


class Polygon(GeometryObject):
    """
    A polygon object consisting of an outer LinearRing, and some collection of
    interior LinearRings representing holes or voids.
    """

    __slots__ = ('_outer_ring', '_inner_rings')
    _type = 'Polygon'

    def __init__(self, coordinates=None):
        """

        Parameters
        ----------
        coordinates : None|List[numpy.ndarray]|List[List[float]]|List[LinearRing]|List[LineString]|Polygon
            The first element is the outer ring, any remaining will be inner rings.
        """

        self._outer_ring = None  # type: Union[None, LinearRing]
        self._inner_rings = None  # type: Union[None, List[LinearRing]]
        if isinstance(coordinates, Polygon):
            coordinates = coordinates.get_coordinate_list()
        if coordinates is None:
            return
        if not isinstance(coordinates, list):
            raise TypeError('coordinates must be a list of linear ring coordinate arrays.')
        if len(coordinates) < 1:
            return
        self.set_outer_ring(coordinates[0])
        for entry in coordinates[1:]:
            self.add_inner_ring(entry)

    @classmethod
    def from_dict(cls, geometry):
        # type: (Dict) -> Polygon
        if not geometry.get('type', None) == cls._type:
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])

    def get_bbox(self):
        if self._outer_ring is None:
            return None
        else:
            return self._outer_ring.get_bbox()

    def get_coordinate_list(self):
        if self._outer_ring is None:
            return None

        out = [self._outer_ring.get_coordinate_list(), ]
        if self._inner_rings is not None:
            for ir in self._inner_rings:
                ir_reversed = LinearRing(ir.coordinates[::-1, :])
                out.append(ir_reversed.get_coordinate_list())
        return out

    def set_outer_ring(self, coordinates):
        """
        Set the outer ring for the Polygon.

        Parameters
        ----------
        coordinates : LinearRing|numpy.ndarray|list

        Returns
        -------
        None
        """

        if coordinates is None:
            self._outer_ring = None
            self._inner_rings = None
            return
        if isinstance(coordinates, (LinearRing, LineString)):
            outer_ring = LinearRing(coordinates=coordinates.coordinates)
        else:
            outer_ring = LinearRing(coordinates=coordinates)
        area = outer_ring.get_area()
        if area == 0:
            logging.warning("The outer ring for this Polygon has zero area. This is likely an error.")
        elif area < 0:
            logging.info(
                "The outer ring of a Polygon is required to have counter-clockwise orientation. "
                "This outer ring has clockwise orientation, so the orientation will be reversed.")
            outer_ring.reverse_orientation()
        self._outer_ring = outer_ring

    def add_inner_ring(self, coordinates):
        if coordinates is None:
            return
        if self._outer_ring is None:
            raise ValueError('A Polygon cannot have an inner ring with no outer ring defined.')

        if self._inner_rings is None:
            self._inner_rings = []
        if isinstance(coordinates, (LinearRing, LineString)):
            inner_ring = LinearRing(coordinates=coordinates.coordinates)
        else:
            inner_ring = LinearRing(coordinates=coordinates)
        area = inner_ring.get_area()
        if area == 0:
            logging.warning("The defined inner ring for this Polygon has zero area. This is likely an error.")
        elif area < 0:
            inner_ring.reverse_orientation()
        self._inner_rings.append(inner_ring)

    def get_perimeter(self):
        """
        Gets the perimeter of the linear ring.

        Returns
        -------
        None|float
        """

        if self._outer_ring is None:
            return None

        perimeter = self._outer_ring.get_perimeter()
        if self._inner_rings is not None:
            for entry in self._inner_rings:
                perimeter += entry.get_perimeter()
        return perimeter

    def get_area(self):
        """
        Gets the area of the polygon.

        Returns
        -------
        None|float
        """

        if self._outer_ring is None:
            return None

        area = self._outer_ring.get_area()  # positive
        if self._inner_rings is not None:
            for entry in self._inner_rings:
                area -= entry.get_area()  # negative
        return area

    def get_centroid(self):
        """
        Gets the centroid of the outer ring of the polygon - note that this may not actually lie in
        the polygon interior for non-convex polygon. This will result in an undefined value
        if the polygon is degenerate.

        Returns
        -------
        numpy.ndarray
        """

        if self._outer_ring is None:
            return None
        return self._outer_ring.get_centroid()

    def contain_coordinates(self, pts_x, pts_y, block_size=None):
        """
        Determines inclusion of the given points in the interior of the polygon.
        The methodology here is based on the Jordan curve theorem approach.

        ** Warning - This method may provide erroneous results for a lat/lon polygon
        crossing the bound of discontinuity and/or surrounding a pole.**

        Note - If the points constitute an x/y grid, then the grid contained method will
        be much more performant.

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

        pts_x, pts_y = _validate_contain_arguments(pts_x, pts_y)

        if self._outer_ring is None:
            return numpy.zeros(pts_x.shape, dtype=numpy.bool)

        o_shape = pts_x.shape
        in_poly = self._outer_ring.contain_coordinates(pts_x, pts_y, block_size=block_size)
        if self._inner_rings is not None:
            for ir in self._inner_rings:
                in_poly &= ~ir.contain_coordinates(pts_x, pts_y, block_size=block_size)

        if len(o_shape) == 0:
            return in_poly
        else:
            return numpy.reshape(in_poly, o_shape)

    def grid_contained(self, grid_x, grid_y):
        """
        Determines inclusion of a coordinate grid inside the polygon. The coordinate
        grid is defined by the two one-dimensional coordinate arrays `grid_x` and `grid_y`.

        Parameters
        ----------
        grid_x : numpy.ndarray
        grid_y : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            boolean mask for point inclusion of the grid. Output is of shape
            `(grid_x.size, grid_y.size)`.
        """

        grid_x, grid_y = _validate_grid_contain_arguments(grid_x, grid_y)

        if self._outer_ring is None:
            return numpy.zeros((grid_x.size, grid_y.size), dtype=numpy.bool)

        in_poly = self._outer_ring.grid_contained(grid_x, grid_y)
        if self._inner_rings is not None:
            for ir in self._inner_rings:
                in_poly &= ~ir.grid_contained(grid_x, grid_y)
        return in_poly

    def add_to_kml(self, doc, parent, coord_transform):
        if self._outer_ring is None:
            return
        outCoords = _get_kml_coordinate_string(self._outer_ring.coordinates, coord_transform)
        inCoords = []
        if self._inner_rings is not None:
            inCoords = [_get_kml_coordinate_string(ir.coordinates, coord_transform) for ir in self._inner_rings]
        doc.add_polygon(outCoords, inCoords=inCoords, par=parent)

    def apply_projection(self, proj_method):
        # type: (callable) -> Polygon
        coords = [self._outer_ring.apply_projection(proj_method), ]
        if self._inner_rings is not None:
            coords.extend([lr.apply_projection(proj_method) for lr in self._inner_rings])
        return Polygon(coordinates=coords)


class MultiPolygon(GeometryObject):
    """
    A collection of polygon objects.
    """

    __slots__ = ('_polygons', )
    _type = 'MultiPolygon'

    def __init__(self, coordinates=None):
        """

        Parameters
        ----------
        coordinates : None|List[List[List[float]]]|List[Polygon]|MultiPolygon
        """

        self._polygons = None
        if isinstance(coordinates, MultiPolygon):
            coordinates = coordinates.get_coordinate_list()
        if coordinates is not None:
            self.polygons = coordinates

    @property
    def polygons(self):
        # type: () -> List[Polygon]
        """
        List[Polygon]: The polygon collection.
        """

        return self._polygons

    @polygons.setter
    def polygons(self, polygons):
        if polygons is None:
            self._polygons = None
            return

        if not isinstance(polygons, list):
            raise TypeError(
                'MultiPolygon requires the polygons is None or a list of Polygons. '
                'Got type {}'.format(type(polygons)))
        polys = []
        for entry in polygons:
            if isinstance(entry, Polygon):
                polys.append(entry)
            else:
                polys.append(Polygon(coordinates=entry))
        self._polygons = polys

    def get_bbox(self):
        if self._polygons is None:
            return None
        else:
            mins = []
            maxs = []
            for polygon in self.polygons:
                t_bbox = polygon.get_bbox()
                for i, entry in enumerate(t_bbox):
                    if len(mins) < i:
                        mins.append(entry)
                    elif entry < mins[i]:
                        mins[i] = entry
                    if len(maxs) < i:
                        maxs.append(entry)
                    elif entry > maxs[i]:
                        maxs[i] = entry
            return mins.extend(maxs)

    @classmethod
    def from_dict(cls, geometry):
        # type: (Dict) -> MultiPolygon
        if not geometry.get('type', None) == cls._type:
            raise ValueError('Poorly formed json {}'.format(geometry))
        cls(coordinates=geometry['coordinates'])

    def get_coordinate_list(self):
        if self._polygons is None:
            return None
        return [polygon.get_coordinate_list() for polygon in self._polygons]

    def get_perimeter(self):
        """
        Gets the perimeter of the linear ring.

        Returns
        -------
        None|float
        """

        if self._polygons is None:
            return None
        return sum(entry.get_perimeter() for entry in self._polygons)

    def get_area(self):
        """
        Gets the area of the polygon.

        Returns
        -------
        None|float
        """

        if self._polygons is None:
            return None
        return sum(entry.get_area() for entry in self._polygons)

    def contain_coordinates(self, pts_x, pts_y, block_size=None):
        """
        Determines inclusion of the given points in the interior of the polygon.
        The methodology here is based on the Jordan curve theorem approach.

        ** Warning - This method may provide erroneous results for a lat/lon polygon
        crossing the bound of discontinuity and/or surrounding a pole.**

        Note - If the points constitute an x/y grid, then the grid contained method will
        be much more performant.

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

        pts_x, pts_y = _validate_contain_arguments(pts_x, pts_y)

        if self._polygons is None or len(self._polygons) == 0:
            return numpy.zeros(pts_x.shape, dtype=numpy.bool)

        in_poly = self._polygons[0].contain_coordinates(pts_x, pts_y, block_size=block_size)
        for entry in self._polygons[1:]:
            in_poly |= entry.contain_coordinates(pts_x, pts_y, block_size=block_size)
        return in_poly

    def grid_contained(self, grid_x, grid_y):
        """
        Determines inclusion of a coordinate grid inside the polygon. The coordinate
        grid is defined by the two one-dimensional coordinate arrays `grid_x` and `grid_y`.

        Parameters
        ----------
        grid_x : numpy.ndarray
        grid_y : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            boolean mask for point inclusion of the grid. Output is of shape
            `(grid_x.size, grid_y.size)`.
        """

        grid_x, grid_y = _validate_grid_contain_arguments(grid_x, grid_y)

        if self._polygons is None or len(self._polygons) == 0:
            return numpy.zeros((grid_x.size, grid_y.size), dtype=numpy.bool)

        in_poly = self._polygons[0].grid_contained(grid_x, grid_y)
        for entry in self._polygons[1:]:
            in_poly |= entry.grid_contained(grid_x, grid_y)
        return in_poly

    def add_to_kml(self, doc, parent, coord_transform):
        if self._polygons is None:
            return
        multigeometry = doc.add_multi_geometry(parent)
        for geometry in self._polygons:
            if geometry is not None:
                geometry.add_to_kml(doc, multigeometry, coord_transform)

    def apply_projection(self, proj_method):
        # type: (callable) -> MultiPolygon
        return MultiPolygon(coordinates=[poly.apply_projection(proj_method) for poly in self.polygons])
