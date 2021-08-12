"""
The AnnotationsType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from typing import Union, List

import numpy

from .base import DEFAULT_STRICT

from sarpy.io.xml.base import Serializable, create_new_node, create_text_node, \
    get_node_value, find_first_child, find_children
from sarpy.io.xml.descriptors import SerializableDescriptor, SerializableListDescriptor, \
    FloatDescriptor, StringDescriptor, StringListDescriptor
from sarpy.geometry.geometry_elements import Point as PointType, \
    LineString as LineStringType, \
    LinearRing as LinearRingType, \
    Polygon as PolygonType, \
    MultiPoint as MultiPointType, \
    MultiLineString as MultiLineStringType, \
    MultiPolygon as MultiPolygonType

logger = logging.getLogger(__name__)


class ParameterType(Serializable):
    """
    The parameter type.
    """

    _fields = ('ParameterName', 'Value')
    _required = _fields
    _numeric_format = {'Value': '0.16G'}
    # Descriptor
    ParameterName = StringDescriptor(
        'ParameterName', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    Value = FloatDescriptor(
        'Value', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: float

    def __init__(self, ParameterName=None, Value=None, **kwargs):
        """

        Parameters
        ----------
        ParameterName : str
        Value : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ParameterName = ParameterName
        self.Value = Value
        super(ParameterType, self).__init__(**kwargs)


class ProjectionType(Serializable):
    """
    The projection type.
    """

    _fields = ('ProjectionName', )
    _required = ('ProjectionName', )
    # Descriptor
    ProjectionName = StringDescriptor(
        'ProjectionName', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

    def __init__(self, ProjectionName=None, **kwargs):
        """

        Parameters
        ----------
        ProjectionName : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ProjectionName = ProjectionName
        super(ProjectionType, self).__init__(**kwargs)


class PrimeMeridianType(Serializable):
    """
    The prime meridian location.
    """

    _fields = ('Name', 'Longitude')
    _required = _fields
    _numeric_format = {'Longitude': '0.16G'}
    # Descriptor
    Name = StringDescriptor(
        'Name', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    Longitude = FloatDescriptor(
        'Longitude', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: float

    def __init__(self, Name=None, Longitude=None, **kwargs):
        """

        Parameters
        ----------
        Name : str
        Longitude : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.Longitude = Longitude
        super(PrimeMeridianType, self).__init__(**kwargs)


class SpheroidType(Serializable):
    """

    """
    _fields = ('SpheroidName', 'SemiMajorAxis', 'InverseFlattening')
    _required = _fields
    _numeric_format = {'SemiMajorAxis': '0.16G', 'InverseFlattening': '0.16G'}
    # Descriptor
    SpheroidName = StringDescriptor(
        'SpheroidName', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    SemiMajorAxis = FloatDescriptor(
        'SemiMajorAxis', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: float
    InverseFlattening = FloatDescriptor(
        'InverseFlattening', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: float

    def __init__(self, SpheroidName=None, SemiMajorAxis=None, InverseFlattening=None, **kwargs):
        """

        Parameters
        ----------
        SpheroidName : str
        SemiMajorAxis : float
        InverseFlattening : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SpheroidName = SpheroidName
        self.SemiMajorAxis = SemiMajorAxis
        self.InverseFlattening = InverseFlattening
        super(SpheroidType, self).__init__(**kwargs)


class DatumType(Serializable):
    """

    """
    _fields = ('Spheroid', )
    _required = ('Spheroid', )
    # Descriptor
    Spheroid = SerializableDescriptor(
        'Spheroid', SpheroidType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: SpheroidType

    def __init__(self, Spheroid=None, **kwargs):
        """

        Parameters
        ----------
        Spheroid : SpheroidType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Spheroid = Spheroid
        super(DatumType, self).__init__(**kwargs)


class GeographicCoordinateSystemType(Serializable):
    """

    """
    _fields = ('Csname', 'Datum', 'PrimeMeridian', 'AngularUnit', 'LinearUnit')
    _required = ('Csname', 'Datum', 'PrimeMeridian', 'AngularUnit')
    # Descriptor
    Csname = StringDescriptor(
        'Csname', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    Datum = SerializableDescriptor(
        'Datum', DatumType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: DatumType
    PrimeMeridian = SerializableDescriptor(
        'PrimeMeridian', PrimeMeridianType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PrimeMeridianType
    AngularUnit = StringDescriptor(
        'AngularUnit', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    LinearUnit = StringDescriptor(
        'LinearUnit', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

    def __init__(self, Csname=None, Datum=None, PrimeMeridian=None, AngularUnit=None, LinearUnit=None, **kwargs):
        """

        Parameters
        ----------
        Csname : str
        Datum : DatumType
        PrimeMeridian : PrimeMeridianType
        AngularUnit : str
        LinearUnit : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Csname = Csname
        self.Datum = Datum
        self.PrimeMeridian = PrimeMeridian
        self.AngularUnit = AngularUnit
        self.LinearUnit = LinearUnit
        super(GeographicCoordinateSystemType, self).__init__(**kwargs)


class ProjectedCoordinateSystemType(Serializable):
    """

    """
    _fields = ('Csname', 'GeographicCoordinateSystem', 'Projection', 'Parameter', 'LinearUnit')
    _required = ('Csname', 'GeographicCoordinateSystem', 'Projection', 'LinearUnit')
    # Descriptor
    Csname = StringDescriptor(
        'Csname', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    GeographicCoordinateSystem = SerializableDescriptor(
        'GeographicCoordinateSystem', GeographicCoordinateSystemType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: GeographicCoordinateSystemType
    Projection = SerializableDescriptor(
        'Projection', ProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: ProjectionType
    Parameter = SerializableDescriptor(
        'Parameter', ParameterType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, ParameterType]
    LinearUnit = StringDescriptor(
        'LinearUnit', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

    def __init__(self, Csname=None, GeographicCoordinateSystem=None, Projection=None, Parameter=None,
                 LinearUnit=None, **kwargs):
        """

        Parameters
        ----------
        Csname : str
        GeographicCoordinateSystem : GeographicCoordinateSystemType
        Projection : ProjectionType
        Parameter : None|ParameterType
        LinearUnit : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Csname = Csname
        self.GeographicCoordinateSystem = GeographicCoordinateSystem
        self.Projection = Projection
        self.Parameter = Parameter
        self.LinearUnit = LinearUnit
        super(ProjectedCoordinateSystemType, self).__init__(**kwargs)


class GeocentricCoordinateSystemType(Serializable):
    """

    """
    _fields = ('Name', 'Datum', 'PrimeMeridian', 'LinearUnit')
    _required = ('Name', 'Datum', 'PrimeMeridian', 'LinearUnit')
    # Descriptor
    Name = StringDescriptor(
        'Name', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    Datum = SerializableDescriptor(
        'Datum', DatumType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: DatumType
    PrimeMeridian = SerializableDescriptor(
        'PrimeMeridian', PrimeMeridianType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PrimeMeridianType
    LinearUnit = StringDescriptor(
        'LinearUnit', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

    def __init__(self, Name=None, Datum=None, PrimeMeridian=None, LinearUnit=None, **kwargs):
        """

        Parameters
        ----------
        Name : str
        Datum : DatumType
        PrimeMeridian : PrimeMeridianType
        LinearUnit : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.Datum = Datum
        self.PrimeMeridian = PrimeMeridian
        self.LinearUnit = LinearUnit
        super(GeocentricCoordinateSystemType, self).__init__(**kwargs)


class ReferenceSystemType(Serializable):
    """
    The reference system.
    """
    _fields = ('ProjectedCoordinateSystem', 'GeographicCoordinateSystem', 'GeocentricCoordinateSystem', 'AxisNames')
    _required = ('ProjectedCoordinateSystem', 'GeographicCoordinateSystem', 'GeocentricCoordinateSystem', 'AxisNames')
    _collections_tags = {'AxisNames': {'array': False, 'child_tag': 'AxisName'}}
    # Descriptor
    ProjectedCoordinateSystem = SerializableDescriptor(
        'ProjectedCoordinateSystem', ProjectedCoordinateSystemType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: ProjectedCoordinateSystemType
    GeographicCoordinateSystem = SerializableDescriptor(
        'GeographicCoordinateSystem', GeographicCoordinateSystemType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: GeographicCoordinateSystemType
    GeocentricCoordinateSystem = SerializableDescriptor(
        'GeocentricCoordinateSystem', GeocentricCoordinateSystemType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: GeocentricCoordinateSystemType
    AxisNames = StringListDescriptor(
        'AxisNames', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: List[str]

    def __init__(self, ProjectedCoordinateSystem=None, GeographicCoordinateSystem=None,
                 GeocentricCoordinateSystem=None, AxisNames=None, **kwargs):
        """

        Parameters
        ----------
        ProjectedCoordinateSystem : ProjectedCoordinateSystemType
        GeographicCoordinateSystem : GeographicCoordinateSystemType
        GeocentricCoordinateSystem : GeocentricCoordinateSystemType
        AxisNames : List[str]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ProjectedCoordinateSystem = ProjectedCoordinateSystem
        self.GeographicCoordinateSystem = GeographicCoordinateSystem
        self.GeocentricCoordinateSystem = GeocentricCoordinateSystem
        self.AxisNames = AxisNames
        super(ReferenceSystemType, self).__init__(**kwargs)


class AnnotationObjectType(Serializable):
    """
    Geometrical representation of the annotation. Only one of the geometry elements should be populated.
    It will not be enforced, but the order of preference will be `('Point', 'Line', 'LinearRing', 'Polygon',
    'MultiPoint', 'MultiLineString', 'MultiPolygon')`.

    Note that PolyhedralSurface is not currently supported.
    """
    _fields = ('Point', 'Line', 'LinearRing', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon')
    _choice = ({'required': True, 'collection': _fields}, )

    def __init__(self, Point=None, Line=None, LinearRing=None, Polygon=None, MultiPoint=None,
                 MultiLineString=None, MultiPolygon=None, **kwargs):
        """

        Parameters
        ----------
        Point : sarpy.geometry.geometry_elements.Point|numpy.array|list|tuple
        Line : sarpy.geometry.geometry_elements.LineString|numpy.array|list|tuple
        LinearRing : sarpy.geometry.geometry_elements.LinearRing|numpy.array|list|tuple
        Polygon : sarpy.geometry.geometry_elements.Polygon|list
        MultiPoint : sarpy.geometry.geometry_elements.MultiPoint|list
        MultiLineString : sarpy.geometry.geometry_elements.MultiLineString|list
        MultiPolygon : sarpy.geometry.geometry_elements.MultiPolygon|list
        kwargs
        """

        self._Point = None
        self._Line = None
        self._LinearRing = None
        self._Polygon = None
        self._MultiPoint = None
        self._MultiLineString = None
        self._MultiPolygon = None

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']

        if Point is not None:
            self.Point = Point
        elif Line is not None:
            self.Line = Line
        elif LinearRing is not None:
            self.LinearRing = LinearRing
        elif Polygon is not None:
            self.Polygon = Polygon
        elif MultiPoint is not None:
            self.MultiPoint = MultiPoint
        elif MultiLineString is not None:
            self.MultiLineString = MultiLineString
        elif MultiPolygon is not None:
            self.MultiPolygon = MultiPolygon
        else:
            logger.error(
                "One of (Point, Line, LinearRing, Polygon, MultiPoint, MultiLineString, MultiPolygon)\n\t"
                "should have been provided to the AnnotationObjectType constructor.")
        super(AnnotationObjectType, self).__init__(**kwargs)

    @property
    def GeometryType(self):
        """
        str: **READ ONLY** The type of geometric element which is set, from
        ('Point', 'Line', 'LinearRing', 'Polygon', 'MultiPoint',
        'MultiLineString', 'MultiPolygon')
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return None

    @property
    def Point(self):
        """
        None|sarpy.geometry.geometry_elements.Point: The point.
        """

        return self._Point

    @Point.setter
    def Point(self, value):
        if value is None:
            self._Point = None
        elif isinstance(value, (numpy.ndarray, list, tuple)):
            self._Point = PointType(coordinates=value)
        elif isinstance(value, PointType):
            self._Point = value
        else:
            raise TypeError(
                'Point requires and instance of sarpy.geometry.geometry_elements.Point, '
                'got {}'.format(type(value)))

    @property
    def Line(self):
        """
        None|sarpy.geometry.geometry_elements.LineString: The line.
        """

        return self._Line

    @Line.setter
    def Line(self, value):
        if value is None:
            self._Line = None
        elif isinstance(value, (numpy.ndarray, list, tuple)):
            self._Line = LineStringType(coordinates=value)
        elif isinstance(value, LineStringType):
            self._Line = value
        else:
            raise TypeError(
                'Line requires and instance of sarpy.geometry.geometry_elements.LineString, '
                'got {}'.format(type(value)))

    @property
    def LinearRing(self):
        """
        None|sarpy.geometry.geometry_elements.LinearRing: The linear ring.
        """

        return self._LinearRing

    @LinearRing.setter
    def LinearRing(self, value):
        if value is None:
            self._LinearRing = None
        elif isinstance(value, (numpy.ndarray, list, tuple)):
            self._LinearRing = LinearRingType(coordinates=value)
        elif isinstance(value, LinearRingType):
            self._LinearRing = value
        else:
            raise TypeError(
                'LinearRing requires and instance of sarpy.geometry.geometry_elements.LinearRing, '
                'got {}'.format(type(value)))

    @property
    def Polygon(self):
        """
        None|sarpy.geometry.geometry_elements.Polygon: The polygon.
        """

        return self._Polygon

    @Polygon.setter
    def Polygon(self, value):
        if value is None:
            self._Polygon = None
        elif isinstance(value, list):
            self._Polygon = PolygonType(coordinates=value)
        elif isinstance(value, PolygonType):
            self._Polygon = value
        else:
            raise TypeError(
                'Polygon requires and instance of sarpy.geometry.geometry_elements.Polygon, '
                'got {}'.format(type(value)))

    @property
    def MultiPoint(self):
        """
        None|sarpy.geometry.geometry_elements.MultiPoint: The multipoint.
        """

        return self._MultiPoint

    @MultiPoint.setter
    def MultiPoint(self, value):
        if value is None:
            self._MultiPoint = None
        elif isinstance(value, list):
            self._MultiPoint = MultiPointType(coordinates=value)
        elif isinstance(value, MultiPointType):
            self._MultiPoint = value
        else:
            raise TypeError(
                'MultiPoint requires and instance of sarpy.geometry.geometry_elements.MultiPoint, '
                'got {}'.format(type(value)))

    @property
    def MultiLineString(self):
        """
        None|sarpy.geometry.geometry_elements.MultiLineString: The multi-linestring.
        """

        return self._MultiLineString

    @MultiLineString.setter
    def MultiLineString(self, value):
        if value is None:
            self._MultiLineString = None
        elif isinstance(value, list):
            self._MultiLineString = MultiLineStringType(coordinates=value)
        elif isinstance(value, MultiLineStringType):
            self._MultiLineString = value
        else:
            raise TypeError(
                'MultiLineString requires and instance of sarpy.geometry.geometry_elements.MultiLineString, '
                'got {}'.format(type(value)))

    @property
    def MultiPolygon(self):
        """
        None|sarpy.geometry.geometry_elements.MultiPolygon: The multi-polygon.
        """

        return self._MultiPolygon

    @MultiPolygon.setter
    def MultiPolygon(self, value):
        if value is None:
            self._MultiPolygon = None
        elif isinstance(value, list):
            self._MultiPolygon = MultiPolygonType(coordinates=value)
        elif isinstance(value, MultiPolygonType):
            self._MultiPolygon = value
        else:
            raise TypeError(
                'MultiPolygon requires and instance of sarpy.geometry.geometry_elements.MultiPolygon, '
                'got {}'.format(type(value)))

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        for attr in self._fields:
            val = getattr(self, attr)
            if val is not None:
                return val.to_dict()
        return {}

    @classmethod
    def from_dict(cls, input_dict):
        typ = input_dict.get('type', None)
        if typ is None:
            return cls()
        elif typ == 'Point':
            return cls(Point=PointType.from_dict(input_dict))
        elif typ == 'LineString':
            return cls(Line=LineStringType.from_dict(input_dict))
        elif typ == 'LinearRing':
            return cls(LinearRing=LinearRingType.from_dict(input_dict))
        elif typ == 'Polygon':
            return cls(Polygon=PolygonType.from_dict(input_dict))
        elif typ == 'MultiPoint':
            return cls(MultiPoint=MultiPointType.from_dict(input_dict))
        elif typ == 'MultiLineString':
            return cls(MultiLineString=MultiLineStringType.from_dict(input_dict))
        elif typ == 'MultiPolygon':
            return cls(MultiPolygon=MultiPolygonType.from_dict(input_dict))
        else:
            logger.error(
                'AnnotationObjectType got unsupported input dictionary {}.\n\t'
                'Returning None.'.format(input_dict))
            return None

    @staticmethod
    def _serialize_point(coords, doc, tag, parent):
        if len(coords) < 2:
            raise ValueError('coords must have at least two elements')
        fmt_func = '{0:0.16G}'.format
        node = create_new_node(doc, tag, parent=parent)
        create_text_node(doc, 'sfa:X', fmt_func(coords[0]), parent=node)
        create_text_node(doc, 'sfa:Y', fmt_func(coords[1]), parent=node)
        if len(coords) > 2:
            create_text_node(doc, 'sfa:Z', fmt_func(coords[2]), parent=node)
        if len(coords) > 3:
            create_text_node(doc, 'sfa:M', fmt_func(coords[3]), parent=node)

    def _serialize_line(self, coords, doc, tag, parent):
        node = create_new_node(doc, tag, parent=parent)
        for entry in coords:
            self._serialize_point(entry, doc, 'sfa:Vertex', node)

    def _serialize_polygon(self, coords, doc, tag, parent):
        node = create_new_node(doc, tag, parent=parent)
        for entry in coords:
            self._serialize_line(entry, doc, 'sfa:Ring', node)

    def _serialize_multilinestring(self, coords, doc, tag, parent):
        node = create_new_node(doc, tag, parent=parent)
        for entry in coords:
            self._serialize_line(entry, doc, 'sfa:Element', node)

    def _serialize_multipolygon(self, coords, doc, tag, parent):
        node = create_new_node(doc, tag, parent=parent)
        for entry in coords:
            self._serialize_polygon(entry, doc, 'sfa:Element', node)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        if parent is None:
            parent = doc.getroot()
        if ns_key is None:
            node = create_new_node(doc, tag, parent=parent)
        else:
            node = create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)
        typ = self.GeometryType
        if typ is None:
            return node

        coords = getattr(self, typ).get_coordinates_list()
        if typ == 'Point':
            self._serialize_point(coords, doc, 'sfa:Point', node)
        elif typ == 'Line':
            self._serialize_line(coords, doc, 'sfa:Line', node)
        elif typ == 'LinearRing':
            self._serialize_line(coords, doc, 'sfa:LinearRing', node)
        elif typ == 'Polygon':
            self._serialize_polygon(coords, doc, 'sfa:Polygon', node)
        elif typ == 'MultiPoint':
            self._serialize_line(coords, doc, 'sfa:MultiPoint', node)
        elif typ == 'MultiLineString':
            self._serialize_multilinestring(coords, doc, 'sfa:MultiLineString', node)
        elif typ == 'MultiPolygon':
            self._serialize_multipolygon(coords, doc, 'sfa:MultiPolygon', node)
        else:
            raise ValueError('Unsupported serialization type {}'.format(typ))
        return node

    @staticmethod
    def _get_value(node, tag, xml_ns, ns_key):
        t_node = find_first_child(node, tag, xml_ns, ns_key)
        if t_node is None:
            return None
        else:
            return float(get_node_value(t_node))

    @classmethod
    def _extract_point(cls, node, xml_ns):
        out = [cls._get_value(node, 'X', xml_ns, 'sfa'), cls._get_value(node, 'Y', xml_ns, 'sfa')]
        z = cls._get_value(node, 'Z', xml_ns, 'sfa')
        if z in None:
            return out
        out.append(z)
        m = cls._get_value(node, 'M', xml_ns, 'sfa')
        if m in None:
            return out
        out.append(m)
        return out

    @classmethod
    def _extract_line(cls, node, xml_ns, tag='Vertex'):
        v_nodes = find_children(node, tag, xml_ns, 'sfa')
        return [cls._extract_point(v_node, xml_ns) for v_node in v_nodes]

    @classmethod
    def _extract_polygon(cls, node, xml_ns, tag='Ring'):
        v_nodes = find_children(node, tag, xml_ns, 'sfa')
        return [cls._extract_line(v_node, xml_ns, tag='Vertex') for v_node in v_nodes]

    @classmethod
    def _deserialize_point(cls, node, xml_ns, tag='Point'):
        point_node = find_first_child(node, tag, xml_ns, 'sfa')
        if point_node is None:
            return None
        return cls._extract_point(point_node, xml_ns)

    @classmethod
    def _deserialize_line(cls, node, xml_ns, tag='Line'):
        line_node = find_first_child(node, tag, xml_ns, 'sfa')
        if line_node is None:
            return None
        return cls._extract_line(line_node, xml_ns, tag='Vertex')

    @classmethod
    def _deserialize_polygon(cls, node, xml_ns, tag='Polygon'):
        poly_node = find_first_child(node, tag, xml_ns, 'sfa')
        if poly_node is None:
            return None
        return cls._extract_polygon(poly_node, xml_ns, tag='Ring')

    @classmethod
    def _deserialize_multilinestring(cls, node, xml_ns, tag='MultiLineString'):
        mls_node = find_first_child(node, tag, xml_ns, 'sfa')
        if mls_node is None:
            return None
        ls_nodes = find_children(mls_node, 'Element', xml_ns, 'sfa')
        return [cls._extract_line(ls_node, xml_ns, tag='Vertex') for ls_node in ls_nodes]

    @classmethod
    def _deserialize_multipolygon(cls, node, xml_ns, tag='MultiPolygon'):
        mp_node = find_first_child(node, tag, xml_ns, 'sfa')
        if mp_node is None:
            return None
        p_nodes = find_children(mp_node, 'Element', xml_ns, 'sfa')
        return [cls._extract_polygon(p_node, xml_ns, tag='Ring') for p_node in p_nodes]

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if xml_ns is None or 'sfa' not in xml_ns:
            raise ValueError('xml_ns must contain an entry for key "sfa"')

        coords = cls._deserialize_point(node, xml_ns, 'Point')
        if coords is not None:
            return cls(Point=PointType(coordinates=coords))

        coords = cls._deserialize_line(node, xml_ns, tag='Line')
        if coords is not None:
            return cls(Line=LineStringType(coordinates=coords))

        coords = cls._deserialize_line(node, xml_ns, tag='LinearRing')
        if coords is not None:
            return cls(LinearRing=LinearRingType(coordinates=coords))

        coords = cls._deserialize_polygon(node, xml_ns, tag='Polygon')
        if coords is not None:
            return cls(Polygon=PolygonType(coordinates=coords))

        coords = cls._deserialize_line(node, xml_ns, tag='MultiPoint')
        if coords is not None:
            return cls(MultiPoint=MultiPointType(coordinates=coords))

        coords = cls._deserialize_multilinestring(node, xml_ns, tag='MultiLineString')
        if coords is not None:
            return cls(MultiLineString=MultiLineStringType(coordinates=coords))

        coords = cls._deserialize_multipolygon(node, xml_ns, tag='MultiPolygon')
        if coords is not None:
            return cls(MultiPolygon=MultiPolygonType(coordinates=coords))

        return cls()


class AnnotationType(Serializable):
    """
    The annotation type.
    """

    _fields = ('Identifier', 'SpatialReferenceSystem', 'Objects')
    _required = ('Identifier', 'Objects')
    _collections_tags = {'Objects': {'array': False, 'child_tag': 'Object'}}
    # Descriptor
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    SpatialReferenceSystem = SerializableDescriptor(
        'SpatialReferenceSystem', ReferenceSystemType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, ReferenceSystemType]
    Objects = SerializableListDescriptor(
        'Objects', AnnotationObjectType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: List[AnnotationObjectType]

    def __init__(self, Identifier=None, SpatialReferenceSystem=None, Objects=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        SpatialReferenceSystem : None|ReferenceSystemType
        Objects : List[AnnotationObjectType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.SpatialReferenceSystem = SpatialReferenceSystem
        self.Objects = Objects
        super(AnnotationType, self).__init__(**kwargs)


class AnnotationsType(Serializable):
    """
    The list of annotations.
    """

    _fields = ('Annotations', )
    _required = ('Annotations', )
    _collections_tags = {'Annotations': {'array': False, 'child_tag': 'Annotation'}}
    # Descriptor
    Annotations = SerializableListDescriptor(
        'Annotations', AnnotationType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: List[AnnotationType]

    def __init__(self, Annotations=None, **kwargs):
        """

        Parameters
        ----------
        Annotations : List[AnnotationType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Annotations = Annotations
        super(AnnotationsType, self).__init__(**kwargs)

    def __len__(self):
        return len(self.Annotations)

    def __getitem__(self, item):
        return self.Annotations[item]
