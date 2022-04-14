"""
The GeoInfo definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from collections import OrderedDict
from xml.etree import ElementTree
from typing import List, Union, Dict

import numpy
from sarpy.io.xml.base import Serializable, ParametersCollection, \
    find_children, create_new_node
from sarpy.io.xml.descriptors import StringDescriptor, \
    ParametersDescriptor, SerializableListDescriptor
from sarpy.io.complex.sicd_elements.blocks import LatLonRestrictionType, LatLonArrayElementType

from .base import DEFAULT_STRICT


class LineType(Serializable):
    _fields = ('EndPoint', 'size')
    _required = ('EndPoint', 'size')

    def __init__(self, EndPoint=None, **kwargs):
        """

        Parameters
        ----------
        EndPoint : List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        kwargs
        """

        self._array = None
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.EndPoint = EndPoint
        super(LineType, self).__init__(**kwargs)

    @property
    def size(self):
        """
        int: The size attribute
        """

        return 0 if self._array is None else self._array.size

    @property
    def EndPoint(self):
        """
        numpy.ndarray: The array of points.
        """

        return numpy.array([], dtype='object') if self._array is None else self._array

    @EndPoint.setter
    def EndPoint(self, value):
        if value is None:
            self._array = None
            return

        if isinstance(value, numpy.ndarray):
            is_type = True
            for entry in value:
                is_type &= isinstance(entry, LatLonArrayElementType)
            if is_type:
                self._array = value
                return

        if isinstance(value, (numpy.ndarray, list, tuple)):
            use_value = []
            for i, entry in enumerate(value):
                if isinstance(entry, LatLonArrayElementType):
                    entry.index = i+1
                    use_value.append(entry)
                elif isinstance(entry, dict):
                    e_val = LatLonArrayElementType.from_dict(entry)
                    e_val.index = i+1
                    use_value.append(e_val)
                elif isinstance(entry, (numpy.ndarray, list, tuple)):
                    use_value.append(LatLonArrayElementType.from_array(entry, index=i+1))
                else:
                    raise TypeError('Got unexpected type for element of EndPoint array `{}`'.format(type(entry)))
            self._array = numpy.array(use_value, dtype='object')
        else:
            raise TypeError('Got unexpected type for EndPoint array `{}`'.format(type(value)))

    def __getitem__(self, item):
        return self._array.__getitem__(item)

    def __setitem__(self, key, value):
        self._array.__setitem__(key, value)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        """
        Parameters
        ----------
        node
        xml_ns : None|dict
        ns_key : None|str
        kwargs : dict

        Returns
        -------
        LineType
        """

        end_point_key = cls._child_xml_ns_key.get('EndPoint', ns_key)
        end_points = []
        for cnode in find_children(node, 'EndPoint', xml_ns, end_point_key):
            end_points.append(LatLonArrayElementType.from_node(cnode, xml_ns, ns_key=end_point_key))
        return cls(EndPoint=end_points)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        if parent is None:
            parent = doc.getroot()

        if ns_key is None:
            node = create_new_node(doc, tag, parent=parent)
        else:
            node = create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)

        node.attrib['size'] = str(self.size)
        end_point_key = self._child_xml_ns_key.get('EndPoint', ns_key)

        for entry in self.EndPoint:
            entry.to_node(doc, 'EndPoint', ns_key=end_point_key, parent=node,
                          check_validity=check_validity, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        return OrderedDict([
            ('EndPoint', [entry.to_dict() for entry in self.EndPoint]),
            ('size', self.size)])


class PolygonType(Serializable):
    _fields = ('Vertex', 'size')
    _required = ('Vertex', 'size')

    def __init__(self, Vertex=None, **kwargs):
        """

        Parameters
        ----------
        Vertex : List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        kwargs
        """

        self._array = None
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Vertex = Vertex
        super(PolygonType, self).__init__(**kwargs)

    @property
    def size(self):
        """
        int: The size attribute
        """

        if self._array is None:
            return 0
        else:
            return self._array.size

    @property
    def Vertex(self):
        """
        numpy.ndarray: The array of points.
        """

        if self._array is None:
            return numpy.array((0,), dtype='object')
        else:
            return self._array

    @Vertex.setter
    def Vertex(self, value):
        if value is None:
            self._array = None
            return

        if isinstance(value, numpy.ndarray):
            is_type = True
            for entry in value:
                is_type &= isinstance(entry, LatLonArrayElementType)
            if is_type:
                self._array = value
                return

        if isinstance(value, (numpy.ndarray, list, tuple)):
            use_value = []
            for i, entry in enumerate(value):
                if isinstance(entry, LatLonArrayElementType):
                    entry.index = i + 1
                    use_value.append(entry)
                elif isinstance(entry, dict):
                    e_val = LatLonArrayElementType.from_dict(entry)
                    e_val.index = i + 1
                    use_value.append(e_val)
                elif isinstance(entry, (numpy.ndarray, list, tuple)):
                    use_value.append(LatLonArrayElementType.from_array(entry, index=i + 1))
                else:
                    raise TypeError('Got unexpected type for element of Vertex array `{}`'.format(type(entry)))
            self._array = numpy.array(use_value, dtype='object')
        else:
            raise TypeError('Got unexpected type for Vertex array `{}`'.format(type(value)))

    def __getitem__(self, item):
        return self._array.__getitem__(item)

    def __setitem__(self, key, value):
        self._array.__setitem__(key, value)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        """
        Parameters
        ----------
        node
        xml_ns : None|dict
        ns_key : None|str
        kwargs : dict

        Returns
        -------
        PolygonType
        """

        vertex_key = cls._child_xml_ns_key.get('Vertex', ns_key)
        vertices = []
        for cnode in find_children(node, 'Vertex', xml_ns, vertex_key):
            vertices.append(LatLonArrayElementType.from_node(cnode, xml_ns, ns_key=vertex_key))
        return cls(Vertex=vertices)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        if parent is None:
            parent = doc.getroot()

        if ns_key is None:
            node = create_new_node(doc, tag, parent=parent)
        else:
            node = create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)

        node.attrib['size'] = str(self.size)
        end_point_key = self._child_xml_ns_key.get('Vertex', ns_key)

        for entry in self.Vertex:
            entry.to_node(doc, 'Vertex', ns_key=end_point_key, parent=node,
                          check_validity=check_validity, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        return OrderedDict([
            ('Vertex', [entry.to_dict() for entry in self.Vertex]),
            ('size', self.size)])


class GeoInfoType(Serializable):
    """
    A geographic feature.
    """

    _fields = ('name', 'Descriptions', 'Point', 'Line', 'Polygon', 'GeoInfo')
    _required = ('name', )
    _set_as_attribute = ('name', )
    _collections_tags = {
        'Descriptions': {'array': False, 'child_tag': 'Desc'},
        'Point': {'array': False, 'child_tag': 'Point'},
        'Line': {'array': False, 'child_tag': 'Line'},
        'Polygon': {'array': False, 'child_tag': 'Polygon'},
        'GeoInfo': {'array': False, 'child_tag': 'GeoInfo'}
    }
    # descriptors
    name = StringDescriptor(
        'name', _required, strict=DEFAULT_STRICT,
        docstring='The name.')  # type: str
    Descriptions = ParametersDescriptor(
        'Descriptions', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Descriptions of the geographic feature.')  # type: ParametersCollection
    Point = SerializableListDescriptor(
        'Point', LatLonRestrictionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Geographic points with WGS-84 coordinates.'
    )  # type: List[LatLonRestrictionType]
    Line = SerializableListDescriptor(
        'Line', LineType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Geographic lines (array) with WGS-84 coordinates.'
    )  # type: List[LineType]
    Polygon = SerializableListDescriptor(
        'Polygon', PolygonType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Geographic polygons (array) with WGS-84 coordinates.'
    )  # type: List[PolygonType]

    def __init__(self, name=None, Descriptions=None, Point=None, Line=None,
                 Polygon=None, GeoInfo=None, **kwargs):
        """

        Parameters
        ----------
        name : str
        Descriptions : ParametersCollection|dict
        Point : List[LatLonRestrictionType]
        Line : List[LineType]
        Polygon : List[PolygonType]
        GeoInfo : Dict[GeoInfoTpe]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.name = name
        self.Descriptions = Descriptions
        self.Point = Point
        self.Line = Line
        self.Polygon = Polygon

        self._GeoInfo = []
        if GeoInfo is None:
            pass
        elif isinstance(GeoInfo, GeoInfoType):
            self.addGeoInfo(GeoInfo)
        elif isinstance(GeoInfo, (list, tuple)):
            for el in GeoInfo:
                self.addGeoInfo(el)
        else:
            raise ValueError('GeoInfo got unexpected type {}'.format(type(GeoInfo)))
        super(GeoInfoType, self).__init__(**kwargs)

    @property
    def GeoInfo(self):
        """
        List[GeoInfoType]: list of GeoInfo objects.
        """

        return self._GeoInfo

    def getGeoInfo(self, key):
        """
        Get GeoInfo(s) with name attribute == `key`.

        Parameters
        ----------
        key : str

        Returns
        -------
        List[GeoInfoType]
        """

        return [entry for entry in self._GeoInfo if entry.name == key]

    def addGeoInfo(self, value):
        """
        Add the given GeoInfo to the GeoInfo list.

        Parameters
        ----------
        value : GeoInfoType

        Returns
        -------
        None
        """

        if isinstance(value, ElementTree.Element):
            gi_key = self._child_xml_ns_key.get('GeoInfo', self._xml_ns_key)
            value = GeoInfoType.from_node(value, self._xml_ns, ns_key=gi_key)
        elif isinstance(value, dict):
            value = GeoInfoType.from_dict(value)

        if isinstance(value, GeoInfoType):
            self._GeoInfo.append(value)
        else:
            raise TypeError('Trying to set GeoInfo element with unexpected type {}'.format(type(value)))

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is None:
            kwargs = OrderedDict()
        gi_key = cls._child_xml_ns_key.get('GeoInfo', ns_key)
        kwargs['GeoInfo'] = find_children(node, 'GeoInfo', xml_ns, gi_key)
        return super(GeoInfoType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        node = super(GeoInfoType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity,
            strict=strict, exclude=exclude+('GeoInfo', ))
        # slap on the GeoInfo children
        if self._GeoInfo is not None and len(self._GeoInfo) > 0:
            for entry in self._GeoInfo:
                entry.to_node(doc, tag, ns_key=ns_key, parent=node, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = super(GeoInfoType, self).to_dict(
            check_validity=check_validity, strict=strict, exclude=exclude+('GeoInfo', ))
        # slap on the GeoInfo children
        if self.GeoInfo is not None and len(self.GeoInfo) > 0:
            out['GeoInfo'] = [entry.to_dict(check_validity=check_validity, strict=strict) for entry in self._GeoInfo]
        return out
