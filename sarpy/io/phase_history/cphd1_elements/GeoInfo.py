"""
The GeoInfo definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from collections import OrderedDict
from xml.etree import ElementTree
from typing import List, Union, Dict

import numpy
from sarpy.io.xml.base import Serializable, SerializableArray, ParametersCollection, \
    find_children
from sarpy.io.xml.descriptors import StringDescriptor, SerializableArrayDescriptor, \
    ParametersDescriptor
from sarpy.io.complex.sicd_elements.blocks import LatLonRestrictionType, LatLonArrayElementType

from .base import DEFAULT_STRICT


class LineType(Serializable):
    _fields = ('EndPoints', )
    _required = ('EndPoints', )
    _collections_tags = {'EndPoints': {'array': True, 'child_tag': 'Endpoint'}}
    # descriptors
    EndPoints = SerializableArrayDescriptor(
        'EndPoints', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='A geographic line (array) with WGS-84 coordinates.'
    )  # type: Union[SerializableArray, List[LatLonArrayElementType]]

    def __init__(self, EndPoints=None, **kwargs):
        """

        Parameters
        ----------
        EndPoints : SerializableArray|List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.EndPoints = EndPoints
        super(LineType, self).__init__(**kwargs)


class PolygonType(Serializable):
    _fields = ('Polygon', )
    _required = ('Polygon', )
    _collections_tags = {'Polygon': {'array': True, 'child_tag': 'Vertex'}}
    # descriptors
    Polygon = SerializableArrayDescriptor(
        'Polygon', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='A geographic polygon (array) with WGS-84 coordinates.'
    )  # type: Union[SerializableArray, List[LatLonArrayElementType]]

    def __init__(self, Polygon=None, **kwargs):
        """

        Parameters
        ----------
        Polygon: SerializableArray|List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Polygon = Polygon
        super(PolygonType, self).__init__(**kwargs)


class GeoInfoType(Serializable):
    """
    A geographic feature.
    """

    _fields = ('name', 'Descriptions', 'Point', 'Line', 'Polygon', 'GeoInfo')
    _required = ('name', )
    _set_as_attribute = ('name', )
    _collections_tags = {
        'Descriptions': {'array': False, 'child_tag': 'Desc'},
        'Point': {'array': True, 'child_tag': 'Point'},
        'Line': {'array': True, 'child_tag': 'Line'},
        'Polygon': {'array': True, 'child_tag': 'Polygon'},
        'GeoInfo': {'array': False, 'child_tag': 'GeoInfo'}
    }
    # descriptors
    name = StringDescriptor(
        'name', _required, strict=DEFAULT_STRICT,
        docstring='The name.')  # type: str
    Descriptions = ParametersDescriptor(
        'Descriptions', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Descriptions of the geographic feature.')  # type: ParametersCollection
    Point = SerializableArrayDescriptor(
        'Point', LatLonRestrictionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Geographic points with WGS-84 coordinates.'
    )  # type: Union[SerializableArray, List[LatLonRestrictionType]]
    Line = SerializableArrayDescriptor(
        'Line', LineType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Geographic lines (array) with WGS-84 coordinates.'
    )  # type: Union[SerializableArray, List[LineType]]
    Polygon = SerializableArrayDescriptor(
        'Polygon', PolygonType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Geographic polygons (array) with WGS-84 coordinates.'
    )  # type: Union[SerializableArray, List[PolygonType]]

    def __init__(self, name=None, Descriptions=None, Point=None, Line=None,
                 Polygon=None, GeoInfo=None, **kwargs):
        """

        Parameters
        ----------
        name : str
        Descriptions : ParametersCollection|dict
        Point : SerializableArray|List[LatLonRestrictionType]|numpy.ndarray|list|tuple
        Line : SerializableArray|List[LineType]
        Polygon : SerializableArray|List[PolygonType]
        GeoInfo : Dict[GeoInfoTpe]
        kwargs : dict
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
