"""
The GeoDataType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from collections import OrderedDict
from typing import Union, List
from xml.etree import ElementTree

from sarpy.io.xml.base import Serializable, SerializableArray, find_children
from sarpy.io.xml.descriptors import SerializableArrayDescriptor, StringEnumDescriptor
from sarpy.io.complex.sicd_elements.base import SerializableCPArray, SerializableCPArrayDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import LatLonCornerStringType, LatLonArrayElementType, GeoInfoType


class GeoDataType(Serializable):
    """
    Container specifying the image coverage area in geographic coordinates.

    .. Note: The SICD.GeoData class is an extension of this class. Implementation remain
        separate to allow the possibility of different functionality.
    """

    _fields = ('EarthModel', 'ImageCorners', 'ValidData')
    _required = ('EarthModel', 'ImageCorners', 'ValidData')
    _collections_tags = {
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
        'ImageCorners': {'array': True, 'child_tag': 'ICP'}}
    _numeric_format = {'ImageCorners': FLOAT_FORMAT, 'ValidData': FLOAT_FORMAT}
    # other class variables
    _EARTH_MODEL_VALUES = ('WGS_84', )
    # descriptors
    EarthModel = StringEnumDescriptor(
        'EarthModel', _EARTH_MODEL_VALUES, _required, strict=True, default_value='WGS_84',
        docstring='Identifies the earth model used for latitude, longitude and height parameters. '
                  'All height values are *Height Above The Ellipsoid '
                  '(HAE)*.'.format(_EARTH_MODEL_VALUES))  # type: str
    ImageCorners = SerializableCPArrayDescriptor(
        'ImageCorners', LatLonCornerStringType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The geographic image corner points array. Image corners points projected to the '
                  'ground/surface level. Points may be projected to the same height as the SCP if ground/surface '
                  'height data is not available. The corner positions are approximate geographic locations and '
                  'not intended for analytical '
                  'use.')  # type: Union[SerializableCPArray, List[LatLonCornerStringType]]
    ValidData = SerializableArrayDescriptor(
        'ValidData', LatLonArrayElementType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=3,
        docstring='The full image array includes both valid data and some zero filled pixels. Simple convex '
                  'polygon enclosed the valid data (may include some zero filled pixels for simplification). '
                  'Vertices in clockwise order.')  # type: Union[SerializableArray, List[LatLonArrayElementType]]

    def __init__(self, EarthModel='WGS_84', ImageCorners=None, ValidData=None, GeoInfos=None, **kwargs):
        """

        Parameters
        ----------
        EarthModel : str
        ImageCorners : SerializableCPArray|List[LatLonCornerStringType]|numpy.ndarray|list|tuple
        ValidData : SerializableArray|List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        GeoInfos : List[GeoInfoType]
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.EarthModel = EarthModel
        self.ImageCorners = ImageCorners
        self.ValidData = ValidData

        self._GeoInfos = []
        if GeoInfos is None:
            pass
        elif isinstance(GeoInfos, GeoInfoType):
            self.setGeoInfo(GeoInfos)
        elif isinstance(GeoInfos, (list, tuple)):
            for el in GeoInfos:
                self.setGeoInfo(el)
        else:
            raise ValueError('GeoInfos got unexpected type {}'.format(type(GeoInfos)))
        super(GeoDataType, self).__init__(**kwargs)

    @property
    def GeoInfos(self):
        """
        List[GeoInfoType]: list of GeoInfos.
        """

        return self._GeoInfos

    def getGeoInfo(self, key):
        """
        Get the GeoInfo(s) with name attribute == `key`

        Parameters
        ----------
        key : str

        Returns
        -------
        List[GeoInfoType]
        """

        return [entry for entry in self._GeoInfos if entry.name == key]

    def setGeoInfo(self, value):
        """
        Add the given GeoInfo to the GeoInfos list.

        Parameters
        ----------
        value : GeoInfoType

        Returns
        -------
        None
        """

        if isinstance(value, ElementTree.Element):
            gi_key = self._child_xml_ns_key.get('GeoInfos', self._xml_ns_key)
            value = GeoInfoType.from_node(value, self._xml_ns, ns_key=gi_key)
        elif isinstance(value, dict):
            value = GeoInfoType.from_dict(value)

        if isinstance(value, GeoInfoType):
            self._GeoInfos.append(value)
        else:
            raise TypeError('Trying to set GeoInfo element with unexpected type {}'.format(type(value)))

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is None:
            kwargs = OrderedDict()
        gkey = cls._child_xml_ns_key.get('GeoInfos', ns_key)
        kwargs['GeoInfos'] = find_children(node, 'GeoInfo', xml_ns, gkey)
        return super(GeoDataType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        node = super(GeoDataType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity, strict=strict, exclude=exclude)
        # slap on the GeoInfo children
        for entry in self._GeoInfos:
            entry.to_node(doc, 'GeoInfo', ns_key=ns_key, parent=node, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = super(GeoDataType, self).to_dict(
            check_validity=check_validity, strict=strict, exclude=exclude)
        # slap on the GeoInfo children
        if len(self.GeoInfos) > 0:
            out['GeoInfos'] = [entry.to_dict(check_validity=check_validity, strict=strict) for entry in self._GeoInfos]
        return out
