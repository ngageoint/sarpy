"""
The GeoInfo definition.
"""

import logging
import numpy
from typing import List, Union

from .base import Serializable, DEFAULT_STRICT, _StringDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from .blocks import ParameterType, LatLonRestrictionType, LatLonArrayElementType


__classification__ = "UNCLASSIFIED"


class GeoInfoType(Serializable):
    """A geographic feature."""
    # TODO: VERIFY - The word document/pdf doesn't match the xsd.
    #   Is the standard really self-referential here? I find that confusing.
    #   I suspect this part of the standard may not have gotten much attention.
    _fields = ('name', 'Descriptions', 'Point', 'Line', 'Polygon')
    _required = ('name', )
    _set_as_attribute = ('name', )
    _choice = ({'required': False, 'collection': ('Point', 'Line', 'Polygon')}, )
    _collections_tags = {
        'Descriptions': {'array': False, 'child_tag': 'Desc'},
        'Line': {'array': True, 'child_tag': 'Endpoint'},
        'Polygon': {'array': True, 'child_tag': 'Vertex'}, }
    # descriptors
    name = _StringDescriptor(
        'name', _required, strict=True,
        docstring='The name.')  # type: str
    Descriptions = _SerializableArrayDescriptor(
        'Descriptions', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Descriptions of the geographic feature.')  # type: List[ParameterType]
    Point = _SerializableDescriptor(
        'Point', LatLonRestrictionType, _required, strict=DEFAULT_STRICT,
        docstring='A geographic point with WGS-84 coordinates.')  # type: LatLonRestrictionType
    Line = _SerializableArrayDescriptor(
        'Line', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='A geographic line (array) with WGS-84 coordinates.'
    )  # type: Union[numpy.ndarray, List[LatLonArrayElementType]]
    Polygon = _SerializableArrayDescriptor(
        'Polygon', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='A geographic polygon (array) with WGS-84 coordinates.'
    )  # type: Union[numpy.ndarray, List[LatLonArrayElementType]]

    @property
    def FeatureType(self):  # type: () -> Union[None, str]
        """
        str: READ ONLY attribute. Identifies the feature type among. This is determined by
        returning the (first) attribute among `Point`, `Line`, `Polygon` which is populated. None will be returned if
        none of them are populated.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return None

    def _validate_features(self):
        if self.Line is not None and self.Line.size < 2:
            logging.error('GeoInfo has a Line feature with {} points defined.'.format(self.Line.size))
            return False
        if self.Polygon is not None and self.Polygon.size < 3:
            logging.error('GeoInfo has a Polygon feature with {} points defined.'.format(self.Polygon.size))
            return False
        return True

    def _basic_validity_check(self):
        condition = super(GeoInfoType, self)._basic_validity_check()
        return condition & self._validate_features()
