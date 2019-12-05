"""
The GeoData definition.
"""

from typing import List, Union

import numpy

from .base import Serializable, DEFAULT_STRICT, _StringEnumDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from .blocks import XYZType, LatLonHAERestrictionType, LatLonCornerStringType, LatLonArrayElementType
from .GeoInfo import GeoInfoType

from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic

__classification__ = "UNCLASSIFIED"


class SCPType(Serializable):
    """Scene Center Point (SCP) in full (global) image. This is the precise location."""
    _fields = ('ECF', 'LLH')
    _required = _fields  # isn't this redundant?
    ECF = _SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The ECF coordinates.')  # type: XYZType
    LLH = _SerializableDescriptor(
        'LLH', LatLonHAERestrictionType, _required, strict=DEFAULT_STRICT,
        docstring='The WGS-84 coordinates.')  # type: LatLonHAERestrictionType

    def derive(self):
        """
        Populates any potential derived data in SCP.

        Returns
        -------
        None
        """

        if self.ECF is None and self.LLH is not None:
            self.ECF = XYZType(coords=geodetic_to_ecf(self.LLH.get_array(order='LAT')))
        elif self.LLH is None and self.ECF is not None:
            self.LLH = LatLonHAERestrictionType(coords=ecf_to_geodetic(self.ECF.get_array()))


class GeoDataType(Serializable):
    """Container specifying the image coverage area in geographic coordinates."""
    _fields = ('EarthModel', 'SCP', 'ImageCorners', 'ValidData', 'GeoInfos')
    _required = ('EarthModel', 'SCP', 'ImageCorners')
    _collections_tags = {
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
        'ImageCorners': {'array': True, 'child_tag': 'ICP'},
        'GeoInfos': {'array': False, 'child_tag': 'GeoInfo'},
    }
    # other class variables
    _EARTH_MODEL_VALUES = ('WGS_84', )
    # descriptors
    EarthModel = _StringEnumDescriptor(
        'EarthModel', _EARTH_MODEL_VALUES, _required, strict=True, default_value='WGS_84',
        docstring='The Earth Model.'.format(_EARTH_MODEL_VALUES))  # type: str
    SCP = _SerializableDescriptor(
        'SCP', SCPType, _required, strict=DEFAULT_STRICT,
        docstring='The Scene Center Point (SCP) in full (global) image. This is the '
                  'precise location.')  # type: SCPType
    ImageCorners = _SerializableArrayDescriptor(
        'ImageCorners', LatLonCornerStringType, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring='The geographic image corner points array. Image corners points projected to the '
                  'ground/surface level. Points may be projected to the same height as the SCP if ground/surface '
                  'height data is not available. The corner positions are approximate geographic locations and '
                  'not intended for analytical use.')  # type: Union[numpy.ndarray, List[LatLonCornerStringType]]
    ValidData = _SerializableArrayDescriptor(
        'ValidData', LatLonArrayElementType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=3,
        docstring='The full image array includes both valid data and some zero filled pixels.'
    )  # type: Union[numpy.ndarray, List[LatLonArrayElementType]]
    GeoInfos = _SerializableArrayDescriptor(
        'GeoInfos', GeoInfoType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Relevant geographic features list.')  # type: List[GeoInfoType]

    def derive(self):
        """
        Populates any potential derived data in GeoData.

        Returns
        -------
        None
        """

        if self.SCP is not None:
            self.SCP.derive()
