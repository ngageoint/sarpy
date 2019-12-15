"""
The GeoData definition.
"""


import logging
from typing import List, Union

import numpy

from .base import Serializable, DEFAULT_STRICT, _StringDescriptor, _StringEnumDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor, \
    _ParametersDescriptor, ParametersCollection, SerializableArray, \
    _SerializableCPArrayDescriptor, SerializableCPArray, _SerializableListDescriptor
from .blocks import XYZType, LatLonRestrictionType, LatLonHAERestrictionType, \
    LatLonCornerStringType, LatLonArrayElementType

from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic

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
    Descriptions = _ParametersDescriptor(
        'Descriptions', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Descriptions of the geographic feature.')  # type: ParametersCollection
    Point = _SerializableDescriptor(
        'Point', LatLonRestrictionType, _required, strict=DEFAULT_STRICT,
        docstring='A geographic point with WGS-84 coordinates.')  # type: LatLonRestrictionType
    Line = _SerializableArrayDescriptor(
        'Line', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='A geographic line (array) with WGS-84 coordinates.'
    )  # type: Union[SerializableArray, List[LatLonArrayElementType]]
    Polygon = _SerializableArrayDescriptor(
        'Polygon', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='A geographic polygon (array) with WGS-84 coordinates.'
    )  # type: Union[SerializableArray, List[LatLonArrayElementType]]

    def __init__(self, name=None, Descriptions=None, Point=None, Line=None, Polygon=None, **kwargs):
        """

        Parameters
        ----------
        name : str
        Descriptions : ParametersCollection|dict
        Point : LatLonRestrictionType|numpy.ndarray|list|tuple
        Line : SerializableArray|List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        Polygon : SerializableArray|List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        kwargs : dict
        """
        self.name = name
        self.Descriptions = Descriptions
        self.Point = Point
        self.Line = Line
        self.Polygon = Polygon
        super(GeoInfoType, self).__init__(**kwargs)

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

    def __init__(self, ECF=None, LLH=None, **kwargs):
        """

        Parameters
        ----------
        ECF : XYZType|numpy.ndarray|list|tuple
        LLH : LatLonHAERestrictionType|numpy.ndarray|list|tuple
        kwargs : dict
        """
        self.ECF = ECF
        self.LLH = LLH

        # TODO: this constructor should probably be changed to use the first of ECF
        #   and LLH which is not None, and derive the other. You can absolutely
        #   construct this with non-matching points, and that's silly. At least we
        #   should put this in the validity check.

        super(SCPType, self).__init__(**kwargs)

    def derive(self):
        """
        If only one of `ECF` or `LLH` is populated, this populates the one missing from the one present.

        Returns
        -------
        None
        """

        if self.ECF is None and self.LLH is not None:
            self.ECF = XYZType.from_array(geodetic_to_ecf(self.LLH.get_array(order='LAT'))[0])
            # TODO: this 2-d thing feels wrong - the [0] above.
        elif self.LLH is None and self.ECF is not None:
            self.LLH = LatLonHAERestrictionType.from_array(ecf_to_geodetic(self.ECF.get_array())[0])


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
    ImageCorners = _SerializableCPArrayDescriptor(
        'ImageCorners', LatLonCornerStringType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The geographic image corner points array. Image corners points projected to the '
                  'ground/surface level. Points may be projected to the same height as the SCP if ground/surface '
                  'height data is not available. The corner positions are approximate geographic locations and '
                  'not intended for analytical use.')  # type: Union[SerializableCPArray, List[LatLonCornerStringType]]
    ValidData = _SerializableArrayDescriptor(
        'ValidData', LatLonArrayElementType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=3,
        docstring='The full image array includes both valid data and some zero filled pixels.'
    )  # type: Union[SerializableArray, List[LatLonArrayElementType]]
    GeoInfos = _SerializableListDescriptor(
        'GeoInfos', GeoInfoType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Relevant geographic features list.')  # type: List[GeoInfoType]

    def __init__(self, EarthModel='WGS_84', SCP=None, ImageCorners=None, ValidData=None, GeoInfos=None, **kwargs):
        """

        Parameters
        ----------
        EarthModel : str
        SCP : SCPType
        ImageCorners : SerializableCPArray|List[LatLonCornerStringType]|numpy.ndarray|list|tuple
        ValidData : SerializableArray|List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        GeoInfos : List[GeoInfoType]
        kwargs : dict
        """
        self.EarthModel = EarthModel
        self.SCP = SCP
        self.ImageCorners = ImageCorners
        self.ValidData = ValidData
        self.GeoInfos = GeoInfos
        super(GeoDataType, self).__init__(**kwargs)

    def derive(self):
        """
        Populates any potential derived data in GeoData. In this case, just calls :func:`SCP.derive()`, and is expected
        to be called by the `SICD` parent as part of a more extensive derived data effort.

        Returns
        -------
        None
        """

        if self.SCP is not None:
            self.SCP.derive()
