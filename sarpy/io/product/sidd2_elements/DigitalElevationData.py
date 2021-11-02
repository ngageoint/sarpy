"""
The DigitalElevationDataType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List

import numpy

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import SerializableDescriptor, IntegerDescriptor, \
    FloatDescriptor, FloatListDescriptor, StringEnumDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import LatLonType


class GeographicCoordinatesType(Serializable):
    """
    Describes the Local Geographic Coordinate system linking row/column to the absolute
    geographic coordinate (lat/lon).
    """

    _fields = ('LongitudeDensity', 'LatitudeDensity', 'ReferenceOrigin')
    _required = ('LongitudeDensity', 'LatitudeDensity', 'ReferenceOrigin')
    _numeric_format = {'LongitudeDensity': FLOAT_FORMAT, 'LatitudeDensity': FLOAT_FORMAT}
    # Descriptor
    LongitudeDensity = FloatDescriptor(
        'LongitudeDensity', _required, strict=DEFAULT_STRICT,
        docstring='Pixel ground spacing in E/W direction that is the number of pixels '
                  'or element intervals in 360 degrees.')  # type: float
    LatitudeDensity = FloatDescriptor(
        'LatitudeDensity', _required, strict=DEFAULT_STRICT,
        docstring='Pixel ground spacing in N/S direction that is the number of pixels '
                  'or element intervals in 360 degrees.')  # type: float
    ReferenceOrigin = SerializableDescriptor(
        'ReferenceOrigin', LatLonType, _required, strict=DEFAULT_STRICT,
        docstring='Northwest corner Latitude/Longitude - product NW corner')  # type: LatLonType

    def __init__(self, LongitudeDensity=None, LatitudeDensity=None, ReferenceOrigin=None, **kwargs):
        """

        Parameters
        ----------
        LongitudeDensity : float
        LatitudeDensity : float
        ReferenceOrigin : LatLonType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.LongitudeDensity = LongitudeDensity
        self.LatitudeDensity = LatitudeDensity
        self.ReferenceOrigin = ReferenceOrigin
        super(GeographicCoordinatesType, self).__init__(**kwargs)


class GeopositioningType(Serializable):
    """
    Describes the absolute coordinate system to which the data is referenced.
    """
    _fields = (
        'CoordinateSystemType', 'GeodeticDatum', 'ReferenceEllipsoid',
        'VerticalDatum', 'SoundingDatum', 'FalseOrigin', 'UTMGridZoneNumber')
    _required = (
        'CoordinateSystemType', 'GeodeticDatum', 'ReferenceEllipsoid',
        'VerticalDatum', 'SoundingDatum', 'FalseOrigin')
    # Descriptor
    CoordinateSystemType = StringEnumDescriptor(
        'CoordinateSystemType', ('GGS', 'UTM'), _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    GeodeticDatum = StringEnumDescriptor(
        'GeodeticDatum', ('World Geodetic System 1984', ), _required, strict=DEFAULT_STRICT,
        default_value='World Geodetic System 1984',
        docstring='')  # type: str
    ReferenceEllipsoid = StringEnumDescriptor(
        'ReferenceEllipsoid', ('World Geodetic System 1984', ), _required, strict=DEFAULT_STRICT,
        default_value='World Geodetic System 1984',
        docstring='')  # type: str
    VerticalDatum = StringEnumDescriptor(
        'VerticalDatum', ('Mean Sea Level', ), _required, strict=DEFAULT_STRICT,
        default_value='Mean Sea Level',
        docstring='')  # type: str
    SoundingDatum = StringEnumDescriptor(
        'SoundingDatum', ('Mean Sea Level', ), _required, strict=DEFAULT_STRICT,
        default_value='Mean Sea Level',
        docstring='')  # type: str
    FalseOrigin = IntegerDescriptor(
        'FalseOrigin', _required, strict=DEFAULT_STRICT,
        docstring='Z values false origin.')  # type: int
    UTMGridZoneNumber = IntegerDescriptor(
        'UTMGridZoneNumber', _required, strict=DEFAULT_STRICT,
        docstring='Gride zone number, required for UTM, not include for GCS. '
                  '**Values -** `+001` to `+060` (northern hemisphere) and `-001` to `-060` '
                  '(southern hemisphere)')  # type: int

    def __init__(self, CoordinateSystemType=None, GeodeticDatum='World Geodetic System 1984',
                 ReferenceEllipsoid='World Geodetic System 1984', VerticalDatum='Mean Sea Level',
                 SoundingDatum='Mean Sea Level', FalseOrigin=None, UTMGridZoneNumber=None, **kwargs):
        """

        Parameters
        ----------
        CoordinateSystemType : str
        GeodeticDatum : str
        ReferenceEllipsoid : str
        VerticalDatum : str
        SoundingDatum : str
        FalseOrigin : int
        UTMGridZoneNumber : None|int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.CoordinateSystemType = CoordinateSystemType
        self.GeodeticDatum = GeodeticDatum
        self.ReferenceEllipsoid = ReferenceEllipsoid
        self.VerticalDatum = VerticalDatum
        self.SoundingDatum = SoundingDatum
        self.FalseOrigin = FalseOrigin
        self.UTMGridZoneNumber = UTMGridZoneNumber
        super(GeopositioningType, self).__init__(**kwargs)


class AccuracyType(Serializable):
    """
    The accuracy estimate.
    """

    _fields = ('Horizontals', 'Verticals')
    _required = ('Horizontals', 'Verticals')
    _collections_tags = {
        'Horizontals': {'array': False, 'child_tag': 'Horizontal'},
        'Verticals': {'array': False, 'child_tag': 'Vertical'}}
    _numeric_format = {key: FLOAT_FORMAT for key in _fields}
    # Descriptor
    Horizontals = FloatListDescriptor(
        'Horizontals', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: List[float]
    Verticals = FloatListDescriptor(
        'Verticals', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: List[float]

    def __init__(self, Horizontals=None, Verticals=None, **kwargs):
        """

        Parameters
        ----------
        Horizontals : List[float]
        Verticals : List[float]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.Horizontals = Horizontals
        self.Verticals = Verticals
        super(AccuracyType, self).__init__(**kwargs)


class PositionalAccuracyType(Serializable):
    """
    Describes the horizontal and vertical point and regional information for the DED.
    """

    _fields = ('NumRegions', 'AbsoluteAccuracy', 'PointToPointAccuracy')
    _required = ('NumRegions', 'AbsoluteAccuracy', 'PointToPointAccuracy')
    # Descriptor
    NumRegions = IntegerDescriptor(
        'NumRegions', _required, strict=DEFAULT_STRICT,
        docstring='Number of positional accuracy regions.')  # type: int
    AbsoluteAccuracy = SerializableDescriptor(
        'AbsoluteAccuracy', AccuracyType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: AccuracyType
    PointToPointAccuracy = SerializableDescriptor(
        'PointToPointAccuracy', AccuracyType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: AccuracyType

    def __init__(self, NumRegions=None, AbsoluteAccuracy=None, PointToPointAccuracy=None, **kwargs):
        """

        Parameters
        ----------
        NumRegions : int
        AbsoluteAccuracy : AccuracyType
        PointToPointAccuracy : AccuracyType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NumRegions = NumRegions
        self.AbsoluteAccuracy = AbsoluteAccuracy
        self.PointToPointAccuracy = PointToPointAccuracy
        super(PositionalAccuracyType, self).__init__(**kwargs)


class DigitalElevationDataType(Serializable):
    """
    This describes any Digital Elevation Data included with the SIDD product.
    """

    _fields = ('GeographicCoordinates', 'Geopositioning', 'PositionalAccuracy', 'NullValue')
    _required = ('GeographicCoordinates', 'Geopositioning', 'PositionalAccuracy')
    # Descriptor
    GeographicCoordinates = SerializableDescriptor(
        'GeographicCoordinates', GeographicCoordinatesType, _required, strict=DEFAULT_STRICT,
        docstring='Describes the Local Geographic Coordinate system linking row/column to the '
                  'absolute geographic coordinate (lat/lon)')  # type: GeographicCoordinatesType
    Geopositioning = SerializableDescriptor(
        'Geopositioning', GeopositioningType, _required, strict=DEFAULT_STRICT,
        docstring='Describes the absolute coordinate system to which the data is '
                  'referenced.')  # type: GeopositioningType
    PositionalAccuracy = SerializableDescriptor(
        'PositionalAccuracy', PositionalAccuracyType, _required, strict=DEFAULT_STRICT,
        docstring='Describes the horizontal and vertical point and regional information '
                  'for the DED.')  # type: PositionalAccuracyType
    NullValue = IntegerDescriptor(
        'NullValue', _required, strict=DEFAULT_STRICT,
        docstring='The value in the DEM corresponding to `No Value`.')  # type: Union[None, int]

    def __init__(self, GeographicCoordinates=None, Geopositioning=None, PositionalAccuracy=None,
                 NullValue=None, **kwargs):
        """

        Parameters
        ----------
        GeographicCoordinates : GeographicCoordinatesType
        Geopositioning : GeopositioningType
        PositionalAccuracy : PositionalAccuracyType
        NullValue : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.GeographicCoordinates = GeographicCoordinates
        self.Geopositioning = Geopositioning
        self.PositionalAccuracy = PositionalAccuracy
        self.NullValue = NullValue
        super(DigitalElevationDataType, self).__init__(**kwargs)
