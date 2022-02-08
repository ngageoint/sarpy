"""
The MeasurementType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List

from sarpy.io.xml.base import Serializable, SerializableArray
from sarpy.io.xml.descriptors import SerializableDescriptor, UnitVectorDescriptor, \
    FloatDescriptor, StringEnumDescriptor, SerializableArrayDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import ReferencePointType, RowColDoubleType, Poly2DType, XYZType, RowColIntType, \
    XYZPolyType, RowColArrayElement


class BaseProjectionType(Serializable):
    """
    The base geometric projection system.
    """

    _fields = ('ReferencePoint', )
    _required = ('ReferencePoint', )
    # Descriptor
    ReferencePoint = SerializableDescriptor(
        'ReferencePoint', ReferencePointType, _required, strict=DEFAULT_STRICT,
        docstring='Reference point for the geometrical system.')  # type: ReferencePointType

    def __init__(self, ReferencePoint=None, **kwargs):
        """

        Parameters
        ----------
        ReferencePoint : ReferencePointType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ReferencePoint = ReferencePoint
        super(BaseProjectionType, self).__init__(**kwargs)


class MeasurableProjectionType(BaseProjectionType):
    """
    A physical base projection.
    """

    _fields = ('ReferencePoint', 'SampleSpacing', 'TimeCOAPoly')
    _required = ('ReferencePoint', 'SampleSpacing', 'TimeCOAPoly')
    # Descriptor
    SampleSpacing = SerializableDescriptor(
        'SampleSpacing', RowColDoubleType, _required, strict=DEFAULT_STRICT,
        docstring='Sample spacing in row and column.')  # type: RowColDoubleType
    TimeCOAPoly = SerializableDescriptor(
        'TimeCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Time (units = seconds) at which center of aperture for a given pixel '
                  'coordinate in the product occurs.')  # type: Poly2DType

    def __init__(self, ReferencePoint=None, SampleSpacing=None, TimeCOAPoly=None, **kwargs):
        """

        Parameters
        ----------
        ReferencePoint : ReferencePointType
        SampleSpacing : RowColDoubleType|numpy.ndarray|list|tuple
        TimeCOAPoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(MeasurableProjectionType, self).__init__(ReferencePoint=ReferencePoint, **kwargs)
        self.SampleSpacing = SampleSpacing
        self.TimeCOAPoly = TimeCOAPoly


class ProductPlaneType(Serializable):
    """
    Plane definition for the product.
    """

    _fields = ('RowUnitVector', 'ColUnitVector')
    _required = _fields
    # Descriptor
    RowUnitVector = UnitVectorDescriptor(
        'RowUnitVector', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Unit vector of the plane defined to be aligned in the increasing row direction '
                  'of the product. (Defined as Rpgd in Design and Exploitation document)')  # type: XYZType
    ColUnitVector = UnitVectorDescriptor(
        'ColUnitVector', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Unit vector of the plane defined to be aligned in the increasing column direction '
                  'of the product. (Defined as Cpgd in Design and Exploitation document)')  # type: XYZType

    def __init__(self, RowUnitVector=None, ColUnitVector=None, **kwargs):
        """

        Parameters
        ----------
        RowUnitVector : XYZType|numpy.ndarray|list|tuple
        ColUnitVector : XYZType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RowUnitVector = RowUnitVector
        self.ColUnitVector = ColUnitVector
        super(ProductPlaneType, self).__init__(**kwargs)


class PlaneProjectionType(MeasurableProjectionType):
    """
    Planar representation of the pixel grid.
    """

    _fields = ('ReferencePoint', 'SampleSpacing', 'TimeCOAPoly', 'ProductPlane')
    _required = ('ReferencePoint', 'SampleSpacing', 'TimeCOAPoly', 'ProductPlane')
    # Descriptor
    ProductPlane = SerializableDescriptor(
        'ProductPlane', ProductPlaneType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: ProductPlaneType

    def __init__(self, ReferencePoint=None, SampleSpacing=None, TimeCOAPoly=None, ProductPlane=None, **kwargs):
        """

        Parameters
        ----------
        ReferencePoint : ReferencePointType
        SampleSpacing : RowColDoubleType|numpy.ndarray|list|tuple
        TimeCOAPoly : Poly2DType|numpy.ndarray|list|tuple
        ProductPlane : ProductPlaneType
        kwargs
        """
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(PlaneProjectionType, self).__init__(
            ReferencePoint=ReferencePoint, SampleSpacing=SampleSpacing, TimeCOAPoly=TimeCOAPoly, **kwargs)
        self.ProductPlane = ProductPlane


class GeographicProjectionType(MeasurableProjectionType):
    """
    Geographic mapping of the pixel grid.
    """

    pass


class CylindricalProjectionType(MeasurableProjectionType):
    """
    Cylindrical mapping of the pixel grid.
    """

    _fields = ('ReferencePoint', 'SampleSpacing', 'TimeCOAPoly', 'StripmapDirection', 'CurvatureRadius')
    _required = ('ReferencePoint', 'SampleSpacing', 'TimeCOAPoly', 'StripmapDirection')
    _numeric_format = {'CurvatureRadius': FLOAT_FORMAT}
    # Descriptor
    StripmapDirection = SerializableDescriptor(
        'StripmapDirection', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Along stripmap direction.')  # type: XYZType
    CurvatureRadius = FloatDescriptor(
        'CurvatureRadius', _required, strict=DEFAULT_STRICT,
        docstring='Radius of Curvature defined at scene center.  If not present, the radius of '
                  'curvature will be derived based upon the equations provided in the '
                  'Design and Exploitation Document')  # type: Union[None, float]

    def __init__(self, ReferencePoint=None, SampleSpacing=None, TimeCOAPoly=None,
                 StripmapDirection=None, CurvatureRadius=None, **kwargs):
        """

        Parameters
        ----------
        ReferencePoint : ReferencePointType
        SampleSpacing : RowColDoubleType|numpy.ndarray|list|tuple
        TimeCOAPoly : Poly2DType|numpy.ndarray|list|tuple
        StripmapDirection : XYZType|numpy.ndarray|list|tuple
        CurvatureRadius : None|float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(CylindricalProjectionType, self).__init__(
            ReferencePoint=ReferencePoint, SampleSpacing=SampleSpacing, TimeCOAPoly=TimeCOAPoly, **kwargs)
        self.StripmapDirection = StripmapDirection
        self.CurvatureRadius = CurvatureRadius


class PolynomialProjectionType(BaseProjectionType):
    """
    Polynomial pixel to ground.  This should only used for sensor systems where the radar
    geometry parameters are not recorded.
    """

    _fields = ('ReferencePoint', 'RowColToLat', 'RowColToLon', 'RowColToAlt', 'LatLonToRow', 'LatLonToCol')
    _required = ('ReferencePoint', 'RowColToLat', 'RowColToLon', 'LatLonToRow', 'LatLonToCol')
    # Descriptor
    RowColToLat = SerializableDescriptor(
        'RowColToLat', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that converts Row/Col to Latitude (degrees).')  # type: Poly2DType
    RowColToLon = SerializableDescriptor(
        'RowColToLon', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that converts Row/Col to Longitude (degrees).')  # type: Poly2DType
    RowColToAlt = SerializableDescriptor(
        'RowColToAlt', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that converts Row/Col to Altitude (meters above '
                  'WGS-84 ellipsoid).')  # type: Union[None, Poly2DType]
    LatLonToRow = SerializableDescriptor(
        'LatLonToRow', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that converts Latitude (degrees) and Longitude (degrees) to '
                  'pixel row location.')  # type: Poly2DType
    LatLonToCol = SerializableDescriptor(
        'LatLonToCol', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that converts Latitude (degrees) and Longitude (degrees) to '
                  'pixel col location.')  # type: Poly2DType

    def __init__(self, ReferencePoint=None, RowColToLat=None, RowColToLon=None, RowColToAlt=None,
                 LatLonToRow=None, LatLonToCol=None, **kwargs):
        """

        Parameters
        ----------
        ReferencePoint : ReferencePointType
        RowColToLat : Poly2DType|numpy.ndarray|list|tuple
        RowColToLon : Poly2DType|numpy.ndarray|list|tuple
        RowColToAlt : None|Poly2DType|numpy.ndarray|list|tuple
        LatLonToRow : Poly2DType|numpy.ndarray|list|tuple
        LatLonToCol : Poly2DType|numpy.ndarray|list|tuple
        kwargs
        """
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(PolynomialProjectionType, self).__init__(ReferencePoint=ReferencePoint, **kwargs)
        self.RowColToLat = RowColToLat
        self.RowColToLon = RowColToLon
        self.RowColToAlt = RowColToAlt
        self.LatLonToRow = LatLonToRow
        self.LatLonToCol = LatLonToCol


class MeasurementType(Serializable):
    """
    Geometric SAR information required for measurement/geolocation.
    """

    _fields = (
        'PolynomialProjection', 'GeographicProjection', 'PlaneProjection', 'CylindricalProjection',
        'PixelFootprint', 'ARPFlag', 'ARPPoly', 'ValidData')
    _required = ('PixelFootprint', 'ARPPoly', 'ValidData')
    _collections_tags = {'ValidData': {'array': True, 'child_tag': 'Vertex'}}
    _numeric_format = {'ValidData': FLOAT_FORMAT}
    _choice = ({'required': False, 'collection': ('PolynomialProjection', 'GeographicProjection',
                                                  'PlaneProjection', 'CylindricalProjection')}, )
    # Descriptor
    PolynomialProjection = SerializableDescriptor(
        'PolynomialProjection', PolynomialProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial pixel to ground. Should only used for sensor systems where the radar '
                  'geometry parameters are not recorded.')  # type: Union[None, PolynomialProjectionType]
    GeographicProjection = SerializableDescriptor(
        'GeographicProjection', GeographicProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='Geographic mapping of the pixel grid referred to as GGD in the '
                  'Design and Exploitation document.')  # type: Union[None, GeographicProjectionType]
    PlaneProjection = SerializableDescriptor(
        'PlaneProjection', PlaneProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='Planar representation of the pixel grid referred to as PGD in the '
                  'Design and Exploitation document.')  # type: Union[None, PlaneProjectionType]
    CylindricalProjection = SerializableDescriptor(
        'CylindricalProjection', CylindricalProjectionType, _required, strict=DEFAULT_STRICT,
        docstring='Cylindrical mapping of the pixel grid referred to as CGD in the '
                  'Design and Exploitation document.')  # type: Union[None, CylindricalProjectionType]
    PixelFootprint = SerializableDescriptor(
        'PixelFootprint', RowColIntType, _required, strict=DEFAULT_STRICT,
        docstring='Size of the image in pixels.')  # type: RowColIntType
    ARPFlag = StringEnumDescriptor(
        'ARPFlag', ('REALTIME', 'PREDICTED', 'POST PROCESSED'), _required, strict=DEFAULT_STRICT,
        docstring='Flag indicating whether ARP polynomial is based on the best available (`collect time` or '
                  '`predicted`) ephemeris.')  # type: Union[None, str]
    ARPPoly = SerializableDescriptor(
        'ARPPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: XYZPolyType
    ValidData = SerializableArrayDescriptor(
        'ValidData', RowColArrayElement, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='Indicates the full image includes both valid data and some zero filled pixels. '
                  'Simple polygon encloses the valid data (may include some zero filled pixels for simplification). '
                  'Vertices in clockwise order.')  # type: Union[SerializableArray, List[RowColArrayElement]]

    def __init__(self, PolynomialProjection=None, GeographicProjection=None, PlaneProjection=None,
                 CylindricalProjection=None, PixelFootprint=None, ARPFlag=None, ARPPoly=None, ValidData=None, **kwargs):
        """

        Parameters
        ----------
        PolynomialProjection : PolynomialProjectionType
        GeographicProjection : GeographicProjectionType
        PlaneProjection : PlaneProjectionType
        CylindricalProjection : CylindricalProjectionType
        PixelFootprint : RowColIntType|numpy.ndarray|list|tuple
        ARPFlag : str
        ARPPoly : XYZPolyType|numpy.ndarray|list|tuple
        ValidData : SerializableArray|List[RowColArrayElement]|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PolynomialProjection = PolynomialProjection
        self.GeographicProjection = GeographicProjection
        self.PlaneProjection = PlaneProjection
        self.CylindricalProjection = CylindricalProjection
        self.PixelFootprint = PixelFootprint
        self.ARPFlag = ARPFlag
        self.ARPPoly = ARPPoly
        self.ValidData = ValidData
        super(MeasurementType, self).__init__(**kwargs)

    @property
    def ProjectionType(self):
        """str: *READ ONLY* Identifies the specific image projection type supplied."""
        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return None

    @property
    def ReferencePoint(self):
        """
        None|ReferencePointType: *READ ONLY* Gets the reference point.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute.ReferencePoint
        return None
