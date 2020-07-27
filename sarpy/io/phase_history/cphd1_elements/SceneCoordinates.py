# -*- coding: utf-8 -*-
"""
The SceneCoordinates type definition.
"""

from typing import Union, List

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _FloatDescriptor, _IntegerDescriptor, \
    _StringDescriptor, _StringEnumDescriptor, _SerializableDescriptor, \
    SerializableArray, _SerializableArrayDescriptor, SerializableCPArray, \
    _SerializableCPArrayDescriptor, _create_text_node, _UnitVectorDescriptor
from sarpy.io.complex.sicd_elements.blocks import XYZType, LatLonType, LatLonCornerType
from sarpy.io.complex.sicd_elements.GeoData import SCPType
from .blocks import AreaType, LSType, LSVertexType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class IARPType(SCPType):
    """
    The Image Area Reference Point (IARP), which is the origin of the Image Area Coordinate system.
    Note that setting one of ECF or LLH will implicitly set the other to its corresponding matched value.
    """


class ECFPlanarType(Serializable):
    """
    Parameters for a planar surface defined in ECF coordinates. The reference
    surface is a plane that contains the IARP.
    """

    _fields = ('uIAX', 'uIAY')
    _required = _fields
    # descriptors
    uIAX = _UnitVectorDescriptor(
        'uIAX', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Image Area X-coordinate (IAX) unit vector in ECF coordinates. '
                  'For stripmap collections, uIAX ALWAYS points in the direction '
                  'of the scanning footprint.')  # type: XYZType
    uIAY = _UnitVectorDescriptor(
        'uIAY', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Image Area Y-coordinate (IAY) unit vector in ECF '
                  'coordinates. This should be perpendicular to '
                  'uIAX.')  # type: XYZType

    def __init__(self, uIAX=None, uIAY=None, **kwargs):
        """

        Parameters
        ----------
        uIAX : XYZType|numpy.ndarray|list|tuple
        uIAY : XYZType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.uIAX = uIAX
        self.uIAY = uIAY
        super(ECFPlanarType, self).__init__(**kwargs)


class LLPlanarType(Serializable):
    """
    Parameters for Lat/Lon planar surface of constant HAE, implicitly assumed to be
    the HAE at the `IARP`.
    """

    _fields = ('uIAXLL', 'uIAYLL')
    _required = _fields
    # descriptors
    uIAXLL = _SerializableDescriptor(
        'uIAXLL', LatLonType, _required, strict=DEFAULT_STRICT,
        docstring='Image coordinate IAX *"unit vector"* expressed as an increment '
                  'in latitude and longitude corresponding to a 1.0 meter increment '
                  'in image coordinate `IAX`.')  # type: LatLonType
    uIAYLL = _SerializableDescriptor(
        'uIAYLL', LatLonType, _required, strict=DEFAULT_STRICT,
        docstring='Image coordinate IAY *"unit vector"* expressed as an increment '
                  'in latitude and longitude corresponding to a 1.0 meter increment '
                  'in image coordinate `IAY`.')  # type: LatLonType

    def __init__(self, uIAXLL=None, uIAYLL=None, **kwargs):
        """

        Parameters
        ----------
        uIAXLL : LatLonType|numpy.ndarray|list|tuple
        uIAYLL : LatLonType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.uIAXLL = uIAXLL
        self.uIAYLL = uIAYLL
        super(LLPlanarType, self).__init__(**kwargs)


class ReferenceSurfaceType(Serializable):
    """
    Parameters that define the Reference Surface used for the product.
    """
    _fields = ('Planar', 'HAE')
    _required = ()
    _choice = ({'required': True, 'collection': _fields}, )
    # descriptors
    Planar = _SerializableDescriptor(
        'Planar', ECFPlanarType, _required, strict=DEFAULT_STRICT,
        docstring='The ECF planar surface definition.')  # type: Union[None, ECFPlanarType]
    HAE = _SerializableDescriptor(
        'HAE', LLPlanarType, _required, strict=DEFAULT_STRICT,
        docstring='The HAE surface definition.')  # type: Union[None, LLPlanarType]

    def __init__(self, Planar=None, HAE=None, **kwargs):
        """

        Parameters
        ----------
        Planar : ECFPlanarType|None
        HAE : LLPlanarType|None
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Planar = Planar
        self.HAE = HAE
        super(ReferenceSurfaceType, self).__init__(**kwargs)


###########
# Image grid definition

class IAXExtentType(Serializable):
    """
    Increasing line index is in the +IAX direction.
    """
    _fields = ('LineSpacing', 'FirstLine', 'NumLines')
    _required = _fields
    _numeric_format = {'LineSpacing': '0.16G'}
    # descriptors
    LineSpacing = _FloatDescriptor(
        'LineSpacing', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The line spacing, in meters.')  # type: float
    FirstLine = _IntegerDescriptor(
        'FirstLine', _required, strict=DEFAULT_STRICT,
        docstring='Index of the first line.')  # type: int
    NumLines = _IntegerDescriptor(
        'NumLines', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of lines.')  # type: int

    def __init__(self, LineSpacing=None, FirstLine=None, NumLines=None, **kwargs):
        """

        Parameters
        ----------
        LineSpacing : float
        FirstLine : int
        NumLines : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.LineSpacing = LineSpacing
        self.FirstLine = FirstLine
        self.NumLines = NumLines
        super(IAXExtentType, self).__init__(**kwargs)


class IAYExtentType(Serializable):
    """
    Increasing sample index is in the +IAY direction.
    """

    _fields = ('SampleSpacing', 'FirstSample', 'NumSamples')
    _required = _fields
    _numeric_format = {'SampleSpacing': '0.16G'}
    # descriptors
    SampleSpacing = _FloatDescriptor(
        'SampleSpacing', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Sample spacing, in meters.')  # type: float
    FirstSample = _IntegerDescriptor(
        'FirstSample', _required, strict=DEFAULT_STRICT,
        docstring='Index of the first sample.')  # type: int
    NumSamples = _IntegerDescriptor(
        'NumSamples', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of samples.')  # type: int

    def __init__(self, SampleSpacing=None, FirstSample=None, NumSamples=None, **kwargs):
        """

        Parameters
        ----------
        SampleSpacing : float
        FirstSample : int
        NumSamples : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SampleSpacing = SampleSpacing
        self.FirstSample = FirstSample
        self.NumSamples = NumSamples
        super(IAYExtentType, self).__init__(**kwargs)


class SegmentType(Serializable):
    """
    Rectangle segment.
    """

    _fields = ('Identifier', 'StartLine', 'StartSample', 'EndLine', 'EndSample', 'SegmentPolygon')
    _required = ('Identifier', 'StartLine', 'StartSample', 'EndLine', 'EndSample')
    _collections_tags = {'SegmentPolygon': {'array': True, 'child_tag': 'SV'}}
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies the Image Segment.')  # type: str
    StartLine = _IntegerDescriptor(
        'StartLine', _required, strict=DEFAULT_STRICT,
        docstring='Start line of the segment.')  # type: int
    StartSample = _IntegerDescriptor(
        'StartSample', _required, strict=DEFAULT_STRICT,
        docstring='Start sample of the segment.')  # type: int
    EndLine = _IntegerDescriptor(
        'EndLine', _required, strict=DEFAULT_STRICT,
        docstring='End line of the segment.')  # type: int
    EndSample = _IntegerDescriptor(
        'EndSample', _required, strict=DEFAULT_STRICT,
        docstring='End sample of the segment.')  # type: int
    SegmentPolygon = _SerializableArrayDescriptor(
        'SegmentPolygon', LSVertexType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=3,
        docstring='Polygon that describes a portion of the segment '
                  'rectangle.')  # type: Union[SerializableArray, List[LSVertexType]]

    def __init__(self, Identifier=None, StartLine=None, StartSample=None, EndLine=None,
                 EndSample=None, SegmentPolygon=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        StartLine : int
        StartSample : int
        EndLine : int
        EndSample : int
        SegmentPolygon : SerializableArray|List[LSVertexType]|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.StartLine = StartLine
        self.StartSample = StartSample
        self.EndLine = EndLine
        self.EndSample = EndSample
        self.SegmentPolygon = SegmentPolygon
        super(SegmentType, self).__init__(**kwargs)


class SegmentListType(SerializableArray):
    _set_size = False
    _set_index = False

    @property
    def NumSegments(self):
        return self.size

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT):
        anode = super(SegmentListType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity, strict=strict)
        _create_text_node(
            doc, 'NumSegments' if ns_key is None else '{}:NumSegments'.format(ns_key),
            '{0:d}'.format(self.NumSegments), parent=anode)
        return anode


class ImageGridType(Serializable):
    """
    Parameters that describe a geo-referenced image grid for image data products
    that may be formed from the CPHD signal array(s).
    """

    _fields = ('Identifier', 'IARPLocation', 'IAXExtent', 'IAYExtent', 'SegmentList')
    _required = ('IARPLocation', 'IAXExtent', 'IAYExtent')
    _collections_tags = {'SegmentList': {'array': True, 'child_tag': 'Segemnt'}}
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies the Image Grid.')  # type: Union[None, str]
    IARPLocation = _SerializableDescriptor(
        'IARPLocation', LSType, _required, strict=DEFAULT_STRICT,
        docstring='IARP grid location. Grid locations indexed by (line, sample) or (L,S). '
                  'Image grid line and sample are pixel-centered indices.')  # type: LSType
    IAXExtent = _SerializableDescriptor(
        'IAXExtent', IAXExtentType, _required, strict=DEFAULT_STRICT,
        docstring='Increasing line index is in the +IAX direction.')  # type: IAXExtentType
    IAYExtent = _SerializableDescriptor(
        'IAYExtent', IAYExtentType, _required, strict=DEFAULT_STRICT,
        docstring='Increasing sample index is in the +IAY direction.')  # type: IAYExtentType
    SegmentList = _SerializableArrayDescriptor(
        'SegmentList', SegmentType, _collections_tags, _required, strict=DEFAULT_STRICT,
        array_extension=SegmentListType,
        docstring='List of image grid segments defined relative to the image '
                  'grid.')  # type: Union[SegmentListType, List[SegmentType]]

    def __init__(self, Identifier=None, IARPLocation=None, IAXExtent=None, IAYExtent=None,
                 SegmentList=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : None|str
        IARPLocation : LSType
        IAXExtent : IAXExtentType
        IAYExtent : IAYExtentType
        SegmentList : SegmentListType|List[SegmentType]|numpy.array|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.IARPLocation = IARPLocation
        self.IAXExtent = IAXExtent
        self.IAYExtent = IAYExtent
        self.SegmentList = SegmentList
        super(ImageGridType, self).__init__(**kwargs)


###############

class SceneCoordinatesType(Serializable):
    """
    Parameters that define geographic coordinates for in the imaged scene.
    """

    _fields = (
        'EarthModel', 'IARP', 'ReferenceSurface', 'ImageArea', 'ImageAreaCornerPoints',
        'ExtendedArea', 'ImageGrid')
    _required = ('EarthModel', 'IARP', 'ReferenceSurface', 'ImageArea', 'ImageAreaCornerPoints')
    _collections_tags = {
        'ImageAreaCornerPoints': {'array': True, 'child_tag': 'IACP'}}
    # descriptors
    EarthModel = _StringEnumDescriptor(
        'EarthModel', ('WGS_84', ), _required, strict=DEFAULT_STRICT, default_value='WGS_84',
        docstring='Specifies the earth model used for specifying geodetic coordinates. All heights are '
                  'Height Above the Ellipsoid (HAE) unless specifically '
                  'noted.')  # type: str
    IARP = _SerializableDescriptor(
        'IARP', IARPType, _required, strict=DEFAULT_STRICT,
        docstring='Image Area Reference Point (IARP). The IARP is the origin of '
                  'the Image Area Coordinate system.')  # type: IARPType
    ReferenceSurface = _SerializableDescriptor(
        'ReferenceSurface', ReferenceSurfaceType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that define the Reference Surface used for the '
                  'product.')  # type: ReferenceSurfaceType
    ImageArea = _SerializableDescriptor(
        'ImageArea', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='Image Area is defined by a rectangle aligned with Image Area coordinates (IAX, IAY). '
                  'May be reduced by the optional polygon.')  # type: AreaType
    ImageAreaCornerPoints = _SerializableCPArrayDescriptor(
        'ImageAreaCornerPoints', LatLonCornerType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Image Area Corner Points (IACPs) that bound the full resolution '
                  'image area.')  # type: Union[SerializableCPArray, List[LatLonCornerType]]
    ExtendedArea = _SerializableDescriptor(
        'ExtendedArea', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='Extended Area is defined by a rectangle aligned with Image Area coordinates '
                  '(IAX, IAY). May be reduced by the optional polygon.')  # type: Union[None, AreaType]
    ImageGrid = _SerializableDescriptor(
        'ImageGrid', ImageGridType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe a geo-referenced image grid for image data '
                  'products that may be formed from the CPHD signal '
                  'array(s).')  # type: ImageGridType

    def __init__(self, EarthModel='WGS_84', IARP=None, ReferenceSurface=None,
                 ImageArea=None, ImageAreaCornerPoints=None, ExtendedArea=None,
                 ImageGrid=None, **kwargs):
        """

        Parameters
        ----------
        EarthModel : None|str
        IARP : IARPType
        ReferenceSurface : ReferenceSurfaceType
        ImageArea : AreaType
        ImageAreaCornerPoints : SerializableCPArray|List[LatLonCornerType]|numpy.ndarray|list|tuple
        ExtendedArea : None|AreaType
        ImageGrid : None|ImageGridType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.EarthModel = EarthModel
        self.IARP = IARP
        self.ReferenceSurface = ReferenceSurface
        self.ImageArea = ImageArea
        self.ImageAreaCornerPoints = ImageAreaCornerPoints
        self.ExtendedArea = ExtendedArea
        self.ImageGrid = ImageGrid
        super(SceneCoordinatesType, self).__init__(**kwargs)
