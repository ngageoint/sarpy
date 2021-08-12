"""
Definition for the DetailObjectInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

from typing import Optional, List

import numpy

from sarpy.compliance import string_types
from sarpy.io.xml.base import Serializable, Arrayable, create_text_node, create_new_node
from sarpy.io.xml.descriptors import StringDescriptor, FloatDescriptor, \
    IntegerDescriptor, SerializableDescriptor, SerializableListDescriptor
from sarpy.io.complex.sicd_elements.blocks import RowColType

from .base import DEFAULT_STRICT
from .blocks import RangeCrossRangeType, RowColDoubleType, LatLonEleType

# TODO: the articulation and configuration information is really not usable in
#  its current form, and should be replaced with a (`name`, `value`) pair.
#  I am omitting for now.


# the Object and sub-component definitions

class PhysicalType(Serializable):
    _fields = ('ChipSize', 'CenterPixel')
    _required = _fields
    ChipSize = SerializableDescriptor(
        'ChipSize', RangeCrossRangeType, _required, strict=DEFAULT_STRICT,
        docstring='The chip size of the physical object, '
                  'in the appropriate plane')  # type: RangeCrossRangeType
    CenterPixel = SerializableDescriptor(
        'CenterPixel', RowColDoubleType, _required, strict=DEFAULT_STRICT,
        docstring='The center pixel of the physical object, '
                  'in the appropriate plane')  # type: RowColDoubleType

    def __init__(self, ChipSize=None, CenterPixel=None, **kwargs):
        """
        Parameters
        ----------
        ChipSize : RangeCrossRangeType|numpy.ndarray|list|tuple
        CenterPixel : RowColDoubleType|numpy.ndarray|list|tuple
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ChipSize = ChipSize
        self.CenterPixel = CenterPixel
        super(PhysicalType, self).__init__(**kwargs)


class PlanePhysicalType(Serializable):
    _fields = (
        'Physical', 'PhysicalWithShadows')
    _required = _fields
    Physical = SerializableDescriptor(
        'Physical', PhysicalType, _required,
        docstring='Chip details for the physical object in the appropriate plane')  # type: PhysicalType
    PhysicalWithShadows = SerializableDescriptor(
        'PhysicalWithShadows', PhysicalType, _required,
        docstring='Chip details for the physical object including shadows in '
                  'the appropriate plane')  # type: PhysicalType

    def __init__(self, Physical=None, PhysicalWithShadows=None, **kwargs):
        """

        Parameters
        ----------
        Physical : PhysicalType
        PhysicalWithShadows : PhysicalType
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Physical = Physical
        self.PhysicalWithShadows = PhysicalWithShadows
        super(PlanePhysicalType, self).__init__(**kwargs)


class SizeType(Serializable, Arrayable):
    _fields = ('Length', 'Width', 'Height')
    _required = _fields
    _numeric_format = {key: '0.16G' for key in _fields}
    # Descriptors
    Length = FloatDescriptor(
        'Length', _required, strict=True, docstring='The Length attribute.')  # type: float
    Width = FloatDescriptor(
        'Width', _required, strict=True, docstring='The Width attribute.')  # type: float
    Height = FloatDescriptor(
        'Height', _required, strict=True, docstring='The Height attribute.')  # type: float

    def __init__(self, Length=None, Width=None, Height=None, **kwargs):
        """
        Parameters
        ----------
        Length : float
        Width : float
        Height : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Length, self.Width, self.Height = Length, Width, Height
        super(SizeType, self).__init__(**kwargs)

    def get_array(self, dtype='float64'):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [Length, Width, Height]
        """

        return numpy.array([self.Length, self.Width, self.Height], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Length, Width, Height]

        Returns
        -------
        SizeType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(Length=array[0], Width=array[1], Heigth=array[2])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class OrientationType(Serializable):
    _fields = ('Roll', 'Pitch', 'Yaw', 'AzimuthAngle')
    _required = ()
    _numeric_format = {key: '0.16G' for key in _fields}
    # descriptors
    Roll = FloatDescriptor(
        'Roll', _required)  # type: float
    Pitch = FloatDescriptor(
        'Pitch', _required)  # type: float
    Yaw = FloatDescriptor(
        'Yaw', _required)  # type: float
    AzimuthAngle = FloatDescriptor(
        'AzimuthAngle', _required)  # type: float

    def __init__(self, Roll=None, Pitch=None, Yaw=None, AzimuthAngle=None, **kwargs):
        """
        Parameters
        ----------
        Roll : float
        Pitch : float
        Yaw : float
        AzimuthAngle : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Roll = Roll
        self.Pitch = Pitch
        self.Yaw = Yaw
        self.AzimuthAngle = AzimuthAngle
        super(OrientationType, self).__init__(**kwargs)


class ImageLocationType(Serializable):
    _fields = (
        'CenterPixel', 'LeftFrontPixel', 'RightFrontPixel', 'RightRearPixel',
        'LeftRearPixel')
    _required = _fields
    # descriptors
    CenterPixel = SerializableDescriptor(
        'CenterPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColType
    LeftFrontPixel = SerializableDescriptor(
        'LeftFrontPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColType
    RightFrontPixel = SerializableDescriptor(
        'RightFrontPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColType
    RightRearPixel = SerializableDescriptor(
        'RightRearPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColType
    LeftRearPixel = SerializableDescriptor(
        'LeftRearPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColType

    def __init__(self, CenterPixel=None, LeftFrontPixel=None, RightFrontPixel=None,
                 RightRearPixel=None, LeftRearPixel=None, **kwargs):
        """
        Parameters
        ----------
        CenterPixel : RowColType|numpy.ndarray|list|tuple
        LeftFrontPixel : RowColType|numpy.ndarray|list|tuple
        RightFrontPixel : RowColType|numpy.ndarray|list|tuple
        RightRearPixel : RowColType|numpy.ndarray|list|tuple
        LeftRearPixel : RowColType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CenterPixel = CenterPixel
        self.LeftFrontPixel = LeftFrontPixel
        self.RightFrontPixel = RightFrontPixel
        self.RightRearPixel = RightRearPixel
        self.LeftRearPixel = LeftRearPixel
        super(ImageLocationType, self).__init__(**kwargs)


class GeoLocationType(Serializable):
    _fields = (
        'CenterPixel', 'LeftFrontPixel', 'RightFrontPixel', 'RightRearPixel',
        'LeftRearPixel')
    _required = _fields
    # descriptors
    CenterPixel = SerializableDescriptor(
        'CenterPixel', LatLonEleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonEleType
    LeftFrontPixel = SerializableDescriptor(
        'LeftFrontPixel', LatLonEleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonEleType
    RightFrontPixel = SerializableDescriptor(
        'RightFrontPixel', LatLonEleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonEleType
    RightRearPixel = SerializableDescriptor(
        'RightRearPixel', LatLonEleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonEleType
    LeftRearPixel = SerializableDescriptor(
        'LeftRearPixel', LatLonEleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonEleType

    def __init__(self, CenterPixel=None, LeftFrontPixel=None, RightFrontPixel=None,
                 RightRearPixel=None, LeftRearPixel=None, **kwargs):
        """
        Parameters
        ----------
        CenterPixel : LatLonEleType|numpy.ndarray|list|tuple
        LeftFrontPixel : LatLonEleType|numpy.ndarray|list|tuple
        RightFrontPixel : LatLonEleType|numpy.ndarray|list|tuple
        RightRearPixel : LatLonEleType|numpy.ndarray|list|tuple
        LeftRearPixel : LatLonEleType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CenterPixel = CenterPixel
        self.LeftFrontPixel = LeftFrontPixel
        self.RightFrontPixel = RightFrontPixel
        self.RightRearPixel = RightRearPixel
        self.LeftRearPixel = LeftRearPixel
        super(GeoLocationType, self).__init__(**kwargs)


class FreeFormType(Serializable):
    _fields = ('Name', 'Value')
    _required = _fields
    Name = StringDescriptor(
        'Name', _required)  # type: str
    Value = StringDescriptor(
        'Value', _required)  # type: str

    def __init__(self, Name=None, Value=None, **kwargs):
        """

        Parameters
        ----------
        Name : str
        Value : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.Value = Value
        super(FreeFormType, self).__init__(**kwargs)


class CompoundCommentType(Serializable):
    _fields = ('Value', 'Comments')
    _required = ()
    _collections_tags = {'Comments': {'array': False, 'child_tag': 'NULL'}}
    # descriptors
    Value = StringDescriptor(
        'Value', _required,
        docstring='A single comment, this will take precedence '
                  'over the list')  # type: Optional[str]
    Comments = SerializableListDescriptor(
        'Comments', FreeFormType, _collections_tags, _required,
        docstring='A collection of comments')  # type: Optional[List[FreeFormType]]

    def __init__(self, Value=None, Comments=None, **kwargs):
        """
        Parameters
        ----------
        Value : None|str
        Comments : None|List[FreeFormType]
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Value = Value
        self.Comments = Comments
        super(CompoundCommentType, self).__init__(**kwargs)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if xml_ns is None:
            tag_start = ''
        elif ns_key is None:
            tag_start = xml_ns['default'] + ':'
        else:
            tag_start = xml_ns[ns_key] + ':'

        if node.text:
            kwargs['Value'] = node.text
            kwargs['Comments'] = None
        else:
            value = []
            for element in node:
                tag_name = element.tag[len(tag_start):]
                value.append(FreeFormType(Name=tag_name, Value=element.text))
            kwargs['Value'] = None
            kwargs['Comments'] = value
        return super(CompoundCommentType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        the_tag = '{}:{}'.format(ns_key, tag) if ns_key is not None else tag
        if self.Value is not None:
            node = create_text_node(doc, the_tag, self.Value, parent=parent)
        else:
            node = create_new_node(doc, the_tag, parent=parent)
            if self.Comments is not None:
                for entry in self.Comments:
                    child_tag = '{}:{}'.format(ns_key, entry.Name) if ns_key is not None else entry.Name
                    create_text_node(doc, child_tag, entry.Value, parent=node)
        return node


class TheObjectType(Serializable):
    _fields = (
        'SystemName', 'SystemComponent', 'NATOName', 'Function', 'Version', 'DecoyType', 'SerialNumber',
        'ObjectClass', 'ObjectSubClass', 'ObjectTypeClass', 'ObjectType', 'ObjectLabel',
        'SlantPlane', 'GroundPlane', 'Size', 'Orientation',
        'Articulation', 'Configuration',
        'Accessories', 'PaintScheme', 'Camouflage', 'Obscuration', 'ObscurationPercent', 'ImageLevelObscuration',
        'ImageLocation', 'GeoLocation',
        'TargetToClutterRatio', 'VisualQualityMetric',
        'UnderlyingTerrain', 'OverlyingTerrain', 'TerrainTexture', 'SeasonalCover')
    _required = ('SystemName', 'ImageLocation', 'GeoLocation')
    # descriptors
    SystemName = StringDescriptor(
        'SystemName', _required, strict=DEFAULT_STRICT,
        docstring='Name of the object.')  # type: str
    SystemComponent = StringDescriptor(
        'SystemComponent', _required, strict=DEFAULT_STRICT,
        docstring='Name of the weapon system component.')  # type: Optional[str]
    NATOName = StringDescriptor(
        'NATOName', _required, strict=DEFAULT_STRICT,
        docstring='Name of the object in NATO naming convention.')  # type: Optional[str]
    Function = StringDescriptor(
        'Function', _required, strict=DEFAULT_STRICT,
        docstring='Function of the object.')  # type: Optional[str]
    Version = StringDescriptor(
        'Version', _required, strict=DEFAULT_STRICT,
        docstring='Version number of the object.')  # type: Optional[str]
    DecoyType = StringDescriptor(
        'DecoyType', _required, strict=DEFAULT_STRICT,
        docstring='Object is a decoy or surrogate.')  # type: Optional[str]
    SerialNumber = StringDescriptor(
        'SerialNumber', _required, strict=DEFAULT_STRICT,
        docstring='Serial number of the object.')  # type: Optional[str]
    # label elements
    ObjectClass = StringDescriptor(
        'ObjectClass', _required, strict=DEFAULT_STRICT,
        docstring='Top level class indicator; e.g., Aircraft, Ship, '
                  'Ground Vehicle, Missile Launcher, etc.')  # type: Optional[str]
    ObjectSubClass = StringDescriptor(
        'ObjectSubClass', _required, strict=DEFAULT_STRICT,
        docstring='Sub-class indicator; e.g., military, commercial')  # type: Optional[str]
    ObjectTypeClass = StringDescriptor(
        'ObjectTypeClass', _required, strict=DEFAULT_STRICT,
        docstring='Object type class indicator; e.g., '
                  'for Aircraft/Military - Propeller, Jet')  # type: Optional[str]
    ObjectType = StringDescriptor(
        'ObjectType', _required, strict=DEFAULT_STRICT,
        docstring='Object type indicator, e.g., '
                  'for Aircraft/Military/Jet - Bomber, Fighter')  # type: Optional[str]
    ObjectLabel = StringDescriptor(
        'ObjectLabel', _required, strict=DEFAULT_STRICT,
        docstring='Object label indicator, e.g., '
                  'for Bomber - Il-28, Tu-22M, Tu-160')  # type: Optional[str]
    SlantPlane = SerializableDescriptor(
        'SlantPlane', PlanePhysicalType, _required, strict=DEFAULT_STRICT,
        docstring='Object physical definition in the slant plane')  # type: Optional[PlanePhysicalType]
    GroundPlane = SerializableDescriptor(
        'GroundPlane', PlanePhysicalType, _required, strict=DEFAULT_STRICT,
        docstring='Object physical definition in the ground plane')  # type: Optional[PlanePhysicalType]
    # specific physical quantities
    Size = SerializableDescriptor(
        'Size', SizeType, _required, strict=DEFAULT_STRICT,
        docstring='The actual physical size of the object')  # type: Optional[SizeType]
    Orientation = SerializableDescriptor(
        'Orientation', OrientationType, _required, strict=DEFAULT_STRICT,
        docstring='The actual orientation size of the object')  # type: Optional[OrientationType]
    Articulation = SerializableDescriptor(
        'Articulation', CompoundCommentType, _required,
        docstring='Articulation description(s)')  # type: Optional[CompoundCommentType]
    Configuration = SerializableDescriptor(
        'Configuration', CompoundCommentType, _required,
        docstring='Configuration description(s)')  # type: Optional[CompoundCommentType]
    Accessories = StringDescriptor(
        'Accessories', _required, strict=DEFAULT_STRICT,
        docstring='Defines items that are out of the norm, or have been added or removed.')  # type: Optional[str]
    PaintScheme = StringDescriptor(
        'PaintScheme', _required, strict=DEFAULT_STRICT,
        docstring='Paint scheme of object (e.g. olive drab, compass ghost grey, etc.).')  # type: Optional[str]
    Camouflage = StringDescriptor(
        'Camouflage', _required, strict=DEFAULT_STRICT,
        docstring='Details the camouflage on the object.')  # type: Optional[str]
    Obscuration = StringDescriptor(
        'Obscuration', _required, strict=DEFAULT_STRICT,
        docstring='General description of the obscuration.')  # type: Optional[str]
    ObscurationPercent = FloatDescriptor(
        'ObscurationPercent', _required, strict=DEFAULT_STRICT,
        docstring='The percent obscuration.')  # type: Optional[float]
    ImageLevelObscuration = StringDescriptor(
        'ImageLevelObscuration', _required, strict=DEFAULT_STRICT,
        docstring='Specific description of the obscuration based on the sensor look angle.')  # type: Optional[str]
    # location of the labeled item
    ImageLocation = SerializableDescriptor(
        'ImageLocation', ImageLocationType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: ImageLocationType
    GeoLocation = SerializableDescriptor(
        'GeoLocation', GeoLocationType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: GeoLocationType
    # text quality descriptions
    TargetToClutterRatio = StringDescriptor(
        'TargetToClutterRatio', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[str]
    VisualQualityMetric = StringDescriptor(
        'VisualQualityMetric', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[str]
    UnderlyingTerrain = StringDescriptor(
        'UnderlyingTerrain', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[str]
    OverlyingTerrain = StringDescriptor(
        'OverlyingTerrain', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[str]
    TerrainTexture = StringDescriptor(
        'TerrainTexture', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[str]
    SeasonalCover = StringDescriptor(
        'SeasonalCover', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[str]

    def __init__(self, SystemName=None, SystemComponent=None, NATOName=None,
                 Function=None, Version=None, DecoyType=None, SerialNumber=None,
                 ObjectClass=None, ObjectSubClass=None, ObjectTypeClass=None,
                 ObjectType=None, ObjectLabel=None, Size=None, Orientation=None,
                 Articulation=None, Configuration=None,
                 Accessories=None, PaintScheme=None, Camouflage=None,
                 Obscuration=None, ObscurationPercent=None, ImageLevelObscuration=None,
                 ImageLocation=None, GeoLocation=None,
                 TargetToClutterRatio=None, VisualQualityMetric=None,
                 UnderlyingTerrain=None, OverlyingTerrain=None,
                 TerrainTexture=None, SeasonalCover=None,
                 **kwargs):
        """
        Parameters
        ----------
        SystemName : str
        SystemComponent : None|str
        NATOName : None|str
        Function : None|str
        Version : None|str
        DecoyType : None|str
        SerialNumber : None|str
        ObjectClass : None|str
        ObjectSubClass : None|str
        ObjectTypeClass : None|str
        ObjectType : None|str
        ObjectLabel : None|str
        Size : None|SizeType|numpy.ndarray|list|tuple
        Orientation : OrientationType
        Articulation : None|CompoundCommentType|str|List[FreeFormType]
        Configuration : None|CompoundCommentType|str|List[FreeFormType]
        Accessories : None|str
        PaintScheme : None|str
        Camouflage : None|str
        Obscuration : None|str
        ObscurationPercent : None|float
        ImageLevelObscuration : None|str
        ImageLocation : ImageLocationType
        GeoLocation : GeoLocationType
        TargetToClutterRatio : None|str
        VisualQualityMetric : None|str
        UnderlyingTerrain : None|str
        OverlyingTerrain : None|str
        TerrainTexture : None|str
        SeasonalCover : None|str
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SystemName = SystemName
        self.SystemComponent = SystemComponent
        self.NATOName = NATOName
        self.Function = Function
        self.Version = Version
        self.DecoyType = DecoyType
        self.SerialNumber = SerialNumber

        self.ObjectClass = ObjectClass
        self.ObjectSubClass = ObjectSubClass
        self.ObjectTypeClass = ObjectTypeClass
        self.ObjectType = ObjectType
        self.ObjectLabel = ObjectLabel

        self.Size = Size
        self.Orientation = Orientation

        if isinstance(Articulation, string_types):
            self.Articulation = CompoundCommentType(Value=Articulation)
        elif isinstance(Articulation, list):
            self.Articulation = CompoundCommentType(Comments=Articulation)
        elif isinstance(Articulation, dict):
            self.Articulation = CompoundCommentType(**Articulation)
        else:
            self.Articulation = Articulation

        if isinstance(Configuration, string_types):
            self.Configuration = CompoundCommentType(Value=Configuration)
        elif isinstance(Configuration, list):
            self.Configuration = CompoundCommentType(Comments=Configuration)
        elif isinstance(Configuration, dict):
            self.Configuration = CompoundCommentType(**Configuration)
        else:
            self.Configuration = Configuration

        self.Accessories = Accessories
        self.PaintScheme = PaintScheme
        self.Camouflage = Camouflage
        self.Obscuration = Obscuration
        self.ObscurationPercent = ObscurationPercent
        self.ImageLevelObscuration = ImageLevelObscuration

        self.ImageLocation = ImageLocation
        self.GeoLocation = GeoLocation

        self.TargetToClutterRatio = TargetToClutterRatio
        self.VisualQualityMetric = VisualQualityMetric
        self.UnderlyingTerrain = UnderlyingTerrain
        self.OverlyingTerrain = OverlyingTerrain
        self.TerrainTexture = TerrainTexture
        self.SeasonalCover = SeasonalCover
        super(TheObjectType, self).__init__(**kwargs)


# other types for the DetailObjectInfo

class NominalType(Serializable):
    _fields = ('ChipSize', )
    _required = _fields
    ChipSize = SerializableDescriptor(
        'ChipSize', RangeCrossRangeType, _required, strict=DEFAULT_STRICT,
        docstring='The nominal chip size used for every object in the dataset, '
                  'in the appropriate plane')  # type: RangeCrossRangeType

    def __init__(self, ChipSize=None, **kwargs):
        """
        Parameters
        ----------
        ChipSize : RangeCrossRangeType|numpy.ndarray|list|tuple
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ChipSize = ChipSize
        super(NominalType, self).__init__(**kwargs)


class PlaneNominalType(Serializable):
    _fields = ('Nominal', )
    _required = _fields
    Nominal = SerializableDescriptor(
        'Nominal', NominalType, _required,
        docstring='Nominal chip details in the appropriate plane')  # type: NominalType

    def __init__(self, Nominal=None, **kwargs):
        """

        Parameters
        ----------
        Nominal : NominalType
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Nominal = Nominal
        super(PlaneNominalType, self).__init__(**kwargs)


# the main type

class DetailObjectInfoType(Serializable):
    _fields = (
        'NumberOfObjectsInImage', 'NumberOfObjectsInScene',
        'SlantPlane', 'GroundPlane', 'Objects')
    _required = (
        'NumberOfObjectsInImage', 'NumberOfObjectsInScene', 'Objects')
    _collections_tags = {'Objects': {'array': False, 'child_tag': 'Object'}}
    # descriptors
    NumberOfObjectsInImage = IntegerDescriptor(
        'NumberOfObjectsInImage', _required, strict=DEFAULT_STRICT,
        docstring='Number of ground truthed objects in the image.')  # type: int
    NumberOfObjectsInScene = IntegerDescriptor(
        'NumberOfObjectsInScene', _required, strict=DEFAULT_STRICT,
        docstring='Number of ground truthed objects in the scene.')  # type: int
    SlantPlane = SerializableDescriptor(
        'SlantPlane', PlaneNominalType, _required,
        docstring='Default chip sizes in the slant plane.')  # type: Optional[PlaneNominalType]
    GroundPlane = SerializableDescriptor(
        'GroundPlane', PlaneNominalType, _required,
        docstring='Default chip sizes in the ground plane.')  # type: Optional[PlaneNominalType]
    Objects = SerializableListDescriptor(
        'Objects', TheObjectType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The object collection')  # type: List[TheObjectType]

    def __init__(self, NumberOfObjectsInImage=None, NumberOfObjectsInScene=None,
                 SlantPlane=None, GroundPlane=None, Objects=None, **kwargs):
        """
        Parameters
        ----------
        NumberOfObjectsInImage : int
        NumberOfObjectsInScene : int
        SlantPlane : None|SlantPlaneNominalType
        GroundPlane : None|GroundPlaneNominalType
        Objects : List[ObjectType]
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NumberOfObjectsInImage = NumberOfObjectsInImage
        self.NumberOfObjectsInScene = NumberOfObjectsInScene
        self.SlantPlane = SlantPlane
        self.GroundPlane = GroundPlane
        self.Objects = Objects
        super(DetailObjectInfoType, self).__init__(**kwargs)
