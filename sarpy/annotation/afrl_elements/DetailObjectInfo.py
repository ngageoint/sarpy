"""
Definition for the DetailObjectInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

import logging
from typing import Optional, List

import numpy

from sarpy.compliance import string_types
from sarpy.io.xml.base import Serializable, Arrayable, create_text_node, create_new_node
from sarpy.io.xml.descriptors import StringDescriptor, FloatDescriptor, \
    IntegerDescriptor, SerializableDescriptor, SerializableListDescriptor
from sarpy.io.complex.sicd_elements.blocks import RowColType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.product.sidd2_elements.SIDD import SIDDType
from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic, wgs_84_norm
from sarpy.geometry.geometry_elements import Point, Polygon, GeometryCollection, Geometry

from .base import DEFAULT_STRICT
from .blocks import RangeCrossRangeType, RowColDoubleType, LatLonEleType

# TODO: Issue - do we need to set the nominal chip size?
#  Comment - the articulation and configuration information is really not usable in
#       its current form, and should be replaced with a (`name`, `value`) pair.

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_ranges(cls, row_range, col_range, row_limit, col_limit):
        """
        Construct from the row/column ranges and limits.

        Parameters
        ----------
        row_range
        col_range
        row_limit
        col_limit

        Returns
        -------
        PhysicalType
        """

        first_row, last_row = max(0, row_range[0]), min(row_limit, row_range[1])
        first_col, last_col = max(0, col_range[0]), min(col_limit, col_range[1])
        return PhysicalType(
            ChipSize=(last_row-first_row, last_col-first_col),
            CenterPixel=(0.5*(last_row+first_row), 0.5*(last_col+first_col)))


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

    def get_max_diameter(self):
        """
        Gets the nominal maximum diameter for the item, in meters.

        Returns
        -------
        float
        """

        return float(numpy.sqrt(self.Length*self.Length + self.Width*self.Width))

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
            return cls(Length=array[0], Width=array[1], Height=array[2])
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

    @classmethod
    def from_geolocation(cls, geo_location, the_structure):
        """
        Construct the image location from the geographical location via
        projection using the SICD model.

        Parameters
        ----------
        geo_location : GeoLocationType
        the_structure : SICDType|SIDDType

        Returns
        -------
        None|ImageLocationType
            None if projection fails, the value otherwise
        """

        if geo_location is None:
            return None

        if not the_structure.can_project_coordinates():
            logger.warning(
                'This sicd does not permit projection,\n\t'
                'so the image location can not be inferred')
            return None

        # make sure this is defined, for the sake of efficiency
        the_structure.define_coa_projection(overide=False)

        kwargs = {}

        if isinstance(the_structure, SICDType):
            image_shift = numpy.array(
                [the_structure.ImageData.FirstRow, the_structure.ImageData.FirstCol], dtype='float64')
        else:
            image_shift = numpy.zeros((2, ), dtype='float64')

        for attribute in cls._fields:
            value = getattr(geo_location, attribute)
            if value is not None:
                absolute_pixel_location = the_structure.project_ground_to_image_geo(
                    value.get_array(dtype='float64'), ordering='latlong')
                if numpy.any(numpy.isnan(absolute_pixel_location)):
                    return None

                kwargs[attribute] = absolute_pixel_location - image_shift
        out = ImageLocationType(**kwargs)
        out.infer_center_pixel()
        return out

    def infer_center_pixel(self):
        """
        Infer the center pixel, if not populated.

        Returns
        -------
        None
        """

        if self.CenterPixel is not None:
            return

        current = numpy.zeros((2, ), dtype='float64')
        for entry in self._fields:
            if entry == 'CenterPixel':
                continue
            value = getattr(self, entry)
            if value is None:
                return
            current += 0.25*value.get_array(dtype='float64')
        self.CenterPixel = RowColType.from_array(current)

    def get_nominal_box(self, row_length=10, col_length=10):
        """
        Get a nominal box containing the object, using the default side length if necessary.

        Parameters
        ----------
        row_length : int|float
            The side length to use for the rectangle, if not defined.
        col_length : int|float
            The side length to use for the rectangle, if not defined.

        Returns
        -------
        None|numpy.ndarray
        """

        if self.LeftFrontPixel is not None and self.RightFrontPixel is not None and \
                self.LeftRearPixel is not None and self.RightRearPixel is not None:
            out = numpy.zeros((4, 2), dtype='float64')
            out[0, :] = self.LeftFrontPixel.get_array()
            out[1, :] = self.RightFrontPixel.get_array()
            out[2, :] = self.RightRearPixel.get_array()
            out[3, :] = self.LeftRearPixel.get_array()
            return out

        if self.CenterPixel is None:
            return None

        shift = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype='float64')
        shift[:, 0] *= row_length
        shift[:, 1] *= col_length
        return self.CenterPixel.get_array(dtype='float64') + shift

    def get_geometry_object(self):
        """
        Gets the geometry for the given image section.

        Returns
        -------
        Geometry
        """

        point = None
        polygon = None
        if self.CenterPixel is not None:
            point = Point(coordinates=self.CenterPixel.get_array(dtype='float64'))
        if self.LeftFrontPixel is not None and \
                self.RightFrontPixel is not None and \
                self.RightRearPixel is not None and \
                self.LeftRearPixel is not None:
            ring = numpy.zeros((4, 2), dtype='float64')
            ring[0, :] = self.LeftFrontPixel.get_array(dtype='float64')
            ring[1, :] = self.RightFrontPixel.get_array(dtype='float64')
            ring[2, :] = self.RightRearPixel.get_array(dtype='float64')
            ring[3, :] = self.LeftRearPixel.get_array(dtype='float64')
            polygon = Polygon(coordinates=[ring, ])
        if point is not None and polygon is not None:
            return GeometryCollection(geometries=[point, polygon])
        elif point is not None:
            return point
        elif polygon is not None:
            return polygon
        else:
            return None


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

    # noinspection PyUnusedLocal
    @classmethod
    def from_image_location(cls, image_location, the_structure, projection_type='HAE', **kwargs):
        """
        Construct the geographical location from the image location via
        projection using the SICD model.

        .. Note::
            This assumes that the image coordinates are with respect to the given
            image (chip), and NOT including any sicd.ImageData.FirstRow/Col values,
            which will be added here.

        Parameters
        ----------
        image_location : ImageLocationType
        the_structure : SICDType|SIDDType
        projection_type : str
            The projection type selector, one of `['PLANE', 'HAE', 'DEM']`. Using `'DEM'`
            requires configuration for the DEM pathway described in
            :func:`sarpy.geometry.point_projection.image_to_ground_dem`.
        kwargs
            The keyword arguments for the :func:`SICDType.project_image_to_ground_geo` method.

        Returns
        -------
        None|GeoLocationType
            Coordinates may be populated as `NaN` if projection fails.
        """

        if image_location is None:
            return None

        if not the_structure.can_project_coordinates():
            logger.warning(
                'This sicd does not permit projection,\n\t'
                'so the image location can not be inferred')
            return None

        # make sure this is defined, for the sake of efficiency
        the_structure.define_coa_projection(overide=False)

        kwargs = {}

        if isinstance(the_structure, SICDType):
            image_shift = numpy.array(
                [the_structure.ImageData.FirstRow, the_structure.ImageData.FirstCol], dtype='float64')
        else:
            image_shift = numpy.zeros((2, ), dtype='float64')

        for attribute in cls._fields:
            value = getattr(image_location, attribute)
            if value is not None:
                coords = value.get_array(dtype='float64') + image_shift
                geo_coords = the_structure.project_image_to_ground_geo(
                    coords, ordering='latlong', projection_type=projection_type, **kwargs)

                kwargs[attribute] = geo_coords
        out = GeoLocationType(**kwargs)
        out.infer_center_pixel()
        return out

    def infer_center_pixel(self):
        """
        Infer the center pixel, if not populated.

        Returns
        -------
        None
        """

        if self.CenterPixel is not None:
            return

        current = numpy.zeros((3, ), dtype='float64')
        for entry in self._fields:
            if entry == 'CenterPixel':
                continue
            value = getattr(self, entry)
            if value is None:
                return
            current += 0.25*geodetic_to_ecf(value.get_array(dtype='float64'))
        self.CenterPixel = LatLonEleType.from_array(ecf_to_geodetic(current))


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
                 ObjectType=None, ObjectLabel=None,
                 SlantPlane=None, GroundPlane=None,
                 Size=None, Orientation=None,
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
        SlantPlane : None|PlanePhysicalType
        GroundPlane : None|PlanePhysicalType
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

        self.SlantPlane = SlantPlane
        self.GroundPlane = GroundPlane
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

    @staticmethod
    def _check_placement(rows, cols, row_bounds, col_bounds, overlap_cutoff=0.5):
        """
        Checks the bounds condition for the provided box.

        Here inclusion is defined by what proportion of the area of the proposed
        chip is actually contained inside the image bounds.

        Parameters
        ----------
        rows : int|float
            The number of rows in the image.
        cols : int|float
            The number of columns in the image.
        row_bounds : List
            Of the form `[row min, row max]`
        col_bounds : List
            Of the form `[col min, col max]`
        overlap_cutoff : float
            Determines the transition from in the periphery to out of the image.

        Returns
        -------
        int
            1 - completely in the image
            2 - the proposed chip has `overlap_cutoff <= fractional contained area < 1`
            3 - the proposed chip has `fractional contained area < overlap_cutoff`
        """

        if row_bounds[1] <= row_bounds[0] or col_bounds[1] <= col_bounds[0]:
            raise ValueError('bounds out of order')
        if 0 <= row_bounds[0] and rows < row_bounds[1] and 0 <= col_bounds[0] and cols < col_bounds[1]:
            return 1  # completely in bounds

        row_size = row_bounds[1] - row_bounds[0]
        col_size = col_bounds[1] - col_bounds[0]

        first_row, last_row = max(0, row_bounds[0]), min(rows, row_bounds[1])
        first_col, last_col = max(0, col_bounds[0]), min(cols, col_bounds[1])

        area_overlap = (last_row - first_row)*(last_col - first_col)
        if area_overlap >= overlap_cutoff*row_size*col_size:
            return 2  # the item is at the periphery
        else:
            return 3  # it should be considered out of range

    def set_image_location_from_sicd(self, sicd, populate_in_periphery=False):
        """
        Set the image location information with respect to the given SICD,
        assuming that the physical coordinates are populated.

        Parameters
        ----------
        sicd : SICDType
        populate_in_periphery : bool

        Returns
        -------
        int
            -1 - insufficient metadata to proceed or other failure
            0 - nothing to be done
            1 - successful
            2 - object in the image periphery, populating based on `populate_in_periphery`
            3 - object not in the image field
        """

        if self.ImageLocation is not None:
            # no need to infer anything, it's already populated
            return 0

        if self.GeoLocation is None:
            logger.warning(
                'GeoLocation is not populated,\n\t'
                'so the image location can not be inferred')
            return -1

        if not sicd.can_project_coordinates():
            logger.warning(
                'This sicd does not permit projection,\n\t'
                'so the image location can not be inferred')
            return -1

        # gets the prospective image location
        image_location = ImageLocationType.from_geolocation(self.GeoLocation, sicd)
        if image_location is None:
            return -1

        # get nominal object size in meters and pixels
        if self.Size is None:
            row_size = 2.0
            col_size = 2.0
        else:
            max_size = self.Size.get_max_diameter()
            row_size = max_size/sicd.Grid.Row.SS
            col_size = max_size/sicd.Grid.Col.SS

        # check bounding information
        rows = sicd.ImageData.NumRows
        cols = sicd.ImageData.NumCols
        center_pixel = image_location.CenterPixel.get_array(dtype='float64')
        row_bounds = [center_pixel[0] - 0.5*row_size, center_pixel[0] + 0.5*row_size]
        col_bounds = [center_pixel[1] - 0.5*col_size, center_pixel[1] + 0.5*col_size]

        placement = self._check_placement(rows, cols, row_bounds, col_bounds)

        if placement == 3:
            return placement
        if placement == 2 and not populate_in_periphery:
            return placement

        self.ImageLocation = image_location
        return placement

    def set_geo_location_from_sicd(self, sicd, projection_type='HAE', **kwargs):
        """
        Set the geographical location information with respect to the given SICD,
        assuming that the image coordinates are populated.

        .. Note::
            This assumes that the image coordinates are with respect to the given
            image (chip), and NOT including any sicd.ImageData.FirstRow/Col values,
            which will be added here.

        Parameters
        ----------
        sicd : SICDType
        projection_type : str
            The projection type selector, one of `['PLANE', 'HAE', 'DEM']`. Using `'DEM'`
            requires configuration for the DEM pathway described in
            :func:`sarpy.geometry.point_projection.image_to_ground_dem`.
        kwargs
            The keyword arguments for the :func:`SICDType.project_image_to_ground_geo` method.
        """

        if self.GeoLocation is not None:
            # no need to infer anything, it's already populated
            return

        if self.ImageLocation is None:
            logger.warning(
                'ImageLocation is not populated,\n\t'
                'so the geographical location can not be inferred')
            return

        if not sicd.can_project_coordinates():
            logger.warning(
                'This sicd does not permit projection,\n\t'
                'so the geographical location can not be inferred')
            return

        self.GeoLocation = GeoLocationType.from_image_location(
            self.ImageLocation, sicd, projection_type=projection_type, **kwargs)

    def set_chip_details_from_sicd(self, sicd, layover_shift=False, populate_in_periphery=False):
        """
        Set the chip information with respect to the given SICD, assuming that the
        image location and size are defined.

        Parameters
        ----------
        sicd : SICDType
        layover_shift : bool
            Shift based on layover direction? This should be `True` if the identification of
            the bounds and/or center pixel do not include any layover, as in
            populating location from known ground truth. This should be `False` if
            the identification of bounds and/or center pixel do include layover,
            potentially as based on annotation of the imagery itself in pixel
            space.
        populate_in_periphery : bool
            Should we populate for peripheral?

        Returns
        -------
        int
            -1 - insufficient metadata to proceed
            0 - nothing to be done
            1 - successful
            2 - object in the image periphery, populating based on `populate_in_periphery`
            3 - object not in the image field
        """

        if self.SlantPlane is not None:
            # no need to infer anything, it's already populated
            return 0

        if self.Size is None:
            logger.warning(
                'Size is not populated,\n\t'
                'so the chip size can not be inferred')
            return -1

        if self.ImageLocation is None:
            # try to set from geolocation
            return_value = self.set_image_location_from_sicd(sicd, populate_in_periphery=populate_in_periphery)
            if return_value in [-1, 3] or (return_value == 2 and not populate_in_periphery):
                return return_value

        # get nominal object size, in meters
        max_size = self.Size.get_max_diameter()  # in meters
        row_size = max_size/sicd.Grid.Row.SS  # in pixels
        col_size = max_size/sicd.Grid.Col.SS  # in pixels

        # get nominal image box
        image_location = self.ImageLocation
        pixel_box = image_location.get_nominal_box(row_length=row_size, col_length=col_size)

        ground_unit_norm = wgs_84_norm(sicd.GeoData.SCP.ECF.get_array())
        slant_plane_unit_norm = numpy.cross(sicd.Grid.Row.UVectECF.get_array(), sicd.Grid.Col.UVectECF.get_array())
        magnitude_factor = ground_unit_norm.dot(slant_plane_unit_norm)
        # determines the relative size of things in slant plane versus ground plane

        # get nominal layover vector - should be pointed generally towards the top (negative rows value)
        layover_magnitude = sicd.SCPCOA.LayoverMagnitude
        if layover_magnitude is None:
            layover_magnitude = 0.25
        layover_size = self.Size.Height*layover_magnitude*magnitude_factor
        if sicd.SCPCOA.LayoverAng is None:
            layover_angle = 0.0
        else:
            layover_angle = numpy.deg2rad(sicd.SCPCOA.LayoverAng - sicd.SCPCOA.AzimAng)
        layover_vector = layover_size*numpy.array(
            [numpy.cos(layover_angle)/sicd.Grid.Row.SS, numpy.sin(layover_angle)/sicd.Grid.Col.SS])

        # craft the layover box
        if layover_shift:
            layover_box = pixel_box + layover_vector
        else:
            layover_box = pixel_box

        # determine the maximum and minimum pixel values here
        min_rows = min(numpy.min(pixel_box[:, 0]), numpy.min(layover_box[:, 0]))
        max_rows = max(numpy.max(pixel_box[:, 0]), numpy.max(layover_box[:, 0]))
        min_cols = min(numpy.min(pixel_box[:, 1]), numpy.min(layover_box[:, 1]))
        max_cols = max(numpy.max(pixel_box[:, 1]), numpy.max(layover_box[:, 1]))

        # determine the padding amount
        row_pad = min(5, 0.3*(max_rows-min_rows))
        col_pad = min(5, 0.3*(max_cols-min_cols))

        # check our bounding information
        rows = sicd.ImageData.NumRows
        cols = sicd.ImageData.NumCols

        chip_rows = [min_rows - row_pad, max_rows + row_pad]
        chip_cols = [min_cols - col_pad, max_cols + col_pad]

        placement = self._check_placement(rows, cols, chip_rows, chip_cols)
        if placement == 3 or (placement == 2 and not populate_in_periphery):
            return placement

        # set the physical data ideal chip size
        physical = PhysicalType.from_ranges(chip_rows, chip_cols, rows, cols)

        # determine nominal shadow vector
        shadow_magnitude = sicd.SCPCOA.ShadowMagnitude
        if shadow_magnitude is None:
            shadow_magnitude = 1.0
        shadow_size = self.Size.Height*shadow_magnitude*magnitude_factor
        shadow_angle = sicd.SCPCOA.Shadow
        shadow_angle = numpy.pi if shadow_angle is None else numpy.deg2rad(shadow_angle)
        shadow_vector = shadow_size*numpy.array(
            [numpy.cos(shadow_angle)/sicd.Grid.Row.SS, numpy.sin(shadow_angle)/sicd.Grid.Col.SS])

        shadow_box = pixel_box + shadow_vector

        min_rows = min(min_rows, numpy.min(shadow_box[:, 0]))
        max_rows = max(max_rows, numpy.max(shadow_box[:, 0]))
        min_cols = min(min_cols, numpy.min(shadow_box[:, 1]))
        max_cols = max(max_cols, numpy.max(shadow_box[:, 1]))

        chip_rows = [min_rows - row_pad, max_rows + row_pad]
        chip_cols = [min_cols - col_pad, max_cols + col_pad]
        # set the physical with shadows data ideal chip size
        physical_with_shadows = PhysicalType.from_ranges(chip_rows, chip_cols, rows, cols)

        self.SlantPlane = PlanePhysicalType(
            Physical=physical,
            PhysicalWithShadows=physical_with_shadows)
        return placement

    def get_image_geometry_object_for_sicd(self, include_chip=False):
        """
        Gets the geometry element describing the image geometry for a sicd.

        Returns
        -------
        Geometry
        """

        if self.ImageLocation is None:
            raise ValueError('No ImageLocation defined.')

        image_geometry_object = self.ImageLocation.get_geometry_object()

        if include_chip and self.SlantPlane is not None:
            center_pixel = self.SlantPlane.Physical.CenterPixel.get_array()
            chip_size = self.SlantPlane.Physical.ChipSize.get_array()
            shift = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype='float64')
            shift[:, 0] *= chip_size[0]
            shift[:, 1] *= chip_size[1]
            chip_rect = center_pixel + shift
            chip_area = Polygon(coordinates=[chip_rect, ])
            if isinstance(image_geometry_object, GeometryCollection):
                image_geometry_object.geometries.append(chip_area)
                return image_geometry_object
            else:
                return GeometryCollection(geometries=[image_geometry_object, chip_area])
        return image_geometry_object


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

    def set_image_location_from_sicd(
            self, sicd, layover_shift=True, populate_in_periphery=False, include_out_of_range=False):
        """
        Set the image location information with respect to the given SICD,
        assuming that the physical coordinates are populated. The `NumberOfObjectsInImage`
        will be set, and `NumberOfObjectsInScene` will be left unchanged.

        Parameters
        ----------
        sicd : SICDType
        layover_shift : bool
            Account for possible layover shift in calculated chip sizes?
        populate_in_periphery : bool
            Populate image information for objects on the periphery?
        include_out_of_range : bool
            Include the objects which are out of range (with no image location information)?
        """

        def update_object(temp_object, in_image_count):
            status = temp_object.set_image_location_from_sicd(
                sicd, populate_in_periphery=populate_in_periphery)
            use_object = False
            if status == 0:
                raise ValueError('Object already has image details set')
            if status == 1 or (status == 2 and populate_in_periphery):
                use_object = True
                temp_object.set_chip_details_from_sicd(
                    sicd, layover_shift=layover_shift, populate_in_periphery=True)
                in_image_count += 1
            return use_object, in_image_count

        objects_in_image = 0
        if include_out_of_range:
            # the objects list is just modified in place
            for the_object in self.Objects:
                _, objects_in_image = update_object(the_object, objects_in_image)
        else:
            # we make a new objects list
            objects = []
            for the_object in self.Objects:
                use_this_object, objects_in_image = update_object(the_object, objects_in_image)
                if use_this_object:
                    objects.append(the_object)
            self.Objects = objects
        self.NumberOfObjectsInImage = objects_in_image
