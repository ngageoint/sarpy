"""
Definition for the DetailFiducialInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

# TODO: comments on difficulties
#   - the field names starting with #db have been excluded, since poorly formed
#   - The PhyscialType seems half complete or something?

from typing import Optional, List

# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, \
    _IntegerDescriptor, _SerializableDescriptor, _SerializableListDescriptor, \
    _StringDescriptor
from sarpy.io.complex.sicd_elements.blocks import RowColType
from .base import DEFAULT_STRICT
from .blocks import LatLonEleType


class ImageLocationType(Serializable):
    _fields = ('CenterPixel', )
    _required = _fields
    # descriptors
    CenterPixel = _SerializableDescriptor(
        'CenterPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='The pixel location of the center of the object')  # type: RowColType

    def __init__(self, CenterPixel=None, **kwargs):
        """
        Parameters
        ----------
        CenterPixel : RowColType|numpy.ndarray|list|tuple
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CenterPixel = CenterPixel
        super(ImageLocationType, self).__init__(**kwargs)


class GeoLocationType(Serializable):
    _fields = ('CenterPixel', )
    _required = _fields
    # descriptors
    CenterPixel = _SerializableDescriptor(
        'CenterPixel', LatLonEleType, _required, strict=DEFAULT_STRICT,
        docstring='The physical location of the center of the object')  # type: LatLonEleType

    def __init__(self, CenterPixel=None, **kwargs):
        """
        Parameters
        ----------
        CenterPixel : LatLonEleType|numpy.ndarray|list|tuple
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CenterPixel = CenterPixel
        super(GeoLocationType, self).__init__(**kwargs)


class PhysicalLocationType(Serializable):
    _fields = ('Physical', )
    _required = _fields
    # descriptors
    Physical = _SerializableDescriptor(
        'Physical', ImageLocationType, _required, strict=DEFAULT_STRICT,
    )  # type: ImageLocationType

    def __init__(self, Physical=None, **kwargs):
        """
        Parameters
        ----------
        Physical : ImageLocationType
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Physical = Physical
        super(PhysicalLocationType, self).__init__(**kwargs)


class TheFiducialType(Serializable):
    _fields = (
        'Name', 'SerialNumber', 'FiducialType', 'DatasetFiducialNumber',
        'ImageLocation', 'GeoLocation', 'SlantPlane', 'GroundPlane')
    _required = (
        'FiducialType', 'ImageLocation', 'GeoLocation')
    # descriptors
    Name = _StringDescriptor(
        'Name', _required, strict=DEFAULT_STRICT,
        docstring='Name of the fiducial.')  # type: Optional[str]
    SerialNumber = _StringDescriptor(
        'SerialNumber', _required, strict=DEFAULT_STRICT,
        docstring='The serial number of the fiducial')  # type: Optional[str]
    FiducialType = _StringDescriptor(
        'FiducialType', _required, strict=DEFAULT_STRICT,
        docstring='Description for the type of fiducial')  # type: str
    DatasetFiducialNumber = _IntegerDescriptor(
        'DatasetFiducialNumber', _required,
        docstring='Unique number of the fiducial within the selected dataset, '
                  'defined by the RDE system')  # type: Optional[int]
    ImageLocation = _SerializableDescriptor(
        'ImageLocation', ImageLocationType, _required,
        docstring='Center of the fiducial in the image'
    )  # type: Optional[ImageLocationType]
    GeoLocation = _SerializableDescriptor(
        'GeoLocation', GeoLocationType, _required,
        docstring='Real physical location of the fiducial'
    )  # type: Optional[GeoLocationType]
    SlantPlane = _SerializableDescriptor(
        'SlantPlane', PhysicalLocationType, _required,
        docstring='Center of the object in the slant plane'
    )  # type: Optional[PhysicalLocationType]
    GroundPlane = _SerializableDescriptor(
        'GroundPlane', PhysicalLocationType, _required,
        docstring='Center of the object in the ground plane'
    )  # type: Optional[PhysicalLocationType]

    def __init__(self, Name=None, SerialNumber=None, FiducialType=None,
                 DatasetFiducialNumber=None, ImageLocation=None,
                 GeoLocation=None, SlantPlane=None, GroundPlane=None,
                 **kwargs):
        """
        Parameters
        ----------
        Name : str
        SerialNumber : None|str
        FiducialType : str
        DatasetFiducialNumber : None|int
        ImageLocation : ImageLocationType
        GeoLocation : GeoLocationType
        SlantPlane : None|PhysicalLocationType
        GroundPlane : None|PhysicalLocationType
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.SerialNumber = SerialNumber
        self.FiducialType = FiducialType
        self.DatasetFiducialNumber = DatasetFiducialNumber
        self.ImageLocation = ImageLocation
        self.GeoLocation = GeoLocation
        self.SlantPlane = SlantPlane
        self.GroundPlane = GroundPlane
        super(TheFiducialType, self).__init__(**kwargs)


class DetailFiducialInfoType(Serializable):
    _fields = (
        'NumberOfFiducialsInImage', 'NumberOfFiducialsInScene', 'Fiducials')
    _required = (
        'NumberOfFiducialsInImage', 'NumberOfFiducialsInScene', 'Fiducials')
    _collections_tags = {'Fiducials': {'array': False, 'child_tag': 'Fiducial'}}
    # descriptors
    NumberOfFiducialsInImage = _IntegerDescriptor(
        'NumberOfFiducialsInImage', _required, strict=DEFAULT_STRICT,
        docstring='Number of ground truthed objects in the image.')  # type: int
    NumberOfFiducialsInScene = _IntegerDescriptor(
        'NumberOfFiducialsInScene', _required, strict=DEFAULT_STRICT,
        docstring='Number of ground truthed objects in the scene.')  # type: int
    Fiducials = _SerializableListDescriptor(
        'Fiducials', TheFiducialType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The object collection')  # type: List[TheFiducialType]

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        super(DetailFiducialInfoType, self).__init__(**kwargs)
