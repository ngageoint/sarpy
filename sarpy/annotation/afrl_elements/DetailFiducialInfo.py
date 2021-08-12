"""
Definition for the DetailFiducialInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

# TODO: comments on difficulties
#   - The PhysicalType seems half complete or something?

from typing import Optional, List

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import IntegerDescriptor, SerializableDescriptor, \
    SerializableListDescriptor, StringDescriptor
from sarpy.io.complex.sicd_elements.blocks import RowColType

from .base import DEFAULT_STRICT
from .blocks import LatLonEleType, RangeCrossRangeType


class ImageLocationType(Serializable):
    _fields = ('CenterPixel', )
    _required = _fields
    # descriptors
    CenterPixel = SerializableDescriptor(
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
    CenterPixel = SerializableDescriptor(
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
    Physical = SerializableDescriptor(
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
        'ImageLocation', 'GeoLocation',
        'Width_3dB', 'Width_18dB', 'Ratio_3dB_18dB',
        'PeakSideLobeRatio', 'IntegratedSideLobeRatio',
        'SlantPlane', 'GroundPlane')
    _required = (
        'FiducialType', 'ImageLocation', 'GeoLocation')
    _tag_overide = {
        'Width_3dB': '_3dBWidth', 'Width_18dB': '_18dBWidth', 'Ratio_3dB_18dB': '_3dB_18dBRatio'}
    # descriptors
    Name = StringDescriptor(
        'Name', _required, strict=DEFAULT_STRICT,
        docstring='Name of the fiducial.')  # type: Optional[str]
    SerialNumber = StringDescriptor(
        'SerialNumber', _required, strict=DEFAULT_STRICT,
        docstring='The serial number of the fiducial')  # type: Optional[str]
    FiducialType = StringDescriptor(
        'FiducialType', _required, strict=DEFAULT_STRICT,
        docstring='Description for the type of fiducial')  # type: str
    DatasetFiducialNumber = IntegerDescriptor(
        'DatasetFiducialNumber', _required,
        docstring='Unique number of the fiducial within the selected dataset, '
                  'defined by the RDE system')  # type: Optional[int]
    ImageLocation = SerializableDescriptor(
        'ImageLocation', ImageLocationType, _required,
        docstring='Center of the fiducial in the image'
    )  # type: Optional[ImageLocationType]
    GeoLocation = SerializableDescriptor(
        'GeoLocation', GeoLocationType, _required,
        docstring='Real physical location of the fiducial'
    )  # type: Optional[GeoLocationType]
    Width_3dB = SerializableDescriptor(
        'Width_3dB', RangeCrossRangeType, _required,
        docstring='The 3 dB impulse response width, in meters'
    )  # type: Optional[RangeCrossRangeType]
    Width_18dB = SerializableDescriptor(
        'Width_18dB', RangeCrossRangeType, _required,
        docstring='The 18 dB impulse response width, in meters'
    )  # type: Optional[RangeCrossRangeType]
    Ratio_3dB_18dB = SerializableDescriptor(
        'Ratio_3dB_18dB', RangeCrossRangeType, _required,
        docstring='Ratio of the 3 dB to 18 dB system impulse response width'
    )  # type: Optional[RangeCrossRangeType]
    PeakSideLobeRatio = SerializableDescriptor(
        'PeakSideLobeRatio', RangeCrossRangeType, _required,
        docstring='Ratio of the peak sidelobe intensity to the peak mainlobe intensity, '
                  'in dB')  # type: Optional[RangeCrossRangeType]
    IntegratedSideLobeRatio = SerializableDescriptor(
        'IntegratedSideLobeRatio', RangeCrossRangeType, _required,
        docstring='Ratio of all the energies in the sidelobes of the '
                  'system impulse response to the energy in the mainlobe, '
                  'in dB')  # type: Optional[RangeCrossRangeType]
    SlantPlane = SerializableDescriptor(
        'SlantPlane', PhysicalLocationType, _required,
        docstring='Center of the object in the slant plane'
    )  # type: Optional[PhysicalLocationType]
    GroundPlane = SerializableDescriptor(
        'GroundPlane', PhysicalLocationType, _required,
        docstring='Center of the object in the ground plane'
    )  # type: Optional[PhysicalLocationType]

    def __init__(self, Name=None, SerialNumber=None, FiducialType=None,
                 DatasetFiducialNumber=None, ImageLocation=None, GeoLocation=None,
                 Width_3dB=None, Width_18dB=None, Ratio_3dB_18dB=None,
                 PeakSideLobeRatio=None, IntegratedSideLobeRatio=None,
                 SlantPlane=None, GroundPlane=None,
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
        Width_3dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        Width_18dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        Ratio_3dB_18dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        PeakSideLobeRatio : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        IntegratedSideLobeRatio : None|RangeCrossRangeType|numpy.ndarray|list|tuple
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
        self.Width_3dB = Width_3dB
        self.Width_18dB = Width_18dB
        self.Ratio_3dB_18dB = Ratio_3dB_18dB
        self.PeakSideLobeRatio = PeakSideLobeRatio
        self.IntegratedSideLobeRatio = IntegratedSideLobeRatio
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
    NumberOfFiducialsInImage = IntegerDescriptor(
        'NumberOfFiducialsInImage', _required, strict=DEFAULT_STRICT,
        docstring='Number of ground truthed objects in the image.')  # type: int
    NumberOfFiducialsInScene = IntegerDescriptor(
        'NumberOfFiducialsInScene', _required, strict=DEFAULT_STRICT,
        docstring='Number of ground truthed objects in the scene.')  # type: int
    Fiducials = SerializableListDescriptor(
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
