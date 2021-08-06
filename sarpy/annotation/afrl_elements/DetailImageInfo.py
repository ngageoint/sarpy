"""
Definition for the DetailImageInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

from typing import Optional

import numpy

# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import _StringDescriptor, Serializable, \
    _SerializableDescriptor, _IntegerDescriptor, _StringEnumDescriptor, \
    _DateTimeDescriptor, _FloatDescriptor
from sarpy.io.complex.sicd_elements.blocks import RowColType
from .base import DEFAULT_STRICT
from .blocks import RangeCrossRangeType, RowColDoubleType

# TODO: Review what's marked required/optional - I'm sure it makes little sense
#  Questionable field definitions:
#   - there is a PixelSpacing and then slant/ground plane elements for pixel spacing.
#   - 3dbWidth is a poorly formed name, so I am omitting for now
#   - ZuluOffset seemingly assumes that the only possible offsets are integer valued - this is wrong
#   - DataCalibrated should obviously be xs:boolean - this is kludged badly for no reason
#   - DataCheckSum - it is unclear what this is the checksum of, and which checksum it would be (CRC-32?)
#   - DataByteOrder - why in the world is this even here?


class ClassificationMarkingsType(Serializable):
    _fields = (
        'Classification', 'Restrictions', 'ClassifiedBy', 'DeclassifyOn', 'DerivedFrom')
    _required = ('Classification', 'Restrictions')
    # descriptors
    Classification = _StringDescriptor(
        'Classification', _required, default_value='',
        docstring='The image classification')  # type: str
    Restrictions = _StringDescriptor(
        'Restrictions', _required, default_value='',
        docstring='Additional caveats to the classification')  # type: str
    ClassifiedBy = _StringDescriptor(
        'ClassifiedBy', _required)  # type: Optional[str]
    DeclassifyOn = _StringDescriptor(
        'DeclassifyOn', _required)  # type: Optional[str]
    DerivedFrom = _StringDescriptor(
        'DerivedFrom', _required)  # type: Optional[str]

    def __init__(self, Classification='', Restrictions='', ClassifiedBy=None,
                 DeclassifyOn=None, DerivedFrom=None, **kwargs):
        """
        Parameters
        ----------
        Classification : str
        Restrictions : str
        ClassifiedBy : None|str
        DeclassifyOn : None|str
        DerivedFrom : None|str
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Classification = Classification
        self.Restrictions = Restrictions
        self.ClassifiedBy = ClassifiedBy
        self.DeclassifyOn = DeclassifyOn
        self.DerivedFrom = DerivedFrom
        super(ClassificationMarkingsType, self).__init__(**kwargs)


class StringRangeCrossRangeType(Serializable):
    """
    A range and cross range attribute container
    """
    _fields = ('Range', 'CrossRange')
    _required = _fields
    # descriptors
    Range = _StringDescriptor(
        'Range', _required, strict=True, docstring='The Range attribute.')  # type: str
    CrossRange = _StringDescriptor(
        'CrossRange', _required, strict=True, docstring='The Cross Range attribute.')  # type: str

    def __init__(self, Range=None, CrossRange=None, **kwargs):
        """
        Parameters
        ----------
        Range : str
        CrossRange : str
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Range, self.CrossRange = Range, CrossRange
        super(StringRangeCrossRangeType, self).__init__(**kwargs)


class ImageCornerType(Serializable):
    _fields = (
        'UpperLeft', 'UpperRight', 'LowerRight', 'LowerLeft')
    _required = _fields
    # descriptors
    UpperLeft = _SerializableDescriptor(
        'UpperLeft', RowColDoubleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColDoubleType
    UpperRight = _SerializableDescriptor(
        'UpperRight', RowColDoubleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColDoubleType
    LowerRight = _SerializableDescriptor(
        'LowerRight', RowColDoubleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColDoubleType
    LowerLeft = _SerializableDescriptor(
        'LowerLeft', RowColDoubleType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: RowColDoubleType

    def __init__(self, UpperLeft=None, UpperRight=None,
                 LowerRight=None, LowerLeft=None, **kwargs):
        """
        Parameters
        ----------
        UpperLeft : RowColDoubleType|numpy.ndarray|list|tuple
        UpperRight : RowColDoubleType|numpy.ndarray|list|tuple
        LowerRight : RowColDoubleType|numpy.ndarray|list|tuple
        LowerLeft : RowColDoubleType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.UpperLeft = UpperLeft
        self.UpperRight = UpperRight
        self.LowerRight = LowerRight
        self.LowerLeft = LowerLeft
        super(ImageCornerType, self).__init__(**kwargs)


class PixelSpacingType(Serializable):
    _fields = ('PixelSpacing', )
    _required = _fields
    # descriptors
    PixelSpacing = _SerializableDescriptor(
        'PixelSpacing', RangeCrossRangeType, _required, strict=DEFAULT_STRICT,
        docstring='The center-to-center pixel spacing in meters.')  # type: RangeCrossRangeType

    def __init__(self, PixelSpacing=None, **kwargs):
        """
        Parameters
        ----------
        PixelSpacing : RangeCrossRangeType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PixelSpacing = PixelSpacing
        super(PixelSpacingType, self).__init__(**kwargs)


class DetailImageInfoType(Serializable):
    _fields = (
        'DataFilename', 'ClassificationMarkings', 'Filetype', 'DataCheckSum',
        'DataSize', 'DataPlane', 'DataDomain', 'DataType', 'BitsPerSample',
        'DataFormat', 'DataByteOrder', 'NumPixels', 'ImageCollectionDate', 'ZuluOffset',
        'SensorReferencePoint', 'SensorCalibrationFactor', 'DataCalibrated',
        'Resolution', 'PixelSpacing', 'WeightingType', 'OverSamplingFactor',
        'ImageQualityDescription', 'ImageHeading',
        'ImageCorners', 'SlantPlane', 'GroundPlane')
    _required = (
        'DataFilename', 'ClassificationMarkings', 'DataPlane', 'DataType',
        'DataFormat', 'NumPixels', 'ImageCollectionDate', 'SensorReferencePoint',
        'Resolution', 'PixelSpacing', 'WeightingType', 'ImageCorners')
    # descriptors
    DataFilename = _StringDescriptor(
        'DataFilename', _required,
        docstring='The base file name to which this information pertains')  # type: str
    ClassificationMarkings = _SerializableDescriptor(
        'ClassificationMarkings', ClassificationMarkingsType, _required,
        docstring='The classification information')  # type: ClassificationMarkingsType
    Filetype = _StringDescriptor(
        'Filetype', _required,
        docstring='The image file type')  # type: Optional[str]
    DataCheckSum = _StringDescriptor(
        'DataCheckSum', _required,
        docstring='The unique 32-bit identifier for the sensor block data')  # type: Optional[str]
    DataSize = _IntegerDescriptor(
        'DataSize', _required,
        docstring='The image size in bytes')  # type: Optional[int]
    DataPlane = _StringEnumDescriptor(
        'DataPlane', {'Slant', 'Ground'}, _required, default_value='Slant',
        docstring='The image plane.')  # type: str
    DataDomain = _StringDescriptor(
        'DataDomain', _required,
        docstring='The image data domain')  # type: Optional[str]
    DataType = _StringDescriptor(
        'DataType', _required,
        docstring='The image data type')  # type: Optional[str]
    BitsPerSample = _IntegerDescriptor(
        'BitsPerSample', _required,
        docstring='The number of bits per sample')  # type: Optional[int]
    DataFormat = _StringDescriptor(
        'DataFormat', _required,
        docstring='The image data format')  # type: str
    DataByteOrder = _StringEnumDescriptor(
        'DataByteOrder', {'Big-Endian', 'Little-Endian'}, _required,
        docstring='The image data byte order.')  # type: Optional[str]
    NumPixels = _SerializableDescriptor(
        'NumPixels', RowColType, _required,
        docstring='The number of image pixels')  # type: RowColType
    ImageCollectionDate = _DateTimeDescriptor(
        'ImageCollectionDate', _required,
        docstring='The date/time of the image collection in UTC')  # type: Optional[numpy.datetime64]
    ZuluOffset = _IntegerDescriptor(
        'ZuluOffset', _required,
        docstring='The local time offset from UTC')  # type: Optional[int]  # TODO: this isn't always integer
    SensorReferencePoint = _StringEnumDescriptor(
        'DataPlane', {'Left', 'Right', 'Top', 'Bottom'}, _required,
        docstring='Description of the sensor location relative to the scene.')  # type: Optional[str]
    SensorCalibrationFactor = _FloatDescriptor(
        'SensorCalibrationFactor', _required,
        docstring='Multiplicative factor used to scale raw image data to the return '
                  'of a calibrated reference reflector or active source')  # type: Optional[float]
    DataCalibrated = _StringDescriptor(
        'DataCalibrated', _required,
        docstring='Has the data been calibrated?')  # type: Optional[str]  # TODO: this obviously should be a xs:boolean
    Resolution = _SerializableDescriptor(
        'Resolution', RangeCrossRangeType, _required,
        docstring='Resolution (intrinsic) of the sensor system/mode in meters.')  # type: RangeCrossRangeType
    PixelSpacing = _SerializableDescriptor(
        'PixelSpacing', RangeCrossRangeType, _required,
        docstring='Pixel spacing of the image in meters.')  # type: RangeCrossRangeType
    WeightingType = _SerializableDescriptor(
        'WeightingType', StringRangeCrossRangeType, _required,
        docstring='Weighting function applied to the image during formation.')  # type: StringRangeCrossRangeType
    OverSamplingFactor = _SerializableDescriptor(
        'OverSamplingFactor', RangeCrossRangeType, _required,
        docstring='The factor by which the pixel space is oversampled.')  # type: RangeCrossRangeType
    ImageQualityDescription = _StringDescriptor(
        'ImageQualityDescription', _required,
        docstring='General description of image quality')  # type: Optional[str]
    ImageHeading = _FloatDescriptor(
        'ImageHeading', _required,
        docstring='Image heading relative to True North, in degrees')  # type: Optional[float]
    ImageCorners = _SerializableDescriptor(
        'ImageCorners', ImageCornerType, _required,
        docstring='The image corners')  # type: ImageCornerType
    SlantPlane = _SerializableDescriptor(
        'SlantPlane', PixelSpacingType, _required,
        docstring='The slant plane pixel spacing')  # type: Optional[PixelSpacingType]
    GroundPlane = _SerializableDescriptor(
        'GroundPlane', PixelSpacingType, _required,
        docstring='The ground plane pixel spacing')  # type: Optional[PixelSpacingType]

    def __init__(self, DataFilename=None, ClassificationMarkings=None,
                 FileType=None, DataCheckSum=None, DataSize=None,
                 DataPlane='Slant', DataDomain=None, DataType=None,
                 BitsPerSample=None, DataFormat=None, DataByteOrder=None, NumPixels=None,
                 ImageCollectionDate=None, ZuluOffset=None,
                 SensorReferencePoint=None, SensorCalibrationFactor=None,
                 DataCalibrated=None, Resolution=None, PixelSpacing=None,
                 WeightingType=None, OverSamplingFactor=None,
                 ImageQualityDescription=None, ImageHeading=None, ImageCorners=None,
                 SlantPlane=None, GroundPlane=None, **kwargs):
        """
        Parameters
        ----------
        DataFilename : str
        ClassificationMarkings : ClassificationMarkingsType
        FileType : str
        DataCheckSum : None|str
        DataSize : int
        DataPlane : str
        DataDomain : None|str
        DataType : None|str
        BitsPerSample : None|int
        DataFormat : None|str
        DataByteOrder : None|str
        NumPixels : RowColType|numpy.ndarray|list|tuple
        ImageCollectionDate : numpy.datetime64|datetime|date|str
        ZuluOffset : None|int
        SensorReferencePoint : None|str
        SensorCalibrationFactor : None|float
        DataCalibrated : None|str
        Resolution : RangeCrossRangeType|numpy.ndarray|list|tuple
        PixelSpacing : RangeCrossRangeType|numpy.ndarray|list|tuple
        WeightingType : StringRangeCrossRangeType
        OverSamplingFactor : RangeCrossRangeType
        ImageQualityDescription : None|str
        ImageHeading : None|float
        ImageCorners : ImageCornerType
        SlantPlane : None|PixelSpacingType
        GroundPlane : None|PixelSpacingType
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DataFilename = DataFilename
        if ClassificationMarkings is None:
            self.ClassificationMarkings = ClassificationMarkingsType()
        else:
            self.ClassificationMarkings = ClassificationMarkings
        self.Filetype = FileType
        self.DataCheckSum = DataCheckSum
        self.DataSize = DataSize
        self.DataPlane = DataPlane
        self.DataDomain = DataDomain
        self.DataType = DataType
        self.BitsPerSample = BitsPerSample
        self.DataFormat = DataFormat
        self.DataByteOrder = DataByteOrder
        self.NumPixels = NumPixels
        self.ImageCollectionDate = ImageCollectionDate
        self.ZuluOffset = ZuluOffset
        self.SensorReferencePoint = SensorReferencePoint
        self.SensorCalibrationFactor = SensorCalibrationFactor
        self.DataCalibrated = DataCalibrated
        self.Resolution = Resolution
        self.PixelSpacing = PixelSpacing
        self.WeightingType = WeightingType
        self.OverSamplingFactor = OverSamplingFactor
        self.ImageQualityDescription = ImageQualityDescription
        self.ImageHeading = ImageHeading
        self.ImageCorners = ImageCorners
        self.SlantPlane = SlantPlane
        self.GroundPlane = GroundPlane
        super(DetailImageInfoType, self).__init__(**kwargs)
