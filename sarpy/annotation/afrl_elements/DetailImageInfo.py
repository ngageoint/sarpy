"""
Definition for the DetailImageInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")

from typing import Optional
import os
import numpy

from sarpy.io.xml.base import Serializable, Arrayable
from sarpy.io.xml.descriptors import StringDescriptor, SerializableDescriptor, \
    IntegerDescriptor, StringEnumDescriptor, DateTimeDescriptor, FloatDescriptor
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.blocks import LatLonType

from .base import DEFAULT_STRICT
from .blocks import RangeCrossRangeType

# TODO: Review what's marked required/optional - I'm sure it makes little sense
#  Questionable field definitions:
#   - there is a PixelSpacing and then slant/ground plane elements for pixel spacing?
#   - ZuluOffset seemingly assumes that the only possible offsets are integer valued - this is wrong.
#   - DataCalibrated should obviously be xs:boolean - this is kludged badly for no reason
#   - DataCheckSum - it is unclear what this is the checksum of, and which checksum it would be (CRC-32?)
#   - DataByteOrder - why in the world is this even here?


class NumPixelsType(Serializable, Arrayable):
    """A row and column attribute container - used as indices into array(s)."""
    _fields = ('NumRows', 'NumCols')
    _required = _fields
    NumRows = IntegerDescriptor(
        'NumRows', _required, strict=True, docstring='The number of rows.')  # type: int
    NumCols = IntegerDescriptor(
        'NumCols', _required, strict=True, docstring='The number of columns.')  # type: int

    def __init__(self, NumRows=None, NumCols=None, **kwargs):
        """
        Parameters
        ----------
        NumRows : int
        NumCols : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NumRows, self.NumCols = NumRows, NumCols
        super(NumPixelsType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.int64):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [NumRows, NumCols]
        """

        return numpy.array([self.NumRows, self.NumCols], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [NumRows, NumCols]

        Returns
        -------
        NumPixelsType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(NumRows=array[0], NumCols=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class ClassificationMarkingsType(Serializable):
    _fields = (
        'Classification', 'Restrictions', 'ClassifiedBy', 'DeclassifyOn', 'DerivedFrom')
    _required = ('Classification', 'Restrictions')
    # descriptors
    Classification = StringDescriptor(
        'Classification', _required, default_value='',
        docstring='The image classification')  # type: str
    Restrictions = StringDescriptor(
        'Restrictions', _required, default_value='',
        docstring='Additional caveats to the classification')  # type: str
    ClassifiedBy = StringDescriptor(
        'ClassifiedBy', _required)  # type: Optional[str]
    DeclassifyOn = StringDescriptor(
        'DeclassifyOn', _required)  # type: Optional[str]
    DerivedFrom = StringDescriptor(
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
    Range = StringDescriptor(
        'Range', _required, strict=True, docstring='The Range attribute.')  # type: str
    CrossRange = StringDescriptor(
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
    UpperLeft = SerializableDescriptor(
        'UpperLeft', LatLonType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonType
    UpperRight = SerializableDescriptor(
        'UpperRight', LatLonType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonType
    LowerRight = SerializableDescriptor(
        'LowerRight', LatLonType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonType
    LowerLeft = SerializableDescriptor(
        'LowerLeft', LatLonType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: LatLonType

    def __init__(self, UpperLeft=None, UpperRight=None,
                 LowerRight=None, LowerLeft=None, **kwargs):
        """
        Parameters
        ----------
        UpperLeft : LatLonType|numpy.ndarray|list|tuple
        UpperRight : LatLonType|numpy.ndarray|list|tuple
        LowerRight : LatLonType|numpy.ndarray|list|tuple
        LowerLeft : LatLonType|numpy.ndarray|list|tuple
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
    PixelSpacing = SerializableDescriptor(
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
        'Width_3dB', 'ImageQualityDescription', 'ImageHeading',
        'ImageCorners', 'SlantPlane', 'GroundPlane', 'SceneCenterReferenceLine')
    _required = (
        'DataFilename', 'ClassificationMarkings', 'DataPlane', 'DataType',
        'DataFormat', 'NumPixels', 'ImageCollectionDate', 'SensorReferencePoint',
        'Resolution', 'PixelSpacing', 'WeightingType', 'ImageCorners')
    _tag_overide = {'Width_3dB': '_3dBWidth'}
    # descriptors
    DataFilename = StringDescriptor(
        'DataFilename', _required,
        docstring='The base file name to which this information pertains')  # type: str
    ClassificationMarkings = SerializableDescriptor(
        'ClassificationMarkings', ClassificationMarkingsType, _required,
        docstring='The classification information')  # type: ClassificationMarkingsType
    Filetype = StringDescriptor(
        'Filetype', _required,
        docstring='The image file type')  # type: Optional[str]
    DataCheckSum = StringDescriptor(
        'DataCheckSum', _required,
        docstring='The unique 32-bit identifier for the sensor block data')  # type: Optional[str]
    DataSize = IntegerDescriptor(
        'DataSize', _required,
        docstring='The image size in bytes')  # type: Optional[int]
    DataPlane = StringEnumDescriptor(
        'DataPlane', {'Slant', 'Ground'}, _required, default_value='Slant',
        docstring='The image plane.')  # type: str
    DataDomain = StringDescriptor(
        'DataDomain', _required,
        docstring='The image data domain')  # type: Optional[str]
    DataType = StringDescriptor(
        'DataType', _required,
        docstring='The image data type')  # type: Optional[str]
    BitsPerSample = IntegerDescriptor(
        'BitsPerSample', _required,
        docstring='The number of bits per sample')  # type: Optional[int]
    DataFormat = StringDescriptor(
        'DataFormat', _required,
        docstring='The image data format')  # type: str
    DataByteOrder = StringEnumDescriptor(
        'DataByteOrder', {'Big-Endian', 'Little-Endian'}, _required,
        docstring='The image data byte order.')  # type: Optional[str]
    NumPixels = SerializableDescriptor(
        'NumPixels', NumPixelsType, _required,
        docstring='The number of image pixels')  # type: NumPixelsType
    ImageCollectionDate = DateTimeDescriptor(
        'ImageCollectionDate', _required,
        docstring='The date/time of the image collection in UTC')  # type: Optional[numpy.datetime64]
    ZuluOffset = IntegerDescriptor(
        'ZuluOffset', _required,
        docstring='The local time offset from UTC')  # type: Optional[int]  # TODO: this isn't always integer
    SensorReferencePoint = StringEnumDescriptor(
        'DataPlane', {'Left', 'Right', 'Top', 'Bottom'}, _required,
        docstring='Description of the sensor location relative to the scene.')  # type: Optional[str]
    SensorCalibrationFactor = FloatDescriptor(
        'SensorCalibrationFactor', _required,
        docstring='Multiplicative factor used to scale raw image data to the return '
                  'of a calibrated reference reflector or active source')  # type: Optional[float]
    DataCalibrated = StringDescriptor(
        'DataCalibrated', _required,
        docstring='Has the data been calibrated?')  # type: Optional[str]  # TODO: this obviously should be a xs:boolean
    Resolution = SerializableDescriptor(
        'Resolution', RangeCrossRangeType, _required,
        docstring='Resolution (intrinsic) of the sensor system/mode in meters.')  # type: RangeCrossRangeType
    PixelSpacing = SerializableDescriptor(
        'PixelSpacing', RangeCrossRangeType, _required,
        docstring='Pixel spacing of the image in meters.')  # type: RangeCrossRangeType
    WeightingType = SerializableDescriptor(
        'WeightingType', StringRangeCrossRangeType, _required,
        docstring='Weighting function applied to the image during formation.')  # type: StringRangeCrossRangeType
    OverSamplingFactor = SerializableDescriptor(
        'OverSamplingFactor', RangeCrossRangeType, _required,
        docstring='The factor by which the pixel space is oversampled.')  # type: Optional[RangeCrossRangeType]
    Width_3dB = SerializableDescriptor(
        'Width_3dB', RangeCrossRangeType, _required,
        docstring='The 3 dB system impulse response with, in meters')  # type: Optional[RangeCrossRangeType]
    ImageQualityDescription = StringDescriptor(
        'ImageQualityDescription', _required,
        docstring='General description of image quality')  # type: Optional[str]
    ImageHeading = FloatDescriptor(
        'ImageHeading', _required,
        docstring='Image heading relative to True North, in degrees')  # type: Optional[float]
    ImageCorners = SerializableDescriptor(
        'ImageCorners', ImageCornerType, _required,
        docstring='The image corners')  # type: ImageCornerType
    SlantPlane = SerializableDescriptor(
        'SlantPlane', PixelSpacingType, _required,
        docstring='The slant plane pixel spacing')  # type: Optional[PixelSpacingType]
    GroundPlane = SerializableDescriptor(
        'GroundPlane', PixelSpacingType, _required,
        docstring='The ground plane pixel spacing')  # type: Optional[PixelSpacingType]
    SceneCenterReferenceLine = FloatDescriptor(
        'SceneCenterReferenceLine', _required,
        docstring='The ideal line (heading) at the intersection of the radar '
                  'line-of-sight with the horizontal reference plane '
                  'created by the forward motion of the aircraft, '
                  'in degrees')  # type: Optional[float]

    def __init__(self, DataFilename=None, ClassificationMarkings=None,
                 FileType=None, DataCheckSum=None, DataSize=None,
                 DataPlane='Slant', DataDomain=None, DataType=None,
                 BitsPerSample=None, DataFormat=None, DataByteOrder=None, NumPixels=None,
                 ImageCollectionDate=None, ZuluOffset=None,
                 SensorReferencePoint=None, SensorCalibrationFactor=None,
                 DataCalibrated=None, Resolution=None, PixelSpacing=None,
                 WeightingType=None, OverSamplingFactor=None, Width_3dB=None,
                 ImageQualityDescription=None, ImageHeading=None, ImageCorners=None,
                 SlantPlane=None, GroundPlane=None, SceneCenterReferenceLine=None,
                 **kwargs):
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
        NumPixels : NumPixelsType|numpy.ndarray|list|tuple
        ImageCollectionDate : numpy.datetime64|datetime|date|str
        ZuluOffset : None|int
        SensorReferencePoint : None|str
        SensorCalibrationFactor : None|float
        DataCalibrated : None|str
        Resolution : RangeCrossRangeType|numpy.ndarray|list|tuple
        PixelSpacing : RangeCrossRangeType|numpy.ndarray|list|tuple
        WeightingType : StringRangeCrossRangeType
        OverSamplingFactor : None|RangeCrossRangeType
        Width_3dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        ImageQualityDescription : None|str
        ImageHeading : None|float
        ImageCorners : ImageCornerType
        SlantPlane : None|PixelSpacingType
        GroundPlane : None|PixelSpacingType
        SceneCenterReferenceLine : None|float
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
        self.Width_3dB = Width_3dB

        self.ImageQualityDescription = ImageQualityDescription
        self.ImageHeading = ImageHeading
        self.ImageCorners = ImageCorners
        self.SlantPlane = SlantPlane
        self.GroundPlane = GroundPlane
        self.SceneCenterReferenceLine = SceneCenterReferenceLine
        super(DetailImageInfoType, self).__init__(**kwargs)

    @classmethod
    def from_sicd(cls, sicd, base_file_name, file_type='NITF02.10'):
        """
        Construct the ImageInfo from the sicd object and given image file name.

        Parameters
        ----------
        sicd : SICDType
        base_file_name : str
        file_type : str
            The file type. This should probably always be NITF02.10 for now.

        Returns
        -------
        DetailImageInfoType
        """

        pixel_type = sicd.ImageData.PixelType
        if pixel_type == 'RE32F_IM32F':
            data_type = 'in-phase/quadrature'
            bits_per_sample = 32
            data_format = 'float'
        elif pixel_type == 'RE16I_IM16I':
            data_type = 'in-phase/quadrature'
            bits_per_sample = 16
            data_format = 'integer'
        elif pixel_type == 'AMP8I_PHS8I':
            data_type = 'magnitude-phase'
            bits_per_sample = 8
            data_format = 'unsigned integer'
        else:
            raise ValueError('Unhandled')

        icps = ImageCornerType(
            UpperLeft=sicd.GeoData.ImageCorners.FRFC,
            UpperRight=sicd.GeoData.ImageCorners.FRLC,
            LowerRight=sicd.GeoData.ImageCorners.LRLC,
            LowerLeft=sicd.GeoData.ImageCorners.LRFC)

        if sicd.Grid.ImagePlane == 'SLANT':
            data_plane = 'Slant'
        elif sicd.Grid.ImagePlane == 'Ground':
            data_plane = 'Ground'
        else:
            data_plane = None

        return DetailImageInfoType(
            DataFilename=base_file_name,
            ClassificationMarkings=ClassificationMarkingsType(
                Classification=sicd.CollectionInfo.Classification),
            FileType=file_type,
            DataPlane=data_plane,
            DataType=data_type,
            BitsPerSample=bits_per_sample,
            DataFormat=data_format,
            DataByteOrder='Big-Endian',
            NumPixels=(sicd.ImageData.NumRows, sicd.ImageData.NumCols),
            ImageCollectionDate=sicd.Timeline.CollectStart,
            SensorReferencePoint='Top',
            Resolution=(sicd.Grid.Row.ImpRespWid, sicd.Grid.Col.ImpRespWid),
            PixelSpacing=(sicd.Grid.Row.SS, sicd.Grid.Col.SS),
            WeightingType=StringRangeCrossRangeType(
                Range=sicd.Grid.Row.WgtType.WindowName,
                CrossRange=sicd.Grid.Col.WgtType.WindowName),
            Width_3dB=(sicd.Grid.Row.ImpRespWid, sicd.Grid.Col.ImpRespWid),  # TODO: I don't think that this is correct?
            ImageHeading=sicd.SCPCOA.AzimAng,
            ImageCorners=icps)
