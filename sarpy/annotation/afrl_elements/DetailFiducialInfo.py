"""
Definition for the DetailFiducialInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")


from typing import Optional, List
import logging
import numpy

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import IntegerDescriptor, SerializableDescriptor, \
    SerializableListDescriptor, StringDescriptor
from sarpy.io.complex.sicd_elements.blocks import RowColType
from sarpy.io.complex.sicd_elements.SICD import SICDType

from .base import DEFAULT_STRICT
from .blocks import LatLonEleType, RangeCrossRangeType

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_geolocation(cls, geo_location, the_structure):
        """
        Construct from the corresponding Geolocation.

        Parameters
        ----------
        geo_location : GeoLocationType
        the_structure : SICDType|SIDDType

        Returns
        -------
        None|ImageLocationType
        """

        if geo_location is None or geo_location.CenterPixel is None:
            return None

        if not the_structure.can_project_coordinates():
            logger.warning(
                'This sicd does not permit projection,\n\t'
                'so the image location can not be inferred')
            return None

        if isinstance(the_structure, SICDType):
            image_shift = numpy.array(
                [the_structure.ImageData.FirstRow, the_structure.ImageData.FirstCol], dtype='float64')
        else:
            image_shift = numpy.zeros((2, ), dtype='float64')

        absolute_pixel_location = the_structure.project_ground_to_image_geo(
            geo_location.CenterPixel.get_array(dtype='float64'), ordering='latlong')

        if numpy.any(numpy.isnan(absolute_pixel_location)):
            return None

        return ImageLocationType(CenterPixel=absolute_pixel_location - image_shift)


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

    def set_default_width_from_sicd(self, sicd, override=False):
        """
        Sets a default value for the 3dB Width from the given SICD.

        Parameters
        ----------
        sicd : SICDType
        override : bool
            Override any present value?
        """

        if self.Width_3dB is None or override:
            self.Width_3dB = RangeCrossRangeType.from_array((sicd.Grid.Row.ImpRespWid, sicd.Grid.Col.ImpRespWid))
            # TODO: this seems questionable to me?

    def set_image_location_from_sicd(self, sicd):
        """
        Set the image location information with respect to the given SICD.

        Parameters
        ----------
        sicd : SICDType

        Returns
        -------
        int
            -1 - insufficient metadata to proceed
            0 - nothing to be done
            1 - successful
            2 - object in image periphery, not populating
            3 - object not in image field
        """

        if self.ImageLocation is not None or self.SlantPlane is not None:
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

        image_location = ImageLocationType.from_geolocation(self.GeoLocation, sicd)
        # check bounding information
        rows = sicd.ImageData.NumRows
        cols = sicd.ImageData.NumCols
        center_pixel = image_location.CenterPixel.get_array(dtype='float64')

        if (0 < center_pixel[0] < rows - 1) and (0 < center_pixel[1] < cols - 1):
            placement = 1
        elif (-3 < center_pixel[0] < rows + 2) and (-3 < center_pixel[1] < cols  + 2):
            placement = 2
        else:
            placement = 3

        if placement in [2, 3]:
            return placement

        self.ImageLocation = image_location
        self.SlantPlane = PhysicalLocationType(Physical=image_location)


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
