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
        Construct from the corresponding Geolocation using the sicd
        projection model.

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

        absolute_pixel_location, _, _ = the_structure.project_ground_to_image_geo(
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

        if image_location is None or image_location.CenterPixel is None:
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

        coords = image_location.CenterPixel.get_array(dtype='float64') + image_shift
        geo_coords = the_structure.project_image_to_ground_geo(
            coords, ordering='latlong', projection_type=projection_type, **kwargs)

        out = GeoLocationType(CenterPixel=geo_coords)
        return out


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
        'IPRWidth3dB', 'IPRWidth18dB', 'IPRWidth3dB18dBRatio',
        'PeakSideLobeRatio', 'IntegratedSideLobeRatio',
        'SlantPlane', 'GroundPlane')
    _required = (
        'FiducialType', 'ImageLocation', 'GeoLocation')
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
    IPRWidth3dB = SerializableDescriptor(
        'IPRWidth3dB', RangeCrossRangeType, _required,
        docstring='The 3 dB impulse response width, in meters'
    )  # type: Optional[RangeCrossRangeType]
    IPRWidth18dB = SerializableDescriptor(
        'IPRWidth18dB', RangeCrossRangeType, _required,
        docstring='The 18 dB impulse response width, in meters'
    )  # type: Optional[RangeCrossRangeType]
    IPRWidth3dB18dBRatio = SerializableDescriptor(
        'IPRWidth3dB18dBRatio', RangeCrossRangeType, _required,
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
                 IPRWidth3dB=None, IPRWidth18dB=None, IPRWidth3dB18dBRatio=None,
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
        IPRWidth3dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        IPRWidth18dB : None|RangeCrossRangeType|numpy.ndarray|list|tuple
        IPRWidth3dB18dBRatio : None|RangeCrossRangeType|numpy.ndarray|list|tuple
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
        self.IPRWidth3dB = IPRWidth3dB
        self.IPRWidth18dB = IPRWidth18dB
        self.IPRWidth3dB18dBRatio = IPRWidth3dB18dBRatio
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

        if self.IPRWidth3dB is None or override:
            self.IPRWidth3dB = RangeCrossRangeType.from_array(
                (sicd.Grid.Row.ImpRespWid, sicd.Grid.Col.ImpRespWid))
            # TODO: this seems questionable to me?

    def set_image_location_from_sicd(self, sicd, populate_in_periphery=False):
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
        elif (-3 < center_pixel[0] < rows + 2) and (-3 < center_pixel[1] < cols + 2):
            placement = 2
        else:
            placement = 3

        if placement == 3 or (placement == 2 and not populate_in_periphery):
            return placement

        self.ImageLocation = image_location
        self.SlantPlane = PhysicalLocationType(Physical=image_location)
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

    def __init__(self, NumberOfFiducialsInImage=None, NumberOfFiducialsInScene=None,
                 Fiducials=None, **kwargs):
        """
        Parameters
        ----------
        NumberOfFiducialsInImage : int
        NumberOfFiducialsInScene : int
        Fiducials : None|List[TheFiducialType]
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NumberOfFiducialsInImage = NumberOfFiducialsInImage
        self.NumberOfFiducialsInScene = NumberOfFiducialsInScene
        self.Fiducials = Fiducials
        super(DetailFiducialInfoType, self).__init__(**kwargs)

    def set_image_location_from_sicd(
            self, sicd, populate_in_periphery=False, include_out_of_range=False):
        """
        Set the image location information with respect to the given SICD,
        assuming that the physical coordinates are populated. The `NumberOfFiducialsInImage`
        will be set, and `NumberOfFiducialsInScene` will be left unchanged.

        Parameters
        ----------
        sicd : SICDType
        populate_in_periphery : bool
            Populate image information for objects on the periphery?
        include_out_of_range : bool
            Include the objects which are out of range (with no image location information)?
        """

        def update_fiducial(temp_fid, in_image_count):
            temp_fid.set_default_width_from_sicd(sicd)  # todo: I'm not sure that this is correct?
            status = temp_fid.set_image_location_from_sicd(sicd, populate_in_periphery=populate_in_periphery)
            use_fid = False
            if status == 0:
                raise ValueError('Fiducial already has image details set')
            if status == 1 or (status == 2 and populate_in_periphery):
                use_fid = True
                in_image_count += 1
            return use_fid, in_image_count

        fid_in_image = 0
        if include_out_of_range:
            # the fiducials list is just modified in place
            for the_fid in self.Fiducials:
                _, fid_in_image = update_fiducial(the_fid, fid_in_image)
        else:
            # the fiducials list is just modified in place
            fiducials = []
            for the_fid in self.Fiducials:
                use_this_fid, fid_in_image = update_fiducial(the_fid, fid_in_image)
                if use_this_fid:
                    fiducials.append(the_fid)
            self.Fiducials = fiducials
        self.NumberOfFiducialsInImage = fid_in_image
