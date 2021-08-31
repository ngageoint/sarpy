"""
Simple helper functions for constructing the AFRL structure assuming either a
known ground truth scenario or inferred analyst truth scenario.
"""

__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

from typing import Dict

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd import SICDReader

from sarpy.annotation.afrl_elements.Research import ResearchType
from sarpy.annotation.afrl_elements.DetailCollectionInfo import DetailCollectionInfoType
from sarpy.annotation.afrl_elements.DetailSubCollectionInfo import DetailSubCollectionInfoType
from sarpy.annotation.afrl_elements.DetailObjectInfo import DetailObjectInfoType, \
    PlaneNominalType, NominalType, TheObjectType, \
    GeoLocationType as ObjectGeoLocation, \
    ImageLocationType as ObjectImageLocation
from sarpy.annotation.afrl_elements.DetailFiducialInfo import DetailFiducialInfoType, \
    TheFiducialType, GeoLocationType as FiducialGeoLocation, \
    ImageLocationType as FiducialImageLocation
from sarpy.annotation.afrl_elements.DetailImageInfo import DetailImageInfoType
from sarpy.annotation.afrl_elements.DetailSensorInfo import DetailSensorInfoType

from sarpy.annotation.label import LabelSchema, FileLabelCollection, LabelCollection, \
    LabelFeature, LabelMetadataList, LabelMetadata


class GroundTruthConstructor(object):
    """
    This class is a helper for performing a ground truth construction.
    """

    __slots__ = (
        '_collection_info', '_subcollection_info', '_objects', '_fiducials')

    def __init__(self, collection_info, subcollection_info):
        """

        Parameters
        ----------
        collection_info : DetailCollectionInfoType
        subcollection_info : DetailSubCollectionInfoType
        """

        self._collection_info = collection_info
        self._subcollection_info = subcollection_info
        self._objects = []
        self._fiducials = []

    def add_fiducial(self, the_fiducial):
        """
        Adds the given fiducial to the collection.

        Parameters
        ----------
        the_fiducial : TheFiducialType
        """

        if not isinstance(the_fiducial, TheFiducialType):
            raise TypeError('Requires an object of type `TheFiducialType`, got `{}`'.format(type(the_fiducial)))
        if the_fiducial.ImageLocation is not None:
            raise ValueError('The fiducial has ImageLocation already set.')
        if the_fiducial.SlantPlane is not None or the_fiducial.GroundPlane is not None:
            raise ValueError('The fiducial already has the SlantPlane or GroundPlane set.')
        self._fiducials.append(the_fiducial)

    def add_fiducial_from_arguments(self, Name=None, SerialNumber=None, FiducialType=None, GeoLocation=None):
        """
        Adds a fiducial to the collection.

        Parameters
        ----------
        Name : str
        SerialNumber : None|str
        FiducialType : None|str
        GeoLocation : FiducialGeoLocation
        """

        self.add_fiducial(
            TheFiducialType(
                Name=Name,
                SerialNumber=SerialNumber,
                FiducialType=FiducialType,
                GeoLocation=GeoLocation))

    def add_object(self, the_object):
        """
        Adds the given object to the collection.

        Parameters
        ----------
        the_object : TheObjectType
        """

        if not isinstance(the_object, TheObjectType):
            raise TypeError('Requires an object of type `TheObjectType`, got `{}`'.format(type(the_object)))
        if the_object.ImageLocation is not None:
            raise ValueError('The object has ImageLocation already set.')
        if the_object.SlantPlane is not None or the_object.GroundPlane is not None:
            raise ValueError('The object already has the SlantPlane or GroundPlane set.')
        self._objects.append(the_object)

    def add_object_from_arguments(
            self, SystemName=None, SystemComponent=None, NATOName=None,
            Function=None, Version=None, DecoyType=None, SerialNumber=None,
            ObjectClass='Unknown', ObjectSubClass='Unknown', ObjectTypeClass='Unknown',
            ObjectType='Unknown', ObjectLabel=None, Size=None,
            Orientation=None,
            Articulation=None, Configuration=None,
            Accessories=None, PaintScheme=None, Camouflage=None,
            Obscuration=None, ObscurationPercent=None, ImageLevelObscuration=None,
            GeoLocation=None, TargetToClutterRatio=None, VisualQualityMetric=None,
            UnderlyingTerrain=None, OverlyingTerrain=None,
            TerrainTexture=None, SeasonalCover=None):
        """
        Adds an object to the collection.

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
        GeoLocation : ObjectGeoLocation
        TargetToClutterRatio : None|str
        VisualQualityMetric : None|str
        UnderlyingTerrain : None|str
        OverlyingTerrain : None|str
        TerrainTexture : None|str
        SeasonalCover : None|str
        """

        self.add_object(
            TheObjectType(SystemName=SystemName,
                          SystemComponent=SystemComponent,
                          NATOName=NATOName,
                          Function=Function,
                          Version=Version,
                          DecoyType=DecoyType,
                          SerialNumber=SerialNumber,
                          ObjectClass=ObjectClass,
                          ObjectSubClass=ObjectSubClass,
                          ObjectTypeClass=ObjectTypeClass,
                          ObjectType=ObjectType,
                          ObjectLabel=ObjectLabel,
                          Size=Size,
                          Orientation=Orientation,
                          Articulation=Articulation,
                          Configuration=Configuration,
                          Accessories=Accessories,
                          PaintScheme=PaintScheme,
                          Camouflage=Camouflage,
                          Obscuration=Obscuration,
                          ObscurationPercent=ObscurationPercent,
                          ImageLevelObscuration=ImageLevelObscuration,
                          GeoLocation=GeoLocation,
                          TargetToClutterRatio=TargetToClutterRatio,
                          VisualQualityMetric=VisualQualityMetric,
                          UnderlyingTerrain=UnderlyingTerrain,
                          OverlyingTerrain=OverlyingTerrain,
                          TerrainTexture=TerrainTexture,
                          SeasonalCover=SeasonalCover))

    def get_final_structure(self):
        """
        It is anticipated that this might be reused to localize for a whole series
        of different sicd files.

        Gets **a static copy** of the constructed AFRL Research structure. This has the
        provided DetailCollectionInfo and DetailSubCollectionInfo populated. It also
        has the DetailObjectInfo and DetailFiducialInfo with the GeoLocation
        ground truth details that have been provided.

        No image location information has been populated, and there are no
        DetailImageInfo or DetailSensorInfo populated, because these are independent
        of ground truth.

        Returns
        -------
        ResearchType
        """

        return ResearchType(
            DetailCollectionInfo=self._collection_info,
            DetailSubCollectionInfo=self._subcollection_info,
            DetailFiducialInfo=DetailFiducialInfoType(
                NumberOfFiducialsInScene=len(self._fiducials),
                Fiducials=self._fiducials),
            DetailObjectInfo=DetailObjectInfoType(
                NumberOfObjectsInScene=len(self._objects),
                Objects=self._objects)).copy()

    def localize_for_sicd(self, sicd, base_sicd_file, populate_in_periphery=False, include_out_of_range=False):
        """
        Localize the AFRL structure for the given sicd structure.

        This returns **a static copy** of the AFRL structure, and this method
        can be repeatedly applied for a sequence of different sicd files which all
        apply to the same ground truth scenario.

        Parameters
        ----------
        sicd : SICDType
        base_sicd_file : str
        populate_in_periphery : bool
        include_out_of_range : bool

        Returns
        -------
        ResearchType
        """

        out_research = self.get_final_structure()
        # TODO: nominal chip size?
        out_research.apply_sicd(
            sicd,
            base_sicd_file,
            populate_in_periphery=populate_in_periphery,
            include_out_of_range=include_out_of_range)
        return out_research

    def localize_for_sicd_reader(self, sicd_reader, populate_in_periphery=False, include_out_of_range=False):
        """
        Localize the AFRL structure for the given sicd file.

        This returns **a static copy** of the AFRL structure, and this method
        can be repeatedly applied for a sequence of different sicd files which all
        apply to the same ground truth scenario.

        Parameters
        ----------
        sicd_reader : SICDReader
        populate_in_periphery : bool
        include_out_of_range : bool

        Returns
        -------
        ResearchType
        """

        out_research = self.get_final_structure()
        out_research.apply_sicd_reader(
            sicd_reader,
            populate_in_periphery=populate_in_periphery,
            include_out_of_range=include_out_of_range)
        return out_research


class AnalystTruthConstructor(object):
    """
    This class is a helper for performing an analyst truth construction.
    """

    __slots__ = (
        '_sicd', '_base_file', '_nominal_chip_size',
        '_collection_info', '_subcollection_info', '_image_info', '_sensor_info',
        '_objects', '_fiducials',
        '_projection_type', '_proj_kwargs')

    def __init__(self, sicd, base_file, collection_info, subcollection_info,
                 nominal_chip_size=40, projection_type='HAE', proj_kwargs=None):
        """

        Parameters
        ----------
        sicd : SICDType
        base_file : str
        collection_info : DetailCollectionInfoType
        subcollection_info : DetailSubCollectionInfoType
        nominal_chip_size : int|float
            The nominal chip size in meters.
        projection_type : str
            One of 'PLANE', 'HAE', or 'DEM'. The value of `proj_kwargs`
            will need to be appropriate.
        proj_kwargs : None|Dict
            The keyword arguments for the :func:`SICDType.project_image_to_ground_geo` method.
        """

        self._sicd = sicd
        self._base_file = base_file
        self._nominal_chip_size = float(nominal_chip_size)
        if self._nominal_chip_size <= 1:
            raise ValueError('noiminal chip size must be at least 1, got {}'.format(self._nominal_chip_size))

        # TODO: should we create a decent shell for general Analyst Truth
        #  collection and subcollection info?
        self._collection_info = collection_info
        self._subcollection_info = subcollection_info
        self._image_info = DetailImageInfoType.from_sicd(self._sicd, self._base_file)
        self._sensor_info = DetailSensorInfoType.from_sicd(self._sicd)
        self._objects = []
        self._fiducials = []

        self._projection_type = projection_type
        self._proj_kwargs = {} if proj_kwargs is None else proj_kwargs

    @property
    def image_info(self):
        """
        DetailImageInfoType: The basic image info object derived from the sicd
        """

        return self._image_info

    @property
    def sensor_info(self):
        """
        DetailSensorInfoType: The basic sensor info object derived from the sicd.
        """

        return self._sensor_info

    def add_fiducial(self, the_fiducial):
        """
        Adds the given fiducial to the collection. Note that this object will be modified in place.

        Parameters
        ----------
        the_fiducial : TheFiducialType
        """

        if not isinstance(the_fiducial, TheFiducialType):
            raise TypeError('Requires an object of type `TheFiducialType`, got `{}`'.format(type(the_fiducial)))
        if the_fiducial.GeoLocation is not None:
            raise ValueError('The fiducial has GeoLocation already set.')
        the_fiducial.set_geo_location_from_sicd(self._sicd, projection_type=self._projection_type, **self._proj_kwargs)
        self._fiducials.append(the_fiducial)

    def add_fiducial_from_arguments(self, Name=None, SerialNumber=None, FiducialType=None, ImageLocation=None):
        """
        Adds a fiducial to the collection.

        Parameters
        ----------
        Name : str
        SerialNumber : None|str
        FiducialType : None|str
        ImageLocation : FiducialImageLocation
        """

        self.add_fiducial(
            TheFiducialType(
                Name=Name,
                SerialNumber=SerialNumber,
                FiducialType=FiducialType,
                ImageLocation=ImageLocation))

    def add_object(self, the_object):
        """
        Adds the object to the collection. Note that this object will be modified in place.

        Parameters
        ----------
        the_object : TheObjectType
        """

        if not isinstance(the_object, TheObjectType):
            raise TypeError('Requires an object of type `TheObjectType`, got `{}`'.format(type(the_object)))
        if the_object.GeoLocation is not None:
            raise ValueError('The object has GeoLocation already set.')
        the_object.set_geo_location_from_sicd(
            self._sicd, projection_type=self._projection_type, **self._proj_kwargs)
        the_object.set_chip_details_from_sicd(self._sicd, populate_in_periphery=True)
        self._objects.append(the_object)

    def add_object_from_arguments(
            self, SystemName=None, SystemComponent=None, NATOName=None,
            Function=None, Version=None, DecoyType=None, SerialNumber=None,
            ObjectClass='Unknown', ObjectSubClass='Unknown', ObjectTypeClass='Unknown',
            ObjectType='Unknown', ObjectLabel=None, Size=None,
            Orientation=None,
            Articulation=None, Configuration=None,
            Accessories=None, PaintScheme=None, Camouflage=None,
            Obscuration=None, ObscurationPercent=None, ImageLevelObscuration=None,
            ImageLocation=None,
            TargetToClutterRatio=None, VisualQualityMetric=None,
            UnderlyingTerrain=None, OverlyingTerrain=None,
            TerrainTexture=None, SeasonalCover=None):
        """
        Adds an object to the collection.

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
        ImageLocation : ObjectImageLocation
        TargetToClutterRatio : None|str
        VisualQualityMetric : None|str
        UnderlyingTerrain : None|str
        OverlyingTerrain : None|str
        TerrainTexture : None|str
        SeasonalCover : None|str
        """

        self.add_object(
            TheObjectType(SystemName=SystemName,
                          SystemComponent=SystemComponent,
                          NATOName=NATOName,
                          Function=Function,
                          Version=Version,
                          DecoyType=DecoyType,
                          SerialNumber=SerialNumber,
                          ObjectClass=ObjectClass,
                          ObjectSubClass=ObjectSubClass,
                          ObjectTypeClass=ObjectTypeClass,
                          ObjectType=ObjectType,
                          ObjectLabel=ObjectLabel,
                          Size=Size,
                          Orientation=Orientation,
                          Articulation=Articulation,
                          Configuration=Configuration,
                          Accessories=Accessories,
                          PaintScheme=PaintScheme,
                          Camouflage=Camouflage,
                          Obscuration=Obscuration,
                          ObscurationPercent=ObscurationPercent,
                          ImageLevelObscuration=ImageLevelObscuration,
                          ImageLocation=ImageLocation,
                          TargetToClutterRatio=TargetToClutterRatio,
                          VisualQualityMetric=VisualQualityMetric,
                          UnderlyingTerrain=UnderlyingTerrain,
                          OverlyingTerrain=OverlyingTerrain,
                          TerrainTexture=TerrainTexture,
                          SeasonalCover=SeasonalCover))

    def get_final_structure(self):
        """
        This is not anticipated to be reused, so the raw progress to date is returned.
        Care should be taken in modifying the returned structure directly.

        Returns
        -------
        ResearchType
        """

        nominal_rows = self._nominal_chip_size/self._sicd.Grid.Row.SS
        nominal_cols = self._nominal_chip_size/self._sicd.Grid.Col.SS
        slant = PlaneNominalType(Nominal=NominalType(ChipSize=(nominal_rows, nominal_cols)))

        return ResearchType(
            DetailCollectionInfo=self._collection_info,
            DetailSubCollectionInfo=self._subcollection_info,
            DetailImageInfo=self._image_info,
            DetailSensorInfo=self._sensor_info,
            DetailFiducialInfo=DetailFiducialInfoType(
                NumberOfFiducialsInScene=len(self._fiducials),
                Fiducials=self._fiducials),
            DetailObjectInfo=DetailObjectInfoType(
                NumberOfObjectsInScene=len(self._objects),
                SlantPlane=slant,
                Objects=self._objects))


def convert_afrl_to_native(research, include_chip=False):
    """
    Converts an AFRL structure to a label structure for simple viewing.

    Parameters
    ----------
    research : ResearchType
    include_chip : bool
        Include the chip definition in the geometry structure?

    Returns
    -------
    FileLabelCollection
    """

    def _convert_object_to_json(t_object):
        # extract the "properties"
        label_metadata = LabelMetadata(label_id=t_object.ObjectLabel)
        label_metadata_list = LabelMetadataList(elements=[label_metadata, ])

        return LabelFeature(
            geometry=t_object.get_image_geometry_object_for_sicd(include_chip=include_chip),
            properties=label_metadata_list)

    if not isinstance(research, ResearchType):
        raise TypeError('Expected ResearchType, got type `{}`'.format(type(research)))
    if research.DetailObjectInfo is None or \
            research.DetailObjectInfo.Objects is None or \
            len(research.DetailObjectInfo.Objects) == 0:
        raise ValueError('Nothing to be done')

    # create our adhoc containers
    label_schema = LabelSchema(version='AdHoc')
    annotation_collection = LabelCollection()
    for the_object in research.DetailObjectInfo.Objects:
        new_key = the_object.ObjectLabel
        if new_key not in label_schema:
            label_schema.add_entry(new_key, new_key)
        annotation_collection.add_feature(_convert_object_to_json(the_object))

    # finalize the collection
    return FileLabelCollection(
        label_schema,
        annotations=annotation_collection,
        image_file_name=research.DetailImageInfo.DataFilename)
