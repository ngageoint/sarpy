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
    TheObjectType, GeoLocationType as ObjectGeolocation, \
    ImageLocationType as ObjectImageLocation
from sarpy.annotation.afrl_elements.DetailFiducialInfo import DetailFiducialInfoType, \
    TheFiducialType, GeoLocationType as FiducialGeolocation, \
    ImageLocationType as FiducialImageLocation
from sarpy.annotation.afrl_elements.DetailImageInfo import DetailImageInfoType
from sarpy.annotation.afrl_elements.DetailSensorInfo import DetailSensorInfoType


class GroundTruthConstructor(object):
    """
    This class is a helper for performing a ground truth construction.
    """

    __slots__ = (
        '_collection_info', '_subcollection_info', '_label_lookup',
        '_objects', '_fiducials')

    def __init__(self, collection_info, subcollection_info, label_lookup):
        """

        Parameters
        ----------
        collection_info : DetailCollectionInfoType
        subcollection_info : DetailSubCollectionInfoType
        label_lookup : Dict[str, Dict]
        """

        self._collection_info = collection_info
        self._subcollection_info = subcollection_info
        self._label_lookup = label_lookup
        self._objects = []
        self._fiducials = []

    def _verify_key(self, the_label):
        label_info = self._label_lookup.get(the_label, None)
        if label_info is None:
            raise KeyError('Not such key `{}` in the label_lookup dictionary'.format(the_label))
        return label_info

    def add_label(self, the_label, **kwargs):
        """
        Add the given label, extract information from the provided
        label_lookup dictionary and passing through kwargs to the object
        or fiducial constructor.

        Parameters
        ----------
        the_label : str
        kwargs
            keyword arguments passed through to the object or fiducial constructor.
            These will override the contents of the label_lookup dictionary.
            Note that image specific arguments are ignored, and GeoLocation must
            be provided in either the lookup table or the keyword arguments.
        """

        label_info = self._verify_key(the_label)
        the_type = label_info.get('type', None).lower()

        if the_type is None or the_type == 'object':
            # by default, we will assume that things are objects
            self._add_object(the_label, label_info, **kwargs)
        elif the_type == 'fiducial':
            self._add_fiducial(the_label, label_info, **kwargs)
        else:
            raise ValueError('Got unhandled type `{}` for key `{}`'.format(the_type, the_label))

    def _add_fiducial(self, the_label, label_info, **kwargs):
        location = label_info.get('GeoLocation', None)
        location = kwargs.get('GeoLocation', location)
        if location is None:
            raise ValueError('GeoLocation must be provided.')

        this_kwargs = {'GeoLocation': FiducialGeolocation(**location)}
        for field in ['Name', 'SerialNumber', 'FiducialType']:
            if field in kwargs:
                this_kwargs[field] = kwargs[field]
            elif field in label_info:
                this_kwargs[field] = label_info[field]
        if 'Name' not in this_kwargs:
            this_kwargs['Name'] = the_label
        self._fiducials.append(TheFiducialType(**this_kwargs))

    def _add_object(self, the_label, label_info, **kwargs):
        location = label_info.get('GeoLocation', None)
        location = kwargs.get('GeoLocation', location)
        if location is None:
            raise ValueError('GeoLocation must be provided.')

        this_kwargs = {'GeoLocation': ObjectGeolocation(**location)}
        for field in [
            'SystemName', 'SystemComponent', 'NATOName',
            'Function', 'Version', 'DecoyType', 'SerialNumber',
            'ObjectClass', 'ObjectSubClass', 'ObjectTypeClass', 'ObjectType', 'ObjectLabel',
            'Size', 'Orientation', 'Articulation', 'Configuration', 'Accessories', 'PaintScheme',
            'Camouflage', 'Obscuration', 'ObscurationPercent', 'ImageLevelObscuration',
            'TargetToClutterRatio', 'VisualQualityMetric',
            'UnderlyingTerrain', 'OverlyingTerrain', 'TerrainTexture', 'SeasonalCover']:
            if field in kwargs:
                this_kwargs[field] = kwargs[field]
            elif field in label_info:
                this_kwargs[field] = label_info[field]

        if 'SystemName' not in this_kwargs:
            this_kwargs['SystemName'] = the_label
        if 'ObjectLabel' not in this_kwargs:
            this_kwargs['ObjectLabel'] = the_label
        self._objects.append(TheObjectType(**this_kwargs))

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
        '_sicd', '_base_file',
        '_collection_info', '_subcollection_info', '_label_lookup',
        '_objects', '_fiducials')

    def __init__(self, sicd, base_file, label_lookup, collection_info, subcollection_info):
        """

        Parameters
        ----------
        sicd : SICDType
        base_file : str
        label_lookup : Dict[str, Dict]
        collection_info : DetailCollectionInfoType
        subcollection_info : DetailSubCollectionInfoType
        """

        self._sicd = sicd
        self._base_file = base_file
        self._label_lookup = label_lookup

        # TODO: should we create a decent shell for general Analyst Truth
        #  collection and subcollection info?
        self._collection_info = collection_info
        self._subcollection_info = subcollection_info
        self._objects = []
        self._fiducials = []

    def _verify_key(self, the_label):
        label_info = self._label_lookup.get(the_label, None)
        if label_info is None:
            raise KeyError('Not such key `{}` in the label_lookup dictionary'.format(the_label))
        return label_info

    def add_label(self, the_label, **kwargs):
        """
        Add the given label, extract information from the provided
        label_lookup dictionary and passing through kwargs to the object
        or fiducial constructor.

        Parameters
        ----------
        the_label : str
        kwargs
            keyword arguments passed through to the object or fiducial constructor.
            These will override the contents of the label_lookup dictionary.
            Note that geolocation specific arguments are ignored, and ImageLocation must
            be provided in either the lookup table or the keyword arguments.
        """

        label_info = self._verify_key(the_label)
        the_type = label_info.get('type', None).lower()

        if the_type is None or the_type == 'object':
            # by default, we will assume that things are objects
            self._add_object(the_label, label_info, **kwargs)
        elif the_type == 'fiducial':
            self._add_fiducial(the_label, label_info, **kwargs)
        else:
            raise ValueError('Got unhandled type `{}` for key `{}`'.format(the_type, the_label))

    def _add_fiducial(self, the_label, label_info, projection_type='HAE', proj_kwargs=None, **kwargs):
        if proj_kwargs is None:
            proj_kwargs = {}
        location = label_info.get('ImageLocation', None)
        location = kwargs.get('ImageLocation', location)
        if location is None:
            raise ValueError('ImageLocation must be provided.')

        this_kwargs = {'ImageLocation': FiducialImageLocation(**location)}
        for field in ['Name', 'SerialNumber', 'FiducialType']:
            if field in kwargs:
                this_kwargs[field] = kwargs[field]
            elif field in label_info:
                this_kwargs[field] = label_info[field]
        if 'Name' not in this_kwargs:
            this_kwargs['Name'] = the_label
        the_fiducial = TheFiducialType(**this_kwargs)
        the_fiducial.set_geo_location_from_sicd(self._sicd, projection_type=projection_type, **proj_kwargs)
        self._fiducials.append(the_fiducial)

    def _add_object(self, the_label, label_info, projection_type='HAE', proj_kwargs=None, **kwargs):
        if proj_kwargs is None:
            proj_kwargs = {}

        location = label_info.get('ImageLocation', None)
        location = kwargs.get('ImageLocation', location)
        if location is None:
            raise ValueError('ImageLocation must be provided.')

        this_kwargs = {'ImageLocation': ObjectImageLocation(**location)}
        for field in [
            'SystemName', 'SystemComponent', 'NATOName',
            'Function', 'Version', 'DecoyType', 'SerialNumber',
            'ObjectClass', 'ObjectSubClass', 'ObjectTypeClass', 'ObjectType', 'ObjectLabel',
            'Size', 'Orientation', 'Articulation', 'Configuration', 'Accessories', 'PaintScheme',
            'Camouflage', 'Obscuration', 'ObscurationPercent', 'ImageLevelObscuration',
            'TargetToClutterRatio', 'VisualQualityMetric',
            'UnderlyingTerrain', 'OverlyingTerrain', 'TerrainTexture', 'SeasonalCover']:
            if field in kwargs:
                this_kwargs[field] = kwargs[field]
            elif field in label_info:
                this_kwargs[field] = label_info[field]

        if 'SystemName' not in this_kwargs:
            this_kwargs['SystemName'] = the_label
        if 'ObjectLabel' not in this_kwargs:
            this_kwargs['ObjectLabel'] = the_label
        the_object = TheObjectType(**this_kwargs)
        the_object.set_geo_location_from_sicd(self._sicd, projection_type=projection_type, **proj_kwargs)
        self._objects.append(the_object)

    def get_final_structure(self):
        """
        This is not anticipated to be reused, so the raw progress to date is returned.
        Care should be taken in modifying the returned structure directly.

        Returns
        -------
        ResearchType
        """

        return ResearchType(
            DetailCollectionInfo=self._collection_info,
            DetailSubCollectionInfo=self._subcollection_info,
            DetailImageInfo=DetailImageInfoType.from_sicd(self._sicd, self._base_file),
            DetailSensorInfo=DetailSensorInfoType.from_sicd(self._sicd),
            DetailFiducialInfo=DetailFiducialInfoType(
                NumberOfFiducialsInScene=len(self._fiducials),
                Fiducials=self._fiducials),
            DetailObjectInfo=DetailObjectInfoType(
                NumberOfObjectsInScene=len(self._objects),
                Objects=self._objects))
