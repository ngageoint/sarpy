"""
Base annotation types for general use - based on the geojson implementation
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
import os
from uuid import uuid4
from typing import Optional, Dict, List, Any, Union
import json

from collections import OrderedDict

from sarpy.geometry.geometry_elements import Jsonable, FeatureCollection, Feature, \
    GeometryCollection, GeometryObject, Geometry, basic_assemble_from_collection


_BASE_VERSION = "Base:1.0"

logger = logging.getLogger(__name__)


class GeometryProperties(Jsonable):
    __slots__ = ('_uid', '_name', '_color')
    _type = 'GeometryProperties'

    def __init__(self, uid=None, name=None, color=None):
        self._name = None
        self._color = None

        if uid is None:
            uid = str(uuid4())
        if not isinstance(uid, str):
            raise TypeError('uid must be a string')
        self._uid = uid

        self.name = name
        self.color = color

    @property
    def uid(self):
        """
        str: A unique identifier for the associated geometry element
        """

        return self._uid

    @property
    def name(self):
        """
        Optional[str]: The name
        """

        return self._name

    @name.setter
    def name(self, value):
        if value is None or isinstance(value, str):
            self._name = value
        else:
            raise TypeError('Got unexpected type for name')

    @property
    def color(self):
        """
        Optional[str]: The color
        """

        return self._color

    @color.setter
    def color(self, value):
        if value is None or isinstance(value, str):
            self._color = value
        else:
            raise TypeError('Got unexpected type for color')

    @classmethod
    def from_dict(cls, the_json):
        """
        Deserialize from json.

        Parameters
        ----------
        the_json : Dict

        Returns
        -------
        GeometryProperties
        """

        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('GeometryProperties cannot be constructed from {}'.format(the_json))
        return cls(
            uid=the_json.get('uid', None),
            name=the_json.get('name', None),
            color=the_json.get('color', None))

    def to_dict(self, parent_dict=None):
        """
        Serialize to json.

        Parameters
        ----------
        parent_dict : None|Dict

        Returns
        -------
        Dict
        """

        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['uid'] = self.uid
        if self.name is not None:
            parent_dict['name'] = self.name
        if self.color is not None:
            parent_dict['color'] = self.color
        return parent_dict


class AnnotationProperties(Jsonable):
    """
    The basic common properties for an annotation
    """

    __slots__ = ('_name', '_description', '_directory', '_geometry_properties', '_parameters')
    _type = 'AnnotationProperties'

    def __init__(self, name=None, description=None, directory=None,
                 geometry_properties=None, parameters=None):
        """

        Parameters
        ----------
        name : Optional[str]
        description : Optional[str]
        directory : Optional[str]
        geometry_properties : None|List[GeometryProperties]
        parameters : Optional[Jsonable]
        """

        self._name = None
        self._description = None
        self._directory = None
        self._geometry_properties = []
        self._parameters = None

        self.name = name
        self.description = description
        self.directory = directory
        self.geometry_properties = geometry_properties
        self.parameters = parameters

    @property
    def name(self):
        """
        Optional[str]: The name
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is None or isinstance(value, str):
            self._name = value
        else:
            raise TypeError('Got unexpected type for name')

    @property
    def description(self):
        """
        Optional[str]: The description
        """
        return self._description

    @description.setter
    def description(self, value):
        if value is None or isinstance(value, str):
            self._description = value
        else:
            raise TypeError('Got unexpected type for description')

    @property
    def directory(self):
        """
        Optional[str]: The directory - for basic display and/or subdivision purposes
        """
        return self._directory

    @directory.setter
    def directory(self, value):
        if value is None:
            self._directory = None
            return

        if not isinstance(value, str):
            raise TypeError('Got unexpected type for directory')

        parts = [entry.strip() for entry in value.split('/')]
        self._directory = '/'.join([entry for entry in parts if entry != ''])

    @property
    def geometry_properties(self):
        # type: () -> List[GeometryProperties]
        """
        List[GeometryProperties]: The geometry properties.
        """

        return self._geometry_properties

    @geometry_properties.setter
    def geometry_properties(self, value):
        if value is None:
            self._geometry_properties = []
            return
        if not isinstance(value, list):
            raise TypeError('Got unexpected value for geometry properties')

        self._geometry_properties = []
        for entry in value:
            self.add_geometry_property(entry)

    def add_geometry_property(self, entry):
        """
        Add a geometry property to the list.

        .. warning::

            Care should be taken that this list stay in sync with the parent geometry.

        Parameters
        ----------
        entry: Dict|GeometryProperties
            The geometry properties instance of serialized version of it.
        """

        if isinstance(entry, dict):
            entry = GeometryProperties.from_dict(entry)

        if not isinstance(entry, GeometryProperties):
            raise TypeError('Got entry of unexpected type for geometry properties list')
        self._geometry_properties.append(entry)

    def get_geometry_property(self, item):
        """
        Fetches the appropriate geometry property.

        Parameters
        ----------
        item : int|str
            The geometry properties uid or integer index.

        Returns
        -------
        GeometryProperties

        Raises
        ------
        KeyError
        """

        return self.get_geometry_property_and_index(item)[0]

    def get_geometry_property_and_index(self, item):
        """
        Fetches the appropriate geometry property and its integer index.

        Parameters
        ----------
        item : int|str
            The geometry properties uid or integer index.

        Returns
        -------
        (GeometryProperties, int)

        Raises
        ------
        KeyError
        """
        if isinstance(item, int):
            return self._geometry_properties[item], item
        elif isinstance(item, str):
            for index, entry in enumerate(self.geometry_properties):
                if entry.uid == item:
                    return entry, index
        raise KeyError('Got unrecognized geometry key `{}`'.format(item))

    @property
    def parameters(self):
        """
        Optional[Jsonable]: The parameters
        """

        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if value is None or isinstance(value, Jsonable):
            self._parameters = value
        else:
            raise TypeError('Got unexpected type for parameters')

    @classmethod
    def from_dict(cls, the_json):
        """
        Deserialize from json.

        Parameters
        ----------
        the_json : Dict

        Returns
        -------
        AnnotationProperties
        """

        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('AnnotationProperties cannot be constructed from {}'.format(the_json))
        return cls(
            name=the_json.get('name', None),
            description=the_json.get('description', None),
            directory=the_json.get('directory', None),
            geometry_properties=the_json.get('geometry_properties', None),
            parameters=the_json.get('parameters', None))

    def to_dict(self, parent_dict=None):
        """
        Serialize to json.

        Parameters
        ----------
        parent_dict : None|Dict

        Returns
        -------
        Dict
        """

        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        for field in ['name', 'description', 'directory']:
            value = getattr(self, field)
            if value is not None:
                parent_dict[field] = value

        if self.geometry_properties is not None:
            parent_dict['geometry_properties'] = [entry.to_dict() for entry in self.geometry_properties]
        if self.parameters is not None:
            parent_dict['parameters'] = self.parameters.to_dict()
        return parent_dict

    def replicate(self):
        geom_properties = None if self.geometry_properties is None else \
            [entry.replicate() for entry in self.geometry_properties]
        params = None if self.parameters is None else self.parameters.replicate()

        the_type = self.__class__
        return the_type(
            name=self.name, description=self.description, directory=self.directory,
            geometry_properties=geom_properties, parameters=params)


class AnnotationFeature(Feature):
    """
    An extension of the Feature class which has the properties attribute
    populated with AnnotationProperties instance.
    """
    _allowed_geometries = None

    @property
    def properties(self):
        """
        The properties.

        Returns
        -------
        None|AnnotationProperties
        """

        return self._properties

    @properties.setter
    def properties(self, properties):
        if properties is None:
            self._properties = AnnotationProperties()
        elif isinstance(properties, AnnotationProperties):
            self._properties = properties
        elif isinstance(properties, dict):
            self._properties = AnnotationProperties.from_dict(properties)
        else:
            raise TypeError('Got an unexpected type for properties attribute of class {}'.format(self.__class__))

    def get_name(self):
        """
        Gets a useful name.

        Returns
        -------
        str
        """

        if self.properties is None or self.properties.name is None:
            return self.uid

        return self.properties.name

    @property
    def geometry(self):
        """
        The geometry object.

        Returns
        -------
        GeometryObject|GeometryCollection
        """

        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        if isinstance(geometry, dict):
            geometry = Geometry.from_dict(geometry)
        if geometry is None:
            self._geometry = None
            return
        if not isinstance(geometry, Geometry):
            raise TypeError('geometry must be an instance of Geometry, got `{}`'.format(type(geometry)))
        if geometry.is_collection:
            geometry = basic_assemble_from_collection(geometry)
        self._geometry = self._validate_geometry_element(geometry)

    @property
    def geometry_count(self):
        """
        int: The number of base geometry elements
        """

        if self.geometry is None:
            return 0
        elif not self.geometry.is_collection:
            return 1
        else:
            return len(self.geometry.collection)

    def get_geometry_name(self, item):
        """
        Gets the name, or a reasonable default, for the geometry.

        Parameters
        ----------
        item : int|str

        Returns
        -------
        str
        """

        geometry, geom_properties = self.get_geometry_and_geometry_properties(item)
        return '<{}>'.format(geometry.__class__.__name__) if geom_properties.name is None else geom_properties.name

    def get_geometry_property(self, item):
        """
        Gets the geometry properties object for the given index/uid.

        Parameters
        ----------
        item : int|str
            The geometry properties uid or integer index.

        Returns
        -------
        GeometryProperties

        Raises
        ------
        KeyError
        """

        return self.properties.get_geometry_property(item)

    def get_geometry_property_and_index(self, item):
        """
        Gets the geometry properties object and integer index for the given index/uid.

        Parameters
        ----------
        item : int|str
            The geometry properties uid or integer index.

        Returns
        -------
        (GeometryProperties, int)

        Raises
        ------
        KeyError
        """

        return self.properties.get_geometry_property_and_index(item)

    def get_geometry_and_geometry_properties(self, item):
        """
        Gets the geometry and geometry properties object for the given index/uid.

        Parameters
        ----------
        item : int|str
            The geometry properties uid or integer index.

        Returns
        -------
        (Point|Line|Polygon, GeometryProperties)

        Raises
        ------
        KeyError
        """

        if self.geometry is None:
            raise ValueError('No geometry defined.')

        geom_prop, index = self.get_geometry_property_and_index(item)

        index = int(index)
        if not (0 <= index < self.geometry_count):
            raise KeyError('invalid geometry index')
        if self.geometry.is_collection:
            return self.geometry.collection[index], geom_prop
        else:
            return self.geometry, geom_prop

    def get_geometry_element(self, item):
        """
        Gets the basic geometry object at the given index.

        Parameters
        ----------
        item : int|str
            The integer index or associated geometry properties uid.

        Returns
        -------
        Point|Line|Polygon

        Raises
        ------
        ValueError|KeyError
        """

        return self.get_geometry_and_geometry_properties(item)[0]

    def _validate_geometry_element(self, geometry):
        if geometry is None:
            return geometry
        if not isinstance(geometry, Geometry):
            raise TypeError('geometry must be an instance of Geometry base class')

        if self._allowed_geometries is not None and geometry.__class__ not in self._allowed_geometries:
            raise TypeError('geometry ({}) is not of one of the allowed types ({})'.format(geometry, self._allowed_geometries))
        return geometry

    def add_geometry_element(self, geometry, properties=None):
        """
        Adds the given geometry to the feature geometry (collection).

        Parameters
        ----------
        geometry : GeometryObject
        properties : None|GeometryProperties
        """

        if not isinstance(geometry, GeometryObject):
            raise TypeError('geometry must be a GeometryObject instance')

        if properties is None:
            properties = GeometryProperties()
        if not isinstance(properties, GeometryProperties):
            raise TypeError('properties must be a GeometryProperties instance')
        if self.properties is None:
            self.properties = AnnotationProperties()

        # handle the geometry
        self._geometry = self._validate_geometry_element(
            basic_assemble_from_collection(self.geometry, geometry))

        # add the geometry property
        self.properties.add_geometry_property(properties)

        # check that they are in sync
        if len(self.properties.geometry_properties) != self.geometry_count:
            logger.warning(
                'There are {} geometry elements defined\n\t'
                'and {} geometry properties populated. '
                'This is likely to cause problems.'.format(
                    self.geometry_count, len(self.properties.geometry_properties)))

    def remove_geometry_element(self, item):
        """
        Remove the geometry element at the given index

        Parameters
        ----------
        item : int|str
        """

        _, index = self.get_geometry_property_and_index(item)

        if self.geometry_count == 1:
            self.geometry = None
            self.properties = None
        elif self.geometry_count == 2:
            del self.geometry.collection[index]
            del self.properties.geometry_properties[index]
            self.geometry = self.geometry.collection[0]
        else:
            del self.geometry.collection[index]
            del self.properties.geometry_properties[index]


class AnnotationCollection(FeatureCollection):
    """
    An extension of the FeatureCollection class which has the features are
    AnnotationFeature instances.
    """

    @property
    def features(self):
        """
        The features list.

        Returns
        -------
        List[AnnotationFeature]
        """

        return self._features

    @features.setter
    def features(self, features):
        if features is None:
            self._features = None
            self._feature_dict = None
            return

        if not isinstance(features, list):
            raise TypeError('features must be a list of AnnotationFeatures. Got {}'.format(type(features)))

        for entry in features:
            self.add_feature(entry)

    def add_feature(self, feature):
        """
        Add an annotation.

        Parameters
        ----------
        feature : AnnotationFeature|Dict
        """

        if isinstance(feature, dict):
            feature = AnnotationFeature.from_dict(feature)
        if not isinstance(feature, AnnotationFeature):
            raise TypeError('This requires an AnnotationFeature instance, got {}'.format(type(feature)))

        if self._features is None:
            self._feature_dict = {feature.uid: 0}
            self._features = [feature, ]
        else:
            self._feature_dict[feature.uid] = len(self._features)
            self._features.append(feature)

    def __getitem__(self, item):
        # type: (Any) -> Union[AnnotationFeature, List[AnnotationFeature]]
        if self._features is None:
            raise StopIteration

        if isinstance(item, str):
            index = self._feature_dict[item]
            return self._features[index]
        return self._features[item]


class FileAnnotationCollection(Jsonable):
    """
    An collection of annotation elements associated with a given single image element file.
    """

    __slots__ = (
         '_version', '_image_file_name', '_image_id', '_core_name', '_annotations')
    _type = 'FileAnnotationCollection'

    def __init__(self, version=None, annotations=None, image_file_name=None, image_id=None, core_name=None):
        if version is None:
            version = _BASE_VERSION
        self._version = version
        self._annotations = None

        if image_file_name is None:
            self._image_file_name = None
        elif isinstance(image_file_name, str):
            self._image_file_name = os.path.split(image_file_name)[1]
        else:
            raise TypeError('image_file_name must be a None or a string')

        self._image_id = image_id
        self._core_name = core_name

        if self._image_file_name is None and self._image_id is None and self._core_name is None:
            logger.error('One of image_file_name, image_id, or core_name should be defined.')

        self.annotations = annotations

    @property
    def version(self):
        """
        str: The version
        """

        return self._version

    @property
    def image_file_name(self):
        """
        The image file name, if appropriate.

        Returns
        -------
        None|str
        """

        return self._image_file_name

    @property
    def image_id(self):
        """
        The image id, if appropriate.

        Returns
        -------
        None|str
        """

        return self._image_id

    @property
    def core_name(self):
        """
        The image core name, if appropriate.

        Returns
        -------
        None|str
        """

        return self._core_name

    @property
    def annotations(self):
        """
        The annotations.

        Returns
        -------
        AnnotationCollection
        """

        return self._annotations

    @annotations.setter
    def annotations(self, annotations):
        # type: (Union[None, AnnotationCollection, Dict]) -> None
        if annotations is None:
            self._annotations = None
            return

        if isinstance(annotations, AnnotationCollection):
            self._annotations = annotations
        elif isinstance(annotations, dict):
            self._annotations = AnnotationCollection.from_dict(annotations)
        else:
            raise TypeError(
                'annotations must be an AnnotationCollection. Got type {}'.format(type(annotations)))

    def add_annotation(self, annotation):
        """
        Add an annotation.

        Parameters
        ----------
        annotation : AnnotationFeature
            The prospective annotation.
        """

        if isinstance(annotation, dict):
            annotation = AnnotationFeature.from_dict(annotation)
        if not isinstance(annotation, AnnotationFeature):
            raise TypeError('This requires an AnnotationFeature instance. Got {}'.format(type(annotation)))

        if self._annotations is None:
            self._annotations = AnnotationCollection()

        self._annotations.add_feature(annotation)

    def delete_annotation(self, annotation_id):
        """
        Deletes the annotation associated with the given id.

        Parameters
        ----------
        annotation_id : str
        """

        del self._annotations[annotation_id]

    @classmethod
    def from_file(cls, file_name):
        """
        Read from (json) file.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        FileAnnotationCollection
        """

        with open(file_name, 'r') as fi:
            the_dict = json.load(fi)
        return cls.from_dict(the_dict)

    @classmethod
    def from_dict(cls, the_dict):
        """
        Define from a dictionary representation.

        Parameters
        ----------
        the_dict : dict

        Returns
        -------
        FileAnnotationCollection
        """

        if not isinstance(the_dict, dict):
            raise TypeError('This requires a dict. Got type {}'.format(type(the_dict)))

        typ = the_dict.get('type', 'NONE')
        if typ != cls._type:
            raise ValueError('FileAnnotationCollection cannot be constructed from the input dictionary')

        return cls(
            version=the_dict.get('version', 'UNKNOWN'),
            annotations=the_dict.get('annotations', None),
            image_file_name=the_dict.get('image_file_name', None),
            image_id=the_dict.get('image_id', None),
            core_name=the_dict.get('core_name', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['version'] = self.version
        if self.image_file_name is not None:
            parent_dict['image_file_name'] = self.image_file_name
        if self.image_id is not None:
            parent_dict['image_id'] = self.image_id
        if self.core_name is not None:
            parent_dict['core_name'] = self.core_name
        if self.annotations is not None:
            parent_dict['annotations'] = self.annotations.to_dict()
        return parent_dict

    def to_file(self, file_name):
        with open(file_name, 'w') as fi:
            json.dump(self.to_dict(), fi, indent=1)
