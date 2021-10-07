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
        if self.name is not None:
            parent_dict['name'] = self.name
        if self.color is not None:
            parent_dict['color'] = self.color


class AnnotationProperties(Jsonable):
    """
    The basic common properties for an annotation
    """

    __slots__ = ('_name', '_description', '_applicable_indices', '_directory', '_geometry_properties', '_parameters')
    _type = 'AnnotationProperties'

    def __init__(self, name=None, description=None, applicable_indices=None,
                 directory=None, geometry_properties=None, parameters=None):
        """

        Parameters
        ----------
        name : Optional[str]
        description : Optional[str]
        applicable_indices : Optional[List[int]]
        directory : Optional[str]
        geometry_properties : Optional[List[GeometryProperties]]
        parameters : Optional[Jsonable]
        """

        self._name = None
        self._description = None
        self._directory = None
        self._applicable_indices = None
        self._geometry_properties = []
        self._parameters = None

        self.name = name
        self.description = description
        self.applicable_indices = applicable_indices
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
        raise TypeError('Got unexpected type for description')

    @property
    def applicable_indices(self):
        """
        None|List[int]: The list of all image indices to which this annotation applies.
        `None` is equivalent to all.
        """

        return self._applicable_indices

    @applicable_indices.setter
    def applicable_indices(self, value):
        if value is None:
            self._applicable_indices = None
            return

        self._applicable_indices = [int(entry) for entry in value]

    def add_applicable_index(self, value):
        """
        Adds an index to the applicable list.

        Parameters
        ----------
        value : int
        """

        if value is None or self._applicable_indices is None:
            return
        value = int(value)
        if value not in self._applicable_indices:
            self._applicable_indices.append(value)
            self._applicable_indices = sorted(self._applicable_indices)

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
        self._description = '/'.join([entry for entry in parts if entry != ''])

    @property
    def geometry_properties(self):
        # type: () -> Optional[List[GeometryProperties]]
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
            self._properties = None
        elif isinstance(properties, AnnotationProperties):
            self._properties = properties
        elif isinstance(properties, dict):
            self._properties = AnnotationProperties.from_dict(properties)
        else:
            raise TypeError('Got an unexpected type for properties attribute of class {}'.format(self.__class__))

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

    def get_geometry_element(self, index):
        """
        Gets the basic geometry object at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        Point|Line|Polygon
        """

        if self.geometry is None:
            raise ValueError('No geometry defined.')

        index = int(index)
        if not (0 <= index < self.geometry_count):
            raise KeyError('invalid geometry index')
        if self.geometry.is_collection:
            return self.geometry.collection[index]
        else:
            return self.geometry

    def _validate_geometry_element(self, geometry):
        if geometry is None:
            return geometry
        if not isinstance(geometry, Geometry):
            raise TypeError('geometry must be an instance of Geometry base class')

        if self._allowed_geometries is not None and \
                geometry not in self._allowed_geometries:
            raise TypeError('geometry is not of one of the allowed types')
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

    def remove_geometry_element(self, index):
        """
        Remove the geometry element at the given index

        Parameters
        ----------
        index : int
        """

        index = int(index)
        if not (0 <= index < self.geometry_count):
            raise KeyError('Got invalid index value')

        if self.geometry_count == 1:
            self._geometry = None
            self._properties = None
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
        if isinstance(item, str):
            index = self._feature_dict[item]
            return self._features[index]
        return self._features[item]


class FileAnnotationCollection(object):
    """
    An collection of annotation elements associated with a given single image element file.
    """

    __slots__ = (
         '_image_file_name', '_image_id', '_core_name', '_annotations')

    def __init__(self, annotations=None, image_file_name=None, image_id=None, core_name=None):
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
        annotation : LabelFeature
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
        return cls(
            annotations=the_dict.get('annotations', None),
            image_file_name=the_dict.get('image_file_name', None),
            image_id=the_dict.get('image_id', None),
            core_name=the_dict.get('core_name', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
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
