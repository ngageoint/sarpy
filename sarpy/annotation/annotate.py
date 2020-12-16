# -*- coding: utf-8 -*-
"""
This module provides utilities for annotating a given sicd file.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import getpass
import time
from collections import OrderedDict
import os
import logging
import json
from typing import Union, List, Any

# noinspection PyProtectedMember
from sarpy.geometry.geometry_elements import _Jsonable, FeatureCollection, Feature
from sarpy.annotation.schema_processing import LabelSchema


class AnnotationMetadata(_Jsonable):
    """
    Basic annotation metadata building block - everything but the geometry object
    """

    __slots__ = ('label_id', 'user_id', 'comment', 'confidence', 'timestamp')
    _type = 'AnnotationMetadata'

    def __init__(self, label_id=None, user_id=None, comment=None, confidence=None, timestamp=None):
        """

        Parameters
        ----------
        label_id : None|str
            The label id
        user_id : None|str
            The user id - will default to current user name
        comment : None|str
        confidence : None|str|int
            The confidence value
        timestamp : None|float|int
            The POSIX timestamp (in seconds) - should be construction time.
        """

        self.label_id = label_id  # type: Union[None, str]
        if user_id is None:
            user_id = getpass.getuser()
        self.user_id = user_id  # type: str
        self.comment = comment  # type: Union[None, str]
        self.confidence = confidence  # type: Union[None, str, int]
        if timestamp is None:
            timestamp = time.time()
        if not isinstance(timestamp, float):
            timestamp = float(timestamp)
        self.timestamp = timestamp  # type: float

    @classmethod
    def from_dict(cls, the_json):
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('AnnotationMetadata cannot be constructed from {}'.format(the_json))
        return cls(
            label_id=the_json.get('label_id', None),
            user_id=the_json.get('user_id', None),
            comment=the_json.get('comment', None),
            confidence=the_json.get('confidence', None),
            timestamp=the_json.get('timestamp', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        for attr in self.__slots__:
            parent_dict[attr] = getattr(self, attr)
        return parent_dict


class AnnotationMetadataList(_Jsonable):
    """
    The collection of AnnotationMetadata elements.
    """

    __slots__ = ('_elements', )
    _type = 'AnnotationMetadataList'

    def __init__(self, elements=None):
        """

        Parameters
        ----------
        elements : None|List[AnnotationMetadata|dict]
        """

        self._elements = None
        if elements is not None:
            self.elements = elements

    def __len__(self):
        if self._elements is None:
            return 0
        return len(self._elements)

    def __getitem__(self, item):
        # type: (Any) -> AnnotationMetadata
        return self._elements[item]

    @property
    def elements(self):
        """
        The AnnotationMetadata elements.

        Returns
        -------
        None|List[AnnotationMetadata]
        """

        return self._elements

    @elements.setter
    def elements(self, elements):
        if elements is None:
            self._elements = None
        if not isinstance(elements, list):
            raise TypeError('elements must be a list of AnnotationMetadata elements')
        self._elements = []
        for element in elements:
            if isinstance(element, AnnotationMetadata):
                self._elements.append(element)
            elif isinstance(element, dict):
                self._elements.append(AnnotationMetadata.from_dict(element))
            else:
                raise TypeError(
                    'Entries of elements must be an AnnotationMetadata, or dict '
                    'which deserialized to an AnnotationMetadata. Got type {}'.format(type(element)))
        self.sort_elements_by_timestamp()

    def sort_elements_by_timestamp(self):
        """
        Ensure that the elements are in decreasing order of timestamp.

        Returns
        -------
        None
        """

        if self._elements is None:
            return

        self._elements = sorted(self._elements, key=lambda x: x.timestamp, reverse=True)

    def insert_new_element(self, element):
        """
        Inserts an element at the head of the elements list.

        Parameters
        ----------
        element : AnnotationMetadata

        Returns
        -------
        None
        """

        if not isinstance(element, AnnotationMetadata):
            raise TypeError('element must be an AnnotationMetadata instance')
        if self._elements is None:
            self._elements = [element, ]
        else:
            if element.timestamp < self._elements[0].timestamp:
                raise ValueError(
                    'Element with timestamp {} cannot be inserted in front of element '
                    'with timestamp {}.'.format(element.timestamp, self._elements[0].timestamp))
            self._elements.insert(0, element)

    @classmethod
    def from_dict(cls, the_json):  # type: (dict) -> AnnotationMetadataList
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('AnnotationMetadataList cannot be constructed from {}'.format(the_json))
        return cls(elements=the_json.get('elements', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        if self._elements is None:
            parent_dict['elements'] = None
        else:
            parent_dict['elements'] = [entry.to_dict() for entry in self._elements]
        return parent_dict


class Annotation(Feature):
    """
    A specific extension of the Feature class which has the properties attribute
    populated with AnnotationMetadataList instance.
    """

    @property
    def properties(self):
        """
        The properties.

        Returns
        -------
        None|AnnotationMetadataList
        """

        return self._properties

    @properties.setter
    def properties(self, properties):
        if properties is None:
            self._properties = None
        elif isinstance(properties, AnnotationMetadataList):
            self._properties = properties
        elif isinstance(properties, dict):
            self._properties = AnnotationMetadataList.from_dict(properties)
        else:
            raise TypeError('properties must be an AnnotationMetadataList')

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['id'] = self.uid
        parent_dict['geometry'] = self.geometry.to_dict()
        parent_dict['properties'] = self.properties.to_dict()
        return parent_dict

    def add_annotation_metadata(self, value):
        if self._properties is None:
            self._properties = AnnotationMetadataList()
        self._properties.insert_new_element(value)


class AnnotationList(FeatureCollection):
    """
    A specific extension of the FeatureCollection class which has the features are
    Annotation instances.
    """

    @property
    def features(self):
        """
        The features list.

        Returns
        -------
        List[Annotation]
        """

        return self._features

    @features.setter
    def features(self, features):
        if features is None:
            self._features = None
            return

        if not isinstance(features, list):
            raise TypeError('features must be a list of Annotations. Got {}'.format(type(features)))

        self._features = []
        for entry in features:
            if isinstance(entry, Annotation):
                self._features.append(entry)
            elif isinstance(entry, dict):
                self._features.append(Annotation.from_dict(entry))
            else:
                raise TypeError(
                    'Entries of features are required to be instances of Annotation or '
                    'dictionary to be deserialized. Got {}'.format(type(entry)))

    def __getitem__(self, item):
        # type: (Any) -> Union[Annotation, List[Annotation]]
        if isinstance(item, str):
            return self._feature_dict[item]
        return self._features[item]


class FileAnnotationCollection(object):
    """
    A collection of file annotation elements.
    """

    __slots__ = (
        '_label_schema', '_image_file_name', '_image_id', '_core_name',
        '_annotations')

    def __init__(self, label_schema, annotations=None, image_file_name=None, image_id=None, core_name=None):
        self._annotations = None

        if isinstance(label_schema, str):
            label_schema = LabelSchema.from_file(label_schema)
        elif isinstance(label_schema, dict):
            label_schema = LabelSchema.from_dict(label_schema)
        if not isinstance(label_schema, LabelSchema):
            raise TypeError('label_schema must be an instance of a LabelSchema.')
        self._label_schema = label_schema

        if image_file_name is None:
            self._image_file_name = None
        elif isinstance(image_file_name, str):
            self._image_file_name = os.path.split(image_file_name)[1]
        else:
            raise TypeError('image_file_name must be a None or a string')

        self._image_id = image_id
        self._core_name = core_name

        if self._image_file_name is None and self._image_id is None and self._core_name is None:
            logging.error('One of image_file_name, image_id, or core_name should be defined.')

        self.annotations = annotations

    @property
    def label_schema(self):
        """
        The label schema.

        Returns
        -------
        LabelSchema
        """
        return self._label_schema

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
        AnnotationList
        """

        return self._annotations

    @annotations.setter
    def annotations(self, annotations):
        # type: (Union[None, AnnotationList, dict]) -> None
        if annotations is None:
            self._annotations = None
            return

        if isinstance(annotations, AnnotationList):
            self._annotations = annotations
        elif isinstance(annotations, dict):
            self._annotations = AnnotationList.from_dict(annotations)
        else:
            raise TypeError(
                'annotations must be an AnnotationList. Got type {}'.format(type(annotations)))
        self.validate_annotations(strict=False)

    def add_annotation(self, annotation, validate_confidence=True, validate_geometry=True):
        """
        Add an annotation, with a check for valid values in confidence and geometry type.

        Parameters
        ----------
        annotation : Annotation
            The prospective annotation.
        validate_confidence : bool
            Should we check that all confidence values follow the schema?
        validate_geometry : bool
            Should we check that all geometries are of allowed type?

        Returns
        -------
        None
        """

        if not isinstance(annotation, Annotation):
            raise TypeError('This requires an Annotation instance. Got {}'.format(type(annotation)))

        if self._annotations is None:
            self._annotations = AnnotationList()

        valid = True
        if validate_confidence:
            valid &= self._valid_confidences(annotation)
        if validate_geometry:
            valid &= self._valid_geometry(annotation)
        if not valid:
            raise ValueError('Annotation does not follow the schema.')
        self._annotations.add_feature(annotation)

    def is_annotation_valid(self, annotation):
        """
        Is the given annotation valid according to the schema?

        Parameters
        ----------
        annotation : Annotation

        Returns
        -------
        bool
        """

        if not isinstance(annotation, Annotation):
            return False

        if self._label_schema is None:
            return True
        valid = self._valid_confidences(annotation)
        valid &= self._valid_geometry(annotation)
        return valid

    def _valid_confidences(self, annotation):
        if self._label_schema is None:
            return True

        if annotation.properties is None or annotation.properties.elements is None:
            return True

        valid = True
        for entry in annotation.properties.elements:
            if not self._label_schema.is_valid_confidence(entry.confidence):
                valid = False
                logging.error('Invalid confidence value {}'.format(entry.confidence))
        return valid

    def _valid_geometry(self, annotation):
        if self._label_schema is None:
            return True
        if not self._label_schema.is_valid_geometry(annotation.geometry):
            logging.error('Invalid geometry type {}'.format(type(annotation.geometry)))
            return False
        return True

    def validate_annotations(self, strict=True):
        if self._annotations is None:
            return True

        valid = True
        for entry in self._annotations:
            valid &= self.is_annotation_valid(entry)
        if strict and not valid:
            raise ValueError('Some annotation does not follow the schema.')
        return valid

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
        if 'label_schema' not in the_dict:
            raise KeyError('this dictionary must contain a label_schema')
        return cls(
            the_dict['label_schema'],
            annotations=the_dict.get('annotations', None),
            image_file_name=the_dict.get('image_file_name', None),
            image_id=the_dict.get('image_id', None),
            core_name=the_dict.get('core_name', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['label_schema'] = self.label_schema.to_dict()
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
