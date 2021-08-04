"""
This module provides structures for annotating a given SICD type file for RCS
calculations.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from collections import OrderedDict
import os
import json
from typing import Union, Any, List, Dict

# noinspection PyProtectedMember
from sarpy.geometry.geometry_elements import _Jsonable, FeatureCollection, Feature, Polygon, MultiPolygon
from sarpy.compliance import string_types, int_func, integer_types

logger = logging.getLogger(__name__)


class RCSStatistics(_Jsonable):
    __slots__ = ('name', 'mean', 'std', 'max', 'min')
    _type = 'RCSStatistics'

    def __init__(self, name=None, mean=None, std=None, max=None, min=None):
        """

        Parameters
        ----------
        name : None|str
        mean : None|float
        std : None|float
        max : None|float
        min : None|float
        """

        if mean is not None:
            mean = float(mean)
        if std is not None:
            std = float(std)
        if max is not None:
            max = float(max)
        if min is not None:
            min = float(min)

        self.name = name  # type: Union[None, str]
        self.mean = mean  # type: Union[None, float]
        self.std = std  # type: Union[None, float]
        self.max = max  # type: Union[None, float]
        self.min = min  # type: Union[None, float]

    @classmethod
    def from_dict(cls, the_json):
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('RCSStatistics cannot be constructed from {}'.format(the_json))
        return cls(
            name=the_json.get('name', None),
            mean=the_json.get('mean', None),
            std=the_json.get('std', None),
            max=the_json.get('max', None),
            min=the_json.get('min', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        for attr in self.__slots__:
            parent_dict[attr] = getattr(self, attr)
        return parent_dict


class RCSValue(_Jsonable):
    """
    The collection of RCSStatistics elements.
    """

    __slots__ = ('polarization', '_statistics', '_name_to_index')
    _type = 'RCSValue'

    def __init__(self, polarization=None, statistics=None):
        """

        Parameters
        ----------
        polarization : None|str
        statistics : None|List[RCSStatistics|dict]
        """

        self._statistics = None  # type: Union[None, List[RCSStatistics]]
        self._name_to_index = None  # type: Union[None, Dict[str, int]]

        self.polarization = polarization  # type: Union[None, str]
        if statistics is not None:
            self.statistics = statistics

    def __len__(self):
        if self._statistics is None:
            return 0
        return len(self._statistics)

    def __getitem__(self, item):
        # type: (Union[int, str]) -> Union[None, RCSStatistics]
        if isinstance(item, string_types):
            return self._statistics[self._name_to_index[item]]
        return self._statistics[item]

    @property
    def statistics(self):
        """
        The RCSStatistics elements.

        Returns
        -------
        None|List[RCSStatistics]
        """

        return self._statistics

    @statistics.setter
    def statistics(self, statistics):
        if statistics is None:
            self._statistics = None
        if not isinstance(statistics, list):
            raise TypeError('statistics must be a list of RCSStatistics elements')
        for element in statistics:
            self.insert_new_element(element)

    def insert_new_element(self, element):
        """
        Inserts an element at the end of the elements list.

        Parameters
        ----------
        element : RCSStatistics

        Returns
        -------
        None
        """

        if isinstance(element, dict):
            element = RCSStatistics.from_dict(element)
        if not isinstance(element, RCSStatistics):
            raise TypeError('element must be an RCSStatistics instance')
        if self._statistics is None:
            self._statistics = [element,]
            self._name_to_index = {element.name : 0}
        else:
            self._statistics.append(element)
            self._name_to_index[element.name] = len(self._statistics) - 1

    @classmethod
    def from_dict(cls, the_json):  # type: (dict) -> RCSValue
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('RCSValue cannot be constructed from {}'.format(the_json))
        return cls(polarization=the_json.get('polarization', None), statistics=the_json.get('statistics', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['polarization'] = self.polarization
        if self._statistics is None:
            parent_dict['statistics'] = None
        else:
            parent_dict['statistics'] = [entry.to_dict() for entry in self._statistics]
        return parent_dict


class RCSValueCollection(_Jsonable):
    """
    The collection of RCSValue elements, one for each polarization. Also, the pixel
    count for the number of integer grid elements contained in the interior of the
    associated geometry interior.
    """

    __slots__ = ('_name', '_description', '_pixel_count', '_elements')
    _type = 'RCSValueCollection'

    def __init__(self, name=None, description=None, pixel_count=None, elements=None):
        """

        Parameters
        ----------
        name : None|str
        description : None|str
        pixel_count : None|int
        elements : None|List[RCSValue|dict]
        """

        self._name = None
        self._description = None
        self._pixel_count = None
        self._elements = None

        self.name = name
        self.description = description
        self.pixel_count = pixel_count
        self.elements = elements

    def __len__(self):
        if self._elements is None:
            return 0
        return len(self._elements)

    def __getitem__(self, item):
        # type: (Union[int, str]) -> Union[None, RCSValue]
        return self._elements[item]

    @property
    def name(self):
        # type: () -> Union[None, str]
        """
        None|str: The name of the associated feature.
        """

        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = None
            return

        if not isinstance(value, string_types):
            raise TypeError('name is required to be of string type.')
        self._name = value

    @property
    def description(self):
        # type: () -> Union[None, str]
        """
        None|str: The description of the associated feature.
        """

        return self._description

    @description.setter
    def description(self, value):
        if value is None:
            self._description = None
            return

        if not isinstance(value, string_types):
            raise TypeError('description is required to be of string type.')
        self._description = value

    @property
    def pixel_count(self):
        # type: () -> Union[None, int]
        """
        None|int: The number of integer pixel grid elements contained in the interior
        of the associated geometry element.
        """

        return self._pixel_count

    @pixel_count.setter
    def pixel_count(self, value):
        if value is None:
            self._pixel_count = None
            return
        if not isinstance(value, integer_types):
            value = int_func(value)
        self._pixel_count = value

    @property
    def elements(self):
        # type: () -> Union[None, List[RCSValue]]
        """
        None|List[RCSValue]: The RCSValue elements.
        """

        return self._elements

    @elements.setter
    def elements(self, elements):
        if elements is None:
            self._elements = None
            return

        if not isinstance(elements, list):
            raise TypeError('elements must be a list of RCSValue elements')
        for element in elements:
            self.insert_new_element(element)

    def insert_new_element(self, element):
        """
        Inserts an element at the end of the elements list.

        Parameters
        ----------
        element : RCSValue
        """

        if isinstance(element, dict):
            element = RCSValue.from_dict(element)
        if not isinstance(element, RCSValue):
            raise TypeError('element must be an RCSValue instance')
        if self._elements is None:
            self._elements = [element,]
        else:
            self._elements.append(element)

    @classmethod
    def from_dict(cls, the_json):  # type: (dict) -> RCSValueCollection
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('RCSValueCollection cannot be constructed from {}'.format(the_json))
        return cls(
            name=the_json.get('name', None), description=the_json.get('description', None),
            pixel_count=the_json.get('pixel_count', None), elements=the_json.get('elements', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        if self.name is not None:
            parent_dict['name'] = self.name
        if self.description is not None:
            parent_dict['description'] = self.description
        parent_dict['pixel_count'] = self.pixel_count
        if self._elements is None:
            parent_dict['elements'] = None
        else:
            parent_dict['elements'] = [entry.to_dict() for entry in self._elements]
        return parent_dict


class RCSFeature(Feature):
    """
    A specific extension of the Feature class which has the properties attribute
    populated with RCSValueCollection instance.
    """

    @property
    def properties(self):
        """
        The properties.

        Returns
        -------
        None|RCSValueCollection
        """

        return self._properties

    @properties.setter
    def properties(self, properties):
        if properties is None:
            self._properties = None
        elif isinstance(properties, RCSValueCollection):
            self._properties = properties
        elif isinstance(properties, dict):
            self._properties = RCSValueCollection.from_dict(properties)
        else:
            raise TypeError('properties must be an RCSValueCollection')

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['id'] = self.uid
        parent_dict['geometry'] = self.geometry.to_dict()
        if self.properties is not None:
            parent_dict['properties'] = self.properties.to_dict()
        return parent_dict


class RCSCollection(FeatureCollection):
    """
    A specific extension of the FeatureCollection class which has the features are
    RCSFeature instances.
    """

    @property
    def features(self):
        """
        The features list.

        Returns
        -------
        List[RCSFeature]
        """

        return self._features

    @features.setter
    def features(self, features):
        if features is None:
            self._features = None
            self._feature_dict = None
            return

        if not isinstance(features, list):
            raise TypeError('features must be a list of RCSFeatures. Got {}'.format(type(features)))

        for entry in features:
            self.add_feature(entry)

    def add_feature(self, feature):
        """
        Add an annotation.

        Parameters
        ----------
        feature : RCSFeature|dict
        """

        if isinstance(feature, dict):
            feature = RCSFeature.from_dict(feature)

        if not isinstance(feature, RCSFeature):
            raise TypeError('This requires an RCSFeature instance, got {}'.format(type(feature)))

        if not isinstance(feature.geometry, (Polygon, MultiPolygon)):
            raise TypeError('RCS annotations require that geometry is a Polygon or Multipolygon.')

        if self._features is None:
            self._feature_dict = {feature.uid: 0}
            self._features = [feature, ]
        else:
            self._feature_dict[feature.uid] = len(self._features)
            self._features.append(feature)

    def __getitem__(self, item):
        # type: (Any) -> Union[RCSFeature, List[RCSFeature]]
        if isinstance(item, string_types):
            index = self._feature_dict[item]
            return self._features[index]
        return self._features[item]


###########
# serialized file object

class FileRCSCollection(object):
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
        LabelCollection
        """

        return self._annotations

    @annotations.setter
    def annotations(self, annotations):
        # type: (Union[None, RCSCollection, dict]) -> None
        if annotations is None:
            self._annotations = None
            return

        if isinstance(annotations, RCSCollection):
            self._annotations = annotations
        elif isinstance(annotations, dict):
            self._annotations = RCSCollection.from_dict(annotations)
        else:
            raise TypeError(
                'annotations must be an RCSCollection. Got type {}'.format(type(annotations)))

    def add_annotation(self, annotation):
        """
        Add an annotation, with a check that the geometry type is a polygon or
        Multipolygon.

        Parameters
        ----------
        annotation : LabelFeature
            The prospective annotation.
        """

        if isinstance(annotation, dict):
            annotation = RCSFeature.from_dict(annotation)
        if not isinstance(annotation, RCSFeature):
            raise TypeError('This requires an RCSFeature instance. Got {}'.format(type(annotation)))

        if self._annotations is None:
            self._annotations = RCSCollection()

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
        FileLabelCollection
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
        FileRCSCollection
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
