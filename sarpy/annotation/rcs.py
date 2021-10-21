"""
This module provides structures for annotating a given SICD type file for RCS
calculations
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from collections import OrderedDict
import json
from typing import Union, Any, List, Optional

from sarpy.geometry.geometry_elements import Jsonable, Polygon, MultiPolygon
from sarpy.annotation.base import AnnotationFeature, AnnotationProperties, \
    AnnotationCollection, FileAnnotationCollection

_RCS_VERSION = "RCS:1.0"
logger = logging.getLogger(__name__)


class RCSStatistics(Jsonable):
    __slots__ = ('mean', 'std', 'max', 'min')
    _type = 'RCSStatistics'

    def __init__(self, mean=None, std=None, max=None, min=None):
        """

        Parameters
        ----------
        mean : None|float
            All values are assumed the be stored here in units of power
        std : None|float
            All values are assumed the be stored here in units of power
        max : None|float
            All values are assumed the be stored here in units of power
        min : None|float
            All values are assumed the be stored here in units of power
        """

        if mean is not None:
            mean = float(mean)
        if std is not None:
            std = float(std)
        if max is not None:
            max = float(max)
        if min is not None:
            min = float(min)

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


class RCSValue(Jsonable):
    """
    The collection of RCSStatistics elements.
    """

    __slots__ = ('polarization', 'units', '_value', '_noise')
    _type = 'RCSValue'

    def __init__(self, polarization, units, value=None, noise=None):
        """

        Parameters
        ----------
        polarization : str
        units: str
        value : None|RCSStatistics
        noise : None|RCSStatistics
        """
        self._value = None
        self._noise = None
        self.polarization = polarization
        self.units = units
        self.value = value
        self.noise = noise

    @property
    def value(self):
        """
        None|RCSStatistics: The value
        """

        return self._value

    @value.setter
    def value(self, val):
        if isinstance(val, dict):
            val = RCSStatistics.from_dict(val)
        if not (val is None or isinstance(val, RCSStatistics)):
            raise TypeError('Got incompatible input for value')
        self._value = val

    @property
    def noise(self):
        """
        None|RCSStatistics: The noise
        """

        return self._value

    @noise.setter
    def noise(self, val):
        if isinstance(val, dict):
            val = RCSStatistics.from_dict(val)
        if not (val is None or isinstance(val, RCSStatistics)):
            raise TypeError('Got incompatible input for noise')
        self._noise = val

    @classmethod
    def from_dict(cls, the_json):  # type: (dict) -> RCSValue
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('RCSValue cannot be constructed from {}'.format(the_json))
        return cls(
            the_json.get('polarization', None),
            the_json.get('units', None),
            value=the_json.get('value', None),
            noise=the_json.get('noise', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['polarization'] = self.polarization
        parent_dict['units'] = self.units
        if self.value is not None:
            parent_dict['value'] = self.value.to_dict()
        if self.noise is not None:
            parent_dict['noise'] = self.noise.to_dict()
        return parent_dict


class RCSValueCollection(Jsonable):
    """
    A specific type for the AnnotationProperties.parameters
    """

    __slots__ = ('_pixel_count', '_elements')
    _type = 'RCSValueCollection'

    def __init__(self, pixel_count=None, elements=None):
        """

        Parameters
        ----------
        pixel_count : None|int
        elements : None|List[RCSValue|dict]
        """

        self._pixel_count = None
        self._elements = None

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
        if not isinstance(value, int):
            value = int(value)
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
            self._elements = [element, ]
        else:
            self._elements.append(element)

    @classmethod
    def from_dict(cls, the_json):  # type: (dict) -> RCSValueCollection
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('RCSValueCollection cannot be constructed from {}'.format(the_json))
        return cls(
            pixel_count=the_json.get('pixel_count', None), elements=the_json.get('elements', None))

    def to_dict(self, parent_dict=None):
        if parent_dict is None:
            parent_dict = OrderedDict()
        parent_dict['type'] = self.type
        parent_dict['pixel_count'] = self.pixel_count
        if self._elements is None:
            parent_dict['elements'] = None
        else:
            parent_dict['elements'] = [entry.to_dict() for entry in self._elements]
        return parent_dict


class RCSProperties(AnnotationProperties):
    _type = 'RCSProperties'

    @property
    def parameters(self):
        """
        Optional[RCSValueCollection]: The parameters
        """

        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if isinstance(value, dict):
            value = RCSValueCollection.from_dict(value)
        if not (value is None or isinstance(value, RCSValueCollection)):
            raise TypeError('Got unexpected type for parameters')
        self._parameters = value


class RCSFeature(AnnotationFeature):
    """
    A specific extension of the Feature class which has the properties attribute
    populated with RCSValueCollection instance.
    """
    _allowed_geometries = (Polygon, MultiPolygon)

    @property
    def properties(self):
        # type: () -> Optional[RCSValueCollection]
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
            self._properties = RCSValueCollection()
        elif isinstance(properties, RCSValueCollection):
            self._properties = properties
        elif isinstance(properties, dict):
            self._properties = RCSValueCollection.from_dict(properties)
        else:
            raise TypeError('properties must be an RCSValueCollection')


class RCSCollection(AnnotationCollection):
    """
    A specific extension of the AnnotationCollection class which has that the
    features are RCSFeature instances.
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

        if self._features is None:
            self._feature_dict = {feature.uid: 0}
            self._features = [feature, ]
        else:
            self._feature_dict[feature.uid] = len(self._features)
            self._features.append(feature)

    def __getitem__(self, item):
        # type: (Any) -> Union[RCSFeature, List[RCSFeature]]
        if self._features is None:
            raise StopIteration

        if isinstance(item, str):
            index = self._feature_dict[item]
            return self._features[index]
        return self._features[item]


###########
# serialized file object

class FileRCSCollection(FileAnnotationCollection):
    """
    An collection of RCS statistics elements.
    """
    _type = 'FileRCSCollection'

    def __init__(self, version=None, annotations=None, image_file_name=None,
                 image_id=None, core_name=None):
        if version is None:
            version = _RCS_VERSION

        FileAnnotationCollection.__init__(
            self, version=version, annotations=annotations, image_file_name=image_file_name,
            image_id=image_id, core_name=core_name)

    @property
    def annotations(self):
        """
        The annotations.

        Returns
        -------
        RCSCollection
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
        Add an annotation.

        Parameters
        ----------
        annotation : RCSFeature
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
        FileRCSCollection
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

        typ = the_dict.get('type', 'NONE')
        if typ != cls._type:
            raise ValueError('FileRCSCollection cannot be constructed from the input dictionary')

        return cls(
            version=the_dict.get('version', 'UNKNOWN'),
            annotations=the_dict.get('annotations', None),
            image_file_name=the_dict.get('image_file_name', None),
            image_id=the_dict.get('image_id', None),
            core_name=the_dict.get('core_name', None))
