"""
This module provides structures for performing data labelling on a background image
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import time
from collections import OrderedDict
import json
from typing import Union, List, Any, Dict
from datetime import datetime
import getpass

from sarpy.annotation.base import AnnotationProperties, FileAnnotationCollection, \
    AnnotationFeature, AnnotationCollection
from sarpy.geometry.geometry_elements import Jsonable, Geometry, Point, MultiPoint, \
    LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection

_LABEL_VERSION = "Label:1.0"
logger = logging.getLogger(__name__)
POSSIBLE_GEOMETRIES = ('point', 'line', 'polygon')


class LabelSchema(object):
    """
    The basic structure for an annotation/labelling schema.

    The label names may certainly be modified in place through use of the `labels`
    property, without worry for causing errors. This is discouraged, because having two
    schemas with same version number/ids and differing names can likely lead to confusion.

    Any modification of label ids of sub-id structure should be performed by using the
    :func:`set_labels_and_subtypes` method, or difficult to diagnose runtime errors
    will likely be introduced.
    """

    __slots__ = (
        '_version', '_labels', '_classification', '_version_date', '_subtypes',
        '_parent_types', '_confidence_values', '_permitted_geometries',
        '_integer_ids', '_maximum_id')

    def __init__(self, version='1.0', labels=None, version_date=None, classification="UNCLASSIFIED",
                 subtypes=None, confidence_values=None, permitted_geometries=None):
        """

        Parameters
        ----------
        version : None|str
            The version of the schema.
        labels : None|Dict[str, str]
            The {<label id> : <label name>} pair dictionary. Each entry must be a string,
            and '' is not a valid label id.
        version_date : None|str
            The date for this schema. If `None`, then the current time will be used.
        classification : str
            The classification for this schema.
        subtypes : None|Dict[str, List[str]]
            The {<label id> : <sub id list>} pairs. The root ids (i.e. those ids
            not belonging as sub-id to some other id) will be populated in the subtypes
            entry with empty string key (i.e. ''). Every key and entry of subtypes
            (excluding the subtypes root '') must correspond to an entry of labels,
            and no id can be a direct subtype of more than one id.
        confidence_values : None|List[Union[str, int]]
            The possible confidence values.
        permitted_geometries : None|List[str]
            The possible geometry types.
        """

        self._version_date = None
        self._labels = None
        self._subtypes = None
        self._parent_types = None
        self._confidence_values = None
        self._permitted_geometries = None
        self._integer_ids = True
        self._maximum_id = None  # type: Union[None, int]

        self._version = version
        self.update_version_date(value=version_date)
        self._classification = classification

        self.confidence_values = confidence_values
        self.permitted_geometries = permitted_geometries
        self.set_labels_and_subtypes(labels, subtypes)

    @property
    def version(self):
        """
        The version of the schema.

        Returns
        -------
        str
        """

        return self._version

    @property
    def version_date(self):
        """
        The date for this schema version - this should be a viable datetime format,
        but this is unenforced.

        Returns
        -------
        str
        """

        return self._version_date

    def update_version_date(self, value=None):
        if isinstance(value, str):
            self._version_date = value
        else:
            self._version_date = datetime.utcnow().isoformat('T')+'Z'

    @property
    def classification(self):
        """
        str: The classification for the contents of this schema.
        """

        return self._classification

    @property
    def suggested_next_id(self):
        """
        None|int: If all ids are integer type, this returns max_id+1. Otherwise, this
        yields None.
        """

        return None if self._maximum_id is None else self._maximum_id + 1

    @property
    def labels(self):
        """
        The complete label dictionary of the form `{label_id : label_name}`.

        Returns
        -------
        Dict[str, str]
        """

        return self._labels

    @property
    def subtypes(self):
        """
        The complete dictionary of subtypes of the form `{parent_id : <subids list>}`.

        Returns
        -------
        Dict[str, List[str]]
        """

        return self._subtypes

    @property
    def parent_types(self):
        """
        The dictionary of parent types of the form `{child_id : <set of parent ids>}`.
        It is canonically defined that an id is a parent of itself. The order of
        the `parent_ids` list is ascending order of parentage, i.e.
        `[<self>, <parent>, <parent of parent>, ...]`.

        Returns
        -------
        Dict[str, List[str]]
        """

        return self._parent_types

    @property
    def confidence_values(self):
        """
        The list of confidence values.

        Returns
        -------
        List
            Each element should be a json type (most likely use cases are str or int).
        """

        return self._confidence_values

    @confidence_values.setter
    def confidence_values(self, conf_values):
        if conf_values is None:
            self._confidence_values = None
            return

        if not isinstance(conf_values, list):
            raise TypeError('confidence_values must be a list. Got type {}'.format(type(conf_values)))
        self._confidence_values = conf_values

    @property
    def permitted_geometries(self):
        """
        The collection of permitted geometry types. None corresponds to all.
        Entries should be one of `{'point', 'line', 'polygon'}`.

        Returns
        -------
        None|List[str]
        """

        return self._permitted_geometries

    @permitted_geometries.setter
    def permitted_geometries(self, values):
        if values is None:
            self._permitted_geometries = None
            return

        if isinstance(values, str):
            values = [values.lower().strip(), ]
        else:
            values = [entry.lower().strip() for entry in values]

        if len(values) == 0:
            self._permitted_geometries = None
            return

        temp_values = []
        for entry in values:
            if entry in temp_values:
                continue
            if entry not in POSSIBLE_GEOMETRIES:
                raise ValueError('Got unknown geometry value `{}`'.format(entry))
            temp_values.append(entry)

        self._permitted_geometries = temp_values

    def get_id_from_name(self, the_name):
        """
        Determine the id from the given name. Get `None` if this fails.

        Parameters
        ----------
        the_name : str

        Returns
        -------
        None|str
        """

        prospective = None
        for key, value in self.labels.items():
            if value == the_name:
                prospective = key
                break
        return prospective

    def get_parent(self, the_id):
        """
        Get the parent id for the given element id. The empty string is returned
        for elements with no parent.

        Parameters
        ----------
        the_id : str

        Returns
        -------
        str
        """

        parents = self.parent_types[the_id]
        return parents[1] if len(parents) > 1 else ''

    def __str__(self):
        return json.dumps(self.to_dict(), indent=1)

    def __repr__(self):
        return json.dumps(self.to_dict())

    def _inspect_new_id_for_integer(self, the_id):
        if not self._integer_ids:
            return  # nothing to do
        if isinstance(the_id, str):
            # noinspection PyBroadException
            try:
                the_id = int(the_id)
            except Exception:
                self._integer_ids = False
                self._maximum_id = None

        if isinstance(the_id, int):
            # noinspection PyTypeChecker
            self._maximum_id = the_id if self._maximum_id is None else \
                max(self._maximum_id, the_id)
        else:
            self._integer_ids = False
            self._maximum_id = None

    def _inspect_ids_for_integer(self):
        for the_id in self._labels:
            self._inspect_new_id_for_integer(the_id)

    @staticmethod
    def _find_inverted_fork(subtypes, labels):
        """
        Look for parents claiming the same child. This assigns all unclaimed children
        to '' parent.

        Parameters
        ----------
        subtypes : dict
        labels : dict

        Returns
        -------
        dict
        """

        # we need to check the reference count for each key in labels
        counts = OrderedDict((key, 0) for key in labels)

        # ensure that every key of subtypes is a string and every value is a list,
        # also that inclusion makes sense
        for key, value in subtypes.items():
            if not isinstance(key, str):
                raise TypeError(
                    'All keys of subtypes must be of type string. Got key `{}` of '
                    'type {}.'.format(key, type(key)))
            if key != '' and key not in labels:
                raise KeyError(
                    'All keys of subtypes must belong to labels. Got key `{}` '
                    'which is missing from labels.'.format(key))

            if not isinstance(value, list):
                raise TypeError(
                    'All values of subtypes must be of type `list`. Got value {} '
                    'for key `{}` of type {}'.format(value, key, type(value)))
            for entry in value:
                if entry not in labels:
                    raise KeyError(
                        'All entries for each value of subtypes must belong to labels. '
                        'Got entry `{}` in key `{}` which is missing from labels.'.format(entry, key))
                counts[entry] += 1
        # create the root entry for subtypes
        if '' not in subtypes:
            subtypes[''] = []
        if isinstance(subtypes, OrderedDict):
            subtypes.move_to_end('', last=False)
        for key in counts:
            value = counts[key]
            if value > 1:
                raise ValueError('key {} is referenced in more than one subtype. This is invalid.'.format(key))
            if value == 0:
                subtypes[''].append(key)
        return subtypes

    @staticmethod
    def _find_cycle(subtypes):
        """
        Find any cycles in the data.

        Parameters
        ----------
        subtypes

        Returns
        -------
        None
        """

        found_cycles = []

        def iterate(current_id, find_id):
            for t_entry in subtypes.get(current_id, []):
                if t_entry == find_id:
                    found_cycles.append((find_id, current_id))
                iterate(t_entry, find_id)

        for the_id in subtypes['']:
            iterate(the_id, the_id)
        if len(found_cycles) > 0:
            for entry in found_cycles:
                logger.error(
                    'Cycle found with ids {} and {}'.format(entry[0], entry[1]))
            raise ValueError('cycles found in graph information')

    def set_labels_and_subtypes(self, labels, subtypes):
        """
        Set the labels and subtypes. **Note that subtypes may be modified in place.**

        Parameters
        ----------
        labels : None|dict
        subtypes : None|dict

        Returns
        -------
        None
        """

        if labels is None:
            labels = OrderedDict()
        if not isinstance(labels, dict):
            raise TypeError('labels is required to be a dict. Got type {}'.format(type(labels)))

        if subtypes is None:
            subtypes = OrderedDict()
        elif not isinstance(subtypes, dict):
            raise TypeError('subtypes is required to be None or a dict. Got type {}'.format(type(subtypes)))

        # ensure that every key and value of labels are strings
        for key in labels:
            if not isinstance(key, str):
                raise TypeError(
                    'All keys of labels must be of type string. Got key `{}` of '
                    'type {}'.format(key, type(key)))
            if key == '':
                raise ValueError('The empty string is not a valid label id.')
            value = labels[key]
            if not isinstance(value, str):
                raise TypeError(
                    'All values of labels must be of type string. Got value {} '
                    'for key `{}` of type {}'.format(value, key, type(value)))

        # look for inverted fork - multiple parents claiming the same child
        subtypes = self._find_inverted_fork(subtypes, labels)
        # look for cycles
        self._find_cycle(subtypes)

        # set the values
        self._labels = labels
        self._subtypes = subtypes
        self._construct_parent_types()
        self._inspect_ids_for_integer()

    def _construct_parent_types(self):
        def iterate(t_key, parents):
            entry = [t_key, ]
            # noinspection PyUnresolvedReferences
            entry.extend(parents)
            self._parent_types[t_key] = entry
            if t_key not in self._subtypes:
                return

            for child_key in self._subtypes[t_key]:
                iterate(child_key, entry)

        self._parent_types = {}
        for key in self._subtypes['']:
            iterate(key, [])

    def _validate_entry(self, the_id, the_name, the_parent):
        """
        Validate the basics for the given entry.

        Parameters
        ----------
        the_id : str
        the_name : str
        the_parent : str

        Returns
        -------
        (str, str, str)
        """

        # validate inputs
        if not (isinstance(the_id, str) and isinstance(the_name, str) and isinstance(the_parent, str)):
            raise TypeError(
                'the_id, the_name, and the_parent must all be string type, got '
                'types {}, {}, {}'.format(type(the_id), type(the_name), type(the_parent)))
        the_id = the_id.strip()
        the_name = the_name.strip()
        the_parent = the_parent.strip()

        # verify that values are permitted and sensible
        if the_id == '':
            raise ValueError('the_id value `` is reserved.')
        if the_name == '':
            raise ValueError('the_name value `` is not permitted.')
        if the_id == the_parent:
            raise ValueError('the_id cannot be the same as the_parent.')

        # try to determine parent from name if not a valid id
        if the_parent != '' and the_parent not in self.labels:
            prospective_parent = self.get_id_from_name(the_parent)
            if prospective_parent is None:
                raise ValueError('the_parent {} matches neither an existing id or name.'.format(the_parent))
            the_parent = prospective_parent

        return the_id, the_name, the_parent

    def add_entry(self, the_id, the_name, the_parent=''):
        """
        Adds a new entry. Note that leading or trailing blanks will be trimmed
        from all input values.

        Parameters
        ----------
        the_id : str
            The id for the label.
        the_name : str
            The name for the label.
        the_parent : str
            The parent id, where blank denotes no parent.

        Returns
        -------
        None
        """

        # validate inputs
        the_id, the_name, the_parent = self._validate_entry(the_id, the_name, the_parent)

        # verify that the_id doesn't already exist
        if the_id in self.labels:
            raise KeyError('the_id = {} already exists'.format(the_id))

        # check if name is already being used, and warn if so
        for key, value in self.labels.items():
            if value == the_name:
                logger.warning(
                    'Note that id {} is already using name {}. Having repeated names is '
                    'permitted, but may lead to confusion.'.format(key, value))

        # add the entry into the labels and subtypes dicts and reset the values
        # perform copy in case of failure
        labels = self.labels.copy()
        subtypes = self.subtypes.copy()
        labels[the_id] = the_name
        if the_parent in subtypes:
            subtypes[the_parent].append(the_id)
        else:
            subtypes[the_parent] = [the_id, ]

        try:
            self.set_labels_and_subtypes(labels, subtypes)
        except (ValueError, KeyError) as e:
            logger.error(
                'Setting new entry id {}, name {}, and parent {} failed with '
                'exception {}'.format(the_id, the_name, the_parent, e))

    def change_entry(self, the_id, the_name, the_parent):
        """
        Modify the values for a schema element.

        Parameters
        ----------
        the_id : str
        the_name : str
        the_parent : str

        Returns
        -------
        bool
            True if anything was actually changed. False otherwise.
        """

        # validate inputs
        the_id, the_name, the_parent = self._validate_entry(the_id, the_name, the_parent)

        # verify that the_id does already exist
        if the_id not in self.labels:
            raise KeyError('the_id = {} does not exist'.format(the_id))

        # check current values
        current_name = self.labels[the_id]
        current_parents = self.parent_types[the_id]
        current_parent = current_parents[1] if len(current_parents) > 1 else ''

        if current_name == the_name and current_parent == the_parent:
            # nothing is changing
            return False

        if current_name != the_name:
            # check if name is already being used by a different element, and warn if so
            if current_name != the_name:
                for key, value in self.labels.items():
                    if value == the_name and key != the_id:
                        logger.warning(
                            'Note that id {} is already using name {}. Having repeated names is '
                            'permitted, but may lead to confusion.'.format(key, value))

        if current_parent != the_parent:
            labels = self.labels.copy()
            labels[the_id] = the_name
            subtypes = self.subtypes.copy()
            # remove the_id from it's current subtype
            subtypes[current_parent].remove(the_id)
            # add it to the new one
            if the_parent in subtypes:
                subtypes[the_parent].append(the_id)
            else:
                subtypes[the_parent] = [the_id, ]
            try:
                self.set_labels_and_subtypes(labels, subtypes)
            except (ValueError, KeyError) as e:
                logger.error(
                    'Modifying entry id {}, name {}, and parent {} failed with '
                    'exception {}.'.format(the_id, the_name, the_parent, e))
                raise e
        else:
            # just changing the name
            self.labels[the_id] = the_name
        return True

    def delete_entry(self, the_id, recursive=False):
        """
        Deletes the entry from the schema.

        If the given element has children and `recursive=False`, a ValueError
        will be raised. If the given element has children and `recursive=True`,
        then all children will be deleted.

        Parameters
        ----------
        the_id : str
        recursive : bool
        """

        if the_id in self._subtypes:
            # handle all the children
            children = self.subtypes[the_id]
            if children is not None and len(children) > 0:
                if not recursive:
                    raise ValueError(
                        'LabelSchema entry for id {} has children. Either move children to a '
                        'different parent, or make recursive=True to delete all children.'.format(the_id))
                the_children = children.copy()  # unsafe to loop over a changing list
                for entry in the_children:
                    self.delete_entry(entry, recursive=True)
            # now, all the children have been deleted.
            del self._subtypes[the_id]
        # remove the entry from the parent's subtypes list
        parent_id = self.get_parent(the_id)
        self.subtypes[parent_id].remove(parent_id)
        # remove entry from labels
        del self._labels[the_id]
        del self._parent_types[the_id]

    def reorder_child_element(self, the_id, spaces=1):
        """
        Move the one space (forward or backward) in the list of children for the
        current parent. This is explicitly changes no actual parent/child
        relationships, and only changes the child list ORDERING.

        Parameters
        ----------
        the_id : str
        spaces : int
            How many spaces to shift the entry.

        Returns
        -------
        bool
            True of something actually changed, False otherwise.
        """

        if the_id not in self._labels:
            raise KeyError('No id {}'.format(the_id))
        parent_id = self.get_parent(the_id)
        children = self.subtypes[parent_id]
        # get the current location
        current_index = children.index(the_id)
        # determine the feasible new location
        if spaces < 0:
            new_index = max(0, current_index + spaces)
        else:
            new_index = min(len(children) - 1, current_index + spaces)
        if current_index == new_index:
            return False  # nothing to be done

        # pop our entry out of its current location
        children.pop(current_index)
        # insert it in its new location
        children.insert(new_index, the_id)
        return True

    @classmethod
    def from_file(cls, file_name):
        """
        Read schema from a file.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        LabelSchema
        """

        with open(file_name, 'r') as fi:
            input_dict = json.load(fi)
        return cls.from_dict(input_dict)

    @classmethod
    def from_dict(cls, input_dict):
        """
        Construct from a dictionary.

        Parameters
        ----------
        input_dict : dict

        Returns
        -------
        LabelSchema
        """

        version = input_dict['version']
        labels = input_dict['labels']
        version_date = input_dict.get('version_date', None)
        classification = input_dict.get('classification', 'UNCLASSIFIED')
        subtypes = input_dict.get('subtypes', None)
        conf_values = input_dict.get('confidence_values', None)
        perm_geometries = input_dict.get('permitted_geometries', None)
        return cls(
            version, labels, version_date=version_date, classification=classification,
            subtypes=subtypes, confidence_values=conf_values, permitted_geometries=perm_geometries)

    def to_dict(self):
        """
        Serialize to a dictionary representation.

        Returns
        -------
        dict
        """

        out = OrderedDict()
        out['version'] = self.version
        out['version_date'] = self.version_date
        out['classification'] = self.classification
        if self.confidence_values is not None:
            out['confidence_values'] = self.confidence_values
        if self.permitted_geometries is not None:
            out['permitted_geometries'] = self.permitted_geometries
        out['labels'] = self._labels
        out['subtypes'] = self._subtypes
        return out

    def to_file(self, file_name):
        """
        Write to a (json) file.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        None
        """

        with open(file_name, 'w') as fi:
            json.dump(self.to_dict(), fi, indent=1)

    def is_valid_confidence(self, value):
        """
        Is the given value a valid confidence (i.e. is in `confidence_values`)?
        Note that `None` is always considered valid here.

        Parameters
        ----------
        value

        Returns
        -------
        bool
        """

        if self._confidence_values is None or value is None:
            return True
        else:
            return value in self._confidence_values

    def is_valid_geometry(self, value):
        """
        Is the given geometry type allowed (i.e. is in `permitted_geometries`)?
        Note that `None` is always considered valid here.

        Parameters
        ----------
        value : str|Geometry

        Returns
        -------
        bool
        """

        def check_geom(geom):
            if isinstance(geom, (Point, MultiPoint)):
                out = 'point' in self._permitted_geometries
                if not out:
                    logger.error('Not allowed point type geometry components')
                return out
            elif isinstance(geom, (LineString, MultiLineString)):
                out = 'line' in self._permitted_geometries
                if not out:
                    logger.error('Not allowed line type geometry components')
                return out
            elif isinstance(geom, (Polygon, MultiPolygon)):
                out = 'polygon' in self._permitted_geometries
                if not out:
                    logger.error('Not allowed polygon type geometry components')
                return out
            elif isinstance(geom, GeometryCollection):
                out = True
                for entry in geom.geometries:
                    out &= check_geom(entry)
                return out
            else:
                raise TypeError('Got unexpected geometry type `{}`'.format(type(geom)))

        if self._permitted_geometries is None or value is None:
            return True

        if isinstance(value, str):
            return value.lower().strip() in self._permitted_geometries
        if not isinstance(value, Geometry):
            raise TypeError('Got unexpected geometry type `{}`'.format(type(value)))
        return check_geom(value)


##########
# elements for labeling a feature

class LabelMetadata(Jsonable):
    """
    Basic annotation metadata building block - everything but the geometry object
    """

    __slots__ = ('label_id', 'user_id', 'comment', 'confidence', 'timestamp')
    _type = 'LabelMetadata'

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
            raise ValueError('LabelMetadata cannot be constructed from {}'.format(the_json))
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

    def replicate(self):
        kwargs = {}
        for attr in self.__slots__:
            if attr not in ['user_id', 'timestamp']:
                kwargs[attr] = getattr(self, attr)
        the_type = self.__class__
        return the_type(**kwargs)


class LabelMetadataList(Jsonable):
    """
    The collection of LabelMetadata elements.
    """

    __slots__ = ('_elements', )
    _type = 'LabelMetadataList'

    def __init__(self, elements=None):
        """

        Parameters
        ----------
        elements : None|List[LabelMetadata|dict]
        """

        self._elements = None
        if elements is not None:
            self.elements = elements

    def __len__(self):
        if self._elements is None:
            return 0
        return len(self._elements)

    def __getitem__(self, item):
        # type: (Any) -> LabelMetadata
        if self._elements is None:
            raise StopIteration
        return self._elements[item]

    @property
    def elements(self):
        """
        The LabelMetadata elements.

        Returns
        -------
        None|List[LabelMetadata]
        """

        return self._elements

    @elements.setter
    def elements(self, elements):
        if elements is None:
            self._elements = None
        if not isinstance(elements, list):
            raise TypeError('elements must be a list of LabelMetadata elements')
        self._elements = []
        for element in elements:
            self.insert_new_element(element)

    def insert_new_element(self, element):
        """
        Inserts an element at the head of the elements list.

        Parameters
        ----------
        element : LabelMetadata

        Returns
        -------
        None
        """

        if isinstance(element, dict):
            element = LabelMetadata.from_dict(element)
        if not isinstance(element, LabelMetadata):
            raise TypeError('element must be an LabelMetadata instance, got type {}'.format(type(element)))

        if self._elements is None:
            self._elements = [element, ]
        elif len(self._elements) == 0:
            self._elements.append(element)
        else:
            if element.timestamp < self._elements[0].timestamp:
                raise ValueError(
                    'Element with timestamp {} cannot be inserted in front of element '
                    'with timestamp {}.'.format(element.timestamp, self._elements[0].timestamp))
            self._elements.insert(0, element)

    @classmethod
    def from_dict(cls, the_json):  # type: (dict) -> LabelMetadataList
        typ = the_json['type']
        if typ != cls._type:
            raise ValueError('LabelMetadataList cannot be constructed from {}'.format(the_json))
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

    def replicate(self):
        kwargs = {}
        elements = self.elements
        if elements is not None:
            kwargs['elements'] = [elements[0].replicate()]
        the_type = self.__class__
        return the_type(**kwargs)

    def get_label_id(self):
        """
        Gets the current label id.

        Returns
        -------
        None|str
        """

        return None if (self.elements is None or len(self.elements) == 0) else self.elements[0].label_id


class LabelProperties(AnnotationProperties):
    _type = 'LabelProperties'

    @property
    def parameters(self):
        """
        LabelMetadataList: The parameters
        """

        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if value is None:
            self._parameters = LabelMetadataList()
            return
        if isinstance(value, dict):
            self._parameters = LabelMetadataList.from_dict(value)
            return
        if isinstance(value, LabelMetadataList):
            self._parameters = value
            return
        raise TypeError('Got unexpected type for parameters `{}`'.format(type(value)))

    def get_label_id(self):
        """
        Gets the current label id.

        Returns
        -------
        None|str
        """

        return None if self.parameters is None else self.parameters.get_label_id()


############
# the feature extensions

class LabelFeature(AnnotationFeature):
    """
    A specific extension of the Feature class which has the properties attribute
    populated with LabelProperties instance.
    """

    @property
    def properties(self):
        """
        The properties.

        Returns
        -------
        None|LabelProperties
        """

        return self._properties

    @properties.setter
    def properties(self, properties):
        if properties is None:
            self._properties = LabelProperties()
            return
        if isinstance(properties, dict):
            self._properties = LabelProperties.from_dict(properties)
            return
        if isinstance(properties, LabelProperties):
            self._properties = properties
            return
        raise TypeError('properties must be an LabelProperties')

    def add_annotation_metadata(self, value):
        """
        Adds the new label to the series of labeling efforts.

        Parameters
        ----------
        value : LabelMetadata
        """

        if self._properties is None:
            self._properties = LabelProperties()
        self._properties.parameters.insert_new_element(value)

    def get_label_id(self):
        """
        Gets the label id.

        Returns
        -------
        None|str
        """

        return None if self.properties is None else self.properties.get_label_id()


class LabelCollection(AnnotationCollection):
    """
    A specific extension of the FeatureCollection class which has the features are
    LabelFeature instances.
    """

    @property
    def features(self):
        """
        The features list.

        Returns
        -------
        List[LabelFeature]
        """

        return self._features

    @features.setter
    def features(self, features):
        if features is None:
            self._features = None
            self._feature_dict = None
            return

        if not isinstance(features, list):
            raise TypeError(
                'features must be a list of LabelFeatures. '
                'Got {}'.format(type(features)))

        for entry in features:
            self.add_feature(entry)

    def add_feature(self, feature):
        """
        Add an annotation.

        Parameters
        ----------
        feature : LabelFeature
        """

        if isinstance(feature, dict):
            feature = LabelFeature.from_dict(feature)
        if not isinstance(feature, LabelFeature):
            raise TypeError('This requires an LabelFeature instance, got {}'.format(type(feature)))

        if self._features is None:
            self._feature_dict = {feature.uid: 0}
            self._features = [feature, ]
        else:
            self._feature_dict[feature.uid] = len(self._features)
            self._features.append(feature)

    def __getitem__(self, item):
        # type: (Any) -> Union[LabelFeature, List[LabelFeature]]
        if self._features is None:
            raise StopIteration

        if isinstance(item, str):
            index = self._feature_dict[item]
            return self._features[index]
        return self._features[item]


###########
# serialized file object

class FileLabelCollection(FileAnnotationCollection):
    """
    An collection of annotation elements associated with a given single image element file.
    """

    __slots__ = (
        '_version', '_label_schema', '_image_file_name', '_image_id', '_core_name', '_annotations')
    _type = 'FileLabelCollection'

    def __init__(self, label_schema, version=None, annotations=None,
                 image_file_name=None, image_id=None, core_name=None):
        if version is None:
            version = _LABEL_VERSION

        if isinstance(label_schema, str):
            label_schema = LabelSchema.from_file(label_schema)
        elif isinstance(label_schema, dict):
            label_schema = LabelSchema.from_dict(label_schema)
        if not isinstance(label_schema, LabelSchema):
            raise TypeError('label_schema must be an instance of a LabelSchema.')
        self._label_schema = label_schema

        FileAnnotationCollection.__init__(
            self, version=version, annotations=annotations, image_file_name=image_file_name,
            image_id=image_id, core_name=core_name)

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
        # type: (Union[None, LabelCollection, dict]) -> None
        if annotations is None:
            self._annotations = None
            return

        if isinstance(annotations, LabelCollection):
            self._annotations = annotations
        elif isinstance(annotations, dict):
            self._annotations = LabelCollection.from_dict(annotations)
        else:
            raise TypeError(
                'annotations must be an LabelCollection. Got type {}'.format(type(annotations)))
        self.validate_annotations(strict=False)

    def add_annotation(self, annotation, validate_confidence=True, validate_geometry=True):
        """
        Add an annotation, with a check for valid values in confidence and geometry type.

        Parameters
        ----------
        annotation : LabelFeature
            The prospective annotation.
        validate_confidence : bool
            Should we check that all confidence values follow the schema?
        validate_geometry : bool
            Should we check that all geometries are of allowed type?

        Returns
        -------
        None
        """

        if not isinstance(annotation, LabelFeature):
            raise TypeError('This requires an LabelFeature instance. Got {}'.format(type(annotation)))

        if self._annotations is None:
            self._annotations = LabelCollection()

        valid = True
        if validate_confidence:
            valid &= self._valid_confidences(annotation)
        if validate_geometry:
            valid &= self._valid_geometry(annotation)
        if not valid:
            raise ValueError('LabelFeature does not follow the schema.')
        self._annotations.add_feature(annotation)

    def is_annotation_valid(self, annotation):
        """
        Is the given annotation valid according to the schema?

        Parameters
        ----------
        annotation : LabelFeature

        Returns
        -------
        bool
        """

        if not isinstance(annotation, LabelFeature):
            return False

        if self._label_schema is None:
            return True
        valid = self._valid_confidences(annotation)
        valid &= self._valid_geometry(annotation)
        return valid

    def _valid_confidences(self, annotation):
        if self._label_schema is None:
            return True

        if annotation.properties is None or annotation.properties.parameters is None:
            return True

        valid = True
        for entry in annotation.properties.parameters:
            if not self._label_schema.is_valid_confidence(entry.confidence):
                valid = False
                logger.error('Invalid confidence value {}'.format(entry.confidence))
        return valid

    def _valid_geometry(self, annotation):
        if self._label_schema is None:
            return True
        if not self._label_schema.is_valid_geometry(annotation.geometry):
            logger.error('Invalid geometry type {}'.format(type(annotation.geometry)))
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
        FileLabelCollection
        """

        if not isinstance(the_dict, dict):
            raise TypeError('This requires a dict. Got type {}'.format(type(the_dict)))
        if 'label_schema' not in the_dict:
            raise KeyError('this dictionary must contain a label_schema')

        typ = the_dict.get('type', 'NONE')
        if typ != cls._type:
            raise ValueError('FileLabelCollection cannot be constructed from the input dictionary')

        return cls(
            the_dict['label_schema'],
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
