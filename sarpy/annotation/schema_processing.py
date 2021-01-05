# -*- coding: utf-8 -*-
"""
This module provides utilities for validating and preparing an annotation schema.
"""

import logging
from collections import OrderedDict
import json
from typing import Dict, List
from datetime import datetime

from sarpy.compliance import string_types


__classification__ = "UNCLASSIFIED"


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
        '_parent_types', '_confidence_values', '_permitted_geometries')

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
        if isinstance(value, string_types):
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
        if not isinstance(values, list):
            values = list(values)
        self._permitted_geometries = values

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
            for entry in subtypes.get(current_id, []):
                if entry == find_id:
                    found_cycles.append((find_id, current_id))
                iterate(entry, find_id)
        for the_id in subtypes['']:
            iterate(the_id, the_id)
        if len(found_cycles) > 0:
            for entry in found_cycles:
                logging.error(
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

    def _construct_parent_types(self):
        def iterate(t_key, parents):
            entry = [t_key, ]
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
        if not (isinstance(the_id, string_types) and isinstance(the_name, string_types) and
                isinstance(the_parent, string_types)):
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
        Adds a new entry. Note that leading or trialing blanks will be trimmed
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
                logging.warning(
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
            logging.error(
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
        None
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
            return

        if current_name != the_name:
            # check if name is already being used by a different element, and warn if so
            if current_name != the_name:
                for key, value in self.labels.items():
                    if value == the_name and key != the_id:
                        logging.warning(
                            'Note that id {} is already using name {}. Having repeated names is '
                            'permitted, but may lead to confusion.'.format(key, value))

        if current_parent != the_parent:
            labels = self.labels.copy()
            labels[the_id] = the_name
            subtypes = self.subtypes.copy()
            # remove the_id from it's current subtype
            subtypes[current_parent].remove(the_id)
            # add it to the new one
            subtypes[the_parent].append(the_id)
            try:
                self.set_labels_and_subtypes(labels, subtypes)
            except (ValueError, KeyError) as e:
                logging.error(
                    'Modifying entry id {}, name {}, and parent {} failed with '
                    'exception {}.'.format(the_id, the_name, the_parent, e))
        else:
            # just changing the name
            self.labels[the_id] = the_name

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
        value
            If string, it should likely be the geometry type string (Point, Linestring, etc).
            For any other object, the exact name of the class will be used for the check.

        Returns
        -------
        bool
        """

        if self._permitted_geometries is None or value is None:
            return True
        elif isinstance(value, str):
            return value in self._permitted_geometries
        else:
            return value.__class__.__name__ in self._permitted_geometries
