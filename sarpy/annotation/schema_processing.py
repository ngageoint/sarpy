# -*- coding: utf-8 -*-
"""
This module provides utilities for validating and preparing an annotation schema.
"""

from collections import OrderedDict
import json
from typing import Dict, List, Set


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

    __slots__ = ('_version', '_labels', '_subtypes', '_parent_types', '_confidence_values')
    _DEFAULT_CONF_VALUES = [0, 1, 2, 3, 4]

    def __init__(self, version, labels, subtypes, confidence_values=None):
        """

        Parameters
        ----------
        version : str
            The version of the schema.
        labels : Dict[str, str]
            The {<label id> : <label name>} pair dictionary. Each entry must be a string,
            and '' is not a valid label id.
        subtypes : None|Dict[str, List[str]]
            The {<label id> : <sub id list>} pairs. The root ids (i.e. those ids
            not belonging as sub-id to some other id) will be populated in the subtypes
            entry with empty string key (i.e. ''). Every key and entry of subtypes
            (excluding the subtypes root '') must correspond to an entry of labels,
            and no id can be a direct subtype of more than one id.
        confidence_values : None|List[str]
        """

        self._version = None
        self._labels = None
        self._subtypes = None
        self._parent_types = None
        self._confidence_values = None
        self.confidence_values = confidence_values
        self.set_labels_and_subtypes(version, labels, subtypes)

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
        It is canonically defined that an id is a parent of itself.

        Returns
        -------
        Dict[str, Set[str]]
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
            self._confidence_values = self._DEFAULT_CONF_VALUES
        elif not isinstance(conf_values, list):
            raise TypeError('confidence_values must be a list. Got type {}'.format(type(conf_values)))
        self._confidence_values = conf_values

    def __str__(self):
        return json.dumps(self.to_dict(), indent=1)

    def __repr__(self):
        return json.dumps(self.to_dict())

    def set_labels_and_subtypes(self, version, labels, subtypes):
        """
        Set the labels and subtypes. This modification must be accompanied by a
        version number modification. **Note that subtypes may be modified in place.**

        Parameters
        ----------
        version : str
        labels : dict
        subtypes : None|dict

        Returns
        -------
        None
        """

        if not isinstance(version, str):
            raise TypeError('version is required to be a string. Got type {}'.format(type(version)))

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

        # we need to check the reference count for each key in labels
        counts = OrderedDict((key, 0) for key in labels)

        # ensure that every key of subtypes is a string and every value is a list,
        # also that inclusion makes sense
        for key in subtypes:
            if not isinstance(key, str):
                raise TypeError(
                    'All keys of subtypes must be of type string. Got key `{}` of '
                    'type {}.'.format(key, type(key)))
            if key != '' and key not in labels:
                raise KeyError(
                    'All keys of subtypes must belong to labels. Got key `{}` '
                    'which is missing from labels.'.format(key))

            value = subtypes[key]
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
        # now, we set the values
        self._version = version
        self._labels = labels
        self._subtypes = subtypes
        self._construct_parent_types()

    def _construct_parent_types(self):
        def iterate(key, entries):
            if key not in self._subtypes:
                self._parent_types[key].union(entries)
                return

            self._parent_types[key].union(entries)
            for entry in self._subtypes[key]:
                iterate(entry, {key, }.union(entries))
            return

        self._parent_types = OrderedDict((key, {key, }) for key in self._labels)
        for key in self._subtypes['']:
            iterate(key, set())

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, 'r') as fi:
            input_dict = json.load(fi)
        return cls.from_dict(input_dict)

    @classmethod
    def from_dict(cls, input_dict):
        version = input_dict['version']
        labels = input_dict['labels']
        subtypes = input_dict.get('subtypes', None)
        return cls(version, labels, subtypes)

    def to_dict(self):
        out = OrderedDict()
        out['version'] = self.version
        out['labels'] = self._labels
        out['subtypes'] = self._subtypes
        return out

    def to_file(self, file_name):
        with open(file_name, 'w') as fi:
            json.dump(self.to_dict(), fi, indent=1)


# TODO: convert annotation from one schema to another
