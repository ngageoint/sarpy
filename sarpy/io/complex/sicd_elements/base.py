# -*- coding: utf-8 -*-
"""
This module contains the base objects for use in the SICD elements, and the base serializable functionality.
"""

import copy
import json

from xml.etree import ElementTree
from collections import OrderedDict
from datetime import datetime, date
import logging
from weakref import WeakKeyDictionary

import numpy
import numpy.polynomial.polynomial
from numpy.linalg import norm

from sarpy.compliance import int_func, integer_types, string_types


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


###
# module level constant - each module in the package will start with this
DEFAULT_STRICT = False
"""
bool: package level default behavior for whether to handle standards compliance strictly (raise exception) or more 
    loosely (by logging a warning)
"""


#################
# dom helper functions


def _get_node_value(nod):
    """XML parsing helper for extracting text value from an ElementTree Element. No error checking performed.

    Parameters
    ----------
    nod : ElementTree.Element
        the xml dom element

    Returns
    -------
    str
        the string value of the node.
    """

    if nod.text is None:
        return None

    val = nod.text.strip()
    if len(val) == 0:
        return None
    else:
        return val


def _create_new_node(doc, tag, parent=None):
    """XML ElementTree node creation helper function.

    Parameters
    ----------
    doc : ElementTree.ElementTree
        The xml Document object.
    tag : str
        Name/tag for new xml element.
    parent : None|ElementTree.Element
        The parent element for the new element. Defaults to the document root element if unspecified.
    Returns
    -------
    ElementTree.Element
        The new element populated as a child of `parent`.
    """

    if parent is None:
        parent = doc.getroot()  # what if there is no root?
    if parent is None:
        element = ElementTree.Element(tag)
        # noinspection PyProtectedMember
        doc._setroot(element)
        return element
    else:
        return ElementTree.SubElement(parent, tag)


def _create_text_node(doc, tag, value, parent=None):
    """XML ElementTree text node creation helper function

    Parameters
    ----------
    doc : ElementTree.ElementTree
        The xml Document object.
    tag : str
        Name/tag for new xml element.
    value : str
        The value for the new element.
    parent : None|ElementTree.Element
        The parent element for the new element. Defaults to the document root element if unspecified.

    Returns
    -------
    ElementTree.Element
        The new element populated as a child of `parent`.
    """

    node = _create_new_node(doc, tag, parent=parent)
    node.text = value
    return node


def _find_first_child(node, tag, xml_ns, ns_key):
    if xml_ns is None:
        return node.find(tag)
    elif ns_key is None:
        return node.find('default:{}'.format(tag), xml_ns)
    else:
        return node.find('{}:{}'.format(ns_key, tag), xml_ns)


def _find_children(node, tag, xml_ns, ns_key):
    if xml_ns is None:
        return node.findall(tag)
    elif ns_key is None:
        return node.findall('default:{}'.format(tag), xml_ns)
    else:
        return node.findall('{}:{}'.format(ns_key, tag), xml_ns)


###
# parsing functions - for reusable functionality in below descriptors or other property definitions


def _parse_str(value, name, instance):
    if value is None:
        return None
    if isinstance(value, string_types):
        return value
    elif isinstance(value, ElementTree.Element):
        return _get_node_value(value)
    else:
        raise TypeError(
            'field {} of class {} requires a string value.'.format(name, instance.__class__.__name__))


def _parse_bool(value, name, instance):
    def parse_string(val):
        if val.lower() in ['0', 'false']:
            return False
        elif val.lower() in ['1', 'true']:
            return True
        else:
            raise ValueError(
                'Boolean field {} of class {} cannot assign from string value {}. '
                'It must be one of ["0", "false", "1", "true"]'.format(name, instance.__class__.__name__, val))

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    elif isinstance(value, integer_types):
        return bool(value)
    elif isinstance(value, ElementTree.Element):
        # from XML deserialization
        return parse_string(_get_node_value(value))
    elif isinstance(value, string_types):
        return parse_string(value)
    else:
        raise ValueError('Boolean field {} of class {} cannot assign from type {}.'.format(
            name, instance.__class__.__name__, type(value)))


def _parse_int(value, name, instance):
    if value is None:
        return None
    if isinstance(value, integer_types):
        return value
    elif isinstance(value, ElementTree.Element):
        # from XML deserialization
        return _parse_int(_get_node_value(value), name, instance)
    elif isinstance(value, string_types):
        try:
            return int_func(value)
        except ValueError as e:
            logging.warning(
                'Got non-integer value {} for integer valued field {} of '
                'class {}'.format(value, name, instance.__class__.__name__))
            try:
                return int_func(float(value))
            except:
                raise e
    else:
        # user or json deserialization
        return int_func(value)


def _parse_float(value, name, instance):
    if value is None:
        return None
    if isinstance(value, float):
        return value
    elif isinstance(value, ElementTree.Element):
        # from XML deserialization
        return float(_get_node_value(value))
    else:
        # user or json deserialization
        return float(value)


def _parse_complex(value, name, instance):
    if value is None:
        return None
    if isinstance(value, complex):
        return value
    elif isinstance(value, ElementTree.Element):
        xml_ns = getattr(instance, '_xml_ns', None)
        if hasattr(instance, '_child_xml_ns_key') and name in instance._child_xml_ns_key:
            xml_ns_key = instance._child_xml_ns_key[name]
        else:
            xml_ns_key = getattr(instance, '_xml_ns_key', None)
        # from XML deserialization
        rnode = _find_children(value, 'Real', xml_ns, xml_ns_key)
        inode = _find_children(value, 'Imag', xml_ns, xml_ns_key)

        if len(rnode) != 1:
            raise ValueError(
                'There must be exactly one Real component of a complex type node '
                'defined for field {} of class {}.'.format(name, instance.__class__.__name__))
        if len(inode) != 1:
            raise ValueError(
                'There must be exactly one Imag component of a complex type node '
                'defined for field {} of class {}.'.format(name, instance.__class__.__name__))
        real = float(_get_node_value(rnode[0]))
        imag = float(_get_node_value(inode[0]))
        return complex(real, imag)
    elif isinstance(value, dict):
        # from json deserialization
        real = None
        for key in ['re', 'real', 'Real']:
            real = value.get(key, real)
        imag = None
        for key in ['im', 'imag', 'Imag']:
            imag = value.get(key, imag)
        if real is None or imag is None:
            raise ValueError(
                'Cannot convert dict {} to a complex number for field {} of '
                'class {}.'.format(value, name, instance.__class__.__name__))
        return complex(real, imag)
    else:
        # from user - I can't imagine that this would ever work
        return complex(value)


def _parse_datetime(value, name, instance, units='us'):
    if value is None:
        return None
    if isinstance(value, numpy.datetime64):
        return value
    elif isinstance(value, string_types):
        # handle Z timezone identifier explicitly - any timezone identifier is deprecated
        if value[-1] == 'Z':
            return numpy.datetime64(value[:-1], units)
        else:
            return numpy.datetime64(value, units)
    elif isinstance(value, ElementTree.Element):
        # from XML deserialization - extract the string
        return _parse_datetime(_get_node_value(value), name, instance, units=units)
    elif isinstance(value, (date, datetime, numpy.int64, numpy.float64)):
        return numpy.datetime64(value, units)
    elif isinstance(value, integer_types):
        # this is less safe, because the units are unknown...
        return numpy.datetime64(value, units)
    else:
        raise TypeError(
            'Field {} for class {} expects datetime convertible input, and '
            'got {}'.format(name, instance.__class__.__name__, type(value)))


def _parse_serializable(value, name, instance, the_type):
    if value is None:
        return None
    if isinstance(value, the_type):
        return value
    elif isinstance(value, dict):
        return the_type.from_dict(value)
    elif isinstance(value, ElementTree.Element):
        xml_ns = getattr(instance, '_xml_ns', None)
        if hasattr(instance, '_child_xml_ns_key'):
            # noinspection PyProtectedMember
            xml_ns_key = instance._child_xml_ns_key.get(name, getattr(instance, '_xml_ns_key', None))
        else:
            xml_ns_key = getattr(instance, '_xml_ns_key', None)
        return the_type.from_node(value, xml_ns, ns_key=xml_ns_key)
    elif isinstance(value, (numpy.ndarray, list, tuple)):
        if issubclass(the_type, Arrayable):
            return the_type.from_array(value)
        else:
            raise TypeError(
                'Field {} of class {} is of type {} (not a subclass of Arrayable) and '
                'got an argument of type {}.'.format(name, instance.__class__.__name__, the_type, type(value)))
    else:
        raise TypeError(
            'Field {} of class {} is expecting type {}, but got an instance of incompatible '
            'type {}.'.format(name, instance.__class__.__name__, the_type, type(value)))


def _parse_serializable_array(value, name, instance, child_type, child_tag):
    if value is None:
        return None
    if isinstance(value, child_type):
        # this is the child element
        return numpy.array([value, ], dtype=numpy.object)
    elif isinstance(value, numpy.ndarray):
        if value.dtype.name != 'object':
            if issubclass(child_type, Arrayable):
                return numpy.array([child_type.from_array(array) for array in value], dtype=numpy.object)
            else:
                raise ValueError(
                    'Attribute {} of array type functionality belonging to class {} got an ndarray of dtype {},'
                    'and child type is not a subclass of Arrayable.'.format(
                        name, instance.__class__.__name__, value.dtype))
        elif len(value.shape) != 1:
            raise ValueError(
                'Attribute {} of array type functionality belonging to class {} got an ndarray of shape {},'
                'but requires a one dimensional array.'.format(
                    name, instance.__class__.__name__, value.shape))
        elif not isinstance(value[0], child_type):
            raise TypeError(
                'Attribute {} of array type functionality belonging to class {} got an ndarray containing '
                'first element of incompatible type {}.'.format(
                    name, instance.__class__.__name__, type(value[0])))
        return value
    elif isinstance(value, ElementTree.Element):
        xml_ns = getattr(instance, '_xml_ns', None)
        if hasattr(instance, '_child_xml_ns_key'):
            # noinspection PyProtectedMember
            xml_ns_key = instance._child_xml_ns_key.get(name, getattr(instance, '_xml_ns_key', None))
        else:
            xml_ns_key = getattr(instance, '_xml_ns_key', None)
        # this is the parent node from XML deserialization
        size = int_func(value.attrib.get('size', -1))  # NB: Corner Point arrays don't have
        # extract child nodes at top level
        child_nodes = _find_children(value, child_tag, xml_ns, xml_ns_key)

        if size == -1:  # fill in, if it's missing
            size = len(child_nodes)
        if len(child_nodes) != size:
            raise ValueError(
                'Attribute {} of array type functionality belonging to class {} got a ElementTree element '
                'with size attribute {}, but has {} child nodes with tag {}.'.format(
                    name, instance.__class__.__name__, size, len(child_nodes), child_tag))
        new_value = numpy.empty((size, ), dtype=numpy.object)
        for i, entry in enumerate(child_nodes):
            new_value[i] = child_type.from_node(entry, xml_ns, ns_key=xml_ns_key)
        return new_value
    elif isinstance(value, (list, tuple)):
        # this would arrive from users or json deserialization
        if len(value) == 0:
            return numpy.empty((0,), dtype=numpy.object)
        elif isinstance(value[0], child_type):
            return numpy.array(value, dtype=numpy.object)
        elif isinstance(value[0], dict):
            # NB: charming errors are possible here if something stupid has been done.
            return numpy.array([child_type.from_dict(node) for node in value], dtype=numpy.object)
        elif isinstance(value[0], (numpy.ndarray, list, tuple)):
            if issubclass(child_type, Arrayable):
                return numpy.array([child_type.from_array(array) for array in value], dtype=numpy.object)
            elif hasattr(child_type, 'Coefs'):
                return numpy.array([child_type(Coefs=array) for array in value], dtype=numpy.object)
            else:
                raise ValueError(
                    'Attribute {} of array type functionality belonging to class {} got an list '
                    'containing elements type {} and construction failed.'.format(
                        name, instance.__class__.__name__, type(value[0])))
        else:
            raise TypeError(
                'Attribute {} of array type functionality belonging to class {} got a list containing first '
                'element of incompatible type {}.'.format(name, instance.__class__.__name__, type(value[0])))
    else:
        raise TypeError(
            'Attribute {} of array type functionality belonging to class {} got incompatible type {}.'.format(
                name, instance.__class__.__name__, type(value)))


def _parse_serializable_list(value, name, instance, child_type):
    if value is None:
        return None
    if isinstance(value, child_type):
        # this is the child element
        return [value, ]

    xml_ns = getattr(instance, '_xml_ns', None)
    if hasattr(instance, '_child_xml_ns_key'):
        # noinspection PyProtectedMember
        xml_ns_key = instance._child_xml_ns_key.get(name, getattr(instance, '_xml_ns_key', None))
    else:
        xml_ns_key = getattr(instance, '_xml_ns_key', None)
    if isinstance(value, ElementTree.Element):
        # this is the child
        return [child_type.from_node(value, xml_ns, ns_key=xml_ns_key), ]
    elif isinstance(value, list) or isinstance(value[0], child_type):
        if len(value) == 0:
            return value
        elif isinstance(value[0], child_type):
            return [entry for entry in value]
        elif isinstance(value[0], dict):
            # NB: charming errors are possible if something stupid has been done.
            return [child_type.from_dict(node) for node in value]
        elif isinstance(value[0], ElementTree.Element):
            return [child_type.from_node(node, xml_ns, ns_key=xml_ns_key) for node in value]
        else:
            raise TypeError(
                'Field {} of list type functionality belonging to class {} got a '
                'list containing first element of incompatible type '
                '{}.'.format(name, instance.__class__.__name__, type(value[0])))
    else:
        raise TypeError(
            'Field {} of class {} got incompatible type {}.'.format(
                name, instance.__class__.__name__, type(value)))


def _parse_parameters_collection(value, name, instance):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    elif isinstance(value, list):
        out = OrderedDict()
        if len(value) == 0:
            return out
        if isinstance(value[0], ElementTree.Element):
            for entry in value:
                out[entry.attrib['name']] = _get_node_value(entry)
            return out
        else:
            raise TypeError(
                'Field {} of list type functionality belonging to class {} got a '
                'list containing first element of incompatible type '
                '{}.'.format(name, instance.__class__.__name__, type(value[0])))
    else:
        raise TypeError(
            'Field {} of class {} got incompatible type {}.'.format(
                name, instance.__class__.__name__, type(value)))


###
# descriptor definitions - these are reusable properties that handle typing and deserialization in one place


class _BasicDescriptor(object):
    """A descriptor object for reusable properties. Note that is is required that the calling instance is hashable."""
    _typ_string = None

    def __init__(self, name, required, strict=DEFAULT_STRICT, default_value=None, docstring=''):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = (name in required)
        self.strict = strict
        self.default_value = default_value

        self.__doc__ = docstring
        self._format_docstring()

    def _format_docstring(self):
        docstring = self.__doc__
        if docstring is None:
            docstring = ''
        if (self._typ_string is not None) and (not docstring.startswith(self._typ_string)):
            docstring = '{} {}'.format(self._typ_string, docstring)

        suff = self._docstring_suffix()
        if suff is not None:
            docstring = '{} {}'.format(docstring, suff)

        lenstr = self._len_string()
        if lenstr is not None:
            docstring = '{} {}'.format(docstring, lenstr)

        if self.required:
            docstring = '{} {}'.format(docstring, ' **Required.**')
        else:
            docstring = '{} {}'.format(docstring, ' **Optional.**')
        self.__doc__ = docstring

    def _len_string(self):
        minl = getattr(self, 'minimum_length', None)
        maxl = getattr(self, 'minimum_length', None)
        def_minl = getattr(self, '_DEFAULT_MIN_LENGTH', None)
        def_maxl = getattr(self, '_DEFAULT_MAX_LENGTH', None)
        if minl is not None and maxl is not None:
            if minl == def_minl and maxl == def_maxl:
                return None

            lenstr = ' Must have length '
            if minl == def_minl:
                lenstr += '<= {0:d}.'.format(maxl)
            elif maxl == def_maxl:
                lenstr += '>= {0:d}.'.format(minl)
            elif minl == maxl:
                lenstr += ' exactly {0:d}.'.format(minl)
            else:
                lenstr += 'in the range [{0:d}, {1:d}].'.format(minl, maxl)
            return lenstr
        else:
            return None

    def _docstring_suffix(self):
        return None

    def __get__(self, instance, owner):
        """The getter.

        Parameters
        ----------
        instance : object
            the calling class instance
        owner : object
            the type of the class - that is, the actual object to which this descriptor is assigned

        Returns
        -------
        object
            the return value
        """

        if instance is None:
            # this has been access on the class, so return the class
            return self

        fetched = self.data.get(instance, self.default_value)
        if fetched is not None or not self.required:
            return fetched
        else:
            msg = 'Required field {} of class {} is not populated.'.format(self.name, instance.__class__.__name__)
            if self.strict:
                raise AttributeError(msg)
            else:
                logging.debug(msg)  # NB: this is at debug level to not be too verbose
            return fetched

    def __set__(self, instance, value):
        """The setter method.

        Parameters
        ----------
        instance : object
            the calling class instance
        value
            the value to use in setting - the type depends of the specific extension of this base class

        Returns
        -------
        bool
            this base class, and only this base class, handles the required compliance and None behavior and has
            a return. This returns True if this the setting value was None, and False otherwise.
        """

        # NOTE: This is intended to handle this case for every extension of this class. Hence the boolean return,
        # which extensions SHOULD NOT implement. This is merely to follow DRY principles.
        if value is None:
            if self.default_value is not None:
                self.data[instance] = self.default_value
                return True
            elif self.required:
                if self.strict:
                    raise ValueError(
                        'Attribute {} of class {} cannot be assigned '
                        'None.'.format(self.name, instance.__class__.__name__))
                else:
                    logging.debug(  # NB: this is at debuglevel to not be too verbose
                        'Required attribute {} of class {} has been set to None.'.format(
                            self.name, instance.__class__.__name__))
            self.data[instance] = None
            return True
        # note that the remainder must be implemented in each extension
        return False  # this is probably a bad habit, but this returns something for convenience alone


class _StringDescriptor(_BasicDescriptor):
    """A descriptor for string type"""
    _typ_string = 'str:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        super(_StringDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):

        if super(_StringDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        self.data[instance] = _parse_str(value, self.name, instance)


class _StringListDescriptor(_BasicDescriptor):
    """A descriptor for properties for an array type item for specified extension of string"""
    _typ_string = 'List[str]:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32

    def __init__(self, name, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None,
                 default_value=None, docstring=None):
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int_func(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int_func(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_StringListDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Attribute {} of class {} is a string list of size {}, and must have length at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.error(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Attribute {} of class {} is a string list of size {}, and must have length no greater than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.error(msg)
            self.data[instance] = new_value

        if super(_StringListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, string_types):
            set_value([value, ])
        elif isinstance(value, ElementTree.Element):
            set_value([_get_node_value(value), ])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], string_types):
                set_value(value)
            elif isinstance(value[0], ElementTree.Element):
                set_value([_get_node_value(nod) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _StringEnumDescriptor(_BasicDescriptor):
    """A descriptor for enumerated (specified) string type.
    **This implicitly assumes that the valid entries are upper case.**"""
    _typ_string = 'str:'

    def __init__(self, name, values, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        self.values = values
        super(_StringEnumDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)
        if (self.default_value is not None) and (self.default_value not in self.values):
            self.default_value = None

    def _docstring_suffix(self):
        suff = ' Takes values in :code:`{}`.'.format(self.values)
        if self.default_value is not None:
            suff += ' Default value is :code:`{}`.'.format(self.default_value)
        return suff

    def __set__(self, instance, value):
        if value is None:
            if self.default_value is not None:
                self.data[instance] = self.default_value
            else:
                super(_StringEnumDescriptor, self).__set__(instance, value)
            return

        val = _parse_str(value, self.name, instance)

        if val in self.values:
            self.data[instance] = val
        else:
            msg = 'Attribute {} of class {} received {}, but values ARE REQUIRED to be ' \
                  'one of {}'.format(self.name, instance.__class__.__name__, value, self.values)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.error(msg)
            self.data[instance] = val


class _BooleanDescriptor(_BasicDescriptor):
    """A descriptor for boolean type"""
    _typ_string = 'bool:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        super(_BooleanDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):
        if super(_BooleanDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return
        try:
            self.data[instance] = _parse_bool(value, self.name, instance)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to `bool` for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None


class _IntegerDescriptor(_BasicDescriptor):
    """A descriptor for integer type"""
    _typ_string = 'int:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, bounds=None, default_value=None, docstring=None):
        self.bounds = bounds
        super(_IntegerDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)
        if (self.default_value is not None) and not self._in_bounds(self.default_value):
            self.default_value = None

    def _docstring_suffix(self):
        if self.bounds is not None:
            return 'Must be in the range [{}, {}]'.format(*self.bounds)
        return ''

    def _in_bounds(self, value):
        if self.bounds is None:
            return True
        return (self.bounds[0] is None or self.bounds[0] <= value) and \
            (self.bounds[1] is None or value <= self.bounds[1])

    def __set__(self, instance, value):
        if super(_IntegerDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            iv = _parse_int(value, self.name, instance)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to `int` for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return

        if self._in_bounds(iv):
            self.data[instance] = iv
        else:
            msg = 'Attribute {} of class {} is required by standard to take value between {}. ' \
                  'Invalid value {}'.format(self.name, instance.__class__.__name__, self.bounds, iv)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.error(msg)
            self.data[instance] = iv


class _IntegerEnumDescriptor(_BasicDescriptor):
    """A descriptor for enumerated (specified) integer type"""
    _typ_string = 'int:'

    def __init__(self, name, values, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        self.values = values
        super(_IntegerEnumDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)
        if (self.default_value is not None) and (self.default_value not in self.values):
            self.default_value = None

    def _docstring_suffix(self):
        return 'Must take one of the values in {}.'.format(self.values)

    def __set__(self, instance, value):
        if super(_IntegerEnumDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            iv = _parse_int(value, self.name, instance)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to `int` for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return

        if iv in self.values:
            self.data[instance] = iv
        else:
            msg = 'Attribute {} of class {} must take value in {}. Invalid value {}.'.format(
                self.name, instance.__class__.__name__, self.values, iv)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.error(msg)
            self.data[instance] = iv


class _IntegerListDescriptor(_BasicDescriptor):
    """A descriptor for integer list type properties"""
    _typ_string = 'list[int]:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int_func(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int_func(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_IntegerListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Attribute {} of class {} is an integer list of size {}, and must have size at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.info(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Attribute {} of class {} is an integer list of size {}, and must have size no larger than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.info(msg)
            self.data[instance] = new_value

        if super(_IntegerListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, integer_types):
            set_value([value, ])
        elif isinstance(value, ElementTree.Element):
            set_value([int_func(_get_node_value(value)), ])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], integer_types):
                set_value(value)
            elif isinstance(value[0], ElementTree.Element):
                set_value([int_func(_get_node_value(nod)) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _FloatDescriptor(_BasicDescriptor):
    """A descriptor for float type properties"""
    _typ_string = 'float:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, bounds=None, default_value=None, docstring=None):
        self.bounds = bounds
        super(_FloatDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)
        if (self.default_value is not None) and not self._in_bounds(self.default_value):
            self.default_value = None

    def _docstring_suffix(self):
        if self.bounds is not None:
            return 'Must be in the range [{}, {}]'.format(*self.bounds)
        return ''

    def _in_bounds(self, value):
        if self.bounds is None:
            return True

        return (self.bounds[0] is None or self.bounds[0] <= value) and \
            (self.bounds[1] is None or value <= self.bounds[1])

    def __set__(self, instance, value):
        if super(_FloatDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            iv = _parse_float(value, self.name, instance)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to `float` for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return

        if self._in_bounds(iv):
            self.data[instance] = iv
        else:
            msg = 'Attribute {} of class {} is required by standard to take value between {}.'.format(
                self.name, instance.__class__.__name__, self.bounds)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.info(msg)
            self.data[instance] = iv


class _FloatListDescriptor(_BasicDescriptor):
    """A descriptor for float list type properties"""
    _typ_string = 'list[float]:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int_func(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int_func(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_FloatListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Attribute {} of class {} is an float list of size {}, and must have size at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.info(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Attribute {} of class {} is a float list of size {}, and must have size no larger than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.info(msg)
            self.data[instance] = new_value

        if super(_FloatListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, float):
            set_value([value, ])
        elif isinstance(value, ElementTree.Element):
            set_value([float(_get_node_value(value)), ])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], float):
                set_value(value)
            elif isinstance(value[0], ElementTree.Element):
                set_value([float(_get_node_value(nod)) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _ComplexDescriptor(_BasicDescriptor):
    """A descriptor for complex valued properties"""
    _typ_string = 'complex:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        super(_ComplexDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):
        if super(_ComplexDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            self.data[instance] = _parse_complex(value, self.name, instance)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to `complex` for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None


class _FloatArrayDescriptor(_BasicDescriptor):
    """A descriptor for float array type properties"""
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32
    _typ_string = 'numpy.ndarray[float64]:'

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None,
                 docstring=None):

        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int_func(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int_func(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_FloatArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        def set_value(new_val):
            if len(new_val) < self.minimum_length:
                msg = 'Attribute {} of class {} is a double array of size {}, and must have size at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.error(msg)
            if len(new_val) > self.maximum_length:
                msg = 'Attribute {} of class {} is a double array of size {}, and must have size no larger than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.error(msg)
            self.data[instance] = new_val

        if super(_FloatArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, numpy.ndarray):
            if not (len(value) == 1) and (numpy.dtype.name == 'float64'):
                raise ValueError('Only one-dimensional ndarrays of dtype float64 are supported here.')
            set_value(value)
        elif isinstance(value, ElementTree.Element):
            xml_ns = getattr(instance, '_xml_ns', None)
            # noinspection PyProtectedMember
            if hasattr(instance, '_child_xml_ns_key') and self.name in instance._child_xml_ns_key:
                # noinspection PyProtectedMember
                xml_ns_key = instance._child_xml_ns_key[self.name]
            else:
                xml_ns_key = getattr(instance, '_xml_ns_key', None)

            size = int_func(value.attrib['size'])
            child_nodes = _find_children(value, self.child_tag, xml_ns, xml_ns_key)
            if len(child_nodes) != size:
                raise ValueError(
                    'Field {} of double array type functionality belonging to class {} got a ElementTree element '
                    'with size attribute {}, but has {} child nodes with tag {}.'.format(
                        self.name, instance.__class__.__name__, size, len(child_nodes), self.child_tag))
            new_value = numpy.empty((size,), dtype=numpy.float64)
            for i, node in enumerate(child_nodes):
                new_value[i] = float(_get_node_value(node))
            set_value(new_value)
        elif isinstance(value, list):
            # user or json deserialization
            set_value(numpy.array(value, dtype=numpy.float64))
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _DateTimeDescriptor(_BasicDescriptor):
    """A descriptor for date time type properties"""
    _typ_string = 'numpy.datetime64:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, docstring=None, numpy_datetime_units='us'):
        self.units = numpy_datetime_units  # s, ms, us, ns are likely choices here, depending on needs
        super(_DateTimeDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_DateTimeDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return
        self.data[instance] = _parse_datetime(value, self.name, instance, self.units)


class _FloatModularDescriptor(_BasicDescriptor):
    """
    A descriptor for float type which will take values in a range [-limit, limit], set using modular
    arithmetic operations
    """
    _typ_string = 'float:'

    def __init__(self, name, limit, required, strict=DEFAULT_STRICT, docstring=None):
        self.limit = float(limit)
        super(_FloatModularDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_FloatModularDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            val = _parse_float(value, self.name, instance)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to `float` for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return

        # do modular arithmetic manipulations
        val = (val % (2 * self.limit))  # NB: % and * have same precedence, so it can be super dumb
        self.data[instance] = val if val <= self.limit else val - 2 * self.limit


class _SerializableDescriptor(_BasicDescriptor):
    """A descriptor for properties of a specified type assumed to be an extension of Serializable"""

    def __init__(self, name, the_type, required, strict=DEFAULT_STRICT, docstring=None):
        self.the_type = the_type
        self._typ_string = str(the_type).strip().split('.')[-1][:-2] + ':'
        super(_SerializableDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_SerializableDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            self.data[instance] = _parse_serializable(value, self.name, instance, self.the_type)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to Serializable type {} for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard.'.format(
                    value, type(value), self.the_type, self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None


class _UnitVectorDescriptor(_BasicDescriptor):
    """A descriptor for properties of a specified type assumed to be of subtype of Arrayable"""

    def __init__(self, name, the_type, required, strict=DEFAULT_STRICT, docstring=None):
        if not issubclass(the_type, Arrayable):
            raise TypeError(
                'The input type {} for field {} must be a subclass of Arrayable.'.format(the_type, name))
        self.the_type = the_type

        self._typ_string = str(the_type).strip().split('.')[-1][:-2] + ':'
        super(_UnitVectorDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_UnitVectorDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            vec = _parse_serializable(value, self.name, instance, self.the_type)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to Unit Vector Type type {} for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard'.format(
                    value, type(value), self.the_type, self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return None

        # noinspection PyTypeChecker
        coords = vec.get_array(dtype=numpy.float64)
        the_norm = norm(coords)
        if the_norm == 0:
            logging.error(
                'The input for field {} is expected to be made into a unit vector. '
                'In this case, the norm of the input is 0. The value is set to None, '
                'which may be against the standard.'.format(self.name))
            self.data[instance] = None
        elif the_norm == 1:
            self.data[instance] = vec
        else:
            self.data[instance] = self.the_type.from_array(coords/the_norm)


class _ParametersDescriptor(_BasicDescriptor):
    """A descriptor for properties of a Parameter type - that is, dictionary"""

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self._typ_string = 'ParametersCollection:'
        super(_ParametersDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_ParametersDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, ParametersCollection):
            self.data[instance] = value
        else:
            the_inst = self.data.get(instance, None)
            xml_ns = getattr(instance, '_xml_ns', None)
            # noinspection PyProtectedMember
            if hasattr(instance, '_child_xml_ns_key') and self.name in instance._child_xml_ns_key:
                # noinspection PyProtectedMember
                xml_ns_key = instance._child_xml_ns_key[self.name]
            else:
                xml_ns_key = getattr(instance, '_xml_ns_key', None)
            if the_inst is None:
                self.data[instance] = ParametersCollection(
                    collection=value, name=self.name, child_tag=self.child_tag,
                    _xml_ns=xml_ns, _xml_ns_key=xml_ns_key)
            else:
                the_inst.set_collection(value)


class _SerializableCPArrayDescriptor(_BasicDescriptor):
    """A descriptor for properties of a list or array of specified extension of Serializable"""
    minimum_length = 4
    maximum_length = 4

    def __init__(self, name, child_type, tag_dict, required, strict=DEFAULT_STRICT, docstring=None):
        self.child_type = child_type
        tags = tag_dict[name]
        self.array = tags.get('array', False)
        if not self.array:
            raise ValueError(
                'Attribute {} is populated in the `_collection_tags` dictionary without `array`=True. '
                'This is inconsistent with using _SerializableCPArrayDescriptor.'.format(name))

        self.child_tag = tags['child_tag']
        self._typ_string = 'numpy.ndarray[{}]:'.format(str(child_type).strip().split('.')[-1][:-2])

        super(_SerializableCPArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_SerializableCPArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, SerializableCPArray):
            self.data[instance] = value
        else:
            xml_ns = getattr(instance, '_xml_ns', None)
            # noinspection PyProtectedMember
            if hasattr(instance, '_child_xml_ns_key') and self.name in instance._child_xml_ns_key:
                # noinspection PyProtectedMember
                xml_ns_key = instance._child_xml_ns_key[self.name]
            else:
                xml_ns_key = getattr(instance, '_xml_ns_key', None)

            the_inst = self.data.get(instance, None)
            if the_inst is None:
                self.data[instance] = SerializableCPArray(
                    coords=value, name=self.name, child_tag=self.child_tag,
                    child_type=self.child_type, _xml_ns=xml_ns, _xml_ns_key=xml_ns_key)
            else:
                the_inst.set_array(value)


class _SerializableListDescriptor(_BasicDescriptor):
    """A descriptor for properties of a list or array of specified extension of Serializable"""

    def __init__(self, name, child_type, tag_dict, required, strict=DEFAULT_STRICT, docstring=None):
        self.child_type = child_type
        tags = tag_dict[name]
        self.array = tags.get('array', False)
        if self.array:
            raise ValueError(
                'Attribute {} is populated in the `_collection_tags` dictionary with `array`=True. '
                'This is inconsistent with using _SerializableListDescriptor.'.format(name))

        self.child_tag = tags['child_tag']
        self._typ_string = 'List[{}]:'.format(str(child_type).strip().split('.')[-1][:-2])
        super(_SerializableListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_SerializableListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            self.data[instance] = _parse_serializable_list(value, self.name, instance, self.child_type)
        except Exception as e:
            logging.error(
                'Failed converting {} of type {} to serializable list of type {} for field {} of '
                'class {} with exception {} - {}. Setting value to None, '
                'which may be against the standard'.format(
                    value, type(value), self.child_type, self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None

#################
# base Serializable class.


class Serializable(object):
    """
    Basic abstract class specifying the serialization pattern. There are no clearly defined Python conventions
    for this issue. Every effort has been made to select sensible choices, but this is an individual effort.

    Notes
    -----
        All fields MUST BE LISTED in the `_fields` tuple. Everything listed in `_required` tuple will be checked
        for inclusion in `_fields` tuple. Note that special care must be taken to ensure compatibility of `_fields`
        tuple, if inheriting from an extension of this class.
    """
    _fields = ()
    """collection of field names"""
    _required = ()
    """subset of `_fields` defining the required (for the given object, according to the sicd standard) fields"""

    _collections_tags = {}
    """
    Entries only appropriate for list/array type objects. Entry formatting:

    * `{'array': True, 'child_tag': <child_name>}` represents an array object, which will have int attribute `size`.
      It has *size* children with tag=<child_name>, each of which has an attribute `index`, which is not always an
      integer. Even when it is an integer, it apparently sometimes follows the matlab convention (1 based), and
      sometimes the standard convention (0 based). In this case, I will deserialize as though the objects are
      properly ordered, and the deserialized objects will have the `index` property from the xml, but it will not
      be used to determine array order - which will be simply carried over from file order.

    * `{'array': False, 'child_tag': <child_name>}` represents a collection of things with tag=<child_name>.
      This entries are not directly below one coherent container tag, but just dumped into an object.
      For example of such usage search for "Parameter" in the SICD standard.

      In this case, I've have to create an ephemeral variable in the class that doesn't exist in the standard,
      and it's not clear what the intent is for this unstructured collection, so used a list object.
      For example, I have a variable called `Parameters` in `CollectionInfoType`, whose job it to contain the
      parameters objects.
    """

    _numeric_format = {}
    """define dict entries of numeric formatting for serialization"""
    _set_as_attribute = ()
    """serialize these fields as xml attributes"""
    _choice = ()
    """
    Entries appropriate for choice selection between attributes. Entry formatting:

    * `{'required': True, 'collection': <tuple of attribute names>}` - indicates that EXACTLY only one of the 
      attributes should be populated.

    * `{'required': False, 'collection': <tuple of attribute names>}` - indicates that no more than one of the 
      attributes should be populated.
    """
    _child_xml_ns_key = {}
    """
    The expected namespace key for attributes. No entry indicates the default namespace. 
    This is important for SIDD handling, but not required for SICD handling.
    """

    # NB: it may be good practice to use __slots__ to further control class functionality?

    def __init__(self, **kwargs):
        """
        The default constructor. For each attribute name in `self._fields`, fetches the value (or None) from
        the `kwargs` dict, and sets the class instance attribute value. The details for attribute value validation,
        present for virtually every attribute, will be implemented specifically as descriptors.

        Parameters
        ----------
        **kwargs :
            the keyword arguments dictionary - the possible entries match the attributes.
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        unexpected_args = [key for key in kwargs if key not in self._fields and key[0] != '_']
        if len(unexpected_args) > 0:
            raise ValueError(
                'Received unexpected construction argument {} for attribute '
                'collection {}'.format(unexpected_args, self._fields))

        for attribute in self._fields:
            if attribute in kwargs:
                try:
                    setattr(self, attribute, kwargs.get(attribute, None))
                except AttributeError:
                    # NB: this is included to allow for read only properties without breaking the paradigm
                    #   Silently catching errors can potentially cover up REAL issues.
                    pass

    def __str__(self):
        return '{}(**{})'.format(self.__class__.__name__, json.dumps(self.to_dict(check_validity=False), indent=1))

    def __repr__(self):
        return '{}(**{})'.format(self.__class__.__name__, self.to_dict(check_validity=False))

    def __setattr__(self, key, value):
        if not (key.startswith('_') or (key in self._fields) or hasattr(self.__class__, key) or hasattr(self, key)):
            # not expected attribute - descriptors, properties, etc
            logging.warning(
                'Class {} instance receiving unexpected attribute {}.\n'
                '\tEnsure that this is not a typo of an expected field name.'.format(self.__class__.__name__, key))
        object.__setattr__(self, key, value)

    def set_numeric_format(self, attribute, format_string):
        """Sets the numeric format string for the given attribute.

        Parameters
        ----------
        attribute : str
            attribute for which the format applies - must be in `_fields`.
        format_string : str
            format string to be applied
        Returns
        -------
        None
        """
        # Extend this to include format function capabilities. Maybe numeric_format is not the right name?
        if attribute not in self._fields:
            raise ValueError('attribute {} is not permitted for class {}'.format(attribute, self.__class__.__name__))
        self._numeric_format[attribute] = format_string

    def _get_formatter(self, attribute):
        """Return a formatting function for the given attribute. This will default to `str` if no other
        option is presented.

        Parameters
        ----------
        attribute : str
            the given attribute name

        Returns
        -------
        Callable
            format function
        """

        entry = self._numeric_format.get(attribute, None)
        if isinstance(entry, string_types):
            fmt_str = '{0:' + entry + '}'
            return fmt_str.format
        elif callable(entry):
            return entry
        else:
            return str

    def is_valid(self, recursive=False):
        """Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `_required`) are not `None`.

        Parameters
        ----------
        recursive : bool
            True if we recursively check that child are also valid. This may result in verbose (i.e. noisy) logging.

        Returns
        -------
        bool
            condition for validity of this element
        """

        all_required = self._basic_validity_check()
        if not recursive:
            return all_required

        valid_children = self._recursive_validity_check()
        return all_required & valid_children

    def _basic_validity_check(self):
        """
        Perform the basic validity check on the direct attributes with no recursive checking.

        Returns
        -------
        bool
             True if all requirements *AT THIS LEVEL* are satisfied, otherwise False.
        """

        all_required = True
        for attribute in self._required:
            present = (getattr(self, attribute) is not None)
            if not present:
                logging.error(
                    "Class {} is missing required attribute {}".format(self.__class__.__name__, attribute))
            all_required &= present

        choices = True
        for entry in self._choice:
            required = entry.get('required', False)
            collect = entry['collection']
            # verify that no more than one of the entries in collect is set.
            present = []
            for attribute in collect:
                if getattr(self, attribute) is not None:
                    present.append(attribute)
            if len(present) == 0 and required:
                logging.error(
                    "Class {} requires that exactly one of the attributes {} is set, but none are "
                    "set.".format(self.__class__.__name__, collect))
                choices = False
            elif len(present) > 1:
                logging.error(
                    "Class {} requires that no more than one of attributes {} is set, but multiple {} are "
                    "set.".format(self.__class__.__name__, collect, present))
                choices = False

        return all_required and choices

    def _recursive_validity_check(self):
        """
        Perform a recursive validity check on all present attributes.

        Returns
        -------
        bool
             True if requirements are recursively satisfied *BELOW THIS LEVEL*, otherwise False.
        """

        def check_item(value):
            if isinstance(value, (Serializable, SerializableArray)):
                return value.is_valid(recursive=True)
            return True

        valid_children = True
        for attribute in self._fields:
            val = getattr(self, attribute)
            good = True
            if isinstance(val, (Serializable, SerializableArray)):
                good = check_item(val)
            elif isinstance(val, list):
                for entry in val:
                    good &= check_item(entry)
            # any issues will be logged as discovered, but we should help with the "stack"
            if not good:
                logging.error(  # I should probably do better with a stack type situation. This is traceable, at least.
                    "Issue discovered with {} attribute of type {} of class {}.".format(
                        attribute, type(val), self.__class__.__name__))
            valid_children &= good
        return valid_children

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        """For XML deserialization.

        Parameters
        ----------
        node : ElementTree.Element
            dom element for serialized class instance
        xml_ns : None|dict
            The xml namespace dictionary.
        ns_key : None|str
            The xml namespace key. If `xml_ns` is None, then this is ignored. If `None` and `xml_ns` is not None,
            then the string `default` will be used. This will be recursively passed down,
            unless overridden by an entry of the cls._child_xml_ns_key dictionary.
        kwargs : None|dict
            `None` or dictionary of previously serialized attributes. For use in inheritance call, when certain
            attributes require specific deserialization.
        Returns
        -------
            Corresponding class instance
        """

        if len(node) == 0 and len(node.attrib) == 0:
            logging.warning('There are no children or attributes associated with node {} for class {}. Returning None.'.format(node, cls))
            return None

        def handle_attribute(the_tag, the_xml_ns_key):
            if the_xml_ns_key is not None:  # handle namespace, if necessary
                fetch_tag = '{' + xml_ns[the_xml_ns_key] + '}' + the_tag
            else:
                fetch_tag = the_tag
            poo = node.attrib.get(fetch_tag, None)
            kwargs[the_tag] = node.attrib.get(fetch_tag, None)

        def handle_single(the_tag, the_xml_ns_key):
            kwargs[the_tag] = _find_first_child(node, the_tag, xml_ns, the_xml_ns_key)

        def handle_list(attrib, ch_tag, the_xml_ns_key):
            cnodes = _find_children(node, ch_tag, xml_ns, the_xml_ns_key)
            if len(cnodes) > 0:
                kwargs[attrib] = cnodes

        if kwargs is None:
            kwargs = {}
        kwargs['_xml_ns'] = xml_ns
        kwargs['_xml_ns_key'] = ns_key

        if not isinstance(kwargs, dict):
            raise ValueError(
                "Named input argument kwargs for class {} must be dictionary instance".format(cls))

        for attribute in cls._fields:
            if attribute in kwargs:
                continue

            kwargs[attribute] = None
            # This value will be replaced if tags are present
            # Note that we want to try explicitly setting to None to trigger descriptor behavior
            # for required fields (warning or error)

            # determine any expected xml namespace for the given entry
            if attribute in cls._child_xml_ns_key:
                xml_ns_key = cls._child_xml_ns_key[attribute]
            else:
                xml_ns_key = ns_key
            # verify that the xml namespace will work
            if xml_ns_key is not None:
                if xml_ns is None:
                    raise ValueError('Attribute {} in class {} expects an xml namespace entry of {}, '
                                     'but xml_ns is None.'.format(attribute, cls, xml_ns_key))
                elif xml_ns_key not in xml_ns:
                    raise ValueError('Attribute {} in class {} expects an xml namespace entry of {}, '
                                     'but xml_ns does not contain this key.'.format(attribute, cls, xml_ns_key))

            if attribute in cls._set_as_attribute:
                xml_ns_key = cls._child_xml_ns_key.get(attribute, None)
                handle_attribute(attribute, xml_ns_key)
            elif attribute in cls._collections_tags:
                # it's a collection type parameter
                array_tag = cls._collections_tags[attribute]
                array = array_tag.get('array', False)
                child_tag = array_tag.get('child_tag', None)
                if array:
                    handle_single(attribute, xml_ns_key)
                elif child_tag is not None:
                    handle_list(attribute, child_tag, xml_ns_key)
                else:
                    # the metadata is broken
                    raise ValueError(
                        'Attribute {} in class {} is listed in the _collections_tags dictionary, but the '
                        '`child_tag` value is either not populated or None.'.format(attribute, cls))
            else:
                # it's a regular property
                handle_single(attribute, xml_ns_key)
        return cls.from_dict(kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        """For XML serialization, to a dom element.

        Parameters
        ----------
        doc : ElementTree.ElementTree
            The xml Document
        tag : None|str
            The tag name. Defaults to the value of `self._tag` and then the class name if unspecified.
        ns_key : None|str
            The namespace prefix. This will be recursively passed down, unless overridden by an entry in the
            _child_xml_ns_key dictionary.
        parent : None|ElementTree.Element
            The parent element. Defaults to the document root element if unspecified.
        check_validity : bool
            Check whether the element is valid before serializing, by calling :func:`is_valid`.
        strict : bool
            Only used if `check_validity = True`. In that case, if `True` then raise an
            Exception (of appropriate type) if the structure is not valid, if `False` then log a
            hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        ElementTree.Element
            The constructed dom element, already assigned to the parent element.
        """

        def serialize_attribute(node, the_tag, val, format_function, the_xml_ns_key):
            if the_xml_ns_key is None:
                node.attrib[the_tag] = format_function(val)
            else:
                node.attrib['{}:{}'.format(the_xml_ns_key, the_tag)] = format_function(val)

        def serialize_array(node, the_tag, ch_tag, val, format_function, size_attrib, the_xml_ns_key):
            if not isinstance(val, numpy.ndarray):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # _collections_tag or the descriptor at runtime
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be an array based on '
                    'the metadata in the _collections_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(val)))
            if not len(val.shape) == 1:
                # again, I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a one-dimensional numpy.ndarray, but it has shape {}'.format(
                        attribute, self.__class__.__name__, val.shape))
            if val.size == 0:
                return  # serializing an empty array is dumb

            if val.dtype.name == 'float64':
                if the_xml_ns_key is None:
                    anode = _create_new_node(doc, the_tag, parent=node)
                else:
                    anode = _create_new_node(doc, '{}:{}'.format(the_xml_ns_key, the_tag), parent=node)
                anode.attrib[size_attrib] = str(val.size)
                for i, val in enumerate(val):
                    vnode = _create_text_node(doc, ch_tag, format_function(val), parent=anode)
                    vnode.attrib['index'] = str(i) if ch_tag == 'Amplitude' else str(i+1)
            else:
                # I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a numpy.ndarray of dtype float64 or object, but it has dtype {}'.format(
                        attribute, self.__class__.__name__, val.dtype))

        def serialize_list(node, ch_tag, val, format_function, the_xml_ns_key):
            if not isinstance(val, list):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # _collections_tags or the descriptor?
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be a list based on '
                    'the metadata in the _collections_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(val)))
            if len(val) == 0:
                return  # serializing an empty list is dumb
            else:
                for entry in val:
                    serialize_plain(node, ch_tag, entry, format_function, the_xml_ns_key)

        def serialize_plain(node, field, val, format_function, the_xml_ns_key):
            # may be called not at top level - if object array or list is present
            prim_tag = '{}:{}'.format(the_xml_ns_key, field) if the_xml_ns_key is not None else field
            if isinstance(val, (Serializable, SerializableArray)):
                val.to_node(doc, field, ns_key=the_xml_ns_key, parent=node,
                            check_validity=check_validity, strict=strict)
            elif isinstance(val, ParametersCollection):
                val.to_node(doc, ns_key=the_xml_ns_key, parent=node, check_validity=check_validity, strict=strict)
            elif isinstance(val, bool):  # this must come before int, where it would evaluate as true
                _create_text_node(doc, prim_tag, 'true' if val else 'false', parent=node)
            elif isinstance(val, string_types):
                _create_text_node(doc, prim_tag, val, parent=node)
            elif isinstance(val, integer_types):
                _create_text_node(doc, prim_tag, format_function(val), parent=node)
            elif isinstance(val, float):
                _create_text_node(doc, prim_tag, format_function(val), parent=node)
            elif isinstance(val, numpy.datetime64):
                out2 = str(val)
                out2 = out2 + 'Z' if out2[-1] != 'Z' else out2
                _create_text_node(doc, prim_tag, out2, parent=node)
            elif isinstance(val, complex):
                cnode = _create_new_node(doc, prim_tag, parent=node)
                if the_xml_ns_key is None:
                    _create_text_node(doc, 'Real', format_function(val.real), parent=cnode)
                    _create_text_node(doc, 'Imag', format_function(val.imag), parent=cnode)
                else:
                    _create_text_node(doc, '{}:Real'.format(the_xml_ns_key), format_function(val.real), parent=cnode)
                    _create_text_node(doc, '{}:Imag'.format(the_xml_ns_key), format_function(val.imag), parent=cnode)
            elif isinstance(val, date):  # should never exist at present
                _create_text_node(doc, prim_tag, val.isoformat(), parent=node)
            elif isinstance(val, datetime):  # should never exist at present
                _create_text_node(doc, prim_tag, val.isoformat(sep='T'), parent=node)
            else:
                raise ValueError(
                    'An entry for class {} using tag {} is of type {}, and serialization has not '
                    'been implemented'.format(self.__class__.__name__, field, type(val)))

        if check_validity:
            if not self.is_valid():
                msg = "{} is not valid, and cannot be SAFELY serialized to XML according to " \
                      "the SICD standard.".format(self.__class__.__name__)
                if strict:
                    raise ValueError(msg)
                logging.warning(msg)
        # create the main node
        if (ns_key is not None and ns_key != 'default') and not tag.startswith(ns_key+':'):
            nod = _create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)
        else:
            nod = _create_new_node(doc, tag, parent=parent)

        # serialize the attributes
        for attribute in self._fields:
            if attribute in exclude:
                continue

            value = getattr(self, attribute)
            if value is None:
                continue
            fmt_func = self._get_formatter(attribute)
            if attribute in self._set_as_attribute:
                xml_ns_key = self._child_xml_ns_key.get(attribute, None)
                serialize_attribute(nod, attribute, value, fmt_func, xml_ns_key)
            else:
                # should we be using some namespace?
                if attribute in self._child_xml_ns_key:
                    xml_ns_key = self._child_xml_ns_key[attribute]
                else:
                    xml_ns_key = getattr(self, '_xml_ns_key', None)
                    if xml_ns_key == 'default':
                        xml_ns_key = None

                if isinstance(value, (numpy.ndarray, list)):
                    array_tag = self._collections_tags.get(attribute, None)
                    if array_tag is None:
                        raise AttributeError(
                            'The value associated with attribute {} in an instance of class {} is of type {}, '
                            'but nothing is populated in the _collection_tags dictionary.'.format(
                                attribute, self.__class__.__name__, type(value)))
                    child_tag = array_tag.get('child_tag', None)
                    if child_tag is None:
                        raise AttributeError(
                            'The value associated with attribute {} in an instance of class {} is of type {}, '
                            'but `child_tag` is not populated in the _collection_tags dictionary.'.format(
                                attribute, self.__class__.__name__, type(value)))
                    size_attribute = array_tag.get('size_attribute', 'size')
                    if isinstance(value, numpy.ndarray):
                        serialize_array(nod, attribute, child_tag, value, fmt_func, size_attribute, xml_ns_key)
                    else:
                        serialize_list(nod, child_tag, value, fmt_func, xml_ns_key)
                else:
                    serialize_plain(nod, attribute, value, fmt_func, xml_ns_key)
        return nod

    @classmethod
    def from_dict(cls, input_dict):
        """For json deserialization, from dict instance.

        Parameters
        ----------
        input_dict : dict
            Appropriate parameters dict instance for deserialization

        Returns
        -------
            Corresponding class instance
        """

        return cls(**input_dict)

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        """
        For json serialization.

        Parameters
        ----------
        check_validity : bool
            Check whether the element is valid before serializing, by calling :func:`is_valid`.
        strict : bool
            Only used if `check_validity = True`. In that case, if `True` then raise an
            Exception (of appropriate type) if the structure is not valid, if `False` then log a
            hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        OrderedDict
            dict representation of class instance appropriate for direct json serialization.
        """

        # noinspection PyUnusedLocal
        def serialize_array(ch_tag, val):
            if not len(val.shape) == 1:
                # again, I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a one-dimensional numpy.ndarray, but it has shape {}'.format(
                        attribute, self.__class__.__name__, val.shape))

            if val.size == 0:
                return []

            if val.dtype.name == 'float64':
                return [float(el) for el in val]
            else:
                # I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}. This is expected to be'
                    'a numpy.ndarray of dtype float64, but it has dtype {}'.format(
                        attribute, self.__class__.__name__, val.dtype))

        def serialize_list(ch_tag, val):
            if len(val) == 0:
                return []
            else:
                return [serialize_plain(ch_tag, entry) for entry in val]

        def serialize_plain(field, val):
            # may be called not at top level - if object array or list is present
            if isinstance(val, Serializable):
                return val.to_dict(check_validity=check_validity, strict=strict)
            elif isinstance(val, SerializableArray):
                return val.to_json_list(check_validity=check_validity, strict=strict)
            elif isinstance(val, ParametersCollection):
                return val.to_dict()
            elif isinstance(val, integer_types) or isinstance(val, string_types) or isinstance(val, float):
                return val
            elif isinstance(val, numpy.datetime64):
                out2 = str(val)
                return out2 + 'Z' if out2[-1] != 'Z' else out2
            elif isinstance(val, complex):
                return {'Real': val.real, 'Imag': val.imag}
            elif isinstance(val, date):  # probably never present
                return val.isoformat()
            elif isinstance(val, datetime):  # probably never present
                return val.isoformat(sep='T')
            else:
                raise ValueError(
                    'a entry for class {} using tag {} is of type {}, and serialization has not '
                    'been implemented'.format(self.__class__.__name__, field, type(val)))

        if check_validity:
            if not self.is_valid():
                msg = "{} is not valid, and cannot be SAFELY serialized to a dictionary valid in " \
                      "the SICD standard.".format(self.__class__.__name__)
                if strict:
                    raise ValueError(msg)
                logging.warning(msg)

        out = OrderedDict()

        for attribute in self._fields:
            if attribute in exclude:
                continue

            value = getattr(self, attribute)
            if value is None:
                continue

            if isinstance(value, (numpy.ndarray, list)):
                array_tag = self._collections_tags.get(attribute, None)
                if array_tag is None:
                    raise AttributeError(
                        'The value associated with attribute {} in an instance of class {} is of type {}, '
                        'but nothing is populated in the _collection_tags dictionary.'.format(
                            attribute, self.__class__.__name__, type(value)))
                child_tag = array_tag.get('child_tag', None)
                if child_tag is None:
                    raise AttributeError(
                        'The value associated with attribute {} in an instance of class {} is of type {}, '
                        'but `child_tag` is not populated in the _collection_tags dictionary.'.format(
                            attribute, self.__class__.__name__, type(value)))
                if isinstance(value, numpy.ndarray):
                    out[attribute] = serialize_array(child_tag, value)
                else:
                    out[attribute] = serialize_list(child_tag, value)
            else:
                out[attribute] = serialize_plain(attribute, value)
        return out

    def copy(self):
        """
        Create a deep copy.

        Returns
        -------

        """

        return self.__class__.from_dict(copy.deepcopy(self.to_dict(check_validity=False)))

    def to_xml_bytes(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        """
        Gets a bytes array, which corresponds to the xml string in utf-8 encoding,
        identified as using the namespace given by `urn` (if given).

        Parameters
        ----------
        urn : None|str|dict
            The xml namespace string or dictionary describing the xml namespace.
        tag : None|str
            The root node tag to use. If not given, then the class name will be used.
        check_validity : bool
            Check whether the element is valid before serializing, by calling :func:`is_valid`.
        strict : bool
            Only used if `check_validity = True`. In that case, if `True` then raise an
            Exception (of appropriate type) if the structure is not valid, if `False` then log a
            hopefully helpful message.

        Returns
        -------
        bytes
            bytes array from :func:`ElementTree.tostring()` call.
        """
        if tag is None:
            tag = self.__class__.__name__
        etree = ElementTree.ElementTree()
        node = self.to_node(etree, tag, ns_key=getattr(self, '_xml_ns_key', None),
                            check_validity=check_validity, strict=strict)

        if urn is None:
            pass
        elif isinstance(urn, string_types):
            node.attrib['xmlns'] = urn
        elif isinstance(urn, dict):
            for key in urn:
                node.attrib[key] = urn[key]
        else:
            raise TypeError('Expected string or dictionary of string for urn, got type {}'.format(type(urn)))
        return ElementTree.tostring(node, encoding='utf-8', method='xml')

    def to_xml_string(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        """
        Gets an xml string with utf-8 encoding, identified as using the namespace
        given by `urn` (if given).

        Parameters
        ----------
        urn : None|str|dict
            The xml namespace or dictionary describing the xml namespace.
        tag : None|str
            The root node tag to use. If not given, then the class name will be used.
        check_validity : bool
            Check whether the element is valid before serializing, by calling :func:`is_valid`.
        strict : bool
            Only used if `check_validity = True`. In that case, if `True` then raise an
            Exception (of appropriate type) if the structure is not valid, if `False` then log a
            hopefully helpful message.

        Returns
        -------
        str
            xml string from :func:`ElementTree.tostring()` call.
        """

        return self.to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict).decode('utf-8')


##########
#  Some basic collections classes


class Arrayable(object):
    """Abstract class specifying basic functionality for assigning from/to an array"""

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type object.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple

        Returns
        -------
        Arrayable
        """

        raise NotImplementedError

    def get_array(self, dtype=numpy.float64):
        """Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
        """
        raise NotImplementedError

    def __getitem__(self, item):
        return self.get_array()[item]


class SerializableArray(object):
    __slots__ = (
        '_child_tag', '_child_type', '_array', '_name', '_minimum_length',
        '_maximum_length', '_xml_ns', '_xml_ns_key')
    _default_minimum_length = 0
    _default_maximum_length = 2**32
    _set_size = True
    _size_var_name = 'size'
    _set_index = True
    _index_var_name = 'index'

    def __init__(self, coords=None, name=None, child_tag=None, child_type=None,
                 minimum_length=None, maximum_length=None, _xml_ns=None, _xml_ns_key=None):
        self._xml_ns = _xml_ns
        self._xml_ns_key = _xml_ns_key
        self._array = None
        if name is None:
            raise ValueError('The name parameter is required.')
        if not isinstance(name, string_types):
            raise TypeError(
                'The name parameter is required to be an instance of str, got {}'.format(type(name)))
        self._name = name

        if child_tag is None:
            raise ValueError('The child_tag parameter is required.')
        if not isinstance(child_tag, string_types):
            raise TypeError(
                'The child_tag parameter is required to be an instance of str, got {}'.format(type(child_tag)))
        self._child_tag = child_tag

        if child_type is None:
            raise ValueError('The child_type parameter is required.')
        if not issubclass(child_type, Serializable):
            raise TypeError('The child_type is required to be a subclass of Serializable.')
        self._child_type = child_type

        if minimum_length is None:
            self._minimum_length = self._default_minimum_length
        else:
            self._minimum_length = max(int_func(minimum_length), 0)

        if maximum_length is None:
            self._maximum_length = max(self._default_maximum_length, self._minimum_length)
        else:
            self._maximum_length = max(int_func(maximum_length), self._minimum_length)

        self.set_array(coords)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self._array[index]

    def __setitem__(self, index, value):
        if value is None:
            raise TypeError('Elements of {} must be of type {}, not None'.format(self._name, self._child_type))
        self._array[index] = _parse_serializable(value, self._name, self, self._child_type)

    def is_valid(self, recursive=False):
        """Returns the validity of this object according to the schema. This is done by inspecting that the
        array is populated.

        Parameters
        ----------
        recursive : bool
            True if we recursively check that children are also valid. This may result in verbose (i.e. noisy) logging.

        Returns
        -------
        bool
            condition for validity of this element
        """
        if self._array is None:
            logging.error(
                "Field {} has unpopulated array".format(self._name))
            return False
        if not recursive:
            return True
        valid_children = True
        for i, entry in enumerate(self._array):
            good = entry.is_valid(recursive=True)
            if not good:
                logging.error(  # I should probably do better with a stack type situation. This is traceable, at least.
                    "Issue discovered with entry {} array field {}.".format(i, self._name))
            valid_children &= good
        return valid_children

    @property
    def size(self):  # type: () -> int
        """
        int: the size of the array.
        """

        if self._array is None:
            return 0
        else:
            return self._array.size

    def get_array(self, dtype=numpy.object, **kwargs):
        """Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number|numpy.object
            numpy data type of the return.
        kwargs : keyword arguments for calls of the form child.get_array(**kwargs)

        Returns
        -------
        numpy.ndarray
            * If `dtype` in `(numpy.object`, 'object')`, then the literal array of
              child objects is returned. *Note: Beware of mutating the elements.*
            * If `dtype` has any other value, then the return value will be tried
              as `numpy.array([child.get_array(dtype=dtype, **kwargs) for child in array]`.
            * If there is any error, then `None` is returned.
        """

        if dtype in [numpy.object, 'object', numpy.dtype('object')]:
            return self._array
        else:
            # noinspection PyBroadException
            try:
                return numpy.array(
                    [child.get_array(dtype=dtype, **kwargs) for child in self._array], dtype=dtype)
            except Exception:
                return None

    def set_array(self, coords):
        """
        Sets the underlying array.

        Parameters
        ----------
        coords : numpy.ndarray|list|tuple

        Returns
        -------
        None
        """

        if coords is None:
            self._array = None
            return
        array = _parse_serializable_array(
            coords, 'coords', self, self._child_type, self._child_tag)
        if not (self._minimum_length <= array.size <= self._maximum_length):
            raise ValueError(
                'Field {} is required to be an array with {} <= length <= {}, and input of length {} '
                'was received'.format(self._name, self._minimum_length, self._maximum_length, array.size))

        self._array = array
        self._check_indices()

    def _check_indices(self):
        if not self._set_index:
            return
        for i, entry in enumerate(self._array):
            try:
                setattr(entry, self._index_var_name, i+1)
            except (AttributeError, ValueError, TypeError):
                continue

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT):
        if self.size == 0:
            return None  # nothing to be done
        if ns_key is None:
            anode = _create_new_node(doc, tag, parent=parent)
        else:
            anode = _create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)
        if self._set_size:
            anode.attrib[self._size_var_name] = str(self.size)
        for i, entry in enumerate(self._array):
            entry.to_node(doc, self._child_tag, ns_key=ns_key, parent=anode,
                          check_validity=check_validity, strict=strict)
        return anode

    @classmethod
    def from_node(cls, node, name, child_tag, child_type, **kwargs):
        return cls(coords=node, name=name, child_tag=child_tag, child_type=child_type, **kwargs)

    def to_json_list(self, check_validity=False, strict=DEFAULT_STRICT):
        """
        For json serialization.
        Parameters
        ----------
        check_validity : bool
            passed through to child_type.to_dict() method.
        strict : bool
            passed through to child_type.to_dict() method.

        Returns
        -------
        List[dict]
        """

        if self.size == 0:
            return []
        return [entry.to_dict(check_validity=check_validity, strict=strict) for entry in self._array]


class SerializableCPArray(SerializableArray):
    __slots__ = (
        '_child_tag', '_child_type', '_array', '_name', '_minimum_length',
        '_maximum_length', '_index_as_string', '_xml_ns', '_xml_ns_key')

    def __init__(self, coords=None, name=None, child_tag=None, child_type=None, _xml_ns=None, _xml_ns_key=None):
        if hasattr(child_type, '_CORNER_VALUES'):
            self._index_as_string = True
        else:
            self._index_as_string = False
        super(SerializableCPArray, self).__init__(
            coords=coords, name=name, child_tag=child_tag, child_type=child_type, _xml_ns=_xml_ns, _xml_ns_key=_xml_ns_key)
        self._minimum_length = 4
        self._maximum_length = 4

    @property
    def FRFC(self):
        if self._array is None:
            return None
        return self._array[0].get_array()

    @property
    def FRLC(self):
        if self._array is None:
            return None
        return self._array[1].get_array()

    @property
    def LRLC(self):
        if self._array is None:
            return None
        return self._array[2].get_array()

    @property
    def LRFC(self):
        if self._array is None:
            return None
        return self._array[3].get_array()

    def _check_indices(self):
        if not self._index_as_string:
            self._array[0].index = 1
            self._array[1].index = 2
            self._array[2].index = 3
            self._array[3].index = 4
        else:
            self._array[0].index = '1:FRFC'
            self._array[1].index = '2:FRLC'
            self._array[2].index = '3:LRLC'
            self._array[3].index = '4:LRFC'

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT):
        if self.size == 0:
            return None  # nothing to be done
        if ns_key is None:
            anode = _create_new_node(doc, tag, parent=parent)
        else:
            anode = _create_new_node(doc, '{}:{}'.format(ns_key, tag), parent=parent)
        for i, entry in enumerate(self._array):
            entry.to_node(doc, self._child_tag, ns_key=ns_key, parent=anode,
                          check_validity=check_validity, strict=strict)
        return anode


class ParametersCollection(object):
    __slots__ = ('_name', '_child_tag', '_dict', '_xml_ns', '_xml_ns_key')

    def __init__(self, collection=None, name=None, child_tag='Parameters', _xml_ns=None, _xml_ns_key=None):
        self._dict = None
        self._xml_ns = _xml_ns
        self._xml_ns_key = _xml_ns_key
        if name is None:
            raise ValueError('The name parameter is required.')
        if not isinstance(name, string_types):
            raise TypeError(
                'The name parameter is required to be an instance of str, got {}'.format(type(name)))
        self._name = name

        if child_tag is None:
            raise ValueError('The child_tag parameter is required.')
        if not isinstance(child_tag, string_types):
            raise TypeError(
                'The child_tag parameter is required to be an instance of str, got {}'.format(type(child_tag)))
        self._child_tag = child_tag

        self.set_collection(collection)

    def __delitem__(self, key):
        if self._dict is not None:
            del self._dict[key]

    def __getitem__(self, key):
        if self._dict is not None:
            return self._dict[key]
        raise KeyError('Dictionary does not contain key {}'.format(key))

    def __setitem__(self, name, value):
        if not isinstance(name, string_types):
            raise ValueError('Parameter name must be of type str, got {}'.format(type(name)))
        if not isinstance(value, string_types):
            raise ValueError('Parameter name must be of type str, got {}'.format(type(value)))

        if self._dict is None:
            self._dict = OrderedDict()
        self._dict[name] = value

    def get(self, key, default=None):
        if self._dict is not None:
            return self._dict.get(key, default)
        return default

    def set_collection(self, value):
        if value is None:
            self._dict = None
        else:
            self._dict = _parse_parameters_collection(value, self._name, self)

    def get_collection(self):
        return self._dict

    # noinspection PyUnusedLocal
    def to_node(self, doc, ns_key=None, parent=None, check_validity=False, strict=False):
        if self._dict is None:
            return None  # nothing to be done
        for name in self._dict:
            value = self._dict[name]
            if not isinstance(value, string_types):
                value = str(value)
            if ns_key is None:
                node = _create_text_node(doc, self._child_tag, value, parent=parent)
            else:
                node = _create_text_node(doc, '{}:{}'.format(ns_key, self._child_tag), value, parent=parent)
            node.attrib['name'] = name

    # noinspection PyUnusedLocal
    def to_dict(self, check_validity=False, strict=False):
        return copy.deepcopy(self._dict)


######
# Extra descriptor

class _SerializableArrayDescriptor(_BasicDescriptor):
    """A descriptor for properties of a list or array of specified extension of Serializable"""
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32

    def __init__(self, name, child_type, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None,
                 array_extension=SerializableArray):
        if not issubclass(array_extension, SerializableArray):
            raise TypeError('array_extension must be a subclass of SerializableArray.')
        self.child_type = child_type
        tags = tag_dict[name]
        self.array = tags.get('array', False)
        if not self.array:
            raise ValueError(
                'Attribute {} is populated in the `_collection_tags` dictionary without `array`=True. '
                'This is inconsistent with using _SerializableArrayDescriptor.'.format(name))

        self.child_tag = tags['child_tag']
        self._typ_string = 'numpy.ndarray[{}]:'.format(str(child_type).strip().split('.')[-1][:-2])
        self.array_extension = array_extension

        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int_func(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int_func(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_SerializableArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_SerializableArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, self.array_extension):
            self.data[instance] = value
        else:
            xml_ns = getattr(instance, '_xml_ns', None)
            xml_ns_key = getattr(instance, '_xml_ns_key', None)
            the_inst = self.data.get(instance, None)
            if the_inst is None:
                self.data[instance] = self.array_extension(
                    coords=value, name=self.name, child_tag=self.child_tag, child_type=self.child_type,
                    minimum_length=self.minimum_length, maximum_length=self.maximum_length, _xml_ns=xml_ns,
                    _xml_ns_key=xml_ns_key)
            else:
                the_inst.set_array(value)
