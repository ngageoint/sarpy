"""
**This module is a work in progress - python object oriented SICD structure 1.1 (2014-09-30).**

This purpose of doing it this way is to encourage effective documentation and streamlined use of the SICD information.
This provides more robustness than using structures with no built-in validation, and more flexibility than using the
rigidity of C++ based standards validation.
"""

# TODO:
#   1.) implement the necessary sicd version 0.4 & 0.5 compatibility manipulations - noted in the body.
#   2.) determine necessary and appropriate formatting issues for serialization/deserialization
#       i.) proper precision for numeric serialization
#       ii.) is there any ridiculous formatting for latitude or longitude?
#   3.) determine and implement appropriate class methods for proper functionality
#       how are things used, and what helper functions do we need?

from xml.etree import ElementTree
from collections import OrderedDict
from datetime import datetime, date
import logging
from weakref import WeakKeyDictionary
from typing import Union, List

import numpy
import numpy.polynomial.polynomial


#################
# module constants

DEFAULT_STRICT = False
"""
bool: module level default behavior for whether to handle standards compliance strictly (raise exception) or more 
    loosely (by logging a warning)
"""

#################
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

        if self.required:
            docstring = '{} {}'.format(docstring, ' **Required.**')
        else:
            docstring = '{} {}'.format(docstring, ' **Optional.**')
        self.__doc__ = docstring

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
            if self.strict:
                raise ValueError(
                    'Attribute {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
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

        if isinstance(value, str):
            # from user or json deserialization
            self.data[instance] = value
        elif isinstance(value, ElementTree.Element):
            # from XML deserialization
            self.data[instance] = _get_node_value(value)
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))


class _StringListDescriptor(_BasicDescriptor):
    """A descriptor for properties for an array type item for specified extension of string"""
    _typ_string = ':obj:`list` of :obj:`str`:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2**32

    def __init__(self, name, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None,
                 default_value=None, docstring=None):
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_StringListDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def _docstring_suffix(self):
        if self.minimum_length == self._DEFAULT_MIN_LENGTH and \
                self.maximum_length == self._DEFAULT_MAX_LENGTH:
            return ''

        lenstr = ' Must have length '
        if self.minimum_length == self._DEFAULT_MIN_LENGTH:
            lenstr += '<= {0:d}.'.format(self.maximum_length)
        elif self.maximum_length == self._DEFAULT_MAX_LENGTH:
            lenstr += '>= {0:d}.'.format(self.minimum_length)
        elif self.minimum_length == self.maximum_length:
            lenstr += ' exactly {0:d}.'.format(self.minimum_length)
        else:
            lenstr += 'in the range [{0:d}, {1:d}].'.format(self.minimum_length, self.maximum_length)
        return lenstr

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

        if isinstance(value, str):
            set_value([value, ])
        elif isinstance(value, ElementTree.Element):
            set_value([_get_node_value(value), ])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], str):
                set_value(value)
            elif isinstance(value[0], ElementTree.Element):
                set_value([_get_node_value(nod) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _StringEnumDescriptor(_BasicDescriptor):
    """A descriptor for enumerated (specified) string type"""
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
            suff += ' Default values is :code:`{}`.'.format(self.default_value)
        return suff

    def __set__(self, instance, value):
        if value is None:
            if self.default_value is not None:
                self.data[instance] = self.default_value
            else:
                super(_StringEnumDescriptor, self).__set__(instance, value)
            return

        if isinstance(value, str):
            val = value.upper()
        elif isinstance(value, ElementTree.Element):
            val = _get_node_value(value).upper()
        else:
            raise TypeError(
                'Attribute {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))

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

        if isinstance(value, ElementTree.Element):
            # from XML deserialization
            bv = self.parse_string(instance, _get_node_value(value))
        elif isinstance(value, bool):
            bv = value
        elif isinstance(value, int):
            bv = bool(value)
        elif isinstance(value, str):
            bv = self.parse_string(instance, value)
        else:
            raise ValueError('Boolean field {} of class {} cannot assign from type {}.'.format(
                self.name, instance.__class__.__name__, type(value)))
        self.data[instance] = bv

    def parse_string(self, instance, value):
        if value.lower() in ['0', 'false']:
            return False
        elif value.lower() in ['1', 'true']:
            return True
        else:
            raise ValueError(
                'Boolean field {} of class {} cannot assign from string value {}. It must be one of '
                '["0", "false", "1", "true"]'.format(self.name, instance.__class__.__name__, value))


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
            return 'Must be in the range [{0:d}, {1:d}]'.format(self.bounds[0], self.bounds[1])
        return ''

    def _in_bounds(self, value):
        return (self.bounds is None) or (self.bounds[0] <= value <= self.bounds[1])

    def __set__(self, instance, value):
        if super(_IntegerDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, ElementTree.Element):
            # from XML deserialization
            iv = int(_get_node_value(value))
        else:
            # user or json deserialization
            iv = int(value)

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

        if isinstance(value, ElementTree.Element):
            # from XML deserialization
            iv = int(_get_node_value(value))
        else:
            # user or json deserialization
            iv = int(value)

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
    _typ_string = ':obj:`list` of :obj:`int`:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2**32

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_IntegerListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.minimum_length == self._DEFAULT_MIN_LENGTH and \
                self.maximum_length == self._DEFAULT_MAX_LENGTH:
            return ''

        lenstr = ' Must have length '
        if self.minimum_length == self._DEFAULT_MIN_LENGTH:
            lenstr += '<= {0:d}.'.format(self.maximum_length)
        elif self.maximum_length == self._DEFAULT_MAX_LENGTH:
            lenstr += '>= {0:d}.'.format(self.minimum_length)
        elif self.minimum_length == self.maximum_length:
            lenstr += ' exactly {0:d}.'.format(self.minimum_length)
        else:
            lenstr += 'in the range [{0:d}, {1:d}].'.format(self.minimum_length, self.maximum_length)
        return lenstr

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

        if isinstance(value, int):
            set_value([value, ])
        elif isinstance(value, ElementTree.Element):
            set_value([int(_get_node_value(value)), ])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], int):
                set_value(value)
            elif isinstance(value[0], ElementTree.Element):
                set_value([int(_get_node_value(nod)) for nod in value])
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
            return 'Must be in the range [{}, {}]'.format(str(self.bounds[0]), str(self.bounds[1]))
        return ''

    def _in_bounds(self, value):
        return (self.bounds is None) or (self.bounds[0] <= value <= self.bounds[1])

    def __set__(self, instance, value):
        if super(_FloatDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, ElementTree.Element):
            # from XML deserialization
            iv = float(_get_node_value(value))
        else:
            # from user of json deserialization
            iv = float(value)

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


class _ComplexDescriptor(_BasicDescriptor):
    """A descriptor for complex valued properties"""
    _typ_string = 'complex:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        super(_ComplexDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):
        if super(_ComplexDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, ElementTree.Element):
            # from XML deserialization
            rnode = value.getElementsByTagName('Real')
            inode = value.getElementsByTagName('Imag')
            if len(rnode) != 1:
                raise ValueError('There must be exactly one Real component of a complex type node defined.')
            if len(inode) != 1:
                raise ValueError('There must be exactly one Imag component of a complex type node defined.')
            real = float(_get_node_value(rnode))
            imag = float(_get_node_value(inode))
            self.data[instance] = complex(re=real, im=imag)
        elif isinstance(value, dict):
            # from json deserialization
            real = None
            for key in ['re', 'real', 'Real']:
                real = value.get(key, real)
            imag = None
            for key in ['im', 'imag', 'Imag']:
                imag = value.get(key, imag)
            if real is None or imag is None:
                raise ValueError('Cannot convert dict {} to a complex number.'.format(value))
            self.data[instance] = complex(re=real, im=imag)
        else:
            # from user - this could be dumb
            self.data[instance] = complex(value)


class _FloatArrayDescriptor(_BasicDescriptor):
    """A descriptor for float array type properties"""
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2**32
    _typ_string = ':obj:`numpy.ndarray` of :obj:`dtype=float64`:'

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None,
                 docstring=None):

        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_FloatArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.minimum_length == self._DEFAULT_MIN_LENGTH and \
                self.maximum_length == self._DEFAULT_MAX_LENGTH:
            return ''

        lenstr = ' Must have length '
        if self.minimum_length == self._DEFAULT_MIN_LENGTH:
            lenstr += '<= {0:d}.'.format(self.maximum_length)
        elif self.maximum_length == self._DEFAULT_MAX_LENGTH:
            lenstr += '>= {0:d}.'.format(self.minimum_length)
        elif self.minimum_length == self.maximum_length:
            lenstr += ' exactly {0:d}.'.format(self.minimum_length)
        else:
            lenstr += 'in the range [{0:d}, {1:d}].'.format(self.minimum_length, self.maximum_length)
        return lenstr

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
            if not (len(value) == 1) and (numpy.dtype == numpy.float64):
                raise ValueError('Only one-dimensional ndarrays of dtype float64 are supported here.')
            set_value(value)
        elif isinstance(value, ElementTree.Element):
            size = int(value.getAttribute('size'))
            child_nodes = value.findall(self.child_tag)
            if len(child_nodes) != size:
                raise ValueError(
                    'Field {} of double array type functionality belonging to class {} got a ElementTree element '
                    'with size attribute {}, but has {} child nodes with tag {}.'.format(
                        self.name, instance.__class__.__name__, size, len(child_nodes), self.child_tag))
            new_value = numpy.empty((size, ), dtype=numpy.float64)
            for i, node in enumerate(new_value):
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
        if isinstance(value, ElementTree.Element):
            # from XML deserialization
            self.data[instance] = numpy.datetime64(_get_node_value(value), self.units)
        elif isinstance(value, numpy.datetime64):
            # from user
            self.data[instance] = value
        else:
            # from user or json deserialization
            self.data[instance] = numpy.datetime64(value, self.units)


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

        if isinstance(value, ElementTree.Element):
            # from XML deserialization
            val = float(_get_node_value(value))
        else:
            # from user of json deserialization
            val = float(value)

        # do modular arithmatic manipulations
        val = (val % (2*self.limit))  # NB: % and * have same precedence, so it can be super dumb
        self.data[instance] = val if val <= self.limit else val - 2*self.limit


class _SerializableDescriptor(_BasicDescriptor):
    """A descriptor for properties of a specified type assumed to be an extension of Serializable"""

    def __init__(self, name, the_type, required, strict=DEFAULT_STRICT, docstring=None):
        self.the_type = the_type
        self._typ_string = str(the_type).strip().split('.')[-1][:-2]+':'
        super(_SerializableDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_SerializableDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, self.the_type):
            self.data[instance] = value
        elif isinstance(value, dict):
            self.data[instance] = self.the_type.from_dict(value)
        elif isinstance(value, ElementTree.Element):
            self.data[instance] = self.the_type.from_node(value)
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _SerializableArrayDescriptor(_BasicDescriptor):
    """A descriptor for properties of a list or array of specified extension of Serializable"""
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2**32

    def __init__(self, name, child_type, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_type = child_type
        tags = tag_dict[name]
        self.array = tags.get('array', False)
        self.child_tag = tags['child_tag']
        if self.array:
            self._typ_string = ':obj:`numpy.ndarray` of :obj:`'+str(child_type).strip().split('.')[-1][:-2]+'`:'
        else:
            self._typ_string = ':obj:`list` of :obj:`'+str(child_type).strip().split('.')[-1][:-2]+'`:'

        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_SerializableArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.minimum_length == self._DEFAULT_MIN_LENGTH and \
                self.maximum_length == self._DEFAULT_MAX_LENGTH:
            return ''

        lenstr = ' Must have length '
        if self.minimum_length == self._DEFAULT_MIN_LENGTH:
            lenstr += '<= {0:d}.'.format(self.maximum_length)
        elif self.maximum_length == self._DEFAULT_MAX_LENGTH:
            lenstr += '>= {0:d}.'.format(self.minimum_length)
        elif self.minimum_length == self.maximum_length:
            lenstr += ' exactly {0:d}.'.format(self.minimum_length)
        else:
            lenstr += 'in the range [{0:d}, {1:d}].'.format(self.minimum_length, self.maximum_length)
        return lenstr

    def __actual_set(self, instance, value):
        if len(value) < self.minimum_length:
            msg = 'Attribute {} of class {} is of size {}, and must have size at least ' \
                  '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.error(msg)
        if len(value) > self.maximum_length:
            msg = 'Attribute {} of class {} is of size {}, and must have size no greater than ' \
                  '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.error(msg)
        self.data[instance] = value

    def __array_set(self, instance, value):
        if isinstance(value, numpy.ndarray):
            if value.dtype != numpy.object:
                raise ValueError(
                    'Attribute {} of array type functionality belonging to class {} got an ndarray of dtype {},'
                    'but contains objects, so requires dtype=numpy.object.'.format(
                        self.name, instance.__class__.__name__, value.dtype))
            elif len(value.shape) != 1:
                raise ValueError(
                    'Attribute {} of array type functionality belonging to class {} got an ndarray of shape {},'
                    'but requires a one dimensional array.'.format(
                        self.name, instance.__class__.__name__, value.shape))
            elif not isinstance(value[0], self.child_type):
                raise TypeError(
                    'Attribute {} of array type functionality belonging to class {} got an ndarray containing '
                    'first element of incompatible type {}.'.format(
                        self.name, instance.__class__.__name__, type(value[0])))
            self.__actual_set(instance, value)
        elif isinstance(value, ElementTree.Element):
            # this is the parent node from XML deserialization
            size = int(value.getAttribute('size'))
            # extract child nodes at top level
            child_nodes = value.findall(self.child_tag)
            if len(child_nodes) != size:
                raise ValueError(
                    'Attribute {} of array type functionality belonging to class {} got a ElementTree element '
                    'with size attribute {}, but has {} child nodes with tag {}.'.format(
                        self.name, instance.__class__.__name__, size, len(child_nodes), self.child_tag))
            new_value = numpy.empty((size, ), dtype=numpy.object)
            for i, entry in enumerate(child_nodes):
                new_value[i] = self.child_type.from_node(entry)
            self.__actual_set(instance, new_value)
        elif isinstance(value, list):
            # this would arrive from users or json deserialization
            if len(value) == 0:
                self.__actual_set(instance, numpy.empty((0, ), dtype=numpy.object))
            elif isinstance(value[0], self.child_type):
                self.__actual_set(instance, numpy.array(value, dtype=numpy.object))
            elif isinstance(value[0], dict):
                # NB: charming errors are possible here if something stupid has been done.
                self.__actual_set(instance, numpy.array(
                    [self.child_type.from_dict(node) for node in value], dtype=numpy.object))
            else:
                raise TypeError(
                    'Attribute {} of array type functionality belonging to class {} got a list containing first '
                    'element of incompatible type {}.'.format(self.name, instance.__class__.__name__, type(value[0])))
        else:
            raise TypeError(
                'Attribute {} of array type functionality belonging to class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))

    def __list_set(self, instance, value):
        if isinstance(value, self.child_type):
            # this is the child element
            self.__actual_set(instance, [value, ])
        elif isinstance(value, ElementTree.Element):
            # this is the child
            self.__actual_set(instance, [self.child_type.from_node(value), ])
        elif isinstance(value, list) or isinstance(value[0], self.child_type):
            if len(value) == 0:
                self.__actual_set(instance, value)
            elif isinstance(value[0], dict):
                # NB: charming errors are possible if something stupid has been done.
                self.__actual_set(instance, [self.child_type.from_dict(node) for node in value])
            elif isinstance(value[0], ElementTree.Element):
                self.__actual_set(instance, [self.child_type.from_node(node) for node in value])
            else:
                raise TypeError(
                    'Field {} of list type functionality belonging to class {} got a list containing first element of '
                    'incompatible type {}.'.format(self.name, instance.__class__.__name__, type(value[0])))
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))

    def __set__(self, instance, value):
        if super(_SerializableArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if self.array:
            self.__array_set(instance, value)
        else:
            self.__list_set(instance, value)


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
        parent = doc.getroot()
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

        for attribute in self._fields:
            try:
                setattr(self, attribute, kwargs.get(attribute, None))
            except AttributeError:
                # NB: this is included to allow for read only properties without breaking the paradigm
                #   Silently catching errors can potentially cover up REAL issues.
                pass

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
        if isinstance(entry, str):
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
                    "Class {} has missing required attribute {}".format(self.__class__.__name__, attribute))
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
                    "Class {} has requires that exactly one of the attributes {} is set, but none are "
                    "set.".format(self.__class__.__name__, collect))
                choices = False
            elif len(present) > 1:
                logging.error(
                    "Class {} has requires that no more than one of attributes {} is set, but multiple {} are "
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
            if isinstance(value, Serializable):
                return value.is_valid(recursive=True)
            return True

        valid_children = True
        for attribute in self._fields:
            val = getattr(self, attribute)
            good = True
            if isinstance(val, Serializable):
                good = check_item(val)
            elif isinstance(val, list) or (isinstance(val, numpy.ndarray) and val.dtype == numpy.object):
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
    def from_node(cls, node, kwargs=None):
        """For XML deserialization.

        Parameters
        ----------
        node : ElementTree.Element
            dom element for serialized class instance
        kwargs : None|dict
            `None` or dictionary of previously serialized attributes. For use in inheritance call, when certain
            attributes require specific deserialization.
        Returns
        -------
        Serializable
            corresponding class instance
        """

        def handle_attribute(the_tag):
            kwargs[the_tag] = node.attrib.get(the_tag, None)

        def handle_single(the_tag):
            kwargs[the_tag] = node.find(the_tag)

        def handle_list(attrib, ch_tag):
            cnodes = node.findall(ch_tag)
            if len(cnodes) > 0:
                kwargs[attrib] = cnodes

        if kwargs is None:
            kwargs = {}
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

            if attribute in cls._set_as_attribute:
                handle_attribute(attribute)
            elif attribute in cls._collections_tags:
                # it's an array type of a list type parameter
                array_tag = cls._collections_tags[attribute]
                array = array_tag.get('array', False)
                child_tag = array_tag.get('child_tag', None)
                if array:
                    handle_single(attribute)
                elif child_tag is not None:
                    # it's a list
                    handle_list(attribute, child_tag)
                else:
                    # the metadata is broken
                    raise ValueError(
                        'Attribute {} in class {} is listed in the _collections_tags dictionary, but the '
                        '"child_tags" value is either missing or None.'.format(attribute, cls))
            else:
                # it's a regular property
                handle_single(attribute)
        return cls.from_dict(kwargs)

    def to_node(self, doc, tag, parent=None, strict=DEFAULT_STRICT, exclude=()):
        """For XML serialization, to a dom element.

        Parameters
        ----------
        doc : ElementTree.ElementTree
            The xml Document
        tag : None|str
            The tag name. Defaults to the value of `self._tag` and then the class name if unspecified.
        parent : None|ElementTree.Element
            The parent element. Defaults to the document root element if unspecified.
        strict : bool
            If `True`, then raise an Exception (of appropriate type) if the structure is not valid.
            Otherwise, log a hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        ElementTree.Element
            The constructed dom element, already assigned to the parent element.
        """

        def serialize_array(node, the_tag, ch_tag, val, format_function):
            if not isinstance(val, numpy.ndarray):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # _collections_tag or the descriptor, probably at runtime
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

            if val.dtype == numpy.float64:
                anode = _create_new_node(doc, the_tag, parent=node)
                anode.attrib['size'] = str(val.size)
                for i, val in enumerate(val):
                    vnode = _create_text_node(doc, ch_tag, format_function(val), parent=anode)
                    vnode.attrib['index'] = str(i)
            elif val.dtype == numpy.object:
                anode = _create_new_node(doc, the_tag, parent=node)
                anode.attrib['size'] = str(val.size)
                for i, entry in enumerate(val):
                    if not isinstance(entry, Serializable):
                        raise TypeError(
                            'The value associated with attribute {} is an instance of class {} should be an object '
                            'array based on the standard, but entry {} is of type {} and not an instance of '
                            'Serializable'.format(attribute, self.__class__.__name__, i, type(entry)))
                    serialize_plain(anode, ch_tag, entry, format_function)
            else:
                # I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a numpy.ndarray of dtype float64 or object, but it has dtype {}'.format(
                        attribute, self.__class__.__name__, val.dtype))

        def serialize_list(node, ch_tag, val, format_function):
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
                    serialize_plain(node, ch_tag, entry, format_function)

        def serialize_plain(node, field, val, format_function):
            # may be called not at top level - if object array or list is present
            if isinstance(val, Serializable):
                val.to_node(doc, field, parent=node, strict=strict)
            elif isinstance(val, str):
                _create_text_node(doc, field, val, parent=node)
            elif isinstance(val, int) or isinstance(val, float):
                _create_text_node(doc, field, format_function(val), parent=node)
            elif isinstance(val, bool):
                _create_text_node(doc, field, 'true' if val else 'false', parent=node)
            elif isinstance(val, date):
                _create_text_node(doc, field, val.isoformat(), parent=node)
            elif isinstance(val, datetime):
                _create_text_node(doc, field, val.isoformat(sep='T'), parent=node)
            elif isinstance(val, numpy.datetime64):
                _create_text_node(doc, field, str(val), parent=node)
            elif isinstance(val, complex):
                cnode = _create_new_node(doc, field, parent=node)
                _create_text_node(doc, 'Real', format_function(val.real), parent=cnode)
                _create_text_node(doc, 'Imag', format_function(val.imag), parent=cnode)
            else:
                raise ValueError(
                    'An entry for class {} using tag {} is of type {}, and serialization has not '
                    'been implemented'.format(self.__class__.__name__, field, type(val)))

        if not self.is_valid():
            msg = "{} is not valid, and cannot be SAFELY serialized to XML according to " \
                  "the SICD standard.".format(self.__class__.__name__)
            if strict:
                raise ValueError(msg)
            logging.warning(msg)

        nod = _create_new_node(doc, tag, parent=parent)

        for attribute in self._fields:
            if attribute in exclude:
                continue

            value = getattr(self, attribute)
            if value is None:
                continue

            fmt_func = self._get_formatter(attribute)
            array_tag = self._collections_tags.get(attribute, None)
            if attribute in self._set_as_attribute:
                nod.attrib[attribute] = fmt_func(value)
            elif array_tag is not None:
                array = array_tag.get('array', False)
                child_tag = array_tag.get('child_tag', None)
                if array:
                    # this will be an numpy.ndarray
                    serialize_array(nod, attribute, child_tag, value, fmt_func)
                elif child_tag is not None:
                    # this will be a list
                    serialize_list(nod, child_tag, value, fmt_func)
                else:
                    # the metadata is broken
                    raise ValueError(
                        'Attribute {} in class {} is listed in the _collections_tags dictionary, but the '
                        '"child_tag" value is either missing or None.'.format(attribute, self.__class__.__name__))
            else:
                serialize_plain(nod, attribute, value, fmt_func)
        return nod

    @classmethod
    def from_dict(cls, input_dict):  # type: (dict) -> Serializable
        """For json deserialization, from dict instance.

        Parameters
        ----------
        input_dict : dict
            Appropriate parameters dict instance for deserialization

        Returns
        -------
        Serializable
            Corresponding class instance
        """

        return cls(**input_dict)

    def to_dict(self, strict=DEFAULT_STRICT, exclude=()):
        """For json serialization.

        Parameters
        ----------
        strict : bool
            If `True`, then raise an Exception (of appropriate type) if the structure is not valid.
            Otherwise, log a hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        OrderedDict
            dict representation of class instance appropriate for direct json serialization.
        """

        def serialize_array(ch_tag, val):
            if not isinstance(val, numpy.ndarray):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # _collections_tag or the descriptor, probably at runtime
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
                return []

            if val.dtype == numpy.float64:
                return [float(el) for el in val]
            elif val.dtype == numpy.object:
                return [serialize_plain(ch_tag, entry) for entry in val]
            else:
                # I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a numpy.ndarray of dtype float64 or object, but it has dtype {}'.format(
                        attribute, self.__class__.__name__, val.dtype))

        def serialize_list(ch_tag, val):
            if not isinstance(val, list):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # _collections_tags or the descriptor?
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be a list based on '
                    'the metadata in the _collections_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(val)))

            if len(val) == 0:
                return []
            else:
                return [serialize_plain(ch_tag, entry) for entry in val]

        def serialize_plain(field, val):
            # may be called not at top level - if object array or list is present
            if isinstance(val, Serializable):
                return val.to_dict(strict=strict)
            elif isinstance(val, (str, int, float, bool)):
                return val
            elif isinstance(val, numpy.datetime64):
                return str(val)
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

            array_tag = self._collections_tags.get(attribute, None)

            if array_tag is not None:
                array = array_tag.get('array', False)
                child_tag = array_tag.get('child_tag', None)
                if array:
                    out[attribute] = serialize_array(child_tag, value)
                elif child_tag is not None:
                    # this will be a list
                    out[attribute] = serialize_list(child_tag, value)
                else:
                    # the metadata is broken
                    raise ValueError(
                        'Attribute {} in class {} is listed in the _collections_tags dictionary, but the '
                        '"child_tags" value is either missing or None.'.format(attribute, self.__class__.__name__))
            else:
                out[attribute] = serialize_plain(attribute, value)
        return out


#############
# Basic building blocks for SICD standard

class PlainValueType(Serializable):
    """This is a basic xml building block element, and not actually specified in the SICD standard."""
    _fields = ('value', )
    _required = _fields
    # descriptor
    value = _StringDescriptor('value', _required, strict=True, docstring='The value')  # type: str

    @classmethod
    def from_node(cls, node, kwargs=None):
        return cls(value=_get_node_value(node))

    def to_node(self, doc, tag, parent=None, strict=DEFAULT_STRICT, exclude=()):
        # we have to short-circuit the super call here, because this is a really primitive element
        node = _create_text_node(doc, tag, self.value, parent=parent)
        return node


class ParameterType(PlainValueType):
    """A parameter - just a name attribute and associated value"""
    _fields = ('name', 'value')
    _required = _fields
    _set_as_attribute = ('name', )
    # descriptor
    name = _StringDescriptor(
        'name', _required, strict=True, docstring='The name.')  # type: str
    value = _StringDescriptor(
        'value', _required, strict=True, docstring='The value.')  # type: str


class XYZType(Serializable):
    """A spatial point in ECF coordinates."""
    _fields = ('X', 'Y', 'Z')
    _required = _fields
    _numeric_format = {'X': '0.8f', 'Y': '0.8f', 'Z': '0.8f'}
    # descriptors
    X = _FloatDescriptor(
        'X', _required, strict=DEFAULT_STRICT,
        docstring='The X attribute. Assumed to ECF or other, similar coordinates.')  # type: float
    Y = _FloatDescriptor(
        'Y', _required, strict=DEFAULT_STRICT,
        docstring='The Y attribute. Assumed to ECF or other, similar coordinates.')  # type: float
    Z = _FloatDescriptor(
        'Z', _required, strict=DEFAULT_STRICT,
        docstring='The Z attribute. Assumed to ECF or other, similar coordinates.')  # type: float

    def getArray(self, dtype=numpy.float64):
        """Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : numpy.dtype
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [X,Y,Z]
        """

        return numpy.array([self.X, self.Y, self.Z], dtype=dtype)


class LatLonType(Serializable):
    """A two-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon')
    _required = _fields
    _numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}
    # descriptors
    Lat = _FloatDescriptor(
        'Lat', _required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')  # type: float
    Lon = _FloatDescriptor(
        'Lon', _required, strict=DEFAULT_STRICT,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')  # type: float

    def getArray(self, order='LON', dtype=numpy.float64):
        """Gets an array representation of the data.

        Parameters
        ----------
        order : str
            Determines array order. 'LAT' yields [Lat, Lon], and anything else yields  [Lon, Lat].
        dtype : numpy.dtype
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array with appropriate entry order
        """

        if order.upper() == 'LAT':
            return numpy.array([self.Lat, self.Lon], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat], dtype=dtype)


class LatLonArrayElementType(LatLonType):
    """An geographic point in an array"""
    _fields = ('Lat', 'Lon', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    _numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}
    index = _IntegerDescriptor(
        'index', _required, strict=False, docstring="The array index")  # type: int


class LatLonRestrictionType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon')
    _required = _fields
    _numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}
    # descriptors
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, _required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')  # type: float
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, _required, strict=DEFAULT_STRICT,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')  # type: float


class LatLonHAEType(LatLonType):
    """A three-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon', 'HAE')
    _required = _fields
    _numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.8f'}
    # descriptors
    HAE = _FloatDescriptor(
        'HAE', _required, strict=DEFAULT_STRICT,
        docstring='The Height Above Ellipsoid (in meters) attribute. Assumed to be WGS-84 coordinates.')  # type: float

    def getArray(self, order='LON', dtype=numpy.float64):
        """Gets an array representation of the data.

        Parameters
        ----------
        order : str
            Determines array order. 'LAT' yields [Lat, Lon, HAE], and anything else yields  [Lon, Lat, HAE].
        dtype : numpy.dtype
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array with appropriate entry order
        """

        if order.upper() == 'LAT':
            return numpy.array([self.Lat, self.Lon, self.HAE], dtype=dtype)
        else:
            return numpy.array([self.Lon, self.Lat, self.HAE], dtype=dtype)


class LatLonHAERestrictionType(LatLonHAEType):
    _fields = ('Lat', 'Lon', 'HAE')
    _required = _fields
    """A three-dimensional geographic point in WGS-84 coordinates."""
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, _required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')  # type: float
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, _required, strict=DEFAULT_STRICT,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')  # type: float


class LatLonCornerType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', _required, strict=True, bounds=(1, 4),
        docstring='The integer index. This represents a clockwise enumeration of the rectangle vertices '
                  'wrt the frame of reference of the collector.')  # type: int


class LatLonCornerStringType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # other specific class variable
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = _StringEnumDescriptor(
        'index', _CORNER_VALUES, _required, strict=True,
        docstring="The string index.")  # type: int


class LatLonHAECornerRestrictionType(LatLonHAERestrictionType):
    """A three-dimensional geographic point in WGS-84 coordinates. Represents a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'HAE', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', _required, strict=True,
        docstring='The integer index. This represents a clockwise enumeration of the rectangle vertices '
                  'wrt the frame of reference of the collector.')  # type: int


class LatLonHAECornerStringType(LatLonHAEType):
    """A three-dimensional geographic point in WGS-84 coordinates. Represents a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'HAE', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = _StringEnumDescriptor(
        'index', _CORNER_VALUES, _required, strict=True, docstring="The string index.")  # type: int


class RowColType(Serializable):
    """A row and column attribute container - used as indices into array(s)."""
    _fields = ('Row', 'Col')
    _required = _fields
    Row = _IntegerDescriptor(
        'Row', _required, strict=DEFAULT_STRICT, docstring='The Row attribute.')  # type: int
    Col = _IntegerDescriptor(
        'Col', _required, strict=DEFAULT_STRICT, docstring='The Column attribute.')  # type: int


class RowColArrayElement(RowColType):
    """A array element row and column attribute container - used as indices into other array(s)."""
    # Note - in the SICD standard this type is listed as RowColvertexType. This is not a descriptive name
    # and has an inconsistency in camel case
    _fields = ('Row', 'Col', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The array index attribute.')  # type: int


class Poly1DType(Serializable):
    """Represents a one-variable polynomial, defined by one-dimensional coefficient array."""
    _fields = ('Coefs', 'order1')
    _required = ('Coefs', )
    _numeric_format = {'Coefs': '0.8f'}
    # other class variables
    _Coefs = None

    def __call__(self, x):
        """
        Evaluate a polynomial at points `x`. This passes `x` straight through to :func:`polyval` of
        :module:`numpy.polynomial.polynomial`.

        Parameters
        ----------
        x : numpy.ndarray
            The point(s) at which to evaluate.

        Returns
        -------
        numpy.ndarray
        """

        if self.Coefs is None:
            return None
        return numpy.polynomial.polynomial.polyval(x, self.Coefs)

    @property
    def order1(self):
        """
        int: The order1 attribute [READ ONLY]  - that is, largest exponent presented in the monomial terms of coefs.
        """

        if self.Coefs is None:
            return None
        else:
            return self.Coefs.size - 1

    @property
    def Coefs(self):
        """
        numpy.ndarray: The one-dimensional polynomial coefficient array of dtype=float64. Assignment object must be a
        one-dimensional numpy.ndarray, or naively convertible to one.
        """

        return self._Coefs

    @Coefs.setter
    def Coefs(self, value):
        if value is None:
            self._order1 = None
            self._Coefs = None
            return

        if isinstance(value, (list, tuple)):
            value = numpy.array(value, dtype=numpy.float64)

        if not isinstance(value, numpy.ndarray):
            raise ValueError(
                'Coefs for class Poly1D must be a list or numpy.ndarray. Received type {}.'.format(type(value)))
        elif len(value.shape) != 1:
            raise ValueError(
                'Coefs for class Poly1D must be one-dimensional. Received numpy.ndarray of shape {}.'.format(value.shape))
        elif not value.dtype == numpy.float64:
            raise ValueError(
                'Coefs for class Poly1D must have dtype=float64. Received numpy.ndarray of dtype {}.'.format(value.dtype))
        self._Coefs = value

    @classmethod
    def from_node(cls, node, kwargs=None):
        """For XML deserialization.

        Parameters
        ----------
        node : ElementTree.Element
            dom element for serialized class instance
        kwargs : None|dict
            `None` or dictionary of previously serialized attributes. For use in inheritance call, when certain
            attributes require specific deserialization.

        Returns
        -------
        Serializable
            corresponding class instance
        """

        order1 = int(node.attrib['order1'])
        coefs = numpy.zeros((order1+1, ), dtype=numpy.float64)
        for cnode in node.findall('Coef'):
            ind = int(cnode.attrib['exponent1'])
            val = float(_get_node_value(cnode))
            coefs[ind] = val
        return cls(Coefs=coefs)

    def to_node(self, doc, tag, parent=None, strict=DEFAULT_STRICT, exclude=()):
        """For XML serialization, to a dom element.

        Parameters
        ----------
        doc : ElementTree.ElementTree
            The xml Document
        tag : None|str
            The tag name. Defaults to the value of `self._tag` and then the class name if unspecified.
        parent : None|ElementTree.Element
            The parent element. Defaults to the document root element if unspecified.
        strict : bool
            If `True`, then raise an Exception (of appropriate type) if the structure is not valid.
            Otherwise, log a hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        ElementTree.Element
            The constructed dom element, already assigned to the parent element.
        """

        if parent is None:
            parent = doc.getroot()
        node = _create_new_node(doc, tag, parent=parent)
        if self.Coefs is None:
            return node

        node.attrib['order1'] = str(self.order1)
        fmt_func = self._get_formatter('Coef')
        for i, val in enumerate(self.Coefs):
            # if val != 0.0:  # should we serialize it sparsely?
            cnode = _create_text_node(doc, 'Coefs', fmt_func(val), parent=node)
            cnode.attrib['exponent1'] = str(i)
        return node

    def to_dict(self, strict=DEFAULT_STRICT, exclude=()):
        """For json serialization.

        Parameters
        ----------
        strict : bool
            If `True`, then raise an Exception (of appropriate type) if the structure is not valid.
            Otherwise, log a hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        OrderedDict
            dict representation of class instance appropriate for direct json serialization.
        """

        out = OrderedDict()
        out['Coefs'] = self.Coefs.tolist()
        return out


class Poly2DType(Serializable):
    """Represents a one-variable polynomial, defined by two-dimensional coefficient array."""
    _fields = ('Coefs', 'order1', 'order2')
    _required = ('Coefs', )
    _numeric_format = {'Coefs': '0.8f'}
    # other class variables
    _Coefs = None

    def __call__(self, x, y):
        """
        Evaluate a polynomial at points [`x`, `y`]. This passes `x`,`y` straight through to :func:`polyval2d` of
        :module:`numpy.polynomial.polynomial`.

        Parameters
        ----------
        x : numpy.ndarray
            The first dependent variable of point(s) at which to evaluate.
        y : numpy.ndarray
            The second dependent variable of point(s) at which to evaluate.

        Returns
        -------
        numpy.ndarray
        """

        if self.Coefs is None:
            return None
        return numpy.polynomial.polynomial.polyval2d(x, y, self.Coefs)

    @property
    def order1(self):
        """
        int: The order1 attribute [READ ONLY]  - that is, largest exponent1 presented in the monomial terms of coefs.
        """

        if self.Coefs is None:
            return None
        else:
            return self.Coefs.shape[0] - 1

    @property
    def order2(self):
        """
        int: The order1 attribute [READ ONLY]  - that is, largest exponent2 presented in the monomial terms of coefs.
        """

        if self.Coefs is None:
            return None
        else:
            return self.Coefs.shape[1] - 1

    @property
    def Coefs(self):
        """
        numpy.ndarray: The two-dimensional polynomial coefficient array of dtype=float64. Assignment object must be a
        two-dimensional numpy.ndarray, or naively convertible to one.
        """

        return self._Coefs

    @Coefs.setter
    def Coefs(self, value):
        if value is None:
            self._Coefs = None
            return

        if isinstance(value, (list, tuple)):
            value = numpy.array(value, dtype=numpy.float64)

        if not isinstance(value, numpy.ndarray):
            raise ValueError(
                'Coefs for class Poly2D must be a list or numpy.ndarray. Received type {}.'.format(type(value)))
        elif len(value.shape) != 2:
            raise ValueError(
                'Coefs for class Poly2D must be two-dimensional. Received numpy.ndarray of shape {}.'.format(value.shape))
        elif not value.dtype == numpy.float64:
            raise ValueError(
                'Coefs for class Poly2D must have dtype=float64. Received numpy.ndarray of dtype {}.'.format(value.dtype))
        self._Coefs = value

    @classmethod
    def from_node(cls, node, kwargs=None):
        """For XML deserialization.

        Parameters
        ----------
        node : ElementTree.Element
            dom element for serialized class instance
        kwargs : None|dict
            `None` or dictionary of previously serialized attributes. For use in inheritance call, when certain
            attributes require specific deserialization.

        Returns
        -------
        Serializable
            corresponding class instance
        """

        order1 = int(node.attrib['order1'])
        order2 = int(node.attrib['order2'])
        coefs = numpy.zeros((order1+1, order2+1), dtype=numpy.float64)
        for cnode in node.findall('Coef'):
            ind1 = int(cnode.attrib['exponent1'])
            ind2 = int(cnode.attrib['exponent2'])
            val = float(_get_node_value(cnode))
            coefs[ind1, ind2] = val
        return cls(Coefs=coefs)

    def to_node(self, doc, tag, parent=None, strict=DEFAULT_STRICT, exclude=()):
        """For XML serialization, to a dom element.

        Parameters
        ----------
        doc : ElementTree.ElementTree
            The xml Document
        tag : None|str
            The tag name. Defaults to the value of `self._tag` and then the class name if unspecified.
        parent : None|ElementTree.Element
            The parent element. Defaults to the document root element if unspecified.
        strict : bool
            If `True`, then raise an Exception (of appropriate type) if the structure is not valid.
            Otherwise, log a hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        ElementTree.Element
            The constructed dom element, already assigned to the parent element.
        """

        if parent is None:
            parent = doc.getroot()
        node = _create_new_node(doc, tag, parent=parent)
        if self.Coefs is None:
            return node

        node.attrib['order1'] = str(self.order1)
        node.attrib['order2'] = str(self.order2)
        fmt_func = self._get_formatter('Coefs')
        for i, val1 in enumerate(self.Coefs):
            for j, val in enumerate(val1):
                # if val != 0.0:  # should we serialize it sparsely?
                cnode = _create_text_node(doc, 'Coef', fmt_func(val), parent=node)
                cnode.attrib['exponent1'] = str(i)
                cnode.attrib['exponent2'] = str(j)
        return node

    def to_dict(self, strict=DEFAULT_STRICT, exclude=()):
        """For json serialization.

        Parameters
        ----------
        strict : bool
            If `True`, then raise an Exception (of appropriate type) if the structure is not valid.
            Otherwise, log a hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        OrderedDict
            dict representation of class instance appropriate for direct json serialization.
        """

        out = OrderedDict()
        out['Coefs'] = self.Coefs.tolist()
        return out


class XYZPolyType(Serializable):
    """
    Represents a single variable polynomial for each of `X`, `Y`, and `Z`. This gives position in ECF coordinates
    as a function of a single dependent variable.
    """

    _fields = ('X', 'Y', 'Z')
    _required = _fields
    # descriptors
    X = _SerializableDescriptor(
        'X', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='The polynomial for the X coordinate.')  # type: Poly1DType
    Y = _SerializableDescriptor(
        'Y', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='The polynomial for the Y coordinate.')  # type: Poly1DType
    Z = _SerializableDescriptor(
        'Z', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='The polynomial for the Z coordinate.')  # type: Poly1DType


class XYZPolyAttributeType(XYZPolyType):
    """
    An array element of X, Y, Z polynomials. The output of these polynomials are expected to spatial variables in
    the ECF coordinate system.
    """
    _fields = ('X', 'Y', 'Z', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The array index value.')  # type: int


class GainPhasePolyType(Serializable):
    """A container for the Gain and Phase Polygon definitions."""

    _fields = ('GainPoly', 'PhasePoly')
    _required = _fields
    # descriptors
    GainPoly = _SerializableDescriptor(
        'GainPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='One-way signal gain (in dB) as a function of X-axis direction cosine (DCX) (variable 1) '
                  'and Y-axis direction cosine (DCY) (variable 2). Gain relative to gain at DCX = 0 '
                  'and DCY = 0, so constant coefficient is always 0.0.')  # type: Poly2DType
    PhasePoly = _SerializableDescriptor(
        'GainPhasePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='One-way signal phase (in cycles) as a function of DCX (variable 1) and '
                  'DCY (variable 2). Phase relative to phase at DCX = 0 and DCY = 0, '
                  'so constant coefficient is always 0.0.')  # type: Poly2DType


class ErrorDecorrFuncType(Serializable):
    """
    This container allows parameterization of linear error decorrelation rate model.
    If `(Delta t) = |t2 – t1|`, then `CC(Delta t) = Min(1.0, Max(0.0, CC0 – DCR*(Delta t)))`.
    """

    _fields = ('CorrCoefZero', 'DecorrRate')
    _required = _fields
    _numeric_format = {'CorrCoefZero': '0.8f', 'DecorrRate': '0.8f'}
    # descriptors
    CorrCoefZero = _FloatDescriptor(
        'CorrCoefZero', _required, strict=DEFAULT_STRICT,
        docstring='Error correlation coefficient for zero time difference (CC0).')  # type: float
    DecorrRate = _FloatDescriptor(
        'DecorrRate', _required, strict=DEFAULT_STRICT,
        docstring='Error decorrelation rate. Simple linear decorrelation rate (DCR).')  # type: float


####################################################################
# Direct building blocks for SICD

#############
# CollectionInfoType section


class RadarModeType(Serializable):
    """Radar mode type container class"""
    _fields = ('ModeType', 'ModeId')
    _required = ('ModeType', )
    # other class variable
    _MODE_TYPE_VALUES = ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP')
    # descriptors
    ModeId = _StringDescriptor(
        'ModeId', _required,
        docstring='Radar imaging mode per Program Specific Implementation Document.')  # type: str
    ModeType = _StringEnumDescriptor(
        'ModeType', _MODE_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="The Radar imaging mode.")  # type: str


class CollectionInfoType(Serializable):
    """General information about the collection."""
    _collections_tags = {
        'Parameters': {'array': False, 'child_tag': 'Parameter'},
        'CountryCode': {'array': False, 'child_tag': 'CountryCode'},
    }
    _fields = (
        'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
        'RadarMode', 'Classification', 'Parameters', 'CountryCodes')
    _required = ('CollectorName', 'CoreName', 'RadarMode', 'Classification')
    # other class variable
    _COLLECT_TYPE_VALUES = ('MONOSTATIC', 'BISTATIC')
    # descriptors
    CollectorName = _StringDescriptor(
        'CollectorName', _required, strict=DEFAULT_STRICT,
        docstring='Radar platform identifier. For Bistatic collections, list the Receive platform.')  # type: str
    IlluminatorName = _StringDescriptor(
        'IlluminatorName', _required, strict=DEFAULT_STRICT,
        docstring='Radar platform identifier that provided the illumination. For Bistatic collections, '
                  'list the transmit platform.')  # type: str
    CoreName = _StringDescriptor(
        'CoreName', _required, strict=DEFAULT_STRICT,
        docstring='Collection and imaging data set identifier. Uniquely identifies imaging collections per '
                  'Program Specific Implementation Doc.')  # type: str
    CollectType = _StringEnumDescriptor(
        'CollectType', _COLLECT_TYPE_VALUES, _required,
        docstring="Collection type identifier. Monostatic collections include single platform collections with "
                  "unique transmit and receive apertures.")  # type: str
    RadarMode = _SerializableDescriptor(
        'RadarMode', RadarModeType, _required, strict=DEFAULT_STRICT,
        docstring='The radar mode.')  # type: RadarModeType
    Classification = _StringDescriptor(
        'Classification', _required, strict=DEFAULT_STRICT, default_value='UNCLASSIFIED',
        docstring='Contains the human-readable banner. Contains classification, file control and handling, '
                  'file releasing, and/or proprietary markings. Specified per Program Specific '
                  'Implementation Document.')  # type: str
    CountryCodes = _StringListDescriptor(
        'CountryCodes', _required, strict=DEFAULT_STRICT,
        docstring="List of country codes for region covered by the image.")  # type: List[str]
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Free form paramaters object list.')  # type: List[ParameterType]


###############
# ImageCreation section


class ImageCreationType(Serializable):
    """General information about the image creation."""
    _fields = ('Application', 'DateTime', 'Site', 'Profile')
    _required = ()
    # descriptors
    Application = _StringDescriptor(
        'Application', _required, strict=DEFAULT_STRICT,
        docstring='Name and version of the application used to create the image.')  # type: str
    DateTime = _DateTimeDescriptor(
        'DateTime', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Date and time the image creation application processed the image (UTC).')  # type: numpy.datetime64
    Site = _StringDescriptor(
        'Site', _required, strict=DEFAULT_STRICT,
        docstring='The creation site of this SICD product.')  # type: str
    Profile = _StringDescriptor(
        'Profile', _required, strict=DEFAULT_STRICT,
        docstring='Identifies what profile was used to create this SICD product.')  # type: str


###############
# ImageData section


class FullImageType(Serializable):
    """The full image product attributes."""
    _fields = ('NumRows', 'NumCols')
    _required = _fields
    # descriptors
    NumRows = _IntegerDescriptor(
        'NumRows', _required, strict=DEFAULT_STRICT,
        docstring='Number of rows in the original full image product. May include zero pixels.')  # type: int
    NumCols = _IntegerDescriptor(
        'NumCols', _required, strict=DEFAULT_STRICT,
        docstring='Number of columns in the original full image product. May include zero pixels.')  # type: int


class ImageDataType(Serializable):
    """The image pixel data."""

    _collections_tags = {
        'AmpTable': {'array': True, 'child_tag': 'Amplitude'},
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
    }
    _fields = (
        'PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel', 'ValidData')
    _required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')
    _numeric_format = {'AmpTable': '0.8f'}
    _PIXEL_TYPE_VALUES = ("RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I")
    # descriptors
    PixelType = _StringEnumDescriptor(
        'PixelType', _PIXEL_TYPE_VALUES, _required, strict=True,
        docstring="The PixelType attribute which specifies the interpretation of the file data.")  # type: str
    AmpTable = _FloatArrayDescriptor(
        'AmpTable', _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=256, maximum_length=256,
        docstring="The amplitude look-up table. This is required if "
                  "`PixelType == 'AMP8I_PHS8I'`")  # type: numpy.ndarray
    NumRows = _IntegerDescriptor(
        'NumRows', _required, strict=DEFAULT_STRICT,
        docstring='The number of Rows in the product. May include zero rows.')  # type: int
    NumCols = _IntegerDescriptor(
        'NumCols', _required, strict=DEFAULT_STRICT,
        docstring='The number of Columns in the product. May include zero rows.')  # type: int
    FirstRow = _IntegerDescriptor(
        'FirstRow', _required, strict=DEFAULT_STRICT,
        docstring='Global row index of the first row in the product. '
                  'Equal to 0 in full image product.')  # type: int
    FirstCol = _IntegerDescriptor(
        'FirstCol', _required, strict=DEFAULT_STRICT,
        docstring='Global column index of the first column in the product. '
                  'Equal to 0 in full image product.')  # type: int
    FullImage = _SerializableDescriptor(
        'FullImage', FullImageType, _required, strict=DEFAULT_STRICT,
        docstring='Original full image product.')  # type: FullImageType
    SCPPixel = _SerializableDescriptor(
        'SCPPixel', RowColType, _required, strict=DEFAULT_STRICT,
        docstring='Scene Center Point pixel global row and column index. Should be located near the '
                  'center of the full image.')  # type: RowColType
    ValidData = _SerializableArrayDescriptor(
        'ValidData', RowColArrayElement, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='Indicates the full image includes both valid data and some zero filled pixels. '
                  'Simple polygon encloses the valid data (may include some zero filled pixels for simplification). '
                  'Vertices in clockwise order.')  # type: Union[numpy.ndarray, List[RowColArrayElement]]

    def _basic_validity_check(self):
        condition = super(ImageDataType, self)._basic_validity_check()
        if (self.PixelType == 'AMP8I_PHS8I') and (self.AmpTable is None):
            logging.error("We have `PixelType='AMP8I_PHS8I'` and `AmpTable` is not defined for ImageDataType.")
            condition = False
        if (self.ValidData is not None) and (len(self.ValidData) < 3):
            logging.error("We have `ValidData` defined, with fewer than 3 entries.")
            condition = False
        return condition


###############
# GeoInfo section


class GeoInfoType(Serializable):
    """A geographic feature."""
    # TODO: This needs to be verified with Wade. The word document/pdf doesn't match the xsd.
    #   Is the standard really self-referential here? I find that confusing.
    _fields = ('name', 'Descriptions', 'Point', 'Line', 'Polygon')
    _required = ('name', )
    _set_as_attribute = ('name', )
    _choice = ({'required': False, 'collection': ('Point', 'Line', 'Polygon')}, )
    _collections_tags = {
        'Descriptions': {'array': False, 'child_tag': 'Desc'},
        'Line': {'array': True, 'child_tag': 'Endpoint'},
        'Polygon': {'array': True, 'child_tag': 'Vertex'}, }
    # descriptors
    name = _StringDescriptor(
        'name', _required, strict=True,
        docstring='The name.')  # type: str
    Descriptions = _SerializableArrayDescriptor(
        'Descriptions', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Descriptions of the geographic feature.')  # type: List[ParameterType]
    Point = _SerializableDescriptor(
        'Point', LatLonRestrictionType, _required, strict=DEFAULT_STRICT,
        docstring='A geographic point with WGS-84 coordinates.')  # type: LatLonRestrictionType
    Line = _SerializableArrayDescriptor(
        'Line', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='A geographic line (array) with WGS-84 coordinates.'
    )  # type: Union[numpy.ndarray, List[LatLonArrayElementType]]
    Polygon = _SerializableArrayDescriptor(
        'Polygon', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='A geographic polygon (array) with WGS-84 coordinates.'
    )  # type: Union[numpy.ndarray, List[LatLonArrayElementType]]

    @property
    def FeatureType(self):  # type: () -> Union[None, str]
        """
        str: READ ONLY attribute. Identifies the feature type among. This is determined by
        returning the (first) attribute among `Point`, `Line`, `Polygon` which is populated. None will be returned if
        none of them are populated.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return None

    def _validate_features(self):
        if self.Line is not None and self.Line.size < 2:
            logging.error('GeoInfo has a Line feature with {} points defined.'.format(self.Line.size))
            return False
        if self.Polygon is not None and self.Polygon.size < 3:
            logging.error('GeoInfo has a Polygon feature with {} points defined.'.format(self.Polygon.size))
            return False
        return True

    def _basic_validity_check(self):
        condition = super(GeoInfoType, self)._basic_validity_check()
        return condition & self._validate_features()


###############
# GeoData section


class SCPType(Serializable):
    """Scene Center Point (SCP) in full (global) image. This is the precise location."""
    _fields = ('ECF', 'LLH')
    _required = _fields  # isn't this redundant?
    ECF = _SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The ECF coordinates.')  # type: XYZType
    LLH = _SerializableDescriptor(
        'LLH', LatLonHAERestrictionType, _required, strict=DEFAULT_STRICT,
        docstring='The WGS-84 coordinates.')  # type: LatLonHAERestrictionType


class GeoDataType(Serializable):
    """Container specifying the image coverage area in geographic coordinates."""
    _fields = ('EarthModel', 'SCP', 'ImageCorners', 'ValidData', 'GeoInfos')
    _required = ('EarthModel', 'SCP', 'ImageCorners')
    _collections_tags = {
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
        'ImageCorners': {'array': True, 'child_tag': 'ICP'},
        'GeoInfos': {'array': False, 'child_tag': 'GeoInfo'},
    }
    # other class variables
    _EARTH_MODEL_VALUES = ('WGS_84', )
    # descriptors
    EarthModel = _StringEnumDescriptor(
        'EarthModel', _EARTH_MODEL_VALUES, _required, strict=True, default_value='WGS_84',
        docstring='The Earth Model.'.format(_EARTH_MODEL_VALUES))  # type: str
    SCP = _SerializableDescriptor(
        'SCP', SCPType, _required, strict=DEFAULT_STRICT,
        docstring='The Scene Center Point (SCP) in full (global) image. This is the '
                  'precise location.')  # type: SCPType
    ImageCorners = _SerializableArrayDescriptor(
        'ImageCorners', LatLonCornerStringType, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring='The geographic image corner points array. Image corners points projected to the '
                  'ground/surface level. Points may be projected to the same height as the SCP if ground/surface '
                  'height data is not available. The corner positions are approximate geographic locations and '
                  'not intended for analytical use.')  # type: Union[numpy.ndarray, List[LatLonCornerStringType]]
    ValidData = _SerializableArrayDescriptor(
        'ValidData', LatLonArrayElementType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=3,
        docstring='The full image array includes both valid data and some zero filled pixels.'
    )  # type: Union[numpy.ndarray, List[LatLonArrayElementType]]
    GeoInfos = _SerializableArrayDescriptor(
        'GeoInfos', GeoInfoType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Relevant geographic features list.')  # type: List[GeoInfoType]


###############
# GridType section


class WgtTypeType(Serializable):
    """The weight type parameters of the direction parameters"""
    _fields = ('WindowName', 'Parameters')
    _required = ('WindowName', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    WindowName = _StringDescriptor(
        'WindowName', _required, strict=DEFAULT_STRICT,
        docstring='Type of aperture weighting applied in the spatial frequency domain (Krow) to yield '
                  'the impulse response in the row direction. '
                  '*Example values - "UNIFORM", "TAYLOR", "UNKNOWN", "HAMMING"*')  # type: str
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, required=_required, strict=DEFAULT_STRICT,
        docstring='Free form parameters list.')  # type: List[ParameterType]

    # @classmethod
    # def from_node(cls, node, kwargs=None):
    #     # TODO: accommodate SICD version 0.4 WgtType definition as spaced delimited string. See sicd.py line 1074.


class DirParamType(Serializable):
    """The direction parameters container"""
    _fields = (
        'UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2', 'DeltaKCOAPoly',
        'WgtType', 'WgtFunct')
    _required = ('UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2')
    _numeric_format = {
        'SS': '0.8f', 'ImpRespWid': '0.8f', 'Sgn': '+d', 'ImpRespBW': '0.8f', 'KCtr': '0.8f',
        'DeltaK1': '0.8f', 'DeltaK2': '0.8f'}
    _collections_tags = {'WgtFunct': {'array': True, 'child_tag': 'Wgt'}}
    # descriptors
    UVectECF = _SerializableDescriptor(
        'UVectECF', XYZType, required=_required, strict=DEFAULT_STRICT,
        docstring='Unit vector in the increasing (row/col) direction (ECF) at the SCP pixel.')  # type: XYZType
    SS = _FloatDescriptor(
        'SS', _required, strict=DEFAULT_STRICT,
        docstring='Sample spacing in the increasing (row/col) direction. Precise spacing at the SCP.')  # type: float
    ImpRespWid = _FloatDescriptor(
        'ImpRespWid', _required, strict=DEFAULT_STRICT,
        docstring='Half power impulse response width in the increasing (row/col) direction. '
                  'Measured at the scene center point.')  # type: float
    Sgn = _IntegerEnumDescriptor(
        'Sgn', (1, -1), _required, strict=DEFAULT_STRICT,
        docstring='Sign for exponent in the DFT to transform the (row/col) dimension to '
                  'spatial frequency dimension.')  # type: int
    ImpRespBW = _FloatDescriptor(
        'ImpRespBW', _required, strict=DEFAULT_STRICT,
        docstring='Spatial bandwidth in (row/col) used to form the impulse response in the (row/col) direction. '
                  'Measured at the center of support for the SCP.')  # type: float
    KCtr = _FloatDescriptor(
        'KCtr', _required, strict=DEFAULT_STRICT,
        docstring='Center spatial frequency in the given dimension. '
                  'Corresponds to the zero frequency of the DFT in the given (row/col) direction.')  # type: float
    DeltaK1 = _FloatDescriptor(
        'DeltaK1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum (row/col) offset from KCtr of the spatial frequency support for the image.')  # type: float
    DeltaK2 = _FloatDescriptor(
        'DeltaK2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum (row/col) offset from KCtr of the spatial frequency support for the image.')  # type: float
    DeltaKCOAPoly = _SerializableDescriptor(
        'DeltaKCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Offset from KCtr of the center of support in the given (row/col) spatial frequency. '
                  'The polynomial is a function of image given (row/col) coordinate (variable 1) and '
                  'column coordinate (variable 2).')  # type: Poly2DType
    WgtType = _SerializableDescriptor(
        'WgtType', WgtTypeType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters describing aperture weighting type applied in the spatial frequency domain '
                  'to yield the impulse response in the given(row/col) direction.')  # type: WgtTypeType
    WgtFunct = _FloatArrayDescriptor(
        'WgtFunct', _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='Sampled aperture amplitude weighting function (array) applied to form the SCP impulse '
                  'response in the given (row/col) direction.')  # type: numpy.ndarray

    def _basic_validity_check(self):
        condition = super(DirParamType, self)._basic_validity_check()
        if (self.WgtFunct is not None) and (self.WgtFunct.size < 2):
            logging.error(
                'The WgtFunct array has been defined in DirParamType, but there are fewer than 2 entries.')
            condition = False
        return condition


class GridType(Serializable):
    """Collection grid details container"""
    _fields = ('ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col')
    _required = _fields
    _IMAGE_PLANE_VALUES = ('SLANT', 'GROUND', 'OTHER')
    _TYPE_VALUES = ('RGAZIM', 'RGZERO', 'XRGYCR', 'XCTYAT', 'PLANE')
    # descriptors
    ImagePlane = _StringEnumDescriptor(
        'ImagePlane', _IMAGE_PLANE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="Defines the type of image plane that the best describes the sample grid. Precise plane "
                  "defined by Row Direction and Column Direction unit vectors.")  # type: str
    Type = _StringEnumDescriptor(
        'Type', _TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Defines the type of spatial sampling grid represented by the image sample grid. 
        Row coordinate first, column coordinate second:

        * `RGAZIM` - Grid for a simple range, Doppler image. Also, the natural grid for images formed with the Polar 
          Format Algorithm.
        
        * `RGZERO` - A grid for images formed with the Range Migration Algorithm. Used only for imaging near closest 
          approach (i.e. near zero Doppler).
        
        * `XRGYCR` - Orthogonal slant plane grid oriented range and cross range relative to the ARP at a 
          reference time.
        
        * `XCTYAT` – Orthogonal slant plane grid with X oriented cross track.
        
        * `PLANE` – Arbitrary plane with orientation other than the specific `XRGYCR` or `XCTYAT`.
        \n\n
        """)  # type: str
    TimeCOAPoly = _SerializableDescriptor(
        'TimeCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring="*Time of Center Of Aperture* as a polynomial function of image coordinates. "
                  "The polynomial is a function of image row coordinate (variable 1) and column coordinate "
                  "(variable 2).")  # type: Poly2DType
    Row = _SerializableDescriptor(
        'Row', DirParamType, _required, strict=DEFAULT_STRICT,
        docstring="Row direction parameters.")  # type: DirParamType
    Col = _SerializableDescriptor(
        'Col', DirParamType, _required, strict=DEFAULT_STRICT,
        docstring="Column direction parameters.")  # type: DirParamType


##############
# TimelineType section


class IPPSetType(Serializable):
    """The Inter-Pulse Parameter array element container."""
    # NOTE that this is simply defined as a child class ("Set") of the TimelineType in the SICD standard
    #   Defining it at root level clarifies the documentation, and giving it a more descriptive name is
    #   appropriate.
    _fields = ('TStart', 'TEnd', 'IPPStart', 'IPPEnd', 'IPPPoly', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    TStart = _FloatDescriptor(
        'TStart', _required, strict=DEFAULT_STRICT,
        docstring='IPP start time relative to collection start time, i.e. offsets in seconds.')  # type: float
    TEnd = _FloatDescriptor(
        'TEnd', _required, strict=DEFAULT_STRICT,
        docstring='IPP end time relative to collection start time, i.e. offsets in seconds.')  # type: float
    IPPStart = _IntegerDescriptor(
        'IPPStart', _required, strict=True, docstring='Starting IPP index for the period described.')  # type: int
    IPPEnd = _IntegerDescriptor(
        'IPPEnd', _required, strict=True, docstring='Ending IPP index for the period described.')  # type: int
    IPPPoly = _SerializableDescriptor(
        'IPPPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='IPP index polynomial coefficients yield IPP index as a function of time.')  # type: Poly1DType
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The element array index.')  # type: int


class TimelineType(Serializable):
    """The details for the imaging collection timeline."""
    _fields = ('CollectStart', 'CollectDuration', 'IPP')
    _required = ('CollectStart', 'CollectDuration', )
    _collections_tags = {'IPP': {'array': True, 'child_tag': 'Set'}}
    # descriptors
    CollectStart = _DateTimeDescriptor(
        'CollectStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The collection start time. The default precision will be microseconds.')  # type: numpy.datetime64
    CollectDuration = _FloatDescriptor(
        'CollectDuration', _required, strict=DEFAULT_STRICT,
        docstring='The duration of the collection in seconds.')  # type: float
    IPP = _SerializableArrayDescriptor(
        'IPP', IPPSetType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring="The Inter-Pulse Period (IPP) parameters array.")  # type: numpy.ndarray


###################
# PositionType section


class PositionType(Serializable):
    """The details for platform and ground reference positions as a function of time since collection start."""
    _fields = ('ARPPoly', 'GRPPoly', 'TxAPCPoly', 'RcvAPC')
    _required = ('ARPPoly',)
    _collections_tags = {'RcvAPC': {'array': True, 'child_tag': 'RcvAPCPoly'}}

    # descriptors
    ARPPoly = _SerializableDescriptor(
        'ARPPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Aperture Reference Point (ARP) position polynomial in ECF as a function of elapsed '
                  'seconds since start of collection.')  # type: XYZPolyType
    GRPPoly = _SerializableDescriptor(
        'GRPPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Ground Reference Point (GRP) position polynomial in ECF as a function of elapsed '
                  'seconds since start of collection.')  # type: XYZPolyType
    TxAPCPoly = _SerializableDescriptor(
        'TxAPCPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Transmit Aperture Phase Center (APC) position polynomial in ECF as a function of '
                  'elapsed seconds since start of collection.')  # type: XYZPolyType
    RcvAPC = _SerializableArrayDescriptor(
        'RcvAPC', XYZPolyAttributeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Receive Aperture Phase Center polynomials array. '
                  'Each polynomial has output in ECF, and represents a function of elapsed seconds since start of '
                  'collection.')  # type; numpy.ndarray


##################
# RadarCollectionType section


class TxFrequencyType(Serializable):
    """The transmit frequency range"""
    _fields = ('Min', 'Max')
    _required = _fields
    # descriptors
    Min = _FloatDescriptor(
        'Min', required=_required, strict=DEFAULT_STRICT,
        docstring='The transmit minimum frequency in Hz.')  # type: float
    Max = _FloatDescriptor(
        'Max', required=_required, strict=DEFAULT_STRICT,
        docstring='The transmit maximum frequency in Hz.')  # type: float


class WaveformParametersType(Serializable):
    """Transmit and receive demodulation waveform parameters."""
    _fields = (
        'TxPulseLength', 'TxRFBandwidth', 'TxFreqStart', 'TxFMRate', 'RcvDemodType', 'RcvWindowLength',
        'ADCSampleRate', 'RcvIFBandwidth', 'RcvFreqStart', 'RcvFMRate')
    _required = ()
    # other class variables
    _DEMOD_TYPE_VALUES = ('STRETCH', 'CHIRP')
    # descriptors
    TxPulseLength = _FloatDescriptor(
        'TxPulseLength', _required, strict=DEFAULT_STRICT,
        docstring='Transmit pulse length in seconds.')  # type: float
    TxRFBandwidth = _FloatDescriptor(
        'TxRFBandwidth', _required, strict=DEFAULT_STRICT,
        docstring='Transmit RF bandwidth of the transmit pulse in Hz.')  # type: float
    TxFreqStart = _FloatDescriptor(
        'TxFreqStart', _required, strict=DEFAULT_STRICT,
        docstring='Transmit Start frequency for Linear FM waveform in Hz, may be relative '
                  'to reference frequency.')  # type: float
    TxFMRate = _FloatDescriptor(
        'TxFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Transmit FM rate for Linear FM waveform in Hz/second.')  # type: float
    RcvWindowLength = _FloatDescriptor(
        'RcvWindowLength', _required, strict=DEFAULT_STRICT,
        docstring='Receive window duration in seconds.')  # type: float
    ADCSampleRate = _FloatDescriptor(
        'ADCSampleRate', _required, strict=DEFAULT_STRICT,
        docstring='Analog-to-Digital Converter sampling rate in samples/second.')  # type: float
    RcvIFBandwidth = _FloatDescriptor(
        'RcvIFBandwidth', _required, strict=DEFAULT_STRICT,
        docstring='Receive IF bandwidth in Hz.')  # type: float
    RcvFreqStart = _FloatDescriptor(
        'RcvFreqStart', _required, strict=DEFAULT_STRICT,
        docstring='Receive demodulation start frequency in Hz, may be relative to reference frequency.')  # type: float
    RcvFMRate = _FloatDescriptor(
        'RcvFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Receive FM rate. Should be 0 if RcvDemodType = "CHIRP".')  # type: float
    RcvDemodType = _StringEnumDescriptor(
        'RcvDemodType', _DEMOD_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="Receive demodulation used when Linear FM waveform is used on transmit.")  # type: float

    def _basic_validity_check(self):
        valid = super(WaveformParametersType, self)._basic_validity_check()
        if (self.RcvDemodType == 'CHIRP') and (self.RcvFMRate != 0):
            # TODO: should we simply reset?
            logging.error(
                'In WaveformParameters, we have RcvDemodType == "CHIRP" and self.RcvFMRate non-zero.')
            valid = False
        return valid


class TxStepType(Serializable):
    """Transmit sequence step details"""
    _fields = ('WFIndex', 'TxPolarization', 'index')
    _required = ('index', )
    # other class variables
    _POLARIZATION2_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER')
    # descriptors
    WFIndex = _IntegerDescriptor(
        'WFIndex', _required, strict=DEFAULT_STRICT,
        docstring='The waveform number for this step.')  # type: int
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION2_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Transmit signal polarization for this step.')  # type: str
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT,
        docstring='The step index')  # type: int


class ChanParametersType(Serializable):
    """Transmit receive sequence step details"""
    _fields = ('TxRcvPolarization', 'RcvAPCIndex', 'index')
    _required = ('TxRcvPolarization', 'index', )
    # other class variables
    _DUAL_POLARIZATION_VALUES = (
        'V:V', 'V:H', 'H:V', 'H:H', 'RHC:RHC', 'RHC:LHC', 'LHC:RHC', 'LHC:LHC', 'OTHER', 'UNKNOWN')
    # descriptors
    TxRcvPolarization = _StringEnumDescriptor(
        'TxRcvPolarization', _DUAL_POLARIZATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Combined Transmit and Receive signal polarization for the channel.')  # type: str
    RcvAPCIndex = _IntegerDescriptor(
        'RcvAPCIndex', _required, strict=DEFAULT_STRICT,
        docstring='Index of the Receive Aperture Phase Center (Rcv APC). Only include if Receive APC position '
                  'polynomial(s) are included.')  # type: int
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The parameter index')  # type: int


class ReferencePointType(Serializable):
    """The reference point definition"""
    _fields = ('ECF', 'Line', 'Sample', 'name')
    _required = _fields
    _set_as_attribute = ('name', )
    # descriptors
    ECF = _SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The geographical coordinates for the reference point.')  # type: XYZType
    Line = _FloatDescriptor(
        'Line', _required, strict=DEFAULT_STRICT,
        docstring='The reference point line index.')  # type: float
    Sample = _FloatDescriptor(
        'Sample', _required, strict=DEFAULT_STRICT,
        docstring='The reference point sample index.')  # type: float
    name = _StringDescriptor(
        'name', _required, strict=DEFAULT_STRICT,
        docstring='The reference point name.')  # type: str


class XDirectionType(Serializable):
    """The X direction of the collect"""
    _fields = ('UVectECF', 'LineSpacing', 'NumLines', 'FirstLine')
    _required = _fields
    # descriptors
    ECF = _SerializableDescriptor(
        'UVectECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The unit vector in the X direction.')  # type: XYZType
    LineSpacing = _FloatDescriptor(
        'LineSpacing', _required, strict=DEFAULT_STRICT,
        docstring='The collection line spacing in the X direction in meters.')  # type: float
    NumLines = _IntegerDescriptor(
        'NumLines', _required, strict=DEFAULT_STRICT,
        docstring='The number of lines in the X direction.')  # type: int
    FirstLine = _IntegerDescriptor(
        'FirstLine', _required, strict=DEFAULT_STRICT,
        docstring='The first line index.')  # type: int


class YDirectionType(Serializable):
    """The Y direction of the collect"""
    _fields = ('UVectECF', 'LineSpacing', 'NumSamples', 'FirstSample')
    _required = _fields
    # descriptors
    ECF = _SerializableDescriptor(
        'UVectECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The unit vector in the Y direction.')  # type: XYZType
    LineSpacing = _FloatDescriptor(
        'LineSpacing', _required, strict=DEFAULT_STRICT,
        docstring='The collection line spacing in the Y direction in meters.')  # type: float
    NumSamples = _IntegerDescriptor(
        'NumSamples', _required, strict=DEFAULT_STRICT,
        docstring='The number of samples in the Y direction.')  # type: int
    FirstSample = _IntegerDescriptor(
        'FirstSample', _required, strict=DEFAULT_STRICT,
        docstring='The first sample index.')  # type: int


class SegmentArrayElement(Serializable):
    """The reference point definition"""
    _fields = ('StartLine', 'StartSample', 'EndLine', 'EndSample', 'Identifier', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    StartLine = _IntegerDescriptor(
        'StartLine', _required, strict=DEFAULT_STRICT,
        docstring='The starting line number of the segment.')  # type: int
    StartSample = _IntegerDescriptor(
        'StartSample', _required, strict=DEFAULT_STRICT,
        docstring='The starting sample number of the segment.')  # type: int
    EndLine = _IntegerDescriptor(
        'EndLine', _required, strict=DEFAULT_STRICT,
        docstring='The ending line number of the segment.')  # type: int
    EndSample = _IntegerDescriptor(
        'EndSample', _required, strict=DEFAULT_STRICT,
        docstring='The ending sample number of the segment.')  # type: int
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='Identifier for the segment data boundary.')
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT,
        docstring='The array index.')  # type: int


class ReferencePlaneType(Serializable):
    """The reference plane"""
    _fields = ('RefPt', 'XDir', 'YDir', 'SegmentList', 'Orientation')
    _required = ('RefPt', 'XDir', 'YDir')
    _collections_tags = {'SegmentList': {'array': True, 'child_tag': 'SegmentList'}}
    # other class variable
    _ORIENTATION_VALUES = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'ARBITRARY')
    # descriptors
    RefPt = _SerializableDescriptor(
        'RefPt', ReferencePointType, _required, strict=DEFAULT_STRICT,
        docstring='The reference point.')  # type: ReferencePointType
    XDir = _SerializableDescriptor(
        'XDir', XDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The X direction collection plane parameters.')  # type: XDirectionType
    YDir = _SerializableDescriptor(
        'YDir', YDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The Y direction collection plane parameters.')  # type: YDirectionType
    SegmentList = _SerializableArrayDescriptor(
        'SegmentList', SegmentArrayElement, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The segment array.')  # type: Union[numpy.ndarray, List[SegmentArrayElement]]
    Orientation = _StringEnumDescriptor(
        'Orientation', _ORIENTATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Describes the shadow intent of the display plane.')  # type: str


class AreaType(Serializable):
    """The collection area"""
    _fields = ('Corner', 'Plane')
    _required = ('Corner', )
    _collections_tags = {
        'Corner': {'array': False, 'child_tag': 'ACP'}, }
    # descriptors
    Corner = _SerializableArrayDescriptor(
        'Corner', LatLonHAECornerRestrictionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring='The collection area corner point definition array.')  # type: List[LatLonHAECornerRestrictionType]
    Plane = _SerializableDescriptor(
        'Plane', ReferencePlaneType, _required, strict=DEFAULT_STRICT,
        docstring='A rectangular area in a geo-located display plane.')  # type: ReferencePlaneType
    # TODO: try to construct Corner from Plane for sicd 0.5. See sicd.py line 1127.


class RadarCollectionType(Serializable):
    """The Radar Collection Type"""
    _fields = (
        'TxFrequency', 'RefFreqIndex', 'Waveform', 'TxPolarization', 'TxSequence', 'RcvChannels', 'Area', 'Parameters')
    _required = ('TxFrequency', 'TxPolarization', 'RcvChannels')
    _collections_tags = {
        'Waveform': {'array': True, 'child_tag': 'WFParameters'},
        'TxSequence': {'array': True, 'child_tag': 'TxStep'},
        'RcvChannels': {'array': True, 'child_tag': 'RcvChannels'},
        'Parameters': {'array': False, 'child_tag': 'Parameters'}}
    # other class variables
    _POLARIZATION1_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER', 'UNKNOWN', 'SEQUENCE')
    # descriptors
    TxFrequencyType = _SerializableDescriptor(
        'TxFrequency', TxFrequencyType, _required, strict=DEFAULT_STRICT,
        docstring='The transmit frequency range.')  # type: TxFrequencyType
    RefFreqIndex = _IntegerDescriptor(
        'RefFreqIndex', _required, strict=DEFAULT_STRICT,
        docstring='The reference frequency index, if applicable. if present, all RF frequency values are expressed '
                  'as offsets from a reference frequency.')  # type: int
    Waveform = _SerializableArrayDescriptor(
        'Waveform', WaveformParametersType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Transmit and receive demodulation waveform parameters.'
    )  # type: Union[numpy.ndarray, List[WaveformParametersType]]
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION1_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The transmit polarization.')  # type: str
    TxSequence = _SerializableArrayDescriptor(
        'TxSequence', TxStepType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The transmit sequence parameters array. If present, indicates the transmit signal steps through '
                  'a repeating sequence of waveforms and/or polarizations. '
                  'One step per Inter-Pulse Period.')  # type: Union[numpy.ndarray, List[TxStepType]]
    RcvChannels = _SerializableArrayDescriptor(
        'RcvChannels', ChanParametersType, _collections_tags,
        _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Receive data channel parameters.')  # type: Union[numpy.ndarray, List[ChanParametersType]]
    Area = _SerializableDescriptor(
        'Area', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='The imaged area covered by the collection.')  # type: AreaType
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='A parameters list.')  # type: List[ParameterType]

    # TODO: validate that TxPolarization issues from sicd 0.5, see scid.py line 1101.

###############
# ImageFormationType section


class RcvChanProcType(Serializable):
    """The Received Processed Channels."""
    _fields = ('NumChanProc', 'PRFScaleFactor', 'ChanIndices')
    _required = ('NumChanProc', 'ChanIndices')
    _collections_tags = {
        'ChanIndices': {'array': False, 'child_tag': 'ChanIndex'}}
    # descriptors
    NumChanProc = _IntegerDescriptor(
        'NumChanProc', _required, strict=DEFAULT_STRICT,
        docstring='Number of receive data channels processed to form the image.')  # type: int
    PRFScaleFactor = _FloatDescriptor(
        'PRFScaleFactor', _required, strict=DEFAULT_STRICT,
        docstring='Factor indicating the ratio of the effective PRF to the actual PRF.')  # type: float
    ChanIndices = _IntegerListDescriptor(  # TODO: clarify the intent of this one.
        'ChanIndices', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Index of a data channel that was processed.')  # type: List[int]


class TxFrequencyProcType(Serializable):
    """The transmit frequency range."""
    _fields = ('MinProc', 'MaxProc')
    _required = _fields
    # descriptors
    MinProc = _FloatDescriptor(
        'MinProc', _required, strict=DEFAULT_STRICT,
        docstring='The minimum transmit frequency processed to form the image, in Hz.')  # type: float
    MaxProc = _FloatDescriptor(
        'MaxProc', _required, strict=DEFAULT_STRICT,
        docstring='The maximum transmit frequency processed to form the image, in Hz.')  # type: float


class ProcessingType(Serializable):
    """The transmit frequency range"""
    _fields = ('Type', 'Applied', 'Parameters')
    _required = ('Type', 'Applied')
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    Type = _StringDescriptor(
        'Type', _required, strict=DEFAULT_STRICT,
        docstring='The processing type identifier.')  # type: str
    Applied = _BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Indicates whether the given processing type has been applied.')  # type: bool
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The parameters list.')  # type: List[ParameterType]


class DistortionType(Serializable):
    """Distortion"""
    _fields = (
        'CalibrationDate', 'A', 'F1', 'F2', 'Q1', 'Q2', 'Q3', 'Q4',
        'GainErrorA', 'GainErrorF1', 'GainErrorF2', 'PhaseErrorF1', 'PhaseErrorF2')
    _required = ('A', 'F1', 'Q1', 'Q2', 'F2', 'Q3', 'Q4')
    # descriptors
    CalibrationDate = _DateTimeDescriptor(
        'CalibrationDate', _required, strict=DEFAULT_STRICT,
        docstring='The calibration date.')
    A = _FloatDescriptor(
        'A', _required, strict=DEFAULT_STRICT,
        docstring='Absolute amplitude scale factor.')  # type: float
    # receive distorion matrix
    F1 = _ComplexDescriptor(
        'F1', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (2,2).')  # type: complex
    Q1 = _ComplexDescriptor(
        'Q1', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (1,2).')  # type: complex
    Q2 = _ComplexDescriptor(
        'Q2', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (2,1).')  # type: complex
    # transmit distortion matrix
    F2 = _ComplexDescriptor(
        'F2', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (2,2).')  # type: complex
    Q3 = _ComplexDescriptor(
        'Q3', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (2, 1).')  # type: complex
    Q4 = _ComplexDescriptor(
        'Q4', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (1, 2).')  # type: complex
    # gain estimation error
    GainErrorA = _FloatDescriptor(
        'GainErrorA', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter A.')  # type: float
    GainErrorF1 = _FloatDescriptor(
        'GainErrorF1', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter F1.')  # type: float
    GainErrorF2 = _FloatDescriptor(
        'GainErrorF2', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter F2.')  # type: float
    PhaseErrorF1 = _FloatDescriptor(
        'PhaseErrorF1', _required, strict=DEFAULT_STRICT,
        docstring='Phase estimation error standard deviation (in dB) for parameter F1.')  # type: float
    PhaseErrorF2 = _FloatDescriptor(
        'PhaseErrorF2', _required, strict=DEFAULT_STRICT,
        docstring='Phase estimation error standard deviation (in dB) for parameter F2.')  # type: float


class PolarizationCalibrationType(Serializable):
    """The polarization calibration"""
    _fields = ('DistortCorrectApplied', 'Distortion')
    _required = _fields
    # descriptors
    DistortCorrectApplied = _BooleanDescriptor(
        'DistortCorrectApplied', _required, strict=DEFAULT_STRICT,
        docstring='Indicates whether the polarization calibration has been applied.')  # type: bool
    Distortion = _SerializableDescriptor(
        'Distortion', DistortionType, _required, strict=DEFAULT_STRICT,
        docstring='The distortion parameters.')  # type: DistortionType


class ImageFormationType(Serializable):
    """The image formation process parameters."""
    _fields = (
        'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc', 'SegmentIdentifier',
        'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus', 'Processings',
        'PolarizationCalibration')
    _required = (
        'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc',
        'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus')
    _collections_tags = {'Processings': {'array': False, 'child_tag': 'Processing'}}
    # class variables
    _DUAL_POLARIZATION_VALUES = (
        'V:V', 'V:H', 'H:V', 'H:H', 'RHC:RHC', 'RHC:LHC', 'LHC:RHC', 'LHC:LHC', 'OTHER', 'UNKNOWN')
    _IMG_FORM_ALGO_VALUES = ('PFA', 'RMA', 'RGAZCOMP', 'OTHER')
    _ST_BEAM_COMP_VALUES = ('NO', 'GLOBAL', 'SV')
    _IMG_BEAM_COMP_VALUES = ('NO', 'SV')
    _AZ_AUTOFOCUS_VALUES = _ST_BEAM_COMP_VALUES
    _RG_AUTOFOCUS_VALUES = _ST_BEAM_COMP_VALUES
    # descriptors
    RcvChanProc = _SerializableDescriptor(
        'RcvChanProc', RcvChanProcType, _required, strict=DEFAULT_STRICT,
        docstring='The received processed channels.')  # type: RcvChanProcType
    TxRcvPolarizationProc = _StringEnumDescriptor(
        'TxRcvPolarizationProc', _DUAL_POLARIZATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The combined transmit/receive polarization processed to form the image.')  # type: str
    TStartProc = _FloatDescriptor(
        'TStartProc', _required, strict=DEFAULT_STRICT,
        docstring='Earliest slow time value for data processed to form the image from CollectionStart.')  # type: float
    TEndProc = _FloatDescriptor(
        'TEndProc', _required, strict=DEFAULT_STRICT,
        docstring='Latest slow time value for data processed to form the image from CollectionStart.')  # type: float
    TxFrequencyProc = _SerializableDescriptor(
        'TxFrequencyProc', TxFrequencyProcType, _required, strict=DEFAULT_STRICT,
        docstring='The range of transmit frequency processed to form the image.')  # type: TxFrequencyProcType
    SegmentIdentifier = _StringDescriptor(
        'SegmentIdentifier', _required, strict=DEFAULT_STRICT,
        docstring='Identifier that describes the image that was processed. '
                  'Must be included when SICD.RadarCollection.Area.Plane.SegmentList is included.')  # type: str
    ImageFormAlgo = _StringEnumDescriptor(
        'ImageFormAlgo', _IMG_FORM_ALGO_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        The image formation algorithm used:
        
        * `PFA` - Polar Format Algorithm
        
        * `RMA` - Range Migration (Omega-K, Chirp Scaling, Range-Doppler)
        
        * `RGAZCOMP` - Simple range, Doppler compression.
        
        """)  # type: str
    STBeamComp = _StringEnumDescriptor(
        'STBeamComp', _ST_BEAM_COMP_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Indicates if slow time beam shape compensation has been applied.
        
        * `"NO"` - No ST beam shape compensation.
        
        * `"GLOBAL"` - Global ST beam shape compensation applied.
        
        * `"SV"` - Spatially variant beam shape compensation applied.
        
        """)  # type: str
    ImageBeamComp = _StringEnumDescriptor(
        'ImageBeamComp', _IMG_BEAM_COMP_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Indicates if image domain beam shape compensation has been applied.
        
        * `"NO"` - No image domain beam shape compensation.
        
        * `"SV"` - Spatially variant image domain beam shape compensation applied.
        
        """)  # type: str
    AzAutofocus = _StringEnumDescriptor(
        'AzAutofocus', _AZ_AUTOFOCUS_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates if azimuth autofocus correction has been applied, with similar '
                  'interpretation as `STBeamComp`.')  # type: str
    RgAutofocus = _StringEnumDescriptor(
        'RgAutofocus', _RG_AUTOFOCUS_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates if range autofocus correction has been applied, with similar '
                  'interpretation as `STBeamComp`.')  # type: str
    Processings = _SerializableArrayDescriptor(
        'Processings', ProcessingType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters to describe types of specific processing that may have been applied '
                  'such as additional compensations.')  # type: List[ProcessingType]
    PolarizationCalibration = _SerializableDescriptor(
        'PolarizationCalibration', PolarizationCalibrationType, _required, strict=DEFAULT_STRICT,
        docstring='The polarization calibration details.')  # type: PolarizationCalibrationType


###############
# SCPCOAType section


class SCPCOAType(Serializable):
    """Center of Aperture (COA) for the Scene Center Point (SCP)."""
    _fields = (
        'SCPTime', 'ARPPos', 'ARPVel', 'ARPAcc', 'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAng',
        'GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng')
    _required = _fields
    # class variables
    _SIDE_OF_TRACK_VALUES = ('L', 'R')
    # descriptors
    SCPTime = _FloatDescriptor(
        'SCPTime', _required, strict=DEFAULT_STRICT,
        docstring='Center Of Aperture time for the SCP t_COA_SCP, relative to collection '
                  'start in seconds.')  # type: float
    ARPPos = _SerializableDescriptor(
        'ARPPos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Aperture position at t_COA_SCP in ECF.')  # type: XYZType
    ARPVel = _SerializableDescriptor(
        'ARPVel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP Velocity at t_COA_SCP in ECF.')  # type: XYZType
    ARPAcc = _SerializableDescriptor(
        'ARPAcc', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP Acceleration at t_COA_SCP in ECF.')  # type: XYZType
    SideOfTrack = _StringEnumDescriptor(
        'SideOfTrack', _SIDE_OF_TRACK_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Side of track.')  # type: str
    SlantRange = _FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT,
        docstring='Slant range from the ARP to the SCP in meters.')  # type: float
    GroundRange = _FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT,
        docstring='Ground Range from the ARP nadir to the SCP. Distance measured along spherical earth model '
                  'passing through the SCP in meters.')  # type: float
    DopplerConeAng = _FloatDescriptor(
        'DopplerConeAng', _required, strict=DEFAULT_STRICT,
        docstring='The Doppler Cone Angle to SCP at t_COA_SCP in degrees.')  # type: float
    GrazeAng = _FloatDescriptor(
        'GrazeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Grazing Angle between the SCP Line of Sight (LOS) and Earth Tangent Plane (ETP).')  # type: float
    IncidenceAng = _FloatDescriptor(
        'IncidenceAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Incidence Angle between the SCP LOS and ETP normal.')  # type: float
    TwistAng = _FloatDescriptor(
        'TwistAng', _required, strict=DEFAULT_STRICT, bounds=(-90., 90.),
        docstring='Angle between cross range in the ETP and cross range in the slant plane.')  # type: float
    SlopeAng = _FloatDescriptor(
        'SlopeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Slope Angle from the ETP to the slant plane at t_COA_SCP.')  # type: float
    AzimAng = _FloatDescriptor(
        'AzimAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.),
        docstring='Angle from north to the line from the SCP to the ARP Nadir at COA. Measured '
                  'clockwise in the ETP.')  # type: float
    LayoverAng = _FloatDescriptor(
        'LayoverAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.),
        docstring='Angle from north to the layover direction in the ETP at COA. Measured '
                  'clockwise in the ETP.')  # type: float


###############
# RadiometricType section


class NoiseLevelType(Serializable):
    """Noise level structure."""
    _fields = ('NoiseLevelType', 'NoisePoly')
    _required = _fields
    # class variables
    _NOISE_LEVEL_TYPE_VALUES = ('ABSOLUTE', 'RELATIVE')
    # descriptors
    NoiseLevelType = _StringEnumDescriptor(
        'NoiseLevelType', _NOISE_LEVEL_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates that the noise power polynomial yields either absolute power level or power '
                  'level relative to the SCP pixel location.')  # type: str
    NoisePoly = _SerializableDescriptor(
        'NoisePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial coefficients that yield thermal noise power (in dB) in a pixel as a function of '
                  'image row coordinate (variable 1) and column coordinate (variable 2).')  # type: Poly2DType


class RadiometricType(Serializable):
    """The radiometric calibration parameters."""
    _fields = ('NoiseLevel', 'RCSSFPoly', 'SigmaZeroSFPoly', 'BetaZeroSFPoly', 'GammaZeroSFPoly')
    _required = ()
    # descriptors
    NoiseLevel = _SerializableDescriptor(
        'NoiseLevel', NoiseLevelType, _required, strict=DEFAULT_STRICT,
        docstring='Noise level structure.')  # type: NoiseLevelType
    RCSSFPoly = _SerializableDescriptor(
        'RCSSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to RCS (sqm) '
                  'as a function of image row coordinate (variable 1) and column coordinate (variable 2). '
                  'Scale factor computed for a target at HAE = SCP_HAE.')  # type: Poly2DType
    SigmaZeroSFPoly = _SerializableDescriptor(
        'SigmaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to clutter parameter '
                  'Sigma-Zero as a function of image row coordinate (variable 1) and column coordinate (variable 2). '
                  'Scale factor computed for a clutter cell at HAE = SCP_HAE.')  # type: Poly2DType
    BetaZeroSFPoly = _SerializableDescriptor(
        'BetaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to radar brightness '
                  'or Beta-Zero as a function of image row coordinate (variable 1) and column coordinate (variable 2). '
                  'Scale factor computed for a clutter cell at HAE = SCP_HAE.')  # type: Poly2DType
    GammaZeroSFPoly = _SerializableDescriptor(
        'GammaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to clutter parameter '
                  'Gamma-Zero as a function of image row coordinate (variable 1) and column coordinate (variable 2). '
                  'Scale factor computed for a clutter cell at HAE = SCP_HAE.')  # type: Poly2DType
    # TODO: NoiseLevelType and NoisePoly used to be at this level for sicd 0.5. See sicd.py line 1176.


###############
# AntennaType section


class EBType(Serializable):
    """Electrical boresight (EB) steering directions for an electronically steered array."""
    _fields = ('DCXPoly', 'DCYPoly')
    _required = _fields
    # descriptors
    DCXPoly = _SerializableDescriptor(
        'DCXPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight steering X-axis direction cosine (DCX) as a function of '
                  'slow time (variable 1).')  # type: Poly1DType
    DCYPoly = _SerializableDescriptor(
        'DCYPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight steering Y-axis direction cosine (DCY) as a function of '
                  'slow time (variable 1).')  # type: Poly1DType


class AntParamType(Serializable):
    """The antenna parameters container."""
    _fields = (
        'XAxisPoly', 'YAxisPoly', 'FreqZero', 'EB', 'Array', 'Elem', 'GainBSPoly', 'EBFreqShift', 'MLFreqDilation')
    _required = ('XAxisPoly', 'YAxisPoly', 'FreqZero', 'Array')
    # descriptors
    XAxisPoly = _SerializableDescriptor(
        'XAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna X-Axis unit vector in ECF as a function of time (variable 1).')  # type: XYZPolyType
    YAxisPoly = _SerializableDescriptor(
        'YAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna Y-Axis unit vector in ECF as a function of time (variable 1).')  # type: XYZPolyType
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring='RF frequency (f0) used to specify the array pattern and eletrical boresite (EB) '
                  'steering direction cosines.')  # type: float
    EB = _SerializableDescriptor(
        'EB', EBType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight (EB) steering directions for an electronically steered array.')  # type: EBType
    Array = _SerializableDescriptor(
        'Array', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='Array pattern polynomials that define the shape of the mainlobe.')  # type: GainPhasePolyType
    Elem = _SerializableDescriptor(
        'Elem', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='Element array pattern polynomials for electronically steered arrays.')  # type: GainPhasePolyType
    GainBSPoly = _SerializableDescriptor(
        'GainBSPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Gain polynomial (dB) as a function of frequency for boresight (BS) at DCX = 0 and DCY = 0. '
                  'Frequency ratio `(f-f0)/f0` is the input variable (variable 1), and the constant coefficient '
                  'is always 0.0.')  # type: Poly1DType
    EBFreqShift = _BooleanDescriptor(
        'EBFreqShift', _required, strict=DEFAULT_STRICT,
        docstring="""
        Parameter indicating whether the elctronic boresite shifts with frequency for an electronically steered array.
        
        * `False` - No shift with frequency.
        
        * `True` - Shift with frequency per ideal array theory.
        
        """)  # type: bool
    MLFreqDilation = _BooleanDescriptor(
        'MLFreqDilation', _required, strict=DEFAULT_STRICT,
        docstring="""
        Parameter indicating the mainlobe (ML) width changes with frequency.
        
        * `False` - No change with frequency.
        
        * `True` - Change with frequency per ideal array theory.
        
        """)  # type: bool


class AntennaType(Serializable):
    """Parameters that describe the antenna illumination patterns during the collection."""
    _fields = ('Tx', 'Rcv', 'TwoWay')
    _required = ()
    # descriptors
    Tx = _SerializableDescriptor(
        'Tx', AntParamType, _required, strict=DEFAULT_STRICT,
        docstring='The transmit antenna parameters.')  # type: AntParamType
    Rcv = _SerializableDescriptor(
        'Rcv', AntParamType, _required, strict=DEFAULT_STRICT,
        docstring='The receive antenna parameters.')  # type: AntParamType
    TwoWay = _SerializableDescriptor(
        'TwoWay', AntParamType, _required, strict=DEFAULT_STRICT,
        docstring='The bidirectional transmit/receive antenna parameters.')  # type: AntParamType


###############
# ErrorStatisticsType section


class CompositeSCPErrorType(Serializable):
    """
    Composite error statistics for the Scene Center Point. Slant plane range (Rg) and azimuth (Az) error
    statistics. Slant plane defined at SCP COA.
    """
    _fields = ('Rg', 'Az', 'RgAz')
    _required = _fields
    # descriptors
    Rg = _FloatDescriptor(
        'Rg', _required, strict=DEFAULT_STRICT,
        docstring='Estimated range error standard deviation.')  # type: float
    Az = _FloatDescriptor(
        'Az', _required, strict=DEFAULT_STRICT,
        docstring='Estimated azimuth error standard deviation.')  # type: float
    RgAz = _FloatDescriptor(
        'RgAz', _required, strict=DEFAULT_STRICT,
        docstring='Estimated range and azimuth error correlation coefficient.')  # type: float


class CorrCoefsType(Serializable):
    """Correlation Coefficient parameters."""
    _fields = (
        'P1P2', 'P1P3', 'P1V1', 'P1V2', 'P1V3', 'P2P3', 'P2V1', 'P2V2', 'P2V3',
        'P3V1', 'P3V2', 'P3V3', 'V1V2', 'V1V3', 'V2V3')
    _required = _fields
    # descriptors
    P1P2 = _FloatDescriptor(
        'P1P2', _required, strict=DEFAULT_STRICT, docstring='P1 and P2 correlation coefficient.')  # type: float
    P1P3 = _FloatDescriptor(
        'P1P3', _required, strict=DEFAULT_STRICT, docstring='P1 and P3 correlation coefficient.')  # type: float
    P1V1 = _FloatDescriptor(
        'P1V1', _required, strict=DEFAULT_STRICT, docstring='P1 and V1 correlation coefficient.')  # type: float
    P1V2 = _FloatDescriptor(
        'P1V2', _required, strict=DEFAULT_STRICT, docstring='P1 and V2 correlation coefficient.')  # type: float
    P1V3 = _FloatDescriptor(
        'P1V3', _required, strict=DEFAULT_STRICT, docstring='P1 and V3 correlation coefficient.')  # type: float
    P2P3 = _FloatDescriptor(
        'P2P3', _required, strict=DEFAULT_STRICT, docstring='P2 and P3 correlation coefficient.')  # type: float
    P2V1 = _FloatDescriptor(
        'P2V1', _required, strict=DEFAULT_STRICT, docstring='P2 and V1 correlation coefficient.')  # type: float
    P2V2 = _FloatDescriptor(
        'P2V2', _required, strict=DEFAULT_STRICT, docstring='P2 and V2 correlation coefficient.')  # type: float
    P2V3 = _FloatDescriptor(
        'P2V3', _required, strict=DEFAULT_STRICT, docstring='P2 and V3 correlation coefficient.')  # type: float
    P3V1 = _FloatDescriptor(
        'P3V1', _required, strict=DEFAULT_STRICT, docstring='P3 and V1 correlation coefficient.')  # type: float
    P3V2 = _FloatDescriptor(
        'P3V2', _required, strict=DEFAULT_STRICT, docstring='P3 and V2 correlation coefficient.')  # type: float
    P3V3 = _FloatDescriptor(
        'P3V3', _required, strict=DEFAULT_STRICT, docstring='P3 and V3 correlation coefficient.')  # type: float
    V1V2 = _FloatDescriptor(
        'V1V2', _required, strict=DEFAULT_STRICT, docstring='V1 and V2 correlation coefficient.')  # type: float
    V1V3 = _FloatDescriptor(
        'V1V3', _required, strict=DEFAULT_STRICT, docstring='V1 and V3 correlation coefficient.')  # type: float
    V2V3 = _FloatDescriptor(
        'V2V3', _required, strict=DEFAULT_STRICT, docstring='V2 and V3 correlation coefficient.')  # type: float


class PosVelErrType(Serializable):
    """Position and velocity error statistics for the radar platform."""
    _fields = ('Frame', 'P1', 'P2', 'P3', 'V1', 'V2', 'V3', 'CorrCoefs', 'PositionDecorr')
    _required = ('Frame', 'P1', 'P2', 'P3', 'V1', 'V2', 'V3')
    # class variables
    _FRAME_VALUES = ('ECF', 'RIC_ECF', 'RIC_ECI')
    # descriptors
    Frame = _StringEnumDescriptor(
        'Frame', _FRAME_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Coordinate frame used for expressing P,V errors statistics. Note: '
                  '*RIC = Radial, In-Track, Cross-Track*, where radial is defined to be from earth center through '
                  'the platform position. ')  # type: str
    P1 = _FloatDescriptor(
        'P1', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 1 standard deviation.')  # type: float
    P2 = _FloatDescriptor(
        'P2', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 2 standard deviation.')  # type: float
    P3 = _FloatDescriptor(
        'P3', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 3 standard deviation.')  # type: float
    V1 = _FloatDescriptor(
        'V1', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 1 standard deviation.')  # type: float
    V2 = _FloatDescriptor(
        'V2', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 2 standard deviation.')  # type: float
    V3 = _FloatDescriptor(
        'V3', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 3 standard deviation.')  # type: float
    CorrCoefs = _SerializableDescriptor(
        'CorrCoefs', CorrCoefsType, _required, strict=DEFAULT_STRICT,
        docstring='Correlation Coefficient parameters.')  # type: CorrCoefsType
    PositionDecorr = _SerializableDescriptor(
        'PositionDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Platform position error decorrelation function.')  # type: ErrorDecorrFuncType


class RadarSensorErrorType(Serializable):
    """Radar sensor error statistics."""
    _fields = ('RangeBias', 'ClockFreqSF', 'TransmitFreqSF', 'RangeBiasDecorr')
    _required = ('RangeBias', )
    # descriptors
    RangeBias = _FloatDescriptor(
        'RangeBias', _required, strict=DEFAULT_STRICT,
        docstring='Range bias error standard deviation.')  # type: float
    ClockFreqSF = _FloatDescriptor(
        'ClockFreqSF', _required, strict=DEFAULT_STRICT,
        docstring='Payload clock frequency scale factor standard deviation, where SF = (Delta f)/f0.')  # type: float
    TransmitFreqSF = _FloatDescriptor(
        'TransmitFreqSF', _required, strict=DEFAULT_STRICT,
        docstring='Transmit frequency scale factor standard deviation, where SF = (Delta f)/f0.')  # type: float
    RangeBiasDecorr = _SerializableDescriptor(
        'RangeBiasDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Range bias decorrelation rate.')  # type: ErrorDecorrFuncType


class TropoErrorType(Serializable):
    """Troposphere delay error statistics."""
    _fields = ('TropoRangeVertical', 'TropoRangeSlant', 'TropoRangeDecorr')
    _required = ()
    # descriptors
    TropoRangeVertical = _FloatDescriptor(
        'TropoRangeVertical', _required, strict=DEFAULT_STRICT,
        docstring='Troposphere two-way delay error for normal incidence standard deviation. '
                  'Expressed as a range error. `(Delta R) = (Delta T) x c/2`.')  # type: float
    TropoRangeSlant = _FloatDescriptor(
        'TropoRangeSlant', _required, strict=DEFAULT_STRICT,
        docstring='Troposphere two-way delay error for the SCP line of sight at COA standard deviation. '
                  'Expressed as a range error. `(Delta R) = (Delta T) x c/2`.')  # type: float
    TropoRangeDecorr = _SerializableDescriptor(
        'TropoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Troposphere range error decorrelation function.')  # type: ErrorDecorrFuncType


class IonoErrorType(Serializable):
    """Ionosphere delay error statistics."""
    _fields = ('IonoRangeVertical', 'IonoRangeSlant', 'IonoRgRgRateCC', 'IonoRangeDecorr')
    _required = ('IonoRgRgRateCC', )
    # descriptors
    IonoRangeVertical = _FloatDescriptor(
        'IonoRangeVertical', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere two-way delay error for normal incidence standard deviation. '
                  'Expressed as a range error. `(Delta R) = (Delta T) x c/2`.')  # type: float
    IonoRangeSlant = _FloatDescriptor(
        'IonoRangeSlant', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere two-way delay rate of change error for normal incidence standard deviation. '
                  'Expressed as a range rate error. `(Delta Rdot) = (Delta Tdot) x c/2`.')  # type: float
    IonoRgRgRateCC = _FloatDescriptor(
        'IonoRgRgRateCC', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error and range rate error correlation coefficient.')  # type: float
    IonoRangeDecorr = _SerializableDescriptor(
        'IonoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error decorrelation rate.')  # type: ErrorDecorrFuncType


class ErrorComponentsType(Serializable):
    """Error statistics by components."""
    _fields = ('PosVelErr', 'RadarSensor', 'TropoError', 'IonoError')
    _required = ('PosVelErr', 'RadarSensor')
    # descriptors
    PosVelErr = _SerializableDescriptor(
        'PosVelErr', PosVelErrType, _required, strict=DEFAULT_STRICT,
        docstring='Position and velocity error statistics for the radar platform.')  # type: PosVelErrType
    RadarSensor = _SerializableDescriptor(
        'RadarSensor', RadarSensorErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Radar sensor error statistics.')  # type: RadarSensorErrorType
    TropoError = _SerializableDescriptor(
        'TropoError', TropoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Troposphere delay error statistics.')  # type: TropoErrorType
    IonoError = _SerializableDescriptor(
        'IonoError', IonoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere delay error statistics.')  # type: IonoErrorType


class ErrorStatisticsType(Serializable):
    """Parameters used to compute error statistics within the SICD sensor model."""
    _fields = ('CompositeSCP', 'Components', 'AdditionalParms')
    _required = ()
    _collections_tags = {'AdditionalParms': {'array': True, 'child_tag': 'Parameter'}}
    # descriptors
    CompositeSCP = _SerializableDescriptor(
        'CompositeSCP', CompositeSCPErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Composite error statistics for the Scene Center Point. Slant plane range (Rg) and azimuth (Az) '
                  'error statistics. Slant plane defined at SCP COA.')  # type: CompositeSCPErrorType
    Components = _SerializableDescriptor(
        'Components', ErrorComponentsType, _required, strict=DEFAULT_STRICT,
        docstring='Error statistics by components.')  # type: ErrorComponentsType
    AdditionalParms = _SerializableArrayDescriptor(
        'AdditionalParms', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Any additional paremeters.')  # type: numpy.ndarray


###############
# MatchInfoType section


class MatchCollectionType(Serializable):
    """The match collection type."""
    _fields = ('CoreName', 'MatchIndex', 'Parameters')
    _required = ('CoreName', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    CoreName = _StringDescriptor(  # TODO: is there no validator for this? Is it unstructured?
        'CoreName', _required, strict=DEFAULT_STRICT,
        docstring='Unique identifier for the match type.')  # type: str
    MatchIndex = _IntegerDescriptor(
        'MatchIndex', _required, strict=DEFAULT_STRICT,
        docstring='Collection sequence index for the match collection.')  # type: int
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The match parameters.')  # type: list


class MatchType(Serializable):
    """The is an array element for match information."""
    _fields = ('TypeId', 'CurrentIndex', 'NumMatchCollections', 'MatchCollections')
    _required = ('TypeId',)
    _collections_tags = {'MatchCollections': {'array': False, 'child_tag': 'MatchCollection'}}
    # descriptors
    TypeId = _StringDescriptor(
        'TypeId', _required, strict=DEFAULT_STRICT,
        docstring='The match type identifier. *Examples - "COHERENT" or "STEREO"*')  # type: str
    CurrentIndex = _IntegerDescriptor(
        'CurrentIndex', _required, strict=DEFAULT_STRICT,
        docstring='Collection sequence index for the current collection.')  # type: int
    MatchCollections = _SerializableArrayDescriptor(
        'MatchCollections', MatchCollectionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The match collections.')  # type: list

    @property
    def NumMatchCollections(self):
        """int: The number of match collections for this match type."""
        if self.MatchCollections is None:
            return 0
        else:
            return len(self.MatchCollections)


class MatchInfoType(Serializable):
    """The match information container."""
    _fields = ('NumMatchTypes', 'MatchTypes')
    _required = ('MatchTypes', )
    _collections_tags = {'MatchTypes': {'array': False, 'child_tag': 'MatchType'}}
    # descriptors
    MatchTypes = _SerializableArrayDescriptor(
        'MatchTypes', MatchType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The match types list.')  # type: list

    @property
    def NumMatchTypes(self):
        """int: The number of types of matched collections."""
        if self.MatchTypes is None:
            return 0
        else:
            return len(self.MatchTypes)

    # TODO: allow for sicd 0.5 version, see sicd.py line 1196.

###############
# RgAzCompType section


class RgAzCompType(Serializable):
    """Parameters included for a Range, Doppler image."""
    _fields = ('AzSF', 'KazPoly')
    _required = _fields
    # descriptors
    AzSF = _FloatDescriptor(
        'AzSF', _required, strict=DEFAULT_STRICT,
        docstring='Scale factor that scales image coordinate az = ycol (meters) to a delta cosine of the '
                  'Doppler Cone Angle at COA, (in 1/meter)')  # type: float
    KazPoly = _SerializableDescriptor(
        'KazPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields azimuth spatial frequency (Kaz = Kcol) as a function of '
                  'slow time (variable 1). Slow Time (sec) -> Azimuth spatial frequency (cycles/meter). '
                  'Time relative to collection start.')  # type: Poly1DType


###############
# PFAType section


class STDeskewType(Serializable):
    """Parameters to describe image domain ST Deskew processing."""
    _fields = ('Applied', 'STDSPhasePoly')
    _required = _fields
    # descriptors
    Applied = _BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Parameter indicating if slow time (ST) Deskew Phase function has been applied.')  # type: bool
    STDSPhasePoly = _SerializableDescriptor(
        'STDSPhasePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Slow time deskew phase function to perform the ST / Kaz shift. Two-dimensional phase (cycles) '
                  'polynomial function of image range coordinate (variable 1) and '
                  'azimuth coordinate (variable 2).')  # type: Poly2DType


class PFAType(Serializable):
    """Parameters for the Polar Formation Algorithm."""
    _fields = (
        'FPN', 'IPN', 'PolarAngRefTime', 'PolarAngPoly', 'SpatialFreqSFPoly', 'Krg1', 'Krg2', 'Kaz1', 'Kaz2',
        'StDeskew')
    _required = ('FPN', 'IPN', 'PolarAngRefTime', 'PolarAngPoly', 'SpatialFreqSFPoly', 'Krg1', 'Krg2', 'Kaz1', 'Kaz2')
    # descriptors
    FPN = _SerializableDescriptor(
        'FPN', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Focus Plane unit normal (ECF). Unit vector FPN points away from the center of '
                  'the Earth.')  # type: XYZType
    IPN = _SerializableDescriptor(
        'IPN', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Image Formation Plane unit normal (ECF). Unit vector IPN points away from the '
                  'center of the Earth.')  # type: XYZType
    PolarAngRefTime = _FloatDescriptor(
        'PolarAngRefTime', _required, strict=DEFAULT_STRICT,
        docstring='Polar image formation reference time *in seconds*. Polar Angle = 0 at the reference time. '
                  'Measured relative to collection start. Note: Reference time is typically set equal to the SCP '
                  'COA time but may be different.')  # type: float
    PolarAngPoly = _SerializableDescriptor(
        'PolarAngPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Polar Angle (radians) as function of time '
                  'relative to Collection Start.')  # type: Poly1DType
    SpatialFreqSFPoly = _SerializableDescriptor(
        'SpatialFreqSFPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields the Spatial Frequency Scale Factor (KSF) as a function of Polar '
                  'Angle. Polar Angle(radians) -> KSF (dimensionless). Used to scale RF frequency (fx, Hz) to '
                  'aperture spatial frequency (Kap, cycles/m). `Kap = fx x (2/c) x KSF`. Kap is the effective spatial '
                  'frequency in the polar aperture.')  # type: Poly1DType
    Krg1 = _FloatDescriptor(
        'Krg1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum range spatial frequency (Krg) output from the polar to rectangular '
                  'resampling.')  # type: float
    Krg2 = _FloatDescriptor(
        'Krg2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum range spatial frequency (Krg) output from the polar to rectangular '
                  'resampling.')  # type: float
    Kaz1 = _FloatDescriptor(
        'Kaz1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum azimuth spatial frequency (Kaz) output from the polar to rectangular '
                  'resampling.')  # type: float
    Kaz2 = _FloatDescriptor(
        'Kaz2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum azimuth spatial frequency (Kaz) output from the polar to rectangular '
                  'resampling.')  # type: float
    StDeskew = _SerializableDescriptor(
        'StDeskew', STDeskewType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters to describe image domain ST Deskew processing.')  # type: STDeskewType


###############
# RMAType section


class RMRefType(Serializable):
    """Range migration reference element of RMA type."""
    _fields = ('PosRef', 'VelRef', 'DopConeAngRef')
    _required = _fields
    # descriptors
    PosRef = _SerializableDescriptor(
        'PosRef', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Platform reference position (ECF) used to establish the reference slant plane.')  # type: XYZType
    VelRef = _SerializableDescriptor(
        'VelRef', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Platform reference velocity vector (ECF) used to establish the reference '
                  'slant plane.')  # type: XYZType
    DopConeAngRef = _FloatDescriptor(
        'DopConeAngRef', _required, strict=DEFAULT_STRICT,
        docstring='Reference Doppler Cone Angle in degrees.')  # type: float


class INCAType(Serializable):
    """Parameters for Imaging Near Closest Approach (INCA) image description."""
    _fields = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly', 'DopCentroidPoly', 'DopCentroidCOA')
    _required = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly')
    # descriptors
    TimeCAPoly = _SerializableDescriptor(
        'TimeCAPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Time of Closest Approach as function of '
                  'image column (azimuth) coordinate (m). Time (in seconds) relative to '
                  'collection start.')  # type: Poly1DType
    R_CA_SCP = _FloatDescriptor(
        'R_CA_SCP', _required, strict=DEFAULT_STRICT,
        docstring='Range at Closest Approach (R_CA) for the scene center point (SCP) in meters.')  # type: float
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring='RF frequency (f0) in Hz used for computing Doppler Centroid values. Typical f0 set equal '
                  'to center transmit frequency.')  # type: float
    DRateSFPoly = _SerializableDescriptor(
        'DRateSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Doppler Rate scale factor (DRSF) as a function of image '
                  'location. Yields DRSF as a function of image range coordinate (variable 1) and azimuth '
                  'coordinate (variable 2). Used to compute Doppler Rate at closest approach.')  # type: Poly2DType
    DopCentroidPoly = _SerializableDescriptor(
        'DopCentroidPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Doppler Centroid value as a function of image location (fdop_DC). '
                  'The fdop_DC is the Doppler frequency at the peak signal response. The polynomial is a function '
                  'of image range coordinate (variable 1) and azimuth coordinate (variable 2). '
                  '*Only used for Stripmap and Dynamic Stripmap collections.*')  # type: Poly2DType
    DopCentroidCOA = _BooleanDescriptor(
        'DopCentroidCOA', _required, strict=DEFAULT_STRICT,
        docstring="""Flag indicating that the COA is at the peak signal (fdop_COA = fdop_DC). `True` if Pixel COA at 
        peak signal for all pixels. `False` otherwise. *Only used for Stripmap and Dynamic Stripmap.*""")  # type: bool


class RMAType(Serializable):
    """Parameters included when the image is formed using the Range Migration Algorithm."""
    _fields = ('RMAlgoType', 'ImageType', 'RMAT', 'RMCR', 'INCA')
    _required = ('RMAlgoType', 'ImageType')
    _choice = ({'required': True, 'collection': ('RMAT', 'RMCR', 'INCA')}, )
    # class variables
    _RM_ALGO_TYPE_VALUES = ('OMEGA_K', 'CSA', 'RG_DOP')
    # descriptors
    RMAlgoType = _StringEnumDescriptor(
        'RMAlgoType', _RM_ALGO_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Identifies the type of migration algorithm used:
        
        * `OMEGA_K` - Algorithms that employ Stolt interpolation of the Kxt dimension. `Kx = (Kf^2 – Ky^2)^0.5`
        
        * `CSA` - Wave number algorithm that process two-dimensional chirp signals.
        
        * `RG_DOP` - Range-Doppler algorithms that employ RCMC in the compressed range domain.
        
        """)  # type: str
    RMAT = _SerializableDescriptor(
        'RMAT', RMRefType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for RMA with Along Track (RMAT) motion compensation.')  # type: RMRefType
    RMCR = _SerializableDescriptor(
        'RMCR', RMRefType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for RMA with Cross Range (RMCR) motion compensation.')  # type: RMRefType
    INCA = _SerializableDescriptor(
        'INCA', INCAType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for Imaging Near Closest Approach (INCA) image description.')  # type: INCAType

    @property
    def ImageType(self):  # type: () -> Union[None, str]
        """
        str: READ ONLY attribute. Identifies the specific RM image type / metadata type supplied. This is determined by
        returning the (first) attribute among `RMAT`, `RMCR`, `INCA` which is populated. None will be returned if
        none of them are populated.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return None


####################################################################
# the SICD object


class SICDType(Serializable):
    """Sensor Independent Complex Data object, containing all the relevant data to formulate products."""
    _fields = (
        'CollectionInfo', 'ImageCreation', 'ImageData', 'GeoData', 'Grid', 'Timeline', 'Position',
        'RadarCollection', 'ImageFormation', 'SCPCOA', 'Radiometric', 'Antenna', 'ErrorStatistics',
        'MatchInfo', 'RgAzComp', 'PFA', 'RMA')
    _required = (
        'CollectionInfo', 'ImageData', 'GeoData', 'Grid', 'Timeline', 'Position',
        'RadarCollection', 'ImageFormation', 'SCPCOA')
    _choice = ({'required': False, 'collection': ('RgAzComp', 'PFA', 'RMA')}, )
    # descriptors
    CollectionInfo = _SerializableDescriptor(
        'CollectionInfo', CollectionInfoType, _required, strict=DEFAULT_STRICT,
        docstring='General information about the collection.')  # type: CollectionInfoType
    ImageCreation = _SerializableDescriptor(
        'ImageCreation', ImageCreationType, _required, strict=DEFAULT_STRICT,
        docstring='General information about the image creation.')  # type: ImageCreationType
    ImageData = _SerializableDescriptor(
        'ImageData', ImageDataType, _required, strict=DEFAULT_STRICT,
        docstring='The image pixel data.')  # type: ImageDataType
    GeoData = _SerializableDescriptor(
        'GeoData', GeoDataType, _required, strict=DEFAULT_STRICT,
        docstring='The geographic coordinates of the image coverage area.')  # type: GeoDataType
    Grid = _SerializableDescriptor(
        'Grid', GridType, _required, strict=DEFAULT_STRICT,
        docstring='The image sample grid.')  # type: GridType
    Timeline = _SerializableDescriptor(
        'Timeline', TimelineType, _required, strict=DEFAULT_STRICT,
        docstring='The imaging collection time line.')  # type: TimelineType
    Position = _SerializableDescriptor(
        'Position', PositionType, _required, strict=DEFAULT_STRICT,
        docstring='The platform and ground reference point coordinates as a function of time.')  # type: PositionType
    RadarCollection = _SerializableDescriptor(
        'RadarCollection', RadarCollectionType, _required, strict=DEFAULT_STRICT,
        docstring='The radar collection information.')  # type: RadarCollectionType
    ImageFormation = _SerializableDescriptor(
        'ImageFormation', ImageFormationType, _required, strict=DEFAULT_STRICT,
        docstring='The image formation process.')  # type: ImageFormationType
    SCPCOA = _SerializableDescriptor(
        'SCPCOA', SCPCOAType, _required, strict=DEFAULT_STRICT,
        docstring='Center of Aperture (COA) for the Scene Center Point (SCP).')  # type: SCPCOAType
    Radiometric = _SerializableDescriptor(
        'Radiometric', RadiometricType, _required, strict=DEFAULT_STRICT,
        docstring='The radiometric calibration parameters.')  # type: RadiometricType
    Antenna = _SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the antenna illumination patterns during the collection.'
    )  # type: AntennaType
    ErrorStatistics = _SerializableDescriptor(
        'ErrorStatistics', ErrorStatisticsType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters used to compute error statistics within the SICD sensor model.'
    )  # type: ErrorStatisticsType
    MatchInfo = _SerializableDescriptor(
        'MatchInfo', MatchInfoType, _required, strict=DEFAULT_STRICT,
        docstring='Information about other collections that are matched to the current collection. The current '
                  'collection is the collection from which this SICD product was generated.')  # type: MatchInfoType
    RgAzComp = _SerializableDescriptor(
        'RgAzComp', RgAzCompType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters included for a Range, Doppler image.')  # type: RgAzCompType
    PFA = _SerializableDescriptor(
        'PFA', PFAType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters included when the image is formed using the Polar Formation Algorithm.')  # type: PFAType
    RMA = _SerializableDescriptor(
        'RMA', RMAType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters included when the image is formed using the Range Migration Algorithm.')  # type: RMAType

    @property
    def ImageFormType(self):  # type: () -> str
        """
        str: READ ONLY attribute. Identifies the specific image formation type supplied. This is determined by
        returning the (first) attribute among `RgAzComp`, `PFA`, `RMA` which is populated. `OTHER` will be returned if
        none of them are populated.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return 'OTHER'

    def _validate_image_segment_id(self):  # type: () -> bool
        if self.ImageFormation is None or self.RadarCollection is None:
            return False

        # get the segment identifier
        seg_id = self.ImageFormation.SegmentIdentifier
        # get the segment list
        try:
            seg_list = self.RadarCollection.Area.Plane.SegmentList
        except AttributeError:
            seg_list = None

        if seg_id is None:
            if seg_list is None:
                return True
            else:
                logging.error(
                    'ImageFormation.SegmentIdentifier is not populated, but RadarCollection.Area.Plane.SegmentList '
                    'is populated. ImageFormation.SegmentIdentifier should be set to identify the appropriate segment.')
                return False
        else:
            if seg_list is None:
                logging.error(
                    'ImageFormation.SegmentIdentifier is populated as {}, but RadarCollection.Area.Plane.SegmentList '
                    'is not populated.'.format(seg_id))
                return False
            else:
                # let's double check that seg_id is sensibly populated
                the_ids = [entry.Identifier for entry in seg_list]
                if seg_id in the_ids:
                    return True
                else:
                    logging.error(
                        'ImageFormation.SegmentIdentifier is populated as {}, but this is not one of the possible '
                        'identifiers in the RadarCollection.Area.Plane.SegmentList definition {}. '
                        'ImageFormation.SegmentIdentifier should be set to identify the '
                        'appropriate segment.'.format(seg_id, the_ids))
                    return False

    def _validate_image_form(self):  # type: () -> bool
        if self.ImageFormation is None:
            logging.error(
                'ImageFormation attribute is not populated, and ImagFormType is {}. This '
                'cannot be valid.'.format(self.ImageFormType))
            return False  # nothing more to be done.

        alg_types = []
        for alg in ['RgAzComp', 'PFA', 'RMA']:
            if getattr(self, alg) is not None:
                alg_types.append(alg)

        if len(alg_types) > 1:
            logging.error(
                'ImageFormation.ImageFormAlgo is set as {}, and multiple SICD image formation parameters {} are set. '
                'Only one image formation algorithm should be set, and ImageFormation.ImageFormAlgo '
                'should match.'.format(self.ImageFormation.ImageFormAlgo, alg_types))
            return False
        elif len(alg_types) == 0:
            if self.ImageFormation.ImageFormAlgo is None:
                # TODO: is this correct?
                logging.warning(
                    'ImageFormation.ImageFormAlgo is not set, and there is no corresponding RgAzComp, PFA, or RMA '
                    'SICD parameters. Setting it to "OTHER".'.format(self.ImageFormation.ImageFormAlgo))
                self.ImageFormation.ImageFormAlgo = 'OTHER'
                return True
            elif self.ImageFormation.ImageFormAlgo != 'OTHER':
                logging.error(
                    'No RgAzComp, PFA, or RMA SICD parameters exist, but ImageFormation.ImageFormAlgo '
                    'is set as {}.'.format(self.ImageFormation.ImageFormAlgo))
                return False
            return True
        else:
            if self.ImageFormation.ImageFormAlgo == alg_types[0].upper():
                return True
            elif self.ImageFormation.ImageFormAlgo is None:
                logging.warning(
                    'Image formation algorithm(s) {} is populated, but ImageFormation.ImageFormAlgo was not set. '
                    'ImageFormation.ImageFormAlgo has been set.'.format(alg_types[0]))
                self.ImageFormation.ImageFormAlgo = alg_types[0].upper()
                return True
            else:  # they are different values
                # TODO: is resetting it the correct decision?
                logging.warning(
                    'Only the image formation algorithm {} is populated, but ImageFormation.ImageFormAlgo '
                    'was set as {}. ImageFormation.ImageFormAlgo has been '
                    'changed.'.format(alg_types[0], self.ImageFormation.ImageFormAlgo))
                self.ImageFormation.ImageFormAlgo = alg_types[0].upper()
                return True

    def _basic_validity_check(self):
        condition = super(SICDType, self)._basic_validity_check()
        # do our image formation parameters match, as appropriate?
        condition &= self._validate_image_form()
        # does the image formation segment identifier and radar collection make sense?
        condition &= self._validate_image_segment_id()
        return condition


# TODO: properly incorporate derived fields kludgery. See sicd.py line 1261.
#  This is quite long and unmodular. This should be implemented at the proper level,
#  and then recursively called, but not until we are sure that we are done with construction.
