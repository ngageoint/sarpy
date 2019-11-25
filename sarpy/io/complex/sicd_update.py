"""
**This module is a work in progress. The eventual structure of this is yet to be determined.**

Object oriented SICD structure definition. Enabling effective documentation and streamlined use of the SICD information
is the main purpose of this approach, versus the matlab struct based effort or using the Python bindings for the C++
SIX library.

TODO:
    * flesh out docstrings from the sicd standards document
    * determine necessary and appropriate formatting issues for serialization/deserialization
    * determine and implement appropriate class methods for proper functionality
"""


from xml.dom import minidom
import numpy
from collections import OrderedDict
from datetime import datetime, date
import logging
from weakref import WeakKeyDictionary
from typing import Union

#################
# module constants
DEFAULT_STRICT = False  # type: bool
"""
bool: module level default behavior for whether to handle standards compliance strictly (raise exception) or more 
    loosely (by logging a warning)
"""


#################
# descriptor definitions - these are reusable properties that handle typing and deserialization in one place

class _BasicDescriptor(object):
    """A descriptor object for reusable properties. Note that is is required that the calling instance is hashable."""
    __typ_string = None

    def __init__(self, name, required, strict=DEFAULT_STRICT, docstring=''):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = (name in required)
        self.strict = strict
        self.__doc__ = docstring
        self._format_docstring()

    def _format_docstring(self):
        docstring = self.__doc__
        if docstring is None:
            docstring = ''

        if (self.__typ_string is not None) and (not docstring.startswith(self.__typ_string)):
            docstring = '{} {}'.format(self.__typ_string, docstring)

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
        str
            the return value
        """

        fetched = self.data.get(instance, None)
        if fetched is not None or not self.required:
            return fetched
        elif self.strict:
            raise AttributeError(
                'Required field {} of class {} is not populated.'.format(self.name, instance.__class__.__name__))
        else:
            logging.info(
                'Required field {} of class {} is not populated.'.format(self.name, instance.__class__.__name__))
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
        # which extensions SHOULD NOT implement. This is merely to enable following DRY principles.
        if value is None:
            if self.strict:
                raise ValueError(
                    'field {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            elif self.required:
                logging.info(
                    'Required field {} of class {} has been set to None.'.format(
                        self.name, instance.__class__.__name__))
            self.data[instance] = None
            return True
        # note that the remainder must be implemented in each extension
        return False  # this is probably a bad habit, but this returns something for convenience alone


class _StringDescriptor(_BasicDescriptor):
    """A descriptor for string type"""
    __typ_string = 'str:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, docstring=None):
        super(_StringDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):

        if super(_StringDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, str):
            # from user or json deserialization
            self.data[instance] = value
        elif isinstance(value, minidom.Element):
            # from XML deserialization
            self.data[instance] = _get_node_value(value)
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))


class _StringListDescriptor(_BasicDescriptor):
    """A descriptor for properties for an array type item for specified extension of string"""
    __typ_string = 'str:'
    __DEFAULT_MIN_LENGTH = 0
    __DEFAULT_MAX_LENGTH = 2**32

    def __init__(self, name, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None, docstring=None):
        self.minimum_length = self.__DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self.__DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_StringListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.minimum_length == self.__DEFAULT_MIN_LENGTH and \
                self.maximum_length == self.__DEFAULT_MAX_LENGTH:
            return ''

        lenstr = ' Must have length '
        if self.minimum_length == self.__DEFAULT_MIN_LENGTH:
            lenstr += '<= {0:d}.'.format(self.maximum_length)
        elif self.maximum_length == self.__DEFAULT_MAX_LENGTH:
            lenstr += '>= {0:d}.'.format(self.minimum_length)
        elif self.minimum_length == self.maximum_length:
            lenstr += ' exactly {0:d}.'.format(self.minimum_length)
        else:
            lenstr += 'in the range [{0:d}, {1:d}].'.format(self.minimum_length, self.maximum_length)
        return lenstr

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Field {} of class {} is a string list of size {}, and must have length at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Field {} of class {} is a string list of size {}, and must have length nu greater than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            self.data[instance] = new_value

        if super(_StringListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, str):
            set_value([value, ])
        elif isinstance(value, minidom.Element):
            set_value([_get_node_value(value), ])
        elif isinstance(value, minidom.NodeList):
            set_value([_get_node_value(nod) for nod in value])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], str):
                set_value(value)
            elif isinstance(value[0], minidom.Element):
                set_value([_get_node_value(nod) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _StringEnumDescriptor(_BasicDescriptor):
    """A descriptor for enumerated (specified) string type"""
    __typ_string = 'str:'

    def __init__(self, name, values, required, strict=DEFAULT_STRICT, docstring=None, default_value=None):
        self.values = values
        self.default_value = default_value if default_value in values else None
        super(_StringEnumDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        suff = ' Takes values in {}.'.format(self.values)
        if self.default_value is not None:
            suff += ' Default values is {}.'.format(self.default_value)
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
        elif isinstance(value, minidom.Element):
            val = _get_node_value(value).upper()
        else:
            raise TypeError(
                'field {} of class {} requires a string value.'.format(self.name, instance.__class__.__name__))

        if val in self.values:
            self.data[instance] = val
        else:
            if self.strict:
                raise ValueError(
                    'Received value {} for field {} of class {}, but values ARE REQUIRED to be one of {}'.format(
                        value, self.name, instance.__class__.__name__, self.values))
            else:
                logging.warning(
                    'Received value {} for field {} of class {}, but values ARE REQUIRED to be one of {}. '.format(
                        value, self.name, instance.__class__.__name__, self.values))
            self.data[instance] = val


class _BooleanDescriptor(_BasicDescriptor):
    """A descriptor for boolean type"""
    __typ_string = 'bool:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, docstring=None):
        super(_BooleanDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_BooleanDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, minidom.Element):
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
    __typ_string = 'int:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, bounds=None, docstring=None):
        self.bounds = bounds
        super(_IntegerDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.bounds is not None:
            return 'Must be in the range [{0:d}, {1:d}]'.format(self.bounds[0], self.bounds[1])
        return ''

    def __set__(self, instance, value):
        if super(_IntegerDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, minidom.Element):
            # from XML deserialization
            iv = int(_get_node_value(value))
        else:
            # user or json deserialization
            iv = int(value)

        if self.bounds is None or (self.bounds[0] <= iv <= self.bounds[1]):
            self.data[instance] = iv
        else:
            msg = 'Field {} of class {} is required by standard to take value between {}.'.format(
                self.name, instance.__class__.__name__, self.bounds)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.warning(msg)
            self.data[instance] = iv


class _IntegerEnumDescriptor(_BasicDescriptor):
    """A descriptor for enumerated (specified) integer type"""
    __typ_string = 'int:'

    def __init__(self, name, values, required, strict=DEFAULT_STRICT, docstring=None):
        self.values = values
        super(_IntegerEnumDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        return 'Must take one of the values in {}.'.format(self.values)

    def __set__(self, instance, value):
        if super(_IntegerEnumDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, minidom.Element):
            # from XML deserialization
            iv = int(_get_node_value(value))
        else:
            # user or json deserialization
            iv = int(value)

        if iv in self.values:
            self.data[instance] = iv
        else:
            if self.strict:
                raise ValueError(
                    'field {} of class {} must take value in {}'.format(
                        self.name, instance.__class__.__name__, self.values))
            else:
                logging.info(
                    'Required field {} of class {} is required to take value in {}.'.format(
                        self.name, instance.__class__.__name__, self.values))
            self.data[instance] = iv


class _IntegerListDescriptor(_BasicDescriptor):
    """A descriptor for integer list type properties"""
    __typ_string = 'int:'
    __DEFAULT_MIN_LENGTH = 0
    __DEFAULT_MAX_LENGTH = 2**32

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self.__DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self.__DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_IntegerListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.minimum_length == self.__DEFAULT_MIN_LENGTH and \
                self.maximum_length == self.__DEFAULT_MAX_LENGTH:
            return ''

        lenstr = ' Must have length '
        if self.minimum_length == self.__DEFAULT_MIN_LENGTH:
            lenstr += '<= {0:d}.'.format(self.maximum_length)
        elif self.maximum_length == self.__DEFAULT_MAX_LENGTH:
            lenstr += '>= {0:d}.'.format(self.minimum_length)
        elif self.minimum_length == self.maximum_length:
            lenstr += ' exactly {0:d}.'.format(self.minimum_length)
        else:
            lenstr += 'in the range [{0:d}, {1:d}].'.format(self.minimum_length, self.maximum_length)
        return lenstr

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Field {} of class {} is an integer list of size {}, and must have size at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Field {} of class {} is an integer list of size {}, and must have size no larger than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            self.data[instance] = new_value

        if super(_IntegerListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, int):
            set_value([value, ])
        elif isinstance(value, minidom.Element):
            set_value([int(_get_node_value(value)), ])
        elif isinstance(value, minidom.NodeList):
            set_value([int(_get_node_value(nod)) for nod in value])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], int):
                set_value(value)
            elif isinstance(value[0], minidom.Element):
                set_value([int(_get_node_value(nod)) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _FloatDescriptor(_BasicDescriptor):
    """A descriptor for float type properties"""
    __typ_string = 'float:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, bounds=None, docstring=None):
        self.bounds = bounds
        super(_FloatDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.bounds is not None:
            return 'Must be in the range [{}, {}]'.format(str(self.bounds[0]), str(self.bounds[1]))
        return ''

    def __set__(self, instance, value):
        if super(_FloatDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, minidom.Element):
            # from XML deserialization
            iv = float(_get_node_value(value))
        else:
            # from user of json deserialization
            iv = float(value)

        if self.bounds is None or (self.bounds[0] <= iv <= self.bounds[1]):
            self.data[instance] = iv
        else:
            msg = 'Field {} of class {} is required by standard to take value between {}.'.format(
                self.name, instance.__class__.__name__, self.bounds)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.warning(msg)
            self.data[instance] = iv


class _ComplexDescriptor(_BasicDescriptor):
    """A descriptor for float type properties"""
    __typ_string = 'complex:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, docstring=None):
        super(_ComplexDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_ComplexDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, minidom.Element):
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
        else:
            # from user or json deserialization
            self.data[instance] = complex(value)


class _FloatArrayDescriptor(_BasicDescriptor):
    """A descriptor for float array type properties"""
    __typ_string = ':obj:numpy.ndarray of dtype=float64:'
    __DEFAULT_MIN_LENGTH = 0
    __DEFAULT_MAX_LENGTH = 2**32

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self.__DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self.__DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_FloatArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.minimum_length == self.__DEFAULT_MIN_LENGTH and \
                self.maximum_length == self.__DEFAULT_MAX_LENGTH:
            return ''

        lenstr = ' Must have length '
        if self.minimum_length == self.__DEFAULT_MIN_LENGTH:
            lenstr += '<= {0:d}.'.format(self.maximum_length)
        elif self.maximum_length == self.__DEFAULT_MAX_LENGTH:
            lenstr += '>= {0:d}.'.format(self.minimum_length)
        elif self.minimum_length == self.maximum_length:
            lenstr += ' exactly {0:d}.'.format(self.minimum_length)
        else:
            lenstr += 'in the range [{0:d}, {1:d}].'.format(self.minimum_length, self.maximum_length)
        return lenstr

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Field {} of class {} is a double array of size {}, and must have size at least ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Field {} of class {} is a double array of size {}, and must have size no larger than ' \
                      '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            self.data[instance] = new_value

        if super(_FloatArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, numpy.ndarray):
            if not (len(value) == 1) and (numpy.dtype == numpy.float64):
                raise ValueError('Only one-dimensional ndarrays of dtype float64 are supported here.')
            set_value(value)
        elif isinstance(value, minidom.Element):
            size = int(value.getAttribute('size'))
            child_nodes = [entry for entry in value.getElementsByTagName(self.child_tag) if entry.parentNode == value]
            if len(child_nodes) != size:
                raise ValueError(
                    'Field {} of double array type functionality belonging to class {} got a minidom element with size '
                    'attribute {}, but has {} child nodes with tag {}.'.format(
                        self.name, instance.__class__.__name__, size, len(child_nodes), self.child_tag))
            new_value = numpy.empty((size, ), dtype=numpy.float64)
            for i, node in enumerate(new_value):
                new_value[i] = float(_get_node_value(node))
            set_value(new_value)
        elif isinstance(value, list):
            set_value(numpy.array(value, dtype=numpy.float64))
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _DateTimeDescriptor(_BasicDescriptor):
    """A descriptor for date time type properties"""
    __typ_string = 'numpy.datetime64:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, docstring=None, numpy_datetime_units='us'):
        super(_DateTimeDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)
        self.units = numpy_datetime_units  # s, ms, us, ns are likely choices here, depending on needs

    def __set__(self, instance, value):
        if super(_DateTimeDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return
        if isinstance(value, minidom.Element):
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
    __typ_string = 'float:'

    def __init__(self, name, limit, required, strict=DEFAULT_STRICT, docstring=None):
        super(_FloatModularDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)
        self.limit = limit

    def __set__(self, instance, value):
        if super(_FloatModularDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, minidom.Element):
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

    def __init__(self, name, the_type, required, strict=DEFAULT_STRICT, tag=None, docstring=None):
        self.the_type = the_type
        self.__typ_string = the_type.__class__.__name__+':'
        self.tag = tag
        super(_SerializableDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(_SerializableDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, self.the_type):
            self.data[instance] = value
        elif isinstance(value, dict):
            self.data[instance] = self.the_type.from_dict(value).modify_tag(self.tag)
        elif isinstance(value, minidom.Element):
            self.data[instance] = self.the_type.from_node(value).modify_tag(self.tag)
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class _SerializableArrayDescriptor(_BasicDescriptor):
    """A descriptor for properties of a list or array of specified extension of Serializable"""
    __DEFAULT_MIN_LENGTH = 0
    __DEFAULT_MAX_LENGTH = 2**32

    def __init__(self, name, child_type, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_type = child_type
        self.__typ_string = child_type.__class__.__name__+':'
        tags = tag_dict[name]
        self.array = tags.get('array', False)
        self.child_tag = tags['child_tag']
        self.minimum_length = self.__DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self.__DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_SerializableArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def _docstring_suffix(self):
        if self.minimum_length == self.__DEFAULT_MIN_LENGTH and \
                self.maximum_length == self.__DEFAULT_MAX_LENGTH:
            return ''

        lenstr = ' Must have length '
        if self.minimum_length == self.__DEFAULT_MIN_LENGTH:
            lenstr += '<= {0:d}.'.format(self.maximum_length)
        elif self.maximum_length == self.__DEFAULT_MAX_LENGTH:
            lenstr += '>= {0:d}.'.format(self.minimum_length)
        elif self.minimum_length == self.maximum_length:
            lenstr += ' exactly {0:d}.'.format(self.minimum_length)
        else:
            lenstr += 'in the range [{0:d}, {1:d}].'.format(self.minimum_length, self.maximum_length)
        return lenstr

    def __actual_set(self, instance, value):
        if len(value) < self.minimum_length:
            msg = 'Field {} of class {} is of size {}, and must have size at least ' \
                  '{}.'.format(self.name, instance.__class__.__name__, value.size, self.minimum_length)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        if len(value) > self.maximum_length:
            msg = 'Field {} of class {} is of size {}, and must have size no greater than ' \
                  '{}.'.format(self.name, instance.__class__.__name__, value.size, self.maximum_length)
            if self.strict:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        self.data[instance] = value

    def __array_set(self, instance, value):
        if isinstance(value, numpy.ndarray):
            if value.dtype != numpy.object:
                raise ValueError(
                    'Field {} of array type functionality belonging to class {} got a ndarray of dtype {},'
                    'but contains objects, so requires dtype=numpy.object.'.format(
                        self.name, instance.__class__.__name__, value.dtype))
            elif len(value.shape) != 1:
                raise ValueError(
                    'Field {} of array type functionality belonging to class {} got a ndarray of shape {},'
                    'but requires a one dimensional array.'.format(
                        self.name, instance.__class__.__name__, value.shape))
            elif not isinstance(value[0], self.child_type):
                raise TypeError(
                    'Field {} of array type functionality belonging to class {} got a ndarray containing '
                    'first element of incompaticble type {}.'.format(
                        self.name, instance.__class__.__name__, type(value[0])))
            self.__actual_set(instance, value)
        elif isinstance(value, minidom.Element):
            # this is the parent node from XML deserialization
            size = int(value.getAttribute('size'))
            # extract child nodes at top level
            child_nodes = [entry for entry in value.getElementsByTagName(self.child_tag) if entry.parentNode == value]
            if len(child_nodes) != size:
                raise ValueError(
                    'Field {} of array type functionality belonging to class {} got a minidom element with size '
                    'attribute {}, but has {} child nodes with tag {}.'.format(
                        self.name, instance.__class__.__name__, size, len(child_nodes), self.child_tag))
            new_value = numpy.empty((size, ), dtype=numpy.object)
            for i, entry in enumerate(child_nodes):
                new_value[i] = self.child_type.from_node(entry).modify_tag(self.child_tag)
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
                    [self.child_type.from_dict(node).modify_tag(self.child_tag) for node in value], dtype=numpy.object))
            else:
                raise TypeError(
                    'Field {} of array type functionality belonging to class {} got a list containing first element of '
                    'incompatible type {}.'.format(self.name, instance.__class__.__name__, type(value[0])))
        else:
            raise TypeError(
                'Field {} of array type functionality belonging to class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))

    def __list_set(self, instance, value):
        if isinstance(value, self.child_type):
            # this is the child element
            self.__actual_set(instance, [value, ])
        elif isinstance(value, minidom.Element):
            # this is the child
            self.__actual_set(instance, [self.child_type.from_node(value).modify_tag(self.child_tag), ])
        elif isinstance(value, minidom.NodeList):
            new_value = []
            for node in value:  # NB: I am ignoring the index attribute (if it exists) and just leaving it in doc order
                new_value.append(self.child_type.from_node(node).modify_tag(self.child_tag))
            self.__actual_set(instance, value)
        elif isinstance(value, list) or isinstance(value[0], self.child_type):
            if len(value) == 0:
                self.__actual_set(instance, value)
            elif isinstance(value[0], dict):
                # NB: charming errors are possible if something stupid has been done.
                self.__actual_set(instance, [self.child_type.from_dict(node).modify_tag(self.child_tag) for node in value])
            elif isinstance(value[0], minidom.Element):
                self.__actual_set(instance, [self.child_type.from_node(node).modify_tag(self.child_tag) for node in value])
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
# dom helper functions, because minidom is a little weird


def _get_node_value(nod):
    """XML parsing helper for extracting text value from an minidom node. No error checking performed.

    Parameters
    ----------
    nod : minidom.Element
        the xml dom element object

    Returns
    -------
    str
        the string value of the node.
    """

    return nod.firstChild.wholeText.strip()


def _create_new_node(doc, tag, par=None):
    """XML minidom node creation helper function.

    Parameters
    ----------
    doc : minidom.Document
        The xml Document object.
    tag : str
        Name/tag for new xml element.
    par : None|minidom.Element
        The parent element for the new element. Defaults to the document root element if unspecified.
    Returns
    -------
    minidom.Element
        The new element populated as a child of `par`.
    """

    nod = doc.createElement(tag)
    if par is None:
        doc.documentElement.appendChild(nod)
    else:
        par.appendChild(nod)
    return nod


def _create_text_node(doc, tag, value, par=None):
    """XML minidom text node creation helper function

    Parameters
    ----------
    doc : minidom.Document
        The xml Document object.
    tag : str
        Name/tag for new xml element.
    value : str
        The value for the new element.
    par : None|minidom.Element
        The parent element for the new element. Defaults to the document root element if unspecified.

    Returns
    -------
    minidom.Element
        The new element populated as a child of `par`.
    """

    nod = doc.createElement(tag)
    nod.appendChild(doc.createTextNode(value))

    if par is None:
        doc.documentElement.appendChild(nod)
    else:
        par.appendChild(nod)
    return nod


#################
# base Serializable class.

class Serializable(object):
    """
    Basic abstract class specifying the serialization pattern. There are no clearly defined Python conventions
    for this issue. Every effort has been made to select sensible choices, but this is an individual effort.

    Notes
    -----
        All fields MUST BE LISTED in the `__fields` tuple. Everything listed in `__required` tuple will be checked
        for inclusion in `__fields` tuple. Note that special care must be taken to ensure compatibility of `__fields`
        tuple, if inheriting from an extension of this class.
    """

    __tag = None
    """tag name when serializing"""
    __fields = ()
    """collection of field names"""
    __required = ()
    """subset of `__fields` defining the required (for the given object, according to the sicd standard) fields"""

    __collections_tags = {}
    """Entries only appropriate for list/array type objects. Entry formatting:

    * `{'array': True, 'child_tag': <child_name>}` represents an array object, which will have int attribute `size`.
      It has *size* children with tag=<child_name>, each of which has an attribute `index`, which is not always an
      integer. Even when it is an integer, it apparently sometimes follows the matlab convention (1 based), and
      sometimes the standard convention (0 based). In this case, I will deserialize as though the objects are
      properly ordered, and the deserialized objects will have the `index` property from the xml, but it will not
      be used to determine array order - which will be simply carried over from file order.

    - `{'array': False, 'child_tag': <child_name>}` represents a collection of things with tag=<child_name>.
          This entries are not directly below one coherent container tag, but just dumped into an object.
          For example of such usage search for "Parameter" in the SICD standard.

          In this case, I've have to create an ephemeral variable in the class that doesn't exist in the standard,
          and it's not clear what the intent is for this unstructured collection, so used a list object.
          For example, I have a variable called `Parameters` in `CollectionInfoType`, whose job it to contain the
          parameters objects.
    """

    __numeric_format = {}
    """define dict entries of numeric formatting for serialization"""
    __set_as_attribute = ()
    """serialize these fields as xml attributes"""

    # NB: it may be good practice to use __slots__ to further control class functionality?

    def __init__(self, **kwargs):
        """
        The default constructor. For each attribute name in `self.__fields`, fetches the value (or None) from
        the `kwargs` dict, and sets the class instance attribute value. The details for attribute value validation,
        present for virtually every attribute, will be implemented specifically as descriptors.

        Parameters
        ----------
        **kwargs : dict
            the keyword arguments dictionary - the possible entries match the attributes.
        """

        for attribute in self.__fields:
            try:
                setattr(self, attribute, kwargs.get(attribute, None))
            except AttributeError:
                # NB: this is included to allow for read only properties without breaking the paradigm
                pass

    def modify_tag(self, value):
        """Sets the default tag for serialization.

        Parameters
        ----------
        value : str
            the appropriate value for the `__tag` attribute.

        Returns
        -------
        The class instance (i.e. `self`) - this is a convenient signature for chaining method calls.
        """

        if value is None:
            return self
        elif isinstance(value, str):
            self.__tag = value
            return self
        else:
            raise TypeError("tag requires string input")

    def set_numeric_format(self, attribute, format_string):
        """Sets the numeric format string for the given attribute.

        Parameters
        ----------
        attribute : str
            attribute for which the format applies - must be in `__fields`.
        format_string : str
            format string to be applied
        Returns
        -------
        None
        """
        # TODO: extend this to include format function capabilities. numeric_format is not the right name.
        if attribute not in self.__fields:
            raise ValueError('attribute {} is not permitted for class {}'.format(attribute, self.__class__.__name__))
        self.__numeric_format[attribute] = format_string

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

        entry = self.__numeric_format.get(attribute, None)
        if isinstance(entry, str):
            fmt_str = '{0:' + entry + '}'
            return fmt_str.format
        elif callable(entry):
            return entry
        else:
            return str

    def is_valid(self, recursive=False):
        """Returns the validity of this object according to the schema. This is done by inspecting that all required
        fields (i.e. entries of `__required`) are not `None`.

        Parameters
        ----------
        recursive : bool
            True if we recursively check that child are also valid. This may result in noisy logging.

        Returns
        -------
        bool
            condition for validity of this element
        """

        def check_item(value):
            good = True
            if isinstance(value, Serializable):
                good = value.is_valid(recursive=recursive)
            return good

        all_required = True
        if len(self.__required) > 0:
            for attribute in self.__required:
                present = getattr(self, attribute) is not None
                if not present:
                    logging.warning(
                        "class {} has missing required attribute {}".format(self.__class__.__name__, attribute))
                all_required &= present
        if not recursive:
            return all_required

        valid_children = True
        for attribute in self.__fields:
            val = getattr(self, attribute)
            good = True
            if isinstance(val, Serializable):
                good = check_item(val)
            elif isinstance(val, list) or (isinstance(val, numpy.ndarray) and val.dtype == numpy.object):
                for entry in val:
                    good &= check_item(entry)
            valid_children &= good
            # any issues will be logged as discovered, but we should help with the "stack"
            if not good:
                logging.warning(
                    "Issue discovered with {} attribute of type {} of class {}".format(
                    attribute, type(val), self.__class__.__name__))
        return all_required & valid_children

    @classmethod
    def from_node(cls, node, kwargs=None):
        """For XML deserialization.

        Parameters
        ----------
        node : minidom.Element
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
            val = node.getAttribute(the_tag)  # this returns an empty string if the_tag doesn't exist
            if len(val) > 0:
                kwargs[the_tag] = val

        def handle_single(the_tag):
            pnodes = [entry for entry in node.getElementsByTagName(the_tag) if entry.parentNode == node]
            if len(pnodes) > 0:
                kwargs[the_tag] = pnodes[0]

        def handle_list(attribute, child_tag):
            pnodes = [entry for entry in node.getElementsByTagName(child_tag) if entry.parentNode == node]
            if len(pnodes) > 0:
                kwargs[attribute] = pnodes

        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise ValueError(
                "Named input argument kwargs for class {} must be dictionary instance".format(cls.__class__.__name__))

        for attribute in cls.__fields:
            if attribute in kwargs:
                continue

            kwargs[attribute] = None
            # This value will be replaced if tags are present
            # Note that we want to try explicitly setting to None to trigger descriptor behavior
            # for required fields (warning or error)

            if attribute in cls.__set_as_attribute:
                handle_attribute(attribute)
            elif attribute in cls.__collections_tags:
                # it's an array type of a list type parameter
                array_tag = cls.__collections_tags[attribute]
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
                        'Attribute {} in class {} is listed in the __collections_tags dictionary, but the '
                        '"child_tags" value is either missing or None.'.format(attribute, cls.__class__.__name__))
            else:
                # it's a regular property
                handle_single(attribute)
        return cls.from_dict(kwargs)

    def to_node(self, doc, tag=None, par=None, strict=DEFAULT_STRICT, exclude=()):
        """For XML serialization, to a dom element.

        Parameters
        ----------
        doc : minidom.Document
            The xml Document
        tag : None|str
            The tag name. Defaults to the value of `self.__tag` and then the class name if unspecified.
        par : None|minidom.Element
            The parent element. Defaults to the document root element if unspecified.
        strict : bool
            If `True`, then raise an Exception (of appropriate type) if the structure is not valid.
            Otherwise, log a hopefully helpful message.
        exclude : tuple
            Attribute names to exclude from this generic serialization. This allows for child classes
            to provide specific serialization for special properties, after using this super method.

        Returns
        -------
        minidom.Element
            The constructed dom element, already assigned to the parent element.
        """

        def set_attribute(node, field, value, fmt_func):
            node.setAttribute(field, fmt_func(value))

        def serialize_array(node, the_tag, child_tag, value, fmt_func):
            if not isinstance(value, numpy.ndarray):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # __array_tag or the descriptor, probably at runtime
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be an array based on '
                    'the metadata in the __arrays_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(value)))
            if not len(value.shape) == 1:
                # again, I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a one-dimensional numpy.ndarray, but it has shape {}'.format(
                        attribute, self.__class__.__name__, value.shape))
            if value.size == 0:
                return  # serializing an empty array is dumb

            if value.dtype == numpy.float64:
                anode = _create_new_node(doc, the_tag, par=node)
                anode.setAttribute('size', str(value.size))
                for i, val in enumerate(value):
                    vnode = _create_text_node(doc, child_tag, fmt_func(val), par=anode)
                    vnode.setAttribute('index', str(i))  # I think that this is reliable
            elif value.dtype == numpy.object:
                anode = _create_new_node(doc, the_tag, par=node)
                anode.setAttribute('size', str(value.size))
                for i, entry in enumerate(value):
                    if not isinstance(entry, Serializable):
                        raise TypeError(
                            'The value associated with attribute {} is an instance of class {} should be an object '
                            'array based on the standard, but entry {} is of type {} and not an instance of '
                            'Serializable'.format(attribute, self.__class__.__name__, i, type(entry)))
                    serialize_plain(anode, child_tag, entry, fmt_func)
            else:
                # I have no idea how we'd find ourselves here, unless inconsistencies have been introduced
                # into the descriptor
                raise ValueError(
                    'The value associated with attribute {} is an instance of class {}, if None, is required to be'
                    'a numpy.ndarray of dtype float64 or object, but it has dtype {}'.format(
                        attribute, self.__class__.__name__, value.dtype))

        def serialize_list(node, child_tag, value, fmt_func):
            if not isinstance(value, list):
                # this should really never happen, unless someone broke the class badly by fiddling with
                # __collections_tags or the descriptor?
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be a list based on '
                    'the metadata in the __arrays_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(value)))
            if len(value) == 0:
                return  # serializing an empty list is dumb
            else:
                for entry in value:
                    serialize_plain(node, child_tag, entry, fmt_func)

        def serialize_plain(node, field, value, fmt_func):
            # may be called not at top level - if object array or list is present
            if isinstance(value, Serializable):
                value.to_node(doc, tag=field, par=node, strict=strict)
            elif isinstance(value, str):
                # TODO: MEDIUM - unicode issues?
                _create_text_node(doc, field, value, par=node)
            elif isinstance(value, int) or isinstance(value, float):
                _create_text_node(doc, field, fmt_func(value), par=node)
            elif isinstance(value, bool):
                # fmt_func here? note that str doesn't work, so...
                _create_text_node(doc, field, '1' if value else '0', par=node)
            elif isinstance(value, date):
                _create_text_node(doc, field, value.isoformat(), par=node)
            elif isinstance(value, datetime):
                _create_text_node(doc, field, value.isoformat(sep='T'), par=node)
            elif isinstance(value, numpy.datetime64):
                _create_text_node(doc, field, str(value), par=node)
            elif isinstance(value, complex):
                cnode = _create_new_node(doc, field, par=node)
                _create_text_node(doc, 'Real', fmt_func(value.real), par=cnode)
                _create_text_node(doc, 'Imag', fmt_func(value.imag), par=cnode)
            else:
                raise ValueError(
                    'a entry for class {} using tag {} is of type {}, and serialization has not '
                    'been implemented'.format(self.__class__.__name__, field, type(value)))

        if not self.is_valid():
            msg = "{} is not valid, and cannot be safely serialized to XML according to " \
                  "the SICD standard.".format(self.__class__.__name__)
            if strict:
                raise ValueError(msg)
            logging.warning(msg)

        if tag is None:
            tag = self.__tag
        if tag is None:
            tag = self.__class__.__name__
        nod = _create_new_node(doc, tag, par=par)

        for attribute in self.__fields:
            if attribute in exclude:
                continue

            value = getattr(self, attribute)
            if value is None:
                continue

            fmt_func = self._get_formatter(attribute)
            array_tag = self.__collections_tags.get(attribute, None)
            if attribute in self.__set_as_attribute:
                set_attribute(nod, attribute, value, fmt_func)
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
                        'Attribute {} in class {} is listed in the __collections_tags dictionary, but the '
                        '"child_tags" value is either missing or None.'.format(attribute, self.__class__.__name__))
            else:
                serialize_plain(nod, attribute, value, fmt_func)
        return nod

    @classmethod
    def from_dict(cls, inputDict):  # type: (dict) -> Serializable
        """For json deserialization, from dict instance.

        Parameters
        ----------
        inputDict : dict
            Appropriate parameters dict instance for deserialization

        Returns
        -------
        Serializable
            Corresponding class instance
        """

        return cls(**inputDict)

    def to_dict(self, strict=DEFAULT_STRICT, exclude=()):  # type: (bool, tuple) -> OrderedDict
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

        # TODO: finish this as above in xml serialization
        # out = OrderedDict()
        # return out
        raise NotImplementedError("Must provide a concrete implementation.")


#############
# Basic building blocks for SICD standard

class PlainValueType(Serializable):
    """This is a basic xml building block element, and not actually specified in the SICD standard"""
    __fields = ('value', )
    __required = __fields
    # descriptor
    value = _StringDescriptor('value', __required, strict=True, docstring='The value')

    @classmethod
    def from_node(cls, node, kwargs=None):
        return cls(value=_get_node_value(node))

    def to_node(self, doc, tag=None, par=None, strict=DEFAULT_STRICT, exclude=()):
        # we have to short-circuit the super call here, because this is a really primitive element
        if tag is None:
            tag = self.__tag
        if tag is None:
            tag = self.__class__.__name__
        node = _create_text_node(doc, tag, self.value, par=par)
        return node


class FloatValueType(Serializable):
    """This is a basic xml building block element, and not actually specified in the SICD standard"""
    __fields = ('value', )
    __required = __fields
    # descriptor
    value = _FloatDescriptor('value', __required, strict=True, docstring='The value')

    @classmethod
    def from_node(cls, node, kwargs=None):
        return cls(value=_get_node_value(node))

    def to_node(self, doc, tag=None, par=None, strict=DEFAULT_STRICT, exclude=()):
        # we have to short-circuit the call here, because this is a really primitive element
        if tag is None:
            tag = self.__tag
        if tag is None:
            tag = self.__class__.__name__
        fmt_func = self._get_formatter('value')
        node = _create_text_node(doc, tag, fmt_func(self.value), par=par)
        return node


class ComplexType(PlainValueType):
    """A complex number"""
    __fields = ('Real', 'Imag')
    __required = __fields
    # descriptor
    Real = _FloatDescriptor(
        'Real', __required, strict=True, docstring='The real component.')
    Imag = _FloatDescriptor(
        'Imag', __required, strict=True, docstring='The imaginary component.')


class ParameterType(PlainValueType):
    """A Parameter structure - just a name attribute and associated value"""
    __tag = 'Parameter'
    __fields = ('name', 'value')
    __required = __fields
    __set_as_attribute = ('name', )
    # descriptor
    name = _StringDescriptor(
        'name', __required, strict=True, docstring='The name.')
    value = _StringDescriptor(
        'value', __required, strict=True, docstring='The value.')


class XYZType(Serializable):
    """A spatial point in ECF coordinates."""
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    __numeric_format = {'X': '0.8f', 'Y': '0.8f', 'Z': '0.8f'}  # TODO: desired precision? This is usually meters?
    # descriptors
    X = _FloatDescriptor(
        'X', __required, strict=DEFAULT_STRICT,
        docstring='The X attribute. Assumed to ECF or other, similar coordinates.')
    Y = _FloatDescriptor(
        'Y', __required, strict=DEFAULT_STRICT,
        docstring='The Y attribute. Assumed to ECF or other, similar coordinates.')
    Z = _FloatDescriptor(
        'Z', __required, strict=DEFAULT_STRICT,
        docstring='The Z attribute. Assumed to ECF or other, similar coordinates.')

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
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    # descriptors
    Lat = _FloatDescriptor(
        'Lat', __required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatDescriptor(
        'Lon', __required, strict=DEFAULT_STRICT,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')

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
    __fields = ('Lat', 'Lon', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    index = _IntegerDescriptor(
        'index', __required, strict=False, docstring="The array index")


class LatLonRestrictionType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates."""
    __fields = ('Lat', 'Lon')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    # descriptors
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, __required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, __required, strict=DEFAULT_STRICT,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')


class LatLonHAEType(LatLonType):
    """A three-dimensional geographic point in WGS-84 coordinates."""
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    __numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?
    # descriptors
    HAE = _FloatDescriptor(
        'HAE', __required, strict=DEFAULT_STRICT,
        docstring='The Height Above Ellipsoid (in meters) attribute. Assumed to be WGS-84 coordinates.')

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
    __fields = ('Lat', 'Lon', 'HAE')
    __required = __fields
    """A three-dimensional geographic point in WGS-84 coordinates."""
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, __required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, __required, strict=DEFAULT_STRICT,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')


class LatLonCornerType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    __fields = ('Lat', 'Lon', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', __required, strict=True, bounds=(1, 4),
        docstring='The integer index. This represents a clockwise enumeration of the rectangle vertices '
                  'wrt the frame of reference of the collector.')


class LatLonCornerStringType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    __fields = ('Lat', 'Lon', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # other specific class variable
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = _StringEnumDescriptor(
        'index', _CORNER_VALUES, __required, strict=True,
        docstring="The string index.")


class LatLonHAECornerRestrictionType(LatLonHAERestrictionType):
    """A three-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    __fields = ('Lat', 'Lon', 'HAE', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', __required, strict=True,
        docstring='The integer index. This represents a clockwise enumeration of the rectangle vertices '
                  'wrt the frame of reference of the collector.')


class LatLonHAECornerStringType(LatLonHAEType):
    """A three-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    __fields = ('Lat', 'Lon', 'HAE', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = _StringEnumDescriptor(
        'index', _CORNER_VALUES, __required, strict=True, docstring="The string index.")


class RowColType(Serializable):
    """A row and column attribute container - used as indices into array(s)."""
    __fields = ('Row', 'Col')
    __required = __fields
    Row = _IntegerDescriptor(
        'Row', __required, strict=DEFAULT_STRICT, docstring='The Row attribute.')
    Col = _IntegerDescriptor(
        'Col', __required, strict=DEFAULT_STRICT, docstring='The Column attribute.')


class RowColArrayElement(RowColType):
    """A array element row and column attribute container - used as indices into other array(s)."""
    # Note - in the SICD standard this type is listed as RowColvertexType. This is not a descriptive name
    # and has an inconsistency in camel case
    __fields = ('Row', 'Col', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', __required, strict=DEFAULT_STRICT, docstring='The array index attribute.')


class PolyCoef1DType(FloatValueType):
    """Represents a monomial term of the form `value * x^{exponent1}`."""
    __fields = ('value', 'exponent1')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    __set_as_attribute = ('exponent1', )
    # descriptors
    exponent1 = _IntegerDescriptor(
        'exponent1', __required, strict=DEFAULT_STRICT, docstring='The exponent1 attribute.')


class PolyCoef2DType(FloatValueType):
    """Represents a monomial term of the form `value * x^{exponent1} * y^{exponent2}`."""
    # NB: based on field names, one could consider PolyCoef2DType an extension of PolyCoef1DType. This has not
    #   be done here, because I would not want an instance of PolyCoef2DType to evaluate as True when testing if
    #   instance of PolyCoef1DType.

    __fields = ('value', 'exponent1', 'exponent2')
    __required = __fields
    __numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    __set_as_attribute = ('exponent1', 'exponent2')
    # descriptors
    exponent1 = _IntegerDescriptor(
        'exponent1', __required, strict=DEFAULT_STRICT, docstring='The exponent1 attribute.')
    exponent2 = _IntegerDescriptor(
        'exponent2', __required, strict=DEFAULT_STRICT, docstring='The exponent2 attribute.')


class Poly1DType(Serializable):
    """Represents a one-variable polynomial, defined as the sum of the given monomial terms."""
    __fields = ('Coefs', 'order1')
    __required = ('Coefs', )
    __collections_tags = {'Coefs': {'array': False, 'child_tag': 'Coef'}}
    __set_as_attribute = ('order1', )
    # descriptors
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef1DType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='The list of monomial terms.')

    @property
    def order1(self):
        """
        int: The order1 attribute [READ ONLY]  - that is, largest exponent presented in the monomial terms of coefs.
        """

        return 0 if self.Coefs is None else max(entry.exponent1 for entry in self.Coefs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class Poly2DType(Serializable):
    """Represents a one-variable polynomial, defined as the sum of the given monomial terms."""
    __fields = ('Coefs', 'order1', 'order2')
    __required = ('Coefs', )
    __collections_tags = {'Coefs': {'array': False, 'child_tag': 'Coef'}}
    __set_as_attribute = ('order1', 'order2')
    # descriptors
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef2DType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='The list of monomial terms.')

    @property
    def order1(self):
        """
        int: The order1 attribute [READ ONLY]  - that is, the largest exponent1 presented in the monomial terms of coefs.
        """

        return 0 if self.Coefs is None else max(entry.exponent1 for entry in self.Coefs)

    @property
    def order2(self):
        """
        int: The order2 attribute [READ ONLY]  - that is, the largest exponent2 presented in the monomial terms of coefs.
        """

        return 0 if self.Coefs is None else max(entry.exponent2 for entry in self.Coefs)

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class XYZPolyType(Serializable):
    """Represents a single variable polynomial for each of `X`, `Y`, and `Z`."""
    __fields = ('X', 'Y', 'Z')
    __required = __fields
    # descriptors
    X = _SerializableDescriptor(
        'X', Poly1DType, __required, strict=DEFAULT_STRICT, docstring='The X polynomial.')
    Y = _SerializableDescriptor(
        'Y', Poly1DType, __required, strict=DEFAULT_STRICT, docstring='The Y polynomial.')
    Z = _SerializableDescriptor(
        'Z', Poly1DType, __required, strict=DEFAULT_STRICT, docstring='The Z polynomial.')
    # TODO: a better description would be good here

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


class XYZPolyAttributeType(XYZPolyType):
    """
    An array element of X, Y, Z polynomials. The output of these polynomials are expected to spatial variables in
    the ECF coordinate system.
    """
    __fields = ('X', 'Y', 'Z', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', __required, strict=DEFAULT_STRICT, docstring='The array index value.')


class GainPhasePolyType(Serializable):
    """A container for the Gain and Phase Polygon definitions."""
    __fields = ('GainPoly', 'PhasePoly')
    __required = __fields
    # descriptors
    GainPoly = _SerializableDescriptor(
        'GainPoly', Poly2DType, __required, strict=DEFAULT_STRICT, docstring='The Gain Polygon.')
    PhasePoly = _SerializableDescriptor(
        'GainPhasePoly', Poly2DType, __required, strict=DEFAULT_STRICT, docstring='The Phase Polygon.')


class ErrorDecorrFuncType(Serializable):
    """The Error Decorrelation Function?"""
    __fields = ('CorrCoefZero', 'DecorrRate')
    __required = __fields
    __numeric_format = {'CorrCoefZero': '0.8f', 'DecorrRate': '0.8f'}  # TODO: desired precision?
    # descriptors
    CorrCoefZero = _FloatDescriptor(
        'CorrCoefZero', __required, strict=DEFAULT_STRICT, docstring='The CorrCoefZero attribute.')
    DecorrRate = _FloatDescriptor(
        'DecorrRate', __required, strict=DEFAULT_STRICT, docstring='The DecorrRate attribute.')

    # TODO: HIGH - this is supposed to be a "function". We should implement the functionality here.


class RadarModeType(Serializable):
    """Radar mode type container class"""
    __tag = 'RadarMode'
    __fields = ('ModeType', 'ModeId')
    __required = ('ModeType', )
    # other class variable
    _MODE_TYPE_VALUES = ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP')
    # descriptors
    ModeId = _StringDescriptor(
        'ModeId', __required, docstring='The Mode Id.')
    ModeType = _StringEnumDescriptor(
        'ModeType', _MODE_TYPE_VALUES, __required, strict=DEFAULT_STRICT,
        docstring="The mode type, which will be one of {}.".format(_MODE_TYPE_VALUES))


class FullImageType(Serializable):
    """The full image attributes"""
    __tag = 'FullImage'
    __fields = ('NumRows', 'NumCols')
    __required = __fields
    # descriptors
    NumRows = _IntegerDescriptor(
        'NumRows', __required, strict=DEFAULT_STRICT, docstring='The number of rows.')
    NumCols = _IntegerDescriptor(
        'NumCols', __required, strict=DEFAULT_STRICT, docstring='The number of columns.')


####################################################################
# Direct building blocks for SICD
#############
# CollectionInfoType section

class CollectionInfoType(Serializable):
    """The collection information container."""
    __tag = 'CollectionInfo'
    __collections_tags = {
        'Parameters': {'array': False, 'child_tag': 'Parameter'},
        'CountryCode': {'array': False, 'child_tag': 'CountryCode'},
    }
    __fields = (
        'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType',
        'RadarMode', 'Classification', 'Parameters', 'CountryCodes')
    __required = ('CollectorName', 'CoreName', 'RadarMode', 'Classification')
    # other class variable
    _COLLECT_TYPE_VALUES = ('MONOSTATIC', 'BISTATIC')
    # descriptors
    CollectorName = _StringDescriptor(
        'CollectorName', __required, strict=DEFAULT_STRICT, docstring='The collector name.')
    IlluminatorName = _StringDescriptor(
        'IlluminatorName', __required, strict=DEFAULT_STRICT, docstring='The illuminator name.')
    CoreName = _StringDescriptor(
        'CoreName', __required, strict=DEFAULT_STRICT, docstring='The core name.')
    CollectType = _StringEnumDescriptor(
        'CollectType', _COLLECT_TYPE_VALUES, __required,
        docstring="The collect type, one of {}".format(_COLLECT_TYPE_VALUES))
    RadarMode = _SerializableDescriptor(
        'RadarMode', RadarModeType, __required, strict=DEFAULT_STRICT, docstring='The radar mode')
    Classification = _StringDescriptor(
        'Classification', __required, strict=DEFAULT_STRICT, docstring='The classification.')
    # list type descriptors
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='The parameters objects **list**.')
    CountryCodes = _StringListDescriptor(
        'CountryCodes', __required, strict=DEFAULT_STRICT, docstring="The country code **list**.")


###############
# ImageCreation section


class ImageCreationType(Serializable):
    """The image creation data container."""
    __fields = ('Application', 'DateTime', 'Site', 'Profile')
    __required = ()
    # descriptors
    Application = _StringDescriptor('Application', __required, strict=DEFAULT_STRICT, docstring='The application.')
    DateTime = _DateTimeDescriptor(
        'DateTime', __required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The image creation date/time.')
    Site = _StringDescriptor(
        'Site', __required, strict=DEFAULT_STRICT, docstring='The site.')
    Profile = _StringDescriptor(
        'Profile', __required, strict=DEFAULT_STRICT, docstring='The profile.')


###############
# ImageData section


class ImageDataType(Serializable):
    """The image data container."""
    __collections_tags = {
        'AmpTable': {'array': True, 'child_tag': 'Amplitude'},
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
    }
    __fields = ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel',
                'ValidData')
    __required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')
    __numeric_format = {'AmpTable': '0.8f'}  # TODO: precision for AmpTable?
    _PIXEL_TYPE_VALUES = ("RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I")
    # descriptors
    PixelType = _StringEnumDescriptor(
        'PixelType', _PIXEL_TYPE_VALUES, __required, strict=True,
        docstring="The PixelType attribute which specifies the interpretation of the file data")
    NumRows = _IntegerDescriptor(
        'NumRows', __required, strict=DEFAULT_STRICT, docstring='The number of Rows')
    NumCols = _IntegerDescriptor(
        'NumCols', __required, strict=DEFAULT_STRICT, docstring='The number of Columns')
    FirstRow = _IntegerDescriptor(
        'FirstRow', __required, strict=DEFAULT_STRICT, docstring='The first row')
    FirstCol = _IntegerDescriptor(
        'FirstCol', __required, strict=DEFAULT_STRICT, docstring='The first column')
    FullImage = _SerializableDescriptor(
        'FullImage', FullImageType, __required, strict=DEFAULT_STRICT, docstring='The full image')
    SCPPixel = _SerializableDescriptor(
        'SCPPixel', RowColType, __required, strict=DEFAULT_STRICT, docstring='The SCP Pixel')
    ValidData = _SerializableArrayDescriptor(
        'ValidData', RowColArrayElement, __collections_tags, __required, strict=DEFAULT_STRICT,
        minimum_length=3, docstring='The valid data area **array**')
    AmpTable = _FloatArrayDescriptor(
        'AmpTable', __collections_tags, __required, strict=DEFAULT_STRICT,
        minimum_length=256, maximum_length=256,
        docstring="The amplitude look-up table. This must be defined if PixelType == 'AMP8I_PHS8I'")

    def is_valid(self, recursive=False):
        condition = super(ImageDataType, self).is_valid(recursive=recursive)

        pixel_type = not (self.PixelType == 'AMP8I_PHS8I' and self.AmpTable is None)
        if not pixel_type:
            logging.warning("We have `PixelType='AMP8I_PHS8I'` and `AmpTable` is undefined for ImageDataType.")
        return condition and pixel_type


###############
# GeoInfo section


class GeoInfoType(Serializable):
    """The GeoInfo container."""
    __tag = 'GeoInfo'
    __collections_tags = {
        'Descriptions': {'array': False, 'child_tag': 'Desc'},
        'Line': {'array': True, 'child_tag': 'Endpoint'},
        'Polygon': {'array': True, 'child_tag': 'Vertex'}, }
    __fields = ('name', 'Descriptions', 'Point', 'Line', 'Polygon')
    __required = ('name', )
    __set_as_attribute = ('name', )
    # descriptors
    name = _StringDescriptor(
        'name', __required, strict=True, docstring='The name.')
    Descriptions = _SerializableArrayDescriptor(
        'Descriptions', ParameterType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='The descriptions **list**.')
    Point = _SerializableDescriptor(
        'Point', LatLonRestrictionType, __required, strict=DEFAULT_STRICT,
        docstring='A geographic point with WGS-84 coordinates.')
    Line = _SerializableArrayDescriptor(
        'Line', LatLonArrayElementType, __collections_tags, __required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='A geographic line (**array**) with WGS-84 coordinates.')
    Polygon = _SerializableArrayDescriptor(
        'Polygon', LatLonArrayElementType, __collections_tags, __required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='A geographic polygon (**array**) with WGS-84 coordinates.')
    # TODO: is the standard really self-referential here? I find that confusing.


###############
# GeoData section


class SCPType(Serializable):
    """The Scene Center Point container"""
    __tag = 'SCP'
    __fields = ('ECF', 'LLH')
    __required = __fields  # isn't this redundant?
    ECF = _SerializableDescriptor(
        'ECF', XYZType, __required, strict=DEFAULT_STRICT, docstring='The ECF coordinates.')
    LLH = _SerializableDescriptor(
        'LLH', LatLonHAERestrictionType, __required, strict=DEFAULT_STRICT,
        docstring='The WGS 84 coordinates.')


class GeoDataType(Serializable):
    """Container specifying the image coverage area in geographic coordinates."""
    __fields = ('EarthModel', 'SCP', 'ImageCorners', 'ValidData', 'GeoInfos')
    __required = ('EarthModel', 'SCP', 'ImageCorners')
    __collections_tags = {
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
        'ImageCorners': {'array': True, 'child_tag': 'ICP'},
        'GeoInfos': {'array': False, 'child_tag': 'GeoInfo'},
    }
    # other class variables
    _EARTH_MODEL_VALUES = ('WGS_84', )
    # descriptors
    EarthModel = _StringEnumDescriptor(
        'EarthModel', _EARTH_MODEL_VALUES, __required, strict=True, default_value='WGS_84',
        docstring='The Earth Model.'.format(_EARTH_MODEL_VALUES))
    SCP = _SerializableDescriptor(
        'SCP', SCPType, __required, strict=DEFAULT_STRICT, tag='SCP', docstring='The Scene Center Point.')
    ImageCorners = _SerializableArrayDescriptor(
        'ImageCorners', LatLonCornerStringType, __collections_tags, __required, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring='The geographic image corner points **array**.')
    ValidData = _SerializableArrayDescriptor(
        'ValidData', LatLonArrayElementType, __collections_tags, __required,
        strict=DEFAULT_STRICT, minimum_length=3,
        docstring='The full image **array** includes both valid data and some zero filled pixels.')
    GeoInfos = _SerializableArrayDescriptor(
        'GeoInfos', GeoInfoType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='Relevant geographic features **list**.')


################
# DirParam section


class WgtTypeType(Serializable):
    """The weight type parameters of the direction parameters"""
    __fields = ('WindowName', 'Parameters')
    __required = ('WindowName', )
    __collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    WindowName = _StringDescriptor(
        'WindowName', __required, strict=DEFAULT_STRICT, docstring='The window name')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, __collections_tags, required=__required, strict=DEFAULT_STRICT,
        docstring='The parameters **list**')


class DirParamType(Serializable):
    """The direction parameters container"""
    __fields = (
        'UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2', 'DeltaKCOAPoly',
        'WgtType', 'WgtFunct')
    __required = ('UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2')
    __numeric_format = {
        'SS': '0.8f', 'ImpRespWid': '0.8f', 'Sgn': '+d', 'ImpRespBW': '0.8f', 'KCtr': '0.8f',
        'DeltaK1': '0.8f', 'DeltaK2': '0.8f'}
    __collections_tags = {'WgtFunct': {'array': True, 'child_tag': 'Wgt'}}
    # descriptors
    UVectECF = _SerializableDescriptor(
        'UVectECF', XYZType, tag='UVectECF', required=__required, strict=DEFAULT_STRICT,
        docstring='Unit vector in the increasing (row/col) direction (ECF) at the SCP pixel.')
    SS = _FloatDescriptor(
        'SS', __required, strict=DEFAULT_STRICT,
        docstring='Sample spacing in the increasing (row/col) direction. Precise spacing at the SCP.')
    ImpRespWid = _FloatDescriptor(
        'ImpRespWid', __required, strict=DEFAULT_STRICT,
        docstring='Half power impulse response width in the increasing (row/col) direction. Measured at the SCP.')
    Sgn = _IntegerEnumDescriptor(
        'Sgn', __required, values=(1, -1), strict=DEFAULT_STRICT,
        docstring='Sign for exponent in the DFT to transform the (row/col) dimension to '
                  'spatial frequency dimension.')
    ImpRespBW = _FloatDescriptor(
        'ImpRespBW', __required, strict=DEFAULT_STRICT,
        docstring='Spatial bandwidth in (row/col) used to form the impulse response in the (row/col) direction. '
                  'Measured at the center of support for the SCP.')
    KCtr = _FloatDescriptor(
        'KCtr', __required, strict=DEFAULT_STRICT,
        docstring='Center spatial frequency in the given dimension. '
                  'Corresponds to the zero frequency of the DFT in the given (row/col) direction.')
    DeltaK1 = _FloatDescriptor(
        'DeltaK1', __required, strict=DEFAULT_STRICT,
        docstring='Minimum (row/col) offset from KCtr of the spatial frequency support for the image.')
    DeltaK2 = _FloatDescriptor(
        'DeltaK2', __required, strict=DEFAULT_STRICT,
        docstring='Maximum (row/col) offset from KCtr of the spatial frequency support for the image.')
    DeltaKCOAPoly = _SerializableDescriptor(
        'DeltaKCOAPoly', Poly2DType, __required, tag='DeltaKCOAPoly', strict=DEFAULT_STRICT,
        docstring='Offset from KCtr of the center of support in the given (row/col) spatial frequency. '
                  'The polynomial is a function of image given (row/col) coordinate (variable 1) and '
                  'column coordinate (variable 2).')
    WgtType = _SerializableDescriptor(
        'WgtType', WgtTypeType, __required, tag='WgtType', strict=DEFAULT_STRICT,
        docstring='Parameters describing aperture weighting type applied in the spatial frequency domain '
                  'to yield the impulse response in the given(row/col) direction.')
    WgtFunct = _FloatArrayDescriptor(
        'WgtFunct', __collections_tags, __required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='Sampled aperture amplitude weighting function (**array**) applied to form the SCP impulse '
                  'response in the given (row/col) direction.')

    def is_valid(self, recursive=False):
        valid = super(DirParamType, self).is_valid(recursive=recursive)
        if self.WgtFunct is not None:
            if self.WgtFunct.size < 2:
                logging.warning('The WgtFunct array has been provided, but there are fewer than 2 entries.')
                valid = False
        return valid

###############
# GridType section


class GridType(Serializable):
    """Collection grid details container"""
    __fields = ('ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col')
    __required = __fields
    _IMAGE_PLANE_VALUES = ('SLANT', 'GROUND', 'OTHER')
    _TYPE_VALUES = ('RGAZIM', 'RGZERO', 'XRGYCR', 'XCTYAT', 'PLANE')
    # descriptors
    ImagePlane = _StringEnumDescriptor(
        'ImagePlane', _IMAGE_PLANE_VALUES, __required, strict=DEFAULT_STRICT,
        docstring="The image plane. Possible values are {}".format(_IMAGE_PLANE_VALUES))
    Type = _StringEnumDescriptor(
        'Type', _TYPE_VALUES, __required, strict=DEFAULT_STRICT,
        docstring="The possible grid type enumeration.")
    TimeCOAPoly = _SerializableDescriptor(
        'TimeCOAPoly', Poly2DType, __required, strict=DEFAULT_STRICT,
        docstring="*Time of Center Of Aperture* as a polynomial function of image coordinates. "
                  "The polynomial is a function of image row coordinate (variable 1) and column coordinate "
                  "(variable 2).")
    Row = _SerializableDescriptor(
        'Row', DirParamType, __required, strict=DEFAULT_STRICT,
        docstring="Row direction parameters.")
    Col = _SerializableDescriptor(
        'Col', DirParamType, __required, strict=DEFAULT_STRICT,
        docstring="Column direction parameters.")


##############
# TimelineType section


class IPPSetType(Serializable):
    """The Inter-Pulse Parameter array element container."""
    # NOTE that this is simply defined as a child class ("Set") of the TimelineType in the SICD standard
    #   Defining it at root level clarifies the documentation, and giving it a more descriptive name is
    #   appropriate.
    __tag = 'Set'
    __fields = ('TStart', 'TEnd', 'IPPStart', 'IPPEnd', 'IPPPoly', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    TStart = _FloatDescriptor(
        'TStart', __required, strict=DEFAULT_STRICT,
        docstring='IPP start time relative to collection start time, i.e. offsets in seconds.')
    TEnd = _FloatDescriptor(
        'TEnd', __required, strict=DEFAULT_STRICT,
        docstring='IPP end time relative to collection start time, i.e. offsets in seconds.')
    IPPStart = _IntegerDescriptor(
        'IPPStart', __required, strict=True, docstring='Starting IPP index for the period described.')
    IPPEnd = _IntegerDescriptor(
        'IPPEnd', __required, strict=True, docstring='Ending IPP index for the period described.')
    IPPPoly = _SerializableDescriptor(
        'IPPPoly', Poly1DType, __required, strict=DEFAULT_STRICT,
        docstring='IPP index polynomial coefficients yield IPP index as a function of time for TStart to TEnd.')
    index = _IntegerDescriptor(
        'index', __required, strict=DEFAULT_STRICT, docstring='The element array index.')


class TimelineType(Serializable):
    """The details for the imaging collection timeline."""
    __fields = ('CollectStart', 'CollectDuration', 'IPP')
    __required = ('CollectStart', 'CollectDuration', )
    __collections_tags = {'IPP': {'array': True, 'child_tag': 'Set'}}
    # descriptors
    CollectStart = _DateTimeDescriptor(
        'CollectStart', __required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The collection start time. The default precision will be microseconds.')
    CollectDuration = _FloatDescriptor(
        'CollectDuration', __required, strict=DEFAULT_STRICT,
        docstring='The duration of the collection in seconds.')
    IPP = _SerializableArrayDescriptor(
        'IPP', IPPSetType, __collections_tags, __required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring="The Inter-Pulse Period (IPP) parameters **array**.")


###################
# PositionType section


class PositionType(Serializable):
    """The details for platform and ground reference positions as a function of time since collection start."""
    __fields = ('ARPPoly', 'GRPPoly', 'TxAPCPoly', 'RcvAPC')
    __required = ('ARPPoly',)
    __collections_tags = {'RcvAPC': {'array': True, 'child_tag': 'RcvAPCPoly'}}

    # descriptors
    ARPPoly = _SerializableDescriptor(
        'ARPPoly', XYZPolyType, __required, strict=DEFAULT_STRICT,
        docstring='Aperture Reference Point (ARP) position polynomial in ECF as a function of elapsed '
                  'seconds since start of collection.')
    GRPPoly = _SerializableDescriptor(
        'GRPPoly', XYZPolyType, __required, strict=DEFAULT_STRICT,
        docstring='Ground Reference Point (GRP) position polynomial in ECF as a function of elapsed '
                  'seconds since start of collection.')
    TxAPCPoly = _SerializableDescriptor(
        'TxAPCPoly', XYZPolyType, __required, strict=DEFAULT_STRICT,
        docstring='Transmit Aperture Phase Center (APC) position polynomial in ECF as a function of '
                  'elapsed seconds since start of collection.')
    RcvAPC = _SerializableArrayDescriptor(
        'RcvAPC', XYZPolyAttributeType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='Receive Aperture Phase Center polynomials **array**. '
                  'Each polynomial has output in ECF, and represents a function of elapsed seconds since start of '
                  'collection.')


##################
# RadarCollectionType section


class TxFrequencyType(Serializable):
    """The transmit frequency range"""
    __tag = 'TxFrequency'
    __fields = ('Min', 'Max')
    __required = __fields
    # descriptors
    Min = _FloatDescriptor(
        'Min', required=__required, strict=DEFAULT_STRICT,
        docstring='The transmit minimum frequency in Hz.')
    Max = _FloatDescriptor(
        'Max', required=__required, strict=DEFAULT_STRICT,
        docstring='The transmit maximum frequency in Hz.')


class WaveformParametersType(Serializable):
    """Transmit and receive demodulation waveform parameters."""
    __fields = (
        'TxPulseLength', 'TxRFBandwidth', 'TxFreqStart', 'TxFMRate', 'RcvDemodType', 'RcvWindowLength',
        'ADCSampleRate', 'RcvIFBandwidth', 'RcvFreqStart', 'RcvFMRate')
    __required = ()
    # other class variables
    _DEMOD_TYPE_VALUES = ('STRETCH', 'CHIRP')
    # descriptors
    TxPulseLength = _FloatDescriptor(
        'TxPulseLength', __required, strict=DEFAULT_STRICT,
        docstring='Transmit pulse length in seconds.')
    TxRFBandwidth = _FloatDescriptor(
        'TxRFBandwidth', __required, strict=DEFAULT_STRICT,
        docstring='Transmit RF bandwidth of the transmit pulse in Hz.')
    TxFreqStart = _FloatDescriptor(
        'TxFreqStart', __required, strict=DEFAULT_STRICT,
        docstring='Transmit Start frequency for Linear FM waveform in Hz, may be relative to reference frequency.')
    TxFMRate = _FloatDescriptor(
        'TxFMRate', __required, strict=DEFAULT_STRICT,
        docstring='Transmit FM rate for Linear FM waveform in Hz/second.')
    RcvWindowLength = _FloatDescriptor(
        'RcvWindowLength', __required, strict=DEFAULT_STRICT,
        docstring='Receive window duration in seconds.')
    ADCSampleRate = _FloatDescriptor(
        'ADCSampleRate', __required, strict=DEFAULT_STRICT,
        docstring='Analog-to-Digital Converter sampling rate in samples/second.')
    RcvIFBandwidth = _FloatDescriptor(
        'RcvIFBandwidth', __required, strict=DEFAULT_STRICT,
        docstring='Receive IF bandwidth in Hz.')
    RcvFreqStart = _FloatDescriptor(
        'RcvFreqStart', __required, strict=DEFAULT_STRICT,
        docstring='Receive demodulation start frequency in Hz, may be relative to reference frequency.')
    RcvFMRate = _FloatDescriptor(
        'RcvFMRate', __required, strict=DEFAULT_STRICT,
        docstring='Receive FM rate. Should be 0 if RcvDemodType = "CHIRP".')
    RcvDemodType = _StringEnumDescriptor(
        'RcvDemodType', _DEMOD_TYPE_VALUES, __required, strict=DEFAULT_STRICT,
        docstring="Receive demodulation used when Linear FM waveform is used on transmit.")

    def __init__(self, **kwargs):
        super(WaveformParametersType, self).__init__(**kwargs)
        if self.RcvDemodType == 'CHIRP':
            self.RcvFMRate = 0

    def is_valid(self, recursive=False):
        valid = super(WaveformParametersType, self).is_valid(recursive=recursive)
        if self.RcvDemodType == 'CHIRP' and self.RcvFMRate != 0:
            logging.warning('In WaveformParameters, we have RcvDemodType == "CHIRP" and self.RcvFMRate non-zero.')

        return valid


class TxStepType(Serializable):
    """Transmit sequence step details"""
    __fields = ('WFIndex', 'TxPolarization', 'index')
    __required = ('index', )
    # other class variables
    _POLARIZATION2_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER')
    # descriptors
    WFIndex = _IntegerDescriptor(
        'WFIndex', __required, strict=DEFAULT_STRICT, docstring='The waveform number for this step.')
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION2_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='Transmit signal polarization for this step.')
    index = _IntegerDescriptor(
        'index', __required, strict=DEFAULT_STRICT, docstring='The step index')


class ChanParametersType(Serializable):
    """Transmit receive sequence step details"""
    __fields = ('TxRcvPolarization', 'RcvAPCIndex', 'index')
    __required = ('TxRcvPolarization', 'index', )
    # other class variables
    _DUAL_POLARIZATION_VALUES = (
        'V:V', 'V:H', 'H:V', 'H:H', 'RHC:RHC', 'RHC:LHC', 'LHC:RHC', 'LHC:LHC', 'OTHER', 'UNKNOWN')
    # descriptors
    TxRcvPolarization = _StringEnumDescriptor(
        'TxRcvPolarization', _DUAL_POLARIZATION_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='Combined Transmit and Receive signal polarization for the channel.')
    RcvAPCIndex = _IntegerDescriptor(
        'RcvAPCIndex', __required, strict=DEFAULT_STRICT,
        docstring='Index of the Receive Aperture Phase Center (Rcv APC). Only include if Receive APC position '
                  'polynomial(s) are included.')
    index = _IntegerDescriptor(
        'index', __required, strict=DEFAULT_STRICT, docstring='The parameter index')


class ReferencePointType(Serializable):
    """The reference point definition"""
    __fields = ('ECF', 'Line', 'Sample', 'name')
    __required = __fields
    __set_as_attribute = ('name', )
    # descriptors
    ECF = _SerializableDescriptor(
        'ECF', XYZType, __required, strict=DEFAULT_STRICT,
        docstring='The geographical coordinates for the reference point.')
    Line = _FloatDescriptor(
        'Line', __required, strict=DEFAULT_STRICT,
        docstring='The Line?')  # TODO: what is this?
    Sample = _FloatDescriptor(
        'Sample', __required, strict=DEFAULT_STRICT,
        docstring='The Sample?')
    name = _StringDescriptor(
        'name', __required, strict=DEFAULT_STRICT,
        docstring='The reference point name.')


class XDirectionType(Serializable):
    """The X direction of the collect"""
    __fields = ('UVectECF', 'LineSpacing', 'NumLines', 'FirstLine')
    __required = __fields
    # descriptors
    ECF = _SerializableDescriptor(
        'UVectECF', XYZType,__required, strict=DEFAULT_STRICT,
        docstring='The unit vector')
    LineSpacing = _FloatDescriptor(
        'LineSpacing', __required, strict=DEFAULT_STRICT,
        docstring='The collection line spacing in meters.')
    NumLines = _IntegerDescriptor(
        'NumLines', __required, strict=DEFAULT_STRICT,
        docstring='The number of lines')
    FirstLine = _IntegerDescriptor(
        'FirstLine', __required, strict=DEFAULT_STRICT,
        docstring='The first line')


class YDirectionType(Serializable):
    """The Y direction of the collect"""
    __fields = ('UVectECF', 'LineSpacing', 'NumSamples', 'FirstSample')
    __required = __fields
    # descriptors
    ECF = _SerializableDescriptor(
        'UVectECF', XYZType, __required, strict=DEFAULT_STRICT,
        docstring='The unit vector.')
    LineSpacing = _FloatDescriptor(
        'LineSpacing', __required, strict=DEFAULT_STRICT,
        docstring='The collection line spacing in meters.')
    NumSamples = _IntegerDescriptor(
        'NumSamples', __required, strict=DEFAULT_STRICT,
        docstring='The number of samples.')
    FirstSample = _IntegerDescriptor(
        'FirstSample', __required, strict=DEFAULT_STRICT,
        docstring='The first sample.')


class SegmentArrayElement(Serializable):
    """The reference point definition"""
    __fields = ('StartLine', 'StartSample', 'EndLine', 'EndSample', 'index')
    __required = __fields
    __set_as_attribute = ('index', )
    # descriptors
    StartLine = _IntegerDescriptor(
        'StartLine', __required, strict=DEFAULT_STRICT,
        docstring='The starting line number.')
    StartSample = _IntegerDescriptor(
        'StartSample', __required, strict=DEFAULT_STRICT,
        docstring='The starting sample number.')
    EndLine = _IntegerDescriptor(
        'EndLine', __required, strict=DEFAULT_STRICT,
        docstring='The ending line number.')
    EndSample = _IntegerDescriptor(
        'EndSample', __required, strict=DEFAULT_STRICT,
        docstring='The ending sample number.')
    index = _IntegerDescriptor(
        'index', __required, strict=DEFAULT_STRICT,
        docstring='The array index.')


class ReferencePlaneType(Serializable):
    """The reference plane"""
    __fields = ('RefPt', 'XDir', 'YDir', 'SegmentList', 'Orientation')
    __required = ('RefPt', 'XDir', 'YDir')
    __collections_tags = {'SegmentList': {'array': True, 'child_tag': 'SegmentList'}}
    # other class variable
    _ORIENTATION_VALUES = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'ARBITRARY')
    # descriptors
    RefPt = _SerializableDescriptor(
        'RefPt', ReferencePointType, __required, strict=DEFAULT_STRICT,
        docstring='The reference point.')
    XDir = _SerializableDescriptor(
        'XDir', XDirectionType, __required, strict=DEFAULT_STRICT,
        docstring='The X direction collection plane parameters.')
    YDir = _SerializableDescriptor(
        'YDir', YDirectionType, __required, strict=DEFAULT_STRICT,
        docstring='The Y direction collection plane parameters.')
    SegmentList = _SerializableArrayDescriptor(
        'SegmentList', SegmentArrayElement, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='The segment **list**.')
    Orientation = _StringEnumDescriptor(
        'Orientation', _ORIENTATION_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='The orientation value.')


class AreaType(Serializable):
    """The collection area"""
    __fields = ('Corner', 'Plane')
    __required = ('Corner', )
    __collections_tags = {
        'Corner': {'array': False, 'child_tag': 'ACP'},}
    # descriptors
    Corner = _SerializableArrayDescriptor(
        'Corner', LatLonHAECornerRestrictionType, __collections_tags, __required, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring='The collection area corner point definition array.')
    Plane = _SerializableDescriptor(
        'Plane', ReferencePlaneType, __required, strict=DEFAULT_STRICT,
        docstring='The collection area reference plane.')


class RadarCollectionType(Serializable):
    """The Radar Collection Type"""
    __tag = 'RadarCollection'
    __fields = ('TxFrequency', 'RefFreqIndex', 'Waveform', 'TxPolarization', 'TxSequence', 'RcvChannels',
                'Area', 'Parameters')
    __required = ('TxFrequency', 'TxPolarization', 'RcvChannels')
    __collections_tags = {
        'Waveform': {'array': True, 'child_tag': 'WFParameters'},
        'TxSequence': {'array': True, 'child_tag': 'TxStep'},
        'RcvChannels': {'array': True, 'child_tag': 'RcvChannels'},
        'Parameters': {'array': False, 'child_tag': 'Parameters'}}
    # other class variables
    _POLARIZATION1_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER', 'UNKNOWN', 'SEQUENCE')
    # descriptors
    TxFrequencyType = _SerializableDescriptor(
        'TxFrequency', TxFrequencyType, __required, strict=DEFAULT_STRICT,
        docstring='The transmit frequency range.')
    RefFreqIndex = _IntegerDescriptor(
        'RefFreqIndex', __required, strict=DEFAULT_STRICT,
        docstring='The reference frequency index, if applicable.')
    Waveform = _SerializableArrayDescriptor(
        'Waveform', WaveformParametersType, __collections_tags, __required,
        strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The waveform parameters **array**.')
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION1_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='The transmit polarization.')  # TODO: iff SEQUENCE, then TxSequence is defined?
    TxSequence = _SerializableArrayDescriptor(
        'TxSequence', TxStepType, __collections_tags, __required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The transmit sequence parameters **array**.')
    RcvChannels = _SerializableArrayDescriptor(
        'RcvChannels', ChanParametersType, __collections_tags,
        __required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Transmit receive sequence step details **array**.')
    Area = _SerializableDescriptor(
        'Area', AreaType, __required, strict=DEFAULT_STRICT,
        docstring='The collection area.')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='A parameters **list**.')


###############
# ImageFormationType section


class RcvChanProcType(Serializable):
    """Receive Channel Process Type"""
    __fields = ('NumChanProc', 'PRFScaleFactor', 'ChanIndices')
    __required = ('NumChanProc', 'ChanIndices')  # TODO: make proper descriptor
    __collections_tags = {
        'ChanIndices': {'array': False, 'child_tag': 'ChanIndex'}}
    # descriptors
    NumChanProc = _IntegerDescriptor(
        'NumChanProc', __required, strict=DEFAULT_STRICT,
        docstring='The Num Chan Proc?')
    PRFScaleFactor = _FloatDescriptor(
        'PRFScaleFactor', __required, strict=DEFAULT_STRICT,
        docstring='The PRF scale factor.')
    ChanIndices = _IntegerListDescriptor(
        'ChanIndices', __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='The channel index **list**.')


class TxFrequencyProcType(Serializable):
    """The transmit frequency range"""
    __tag = 'TxFrequencyProc'
    __fields = ('MinProc', 'MaxProc')
    __required = __fields
    # descriptors
    MinProc = _FloatDescriptor(
        'MinProc', __required, strict=DEFAULT_STRICT,
        docstring='The transmit minimum frequency in Hz.')
    MaxProc = _FloatDescriptor(
        'MaxProc', __required, strict=DEFAULT_STRICT,
        docstring='The transmit maximum frequency in Hz.')


class ProcessingType(Serializable):
    """The transmit frequency range"""
    __tag = 'Processing'
    __fields = ('Type', 'Applied', 'Parameters')
    __required = ('Type', 'Applied')
    __collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    Type = _StringDescriptor(
        'Type', __required, strict=DEFAULT_STRICT, docstring='The type string.')
    Applied = _BooleanDescriptor(
        'Applied', __required, strict=DEFAULT_STRICT,
        docstring='Whether the given type has been applied.')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='The parameters **list**.')


class DistortionType(Serializable):
    """Distortion"""
    __fields = (
        'CalibrationDate', 'A', 'F1', 'F2', 'Q1', 'Q2', 'Q3', 'Q4',
        'GainErrorA', 'GainErrorF1', 'GainErrorF2', 'PhaseErrorF1', 'PhaseErrorF2')
    __required = ('A', 'F1', 'Q1', 'Q2', 'F2', 'Q3', 'Q4')
    # descriptors
    CalibrationDate = _DateTimeDescriptor(
        'CalibrationDate', __required, strict=DEFAULT_STRICT,
        docstring='The calibration date.')
    A = _FloatDescriptor(
        'A', __required, strict=DEFAULT_STRICT, docstring='The A attribute.')
    F1 = _ComplexDescriptor(
        'F1', __required, strict=DEFAULT_STRICT, docstring='The F1 attribute.')
    F2 = _ComplexDescriptor(
        'F2', __required, strict=DEFAULT_STRICT, docstring='The F2 attribute.')
    Q1 = _ComplexDescriptor(
        'Q1', __required, strict=DEFAULT_STRICT, docstring='The Q1 attribute.')
    Q2 = _ComplexDescriptor(
        'Q2', __required, strict=DEFAULT_STRICT, docstring='The Q2 attribute.')
    Q3 = _ComplexDescriptor(
        'Q3', __required, strict=DEFAULT_STRICT, docstring='The Q3 attribute.')
    Q4 = _ComplexDescriptor(
        'Q4', __required, strict=DEFAULT_STRICT, docstring='The Q4 attribute.')
    GainErrorA = _FloatDescriptor(
        'GainErrorA', __required, strict=DEFAULT_STRICT,
        docstring='The GainErrorA attribute.')
    GainErrorF1 = _FloatDescriptor(
        'GainErrorF1', __required, strict=DEFAULT_STRICT,
        docstring='The GainErrorF1 attribute.')
    GainErrorF2 = _FloatDescriptor(
        'GainErrorF2', __required, strict=DEFAULT_STRICT,
        docstring='The GainErrorF2 attribute.')
    PhaseErrorF1 = _FloatDescriptor(
        'PhaseErrorF1', __required, strict=DEFAULT_STRICT,
        docstring='The PhaseErrorF1 attribute.')
    PhaseErrorF2 = _FloatDescriptor(
        'PhaseErrorF2', __required, strict=DEFAULT_STRICT,
        docstring='The PhaseErrorF2 attribute.')


class PolarizationCalibrationType(Serializable):
    """The polarization calibration"""
    __fields = ('DistortCorrectApplied', 'Distortion')
    __required = __fields
    # descriptors
    DistortCorrectApplied = _BooleanDescriptor(
        'DistortCorrectApplied', __required, strict=DEFAULT_STRICT,
        docstring='Whether the distortion correction has been applied')
    Distortion = _SerializableDescriptor(
        'Distortion', DistortionType, __required, strict=DEFAULT_STRICT,
        docstring='The distortion parameters.')


class ImageFormationType(Serializable):
    """The image formation parameters type"""
    __fields = (
        'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc', 'SegmentIdentifier',
        'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus', 'Processings',
        'PolarizationCalibration')
    __required = (
        'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc',
        'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus')
    __collections_tags = {'Processings': {'array': False, 'child_tag': 'Processing'}}
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
        'RcvChanProc', RcvChanProcType, __required, strict=DEFAULT_STRICT,
        docstring='The received channels processed?')
    TxRcvPolarizationProc = _StringEnumDescriptor(
        'TxRcvPolarizationProc', _DUAL_POLARIZATION_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='The transmit/receive polarization.')
    TStartProc = _FloatDescriptor(
        'TStartProc', __required, strict=DEFAULT_STRICT, docstring='The processing start time.')
    TEndProc = _FloatDescriptor(
        'TEndProc', __required, strict=DEFAULT_STRICT, docstring='The processing end time.')
    TxFrequencyProc = _SerializableDescriptor(
        'TxFrequencyProc', TxFrequencyProcType, __required, strict=DEFAULT_STRICT,
        docstring='The processing frequency range.')
    SegmentIdentifier = _StringDescriptor(
        'SegmentIdentifier', __required, strict=DEFAULT_STRICT, docstring='The segment identifier.')
    ImageFormAlgo = _StringEnumDescriptor(
        'ImageFormAlgo', _IMG_FORM_ALGO_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='The image formation algorithm used.')
    STBeamComp = _StringEnumDescriptor(
        'STBeamComp', _ST_BEAM_COMP_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='The ST beam comp.')
    ImageBeamComp = _StringEnumDescriptor(
        'ImageBeamComp', _IMG_BEAM_COMP_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='The image beam comp.')
    AzAutofocus = _StringEnumDescriptor(
        'AzAutofocus', _AZ_AUTOFOCUS_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='The azimuth autofocus.')
    RgAutofocus = _StringEnumDescriptor(
        'RgAutofocus', _RG_AUTOFOCUS_VALUES, __required, strict=DEFAULT_STRICT,
        docstring='The range autofocus.')
    Processings = _SerializableArrayDescriptor(
        'Processings', ProcessingType, __collections_tags, __required, strict=DEFAULT_STRICT,
        docstring='The processing collection.')
    PolarizationCalibration = _SerializableDescriptor(
        'PolarizationCalibration', PolarizationCalibrationType, __required, strict=DEFAULT_STRICT,
        docstring='The polarization calibration details.')


###############
# SCPCOAType section


class SCPCOAType(Serializable):
    """The scene center point - COA?"""
    __fields = (
        'SCPTime', 'ARPPos', 'ARPVel', 'ARPAcc', 'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAng',
        'GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng')
    __required = __fields
    # class variables
    _SIDE_OF_TRACK_VALUES = ('L', 'R')
    # descriptors
    SCPTime = _FloatDescriptor(
        'SCPTime', __required, strict=DEFAULT_STRICT, docstring='The scene center point time in seconds?')
    ARPPos = _SerializableDescriptor(
        'ARPPos', XYZType, __required, strict=DEFAULT_STRICT, docstring='The aperture position.')
    ARPVel = _SerializableDescriptor(
        'ARPVel', XYZType, __required, strict=DEFAULT_STRICT, docstring='The aperture velocity.')
    ARPAcc = _SerializableDescriptor(
        'ARPAcc', XYZType, __required, strict=DEFAULT_STRICT, docstring='The aperture acceleration.')
    SideOfTrack = _StringEnumDescriptor(
        'SideOfTrack', _SIDE_OF_TRACK_VALUES, __required, strict=DEFAULT_STRICT, docstring='The side of track.')
    SlantRange = _FloatDescriptor(
        'SlantRange', __required, strict=DEFAULT_STRICT, docstring='The slant range.')
    GroundRange = _FloatDescriptor(
        'GroundRange', __required, strict=DEFAULT_STRICT, docstring='The ground range.')
    DopplerConeAng = _FloatDescriptor(
        'DopplerConeAng', __required, strict=DEFAULT_STRICT, docstring='The Doppler cone angle.')
    GrazeAng = _FloatDescriptor(
        'GrazeAng', __required, strict=DEFAULT_STRICT, bounds=(0., 90.), docstring='The graze angle.')
    IncidenceAng = _FloatDescriptor(
        'IncidenceAng', __required, strict=DEFAULT_STRICT, bounds=(0., 90.), docstring='The incidence angle.')
    TwistAng = _FloatDescriptor(
        'TwistAng', __required, strict=DEFAULT_STRICT, bounds=(-90., 90.), docstring='The twist angle.')
    SlopeAng = _FloatDescriptor(
        'SlopeAng', __required, strict=DEFAULT_STRICT, bounds=(0., 90.), docstring='The slope angle.')
    AzimAng = _FloatDescriptor(
        'AzimAng', __required, strict=DEFAULT_STRICT, bounds=(0., 360.), docstring='The azimuth angle.')
    LayoverAng = _FloatDescriptor(
        'LayoverAng', __required, strict=DEFAULT_STRICT, bounds=(0., 360.), docstring='The layover angle.')


###############
# RadiometricType section
###############
# AntParamType section
###############
# AntennaType section
###############
# ErrorStatisticsType section
###############
# MatchInfoType section
###############
# RgAzCompType section
###############
# PFAType section
###############
# RMAType section
####################################################################
# the SICD object


# class ShellType(Serializable):
#     """"""
#     __fields = ()
#     __required = ()
#     __collections_tags = {'': {'array': False, 'child_tag': ''}}
#     # descriptors
