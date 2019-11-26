"""
**This module is a work in progress. The eventual structure of this is yet to be determined.**

Object oriented SICD structure definition. Enabling effective documentation and streamlined use of the SICD information
is the main purpose of this approach, versus the matlab struct based effort or using the Python bindings for the C++
SIX library.
"""

# TODO: 1.) complete to_dict functionality for serializable.
# TODO: 2.) flesh out docstrings from the sicd standards document
# TODO: 3.) determine necessary and appropriate formatting issues for serialization/deserialization
# TODO: 4.) determine and implement appropriate class methods for proper functionality

from xml.dom import minidom
import numpy
from collections import OrderedDict
from datetime import datetime, date
import logging
from weakref import WeakKeyDictionary

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
    _typ_string = 'str:'

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
    _typ_string = ':obj:`list` of :obj:`str`:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2**32

    def __init__(self, name, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None, docstring=None):
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(_StringListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

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
    _typ_string = 'str:'

    def __init__(self, name, values, required, strict=DEFAULT_STRICT, docstring=None, default_value=None):
        self.values = values
        self.default_value = default_value if default_value in values else None
        super(_StringEnumDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

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
    _typ_string = 'bool:'

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
    _typ_string = 'int:'

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
    _typ_string = 'int:'

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
    _typ_string = 'float:'

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
    _typ_string = 'complex:'

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

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None, docstring=None):

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
    _typ_string = 'float:'

    def __init__(self, name, limit, required, strict=DEFAULT_STRICT, docstring=None):
        self.limit = limit
        super(_FloatModularDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

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
        elif isinstance(value, minidom.Element):
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
            self.__actual_set(instance, [self.child_type.from_node(value), ])
        elif isinstance(value, minidom.NodeList):
            new_value = []
            for node in value:  # NB: I am ignoring the index attribute (if it exists) and just leaving it in doc order
                new_value.append(self.child_type.from_node(node))
            self.__actual_set(instance, value)
        elif isinstance(value, list) or isinstance(value[0], self.child_type):
            if len(value) == 0:
                self.__actual_set(instance, value)
            elif isinstance(value[0], dict):
                # NB: charming errors are possible if something stupid has been done.
                self.__actual_set(instance, [self.child_type.from_dict(node) for node in value])
            elif isinstance(value[0], minidom.Element):
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
        All fields MUST BE LISTED in the `_fields` tuple. Everything listed in `_required` tuple will be checked
        for inclusion in `_fields` tuple. Note that special care must be taken to ensure compatibility of `_fields`
        tuple, if inheriting from an extension of this class.
    """

    _fields = ()
    """collection of field names"""
    _required = ()
    """subset of `_fields` defining the required (for the given object, according to the sicd standard) fields"""

    _collections_tags = {}
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

    _numeric_format = {}
    """define dict entries of numeric formatting for serialization"""
    _set_as_attribute = ()
    """serialize these fields as xml attributes"""

    # NB: it may be good practice to use __slots__ to further control class functionality?

    def __init__(self, **kwargs):
        """
        The default constructor. For each attribute name in `self._fields`, fetches the value (or None) from
        the `kwargs` dict, and sets the class instance attribute value. The details for attribute value validation,
        present for virtually every attribute, will be implemented specifically as descriptors.

        Parameters
        ----------
        **kwargs : dict
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
        # TODO: extend this to include format function capabilities. numeric_format is not the right name.
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
            True if we recursively check that child are also valid. This may result in noisy logging.

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
                logging.warning(
                    "Class {} has missing required attribute {}".format(self.__class__.__name__, attribute))
            all_required &= present
        return all_required

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
                logging.warning(
                    "Issue discovered with {} attribute of type {} of class {}".format(
                    attribute, type(val), self.__class__.__name__))
            valid_children &= good
        return valid_children

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

    def to_node(self, doc, tag, par=None, strict=DEFAULT_STRICT, exclude=()):
        """For XML serialization, to a dom element.

        Parameters
        ----------
        doc : minidom.Document
            The xml Document
        tag : None|str
            The tag name. Defaults to the value of `self._tag` and then the class name if unspecified.
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
                # _collections_tag or the descriptor, probably at runtime
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be an array based on '
                    'the metadata in the _collections_tags dictionary, but we received an instance of '
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
                # _collections_tags or the descriptor?
                raise TypeError(
                    'The value associated with attribute {} is an instance of class {} should be a list based on '
                    'the metadata in the _collections_tags dictionary, but we received an instance of '
                    'type {}'.format(attribute, self.__class__.__name__, type(value)))
            if len(value) == 0:
                return  # serializing an empty list is dumb
            else:
                for entry in value:
                    serialize_plain(node, child_tag, entry, fmt_func)

        def serialize_plain(node, field, value, fmt_func):
            # may be called not at top level - if object array or list is present
            if isinstance(value, Serializable):
                value.to_node(doc, field, par=node, strict=strict)
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

        nod = _create_new_node(doc, tag, par=par)

        for attribute in self._fields:
            if attribute in exclude:
                continue

            value = getattr(self, attribute)
            if value is None:
                continue

            fmt_func = self._get_formatter(attribute)
            array_tag = self._collections_tags.get(attribute, None)
            if attribute in self._set_as_attribute:
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
                        'Attribute {} in class {} is listed in the _collections_tags dictionary, but the '
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
    _fields = ('value', )
    _required = _fields
    # descriptor
    value = _StringDescriptor('value', _required, strict=True, docstring='The value')

    @classmethod
    def from_node(cls, node, kwargs=None):
        return cls(value=_get_node_value(node))

    def to_node(self, doc, tag, par=None, strict=DEFAULT_STRICT, exclude=()):
        # we have to short-circuit the super call here, because this is a really primitive element
        node = _create_text_node(doc, tag, self.value, par=par)
        return node


class FloatValueType(Serializable):
    """This is a basic xml building block element, and not actually specified in the SICD standard"""
    _fields = ('value', )
    _required = _fields
    # descriptor
    value = _FloatDescriptor('value', _required, strict=True, docstring='The value')

    @classmethod
    def from_node(cls, node, kwargs=None):
        return cls(value=_get_node_value(node))

    def to_node(self, doc, tag, par=None, strict=DEFAULT_STRICT, exclude=()):
        # we have to short-circuit the call here, because this is a really primitive element
        fmt_func = self._get_formatter('value')
        node = _create_text_node(doc, tag, fmt_func(self.value), par=par)
        return node


class ComplexType(PlainValueType):
    """A complex number"""
    _fields = ('Real', 'Imag')
    _required = _fields
    # descriptor
    Real = _FloatDescriptor(
        'Real', _required, strict=True, docstring='The real component.')
    Imag = _FloatDescriptor(
        'Imag', _required, strict=True, docstring='The imaginary component.')


class ParameterType(PlainValueType):
    """A Parameter structure - just a name attribute and associated value"""
    _tag = 'Parameter'
    _fields = ('name', 'value')
    _required = _fields
    _set_as_attribute = ('name', )
    # descriptor
    name = _StringDescriptor(
        'name', _required, strict=True, docstring='The name.')
    value = _StringDescriptor(
        'value', _required, strict=True, docstring='The value.')


class XYZType(Serializable):
    """A spatial point in ECF coordinates."""
    _fields = ('X', 'Y', 'Z')
    _required = _fields
    _numeric_format = {'X': '0.8f', 'Y': '0.8f', 'Z': '0.8f'}  # TODO: desired precision? This is usually meters?
    # descriptors
    X = _FloatDescriptor(
        'X', _required, strict=DEFAULT_STRICT,
        docstring='The X attribute. Assumed to ECF or other, similar coordinates.')
    Y = _FloatDescriptor(
        'Y', _required, strict=DEFAULT_STRICT,
        docstring='The Y attribute. Assumed to ECF or other, similar coordinates.')
    Z = _FloatDescriptor(
        'Z', _required, strict=DEFAULT_STRICT,
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
    _fields = ('Lat', 'Lon')
    _required = _fields
    _numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    # descriptors
    Lat = _FloatDescriptor(
        'Lat', _required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatDescriptor(
        'Lon', _required, strict=DEFAULT_STRICT,
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
    _fields = ('Lat', 'Lon', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    _numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    index = _IntegerDescriptor(
        'index', _required, strict=False, docstring="The array index")


class LatLonRestrictionType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon')
    _required = _fields
    _numeric_format = {'Lat': '2.8f', 'Lon': '3.8f'}  # TODO: desired precision?
    # descriptors
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, _required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, _required, strict=DEFAULT_STRICT,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')


class LatLonHAEType(LatLonType):
    """A three-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon', 'HAE')
    _required = _fields
    _numeric_format = {'Lat': '2.8f', 'Lon': '3.8f', 'HAE': '0.3f'}  # TODO: desired precision?
    # descriptors
    HAE = _FloatDescriptor(
        'HAE', _required, strict=DEFAULT_STRICT,
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
    _fields = ('Lat', 'Lon', 'HAE')
    _required = _fields
    """A three-dimensional geographic point in WGS-84 coordinates."""
    Lat = _FloatModularDescriptor(
        'Lat', 90.0, _required, strict=DEFAULT_STRICT,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.')
    Lon = _FloatModularDescriptor(
        'Lon', 180.0, _required, strict=DEFAULT_STRICT,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.')


class LatLonCornerType(LatLonType):
    """A two-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', _required, strict=True, bounds=(1, 4),
        docstring='The integer index. This represents a clockwise enumeration of the rectangle vertices '
                  'wrt the frame of reference of the collector.')


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
        docstring="The string index.")


class LatLonHAECornerRestrictionType(LatLonHAERestrictionType):
    """A three-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'HAE', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', _required, strict=True,
        docstring='The integer index. This represents a clockwise enumeration of the rectangle vertices '
                  'wrt the frame of reference of the collector.')


class LatLonHAECornerStringType(LatLonHAEType):
    """A three-dimensional geographic point in WGS-84 coordinates representing a collection area box corner point."""
    _fields = ('Lat', 'Lon', 'HAE', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    _CORNER_VALUES = ('1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC')
    # descriptors
    index = _StringEnumDescriptor(
        'index', _CORNER_VALUES, _required, strict=True, docstring="The string index.")


class RowColType(Serializable):
    """A row and column attribute container - used as indices into array(s)."""
    _fields = ('Row', 'Col')
    _required = _fields
    Row = _IntegerDescriptor(
        'Row', _required, strict=DEFAULT_STRICT, docstring='The Row attribute.')
    Col = _IntegerDescriptor(
        'Col', _required, strict=DEFAULT_STRICT, docstring='The Column attribute.')


class RowColArrayElement(RowColType):
    """A array element row and column attribute container - used as indices into other array(s)."""
    # Note - in the SICD standard this type is listed as RowColvertexType. This is not a descriptive name
    # and has an inconsistency in camel case
    _fields = ('Row', 'Col', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The array index attribute.')


class PolyCoef1DType(FloatValueType):
    """Represents a monomial term of the form `value * x^{exponent1}`."""
    _fields = ('value', 'exponent1')
    _required = _fields
    _numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    _set_as_attribute = ('exponent1', )
    # descriptors
    exponent1 = _IntegerDescriptor(
        'exponent1', _required, strict=DEFAULT_STRICT, docstring='The exponent1 attribute.')


class PolyCoef2DType(FloatValueType):
    """Represents a monomial term of the form `value * x^{exponent1} * y^{exponent2}`."""
    # NB: based on field names, one could consider PolyCoef2DType an extension of PolyCoef1DType. This has not
    #   be done here, because I would not want an instance of PolyCoef2DType to evaluate as True when testing if
    #   instance of PolyCoef1DType.

    _fields = ('value', 'exponent1', 'exponent2')
    _required = _fields
    _numeric_format = {'value': '0.8f'}  # TODO: desired precision?
    _set_as_attribute = ('exponent1', 'exponent2')
    # descriptors
    exponent1 = _IntegerDescriptor(
        'exponent1', _required, strict=DEFAULT_STRICT, docstring='The exponent1 attribute.')
    exponent2 = _IntegerDescriptor(
        'exponent2', _required, strict=DEFAULT_STRICT, docstring='The exponent2 attribute.')


class Poly1DType(Serializable):
    """Represents a one-variable polynomial, defined as the sum of the given monomial terms."""
    _fields = ('Coefs', 'order1')
    _required = ('Coefs', )
    _collections_tags = {'Coefs': {'array': False, 'child_tag': 'Coef'}}
    _set_as_attribute = ('order1', )
    # descriptors
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef1DType, _collections_tags, _required, strict=DEFAULT_STRICT,
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
    _fields = ('Coefs', 'order1', 'order2')
    _required = ('Coefs', )
    _collections_tags = {'Coefs': {'array': False, 'child_tag': 'Coef'}}
    _set_as_attribute = ('order1', 'order2')
    # descriptors
    Coefs = _SerializableArrayDescriptor(
        'Coefs', PolyCoef2DType, _collections_tags, _required, strict=DEFAULT_STRICT,
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
    _fields = ('X', 'Y', 'Z')
    _required = _fields
    # descriptors
    X = _SerializableDescriptor(
        'X', Poly1DType, _required, strict=DEFAULT_STRICT, docstring='The X polynomial.')
    Y = _SerializableDescriptor(
        'Y', Poly1DType, _required, strict=DEFAULT_STRICT, docstring='The Y polynomial.')
    Z = _SerializableDescriptor(
        'Z', Poly1DType, _required, strict=DEFAULT_STRICT, docstring='The Z polynomial.')
    # TODO: a better description would be good here

    # TODO: HIGH - helper method to get numpy.polynomial.polynomial objects and so forth?
    #   Look to SICD functionality for figuring out what we need here.


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
        'index', _required, strict=DEFAULT_STRICT, docstring='The array index value.')


class GainPhasePolyType(Serializable):
    """A container for the Gain and Phase Polygon definitions."""
    _fields = ('GainPoly', 'PhasePoly')
    _required = _fields
    # descriptors
    GainPoly = _SerializableDescriptor(
        'GainPoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='The Gain Polygon.')
    PhasePoly = _SerializableDescriptor(
        'GainPhasePoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='The Phase Polygon.')


class ErrorDecorrFuncType(Serializable):
    """The Error Decorrelation Function?"""
    _fields = ('CorrCoefZero', 'DecorrRate')
    _required = _fields
    _numeric_format = {'CorrCoefZero': '0.8f', 'DecorrRate': '0.8f'}  # TODO: desired precision?
    # descriptors
    CorrCoefZero = _FloatDescriptor(
        'CorrCoefZero', _required, strict=DEFAULT_STRICT, docstring='The CorrCoefZero attribute.')
    DecorrRate = _FloatDescriptor(
        'DecorrRate', _required, strict=DEFAULT_STRICT, docstring='The DecorrRate attribute.')

    # TODO: HIGH - this is supposed to be a "function". We should implement the functionality here.


class RadarModeType(Serializable):
    """Radar mode type container class"""
    _tag = 'RadarMode'
    _fields = ('ModeType', 'ModeId')
    _required = ('ModeType', )
    # other class variable
    _MODE_TYPE_VALUES = ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP')
    # descriptors
    ModeId = _StringDescriptor(
        'ModeId', _required, docstring='The Mode Id.')
    ModeType = _StringEnumDescriptor(
        'ModeType', _MODE_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="The mode type, which will be one of {}.".format(_MODE_TYPE_VALUES))


class FullImageType(Serializable):
    """The full image attributes"""
    _tag = 'FullImage'
    _fields = ('NumRows', 'NumCols')
    _required = _fields
    # descriptors
    NumRows = _IntegerDescriptor(
        'NumRows', _required, strict=DEFAULT_STRICT, docstring='The number of rows.')
    NumCols = _IntegerDescriptor(
        'NumCols', _required, strict=DEFAULT_STRICT, docstring='The number of columns.')


####################################################################
# Direct building blocks for SICD
#############
# CollectionInfoType section

class CollectionInfoType(Serializable):
    """The collection information container."""
    _tag = 'CollectionInfo'
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
        'CollectorName', _required, strict=DEFAULT_STRICT, docstring='The collector name.')
    IlluminatorName = _StringDescriptor(
        'IlluminatorName', _required, strict=DEFAULT_STRICT, docstring='The illuminator name.')
    CoreName = _StringDescriptor(
        'CoreName', _required, strict=DEFAULT_STRICT, docstring='The core name.')
    CollectType = _StringEnumDescriptor(
        'CollectType', _COLLECT_TYPE_VALUES, _required,
        docstring="The collect type, one of {}".format(_COLLECT_TYPE_VALUES))
    RadarMode = _SerializableDescriptor(
        'RadarMode', RadarModeType, _required, strict=DEFAULT_STRICT, docstring='The radar mode')
    Classification = _StringDescriptor(
        'Classification', _required, strict=DEFAULT_STRICT, docstring='The classification.')
    # list type descriptors
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The parameters objects list.')
    CountryCodes = _StringListDescriptor(
        'CountryCodes', _required, strict=DEFAULT_STRICT, docstring="The country code list.")


###############
# ImageCreation section


class ImageCreationType(Serializable):
    """The image creation data container."""
    _fields = ('Application', 'DateTime', 'Site', 'Profile')
    _required = ()
    # descriptors
    Application = _StringDescriptor('Application', _required, strict=DEFAULT_STRICT, docstring='The application.')
    DateTime = _DateTimeDescriptor(
        'DateTime', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The image creation date/time.')
    Site = _StringDescriptor(
        'Site', _required, strict=DEFAULT_STRICT, docstring='The site.')
    Profile = _StringDescriptor(
        'Profile', _required, strict=DEFAULT_STRICT, docstring='The profile.')


###############
# ImageData section


class ImageDataType(Serializable):
    """The image data container."""
    _collections_tags = {
        'AmpTable': {'array': True, 'child_tag': 'Amplitude'},
        'ValidData': {'array': True, 'child_tag': 'Vertex'},
    }
    _fields = ('PixelType', 'AmpTable', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel',
                'ValidData')
    _required = ('PixelType', 'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'FullImage', 'SCPPixel')
    _numeric_format = {'AmpTable': '0.8f'}  # TODO: precision for AmpTable?
    _PIXEL_TYPE_VALUES = ("RE32F_IM32F", "RE16I_IM16I", "AMP8I_PHS8I")
    # descriptors
    PixelType = _StringEnumDescriptor(
        'PixelType', _PIXEL_TYPE_VALUES, _required, strict=True,
        docstring="The PixelType attribute which specifies the interpretation of the file data")
    NumRows = _IntegerDescriptor(
        'NumRows', _required, strict=DEFAULT_STRICT, docstring='The number of Rows')
    NumCols = _IntegerDescriptor(
        'NumCols', _required, strict=DEFAULT_STRICT, docstring='The number of Columns')
    FirstRow = _IntegerDescriptor(
        'FirstRow', _required, strict=DEFAULT_STRICT, docstring='The first row')
    FirstCol = _IntegerDescriptor(
        'FirstCol', _required, strict=DEFAULT_STRICT, docstring='The first column')
    FullImage = _SerializableDescriptor(
        'FullImage', FullImageType, _required, strict=DEFAULT_STRICT, docstring='The full image')
    SCPPixel = _SerializableDescriptor(
        'SCPPixel', RowColType, _required, strict=DEFAULT_STRICT, docstring='The SCP Pixel')
    ValidData = _SerializableArrayDescriptor(
        'ValidData', RowColArrayElement, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=3, docstring='The valid data area array.')
    AmpTable = _FloatArrayDescriptor(
        'AmpTable', _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=256, maximum_length=256,
        docstring="The amplitude look-up table. This must be defined if PixelType == 'AMP8I_PHS8I'")

    def _basic_validity_check(self):
        condition = super(ImageDataType, self)._basic_validity_check()
        if (self.PixelType == 'AMP8I_PHS8I') and (self.AmpTable is None):
            logging.warning("We have `PixelType='AMP8I_PHS8I'` and `AmpTable` is not defined for ImageDataType.")
            condition = False
        if (self.ValidData is not None) and (len(self.ValidData) < 3):
            logging.warning("We have `ValidData` defined, with fewer than 3 entries.")
            condition = False
        return condition


###############
# GeoInfo section


class GeoInfoType(Serializable):
    """The GeoInfo container."""
    _tag = 'GeoInfo'
    _collections_tags = {
        'Descriptions': {'array': False, 'child_tag': 'Desc'},
        'Line': {'array': True, 'child_tag': 'Endpoint'},
        'Polygon': {'array': True, 'child_tag': 'Vertex'}, }
    _fields = ('name', 'Descriptions', 'Point', 'Line', 'Polygon')
    _required = ('name', )
    _set_as_attribute = ('name', )
    # descriptors
    name = _StringDescriptor(
        'name', _required, strict=True, docstring='The name.')
    Descriptions = _SerializableArrayDescriptor(
        'Descriptions', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The descriptions list.')
    Point = _SerializableDescriptor(
        'Point', LatLonRestrictionType, _required, strict=DEFAULT_STRICT,
        docstring='A geographic point with WGS-84 coordinates.')
    Line = _SerializableArrayDescriptor(
        'Line', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='A geographic line (array) with WGS-84 coordinates.')
    Polygon = _SerializableArrayDescriptor(
        'Polygon', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=3,
        docstring='A geographic polygon (array) with WGS-84 coordinates.')
    # TODO: is the standard really self-referential here? I find that confusing.
    # TODO: validate the choice - exactly one of Point/Line/Polygon should be populated.


###############
# GeoData section


class SCPType(Serializable):
    """The Scene Center Point container"""
    _tag = 'SCP'
    _fields = ('ECF', 'LLH')
    _required = _fields  # isn't this redundant?
    ECF = _SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT, docstring='The ECF coordinates.')
    LLH = _SerializableDescriptor(
        'LLH', LatLonHAERestrictionType, _required, strict=DEFAULT_STRICT,
        docstring='The WGS 84 coordinates.')


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
        docstring='The Earth Model.'.format(_EARTH_MODEL_VALUES))
    SCP = _SerializableDescriptor(
        'SCP', SCPType, _required, strict=DEFAULT_STRICT, docstring='The Scene Center Point.')
    ImageCorners = _SerializableArrayDescriptor(
        'ImageCorners', LatLonCornerStringType, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring='The geographic image corner points array.')
    ValidData = _SerializableArrayDescriptor(
        'ValidData', LatLonArrayElementType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=3,
        docstring='The full image array includes both valid data and some zero filled pixels.')
    GeoInfos = _SerializableArrayDescriptor(
        'GeoInfos', GeoInfoType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Relevant geographic features list.')


################
# DirParam section


class WgtTypeType(Serializable):
    """The weight type parameters of the direction parameters"""
    _fields = ('WindowName', 'Parameters')
    _required = ('WindowName', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    WindowName = _StringDescriptor(
        'WindowName', _required, strict=DEFAULT_STRICT, docstring='The window name.')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, required=_required, strict=DEFAULT_STRICT,
        docstring='The parameters list.')


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
        docstring='Unit vector in the increasing (row/col) direction (ECF) at the SCP pixel.')
    SS = _FloatDescriptor(
        'SS', _required, strict=DEFAULT_STRICT,
        docstring='Sample spacing in the increasing (row/col) direction. Precise spacing at the SCP.')
    ImpRespWid = _FloatDescriptor(
        'ImpRespWid', _required, strict=DEFAULT_STRICT,
        docstring='Half power impulse response width in the increasing (row/col) direction. Measured at the SCP.')
    Sgn = _IntegerEnumDescriptor(
        'Sgn', (1, -1), _required, strict=DEFAULT_STRICT,
        docstring='Sign for exponent in the DFT to transform the (row/col) dimension to '
                  'spatial frequency dimension.')
    ImpRespBW = _FloatDescriptor(
        'ImpRespBW', _required, strict=DEFAULT_STRICT,
        docstring='Spatial bandwidth in (row/col) used to form the impulse response in the (row/col) direction. '
                  'Measured at the center of support for the SCP.')
    KCtr = _FloatDescriptor(
        'KCtr', _required, strict=DEFAULT_STRICT,
        docstring='Center spatial frequency in the given dimension. '
                  'Corresponds to the zero frequency of the DFT in the given (row/col) direction.')
    DeltaK1 = _FloatDescriptor(
        'DeltaK1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum (row/col) offset from KCtr of the spatial frequency support for the image.')
    DeltaK2 = _FloatDescriptor(
        'DeltaK2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum (row/col) offset from KCtr of the spatial frequency support for the image.')
    DeltaKCOAPoly = _SerializableDescriptor(
        'DeltaKCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Offset from KCtr of the center of support in the given (row/col) spatial frequency. '
                  'The polynomial is a function of image given (row/col) coordinate (variable 1) and '
                  'column coordinate (variable 2).')
    WgtType = _SerializableDescriptor(
        'WgtType', WgtTypeType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters describing aperture weighting type applied in the spatial frequency domain '
                  'to yield the impulse response in the given(row/col) direction.')
    WgtFunct = _FloatArrayDescriptor(
        'WgtFunct', _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=2,
        docstring='Sampled aperture amplitude weighting function (array) applied to form the SCP impulse '
                  'response in the given (row/col) direction.')

    def _basic_validity_check(self):
        condition = super(DirParamType, self)._basic_validity_check()
        if (self.WgtFunct is not None) and (self.WgtFunct.size < 2):
            logging.warning(
                'The WgtFunct array has been defined in DirParamType, but there are fewer than 2 entries.')
            condition = False
        return condition


###############
# GridType section


class GridType(Serializable):
    """Collection grid details container"""
    _fields = ('ImagePlane', 'Type', 'TimeCOAPoly', 'Row', 'Col')
    _required = _fields
    _IMAGE_PLANE_VALUES = ('SLANT', 'GROUND', 'OTHER')
    _TYPE_VALUES = ('RGAZIM', 'RGZERO', 'XRGYCR', 'XCTYAT', 'PLANE')
    # descriptors
    ImagePlane = _StringEnumDescriptor(
        'ImagePlane', _IMAGE_PLANE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="The image plane. Possible values are {}".format(_IMAGE_PLANE_VALUES))
    Type = _StringEnumDescriptor(
        'Type', _TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="The possible grid type enumeration.")
    TimeCOAPoly = _SerializableDescriptor(
        'TimeCOAPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring="*Time of Center Of Aperture* as a polynomial function of image coordinates. "
                  "The polynomial is a function of image row coordinate (variable 1) and column coordinate "
                  "(variable 2).")
    Row = _SerializableDescriptor(
        'Row', DirParamType, _required, strict=DEFAULT_STRICT,
        docstring="Row direction parameters.")
    Col = _SerializableDescriptor(
        'Col', DirParamType, _required, strict=DEFAULT_STRICT,
        docstring="Column direction parameters.")


##############
# TimelineType section


class IPPSetType(Serializable):
    """The Inter-Pulse Parameter array element container."""
    # NOTE that this is simply defined as a child class ("Set") of the TimelineType in the SICD standard
    #   Defining it at root level clarifies the documentation, and giving it a more descriptive name is
    #   appropriate.
    _tag = 'Set'
    _fields = ('TStart', 'TEnd', 'IPPStart', 'IPPEnd', 'IPPPoly', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    TStart = _FloatDescriptor(
        'TStart', _required, strict=DEFAULT_STRICT,
        docstring='IPP start time relative to collection start time, i.e. offsets in seconds.')
    TEnd = _FloatDescriptor(
        'TEnd', _required, strict=DEFAULT_STRICT,
        docstring='IPP end time relative to collection start time, i.e. offsets in seconds.')
    IPPStart = _IntegerDescriptor(
        'IPPStart', _required, strict=True, docstring='Starting IPP index for the period described.')
    IPPEnd = _IntegerDescriptor(
        'IPPEnd', _required, strict=True, docstring='Ending IPP index for the period described.')
    IPPPoly = _SerializableDescriptor(
        'IPPPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='IPP index polynomial coefficients yield IPP index as a function of time for TStart to TEnd.')
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The element array index.')


class TimelineType(Serializable):
    """The details for the imaging collection timeline."""
    _fields = ('CollectStart', 'CollectDuration', 'IPP')
    _required = ('CollectStart', 'CollectDuration', )
    _collections_tags = {'IPP': {'array': True, 'child_tag': 'Set'}}
    # descriptors
    CollectStart = _DateTimeDescriptor(
        'CollectStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The collection start time. The default precision will be microseconds.')
    CollectDuration = _FloatDescriptor(
        'CollectDuration', _required, strict=DEFAULT_STRICT,
        docstring='The duration of the collection in seconds.')
    IPP = _SerializableArrayDescriptor(
        'IPP', IPPSetType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring="The Inter-Pulse Period (IPP) parameters array.")


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
                  'seconds since start of collection.')
    GRPPoly = _SerializableDescriptor(
        'GRPPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Ground Reference Point (GRP) position polynomial in ECF as a function of elapsed '
                  'seconds since start of collection.')
    TxAPCPoly = _SerializableDescriptor(
        'TxAPCPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Transmit Aperture Phase Center (APC) position polynomial in ECF as a function of '
                  'elapsed seconds since start of collection.')
    RcvAPC = _SerializableArrayDescriptor(
        'RcvAPC', XYZPolyAttributeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Receive Aperture Phase Center polynomials array. '
                  'Each polynomial has output in ECF, and represents a function of elapsed seconds since start of '
                  'collection.')


##################
# RadarCollectionType section


class TxFrequencyType(Serializable):
    """The transmit frequency range"""
    _tag = 'TxFrequency'
    _fields = ('Min', 'Max')
    _required = _fields
    # descriptors
    Min = _FloatDescriptor(
        'Min', required=_required, strict=DEFAULT_STRICT,
        docstring='The transmit minimum frequency in Hz.')
    Max = _FloatDescriptor(
        'Max', required=_required, strict=DEFAULT_STRICT,
        docstring='The transmit maximum frequency in Hz.')


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
        docstring='Transmit pulse length in seconds.')
    TxRFBandwidth = _FloatDescriptor(
        'TxRFBandwidth', _required, strict=DEFAULT_STRICT,
        docstring='Transmit RF bandwidth of the transmit pulse in Hz.')
    TxFreqStart = _FloatDescriptor(
        'TxFreqStart', _required, strict=DEFAULT_STRICT,
        docstring='Transmit Start frequency for Linear FM waveform in Hz, may be relative to reference frequency.')
    TxFMRate = _FloatDescriptor(
        'TxFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Transmit FM rate for Linear FM waveform in Hz/second.')
    RcvWindowLength = _FloatDescriptor(
        'RcvWindowLength', _required, strict=DEFAULT_STRICT,
        docstring='Receive window duration in seconds.')
    ADCSampleRate = _FloatDescriptor(
        'ADCSampleRate', _required, strict=DEFAULT_STRICT,
        docstring='Analog-to-Digital Converter sampling rate in samples/second.')
    RcvIFBandwidth = _FloatDescriptor(
        'RcvIFBandwidth', _required, strict=DEFAULT_STRICT,
        docstring='Receive IF bandwidth in Hz.')
    RcvFreqStart = _FloatDescriptor(
        'RcvFreqStart', _required, strict=DEFAULT_STRICT,
        docstring='Receive demodulation start frequency in Hz, may be relative to reference frequency.')
    RcvFMRate = _FloatDescriptor(
        'RcvFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Receive FM rate. Should be 0 if RcvDemodType = "CHIRP".')
    RcvDemodType = _StringEnumDescriptor(
        'RcvDemodType', _DEMOD_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="Receive demodulation used when Linear FM waveform is used on transmit.")

    def __init__(self, **kwargs):
        super(WaveformParametersType, self).__init__(**kwargs)
        if self.RcvDemodType == 'CHIRP':
            self.RcvFMRate = 0

    def _basic_validity_check(self):
        valid = super(WaveformParametersType, self)._basic_validity_check()
        if (self.RcvDemodType == 'CHIRP') and (self.RcvFMRate != 0):
            logging.warning('In WaveformParameters, we have RcvDemodType == "CHIRP" and self.RcvFMRate non-zero.')
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
        'WFIndex', _required, strict=DEFAULT_STRICT, docstring='The waveform number for this step.')
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION2_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Transmit signal polarization for this step.')
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The step index')


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
        docstring='Combined Transmit and Receive signal polarization for the channel.')
    RcvAPCIndex = _IntegerDescriptor(
        'RcvAPCIndex', _required, strict=DEFAULT_STRICT,
        docstring='Index of the Receive Aperture Phase Center (Rcv APC). Only include if Receive APC position '
                  'polynomial(s) are included.')
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The parameter index')


class ReferencePointType(Serializable):
    """The reference point definition"""
    _fields = ('ECF', 'Line', 'Sample', 'name')
    _required = _fields
    _set_as_attribute = ('name', )
    # descriptors
    ECF = _SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The geographical coordinates for the reference point.')
    Line = _FloatDescriptor(
        'Line', _required, strict=DEFAULT_STRICT,
        docstring='The Line?')  # TODO: what is this?
    Sample = _FloatDescriptor(
        'Sample', _required, strict=DEFAULT_STRICT,
        docstring='The Sample?')
    name = _StringDescriptor(
        'name', _required, strict=DEFAULT_STRICT,
        docstring='The reference point name.')


class XDirectionType(Serializable):
    """The X direction of the collect"""
    _fields = ('UVectECF', 'LineSpacing', 'NumLines', 'FirstLine')
    _required = _fields
    # descriptors
    ECF = _SerializableDescriptor(
        'UVectECF', XYZType,_required, strict=DEFAULT_STRICT,
        docstring='The unit vector')
    LineSpacing = _FloatDescriptor(
        'LineSpacing', _required, strict=DEFAULT_STRICT,
        docstring='The collection line spacing in meters.')
    NumLines = _IntegerDescriptor(
        'NumLines', _required, strict=DEFAULT_STRICT,
        docstring='The number of lines')
    FirstLine = _IntegerDescriptor(
        'FirstLine', _required, strict=DEFAULT_STRICT,
        docstring='The first line')


class YDirectionType(Serializable):
    """The Y direction of the collect"""
    _fields = ('UVectECF', 'LineSpacing', 'NumSamples', 'FirstSample')
    _required = _fields
    # descriptors
    ECF = _SerializableDescriptor(
        'UVectECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The unit vector.')
    LineSpacing = _FloatDescriptor(
        'LineSpacing', _required, strict=DEFAULT_STRICT,
        docstring='The collection line spacing in meters.')
    NumSamples = _IntegerDescriptor(
        'NumSamples', _required, strict=DEFAULT_STRICT,
        docstring='The number of samples.')
    FirstSample = _IntegerDescriptor(
        'FirstSample', _required, strict=DEFAULT_STRICT,
        docstring='The first sample.')


class SegmentArrayElement(Serializable):
    """The reference point definition"""
    _fields = ('StartLine', 'StartSample', 'EndLine', 'EndSample', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    StartLine = _IntegerDescriptor(
        'StartLine', _required, strict=DEFAULT_STRICT,
        docstring='The starting line number.')
    StartSample = _IntegerDescriptor(
        'StartSample', _required, strict=DEFAULT_STRICT,
        docstring='The starting sample number.')
    EndLine = _IntegerDescriptor(
        'EndLine', _required, strict=DEFAULT_STRICT,
        docstring='The ending line number.')
    EndSample = _IntegerDescriptor(
        'EndSample', _required, strict=DEFAULT_STRICT,
        docstring='The ending sample number.')
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT,
        docstring='The array index.')


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
        docstring='The reference point.')
    XDir = _SerializableDescriptor(
        'XDir', XDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The X direction collection plane parameters.')
    YDir = _SerializableDescriptor(
        'YDir', YDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The Y direction collection plane parameters.')
    SegmentList = _SerializableArrayDescriptor(
        'SegmentList', SegmentArrayElement, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The segment list.')
    Orientation = _StringEnumDescriptor(
        'Orientation', _ORIENTATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The orientation value.')


class AreaType(Serializable):
    """The collection area"""
    _fields = ('Corner', 'Plane')
    _required = ('Corner', )
    _collections_tags = {
        'Corner': {'array': False, 'child_tag': 'ACP'},}
    # descriptors
    Corner = _SerializableArrayDescriptor(
        'Corner', LatLonHAECornerRestrictionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring='The collection area corner point definition array.')
    Plane = _SerializableDescriptor(
        'Plane', ReferencePlaneType, _required, strict=DEFAULT_STRICT,
        docstring='The collection area reference plane.')


class RadarCollectionType(Serializable):
    """The Radar Collection Type"""
    _tag = 'RadarCollection'
    _fields = ('TxFrequency', 'RefFreqIndex', 'Waveform', 'TxPolarization', 'TxSequence', 'RcvChannels',
                'Area', 'Parameters')
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
        docstring='The transmit frequency range.')
    RefFreqIndex = _IntegerDescriptor(
        'RefFreqIndex', _required, strict=DEFAULT_STRICT,
        docstring='The reference frequency index, if applicable.')
    Waveform = _SerializableArrayDescriptor(
        'Waveform', WaveformParametersType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The waveform parameters array.')
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION1_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The transmit polarization.')  # TODO: iff SEQUENCE, then TxSequence is defined?
    TxSequence = _SerializableArrayDescriptor(
        'TxSequence', TxStepType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The transmit sequence parameters array.')
    RcvChannels = _SerializableArrayDescriptor(
        'RcvChannels', ChanParametersType, _collections_tags,
        _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Transmit receive sequence step details array.')
    Area = _SerializableDescriptor(
        'Area', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='The collection area.')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='A parameters list.')


###############
# ImageFormationType section


class RcvChanProcType(Serializable):
    """Receive Channel Process Type"""
    _fields = ('NumChanProc', 'PRFScaleFactor', 'ChanIndices')
    _required = ('NumChanProc', 'ChanIndices')  # TODO: make proper descriptor
    _collections_tags = {
        'ChanIndices': {'array': False, 'child_tag': 'ChanIndex'}}
    # descriptors
    NumChanProc = _IntegerDescriptor(
        'NumChanProc', _required, strict=DEFAULT_STRICT,
        docstring='The Num Chan Proc?')
    PRFScaleFactor = _FloatDescriptor(
        'PRFScaleFactor', _required, strict=DEFAULT_STRICT,
        docstring='The PRF scale factor.')
    ChanIndices = _IntegerListDescriptor(
        'ChanIndices', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The channel index list.')


class TxFrequencyProcType(Serializable):
    """The transmit frequency range"""
    _tag = 'TxFrequencyProc'
    _fields = ('MinProc', 'MaxProc')
    _required = _fields
    # descriptors
    MinProc = _FloatDescriptor(
        'MinProc', _required, strict=DEFAULT_STRICT,
        docstring='The transmit minimum frequency in Hz.')
    MaxProc = _FloatDescriptor(
        'MaxProc', _required, strict=DEFAULT_STRICT,
        docstring='The transmit maximum frequency in Hz.')


class ProcessingType(Serializable):
    """The transmit frequency range"""
    _tag = 'Processing'
    _fields = ('Type', 'Applied', 'Parameters')
    _required = ('Type', 'Applied')
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    Type = _StringDescriptor(
        'Type', _required, strict=DEFAULT_STRICT, docstring='The type string.')
    Applied = _BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Whether the given type has been applied.')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The parameters list.')


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
        'A', _required, strict=DEFAULT_STRICT, docstring='The A attribute.')
    F1 = _ComplexDescriptor(
        'F1', _required, strict=DEFAULT_STRICT, docstring='The F1 attribute.')
    F2 = _ComplexDescriptor(
        'F2', _required, strict=DEFAULT_STRICT, docstring='The F2 attribute.')
    Q1 = _ComplexDescriptor(
        'Q1', _required, strict=DEFAULT_STRICT, docstring='The Q1 attribute.')
    Q2 = _ComplexDescriptor(
        'Q2', _required, strict=DEFAULT_STRICT, docstring='The Q2 attribute.')
    Q3 = _ComplexDescriptor(
        'Q3', _required, strict=DEFAULT_STRICT, docstring='The Q3 attribute.')
    Q4 = _ComplexDescriptor(
        'Q4', _required, strict=DEFAULT_STRICT, docstring='The Q4 attribute.')
    GainErrorA = _FloatDescriptor(
        'GainErrorA', _required, strict=DEFAULT_STRICT,
        docstring='The GainErrorA attribute.')
    GainErrorF1 = _FloatDescriptor(
        'GainErrorF1', _required, strict=DEFAULT_STRICT,
        docstring='The GainErrorF1 attribute.')
    GainErrorF2 = _FloatDescriptor(
        'GainErrorF2', _required, strict=DEFAULT_STRICT,
        docstring='The GainErrorF2 attribute.')
    PhaseErrorF1 = _FloatDescriptor(
        'PhaseErrorF1', _required, strict=DEFAULT_STRICT,
        docstring='The PhaseErrorF1 attribute.')
    PhaseErrorF2 = _FloatDescriptor(
        'PhaseErrorF2', _required, strict=DEFAULT_STRICT,
        docstring='The PhaseErrorF2 attribute.')


class PolarizationCalibrationType(Serializable):
    """The polarization calibration"""
    _fields = ('DistortCorrectApplied', 'Distortion')
    _required = _fields
    # descriptors
    DistortCorrectApplied = _BooleanDescriptor(
        'DistortCorrectApplied', _required, strict=DEFAULT_STRICT,
        docstring='Whether the distortion correction has been applied')
    Distortion = _SerializableDescriptor(
        'Distortion', DistortionType, _required, strict=DEFAULT_STRICT,
        docstring='The distortion parameters.')


class ImageFormationType(Serializable):
    """The image formation parameters type"""
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
        docstring='The received channels processed?')
    TxRcvPolarizationProc = _StringEnumDescriptor(
        'TxRcvPolarizationProc', _DUAL_POLARIZATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The transmit/receive polarization.')
    TStartProc = _FloatDescriptor(
        'TStartProc', _required, strict=DEFAULT_STRICT, docstring='The processing start time.')
    TEndProc = _FloatDescriptor(
        'TEndProc', _required, strict=DEFAULT_STRICT, docstring='The processing end time.')
    TxFrequencyProc = _SerializableDescriptor(
        'TxFrequencyProc', TxFrequencyProcType, _required, strict=DEFAULT_STRICT,
        docstring='The processing frequency range.')
    SegmentIdentifier = _StringDescriptor(
        'SegmentIdentifier', _required, strict=DEFAULT_STRICT, docstring='The segment identifier.')
    ImageFormAlgo = _StringEnumDescriptor(
        'ImageFormAlgo', _IMG_FORM_ALGO_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The image formation algorithm used.')
    STBeamComp = _StringEnumDescriptor(
        'STBeamComp', _ST_BEAM_COMP_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The ST beam comp.')
    ImageBeamComp = _StringEnumDescriptor(
        'ImageBeamComp', _IMG_BEAM_COMP_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The image beam comp.')
    AzAutofocus = _StringEnumDescriptor(
        'AzAutofocus', _AZ_AUTOFOCUS_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The azimuth autofocus.')
    RgAutofocus = _StringEnumDescriptor(
        'RgAutofocus', _RG_AUTOFOCUS_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The range autofocus.')
    Processings = _SerializableArrayDescriptor(
        'Processings', ProcessingType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The processing collection.')
    PolarizationCalibration = _SerializableDescriptor(
        'PolarizationCalibration', PolarizationCalibrationType, _required, strict=DEFAULT_STRICT,
        docstring='The polarization calibration details.')


###############
# SCPCOAType section


class SCPCOAType(Serializable):
    """The scene center point - COA?"""
    _fields = (
        'SCPTime', 'ARPPos', 'ARPVel', 'ARPAcc', 'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAng',
        'GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng')
    _required = _fields
    # class variables
    _SIDE_OF_TRACK_VALUES = ('L', 'R')
    # descriptors
    SCPTime = _FloatDescriptor(
        'SCPTime', _required, strict=DEFAULT_STRICT, docstring='The scene center point time in seconds?')
    ARPPos = _SerializableDescriptor(
        'ARPPos', XYZType, _required, strict=DEFAULT_STRICT, docstring='The aperture position.')
    ARPVel = _SerializableDescriptor(
        'ARPVel', XYZType, _required, strict=DEFAULT_STRICT, docstring='The aperture velocity.')
    ARPAcc = _SerializableDescriptor(
        'ARPAcc', XYZType, _required, strict=DEFAULT_STRICT, docstring='The aperture acceleration.')
    SideOfTrack = _StringEnumDescriptor(
        'SideOfTrack', _SIDE_OF_TRACK_VALUES, _required, strict=DEFAULT_STRICT, docstring='The side of track.')
    SlantRange = _FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT, docstring='The slant range.')
    GroundRange = _FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT, docstring='The ground range.')
    DopplerConeAng = _FloatDescriptor(
        'DopplerConeAng', _required, strict=DEFAULT_STRICT, docstring='The Doppler cone angle.')
    GrazeAng = _FloatDescriptor(
        'GrazeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.), docstring='The graze angle.')
    IncidenceAng = _FloatDescriptor(
        'IncidenceAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.), docstring='The incidence angle.')
    TwistAng = _FloatDescriptor(
        'TwistAng', _required, strict=DEFAULT_STRICT, bounds=(-90., 90.), docstring='The twist angle.')
    SlopeAng = _FloatDescriptor(
        'SlopeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.), docstring='The slope angle.')
    AzimAng = _FloatDescriptor(
        'AzimAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.), docstring='The azimuth angle.')
    LayoverAng = _FloatDescriptor(
        'LayoverAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.), docstring='The layover angle.')


###############
# RadiometricType section


class NoiseLevelType(Serializable):
    """Noise level type container."""
    _fields = ('NoiseLevelType', 'NoisePoly')
    _required = _fields
    # class variables
    _NOISE_LEVEL_TYPE_VALUES = ('ABSOLUTE', 'RELATIVE')
    # descriptors
    NoiseLevelType = _StringEnumDescriptor(
        'NoiseLevelType', _NOISE_LEVEL_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The noise level type value.')
    NoisePoly = _SerializableDescriptor(
        'NoisePoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='The noise level polynomial.')


class RadiometricType(Serializable):
    """The radiometric type container."""
    _fields = ('NoiseLevel', 'RCSSFPoly', 'SigmaZeroSFPoly', 'BetaZeroSFPoly', 'GammaZeroSFPoly')
    _required = ()
    # descriptors
    NoiseLevel = _SerializableDescriptor(
        'NoiseLevel', NoiseLevelType, _required, strict=DEFAULT_STRICT, docstring='The noise level.')
    RCSSFPoly = _SerializableDescriptor(
        'RCSSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='The RCSSF polynomial.')
    SigmaZeroSFPoly = _SerializableDescriptor(
        'SigmaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='The Sigma_0 SF polynomial.')
    BetaZeroSFPoly = _SerializableDescriptor(
        'BetaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='The Beta_0 SF polynomial.')
    GammaZeroSFPoly = _SerializableDescriptor(
        'GammaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='The Gamma_0 SF polynomial.')


###############
# AntennaType section


class EBType(Serializable):
    """"""
    _fields = ('DCXPoly', 'DCYPoly')
    _required = _fields
    # descriptors
    DCXPoly = _SerializableDescriptor(
        'DCXPoly', Poly1DType, _required, strict=DEFAULT_STRICT, docstring='The DCX polynomial.')
    DCYPoly = _SerializableDescriptor(
        'DCYPoly', Poly1DType, _required, strict=DEFAULT_STRICT, docstring='The DCY polynomial.')


class AntParamType(Serializable):
    """The antenna parameters container."""
    _fields = ('XAxisPoly', 'YAxisPoly', 'FreqZero', 'EB', 'Array', 'Elem', 'GainBSPoly', 'EBFreqShift', 'MLFreqDilation')
    _required = ('XAxisPoly', 'YAxisPoly', 'FreqZero', 'Array')
    # descriptors
    XAxisPoly = _SerializableDescriptor(
        'XAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT, docstring='The X axis polynomial.')
    YAxisPoly = _SerializableDescriptor(
        'YAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT, docstring='The Y axis polynomial.')
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT, docstring='The frequency zero.')
    EB = _SerializableDescriptor(
        'EB', EBType, _required, strict=DEFAULT_STRICT, docstring='The EB.')
    Array = _SerializableDescriptor(
        'Array', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='The antenna array gain/phase polynomial.')
    Elem = _SerializableDescriptor(
        'Elem', GainPhasePolyType, _required, strict=DEFAULT_STRICT, docstring='The element gain/phase polynomial.')
    GainBSPoly = _SerializableDescriptor(
        'GainBSPoly', Poly1DType, _required, strict=DEFAULT_STRICT, docstring='The Gain BS polynomial.')
    EBFreqShift = _BooleanDescriptor(
        'EBFreqShift', _required, strict=DEFAULT_STRICT, docstring='The EB shift boolean.')
    MLFreqDilation = _BooleanDescriptor(
        'MLFreqDilation', _required, strict=DEFAULT_STRICT, docstring='The ML frequency dilation boolean.')


class AntennaType(Serializable):
    """The antenna parameters."""
    _fields = ('Tx', 'Rcv', 'TwoWay')
    _required = ()
    # descriptors
    Tx = _SerializableDescriptor(
        'Tx', AntParamType, _required, strict=DEFAULT_STRICT, docstring='The transmit antenna.')
    Rcv = _SerializableDescriptor(
        'Rcv', AntParamType, _required, strict=DEFAULT_STRICT, docstring='The receive antenna.')
    TwoWay = _SerializableDescriptor(
        'TwoWay', AntParamType, _required, strict=DEFAULT_STRICT, docstring='The bidirectional transmit/receive antenna.')


###############
# ErrorStatisticsType section


class CompositeSCPErrorType(Serializable):
    """The composite SCP container for the error statistics."""
    _fields = ('Rg', 'Az', 'RgAz')
    _required = _fields
    # descriptors
    Rg = _FloatDescriptor(
        'Rg', _required, strict=DEFAULT_STRICT, docstring='The range.')
    Az = _FloatDescriptor(
        'Az', _required, strict=DEFAULT_STRICT, docstring='The azimuth.')
    RgAz = _FloatDescriptor(
        'RgAz', _required, strict=DEFAULT_STRICT, docstring='The range azimuth.')


class CorrCoefsType(Serializable):
    """Correlation coefficients container for Pos Vel Err parameters of the error statistics components."""
    _fields = (
        'P1P2', 'P1P3', 'P1V1', 'P1V2', 'P1V3', 'P2P3', 'P2V1', 'P2V2', 'P2V3',
        'P3V1', 'P3V2', 'P3V3', 'V1V2', 'V1V3', 'V2V3')
    _required = _fields
    # descriptors
    P1P2 = _FloatDescriptor(
        'P1P2', _required, strict=DEFAULT_STRICT, docstring='P1 and P2 correlation coefficient.')
    P1P3 = _FloatDescriptor(
        'P1P3', _required, strict=DEFAULT_STRICT, docstring='P1 and P3 correlation coefficient.')
    P1V1 = _FloatDescriptor(
        'P1V1', _required, strict=DEFAULT_STRICT, docstring='P1 and V1 correlation coefficient.')
    P1V2 = _FloatDescriptor(
        'P1V2', _required, strict=DEFAULT_STRICT, docstring='P1 and V2 correlation coefficient.')
    P1V3 = _FloatDescriptor(
        'P1V3', _required, strict=DEFAULT_STRICT, docstring='P1 and V3 correlation coefficient.')
    P2P3 = _FloatDescriptor(
        'P2P3', _required, strict=DEFAULT_STRICT, docstring='P2 and P3 correlation coefficient.')
    P2V1 = _FloatDescriptor(
        'P2V1', _required, strict=DEFAULT_STRICT, docstring='P2 and V1 correlation coefficient.')
    P2V2 = _FloatDescriptor(
        'P2V2', _required, strict=DEFAULT_STRICT, docstring='P2 and V2 correlation coefficient.')
    P2V3 = _FloatDescriptor(
        'P2V3', _required, strict=DEFAULT_STRICT, docstring='P2 and V3 correlation coefficient.')
    P3V1 = _FloatDescriptor(
        'P3V1', _required, strict=DEFAULT_STRICT, docstring='P3 and V1 correlation coefficient.')
    P3V2 = _FloatDescriptor(
        'P3V2', _required, strict=DEFAULT_STRICT, docstring='P3 and V2 correlation coefficient.')
    P3V3 = _FloatDescriptor(
        'P3V3', _required, strict=DEFAULT_STRICT, docstring='P3 and V3 correlation coefficient.')
    V1V2 = _FloatDescriptor(
        'V1V2', _required, strict=DEFAULT_STRICT, docstring='V1 and V2 correlation coefficient.')
    V1V3 = _FloatDescriptor(
        'V1V3', _required, strict=DEFAULT_STRICT, docstring='V1 and V3 correlation coefficient.')
    V2V3 = _FloatDescriptor(
        'V2V3', _required, strict=DEFAULT_STRICT, docstring='V2 and V3 correlation coefficient.')


class PosVelErrType(Serializable):
    """The Pos Vel Err container for the error statistics components."""
    _fields = ('Frame', 'P1', 'P2', 'P3', 'V1', 'V2', 'V3', 'CorrCoefs', 'PositionDecorr')
    _required = ('Frame', 'P1', 'P2', 'P3', 'V1', 'V2', 'V3')
    # class variables
    _FRAME_VALUES = ('ECF', 'RIC_ECF', 'RIC_ECI')
    # descriptors
    Frame = _StringEnumDescriptor(
        'Frame', _FRAME_VALUES, _required, strict=DEFAULT_STRICT, docstring='The frame of reference?')
    P1 = _FloatDescriptor('P1', _required, strict=DEFAULT_STRICT, docstring='')
    P2 = _FloatDescriptor('P2', _required, strict=DEFAULT_STRICT, docstring='')
    P3 = _FloatDescriptor('P3', _required, strict=DEFAULT_STRICT, docstring='')
    V1 = _FloatDescriptor('V1', _required, strict=DEFAULT_STRICT, docstring='')
    V2 = _FloatDescriptor('V2', _required, strict=DEFAULT_STRICT, docstring='')
    V3 = _FloatDescriptor('V3', _required, strict=DEFAULT_STRICT, docstring='')
    CorrCoefs = _SerializableDescriptor(
        'CorrCoefs', CorrCoefsType, _required, strict=DEFAULT_STRICT, docstring='The correlation coefficients.')
    PositionDecorr = _SerializableDescriptor(
        'PositionDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='The position decorrelation function.')


class RadarSensorErrorType(Serializable):
    """The radar sensor container for the error statistics components."""
    _fields = ('RangeBias', 'ClockFreqSF', 'TransmitFreqSF', 'RangeBiasDecorr')
    _required = ('RangeBias', )
    # descriptors
    RangeBias = _FloatDescriptor(
        'RangeBias', _required, strict=DEFAULT_STRICT, docstring='The range bias.')
    ClockFreqSF = _FloatDescriptor(
        'ClockFreqSF', _required, strict=DEFAULT_STRICT, docstring='The clock frequency SF.')
    TransmitFreqSF = _FloatDescriptor(
        'TransmitFreqSF', _required, strict=DEFAULT_STRICT, docstring='The tramsit frequency SF.')
    RangeBiasDecorr = _SerializableDescriptor(
        'RangeBiasDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='The range bias decorrelation function.')


class TropoErrorType(Serializable):
    """The Troposphere error container for the error statistics components."""
    _fields = ('TropoRangeVertical', 'TropoRangeSlant', 'TropoRangeDecorr')
    _required = ()
    # descriptors
    TropoRangeVertical = _FloatDescriptor(
        'TropoRangeVertical', _required, strict=DEFAULT_STRICT, docstring='The Troposphere vertical range.')
    TropoRangeSlant = _FloatDescriptor(
        'TropoRangeSlant', _required, strict=DEFAULT_STRICT, docstring='The Troposphere slant range.')
    TropoRangeDecorr = _SerializableDescriptor(
        'TropoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='The Troposphere range decorrelation function.')


class IonoErrorType(Serializable):
    """The Ionosphere error container for the error statistics components."""
    _fields = ('IonoRangeVertical', 'IonoRangeSlant', 'IonoRgRgRateCC', 'IonoRangeDecorr')
    _required = ('IonoRgRgRateCC', )
    # descriptors
    IonoRangeVertical = _FloatDescriptor(
        'IonoRangeVertical', _required, strict=DEFAULT_STRICT, docstring='The Ionosphere vertical range.')
    IonoRangeSlant = _FloatDescriptor(
        'IonoRangeSlant', _required, strict=DEFAULT_STRICT, docstring='The Ionosphere slant range.')
    IonoRgRgRateCC = _FloatDescriptor(
        'IonoRgRgRateCC', _required, strict=DEFAULT_STRICT,
        docstring='The Ionosphere RgRg rate correlation coefficient.')
    IonoRangeDecorr = _SerializableDescriptor(
        'IonoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='The Ionosphere range decorrelation function.')


class ErrorComponentsType(Serializable):
    """The error components container for the error statistics."""
    _fields = ('PosVelErr', 'RadarSensor', 'TropoError', 'IonoError')
    _required = ('PosVelErr', 'RadarSensor')
    # descriptors
    PosVelErr = _SerializableDescriptor(
        'PosVelErr', PosVelErrType, _required, strict=DEFAULT_STRICT,
        docstring='')
    RadarSensor = _SerializableDescriptor(
        'RadarSensor', RadarSensorErrorType, _required, strict=DEFAULT_STRICT,
        docstring='')
    TropoError = _SerializableDescriptor(
        'TropoError', TropoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='')
    IonoError = _SerializableDescriptor(
        'IonoError', IonoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='')


class ErrorStatisticsType(Serializable):
    """"""
    _fields = ('CompositeSCP', 'Components', 'AdditionalParms')
    _required = ()
    _collections_tags = {'AdditionalParms': {'array': True, 'child_tag': 'Parameter'}}
    # descriptors
    CompositeSCP = _SerializableDescriptor(
        'CompositeSCP', CompositeSCPErrorType, _required, strict=DEFAULT_STRICT,
        docstring='')
    Components = _SerializableDescriptor(
        'Components', ErrorComponentsType, _required, strict=DEFAULT_STRICT,
        docstring='')
    AdditionalParms = _SerializableArrayDescriptor(
        'AdditionalParms',ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Any additional paremeters.')


###############
# MatchInfoType section


class MatchCollectionType(Serializable):
    """The match collection type."""
    _fields = ('CoreName', 'MatchIndex', 'Parameters')
    _required = ('CoreName', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    CoreName = _StringDescriptor(
        'CoreName', _required, strict=DEFAULT_STRICT, docstring='')
    MatchIndex = _IntegerDescriptor(
        'MatchIndex', _required, strict=DEFAULT_STRICT, docstring='')
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The extra parameters.')


class MatchType(Serializable):
    """The is an array element for match information."""
    _fields = ('TypeId', 'CurrentIndex', 'NumMatchCollections', 'MatchCollections')
    _required = ('TypeId',)
    _collections_tags = {'MatchCollections': {'array': False, 'child_tag': 'MatchCollection'}}
    # descriptors
    TypeId = _StringDescriptor(
        'TypeId', _required, strict=DEFAULT_STRICT, docstring='The type identifier.')
    CurrentIndex = _IntegerDescriptor(  # TODO: is this to build an iterator?
        'CurrentIndex', _required, strict=DEFAULT_STRICT, docstring='The current index.')
    MatchCollections = _SerializableArrayDescriptor(
        'MatchCollections', MatchCollectionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The match collections.')

    @property
    def NumMatchCollections(self):
        """The number of elements in the match collection."""
        if self.MatchCollections is None:
            return 0
        else:
            return len(self.MatchCollections)


class MatchInfoType(Serializable):
    """The match information container."""
    _fields = ('NumMatchTypes', 'MatchTypes')
    _required = ('MatchTypes', )
    _collections_tags = {'MatchTypes': {'array': False, 'child_tag': ''}}
    # descriptors
    MatchTypes = _SerializableArrayDescriptor(
        'MatchTypes', MatchType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The match types list.')

    @property
    def NumMatchTypes(self):
        """The number of elements in the match collection."""
        if self.MatchTypes is None:
            return 0
        else:
            return len(self.MatchTypes)


###############
# RgAzCompType section


class RgAzCompType(Serializable):
    """"""
    _fields = ('AzSF', 'KazPoly')
    _required = _fields
    # descriptors
    AzSF = _FloatDescriptor(
        'AzSF', _required, strict=DEFAULT_STRICT, docstring='The azimuth SF.')
    KazPoly = _SerializableDescriptor(
        'KazPoly', Poly1DType, _required, strict=DEFAULT_STRICT, docstring='The Kaz polynomial.')


###############
# PFAType section


class STDeskewType(Serializable):
    """"""
    _fields = ('Applied', 'STDSPhasePoly')
    _required = _fields
    # descriptors
    Applied = _BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Whether the deskew polynomial has been applied.')
    STDSPhasePoly = _SerializableDescriptor(
        'STDSPhasePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='The ST DS Phase polynomial.')


class PFAType(Serializable):
    """"""
    _fields = (
        'FPN', 'IPN', 'PolarAngRefTime', 'PolarAngPoly', 'SpatialFreqSFPoly', 'Krg1', 'Krg2', 'Kaz1', 'Kaz2',
        'StDeskew')
    _required = ('FPN', 'IPN', 'PolarAngRefTime', 'PolarAngPoly', 'SpatialFreqSFPoly', 'Krg1', 'Krg2', 'Kaz1', 'Kaz2')
    # descriptors
    FPN = _SerializableDescriptor(
        'FPN', XYZType, _required, strict=DEFAULT_STRICT, docstring='')
    IPN = _SerializableDescriptor(
        'IPN', XYZType, _required, strict=DEFAULT_STRICT, docstring='')
    PolarAngRefTime = _FloatDescriptor(
        'PolarAngRefTime', _required, strict=DEFAULT_STRICT,
        docstring='Polar angle reference time in seconds.')
    PolarAngPoly = _SerializableDescriptor(
        'PolarAngPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polar angle polynomial.')
    SpatialFreqSFPoly = _SerializableDescriptor(
        'SpatialFreqSFPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Spatial frequency SF polynomial.')
    Krg1 = _FloatDescriptor(
        'Krg1', _required, strict=DEFAULT_STRICT, docstring='')
    Krg2 = _FloatDescriptor(
        'Krg2', _required, strict=DEFAULT_STRICT, docstring='')
    Kaz1 = _FloatDescriptor(
        'Kaz1', _required, strict=DEFAULT_STRICT, docstring='')
    Kaz2 = _FloatDescriptor(
        'Kaz2', _required, strict=DEFAULT_STRICT, docstring='')
    StDeskew = _SerializableDescriptor(
        'StDeskew', STDeskewType, _required, strict=DEFAULT_STRICT, docstring='')


###############
# RMAType section


class RMRefType(Serializable):
    """Range migration reference element of RMA type."""
    _fields = ('PosRef', 'VelRef', 'DopConeAngRef')
    _required = _fields
    # descriptors
    PosRef = _SerializableDescriptor(
        'PosRef', XYZType, _required, strict=DEFAULT_STRICT, docstring='')
    VelRef = _SerializableDescriptor(
        'VelRef', XYZType, _required, strict=DEFAULT_STRICT, docstring='')
    DopConeAngRef = _FloatDescriptor(
        'DopConeAngRef', _required, strict=DEFAULT_STRICT, docstring='')


class INCAType(Serializable):
    """"""
    _fields = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly', 'DopCentroidPoly', 'DopCentroidCOA')
    _required = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly')
    # descriptors
    TimeCAPoly = _SerializableDescriptor(
        'TimeCAPoly', Poly1DType, _required, strict=DEFAULT_STRICT, docstring='')
    R_CA_SCP = _FloatDescriptor(
        'R_CA_SCP', _required, strict=DEFAULT_STRICT, docstring='')
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT, docstring='')
    DRateSFPoly = _SerializableDescriptor(
        'DRateSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='')
    DopCentroidPoly = _SerializableDescriptor(
        'DopCentroidPoly', Poly2DType, _required, strict=DEFAULT_STRICT, docstring='')
    DopCentroidCOA = _BooleanDescriptor(
        'DopCentroidCOA', _required, strict=DEFAULT_STRICT, docstring='')


class RMAType(Serializable):
    """"""
    _fields = ('RMAlgoType', 'ImageType', 'RMAT', 'RMCR', 'INCA')
    _required = ('RMAlgoType', 'ImageType')
    # class variables
    _RM_ALGO_TYPE_VALUES = ('OMEGA_K', 'CSA', 'RG_DOP')
    _IMAGE_TYPE_VALUES = ('RMAT', 'RMCR', 'INCA')
    # descriptors
    RMAlgoType = _StringEnumDescriptor(
        'RMAlgoType', _RM_ALGO_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The range migration algorithm type.')
    ImageType = _StringEnumDescriptor(
        'ImageType', _IMAGE_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The image type.')
    RMAT = _SerializableDescriptor(
        'RMAT', RMRefType, _required, strict=DEFAULT_STRICT, docstring='')
    RMCR = _SerializableDescriptor(
        'RMCR', RMRefType, _required, strict=DEFAULT_STRICT, docstring='')
    INCA = _SerializableDescriptor(
        'INCA', INCAType, _required, strict=DEFAULT_STRICT, docstring='')
    # TODO: validate the choice - exactly one of RMAT, RMCR, INCA should be populated,
    #   and it should be in-keeping with ImageType selection - should be property?

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
    # descriptors
    CollectionInfo = _SerializableDescriptor(
        'CollectionInfo', CollectionInfoType, _required, strict=DEFAULT_STRICT,
        docstring='')
    ImageCreation = _SerializableDescriptor(
        'ImageCreation', ImageCreationType, _required, strict=DEFAULT_STRICT,
        docstring='')
    ImageData = _SerializableDescriptor(
        'ImageData', ImageDataType, _required, strict=DEFAULT_STRICT,
        docstring='')
    GeoData = _SerializableDescriptor(
        'GeoData', GeoDataType, _required, strict=DEFAULT_STRICT,
        docstring='')
    Grid = _SerializableDescriptor(
        'Grid', GridType, _required, strict=DEFAULT_STRICT,
        docstring='')
    Timeline = _SerializableDescriptor(
        'Timeline', TimelineType, _required, strict=DEFAULT_STRICT,
        docstring='')
    Position = _SerializableDescriptor(
        'Position', PositionType, _required, strict=DEFAULT_STRICT,
        docstring='')
    RadarCollection = _SerializableDescriptor(
        'RadarCollection', RadarCollectionType, _required, strict=DEFAULT_STRICT,
        docstring='')
    ImageFormation = _SerializableDescriptor(
        'ImageFormation', ImageFormationType, _required, strict=DEFAULT_STRICT,
        docstring='')
    SCPCOA = _SerializableDescriptor(
        'SCPCOA', SCPCOAType, _required, strict=DEFAULT_STRICT,
        docstring='')
    Radiometric = _SerializableDescriptor(
        'Radiometric', RadiometricType, _required, strict=DEFAULT_STRICT,
        docstring='')
    Antenna = _SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='')
    ErrorStatistics = _SerializableDescriptor(
        'ErrorStatistics', ErrorStatisticsType, _required, strict=DEFAULT_STRICT,
        docstring='')
    MatchInfo = _SerializableDescriptor(
        'MatchInfo', MatchInfoType, _required, strict=DEFAULT_STRICT,
        docstring='')
    RgAzComp = _SerializableDescriptor(
        'RgAzComp', RgAzCompType, _required, strict=DEFAULT_STRICT,
        docstring='')
    PFA = _SerializableDescriptor(
        'PFA', PFAType, _required, strict=DEFAULT_STRICT,
        docstring='')
    RMA = _SerializableDescriptor(
        'RMA', RMAType, _required, strict=DEFAULT_STRICT,
        docstring='')
    # TODO: validate the choice for RgAzComp/PFA/RMA - none are required.
