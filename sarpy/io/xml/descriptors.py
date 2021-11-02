"""
This module contains the base objects for use in base xml/serializable functionality.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from xml.etree import ElementTree
from weakref import WeakKeyDictionary

import numpy
from numpy.linalg import norm

from sarpy.io.xml.base import DEFAULT_STRICT, get_node_value, find_children, \
    Arrayable, ParametersCollection, SerializableArray, \
    parse_str, parse_bool, parse_int, parse_float, parse_complex, parse_datetime, \
    parse_serializable, parse_serializable_list

logger = logging.getLogger(__name__)


class BasicDescriptor(object):
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
                logger.debug(msg)  # NB: this is at debug level to not be too verbose
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
                    logger.debug(  # NB: this is at debuglevel to not be too verbose
                        'Required attribute {} of class {} has been set to None.'.format(
                            self.name, instance.__class__.__name__))
            self.data[instance] = None
            return True
        # note that the remainder must be implemented in each extension
        return False  # this is probably a bad habit, but this returns something for convenience alone


class StringDescriptor(BasicDescriptor):
    """A descriptor for string type"""
    _typ_string = 'str:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        super(StringDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):

        if super(StringDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        self.data[instance] = parse_str(value, self.name, instance)


class StringListDescriptor(BasicDescriptor):
    """A descriptor for properties for an array type item for specified extension of string"""
    _typ_string = 'List[str]:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32

    def __init__(self, name, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None,
                 default_value=None, docstring=None):
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(StringListDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Attribute {} of class {} is a string list of size {},\n\t' \
                      'and must have length at least {}.'.format(
                        self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.error(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Attribute {} of class {} is a string list of size {},\n\t' \
                      'and must have length no greater than {}.'.format(
                        self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.error(msg)
            self.data[instance] = new_value

        if super(StringListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, str):
            set_value([value, ])
        elif isinstance(value, ElementTree.Element):
            set_value([get_node_value(value), ])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], str):
                set_value(value)
            elif isinstance(value[0], ElementTree.Element):
                set_value([get_node_value(nod) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class StringEnumDescriptor(BasicDescriptor):
    """A descriptor for enumerated (specified) string type.
    **This implicitly assumes that the valid entries are upper case.**"""
    _typ_string = 'str:'

    def __init__(self, name, values, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        self.values = values
        super(StringEnumDescriptor, self).__init__(
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
                super(StringEnumDescriptor, self).__set__(instance, value)
            return

        val = parse_str(value, self.name, instance)

        if val in self.values:
            self.data[instance] = val
        else:
            msg = 'Attribute {} of class {} received {},\n\t' \
                  'but values ARE REQUIRED to be one of {}'.format(
                    self.name, instance.__class__.__name__, value, self.values)
            if self.strict:
                raise ValueError(msg)
            else:
                logger.error(msg)
            self.data[instance] = val


class BooleanDescriptor(BasicDescriptor):
    """A descriptor for boolean type"""
    _typ_string = 'bool:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        super(BooleanDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):
        if super(BooleanDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return
        try:
            self.data[instance] = parse_bool(value, self.name, instance)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to `bool`\n\t'
                'for field {} of class {} with exception {} - {}\n\t'
                'Setting value to None, which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None


class IntegerDescriptor(BasicDescriptor):
    """A descriptor for integer type"""
    _typ_string = 'int:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, bounds=None, default_value=None, docstring=None):
        self.bounds = bounds
        super(IntegerDescriptor, self).__init__(
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
        if super(IntegerDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            iv = parse_int(value, self.name, instance)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to `int`\n\t'
                'for field {} of class {} with exception {} - {}.\n\t'
                'Setting value to None, which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return

        if self._in_bounds(iv):
            self.data[instance] = iv
        else:
            msg = 'Attribute {} of class {} is required by standard\n\t' \
                  'to take value between {}. Invalid value {}'.format(
                    self.name, instance.__class__.__name__, self.bounds, iv)
            if self.strict:
                raise ValueError(msg)
            else:
                logger.error(msg)
            self.data[instance] = iv


class IntegerEnumDescriptor(BasicDescriptor):
    """A descriptor for enumerated (specified) integer type"""
    _typ_string = 'int:'

    def __init__(self, name, values, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        self.values = values
        super(IntegerEnumDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)
        if (self.default_value is not None) and (self.default_value not in self.values):
            self.default_value = None

    def _docstring_suffix(self):
        return 'Must take one of the values in {}.'.format(self.values)

    def __set__(self, instance, value):
        if super(IntegerEnumDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            iv = parse_int(value, self.name, instance)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to `int`\n\t'
                'for field {} of class {} with exception {} - {}.\n\t'
                'Setting value to None, which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return

        if iv in self.values:
            self.data[instance] = iv
        else:
            msg = 'Attribute {} of class {} must take value in {}.\n\t' \
                  'Invalid value {}.'.format(
                    self.name, instance.__class__.__name__, self.values, iv)
            if self.strict:
                raise ValueError(msg)
            else:
                logger.error(msg)
            self.data[instance] = iv


class IntegerListDescriptor(BasicDescriptor):
    """A descriptor for integer list type properties"""
    _typ_string = 'list[int]:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(IntegerListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Attribute {} of class {} is an integer list of size {},\n\t' \
                      'and must have size at least {}.'.format(
                        self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.info(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Attribute {} of class {} is an integer list of size {},\n\t' \
                      'and must have size no larger than {}.'.format(
                        self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.info(msg)
            self.data[instance] = new_value

        if super(IntegerListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, int):
            set_value([value, ])
        elif isinstance(value, ElementTree.Element):
            set_value([int(get_node_value(value)), ])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], int):
                set_value(value)
            elif isinstance(value[0], ElementTree.Element):
                set_value([int(get_node_value(nod)) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class FloatDescriptor(BasicDescriptor):
    """A descriptor for float type properties"""
    _typ_string = 'float:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, bounds=None, default_value=None, docstring=None):
        self.bounds = bounds
        super(FloatDescriptor, self).__init__(
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
        if super(FloatDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            iv = parse_float(value, self.name, instance)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to `float`\n\t'
                'for field {} of class {} with exception {} - {}.\n\t'
                'Setting value to None, which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return

        if self._in_bounds(iv):
            self.data[instance] = iv
        else:
            msg = 'Attribute {} of class {}\n\t' \
                  'is required by standard to take value between {}.'.format(
                    self.name, instance.__class__.__name__, self.bounds)
            if self.strict:
                raise ValueError(msg)
            else:
                logger.info(msg)
            self.data[instance] = iv


class FloatListDescriptor(BasicDescriptor):
    """A descriptor for float list type properties"""
    _typ_string = 'list[float]:'
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(FloatListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        def set_value(new_value):
            if len(new_value) < self.minimum_length:
                msg = 'Attribute {} of class {} is an float list of size {},\n\t' \
                      'and must have size at least {}.'.format(
                        self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.info(msg)
            if len(new_value) > self.maximum_length:
                msg = 'Attribute {} of class {} is a float list of size {},\n\t' \
                      'and must have size no larger than {}.'.format(
                        self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.info(msg)
            self.data[instance] = new_value

        if super(FloatListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        if isinstance(value, float):
            set_value([value, ])
        elif isinstance(value, ElementTree.Element):
            set_value([float(get_node_value(value)), ])
        elif isinstance(value, list):
            if len(value) == 0 or isinstance(value[0], float):
                set_value(value)
            elif isinstance(value[0], ElementTree.Element):
                set_value([float(get_node_value(nod)) for nod in value])
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class ComplexDescriptor(BasicDescriptor):
    """A descriptor for complex valued properties"""
    _typ_string = 'complex:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, default_value=None, docstring=None):
        super(ComplexDescriptor, self).__init__(
            name, required, strict=strict, default_value=default_value, docstring=docstring)

    def __set__(self, instance, value):
        if super(ComplexDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            self.data[instance] = parse_complex(value, self.name, instance)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to `complex`\n\t'
                'for field {} of class {} with exception {} - {}.\n\t'
                'Setting value to None, which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None


class FloatArrayDescriptor(BasicDescriptor):
    """A descriptor for float array type properties"""
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32
    _typ_string = 'numpy.ndarray[float64]:'

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT, minimum_length=None, maximum_length=None,
                 docstring=None):

        self.child_tag = tag_dict[name]['child_tag']
        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(FloatArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        def set_value(new_val):
            if len(new_val) < self.minimum_length:
                msg = 'Attribute {} of class {} is a double array of size {},\n\t' \
                      'and must have size at least {}.'.format(
                        self.name, instance.__class__.__name__, value.size, self.minimum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.error(msg)
            if len(new_val) > self.maximum_length:
                msg = 'Attribute {} of class {} is a double array of size {},\n\t' \
                      'and must have size no larger than {}.'.format(
                        self.name, instance.__class__.__name__, value.size, self.maximum_length)
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.error(msg)
            self.data[instance] = new_val

        if super(FloatArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
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

            size = int(value.attrib['size'])
            child_nodes = find_children(value, self.child_tag, xml_ns, xml_ns_key)
            if len(child_nodes) != size:
                raise ValueError(
                    'Field {} of double array type functionality belonging to class {} got a ElementTree element '
                    'with size attribute {}, but has {} child nodes with tag {}.'.format(
                        self.name, instance.__class__.__name__, size, len(child_nodes), self.child_tag))
            new_value = numpy.empty((size,), dtype=numpy.float64)
            for i, node in enumerate(child_nodes):
                new_value[i] = float(get_node_value(node))
            set_value(new_value)
        elif isinstance(value, list):
            # user or json deserialization
            set_value(numpy.array(value, dtype=numpy.float64))
        else:
            raise TypeError(
                'Field {} of class {} got incompatible type {}.'.format(
                    self.name, instance.__class__.__name__, type(value)))


class DateTimeDescriptor(BasicDescriptor):
    """A descriptor for date time type properties"""
    _typ_string = 'numpy.datetime64:'

    def __init__(self, name, required, strict=DEFAULT_STRICT, docstring=None, numpy_datetime_units='us'):
        self.units = numpy_datetime_units  # s, ms, us, ns are likely choices here, depending on needs
        super(DateTimeDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(DateTimeDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return
        self.data[instance] = parse_datetime(value, self.name, instance, self.units)


class FloatModularDescriptor(BasicDescriptor):
    """
    A descriptor for float type which will take values in a range [-limit, limit], set using modular
    arithmetic operations
    """
    _typ_string = 'float:'

    def __init__(self, name, limit, required, strict=DEFAULT_STRICT, docstring=None):
        self.limit = float(limit)
        super(FloatModularDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(FloatModularDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            val = parse_float(value, self.name, instance)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to `float`\n\t'
                'for field {} of class {} with exception {} - {}.\n\t'
                'Setting value to None, which may be against the standard'.format(
                    value, type(value), self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return

        # do modular arithmetic manipulations
        val = (val % (2 * self.limit))  # NB: % and * have same precedence, so it can be super dumb
        self.data[instance] = val if val <= self.limit else val - 2 * self.limit


class SerializableDescriptor(BasicDescriptor):
    """A descriptor for properties of a specified type assumed to be an extension of Serializable"""

    def __init__(self, name, the_type, required, strict=DEFAULT_STRICT, docstring=None):
        self.the_type = the_type
        self._typ_string = str(the_type).strip().split('.')[-1][:-2] + ':'
        super(SerializableDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(SerializableDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            self.data[instance] = parse_serializable(value, self.name, instance, self.the_type)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to Serializable type {}\n\t'
                'for field {} of class {} with exception {} - {}.\n\t'
                'Setting value to None, which may be against the standard.'.format(
                    value, type(value), self.the_type, self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None


class UnitVectorDescriptor(BasicDescriptor):
    """A descriptor for properties of a specified type assumed to be of subtype of Arrayable"""

    def __init__(self, name, the_type, required, strict=DEFAULT_STRICT, docstring=None):
        if not issubclass(the_type, Arrayable):
            raise TypeError(
                'The input type {} for field {} must be a subclass of Arrayable.'.format(the_type, name))
        self.the_type = the_type

        self._typ_string = str(the_type).strip().split('.')[-1][:-2] + ':'
        super(UnitVectorDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(UnitVectorDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            vec = parse_serializable(value, self.name, instance, self.the_type)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to Unit Vector Type type {}\n\t'
                'for field {} of class {} with exception {} - {}.\n\t'
                'Setting value to None, which may be against the standard'.format(
                    value, type(value), self.the_type, self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None
            return None

        # noinspection PyTypeChecker
        coords = vec.get_array(dtype=numpy.float64)
        the_norm = norm(coords)
        if the_norm == 0:
            logger.error(
                'The input for field {} is expected to be made into a unit vector.\n\t'
                'In this case, the norm of the input is 0.\n\t'
                'The value is set to None, which may be against the standard.'.format(
                    self.name))
            self.data[instance] = None
        elif the_norm == 1:
            self.data[instance] = vec
        else:
            self.data[instance] = self.the_type.from_array(coords/the_norm)


class ParametersDescriptor(BasicDescriptor):
    """A descriptor for properties of a Parameter type - that is, dictionary"""

    def __init__(self, name, tag_dict, required, strict=DEFAULT_STRICT, docstring=None):
        self.child_tag = tag_dict[name]['child_tag']
        self._typ_string = 'ParametersCollection:'
        super(ParametersDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(ParametersDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
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


class SerializableListDescriptor(BasicDescriptor):
    """A descriptor for properties of a list or array of specified extension of Serializable"""

    def __init__(self, name, child_type, tag_dict, required, strict=DEFAULT_STRICT, docstring=None):
        self.child_type = child_type
        tags = tag_dict[name]
        self.array = tags.get('array', False)
        if self.array:
            raise ValueError(
                'Attribute {} is populated in the `_collection_tags` dictionary with `array`=True. '
                'This is inconsistent with using SerializableListDescriptor.'.format(name))

        self.child_tag = tags['child_tag']
        self._typ_string = 'List[{}]:'.format(str(child_type).strip().split('.')[-1][:-2])
        super(SerializableListDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(SerializableListDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        try:
            self.data[instance] = parse_serializable_list(value, self.name, instance, self.child_type)
        except Exception as e:
            logger.error(
                'Failed converting {} of type {} to serializable list of type {}\n\t'
                'for field {} of class {} with exception {} - {}.\n\t'
                'Setting value to None, which may be against the standard'.format(
                    value, type(value), self.child_type, self.name, instance.__class__.__name__, type(e), e))
            self.data[instance] = None


class SerializableArrayDescriptor(BasicDescriptor):
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
                'This is inconsistent with using SerializableArrayDescriptor.'.format(name))

        self.child_tag = tags['child_tag']
        self._typ_string = 'numpy.ndarray[{}]:'.format(str(child_type).strip().split('.')[-1][:-2])
        self.array_extension = array_extension

        self.minimum_length = self._DEFAULT_MIN_LENGTH if minimum_length is None else int(minimum_length)
        self.maximum_length = self._DEFAULT_MAX_LENGTH if maximum_length is None else int(maximum_length)
        if self.minimum_length > self.maximum_length:
            raise ValueError(
                'Specified minimum length is {}, while specified maximum length is {}'.format(
                    self.minimum_length, self.maximum_length))
        super(SerializableArrayDescriptor, self).__init__(name, required, strict=strict, docstring=docstring)

    def __set__(self, instance, value):
        if super(SerializableArrayDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
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
