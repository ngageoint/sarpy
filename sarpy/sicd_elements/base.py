"""
This module contains the base objects for use in the SICD elements, and the base serializable functionality.
"""

from xml.etree import ElementTree
from collections import OrderedDict
from datetime import datetime, date
import logging
from weakref import WeakKeyDictionary

import numpy
import numpy.polynomial.polynomial

__classification__ = "UNCLASSIFIED"

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
    _DEFAULT_MAX_LENGTH = 2 ** 32

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
            self.data[instance] = complex(real, imag)
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
            self.data[instance] = complex(real, imag)
        else:
            # from user - this could be dumb
            self.data[instance] = complex(value)


class _FloatArrayDescriptor(_BasicDescriptor):
    """A descriptor for float array type properties"""
    _DEFAULT_MIN_LENGTH = 0
    _DEFAULT_MAX_LENGTH = 2 ** 32
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
            new_value = numpy.empty((size,), dtype=numpy.float64)
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
    _DEFAULT_MAX_LENGTH = 2 ** 32

    def __init__(self, name, child_type, tag_dict, required, strict=DEFAULT_STRICT,
                 minimum_length=None, maximum_length=None, docstring=None):
        self.child_type = child_type
        tags = tag_dict[name]
        self.array = tags.get('array', False)
        self.child_tag = tags['child_tag']
        if self.array:
            self._typ_string = ':obj:`numpy.ndarray` of :obj:`' + str(child_type).strip().split('.')[-1][:-2] + '`:'
        else:
            self._typ_string = ':obj:`list` of :obj:`' + str(child_type).strip().split('.')[-1][:-2] + '`:'

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
            new_value = numpy.empty((size,), dtype=numpy.object)
            for i, entry in enumerate(child_nodes):
                new_value[i] = self.child_type.from_node(entry)
            self.__actual_set(instance, new_value)
        elif isinstance(value, list):
            # this would arrive from users or json deserialization
            if len(value) == 0:
                self.__actual_set(instance, numpy.empty((0,), dtype=numpy.object))
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
            if attribute in kwargs:
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
            Corresponding class instance
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
