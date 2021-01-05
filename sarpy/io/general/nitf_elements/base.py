# -*- coding: utf-8 -*-
"""
Base NITF Header functionality definition.
"""

import logging
from weakref import WeakKeyDictionary
from typing import Union, List, Tuple
from collections import OrderedDict

import numpy

from sarpy.compliance import int_func, integer_types, string_types, bytes_to_string
from .tres.registration import find_tre


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


# Base NITF type

class BaseNITFElement(object):

    @classmethod
    def minimum_length(cls):
        """
        The minimum size in bytes that takes to write this header element.

        Returns
        -------
        int
        """

        raise NotImplementedError

    def get_bytes_length(self):
        """
        Get the length of the serialized bytes array

        Returns
        -------
        int
        """

        raise NotImplementedError

    def to_bytes(self):
        """
        Write the object to a properly packed str.

        Returns
        -------
        bytes
        """

        raise NotImplementedError

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value: bytes|str
            the header string to scrape
        start : int
            the beginning location in the string

        Returns
        -------

        """

        raise NotImplementedError

    def to_json(self):
        """
        Serialize element to a json representation. This is intended to allow
        a simple presentation of the element.

        Returns
        -------
        dict
        """

        raise NotImplementedError

# Basic input and output interpreters

def _get_bytes(val, length):
    if val is None:
        return b''
    elif isinstance(val, integer_types):
        frm_str = '{0:0' + str(length) + 'd}'
        return frm_str.format(val).encode('utf-8')
    elif isinstance(val, string_types):
        frm_str = '{0:' + str(length) + 's}'
        return frm_str.format(val).encode('utf-8')
    elif isinstance(val, bytes):
        if len(val) >= length:
            return val[:length]
        else:
            return val + b'\x00' * (length - len(val))
    else:
        raise TypeError('Unhandled type {}'.format(type(val)))


def _parse_int(val, length, default, name, instance):
    """
    Parse and/or validate the integer input.

    Parameters
    ----------
    val : None|int|bytes
    length : int
    default : None|int

    Returns
    -------
    int
    """

    if val is None:
        return default
    else:
        val = int_func(val)

    if -int_func(10)**(length-1) < val < int_func(10)**length:
        return val
    raise ValueError(
        'Integer {} cannot be rendered as a string of {} characters for '
        'attribute {} of class {}'.format(val, length, name, instance.__class__.__name__))


def _parse_str(val, length, default, name, instance):
    """
    Parse and/or validate the string input.

    Parameters
    ----------
    val : None|str|bytes
    length : int
    default : None|str

    Returns
    -------
    str
    """

    if val is None:
        return default

    if isinstance(val, bytes):
        val = bytes_to_string(val)
    elif not isinstance(val, string_types):
        val = str(val)

    val = val.rstrip()
    if len(val) <= length:
        return val
    else:
        logging.warning(
            'Got string input value of length {} for attribute {} of class {}, '
            'which is longer than the allowed length {}, so '
            'truncating'.format(len(val), name, instance.__class__.__name__, length))
        return val[:length]


def _parse_bytes(val, length, default, name, instance):
    """
    Validate the raw/bytes input.

    Parameters
    ----------
    val : None|bytes
    length : int
    default : None|int

    Returns
    -------
    int
    """

    if val is None:
        return default
    elif isinstance(val, bytes):
        if len(val) <= length:
            return val
        else:
            logging.warning(
                'Got string input value of length {} for attribute {} of class {}, '
                'which is longer than the allowed length {}, so '
                'truncating'.format(len(val), name, instance.__class__.__name__, length))
            return val[:length]
    else:
        raise TypeError(
            'Expected type int or bytes for attribute {} of class {}, '
            'and got {}'.format(name, instance.__class__.__name__, type(val)))


def _parse_nitf_element(val, nitf_type, default_args, name, instance):
    if not issubclass(nitf_type, BaseNITFElement):
        raise TypeError(
            'nitf_type for attribute {} of class {} must be a subclass of '
            'BaseNITFElement'.format(name, nitf_type.__class__.__name__))

    if val is None:
        if default_args is None:
            return None
        return nitf_type(**default_args)
    elif isinstance(val, bytes):
        return nitf_type.from_bytes(val, 0)
    elif isinstance(val, nitf_type):
        return val
    else:
        raise ValueError(
            'Attribute {} for class {} requires an input of type bytes or {}. '
            'Got {}'.format(name, instance.__class__.__name__, nitf_type, type(val)))


# NITF Descriptors

class _BasicDescriptor(object):
    """A descriptor object for reusable properties. Note that is is required that the calling instance is hashable."""
    _typ_string = None

    def __init__(self, name, required, length, docstring=''):
        self.data = WeakKeyDictionary()  # our instance reference dictionary
        # WeakDictionary use is subtle here. A reference to a particular class instance in this dictionary
        # should not be the thing keeping a particular class instance from being destroyed.
        self.name = name
        self.required = required
        self.length = length

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

        if not self.required:
            docstring = '{} {}'.format(docstring, ' **Conditional.**')
        self.__doc__ = docstring

    def _docstring_suffix(self):
        return None

    def _get_default(self, instance):
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

        fetched = self.data.get(instance, None)
        if fetched is not None or not self.required:
            return fetched
        else:
            msg = 'Required field {} of class {} is not populated.'.format(self.name, instance.__class__.__name__)
            raise AttributeError(msg)

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
            default_value = self._get_default(instance)
            if default_value is not None:
                self.data[instance] = default_value
                return True
            elif self.required:
                raise ValueError(
                    'Attribute {} of class {} cannot be assigned None.'.format(self.name, instance.__class__.__name__))
            self.data[instance] = None
            return True
        # note that the remainder must be implemented in each extension
        return False  # this is probably a bad habit, but this returns something for convenience alone


class _StringDescriptor(_BasicDescriptor):
    """A descriptor for string type"""
    _typ_string = 'str:'

    def __init__(self, name, required, length, default_value='', docstring=None):
        self._default_value = default_value
        super(_StringDescriptor, self).__init__(
            name, required, length, docstring=docstring)

    def _get_default(self, instance):
        return self._default_value

    def _docstring_suffix(self):
        if self._default_value is not None and len(self._default_value) > 0:
            return ' Default value is :code:`{}`.'.format(self._default_value)

    def __set__(self, instance, value):

        if super(_StringDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return
        self.data[instance] = _parse_str(value, self.length, self._default_value, self.name, instance)


class _StringEnumDescriptor(_BasicDescriptor):
    """A descriptor for enumerated (specified) string type.
    **The valid entries are case-sensitive and should be stripped of white space on each end.**"""
    _typ_string = 'str:'

    def __init__(self, name, required, length, values, default_value=None, docstring=None):
        self.values = values
        self._default_value = default_value
        super(_StringEnumDescriptor, self).__init__(
            name, required, length, docstring=docstring)
        if (self._default_value is not None) and (self._default_value not in self.values):
            self._default_value = None

    def _get_default(self, instance):
        return self._default_value

    def _docstring_suffix(self):
        suff = ' Takes values in :code:`{}`.'.format(self.values)
        if self._default_value is not None and len(self._default_value) > 0:
            suff += ' Default value is :code:`{}`.'.format(self._default_value)
        return suff

    def __set__(self, instance, value):
        if value is None:
            if self._default_value is not None:
                self.data[instance] = self._default_value
            else:
                super(_StringEnumDescriptor, self).__set__(instance, value)
            return

        val = _parse_str(value, self.length, self._default_value, self.name, instance)

        if val in self.values:
            self.data[instance] = val
        elif self._default_value is not None:
            msg = 'Attribute {} of class {} received {}, but values ARE REQUIRED to be ' \
                  'one of {}. It has been set to the default ' \
                  'value.'.format(self.name, instance.__class__.__name__, value, self.values)
            logging.error(msg)
            self.data[instance] = self._default_value
        else:
            msg = 'Attribute {} of class {} received {}, but values ARE REQUIRED to be ' \
                  'one of {}. This should be resolved, or it may cause unexpected ' \
                  'issues.'.format(self.name, instance.__class__.__name__, value, self.values)
            logging.error(msg)
            self.data[instance] = val


class _IntegerDescriptor(_BasicDescriptor):
    """A descriptor for integer type"""
    _typ_string = 'int:'

    def __init__(self, name, required, length, default_value=0, docstring=None):
        self._default_value = default_value
        super(_IntegerDescriptor, self).__init__(
            name, required, length, docstring=docstring)

    def _get_default(self, instance):
        return self._default_value

    def _docstring_suffix(self):
        if self._default_value is not None:
            return ' Default value is :code:`{}`.'.format(self._default_value)

    def __set__(self, instance, value):
        if super(_IntegerDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        iv = _parse_int(value, self.length, self._default_value, self.name, instance)
        self.data[instance] = iv


class _RawDescriptor(_BasicDescriptor):
    """A descriptor for bytes type"""
    _typ_string = 'bytes:'

    def __init__(self, name, required, length, default_value=None, docstring=None):
        self._default_value = default_value
        super(_RawDescriptor, self).__init__(
            name, required, length, docstring=docstring)

    def _get_default(self, instance):
        return self._default_value

    def __set__(self, instance, value):
        if super(_RawDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        iv = _parse_bytes(value, self.length, self._default_value, self.name, instance)
        self.data[instance] = iv


class _NITFElementDescriptor(_BasicDescriptor):
    """A descriptor for properties of a specified type assumed to be an extension of Serializable"""

    def __init__(self, name, required, the_type, default_args=None, docstring=None):
        self.the_type = the_type
        self._typ_string = str(the_type).strip().split('.')[-1][:-2] + ':'
        self._default_args = default_args
        super(_NITFElementDescriptor, self).__init__(name, required, None, docstring=docstring)

    def _get_default(self, instance):
        if self._default_args is not None:
            return self.the_type(**self._default_args)
        return None

    def __set__(self, instance, value):
        if super(_NITFElementDescriptor, self).__set__(instance, value):  # the None handler...kinda hacky
            return

        self.data[instance] = _parse_nitf_element(value, self.the_type, self._default_args, self.name, instance)


# Concrete NITF element types

class NITFElement(BaseNITFElement):
    _ordering = ()
    _lengths = {}

    def __init__(self, **kwargs):
        for fld in self._ordering:
            try:
                setattr(self, fld, kwargs.get(fld, None))
            except:
                logging.critical('Failed setting attribute {} for class {}'.format(fld, self.__class__))
                raise

    @classmethod
    def minimum_length(cls):
        """
        The minimum size in bytes that takes to write this header element.

        Returns
        -------
        int
        """

        return sum(cls._lengths.values())

    def _get_attribute_length(self, fld):
        if fld not in self._ordering:
            return 0

        if fld in self._lengths:
            return self._lengths[fld]
        else:
            val = getattr(self, fld)
            if val is None:
                return 0
            elif isinstance(val, BaseNITFElement):
                return val.get_bytes_length()
            else:
                raise TypeError(
                    'Unhandled type {} for attribute {} of '
                    'class {}'.format(type(val), fld, self.__class__.__name__))

    def _get_attribute_bytes(self, fld):
        if fld not in self._ordering:
            return b''

        val = getattr(self, fld)
        if isinstance(val, BaseNITFElement):
            return val.to_bytes()
        elif fld in self._lengths:
            return _get_bytes(val, self._lengths[fld])
        else:
            raise ValueError(
                'Unhandled attribute {} for class {}'.format(fld, self.__class__.__name__))

    def get_bytes_length(self):
        """
        Get the length of the serialized bytes array

        Returns
        -------
        int
        """

        return sum(self._get_attribute_length(fld) for fld in self._ordering)

    def to_bytes(self):
        """
        Write the object to a properly packed str.

        Returns
        -------
        bytes
        """

        return b''.join(self._get_attribute_bytes(fld) for fld in self._ordering)

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        """

        Parameters
        ----------
        fields : dict
            The attribute:value dictionary.
        attribute : str
            The attribute name.
        value : bytes
            The bytes array to be parsed.
        start : int
            The present position in `value`.

        Returns
        -------
        int
            The position in `value` after parsing this attribute.
        """

        if attribute not in cls._ordering:
            raise ValueError('Unexpected attribute {}'.format(attribute))

        if attribute in fields:
            return start
        if attribute in cls._lengths:
            end = start + cls._lengths[attribute]
            fields[attribute] = value[start:end]
            return end
        elif hasattr(cls, attribute):
            the_typ = getattr(cls, attribute).the_type
            assert issubclass(the_typ, BaseNITFElement)
            the_value = the_typ.from_bytes(value, start)
            fields[attribute] = the_value
            return start + the_value.get_bytes_length()
        else:
            raise ValueError('Cannot parse attribute {} for class {}'.format(attribute, cls))

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value: bytes|str
            the header string to scrape
        start : int
            the beginning location in the string

        Returns
        -------

        """

        fields = {}
        loc = start
        for fld in cls._ordering:
            loc = cls._parse_attribute(fields, fld, value, loc)
        return cls(**fields)

    def to_json(self):
        out = OrderedDict()
        for fld in self._ordering:
            if self._get_attribute_length(fld) == 0:
                continue
            value = getattr(self, fld)
            if value is None:
                out[fld] = ''
            elif isinstance(value, string_types) or isinstance(value, integer_types) \
                    or isinstance(value, bytes):
                out[fld] = value
            elif isinstance(value, BaseNITFElement):
                out[fld] = value.to_json()
            else:
                logging.error(
                    'Got unhandled type `{}` for json serialization for '
                    'attribute `{}` of class {}'.format(type(value), fld, self.__class__))
        return out


class NITFLoop(NITFElement):
    __slots__ = ('_values', )
    _ordering = ('values', )
    _child_class = None  # must be a subclass of NITFElement
    _count_size = 0  # type: int

    def __init__(self, values=None, **kwargs):
        if not issubclass(self._child_class, NITFElement):
            raise TypeError('_child_class for {} must be a subclass of NITFElement'.format(self.__class__.__name__))
        self._values = tuple()
        super(NITFLoop, self).__init__(values=values, **kwargs)

    @property
    def values(self):  # type: () -> Tuple[_child_class, ...]
        return self._values

    @values.setter
    def values(self, value):
        if value is None:
            self._values = ()
            return
        if not isinstance(value, tuple):
            value = tuple(value)
        for i, entry in enumerate(value):
            if not isinstance(entry, self._child_class):
                raise TypeError(
                    'values must be of type {}, got entry {} of type {}'.format(self._child_class, i, type(entry)))
        self._values = value

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):  # type: (Union[int, slice]) -> Union[_child_class, List[_child_class]]
        return self._values[item]

    def get_bytes_length(self):
        return self._count_size + sum(entry.get_bytes_length() for entry in self._values)

    @classmethod
    def minimum_length(cls):
        return cls._count_size

    @classmethod
    def _parse_count(cls, value, start):
        loc = start
        count = int_func(value[loc:loc+cls._count_size])
        loc += cls._count_size
        return count, loc

    @classmethod
    def from_bytes(cls, value, start):
        if not issubclass(cls._child_class, NITFElement):
            raise TypeError('_child_class for {} must be a subclass of NITFElement'.format(cls.__name__))

        count, loc = cls._parse_count(value, start)
        if count == 0:
            return cls(values=None)

        values = []
        for i in range(count):
            val = cls._child_class.from_bytes(value, loc)
            loc += val.get_bytes_length()
            values.append(val)
        return cls(values=values)

    def _counts_bytes(self):
        frm_str = '{0:0'+str(self._count_size) + 'd}'
        return frm_str.format(len(self.values)).encode('utf-8')

    def to_bytes(self):
        return self._counts_bytes() + b''.join(entry.to_bytes() for entry in self._values)

    def to_json(self):
        return [entry.to_json() for entry in self._values]


class Unstructured(NITFElement):
    """
    A possible NITF element pattern which is largely unparsed -
    just a bytes array of a given length
    """

    __slots__ = ('_data', )
    _ordering = ('data', )
    _size_len = 1

    def __init__(self, data=None, **kwargs):
        self._data = None
        if not (isinstance(self._size_len, int) and self._size_len > 0):
            raise TypeError(
                'class variable _size_len for {} must be a positive '
                'integer'.format(self.__class__.__name__))
        super(Unstructured, self).__init__(data=data, **kwargs)

    @property
    def data(self):  # type: () -> Union[None, bytes, NITFElement]
        return self._data

    @data.setter
    def data(self, value):
        if value is None:
            self._data = None
            return

        if not isinstance(value, (bytes, NITFElement)):
            raise TypeError(
                'data requires bytes or NITFElement type. '
                'Got type {}'.format(type(value)))
        siz_lim = 10**self._size_len - 1
        if isinstance(value, bytes):
            len_cond = (len(value) > siz_lim)
        else:
            len_cond = value.get_bytes_length() > siz_lim
        if len_cond:
            raise ValueError('The provided data is longer than {}'.format(siz_lim))
        self._data = value
        self._populate_data()

    def _populate_data(self):
        """
        Populate the _data attribute from bytes to some other appropriate object.

        Returns
        -------
        None
        """

        pass

    @classmethod
    def minimum_length(cls):
        return cls._size_len

    def _get_attribute_bytes(self, attribute):
        if attribute == 'data':
            siz_frm = '{0:0' + str(self._size_len) + '}'
            data = self.data
            if data is None:
                return b'0'*self._size_len
            if isinstance(data, NITFElement):
                data = data.to_bytes()
            if isinstance(data, bytes):
                return siz_frm.format(len(data)).encode('utf-8') + data
            else:
                raise TypeError(
                    'Got unexpected data type {} for attribute {} of class {}'.format(
                        type(data), attribute, self.__class__))
        return super(Unstructured, self)._get_attribute_bytes(attribute)

    def _get_attribute_length(self, attribute):
        if attribute == 'data':
            data = self.data
            if data is None:
                return self._size_len
            elif isinstance(data, NITFElement):
                return self._size_len + data.get_bytes_length()
            else:
                return self._size_len + len(data)
        return super(Unstructured, self)._get_attribute_length(attribute)

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'data':
            length = int_func(value[start:start + cls._size_len])
            start += cls._size_len
            fields['data'] = value[start:start + length]
            return start + length
        return super(Unstructured, cls)._parse_attribute(fields, attribute, value, start)


class _ItemArrayHeaders(BaseNITFElement):
    """
    Item array in the NITF header (i.e. Image Segment, Text Segment).
    This is not really meant to be used directly.
    """

    __slots__ = ('subhead_sizes', 'item_sizes')
    _subhead_len = 0
    _item_len = 0

    def __init__(self, subhead_sizes=None, item_sizes=None, **kwargs):
        """

        Parameters
        ----------
        subhead_sizes : numpy.ndarray|None
        item_sizes : numpy.ndarray|None
        """

        if subhead_sizes is None or item_sizes is None:
            subhead_sizes = numpy.zeros((0, ), dtype=numpy.int64)
            item_sizes = numpy.zeros((0,), dtype=numpy.int64)
        if subhead_sizes.shape != item_sizes.shape or len(item_sizes.shape) != 1:
            raise ValueError(
                'the subhead_offsets and item_offsets arrays must one-dimensional and the same length')

        self.subhead_sizes = subhead_sizes
        """
        numpy.ndarray: the subheader sizes
        """

        self.item_sizes = item_sizes
        """
        numpy.ndarray: the item size
        """

        super(_ItemArrayHeaders, self).__init__(**kwargs)

    def get_bytes_length(self):
        return 3 + (self._subhead_len + self._item_len)*self.subhead_sizes.size

    @classmethod
    def minimum_length(cls):
        return 3

    @classmethod
    def from_bytes(cls, value, start, subhead_len=0, item_len=0):
        """

        Parameters
        ----------
        value : bytes|str
        start : int
        subhead_len : int
        item_len : int

        Returns
        -------
        _ItemArrayHeaders
        """

        subhead_len, item_len = int_func(cls._subhead_len), int_func(cls._item_len)
        if len(value) < start + 3:
            raise ValueError('value must have length at least {}. Got {}'.format(start+3, len(value)))
        loc = start
        count = int_func(value[loc:loc+3])
        length = 3 + count*(subhead_len + item_len)
        if len(value) < start + length:
            raise ValueError('value must have length at least {}. Got {}'.format(start+length, len(value)))
        loc += 3
        subhead_sizes = numpy.zeros((count, ), dtype=numpy.int64)
        item_sizes = numpy.zeros((count, ), dtype=numpy.int64)
        for i in range(count):
            subhead_sizes[i] = int_func(value[loc: loc+subhead_len])
            loc += subhead_len
            item_sizes[i] = int_func(value[loc: loc+item_len])
            loc += item_len
        return cls(subhead_sizes, item_sizes)

    def to_bytes(self):
        out = '{0:03d}'.format(self.subhead_sizes.size)
        subh_frm = '{0:0' + str(self._subhead_len) + 'd}'
        item_frm = '{0:0' + str(self._item_len) + 'd}'
        for sh_off, it_off in zip(self.subhead_sizes, self.item_sizes):
            out += subh_frm.format(sh_off) + item_frm.format(it_off)
        return out.encode()

    def to_json(self):
        return OrderedDict([
            ('subheader_sizes', self.subhead_sizes.tolist()),
            ('item_sizes', self.item_sizes.tolist())])


######
# TRE Elements

class TRE(BaseNITFElement):
    """
    An abstract TRE class - this should not be instantiated directly.
    """

    @property
    def TAG(self):
        """
        str: The TRE tag.
        """

        raise NotImplementedError

    @property
    def DATA(self):
        """
        The TRE data.
        """

        raise NotImplementedError

    @property
    def EL(self):
        """
        int: The TRE element length.
        """

        raise NotImplementedError

    def get_bytes_length(self):
        raise NotImplementedError

    def to_bytes(self):
        raise NotImplementedError

    @classmethod
    def minimum_length(cls):
        return 11

    @classmethod
    def from_bytes(cls, value, start):
        tag = value[start:start+6]
        known_tre = find_tre(tag)
        if known_tre is not None:
            try:
                return known_tre.from_bytes(value, start)
            except Exception as e:
                logging.error(
                    "Returning unparsed tre, because we failed parsing tre as "
                    "type {} with exception\n\t{}".format(known_tre.__name__, e))
        return UnknownTRE.from_bytes(value, start)

    def to_json(self):
        out = OrderedDict([('tag', self.TAG), ('length', self.EL)])
        if isinstance(self.DATA, bytes):
            out['data'] = self.DATA
        else:
            out['data'] = self.DATA.to_json()
        return out


class UnknownTRE(TRE):
    __slots__ = ('_TAG', '_data')

    def __init__(self, TAG, data):
        """

        Parameters
        ----------
        TAG : str
        data : bytes
        """

        self._data = None
        if isinstance(TAG, bytes):
            TAG = TAG.decode('utf-8')

        if not isinstance(TAG, string_types):
            raise TypeError('TAG must be a string. Got {}'.format(type(TAG)))
        if len(TAG) > 6:
            raise ValueError('TAG must be 6 or fewer characters')

        self._TAG = TAG
        self._data = data

    @property
    def TAG(self):
        return self._TAG

    @property
    def DATA(self):  # type: () -> bytes
        return self._data

    @DATA.setter
    def DATA(self, value):
        if not isinstance(value, bytes):
            raise TypeError('data must be a bytes instance. Got {}'.format(type(value)))
        self._data = value

    @property
    def EL(self):
        return len(self._data)

    def get_bytes_length(self):
        return 11 + self.EL

    def to_bytes(self):
        return '{0:6s}{1:05d}'.format(self.TAG, self.EL).encode('utf-8') + self._data

    @classmethod
    def from_bytes(cls, value, start):
        tag = value[start:start+6]
        lng = int_func(value[start+6:start+11])
        return cls(tag, value[start+11:start+11+lng])


class TREList(NITFElement):
    """
    A list of TREs. This is meant to be used indirectly through one of the header
    type objects, which controls the parsing appropriately.
    """

    __slots__ = ('_tres', )
    _ordering = ('tres', )

    def __init__(self, tres=None, **kwargs):
        self._tres = []
        super(TREList, self).__init__(tres=tres, **kwargs)

    @property
    def tres(self):
        # type: () -> List[TRE]
        return self._tres

    @tres.setter
    def tres(self, value):
        if value is None:
            self._tres = []
            return

        if not isinstance(value, (list, tuple)):
            raise TypeError('tres must be a list or tuple')

        for i, entry in enumerate(value):
            if not isinstance(entry, TRE):
                raise TypeError(
                    'Each entry of tres must be of type TRE. '
                    'Entry {} is type {}'.format(i, type(entry)))
        self._tres = value

    def _get_attribute_bytes(self, attribute):
        if attribute == 'tres':
            if len(self._tres) == 0:
                return b''
            return b''.join(entry.to_bytes() for entry in self._tres)
        return super(TREList, self)._get_attribute_bytes(attribute)

    def _get_attribute_length(self, attribute):
        if attribute == 'tres':
            if len(self._tres) == 0:
                return 0
            return sum(entry.get_bytes_length() for entry in self._tres)
        return super(TREList, self)._get_attribute_length(attribute)

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'tres':
            if len(value) == start:
                fields['tres'] = []
                return start
            tres = []
            loc = start
            while loc < len(value):
                anticipated_length = int_func(value[loc+6:loc+11]) + 11
                tre = TRE.from_bytes(value, loc)
                parsed_length = tre.get_bytes_length()
                if parsed_length != anticipated_length:
                    logging.error(
                        'The given length for TRE {} instance is {}, but the constructed length is {}. '
                        'This is the result of a malformed TRE object definition. '
                        'If possible, this should be reported to the sarpy team for review/repair.'.format(
                            tre.TAG, anticipated_length, parsed_length))
                loc += anticipated_length
                tres.append(tre)
            fields['tres'] = tres
            return len(value)
        return super(TREList, cls)._parse_attribute(fields, attribute, value, start)

    def __len__(self):
        return len(self._tres)

    def __getitem__(self, item):
        # type: (Union[int, slice, str]) -> Union[None, TRE, List[TRE]]
        if isinstance(item, (int, slice)):
            return self._tres[item]
        elif isinstance(item, string_types):
            for entry in self.tres:
                if entry.TAG == item:
                    return entry
            return None
        else:
            raise TypeError('Got unhandled type {}'.format(type(item)))

    def to_json(self):
        return [entry.to_json() for entry in self._tres]


class TREHeader(Unstructured):
    def _populate_data(self):
        if isinstance(self._data, bytes):
            data = TREList.from_bytes(self._data, 0)
            self._data = data


class UserHeaderType(Unstructured):
    __slots__ = ('_data', '_ofl')
    _ordering = ('data', )
    _size_len = 5
    _ofl_len = 3

    def __init__(self, OFL=None, data=None, **kwargs):
        self._ofl = None
        self._data = None
        self.OFL = OFL
        super(UserHeaderType, self).__init__(data=data, **kwargs)

    @property
    def OFL(self):  # type: () -> int
        return self._ofl

    @OFL.setter
    def OFL(self, value):
        if value is None:
            self._ofl = 0
            return

        value = int_func(value)
        if not (0 <= value <= 999):
            raise ValueError('ofl requires an integer value in the range 0-999.')
        self._ofl = value

    def _populate_data(self):
        if isinstance(self._data, bytes):
            data = TREList.from_bytes(self._data, 0)
            self._data = data

    @classmethod
    def minimum_length(cls):
        return cls._size_len

    def _get_attribute_bytes(self, attribute):
        if attribute == 'data':
            siz_frm = '{0:0' + str(self._size_len) + 'd}'
            ofl_frm = '{0:0' + str(self._ofl_len) + 'd}'
            data = self.data
            if data is None:
                return b'0'*self._size_len
            if isinstance(data, NITFElement):
                data = data.to_bytes()
            if isinstance(data, bytes):
                return siz_frm.format(len(data) + self._ofl_len).encode('utf-8') + \
                    ofl_frm.format(self._ofl).encode('utf-8') + data
            else:
                raise TypeError('Got unexpected data type {}'.format(type(data)))
        return super(Unstructured, self)._get_attribute_bytes(attribute)

    def _get_attribute_length(self, attribute):
        if attribute == 'data':
            data = self.data
            if data is None:
                return self._size_len
            elif isinstance(data, NITFElement):
                return self._size_len + self._ofl_len + data.get_bytes_length()
            else:
                return self._size_len + self._ofl_len + len(data)
        return super(UserHeaderType, self)._get_attribute_length(attribute)

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'data':
            length = int_func(value[start:start + cls._size_len])
            start += cls._size_len
            if length > 0:
                ofl = int_func(value[start:start+cls._ofl_len])
                fields['OFL'] = ofl
                fields['data'] = value[start+cls._ofl_len:start + length]
            else:
                fields['OFL'] = 0
                fields['data'] = None
            return start + length
        return super(UserHeaderType, cls)._parse_attribute(fields, attribute, value, start)
