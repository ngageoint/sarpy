# -*- coding: utf-8 -*-
"""
Functionality for dealing with NITF file header information. This is specifically
geared towards SICD file usage, and some functionality may not allow full generality.
See MIL-STD-2500C for specification details.
"""

import logging
import sys
from collections import OrderedDict
from typing import Union, Tuple, List, Set
import struct

import numpy

from .tres.registration import find_tre

########
# NITF header details specific to SICD files

_SICD_SPECIFICATION_IDENTIFIER = 'SICD Volume 1 Design & Implementation Description Document'
_SICD_SPECIFICATION_VERSION = '1.2'
_SICD_SPECIFICATION_DATE = '2018-12-13T00:00:00Z'
_SICD_SPECIFICATION_NAMESPACE = 'urn:SICD:1.2.1'  # must be of the form 'urn:SICD:<version>'


integer_types = (int, )
string_types = (str, )
int_func = int
if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    int_func = long  # to accommodate for 32-bit python 2
    # noinspection PyUnresolvedReferences
    integer_types = (int, long)
    # noinspection PyUnresolvedReferences
    string_types = (str, unicode)

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


#####
# Basic Components

def _get_default(frmt):
    """
    Get a default primitive value according to a schema.

    Parameters
    ----------
    frmt : str

    Returns
    -------

    """

    length = int_func(frmt[:-1])
    typ = frmt[-1]
    if typ == 's':
        return '\x20'*length
    elif typ == 'd':
        return 0
    elif typ == 'f':
        return 0.0
    elif typ == 'b':
        return b'\x00'*length
    else:
        raise ValueError('Got unparseable format string {}'.format(frmt))


def _validate_value(frmt, value, name):
    """
    Validate a primitive value according to the schema.

    Parameters
    ----------
    frmt : str
    value : str|bytes|int|float
    name : str

    Returns
    -------

    """

    if value is None:
        return _get_default(frmt)

    length = int_func(frmt[:-1])
    typ = frmt[-1]
    if typ == 's':
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if not isinstance(value, str):
            raise TypeError('Field {} requires a string, got type {}'.format(name, type(value)))
        if len(value) > length:
            logging.warning(
                'Field {} requires a string of length {}, got one of length {}, '
                'so truncating'.format(name, length, len(value)))
            return value[:length]
        fmt_out = '{0:' + frmt + '}'
        return fmt_out.format(value)
    elif typ == 'd':
        if isinstance(value, (str, bytes)):
            try:
                value = int_func(value)
            except Exception as e:
                logging.error('Failed converting attribute {} with error {}'.format(name, e))
                raise e
        if value > 10**length:
            raise ValueError(
                'Field {} requires an integer of {} or fewer digits. '
                'Got {}'.format(name, length, value))
        return value
    elif typ == 'b':
        if isinstance(value, str):
            value = value.encode()
        if not isinstance(value, bytes):
            raise ValueError('Field {} requires bytes. Got type {}'.format(name, type(value)))
        if len(value) > length:
            return value[:length]
        if len(value) < length:
            value += b'\x00'*(length - len(value))
        return value
    else:
        raise ValueError('Got unparseable format {}'.format(frmt))


def _get_bytes(frmt, value, name):
    length = int_func(frmt[:-1])
    typ = frmt[-1]
    if typ == 's':
        fmt_out = '{0:' + frmt + '}'
        if len(value) > length:
            return value[:length].encode('utf-8')
        return fmt_out.format(value).encode('utf-8')
    elif typ == 'd':
        fmt_out = '{0:0' + frmt + '}'
        return fmt_out.format(value).encode('utf-8')
    elif typ == 'b':
        if len(value) >= length:
            return value[:length]
        else:
            return value + b'\x00'*(length - len(value))
    else:
        raise ValueError('Unparseable format {} for field {}'.format(frmt, name))


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


class NITFElement(BaseNITFElement):
    """
    Describes the basic nitf header reading and writing functionality. This is
    an abstract element, and not meant to be used directly.
    """

    __slots__ = ('_skips', )  # the possible attribute collection
    _formats = {}  # {attribute: '<length>d/s/f'} for each primitive element
    _types = {}  # provides the types for non-primitive elements
    _defaults = {}  # default values for elements
    # for non-primitive element, this should be a dictionary with keys
    # 'args' and 'kwargs' to pass into the constructor. If the default constructor
    # is desired, then put an empty dict.
    _enums = {}  # allowed values for enum type variables, entries should be set type
    # NOTE: the string entries of enums should be stripped of the white space characters
    _ranges = {}  # ranges for limited int variables, entries should be (min|None, max|None) tuples.
    _if_skips = {}  # dictionary of the form {<variable>: {'condition': <condition, 'vars': []}}

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
            The appropriate keyword arguments
        """

        for attribute in kwargs:
            setattr(self, attribute, kwargs[attribute])
            # NB: this may trigger an attribute error, if something is dumb
        if not hasattr(self, '_skips'):
            self._skips = set()
        for attribute in self.__slots__:
            if not hasattr(self, attribute):
                # this attribute wasn't defined above, so let's initialize it
                setattr(self, attribute, self.get_default_value(attribute))

    @classmethod
    def get_default_value(cls, attribute):
        if attribute in cls._defaults:
            if attribute in cls._types:
                arg_dict = cls._defaults[attribute]
                typ = cls._types[attribute]
                return typ(*arg_dict.get('args', []), **arg_dict.get('kwargs', {}))
            return cls._defaults[attribute]
        elif attribute in cls._formats:
            return _get_default(cls._formats[attribute])
        else:
            return None

    def _set_attribute(self, attribute, value):
        """
        Override this for individual behavior. This should be overridden with
        care, to avoid a recursive loop.

        The final step of setting should look like
        :code:`object.__setattr__(self, attribute, value)`

        Parameters
        ----------
        attribute : str
            The attribute name
        value
            The attribute value.

        Returns
        -------
        None
        """

        def validate_enum():
            if attribute not in self._enums or value is None:
                return
            contained = value in self._enums[attribute]
            if not contained and isinstance(value, str):
                contained = value.strip() in self._enums[attribute]

            if not contained:
                logging.error(
                    "Attribute {} must take values in {}. "
                    "Got {}".format(attribute, list(self._enums[attribute]), value))

        def validate_range():
            if attribute not in self._ranges or value is None:
                return
            rng = self._ranges[attribute]
            if rng[0] is not None and value < rng[0]:
                logging.error('Attribute {} must be >= {}. Got {}'.format(attribute, rng[0], value))
            if rng[1] is not None and value > rng[1]:
                logging.error('Attribute {} must be <= {}. Got {}'.format(attribute, rng[1], value))

        # noinspection PyUnusedLocal
        def check_conditions(the_value):
            dd = self._if_skips.get(attribute, None)
            if dd is None:
                return
            cstr = 'the_value {}'.format(dd['condition'])
            condition = eval(cstr)
            if condition:
                self._skips = self._skips.union(dd['vars'])
            else:
                self._skips = self._skips.difference(dd['vars'])

        if attribute in self._types:
            typ = self._types[attribute]
            if value is None:
                value = self.get_default_value(attribute)  # NB: might still be None
            if isinstance(value, bytes):
                value = typ.from_bytes(value, 0)
            if not (value is None or isinstance(value, BaseNITFElement)):
                raise ValueError('Attribute {} is expected to be of type {}, '
                                 'got {}'.format(attribute, typ, type(value)))
            check_conditions(value)
            object.__setattr__(self, attribute, value)
        elif attribute in self._formats:
            frmt = self._formats[attribute]
            if value is None:
                value = _get_default(frmt)
            else:
                value = _validate_value(frmt, value, attribute)
            validate_enum()
            validate_range()
            check_conditions(value)
            object.__setattr__(self, attribute, value)
        else:
            raise ValueError(
                'This method should not be called for attribute not listed in _formats or _types')

    def __setattr__(self, attribute, value):
        if attribute in self._types or attribute in self._formats:
            self._set_attribute(attribute, value)
        else:
            object.__setattr__(self, attribute, value)

    def _get_bytes_attribute(self, attribute):
        """
        Get the bytes representation for a given attribute.

        Parameters
        ----------
        attribute : str

        Returns
        -------
        bytes
        """

        if not hasattr(self, attribute):
            raise AttributeError('No attribute {}'.format(attribute))
        if (attribute in self._skips) or not (attribute in self._formats or attribute in self._types):
            return b''
        val = getattr(self, attribute)
        if val is None:
            return b''
        if attribute in self._types:
            if not isinstance(val, BaseNITFElement):
                raise TypeError(
                    'Elements in _bytes must be an instance of NITFElement. '
                    'Got type {} for attribute {}'.format(type(val), attribute))
            return val.to_bytes()
        else:
            return _get_bytes(self._formats[attribute], val, attribute)

    def _get_length_attribute(self, attribute):
        if not hasattr(self, attribute):
            raise AttributeError('No attribute {}'.format(attribute))
        if (attribute in self._skips) or not (attribute in self._formats or attribute in self._types):
            return 0
        if attribute in self._types:
            val = getattr(self, attribute)
            if val is None:
                return 0
            if not isinstance(val, BaseNITFElement):
                raise TypeError(
                    'Elements in _bytes must be an instance of NITFElement. '
                    'Got type {} for attribute {}'.format(type(val), attribute))
            return val.get_bytes_length()
        elif attribute in self._formats:
            return int_func(self._formats[attribute][:-1])
        else:
            return 0

    def get_bytes_length(self):
        return sum(self._get_length_attribute(attribute) for attribute in self.__slots__)

    @classmethod
    def minimum_length(cls):
        """
        The minimum size in bytes that takes to write this header element.

        Returns
        -------
        int
        """

        min_length = 0
        for attribute in cls.__slots__:
            if attribute in cls._types:
                typ = cls._types[attribute]
                if issubclass(typ, BaseNITFElement):
                    min_length += typ.minimum_length()
            elif attribute in cls._formats:
                min_length += int_func(cls._formats[attribute][:-1])
        return min_length

    @classmethod
    def _parse_attribute(cls, fields, skips, attribute, value, start):
        """

        Parameters
        ----------
        fields : dict
            The attribute:value dictionary.
        skips : Set
            The set of attributes to skip - for conditional values.
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

        # noinspection PyUnusedLocal
        def check_conditions(the_value):
            dd = cls._if_skips.get(attribute, None)
            if dd is None:
                return
            cstr = 'the_value {}'.format(dd['condition'])
            condition = eval(cstr)
            if condition:
                for var in dd['vars']:
                    skips.add(var)
                    fields[var] = None
            else:
                for var in dd['vars']:
                    skips.remove(var)

        if attribute not in cls.__slots__:
            raise ValueError('No attribute {}'.format(attribute))
        if attribute in skips or attribute in fields:
            return start
        if attribute in cls._types:
            typ = cls._types[attribute]
            if not issubclass(typ, NITFElement):
                raise TypeError(
                    'Class {}, attribute {} must be a type which is a subclass of NITFElement. '
                    'Got {}'.format(cls.__name__, attribute, typ))
            val = typ.from_bytes(value, start)
            fields[attribute] = val
            check_conditions(val)
            return start+val.get_bytes_length()
        elif attribute in cls._formats:
            frmt = cls._formats[attribute]
            length = int_func(frmt[:-1])
            end = start + length
            if len(value) < end:
                raise ValueError(
                    'Class {}, attribute {} must have value string of length {}, '
                    'but is only length {}'.format(cls.__name__, attribute, end, len(value)))
            val = _validate_value(frmt, value[start:end], attribute)
            fields[attribute] = val
            check_conditions(val)
            return end
        else:
            return start

    @classmethod
    def _parse_attributes(cls, value, start):
        """
        Parse the byte array.

        Parameters
        ----------
        value : bytes
            The bytes array to be parsed.
        start : int
            The present position in `value`.

        Returns
        -------
        (dict, set)
            The parsed fields dictionary and set of fields to be skipped.
        """

        fields = OrderedDict()
        skips = set()
        loc = start
        for attribute in cls.__slots__:
            loc = cls._parse_attribute(fields, skips, attribute, value, loc)
        return fields, skips

    @classmethod
    def from_bytes(cls, value, start):
        fields, skips = cls._parse_attributes(value, start)
        fields['_skips'] = skips
        return cls(**fields)

    def to_bytes(self):
        items = [self._get_bytes_attribute(attribute) for attribute in self.__slots__]
        return b''.join(items)


class NITFLoop(NITFElement):
    __slots__ = ('_values', )
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


class Unstructured(NITFElement):
    """
    A possible NITF element pattern which is largely unparsed -
    just a bytes array of a given length
    """

    __slots__ = ('_data', )
    _size_len = 1

    def __init__(self, **kwargs):
        if not (isinstance(self._size_len, int) and self._size_len > 0):
            raise TypeError(
                'class variable _size_len for {} must be a positive '
                'integer'.format(self.__class__.__name__))

        self._data = None
        super(Unstructured, self).__init__(**kwargs)

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

    def _get_bytes_attribute(self, attribute):
        if attribute == '_data':
            siz_frm = '{0:0' + str(self._size_len) + '}'
            data = self.data
            if data is None:
                return b'0'*self._size_len
            if isinstance(data, NITFElement):
                data = data.to_bytes()
            if isinstance(data, bytes):
                return siz_frm.format(len(data)).encode('utf-8') + data
            else:
                raise TypeError('Got unexpected data type {}'.format(type(data)))
        return super(Unstructured, self)._get_bytes_attribute(attribute)

    def _get_length_attribute(self, attribute):
        if attribute == '_data':
            data = self.data
            if data is None:
                return self._size_len
            elif isinstance(data, NITFElement):
                return self._size_len + data.get_bytes_length()
            else:
                return self._size_len + len(data)
        return super(Unstructured, self)._get_length_attribute(attribute)

    @classmethod
    def _parse_attribute(cls, fields, skips, attribute, value, start):
        if attribute == '_data':
            length = int_func(value[start:start + cls._size_len])
            start += cls._size_len
            fields['data'] = value[start:start + length]
            return start + length
        return super(Unstructured, cls)._parse_attribute(fields, skips, attribute, value, start)


class _ItemArrayHeaders(NITFElement):
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
        self.item_sizes = item_sizes
        self._skips = set()
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


#######
# Security tags - this is used in a variety of places

class NITFSecurityTags(NITFElement):
    """
    The NITF security tags - described in SICD standard 2014-09-30, Volume II, page 20
    """

    __slots__ = (
        'CLAS', 'CLSY', 'CODE', 'CTLH',
        'REL', 'DCTP', 'DCDT', 'DCXM',
        'DG', 'DGDT', 'CLTX', 'CAPT',
        'CAUT', 'CRSN', 'SRDT', 'CTLN')
    _formats = {
        'CLAS': '1s', 'CLSY': '2s', 'CODE': '11s', 'CTLH': '2s',
        'REL': '20s', 'DCTP': '2s', 'DCDT': '8s', 'DCXM': '4s',
        'DG': '1s', 'DGDT': '8s', 'CLTX': '43s', 'CAPT': '1s',
        'CAUT': '40s', 'CRSN': '1s', 'SRDT': '8s', 'CTLN': '15s'}
    _defaults = {'CLAS': 'U', }
    _enums = {
        'CLAS': {'U', 'R', 'C', 'S', 'T'},
        'DCTP': {'', 'DD', 'DE', 'GD', 'GE', 'O', 'X'},
        'DCXM': {
            '', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
            'X251', 'X252', 'X253', 'X254', 'X255', 'X256', 'X257', 'X258'},
        'DG': {'', 'S', 'C', 'R'},
        'CATP': {'', 'O', 'D', 'M'},
        'CRSN': {'', 'A', 'B', 'C', 'D', 'E', 'F', 'G'}}


######
# TRE

class TRE(BaseNITFElement):
    """
    An abstract TRE class - this should not be instantiated directly.
    """

    @property
    def TAG(self):
        """
        The TRE tag.

        Returns
        -------
        str
        """

        raise NotImplementedError

    @property
    def DATA(self):
        """
        The TRE data.

        Returns
        -------

        """

        raise NotImplementedError

    @property
    def EL(self):
        """
        The TRE element length.

        Returns
        -------
        int
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
                    "Failed parsing tre as type {} with error {}. "
                    "Returning unparsed.".format(known_tre.__name__, e))
        return UnknownTRE.from_bytes(value, start)


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

        if not isinstance(TAG, str):
            raise TypeError('TAG must be a string. Got {}'.format(type(TAG)))
        if len(TAG) > 6:
            raise ValueError('TAG must be 6 or fewer characters')

        self._TAG = TAG
        self.data = data

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

    def __init__(self, tres=None, **kwargs):
        self._tres = []
        super(TREList, self).__init__(tres=tres, **kwargs)

    @property
    def tres(self):  # type: () -> List[TRE]
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

    def _get_bytes_attribute(self, attribute):
        if attribute == '_tres':
            if len(self._tres) == 0:
                return b''
            return b''.join(entry.to_bytes() for entry in self._tres)
        return super(TREList, self)._get_bytes_attribute(attribute)

    def _get_length_attribute(self, attribute):
        if attribute == '_tres':
            if len(self._tres) == 0:
                return 0
            return sum(entry.get_bytes_length() for entry in self._tres)
        return super(TREList, self)._get_length_attribute(attribute)

    @classmethod
    def _parse_attribute(cls, fields, skips, attribute, value, start):
        if attribute == '_tres':
            if len(value) == start:
                fields['tres'] = []
                return start
            tres = []
            loc = start
            while loc < len(value):
                tre = TRE.from_bytes(value, loc)
                loc += tre.get_bytes_length()
                tres.append(tre)
            fields['tres'] = tres
            return len(value)
        return super(TREList, cls)._parse_attribute(fields, skips, attribute, value, start)

    def __len__(self):
        return len(self._tres)

    def __getitem__(self, item):  # type: (Union[int, slice]) -> Union[TRE, List[TRE]]
        return self._tres[item]


class TREHeader(Unstructured):
    def _populate_data(self):
        if isinstance(self._data, bytes):
            data = TREList.from_bytes(self._data, 0)
            self._data = data


class UserHeaderType(Unstructured):
    __slots__ = ('_ofl', '_data')
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

    def _get_bytes_attribute(self, attribute):
        if attribute == '_data':
            siz_frm = '{0:0' + str(self._size_len) + '}'
            ofl_frm = '{0:0' + str(self._ofl_len) + '}'
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
        return super(Unstructured, self)._get_bytes_attribute(attribute)

    def _get_length_attribute(self, attribute):
        if attribute == '_data':
            data = self.data
            if data is None:
                return self._size_len
            elif isinstance(data, NITFElement):
                return self._size_len + self._ofl_len + data.get_bytes_length()
            else:
                return self._size_len + self._ofl_len + len(data)
        return super(UserHeaderType, self)._get_length_attribute(attribute)

    @classmethod
    def _parse_attribute(cls, fields, skips, attribute, value, start):
        if attribute == '_data':
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
        return super(UserHeaderType, cls)._parse_attribute(fields, skips, attribute, value, start)


######
# Partially general image segment header

class ImageBand(NITFElement):
    """
    Single image band, part of the image bands collection
    """
    __slots__ = ('IREPBAND', 'ISUBCAT', 'IFC', 'IMFLT', '_LUTD')
    _formats = {'IREPBAND': '2s', 'ISUBCAT': '6s', 'IFC': '1s', 'IMFLT': '3s'}
    _defaults = {'IFC': 'N'}
    _enums = {'IFC': {'N'}}

    def __init__(self, **kwargs):
        self._LUTD = None
        super(ImageBand, self).__init__(**kwargs)

    @classmethod
    def minimum_length(cls):
        return 13

    @property
    def LUTD(self):  # type: () -> Union[None, numpy.ndarray]
        return self._LUTD

    @LUTD.setter
    def LUTD(self, value):
        if value is None:
            self._LUTD = None
            return

        if not isinstance(value, numpy.ndarray):
            raise TypeError('LUTD must be a numpy array')
        if value.dtype.name != 'uint8':
            raise ValueError('LUTD must be a numpy array of dtype uint8, got {}'.format(value.dtype.name))
        if value.ndim != 2:
            raise ValueError('LUTD must be a two-dimensional array')
        if value.shape[0] > 4:
            raise ValueError(
                'The number of LUTD bands (axis 0) must be 4 or fewer. '
                'Got LUTD shape {}'.format(value.shape))
        if value.shape[1] > 65536:
            raise ValueError(
                'The number of LUTD elemnts (axis 1) must be 65536 or fewer. '
                'Got LUTD shape {}'.format(value.shape))
        self._LUTD = value

    @property
    def NLUTS(self):
        return 0 if self._LUTD is None else self._LUTD.shape[0]

    @property
    def NELUTS(self):
        return 0 if self._LUTD is None else self._LUTD.shape[1]

    def _get_bytes_attribute(self, attribute):
        if attribute == '_LUTD':
            if self.NLUTS == 0:
                out = b'0'
            else:
                out = '{0:d}{1:05d}'.format(self.NLUTS, self.NELUTS).encode() + \
                      struct.pack('{}B'.format(self.NLUTS * self.NELUTS, *self.LUTD.flatten()))
            return out
        else:
            return super(ImageBand, self)._get_bytes_attribute(attribute)

    def _get_length_attribute(self, attribute):
        if attribute == '_LUTD':
            nluts = self.NLUTS
            if nluts == 0:
                return 1
            else:
                neluts = self.NELUTS
                return 6 + nluts * neluts
        else:
            return super(ImageBand, self)._get_length_attribute(attribute)

    @classmethod
    def _parse_attribute(cls, fields, skips, attribute, value, start):
        if attribute == '_LUTD':
            loc = start
            nluts = int_func(value[loc:loc + 1])
            loc += 1
            if nluts == 0:
                fields['LUTD'] = None
            else:
                neluts = int_func(value[loc:loc + 5])
                loc += 5
                siz = nluts * neluts
                lutd = numpy.array(
                    struct.unpack('{}B'.format(siz), value[loc:loc + siz]), dtype=numpy.uint8).reshape(
                    (nluts, neluts))
                fields['LUTD'] = lutd
                loc += siz
            return loc
        return super(ImageBand, cls)._parse_attribute(fields, skips, attribute, value, start)


class ImageBands(NITFLoop):
    _child_class = ImageBand
    _count_size = 1

    @classmethod
    def _parse_count(cls, value, start):
        loc = start
        count = int_func(value[loc:loc + cls._count_size])
        loc += cls._count_size
        if count == 0:
            # (only) if there are more than 9, a longer field is used
            count = int_func(value[loc:loc + 5])
            loc += 5
        return count, loc

    def _counts_bytes(self):
        siz = len(self.values)
        if siz <= 9:
            return '{0:1d}'.format(siz).encode()
        else:
            return '0{0:05d}'.format(siz).encode()


class ImageComment(NITFElement):
    __slots__ = ('comment',)
    _formats = {'comment': '80s'}


class ImageComments(NITFLoop):
    _child_class = ImageComment
    _count_size = 1


class ImageSegmentHeader(NITFElement):
    """
    The Image Segment header - described in SICD standard 2014-09-30, Volume II, page 24
    """

    __slots__ = (
        'IM', 'IID1', 'IDATIM', 'TGTID',
        'IID2', '_Security', 'ENCRYP', 'ISORCE',
        'NROWS', 'NCOLS', 'PVTYPE', 'IREP',
        'ICAT', 'ABPP', 'PJUST', 'ICORDS',
        'IGEOLO', '_ImageComments', 'IC', 'COMRAT', '_ImageBands',
        'ISYNC', 'IMODE', 'NBPR', 'NBPC', 'NPPBH',
        'NPPBV', 'NBPP', 'IDLVL', 'IALVL',
        'ILOC', 'IMAG', '_UserHeader', '_ExtendedHeader')
    _formats = {
        'IM': '2s', 'IID1': '10s', 'IDATIM': '14s', 'TGTID': '17s',
        'IID2': '80s', 'ENCRYP': '1s', 'ISORCE': '42s',
        'NROWS': '8d', 'NCOLS': '8d', 'PVTYPE': '3s', 'IREP': '8s',
        'ICAT': '8s', 'ABPP': '2d', 'PJUST': '1s', 'ICORDS': '1s',
        'IGEOLO': '60s', 'IC': '2s', 'COMRAT': '4s', 'ISYNC': '1d', 'IMODE': '1s',
        'NBPR': '4d', 'NBPC': '4d', 'NPPBH': '4d', 'NPPBV': '4d',
        'NBPP': '2d', 'IDLVL': '3d', 'IALVL': '3d', 'ILOC': '10s',
        'IMAG': '4s', 'UDIDL': '5d', 'IXSHDL': '5d'}
    _defaults = {
        'IM': 'IM', 'ENCRYP': '0',
        'IREP': 'NODISPLY', 'ICAT': 'SAR', 'PJUST': 'R',
        'ICORDS': 'G', 'IC': 'NC', 'ISYNC': 0, 'IMODE': 'P',
        'NBPR': 1, 'NBPC': 1, 'IMAG': '1.0 ', 'UDIDL': 0, 'IXSHDL': 0,
        '_Security': {}, '_ImageComments': {}, '_ImageBands': {},
        '_UserHeader': {}, '_ExtendedHeader': {}}
    _types = {
        '_Security': NITFSecurityTags,
        '_ImageComments': ImageComments,
        '_ImageBands': ImageBands,
        '_UserHeader': UserHeaderType,
        '_ExtendedHeader': UserHeaderType}
    _enums = {
        'IC': {
            'NC', 'NM', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'I1',
            'M1', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'},
        'ENCRYP': {'0', },
        'PVTYPE': {'INT', 'B', 'SI', 'R', 'C'},
        'IREP': {
            'MONO', 'RGB', 'RGB/LUT', 'MULTI', 'NODISPLY', 'NVECTOR',
            'POLAR', 'VPH', 'YCbCr601'},
        'ICAT': {
            'VIS', 'SL', 'TI', 'FL', 'RD', 'EO', 'OP', 'HR', 'HS', 'CP',
            'BP', 'SAR', 'SARIQ', 'IR', 'MAP', 'MS', 'FP', 'MRI', 'XRAY',
            'CAT', 'VD', 'PAT', 'LEG', 'DTEM', 'MATR', 'LOCG', 'BARO',
            'CURRENT', 'DEPTH', 'WIND'},
        'PJUST': {'L', 'R'},
        'ICOORDS': {'', 'U', 'G', 'N', 'S', 'D'}}
    _if_skips = {
        'IC': {'condition': 'in ("NC", "NM")', 'vars': ['COMRAT', ]}}

    def __init__(self, **kwargs):
        self._skips = set()
        super(ImageSegmentHeader, self).__init__(**kwargs)

    @property
    def ImageComments(self):  # type: () -> ImageComments
        """ImageComments: the image comments instance"""
        return self._ImageComments

    @ImageComments.setter
    def ImageComments(self, value):
        # noinspection PyAttributeOutsideInit
        self._ImageComments = value

    @property
    def ImageBands(self):  # type: () -> ImageBands
        """ImageBands: the image bands instance"""
        return self._ImageBands

    @ImageBands.setter
    def ImageBands(self, value):
        # noinspection PyAttributeOutsideInit
        self._ImageBands = value

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        return self._Security

    @Security.setter
    def Security(self, value):
        # noinspection PyAttributeOutsideInit
        self._Security = value

    @property
    def UserHeader(self):  # type: () -> Union[UserHeaderType, NITFElement]
        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        # noinspection PyAttributeOutsideInit
        self._UserHeader = value

    @property
    def ExtendedHeader(self):  # type: () -> Union[UserHeaderType, NITFElement]
        return self._ExtendedHeader

    @ExtendedHeader.setter
    def ExtendedHeader(self, value):
        # noinspection PyAttributeOutsideInit
        self._ExtendedHeader = value

    @classmethod
    def minimum_length(cls):
        # COMRAT may not be there
        return super(ImageSegmentHeader, cls).minimum_length() - 4


######
# Graphics segment header

class GraphicsSegmentHeader(NITFElement):
    """
    Graphics Segment Subheader
    """

    __slots__ = (
        'SY', 'SID', 'SNAME', '_Security', 'ENCRYP', 'SFMT',
        'SSTRUCT', 'SDLVL', 'SALVL', 'SLOC', 'SBND1',
        'SCOLOR', 'SBND2', 'SRES2', '_UserHeader')
    _formats = {
        'SY': '2s', 'SID': '10s', 'SNAME': '20s', 'ENCRYP': '1d',
        'SFMT': '1s', 'SSTRUCT': '13d', 'SDLVL': '3d', 'SALVL': '3d',
        'SLOC': '10d', 'SBND1': '10d', 'SCOLOR': '1s', 'SBND2': '10d',
        'SRES2': '2d'}
    _defaults = {
        'SY': 'SY', 'SFMT': 'C', 'SDLVL': 1,
        '_Security': {}, '_UserHeader': {}}
    _types = {
        '_Security': NITFSecurityTags,
        '_UserHeader': UserHeaderType}
    _enums = {'SCOLOR': {'C', 'M'}}

    def __init__(self, **kwargs):
        super(GraphicsSegmentHeader, self).__init__(**kwargs)

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        return self._Security

    @Security.setter
    def Security(self, value):
        # noinspection PyAttributeOutsideInit
        self._Security = value

    @property
    def UserHeader(self):  # type: () -> Union[NITFElement, UserHeaderType]
        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        # noinspection PyAttributeOutsideInit
        self._UserHeader = value


######
# Text segment header

class TextSegmentHeader(NITFElement):
    """
    Text Segment Subheader
    """

    __slots__ = (
        'TE', 'TEXTID', 'TXTALVL',
        'TXTDT', 'TXTITL', '_Security',
        'ENCRYP', 'TXTFMT', '_UserHeader')
    _formats = {
        'TE': '2s', 'TEXTID': '7s', 'TXTALVL': '3d',
        'TXTDT': '14s', 'TXTITL': '80s', 'ENCRYP': '1d',
        'TXTFMT': '3s', 'TXSHDL': '5d', 'TXSOFL': '3d'}
    _defaults = {'TE': 'TE', '_Security': {}, '_UserHeader': {}}
    _types = {
        '_Security': NITFSecurityTags,
        '_UserHeader': UserHeaderType}

    def __init__(self, **kwargs):
        self._skips = set()
        super(TextSegmentHeader, self).__init__(**kwargs)

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        return self._Security

    @Security.setter
    def Security(self, value):
        # noinspection PyAttributeOutsideInit
        self._Security = value

    @property
    def UserHeader(self):  # type: () -> Union[NITFElement, UserHeaderType]
        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        # noinspection PyAttributeOutsideInit
        self._UserHeader = value


######
# Data Extension header

class DataExtensionHeader(NITFElement):
    """
    This requires an extension for essentially any non-trivial DES.
    """

    class DESUserHeader(Unstructured):
        _size_len = 4

    __slots__ = ('DE', 'DESID', 'DESVER', '_Security', 'DESOFLOW', 'DESITEM', '_UserHeader')
    _formats = {
        'DE': '2s', 'DESID': '25s', 'DESVER': '2d',
        'DESOFLOW': '6s', 'DESITEM': '3d'}
    _defaults = {
        'DE': 'DE', 'DESID': 'XML_DATA_CONTENT', 'DESVER': 1,
        '_Security': {}, '_UserHeader': {}}
    _types = {
        '_Security': NITFSecurityTags,
        '_UserHeader': DESUserHeader}
    _enums = {'DESOFLOW': {'', 'XHD', 'IXSHD', 'SXSHD', 'TXSHD', 'UDHD', 'UDID'}}
    _if_skips = {
        'DESID': {'condition': '!= "TRE_OVERFLOW"', 'vars': ['DESOFLOW', 'DESITEM']}}

    def __init__(self, **kwargs):
        self._skips = set()
        super(DataExtensionHeader, self).__init__(**kwargs)

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        return self._Security

    @Security.setter
    def Security(self, value):
        # noinspection PyAttributeOutsideInit
        self._Security = value

    @property
    def UserHeader(self):  # type: () -> Union[DESUserHeader, SICDDESSubheader]
        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        # noinspection PyAttributeOutsideInit
        self._UserHeader = value

    def _set_attribute(self, attribute, value):
        NITFElement._set_attribute(self, attribute, value)
        if attribute == '_UserHeader':
            self._load_header_data()

    def _load_header_data(self):
        """
        Load any user defined header specifics.

        Returns
        -------
        None
        """

        if not isinstance(self._UserHeader, DataExtensionHeader.DESUserHeader):
            return
        if self.DESID.strip() == 'XML_DATA_CONTENT':
            # try loading sicd
            if self._UserHeader.get_bytes_length() == 777:
                # It could be a version 1.0 or greater SICD
                data = self._UserHeader.to_bytes()
                try:
                    data = SICDDESSubheader.from_bytes(data, 0)
                    self._UserHeader = data
                except Exception as e:
                    logging.error(
                        'DESID is "XML_DATA_CONTENT" and data is the right length for SICD, '
                        'but parsing failed with error {}'.format(e))
        elif self.DESID.strip() == 'STREAMING_FILE_HEADER':
            # TODO: LOW Priority - I think that this is deprecated?
            pass

    @classmethod
    def minimum_length(cls):
        return 200


######
# Reserved Extension header

class ReservedExtensionHeader(NITFElement):
    """
    This requires an extension for essentially any non-trivial RES.
    """

    class RESUserHeader(Unstructured):
        _size_len = 4

    __slots__ = ('RE', 'RESID', 'RESVER', '_Security', '_UserHeader')
    _formats = {'RE': '2s', 'RESID': '25s', 'RESVER': '2d'}
    _defaults = {'RE': 'RE', 'RESVER': 1, '_Security': {}, '_UserHeader': {}}
    _types = {
        '_Security': NITFSecurityTags,
        '_UserHeader': RESUserHeader}

    def __init__(self, **kwargs):
        self._skips = set()
        super(ReservedExtensionHeader, self).__init__(**kwargs)

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        return self._Security

    @Security.setter
    def Security(self, value):
        # noinspection PyAttributeOutsideInit
        self._Security = value

    @property
    def UserHeader(self):  # type: () -> RESUserHeader
        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        # noinspection PyAttributeOutsideInit
        self._UserHeader = value

    def _set_attribute(self, attribute, value):
        NITFElement._set_attribute(self, attribute, value)
        if attribute == '_UserHeader':
            self._load_header_data()

    def _load_header_data(self):
        """
        Load any user defined header specifics.

        Returns
        -------
        None
        """

        if not isinstance(self._UserHeader, ReservedExtensionHeader.RESUserHeader):
            return
        pass

    @classmethod
    def minimum_length(cls):
        return 200


######
# The overall header

class NITFHeader(NITFElement):
    """
    The NITF file header - described in SICD standard 2014-09-30, Volume II, page 17
    """

    class ImageSegmentsType(_ItemArrayHeaders):
        _subhead_len = 6
        _item_len = 10

    class GraphicsSegmentsType(_ItemArrayHeaders):
        _subhead_len = 4
        _item_len = 6

    class TextSegmentsType(_ItemArrayHeaders):
        _subhead_len = 4
        _item_len = 5

    class DataExtensionsType(_ItemArrayHeaders):
        _subhead_len = 4
        _item_len = 9

    class ReservedExtensionsType(_ItemArrayHeaders):
        _subhead_len = 4
        _item_len = 7

    __slots__ = (
        'FHDR', 'FVER', 'CLEVEL', 'STYPE',
        'OSTAID', 'FDT', 'FTITLE', '_Security',
        'FSCOP', 'FSCPYS', 'ENCRYP', 'FBKGC',
        'ONAME', 'OPHONE', 'FL', 'HL',
        '_ImageSegments', '_GraphicsSegments', 'NUMX',
        '_TextSegments', '_DataExtensions', '_ReservedExtensions',
        '_UserHeader', '_ExtendedHeader')
    # NB: NUMX is truly reserved for future use, and should always be 0
    _formats = {
        'FHDR': '4s', 'FVER': '5s', 'CLEVEL': '2d', 'STYPE': '4s',
        'OSTAID': '10s', 'FDT': '14s', 'FTITLE': '80s',
        'FSCOP': '5d', 'FSCPYS': '5d', 'ENCRYP': '1s', 'FBKGC': '3b',
        'ONAME': '24s', 'OPHONE': '18s', 'FL': '12d', 'HL': '6d',
        'NUMX': '3d', }
    _defaults = {
        'FHDR': 'NITF', 'FVER': '02.10', 'STYPE': 'BF01',
        'ENCRYP': '0', 'FBKGC': b'\x00\x00\x00',
        'HL': 338, 'NUMX': 0,
        '_Security': {}, '_ImageSegments': {}, '_GraphicsSegments': {},
        '_TextSegments': {}, '_DataExtensions': {}, '_ReservedExtensions': {},
        '_UserHeader': {}, '_ExtendedHeader': {}}
    _types = {
        '_Security': NITFSecurityTags,
        '_ImageSegments': ImageSegmentsType,
        '_GraphicsSegments': GraphicsSegmentsType,
        '_TextSegments': TextSegmentsType,
        '_DataExtensions': DataExtensionsType,
        '_ReservedExtensions': ReservedExtensionsType,
        '_UserHeader': UserHeaderType,
        '_ExtendedHeader': UserHeaderType, }

    def __init__(self, **kwargs):
        super(NITFHeader, self).__init__(**kwargs)
        self._set_HL()

    @property
    def Security(self):  # type: () -> NITFSecurityTags
        """NITFSecurityTags: the security tags instance"""
        return self._Security

    @Security.setter
    def Security(self, value):
        # noinspection PyAttributeOutsideInit
        self._Security = value

    @property
    def ImageSegments(self):  # type: () -> ImageSegmentsType
        return self._ImageSegments

    @ImageSegments.setter
    def ImageSegments(self, value):
        # noinspection PyAttributeOutsideInit
        self._ImageSegments = value

    @property
    def GraphicsSegments(self):  # type: () -> GraphicsSegmentsType
        return self._GraphicsSegments

    @GraphicsSegments.setter
    def GraphicsSegments(self, value):
        # noinspection PyAttributeOutsideInit
        self._GraphicsSegments = value

    @property
    def TextSegments(self):  # type: () -> TextSegmentsType
        return self._TextSegments

    @TextSegments.setter
    def TextSegments(self, value):
        # noinspection PyAttributeOutsideInit
        self._TextSegments = value

    @property
    def DataExtensions(self):  # type: () -> DataExtensionsType
        return self._DataExtensions

    @DataExtensions.setter
    def DataExtensions(self, value):
        # noinspection PyAttributeOutsideInit
        self._DataExtensions = value

    @property
    def ReservedExtensions(self):  # type: () -> ReservedExtensionsType
        return self._ReservedExtensions

    @ReservedExtensions.setter
    def ReservedExtensions(self, value):
        # noinspection PyAttributeOutsideInit
        self._ReservedExtensions = value

    @property
    def UserHeader(self):  # type: () -> Union[UserHeaderType, NITFElement]
        return self._UserHeader

    @UserHeader.setter
    def UserHeader(self, value):
        # noinspection PyAttributeOutsideInit
        self._UserHeader = value

    @property
    def ExtendedHeader(self):  # type: () -> Union[UserHeaderType, NITFElement]
        return self._ExtendedHeader

    @ExtendedHeader.setter
    def ExtendedHeader(self, value):
        # noinspection PyAttributeOutsideInit
        self._ExtendedHeader = value

    def _set_HL(self):
        self.HL = self.get_bytes_length()


#####
# A general nitf header interpreter - intended for extension

class NITFDetails(object):
    """
    This class allows for somewhat general parsing of the header information in
    a NITF 2.1 file. It is expected that the text segment and data extension
    segments must be handled individually, probably by extending this class.
    """

    __slots__ = (
        '_file_name', '_nitf_header',
        'img_subheader_offsets', 'img_segment_offsets',
        'graphics_subheader_offsets', 'graphics_segment_offsets',
        'text_subheader_offsets', 'text_segment_offsets',
        'des_subheader_offsets', 'des_segment_offsets',
        'res_subheader_offsets', 'res_segment_offsets')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF 2.1 file
        """

        self._file_name = file_name

        with open(file_name, mode='rb') as fi:
            # Read the first 9 bytes to verify NITF
            version_info = fi.read(9).decode('utf-8')
            if version_info != 'NITF02.10':
                raise IOError('Not a NITF 2.1 file.')
            # get the header length
            fi.seek(354)  # offset to first field of interest
            header_length = int_func(fi.read(6))
            # go back to the beginning of the file, and parse the whole header
            fi.seek(0)
            header_string = fi.read(header_length)
            self._nitf_header = NITFHeader.from_bytes(header_string, 0)

        curLoc = self._nitf_header.HL
        # populate image segment offset information
        curLoc, self.img_subheader_offsets, self.img_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.ImageSegments)
        # populate graphics segment offset information
        curLoc, self.graphics_subheader_offsets, self.graphics_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.GraphicsSegments)
        # populate text segment offset information
        curLoc, self.text_subheader_offsets, self.text_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.TextSegments)
        # populate data extension offset information
        curLoc, self.des_subheader_offsets, self.des_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.DataExtensions)
        # populate data extension offset information
        curLoc, self.res_subheader_offsets, self.res_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.ReservedExtensions)

    @staticmethod
    def _element_offsets(curLoc, item_array_details):
        # type: (int, _ItemArrayHeaders) -> Tuple[int, Union[None, numpy.ndarray], Union[None, numpy.ndarray]]
        subhead_sizes = item_array_details.subhead_sizes
        item_sizes = item_array_details.item_sizes
        if subhead_sizes.size == 0:
            return curLoc, None, None

        subhead_offsets = numpy.full(subhead_sizes.shape, curLoc, dtype=numpy.int64)
        subhead_offsets[1:] += numpy.cumsum(subhead_sizes[:-1]) + numpy.cumsum(item_sizes[:-1])
        item_offsets = subhead_offsets + subhead_sizes
        curLoc = item_offsets[-1] + item_sizes[-1]
        return curLoc, subhead_offsets, item_offsets

    @property
    def file_name(self):
        """str: the file name."""
        return self._file_name

    @property
    def nitf_header(self):  # type: () -> NITFHeader
        """NITFHeader: the nitf header object"""
        return self._nitf_header

    def parse_image_subheader(self, index):
        """
        Parse the image segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ImageSegmentHeader
        """

        if index >= self.img_subheader_offsets:
            raise IndexError(
                'There are only {} image segments, invalid image segment position {}'.format(self.img_subheader_offsets, index))
        offset = self.img_subheader_offsets[index]
        subhead_size = self._nitf_header.ImageSegments.subhead_sizes[index]
        with open(self._file_name, mode='rb') as fi:
            fi.seek(int_func(offset))
            header_string = fi.read(int_func(subhead_size))
            out = ImageSegmentHeader.from_bytes(header_string, 0)
        return out

    def parse_text_subheader(self, index):
        """
        Parse the text segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        TextSegmentHeader
        """

        if index >= self.text_subheader_offsets.size:
            raise IndexError(
                'There are only {} image segments, invalid image segment position {}'.format(
                    self.text_subheader_offsets.size, index))
        offset = self.text_subheader_offsets[index]
        subhead_size = self._nitf_header.TextSegments.subhead_sizes[index]
        with open(self._file_name, mode='rb') as fi:
            fi.seek(int_func(offset))
            header_string = fi.read(int_func(subhead_size))
            out = TextSegmentHeader.from_bytes(header_string, 0)
        return out

    def parse_graphics_subheader(self, index):
        """
        Parse the graphics segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        GraphicsSegmentHeader
        """

        if index >= self.graphics_subheader_offsets.size:
            raise IndexError(
                'There are only {} image segments, invalid image segment position {}'.format(
                    self.graphics_subheader_offsets.size, index))
        offset = self.graphics_subheader_offsets[index]
        subhead_size = self._nitf_header.GraphicsSegments.subhead_sizes[index]
        with open(self._file_name, mode='rb') as fi:
            fi.seek(int_func(offset))
            header_string = fi.read(int_func(subhead_size))
            out = GraphicsSegmentHeader.from_bytes(header_string, 0)
        return out

    def parse_des_subheader(self, index):
        """
        Parse the data extension segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        DataExtensionHeader
        """

        if index >= self.des_subheader_offsets.size:
            raise IndexError(
                'There are only {} image segments, invalid image segment position {}'.format(
                    self.des_subheader_offsets.size, index))
        offset = self.des_subheader_offsets[index]
        subhead_size = self._nitf_header.DataExtensions.subhead_sizes[index]
        with open(self._file_name, mode='rb') as fi:
            fi.seek(int_func(offset))
            header_string = fi.read(int_func(subhead_size))
            out = DataExtensionHeader.from_bytes(header_string, 0)
        return out

    def parse_res_subheader(self, index):
        """
        Parse the reserved extension subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ReservedExtensionHeader
        """

        if index >= self.res_subheader_offsets.size:
            raise IndexError(
                'There are only {} image segments, invalid image segment position {}'.format(
                    self.res_subheader_offsets.size, index))
        offset = self.res_subheader_offsets[index]
        subhead_size = self._nitf_header.ReservedExtensions.subhead_sizes[index]
        with open(self._file_name, mode='rb') as fi:
            fi.seek(int_func(offset))
            header_string = fi.read(int_func(subhead_size))
            out = ReservedExtensionHeader.from_bytes(header_string, 0)
        return out


##############
# SICD specific elements

class SICDDESSubheader(NITFElement):
    """
    The SICD Data Extension header - described in SICD standard 2014-09-30, Volume II, page 29
    """

    __slots__ = (
        'DESSHL', 'DESCRC', 'DESSHFT', 'DESSHDT',
        'DESSHRP', 'DESSHSI', 'DESSHSV', 'DESSHSD',
        'DESSHTN', 'DESSHLPG', 'DESSHLPT', 'DESSHLI',
        'DESSHLIN', 'DESSHABS', )
    _formats = {
        'DESSHL': '4d', 'DESCRC': '5d', 'DESSHFT': '8s', 'DESSHDT': '20s',
        'DESSHRP': '40s', 'DESSHSI': '60s', 'DESSHSV': '10s',
        'DESSHSD': '20s', 'DESSHTN': '120s', 'DESSHLPG': '125s',
        'DESSHLPT': '25s', 'DESSHLI': '20s', 'DESSHLIN': '120s',
        'DESSHABS': '200s', }
    _defaults = {
        'DESSHL': 773, 'DESCRC': 99999, 'DESSHFT': 'XML',
        'DESSHSI': _SICD_SPECIFICATION_IDENTIFIER,
        'DESSHSV': _SICD_SPECIFICATION_VERSION,
        'DESSHSD': _SICD_SPECIFICATION_DATE,
        'DESSHTN': _SICD_SPECIFICATION_NAMESPACE}
