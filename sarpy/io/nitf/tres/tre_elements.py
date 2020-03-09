# -*- coding: utf-8 -*-
"""
Module contained elements for defining TREs - really intended as read only objects.
"""

import sys
from collections import OrderedDict
from typing import Union, List


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


def _parse_type(typ_string, leng, value, start):
    """

    Parameters
    ----------
    typ_string : str
    leng : int
    value : bytes
    start : int

    Returns
    -------
    str|int|bytes
    """

    byt = value[start:start + leng]
    if typ_string == 's':
        return byt.decode('utf-8').strip()
    elif typ_string == 'd':
        return int_func(byt)
    elif typ_string == 'b':
        return byt
    else:
        raise ValueError('Got unrecognized type string {}'.format(typ_string))


def _create_format(typ_string, leng):
    if typ_string == 's':
        return '{0:' + '{0:d}'.format(leng) + 's}'
    elif typ_string == 'd':
        return '{0:0' + '{0:d}'.format(leng) + 'd}'
    else:
        return ValueError('Unknown typ_string {}'.format(typ_string))


class TREElement(object):
    def __init__(self):
        self._field_ordering = []
        self._field_format = {}
        self._bytes_length = 0

    def add_field(self, attribute, typ_string, leng, value):
        val = _parse_type(typ_string, leng, value, self._bytes_length)
        setattr(self, attribute, val)
        self._bytes_length += leng
        self._field_ordering.append(attribute)
        self._field_format[attribute] = _create_format(typ_string, leng)
        return

    def add_loop(self, attribute, length, child_type, value):
        obj = TRELoop(length, child_type, value, self._bytes_length)
        setattr(self, attribute, obj)
        self._bytes_length += obj.get_bytes_length()
        self._field_ordering.append(attribute)

    def _attribute_to_bytes(self, attribute):
        val = getattr(self, attribute, None)
        if val is None:
            return b''
        elif isinstance(val, TREElement):
            return val.to_bytes()
        elif isinstance(val, bytes):
            return val
        elif isinstance(val, (int, str)):
            return self._field_format[attribute].format(val)

    def to_dict(self):
        out = OrderedDict()
        for fld in self._field_ordering:
            val = getattr(self, fld)
            if val is None or isinstance(val, (int, str, bytes)):
                out[fld] = val
            elif isinstance(val, TREElement):
                out[fld] = val.to_dict()
            else:
                raise TypeError('Unhandled type {}'.format(type(val)))
        return out

    def to_bytes(self):
        """
        Get the bytes string

        Returns
        -------
        bytes
        """

        items = [self._attribute_to_bytes(fld) for fld in self._field_ordering]
        return b''.join(items)

    def get_bytes_length(self):
        """
        Get the length of the bytes string.

        Returns
        -------
        int
        """

        return self._bytes_length


class TRELoop(TREElement):
    def __init__(self, length, child_type, value, start):
        """

        Parameters
        ----------
        length : int
        child_type : type
        value : bytes
        start : int
        """

        if not issubclass(child_type, TREElement):
            raise TypeError('child_class must be a subclass of TREElement.')

        super(TRELoop, self).__init__()
        self._data = []
        loc = start
        for i in range(length):
            entry = child_type(value)
            leng = entry.get_bytes_length()
            self._bytes_length += leng
            loc += leng
            self._data.append(entry)

    def to_dict(self):
        return [entry.to_dict() for entry in self._data]

    def to_bytes(self):
        return b''.join(entry.to_bytes() for entry in self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):  # type: (Union[int, slice]) -> Union[TREElement, List[TREElement]]
        return self._data[item]


class TREExtension(object):
    __slots__ = ('_data', )
    _tag_value = None
    _data_type = None

    def __init__(self, value):
        if not issubclass(self._data_type, TREElement):
            raise TypeError('_data_type must be a subclass of TREElement. Got type {}'.format(self._data_type))
        if not isinstance(self._tag_value, str):
            raise TypeError('_tag_value must be a string')
        if len(self._tag_value) > 6:
            raise ValueError('Tag value must have 6 or fewer characters.')
        self._data = None
        self.data = value

    @property
    def TAG(self):
        return self._tag_value

    @property
    def data(self):  # type: () -> _data_type
        return self._data

    @data.setter
    def data(self, value):
        # type: (Union[bytes, _data_type]) -> None
        if isinstance(value, self._data_type):
            self._data = value
        elif isinstance(value, bytes):
            self._data = self._data_type(value)
        else:
            raise TypeError(
                'data must be of {} type or a bytes array. '
                'Got {}'.format(self._data_type, type(value)))

    def get_bytes_length(self):
        return 11 + self.data.get_bytes_length()

    def to_bytes(self):
        byts = self.data.to_bytes()
        return '{0:6s}{1:05d}'.format(self.TAG, len(byts)).encode('utf-8') + byts

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value : bytes
        start : int

        Returns
        -------

        """

        tag_value = value[start:start+6].decode('utf-8').strip()
        lng = int_func(value[start+6:start+11])
        if tag_value != cls._tag_value:
            raise ValueError('tag value must be {}. Got {}'.format(cls._tag_value, tag_value))
        return cls(value[start+11:start+11+lng])
