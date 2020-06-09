# -*- coding: utf-8 -*-
"""
Common utils for CPHD functionality.
"""

import numpy

_format_mapping = {
    'F8': 'd', 'F4': 'f', 'I8': 'q', 'I4': 'i', 'I2': 'h', 'I1': 'b'}


def _map_part(part):
    parts = part.split('=')
    if len(parts) != 2:
        raise ValueError('Cannot parse format part {}'.format(part))
    res = _format_mapping.get(parts[1].strip(), None)
    if res is None:
        raise ValueError('Cannot parse format part {}'.format(part))
    return res


def parse_format(frm):
    """
    Determine a struct format from a CPHD format string.

    Parameters
    ----------
    frm : str

    Returns
    -------
    Tuple[str, ...]
    """

    return tuple(_map_part(el) for el in frm.strip().split(';'))

def homogeneous_format(frm, return_length=False):
    """
    Determine a struct format from a CPHD format string, requiring that any multiple
    parts are all identical.

    Parameters
    ----------
    frm : str
    return_length : bool
        Return the number of elements?

    Returns
    -------
    str
    """

    entries = parse_format(frm)
    entry_set = set(entries)
    if len(entry_set) == 1:
        return entry_set.pop(), len(entries) if return_length else entry_set.pop()
    else:
        raise ValueError('Non-homogeneous format required {}'.format(entries))

def homogeneous_dtype(frm, return_length=False):
    """
    Determine a numpy.dtype (including endianness) from a CPHD format string, requiring
    that any multiple parts are all identical.

    Parameters
    ----------
    frm : str
    return_length : bool
        Return the number of elements?

    Returns
    -------
    numpy.dtype
    """

    entries = ['>'+el.lower() for el in frm.strip().split(';')]
    entry_set = set(entries)
    if len(entry_set) == 1:
        return numpy.dtype(entry_set.pop()), len(entries) if return_length else numpy.dtype(entry_set.pop())
    else:
        raise ValueError('Non-homogeneous format required {}'.format(entries))
