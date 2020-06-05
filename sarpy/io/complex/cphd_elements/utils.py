# -*- coding: utf-8 -*-
"""
Common utils for CPHD functionality.
"""

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
    str
    """

    entries = list(set(_map_part(el) for el in frm.strip().split(';')))
    if len(entries) == 1:
        return entries[0]
    else:
        raise ValueError('The format {} requires multiple struct parts'.format(frm))
