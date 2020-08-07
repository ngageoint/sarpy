# -*- coding: utf-8 -*-
"""
Utilities for parsing slice input.
"""

from sarpy.compliance import integer_types, int_func

__classification__ = "UNCLASSIFIED"
__author__ = 'Thomas McCullough'


def validate_slice_int(the_int, bound, include=True):
    """
    Ensure that the given integer makes sense as a slice entry, and move to
    a normalized form.

    Parameters
    ----------
    the_int : int
    bound : int
    include : bool

    Returns
    -------
    int
    """

    if not isinstance(bound, integer_types) or bound <= 0:
        raise TypeError('bound must be a positive integer.')
    if include:
        if the_int <= -bound or the_int >= bound:
            raise ValueError('Slice argument {} does not fit with bound {}'.format(the_int, bound))
    else:
        if the_int <= -bound or the_int > bound:
            raise ValueError('Slice argument {} does not fit with bound {}'.format(the_int, bound))
    if the_int < 0:
        return the_int + bound
    return the_int


def validate_slice(the_slice, bound):
    """
    Parse a slice into a normalized form.

    Parameters
    ----------
    the_slice : slice
    bound : int

    Returns
    -------
    slice
    """

    if not isinstance(the_slice, slice):
        raise TypeError('the_slice must be a of type slice, got type {}.'.format(type(the_slice)))
    if not isinstance(bound, integer_types) or bound <= 0:
        raise TypeError('bound must be a positive integer.')

    t_start = the_slice.start
    t_stop = the_slice.stop
    t_step = 1 if the_slice.step is None else the_slice.step
    if t_start is None and t_stop is None:
        t_start, t_stop = 0, bound
    elif t_start is None:
        t_stop = validate_slice_int(t_stop, bound, include=False)
        t_start = 0 if t_stop >= 0 else bound - 1
    elif t_stop is None:
        t_start = validate_slice_int(t_start, bound)
        t_stop = -1 if t_step < 0 else bound
    else:
        t_start = validate_slice_int(t_start, bound)
        t_stop = validate_slice_int(t_stop, bound, include=False)
    if (t_step < 0 and t_start < t_stop) or (t_step > 0 and t_start > t_stop):
        raise ValueError(
            'The slice values start={}, stop={}, step={} are not viable'.format(t_start, t_stop, t_step))
    return slice(t_start, t_stop, t_step)
