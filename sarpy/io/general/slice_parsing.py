"""
Utilities for parsing slice input.
"""

__classification__ = "UNCLASSIFIED"
__author__ = 'Thomas McCullough'

from typing import Union, Tuple, Sequence
import numpy


def validate_slice_int(the_int: int, bound: int, include: bool = True) -> int:
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

    if not isinstance(bound, int) or bound <= 0:
        raise TypeError('bound must be a positive integer.')
    if include:
        if not -bound <= the_int < bound:
            raise ValueError('Slice argument {} does not fit with bound {}'.format(the_int, bound))
    else:
        if not -bound < the_int <= bound:
            raise ValueError('Slice argument {} does not fit with bound {}'.format(the_int, bound))

    if the_int < 0:
        return the_int + bound
    return the_int


def verify_slice(item: Union[None, int, slice, Tuple[int, ...]], max_element: int) -> slice:
    """
    Verify a given slice against a bound.

    **New in version 1.3.0.**

    Parameters
    ----------
    item : None|int|slice|Tuple[int, ...]
    max_element : int

    Returns
    -------
    slice
        This will certainly have `start` and `step` populated, and will have `stop`
        populated unless `step < 0` and `stop` must be `None`.
    """

    def check_bound(entry: Union[None, int]) -> Union[None, int]:
        if entry is None:
            return entry
        elif -max_element <= entry < 0:
            entry += max_element
            return entry
        elif 0 <= entry <= max_element:
            return entry
        else:
            raise ValueError('Got out of bounds argument ({}) in slice limited by `{}`'.format(entry, max_element))

    if not isinstance(max_element, int) or max_element < 1:
        raise ValueError('slice verification requires a positive integer limit')

    if isinstance(item, Sequence):
        item = slice(*item)

    if item is None:
        return slice(0, max_element, 1)
    elif isinstance(item, int):
        item = check_bound(item)
        return slice(item, item+1, 1)
    elif isinstance(item, slice):
        start = check_bound(item.start)
        stop = check_bound(item.stop)
        step = 1 if item.step is None else item.step
        if step > 0:
            if start is None:
                start = 0
            if stop is None:
                stop = max_element
        if step < 0:
            if start is None:
                start = max_element - 1
        if start is not None and stop is not None:
            if numpy.sign(stop - start) != numpy.sign(step):
                raise ValueError('slice {} is not well formed'.format(item))
        return slice(start, stop, step)
    else:
        raise ValueError('Got unexpected argument of type {} in slice'.format(type(item)))


def verify_subscript(
        subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
        corresponding_shape: Tuple[int, ...]) -> Tuple[slice, ...]:
    """
    Verify a subscript like item against a corresponding shape.

    **New in version 1.3.0**

    Parameters
    ----------
    subscript : None|int|slice|Sequence[int|slice|Tuple[int, ...]]
    corresponding_shape : Tuple[int, ...]

    Returns
    -------
    Tuple[slice, ...]
    """

    ndim = len(corresponding_shape)

    if subscript is None or subscript is Ellipsis:
        return tuple([slice(0, corresponding_shape[i], 1) for i in range(ndim)])
    elif isinstance(subscript, int):
        out = [verify_slice(slice(subscript, subscript + 1, 1), corresponding_shape[0]), ]
        out.extend([slice(0, corresponding_shape[i], 1) for i in range(1, ndim)])
        return tuple(out)
    elif isinstance(subscript, slice):
        out = [verify_slice(subscript, corresponding_shape[0]), ]
        out.extend([slice(0, corresponding_shape[i], 1) for i in range(1, ndim)])
        return tuple(out)
    elif isinstance(subscript, Sequence):
        # check for Ellipsis usage...
        ellipsis_location = None
        for index, entry in enumerate(subscript):
            if entry is Ellipsis:
                if ellipsis_location is None:
                    ellipsis_location = index
                else:
                    raise KeyError('slice definition cannot contain more than one ellipsis')

        if ellipsis_location is not None:
            if len(subscript) > ndim:
                raise ValueError('More subscript entries ({}) than shape dimensions ({}).'.format(len(subscript), ndim))

            if ellipsis_location == len(subscript)-1:
                subscript = subscript[:ellipsis_location]
            elif ellipsis_location == 0:
                init_pad = ndim - len(subscript) + 1
                subscript = tuple([None, ]*init_pad) + subscript[1:]
            else:  # ellipsis in the middle
                middle_pad = ndim - len(subscript) + 1
                subscript = subscript[:ellipsis_location] + tuple([None, ]*middle_pad) + subscript[ellipsis_location+1:]

        if len(subscript) > ndim:
            raise ValueError('More subscript entries ({}) than shape dimensions ({}).'.format(len(subscript), ndim))

        out = [verify_slice(item_i, corresponding_shape[i]) for i, item_i in enumerate(subscript)]
        if len(out) < ndim:
            out.extend([slice(0, corresponding_shape[i], 1) for i in range(len(out), ndim)])
        return tuple(out)
    else:
        raise ValueError('Got unhandled subscript {}'.format(subscript))

def get_slice_result_size(slice_in: slice) -> int:
    """
    Gets the size of the slice result. This assumes a normalized slice definition.

    **New in version 1.3.0.**

    Parameters
    ----------
    slice_in : slice

    Returns
    -------
    int
    """

    # NB: this assumes a normalized slice definition
    if slice_in.step > 0:
        return int(numpy.floor((slice_in.stop - 1 - slice_in.start) / slice_in.step) + 1)
    elif slice_in.stop is None:
        return int(numpy.floor(slice_in.start / abs(slice_in.step)) + 1)
    else:
        return int(numpy.floor((slice_in.stop + 1 - slice_in.start) / slice_in.step) + 1)


def get_subscript_result_size(
        subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
        corresponding_shape: Tuple[int, ...]) -> Tuple[Tuple[slice, ...], Tuple[int, ...]]:
    """
    Validate the given subscript against the corresponding shape, and also determine
    the shape of the resultant data reading result.

    **New in version 1.3.0**

    Parameters
    ----------
    subscript : None|int|slice|Tuple[slice, ...]
    corresponding_shape : Tuple[int, ...]

    Returns
    -------
    valid_subscript : Tuple[slice, ...]
    output_shape : Tuple[int, ...]
    """

    subscript = verify_subscript(subscript, corresponding_shape)
    the_shape = tuple([get_slice_result_size(sl) for sl in subscript])
    return subscript, the_shape
