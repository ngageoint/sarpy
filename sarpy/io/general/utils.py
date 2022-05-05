"""
Common functionality for converting metadata
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Union, Tuple
import hashlib
import os
import warnings

import numpy

try:
    import h5py
except ImportError:
    h5py = None

# TODO: are validate_range, reverse_range still necessary here?

def validate_range(arg, siz):
    # type: (Union[None, int, Tuple[int, int], Tuple[int, int, int]], int) -> tuple
    """
    Validate the range definition.

    Parameters
    ----------
    arg : None|int|Tuple[int, int]|Tuple[int, int, int]
    siz : int

    Returns
    -------
    tuple
        Of the form `(start, stop, step)`.
    """

    start, stop, step = None, None, None
    if arg is None:
        pass
    elif isinstance(arg, int):
        start = arg
        stop = arg + 1
        step = 1
    else:
        # NB: following this pattern to avoid confused pycharm inspection
        if len(arg) == 1:
            start = int(arg[0])
            stop = start + 1
        elif len(arg) == 2:
            start, stop = arg
        elif len(arg) == 3:
            start, stop, step = arg
    start = 0 if start is None else int(start)
    stop = siz if stop is None else int(stop)
    step = 1 if step is None else int(step)
    # basic validity check
    if not (-siz < start < siz):
        raise ValueError(
            'Range argument {} has extracted start {}, which is required '
            'to be in the range [0, {})'.format(arg, start, siz))
    if not (-siz < stop <= siz):
        raise ValueError(
            'Range argument {} has extracted "stop" {}, which is required '
            'to be in the range [0, {}]'.format(arg, stop, siz))
    if not (0 < abs(step) <= siz):
        raise ValueError(
            'Range argument {} has extracted step {}, for an axis of length '
            '{}'.format(arg, step, siz))
    if ((step < 0) and (stop > start)) or ((step > 0) and (start > stop)):
        raise ValueError(
            'Range argument {} has extracted start {}, stop {}, step {}, '
            'which is not valid.'.format(arg, start, stop, step))

    # reform negative values for start/stop appropriately
    if start < 0:
        start += siz
    if stop < 0:
        stop += siz
    return start, stop, step


def reverse_range(arg, siz):
    # type: (Union[None, int, Tuple[int, int], Tuple[int, int, int]], int) -> Tuple[int, int, int]
    """
    Reverse the range definition.

    Parameters
    ----------
    arg : None|int|Tuple[int,int]|Tuple[int,int,int]
    siz : int

    Returns
    -------
    Tuple[int,int,int]
        Of the form `(start, stop, step)`.
    """

    start, stop, step = validate_range(arg, siz)
    # read backwards
    return (siz - 1) - start, (siz - 1) - stop, -step


###########

def verify_slice(item: Union[None, int, slice, Tuple[int, ...]], max_element: int) -> slice:
    """
    Verify a given slice against a bound.

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

    if isinstance(item, tuple):
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
        subscript: Union[None, int, slice, Ellipsis, Tuple[slice, ...]],
        corresponding_shape: Tuple[int, ...]) -> Tuple[slice, ...]:
    """
    Verify a subscript like item against a corresponding shape.

    Parameters
    ----------
    subscript : None|int|slice|Ellipsis|Tuple[slice, ...]
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
    elif isinstance(subscript, tuple):
        # check for Ellipsis usage...
        ellipsis_location = None
        for index, entry in subscript:
            if entry is Ellipsis:
                if ellipsis_location is None:
                    ellipsis_location = index
                else:
                    raise KeyError('slice definition cannot contain more than one ellipsis')

        if ellipsis_location is not None:
            if len(subscript) > ndim-1:
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


def result_size(
        subscript: Union[None, int, slice, Tuple[slice, ...]],
        corresponding_shape: Tuple[int, ...]) -> (Tuple[slice, ...], Tuple[int, ...]):
    """
    Validate the given subscript against the corresponding shape, and also determine
    the shape of the resultant data reading result.

    Parameters
    ----------
    subscript : None|int|slice|Tuple[slice, ...]
    corresponding_shape : Tuple[int, ...]

    Returns
    -------
    valid_subscript : Tuple[slice, ...]
    output_shape : Tuple[int, ...]
    """

    def out_size(sl_in):
        if sl_in.stop is None:
            return int(sl_in.start/abs(sl_in.step))
        else:
            return int((sl_in.stop - sl_in.start )/sl_in.step)

    subscript = verify_subscript(subscript, corresponding_shape)
    the_shape = tuple([out_size(sl) for sl in subscript])
    return subscript, the_shape


def parse_timestring(str_in, precision='us'):
    if str_in.strip()[-1] == 'Z':
        return numpy.datetime64(str_in[:-1], precision)
    return numpy.datetime64(str_in, precision)


def get_seconds(dt1, dt2, precision='us'):
    """
    The number of seconds between two numpy.datetime64 elements.

    Parameters
    ----------
    dt1 : numpy.datetime64
    dt2 : numpy.datetime64
    precision : str
        one of 's', 'ms', 'us', or 'ns'

    Returns
    -------
    float
        the number of seconds between dt2 and dt1 (i.e. dt1 - dt2).
    """

    if precision == 's':
        scale = 1
    elif precision == 'ms':
        scale = 1e-3
    elif precision == 'us':
        scale = 1e-6
    elif precision == 'ns':
        scale = 1e-9
    else:
        raise ValueError('unrecognized precision {}'.format(precision))

    dtype = 'datetime64[{}]'.format(precision)
    tdt1 = dt1.astype(dtype)
    tdt2 = dt2.astype(dtype)
    return float((tdt1.astype('int64') - tdt2.astype('int64'))*scale)


def is_file_like(the_input):
    """
    Verify whether the provided input appear to provide a "file-like object". This
    term is used ubiquitously, but not all usages are identical. In this case, we
    mean that there exist callable attributes `read`, `write`, `seek`, and `tell`.

    Note that this does not check the mode (binary/string or read/write/append),
    as it is not clear that there is any generally accessible way to do so.

    Parameters
    ----------
    the_input

    Returns
    -------
    bool
    """

    out = True
    for attribute in ['read', 'write', 'seek', 'tell']:
        value = getattr(the_input, attribute, None)
        out &= callable(value)
    return out


def calculate_md5(the_path, chunk_size=1024*1024):
    """
    Calculate the md5 checksum of a given file defined by a path.

    Parameters
    ----------
    the_path : str
        The path to the file
    chunk_size : int
        The chunk size for processing

    Returns
    -------
    str
        The 32 character MD5 hex digest of the given file
    """

    md5_hash = hashlib.md5()
    with open(the_path, 'rb') as fi:
        for chunk in iter(lambda: fi.read(chunk_size), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def is_hdf5(file_name):
    """
    Test whether the given input is a hdf5 file.

    Parameters
    ----------
    file_name : str|BinaryIO

    Returns
    -------
    bool
    """

    if is_file_like(file_name):
        current_location = file_name.tell()
        file_name.seek(0, os.SEEK_SET)
        header = file_name.read(4)
        file_name.seek(current_location, os.SEEK_SET)
    elif isinstance(file_name, str):
        if not os.path.isfile(file_name):
            return False

        with open(file_name, 'rb') as fi:
            header = fi.read(4)
    else:
        return False

    out = (header == b'\x89HDF')
    if out and h5py is None:
        warnings.warn('The h5py library was not successfully imported, and no hdf5 files can be read')
    return out

