"""
Common functionality for converting metadata
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Union, Tuple, BinaryIO, Any, Optional
import hashlib
import os
import warnings
import struct
import io

import numpy

try:
    import h5py
except ImportError:
    h5py = None


# TODO: validate_range still necessary here?

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


###########
# general file type checks

def is_file_like(the_input: Any) -> bool:
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


def is_real_file(the_input: BinaryIO) -> bool:
    """
    Determine if the file-like object is associated with an actual file.
    This is mainly to consider suitability for establishment of a numpy.memmap.

    Parameters
    ----------
    the_input : BinaryIO

    Returns
    -------
    bool
    """

    if hasattr(the_input, 'fileno'):
        try:
            # check that fileno actually works, not just exists.
            the_input.fileno()
            return True
        except io.UnsupportedOperation:
            return False


def _fetch_initial_bytes(file_name: Union[str, BinaryIO], size: int) -> Union[None, bytes]:
    header = b''
    if is_file_like(file_name):
        current_location = file_name.tell()
        file_name.seek(0, os.SEEK_SET)
        header = file_name.read(size)
        file_name.seek(current_location, os.SEEK_SET)
    elif isinstance(file_name, str):
        if not os.path.isfile(file_name):
            return None

        with open(file_name, 'rb') as fi:
            header = fi.read(size)

    if len(header) != size:
        return None
    return header


def is_nitf(
        file_name: Union[str, BinaryIO],
        return_version=False) -> Union[bool, Tuple[bool, Optional[str]]]:
    """
    Test whether the given input is a NITF 2.0 or 2.1 file.

    Parameters
    ----------
    file_name : str|BinaryIO
    return_version : bool


    Returns
    -------
    is_nitf_file: bool
        Is the file a NITF file, based solely on checking initial bytes.
    nitf_version: None|str
        Only returned is `return_version=True`. Will be `None` in the event that
        `is_nitf_file=False`.
    """

    header = _fetch_initial_bytes(file_name, 9)
    if header is None:
        if return_version:
            return False, None
        else:
            return False

    ihead = header[:4]
    vers = header[4:]
    if ihead == b'NITF':
        try:
            vers = vers.decode('utf-8')
            return (True, vers) if return_version else True
        except ValueError:
            pass

    return (False, None) if return_version else False


def is_tiff(
        file_name: Union[str, BinaryIO],
        return_details=False) -> Union[bool, Tuple[bool, Optional[str], Optional[int]]]:
    """
    Test whether the given input is a tiff or big_tiff file.

    Parameters
    ----------
    file_name : str|BinaryIO
    return_details : bool
        Return the tiff details of endianess and magic number?

    Returns
    -------
    is_tiff_file : bool
    endianness : None|str
        Only returned if `return_details` is `True`. One of `['>', '<']`.
    magic_number : None|int
        Only returned if `return_details` is `True`. One of `[42, 43]`.
    """

    header = _fetch_initial_bytes(file_name, 4)
    if header is None:
        return (False, None, None) if return_details else False

    try:
        endian_part = header[:2].decode('utf-8')
    except ValueError:
        return (False, None, None) if return_details else False

    if endian_part not in ['II', 'MM']:
        return (False, None, None) if return_details else False

    if endian_part == 'II':
        endianness = '<'
    else:
        endianness = '>'
    magic_number = struct.unpack('{}h'.format(endianness), header[2:])[0]

    if magic_number in [42, 43]:
        # NB: 42 is regular tiff, while 43 is big tiff
        return (True, magic_number, endianness) if return_details else True
    return (False, None, None) if return_details else False


def is_hdf5(file_name: Union[str, BinaryIO]) -> bool:
    """
    Test whether the given input is a hdf5 file.

    Parameters
    ----------
    file_name : str|BinaryIO

    Returns
    -------
    bool
    """

    header = _fetch_initial_bytes(file_name, 4)
    if header is None:
        return False

    out = (header == b'\x89HDF')
    if out and h5py is None:
        warnings.warn('The h5py library was not successfully imported, and no hdf5 files can be read')
    return out


###########

def parse_timestring(str_in: str, precision: str='us') -> numpy.datetime64:
    """
    Parse (naively) a timestring to numpy.datetime64 of the given precision.

    Parameters
    ----------
    str_in : str
    precision : str
        See numpy.datetime64 for precision options.

    Returns
    -------
    numpy.datetime64
    """

    if str_in.strip()[-1] == 'Z':
        return numpy.datetime64(str_in[:-1], precision)
    return numpy.datetime64(str_in, precision)


def get_seconds(dt1: numpy.datetime64,
                dt2: numpy.datetime64,
                precision: str='us') -> float:
    """
    The number of seconds between two numpy.datetime64 elements.

    Parameters
    ----------
    dt1 : numpy.datetime64
    dt2 : numpy.datetime64
    precision : str
        one of 's', 'ms', 'us', or 'ns'.

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


def calculate_md5(the_path: str, chunk_size: int=1024*1024) -> str:
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
