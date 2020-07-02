"""Module for remapping complex data for display."""

from inspect import getmembers, isfunction
import sys

import numpy
from scipy.stats import scoreatpercentile as prctile

__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"


def get_remap_list():
    """
    Create list of remap functions accessible from this module.

    Returns
    -------
    List[Tuple[str, callable], ...]
        List of tuples of the form `(<function name>, <function>)`.
    """

    # We specifically list these as the only funtions in is this module that are
    # not remaps.  If we later add other utility functions to this module, we
    # will have to manually add them to this list as well.  However, we don't
    # have to do anything if we are just adding more remap functions.
    names_nonremap_funs = [
        'get_remap_list', 'amplitude_to_density', '_clip_cast', 'linear_discretization']
    # Get all functions from this module
    all_funs = getmembers(sys.modules[__name__], isfunction)
    # all_funs is list of (function name, function object) tuples.  fun[0] is name.
    just_remap_funs = [fun for fun in all_funs if fun[0] not in names_nonremap_funs]
    return just_remap_funs


def amplitude_to_density(a, dmin=30, mmult=40, data_mean=None):
    """
    Convert to density data for remap.

    Parameters
    ----------
    a : numpy.ndarray
    dmin : float|int
    mmult : float|int
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    EPS = 1e-5

    if (a==0).all():
        return numpy.zeros(a.shape)
    else:
        a = abs(a)
        if not data_mean:
            data_mean = numpy.mean(a[numpy.isfinite(a)])
        cl = 0.8 * data_mean
        ch = mmult * cl
        m = (255 - dmin)/numpy.log10(ch/cl)
        b = dmin - (m * numpy.log10(cl))

        return (m * numpy.log10(numpy.maximum(a, EPS))) + b


# Does Python not have a builtin way to do this fundamental operation???
def _clip_cast(x, dtype='uint8'):
    """
    Cast by clipping values outside of valid range, rather than wrapping.

    Parameters
    ----------
    x : numpy.ndarray
    dtype : str|numpy.dtype

    Returns
    -------
    numpy.ndarray
    """

    np_type = numpy.dtype(dtype)
    return numpy.clip(x, numpy.iinfo(np_type).min, numpy.iinfo(np_type).max).astype(dtype)


def density(x):
    """
    Standard set of parameters for density remap.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    return _clip_cast(amplitude_to_density(x))


def brighter(x):
    """
    Brighter set of parameters for density remap.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    return _clip_cast(amplitude_to_density(x, 60, 40))


def darker(x):
    """
    Darker set of parameters for density remap.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    return _clip_cast(amplitude_to_density(x, 0, 40))


def highcontrast(x):
    """
    Increased contrast set of parameters for density remap.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    return _clip_cast(amplitude_to_density(x, 30, 4))


def linear(x):
    """
    Linear remap - just the magnitude.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    if numpy.iscomplexobj(x):
        return numpy.abs(x)
    else:
        return x


def log(x):
    """
    Logarithmic remap.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    out = numpy.log(numpy.abs(x))
    out[numpy.logical_not(numpy.isfinite(out))] = numpy.min(out[numpy.isfinite(out)])
    return out


def pedf(x):
    """
    Piecewise extended density format remap.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    out = amplitude_to_density(x)
    out[out > 128] = 0.5 * (out[out > 128] + 128)
    return _clip_cast(out)


def nrl(x, a=1, c=220):
    """
    Lin-log style remap.

    Parameters
    ----------
    x : numpy.ndarray
        data to remap
    a : float
        scale factor of 99th percentile for input "knee"
    c : float
        output "knee" in lin-log curve
    Returns
    -------
    numpy.ndarray
    """

    x = numpy.abs(x)
    xmin = numpy.min(x)
    p99 = prctile(x[numpy.isfinite(x)], 99)
    b = (255 - c) / numpy.log10((numpy.max(x) - xmin) / ((a * p99) - xmin))

    out = numpy.zeros_like(x, numpy.uint8)
    linear_region = (x <= a*p99)
    out[linear_region] = (x[linear_region] - xmin) * c / ((a * p99) - xmin)
    out[numpy.logical_not(linear_region)] = c + (b *
                                              numpy.log10((x[numpy.logical_not(linear_region)] - xmin) / ((a * p99) - xmin)))
    return out


def linear_discretization(array, max_value=None, min_value=None, bit_depth=8):
    """
    Make a linearly discretized version of the input array.

    Parameters
    ----------
    array : numpy.ndarray
    max_value : None|int|float
        Value above which to clip (down).
    min_value : None|int|float
        Value below which to clip (up).
    bit_depth : int
        Must be 8 or 16.

    Returns
    -------
    numpy.ndarray
    """

    if bit_depth not in (8, 16):
        raise ValueError('bit_depth must be 8 or 16, got {}'.format(bit_depth))

    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(
            'If both provided, min_value ({}) must be strictly less than '
            'max_value ({}).'.format(min_value, max_value))

    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be an numpy.ndarray, got type {}'.format(type(array)))

    if numpy.iscomplexobj(array):
        array = numpy.abs(array)


    if min_value is None:
        min_value = numpy.min(array)
    if max_value is None:
        max_value = numpy.max(array)

    if min_value == max_value:
        return numpy.zeros(array.shape, dtype=numpy.uint8)

    if bit_depth == 8:
        out = numpy.zeros(array.shape, dtype=numpy.uint8)
        out[:] = (255.0*(numpy.clip(array, min_value, max_value) - min_value))/(max_value - min_value)
        return out
    elif bit_depth == 16:
        out = numpy.zeros(array.shape, dtype=numpy.uint16)
        out[:] = (65535.0*(numpy.clip(array, min_value, max_value) - min_value))/(max_value - min_value)
        return out
    else:
        raise ValueError('Got unhandled bit_depth {}'.format(bit_depth))
