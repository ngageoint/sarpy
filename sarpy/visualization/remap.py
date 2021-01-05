"""
Provides common methods for remapping a complex image to an 8-bit image.
"""

import logging
from collections import OrderedDict

import numpy
from scipy.stats import scoreatpercentile as prctile

from sarpy.compliance import string_types


__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"


_DEFAULTS_REGISTERED = False
_REMAP_DICT = OrderedDict()


def register_remap(remap_name, remap_function, overwrite=False):
    """
    Register a remap function for general usage.

    Parameters
    ----------
    remap_name : str
    remap_function : callable
    overwrite : bool
        Should we overwrite any currently existing remap of the given name?

    Returns
    -------
    None
    """

    if not isinstance(remap_name, string_types):
        raise TypeError('remap_name must be a string, got type {}'.format(type(remap_name)))
    if not callable(remap_function):
        raise TypeError('remap_function must be callable.')

    if remap_name not in _REMAP_DICT:
        _REMAP_DICT[remap_name] = remap_function
    elif overwrite:
        logging.info('Overwriting the remap {}'.format(remap_name))
        _REMAP_DICT[remap_name] = remap_function
    else:
        logging.info('Remap {} already exists and is not being replaced'.format(remap_name))


def _register_defaults():
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    for remap_name, remap_func in [
            ('density', density),
            ('high_contrast', high_contrast),
            ('brighter', brighter),
            ('darker', darker),
            ('linear', linear),
            ('log', log),
            ('pedf', pedf),
            ('nrl', nrl)]:
        register_remap(remap_name, remap_func)
    _DEFAULTS_REGISTERED = True


def get_remap_list():
    """
    Create list of remap functions accessible from this module.

    Returns
    -------
    List[Tuple[str, callable], ...]
        List of tuples of the form `(<function name>, <function>)`.
    """

    _register_defaults()

    # NB: this was originally implemented via inspection of the callable members
    # of this module, but that ends up requiring more care in excluding
    # undesirable elements than this method
    return [(the_key, the_value) for the_key, the_value in _REMAP_DICT.items()]


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


def clip_cast(x, dtype='uint8'):
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
    return numpy.clip(x, numpy.iinfo(np_type).min, numpy.iinfo(np_type).max).astype(np_type)


def density(x, data_mean=None):
    """
    Standard set of parameters for density remap.

    Parameters
    ----------
    x : numpy.ndarray
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    return clip_cast(amplitude_to_density(x, data_mean=data_mean))


def brighter(x, data_mean=None):
    """
    Brighter set of parameters for density remap.

    Parameters
    ----------
    x : numpy.ndarray
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    return clip_cast(amplitude_to_density(x, dmin=60, mmult=40, data_mean=data_mean))


def darker(x, data_mean=None):
    """
    Darker set of parameters for density remap.

    Parameters
    ----------
    x : numpy.ndarray
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    return clip_cast(amplitude_to_density(x, dmin=0, mmult=40, data_mean=data_mean))


def high_contrast(x, data_mean=None):
    """
    Increased contrast set of parameters for density remap.

    Parameters
    ----------
    x : numpy.ndarray
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    return clip_cast(amplitude_to_density(x, dmin=30, mmult=4, data_mean=data_mean))


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


def pedf(x, data_mean=None):
    """
    Piecewise extended density format remap.

    Parameters
    ----------
    x : numpy.ndarray
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    out = amplitude_to_density(x, data_mean=data_mean)
    out[out > 128] = 0.5 * (out[out > 128] + 128)
    return clip_cast(out)


def nrl(x, a=1., c=220.):
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
    xmax = numpy.max(x)
    xmin = numpy.min(x)
    p99 = prctile(x[numpy.isfinite(x)], 99)
    b = (255 - c)/(numpy.log10(xmax - xmin)*(a*p99 - xmin))

    out = numpy.zeros_like(x, numpy.uint8)
    linear_region = (x <= a*p99)
    out[linear_region] = c*(x[linear_region] - xmin)/(a*p99 - xmin)
    out[~linear_region] = c + b*numpy.log10((x[~linear_region] - xmin)/(a*p99 - xmin))
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
