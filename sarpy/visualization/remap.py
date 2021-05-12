"""
Provides common methods for remapping a complex image to an 8-bit image.
"""

import logging
from collections import OrderedDict

import numpy
from scipy.stats import scoreatpercentile as prctile

from sarpy.compliance import string_types


__classification__ = "UNCLASSIFIED"
__author__ = ("Wade Schwartzkopf", "Thomas McCullough")


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


def amplitude_to_density(data, dmin=30, mmult=40, data_mean=None):
    """
    Convert to density data for remap.

    This is a digested version of contents presented in a 1994 pulication
    entitled "Softcopy Display of SAR Data" by Kevin Mangis. It is unclear where
    this was first published or where it may be publically available.

    Parameters
    ----------
    data : numpy.ndarray
        The (presumably complex) data to remap
    dmin : float|int
        A dynamic range parameter. Lower this widens the range, will raising it
        narrows the range. This was historically fixed at 30.
    mmult : float|int
        A contrast parameter. Low values will result is higher contrast and quicker
        saturation, while high values will decrease contrast and slower saturation.
        There is some balance between the competing effects in the `dmin` and `mmult`
        parameters.
    data_mean : None|float|int
        The data mean (for this or the parent array for continuity), which will
        be calculated if not provided.

    Returns
    -------
    numpy.ndarray
    """

    dmin = float(dmin)
    if not (0 <= dmin < 255):
        raise ValueError('Invalid dmin value {}'.format(dmin))

    mmult = float(mmult)
    if mmult < 1:
        raise ValueError('Invalid mmult value {}'.format(mmult))

    EPS = 1e-5
    amplitude = numpy.abs(data)
    if numpy.all(amplitude == 0):
        return amplitude
    else:
        if not data_mean:
            data_mean = numpy.mean(amplitude[numpy.isfinite(amplitude)])
        # remap parameters
        C_L = 0.8*data_mean
        C_H = mmult*C_L  # decreasing mmult will result in higher contrast (and quicker saturation)
        slope = (255 - dmin)/numpy.log10(C_H/C_L)
        constant = dmin - (slope*numpy.log10(C_L))
        # NB: C_H/C_L trivially collapses to mmult, but this is maintained for
        # clarity in historical reference
        # Originally, C_L and C_H were static values drawn from a determined set
        # of remap look-up tables. The C_L/C_H values were presumably based roughly
        # on mean amplitude and desired rempa brightness/contrast. The dmin value
        # was fixed as 30.
        return (slope*numpy.log10(numpy.maximum(amplitude, EPS))) + constant


def clip_cast(array, dtype='uint8'):
    """
    Cast by clipping values outside of valid range, rather than wrapping.

    Parameters
    ----------
    array : numpy.ndarray
    dtype : str|numpy.dtype

    Returns
    -------
    numpy.ndarray
    """

    np_type = numpy.dtype(dtype)
    return numpy.clip(array, numpy.iinfo(np_type).min, numpy.iinfo(np_type).max).astype(np_type)


def density(data, data_mean=None):
    """
    Standard set of parameters for density remap.

    Parameters
    ----------
    data : numpy.ndarray
        The data to remap.
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    return clip_cast(amplitude_to_density(data, data_mean=data_mean))


def brighter(data, data_mean=None):
    """
    Brighter set of parameters for density remap.

    Parameters
    ----------
    data : numpy.ndarray
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    return clip_cast(amplitude_to_density(data, dmin=60, mmult=40, data_mean=data_mean))


def darker(data, data_mean=None):
    """
    Darker set of parameters for density remap.

    Parameters
    ----------
    data : numpy.ndarray
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    return clip_cast(amplitude_to_density(data, dmin=0, mmult=40, data_mean=data_mean))


def high_contrast(data, data_mean=None):
    """
    Increased contrast set of parameters for density remap.

    Parameters
    ----------
    data : numpy.ndarray
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    return clip_cast(amplitude_to_density(data, dmin=30, mmult=4, data_mean=data_mean))


def linear(data):
    """
    Linear remap - just the magnitude.

    Parameters
    ----------
    data : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    if numpy.iscomplexobj(data):
        amplitude = numpy.abs(data)
    else:
        amplitude = numpy.copy(data)

    finite_mask = numpy.isfinite(amplitude)
    min_value = numpy.min(amplitude[finite_mask])
    max_value = numpy.max(amplitude[finite_mask])

    return clip_cast(255.*(amplitude - min_value)/(max_value - min_value), 'uint8')


def log(data):
    """
    Logarithmic remap.

    Parameters
    ----------
    data : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    out = numpy.abs(data)
    out[out < 0] = 0
    out += 1  # bump values up over 1.

    finite_mask = numpy.isfinite(out)
    if not numpy.any(finite_mask):
        out[:] = 0
        return out.astype('uint8')

    log_values = numpy.log(out[finite_mask])
    min_value = numpy.min(log_values)
    max_value = numpy.max(log_values)

    out[finite_mask] = 255*(log_values - min_value)/(max_value - min_value)
    out[~finite_mask] = 255
    return out.astype('uint8')


def pedf(data, data_mean=None):
    """
    Piecewise extended density format remap.

    Parameters
    ----------
    data : numpy.ndarray
        The array to be remapped.
    data_mean : None|float|int

    Returns
    -------
    numpy.ndarray
    """

    out = amplitude_to_density(data, data_mean=data_mean)
    out[out > 128] = 0.5 * (out[out > 128] + 128)
    return clip_cast(out)


def _nrl_stats(amplitude):
    """
    Calculate the statistiucs for input into the nrl remap.

    Parameters
    ----------
    amplitude : numpy.ndarray
        The amplitude array, assumed real valued.

    Returns
    -------
    tuple
        Of the form `(minimum, maximum, 99th percentile)
    """

    finite_mask = numpy.isfinite(amplitude)
    if numpy.any(finite_mask):
        return numpy.min(amplitude[finite_mask]), numpy.max(amplitude[finite_mask]), prctile(amplitude[finite_mask], 99)
    else:
        return 0, 0, 0


def nrl(data, knee=220, stats=None):
    """
    A lin-log style remap.

    Parameters
    ----------
    data : numpy.ndarray
        The data array to remap
    knee : float|int
        The knee of the lin-log transition.
    stats : None|tuple
        This is calculated if not provided. Expected to be of the form
        `(minimum, maximum, 99th percentile)`.

    Returns
    -------
    numpy.ndarray
    """

    if not (0 < knee < 255):
        raise ValueError('The knee value must be strictly between 0 and 255.')
    knee = float(knee)

    out = numpy.abs(data)  # starts as amplitude, and will be redefined in place
    if stats is None:
        stats = _nrl_stats(out)

    amplitude_min, amplitude_max, amplitude_99 = stats
    if not (amplitude_min <= amplitude_99 <= amplitude_max):
        raise ValueError('Got inconsistent stats values {}'.format(stats))

    if amplitude_min == amplitude_max:
        out[:] = 0
        return out.astype('uint8')

    linear_region = (out <= amplitude_99)
    if amplitude_99 > amplitude_min:
        out[linear_region] = knee*(out[linear_region] - amplitude_min)/(amplitude_99 - amplitude_min)
    else:
        logging.warning(
            'The remap array is at least 99% constant, the nrl remap may return '
            'strange results.')
        out[linear_region] = 0

    if amplitude_99 == amplitude_max:
        out[~linear_region] = knee
    else:
        # calulate the log values
        log_values = (out[~linear_region] - amplitude_99)/(amplitude_max - amplitude_99) + 1
        # this is now linearly scaled from 1 to 2, apply log_2 and then scale appropriately
        out[~linear_region] = numpy.log2(log_values)*(255 - knee) + knee
    return clip_cast(out, 'uint8')


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
