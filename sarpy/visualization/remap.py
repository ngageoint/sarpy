"""Module for remapping complex data for display."""

from typing import List, Tuple
from inspect import getmembers, isfunction
import sys

import numpy as np
from scipy.stats import scoreatpercentile as prctile

__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"


def get_remap_list():
    """
    Create list of remap functions accessible from this module.

    Returns
    -------
    List[Tuple[str, callable]]
        List of tuples of the form `(<function name>, <function>)`.
    """

    # We specifically list these as the only funtions in is this module that are
    # not remaps.  If we later add other utility functions to this module, we
    # will have to manually add them to this list as well.  However, we don't
    # have to do anything if we are just adding more remap functions.
    names_nonremap_funs = ['get_remap_list', 'amplitude_to_density', '_clip_cast']
    # Get all functions from this module
    all_funs = getmembers(sys.modules[__name__], isfunction)
    # all_funs is list of (funcion name, function object) tuples.  fun[0] is name.
    just_remap_funs = [fun for fun in all_funs if fun[0] not in names_nonremap_funs]
    # TODO: LOW - although this is intended to be helpful, its not particularly robust
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

    a = np.abs(a)
    if not data_mean:
        data_mean = np.mean(a[np.isfinite(a)])
    cl = 0.8 * data_mean
    ch = mmult * cl
    m = (255 - dmin)/np.log10(ch/cl)
    b = dmin - (m * np.log10(cl))

    return (m * np.log10(np.maximum(a, EPS))) + b


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

    np_type = np.dtype(dtype)
    return np.clip(x, np.iinfo(np_type).min, np.iinfo(np_type).max).astype(dtype)


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

    if np.iscomplexobj(x):
        return np.abs(x)
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

    out = np.log(np.abs(x))
    out[np.logical_not(np.isfinite(out))] = np.min(out[np.isfinite(out)])
    return out


def pedf(x):
    """
    Piecewise extended density format remap.

    Parameters
    ----------
    x : numnpy.ndarray

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

    x = np.abs(x)
    xmin = np.min(x)
    p99 = prctile(x[np.isfinite(x)], 99)
    b = (255 - c) / np.log10((np.max(x) - xmin) / ((a * p99) - xmin))

    out = np.zeros_like(x, np.uint8)
    linear_region = (x <= a*p99)
    out[linear_region] = (x[linear_region] - xmin) * c / ((a * p99) - xmin)
    out[np.logical_not(linear_region)] = c + (b *
                                              np.log10((x[np.logical_not(linear_region)] - xmin) / ((a * p99) - xmin)))
    return out
