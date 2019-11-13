"""Module for remapping complex data for display."""

import numpy as np

__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"
__email__ = "wschwartzkopf@integrity-apps.com"

# TODO: there's a whole package for this. Is this necessary? Where are these even used?
#   I question this collection of methods, so I'm setting aside docstring editing until later.
#   I think that methods are only called in the example code in docs, so let's look there.


def get_remap_list():
    """Create list of remap functions accessible from this module.

    Returned list is of (function name, function object) tuples.

    """

    # TODO: LOW - what in the world is this for? It's not called anywhere in the code base.

    # These imports are only used within this function.  Additionally, we don't
    # expect this function to be called frequently, so the small overhead of
    # importing each call is insignicant.
    from inspect import getmembers, isfunction
    import sys

    # We specifically list these as the only funtions in is this module that are
    # not remaps.  If we later add other utility functions to this module, we
    # will have to manually add them to this list as well.  However, we don't
    # have to do anything if we are just adding more remap functions.
    names_nonremap_funs = ['get_remap_list', 'amplitude_to_density', '_clip_cast']
    # Get all functions from this module
    all_funs = getmembers(sys.modules[__name__], isfunction)
    # all_funs is list of (funcion name, function object) tuples.  fun[0] is name.
    just_remap_funs = [fun for fun in all_funs if fun[0] not in names_nonremap_funs]

    return just_remap_funs


def amplitude_to_density(a, dmin=30, mmult=40, data_mean=None):
    """Convert to density data for remap."""
    # TODO: LOW - mixed numpy and native methods and terrible variables names.
    #   Why all these hard-coded parameter values?

    EPS = 1e-5

    a = abs(a)
    if not data_mean:
        data_mean = np.mean(a[np.isfinite(a)])
    cl = 0.8 * data_mean
    ch = mmult * cl
    m = (255 - dmin)/np.log10(ch/cl)
    b = dmin - (m * np.log10(cl))

    return (m * np.log10(np.maximum(a, EPS))) + b


# Does Python not have a builtin way to do this fundamental operation???
def _clip_cast(x, dtype='uint8'):
    """Cast by clipping values outside of valid range, rather than wrapping."""

    # TODO: LOW - dtype = uint8 or uint16, then just scale anyways...max -> top value, min -> minimum value?
    #   I think that the behavior presented here (clipping to min/max values) is native numpy behavior when
    #   converting float types to int types. Converting int types to lower bit depth int type is simply truncation
    #   (i.e. rollover observed)
    #   Am i right in assuming that discretization is actually what's desired, for colormap usage?
    #   This should be made clear...

    np_type = np.dtype(dtype)
    return np.clip(x, np.iinfo(np_type).min, np.iinfo(np_type).max).astype(dtype)


# TODO: hard-coded parameters applied to amplitude_to_density...why?

def density(x):
    """Standard set of parameters for density remap."""
    return _clip_cast(amplitude_to_density(x))


def brighter(x):
    """Brighter set of parameters for density remap."""
    return _clip_cast(amplitude_to_density(x, 60, 40))


def darker(x):
    """Darker set of parameters for density remap."""
    return _clip_cast(amplitude_to_density(x, 0, 40))


def highcontrast(x):
    """Increased contrast set of parameters for density remap."""
    return _clip_cast(amplitude_to_density(x, 30, 4))


def linear(x):
    """Dumb linear remap."""
    if np.iscomplexobj(x):
        return abs(x)  # native method...
    else:
        return x


def log(x):
    """Logarithmic remap."""
    out = np.log(abs(x))  # native method...
    out[np.logical_not(np.isfinite(out))] = np.min(out[np.isfinite(out)])
    return out


def pedf(x):
    """Piecewise extended density format remap."""
    out = amplitude_to_density(x)
    out[out > 128] = 0.5 * (out[out > 128] + 128)
    return _clip_cast(out)


def nrl(x, a=1, c=220):
    """Lin-log style remap.

    Input parameters
    x : data to remap
    a : scale factor of 99th percentile for input "knee"
    c : output "knee" in lin-log curve

    """

    # We include this import inside the function since it is only used here, and
    # since we don't want importing of this module to fail if the user does not
    # have scipy.
    from scipy.stats import scoreatpercentile as prctile

    x = abs(x)  # native method...
    xmin = np.min(x)
    p99 = prctile(x[np.isfinite(x)], 99)
    b = (255 - c) / np.log10((np.max(x) - xmin) / ((a * p99) - xmin))

    out = np.zeros_like(x, np.uint8)  # np.zeros()? jeez.
    linear_region = (x <= a*p99)
    out[linear_region] = (x[linear_region] - xmin) * c / ((a * p99) - xmin)
    out[np.logical_not(linear_region)] = c + (b *
                                              np.log10((x[np.logical_not(linear_region)] - xmin) / ((a * p99) - xmin)))
    return out
