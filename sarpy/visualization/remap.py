"""
Provides common methods for remapping a complex image to an 8-bit image.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Wade Schwartzkopf", "Thomas McCullough")


import logging
from collections import OrderedDict

import numpy
from scipy.stats import scoreatpercentile as prctile

from sarpy.compliance import string_types


logger = logging.getLogger(__name__)

_DEFAULTS_REGISTERED = False
_REMAP_DICT = OrderedDict()

###########
# helper functions


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
        return slope*numpy.log10(numpy.maximum(amplitude, EPS)) + constant


def _linear_map(data, min_value, max_value):
    """
    Helper function which maps the input data, assumed to be of the correct from,
    into [0, 1] via a linear mapping (data - min_value)(max_value - min_value)
    and then clipping.

    Parameters
    ----------
    data : numpy.ndarray
    min_value : float
    max_value : float

    Returns
    -------
    numpy.ndarray
    """

    return numpy.clip((data - min_value)/float(max_value - min_value), 0, 1)


###########
# registration function for maintaining the list

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
        logger.info('Overwriting the remap {}'.format(remap_name))
        _REMAP_DICT[remap_name] = remap_function
    else:
        logger.info('Remap {} already exists and is not being replaced'.format(remap_name))


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


############
# remap callable classes

class RemapFunction(object):
    _name = '_RemapFunction'
    __slots__ = ('_bit_depth', '_dimension')

    def __init__(self, bit_depth=8, dimension=0):
        """

        Parameters
        ----------
        bit_depth : int
            Should be one of 8 or 16
        dimension : int
            Is expected to be one of 0 (monochromatic) or 3 (rgb)
        """
        self._bit_depth = None
        self._dimension = None

        self._set_bit_depth(bit_depth)
        self._set_dimension(dimension)

    @property
    def name(self):
        """
        str: The (read-only) name for the remap function, which should be (globally)
        unique.
        """

        return self._name

    @property
    def bit_depth(self):
        """
        int: The (read-only) bit depth, which should be either 8 or 16.
        This is expected to be enforced by the implementation directly.
        """

        return self._bit_depth

    def _set_bit_depth(self, value):
        """
        This is intended to be read-only.

        Parameters
        ----------
        value : int
        """

        value = int(value)

        if value not in [8, 16]:
            raise ValueError('Bit depth is required to be one of 8 or 16, got `{}`'.format(value))
        self._bit_depth = value

    @property
    def dimension(self):
        """
        int: The (read-only) size of the final dimension. The value 0 is monochromatic,
        and the output should have identical shape as input. Any other value
        IS EXPECTED to have additional final dimension of this size added.
        """

        return self._dimension

    def _set_dimension(self, value):
        """
        The property is intended to be read-only.

        Parameters
        ----------
        value : int
        """

        value = int(value)
        if value not in [0, 1, 3]:
            raise ValueError('Dimension is required to be one of 8 or 16, got `{}`'.format(value))
        self._dimension = value

    @property
    def output_dtype(self):
        """
        numpy.dtype: The output data type.
        """

        if self._bit_depth == 8:
            return numpy.dtype('u1')
        elif self._bit_depth == 16:
            return numpy.dtype('u2')
        else:
            raise ValueError('Unhandled bit_depth `{}`'.format(self._bit_depth))

    def __call__(self, data):
        """
        This performs the mapping from input data to output discrete version.

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError


class Density(RemapFunction):
    """
    A monochromatic logarithmic density remapping function.

    This is a digested version of contents presented in a 1994 publication
    entitled "Softcopy Display of SAR Data" by Kevin Mangis. It is unclear where
    this was first published or where it may be publicly available.
    """

    __slots__ = ('_bit_depth', '_dimension', '_dmin', '_mmult', '_eps', '_data_mean')
    _name = 'density'

    def __init__(self, bit_depth=8, dmin=30, mmult=40, eps=1e-5, data_mean=None):
        """

        Parameters
        ----------
        bit_depth : int
        dmin : float|int
            A dynamic range parameter. Lower this widens the range, will raising it
            narrows the range. This was historically fixed at 30.
        mmult : float|int
            A contrast parameter. Low values will result is higher contrast and quicker
            saturation, while high values will decrease contrast and slower saturation.
            There is some balance between the competing effects in the `dmin` and `mmult`
            parameters.
        eps : float
            small offset to create a nominal floor when mapping data containing 0's.
        data_mean : None|float|int
            The global data mean (for continuity). The appropriate value will be
            calculated on a per calling array basis if not provided.
        """

        RemapFunction.__init__(self, bit_depth=bit_depth, dimension=0)
        self._data_mean = None
        self._dmin = None
        self._mmult = None
        self._eps = float(eps)

        self._set_dmin(dmin)
        self._set_mmult(mmult)
        self.data_mean = data_mean

    @property
    def dmin(self):
        """
        float: The dynamic range parameter. This is read-only.
        """

        return self._dmin

    def _set_dmin(self, value):
        value = float(value)
        if not (0 <= value < 255):
            raise ValueError('dmin must be in the interval [0, 255), got value {}'.format(value))
        self._dmin = value

    @property
    def mmult(self):
        """
        float: The contrast parameter. This is read only.
        """
        return self._mmult

    def _set_mmult(self, value):
        value = float(value)
        if value < 1:
            raise ValueError('mmult must be < 1, got {}'.format(value))
        self._mmult = value

    @property
    def data_mean(self):
        """
        None|float: The data mean for global use.
        """
        return self._data_mean

    @data_mean.setter
    def data_mean(self, value):
        if value is None:
            self._data_mean = None
            return

        self._data_mean = float(value)

    def __call__(self, data, data_mean=None):

        data_mean = float(data_mean) if data_mean is not None else self._data_mean

        if self.bit_depth == 8:
            return clip_cast(amplitude_to_density(data, dmin=self.dmin, mmult=self.mmult, data_mean=data_mean), self.output_dtype)
        elif self.bit_depth == 16:
            return clip_cast(amplitude_to_density(data, dmin=self.dmin, mmult=self.mmult, data_mean=data_mean), self.output_dtype)
        else:
            raise ValueError('Unsupported bit depth `{}`'.format(self.bit_depth))


class Brighter(Density):
    """
    The density remap using parameters for brighter results.
    """

    _name = 'brighter'

    def __init__(self, bit_depth=8, eps=1e-5, data_mean=None):
        Density.__init__(self, bit_depth=bit_depth, dmin=60, mmult=40, eps=eps, data_mean=data_mean)


class Darker(Density):
    """
    The density remap using parameters for darker results.
    """

    _name = 'darker'

    def __init__(self, bit_depth=8, eps=1e-5, data_mean=None):
        Density.__init__(self, bit_depth=bit_depth, dmin=0, mmult=40, eps=eps, data_mean=data_mean)


class High_Contrast(Density):
    """
    The density remap using parameters for high contrast results.
    """

    _name = 'high_contrast'

    def __init__(self, bit_depth=8, eps=1e-5, data_mean=None):
        Density.__init__(self, bit_depth=bit_depth, dmin=30, mmult=4, eps=eps, data_mean=data_mean)


class Linear(RemapFunction):
    """
    A monochromatic linear remap function.
    """

    __slots__ = ('_bit_depth', '_dimension', '_max_value', '_min_value')
    _name = 'linear'

    def __init__(self, bit_depth=8, min_value=None, max_value=None):
        """

        Parameters
        ----------
        bit_depth : int
        min_value : None|float
        max_value : None|float
        """
        RemapFunction.__init__(self, bit_depth=bit_depth, dimension=0)

        if min_value is not None:
            min_value = float(min_value)
        if max_value is not None:
            max_value = float(max_value)
        self._min_value = min_value
        self._max_value = max_value

    @property
    def min_value(self):
        """
        None|float: The minimum value allowed (clipped below this)
        """
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        if value is None:
            self._min_value = None
        else:
            value = float(value)
            if not numpy.isfinite(value):
                raise ValueError('Got unsupported minimum value `{}`'.format(value))
            self._min_value = value

    @property
    def max_value(self):
        """
        None|float:  The minimum value allowed (clipped above this)
        """

        return self._max_value

    @max_value.setter
    def max_value(self, value):
        if value is None:
            self._max_value = None
        else:
            value = float(value)
            if not numpy.isfinite(value):
                raise ValueError('Got unsupported maximum value `{}`'.format(value))
            self._max_value = value

    def _get_extrema(self, amplitude, min_value, max_value):
        if min_value is not None:
            min_value = float(min_value)
        if max_value is not None:
            max_value = float(max_value)

        if min_value is None:
            min_value = self.min_value
        if min_value is None:
            min_value = numpy.min(amplitude)

        if max_value is None:
            max_value = self.max_value
        if max_value is None:
            max_value = numpy.max(amplitude)

        # sanity check
        if min_value > max_value:
            min_value, max_value = max_value, min_value

        return min_value, max_value

    def __call__(self, data, min_value=None, max_value=None):
        if numpy.iscomplexobj(data):
            amplitude = numpy.abs(data)
        else:
            amplitude = data

        dtype = self.output_dtype
        out = numpy.empty(amplitude.shape, dtype=dtype)

        finite_mask = numpy.isfinite(amplitude)
        out[~finite_mask] = numpy.iinfo(dtype).max

        if numpy.any(finite_mask):
            temp_data = amplitude[finite_mask]

            min_value, max_value = self._get_extrema(temp_data, min_value, max_value)

            if min_value == max_value:
                out[finite_mask] = 0
            else:
                out[finite_mask] = clip_cast(numpy.iinfo(dtype).max*_linear_map(amplitude[finite_mask], min_value, max_value), dtype)
        return out


class Log(RemapFunction):
    """
    A logarithmic remap function.
    """

    __slots__ = ('_bit_depth', '_dimension', '_max_value', '_min_value')
    _name = 'linear'

    def __init__(self, bit_depth=8, min_value=None, max_value=None):
        """

        Parameters
        ----------
        bit_depth : int
        min_value : None|float
        max_value : None|float
        """

        RemapFunction.__init__(self, bit_depth=bit_depth, dimension=0)

        if min_value is not None:
            min_value = float(min_value)
        if max_value is not None:
            max_value = float(max_value)
        self._min_value = min_value
        self._max_value = max_value

    @property
    def min_value(self):
        """
        None|float: The minimum value allowed (clipped below this)
        """
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        if value is None:
            self._min_value = None
        else:
            value = float(value)
            if not numpy.isfinite(value):
                raise ValueError('Got unsupported minimum value `{}`'.format(value))
            self._min_value = value

    @property
    def max_value(self):
        """
        None|float:  The minimum value allowed (clipped above this)
        """

        return self._max_value

    @max_value.setter
    def max_value(self, value):
        if value is None:
            self._max_value = None
        else:
            value = float(value)
            if not numpy.isfinite(value):
                raise ValueError('Got unsupported maximum value `{}`'.format(value))
            self._max_value = value

    def _get_extrema(self, amplitude, min_value, max_value):
        if min_value is not None:
            min_value = float(min_value)
        if max_value is not None:
            max_value = float(max_value)

        if min_value is None:
            min_value = self.min_value
        if min_value is None:
            min_value = numpy.min(amplitude)

        if max_value is None:
            max_value = self.max_value
        if max_value is None:
            max_value = numpy.max(amplitude)

        # sanity check
        if min_value > max_value:
            min_value, max_value = max_value, min_value

        return numpy.log(min_value), numpy.log(max_value)

    def __call__(self, data, min_value=None, max_value=None):
        amplitude = numpy.abs(data)

        dtype = self.output_dtype
        out = numpy.empty(amplitude.shape, dtype=dtype)
        finite_mask = numpy.isfinite(amplitude)
        zero_mask = (amplitude == 0)
        use_mask = finite_mask & (~zero_mask)

        out[~finite_mask] = numpy.iinfo(dtype).max
        out[zero_mask] = 0

        if numpy.any(use_mask):
            temp_data = amplitude[use_mask]
            min_value, max_value = self._get_extrema(temp_data, min_value, max_value)

            if min_value == max_value:
                out[use_mask] = 0
            else:
                out[use_mask] = clip_cast(numpy.iinfo(dtype).max*_linear_map(numpy.log(temp_data), min_value, max_value), dtype)
        return out


class PEDF(RemapFunction):
    """
    A monochromatic piecewise extended density format remap.
    """

    __slots__ = ('_bit_depth', '_dimension')
    _name = 'pedf'

    def __init__(self, bit_depth=8):
        RemapFunction.__init__(self, bit_depth=bit_depth, dimension=0)

        # todo
        pass

    def __call__(self, data):
        raise NotImplementedError


class NRL(RemapFunction):
    """
    A monochromatic remap which is linear for percentile of the data, then
    transitions to logarithmic.
    """

    __slots__ = ('_bit_depth', '_dimension')
    _name = 'nrl'

    def __init__(self, bit_depth=8):
        RemapFunction.__init__(self, bit_depth=bit_depth, dimension=0)

        # todo
        pass

    def __call__(self, data):
        raise NotImplementedError


class RGBLookup(RemapFunction):
    """
    A remap which uses a monochromatic remap function and a 256 color lookup
    table to produce a color image output
    """

    __slots__ = ('_bit_depth', '_dimension', '_mono_remap', '_lookup_table')
    _name= '_rgb_lookup'

    def __init__(self, mono_remap, lookup_table):
        """

        Parameters
        ----------
        mono_remap : RemapFunction
        lookup_table : numpy.ndarray
        """

        self._mono_remap = None
        self._lookup_table = None
        RemapFunction.__init__(self, bit_depth=8, dimension=3)
        self._set_mono_remap(mono_remap)
        self._set_lookup_table(lookup_table)

    @property
    def mono_remap(self):
        """
        RemapFunction: The monochromatic remap being used.
        """
        return self._mono_remap

    def _set_mono_remap(self, value):
        if not (isinstance(value, RemapFunction) and value.dimension == 0 and value.bit_depth == 8):
            raise ValueError('mon_remap requires a monochromatic remap instance with bit_depth=8')
        self._mono_remap = value

    @property
    def lookup_table(self):
        """
        numpy.ndarray: The 256 x 3 8-bit lookup table.
        """

        return self._lookup_table

    def _set_lookup_table(self, value):
        if not (isinstance(value, numpy.ndarray) and value.shape == (256, 3) and value.dtype.name == 'uint8'):
            raise ValueError('lookup_table requires a numpy array of shape (256, 3) and dtype = uint8')
        self._lookup_table = value

    def __call__(self, data, *args, **kwargs):
        return self._lookup_table[self._mono_remap(data, *args, **kwargs)]


###########
# the original flat methods, for backwards compatibility

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
        logger.warning(
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
