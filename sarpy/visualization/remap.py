"""
Provides common methods for remapping a complex or other array to 8 or 16-bit
image type arrays.

Note: The original function and 8-bit implementation has been replaced with a
class based solution which allows state variables associated with the remap
function, and support for 16-bit versions, as well as an 8-bit MA, RGB or RGBA
lookup tables.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Wade Schwartzkopf", "Thomas McCullough")


import logging
from collections import OrderedDict
from typing import Dict
import warnings

import numpy

from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.utils import get_data_mean_magnitude, stats_calculation, \
    get_data_extrema

try:
    from matplotlib import cm
except ImportError:
    cm = None


logger = logging.getLogger(__name__)

_DEFAULTS_REGISTERED = False
_REMAP_DICT = OrderedDict()  # type: Dict[str, RemapFunction]

###########
# helper functions


def clip_cast(array, dtype='uint8', min_value=None, max_value=None):
    """
    Cast by clipping values outside of valid range, rather than truncating.

    Parameters
    ----------
    array : numpy.ndarray
    dtype : str|numpy.dtype
    min_value : None|int|float
    max_value : None|int|float

    Returns
    -------
    numpy.ndarray
    """

    np_type = numpy.dtype(dtype)
    min_value = numpy.iinfo(np_type).min if min_value is None else max(min_value, numpy.iinfo(np_type).min)
    max_value = numpy.iinfo(np_type).max if max_value is None else max(max_value, numpy.iinfo(np_type).max)
    return numpy.clip(array, min_value, max_value).astype(np_type)


def amplitude_to_density(data, dmin=30, mmult=40, data_mean=None):
    """
    Convert to density data for remap.

    This is a digested version of contents presented in a 1994 pulication
    entitled "Softcopy Display of SAR Data" by Kevin Mangis. It is unclear where
    this was first published or where it may be publicly available.

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


def _nrl_stats(amplitude, percentile=99):
    """
    Calculate the statistics for input into the nrl remap.

    Parameters
    ----------
    amplitude : numpy.ndarray
        The amplitude array, assumed real valued.
    percentile : float|int
        Which percentile to calculate

    Returns
    -------
    tuple
        Of the form `(minimum, maximum, `percentile` percentile)
    """

    finite_mask = numpy.isfinite(amplitude)
    if numpy.any(finite_mask):
        return stats_calculation(amplitude[finite_mask], percentile=percentile)
    else:
        return 0, 0, 0


###########
# registration function for maintaining the list

def register_remap(remap_function, overwrite=False):
    """
    Register a remap function for general usage.

    Parameters
    ----------
    remap_function : RemapFunction|Type
    overwrite : bool
        Should we overwrite any currently existing remap of the given name?

    Returns
    -------
    None
    """

    if isinstance(remap_function, type) and issubclass(remap_function, RemapFunction):
        remap_function = remap_function()
    if not isinstance(remap_function, RemapFunction):
        raise TypeError('remap_function must be an instance of RemapFunction.')

    remap_name = remap_function.name

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
    register_remap(NRL(bit_depth=8), overwrite=False)
    register_remap(Density(bit_depth=8), overwrite=False)
    register_remap(High_Contrast(bit_depth=8), overwrite=False)
    register_remap(Brighter(bit_depth=8), overwrite=False)
    register_remap(Darker(bit_depth=8), overwrite=False)
    register_remap(Linear(bit_depth=8), overwrite=False)
    register_remap(Logarithmic(bit_depth=8), overwrite=False)
    register_remap(PEDF(bit_depth=8), overwrite=False)
    if cm is not None:
        try:
            register_remap(LUT8bit(NRL(bit_depth=8), 'viridis', use_alpha=False), overwrite=False)
        except KeyError:
            pass
        try:
            register_remap(LUT8bit(NRL(bit_depth=8), 'magma', use_alpha=False), overwrite=False)
        except KeyError:
            pass
        try:
            register_remap(LUT8bit(NRL(bit_depth=8), 'rainbow', use_alpha=False), overwrite=False)
        except KeyError:
            pass
        try:
            register_remap(LUT8bit(NRL(bit_depth=8), 'bone', use_alpha=False), overwrite=False)
        except KeyError:
            pass

    _DEFAULTS_REGISTERED = True


def get_remap_names():
    """
    Gets a list of currently registered remap function names.

    Returns
    -------
    List[str]
    """

    if not _DEFAULTS_REGISTERED:
        _register_defaults()
    return list(_REMAP_DICT.keys())


def get_remap_list():
    """
    Gets a list of currently registered remaps.

    Returns
    -------
    List[Tuple[str, RemapFunction], ...]
        List of tuples of the form `(<name>, <RemapFunction instance>)`.
    """

    if not _DEFAULTS_REGISTERED:
        _register_defaults()

    # NB: this was originally implemented via inspection of the callable members
    # of this module, but that ends up requiring more care in excluding
    # undesirable elements than this method
    return [(the_key, the_value) for the_key, the_value in _REMAP_DICT.items()]


def get_registered_remap(remap_name, default=None):
    """
    Gets a remap function from it's registered name.

    Parameters
    ----------
    remap_name : str
    default : None|RemapFunction

    Returns
    -------
    RemapFunction

    Raises
    ------
    KeyError
    """

    if not _DEFAULTS_REGISTERED:
        _register_defaults()

    if remap_name in _REMAP_DICT:
        return _REMAP_DICT[remap_name]
    if default is not None:
        return default
    raise KeyError('Unregistered remap name `{}`'.format(remap_name))


############
# remap callable classes

class RemapFunction(object):
    """
    Abstract remap class which is callable.

    See the :func:`call` implementation for the given class to understand
    what specific keyword arguments are allowed for the specific instance.
    """

    _name = '_RemapFunction'
    __slots__ = ('_override_name', '_bit_depth', '_dimension')
    _allowed_dimension = {0, 1, 2, 3, 4}

    def __init__(self, override_name=None, bit_depth=8, dimension=0):
        """

        Parameters
        ----------
        override_name : None|str
            Override name for a specific class instance
        bit_depth : int
            Should be one of 8 or 16
        dimension : int
        """
        self._override_name = None
        self._bit_depth = None
        self._dimension = None
        self._set_name(override_name)
        self._set_bit_depth(bit_depth)
        self._set_dimension(dimension)

    @property
    def name(self):
        """
        str: The (read-only) name for the remap function. This will be the
        override_name if one has been provided for this instance, otherwise it
        will be the generic `_name` class value.
        """

        return self._name if self._override_name is None else self._override_name

    def _set_name(self, value):
        if value is None or isinstance(value, str):
            self._override_name = value
        else:
            raise ValueError('Got incompatible name')

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

        if value not in [8, 16, 32]:
            raise ValueError('Bit depth is required to be one of 8, 16, or 32 and we got `{}`'.format(value))
        self._bit_depth = value

    @property
    def dimension(self):
        """
        int: The (read-only) size of the (additional) output final dimension.
        The value 0 is monochromatic, where the retuned output will have identical
        shape as input. Any other value should have additional final dimension of this size.
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
        if self._allowed_dimension is not None and value not in self._allowed_dimension:
            raise ValueError(
                'Dimension is required to be one of `{}`, got `{}`'.format(self._allowed_dimension, value))
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
        elif self._bit_depth == 32:
            return numpy.dtype('u4')
        else:
            raise ValueError('Unhandled bit_depth `{}`'.format(self._bit_depth))

    @property
    def are_global_parameters_set(self):
        """
        bool: Are (all) global parameters used for applying this remap function
        set? This should return `True` if there are no global parameters.
        """

        return True

    def raw_call(self, data, **kwargs):
        """
        This performs the mapping from input data to output floating point
        version, this is directly used by the :func:`call` method.

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        kwargs
            Some keyword arguments may be allowed here

        Returns
        -------
        numpy.ndarray
            This should generally have `float64` dtype.
        """

        raise NotImplementedError

    def call(self, data, **kwargs):
        """
        This performs the mapping from input data to output discrete version.

        This method os directly called by the :func:`__call__` method, so the
        class instance (once constructed) is itself callable, as follows:

        >>> remap = RemapFunction()
        >>> discrete_data = remap(data, **kwargs)

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        kwargs
            Some keyword arguments may be allowed here

        Returns
        -------
        numpy.ndarray
        """

        return clip_cast(self.raw_call(data, **kwargs), dtype=self.output_dtype)

    def __call__(self, data, **kwargs):
        return self.call(data, **kwargs)

    @staticmethod
    def _validate_pixel_bounds(reader, index, pixel_bounds):
        data_size = reader.get_data_size_as_tuple()[index]
        if pixel_bounds is None:
            return 0, data_size[0], 0, data_size[1]

        if not (
                (-data_size[0] <= pixel_bounds[0] <= data_size[0]) and
                (-data_size[0] <= pixel_bounds[1] <= data_size[0]) and
                (-data_size[1] <= pixel_bounds[2] <= data_size[1]) and
                (-data_size[1] <= pixel_bounds[3] <= data_size[1])):
            raise ValueError('invalid pixel bounds `{}` for data of shape `{}`'.format(pixel_bounds, data_size))
        return pixel_bounds

    def calculate_global_parameters_from_reader(self, reader, index=0, pixel_bounds=None):
        """
        Calculates any useful global bounds for the specified reader, the given
        index, and inside the given pixel bounds.

        This is expected to save ny necessary state here.

        Parameters
        ----------
        reader : SICDTypeReader
        index : int
        pixel_bounds : None|tuple|list|numpy.ndarray
            If provided, is of the form `(row min, row max, column min, column max)`.

        Returns
        -------
        None
        """

        raise NotImplementedError


class MonochromaticRemap(RemapFunction):
    """
    Abstract monochromatic remap class.
    """

    _name = '_Monochromatic'
    __slots__ = ('_override_name', '_bit_depth', '_dimension','_max_output_value')
    _allowed_dimension = {0, }

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None):
        r"""

        Parameters
        ----------
        override_name : None|str
            Override name for a specific class instance
        bit_depth : int
        max_output_value : None|int
            The maximum output value. If provided, this must be in the interval
            :math:`[0, 2^{bit\_depth}]`
        """

        self._max_output_value = None
        RemapFunction.__init__(self, override_name=override_name, bit_depth=bit_depth, dimension=0)
        self._set_max_output_value(max_output_value)

    @property
    def max_output_value(self):
        """
        int: The (read-only) maximum output value size.
        """

        return self._max_output_value

    def _set_max_output_value(self, value):
        max_possible = numpy.iinfo(self.output_dtype).max
        if value is None:
            value = max_possible
        else:
            value = int(value)

        if 0 < value <= max_possible:
            self._max_output_value = value
        else:
            raise ValueError(
                'the max_output_value must be between 0 and {}, '
                'got {}'.format(max_possible, value))

    def raw_call(self, data, **kwargs):
        raise NotImplementedError

    def calculate_global_parameters_from_reader(self, reader, index=0, pixel_bounds=None):
        raise NotImplementedError


############
# basic monchromatic collection

class Density(MonochromaticRemap):
    """
    A monochromatic logarithmic density remapping function.

    This is a digested version of contents presented in a 1994 publication
    entitled "Softcopy Display of SAR Data" by Kevin Mangis. It is unclear where
    this was first published or where it may be publicly available.
    """

    __slots__ = ('_override_name', '_bit_depth', '_dimension', '_dmin', '_mmult', '_eps', '_data_mean')
    _name = 'density'

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None,
                 dmin=30, mmult=40, eps=1e-5, data_mean=None):
        r"""

        Parameters
        ----------
        override_name : None|str
            Override name for a specific class instance
        bit_depth : int
        max_output_value : None|int
            The maximum output value. If provided, this must be in the interval
            :math:`[0, 2^{bit\_depth}]`
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

        MonochromaticRemap.__init__(self, override_name=override_name, bit_depth=bit_depth, max_output_value=max_output_value)
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

    @property
    def are_global_parameters_set(self):
        """
        bool: Is the global parameters used for applying this remap function
        set? In this case, this is the `data_mean` property.
        """

        return self._data_mean is not None

    def raw_call(self, data, data_mean=None):
        """
        This performs the mapping from input data to output floating point
        version, this is directly used by the :func:`call` method.

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        data_mean : None|float
            The pre-calculated data mean, for consistent global use. The order
            of preference is the value provided here, the class data_mean property
            value, then the value calculated from the present sample.

        Returns
        -------
        numpy.ndarray
        """

        data_mean = float(data_mean) if data_mean is not None else self._data_mean
        # the amplitude_to_density function is ort specifically geared towards
        # dynamic range of 0 - 255, just adjust it.
        multiplier = float(self.max_output_value)/255.0
        return multiplier*amplitude_to_density(data, dmin=self.dmin, mmult=self.mmult, data_mean=data_mean)

    def call(self, data, data_mean=None):
        """
        This performs the mapping from input data to output discrete version.

        This method os directly called by the :func:`__call__` method, so the
        class instance (once constructed) is itself callable, as follows:

        >>> remap = Density()
        >>> discrete_data = remap(data, data_mean=85.2)

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        data_mean : None|float
            The pre-calculated data mean, for consistent global use. The order
            of preference is the value provided here, the class data_mean property
            value, then the value calculated from the present sample.

        Returns
        -------
        numpy.ndarray
        """

        return clip_cast(
            self.raw_call(data, data_mean=data_mean),
            dtype=self.output_dtype, min_value=0, max_value=self.max_output_value)

    def calculate_global_parameters_from_reader(self, reader, index=0, pixel_bounds=None):
        pixel_bounds = self._validate_pixel_bounds(reader, index, pixel_bounds)
        self.data_mean = get_data_mean_magnitude(pixel_bounds, reader, index, 25*1024*1024)


class Brighter(Density):
    """
    The density remap using parameters for brighter results.
    """

    _name = 'brighter'

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None, eps=1e-5, data_mean=None):
        Density.__init__(self, override_name=override_name, bit_depth=bit_depth, max_output_value=max_output_value,
                         dmin=60, mmult=40, eps=eps, data_mean=data_mean)


class Darker(Density):
    """
    The density remap using parameters for darker results.
    """

    _name = 'darker'

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None, eps=1e-5, data_mean=None):
        Density.__init__(self, override_name=override_name, bit_depth=bit_depth, max_output_value=max_output_value,
                         dmin=0, mmult=40, eps=eps, data_mean=data_mean)


class High_Contrast(Density):
    """
    The density remap using parameters for high contrast results.
    """

    _name = 'high_contrast'

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None, eps=1e-5, data_mean=None):
        Density.__init__(self, override_name=override_name, bit_depth=bit_depth, max_output_value=max_output_value,
                         dmin=30, mmult=4, eps=eps, data_mean=data_mean)


class Linear(MonochromaticRemap):
    """
    A monochromatic linear remap function.
    """

    __slots__ = ('_override_name', '_bit_depth', '_dimension', '_max_value', '_min_value')
    _name = 'linear'

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None, min_value=None, max_value=None):
        """

        Parameters
        ----------
        override_name : None|str
            Override name for a specific class instance
        bit_depth : int
        min_value : None|float
        max_value : None|float
        """
        MonochromaticRemap.__init__(self, override_name=override_name, bit_depth=bit_depth, max_output_value=max_output_value)

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

    @property
    def are_global_parameters_set(self):
        """
        bool: Are (all) global parameters used for applying this remap function
        set? In this case, this is the `min_value` and `max_value` properties.
        """

        return self._min_value is not None and self._max_value is not None

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

    def raw_call(self, data, min_value=None, max_value=None):
        """
        This performs the mapping from input data to output floating point
        version, this is directly used by the :func:`call` method.

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        min_value : None|float
            A minimum threshold, or pre-calculated data minimum, for consistent
            global use. The order of preference is the value provided here, the
            class `min_value` property value, then calculated from the present
            sample.
        max_value : None|float
            A maximum value threshold, or pre-calculated data maximum, for consistent
            global use. The order of preference is the value provided here, the
            class `max_value` property value, then calculated from the present
            sample.

        Returns
        -------
        numpy.ndarray
        """

        if numpy.iscomplexobj(data):
            amplitude = numpy.abs(data)
        else:
            amplitude = data

        out = numpy.empty(amplitude.shape, dtype='float64')
        max_output_value = self.max_output_value

        finite_mask = numpy.isfinite(amplitude)
        out[~finite_mask] = max_output_value

        if numpy.any(finite_mask):
            temp_data = amplitude[finite_mask]

            min_value, max_value = self._get_extrema(temp_data, min_value, max_value)

            if min_value == max_value:
                out[finite_mask] = 0
            else:
                out[finite_mask] = max_output_value*_linear_map(amplitude[finite_mask], min_value, max_value)
        return out

    def call(self, data, min_value=None, max_value=None):
        """
        This performs the mapping from input data to output discrete version.

        This method os directly called by the :func:`__call__` method, so the
        class instance (once constructed) is itself callable, as follows:

        >>> remap = Linear()
        >>> discrete_data = remap(data, min_value=0, max_value=100)

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        min_value : None|float
            A minimum threshold, or pre-calculated data minimum, for consistent
            global use. The order of preference is the value provided here, the
            class `min_value` property value, then calculated from the present
            sample.
        max_value : None|float
            A maximum value threshold, or pre-calculated data maximum, for consistent
            global use. The order of preference is the value provided here, the
            class `max_value` property value, then calculated from the present
            sample.

        Returns
        -------
        numpy.ndarray
        """

        return clip_cast(
            self.raw_call(data, min_value=min_value, max_value=max_value),
            dtype=self.output_dtype, min_value=0, max_value=self.max_output_value)

    def calculate_global_parameters_from_reader(self, reader, index=0, pixel_bounds=None):
        pixel_bounds = self._validate_pixel_bounds(reader, index, pixel_bounds)
        self.min_value, self.max_value = get_data_extrema(pixel_bounds, reader, index, 25*1024*1024, percentile=None)


class Logarithmic(MonochromaticRemap):
    """
    A logarithmic remap function.
    """

    __slots__ = ('_override_name', '_bit_depth', '_dimension', '_max_value', '_min_value')
    _name = 'log'

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None, min_value=None, max_value=None):
        """

        Parameters
        ----------
        override_name : None|str
            Override name for a specific class instance
        bit_depth : int
        min_value : None|float
        max_value : None|float
        """

        MonochromaticRemap.__init__(self, override_name=override_name, bit_depth=bit_depth, max_output_value=max_output_value)

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

    @property
    def are_global_parameters_set(self):
        """
        bool: Are (all) global parameters used for applying this remap function
        set? In this case, this is the `min_value` and `max_value` properties.
        """

        return self._min_value is not None and self._max_value is not None

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

    def raw_call(self, data, min_value=None, max_value=None):
        """
        This performs the mapping from input data to output floating point
        version, this is directly used by the :func:`call` method.

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        min_value : None|float
            A minimum threshold, or pre-calculated data minimum, for consistent
            global use. The order of preference is the value provided here, the
            class `min_value` property value, then calculated from the present
            sample.
        max_value : None|float
            A maximum value threshold, or pre-calculated data maximum, for consistent
            global use. The order of preference is the value provided here, the
            class `max_value` property value, then calculated from the present
            sample.

        Returns
        -------
        numpy.ndarray
        """

        amplitude = numpy.abs(data)

        out = numpy.empty(amplitude.shape, dtype='float64')
        max_output_value = self.max_output_value

        finite_mask = numpy.isfinite(amplitude)
        zero_mask = (amplitude == 0)
        use_mask = finite_mask & (~zero_mask)

        out[~finite_mask] = max_output_value
        out[zero_mask] = 0

        if numpy.any(use_mask):
            temp_data = amplitude[use_mask]
            min_value, max_value = self._get_extrema(temp_data, min_value, max_value)

            if min_value == max_value:
                out[use_mask] = 0
            else:
                temp_data = (numpy.clip(temp_data, min_value, max_value) - min_value)/(max_value - min_value) + 1
                out[use_mask] = max_output_value*numpy.log2(temp_data)
        return out

    def call(self, data, min_value=None, max_value=None):
        """
        This performs the mapping from input data to output discrete version.

        This method os directly called by the :func:`__call__` method, so the
        class instance (once constructed) is itself callable, as follows:

        >>> remap = Logarithmic()
        >>> discrete_data = remap(data, min_value=1.8, max_value=1.2e6)

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        min_value : None|float
            A minimum threshold, or pre-calculated data minimum, for consistent
            global use. The order of preference is the value provided here, the
            class `min_value` property value, then calculated from the present
            sample.
        max_value : None|float
            A maximum value threshold, or pre-calculated data maximum, for consistent
            global use. The order of preference is the value provided here, the
            class `max_value` property value, then calculated from the present
            sample.

        Returns
        -------
        numpy.ndarray
        """

        return clip_cast(
            self.raw_call(data, min_value=min_value, max_value=max_value),
            dtype=self.output_dtype, min_value=0, max_value=self.max_output_value)

    def calculate_global_parameters_from_reader(self, reader, index=0, pixel_bounds=None):
        pixel_bounds = self._validate_pixel_bounds(reader, index, pixel_bounds)
        self.min_value, self.max_value = get_data_extrema(pixel_bounds, reader, index, 25*1024*1024, percentile=None)


class PEDF(MonochromaticRemap):
    """
    A monochromatic piecewise extended density format remap.
    """

    __slots__ = ('_override_name', '_bit_depth', '_dimension', '_density')
    _name = 'pedf'

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None,
                 dmin=30, mmult=40, eps=1e-5, data_mean=None):
        """

        Parameters
        ----------
        override_name : None|str
            Override name for a specific class instance
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

        MonochromaticRemap.__init__(
            self, override_name=override_name, bit_depth=bit_depth, max_output_value=max_output_value)
        self._density = Density(
            bit_depth=bit_depth, max_output_value=max_output_value,
            dmin=dmin, mmult=mmult, eps=eps, data_mean=data_mean)

    @property
    def are_global_parameters_set(self):
        """
        bool: Are (all) global parameters used for applying this remap function
        set? In this case, this is the `min_value` and `max_value` properties.
        """

        return self._density.are_global_parameters_set

    def raw_call(self, data, data_mean=None):
        """
        This performs the mapping from input data to output floating point
        version, this is directly used by the :func:`call` method.

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        data_mean : None|float
            The pre-calculated data mean, for consistent global use. The order
            of preference is the value provided here, the class data_mean property
            value, then the value calculated from the present sample.

        Returns
        -------
        numpy.ndarray
        """

        half_value = 0.5*self.max_output_value
        out = self._density.raw_call(data, data_mean=data_mean)
        top_mask = (out > half_value)
        out[top_mask] = 0.5*(out[top_mask] + half_value)
        return out

    def call(self, data, data_mean=None):
        """
        This performs the mapping from input data to output discrete version.

        This method os directly called by the :func:`__call__` method, so the
        class instance (once constructed) is itself callable, as follows:

        >>> remap = PEDF()
        >>> discrete_data = remap(data, data_mean=85.2)

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        data_mean : None|float
            The pre-calculated data mean, for consistent global use. The order
            of preference is the value provided here, the class data_mean property
            value, then the value calculated from the present sample.

        Returns
        -------
        numpy.ndarray
        """

        return clip_cast(
            self.raw_call(data, data_mean=data_mean),
            dtype=self.output_dtype, min_value=0, max_value=self.max_output_value)

    def calculate_global_parameters_from_reader(self, reader, index=0, pixel_bounds=None):
        self._density.calculate_global_parameters_from_reader(
            reader, index=index, pixel_bounds=pixel_bounds)


class NRL(MonochromaticRemap):
    """
    A monochromatic remap which is linear for percentile of the data, then
    transitions to logarithmic.
    """

    __slots__ = ('_override_name', '_bit_depth', '_dimension', '_knee', '_percentile', '_stats')
    _name = 'nrl'

    def __init__(self, override_name=None, bit_depth=8, max_output_value=None, knee=None, percentile=99, stats=None):
        """
        Parameters
        ----------
        override_name : None|str
            Override name for a specific class instance
        bit_depth : int
        knee : int
            Where the knee for switching from linear to logarithmic occurs in the
            colormap regime - this should be in keeping with bit-depth.
        percentile : int|float
            In the event that we are calculating the stats, which percentile
            is the cut-off for lin-log switch-over?
        stats : None|tuple
            If provided, this should be of the form `(minimum, maximum, changeover)`.
        """

        self._knee = None
        self._percentile = None
        self._stats = None
        MonochromaticRemap.__init__(self, override_name=override_name, bit_depth=bit_depth, max_output_value=max_output_value)
        self._set_knee(knee)
        self._set_percentile(percentile)
        self._set_stats(stats)

    @property
    def knee(self):
        """
        float: The for switching from linear to logarithmic occurs in the colormap regime
        """

        return self._knee

    def _set_knee(self, knee):
        max_value = self.max_output_value
        if knee is None:
            knee = 0.8*max_value
        knee = float(knee)
        if not (0 < knee < max_value):
            raise ValueError(
                'In keeping with bit-depth, knee must take a value strictly '
                'between 0 and {}'.format(max_value))
        self._knee = knee

    @property
    def percentile(self):
        """
        float: In the event that we are calculating the stats, which percentile
        is the cut-off for lin-log switch-over?
        """

        return self._percentile

    def _set_percentile(self, percentile):
        if percentile is None:
            percentile = 99.0
        else:
            percentile = float(percentile)

        if not (0 < percentile < 100):
            raise ValueError('percentile must fall strictly between 0 and 100')
        self._percentile = percentile

    @property
    def stats(self):
        """
        None|tuple: If populated, this is a tuple of the form `(minimum, maximum, changeover)`.
        """

        return self._stats

    def _set_stats(self, value):
        if value is None:
            self._stats = None
        else:
            self._stats = self._validate_stats(None, value)

    def _validate_stats(self, amplitude, stats):
        if stats is None:
            stats = self.stats
        if stats is None and amplitude is not None:
            stats = _nrl_stats(amplitude, self.percentile)
        if stats is not None:
            min_value = float(stats[0])
            max_value = float(stats[1])
            changeover_value = float(stats[2])
            if not (min_value <= changeover_value <= max_value):
                raise ValueError('Got inconsistent stats value `{}`'.format(stats))
            stats = (min_value, max_value, changeover_value)
        return stats

    @property
    def are_global_parameters_set(self):
        """
        bool: Are (all) global parameters used for applying this remap function
        set? In this case, this is the `stats` property.
        """

        return self._stats is not None

    def raw_call(self, data, stats=None):
        """
        This performs the mapping from input data to output floating point
        version, this is directly used by the :func:`call` method.

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        stats : None|tuple
            The stats `(minimum, maximum, chnageover)`, for consistent
            global use. The order of preference is the value provided here, the
            class `stats` property value, then calculated from the present
            sample.

        Returns
        -------
        numpy.ndarray
        """

        output_dtype = self.output_dtype
        max_index = self.max_output_value

        amplitude = numpy.abs(data)
        amplitude_min, amplitude_max, changeover = self._validate_stats(amplitude, stats)
        out = numpy.empty(amplitude.shape, dtype='float64')
        if amplitude_min == amplitude_max:
            out[:] = 0
            return out

        linear_region = (amplitude <= changeover)
        if changeover > amplitude_min:

            out[linear_region] = numpy.clip(
                self.knee*_linear_map(amplitude[linear_region], amplitude_min, changeover),
                0,
                max_index)
        else:
            logger.warning(
                'The remap array is at least significantly constant, the nrl remap may return '
                'strange results.')
            out[linear_region] = 0

        if changeover == amplitude_max:
            out[~linear_region] = self.knee
        else:
            # calculate the log values
            extreme_data = numpy.clip(amplitude[~linear_region], changeover, amplitude_max)
            log_values = (extreme_data - changeover)/(amplitude_max - changeover) + 1
            # this is now linearly scaled from 1 to 2, apply log_2 and then scale appropriately
            out[~linear_region] = numpy.log2(log_values)*(max_index - self.knee) + self.knee
        return out

    def call(self, data, stats=None):
        """
        This performs the mapping from input data to output discrete version.

        This method os directly called by the :func:`__call__` method, so the
        class instance (once constructed) is itself callable as follows:

        >>> remap = NRL()
        >>> discrete_data = remap(data, stats=(2.3, 1025.0, 997.2))

        Parameters
        ----------
        data : numpy.ndarray
            The (presumably) complex data to remap.
        stats : None|tuple
            The stats `(minimum, maximum, chnageover)`, for consistent
            global use. The order of preference is the value provided here, the
            class `stats` property value, then calculated from the present
            sample.

        Returns
        -------
        numpy.ndarray
        """

        return clip_cast(
            self.raw_call(data, stats=stats),
            dtype=self.output_dtype, min_value=0, max_value=self.max_output_value)

    def calculate_global_parameters_from_reader(self, reader, index=0, pixel_bounds=None):
        pixel_bounds = self._validate_pixel_bounds(reader, index, pixel_bounds)
        self._set_stats(get_data_extrema(pixel_bounds, reader, index, 25*1024*1024, percentile=self.percentile))


class LUT8bit(RemapFunction):
    """
    A remap which uses a monochromatic remap function and an 8-bit lookup table
    to produce a (color) image output
    """

    __slots__ = ('_override_name', '_bit_depth', '_dimension', '_mono_remap', '_lookup_table')
    _name = '_lut_8bit'
    _allowed_dimension = None

    def __init__(self, mono_remap, lookup_table, override_name=None, use_alpha=False):
        """

        Parameters
        ----------
        mono_remap : MonochromaticRemap
            The remap to apply before using the lookup table. Note that the `max_output_value`
            and lookup_table first dimension size are required to be the same.
        lookup_table : str|numpy.ndarray
            A string name for a registered matplotlib colormap or the 256 element
            rgb or rgba array.
        override_name : None|str
            Override name for a specific class instance. If this is not provided and
            the `lookup_table` will be constructed from a matplotlib colormap name,
            then that name will be used.
        use_alpha : bool
            Only used if `mono_remap` is the name of a matplotlib colormap, this
            specifies whether or not to use the alpha channel.
        """

        self._mono_remap = None
        self._lookup_table = None
        if override_name is None and isinstance(lookup_table, str):
            override_name = lookup_table
        RemapFunction.__init__(self, override_name=override_name, bit_depth=8, dimension=0)
        # NB: dimension will be determined by the lookup table
        self._set_mono_remap(mono_remap)
        self._set_lookup_table(lookup_table, use_alpha)

    def _set_dimension(self, value):
        """
        The property is intended to be read-only.

        Parameters
        ----------
        value : int
        """

        self._dimension = value

    @property
    def mono_remap(self):
        # type: () -> MonochromaticRemap
        """
        MonochromaticRemap: The monochromatic remap being used.
        """
        return self._mono_remap

    def _set_mono_remap(self, value):
        if not isinstance(value, MonochromaticRemap):
            raise ValueError('mono_remap requires a monochromatic remap instance')
        self._mono_remap = value

    @property
    def lookup_table(self):
        """
        numpy.ndarray: The 8-bit lookup table.
        """

        return self._lookup_table

    def _set_lookup_table(self, value, use_alpha):
        max_out_size = self.mono_remap.max_output_value
        if isinstance(value, str):
            if cm is None:
                raise ImportError(
                    'The lookup_table has been specified by providing a matplotlib '
                    'colormap name, but matplotlib can not be imported.')
            cmap = cm.get_cmap(value, max_out_size+1)
            color_array = cmap(numpy.arange(max_out_size+1))
            value = clip_cast(max_out_size*color_array, dtype='uint8')
            if value.shape[1] > 3 and not use_alpha:
                value = value[:, :3]
        if not (isinstance(value, numpy.ndarray) and value.ndim == 2 and value.dtype.name == 'uint8'):
            raise ValueError(
                'lookup_table requires a two-dimensional numpy array of dtype = uint8')
        if value.shape[0] != max_out_size+1:
            raise ValueError(
                'lookup_table size (first dimension) must agree with mono_remap.max_output_value')

        self._lookup_table = value
        self._dimension = value.shape[1]

    @property
    def are_global_parameters_set(self):
        """
        bool: Are (all) global parameters used for applying this remap function set?
        """

        return self.mono_remap.are_global_parameters_set

    def raw_call(self, data, **kwargs):
        """
        Contrary to monochromatic remaps, this is identical to :func:`call`.

        Parameters
        ----------
        data : numpy.ndarray
        kwargs
            The keyword arguments passed through to mono_remap.

        Returns
        -------
        numpy.ndarray
        """

        return self._lookup_table[self._mono_remap(data, **kwargs)]

    def call(self, data, **kwargs):
        return self.raw_call(data, **kwargs)

    def calculate_global_parameters_from_reader(self, reader, index=0, pixel_bounds=None):
        self.mono_remap.calculate_global_parameters_from_reader(
            reader, index=index, pixel_bounds=pixel_bounds)


#################
# DEPRECATED!
#################
# the original flat methods, maintained for a while
# for backwards compatibility

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

    warnings.warn(
        'the density() method is deprecated,\n\t'
        'use the Density class, which is also callable', DeprecationWarning)
    remapper = get_registered_remap('density')
    return remapper(data, data_mean=data_mean)


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

    warnings.warn(
        'the brighter() method is deprecated,\n\t'
        'use the Brighter class, which is also callable', DeprecationWarning)
    remapper = get_registered_remap('brighter')
    return remapper(data, data_mean=data_mean)


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

    warnings.warn(
        'the darker() method is deprecated,\n\t'
        'use the Darker class, which is also callable', DeprecationWarning)
    remapper = get_registered_remap('darker')
    return remapper(data, data_mean=data_mean)


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

    warnings.warn(
        'the high_contrast() method is deprecated,\n\t'
        'use the HighContrast class, which is also callable', DeprecationWarning)
    remapper = get_registered_remap('high_contrast')
    return remapper(data, data_mean=data_mean)


def linear(data, min_value=None, max_value=None):
    """
    Linear remap - just the magnitude.

    Parameters
    ----------
    data : numpy.ndarray
    min_value : None|float
        The minimum allowed value for the dynamic range.
    max_value : None|float
        The maximum allowed value for the dynamic range.

    Returns
    -------
    numpy.ndarray
    """

    warnings.warn(
        'the linear() method is deprecated,\n\t'
        'use the Linear class, which is also callable', DeprecationWarning)
    remapper = get_registered_remap('linear')
    return remapper(data, min_value=min_value, max_value=max_value)


def log(data, min_value=None, max_value=None):
    """
    Logarithmic remap.

    Parameters
    ----------
    data : numpy.ndarray
    min_value : None|float
        The minimum allowed value for the dynamic range.
    max_value : None|float
        The maximum allowed value for the dynamic range.

    Returns
    -------
    numpy.ndarray
    """

    warnings.warn(
        'the log() method is deprecated,\n\t'
        'use the Logarithmic class, which is also callable', DeprecationWarning)
    remapper = get_registered_remap('log')
    return remapper(data, min_value=min_value, max_value=max_value)


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

    warnings.warn(
        'the pedf() method is deprecated,\n\t'
        'use the PEDF class, which is also callable', DeprecationWarning)
    remapper = get_registered_remap('pedf')
    return remapper(data, data_mean=data_mean)


def nrl(data, stats=None):
    """
    A lin-log style remap.

    Parameters
    ----------
    data : numpy.ndarray
        The data array to remap
    stats : None|tuple
        This is calculated if not provided. Expected to be of the form
        `(minimum, maximum, 99th percentile)`.

    Returns
    -------
    numpy.ndarray
    """

    warnings.warn(
        'the nrl() method is deprecated,\n\t'
        'use the NRL class, which is also callable', DeprecationWarning)
    remapper = get_registered_remap('nrl')
    return remapper(data, stats=stats)
