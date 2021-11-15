"""
Methods for transforming SICD data to a common state.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from tempfile import mkstemp
import os
from typing import Tuple

import numpy
from numpy.polynomial import polynomial
from numpy.random import randn
from scipy.signal import resample

from sarpy.io.general.base import SarpyIOError
from sarpy.processing.ortho_rectify import FullResolutionFetcher
from sarpy.processing.fft_base import fft, ifft, fftshift, ifftshift, \
    fft_sicd, ifft_sicd

from sarpy.io.complex.base import FlatSICDReader, SICDTypeReader
from sarpy.io.complex.converter import open_complex
from sarpy.io.complex.sicd import SICDWriter
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.Grid import WgtTypeType

logger = logging.getLogger(__name__)


##################
# helper functions

def apply_skew_poly(input_data, delta_kcoa_poly, row_array, col_array, fft_sgn,
                    dimension, forward=False):
    """
    Performs the skew operation on the complex array, according to the provided
    delta kcoa polynomial.

    Parameters
    ----------
    input_data : numpy.ndarray
        The input data.
    delta_kcoa_poly : numpy.ndarray
        The delta kcoa polynomial to use.
    row_array : numpy.ndarray
        The row array, should agree with input_data first dimension definition.
    col_array : numpy.ndarray
        The column array, should agree with input_data second dimension definition.
    fft_sgn : int
        The fft sign to use.
    dimension : int
        The dimension to apply along.
    forward : bool
        If True, this shifts forward (i.e. skews), otherwise applies in inverse
        (i.e. deskew) direction.

    Returns
    -------
    numpy.ndarray
    """

    if numpy.all(delta_kcoa_poly == 0):
        return input_data

    delta_kcoa_poly_int = polynomial.polyint(delta_kcoa_poly, axis=dimension)
    if forward:
        fft_sgn *= -1
    return input_data*numpy.exp(1j*fft_sgn*2*numpy.pi*polynomial.polygrid2d(
        row_array, col_array, delta_kcoa_poly_int))


def determine_weight_array(input_data_shape, weight_array, oversample_rate, dimension):
    """
    Determine the appropriate resampled weight array and bounds.

    Parameters
    ----------
    input_data_shape : tuple
        The shape of the input data, which should be a two element tuple.
    weight_array : numpy.ndarray
    oversample_rate : int|float
    dimension : int

    Returns
    -------
    (numpy.ndarray, int, int)
        The weight array, start index, and end index
    """

    if not (isinstance(weight_array, numpy.ndarray) and weight_array.ndim == 1):
        raise ValueError('The weight array must be one-dimensional')

    weight_size = round(input_data_shape[dimension]/oversample_rate)
    if weight_array.ndim != 1:
        raise ValueError('weight_array must be one dimensional.')

    weight_ind_start = int(numpy.floor(0.5*(input_data_shape[dimension] - weight_size)))
    weight_ind_end = weight_ind_start + weight_size

    if weight_array.size == weight_size:
        return weight_array, weight_ind_start, weight_ind_end
    else:
        return resample(weight_array, weight_size), weight_ind_start, weight_ind_end


def apply_weight_array(input_data, weight_array, oversample_rate, dimension, inverse=False):
    """
    Apply the weight array along the given dimension.

    Parameters
    ----------
    input_data : numpy.ndarray
        The complex data array to weight.
    weight_array : numpy.ndarray
        The weight array.
    oversample_rate : int|float
        The oversample rate.
    dimension : int
        Along which dimension to apply the weighting? Must be one of `{0, 1}`.
    inverse : bool
        If `True`, this divides the weight (i.e. de-weight), otherwise it multiplies.

    Returns
    -------
    numpy.ndarray
    """

    if not (isinstance(input_data, numpy.ndarray) and input_data.ndim == 2):
        raise ValueError('The data array must be two-dimensional')

    if weight_array is None:
        # nothing to be done
        return input_data

    weight_array, weight_ind_start, weight_ind_end = determine_weight_array(
        input_data.shape, weight_array, oversample_rate, dimension)

    if inverse and numpy.any(weight_array == 0):
        raise ValueError('inverse=True and the weight array contains some zero entries.')

    output_data = fftshift(fft(input_data, axis=dimension), axes=dimension)
    if dimension == 0:
        if inverse:
            output_data[weight_ind_start:weight_ind_end, :] /= weight_array[:, numpy.newaxis]
        else:
            output_data[weight_ind_start:weight_ind_end, :] *= weight_array[:, numpy.newaxis]
    else:
        if inverse:
            output_data[:, weight_ind_start:weight_ind_end] /= weight_array
        else:
            output_data[:, weight_ind_start:weight_ind_end] *= weight_array
    return ifft(ifftshift(output_data, axes=dimension), axis=dimension)


def _add_poly(poly1, poly2):
    """
    Add two-dimensional polynomials together.

    Parameters
    ----------
    poly1 : numpy.ndarray
    poly2 : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    if not isinstance(poly1, numpy.ndarray) and poly1.ndim == 2:
        raise TypeError('poly1 must be a two-dimensional numpy array.')
    if not isinstance(poly2, numpy.ndarray) and poly2.ndim == 2:
        raise TypeError('poly2 must be a two-dimensional numpy array.')
    out = numpy.zeros((max(poly1.shape[0], poly2.shape[0]), max(poly1.shape[1], poly2.shape[1])), dtype='float64')
    out[:poly1.shape[0], :poly1.shape[1]] += poly1
    out[:poly2.shape[0], :poly2.shape[1]] += poly2
    return out


def _get_deskew_params(the_sicd, dimension):
    """
    Gets the basic deskew parameters.

    Parameters
    ----------
    the_sicd : SICDType
    dimension : int

    Returns
    -------
    (numpy.ndarray, int)
        The delta_kcoa_poly and fft sign along the given dimension
    """

    # define the derived variables
    delta_kcoa_poly = numpy.array([[0, ], ], dtype=numpy.float64)
    fft_sign = -1
    if dimension == 0:
        try:
            delta_kcoa_poly = the_sicd.Grid.Row.DeltaKCOAPoly.get_array(dtype='float64')
        except (ValueError, AttributeError):
            pass
        try:
            fft_sign = the_sicd.Grid.Row.Sgn
        except (ValueError, AttributeError):
            pass

    else:
        try:
            delta_kcoa_poly = the_sicd.Grid.Col.DeltaKCOAPoly.get_array(dtype='float64')
        except (ValueError, AttributeError):
            pass
        try:
            fft_sign = the_sicd.Grid.Col.Sgn
        except (ValueError, AttributeError):
            pass
    return delta_kcoa_poly, fft_sign


##########
# sicd state checking functions

def is_not_skewed(sicd, dimension):
    """
    Check if the sicd structure is not skewed along the provided dimension.

    Parameters
    ----------
    sicd : SICDType
    dimension : int

    Returns
    -------
    bool
    """

    if dimension == 0:
        if sicd.Grid is None or sicd.Grid.Row is None or sicd.Grid.Row.DeltaKCOAPoly is None:
            return True
        return numpy.all(sicd.Grid.Row.DeltaKCOAPoly.get_array(dtype='float64') == 0)
    else:
        if sicd.Grid is None or sicd.Grid.Col is None or sicd.Grid.Col.DeltaKCOAPoly is None:
            return True
        return numpy.all(sicd.Grid.Col.DeltaKCOAPoly.get_array(dtype='float64') == 0)


def is_uniform_weight(sicd, dimension):
    """
    Check if the sicd structure is has uniform weight along the provided dimension.

    Parameters
    ----------
    sicd : SICDType
    dimension : int

    Returns
    -------
    bool
    """

    if dimension == 0:
        if sicd.Grid is None or sicd.Grid.Row is None:
            return False
        dir_param = sicd.Grid.Row
    else:
        if sicd.Grid is None or sicd.Grid.Col is None:
            return False
        dir_param = sicd.Grid.Col

    if dir_param.WgtType is not None and dir_param.WgtType.WindowName == 'UNIFORM':
        return True
    if dir_param.WgtFunct is not None:
        return numpy.all(dir_param.WgtFunct == dir_param.WgtFunct[0])
    return True


def is_normalized(sicd, dimension=1):
    """
    Check if the sicd structure is normalized along the provided dimension.

    Parameters
    ----------
    sicd : SICDType
        The SICD structure.
    dimension : int
        The dimension to test.

    Returns
    -------
    bool
        normalization state in the given dimension
    """

    def _is_fft_sgn_negative():
        if dimension == 0:
            if sicd.Grid is None or sicd.Grid.Row is None or sicd.Grid.Row.Sgn is None:
                return True
            return sicd.Grid.Row.Sgn == -1
        else:
            if sicd.Grid is None or sicd.Grid.Col is None or sicd.Grid.Col.Sgn is None:
                return True
            return sicd.Grid.Col.Sgn == -1

    dimension = int(dimension)
    if dimension not in [0, 1]:
        raise ValueError('dimension must be either 0 or 1, got {}'.format(dimension))

    return is_not_skewed(sicd, dimension) and is_uniform_weight(sicd, dimension) and \
        _is_fft_sgn_negative()


###########
# calculator class, intended mainly for use in aperture tool

class DeskewCalculator(FullResolutionFetcher):
    """
    This is a calculator for deskewing/deweighting which requires full resolution
    in both dimensions.
    """

    __slots__ = (
        '_apply_deskew', '_apply_deweighting', '_apply_off_axis', '_delta_kcoa_poly_axis', '_delta_kcoa_poly_off_axis',
        '_row_fft_sgn', '_col_fft_sgn',
        '_row_shift', '_row_mult', '_col_shift', '_col_mult',
        '_row_weight', '_row_pad', '_col_weight', '_col_pad',
        '_is_normalized', '_is_not_skewed_row', '_is_not_skewed_col',
        '_is_uniform_weight_row', '_is_uniform_weight_col', )

    def __init__(self, reader, dimension=1, index=0, apply_deskew=True,
                 apply_deweighting=False, apply_off_axis=True):
        """

        Parameters
        ----------
        reader : SICDTypeReader
        dimension : int
            The dimension in `{0, 1}` along which to deskew. `0` is row/range/fast-time,
            and `1` is column/azimuth/slow-time.
        index : int
            The reader index to utilize
        apply_deskew : bool
            Deskew along the given axis?
        apply_deweighting : bool
            Deweight?
        apply_off_axis : bool
            Deskew off axis, to the extent possible?
        """

        self._apply_deskew = apply_deskew
        self._apply_deweighting = apply_deweighting
        self._apply_off_axis = apply_off_axis
        self._delta_kcoa_poly_axis = None
        self._delta_kcoa_poly_off_axis = None
        self._row_fft_sgn = None
        self._row_shift = None
        self._row_mult = None
        self._col_fft_sgn = None
        self._col_shift = None
        self._col_mult = None
        self._row_weight = None
        self._row_pad = None
        self._col_weight = None
        self._col_pad = None
        self._is_normalized = None
        self._is_not_skewed_row = None
        self._is_not_skewed_col = None
        self._is_uniform_weight_row = None
        self._is_uniform_weight_col = None
        super(DeskewCalculator, self).__init__(
            reader, dimension=dimension, index=index, block_size=None)

    @property
    def dimension(self):
        # type: () -> int
        """
        int: The dimension along which to perform the color subaperture split.
        """

        return self._dimension

    @dimension.setter
    def dimension(self, value):
        value = int(value)
        if value not in [0, 1]:
            raise ValueError('dimension must be 0 or 1, got {}'.format(value))
        self._dimension = value
        if self._sicd is not None:
            self._set_sicd(self._sicd)

    def _set_index(self, value):
        value = int(value)
        if value < 0:
            raise ValueError('The index must be a non-negative integer, got {}'.format(value))

        # noinspection PyUnresolvedReferences
        sicds = self.reader.get_sicds_as_tuple()
        if value >= len(sicds):
            raise ValueError('The index must be less than the sicd count.')
        self._index = value
        self._set_sicd(sicds[value])
        self._data_size = self.reader.get_data_size_as_tuple()[value]

    def _set_sicd(self, the_sicd):
        # type : (SICDType) -> None
        if the_sicd is None:
            self._sicd = None
            return

        if not isinstance(the_sicd, SICDType):
            raise TypeError('the_sicd must be an insatnce of SICDType, got type {}'.format(type(the_sicd)))

        self._sicd = the_sicd
        row_delta_kcoa_poly, self._row_fft_sgn = _get_deskew_params(the_sicd, 0)
        col_delta_kcoa_poly, self._col_fft_sgn = _get_deskew_params(the_sicd, 1)
        if self.dimension == 0:
            self._delta_kcoa_poly_axis = row_delta_kcoa_poly
            delta_kcoa_poly_int = polynomial.polyint(row_delta_kcoa_poly, axis=0)
            self._delta_kcoa_poly_off_axis = _add_poly(-polynomial.polyder(delta_kcoa_poly_int, axis=1),
                                                       col_delta_kcoa_poly)
        else:
            self._delta_kcoa_poly_axis = col_delta_kcoa_poly
            delta_kcoa_poly_int = polynomial.polyint(col_delta_kcoa_poly, axis=1)
            self._delta_kcoa_poly_off_axis = _add_poly(-polynomial.polyder(delta_kcoa_poly_int, axis=0),
                                                       row_delta_kcoa_poly)

        self._row_shift = the_sicd.ImageData.SCPPixel.Row - the_sicd.ImageData.FirstRow
        self._row_mult = the_sicd.Grid.Row.SS
        self._col_shift = the_sicd.ImageData.SCPPixel.Col - the_sicd.ImageData.FirstCol
        self._col_mult = the_sicd.Grid.Col.SS
        self._row_pad = max(1., 1./(the_sicd.Grid.Row.SS*the_sicd.Grid.Row.ImpRespBW))
        self._row_weight = the_sicd.Grid.Row.WgtFunct.copy() if the_sicd.Grid.Row.WgtFunct is not None else None
        self._col_pad = max(1., 1./(the_sicd.Grid.Col.SS*the_sicd.Grid.Col.ImpRespBW))
        self._col_weight = the_sicd.Grid.Col.WgtFunct.copy() if the_sicd.Grid.Col.WgtFunct is not None else None
        self._is_normalized = is_normalized(the_sicd, self.dimension)
        self._is_not_skewed_row = is_not_skewed(the_sicd, 0)
        self._is_not_skewed_col = is_not_skewed(the_sicd, 1)
        self._is_uniform_weight_row = is_uniform_weight(the_sicd, 0)
        self._is_uniform_weight_col = is_uniform_weight(the_sicd, 1)

    @property
    def apply_deskew(self):
        """
        bool: Apply deskew to calculated value. This is for API completeness.
        """

        return self._apply_deskew

    @apply_deskew.setter
    def apply_deskew(self, value):
        self._apply_deskew = (value is True)

    @property
    def apply_deweighting(self):
        """
        bool: Apply deweighting to calculated values.
        """

        return self._apply_deweighting

    @apply_deweighting.setter
    def apply_deweighting(self, value):
        self._apply_deweighting = (value is True)

    def _get_index_arrays(self, row_range, row_step, col_range, col_step):
        """
        Get index array data for polynomial evaluation.

        Parameters
        ----------
        row_range : tuple
        row_step : int
        col_range : tuple
        col_step : int

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
        """

        row_array = self._row_mult*(numpy.arange(row_range[0], row_range[1], row_step) - self._row_shift)
        col_array = self._col_mult*(numpy.arange(col_range[0], col_range[1], col_step) - self._col_shift)
        return row_array, col_array

    def __getitem__(self, item):
        """
        Fetches the processed data based on the input slice.

        Parameters
        ----------
        item

        Returns
        -------
        numpy.ndarray
        """

        def on_axis_deskew(t_full_data, fft_sgn):
            return apply_skew_poly(
                t_full_data, self._delta_kcoa_poly_axis, row_array, col_array,
                fft_sgn, self.dimension, forward=False)

        def other_axis_deskew(t_full_data, fft_sgn):
            # We cannot generally deskew in both directions at once, but we
            # can recenter the nonskewed dimension with a uniform shift
            if numpy.any(self._delta_kcoa_poly_off_axis != 0):
                # get deltakcoa at midpoint, and treat as a constant polynomial
                row_mid = row_array[int(round(0.5 * row_array.size)) - 1]
                col_mid = col_array[int(round(0.5 * col_array.size)) - 1]
                delta_kcoa_new_const = numpy.zeros((1, 1), dtype='float64')
                delta_kcoa_new_const[0, 0] = polynomial.polyval2d(
                    row_mid, col_mid, self._delta_kcoa_poly_off_axis)
                # apply this uniform shift
                t_full_data = apply_skew_poly(
                    t_full_data, delta_kcoa_new_const, row_array, col_array,
                    fft_sgn, 1-self.dimension, forward=False)
            return t_full_data

        if self._is_normalized or not self.apply_deskew:
            # just fetch the data and return
            if not isinstance(item, tuple) or len(item) != 2:
                raise KeyError(
                    'Slicing in the deskew calculator must be two dimensional. '
                    'Got slice item {}'.format(item))
            return self.reader.__getitem__((item[0], item[1], self.index))

        # parse the slicing to ensure consistent structure
        row_range, col_range, _ = self._parse_slicing(item)
        # get full resolution data in both directions
        row_step = 1 if row_range[2] > 0 else -1
        col_step = 1 if col_range[2] > 0 else -1
        full_data = self.reader[
               row_range[0]:row_range[1]:row_step,
               col_range[0]:col_range[1]:col_step,
               self.index]
        # de-weight in each applicable direction
        if self._apply_deweighting and self._is_not_skewed_row and not self._is_uniform_weight_row:
            full_data = apply_weight_array(full_data, self._row_weight, self._row_pad, 0, inverse=True)
        if self._apply_deweighting and self._is_not_skewed_col and not self._is_uniform_weight_col:
            full_data = apply_weight_array(full_data, self._col_weight, self._col_pad, 1, inverse=True)
        # deskew in our given dimension
        row_array, col_array = self._get_index_arrays(row_range, row_step, col_range, col_step)
        if self.dimension == 0:
            # deskew on axis, if necessary
            if not self._is_not_skewed_row:
                full_data = on_axis_deskew(full_data, self._row_fft_sgn)
                if self._apply_deweighting:
                    full_data = apply_weight_array(full_data, self._row_weight, self._row_pad, 0, inverse=True)
            if self._apply_off_axis:
                # deskew off axis, to the extent possible
                full_data = other_axis_deskew(full_data, self._col_fft_sgn)
        elif self.dimension == 1:
            # deskew on axis, if necessary
            if not self._is_not_skewed_col:
                full_data = on_axis_deskew(full_data, self._col_fft_sgn)
                if self._apply_deweighting:
                    full_data = apply_weight_array(full_data, self._col_weight, self._col_pad, 1, inverse=True)
            if self._apply_off_axis:
                # deskew off axis, to the extent possible
                full_data = other_axis_deskew(full_data, self._row_fft_sgn)
        return full_data[::abs(row_range[2]), ::abs(col_range[2])]


def aperture_dimension_limits(sicd, dimension, dimension_limits=None, aperture_limits=None):
    """
    This is a helper method to determine the "correct" effective limits for aperture
    processing along the given dimension, considering the ImpRespBW values.

    Parameters
    ----------
    sicd : SICDType
    dimension : int
        One of `{0, 1}`, for the processing dimension
    dimension_limits : None|Tuple[int|float, int|float]
        The base limits along the given dimension, will default to `(0, rows)` for `dimension=0`
        or `(0, columns)` for `dimension=1`, if not provided.
    aperture_limits : None|Tuple[int|float, int|float]
        The desired aperture limits, relative to `dimension_limits`.

    Returns
    -------
    dimension_limits : Tuple[int, int]
        The explicitly populated effective limits along the given dimension.
    aperture_limits : Tuple[int, int]
        The valid aperture limits, relative to `dimension_limits`, after considerations
        of the impulse response bandwidth along the dimension.
    """

    def validate_tuple(tup, limit):
        # type: (None|tuple, int) -> Tuple[int, int]
        if tup is None:
            return 0, limit

        out = int(numpy.floor(tup[0])), int(numpy.ceil(tup[1]))
        if not (0 <= out[0] < out[1] <= limit):
            raise ValueError('Got invalid tuple `{}` for limit `{}`'.format(tup, limit))
        return out

    def extrema_tuple(tup1, tup2):
        # type: (tuple, tuple) -> Tuple[int, int]
        return int(numpy.floor(max(tup1[0], tup2[0]))), int(numpy.ceil(min(tup1[1], tup2[1])))

    dimension = int(dimension)
    if dimension not in [0, 1]:
        raise ValueError('Got invalid dimension value')

    if dimension == 0:
        the_limit = sicd.ImageData.NumRows
        the_oversample = sicd.Grid.Row.get_oversample_rate()
    else:
        the_limit = sicd.ImageData.NumCols
        the_oversample = sicd.Grid.Col.get_oversample_rate()

    dimension_limits = validate_tuple(dimension_limits, the_limit)
    dimension_count = dimension_limits[1] - dimension_limits[0]
    aperture_limits = validate_tuple(aperture_limits, dimension_count)

    ap_size = dimension_count/the_oversample
    ap_limits = (0.5 * (dimension_count - ap_size), 0.5 * (dimension_count + ap_size))
    ap_limits = extrema_tuple(aperture_limits, ap_limits)
    return dimension_limits, ap_limits


def aperture_dimension_params(
        sicd, dimension, dimension_limits=None, aperture_limits=None, new_weight_function=None):
    """
    Gets the aperture processing parameters along the given dimension.

    Parameters
    ----------
    sicd : SICDType
    dimension : int
        One of `{0, 1}`, for the processing dimension
    dimension_limits : None|tuple[int|float, int|float]
        The base limits along the given dimension, will default to `(0, rows)`
        for `dimension=0` or `(0, columns)` for `dimension=1`, if not provided.
    aperture_limits : tuple[int, int]
        The valid aperture limits, relative to `dimension_limits`, after
        considerations of the impulse response bandwidth along the dimension.
    new_weight_function : None|numpy.ndarray
        The new weight function. This will default to the current weight function
        if not provided.

    Returns
    -------
    dimension_limits : tuple[int, int]
        The explicitly populated effective limits along the given dimension.
    cur_aperture_limits : tuple[int, int]
        The current valid aperture limits, relative to `dimension_limits`, after
        considerations of the impulse response bandwidth along the dimension.
    cur_aperture_weighting : numpy.ndarray
    new_aperture_limits : tuple[int, int]
        The new valid aperture limits, relative to `dimension_limits`, after
        considerations of the impulse response bandwidth along the dimension.
    new_aperture_weighting : numpy.ndarray
    """

    dimension = int(dimension)
    if dimension not in [0, 1]:
        raise ValueError('Got invalid dimension value')

    dimension_limits, cur_aperture_limits = aperture_dimension_limits(sicd, dimension, dimension_limits, None)
    _, new_aperture_limits = aperture_dimension_limits(sicd, dimension, dimension_limits, aperture_limits)

    cur_ap_count = cur_aperture_limits[1] - cur_aperture_limits[0]
    new_ap_count = new_aperture_limits[1] - new_aperture_limits[0]
    if dimension == 0:
        cur_weight_function = resample(sicd.Grid.Row.WgtFunct, cur_ap_count)
        if new_weight_function is None:
            new_weight_function = resample(sicd.Grid.Row.WgtFunct, new_ap_count)
        else:
            new_weight_function = resample(new_weight_function, new_ap_count)
    else:
        cur_weight_function = resample(sicd.Grid.Col.WgtFunct, cur_ap_count)
        if new_weight_function is None:
            new_weight_function = resample(sicd.Grid.Col.WgtFunct, new_ap_count)
        else:
            new_weight_function = resample(new_weight_function, new_ap_count)
    return dimension_limits, cur_aperture_limits, cur_weight_function, new_aperture_limits, new_weight_function


def noise_scaling(cur_ap_limits, cur_weighting, new_ap_limits, new_weighting):
    """
    Gets noise scaling due to sub-aperture degradation and re-weighting along one
    dimension.

    Parameters
    ----------
    cur_ap_limits : tuple
    cur_weighting : numpy.ndarray
    new_ap_limits : tuple
    new_weighting : numpy.ndarray

    Returns
    -------
    noise_multiplier : float
    """

    start_lim = new_ap_limits[0]-cur_ap_limits[0]
    end_lim = start_lim + new_ap_limits[1] - new_ap_limits[0]
    weight_change = new_weighting/cur_weighting[start_lim:end_lim]
    second_moment = numpy.sum(weight_change**2)/float(cur_weighting.size)
    return second_moment


def sicd_degrade_reweight(
        reader, output_file=None, index=0,
        row_limits=None, column_limits=None,
        row_aperture=None, row_weighting=None,
        column_aperture=None, column_weighting=None,
        add_noise=None, pixel_threshold=1500*1500,
        check_existence=True, check_older_version=False, repopulate_rniirs=True):
    r"""
    Given input, create a SICD (file or reader) with modified weighting/subaperture
    parameters. Any additional noise will be added **before** performing any sub-aperture
    degradation processing.

    Recall that reducing the size of the impulse response bandwidth via sub-aperture
    degradation in a single direction by by :math:`ratio`, actually decreases the
    magnitude of the noise in pixel power by :math:`ratio`, or subtracts
    :math:`10*\log_{10}(ratio)` from the noise given in dB.

    .. warning::

        To ensure correctness of metadata, if the Noise Polynomial is present,
        then then the NoiseLevelType must be `'ABSOLUTE'`. Otherwise an exception
        will be raised.

    Parameters
    ----------
    reader : str|SICDTypeReader
        A sicd type reader.
    output_file : None|str
        If `None`, an in-memory SICD reader instance will be returned. Otherwise,
        this is the path for the produced output SICD file.
    index : int
        The reader index to be used.
    row_limits : None|(int, int)
        Row limits for the underlying data.
    column_limits : None|(int, int)
        Column limits for the underlying data.
    row_aperture : None|tuple
        `None` (no row subaperture), or integer valued `start_row, end_row` for the
        row subaperture definition. This is with respect to row values AFTER
        considering `row_limits`. Note that this will reduce the noise, so the
        noise polynomial (if it is populated) will be modified.
    row_weighting : None|dict
        `None` (no row weighting change), or the new row weighting parameters
        `{'WindowName': <name>, 'Parameters: {}, 'WgtFunct': array}`.
    column_aperture : None|tuple
        `None` (no column subaperture), or integer valued `start_col, end_col` for the
        column sub-aperture definition. This is with respect to row values AFTER
        considering `column_limits`. Note that this will reduce the noise, so the
        noise polynomial (if it is populated) will be modified.
    column_weighting : None|dict
        `None` (no columng weighting change), or the new column weighting parameters
        `{'WindowName': <name>, 'Parameters: {}, 'WgtFunct': array}`.
    add_noise : None|float
        If provided, Gaussian white noise of pixel power `add_noise` will be
        added, prior to any subbaperture processing. Note that performing subaperture
        processing actually reduces the resulting noise, which will also be considered.
    pixel_threshold : None|int
        Approximate pixel area threshold for performing this directly in memory.
    check_existence : bool
        Should we check if the given file already exists, and raise an exception if so?
    check_older_version : bool
        Try to use a less recent version of SICD (1.1), for possible application compliance issues?
    repopulate_rniirs : bool
        Should we try to repopulate the estimated RNIIRS value?

    Returns
    -------
    None|FlatSICDReader
        No return if `output_file` is provided, otherwise the returns the in-memory
        reader object.
    """

    def validate_filename():
        if output_file is None:
            return

        if check_existence and os.path.exists(output_file):
            raise SarpyIOError('The file {} already exists.'.format(output_file))

    def validate_sicd(the_sicd):
        if the_sicd.Grid is None or the_sicd.Grid.Row is None or the_sicd.Grid.Col is None:
            raise ValueError('Grid.Row and Grid.Col must be populated')
        for direction in ['Row', 'Col']:
            el = getattr(the_sicd.Grid, direction)
            if el.DeltaKCOAPoly is None or el.WgtFunct is None:
                raise ValueError('DeltaKCOAPoly and WgtFunct must be populated for both Row and Col')

    def validate_limits(lims, max_index):
        if lims is None:
            return 0, max_index
        _lims = (int(lims[0]), int(lims[1]))
        if not (0 <= _lims[0] < _lims[1] <= max_index):
            raise ValueError('Got poorly formatted index limit {}'.format(lims))
        return _lims

    def get_iterations(max_index, other_index):
        if in_memory:
            return [(0, max_index), ]
        out = []
        block = int(pixel_threshold / float(other_index))
        _start_ind = 0
        while _start_ind < max_index:
            _end_ind = min(_start_ind + block, max_index)
            out.append((_start_ind, _end_ind))
            _start_ind = _end_ind
        return out

    def get_direction_array_meters(dimension, start_index, end_index):
        if dimension == 0:
            shift = sicd.ImageData.FirstRow - sicd.ImageData.SCPPixel.Row
            multiplier = sicd.Grid.Row.SS
        else:
            shift = sicd.ImageData.FirstCol - sicd.ImageData.SCPPixel.Col
            multiplier = sicd.Grid.Col.SS
        return (numpy.arange(start_index, end_index) + shift)*multiplier

    def do_add_noise():
        if add_noise is None:
            return

        # noinspection PyBroadException
        try:
            variance = float(add_noise)
            if variance <= 0:
                logger.error('add_noise was provided as `{}`, but must be a positive number'.format(add_noise))
                return
        except Exception:
            logger.error('add_noise was provided as `{}`, but must be a positive number'.format(add_noise))
            return

        sigma = numpy.sqrt(0.5*variance)
        if noise_level is not None:
            noise_constant_power = numpy.exp(numpy.log(10)*0.1*noise_level.NoisePoly[0, 0])
            noise_constant_power += variance
            noise_constant_db = 10*numpy.log10(noise_constant_power)
            noise_level.NoisePoly[0, 0] = noise_constant_db

        for (_start_ind, _stop_ind) in row_iterations:
            d_shape = (_stop_ind - _start_ind, data_shape[1])
            added_noise = numpy.empty(d_shape, dtype='complex64')
            added_noise[:].real = randn(*d_shape).astype('float32')
            added_noise[:].imag = randn(*d_shape).astype('float32')
            added_noise *= sigma
            working_data[_start_ind:_stop_ind, :] += added_noise

    def do_dimension(dimension):
        if dimension == 0:
            dir_params = sicd.Grid.Row
            aperture_in = row_aperture
            weighting_in = row_weighting
            dimension_limits = row_limits
        else:
            dir_params = sicd.Grid.Col
            aperture_in = column_aperture
            weighting_in = column_weighting
            dimension_limits = column_limits

        not_skewed = is_not_skewed(sicd, dimension)
        uniform_weight = is_uniform_weight(sicd, dimension)
        delta_kcoa = dir_params.DeltaKCOAPoly.get_array(dtype='float64')

        st_beam_comp = sicd.ImageFormation.STBeamComp if sicd.ImageFormation is not None else None

        if dimension == 1 and (not uniform_weight or weighting_in is not None):
            if st_beam_comp is None:
                logger.warning(
                    'Processing along the column direction requires modification\n\t'
                    'of the original weighting scheme, and the value for\n\t'
                    'ImageFormation.STBeamComp is not populated.\n\t'
                    'It is unclear how imperfect the de-weighting effort along the column may be.')
            elif st_beam_comp == 'NO':
                logger.warning(
                    'Processing along the column direction requires modification\n\t'
                    'of the original weighting scheme, and the value for\n\t'
                    'ImageFormation.STBeamComp is populated as `NO`.\n\t'
                    'It is likely that the de-weighting effort along the column is imperfect.')

        if aperture_in is None and weighting_in is None:
            # nothing to be done in this dimension
            return noise_adjustment_multiplier

        new_weight = None if weighting_in is None else weighting_in['WgtFunct']
        dimension_limits, cur_aperture_limits, cur_weight_function, \
            new_aperture_limits, new_weight_function = aperture_dimension_params(
                old_sicd, dimension, dimension_limits=dimension_limits,
                aperture_limits=aperture_in, new_weight_function=new_weight)
        index_count = dimension_limits[1] - dimension_limits[0]
        cur_center_index = 0.5*(cur_aperture_limits[0] + cur_aperture_limits[1])
        new_center_index = 0.5*(new_aperture_limits[0] + new_aperture_limits[1])
        noise_multiplier = noise_scaling(
            cur_aperture_limits, cur_weight_function, new_aperture_limits, new_weight_function)

        # perform deskew, if necessary
        if not not_skewed:
            if dimension == 0:
                row_array = get_direction_array_meters(0, 0, out_data_shape[0])
                for (_start_ind, _stop_ind) in col_iterations:
                    col_array = get_direction_array_meters(1, _start_ind, _stop_ind)
                    working_data[:, _start_ind:_stop_ind] = apply_skew_poly(
                        working_data[:, _start_ind:_stop_ind], delta_kcoa,
                        row_array, col_array, dir_params.Sgn, 0, forward=False)
            else:
                col_array = get_direction_array_meters(1, 0, out_data_shape[1])
                for (_start_ind, _stop_ind) in row_iterations:
                    row_array = get_direction_array_meters(0, _start_ind, _stop_ind)
                    working_data[_start_ind:_stop_ind, :] = apply_skew_poly(
                        working_data[_start_ind:_stop_ind, :], delta_kcoa,
                        row_array, col_array, dir_params.Sgn, 1, forward=False)

        # perform fourier transform along the given dimension
        if dimension == 0:
            for (_start_ind, _stop_ind) in col_iterations:
                working_data[:, _start_ind:_stop_ind] = fftshift(
                    fft_sicd(working_data[:, _start_ind:_stop_ind], dimension, sicd), axes=dimension)
        else:
            for (_start_ind, _stop_ind) in row_iterations:
                working_data[_start_ind:_stop_ind, :] = fftshift(
                    fft_sicd(working_data[_start_ind:_stop_ind, :], dimension, sicd), axes=dimension)

        # perform deweight, if necessary
        if not uniform_weight:
            if dimension == 0:
                working_data[cur_aperture_limits[0]:cur_aperture_limits[1], :] /= cur_weight_function[:, numpy.newaxis]
            else:
                working_data[:, cur_aperture_limits[0]:cur_aperture_limits[1]] /= cur_weight_function

        # do sub-aperture, if necessary
        if aperture_in is not None:
            if dimension == 0:
                working_data[:new_aperture_limits[0], :] = 0
                working_data[new_aperture_limits[1]:, :] = 0
            else:
                working_data[:, :new_aperture_limits[0]] = 0
                working_data[:, new_aperture_limits[1]:] = 0

            the_ratio = float(new_aperture_limits[1] - new_aperture_limits[0]) / \
                float(cur_aperture_limits[1] - cur_aperture_limits[0])
            # modify the ImpRespBW value (derived ImpRespWid handled at the end)
            dir_params.ImpRespBW *= the_ratio

        # perform reweight, if necessary
        if weighting_in is not None:
            if dimension == 0:
                working_data[new_aperture_limits[0]:new_aperture_limits[1], :] *= new_weight_function[:, numpy.newaxis]
            else:
                working_data[:, new_aperture_limits[0]:new_aperture_limits[1]] *= new_weight_function
            # modify the weight definition
            dir_params.WgtType = WgtTypeType(
                WindowName=weighting_in['WindowName'],
                Parameters=weighting_in.get('Parameters', None))
            dir_params.WgtFunct = weighting_in['WgtFunct'].copy()
        elif not uniform_weight:
            if dimension == 0:
                working_data[new_aperture_limits[0]:new_aperture_limits[1], :] *= new_weight_function[:, numpy.newaxis]
            else:
                working_data[:, new_aperture_limits[0]:new_aperture_limits[1]] *= new_weight_function

        # perform inverse fourier transform along the given dimension
        if dimension == 0:
            for (_start_ind, _stop_ind) in col_iterations:
                working_data[:, _start_ind:_stop_ind] = ifft_sicd(
                    ifftshift(working_data[:, _start_ind:_stop_ind], axes=dimension), dimension, sicd)
        else:
            for (_start_ind, _stop_ind) in row_iterations:
                working_data[_start_ind:_stop_ind, :] = ifft_sicd(
                    ifftshift(working_data[_start_ind:_stop_ind, :], axes=dimension), dimension, sicd)

        # perform the (original) reskew, if necessary
        if not numpy.all(delta_kcoa == 0):
            if dimension == 0:
                row_array = get_direction_array_meters(0, 0, out_data_shape[0])
                for (_start_ind, _stop_ind) in col_iterations:
                    col_array = get_direction_array_meters(1, _start_ind, _stop_ind)
                    working_data[:, _start_ind:_stop_ind] = apply_skew_poly(
                        working_data[:, _start_ind:_stop_ind], delta_kcoa,
                        row_array, col_array, dir_params.Sgn, 0, forward=True)
            else:
                col_array = get_direction_array_meters(1, 0, out_data_shape[1])
                for (_start_ind, _stop_ind) in row_iterations:
                    row_array = get_direction_array_meters(0, _start_ind, _stop_ind)
                    working_data[_start_ind:_stop_ind, :] = apply_skew_poly(
                        working_data[_start_ind:_stop_ind, :], delta_kcoa,
                        row_array, col_array, dir_params.Sgn, 1, forward=True)

        # modify the delta_kcoa_poly - introduce the shift necessary for additional offset
        if new_center_index != cur_center_index:
            additional_shift = dir_params.Sgn*(cur_center_index - new_center_index)/float(index_count*dir_params.SS)
            delta_kcoa[0, 0] += additional_shift
            dir_params.DeltaKCOAPoly = delta_kcoa

        return noise_adjustment_multiplier*noise_multiplier

    if isinstance(reader, str):
        reader = open_complex(reader)

    if not isinstance(reader, SICDTypeReader):
        raise TypeError('reader must be sicd type reader, got {}'.format(reader))

    # noinspection PyUnresolvedReferences
    old_sicd = reader.get_sicds_as_tuple()[index]
    validate_sicd(old_sicd)
    validate_filename()

    data_shape = reader.get_data_size_as_tuple()[index]

    redo_geo = (row_limits is not None or column_limits is not None)

    row_limits = validate_limits(row_limits, data_shape[0])
    column_limits = validate_limits(column_limits, data_shape[1])

    # prepare our working sicd structure
    sicd = old_sicd.copy()  # type: SICDType

    noise_level = None if sicd.Radiometric is None else sicd.Radiometric.NoiseLevel
    # NB: as an alternative, we could drop the noise polynomial?
    if add_noise is not None and \
            noise_level is not None and \
            noise_level.NoiseLevelType != 'ABSOLUTE':
        raise ValueError(
            'add_noise is provided, but the radiometric noise is populated,\n\t'
            'with `NoiseLevelType={}`'.format(noise_level.NoiseLevelType))

    sicd.ImageData.FirstRow += row_limits[0]
    sicd.ImageData.NumRows = row_limits[1] - row_limits[0]
    sicd.ImageData.FirstCol += column_limits[0]
    sicd.ImageData.NumCols = column_limits[1] - column_limits[0]
    if redo_geo:
        sicd.define_geo_image_corners(override=True)

    out_data_shape = (sicd.ImageData.NumRows, sicd.ImageData.NumCols)

    pixel_area = out_data_shape[0]*out_data_shape[1]
    temp_file = None
    in_memory = True if (output_file is None or pixel_threshold is None) else (pixel_area < pixel_threshold)

    row_iterations = get_iterations(out_data_shape[0], out_data_shape[1])
    col_iterations = get_iterations(out_data_shape[1], out_data_shape[0])

    if in_memory:
        working_data = reader[row_limits[0]:row_limits[1], column_limits[0]:column_limits[1], index]
    else:
        _, temp_file = mkstemp(suffix='.sarpy.cache', text=False)
        working_data = numpy.memmap(temp_file, dtype='complex64', mode='r+', offset=0, shape=out_data_shape)
        for (start_ind, stop_ind) in row_iterations:
            working_data[start_ind:stop_ind, :] = reader[
                                                  start_ind+row_limits[0]:stop_ind+row_limits[0],
                                                  column_limits[0]:column_limits[1], index]

    noise_adjustment_multiplier = 1.0

    # NB: I'm adding Gaussian white noise first
    do_add_noise()

    # do, as necessary, along the row
    noise_adjustment_multiplier = do_dimension(0)
    # do, as necessary, along the row
    noise_adjustment_multiplier = do_dimension(1)

    # re-derive the various ImpResp parameters
    sicd.Grid.derive_direction_params(sicd.ImageData, populate=True)

    # adjust the noise polynomial
    if noise_level is not None:
        noise_level.NoisePoly[0, 0] += 10*numpy.log10(noise_adjustment_multiplier)

    if sicd.Radiometric is not None:
        sf_adjust = old_sicd.Grid.get_slant_plane_area()/noise_adjustment_multiplier
        sicd.Radiometric.RCSSFPoly.Coefs = sicd.Radiometric.RCSSFPoly.get_array()*sf_adjust
        sicd.Radiometric.BetaZeroSFPoly.Coefs = sicd.Radiometric.BetaZeroSFPoly.get_array() / \
            noise_adjustment_multiplier
        sicd.Radiometric.SigmaZeroSFPoly.Coefs = sicd.Radiometric.SigmaZeroSFPoly.get_array() / \
            noise_adjustment_multiplier
        sicd.Radiometric.GammaZeroSFPoly.Coefs = sicd.Radiometric.GammaZeroSFPoly.get_array() / \
            noise_adjustment_multiplier

    if sicd.RMA is not None and sicd.RMA.INCA is not None and sicd.RMA.INCA.TimeCAPoly is not None:
        # redefine the INCA doppler centroid poly to be in keeping with any redefinition of our Col.DeltaKCOAPoly?
        sicd.RMA.INCA.DopCentroidPoly = sicd.Grid.Col.DeltaKCOAPoly.get_array(dtype='float64') / \
            sicd.RMA.INCA.TimeCAPoly[1]

    if repopulate_rniirs:
        sicd.populate_rniirs(override=True)

    if output_file is None:
        return FlatSICDReader(sicd, working_data)
    else:
        # write out the new sicd file
        with SICDWriter(
                output_file, sicd,
                check_older_version=check_older_version, check_existence=check_existence) as writer:
            for (start_ind, stop_ind) in row_iterations:
                writer.write_chip(working_data[start_ind:stop_ind, :], start_indices=(start_ind, 0))

        if temp_file is not None and os.path.exists(temp_file):
            working_data = None
            os.remove(temp_file)
