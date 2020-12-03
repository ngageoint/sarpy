# -*- coding: utf-8 -*-
"""
Methods for transforming SICD data to a common state.
"""

import numpy
from numpy.polynomial import polynomial
import scipy.signal

from sarpy.compliance import int_func
from sarpy.processing.ortho_rectify import FullResolutionFetcher
from sarpy.processing.fft_base import fft, ifft, fftshift, ifftshift
from sarpy.io.complex.sicd_elements.SICD import SICDType


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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


def _is_not_skewed(sicd, dimension):
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


def _is_uniform_weight(sicd, dimension):
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


def _is_fft_sgn_negative(sicd, dimension):
    """
    Check if the sicd structure has negative fft sign along the given dimension.

    Parameters
    ----------
    sicd : SICDType
    dimension : int

    Returns
    -------
    bool
    """

    if dimension == 0:
        if sicd.Grid is None or sicd.Grid.Row is None or sicd.Grid.Row.Sgn is None:
            return True
        return sicd.Grid.Row.Sgn == -1
    else:
        if sicd.Grid is None or sicd.Grid.Col is None or sicd.Grid.Col.Sgn is None:
            return True
        return sicd.Grid.Col.Sgn == -1


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

    dimension = int_func(dimension)
    if dimension not in [0, 1]:
        raise ValueError('dimension must be either 0 or 1, got {}'.format(dimension))

    return _is_not_skewed(sicd, dimension) and _is_uniform_weight(sicd, dimension) and \
           _is_fft_sgn_negative(sicd, dimension)


class DeskewCalculator(FullResolutionFetcher):
    """
    This is a calculator for deskewing/deweighting which requires full resolution
    in both dimensions.
    """

    __slots__ = (
        '_apply_deskew', '_apply_deweighting', '_delta_kcoa_poly_axis', '_delta_kcoa_poly_off_axis',
        '_row_fft_sgn', '_col_fft_sgn',
        '_row_shift', '_row_mult', '_col_shift', '_col_mult',
        '_row_weight', '_row_pad', '_col_weight', '_col_pad',
        '_is_normalized', '_is_not_skewed_row', '_is_not_skewed_col',
        '_is_uniform_weight_row', '_is_uniform_weight_col', )

    def __init__(self, reader, dimension=1, index=0, apply_deskew=True, apply_deweighting=False):
        self._apply_deskew = apply_deskew
        self._apply_deweighting = apply_deweighting
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
        value = int_func(value)
        if value not in [0, 1]:
            raise ValueError('dimension must be 0 or 1, got {}'.format(value))
        self._dimension = value
        if self._sicd is not None:
            self._set_sicd(self._sicd)

    def _set_index(self, value):
        value = int_func(value)
        if value < 0:
            raise ValueError('The index must be a non-negative integer, got {}'.format(value))

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
        self._row_pad = max(1, 1/(the_sicd.Grid.Row.SS*the_sicd.Grid.Row.ImpRespBW))
        self._row_weight = the_sicd.Grid.Row.WgtFunct.copy() if the_sicd.Grid.Row.WgtFunct is not None else None
        self._col_pad = max(1, 1/(the_sicd.Grid.Col.SS*the_sicd.Grid.Col.ImpRespBW))
        self._col_weight = the_sicd.Grid.Col.WgtFunct.copy() if the_sicd.Grid.Col.WgtFunct is not None else None
        self._is_normalized = is_normalized(the_sicd, self.dimension)
        self._is_not_skewed_row = _is_not_skewed(the_sicd, 0)
        self._is_not_skewed_col = _is_not_skewed(the_sicd, 1)
        self._is_uniform_weight_row = _is_uniform_weight(the_sicd, 0)
        self._is_uniform_weight_col = _is_uniform_weight(the_sicd, 1)

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
            return _deskew_array(
                t_full_data, self._delta_kcoa_poly_axis, row_array, col_array, fft_sgn, self.dimension)

        def other_axis_deskew(t_full_data, fft_sgn):
            # We cannot generally deskew in both directions at once, but we
            # can recenter the nonskewed dimension with a uniform shift
            if numpy.any(self._delta_kcoa_poly_off_axis != 0):
                # get deltakcoa at midpoint, and treat as a constant polynomial
                row_mid = row_array[int_func(round(0.5 * row_array.size)) - 1]
                col_mid = col_array[int_func(round(0.5 * col_array.size)) - 1]
                delta_kcoa_new_const = numpy.zeros((1, 1), dtype='float64')
                delta_kcoa_new_const[0, 0] = polynomial.polyval2d(
                    row_mid, col_mid, self._delta_kcoa_poly_off_axis)
                # apply this uniform shift
                t_full_data = _deskew_array(
                    t_full_data, delta_kcoa_new_const, row_array, col_array, fft_sgn, 1-self.dimension)
            return t_full_data

        if self._is_normalized or not self.apply_deskew:
            # just fetch the data and return
            if not isinstance(item, tuple) or len(item) != 2:
                raise KeyError('Slicing in the deskew calculator must be two dimensional. Got slice item {}'.format(item))
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
            full_data = _deweight_array(full_data, self._row_weight, self._row_pad, 0)
        if self._apply_deweighting and self._is_not_skewed_col and not self._is_uniform_weight_col:
            full_data = _deweight_array(full_data, self._col_weight, self._col_pad, 1)
        # deskew in our given dimension
        row_array, col_array = self._get_index_arrays(row_range, row_step, col_range, col_step)
        if self.dimension == 0:
            # deskew on axis, if necessary
            if not self._is_not_skewed_row:
                full_data = on_axis_deskew(full_data, self._row_fft_sgn)
                if self._apply_deweighting:
                    full_data = _deweight_array(full_data, self._row_weight, self._row_pad, 0)
            # deskew off axis, as possible
            full_data = other_axis_deskew(full_data, self._col_fft_sgn)
        elif self.dimension == 1:
            # deskew on axis, if necessary
            if not self._is_not_skewed_col:
                full_data = on_axis_deskew(full_data, self._col_fft_sgn)
                if self._apply_deweighting:
                    full_data = _deweight_array(full_data, self._col_weight, self._col_pad, 1)
            # deskew off axis, as possible
            full_data = other_axis_deskew(full_data, self._row_fft_sgn)
        return full_data[::abs(row_range[2]), ::abs(col_range[2])]


def _get_deskew_params(the_sicd, dimension):
    """
    Gets the basic deskew parameters.

    Parameters
    ----------
    the_sicd : SICDType
    dimension : int

    Returns
    -------
    numpy.ndarray, int
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


def _deskew_array(input_data, delta_kcoa_poly, row_array, col_array, fft_sgn, dimension):
    """
    Performs deskew (centering of the spectrum on zero frequency) on a complex array.

    Parameters
    ----------
    input_data : numpy.ndarray
    delta_kcoa_poly : numpy.ndarray
    row_array : numpy.ndarray
    col_array : numpy.ndarray
    fft_sgn : int

    Returns
    -------
    numpy.ndarray
    """

    delta_kcoa_poly_int = polynomial.polyint(delta_kcoa_poly, axis=dimension)
    return input_data*numpy.exp(1j*fft_sgn*2*numpy.pi*polynomial.polygrid2d(
        row_array, col_array, delta_kcoa_poly_int))


def _deweight_array(input_data, weight_array, oversample_rate, dimension):
    """
    Uniformly weight complex SAR data along the given dimension.

    Parameters
    ----------
    input_data : numpy.ndarray
    weight_array : numpy.ndarray
    oversample_rate : int|float
    dimension : int

    Returns
    -------
    numpy.ndarray
    """

    if weight_array is None:
        # nothing to be done
        return input_data

    weight_size = round(input_data.shape[dimension]/oversample_rate)
    if weight_array.ndim != 1:
        raise ValueError('weight_array must be one dimensional.')
    if weight_array.size != weight_size:
        weight_array = scipy.signal.resample(weight_array, weight_size)
    weight_ind_start = int_func(numpy.floor(0.5*(input_data.shape[dimension] - weight_size)))
    weight_ind_end = weight_ind_start + weight_size

    output_data = fftshift(fft(input_data, axis=dimension), axes=dimension)
    if dimension == 0:
        output_data[weight_ind_start:weight_ind_end, :] /= weight_array[:, numpy.newaxis]
    else:
        output_data[:, weight_ind_start:weight_ind_end] /= weight_array
    return ifft(ifftshift(output_data, axes=dimension), axis=dimension)


################################
# The below should be deprecated

def deskewparams(sicd_meta, dim):
    """

    Parameters
    ----------
    sicd_meta: sarpy.io.complex.sicd_elements.SICD.SICDType
        the sicd structure
    dim : int
        the dimension to test

    Returns
    -------
    Tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray,float]
    """

    DeltaKCOAPoly, fft_sgn = _get_deskew_params(sicd_meta, dimension=dim)

    # Vectors describing range and azimuth distances from SCP (in meters) for rows and columns
    rg_coords_m = (numpy.arange(0, sicd_meta.ImageData.NumRows, dtype=numpy.float32) +
                   sicd_meta.ImageData.FirstRow - sicd_meta.ImageData.SCPPixel.Row)*sicd_meta.Grid.Row.SS
    az_coords_m = (numpy.arange(0, sicd_meta.ImageData.NumCols, dtype=numpy.float32) +
                   sicd_meta.ImageData.FirstCol - sicd_meta.ImageData.SCPPixel.Col)*sicd_meta.Grid.Col.SS
    return DeltaKCOAPoly, rg_coords_m, az_coords_m, fft_sgn


def deskewmem(input_data, DeltaKCOAPoly, dim0_coords_m, dim1_coords_m, dim, fft_sgn=-1):
    """
    Performs deskew (centering of the spectrum on zero frequency) on a complex dataset.

    Parameters
    ----------
    input_data : numpy.ndarray
        Complex FFT Data
    DeltaKCOAPoly : numpy.ndarray
        Polynomial that describes center of frequency support of data.
    dim0_coords_m : numpy.ndarray
    dim1_coords_m : numpy.ndarray
    dim : int
    fft_sgn : int|float

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        * `output_data` - Deskewed data
        * `new_DeltaKCOAPoly` - Frequency support shift in the non-deskew dimension caused by the deskew.
    """

    # Integrate DeltaKCOA polynomial (in meters) to form new polynomial DeltaKCOAPoly_int
    DeltaKCOAPoly_int = polynomial.polyint(DeltaKCOAPoly, axis=dim)
    # New DeltaKCOAPoly in other dimension will be negative of the derivative of
    # DeltaKCOAPoly_int in other dimension (assuming it was zero before).
    new_DeltaKCOAPoly = - polynomial.polyder(DeltaKCOAPoly_int, axis=dim-1)
    # Apply phase adjustment from polynomial
    dim1_coords_m_2d, dim0_coords_m_2d = numpy.meshgrid(dim1_coords_m, dim0_coords_m)
    output_data = numpy.multiply(input_data, numpy.exp(1j * fft_sgn * 2 * numpy.pi *
                                                 polynomial.polyval2d(
                                                     dim0_coords_m_2d,
                                                     dim1_coords_m_2d,
                                                     DeltaKCOAPoly_int)))
    return output_data, new_DeltaKCOAPoly


def deweightmem(input_data, weight_fun=None, oversample_rate=1, dim=1):
    """
    Uniformly weights complex SAR data in given dimension.

    .. Note:: This implementation ASSUMES that the data has already been de-skewed and that the frequency support
        is centered.

    Parameters
    ----------
    input_data : numpy.ndarray
        The complex data
    weight_fun : callable|numpy.ndarray
        Can be an array that explicitly provides the (inverse) weighting, or a function of a
        single numeric argument (number of elements) which produces the (inverse) weighting vector.
    oversample_rate : int|float
        Amount of sampling beyond the ImpRespBW in the processing dimension.
    dim : int
        Dimension over which to perform deweighting.

    Returns
    -------
    numpy.ndarray
    """
    # TODO: HIGH - there was a prexisting comment "Test this function"

    # Weighting only valid across ImpRespBW
    weight_size = round(input_data.shape[dim]/oversample_rate)
    if weight_fun is None:  # No weighting passed in.  Do nothing.
        return input_data
    elif callable(weight_fun):
        weighting = weight_fun(weight_size)
    elif numpy.array(weight_fun).ndim == 1:
        weighting = scipy.signal.resample(weight_fun, weight_size)
    # TODO: HIGH - half complete condition

    weight_zp = numpy.ones((input_data.shape[dim], ), dtype=numpy.float64)  # Don't scale outside of ImpRespBW
    weight_zp[numpy.floor((input_data.shape[dim]-weight_size)/2)+numpy.arange(weight_size)] = weighting

    # Divide out weighting in spatial frequency domain
    output_data = numpy.fft.fftshift(numpy.fft.fft(input_data, axis=dim), axes=dim)
    output_data = output_data/weight_zp
    output_data = numpy.fft.ifft(numpy.fft.ifftshift(output_data, axes=dim), axis=dim)

    return output_data
