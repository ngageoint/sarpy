"""Functions to transform SICD data to a common state"""

import numpy as np
from numpy.polynomial import polynomial
import scipy.signal

__classification__ = "UNCLASSIFIED"
__author__ = ("Wade Schwartzkopf", "Daniel Haverporth")


# TODO: It seems clear that these three methods are part of a single workflow
#   which should be tidied up. How does it happen?


def is_normalized(sicd_meta, dim=1):
    """

    Parameters
    ----------
    sicd_meta : sarpy.io.complex.sicd_elements.SICD.SICDType
        the sicd structure
    dim : int
        the dimension to test

    Returns
    -------
    bool
        normalization state in the given dimension
    """

    def dimension_check(dir_param):
        # type: (sarpy.io.complex.sicd_elements.Grid.DirParamType) -> bool
        # Several reasons that we might need to applied normalization
        needs_deskew = ((dir_param.DeltaKCOAPoly is not None) and
                        np.any(dir_param.DeltaKCOAPoly.get_array() != 0))
        not_uniform_weight = (
                (dir_param.WgtType is not None and dir_param.WgtType.WindowName is not None
                 and dir_param.WgtType.WindowName != 'UNIFORM') or
                (dir_param.WgtFunct is not None and np.any(np.diff(dir_param.WgtFunct))))
        needs_fft_sgn_flip = (dir_param.Sgn == 1)
        return not (needs_deskew or not_uniform_weight or needs_fft_sgn_flip)

    if dim == 1:  # slow-time
        return dimension_check(sicd_meta.Grid.Col)
    else:  # fast-time
        return dimension_check(sicd_meta.Grid.Row)


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

    # TODO: HIGH - this should be class method of SICD class (or something like that)

    # DeltaKCOAPoly
    DeltaKCOAPoly = np.array([[0, ], ], dtype=np.float64)
    fft_sgn = -1
    if dim == 0:
        try:
            DeltaKCOAPoly = sicd_meta.Grid.Row.DeltaKCOAPoly.get_array()
        except (ValueError, AttributeError):
            pass
        try:
            fft_sgn = sicd_meta.Grid.Row.Sgn
        except (ValueError, AttributeError):
            pass
    else:
        try:
            DeltaKCOAPoly = sicd_meta.Grid.Col.DeltaKCOAPoly.get_array()
        except (ValueError, AttributeError):
            pass
        try:
            fft_sgn = sicd_meta.Grid.Col.Sgn
        except (ValueError, AttributeError):
            pass

    # Vectors describing range and azimuth distances from SCP (in meters) for rows and columns
    rg_coords_m = (np.arange(0, sicd_meta.ImageData.NumRows, dtype=np.float32) +
                   sicd_meta.ImageData.FirstRow - sicd_meta.ImageData.SCPPixel.Row)*sicd_meta.Grid.Row.SS
    az_coords_m = (np.arange(0, sicd_meta.ImageData.NumCols, dtype=np.float32) +
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
    dim1_coords_m_2d, dim0_coords_m_2d = np.meshgrid(dim1_coords_m, dim0_coords_m)
    output_data = np.multiply(input_data, np.exp(1j * fft_sgn * 2 * np.pi *
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
    elif np.array(weight_fun).ndim == 1:
        weighting = scipy.signal.resample(weight_fun, weight_size)
    # TODO: HIGH - half complete condition

    weight_zp = np.ones((input_data.shape[dim], ), dtype=np.float64)  # Don't scale outside of ImpRespBW
    weight_zp[np.floor((input_data.shape[dim]-weight_size)/2)+np.arange(weight_size)] = weighting

    # Divide out weighting in spatial frequency domain
    output_data = np.fft.fftshift(np.fft.fft(input_data, axis=dim), axes=dim)
    output_data = output_data/weight_zp
    output_data = np.fft.ifft(np.fft.ifftshift(output_data, axes=dim), axis=dim)

    return output_data
