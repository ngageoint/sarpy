"""This module includes functions to assist in transforming SICD data to a common state."""

import numpy as np
from numpy.polynomial import polynomial

__author__ = ["Wade Schwartzkopf", "Daniel Haverporth"]
__classification__ = "UNCLASSIFIED"
__email__ = "Wade.C.Schwartzkopf.ctr@nga.mil"


def is_normalized(sicd_meta, dim=1):
    """Test whether SAR data is already normalized in a given dimension."""
    if dim == 1:  # slow-time
        sicd_grid_struct = sicd_meta.Grid.Col
    else:  # fast-time
        sicd_grid_struct = sicd_meta.Grid.Row

    # Several reasons that we might need to applied normalization
    needs_deskew = (hasattr(sicd_grid_struct, 'DeltaKCOAPoly') and
                    np.any(sicd_grid_struct.DeltaKCOAPoly != 0))
    not_uniform_weight = ((hasattr(sicd_grid_struct, 'WgtType') and
                           hasattr(sicd_grid_struct.WgtType, 'WindowName') and
                           not sicd_grid_struct.WgtType.WindowName == 'UNIFORM') or
                          (hasattr(sicd_grid_struct, 'WgtFunct') and
                           np.any(np.diff(sicd_grid_struct.WgtFunct))))
    needs_fft_sgn_flip = sicd_grid_struct.Sgn(1) == 1

    return not needs_deskew and not not_uniform_weight and not needs_fft_sgn_flip


def deskewparams(sicd_meta, dim):
    """Extract from SICD structure the parameters required for deskewmem."""

    # DeltaKCOAPoly
    if (dim == 0 and hasattr(sicd_meta, 'Grid') and
       hasattr(sicd_meta.Grid, 'Row') and
       hasattr(sicd_meta.Grid.Row, 'DeltaKCOAPoly')):
        DeltaKCOAPoly = sicd_meta.Grid.Row.DeltaKCOAPoly
    elif (dim == 1 and hasattr(sicd_meta, 'Grid') and
          hasattr(sicd_meta.Grid, 'Col') and
          hasattr(sicd_meta.Grid.Col, 'DeltaKCOAPoly')):
        DeltaKCOAPoly = sicd_meta.Grid.Col.DeltaKCOAPoly
    else:  # assum to be 0 if undefined
        DeltaKCOAPoly = 0
    # Vectors describing range and azimuth distances from SCP (in meters) for rows and columns
    rg_coords_m = (np.arange(0, sicd_meta.ImageData.NumRows, dtype=np.float32) +
                   float(sicd_meta.ImageData.FirstRow) -
                   float(sicd_meta.ImageData.SCPPixel.Row)) * sicd_meta.Grid.Row.SS
    az_coords_m = (np.arange(0, sicd_meta.ImageData.NumCols, dtype=np.float32) +
                   float(sicd_meta.ImageData.FirstCol) -
                   float(sicd_meta.ImageData.SCPPixel.Col)) * sicd_meta.Grid.Col.SS
    # FFT sign required to transform data to spatial frequency domain
    if (dim == 0 and hasattr(sicd_meta, 'Grid') and
       hasattr(sicd_meta.Grid, 'Row') and hasattr(sicd_meta.Grid.Row, 'Sgn')):
        fft_sgn = sicd_meta.Grid.Row.Sgn
    elif (dim == 1 and hasattr(sicd_meta, 'Grid') and
          hasattr(sicd_meta.Grid, 'Col') and hasattr(sicd_meta.Grid.Col, 'Sgn')):
        fft_sgn = sicd_meta.Grid.Col.Sgn
    else:
        fft_sgn = -1  # Most common
    return DeltaKCOAPoly, rg_coords_m, az_coords_m, fft_sgn


def deskewmem(input_data, DeltaKCOAPoly, dim0_coords_m, dim1_coords_m, dim, fft_sgn=-1):
    """Performs deskew (centering of the spectrum on zero frequency) on a complex dataset.

    INPUTS:
       input_data:  Complex FFT Data
       DeltaKCOAPoly:  Polynomial that describes center of frequency support of data.
       dim0_coords_m:  Coordinate of each "row" in dimension 0
       dim1_coords_m:  Coordinate of each "column" in dimension 1
       dim:  Dimension over which to perform deskew
       fft_sgn:  FFT sign required to transform data to spatial frequency domain
    OUTPUTS:
       output_data:  Deskewed data
       new_DeltaKCOAPoly:  Frequency support shift in the non-deskew dimension
          caused by the deskew.
    """

    # Integrate DeltaKCOA polynomial (in meters) to form new polynomial DeltaKCOAPoly_int
    DeltaKCOAPoly_int = polynomial.polyint(DeltaKCOAPoly, axis=dim)
    # New DeltaKCOAPoly in other dimension will be negative of the derivative of
    # DeltaKCOAPoly_int in other dimension (assuming it was zero before).
    new_DeltaKCOAPoly = - polynomial.polyder(DeltaKCOAPoly_int, axis=dim-1)
    # Apply phase adjustment from polynomial
    [dim1_coords_m_2d, dim0_coords_m_2d] = np.meshgrid(dim1_coords_m, dim0_coords_m)
    output_data = np.multiply(input_data, np.exp(1j * fft_sgn * 2 * np.pi *
                                                 polynomial.polyval2d(
                                                     dim0_coords_m_2d,
                                                     dim1_coords_m_2d,
                                                     DeltaKCOAPoly_int)))
    return output_data, new_DeltaKCOAPoly


def deweightmem(input_data, weight_fun=None, oversample_rate=1, dim=1):
    """DEWEIGHTMEM Make complex SAR uniformly weighted in one dimension.

          Parameter name    Description

          input_data        Array of complex values to deweight
          weight_fun        Description of weighting applied.  Either a
                               function that takes a single argument (number
                               of elements) and produces the weighting to
                               apply, or a vector that is the weighting
                               function sampled.
          oversample_rate   Amount of sampling beyond the ImpRespBW in the
                               processing dimension. (Default is Nyquist
                               sampling = 1).
          dim               Dimension over which to perform deweighting.
                               Default is 1.

    This implementation assumes that the data has already been "deskewed" and
    that the frequency support is centered.
    """
    # TODO: Test this function

    # Weighting only valid across ImpRespBW
    weight_size = round(input_data.shape[dim]/oversample_rate)
    if weight_fun is None:  # No weighting passed in.  Do nothing.
        return input_data
    elif callable(weight_fun):
        weighting = weight_fun(weight_size)
    elif np.array(weight_fun).ndim == 1:
        import scipy.signal as sig
        weighting = sig.resample(weight_fun, weight_size)

    weight_zp = np.ones(input_data.shape[dim])  # Don't scale outside of ImpRespBW
    weight_zp[np.floor((input_data.shape[dim]-weight_size)/2)+np.arange(weight_size)] = weighting

    # Divide out weighting in spatial frequency domain
    output_data = np.fft.fftshift(np.fft.fft(input_data, axis=dim), axes=dim)
    output_data = output_data/weight_zp
    output_data = np.fft.ifft(np.fft.ifftshift(output_data, axes=dim), axis=dim)

    return output_data
