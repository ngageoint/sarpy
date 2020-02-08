# -*- coding: utf-8 -*-
"""
Common functionality for converting metadata
"""

import logging
import numpy
from numpy.polynomial import polynomial

from .sicd_elements.blocks import Poly2DType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


def get_seconds(dt1, dt2, precision='us'):
    """
    The number of seconds between two numpy.datetime64 elements.

    Parameters
    ----------
    dt1 : numpy.datetime64
    dt2 : numpy.datetime64
    precision : str
        one of 's', 'ms', 'us', or 'ns'
    Returns
    -------
    float
        the number of seconds between dt2 and dt1 (i.e. dt1 - dt2).
    """
    if precision == 's':
        scale = 1
    elif precision == 'ms':
        scale = 1e-3
    elif precision == 'us':
        scale = 1e-6
    elif precision == 'ns':
        scale = 1e-9
    else:
        raise ValueError('unrecognized precision {}'.format(precision))

    dtype = 'datetime64[{}]'.format(precision)
    tdt1 = dt1.astype(dtype)
    tdt2 = dt2.astype(dtype)
    return float((tdt1.astype('int64') - tdt2.astype('int64'))*scale)


def two_dim_poly_fit(x, y, z, x_order=2, y_order=2, x_scale=1, y_scale=1, rcond=None):
    """
    Perform fit of data to two dimensional polynomial.

    Parameters
    ----------
    x : numpy.ndarray
        the x data
    y : numpy.ndarray
        the y data
    z : numpy.ndarray
        the z data
    x_order : int
        the order for x
    y_order : int
        the order for y
    x_scale : float
        In order to help the fitting problem to become better conditioned, the independent
        variables can be scaled, the fit performed, and then the solution rescaled.
    y_scale : float
    rcond : None|float
        passed through to :func:`numpy.linalg.lstsq`.
    Returns
    -------
    numpy.ndarray
        the coefficient array
    """

    if not isinstance(x, numpy.ndarray) or not isinstance(y, numpy.ndarray) or not isinstance(z, numpy.ndarray):
        raise TypeError('x, y, z must be numpy arrays')
    if (x.size != z.size) or (y.size != z.size):
        raise ValueError('x, y, z must have the same cardinality size.')

    x = x.flatten()*x_scale
    y = y.flatten()*y_scale
    z = z.flatten()
    # first, we need to formulate this as A*t = z
    # where A has shape (x.size, (x_order+1)*(y_order+1))
    # and t has shape ((x_order+1)*(y_order+1), )
    A = numpy.empty((x.size, (x_order+1)*(y_order+1)), dtype=numpy.float64)
    for i, index in enumerate(numpy.ndindex((x_order+1, y_order+1))):
        A[:, i] = numpy.power(x, index[0])*numpy.power(y, index[1])
    # perform least squares fit
    sol, residuals, rank, sing_values = numpy.linalg.lstsq(A, z, rcond=rcond)
    # NB: it seems like this problem is not always well-conditioned (TimeCOAPoly, at least)
    if len(residuals) != 0:
        residuals /= float(x.size)
    sol = numpy.power(x_scale, numpy.arange(x_order+1))[:, numpy.newaxis] * \
          numpy.reshape(sol, (x_order+1, y_order+1)) * \
          numpy.power(y_scale, numpy.arange(y_order+1))
    print('sol = {}'.format(sol))
    return sol, residuals, rank, sing_values


def get_im_physical_coords(array, grid, image_data, direction):
    """
    Converts one dimension of "pixel" image (row or column) coordinates to
    "physical" image (range or azimuth in meters) coordinates, for use in the
    various two-variable sicd polynomials.

    Parameters
    ----------
    array : numpy.array|float|int
        either row or col coordinate component
    grid : sarpy.io.complex.sicd_elements.Grid.GridType
    image_data : sarpy.io.complex.sicd_elements.ImageData.ImageDataType
    direction : str
        one of 'Row' or 'Col' (case insensitive) to determine which
    Returns
    -------
    numpy.array|float
    """

    if direction.upper() == 'ROW':
        return (array - image_data.SCPPixel.Row)*grid.Row.SS
    elif direction.upper() == 'COL':
        return (array - image_data.SCPPixel.Col)*grid.Col.SS
    else:
        raise ValueError('Unrecognized direction {}'.format(direction))


def fit_time_coa_polynomial(inca, image_data, grid, dop_rate_scaled_coeffs, poly_order=2):
    """

    Parameters
    ----------
    inca : sarpy.io.complex.sicd_elements.RMA.INCAType
    image_data : sarpy.io.complex.sicd_elements.ImageData.ImageDataType
    grid : sarpy.io.complex.sicd_elements.Grid.GridType
    dop_rate_scaled_coeffs : numpy.ndarray
        the dop rate polynomial relative to physical coordinates - the is a
        common construct in converting metadata for csk/sentinel/radarsat
    poly_order : int
        the degree of the polynomial to fit.
    Returns
    -------
    Poly2DType
    """

    grid_samples = poly_order + 3
    coords_az = get_im_physical_coords(
        numpy.linspace(0, image_data.NumCols - 1, grid_samples, dtype=numpy.float64), grid, image_data, 'col')
    coords_rg = get_im_physical_coords(
        numpy.linspace(0, image_data.NumRows - 1, grid_samples, dtype=numpy.float64), grid, image_data, 'row')
    coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)
    time_ca_sampled = inca.TimeCAPoly(coords_az_2d)
    dop_centroid_sampled = inca.DopCentroidPoly(coords_rg_2d, coords_az_2d)
    doppler_rate_sampled = polynomial.polyval(coords_rg_2d, dop_rate_scaled_coeffs)
    time_coa_sampled = time_ca_sampled + dop_centroid_sampled / doppler_rate_sampled
    coefs, residuals, rank, sing_values = two_dim_poly_fit(
        coords_rg_2d, coords_az_2d, time_coa_sampled,
        x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
    logging.info('The time_coa_fit details:\nroot mean square residuals = {}\nrank = {}\nsingular values = {}'.format(residuals, rank, sing_values))
    return Poly2DType(Coefs=coefs)
