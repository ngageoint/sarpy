# -*- coding: utf-8 -*-
"""
Common functionality for converting metadata
"""
import sys
import logging
from xml.etree import ElementTree
from typing import Union, Tuple

import numpy
from numpy.polynomial import polynomial

from .sicd_elements.blocks import Poly2DType

integer_types = (int, )
string_types = (str, )
int_func = int
if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    from cStringIO import StringIO
    # noinspection PyUnresolvedReferences
    int_func = long
    # noinspection PyUnresolvedReferences
    integer_types = (int, long)
    # noinspection PyUnresolvedReferences
    string_types = (str, unicode)
else:
    # noinspection PyUnresolvedReferences
    from io import StringIO

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


def validate_range(arg, siz):
    # type: (Union[None, int, Tuple[int, int], Tuple[int, int, int]], int) -> Tuple[int, int, int]
    """
    Validate the range definition.

    Parameters
    ----------
    arg : None|int|Tuple[int, int]|Tuple[int, int, int]
    siz : int

    Returns
    -------
    Tuple[int, int, int]
        Of the form `(start, stop, step)`.
    """

    start, stop, step = None, None, None
    if arg is None:
        pass
    elif isinstance(arg, integer_types):
        step = arg
    else:
        # NB: following this pattern to avoid confused pycharm inspection
        if len(arg) == 1:
            step = arg[0]
        elif len(arg) == 2:
            stop, step = arg
        elif len(arg) == 3:
            start, stop, step = arg
    start = 0 if start is None else int_func(start)
    stop = siz if stop is None else int_func(stop)
    step = 1 if step is None else int_func(step)
    # basic validity check
    if not (-siz < start < siz):
        raise ValueError(
            'Range argument {} has extracted start {}, which is required '
            'to be in the range [0, {})'.format(arg, start, siz))
    if not (-siz < stop <= siz):
        raise ValueError(
            'Range argument {} has extracted "stop" {}, which is required '
            'to be in the range [0, {}]'.format(arg, stop, siz))
    if not ((0 < step < siz) or (-siz < step < 0)):
        raise ValueError(
            'Range argument {} has extracted step {}, for an axis of length '
            '{}'.format(arg, start, siz))
    if ((step < 0) and (stop > start)) or ((step > 0) and (start > stop)):
        raise ValueError(
            'Range argument {} has extracted start {}, stop {}, step {}, '
            'which is not valid.'.format(arg, start, stop, step))

    # reform negative values for start/stop appropriately
    if start < 0:
        start += siz
    if stop < 0:
        stop += siz
    return start, stop, step


def reverse_range(arg, siz):
    # type: (Union[None, int, Tuple[int, int], Tuple[int, int, int]], int) -> Tuple[int, int, int]
    """
    Reverse the range definition.

    Parameters
    ----------
    arg : None|int|Tuple[int,int]|Tuple[int,int,int]
    siz : int

    Returns
    -------
    Tuple[int,int,int]
        Of the form `(start, stop, step)`.
    """

    start, stop, step = validate_range(arg, siz)
    # read backwards
    return (siz - 1) - start, (siz - 1) - stop, -step


def parse_timestring(str_in, precision='us'):
    if str_in.strip()[-1] == 'Z':
        return numpy.datetime64(str_in[:-1], precision)
    return numpy.datetime64(str_in, precision)


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


def parse_xml_from_string(xml_string):
    """
    Parse the ElementTree root node and xml namespace dict from an xml string.

    Parameters
    ----------
    xml_string : str|bytes

    Returns
    -------
    (ElementTree.Element, dict)
    """

    if not isinstance(xml_string, string_types) and isinstance(xml_string, bytes):
        xml_string = xml_string.decode('utf-8')

    root_node = ElementTree.fromstring(xml_string)
    # define the namespace dictionary
    xml_ns = dict([node for _, node in ElementTree.iterparse(StringIO(xml_string), events=('start-ns',))])
    if len(xml_ns.keys()) == 0:
        xml_ns = None
    elif '' in xml_ns:
        xml_ns['default'] = xml_ns['']
    return root_node, xml_ns


def snr_to_rniirs(bandwidth_area, signal, noise):
    """
    Calculate the information_density and RNIIRS estimate from bandwidth area and
    signal/noise estimates.

    It is assumed that geometric effects for signal and noise have been accounted for
    (i.e. use SigmaZeroSFPoly), and signal and noise have each been averaged to a
    single pixel value.

    This mapping has been empirically determined by fitting Shannon-Hartley channel
    capacity to RNIIRS for some sample images.

    Parameters
    ----------
    bandwidth_area : float
    signal : float
    noise : float

    Returns
    -------
    (float, float)
        The information_density and RNIIRS
    """

    information_density = bandwidth_area*numpy.log2(1 + signal/noise)

    a = numpy.array([3.7555, .3960], dtype=numpy.float64)
    # we have empirically fit so that
    #   rniirs = a_0 + a_1*log_2(information_density)

    # note that if information_density is sufficiently small, it will
    # result in negative values in the above functional form. This would be
    # invalid for RNIIRS by definition, so we must avoid this case.

    # We transition to a linear function of information_density
    # below a certain point. This point will be chosen to be the (unique) point
    # at which the line tangent to the curve intersects the origin, and the
    # linear approximation below that point will be defined by this tangent line.

    # via calculus, we can determine analytically where that happens
    # rniirs_transition = a[1]/numpy.log(2)
    iim_transition = numpy.exp(1 - numpy.log(2)*a[0]/a[1])
    slope = a[1]/(iim_transition*numpy.log(2))

    if information_density > iim_transition:
        return information_density, a[0] + a[1]*numpy.log2(information_density)
    else:
        return information_density, slope*information_density


def fit_position_xvalidation(time_array, position_array, velocity_array, max_degree=5):
    """
    Empirically fit the polynomials for the X, Y, Z ECF position array, using cross
    validation with the velocity array to determine the best fit degree up to a
    given maximum degree.

    Parameters
    ----------
    time_array : numpy.ndarray
    position_array : numpy.ndarray
    velocity_array : numpy.ndarray
    max_degree : int

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray,)
        The X, Y, Z polynomial coefficients.
    """

    if not isinstance(time_array, numpy.ndarray) or \
            not isinstance(position_array, numpy.ndarray) or \
            not isinstance(velocity_array, numpy.ndarray):
        raise TypeError('time_array, position_array, and velocity_array must be numpy.ndarray instances.')

    if time_array.ndim != 1 and time_array.size > 1:
        raise ValueError('time_array must be one-dimensional with at least 2 elements.')

    if position_array.shape != velocity_array.shape:
        raise ValueError('position_array and velocity_array must have the same shape.')

    if position_array.shape[0] != time_array.size:
        raise ValueError('The first dimension of position_array must be the same size as time_array.')

    if position_array.shape[1] != 3:
        raise ValueError('The second dimension of position array must have size 3, '
                         'representing X, Y, Z ECF coordinates.')

    max_degree = int(max_degree)
    if max_degree < 1:
        raise ValueError('max_degree must be at least 1.')
    if max_degree > 6:
        logging.warning('max_degree greater than 6 for polynomial fitting is very likely to lead '
                        'to poorly conditioned (i.e. badly behaved) fit.')

    deg = 1
    prev_vel_error = numpy.inf
    P_x, P_y, P_z = None, None, None
    while deg < min(max_degree, position_array.shape[0]):
        # fit position
        P_x = polynomial.polyfit(time_array, position_array[:, 0], deg=deg)
        P_y = polynomial.polyfit(time_array, position_array[:, 1], deg=deg)
        P_z = polynomial.polyfit(time_array, position_array[:, 2], deg=deg)
        # extract estimated velocities
        vel_est = numpy.hstack(
            (polynomial.polyval(time_array, polynomial.polyder(P_x))[:, numpy.newaxis],
             polynomial.polyval(time_array, polynomial.polyder(P_y))[:, numpy.newaxis],
             polynomial.polyval(time_array, polynomial.polyder(P_z))[:, numpy.newaxis]))
        # check our velocity error
        vel_err = vel_est - velocity_array
        cur_vel_error = numpy.sum((vel_err * vel_err))
        deg += 1
        # stop if the error is not smaller than at the previous step
        if cur_vel_error >= prev_vel_error:
            break
    return P_x, P_y, P_z
