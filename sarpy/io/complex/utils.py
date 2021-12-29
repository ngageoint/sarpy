"""
Common functionality for converting metadata
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from typing import Iterator, Tuple, List

import numpy
from numpy.polynomial import polynomial
from scipy.stats import scoreatpercentile
from scipy.linalg import lstsq

from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.blocks import Poly2DType

logger = logging.getLogger(__name__)


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
        passed through to :func:`scipy.linalg.lstsq` (as `cond`).

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
    # noinspection PyTypeChecker
    for i, index in enumerate(numpy.ndindex((x_order+1, y_order+1))):
        A[:, i] = numpy.power(x, index[0])*numpy.power(y, index[1])
    # perform least squares fit
    sol, residuals, rank, sing_values = lstsq(A, z, cond=rcond)
    if isinstance(residuals, (numpy.ndarray, numpy.number)):
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
        return (array - image_data.SCPPixel.Row + image_data.FirstRow)*grid.Row.SS
    elif direction.upper() == 'COL':
        return (array - image_data.SCPPixel.Col + image_data.FirstCol)*grid.Col.SS
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
        the dop rate polynomial relative to physical coordinates - this a
        common construct in converting metadata for csk/sentinel/radarsat
    poly_order : int
        the degree of the polynomial to fit.
    Returns
    -------
    Poly2DType
    """

    grid_samples = poly_order + 8
    coords_az = get_im_physical_coords(
        numpy.linspace(0, image_data.NumCols - 1, grid_samples, dtype='float64'), grid, image_data, 'col')
    coords_rg = get_im_physical_coords(
        numpy.linspace(0, image_data.NumRows - 1, grid_samples, dtype='float64'), grid, image_data, 'row')
    coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)
    time_ca_sampled = inca.TimeCAPoly(coords_az_2d)
    dop_centroid_sampled = inca.DopCentroidPoly(coords_rg_2d, coords_az_2d)
    doppler_rate_sampled = polynomial.polyval(coords_rg_2d, dop_rate_scaled_coeffs)
    time_coa_sampled = time_ca_sampled + dop_centroid_sampled / doppler_rate_sampled
    coefs, residuals, rank, sing_values = two_dim_poly_fit(
        coords_rg_2d, coords_az_2d, time_coa_sampled,
        x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
    logger.info(
        'The time_coa_fit details:\n\t'
        'root mean square residuals = {}\n\t'
        'rank = {}\n\t'
        'singular values = {}'.format(residuals, rank, sing_values))
    return Poly2DType(Coefs=coefs)


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
    if max_degree > 10:
        logger.warning('max_degree greater than 10 for polynomial fitting may lead\n\t'
                        'to poorly conditioned (i.e. badly behaved) fit.')

    deg = 1
    prev_vel_error = numpy.inf
    P_x, P_y, P_z = None, None, None
    while deg <= min(max_degree, position_array.shape[0]-1):
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


def sicd_reader_iterator(reader, partitions=None, polarization=None, band=None):
    """
    Provides an iterator over a collection of partitions (tuple of tuple of integer
    indices for the reader) for a sicd type reader object.

    Parameters
    ----------
    reader : SICDTypeReader
    partitions : None|tuple
        The partitions collection. If None, then partitioning from
        `reader.get_sicd_partitions()` will be used.
    polarization : None|str
        The polarization string to match.
    band : None|str
        The band to match.

    Returns
    -------
    Iterator[tuple]
        Yields the partition index, the sicd reader index, and the sicd structure.
    """

    def sicd_match():
        match = True
        if band is not None:
            match &= (this_sicd.get_transmit_band_name() == band)
        if polarization is not None:
            match &= (this_sicd.get_processed_polarization() == polarization)
        return match

    if not isinstance(reader, SICDTypeReader):
        raise TypeError('reader must be an instance of SICDTypeReader. Got type {}'.format(type(reader)))
    if reader.reader_type != "SICD":
        raise ValueError('The provided reader must be of SICD type.')

    if partitions is None:
        # noinspection PyUnresolvedReferences
        partitions = reader.get_sicd_partitions()
    # noinspection PyUnresolvedReferences
    the_sicds = reader.get_sicds_as_tuple()
    for this_partition, entry in enumerate(partitions):
        for this_index in entry:
            this_sicd = the_sicds[this_index]
            if sicd_match():
                yield this_partition, this_index, this_sicd


def get_physical_coordinates(the_sicd, row_value, col_value):
    """
    Transform from image coordinates to physical coordinates, for polynomial evaluation.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    row_value : int|float|numpy.ndarray
    col_value : int|float|numpy.ndarray

    Returns
    -------
    tuple
        The row and colummn physical coordinates
    """

    return get_im_physical_coords(row_value, the_sicd.Grid, the_sicd.ImageData, 'row'), \
           get_im_physical_coords(col_value, the_sicd.Grid, the_sicd.ImageData, 'col')



###################
# helper functions

def get_fetch_block_size(start_element, stop_element, block_size_in_bytes, bands=1):
    """
    Gets the appropriate block size, given fetch parameters and constraints.

    Note: this is a helper function, and no checking of argument validity will
    be performed.

    Parameters
    ----------
    start_element : int
    stop_element : int
    block_size_in_bytes : int
    bands : int

    Returns
    -------
    int
    """

    if stop_element == start_element:
        return None
    if block_size_in_bytes is None:
        return None

    full_size = float(abs(stop_element - start_element))
    return max(1, int(numpy.ceil(block_size_in_bytes / float(bands*8*full_size))))


def extract_blocks(the_range, index_block_size):
    # type: (Tuple[int, int, int], int) -> (List[Tuple[int, int, int]], List[Tuple[int, int]])
    """
    Convert the single range definition into a series of range definitions in
    keeping with fetching of the appropriate block sizes.

    Note: this is a helper function, and no checking of argument validity will
    be performed.

    Parameters
    ----------
    the_range : Tuple[int, int, int]
        The input (off processing axis) range.
    index_block_size : None|int|float
        The size of blocks (number of indices).

    Returns
    -------
    List[Tuple[int, int, int]], List[Tuple[int, int]]
        The sequence of range definitions `(start index, stop index, step)`
        relative to the overall image, and the sequence of start/stop indices
        for positioning of the given range relative to the original range.
    """

    entries = numpy.arange(the_range[0], the_range[1], the_range[2], dtype=numpy.int64)
    if index_block_size is None:
        return [the_range, ], [(0, entries.size), ]

    # how many blocks?
    block_count = int(numpy.ceil(entries.size/float(index_block_size)))
    if index_block_size == 1:
        return [the_range, ], [(0, entries.size), ]

    # workspace for what the blocks are
    out1 = []
    out2 = []
    start_ind = 0
    for i in range(block_count):
        end_ind = start_ind+index_block_size
        if end_ind < entries.size:
            block1 = (int(entries[start_ind]), int(entries[end_ind]), the_range[2])
            block2 = (start_ind, end_ind)
        else:
            block1 = (int(entries[start_ind]), the_range[1], the_range[2])
            block2 = (start_ind, entries.size)
        out1.append(block1)
        out2.append(block2)
        start_ind = end_ind
    return out1, out2


def get_data_mean_magnitude(bounds, reader, index, block_size_in_bytes):
    """
    Gets the mean magnitude in the region defined by bounds.

    Note: this is a helper function, and no checking of argument validity will
    be performed.

    Parameters
    ----------
    bounds : numpy.ndarray|tuple|list
        Of the form `(row_start, row_end, col_start, col_end)`.
    reader : SICDTypeReader
        The data reader.
    index : int
        The reader index to use.
    block_size_in_bytes : int|float
        The block size in bytes.

    Returns
    -------
    float
    """

    # Extract the mean of the data magnitude - for global remap usage
    logger.info(
        'Calculating mean over the block ({}:{}, {}:{}), this may be time consuming'.format(*bounds))
    mean_block_size = get_fetch_block_size(bounds[0], bounds[1], block_size_in_bytes)
    mean_column_blocks, _ = extract_blocks((bounds[2], bounds[3], 1), mean_block_size)
    mean_total = 0.0
    mean_count = 0
    for this_column_range in mean_column_blocks:
        data = numpy.abs(reader[
                         bounds[0]:bounds[1],
                         this_column_range[0]:this_column_range[1],
                         index])
        mask = (data > 0) & numpy.isfinite(data)
        mean_total += numpy.sum(data[mask])
        mean_count += numpy.sum(mask)
    return float(mean_total / mean_count)


def stats_calculation(data, percentile=None):
    """
    Calculate the statistics for input into the nrl remap.

    Parameters
    ----------
    data : numpy.ndarray
        The amplitude array, assumed real valued an finite.
    percentile : None|float|int
        Which percentile to calculate

    Returns
    -------
    tuple
        Of the form `(`minimum, maximum)` or `(minimum, maximum, `percentile` percentile)
    """

    if percentile is None:
        return numpy.min(data), numpy.max(data)
    else:
        return numpy.min(data), numpy.max(data), scoreatpercentile(data, percentile)


def get_data_extrema(bounds, reader, index, block_size_in_bytes, percentile=None):
    """
    Gets the minimum and maximum magnitude in the region defined by bounds,
    optionally, get as **estimate** for the `percentage` percentile point.

    Note: this is a helper function, and no checking of argument validity will
    be performed.

    Parameters
    ----------
    bounds : numpy.ndarray|tuple|list
        Of the form `(row_start, row_end, col_start, col_end)`.
    reader : SICDTypeReader
        The data reader.
    index : int
        The reader index to use.
    block_size_in_bytes : int|float
        The block size in bytes.
    percentile : None|int|float
        The optional percentile to estimate.

    Returns
    -------
    Tuple
        The minimum (finite) observed, maximum (finite) observed, and optionally,
        the desired (of finite) percentile.

        The minimum, maximum, and percentile values will be `None` exactly when the reader
        contains no finite data.
    """

    min_value = None
    max_value = None
    percent = None

    # Extract the mean of the data magnitude - for global remap usage
    logger.info(
        'Calculating extrema over the block ({}:{}, {}:{}), this may be time consuming'.format(*bounds))
    mean_block_size = get_fetch_block_size(bounds[0], bounds[1], block_size_in_bytes)
    mean_column_blocks, _ = extract_blocks((bounds[2], bounds[3], 1), mean_block_size)
    for this_column_range in mean_column_blocks:
        data = numpy.abs(reader[
                         bounds[0]:bounds[1],
                         this_column_range[0]:this_column_range[1],
                         index])
        mask = numpy.isfinite(data)
        if numpy.any(mask):
            temp_values = stats_calculation(data[mask], percentile=percentile)

            min_value = temp_values[0] if min_value is None else min(min_value, temp_values[0])
            max_value = temp_values[1] if max_value is None else max(max_value, temp_values[1])
            percent = temp_values[2] if percentile is not None else None

    if percentile is None:
        return min_value, max_value
    else:
        return min_value, max_value, percent
