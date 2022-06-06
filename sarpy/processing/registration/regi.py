"""
Basic image registration, generally best suited most suited for coherent image
collection. This is based pretty directly on an approach developed at Sandia and
generally referred to by the name "regi".

The relevant matlab code appears to be authored by Terry M. Calloway,
Sandia National Laboratories, and modified by Wade Schwartzkopf, NGA.
"""

__classification__ = 'UNCLASSIFIED'
__author__ = ["Thomas McCullough", "Terry M. Calloway", "Wade Schwartzkopf"]


import logging
from typing import List, Tuple, Dict, Union, Optional

import numpy
from scipy.signal import correlate2d
from scipy.interpolate import LinearNDInterpolator

from sarpy.io.general.base import BaseReader


logger = logging.getLogger(__name__)


def _validate_match_parameters(
        reference_size: Tuple[int, int],
        moving_size: Tuple[int, int],
        match_box_size: Tuple[int, int],
        moving_deviation: Tuple[int, int],
        decimation: Tuple[int, int]) -> None:
    """
    Validate the match paramaters based the size of the images.

    Parameters
    ----------
    reference_size : Tuple[int, int]
    moving_size : Tuple[int, int]
    match_box_size : Tuple[int, int]
    moving_deviation : Tuple[int, int]
    decimation : Tuple[int, int]
    """

    if not ((match_box_size[0] % 2) == 1 and match_box_size[0] > 1 and
            (match_box_size[1] % 2) == 1 and match_box_size[1] > 1):
        raise ValueError('The match box size must have both odd entries greater than 1')

    if not ((moving_deviation[0] % 2) == 1 and moving_deviation[0] > 1 and
            (moving_deviation[1] % 2) == 1 and moving_deviation[1] > 1):
        raise ValueError('The match box size must have both odd entries greater than 1')

    limit_fraction = 0.5
    if match_box_size[0]*decimation[0] > limit_fraction*reference_size[0] or \
            match_box_size[1]*decimation[1] > limit_fraction*reference_size[1]:
        raise ValueError(
            'The size of the match box - {} with decimation - {} is too large\n\t'
            'with respect tothe size of the reference image - {}'.format(
                match_box_size, decimation, reference_size))
    if match_box_size[0]*decimation[0] > limit_fraction*moving_size[0] or \
            match_box_size[1]*decimation[1] > limit_fraction*moving_size[1]:
        raise ValueError(
            'The size of the match box - {} with decimation - {} is too close\n\t'
            'to the size of the moving image - {}'.format(
                match_box_size, decimation, moving_size))


def _populate_difference_structure(
        mapping_values: List[List[Dict]]) -> None:
    """
    Helper function for populating derivative estimates into our structure.

    Parameters
    ----------
    mapping_values: List[List[dict]]
    """

    # NB: this assumes the expected structure

    def do_diff(the_diff, the_count, direction, ref_loc0, mov_loc0, ref_loc1, mov_loc1):
        if mov_loc0 is None or mov_loc1 is None:
            return the_diff, the_count
        the_diff += float(mov_loc1[direction] - mov_loc0[direction]) / \
                    float(ref_loc1[direction] - ref_loc0[direction])
        the_count += 1
        return the_diff, the_count

    def basic_estimate_diff(entry, i, j):
        ref_loc = entry['reference_location']
        mov_loc = entry['moving_location']
        if mov_loc is None:
            return

        if entry.get('row_derivative', None) is None:
            # calculate row derivative
            r_diff = 0.0
            r_count = 0
            # get value based on before
            if i > 0:
                o_entry = mapping_values[i-1][j]
                do_diff(r_diff, r_count, 0, ref_loc, mov_loc,
                        o_entry['reference_location'], o_entry['moving_location'])
            # get value based on after
            if i < len(mapping_values) - 1:
                o_entry = mapping_values[i+1][j]
                do_diff(r_diff, r_count, 0, ref_loc, mov_loc,
                        o_entry['reference_location'], o_entry['moving_location'])
            if r_count > 0:
                row_der = r_diff/float(r_count)
                entry['row_derivative'] = row_der
                if row_der < 0.0:
                    logger.warning('Entry ({}, {}) has negative row derivative ({})'.format(i, j, row_der))

        if entry.get('column_derivative', None) is None:
            # calculate the column derivative
            c_diff = 0.0
            c_count = 0
            # get the value based on before
            if j > 0:
                o_entry = mapping_values[i][j-1]
                do_diff(c_diff, c_count, 1, ref_loc, mov_loc,
                        o_entry['reference_location'], o_entry['moving_location'])
            # get value based on after
            if j < len(mapping_values[0]) - 1:
                o_entry = mapping_values[i][j+1]
                do_diff(c_diff, c_count, 1, ref_loc, mov_loc,
                        o_entry['reference_location'], o_entry['moving_location'])
            if c_count > 0:
                col_der = c_diff/float(c_count)
                entry['column_derivative'] = col_der
                if col_der < 0.0:
                    logger.warning('Entry ({}, {}) has negative column derivative ({})'.format(i, j, col_der))

    for row_index, grid_row in enumerate(mapping_values):
        for col_index, element in enumerate(grid_row):
            basic_estimate_diff(element, row_index, col_index)


def _subpixel_shift(values: numpy.ndarray) -> float:
    """
    This is simplified port of the SAR toolbox matlab function fin_minms. This
    uses data from an empirical fit derived from unknown origins to estimate where
    the "real" minimum occurred.

    Parameters
    ----------
    values : numpy.ndarray
        Must have length 3, with either `values[1] <= min(values[0], values[2])`
        (a minimization problem), or `values[1] >= max(values[0], values[2])`
        (a maximization problem). Maximization problems will be re-cast as
        minimization through inversion.

    Returns
    -------
    shift : float
        This values will be (-1, 1), with -1 corresponding to the first location,
        0 corresponding to the center location, and 1 corresponding to the final
        location.
    """

    if not (isinstance(values, numpy.ndarray) and values.ndim == 1 and values.size == 3):
        raise ValueError('The input must be a 1-d array with three entries.')

    if not (values[1] >= max(values[0], values[2]) or values[1] <= min(values[0], values[2])):
        raise ValueError(
            'The central entry must either be larger than the other two (maximization problem)\n\t'
            'or smaller than the other two (minimization problem) - values {}'.format(values))

    if values[1] >= max(values[0], values[2]):
        # recast maximization problem as minimization
        return _subpixel_shift(-values)

    if values[0] == values[1] or values[1] == values[2]:
        # no real information
        return 0.0
    if values[0] == values[2]:
        # it's symmetric
        return 0.0

    values = (values + numpy.min(values))  # ensure that everything is positive by shifting up

    # Here are the verbatim matlab comments:
    # Algorithm -
    #   xm = min
    #   r = (rmsmid-rmsmin)/(rmsmax-rmsmin)
    #   empirical fit (r,xm) resembles arc of circle centered at xc=-11/8
    #   xm = 2(xc-3/8) + sqrt(4(xc-3/8)**2 - (r-1)((r-1)-2(xc-1)))
    #   empirical fit (r,xm) resembles arc of circle centered at xc=-6/8
    #   xm = 1-xc - sqrt(0.125 + 2(0.25-xc)**2 - (x-xc)**2)

    nsr = 0.5
    noise = nsr*numpy.max(values)
    rms = numpy.sqrt(values - noise)
    rmsmin = rms[1]
    rmsmid = min(rms[0], rms[2])
    rmsmax = max(rms[0], rms[2])
    r = (rmsmid - rmsmin)/(rmsmax - rmsmin)
    rm1 = r - 1.
    fit_val = 12.25 - rm1*(rm1 + 4.75)
    if not (2.5*2.5 < fit_val < 4.5*4.5):
        # probaly impossible...
        return 0.0
    shift = -3.5 + numpy.sqrt(fit_val)
    return -shift if rms[0] < rms[2] else shift


def _max_correlation_step(
        reference_array: numpy.ndarray,
        moving_array: numpy.ndarray,
        do_subpixel: bool = False) -> Tuple[Optional[numpy.ndarray], Optional[float]]:
    """
    Find the best match location of the moving array inside the reference array.

    Parameters
    ----------
    reference_array : numpy.ndarray
    moving_array : numpy.ndarray
    do_subpixel : bool
        Include a subpixel registration effort?

    Returns
    -------
    best_location : None|numpy.ndarray
        Will return `None` if there is no information, i.e. the reference patch
        or moving patch is all 0. Otherwise, this will be a numpy array
        `[row, column]` of the location of highest correlation, determined via
        :func:`numpy.argmax`.
    maximum_correlation : None|float
    """

    if reference_array.ndim != 2 or moving_array != 2:
        raise ValueError('Input arrays must be 2-dimensional')
    if reference_array.shape[0] < moving_array.shape[0] or reference_array.shape[1] < moving_array.shape[1]:
        raise ValueError(
            'It is required that the moving array (shape {}) is strictly contained\n\t'
            'inside  the reference array (shape {})'.format(moving_array.shape, reference_array.shape))

    # NB: sqrt suggested by matlab, presumably to dampen the importance of bright returns?
    reference_array = numpy.sqrt(numpy.abs(reference_array))
    if numpy.all(reference_array == 0):
        return None, None

    moving_array = numpy.sqrt(numpy.abs(moving_array))
    if numpy.all(moving_array == 0):
        return None, None

    kernel = numpy.ones(reference_array.shape, dtype='float32')

    # now, find the best match
    match_values = correlate2d(reference_array, moving_array, mode='valid')
    norm_values = correlate2d(kernel, moving_array*moving_array, mode='valid')
    mask = (norm_values > 0)

    # reduce to dot product of the unit vectors, and we pick the best match
    match_values[mask] /= numpy.sqrt(norm_values[mask])

    # raw maximum location
    raw_max_location = numpy.unravel_index(numpy.argmax(match_values), match_values.shape)
    maximum_value = match_values[raw_max_location[0], raw_max_location[1]]

    if do_subpixel:
        sub_shift = numpy.zeros((2,), dtype='float64')
        sub_shift[0] += _subpixel_shift(match_values[raw_max_location[0]-1:raw_max_location[0]+1, raw_max_location[1]])
        sub_shift[1] += _subpixel_shift(match_values[raw_max_location[0], raw_max_location[1]-1:raw_max_location[1]+1])
        raw_max_location += sub_shift

    if (moving_array.shape[0] % 2) == 0 or (moving_array.shape[1] % 2) == 0:
        shift = numpy.zeros((2, ), dtype='float64')
    else:
        shift = numpy.zeros((2, ), dtype='int64')
    shift[0] = 0.5*(moving_array.shape[0] - 1)
    shift[1] = 0.5*(moving_array.shape[1] - 1)
    return shift + raw_max_location, maximum_value


def _single_step_location(
        reference_data: Union[BaseReader, numpy.ndarray],
        reference_index: Optional[int],
        reference_size: Tuple[int, int],
        moving_data: Union[BaseReader, numpy.ndarray],
        moving_index: Optional[int],
        moving_size: Tuple[int, int],
        reference_location: Tuple[int, int],
        moving_location: Tuple[int, int],
        match_box_size: Tuple[int, int] = (25, 25),
        moving_deviation: Tuple[int, int] = (15, 15),
        decimation: Tuple[int, int] = (1, 1)) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
    """
    Perform a single step of the reference search by finding the best matching
    location at given size and scale.

    Parameters
    ----------
    reference_data : BaseReader|numpy.ndarray
    reference_index : None|int
    reference_size : Tuple[int, int]
    moving_data : BaseReader|numpy.ndarray
    moving_index : None|int
    moving_size : Tuple[int, int]
    reference_location : Tuple[int, int]
    moving_location : Tuple[int, int]
    match_box_size : Tuple[int, int]
    moving_deviation : Tuple[int, int]
    decimation : Tuple[int, int]

    Returns
    -------
    best_location : None|Tuple[int, int]
        Will return `None` if there is no information, i.e. the reference patch
        or moving patch is all 0.
    maximum_correlation : None|float
    """

    # NB: we require odd entries here
    match_box_half = (int((match_box_size[0] - 1)/2), int((match_box_size[1] - 1)/2))
    deviation_half = (int((moving_deviation[0] - 1)/2), int((moving_deviation[1] - 1)/2))
    moving_half = (deviation_half[0] + match_box_half[0], deviation_half[1] + match_box_half[1])

    # vet the reference patch location
    ref_box_start = (
        reference_location[0] - match_box_half[0]*decimation[0],
        reference_location[1] - match_box_half[1]*decimation[1])
    if ref_box_start[0] < 0 or \
            ref_box_start[1] < 0 or \
            ref_box_start[0] > reference_size[0] - match_box_size[0]*decimation[0] + 1 or \
            ref_box_start[1] > reference_size[1] - match_box_size[1]*decimation[1] + 1:
        raise ValueError(
            'Overflows bounds. Cannot proceed with reference image of size {}\n\t'
            'using reference box of size {} and decimation {}\n\t'
            'at reference location {}'.format(reference_size, match_box_size, decimation, reference_location))
    ref_box_end = (
        ref_box_start[0] + match_box_size[0]*decimation[0] + 1,
        ref_box_start[1] + match_box_size[1]*decimation[1] + 1)

    # fetch the reference image patch
    if isinstance(reference_data, BaseReader):
        reference_array = reference_data[
                          ref_box_start[0]:ref_box_end[0]:decimation[0],
                          ref_box_start[1]:ref_box_end[1]:decimation[1],
                          reference_index]
    else:
        reference_array = reference_data[
                          ref_box_start[0]:ref_box_end[0]:decimation[0],
                          ref_box_start[1]:ref_box_end[1]:decimation[1]]

    # vet the moving patch location
    mov_loc_temp = [moving_location[0], moving_location[1]]
    for i in [0, 1]:
        if mov_loc_temp[i] < moving_half[i]*decimation[i]:
            mov_loc_temp[i] = moving_half[i]*decimation[i]
        elif mov_loc_temp[i] > moving_size[i] - moving_half[i]*decimation[i] + 1:
            mov_loc_temp[i] = moving_size[i] - moving_half[i]*decimation[i] + 1
    moving_box_start = (
        mov_loc_temp[0] - moving_half[0]*decimation[0],
        mov_loc_temp[1] - moving_half[1]*decimation[1])
    moving_box_end = (
        mov_loc_temp[0] + moving_half[0]*decimation[0] + 1,
        mov_loc_temp[1] + moving_half[1]*decimation[1] + 1)

    # fetch the moving patch array for max correlation check
    if isinstance(moving_data, BaseReader):
        moving_array = moving_data[
                       moving_box_start[0]:moving_box_end[0]:decimation[0],
                       moving_box_start[1]:moving_box_end[1]:decimation[1],
                       moving_index]
    else:
        moving_array = moving_data[
                       moving_box_start[0]:moving_box_end[0]:decimation[0],
                       moving_box_start[1]:moving_box_end[1]:decimation[1]]

    # find the point of maximum correlation
    do_subpixel = numpy.all(decimation == 1)
    best_temp_location, maximum_correlation = _max_correlation_step(reference_array, moving_array, do_subpixel=do_subpixel)

    if best_temp_location is None:
        return best_temp_location, maximum_correlation

    return ((best_temp_location[0])*decimation[0] + mov_loc_temp[0], (best_temp_location[1])*decimation[1] + mov_loc_temp[1]), maximum_correlation


def _single_step_grid(
        reference_data: Union[BaseReader, numpy.ndarray],
        reference_index: Optional[int],
        reference_size: Tuple[int, int],
        moving_data: Union[BaseReader, numpy.ndarray],
        moving_index: Optional[int],
        moving_size: Tuple[int, int],
        reference_box_rough: Tuple[int, int],
        moving_box_rough: Tuple[int, int],
        match_box_size: Tuple[int, int] = (25, 25),
        moving_deviation: Tuple[int, int] = (15, 15),
        decimation: Tuple[int, int] = (1, 1),
        previous_values: Optional[List[List[Dict]]] = None):
    """
    We will determine a series of best matching (small size) patch locations
    between the pixel area of `reference_data` laid out in `reference_box_rough`
    and the pixel area of `moving_data` laid out in `moving_box_rough` - which
    should be very close to the same size.

    Parameters
    ----------
    reference_data : BaseReader|numpy.ndarray
    reference_index : None|int
    reference_size : Tuple[int, int]
    moving_data : BaseReader|numpy.ndarray
    moving_index : None|int
    moving_size : Tuple[int, int]
    reference_box_rough : Tuple[int, int, int, int]
    moving_box_rough : Tuple[int, int, int, int]
    match_box_size : Tuple[int, int]
    moving_deviation : Tuple[int, int]
    decimation : Tuple[int, int]
    previous_values : None|List[List[dict]]

    Returns
    -------
    result_values : List[List[dict]]
        entry `[i][j]` tell the mapping of the reference location in nominal
        reference grid to moving grid location
        :code:`{'reference_location': (row, column),
        'moving_location': (matched_row, matched_column),
        'max_correlation': <value>}`
    """

    def determine_best_guess(ref_point):
        if row_interp is None:
            return (
                ref_point[0] - reference_box_rough[0] + moving_box_rough[0],
                ref_point[1] - reference_box_rough[2] + moving_box_rough[2])
        else:
            return (
                row_interp(ref_point[0]),
                col_interp(ref_point[1]))

    effective_ref_size = (
        int(reference_box_rough[1] - reference_box_rough[0]),
        int(reference_box_rough[3] - reference_box_rough[2]))
    effective_move_size = (
        int(moving_box_rough[1] - moving_box_rough[0]),
        int(moving_box_rough[3] - moving_box_rough[2]))

    # validate the parameters at this scale
    _validate_match_parameters(
        effective_ref_size, effective_move_size, match_box_size, moving_deviation, decimation)

    # construct the grid which we are going to try to map
    half_row_size = int(0.5*(match_box_size[0] - 1)*decimation[0])
    half_col_size = int(0.5*(match_box_size[1] - 1)*decimation[1])
    row_grid = numpy.arange(reference_box_rough[0] + half_row_size, reference_box_rough[1] + half_row_size, 2*half_row_size)
    col_grid = numpy.arange(reference_box_rough[2] + half_col_size, reference_box_rough[3] + half_col_size, 2*half_col_size)

    # create a mapping which estimates (ref_row, ref_col) -> (mov_row, mov_col)
    row_interp = None
    col_interp = None
    if previous_values is not None:
        # determine the approximate derivative values
        _populate_difference_structure(previous_values)

        # get values from our grid, excluding places which require a negative derivative
        ref_locs = []
        mov_locs = []

        for row_values in previous_values:
            for entry in row_values:
                row_der = entry.get('row_derivative', None)
                col_der = entry.get('column_derivative', None)
                if row_der is not None and 0.25 < row_der < 2.0 and \
                        col_der is not None and 0.25 < col_der < 2.0:
                    ref_locs.append(entry['reference_location'])
                    mov_locs.append(entry['moving_location'])

        if len(ref_locs) > 3:
            ref_locs = numpy.array(ref_locs, dtype='float64')
            mov_locs = numpy.array(mov_locs, dtype='float64')
            # create an interpolation function mapping reference coords -> moving coords (so far)
            row_interp = LinearNDInterpolator(ref_locs, mov_locs[:, 0])
            col_interp = LinearNDInterpolator(ref_locs, mov_locs[:, 1])

    result_values = []
    # generate our grid of locations to compare, and then compare
    for row_grid_entry in row_grid:
        the_row_list = []
        result_values.append(the_row_list)

        for col_grid_entry in col_grid:
            ref_loc = (row_grid_entry, col_grid_entry)
            best_loc, max_correlation = _single_step_location(
                reference_data, reference_index, reference_size,
                moving_data, moving_index, moving_size,
                ref_loc, determine_best_guess(ref_loc),
                match_box_size=match_box_size, moving_deviation=moving_deviation, decimation=decimation)
            the_row_list.append(
                {
                    'reference_location': ref_loc,
                    'moving_location': best_loc,
                    'max_correlation': max_correlation
                })

    return result_values


def register_arrays(
        reference_data: numpy.ndarray,
        moving_data: numpy.ndarray) -> List[List[Dict]]:
    """
    Register the moving_data array to the reference_data array using the regi algorithm.

    Parameters
    ----------
    reference_data : numpy.ndarray
    moving_data : numpy.ndarray

    Returns
    -------
    result_values : List[List[dict]]
        entry `[i][j]` tell the mapping of the reference location in nominal
        reference grid to moving grid location
        :code:`{'reference_location': (row, column),
        'moving_location': (matched_row, matched_column),
        'max_correlation': <value>}`
    """

    if not isinstance(reference_data, numpy.ndarray):
        raise TypeError('reference_data must be a numpy array')
    if not isinstance(moving_data, numpy.ndarray):
        raise TypeError('moving_data must be a numpy array')
    if reference_data.ndim != 2:
        raise ValueError('data arrays must be two-dimensional')
    if reference_data.shape != moving_data.shape:
        raise ValueError('data arrays must have the same (2-d) shape')

    the_size = reference_data.shape
    box_rough = (0, the_size[0], 0, the_size[1])
    size_min = min(*the_size)
    if size_min > 2000:
        decimation = (8, 8)
    elif size_min > 1000:
        decimation = (4, 4)
    elif size_min > 500:
        decimation = (2, 2)
    else:
        decimation = (1, 1)

    match_box = (min(25, the_size[0]), min(25, the_size[1]))
    deviation = (min(15, the_size[0]), min(15, the_size[1]))

    return _single_step_grid(
        reference_data, None, the_size,
        moving_data, None, the_size,
        box_rough, box_rough,
        match_box_size=match_box, moving_deviation=deviation, decimation=decimation)
