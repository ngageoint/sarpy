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
from typing import Tuple

import numpy
from scipy.signal import correlate2d

from sarpy.io.general.base import BaseReader
from sarpy.io.complex.base import SICDTypeReader


logger = logging.getLogger(__name__)

# TODO: notes
#  - multi-stage control point estimate - goes coarse grid to progressively finer (stopping criterion?)
#       * the matlab forces you to choose the number of divisions, keeps the box size
#         and grid spacing the same, and steps down the decimation by a factor of 2
#  - "single stage control point estimate" (cpssestimate.m)
#       * does correlation over box of given size, over search area of given size
#       * at each point of grid, we determine the minimum difference of patch (after scaling and all that),
#         by doing a correlation.
#       * this determines grid of points and offsets at each grid point
#   - matlab appears to use box size of 25 x 25, grid spacing of 15 x 15, search area of 15 x 15,
#     and decimation that steps down by a factor of 2.
#   - the end product is really this final collection of grid points and offsets at each grid point
#     which is then used to formulate a warp transform.
#   - I will worry about the warp transform later, and just focus on point information for now
#   - Note that there is NO ROTATION here


def _validate_reader(the_reader, the_index):
    """
    Validate the array or reader for registration efforts.

    Parameters
    ----------
    the_reader : SICDTypeReader|numpy.ndarray
    the_index : None|int

    Returns
    -------
    the_index : None|int
    the_size : Tuple[int, int]
        The size of the input
    """

    if isinstance(the_reader, SICDTypeReader):
        if the_index is None:
            the_index = 0
        the_size = the_reader.get_data_size_as_tuple()[the_index]
    elif isinstance(the_reader, numpy.ndarray):
        if the_reader.ndim != 2:
            raise ValueError('Provided image array must be two-dimensional')
        the_index = None
        the_size = the_reader.shape
    else:
        raise ValueError('Image must be a reader instance or an array')
    return the_reader, the_index, the_size


def _validate_match_parameters(reference_size, moving_size, match_box_size, moving_deviation):
    if not ((match_box_size[0] % 2) == 1 and match_box_size[0] > 1 and
            (match_box_size[1] % 2) == 1 and match_box_size[1] > 1):
        raise ValueError('The match box size must have both odd entries greater than 1')

    if not ((moving_deviation[0] % 2) == 1 and moving_deviation[0] > 1 and
            (moving_deviation[1] % 2) == 1 and moving_deviation[1] > 1):
        raise ValueError('The match box size must have both odd entries greater than 1')

    if match_box_size[0] > 0.3*reference_size[0] or match_box_size[1] > 0.3*reference_size[1]:
        raise ValueError(
            'The size of the match box ({}) is too close\n\t'
            'to the size of the reference image ({})'.format(match_box_size, reference_size))
    if match_box_size[0] > 0.3*moving_size[0] or match_box_size[1] > 0.3*moving_size[1]:
        raise ValueError(
            'The size of the match box ({}) is too close\n\t'
            'to the size of the moving image ({})'.format(match_box_size, moving_size))


def _subpixel_shift(values):
    """
    This is simplified port of the SAR toolbox matlab function fin_minms. This
    uses data from an empirical fit derived from unknown origins to estimate where
    the "real" minimum occured.

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

    if (values[0] == values[1] or values[1] == values[2]):
        # no information
        return 0.0
    if values[0] == values[2]:
        # it's symmetric
        return 0.0

    values = (values + numpy.min(values))  # ensure that everything is positive by shifting up

    # Here are the verbatim matlab comments - it's not clear to me why this is assumed:
    # Algorithm:
    # xm = min
    # r = (rmsmid-rmsmin)/(rmsmax-rmsmin)
    # empirical fit (r,xm) resembles arc of circle centered at xc=-11/8
    # xm = 2(xc-3/8) + sqrt(4(xc-3/8)**2 - (r-1)((r-1)-2(xc-1)))
    # empirical fit (r,xm) resembles arc of circle centered at xc=-6/8
    # xm = 1-xc - sqrt(0.125 + 2(0.25-xc)**2 - (x-xc)**2)

    nsr = 0.5
    noise = nsr*numpy.max(values)
    rms = numpy.sqrt(y - noise)
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


def _max_correlation_step(reference_array, moving_array, do_subpixel=False):
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
    maximum_correlation : float
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
        reference_data, reference_index, reference_size,
        moving_data, moving_index, moving_size,
        reference_location, moving_location,
        match_box_size=(25, 25), moving_deviation=(15, 15), decimation=(1, 1)):
    """
    Perform a single step of the reference search by finding the best matching
    location at given size and scale.

    Parameters
    ----------
    reference_data : SICDTypeReader|numpy.ndarray
    reference_index : None|int
    reference_size : Tuple[int, int]
    moving_data : SICDTypeReader|numpy.ndarray
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

    return ((best_temp_location[0])*decimation[0] + mov_loc_temp[0], (best_temp_location[1])*decimation[1] + mov_loc_temp[1]), \
        maximum_correlation

def _single_step_grid(
        reference_data, reference_index, reference_size,
        moving_data, moving_index, moving_size,
        reference_box_rough, moving_box_rough,
        match_box_size=(25, 25), moving_deviation=(15, 15), decimation=(1, 1)):
    """
    We will determine a series of best matching (small size) patch locations
    between the pixel area of `reference_data` laid out in `reference_box_rough`
    and the pixel area of `moving_data` laid out in `moving_box_rough` - which
    should be very close to the same size.

    Parameters
    ----------
    reference_data : SICDTypeReader|numpy.ndarray
    reference_index : None|int
    reference_size : Tuple[int, int]
    moving_data : SICDTypeReader|numpy.ndarray
    moving_index : None|int
    moving_size : Tuple[int, int]
    reference_box_rough : Tuple[int, int, int, int]
    moving_box_rough : Tuple[int, int, int, int]
    match_box_size : Tuple[int, int]
    moving_deviation : Tuple[int, int]
    decimation : Tuple[int, int]

    Returns
    -------

    """

    pass
