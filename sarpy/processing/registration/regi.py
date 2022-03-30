"""
Basic image registration, generally best suited most suited for coherent image
collection. This is mostly based on an approach developed at Sandia and generally
referred to by the name "regi".

The relevant matlab code appears to be authored by
Terry M. Calloway, Sandia National Laboratories, and modified by
Wade Schwartzkopf.
"""

__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"


import logging
from typing import Tuple

import numpy
from scipy.signal import correlate2d

from sarpy.io.complex.base import SICDTypeReader


logger = logging.getLogger(__name__)


# TODO: rename this to something more appropriate, and fix the doc string

# TODO: notes
#  - multi-stage control point estimate - goes coarse grid to progressively finer (stopping criterion?)
#       * the matlab forces you to choose the number of divisions, keeps the box size
#         and grid spacing the same, and steps down the decimation by a factor of 2
#  - "single stage control point estimate" (cpssestimate.m)
#       * does correlation over box of given size, over search area of given size
#       * at each point of grid, we determine the minimum difference of patch (after scaling and all that),
#         by doing a correlation. I am going to change that to maximizing the inner product
#         (this eliminates the need for obtuse preprocessing), and could be coherent or incoherent
#       * this determines grid of points and offsets at each grid point
#   - matlab appears to use box size of 25 x 25, grid spacing of 15 x 15, search area of 15 x 15,
#     and decimation that steps down by a factor of 2.
#   - the end product is really this final collection of grid points and offsets at each grid point
#     which is then used to formulate a warp transform.
#   - I will worry about the warp transform later, and just focus on point information for now
#   - Note that there is NO ROTATION here


def _max_correlation_step(reference_array, moving_array):
    """
    Find the best match location of the moving array inside the reference array.

    Parameters
    ----------
    reference_array : numpy.ndarray
    moving_array : numpy.ndarray

    Returns
    -------
    best_location : None|numpy.ndarray
        Will return `None` if there is no information, i.e. the reference patch
        or moving patch is all 0. Otherwise, this will be a numpy array
        `[row, column]` of the location of highest correlation, determined via
        :func:`numpy.argmax`.
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
        return None

    moving_array = numpy.sqrt(numpy.abs(moving_array))
    if numpy.all(moving_array == 0):
        return None

    kernel = numpy.ones(reference_array.shape, dtype='float32')

    # now, find the best match
    match_values = correlate2d(reference_array, moving_array, mode='valid')
    norm_values = correlate2d(kernel, moving_array*moving_array, mode='valid')
    mask = (norm_values > 0)

    # reduce to dot product of the unit vectors, and we pick the best match
    match_values[mask] /= numpy.sqrt(norm_values[mask])

    if (moving_array.shape[0] % 2) == 0 or (moving_array.shape[1] % 2) == 0:
        shift = numpy.zeros((2, ), dtype='float64')
    else:
        shift = numpy.zeros((2, ), dtype='int64')
    shift[0] = 0.5*(moving_array.shape[0] - 1)
    shift[1] = 0.5*(moving_array.shape[1] - 1)
    return shift + numpy.unravel_index(numpy.argmax(match_values), match_values.shape)


def _single_step_reader(
        reference_reader, reference_index, moving_reader, moving_index, reference_location, moving_location,
        match_box_size=(25, 25), moving_deviation=(15, 15), decimation=(1, 1)):
    """
    Perform a single step of the reference search by finding the best matching
    location at given size and scale.

    Parameters
    ----------
    reference_reader : SICDTypeReader
    reference_index : int
    moving_reader : SICDTypeReader
    moving_index : int
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
    """

    ref_size = reference_reader.get_data_size_as_tuple()[reference_index]
    moving_size = moving_reader.get_data_size_as_tuple()[moving_index]

    # TODO: this check should be moved up a level?
    if match_box_size[0] > 0.3*ref_size[0] or match_box_size[1] > 0.3*ref_size[1]:
        raise ValueError(
            'The size of the match box ({}) is too close\n\t'
            'to the size of the reference image ({})'.format(match_box_size, ref_size))
    if match_box_size[0] > 0.3*moving_size[0] or match_box_size[1] > 0.3*moving_size[1]:
        raise ValueError(
            'The size of the match box ({}) is too close\n\t'
            'to the size of the moving image ({})'.format(match_box_size, moving_size))

    # NB: we require odd entries here
    match_box_half = (int((match_box_size[0] - 1)/2), int((match_box_size[1] - 1)/2))
    deviation_half = (int((moving_deviation[0] - 1)/2), int((moving_deviation[1] - 1)/2))
    moving_half = (deviation_half[0] + match_box_half[0], deviation_half[1] + match_box_half[1])

    # fetch the reference image patch, and normalize
    ref_box_start = (
        reference_location[0] - match_box_half[0]*decimation[0],
        reference_location[1] - match_box_half[1]*decimation[1])
    if ref_box_start[0] < 0 or \
            ref_box_start[1] < 0 or \
            ref_box_start[0] > ref_size[0] - match_box_size[0]*decimation[0] + 1 or \
            ref_box_start[1] > ref_size[1] - match_box_size[1]*decimation[1] + 1:
        raise ValueError(
            'Overflows bounds. Cannot proceed with reference image of size {}\n\t'
            'using reference box of size {} and decimation {}\n\t'
            'at reference location {}'.format(ref_size, match_box_size, decimation, reference_location))
    ref_box_end = (
        ref_box_start[0] + match_box_size[0]*decimation[0] + 1,
        ref_box_start[1] + match_box_size[1]*decimation[1] + 1)

    reference_array = reference_reader[
                      ref_box_start[0]:ref_box_end[0]:decimation[0],
                      ref_box_start[1]:ref_box_end[1]:decimation[1],
                      reference_index]

    # vet the suggested moving location
    mov_loc_temp = [moving_location[0], moving_location[1]]
    for i in [0, 1]:
        if mov_loc_temp[i] < moving_half[i]*decimation[i]:
            mov_loc_temp[i] = moving_half[i]*decimation[i]
        elif mov_loc_temp[i] > moving_size[i] - moving_half[i]*decimation[i] + 1:
            mov_loc_temp[i] = moving_size[i] - moving_half[i]*decimation[i] + 1
    # fetch the moving patch area over which we will optimize
    moving_box_start = (
        mov_loc_temp[0] - moving_half[0]*decimation[0],
        mov_loc_temp[1] - moving_half[1]*decimation[1])
    moving_box_end = (
        mov_loc_temp[0] + moving_half[0]*decimation[0] + 1,
        mov_loc_temp[1] + moving_half[1]*decimation[1] + 1)

    moving_array = moving_reader[
                   moving_box_start[0]:moving_box_end[0]:decimation[0],
                   moving_box_start[1]:moving_box_end[1]:decimation[1],
                   moving_index]
    best_temp_location = _max_correlation_step(reference_array, moving_array)

    if best_temp_location is None:
        return None

    return (
        (best_temp_location[0])*decimation[0] + mov_loc_temp[0],
        (best_temp_location[1])*decimation[1] + mov_loc_temp[1])
