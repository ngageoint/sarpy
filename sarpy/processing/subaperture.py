# -*- coding: utf-8 -*-
"""
Sub-aperture processing methods.
"""

import logging
import os
from typing import Union, Generator

import numpy

from sarpy.compliance import int_func, integer_types
from sarpy.processing.fft_base import FFTCalculator, fft, ifft, fftshift
from sarpy.io.general.slice_parsing import validate_slice, validate_slice_int
from sarpy.io.product.sidd_structure_creation import create_sidd_structure
from sarpy.io.product.sidd import SIDDWriter
from sarpy.processing.ortho_rectify import OrthorectificationHelper, OrthorectificationIterator


####################
# Module variables providing default values
_METHOD_VALUES = ('NORMAL', 'FULL', 'MINIMAL')

__author__ = 'Thomas McCullough'
__classification__ = "UNCLASSIFIED"


def frame_definition(array_size, frame_count=9, aperture_fraction=0.2, fill=1, method='FULL'):
    """
    Get the frame definition along the desired axis for subaperture processing.

    Parameters
    ----------
    array_size : int
        The size of the given array.
    frame_count : int
        The number of frames to calculate.
    aperture_fraction : float
        The relative size of each aperture window.
    fill : float|int
        The fft fill value.
    method : str
        The subaperture processing method, which must be one of
        `('NORMAL', 'FULL', 'MINIMAL')`.

    Returns
    -------
    List[Tuple[int, int]], int
        The frame definition and individual output resolution.
    """

    method = method.upper()
    if method not in _METHOD_VALUES:
        raise ValueError('method must be one of {}, got {}'.format(_METHOD_VALUES, method))

    fill = float(fill)
    if fill < 0.9999999:
        raise ValueError('fill must be at least 1.0, got {}'.format(fill))

    # determine our functional array and processing sizes
    functional_array_size = array_size/fill
    left_edge = int_func(numpy.round(0.5*(array_size - functional_array_size)))
    processing_size = array_size - 2*left_edge
    # determine the (static) size of each sub-aperture
    subaperture_size = int_func(numpy.ceil(aperture_fraction*processing_size))
    # determine the step size
    step = 0 if frame_count == 1 else \
        int_func(numpy.floor((processing_size - subaperture_size)/float(frame_count-1)))

    if method == 'NORMAL':
        output_resolution = int_func(numpy.ceil(aperture_fraction*array_size))
    elif method == 'FULL':
        output_resolution = array_size
    elif method == 'MINIMAL':
        output_resolution = int_func(numpy.ceil(processing_size/float(frame_count)))
    else:
        raise ValueError('Got unhandled method {}'.format(method))

    frames = []
    start_offset = left_edge
    for i in range(frame_count):
        frames.append((start_offset, start_offset + subaperture_size))
        start_offset += step

    return frames, output_resolution


#####################################
# The sub-aperture processing methods

def _validate_input(array):
    # type: (numpy.ndarray) -> numpy.ndarray
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a numpy array. Got type {}'.format(type(array)))
    if not numpy.iscomplexobj(array):
        raise ValueError('array must be a complex array, got dtype {}'.format(array.dtype))
    if array.ndim != 2:
        raise ValueError('array must be two dimensional. Got shape {}'.format(array.shape))
    return array


def _validate_dimension(dimension):
    # type: (int) -> int
    dimension = int(dimension)
    if dimension not in (0, 1):
        raise ValueError('dimension must be 0 or 1, got {}'.format(dimension))
    return dimension


def subaperture_processing_array(array, aperture_indices, output_resolution, dimension=0):
    """
    Perform the sub-aperture processing on the given complex array data.

    Parameters
    ----------
    array : numpy.ndarray
        The complex array data. Dimension other than 2 is not supported.
    aperture_indices : Tuple[int, int]
        The start/stop indices for the subaperture processing.
    output_resolution : int
        The output resolution parameter.
    dimension : int
        The dimension along which to perform the sub-aperture processing. Must be
        one of 0 or 1.

    Returns
    -------
    numpy.ndarray
    """

    array = _validate_input(array)
    dimension = _validate_dimension(dimension)

    return subaperture_processing_phase_history(
        fftshift(fft(array, axis=dimension), axes=dimension),
        aperture_indices, output_resolution, dimension=dimension)


def subaperture_processing_phase_history(phase_array, aperture_indices, output_resolution, dimension=0):
    """
    Perform the sub-aperture processing on the given complex phase history data.

    Parameters
    ----------
    phase_array : numpy.ndarray
        The complex array data. Dimension other than 2 is not supported.
    aperture_indices : Tuple[int, int]
        The start/stop indices for the subaperture processing.
    output_resolution : int
        The output resolution parameter.
    dimension : int
        The dimension along which to perform the sub-aperture processing. Must be
        one of 0 or 1.

    Returns
    -------
    numpy.ndarray
    """

    phase_array = _validate_input(phase_array)
    dimension = _validate_dimension(dimension)

    if dimension == 0:
        return ifft(phase_array[aperture_indices[0]:aperture_indices[1], :], axis=0, n=output_resolution)
    else:
        return ifft(phase_array[:, aperture_indices[0]:aperture_indices[1]], axis=1, n=output_resolution)


class SubapertureCalculator(FFTCalculator):
    """
    Class for performing sub-aperture processing from a reader instance.

    It is important to note that full resolution is required for along the
    processing dimension, so sub-sampling along the processing dimension does
    not decrease the amount of data which must be fetched.
    """

    __slots__ = ('_frame_count', '_aperture_fraction', '_method', '_frame_definition')

    def __init__(self, reader, dimension=0, index=0, block_size=10,
                 frame_count=9, aperture_fraction=0.2, method='FULL'):
        """

        Parameters
        ----------
        reader : str|BaseReader
            Input file path or reader object, which must be of sicd type.
        dimension : int
            The dimension over which to split the sub-aperture.
        index : int
            The sicd index to use.
        block_size : int
            The approximate processing block size to fetch, given in MB. The
            minimum value for use here will be 1.
        frame_count : int
            The number of frames to calculate.
        aperture_fraction : float
            The relative size of each aperture window.
        method : str
            The subaperture processing method, which must be one of
            `('NORMAL', 'FULL', 'MINIMAL')`.
        """

        self._frame_count = 9
        self._aperture_fraction = 0.2
        self._method = 'FULL'
        self._frame_definition = None
        super(SubapertureCalculator, self).__init__(
            reader, dimension=dimension, index=index, block_size=block_size)

        self.frame_count = frame_count
        self.aperture_fraction = aperture_fraction
        self.method = method

    @property
    def frame_count(self):
        """
        int: The frame count.
        """

        return self._frame_count

    @frame_count.setter
    def frame_count(self, value):
        value = int_func(value)
        if value < 1:
            raise ValueError('frame_count must be a positive integer.')
        self._frame_count = value

    @property
    def aperture_fraction(self):
        """
        float: The relative aperture fraction size.
        """
        return self._aperture_fraction

    @aperture_fraction.setter
    def aperture_fraction(self, value):
        value = float(value)
        if not (0 < value < 1):
            raise ValueError(
                'aperture_fraction must be in the range (0, 1), got {}'.format(value))
        self._aperture_fraction = value

    @property
    def method(self):
        """
        str: The subaperture method.
        """

        return self._method

    @method.setter
    def method(self, value):
        value = value.upper()
        if value not in _METHOD_VALUES:
            raise ValueError('method must be one of {}, got {}'.format(_METHOD_VALUES, value))
        self._method = value

    def _parse_frame_argument(self, the_frame):
        if the_frame is None:
            return numpy.arange(self.frame_count)
        elif isinstance(the_frame, slice):
            the_frame = validate_slice(the_frame, self.frame_count)
            return numpy.arange(the_frame.start, the_frame.stop, the_frame.step)
        elif isinstance(the_frame, integer_types):
            return validate_slice_int(the_frame, self.frame_count)
        elif isinstance(the_frame, (list, tuple)):
            return self._parse_frame_argument(numpy.array(the_frame))
        elif isinstance(the_frame, numpy.ndarray):
            if not issubclass(the_frame.dtype.type, numpy.integer):
                raise ValueError(
                    'The last slice dimension was a numpy array of unsupported non-integer '
                    'dtype {}'.format(the_frame.dtype))
            if the_frame.ndim != 1:
                raise ValueError(
                    'The last slice dimension was a numpy array which was not one-dimensional, '
                    'which is of unsupported.')
            out = the_frame.copy()
            for i, entry in enumerate(out):
                if (entry <= -self.frame_count) or (entry >= self.frame_count):
                    raise ValueError(
                        'The last slice dimension was a numpy array, and entry {} has '
                        'value {}, which is not sensible for the '
                        'bound {}'.format(i, entry, self.frame_count))
                if entry < 0:
                    out[i] += self.frame_count
            return out
        else:
            raise TypeError(
                'The final slice dimension is of unsupported type {}'.format(type(the_frame)))

    def _parse_slicing(self, item):
        row_range, col_range, the_frame = super(SubapertureCalculator, self)._parse_slicing(item)
        return row_range, col_range, self._parse_frame_argument(the_frame)

    def subaperture_generator(self, row_range, col_range, frames=None):
        # type: (tuple, tuple, Union[None, int, list, tuple, numpy.ndarray]) -> Generator[numpy.ndarray]
        """
        Supplies a generator for the given row and column ranges and frames collection.
        **Note that this IGNORES the block_size parameter in fetching, and fetches the
        entire required block.**

        The full resolution data in the processing dimension is required, even if
        down-sampled by the row_range or col_range parameter.

        Parameters
        ----------
        row_range : Tuple[int, int, int]
            The row range.
        col_range : Tuple[int, int, int]
            The column range.
        frames : None|int|list|tuple|numpy.ndarray
            The frame or frame collection.

        Returns
        -------
        Generator[numpy.ndarray]
        """

        def get_dimension_details(the_range):
            the_snip = -1 if the_range[2] < 0 else 1
            t_full_range = (the_range[0], the_range[1], the_snip)
            t_full_size = the_range[1] - the_range[0]
            t_step = abs(the_range[2])
            return t_full_range, t_full_size, t_step

        if self._fill is None:
            raise ValueError('Unable to proceed unless the index and dimension are set.')

        frames = self._parse_frame_argument(frames)
        if isinstance(frames, integer_types):
            frames = [frames, ]

        if self.dimension == 0:
            # determine the full resolution block of data to fetch
            this_row_range, full_size, step = get_dimension_details(row_range)
            # fetch the necessary data block
            data = self.reader[
                   this_row_range[0]:this_row_range[1]:this_row_range[2], col_range[0]:col_range[1]:col_range[2], self.index]
        else:
            # determine the full resolution block of data to fetch
            this_col_range, full_size, step = get_dimension_details(col_range)
            # fetch the necessary data block, and transform to phase space
            data = self.reader[
                   row_range[0]:row_range[1]:row_range[2], this_col_range[0]:this_col_range[1]:this_col_range[2], self.index]
        # handle any nonsense data as 0
        data[~numpy.isfinite(data)] = 0
        # transform the data to phase space
        data = fftshift(fft(data, axis=self.dimension), axes=self.dimension)
        # define our frame collection
        frame_collection, output_resolution = frame_definition(
            full_size, frame_count=self.frame_count, aperture_fraction=self.aperture_fraction,
            fill=self.fill, method=self.method)
        # iterate over frames and generate the results
        for frame_index in frames:
            frame_def = frame_collection[int_func(frame_index)]
            this_subap_data = subaperture_processing_phase_history(
                data, frame_def, output_resolution=output_resolution, dimension=self.dimension)
            if step == 1:
                yield this_subap_data
            elif self.dimension == 0:
                yield this_subap_data[::step, :]
            else:
                yield this_subap_data[:, ::step]

    def _prepare_output(self, row_range, col_range, frames=None):
        row_count = int_func((row_range[1] - row_range[0]) / float(row_range[2]))
        col_count = int_func((col_range[1] - col_range[0]) / float(col_range[2]))
        if frames is None or len(frames) == 1:
            out_size = (row_count, col_count)
        else:
            out_size = (row_count, col_count, len(frames))
        return numpy.zeros(out_size, dtype=numpy.complex64)

    def __getitem__(self, item):
        """
        Fetches the csi data based on the input slice. Slicing in the final
        dimension using an integer, slice, or integer array is supported. Note
        that this could easily be memory intensive, and should be used with
        some care.

        Parameters
        ----------
        item

        Returns
        -------
        numpy.ndarray
        """

        if self._fill is None:
            raise ValueError('Unable to proceed unless the index and dimension are set.')

        # parse the slicing to ensure consistent structure
        row_range, col_range, frames = self._parse_slicing(item)
        if isinstance(frames, integer_types):
            frames = [frames, ]

        if self.dimension == 0:
            column_block_size = self.get_fetch_block_size(row_range[0], row_range[1])
            # get our block definitions
            column_blocks, result_blocks = self.extract_blocks(col_range, column_block_size)
            if column_blocks == 1 and len(frames) == 1:
                # no need to prepare output, which will take twice the memory, so just return
                out = self.subaperture_generator(row_range, col_range, frames).__next__()
            else:
                out = self._prepare_output(row_range, col_range, frames=frames)
                for this_column_range, result_range in zip(column_blocks, result_blocks):
                    generator = self.subaperture_generator(row_range, this_column_range, frames)
                    if len(frames) == 1:
                        out[:, result_range[0]:result_range[1]] = generator.__next__()
                    else:
                        for i, data in enumerate(generator):
                            out[:, result_range[0]:result_range[1], i] = data
        else:
            row_block_size = self.get_fetch_block_size(col_range[0], col_range[1])
            # get our block definitions
            row_blocks, result_blocks = self.extract_blocks(row_range, row_block_size)
            if row_blocks == 1 and len(frames) == 1:
                out = self.subaperture_generator(row_range, col_range, frames).__next__()
            else:
                out = self._prepare_output(row_range, col_range, frames=frames)
                for this_row_range, result_range in zip(row_blocks, result_blocks):
                    generator = self.subaperture_generator(this_row_range, col_range, frames)
                    if len(frames) == 1:
                        out[result_range[0]:result_range[1], :] = generator.__next__()
                    else:
                        for i, data in enumerate(generator):
                            out[result_range[0]:result_range[1], :, i] = data
        return out


class SubapertureOrthoIterator(OrthorectificationIterator):
    """
    An iterator class for the ortho-rectified subaperture processing.

    Iterating depth first requires the least fetching from the reader once for
    all frames. Otherwise, iterating requires redundantly fetching data once
    for each frame.

    It should be noted that fetching data is not time intensive if working using
    a local file (i.e. on your computer), but it may be if working using some
    kind of network file system.
    """

    __slots__ = ('_depth_first', '_this_frame', '_generator')

    def __init__(self, ortho_helper, calculator, bounds=None, depth_first=True):
        """

        Parameters
        ----------
        ortho_helper : OrthorectificationHelper
        calculator : SubapertureCalculator
        bounds : None|numpy.ndarray|list|tuple
            The pixel bounds of the form `(min row, max row, min col, max col)`.
            This will default to the full image.
        depth_first : bool
            If `True`, by image segment part then frame - this requires the least
            fetching from the reader, once across all frames. Otherwise, iteration
            will proceed by frames and then image segment - this requires more
            fetching from the reader, once per frame.
        """

        self._generator = None
        self._this_frame = None
        self._depth_first = bool(depth_first)

        if not isinstance(calculator, SubapertureCalculator):
            raise TypeError(
                'calculator must be an instance of SubapertureCalculator. Got type {}'.format(type(calculator)))
        super(SubapertureOrthoIterator, self).__init__(ortho_helper, calculator=calculator, bounds=bounds)

    @property
    def calculator(self):
        # type: () -> SubapertureCalculator
        # noinspection PyTypeChecker
        return self._calculator

    def _depth_first_iteration(self):
        if not self._depth_first:
            raise ValueError('Requires depth_first = True')

        # determine our current state
        if self._this_index is None or self._this_frame is None:
            self._this_index = 0
            self._this_frame = 0
        else:
            self._this_frame += 1
            if self._this_frame >= self.calculator.frame_count:
                self._this_index += 1
                self._this_frame = 0
        # at this point, _this_index & _this_frame indicates which entry to return
        if self._this_index >= len(self._iteration_blocks):
            self._this_index = None  # reset the iteration scheme
            self._this_frame = None
            raise StopIteration()

        this_ortho_bounds, this_pixel_bounds = self._get_state_parameters()
        # accommodate for real pixel limits
        this_pixel_bounds = self._ortho_helper.get_real_pixel_bounds(this_pixel_bounds)
        if self._this_frame == 0:
            # set up the iterator from the calculator
            self._generator = self.calculator.subaperture_generator(
                (this_pixel_bounds[0], this_pixel_bounds[1], 1),
                (this_pixel_bounds[2], this_pixel_bounds[3], 1))

        logging.info(
            'Fetching orthorectified coordinate block ({}:{}, {}:{}) of ({}:{}) for frame {}'.format(
                this_ortho_bounds[0] - self.ortho_bounds[0], this_ortho_bounds[1] - self.ortho_bounds[0],
                this_ortho_bounds[2] - self.ortho_bounds[2], this_ortho_bounds[3] - self.ortho_bounds[2],
                self.ortho_bounds[1] - self.ortho_bounds[0], self.ortho_bounds[3] - self.ortho_bounds[2],
                self._this_frame))
        data = self._generator.__next__()
        ortho_data = self._get_orthorectified_version(this_ortho_bounds, this_pixel_bounds,data)
        start_indices = (this_ortho_bounds[0] - self.ortho_bounds[0],
                         this_ortho_bounds[2] - self.ortho_bounds[2])
        return ortho_data, start_indices, self._this_frame

    def _frame_first_iteration(self):
        if self._depth_first:
            raise ValueError('Requires depth_first = False')

        # determine our current state
        if self._this_index is None or self._this_frame is None:
            self._this_index = 0
            self._this_frame = 0
        else:
            self._this_index += 1
            if self._this_index >= len(self._iteration_blocks):
                self._this_frame += 1
                self._this_index = 0

        # at this point, _this_index & _this_frame indicates which entry to return
        if self._this_frame >= self.calculator.frame_count:
            raise StopIteration()

        # calculate our result
        this_ortho_bounds, this_pixel_bounds = self._get_state_parameters()
        # accommodate for real pixel limits
        this_pixel_bounds = self._ortho_helper.get_real_pixel_bounds(this_pixel_bounds)
        logging.info(
            'Fetching orthorectified coordinate block ({}:{}, {}:{}) of ({}:{}) for frame {}'.format(
                this_ortho_bounds[0] - self.ortho_bounds[0], this_ortho_bounds[1] - self.ortho_bounds[0],
                this_ortho_bounds[2] - self.ortho_bounds[2], this_ortho_bounds[3] - self.ortho_bounds[2],
                self.ortho_bounds[1] - self.ortho_bounds[0], self.ortho_bounds[3] - self.ortho_bounds[2],
                self._this_frame))


        data = self.calculator[
               this_pixel_bounds[0]:this_pixel_bounds[1],
               this_pixel_bounds[2]:this_pixel_bounds[3],
               self._this_frame]
        ortho_data = self._get_orthorectified_version(this_ortho_bounds, this_pixel_bounds,data)
        start_indices = (this_ortho_bounds[0] - self.ortho_bounds[0],
                         this_ortho_bounds[2] - self.ortho_bounds[2])
        return ortho_data, start_indices, self._this_frame

    def __next__(self):
        """
        Get the next iteration of ortho-rectified data.

        Returns
        -------
        numpy.ndarray, Tuple[int, int], int
            The data and the (normalized) indices (start_row, start_col) for this section of data, relative
            to overall output shape, and then the frame index.
        """

        # NB: this is the Python 3 pattern for iteration

        if self._depth_first:
            return self._depth_first_iteration()
        else:
            return self._frame_first_iteration()

    def next(self):
        """
        Get the next iteration of ortho-rectified data.

        Returns
        -------
        numpy.ndarray, Tuple[int, int], int
            The data and the (normalized) indices (start_row, start_col) for this section of data, relative
            to overall output shape, and then the frame index.
        """

        # NB: this is the Python 2 pattern for iteration
        return self.__next__()
