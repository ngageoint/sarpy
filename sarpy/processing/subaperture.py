# -*- coding: utf-8 -*-
"""
Sub-aperture processing methods.
"""

import logging
import os
from typing import Generator

import numpy

from sarpy.compliance import int_func, integer_types
from sarpy.processing.fft_base import FFTCalculator, fft, ifft, fftshift
from sarpy.io.general.slice_parsing import validate_slice, validate_slice_int
from sarpy.io.product.sidd_creation_utils import create_sidd
from sarpy.io.product.sidd import SIDDWriter
from sarpy.processing.ortho_rectify import OrthorectificationHelper
from sarpy.visualization.remap import amplitude_to_density, clip_cast


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

    def __init__(self, reader, dimension=0, index=0, block_size=50,
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

        def prepare_output():
            row_count = int_func((row_range[1] - row_range[0]) / float(row_range[2]))
            col_count = int_func((col_range[1] - col_range[0]) / float(col_range[2]))
            out_size = (row_count, col_count) if len(frames) == 1 else (row_count, col_count, len(frames))
            return numpy.zeros(out_size, dtype=numpy.complex64)

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
                out = prepare_output()
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
                out = prepare_output()
                for this_row_range, result_range in zip(row_blocks, result_blocks):
                    generator = self.subaperture_generator(this_row_range, col_range, frames)
                    if len(frames) == 1:
                        out[result_range[0]:result_range[1], :] = generator.__next__()
                    else:
                        for i, data in enumerate(generator):
                            out[result_range[0]:result_range[1], :, i] = data
        return out


def create_dynamic_image_sidd(
        ortho_helper, output_directory, output_file=None, dimension=0, block_size=50,
        bounds=None, frame_count=9, aperture_fraction=0.2, method='FULL', version=2):
    """
    Create a SIDD version of a Dynamic Image (Sub-Aperture Stack) from a SICD type reader.

    Parameters
    ----------
    ortho_helper : OrthorectificationHelper
        The ortho-rectification helper object.
    output_directory : str
        The output directory for the given file.
    output_file : None|str
        The file name, this will default to a sensible value.
    dimension : int
        The dimension over which to split the sub-aperture.
    block_size : int
        The approximate processing block size to fetch, given in MB. The
        minimum value for use here will be 1.
    bounds : None|numpy.ndarray|list|tuple
        The sicd pixel bounds of the form `(min row, max row, min col, max col)`.
        This will default to the full image.
    frame_count : int
        The number of frames to calculate.
    aperture_fraction : float
        The relative size of each aperture window.
    method : str
        The subaperture processing method, which must be one of
        `('NORMAL', 'FULL', 'MINIMAL')`.
    version : int
        The SIDD version to use, must be one of 1 or 2.

    Returns
    -------
    None
    """

    if not os.path.isdir(output_directory):
        raise IOError('output_directory {} does not exist or is not a directory'.format(output_directory))

    if not isinstance(ortho_helper, OrthorectificationHelper):
        raise TypeError(
            'ortho_helper is required to be an instance of OrthorectificationHelper, '
            'got type {}'.format(type(ortho_helper)))

    def get_pixel_data(temp_pixel_bounds):
        this_pixel_row_range = (temp_pixel_bounds[0], temp_pixel_bounds[1], 1)
        this_pixel_col_range = (this_pixel_bounds[2], temp_pixel_bounds[3], 1)
        this_row_array = numpy.arange(this_pixel_row_range[0], this_pixel_row_range[1], 1)
        this_col_array = numpy.arange(this_pixel_col_range[0], this_pixel_col_range[1], 1)
        return this_pixel_row_range, this_pixel_col_range, this_row_array, this_col_array

    def get_orthorectified_version(these_ortho_bounds, this_row_array, this_col_array, this_subap_data):
        return clip_cast(
            amplitude_to_density(
                ortho_helper.get_orthorectified_from_array(these_ortho_bounds, this_row_array, this_col_array, this_subap_data),
                data_mean=the_mean),
            dtype='uint8')

    def log_progress(t_ortho_bounds, t_index):
        logging.info('Writing pixels ({}:{}, {}:{}) of ({}, {}) for index '
                     '{} of {}'.format(t_ortho_bounds[0]-ortho_bounds[0], t_ortho_bounds[1]-ortho_bounds[0],
                                       t_ortho_bounds[2] - ortho_bounds[2], t_ortho_bounds[3] - ortho_bounds[2],
                                       ortho_bounds[1] - ortho_bounds[0], ortho_bounds[3] - ortho_bounds[2],
                                       t_index+1, subap_calculator.frame_count))

    reader = ortho_helper.reader
    index = ortho_helper.index

    # construct the subaperture calculator class
    subap_calculator = SubapertureCalculator(
        reader, dimension=dimension, index=index, block_size=block_size,
        frame_count=frame_count, aperture_fraction=aperture_fraction, method=method)
    # validate the bounds
    if bounds is None:
        bounds = (0, subap_calculator.data_size[0], 0, subap_calculator.data_size[1])
    bounds, pixel_rectangle = ortho_helper.bounds_to_rectangle(bounds)
    # get the corresponding prtho bounds
    ortho_bounds = ortho_helper.get_orthorectification_bounds_from_pixel_object(pixel_rectangle)
    # Extract the mean of the data magnitude - for global remap usage
    the_mean = subap_calculator.get_data_mean_magnitude(bounds)

    # create the sidd structure
    sidd_structure = create_sidd(
        ortho_helper, ortho_bounds,
        product_class='Dynamic Image', pixel_type='MONO8I', version=version)
    # set suggested name
    sidd_structure._NITF = {
        'SUGGESTED_NAME': subap_calculator.sicd.get_suggested_name(subap_calculator.index)+'__DI', }
    the_sidds = []
    for i in range(subap_calculator.frame_count):
        this_sidd = sidd_structure.copy()
        this_sidd.ProductCreation.ProductType = 'Frame {}'.format(i+1)
        the_sidds.append(this_sidd)
    # create the sidd writer
    if output_file is None:
        # noinspection PyProtectedMember
        full_filename = os.path.join(output_directory, sidd_structure._NITF['SUGGESTED_NAME']+'.nitf')
    else:
        full_filename = os.path.join(output_directory, output_file)
    if os.path.exists(os.path.expanduser(full_filename)):
        raise IOError('File {} already exists.'.format(full_filename))
    writer = SIDDWriter(full_filename, the_sidds, subap_calculator.sicd)

    if subap_calculator.dimension == 0:
        # we are using the full resolution row data
        # determine the orthorectified blocks to use
        column_block_size = subap_calculator.get_fetch_block_size(ortho_bounds[0], ortho_bounds[1])
        ortho_column_blocks, ortho_result_blocks = subap_calculator.extract_blocks(
            (ortho_bounds[2], ortho_bounds[3], 1), column_block_size)

        for this_column_range, result_range in zip(ortho_column_blocks, ortho_result_blocks):
            # determine the corresponding pixel ranges to encompass these values
            this_ortho_bounds, this_pixel_bounds = ortho_helper.extract_pixel_bounds(
                (ortho_bounds[0], ortho_bounds[1], this_column_range[0], this_column_range[1]))
            # accommodate for real pixel limits
            this_pixel_bounds = ortho_helper.get_real_pixel_bounds(this_pixel_bounds)
            # create the subaperture generator
            pixel_row_range, pixel_col_range, row_array, col_array = get_pixel_data(this_pixel_bounds)
            start_indices = (this_ortho_bounds[0] - ortho_bounds[0],
                             this_ortho_bounds[2] - ortho_bounds[2])
            for i, subap_data in enumerate(subap_calculator.subaperture_generator(pixel_row_range, pixel_col_range)):
                log_progress(this_ortho_bounds, i)
                ortho_subap_data = get_orthorectified_version(this_ortho_bounds, row_array, col_array, subap_data)
                writer(ortho_subap_data, start_indices=start_indices, index=i)
    else:
        row_block_size = subap_calculator.get_fetch_block_size(ortho_bounds[2], ortho_bounds[3])
        ortho_row_blocks, ortho_result_blocks = subap_calculator.extract_blocks(
            (ortho_bounds[0], ortho_bounds[1], 1), row_block_size)

        for this_row_range, result_range in zip(ortho_row_blocks, ortho_result_blocks):
            # determine the corresponding pixel ranges to encompass these values
            this_ortho_bounds, this_pixel_bounds = ortho_helper.extract_pixel_bounds(
                (this_row_range[0], this_row_range[1], ortho_bounds[2], ortho_bounds[3]))
            # accommodate for real pixel limits
            this_pixel_bounds = ortho_helper.get_real_pixel_bounds(this_pixel_bounds)
            # create the subaperture generator
            pixel_row_range, pixel_col_range, row_array, col_array = get_pixel_data(this_pixel_bounds)
            start_indices = (this_ortho_bounds[0] - ortho_bounds[0],
                             this_ortho_bounds[2] - ortho_bounds[2])
            for i, subap_data in enumerate(subap_calculator.subaperture_generator(pixel_row_range, pixel_col_range)):
                log_progress(this_ortho_bounds, i)
                ortho_subap_data = get_orthorectified_version(this_ortho_bounds, row_array, col_array, subap_data)
                writer(ortho_subap_data, start_indices=start_indices, index=i)
