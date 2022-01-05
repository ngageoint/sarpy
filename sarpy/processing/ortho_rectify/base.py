"""
Common ortho-rectification elements
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
import os
from typing import Union, Tuple, List, Any

import numpy

from sarpy.io.complex.converter import open_complex
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.general.slice_parsing import validate_slice_int, validate_slice
from sarpy.io.complex.utils import get_fetch_block_size, extract_blocks

from sarpy.geometry.geocoords import ecf_to_geodetic
from sarpy.visualization.remap import RemapFunction

from .ortho_methods import OrthorectificationHelper

logger = logging.getLogger(__name__)


class FullResolutionFetcher(object):
    """
    This is a base class for provided a simple API for processing schemes where
    full resolution is required along the processing dimension, so sub-sampling
    along the processing dimension does not decrease the amount of data which
    must be fetched.
    """

    __slots__ = (
        '_reader', '_index', '_sicd', '_dimension', '_data_size', '_block_size')

    def __init__(self, reader, dimension=0, index=0, block_size=10):
        """

        Parameters
        ----------
        reader : str|SICDTypeReader
            Input file path or reader object, which must be of sicd type.
        dimension : int
            The dimension over which to split the sub-aperture.
        index : int
            The sicd index to use.
        block_size : None|int|float
            The approximate processing block size to fetch, given in MB. The
            minimum value for use here will be 0.25. `None` represents processing
            as a single block.
        """

        self._index = None  # set explicitly
        self._sicd = None  # set with index setter
        self._dimension = None  # set explicitly
        self._data_size = None  # set with index setter
        self._block_size = None  # set explicitly

        # validate the reader
        if isinstance(reader, str):
            reader = open_complex(reader)
        if not isinstance(reader, SICDTypeReader):
            raise TypeError('reader is required to be a path name for a sicd-type image, '
                            'or an instance of a reader object.')
        self._reader = reader
        # set the other properties
        self.dimension = dimension
        self.index = index
        self.block_size = block_size

    @property
    def reader(self):
        # type: () -> SICDTypeReader
        """
        SICDTypeReader: The reader instance.
        """

        return self._reader

    @property
    def dimension(self):
        # type: () -> int
        """
        int: The dimension along which to perform the color subaperture split.
        """

        return self._dimension

    @dimension.setter
    def dimension(self, value):
        value = int(value)
        if value not in [0, 1]:
            raise ValueError('dimension must be 0 or 1, got {}'.format(value))
        self._dimension = value

    @property
    def data_size(self):
        # type: () -> Tuple[int, int]
        """
        Tuple[int, int]: The data size for the reader at the given index.
        """

        return self._data_size

    @property
    def index(self):
        # type: () -> int
        """
        int: The index of the reader.
        """

        return self._index

    @index.setter
    def index(self, value):
        self._set_index(value)

    def _set_index(self, value):
        value = int(value)
        if value < 0:
            raise ValueError('The index must be a non-negative integer, got {}'.format(value))

        sicds = self.reader.get_sicds_as_tuple()
        if value >= len(sicds):
            raise ValueError('The index must be less than the sicd count.')
        self._index = value
        self._sicd = sicds[value]
        self._data_size = self.reader.get_data_size_as_tuple()[value]

    @property
    def block_size(self):
        # type: () -> float
        """
        None|float: The approximate processing block size in MB, where `None`
        represents processing in a single block.
        """

        return self._block_size

    @block_size.setter
    def block_size(self, value):
        if value is None:
            self._block_size = None
        else:
            value = float(value)
            if value < 0.25:
                value = 0.25
            self._block_size = value

    @property
    def block_size_in_bytes(self):
        # type: () -> Union[None, int]
        """
        None|int: The approximate processing block size in bytes.
        """

        return None if self._block_size is None else int(self._block_size*(2**20))

    @property
    def sicd(self):
        # type: () -> SICDType
        """
        SICDType: The sicd structure.
        """

        return self._sicd

    def _parse_slicing(self, item):
        # type: (Union[None, int, slice, tuple]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Any]

        def parse(entry, dimension):
            bound = self.data_size[dimension]
            if entry is None:
                return 0, bound, 1
            elif isinstance(entry, int):
                entry = validate_slice_int(entry, bound)
                return entry, entry+1, 1
            elif isinstance(entry, slice):
                entry = validate_slice(entry, bound)
                return entry.start, entry.stop, entry.step
            else:
                raise TypeError('No support for slicing using type {}'.format(type(entry)))

        # this input is assumed to come from slice parsing
        if isinstance(item, tuple):
            if len(item) > 3:
                raise ValueError(
                    'Received slice argument {}. We cannot slice '
                    'on more than two dimensions.'.format(item))
            elif len(item) == 3:
                return parse(item[0], 0), parse(item[1], 1), item[2]
            elif len(item) == 2:
                return parse(item[0], 0), parse(item[1], 1), None
            elif len(item) == 1:
                return parse(item[0], 0), parse(None, 1), None
            else:
                return parse(None, 0), parse(None, 1), None
        elif isinstance(item, slice):
            return parse(item, 0), parse(None, 1), None
        elif isinstance(item, int):
            return parse(item, 0), parse(None, 1), None
        else:
            raise TypeError('Slicing using type {} is unsupported'.format(type(item)))

    def get_fetch_block_size(self, start_element, stop_element):
        """
        Gets the fetch block size for the given full resolution section.
        This assumes that the fetched data will be 8 bytes per pixel, in
        accordance with single band complex64 data.

        Parameters
        ----------
        start_element : int
        stop_element : int

        Returns
        -------
        int
        """

        return get_fetch_block_size(start_element, stop_element, self.block_size_in_bytes, bands=1)

    @staticmethod
    def extract_blocks(the_range, index_block_size):
        # type: (Tuple[int, int, int], Union[None, int, float]) -> (List[Tuple[int, int, int]], List[Tuple[int, int]])
        """
        Convert the single range definition into a series of range definitions in
        keeping with fetching of the appropriate block sizes.

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

        return extract_blocks(the_range, index_block_size)

    def _full_row_resolution(self, row_range, col_range):
        # type: (Tuple[int, int, int], Tuple[int, int, int]) -> numpy.ndarray
        """
        Perform the full row resolution data, with any appropriate calculations.

        Parameters
        ----------
        row_range : Tuple[int, int, int]
        col_range : Tuple[int, int, int]

        Returns
        -------
        numpy.ndarray
        """

        # fetch the data and perform the csi calculation
        if row_range[2] not in [1, -1]:
            raise ValueError('The step for row_range must be +/- 1, for full row resolution data.')
        if row_range[1] == -1:
            data = self.reader[
                   row_range[0]::row_range[2],
                   col_range[0]:col_range[1]:col_range[2],
                   self.index]
        else:
            data = self.reader[
                   row_range[0]:row_range[1]:row_range[2],
                   col_range[0]:col_range[1]:col_range[2],
                   self.index]

        if data.ndim < 2:
            data = numpy.reshape(data, (-1, 1))
        # handle nonsense data with zeros
        data[~numpy.isfinite(data)] = 0
        return data

    def _full_column_resolution(self, row_range, col_range):
        # type: (Tuple[int, int, int], Tuple[int, int, int]) -> numpy.ndarray
        """
        Perform the full column resolution data, with any appropriate calculations.

        Parameters
        ----------
        row_range : Tuple[int, int, int]
        col_range : Tuple[int, int, int]

        Returns
        -------
        numpy.ndarray
        """

        # fetch the data and perform the csi calculation
        if col_range[2] not in [1, -1]:
            raise ValueError('The step for col_range must be +/- 1, for full col resolution data.')
        if col_range[1] == -1:
            data = self.reader[
                   row_range[0]:row_range[1]:row_range[2],
                   col_range[0]::col_range[2],
                   self.index]
        else:
            data = self.reader[
                   row_range[0]:row_range[1]:row_range[2],
                   col_range[0]:col_range[1]:col_range[2],
                   self.index]

        if data.ndim < 2:
            data = numpy.reshape(data, (1, -1))
        # handle nonsense data with zeros
        data[~numpy.isfinite(data)] = 0
        return data

    def _prepare_output(self, row_range, col_range):
        """
        Prepare the output workspace for :func:`__getitem__`.

        Parameters
        ----------
        row_range
        col_range

        Returns
        -------
        numpy.ndarray
        """

        row_count = int((row_range[1] - row_range[0]) / float(row_range[2]))
        col_count = int((col_range[1] - col_range[0]) / float(col_range[2]))
        out_size = (row_count, col_count)
        return numpy.zeros(out_size, dtype=numpy.complex64)

    def __getitem__(self, item):
        """
        Fetches the processed data based on the input slice.

        Parameters
        ----------
        item

        Returns
        -------
        numpy.ndarray
        """

        # parse the slicing to ensure consistent structure
        row_range, col_range, _ = self._parse_slicing(item)
        return self.reader[
               row_range[0]:row_range[1]:row_range[2],
               col_range[0]:col_range[1]:col_range[2],
               self.index]


class OrthorectificationIterator(object):
    """
    This provides a generator for an Orthorectification process on a given
    reader/index/(pixel) bounds.
    """

    __slots__ = (
        '_calculator', '_ortho_helper', '_pixel_bounds', '_ortho_bounds',
        '_this_index', '_iteration_blocks', '_remap_function')

    def __init__(
            self, ortho_helper, calculator=None, bounds=None,
            remap_function=None, recalc_remap_globals=False):
        """

        Parameters
        ----------
        ortho_helper : OrthorectificationHelper
            The ortho-rectification helper.
        calculator : None|FullResolutionFetcher
            The FullResolutionFetcher instance. If not provided, then this will
            default to a base FullResolutionFetcher instance - which is only
            useful for a basic detected image.
        bounds : None|numpy.ndarray|list|tuple
            The pixel bounds of the form `(min row, max row, min col, max col)`.
            This will default to the full image.
        remap_function : None|RemapFunction
            The remap function to apply, if desired.
        recalc_remap_globals : bool
            Only applies if a remap function is provided, should we recalculate
            any required global parameters? This will automatically happen if
            they are not already set.
        """

        self._this_index = None
        self._iteration_blocks = None
        self._remap_function = None

        # validate ortho_helper
        if not isinstance(ortho_helper, OrthorectificationHelper):
            raise TypeError(
                'ortho_helper must be an instance of OrthorectificationHelper, got '
                'type {}'.format(type(ortho_helper)))
        self._ortho_helper = ortho_helper

        # validate calculator
        if calculator is None:
            calculator = FullResolutionFetcher(ortho_helper.reader, index=ortho_helper.index, dimension=0)
        if not isinstance(calculator, FullResolutionFetcher):
            raise TypeError(
                'calculator must be an instance of FullResolutionFetcher, got '
                'type {}'.format(type(calculator)))
        self._calculator = calculator

        if ortho_helper.reader.file_name is not None and calculator.reader.file_name is not None and \
                os.path.abspath(ortho_helper.reader.file_name) != os.path.abspath(calculator.reader.file_name):
            raise ValueError(
                'ortho_helper has reader for file {}, while calculator has reader '
                'for file {}'.format(ortho_helper.reader.file_name, calculator.reader.file_name))
        if ortho_helper.index != calculator.index:
            raise ValueError(
                'ortho_helper is using index {}, while calculator is using '
                'index {}'.format(ortho_helper.index, calculator.index))

        # validate the bounds
        if bounds is not None:
            pixel_bounds, pixel_rectangle = ortho_helper.bounds_to_rectangle(bounds)
            # get the corresponding ortho bounds
            ortho_bounds = ortho_helper.get_orthorectification_bounds_from_pixel_object(pixel_rectangle)
        else:
            ortho_bounds = ortho_helper.get_full_ortho_bounds()
            ortho_bounds, nominal_pixel_bounds = ortho_helper.extract_pixel_bounds(ortho_bounds)
            # extract the values - ensure that things are within proper image bounds
            pixel_bounds = ortho_helper.get_real_pixel_bounds(nominal_pixel_bounds)

        # validate remap function
        if remap_function is None or isinstance(remap_function, RemapFunction):
            self._remap_function = remap_function
        else:
            raise TypeError(
                'remap_function is expected to be an instance of RemapFunction, '
                'got type `{}`'.format(type(remap_function)))

        self._pixel_bounds = pixel_bounds
        self._ortho_bounds = ortho_bounds
        self._prepare_state(recalc_remap_globals=recalc_remap_globals)

    @property
    def ortho_helper(self):
        # type: () -> OrthorectificationHelper
        """
        OrthorectificationHelper: The ortho-rectification helper.
        """

        return self._ortho_helper

    @property
    def calculator(self):
        # type: () -> FullResolutionFetcher
        """
        FullResolutionFetcher : The calculator instance.
        """

        return self._calculator

    @property
    def sicd(self):
        """
        SICDType: The sicd structure.
        """

        return self.calculator.sicd

    @property
    def pixel_bounds(self):
        """
        numpy.ndarray : Of the form `(row min, row max, col min, col max)`.
        """

        return self._pixel_bounds

    @property
    def ortho_bounds(self):
        """
        numpy.ndarray : Of the form `(row min, row max, col min, col max)`. Note that
        these are "unnormalized" orthorectified pixel coordinates.
        """

        return self._ortho_bounds

    @property
    def ortho_data_size(self):
        """
        Tuple[int, int] : The size of the overall ortho-rectified output.
        """

        return (
            int(self.ortho_bounds[1] - self.ortho_bounds[0]),
            int(self.ortho_bounds[3] - self.ortho_bounds[2]))

    @property
    def remap_function(self):
        # type: () -> Union[None, RemapFunction]
        """
        None|RemapFunction: The remap function to be applied.
        """

        return self._remap_function

    def get_ecf_image_corners(self):
        """
        The corner points of the overall ortho-rectified output in ECF
        coordinates. The ordering of these points follows the SICD convention.

        Returns
        -------
        numpy.ndarray
        """

        if self.ortho_bounds is None:
            return None
        _, ortho_pixel_corners = self._ortho_helper.bounds_to_rectangle(self.ortho_bounds)
        return self._ortho_helper.proj_helper.ortho_to_ecf(ortho_pixel_corners)

    def get_llh_image_corners(self):
        """
        The corner points of the overall ortho-rectified output in Lat/Lon/HAE
        coordinates. The ordering of these points follows the SICD convention.

        Returns
        -------
        numpy.ndarray
        """

        ecf_corners = self.get_ecf_image_corners()
        if ecf_corners is None:
            return None
        else:
            return ecf_to_geodetic(ecf_corners)

    def _prepare_state(self, recalc_remap_globals=False):
        """
        Prepare the iteration state.

        Returns
        -------
        None
        """

        if self.calculator.dimension == 0:
            column_block_size = self.calculator.get_fetch_block_size(self.ortho_bounds[0], self.ortho_bounds[1])
            self._iteration_blocks, _ = self.calculator.extract_blocks(
                (self.ortho_bounds[2], self.ortho_bounds[3], 1), column_block_size)
        else:
            row_block_size = self.calculator.get_fetch_block_size(self.ortho_bounds[2], self.ortho_bounds[3])
            self._iteration_blocks, _ = self.calculator.extract_blocks(
                (self.ortho_bounds[0], self.ortho_bounds[1], 1), row_block_size)

        if self.remap_function is not None and \
                (recalc_remap_globals or not self.remap_function.are_global_parameters_set):
            self.remap_function.calculate_global_parameters_from_reader(
                self.ortho_helper.reader, index=self.ortho_helper.index, pixel_bounds=self.pixel_bounds)

    @staticmethod
    def _get_ortho_helper(pixel_bounds, this_data):
        """
        Get helper data for ortho-rectification.

        Parameters
        ----------
        pixel_bounds
        this_data

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
        """

        rows_temp = pixel_bounds[1] - pixel_bounds[0]
        if this_data.shape[0] == rows_temp:
            row_array = numpy.arange(pixel_bounds[0], pixel_bounds[1])
        elif this_data.shape[0] == (rows_temp - 1):
            row_array = numpy.arange(pixel_bounds[0], pixel_bounds[1] - 1)
        else:
            raise ValueError('Unhandled data size mismatch {} and {}'.format(this_data.shape, rows_temp))

        cols_temp = pixel_bounds[3] - pixel_bounds[2]
        if this_data.shape[1] == cols_temp:
            col_array = numpy.arange(pixel_bounds[2], pixel_bounds[3])
        elif this_data.shape[1] == (cols_temp - 1):
            col_array = numpy.arange(pixel_bounds[2], pixel_bounds[3] - 1)
        else:
            raise ValueError('Unhandled data size mismatch {} and {}'.format(this_data.shape, cols_temp))
        return row_array, col_array

    def _get_orthorectified_version(self, this_ortho_bounds, pixel_bounds, this_data):
        """
        Get the orthorectified version from the raw values and pixel information.

        Parameters
        ----------
        this_ortho_bounds
        pixel_bounds
        this_data

        Returns
        -------
        numpy.ndarray
        """

        row_array, col_array = self._get_ortho_helper(pixel_bounds, this_data)
        ortho_data = self._ortho_helper.get_orthorectified_from_array(
            this_ortho_bounds, row_array, col_array, this_data)
        if self.remap_function is None:
            return ortho_data
        else:
            return self.remap_function(ortho_data)

    def _get_state_parameters(self, pad=10):
        """
        Gets the pixel information associated with the current state.

        Parameters
        ----------
        pad : int
            Pad the pixel bounds, to accommodate for any edge cases.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
        """

        if self._calculator.dimension == 0:
            this_column_range = self._iteration_blocks[self._this_index]
            # determine the corresponding pixel ranges to encompass these values
            this_ortho_bounds, this_pixel_bounds = self._ortho_helper.extract_pixel_bounds(
                (self.ortho_bounds[0], self.ortho_bounds[1], this_column_range[0], this_column_range[1]))
        else:
            this_row_range = self._iteration_blocks[self._this_index]
            this_ortho_bounds, this_pixel_bounds = self._ortho_helper.extract_pixel_bounds(
                (this_row_range[0], this_row_range[1], self.ortho_bounds[2], self.ortho_bounds[3]))
        
        this_pixel_bounds[0::2] -= pad
        this_pixel_bounds[1::2] += pad
        return this_ortho_bounds, this_pixel_bounds

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get the next iteration of orthorectified data.

        Returns
        -------
        (numpy.ndarray, Tuple[int, int])
            The data and the (normalized) indices (start_row, start_col) for this section of data, relative
            to overall output shape.
        """

        # NB: this is the Python 3 pattern for iteration
        if self._this_index is None:
            self._this_index = 0
        else:
            self._this_index += 1
        # at this point, _this_index indicates which entry to return
        if self._this_index >= len(self._iteration_blocks):
            self._this_index = None  # reset the iteration scheme
            raise StopIteration()

        this_ortho_bounds, this_pixel_bounds = self._get_state_parameters()
        # accommodate for real pixel limits
        this_pixel_bounds = self._ortho_helper.get_real_pixel_bounds(this_pixel_bounds)
        # extract the csi data and ortho-rectify
        logger.info(
            'Fetching orthorectified coordinate block ({}:{}, {}:{}) of ({}, {})'.format(
                this_ortho_bounds[0] - self.ortho_bounds[0], this_ortho_bounds[1] - self.ortho_bounds[0],
                this_ortho_bounds[2] - self.ortho_bounds[2], this_ortho_bounds[3] - self.ortho_bounds[2],
                self.ortho_bounds[1] - self.ortho_bounds[0], self.ortho_bounds[3] - self.ortho_bounds[2]))
        ortho_data = self._get_orthorectified_version(
            this_ortho_bounds, this_pixel_bounds,
            self._calculator[this_pixel_bounds[0]:this_pixel_bounds[1], this_pixel_bounds[2]:this_pixel_bounds[3]])
        # determine the relative image size
        start_indices = (this_ortho_bounds[0] - self.ortho_bounds[0],
                         this_ortho_bounds[2] - self.ortho_bounds[2])
        return ortho_data, start_indices

    def next(self):
        """
        Get the next iteration of ortho-rectified data.

        Returns
        -------
        numpy.ndarray, Tuple[int, int]
            The data and the (normalized) indices (start_row, start_col) for this section of data, relative
            to overall output shape.
        """

        # NB: this is the Python 2 pattern for iteration
        return self.__next__()
