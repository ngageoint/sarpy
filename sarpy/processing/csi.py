# -*- coding: utf-8 -*-
"""
The methods for computing a color sub-aperture image for SICD type images.

As noted in the CSICalculator class, the full resolution data along the split dimension
is required, so sub-sampling along the split dimension does not decrease the amount of
data which must be fetched and/or processing which must be performed.

**Example Usage**
>>> from matplotlib import pyplot
>>> from sarpy.io.complex.converter import open_complex
>>> from sarpy.processing.csi import CSICalculator
>>> from sarpy.visualization.remap import density

>>> # open a sicd type file
>>> reader = open_complex("<file name>")
>>> # see the sizes of all image segments
>>> reader.get_data_size_as_tuple()

>>> # construct the csi performer instance
>>> # make sure to set the index and dimension as appropriate
>>> csi_calculator = CSICalculator(reader, dimension=0, index=0)
>>> # see the size for this particular image element
>>> # this is identical to the data size from the reader at index
>>> csi_calculator.data_size

>>> # set a different index or change the dimension
>>> csi_calculator.index = 2
>>> csi_calculator.dimension = 1

>>> # calculate the csi for an image segment
>>> csi_data = csi_calculator[300:500, 200:600]

>>> # let's view this csi image using matplotlib
>>> fig, axs = pyplot.subplots(nrows=1, ncols=1)
>>> axs.imshow(density(csi_data), aspect='equal')
>>> pyplot.show()
"""

import logging
import os
from typing import Union, Tuple

import numpy

from sarpy.compliance import string_types, integer_types, int_func
from sarpy.io.complex.converter import open_complex
from sarpy.io.general.base import BaseReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.product.sidd_creation_utils import create_sidd
from sarpy.io.product.sidd import SIDDWriter
from sarpy.processing.ortho_rectify import OrthorectificationHelper
from sarpy.visualization.remap import amplitude_to_density, clip_cast

__classification__ = "UNCLASSIFIED"
__author__ = ('Thomas McCullough',  'Melanie Baker')


def filter_map_construction(siz):
    """
    Provides the RGB filter array for sub-aperture processing.

    Parameters
    ----------
    siz : int
        the size of the colormap

    Returns
    -------
    numpy.ndarray
        the `siz x 3` colormap array
    """

    if siz < 1:
        raise ValueError('Cannot create the filter map with fewer than 4 elements.')

    siz = int_func(round(siz))
    basic_size = int_func(numpy.ceil(0.25*siz))

    # create trapezoidal stack
    trapezoid = numpy.hstack(
        (numpy.arange(1, basic_size+1, dtype=numpy.int32),
         numpy.full((basic_size-1, ), basic_size, dtype=numpy.int32),
         numpy.arange(basic_size, 0, -1, dtype=numpy.int32)))/float(basic_size)
    out = numpy.zeros((siz, 3), dtype=numpy.float64)
    # create red, green, blue indices
    green_inds = int_func(round(0.5*(siz - trapezoid.size))) + numpy.arange(trapezoid.size)
    red_inds = ((green_inds + basic_size) % siz)
    blue_inds = ((green_inds - basic_size) % siz)
    # populate our array
    out[red_inds, 0] = trapezoid
    out[green_inds, 1] = trapezoid
    out[blue_inds, 2] = trapezoid
    return out


def csi_from_array(array, dimension=0, platform_direction='R', fill=1, filter_map=None):
    """
    Creates a color subaperture array from a complex array.

    Parameters
    ----------
    array : numpy.ndarray
        The complex valued SAR data, assumed to be in the "image" domain.
        Required to be two-dimensional.
    dimension : int
        The dimension over which to split the sub-aperture.
    platform_direction : str
        The (case insensitive) platform direction, required to be one of `('R', 'L')`.
    fill : float
        The fill factor. This will be ignored if `filter_map` is provided.
    filter_map : None|numpy.ndarray
        The RGB filter mapping. This is assumed constructed using :func:`filter_map_construction`.

    Returns
    -------
    numpy.ndarray
    """

    if not (isinstance(array, numpy.ndarray) and len(array.shape) == 2 and numpy.iscomplexobj(array)):
        raise ValueError('array must be a two-dimensional numpy array of complex dtype')

    dimension = int_func(dimension)
    if dimension not in [0, 1]:
        raise ValueError('dimension must be 0 or 1, got {}'.format(dimension))
    if dimension == 0:
        array = array.T

    pdir_func = platform_direction.upper()[0]
    if pdir_func not in ['R', 'L']:
        raise ValueError('It is expected that pdir is one of "R" or "L". Got {}'.format(platform_direction))

    # get our filter construction data
    if filter_map is None:
        filter_map = filter_map_construction(array.shape[1]/float(fill))
    if not (isinstance(filter_map, numpy.ndarray) and
            filter_map.dtype.name in ['float32', 'float64'] and
            filter_map.ndim == 2 and filter_map.shape[1] == 3):
        raise ValueError('filter_map must be a N x 3 numpy array of float dtype.')

    # move to phase history domain
    ph_indices = int(numpy.floor(0.5*(array.shape[1] - filter_map.shape[0]))) + \
                 numpy.arange(filter_map.shape[0], dtype=numpy.int32)
    ph0 = numpy.fft.fftshift(numpy.fft.ifft(numpy.cast[numpy.complex128](array), axis=1), axes=1)[:, ph_indices]
    # construct the filtered workspace
    # NB: processing is more efficient with color band in the first dimension
    ph0_RGB = numpy.zeros((3, array.shape[0], filter_map.shape[0]), dtype=numpy.complex128)
    for i in range(3):
        ph0_RGB[i, :, :] = ph0*filter_map[:, i]
    del ph0

    # Shift phase history to avoid having zeropad in middle of filter, to alleviate
    # the purple sidelobe artifact.
    filter_shift = int(numpy.ceil(0.25*filter_map.shape[0]))
    ph0_RGB[0, :] = numpy.roll(ph0_RGB[0, :], -filter_shift)
    ph0_RGB[2, :] = numpy.roll(ph0_RGB[2, :], filter_shift)
    # NB: the green band is already centered

    # FFT back to the image domain
    im0_RGB = numpy.fft.fft(numpy.fft.fftshift(ph0_RGB, axes=2), n=array.shape[1], axis=2)
    del ph0_RGB

    # Replace the intensity with the original image intensity to main full resolution
    # (in intensity, but not in color).
    scale_factor = numpy.abs(array)/numpy.abs(im0_RGB).max(axis=0)
    im0_RGB = numpy.abs(im0_RGB)*scale_factor

    # reorient image so that the color segment is in the final dimension
    if dimension == 0:
        im0_RGB = im0_RGB.transpose([2, 1, 0])
    else:
        im0_RGB = im0_RGB.transpose([1, 2, 0])
    if pdir_func == 'R':
        # reverse the color band order
        im0_RGB = im0_RGB[:, :, ::-1]
    return im0_RGB


class CSICalculator(object):
    """
    Class for creating color sub-aperture image from a reader instance.

    It is important to note that full resolution is required for processing along
    the split dimension, so sub-sampling along the split dimension does not decrease
    the amount of data which must be fetched.
    """

    __slots__ = (
        '_reader', '_index', '_sicd', '_platform_direction', '_dimension', '_data_size',
        '_fill', '_block_size')

    def __init__(self, reader, dimension=0, index=0, block_size=50):
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
        """

        self._index = None # set explicitly
        self._sicd = None  # set with index setter
        self._platform_direction = None  # set with the index setter
        self._dimension = None # set explicitly
        self._data_size = None  # set with index setter
        self._fill = None # set implicitly with _set_fill()
        self._block_size = None # set explicitly

        # validate the reader
        if isinstance(reader, string_types):
            reader = open_complex(reader)
        if not isinstance(reader, BaseReader):
            raise TypeError('reader is required to be a path name for a sicd-type image, '
                            'or an instance of a reader object.')
        if not reader.is_sicd_type:
            raise TypeError('reader is required to be of sicd_type.')
        self._reader = reader
        # set the other properties
        self.dimension = dimension
        self.index = index
        self.block_size = block_size

    @property
    def reader(self):
        # type: () -> BaseReader
        """
        BaseReader: The reader instance.
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
        self._set_fill()

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
        value = int(value)
        if value < 0:
            raise ValueError('The index must be a non-negative integer, got {}'.format(value))

        sicds = self.reader.get_sicds_as_tuple()
        if value >= len(sicds):
            raise ValueError('The index must be less than the sicd count.')
        self._index = value
        self._sicd = sicds[value]

        if self._sicd.SCPCOA is None or self._sicd.SCPCOA.SideOfTrack is None:
            logging.warning(
                'The sicd object at index {} has unpopulated SCPCOA.SideOfTrack. '
                'Defaulting to "R", which may be incorrect.')
            self._platform_direction = 'R'
        else:
            self._platform_direction = self._sicd.SCPCOA.SideOfTrack

        self._data_size = self.reader.get_data_size_as_tuple()[value]
        self._set_fill()

    @property
    def fill(self):
        # type: () -> float
        """
        float: The fill factor for the subaperture splitting.
        """

        return self._fill

    def _set_fill(self):
        self._fill = None
        if self._dimension is None:
            return
        if self._index is None:
            return

        sicd = self._sicd

        if self.dimension == 0:
            try:
                fill = 1/(sicd.Grid.Row.SS*sicd.Grid.Row.ImpRespBW)
            except (ValueError, AttributeError, TypeError):
                fill = 1
        else:
            try:
                fill = 1/(sicd.Grid.Col.SS*sicd.Grid.Col.ImpRespBW)
            except (ValueError, AttributeError, TypeError):
                fill = 1
        self._fill = float(fill)

    @property
    def block_size(self):
        # type: () -> int
        """
        None|int: The approximate processing block size in MB.
        """

        return self._block_size

    @block_size.setter
    def block_size(self, value):
        if value is None:
            self._block_size = None
        value = int(value)
        if value < 1:
            value = 1
        self._block_size = value

    @property
    def sicd(self):
        # type: () -> SICDType
        """
        SICDType: The sicd structure.
        """

        return self._sicd

    def _parse_slicing(self, item):
        # type: (Union[None, int, slice, tuple]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]

        def validate_int(entry, bound):
            if entry <= -bound or entry >= bound:
                raise ValueError('Slice argument {} does not fit with bound {}'.format(entry, bound))
            if entry < 0:
                return entry + bound
            return entry

        def parse(entry, dimension):
            bound = self.data_size[dimension]
            if entry is None:
                return 0, bound, 1
            elif isinstance(entry, integer_types):
                entry = validate_int(entry, bound)
                return entry, entry+1, 1
            elif isinstance(entry, slice):
                t_start = entry.start
                t_stop = entry.stop
                t_step = 1 if entry.step is None else entry.step
                if t_start is None and t_stop is None:
                    t_start, t_stop = 0, bound
                elif t_start is None:
                    t_stop = validate_int(t_stop, bound)
                    t_start = 0 if t_stop >= 0 else bound-1
                elif t_stop is None:
                    t_start = validate_int(t_start, bound)
                    t_stop = -1 if t_step < 0 else bound
                else:
                    t_start = validate_int(t_start, bound)
                    t_stop = validate_int(t_stop, bound)
                if t_start == t_stop:
                    raise ValueError(
                        'Got identical start and stop slice bounds. Empty slicing not '
                        'supported for CSICalculator.')
                if (t_step < 0 and t_start <= t_stop) or (t_step > 0 and t_start >= t_stop):
                    raise ValueError(
                        'The slice values start={}, stop={}, step={} are not viable'.format(t_start, t_stop, t_step))
                return t_start, t_stop, t_step
            else:
                raise TypeError('CSICalculator does not support slicing using type {}'.format(type(entry)))

        # this input is assumed to come from slice parsing
        if isinstance(item, tuple):
            if len(item) > 2:
                raise ValueError(
                    'CSICalculator received slice argument {}. We cannot slice on more than two dimensions.'.format(item))

            return parse(item[0], 0), parse(item[1], 1)
        elif isinstance(item, slice):
            return parse(item, 0), parse(None, 1)
        elif isinstance(item, integer_types):
            return parse(item, 0), parse(None, 1)
        else:
            raise TypeError('CSICalculator does not support slicing using type {}'.format(type(item)))

    def get_fetch_block_size(self, start_element, stop_element):
        """
        Gets the fetch block size for the given full resolution section.
        This assumes that the fetched data will be 14 bytes per pixel, in
        accordance with 3-band complex64 data.

        Parameters
        ----------
        start_element : int
        stop_element : int

        Returns
        -------
        int
        """

        if stop_element == start_element:
            return None

        full_size = float(abs(stop_element - start_element))
        return None if self.block_size is None else \
            max(1, int(numpy.ceil(self.block_size*2**17 /full_size)))

    @staticmethod
    def extract_blocks(the_range, block_size):
        """
        Extract the block definition.

        Parameters
        ----------
        the_range
        block_size : None|int|float

        Returns
        -------
        List[Tuple[int, int, int]], List[Tuple[int, int]]
        """


        entries = numpy.arange(the_range[0], the_range[1], the_range[2], dtype=numpy.int64)
        if block_size is None:
            return [the_range, ], [(0, entries.size), ]

        # how many blocks?
        block_count = int(numpy.ceil(entries.size/block_size))
        if block_size == 1:
            return [the_range, ], [(0, entries.size), ]

        # workspace for what the blocks are
        out1 = []
        out2 = []
        start_ind = 0
        for i in range(block_count):
            end_ind = start_ind+block_size
            if end_ind < entries.size:
                block1 = (int_func(entries[start_ind]), int_func(entries[end_ind]), the_range[2])
                block2 = (start_ind, end_ind)
            else:
                block1 = (int_func(entries[start_ind]), the_range[1], the_range[2])
                block2 = (start_ind, entries.size)
            out1.append(block1)
            out2.append(block2)
            start_ind = end_ind
        return out1, out2

    def _full_row_resolution(self, row_range, col_range, filter_map):
        """
        Perform the full row resolution calculation.

        Parameters
        ----------
        row_range
        col_range
        filter_map : numpy.ndarray

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

        return csi_from_array(
            data, dimension=0, platform_direction=self._platform_direction,
            filter_map=filter_map)

    def _full_column_resolution(self, row_range, col_range, filter_map):
        """
        Perform the full column resolution calculation.

        Parameters
        ----------
        row_range
        col_range
        filter_map : numpy.ndarray

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

        return csi_from_array(
            data, dimension=1, platform_direction=self._platform_direction,
            filter_map=filter_map)

    def __getitem__(self, item):
        """
        Fetches the csi data based on the input slice.

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
            out_size = (row_count, col_count, 3)
            return numpy.zeros(out_size, dtype=numpy.float64)

        # parse the slicing to ensure consistent structure
        row_range, col_range = self._parse_slicing(item)
        if self.dimension == 0:
            # we will proceed fetching full row resolution
            full_row_count = abs(int_func(row_range[1] - row_range[0]))
            row_snip = -1 if row_range[2] < 0 else 1
            filter_map = filter_map_construction(full_row_count/self._fill)
            block_size = self.get_fetch_block_size(row_range[0], row_range[1])
            this_row_range = (row_range[0], row_range[1], row_snip)
            # get our block definitions
            column_blocks, result_blocks = self.extract_blocks(col_range, block_size)
            if len(column_blocks) == 1:
                # it's just a single block
                csi = self._full_row_resolution(this_row_range, col_range, filter_map)
                return csi[::abs(row_range[2]), :, :]
            else:
                # prepare the output space
                out = prepare_output()
                for this_column_range, result_range in zip(column_blocks, result_blocks):
                    csi = self._full_row_resolution(this_row_range, this_column_range, filter_map)
                    out[:, result_range[0]:result_range[1], :] = csi[::abs(row_range[2]), :, :]
                return out
        else:
            # we will proceed fetching full column resolution
            full_col_count = abs(int_func(col_range[1] - col_range[0]))
            col_snip = -1 if col_range[2] < 0 else 1
            filter_map = filter_map_construction(full_col_count/self._fill)
            block_size = self.get_fetch_block_size(col_range[0], col_range[1])
            this_col_range = (col_range[0], col_range[1], col_snip)
            # get our block definitions
            row_blocks, result_blocks = self.extract_blocks(row_range, block_size)
            if len(row_blocks) == 1:
                # it's just a single block
                csi = self._full_column_resolution(row_range, this_col_range, filter_map)
                return csi[:, ::abs(col_range[2]), :]
            else:
                # prepare the output space
                out = prepare_output()
                for this_row_range, result_range in zip(row_blocks, result_blocks):
                    csi = self._full_column_resolution(this_row_range, this_col_range, filter_map)
                    out[result_range[0]:result_range[1], :, :] = csi[:, ::abs(col_range[2]), :]
                return out


def create_csi_sidd(ortho_helper, output_directory, output_file=None, dimension=0, block_size=50, bounds=None, version=2):
    """
    Create a SIDD version of a Color Sub-Aperture Image from a SICD type reader.

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

    def get_orthorectified_version(temp_pixel_bounds):
        csi_data = csi_calculator[temp_pixel_bounds[0]:temp_pixel_bounds[1], temp_pixel_bounds[2]:temp_pixel_bounds[3]]
        # orthorectify this data, and write straight into the sidd file
        rows_temp = temp_pixel_bounds[1] - temp_pixel_bounds[0]
        if csi_data.shape[0] == rows_temp:
            row_array = numpy.arange(temp_pixel_bounds[0], temp_pixel_bounds[1])
        elif csi_data.shape[0] == (rows_temp + 1):
            row_array = numpy.arange(temp_pixel_bounds[0], temp_pixel_bounds[1] + 1)
        else:
            raise ValueError('Unhandled data size mismatch {} and {}'.format(csi_data.shape, rows_temp))
        cols_temp = temp_pixel_bounds[3] - temp_pixel_bounds[2]
        if csi_data.shape[1] == cols_temp:
            col_array = numpy.arange(temp_pixel_bounds[2], temp_pixel_bounds[2])
        elif csi_data.shape[1] == (cols_temp + 1):
            col_array = numpy.arange(temp_pixel_bounds[2], temp_pixel_bounds[2] + 1)
        else:
            raise ValueError('Unhandled data size mismatch {} and {}'.format(csi_data.shape, cols_temp))
        return row_array, col_array, csi_data

    reader = ortho_helper.reader
    index = ortho_helper.index

    # construct the CSI calculator class
    csi_calculator = CSICalculator(reader, dimension=dimension, index=index, block_size=block_size)

    # validate the bounds
    if bounds is None:
        bounds = (0, csi_calculator.data_size[0], 0, csi_calculator.data_size[1])
    bounds, pixel_rectangle = ortho_helper.bounds_to_rectangle(bounds)
    # get the corresponding prtho bounds
    ortho_bounds = ortho_helper.get_orthorectification_bounds_from_pixel_object(pixel_rectangle)

    # Extract the mean of the data magnitude - for global remap usage
    mean_block_size = 3*csi_calculator.get_fetch_block_size(bounds[0], bounds[1])  # this is single band, so make bigger as necessary
    mean_column_blocks, _ = csi_calculator.extract_blocks((ortho_bounds[2], ortho_bounds[3], 1), block_size=mean_block_size)
    mean_total = 0.0
    mean_count = 0
    for this_column_range in mean_column_blocks:
        data = numpy.abs(csi_calculator.reader[bounds[0]:bounds[1], this_column_range[0]:this_column_range[1], csi_calculator.index])
        mean_total += numpy.sum(data)
        mean_count += data.size
    the_mean = mean_total/mean_count

    # create the sidd structure
    sidd_structure = create_sidd(
        ortho_helper, ortho_bounds,
        product_class='Color Sub-Aperture Image', pixel_type='RGB24I', version=version)
    # set suggested name
    sidd_structure._NITF = {
        'SUGGESTED_NAME': csi_calculator.sicd.get_suggested_name(csi_calculator.index)+'__CSI', }
    # create the sidd writer
    if output_file is None:
        full_filename = os.path.join(output_directory, sidd_structure._NITF['SUGGESTED_NAME']+'.nitf')
    else:
        full_filename = os.path.join(output_directory, output_file)
    if os.path.expanduser(full_filename):
        raise IOError('File {} already exists.'.format(full_filename))
    writer = SIDDWriter(full_filename, sidd_structure, csi_calculator.sicd)

    if csi_calculator.dimension == 0:
        # we are using the full resolution row data
        # determine the orthorectified blocks to use
        block_size = csi_calculator.get_fetch_block_size(ortho_bounds[0], ortho_bounds[1])
        ortho_column_blocks, ortho_result_blocks = csi_calculator.extract_blocks(
            (ortho_bounds[2], ortho_bounds[3], 1), block_size=block_size)

        for this_column_range, result_range in zip(ortho_column_blocks, ortho_result_blocks):
            # determine the corresponding pixel ranges to encompass these values
            this_ortho_bounds, this_pixel_bounds = ortho_helper.extract_pixel_bounds(
                (ortho_bounds[0], ortho_bounds[1], this_column_range[0], this_column_range[1]))
            # extract the csi data
            row_array, col_array, csi_data = get_orthorectified_version(this_pixel_bounds)
            ortho_csi_data = clip_cast(
                amplitude_to_density(
                    ortho_helper.get_orthorectified_from_array(ortho_bounds, row_array, col_array, csi_data),
                    data_mean=the_mean),
                dtype='uint8')
            start_indices = (this_ortho_bounds[0] - ortho_bounds[0],
                             result_range[0] + this_ortho_bounds[2] - ortho_bounds[2])
            writer(ortho_csi_data, start_indices=start_indices)
    else:
        # we are using the full resolution column data
        # determine the orthorectified blocks to use
        block_size = csi_calculator.get_fetch_block_size(ortho_bounds[2], ortho_bounds[3])
        ortho_row_blocks, ortho_result_blocks = csi_calculator.extract_blocks(
            (ortho_bounds[0], ortho_bounds[1], 1), block_size=block_size)

        for this_row_range, result_range in zip(ortho_row_blocks, ortho_result_blocks):
            # determine the corresponding pixel ranges to encompass these values
            this_ortho_bounds, this_pixel_bounds = ortho_helper.extract_pixel_bounds(
                (this_row_range[0], this_row_range[1], ortho_bounds[2], ortho_bounds[3]))
            # extract the csi data
            row_array, col_array, csi_data = get_orthorectified_version(this_pixel_bounds)
            ortho_csi_data = clip_cast(
                amplitude_to_density(
                    ortho_helper.get_orthorectified_from_array(ortho_bounds, row_array, col_array, csi_data),
                    data_mean=the_mean),
                dtype='uint8')
            start_indices = (result_range[0] + this_ortho_bounds[0] - ortho_bounds[0],
                             this_ortho_bounds[2] - ortho_bounds[2])
            writer(ortho_csi_data, start_indices=start_indices)
