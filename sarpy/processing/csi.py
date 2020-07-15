# -*- coding: utf-8 -*-
"""
The methods for computing a color sub-aperture image for SICD type images.

As noted in the PerformCSI class, the full resolution data along the split dimension
is required, so sub-sampling along the split dimension does not decrease the amount of
data which must be fetched and/or processing whihc must be performed.

**Example Usage**
>>> from matplotlib import pyplot
>>> from sarpy.io.complex.converter import open_complex
>>> from sarpy.processing.csi import PerformCSI
>>> from sarpy.visualization.remap import density

>>> # open a sicd type file
>>> reader = open_complex(<file name>)
>>> # see the sizes of all image segments
>>> reader.get_data_size_as_tuple()

>>> # construct the csi performer instance
>>> # make sure to set the index and dimension as appropriate
>>> csi_calculator = PerformCSI(reader, dimension=0, index=0)
>>> # see the size for this particular image element
>>> # this is identical to the data size from the reader at index
>>> csi_calculator.data_size

>>> # set a different index or change the dimension
>>> csi_calculator.index = 2
>>> csi_calculator.dimension = 1

>>> # calculate the csi for an image segment
>>> csi_data = csi_calculator[300:500, 200:600]
>>> # recall that even though we are subsampling along both dimensions, the processing
>>> # requires that we fetch the entire segment along the split dimension

>>> # let's view this csi image using matplotlib
>>> fig, axs = pyplot.subplots(nrows=1, ncols=1)
>>> h1 = axs.imshow(density(csi_data), cmap='inferno', aspect='equal')
>>> fig.colorbar(h1, ax=axs)
>>> pyplot.show()
"""

import logging
from typing import Union, Tuple

import numpy

from sarpy.compliance import string_types, integer_types, int_func
from sarpy.io.complex.converter import open_complex
from sarpy.io.general.base import BaseReader

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

    siz = int_func(siz)
    red_siz = max(1, int_func(siz/4))
    # create trapezoidal stack
    trapezoid = numpy.hstack(
        (numpy.arange(1, red_siz+1, dtype=numpy.float32)/red_siz,
         numpy.ones((red_siz, ), dtype=numpy.float32),
         numpy.arange(red_siz, 0, -1, dtype=numpy.float64)/red_siz))
    out = numpy.zeros((siz, 3), dtype=numpy.float32)
    # create red, green, blue indices
    green_inds = int_func(0.5*(siz - trapezoid.size)) + numpy.arange(trapezoid.size)
    red_inds = ((green_inds + red_siz) % siz)
    blue_inds = ((green_inds - red_siz) % siz)
    # populate our array
    out[red_inds, 0] = trapezoid
    out[green_inds, 1] = trapezoid
    out[blue_inds, 2] = trapezoid
    return out


def csi_array(array, dimension=0, platform_direction='R', fill=1, filter_map=None):
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

    dim = int_func(dimension)
    if dim not in [0, 1]:
        raise ValueError('dimension must be 0 or 1, got {}'.format(dim))
    if dim == 0:
        array = array.T  # this is a view

    pdir_func = platform_direction.upper()[0]
    if pdir_func not in ['R', 'L']:
        raise ValueError('It is expected that pdir is one of "R" or "L". Got {}'.format(platform_direction))

    # get our filter construction data
    if filter_map is None:
        filter_map = filter_map_construction(array.shape[1]/float(fill))
    if not (isinstance(filter_map, numpy.ndarray) and filter_map.dtype.name == 'float32' and
            filter_map.ndim == 2 and filter_map.shape[1] == 3):
        raise ValueError('filter_map must be a N x 3 numpy array of dtype float32.')
    # move to phase history domain
    ph_indices = int_func(numpy.floor(0.5*(array.shape[1] - filter_map.shape[0]))) + \
                 numpy.arange(filter_map.shape[0], dtype=numpy.int32)
    ph0 = numpy.fft.fftshift(numpy.fft.ifft(array, axis=1), axes=1)[:, ph_indices]

    # construct the filtered workspace
    ph0_RGB = numpy.zeros((array.shape[0], filter_map.shape[0], 3), dtype=numpy.complex64)
    for i in range(3):
        ph0_RGB[:, :, i] = ph0*filter_map[:, i]
    del ph0

    # Shift phase history to avoid having zeropad in middle of filter.
    # This fixes the purple sidelobe artifact.
    filter_shift = int(filter_map.shape[0]/4)
    ph0_RGB[:, :, 0] = numpy.roll(ph0_RGB[:, :, 0], -filter_shift)
    ph0_RGB[:, :, 2] = numpy.roll(ph0_RGB[:, :, 2], filter_shift)
    # NB: the green band is already centered

    # FFT back to the image domain
    im0_RGB = numpy.fft.fft(numpy.fft.fftshift(ph0_RGB, axes=1), n=array.shape[1], axis=1)
    del ph0_RGB

    # Replace the intensity with the original image intensity to main full resolution
    # (in intensity, but not in color).
    scale_factor = numpy.abs(array)/numpy.abs(im0_RGB).max(axis=2)
    im0_RGB = numpy.abs(im0_RGB)*scale_factor[:, :, numpy.newaxis]

    # reorient images
    if dim == 0:
        im0_RGB = im0_RGB.transpose([1, 0, 2])
    if pdir_func == 'R':
        # reverse the color band order
        im0_RGB = im0_RGB[:, :, ::-1]
    return im0_RGB


class PerformCSI(object):
    """
    Class for creating color subaperture images from a reader instance.

    It is important to note that sub-sampling along the split dimension does
    not decrease the amount of data which must be fetched, because the processing
    methodology requires the entire row/column in the split dimension.
    """

    __slots__ = (
        '_reader', '_index', '_sicd', '_platform_direction', '_dimension', '_data_size', '_fill', '_filter_map',
        '_block_size')

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
        self._fill = None # set implicitly with _set_fill_and_filtermap()
        self._filter_map = None  # set implicitly with _set_fill_and_filtermap()
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
        self._set_fill_and_filtermap()

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
        self._set_fill_and_filtermap()

    @property
    def fill(self):
        # type: () -> float
        """
        float: The fill factor for the subaperture splitting.
        """

        return self._fill

    def _set_fill_and_filtermap(self):
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
        # set filter_map
        self._filter_map = filter_map_construction(self.data_size[self.dimension]/self._fill)

    @property
    def block_size(self):
        # type: () -> int
        """
        int: The approximate processing block size in MB.
        """

        return self._block_size

    @block_size.setter
    def block_size(self, value):
        value = int(value)
        if value < 1:
            value = 1
        self._block_size = value

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
                        'supported for PerformCSI.')
                if (t_step < 0 and t_start <= t_stop) or (t_step > 0 and t_start >= t_stop):
                    raise ValueError(
                        'The slice values start={}, stop={}, step={} are not viable'.format(t_start, t_stop, t_step))
                return t_start, t_stop, t_step
            else:
                raise TypeError('PerformCSI does not support slicing using type {}'.format(type(entry)))

        # this input is assumed to come from slice parsing
        if isinstance(item, tuple):
            if len(item) > 2:
                raise ValueError(
                    'PerformCSI received slice argument {}. We cannot slice on more than two dimensions.'.format(item))

            return parse(item[0], 0), parse(item[1], 1)
        elif isinstance(item, slice):
            return parse(item, 0), parse(None, 1)
        elif isinstance(item, integer_types):
            return parse(item, 0), parse(None, 1)
        else:
            raise TypeError('PerformCSI does not support slicing using type {}'.format(type(item)))

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

        if self._filter_map is None:
            raise ValueError('Unable to proceed unless the index and dimension are set.')

        # parse the slicing
        row_range, col_range = self._parse_slicing(item)
        print('row_range = {}, col_range = {}'.format(row_range, col_range))
        row_count = int_func((row_range[1] - row_range[0])/float(row_range[2]))
        col_count = int_func((col_range[1] - col_range[0])/float(col_range[2]))
        out_size = (row_count, col_count, 3)
        out = numpy.zeros(out_size, dtype=numpy.complex64)

        if self.dimension == 0:
            # how many columns to fetch at a single time, because we have to fetch the entire row
            fetch_max_cols = max(1, int(numpy.ceil(self.block_size * 2 ** 17 / float(self.data_size[0]))))
            cols_array = numpy.arange(col_range[0], col_range[1], col_range[2])
            snip = -1 if col_range[2] < 0 else 1
            # proceed with our block processing
            start_entry = 0
            while start_entry < cols_array.size:
                col_start = cols_array[start_entry]
                actual_end = min(start_entry + fetch_max_cols, cols_array.size-1)
                apparent_end_entry = cols_array[actual_end] + snip
                # fetch the full column data
                if apparent_end_entry == -1:  # step will be negative here
                    full_data = self.reader[:, col_start::col_range[2], self.index]
                else:
                    full_data = self.reader[:, col_start:apparent_end_entry:col_range[2], self.index]
                if full_data.ndim == 1:
                    full_data = numpy.reshape(full_data, (full_data.size, 1))
                # perform the csi of this section
                csi = csi_array(
                    full_data, dimension=self.dimension, platform_direction=self._platform_direction,
                    filter_map=self._filter_map)
                if row_range[1] == -1:
                    out[:, start_entry:actual_end + 1] = csi[row_range[0]::row_range[2], :]
                else:
                    out[:, start_entry:actual_end + 1] = csi[row_range[0]:row_range[1]:row_range[2], :]
                start_entry = actual_end + 1
        else:
            # how many rows to fetch a single time for processing, because we have to fetch the entire column
            fetch_max_rows = max(1, int(numpy.ceil(self.block_size*2**17/float(self.data_size[1]))))
            rows_array = numpy.arange(row_range[0], row_range[1], row_range[2])
            snip = -1 if row_range[2] < 0 else 1
            # proceed with our block processing
            start_entry = 0
            while start_entry < rows_array.size:
                row_start = rows_array[start_entry]
                actual_end = min(start_entry+fetch_max_rows, rows_array.size-1)
                apparent_end_entry = rows_array[actual_end] + snip
                # fetch the full column data
                if apparent_end_entry == -1:  # step will be negative here
                    full_data = self.reader[row_start::row_range[2], :, self.index]
                else:
                    full_data = self.reader[row_start:apparent_end_entry:row_range[2], :, self.index]
                if full_data.ndim == 1:
                    full_data = numpy.reshape(full_data, (1, full_data.size))
                # perform the csi of this section
                csi = csi_array(
                    full_data, dimension=self.dimension, platform_direction=self._platform_direction,
                    filter_map=self._filter_map)
                if col_range[1] == -1:
                    out[start_entry:actual_end+1, :] = csi[:, col_range[0]::col_range[2]]
                else:
                    out[start_entry:actual_end+1, :] = csi[:, col_range[0]:col_range[1]:col_range[2]]
                start_entry = actual_end + 1
        return out
