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
import numpy

from sarpy.compliance import int_func
# noinspection PyProtectedMember
from sarpy.processing.fft_base import FFTCalculator, _get_fetch_block_size, fft, ifft, fftshift
from sarpy.io.general.base import BaseReader
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


def csi_array(array, dimension=0, platform_direction='R', fill=1, filter_map=None):
    """
    Creates a color subaperture array from a complex array.

    .. Note: this ignores any potential sign issues for the fft and ifft, because
        the results would be identical - fft followed by ifft versus ifft followed
        by fft.

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
        fill = max(1.0, float(fill))
        filter_map = filter_map_construction(array.shape[1]/fill)
    if not (isinstance(filter_map, numpy.ndarray) and
            filter_map.dtype.name in ['float32', 'float64'] and
            filter_map.ndim == 2 and filter_map.shape[1] == 3):
        raise ValueError('filter_map must be a N x 3 numpy array of float dtype.')

    # move to phase history domain
    ph_indices = int(numpy.floor(0.5*(array.shape[1] - filter_map.shape[0]))) + \
                 numpy.arange(filter_map.shape[0], dtype=numpy.int32)
    ph0 = fftshift(ifft(numpy.cast[numpy.complex128](array), axis=1), axes=1)[:, ph_indices]
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
    im0_RGB = fft(fftshift(ph0_RGB, axes=2), n=array.shape[1], axis=2)
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


class CSICalculator(FFTCalculator):
    """
    Class for creating color sub-aperture image from a reader instance.

    It is important to note that full resolution is required for processing along
    the split dimension, so sub-sampling along the split dimension does not decrease
    the amount of data which must be fetched.
    """


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

        super(CSICalculator, self).__init__(
            reader, dimension=dimension, index=index, block_size=block_size)

    def get_fetch_block_size(self, start_element, stop_element):
        """
        Gets the fetch block size for the given full resolution section.
        This assumes that the fetched data will be 24 bytes per pixel, in
        accordance with 3-band complex64 data.

        Parameters
        ----------
        start_element : int
        stop_element : int

        Returns
        -------
        int
        """

        return _get_fetch_block_size(start_element, stop_element, self.block_size_in_bytes, bands=3)


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

        return csi_array(
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

        return csi_array(
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

        def get_dimension_details(the_range):
            full_count = abs(int_func(the_range[1] - the_range[0]))
            the_snip = -1 if the_range[2] < 0 else 1
            t_filter_map = filter_map_construction(full_count/self.fill)
            t_block_size = self.get_fetch_block_size(the_range[0], the_range[1])
            t_full_range = (the_range[0], the_range[1], the_snip)
            return t_filter_map, t_block_size, t_full_range

        def prepare_output():
            row_count = int_func((row_range[1] - row_range[0]) / float(row_range[2]))
            col_count = int_func((col_range[1] - col_range[0]) / float(col_range[2]))
            out_size = (row_count, col_count, 3)
            return numpy.zeros(out_size, dtype=numpy.float64)

        # parse the slicing to ensure consistent structure
        row_range, col_range, _ = self._parse_slicing(item)
        if self.dimension == 0:
            # we will proceed fetching full row resolution
            filter_map, row_block_size, this_row_range = get_dimension_details(row_range)
            # get our block definitions
            column_blocks, result_blocks = self.extract_blocks(col_range, row_block_size)
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
            filter_map, column_block_size, this_col_range = get_dimension_details(col_range)
            # get our block definitions
            row_blocks, result_blocks = self.extract_blocks(row_range, column_block_size)
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


def create_csi_sidd(
        ortho_helper, output_directory, output_file=None, dimension=0,
        block_size=50, bounds=None, version=2):
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

    def get_ortho_helper(temp_pixel_bounds, this_csi_data):
        rows_temp = temp_pixel_bounds[1] - temp_pixel_bounds[0]
        if this_csi_data.shape[0] == rows_temp:
            row_array = numpy.arange(temp_pixel_bounds[0], temp_pixel_bounds[1])
        elif this_csi_data.shape[0] == (rows_temp + 1):
            row_array = numpy.arange(temp_pixel_bounds[0], temp_pixel_bounds[1] + 1)
        else:
            raise ValueError('Unhandled data size mismatch {} and {}'.format(this_csi_data.shape, rows_temp))
        cols_temp = temp_pixel_bounds[3] - temp_pixel_bounds[2]
        if this_csi_data.shape[1] == cols_temp:
            col_array = numpy.arange(temp_pixel_bounds[2], temp_pixel_bounds[3])
        elif this_csi_data.shape[1] == (cols_temp + 1):
            col_array = numpy.arange(temp_pixel_bounds[2], temp_pixel_bounds[3] + 1)
        else:
            raise ValueError('Unhandled data size mismatch {} and {}'.format(this_csi_data.shape, cols_temp))
        return row_array, col_array

    def get_orthorectified_version(these_ortho_bounds, temp_pixel_bounds, this_csi_data):
        row_array, col_array = get_ortho_helper(temp_pixel_bounds, this_csi_data)
        return clip_cast(
            amplitude_to_density(
                ortho_helper.get_orthorectified_from_array(these_ortho_bounds, row_array, col_array, this_csi_data),
                data_mean=the_mean),
            dtype='uint8')

    def log_progress(t_ortho_bounds):
        logging.info('Writing pixels ({}:{}, {}:{}) of ({}, {})'.format(
            t_ortho_bounds[0]-ortho_bounds[0], t_ortho_bounds[1]-ortho_bounds[0],
            t_ortho_bounds[2] - ortho_bounds[2], t_ortho_bounds[3] - ortho_bounds[2],
            ortho_bounds[1] - ortho_bounds[0], ortho_bounds[3] - ortho_bounds[2]))

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
    the_mean = csi_calculator.get_data_mean_magnitude(bounds)

    # create the sidd structure
    sidd_structure = create_sidd(
        ortho_helper, ortho_bounds,
        product_class='Color Subaperture Image', pixel_type='RGB24I', version=version)
    # set suggested name
    sidd_structure._NITF = {
        'SUGGESTED_NAME': csi_calculator.sicd.get_suggested_name(csi_calculator.index)+'__CSI', }
    # create the sidd writer
    if output_file is None:
        # noinspection PyProtectedMember
        full_filename = os.path.join(output_directory, sidd_structure._NITF['SUGGESTED_NAME']+'.nitf')
    else:
        full_filename = os.path.join(output_directory, output_file)
    if os.path.exists(os.path.expanduser(full_filename)):
        raise IOError('File {} already exists.'.format(full_filename))
    writer = SIDDWriter(full_filename, sidd_structure, csi_calculator.sicd)

    if csi_calculator.dimension == 0:
        # we are using the full resolution row data
        # determine the orthorectified blocks to use
        column_block_size = csi_calculator.get_fetch_block_size(ortho_bounds[0], ortho_bounds[1])
        ortho_column_blocks, ortho_result_blocks = csi_calculator.extract_blocks(
            (ortho_bounds[2], ortho_bounds[3], 1), column_block_size)

        for this_column_range, result_range in zip(ortho_column_blocks, ortho_result_blocks):
            # determine the corresponding pixel ranges to encompass these values
            this_ortho_bounds, this_pixel_bounds = ortho_helper.extract_pixel_bounds(
                (ortho_bounds[0], ortho_bounds[1], this_column_range[0], this_column_range[1]))
            # accommodate for real pixel limits
            this_pixel_bounds = ortho_helper.get_real_pixel_bounds(this_pixel_bounds)
            # extract the csi data and ortho-rectify
            ortho_csi_data = get_orthorectified_version(
                this_ortho_bounds, this_pixel_bounds,
                csi_calculator[this_pixel_bounds[0]:this_pixel_bounds[1], this_pixel_bounds[2]:this_pixel_bounds[3]])
            # write out to the file
            start_indices = (this_ortho_bounds[0] - ortho_bounds[0],
                             this_ortho_bounds[2] - ortho_bounds[2])
            log_progress(this_ortho_bounds)
            writer(ortho_csi_data, start_indices=start_indices, index=0)
    else:
        # we are using the full resolution column data
        # determine the orthorectified blocks to use
        row_block_size = csi_calculator.get_fetch_block_size(ortho_bounds[2], ortho_bounds[3])
        ortho_row_blocks, ortho_result_blocks = csi_calculator.extract_blocks(
            (ortho_bounds[0], ortho_bounds[1], 1), row_block_size)

        for this_row_range, result_range in zip(ortho_row_blocks, ortho_result_blocks):
            # determine the corresponding pixel ranges to encompass these values
            this_ortho_bounds, this_pixel_bounds = ortho_helper.extract_pixel_bounds(
                (this_row_range[0], this_row_range[1], ortho_bounds[2], ortho_bounds[3]))
            # accommodate for real pixel limits
            this_pixel_bounds = ortho_helper.get_real_pixel_bounds(this_pixel_bounds)
            # extract the csi data and ortho-rectify
            ortho_csi_data = get_orthorectified_version(
                this_ortho_bounds, this_pixel_bounds,
                csi_calculator[this_pixel_bounds[0]:this_pixel_bounds[1], this_pixel_bounds[2]:this_pixel_bounds[3]])
            # write out to the file
            start_indices = (this_ortho_bounds[0] - ortho_bounds[0],
                             this_ortho_bounds[2] - ortho_bounds[2])
            log_progress(this_ortho_bounds)
            writer(ortho_csi_data, start_indices=start_indices, index=0)
