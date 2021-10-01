"""
The methods for computing a color sub-aperture image for SICD type images.

As noted in the CSICalculator class, the full resolution data along the split dimension
is required, so sub-sampling along the split dimension does not decrease the amount of
data which must be fetched and/or processing which must be performed.

Examples
--------
.. code-block:: python

    from matplotlib import pyplot
    from sarpy.io.complex.converter import open_complex
    from sarpy.processing.csi import CSICalculator
    from sarpy.visualization.remap import Density

    # open a sicd type file
    reader = open_complex("<file name>")
    # see the sizes of all image segments
    print(reader.get_data_size_as_tuple())

    # construct the csi performer instance
    # make sure to set the index and dimension as appropriate
    csi_calculator = CSICalculator(reader, dimension=0, index=0)
    # see the size for this particular image element
    # this is identical to the data size from the reader at index
    print(csi_calculator.data_size)

    # set a different index or change the dimension
    # csi_calculator.index = 2
    # csi_calculator.dimension = 1

    # calculate the csi for an image segment
    csi_data = csi_calculator[300:500, 200:600]

    # create our remap function
    density = Density()

    # let's view this csi image using matplotlib
    fig, axs = pyplot.subplots(nrows=1, ncols=1)
    axs.imshow(density(csi_data), aspect='equal')
    pyplot.show()
"""

__classification__ = "UNCLASSIFIED"
__author__ = 'Thomas McCullough'

import numpy

from sarpy.compliance import int_func
from sarpy.processing.fft_base import FFTCalculator, fft, ifft, fftshift
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.utils import get_fetch_block_size


def filter_map_construction(siz):
    """
    Provides the RGB filter array for sub-aperture processing.

    Parameters
    ----------
    siz : int|float
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
        reader : str|SICDTypeReader
            Input file path or reader object, which must be of sicd type.
        dimension : int
            The dimension over which to split the sub-aperture.
        index : int
            The sicd index to use.
        block_size : None|int|float
            The approximate processing block size to fetch, given in MB.
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

        return get_fetch_block_size(start_element, stop_element, self.block_size_in_bytes, bands=3)

    def _full_row_resolution(self, row_range, col_range, filter_map=None):
        data = super(CSICalculator, self)._full_row_resolution(row_range, col_range)
        return csi_array(
            data, dimension=0, platform_direction=self._platform_direction,
            filter_map=filter_map)

    def _full_column_resolution(self, row_range, col_range, filter_map=None):
        data = super(CSICalculator, self)._full_column_resolution(row_range, col_range)
        return csi_array(
            data, dimension=1, platform_direction=self._platform_direction,
            filter_map=filter_map)

    def _prepare_output(self, row_range, col_range):
        row_count = int_func((row_range[1] - row_range[0]) / float(row_range[2]))
        col_count = int_func((col_range[1] - col_range[0]) / float(col_range[2]))
        out_size = (row_count, col_count, 3)
        return numpy.zeros(out_size, dtype=numpy.float64)

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
                out = self._prepare_output(row_range, col_range)
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
                out = self._prepare_output(row_range, col_range)
                for this_row_range, result_range in zip(row_blocks, result_blocks):
                    csi = self._full_column_resolution(this_row_range, this_col_range, filter_map)
                    out[result_range[0]:result_range[1], :, :] = csi[:, ::abs(col_range[2]), :]
                return out
