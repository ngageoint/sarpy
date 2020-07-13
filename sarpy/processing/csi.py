# -*- coding: utf-8 -*-
"""
The module contains methods for computing a color subaperture image
"""

import logging

import numpy
from sarpy.io.complex.converter import open_complex
from sarpy.io.general.base import BaseReader


__classification__ = "UNCLASSIFIED"
__author__ = ('Thomas Mccullough',  'Melanie Baker')


def _jet_wrapped(siz):
    """
    Provides a jet-like colormap array for sub-aperture processing

    Parameters
    ----------
    siz : int
        the size of the colormap

    Returns
    -------
    numpy.ndarray
        the `siz x 3` colormap array
    """

    siz = int(siz)
    red_siz = max(1, int(siz/4))
    # create trapezoidal stack
    trapezoid = numpy.hstack(
        (numpy.arange(1, red_siz+1, dtype=numpy.float64)/float(red_siz),
         numpy.ones((red_siz, ), dtype=numpy.float64),
         numpy.arange(red_siz, 0, -1, dtype=numpy.float64)/float(red_siz)))
    out = numpy.zeros((siz, 3), dtype=numpy.float64)
    # create red, green, blue indices
    green_inds = int(0.5*(siz - trapezoid.size)) + numpy.arange(trapezoid.size)
    red_inds = ((green_inds + red_siz) % siz)
    blue_inds = ((green_inds - red_siz) % siz)
    # populate our array
    out[red_inds, 0] = trapezoid
    out[green_inds, 1] = trapezoid
    out[blue_inds, 2] = trapezoid
    return out


def csi_array(array, dimension=0, platform_direction='R', fill=1):
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
        The fill factor.

    Returns
    -------
    numpy.ndarray
    """

    if not (isinstance(array, numpy.ndarray) and len(array.shape) == 2 and numpy.iscomplexobj(array)):
        raise ValueError('array must be a two-dimensional numpy array of complex dtype')

    dim = int(dimension)
    if dim not in [0, 1]:
        raise ValueError('dimension must be 0 or 1, got {}'.format(dim))
    if dim == 0:
        array = array.T  # this is a view

    pdir_func = platform_direction.upper()[0]
    if pdir_func not in ['R', 'L']:
        raise ValueError('It is expected that pdir is one of "R" or "L". Got {}'.format(platform_direction))

    # get our filter construction data
    cmap = _jet_wrapped(array.shape[1]/float(fill))
    # move to phase history domain
    ph_indices = int(numpy.floor(0.5*(array.shape[1] - cmap.shape[0]))) + numpy.arange(cmap.shape[0], dtype=numpy.int32)
    ph0 = numpy.fft.fftshift(numpy.fft.ifft(array, axis=1), axes=1)[:, ph_indices]

    # construct the filtered workspace
    ph0_RGB = numpy.zeros((array.shape[0], cmap.shape[0], 3), dtype=numpy.complex64)
    for i in range(3):
        ph0_RGB[:, :, i] = ph0*cmap[:, i]
    del ph0

    # Shift phase history to avoid having zeropad in middle of filter.
    # This fixes the purple sidelobe artifact.
    filter_shift = int(numpy.ceil(array.shape[1]/(4*fill)))
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


def from_reader(reader, dimension=0, row_range=None, col_range=None, index=0):
    """
    Creates a color subaperture image (csi) for the specified range from the
    file or reader object.

    Parameters
    ----------
    reader : BaseReader|str
        Reader object or file name for a reader object
    dimension : int
        Passed through to the :func:`csi_array` method.
    row_range : None|tuple|int
        Passed through to `read_chip` method of the reader object for fetching data.
        Should be `None` if `dimension = 0`.
    col_range : None|tuple|int
        Passed through to `read_chip` method of the reader object.
        Should be `None` if `dimension = 1`.
    index : int
        Passed through to `read_chip` method of the reader object.
        Used to determine which sicd/chip to use, if there are multiple.

    Returns
    -------
    numpy.ndarray
        The csi array of dtype=float64
    """

    if isinstance(reader, str):
        reader = open_complex(reader)
    if not isinstance(reader, BaseReader):
        raise TypeError('reader is required to be a path name for a sicd-type image, '
                        'or an instance of a reader object.')
    if not reader.is_sicd_type:
        raise TypeError('reader is required to be of sicd_type.')

    sicd = reader.get_sicds_as_tuple()[index]

    if sicd.SCPCOA is None or sicd.SCPCOA.SideOfTrack is None:
        logging.warning(
            'The sicd object at index {} has unpopulated SCPCOA.SideOfTrack. '
            'Defaulting to "R", which may be incorrect.')
        pdir = 'R'
    else:
        pdir = sicd.SCPCOA.SideOfTrack

    if dimension == 0:
        try:
            fill = 1/(sicd.Grid.Col.SS*sicd.Grid.Col.ImpRespBW)
        except (ValueError, AttributeError, TypeError):
            fill = 1
        if row_range is not None:
            logging.warning(
                'The csi.from_reader method is being called with dimension 0, '
                'where row_range is not None. This is probably a mistake.')
    else:
        try:
            fill = 1/(sicd.Grid.Row.SS*sicd.Grid.Row.ImpRespBW)
        except (ValueError, AttributeError, TypeError):
            fill = 1
        if row_range is not None:
            logging.warning(
                'The csi.from_reader method is being called with dimension 1, '
                'where col_range is not None. This is probably a mistake.')

    array = reader.read_chip(row_range, col_range, index=index)
    return csi_array(array, dimension=dimension, platform_direction=pdir, fill=fill)
