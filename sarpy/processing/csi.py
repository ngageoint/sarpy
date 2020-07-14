# -*- coding: utf-8 -*-
"""
The module contains methods for computing a color subaperture image
"""

import numpy

from sarpy.io.complex.converter import open_complex as file_open
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


def mem(image, dim=0, pdir='R', fill=1):
    """
    Creates a color subaperture image (csi) from full-resolution complex data.

    Parameters
    ----------
    image : numpy.ndarray
        the complex valued SAR data in the image domain.
    dim : int
        dimension over which to split the sub-aperture, defaults to 0.
    pdir : str
        platform direction, 'RIGHT'/'R' (default) or 'LEFT'/'L'. The assumption
        is that 2nd dimension is increasing range.
    fill : int|float
        the fill factor, which defaults to 1.

    Returns
    -------
    numpy.ndarray
        The csi array, of shape `M x N x 3`, where image has shape `M x N`,
        of dtype=float64
    """

    if not (isinstance(image, numpy.ndarray) and len(image.shape) == 2 and numpy.iscomplexobj(image)):
        raise ValueError('image must be a two-dimensional numpy array of complex dtype')

    dim = int(dim)
    if dim not in [0, 1]:
        raise ValueError('dim must take value 0 or 1, got {}'.format(dim))
    if dim == 0:
        image = image.T  # this is a view

    pdir_func = pdir.upper()[0]
    if pdir_func not in ['R', 'L']:
        raise ValueError('It is expected that pdir is one of "R", "RIGHT", "L", or "LEFT". Got {}'.format(pdir))

    # move to phase history domain
    cmap = _jet_wrapped(image.shape[1]/fill)  # which axis?
    ph_indices = int(numpy.floor(0.5*(image.shape[1] - cmap.shape[0]))) + numpy.arange(cmap.shape[0], dtype=numpy.int32)
    ph0 = numpy.fft.fftshift(numpy.fft.ifft(image, axis=1), axes=1)[:, ph_indices]
    # apply the sub-aperture filters
    # ph0_RGB = ph0[:, :, numpy.newaxis]*cmap
    ph0_RGB = numpy.zeros((image.shape[0], cmap.shape[0], 3), dtype=numpy.complex64)
    for i in range(3):
        ph0_RGB[:, :, i] = ph0*cmap[:, i]
    del ph0

    # Shift phase history to avoid having zeropad in middle of filter.
    # This fixes the purple sidelobe artifact.
    filter_shift = int(numpy.ceil(image.shape[1]/(4*fill)))
    ph0_RGB[:, :, 0] = numpy.roll(ph0_RGB[:, :, 0], -filter_shift)
    ph0_RGB[:, :, 2] = numpy.roll(ph0_RGB[:, :, 2], filter_shift)
    # NB: the green band is already centered

    # FFT back to the image domain
    im0_RGB = numpy.fft.fft(numpy.fft.fftshift(ph0_RGB, axes=1), n=image.shape[1], axis=1)
    del ph0_RGB

    # Replace the intensity with the original image intensity to main full resolution
    # (in intensity, but not in color).
    scale_factor = numpy.abs(image)/numpy.abs(im0_RGB).max(axis=2)
    im0_RGB = numpy.abs(im0_RGB)*scale_factor[:, :, numpy.newaxis]

    # reorient images
    if dim == 0:
        im0_RGB = im0_RGB.transpose([1, 0, 2])
    if pdir_func == 'R':
        # reverse the color band order
        im0_RGB = im0_RGB[:, :, ::-1]
    return im0_RGB


def file(reader, dim=1, row_range=None, col_range=None, index=0):
    """
    Creates a color subaperture image (csi) for the specified range from the
    file or reader object.

    Parameters
    ----------
    reader : BaseReader|str
        Reader object or file name for a reader object
    dim : int
        passed through to the `mem` method
    row_range : none|tuple|int
        Passed through to `read_chip` method of the reader object.
    col_range : none|tuple|int
        Passed through to `read_chip` method of the reader object.
    index : int
        Passed through to `read_chip` method of the reader object.
        Used to determine which sicd/chip to use, if there are multiple.

    Returns
    -------
    numpy.ndarray
        The csi array of dtype=float64
    """

    if isinstance(reader, str):
        reader = file_open(reader)
    if not isinstance(reader, BaseReader):
        raise TypeError('reader is required to be a file name for a complex image object, '
                        'or an instance of a reader object.')

    index = int(index)
    sicd = reader.sicd_meta
    if isinstance(sicd, tuple):
        sicd = sicd[index]

    pdir = None if sicd.SCPCOA is None else sicd.SCPCOA.SideOfTrack
    if pdir is None:
        pdir = 'R'

    try:
        fill = 1/(sicd.Grid.Col.SS*sicd.Grid.Col.ImpRespBW)
    except (ValueError, AttributeError, TypeError):
        fill = 1

    image = reader.read_chip(row_range, col_range, index=index)
    return mem(image, dim=dim, pdir=pdir, fill=fill)
