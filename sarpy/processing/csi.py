"""Compute a color subaperture image."""

from ..io import complex
import numpy as np

__classification__ = "UNCLASSIFIED"


def _jet_wrapped(m):
    """Copies functionality of the jet colormap in MATLAB.

    However, "wraps" around blue and red filters, so that filters are
    identically shaped (in a periodic sense).

    Args:
        m: size of colormap to create

    Returns:
        3 bands of color filters
    """

    # TODO: MEDIUM - I have ignored docstring here. This functionality should be cleaned up significantly.
    #   Possibly introduce matplotlib colormap choices? Provides a nice library of colormaps, and is easy to use.
    #   The choice of dimension ordering is ridiculous, so should be changed. Move the default away from jet
    #   with its famously bad luminance issues.

    n = np.ceil(float(m)/4)
    u = np.r_[(np.arange(n)+1)/n, np.ones(int(round(n))-1), np.arange(n, 0, -1)/n]
    g = int(round((m-len(u))/2)) + np.arange(len(u))
    r = g + int(n)
    b = g - int(n)
    g = g % m
    r = r % m
    b = b % m
    J = np.zeros((3, m))
    J[0, r] = J[1, g] = J[2, b] = u
    return J


def mem(im0, dim=1, pdir='right', fill=1):
    """
    Displays subaperture information as color on full resolution data.

    :param im0: complex valued SAR map in the image domain.
    :param dim: dimension over which to split subaperture.
    :param pdir: platform direction, 'right' (default) or 'left'. Assumption is that 2nd dimension is increasing range.
    :param fill: fill factor.
    :return: A 3-dimensional array of complex image data, with each band representing red, green, and blue.
    """

    # TODO: HIGH -
    #   1.) this description is not descriptive. What does this do?
    #   2.) Rename this method to a more descriptive name?
    #   3.) Coherent dimension ordering convention should be followed - this is confusing.
    #   4.) This dim argument handling should be unhacked
    #   5.) There's a bunch of native methods mixed in with numpy methods - should all be numpy methods
    #   6.) Unit test

    # Purely out of laziness, so the same code processes both dimensions
    if dim == 0:
        im0 = im0.transpose()

    # Setup the 3 band of color filters
    cmap = _jet_wrapped(int(round(im0.shape[1]/fill)))  # jet-like colormap, used for filtering
    # Move to the phase history domain
    ph0 = np.fft.fftshift(np.fft.ifft(im0, axis=1), axes=1)

    # Apply the subap filters
    ph_indices = int(np.floor((im0.shape[1] - cmap.shape[1])/2)) + np.arange(cmap.shape[1])
    ph0_RGB = np.zeros((3, im0.shape[0], len(ph_indices)), dtype=complex)
    ph0_RGB[0, :] = ph0[:, ph_indices] * cmap[0, :]  # Red
    ph0_RGB[1, :] = ph0[:, ph_indices] * cmap[1, :]  # Green
    ph0_RGB[2, :] = ph0[:, ph_indices] * cmap[2, :]  # Blue

    # Shift phase history to avoid having zeropad in middle of filter.  This
    # fixes the purple sidelobe artifact.
    filtershift = int(np.ceil(im0.shape[1]/(4*fill)))
    ph0_RGB[0, :] = np.roll(ph0_RGB[0, :], -filtershift)  # Red
    ph0_RGB[2, :] = np.roll(ph0_RGB[2, :], filtershift)  # Blue
    # Green already centered

    # FFT back to the image domain
    im0_RGB = np.fft.fft(np.fft.fftshift(ph0_RGB, axes=2), n=im0.shape[1], axis=2)

    # Replace the intensity with the original image intensity to main full resolution
    # (in intensity, but not in color).
    scale_factor = abs(im0)/abs(im0_RGB).max(0)
    im0_RGB = abs(im0_RGB)*scale_factor

    # Reorient images as necessary
    if dim == 0:
        im0_RGB = im0_RGB.transpose([0, 2, 1])
        im0 = im0.transpose()  # Since Python variables are references, return to original state
    elif pdir == 'right':
        im0_RGB[[0, 2]] = im0_RGB[[2, 0]]
    im0_RGB = im0_RGB.transpose([1, 2, 0])  # Reorient so it displays directly in plt.imshow

    return im0_RGB


def file(fname, dim=1, row_range=None, col_range=None):
    """
    Displays subaperture information as color on full resolution data.

    :param fname: filename of complex image.
    :param dim: dimension over which to split subaperture.
    :param row_range: range of row values for image. If None (default) all image rows will be processed.
    :param col_range: range of column values for image. If None(default) all image columns will be processed.
    :return: A 3-dimensional array of complex image data, with each dimension representing red, green, and blue bands.
    """

    # TODO: HIGH - unit test this thing? Just this? Just above? Both?
    #  Are both of these methods even actually user facing? naming again.

    readerObj = complex.open(fname)
    b = readerObj.read_chip(row_range, col_range)
    if readerObj.sicdmeta.SCPCOA.SideOfTrack == 'L':
        pdir = 'left'
    else:
        pdir = 'right'
    # Get metadata to calculate fill factor
    FF = 1/(readerObj.sicdmeta.Grid.Col.SS*readerObj.sicdmeta.Grid.Col.ImpRespBW)
    return mem(b, dim=dim, pdir=pdir, fill=FF)
