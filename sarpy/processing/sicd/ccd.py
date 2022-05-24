"""
The module contains methods for computing a coherent change detection from registered images
"""

import numpy
import scipy.signal

__classification__ = "UNCLASSIFIED"
__author__ = ('Thomas Mccullough',  'Wade Schwartzkopf', 'Mike Dowell')


def mem(reference_image, match_image, corr_window_size):
    """
    Performs coherent change detection, following the equation as described in
    Jakowatz, et al., "Spotlight-mode Synthetic Aperture radar: A Signal
    Processing Approach".

    .. warning: This assumes that the two arrays have already been properly
        registered with respect to one another, and all processing will proceed
        directly in memory.

    Parameters
    ----------
    reference_image : numpy.ndarray
    match_image : numpy.ndarray
    corr_window_size : int|tuple
        The correlation window size. If int, a square correlation window of
        given size will be used. If tuple, it must be a two element tuple of
        ints which describe the correlation window size.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The ccd and phase arrays
    """

    if isinstance(corr_window_size, int):
        kernel = numpy.ones((corr_window_size, corr_window_size), dtype=numpy.float32)
    elif isinstance(corr_window_size, tuple) and len(corr_window_size) == 2:
        kernel = numpy.ones(corr_window_size, dtype=numpy.float32)
    else:
        raise TypeError('corr_window_size is required to be an int or two element tuple of ints')

    inner_product = scipy.signal.convolve2d(
        numpy.conj(reference_image)*match_image, kernel, mode='same')
    # calculate magnitude of smeared reference image, accounting for numerical errors
    ref_mag = scipy.signal.convolve2d(
        reference_image.real*reference_image.real + reference_image.imag*reference_image.imag,
        kernel, mode='same')
    ref_mag[ref_mag < 0] = 0
    ref_mag = numpy.sqrt(ref_mag)
    # same for match image
    match_mag = scipy.signal.convolve2d(
        match_image.real*match_image.real + match_image.imag*match_image.imag,
        kernel, mode='same')
    match_mag[match_mag < 0] = 0
    match_mag = numpy.sqrt(match_mag)
    # perform the ccd calculation
    ccd = numpy.where((ref_mag > 0) & (match_mag > 0), inner_product/(ref_mag*match_mag), numpy.float32(0.0))
    phase = numpy.angle(inner_product)
    return ccd, phase
