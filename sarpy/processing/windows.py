"""
Window function definitions and a few helper functions. This just passes through
to scipy functions after managing scipy version dependent import structure.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import scipy
import numpy
from scipy.optimize import newton


_version_string_parts = scipy.__version__.split('.')
_version = (int(_version_string_parts[0]), int(_version_string_parts[1]))


############
# some basic window function definitions

if _version >= (1, 1):
    # noinspection PyUnresolvedReferences
    from scipy.signal.windows import general_hamming as _general_hamming, \
        kaiser as _kaiser
else:
    _general_hamming = None
    # noinspection PyUnresolvedReferences
    from scipy.signal import kaiser as _kaiser

if _version >= (1, 6):
    # noinspection PyUnresolvedReferences
    from scipy.signal.windows import taylor as _taylor
else:
    _taylor = None


def general_hamming(M, alpha, sym=True):
    r"""
    Returns a generalized hamming function. Constructed (non-symmetric) as
    :math:`\alpha - (1-\alpha)\cos\left(\frac{2\pi n}{M-1}\right) 0\leq n \leq M-1`

    Parameters
    ----------
    M : int
        Number of points in the output window.
    alpha : float
        The window coefficient.
    sym : bool
        When `True` (default), generates a symmetric window, for use in filter
        design. When `False`, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    numpy.ndarray
    """

    if _general_hamming is not None:
        return _general_hamming(M, alpha, sym=sym)

    if (M % 2) == 0:
        k = int(M / 2)
    else:
        k = int((M + 1) / 2)
    theta = 2 * numpy.pi * numpy.arange(k) / (M - 1)

    weights = numpy.zeros((M,), dtype=numpy.float64)
    if sym:
        weights[:k] = (alpha - (1 - alpha) * numpy.cos(theta))
        weights[k:] = weights[k - 1::-1]
    else:
        weights[:k] = (alpha - (1 - alpha) * numpy.cos(theta))[::-1]
        weights[k:] = weights[k - 1::-1]
    return weights


def hamming(M, sym=True):
    """
    The hamming window, which is a general hamming window with alpha=0.54.

    Parameters
    ----------
    M : int
        Number of points in the output window.
    sym : bool
        When `True` (default), generates a symmetric window, for use in filter
        design. When `False`, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    numpy.ndarray
    """

    return general_hamming(M, 0.54, sym=sym)


def hanning(M, sym=True):
    """
    The hanning or hann window, which is a general hamming window with alpha=0.5.

    Parameters
    ----------
    M : int
        Number of points in the output window.
    sym : bool
        When `True` (default), generates a symmetric window, for use in filter
        design. When `False`, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    numpy.ndarray
    """

    return general_hamming(M, 0.5, sym=sym)


def taylor(M, nbar=4, sll=-30, norm=True, sym=True):
    """
    The Taylor window taper function approximates the Dolph-Chebyshev windows
    constant sidelobe level for a parameterized number of near-in sidelobes,
    but then allows a taper beyond.

    The SAR (synthetic aperature radar) community commonly uses Taylor weighting
    for image formation processing because it provides strong, selectable sidelobe
    suppression with minimum broadening of the mainlobe.

    Parameters
    ----------
    M : int
        Number of points in the output window.
    nbar : int
        Number of nearly constant level sidelobes adjacent to the mainlobe.
    sll : float
        Desired suppression of sidelobe level in decibels (dB) relative to the
        DC gain of the mainlobe. This should be a positive number.
    norm : bool
        When `True` (default), divides the window by the largest (middle) value
        for odd-length windows or the value that would occur between the two
        repeated middle values for even-length windows such that all values are
        less than or equal to 1. When `False` the DC gain will remain at 1 (0 dB)
        and the sidelobes will be sll dB down.
    sym : bool
        When `True` (default), generates a symmetric window, for use in filter
        design. When `False`, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    numpy.ndarray
    """

    if _taylor is not None:
        if sll < 0:
            sll *= -1
        return _taylor(M, nbar=nbar, sll=sll, norm=norm, sym=sym)

    if sll > 0:
        sll *= -1

    a = numpy.arccosh(10**(-sll/20.))/numpy.pi
    # Taylor pulse widening (dilation) factor
    sp2 = (nbar*nbar)/(a*a + (nbar-0.5)*(nbar-0.5))
    # the angular space in n points
    xi = numpy.linspace(-numpy.pi, numpy.pi, M)
    # calculate the cosine weights
    out = numpy.ones((M, ), dtype=numpy.float64)  # the "constant" term
    max_value = 1.0

    coefs = numpy.arange(1, nbar)
    sgn = 1

    for m in coefs:
        coefs1 = (coefs - 0.5)
        coefs2 = coefs[coefs != m]
        numerator = numpy.prod(1 - (m*m)/(sp2*(a*a + coefs1*coefs1)))
        denominator = numpy.prod(1 - (m*m)/(coefs2*coefs2))
        out += sgn*(numerator/denominator)*numpy.cos(m*xi)
        max_value += sgn*(numerator/denominator)
        sgn *= -1

    if not sym:
        k = int(M/2)
        l = M-k
        out2 = numpy.empty((M, ), dtype='float64')
        out2[:k] = out[l:]
        out2[k:] = out[:l]
        out = out2

    if norm:
        out /= max_value
    return out


def kaiser(M, beta, sym=True):
    """
    Return a Kaiser window, which is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window.
    beta : float
        Shape parameter, determines trade-off between main-lobe width and side
        lobe level. As beta gets large, the window narrows.
    sym : bool
        When `True` (default), generates a symmetric window, for use in filter
        design. When `False`, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    numpy.ndarray
    """

    return _kaiser(M, beta, sym=sym)


#################
# helper methods

def hamming_ipr(x, a):
    """
    Evaluate the Hamming impulse response function over the given array.

    Parameters
    ----------
    x : numpy.ndarray|float|int
    a : float
        The Hamming parameter value.

    Returns
    -------
    numpy.ndarray
    """

    return a*numpy.sinc(x) + 0.5*(1-a)*(numpy.sinc(x-1) + numpy.sinc(x+1)) - a/numpy.sqrt(2)


def get_hamming_broadening_factor(coef):
    test_array = numpy.linspace(0.3, 2.5, 100)
    values = hamming_ipr(test_array, coef)
    init_value = test_array[numpy.argmin(numpy.abs(values))]
    zero = newton(hamming_ipr, init_value, args=(coef,), tol=1e-12, maxiter=100)
    return 2 * zero


def find_half_power(wgt_funct, oversample=1024):
    """
    Find the half power point of the impulse response function.

    Parameters
    ----------
    wgt_funct : None|numpy.ndarray
    oversample : int

    Returns
    -------
    None|float
    """

    if wgt_funct is None:
        return None

    # solve for the half-power point in an oversampled impulse response
    impulse_response = numpy.abs(numpy.fft.fft(wgt_funct, wgt_funct.size*oversample))/numpy.sum(wgt_funct)
    ind = numpy.flatnonzero(impulse_response < 1 / numpy.sqrt(2))[0]
    # find first index with less than half power,
    # then linearly interpolate to estimate 1/sqrt(2) crossing
    v0 = impulse_response[ind - 1]
    v1 = impulse_response[ind]
    zero_ind = ind - 1 + (1./numpy.sqrt(2) - v0)/(v1 - v0)
    return 2*zero_ind/oversample
