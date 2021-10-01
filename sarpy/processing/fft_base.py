"""
Helper classes and methods for Fourier processing schemes.
"""

__classification__ = "UNCLASSIFIED"
__author__ = 'Thomas McCullough'

import logging

from sarpy.compliance import int_func
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.processing.ortho_rectify import FullResolutionFetcher

# NB: the below are intended as common imports from other locations
#   leave them here, even if unused
import numpy
import scipy
if scipy.__version__ < '1.4':
    # noinspection PyUnresolvedReferences
    from scipy.fftpack import fft, ifft, fftshift, ifftshift
else:
    # noinspection PyUnresolvedReferences
    from scipy.fft import fft, ifft, fftshift, ifftshift

logger = logging.getLogger(__name__)


class FFTCalculator(FullResolutionFetcher):
    """
    Base Fourier processing calculator class.

    This is intended for processing schemes where full resolution is required
    along the processing dimension, so sub-sampling along the processing
    dimension does not decrease the amount of data which must be fetched.
    """

    __slots__ = (
        '_platform_direction', '_fill')

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
        block_size : int
            The approximate processing block size to fetch, given in MB. The
            minimum value for use here will be 1.
        """

        self._platform_direction = None  # set with the index setter
        self._fill = None # set implicitly with _set_fill()
        super(FFTCalculator, self).__init__(reader, dimension=dimension, index=index, block_size=block_size)

    @property
    def dimension(self):
        # type: () -> int
        """
        int: The dimension along which to perform the color subaperture split.
        """

        return self._dimension

    @dimension.setter
    def dimension(self, value):
        value = int_func(value)
        if value not in [0, 1]:
            raise ValueError('dimension must be 0 or 1, got {}'.format(value))
        self._dimension = value
        self._set_fill()

    @property
    def index(self):
        # type: () -> int
        """
        int: The index of the reader.
        """

        return self._index

    @index.setter
    def index(self, value):
        super(FFTCalculator, self)._set_index(value)

        if self._sicd.SCPCOA is None or self._sicd.SCPCOA.SideOfTrack is None:
            logger.warning(
                'The sicd object at index {} has unpopulated SCPCOA.SideOfTrack.\n\t'
                'Defaulting to "R", which may be incorrect.')
            self._platform_direction = 'R'
        else:
            self._platform_direction = self._sicd.SCPCOA.SideOfTrack
        self._set_fill()

    @property
    def fill(self):
        # type: () -> float
        """
        float: The fill factor for the fourier processing.
        """

        return self._fill

    def _set_fill(self):
        self._fill = None
        if self._dimension is None:
            return
        if self._index is None:
            return

        if self.dimension == 0:
            try:
                fill = 1.0/(self.sicd.Grid.Row.SS*self.sicd.Grid.Row.ImpRespBW)
            except (ValueError, AttributeError, TypeError):
                fill = 1.0
        else:
            try:
                fill = 1.0/(self.sicd.Grid.Col.SS*self.sicd.Grid.Col.ImpRespBW)
            except (ValueError, AttributeError, TypeError):
                fill = 1.0
        self._fill = max(1.0, float(fill))

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

        raise NotImplementedError


def _validate_fft_input(array):
    """
    Validate the fft input.

    Parameters
    ----------
    array : numpy.ndarray

    Returns
    -------
    None
    """

    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a numpy array')
    if not numpy.iscomplexobj(array):
        raise ValueError('array must have a complex data type')
    if array.ndim != 2:
        raise ValueError('array must be a two-dimensional array. Got shape {}'.format(array.shape))


def _determine_direction(sicd, dimension):
    """
    Determine the default sign for the fft.

    Parameters
    ----------
    sicd : SICDType
    dimension : int

    Returns
    -------
    int
    """

    sgn = None
    if dimension == 0:
        try:
            sgn = sicd.Grid.Row.Sgn
        except AttributeError:
            pass
    elif dimension == 1:
        try:
            sgn = sicd.Grid.Col.Sgn
        except AttributeError:
            pass
    else:
        raise ValueError('dimension must be one of 0 or 1.')
    return -1 if sgn is None else sgn


def fft_sicd(array, dimension, sicd):
    """
    Apply the forward one-dimensional forward fft to data associated with the
    given sicd along the given dimension/axis, in accordance with the sign
    populated in the SICD structure (default is -1).

    Parameters
    ----------
    array : numpy.ndarray
        The data array, which must be two-dimensional and complex.
    dimension : int
        Must be one of 0, 1.
    sicd : SICDType
        The associated SICD structure.

    Returns
    -------
    numpy.ndarray
    """

    sgn = _determine_direction(sicd, dimension)
    return fft(array, axis=dimension) if sgn < 0 else ifft(array, axis=dimension)


def ifft_sicd(array, dimension, sicd):
    """
    Apply the inverse one-dimensional fft to data associated with the given sicd
    along the given dimension/axis.

    Parameters
    ----------
    array : numpy.ndarray
        The data array, which must be two-dimensional and complex.
    dimension : int
        Must be one of 0, 1.
    sicd : SICDType
        The associated SICD structure.

    Returns
    -------
    numpy.ndarray
    """

    sgn = _determine_direction(sicd, dimension)
    return ifft(array, axis=dimension) if sgn < 0 else fft(array, axis=dimension)


def fft2_sicd(array, sicd):
    """
    Apply the forward two-dimensional fft (i.e. both axes) to data associated with
    the given sicd.

    Parameters
    ----------
    array : numpy.ndarray
        The data array, which must be two-dimensional and complex.
    sicd : SICDType
        The associated SICD structure.

    Returns
    -------
    numpy.ndarray
    """

    return fft_sicd(fft_sicd(array, 0, sicd), 1, sicd)


def ifft2_sicd(array, sicd):
    """
    Apply the inverse two-dimensional fft (i.e. both axes) to data associated with
    the given sicd.

    Parameters
    ----------
    array : numpy.ndarray
        The data array, which must be two-dimensional and complex.
    sicd : SICDType
        The associated SICD structure.

    Returns
    -------
    numpy.ndarray
    """

    return ifft_sicd(ifft_sicd(array, 0, sicd), 1, sicd)
