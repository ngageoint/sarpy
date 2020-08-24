# -*- coding: utf-8 -*-
"""
Helper methods for aperture tool processing.
"""

from sarpy.processing.fft_base import fft2_sicd, ifft2_sicd, fftshift
from sarpy.processing.normalize_sicd import DeskewCalculator
import numpy


__classification__ = "UNCLASSIFIED"
__author__ = "Jason Casey"


class ApertureFilter(object):
    """
    This is a calculator for filtering SAR imagery using a subregion of complex
    fft data over a full resolution subregion of the original SAR data.
    """

    __slots__ = (
        '_deskew_calculator', '_sub_image_bounds', '_normalized_phase_history', )

    def __init__(self, reader, dimension=1, index=0, apply_deweighting=False):
        self._deskew_calculator = DeskewCalculator(
            reader, dimension=dimension, index=index, apply_deweighting=apply_deweighting)
        self._sub_image_bounds = None
        self._normalized_phase_history = None

    def _get_fft_complex_data(self, cdata):
        """
        Transform the complex image data to phase history data.

        Parameters
        ----------
        cdata : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        return fftshift(ifft2_sicd(cdata, self.sicd))

    def _get_fft_phase_data(self, ph_data):
        """
        Transforms the phase history data to complex image data.

        Parameters
        ----------
        ph_data : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        # TODO: I don't see that they first perform an fftshift in the matlab?
        return fft2_sicd(fftshift(ph_data), self.sicd)

    @property
    def sicd(self):
        """
        sarpy.io.complex.sicd_elements.SICD.SICDType: The associated SICD structure.
        """

        return self._deskew_calculator.sicd

    @property
    def dimension(self):
        """
        int: The processing dimension.
        """

        return self._deskew_calculator.dimension

    @property
    def sub_image_bounds(self):
        """
        Tuple[Tuple[int, int]]: The sub-image bounds used for processing.
        """

        return self._sub_image_bounds

    @property
    def normalized_phase_history(self):
        """None|numpy.ndarray: The normalized phase history"""
        return self._normalized_phase_history

    def set_sub_image_bounds(self, row_bounds, col_bounds):
        self._sub_image_bounds = (row_bounds, col_bounds)
        deskewed_data = self._deskew_calculator[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
        self._normalized_phase_history = self._get_fft_complex_data(deskewed_data)

    def __getitem__(self, item):
        if self._normalized_phase_history is None:
            return None
        filtered_cdata = numpy.zeros(self._normalized_phase_history.shape, dtype='complex64')
        # TODO: do you really mean to pad all around with 0's and then perform?
        filtered_cdata[item] = self._normalized_phase_history[item]
        # do the inverse transform of this sampled portion
        return self._get_fft_phase_data(filtered_cdata)
