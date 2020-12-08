# -*- coding: utf-8 -*-
"""
Helper methods for aperture tool processing.
"""

from sarpy.processing.fft_base import fft2_sicd, ifft2_sicd, fftshift
from sarpy.processing.normalize_sicd import DeskewCalculator
import numpy
from scipy.constants.constants import speed_of_light
from sarpy.io.general.base import BaseReader


__classification__ = "UNCLASSIFIED"
__author__ = "Jason Casey"


class ApertureFilter(object):
    """
    This is a calculator for filtering SAR imagery using a subregion of complex
    fft data over a full resolution subregion of the original SAR data.

    To use this class first it should be initialized with a reader object

    """

    __slots__ = (
        '_deskew_calculator', '_sub_image_bounds', '_normalized_phase_history')

    def __init__(self, reader, dimension=1, index=0, apply_deskew=True, apply_deweighting=False):
        """

        Parameters
        ----------
        reader : BaseReader
        dimension : int
        index : int
        apply_deskew : bool
        apply_deweighting : bool
        """

        self._normalized_phase_history = None
        self._deskew_calculator = DeskewCalculator(
            reader, dimension=dimension, index=index, apply_deskew=apply_deskew, apply_deweighting=apply_deweighting)
        self._sub_image_bounds = None

    @property
    def apply_deskew(self):
        """
        bool: Apply deskew to calculated value.
        """

        return self._deskew_calculator.apply_deskew

    @apply_deskew.setter
    def apply_deskew(self, value):
        self._deskew_calculator.apply_deskew = value
        self._set_normalized_phase_history()

    @property
    def apply_deweighting(self):
        """
        bool: Apply deweighting to calculated values.
        """

        return self._deskew_calculator.apply_deweighting

    @apply_deweighting.setter
    def apply_deweighting(self, val):
        self._deskew_calculator.apply_deweighting = val
        self._set_normalized_phase_history()

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

        return fftshift(fft2_sicd(cdata, self.sicd))

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

        return ifft2_sicd(ph_data, self.sicd)

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
    def data_size(self):
        """
        None|(int, int): The feasible data size
        """

        if self._sub_image_bounds is None:
            return None
        row_bounds, col_bounds = self._sub_image_bounds
        return row_bounds[1] - row_bounds[0], col_bounds[1] - col_bounds[0]

    @dimension.setter
    def dimension(self, val):
        """
        Parameters
        ----------
        val : int

        Returns
        -------
        None
        """

        self._deskew_calculator.dimension = val
        self._set_normalized_phase_history()

    @property
    def flip_x_axis(self):
        try:
            return self.sicd.SCPCOA.SideOfTrack == "R"
        except AttributeError:
            return False

    @property
    def sub_image_bounds(self):
        """
        Tuple[Tuple[int, int]]: The sub-image bounds used for processing.
        """

        return self._sub_image_bounds

    def set_sub_image_bounds(self, row_bounds, col_bounds):
        self._sub_image_bounds = (row_bounds, col_bounds)
        self._set_normalized_phase_history()

    @property
    def normalized_phase_history(self):
        """
        None|numpy.ndarray: The normalized phase history
        """

        return self._normalized_phase_history

    def _set_normalized_phase_history(self):
        if self._sub_image_bounds is None:
            return None

        row_bounds, col_bounds = self._sub_image_bounds
        underlying_size = self._deskew_calculator.data_size
        if row_bounds[0] < 0 or row_bounds[1] > underlying_size[0]:
            raise ValueError(
                'Desired row_bounds given as {}, and underlying data size is {}'.format(row_bounds, underlying_size))
        if col_bounds[0] < 0 or col_bounds[1] > underlying_size[1]:
            raise ValueError(
                'Desired col_bounds given as {}, and underlying data size is {}'.format(col_bounds, underlying_size))

        deskewed_data = self._deskew_calculator[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
        self._normalized_phase_history = self._get_fft_complex_data(deskewed_data)

    @property
    def polar_angles(self):
        angle_width = (1 / self.sicd.Grid.Col.SS) / self.sicd.Grid.Row.KCtr
        if self.sicd.Grid.Col.KCtr:
            angle_ctr = self.sicd.Grid.Col.KCtr
        else:
            angle_ctr = 0
        angle_limits = angle_ctr + numpy.array([-1, 1]) * angle_width / 2
        if self.flip_x_axis:
            angle_limits = angle_limits[1], angle_limits[0]
        angles = numpy.linspace(angle_limits[0], angle_limits[1], self.normalized_phase_history.shape[1])
        return numpy.rad2deg(numpy.arctan(angles))

    @property
    def frequencies(self):
        """
        This returns the subaperture frequencies in units of GHz

        Returns
        -------
        numpy.array
        """
        freq_width = (1 / self.sicd.Grid.Row.SS) * (speed_of_light / 2)
        freq_ctr = self.sicd.Grid.Row.KCtr * (speed_of_light / 2)
        freq_limits = freq_ctr + (numpy.array([-1, 1]) * freq_width / 2)
        if self.sicd.PFA:
            freq_limits = freq_limits / self.sicd.PFA.SpatialFreqSFPoly[0]
        freq_limits = freq_limits/1e9
        frequencies = numpy.linspace(freq_limits[1], freq_limits[0], self.normalized_phase_history.shape[0])
        return frequencies

    def __getitem__(self, item):
        if self.normalized_phase_history is None:
            return None
        filtered_cdata = numpy.zeros(self.normalized_phase_history.shape, dtype='complex64')
        filtered_cdata[item] = self.normalized_phase_history[item]
        # do the inverse transform of this sampled portion
        return self._get_fft_phase_data(filtered_cdata)
