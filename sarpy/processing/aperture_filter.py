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


if __name__ == '__main__':
    import os
    from sarpy.io.complex.sicd import SICDReader
    from sarpy.visualization.remap import density
    from matplotlib import pyplot
    from numpy.polynomial import polynomial
    from sarpy.processing.fft_base import fft2_sicd, fftshift

    fname = os.path.expanduser('~/Desktop/sarpy_testing/sicd/sicd_example_RMA_RGZERO_RE16I_IM16I.nitf')
    reader = SICDReader(fname)
    # extract a test swath
    test_data = reader[300:800, 500:1000]

    old = False

    if old:
        from sarpy.processing.normalize_sicd import deskewparams, deskewmem
        # old version
        DeltaKCOAPoly_orig, rg_coords_m, az_coords_m, fft_sgn = deskewparams(reader.sicd_meta, 1)
        test_deskew_data, DeltaKCOAPoly_new = deskewmem(
            test_data, DeltaKCOAPoly_orig, rg_coords_m[300:800], az_coords_m[500:1000], 1, fft_sgn=fft_sgn)
        # do the uniform shift for the other dimension
        row_mid = rg_coords_m[300 + int((800-300)/2) - 1]
        col_mid = az_coords_m[500 + int((1000-500)/2) - 1]
        delta_kcoa_new_const = numpy.zeros((1, 1), dtype='float64')
        delta_kcoa_new_const[0, 0] = polynomial.polyval2d(row_mid, col_mid, DeltaKCOAPoly_new)
        # apply this uniform shift
        test_deskew_data, junk = deskewmem(test_deskew_data, delta_kcoa_new_const, rg_coords_m[300:800], az_coords_m[500:1000], 0, fft_sgn=reader.sicd_meta.Grid.Row.Sgn)
        test_phd = fftshift(fft2_sicd(test_deskew_data, reader.sicd_meta))

    # new version
    deskew_calc = DeskewCalculator(reader, dimension=1, index=0, apply_deskew=True, apply_deweighting=True)
    test_deskew_data2 = deskew_calc[300:800, 500:1000]
    # let's check the two phase histories
    test_phd2 = fftshift(fft2_sicd(test_deskew_data2, reader.sicd_meta))
    # let's check the aperture filter implementation
    ap_filter = ApertureFilter(reader, dimension=1, index=0, apply_deskew=True, apply_deweighting=True)
    ap_filter.set_sub_image_bounds((300, 800), (500, 1000))
    test_phd2_1 = ap_filter.normalized_phase_history

    diff_magnitude = numpy.abs(test_phd2 - test_phd2_1)
    print(numpy.mean(diff_magnitude), numpy.max(diff_magnitude), numpy.std(diff_magnitude))

    fig, axs = pyplot.subplots(nrows=2, ncols=1)
    axs[0].imshow(density(test_phd2_1))
    axs[1].imshow(density(test_deskew_data2))
    pyplot.show()
