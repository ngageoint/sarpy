from sarpy.processing.normalize_sicd import DeskewCalculator
from scipy.fftpack import fft2, ifft2, fftshift
import numpy


class ApertureFilter:
    """
    This is a calculator for filtering SAR imagery using a subregion of complex fft data over a full resolution
    subregion of the original SAR data
    """

    __slots__ = (
        '_reader', '_dimension', '_deskew_calculator', '_sub_image_bounds', '_normalized_phase_history', )

    def __init__(self, reader, dimension=1, index=0, apply_deweighting=False):
        self._reader = reader
        self._dimension = dimension
        self._deskew_calculator = DeskewCalculator(reader, dimension, index, apply_deweighting)
        self._sub_image_bounds = None
        self._normalized_phase_history = None

    def _get_fft_complex_data(self,
                              cdata,  # type: numpy.ndarray
                              ):
        if self._reader.sicd_meta.Grid.Col.Sgn > 0 and self._reader.sicd_meta.Grid.Row.Sgn > 0:
            # use fft2 to go from image to spatial freq
            ft_cdata = fft2(cdata)
            ft_cdata = fftshift(ft_cdata)
        else:
            # flip using ifft2
            ft_cdata = ifft2(cdata)

        ft_cdata = fftshift(ft_cdata)
        return ft_cdata

    @property
    def sub_image_bounds(self):
        return self._sub_image_bounds

    @property
    def normalized_phase_history(self):
        return self._normalized_phase_history

    def set_sub_image_bounds(self, row_bounds, col_bounds):
        self._sub_image_bounds = (row_bounds, col_bounds)
        deskewed_data = self._deskew_calculator[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
        self._normalized_phase_history = self._get_fft_complex_data(deskewed_data)

    def __getitem__(self, item):
        filtered_cdata = numpy.zeros(self._normalized_phase_history.shape, dtype=numpy.complex)
        filtered_cdata[item] = self._normalized_phase_history[item]
        filtered_cdata = fftshift(filtered_cdata)

        inverse_flag = False
        ro = self._reader
        if ro.sicd_meta.Grid.Col.Sgn > 0 and ro.sicd_meta.Grid.Row.Sgn > 0:
            pass
        else:
            inverse_flag = True

        if inverse_flag:
            cdata_clip = fft2(filtered_cdata)
        else:
            cdata_clip = ifft2(filtered_cdata)
        return cdata_clip

