import numpy
from sarpy_gui_apps.supporting_classes.complex_image_reader import ComplexImageReader


class AppVariables:
    def __init__(self):
        self.sicd_fname = str
        self.sicd_reader_object = None          # type: ComplexImageReader
        self.selected_region = None     # type: tuple
        self.fft_complex_data = None            # type: numpy.ndarray
        self.filtered_data = None           # type: numpy.ndarray
        self.fft_display_data = numpy.ndarray       # type: numpy.ndarray
        self.fft_image_bounds = None          # type: (int, int, int, int)
        self.fft_canvas_bounds = None           # type: (int, int, int, int)
        self.selected_region_complex_data = None            # type: numpy.ndarray
