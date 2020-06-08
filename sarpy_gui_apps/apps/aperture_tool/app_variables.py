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

        self.animation_n_frames = None      # type: int
        self.animation_aperture_faction = None      # type: int
        self.animation_frame_rate = None            # type: int
        self.animation_cycle_continuously = False       # type: bool
        self.animation_current_position = 0             # type: int
        self.animation_is_running = False           # type: bool
        self.animation_stop_pressed = False         # type: bool
        self.animation_min_aperture_percent = None      # type: float
        self.animation_max_aperture_percent = None      # type: float
