from numpy import ndarray
import sarpy.io.complex as sarpy_complex
from sarpy.io.complex import Reader
from tkinter_gui_builder.canvas_image_objects.canvas_image import CanvasDisplayImage


class SarpyCanvasDisplayImage(CanvasDisplayImage):
    def __init__(self):
        super(CanvasDisplayImage, self).__init__()
        self.reader_object = None           # type: Reader

    def init_from_fname(self, fname):
        self.reader_object = sarpy_complex.open(fname)
        stop = 1

    def __init__(self):
        self.canvas_display_image = None              # type: ndarray
        self.image_decimation_factor = 1           # type: int
        self.canvas_full_image_upper_left_yx = (0, 0)  # type: (int, int)



    def get_canvas_display_image_from_full_image_subsection(self,
                                                            full_image_ul_y,
                                                            full_image_ul_x,
                                                            full_image_br_y,
                                                            full_image_br_x):
        pass
