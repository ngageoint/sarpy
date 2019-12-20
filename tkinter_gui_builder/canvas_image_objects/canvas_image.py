import numpy as np
from numpy import ndarray
from six import add_metaclass
import abc


@add_metaclass(abc.ABCMeta)
class CanvasDisplayImage:
    def __init__(self):
        self.canvas_display_image = None                 # type: ndarray
        self.image_decimation_factor = 1           # type: int
        self.full_image_nx = None                   # type: int
        self.full_image_ny = None                        # type: int
        self.canvas_full_image_upper_left_yx = (0, 0)  # type: (int, int)
        self.canvas_ny = None
        self.canvas_nx = None

    @abc.abstractmethod
    def init_from_fname_and_canvas_size(self,
                                        fname,  # type: str
                                        canvas_ny,      # type: int
                                        canvas_nx,      # type: int
                                        ):
        pass

    def get_canvas_display_image_from_full_image_subsection(self, full_image_rect):
        pass

    def canvas_coords_to_full_image_coords(self,
                                           canvas_x,                # type: ndarray
                                           canvas_y,                # type: ndarray
                                           ):
        x = canvas_x * self.image_decimation_factor + self.canvas_full_image_upper_left_yx[1]
        y = canvas_y * self.image_decimation_factor + self.canvas_full_image_upper_left_yx[0]

    def update_canvas_display_image_from_full_image_subsection(self, full_image_rect):
        im_data = self.get_canvas_display_image_from_full_image_subsection(full_image_rect)
        self.update_canvas_display_from_numpy_array(im_data)
        self.set_decimation_from_full_image_rect(full_image_rect)
        self.canvas_full_image_upper_left_yx = (full_image_rect[0], full_image_rect[1])

    def update_canvas_display_image_from_canvas_rect(self, canvas_rect):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        self.update_canvas_display_image_from_full_image_subsection(full_image_rect)

    def update_canvas_display_image_from_full_image(self):
        full_image_rect = (0, 0, self._full_image_ny, self.full_image_nx)
        self.update_canvas_display_image_from_full_image_subsection(full_image_rect)

    def update_canvas_display_from_numpy_array(self,
                                               image_data,  # type: ndarray
                                               ):
        self.canvas_display_image = image_data

    def get_decimation_from_full_image_rect(self, full_image_rect):
        ny = full_image_rect[2] - full_image_rect[0]
        nx = full_image_rect[3] - full_image_rect[1]
        decimation_y = ny / self.canvas_ny
        decimation_x = nx / self.canvas_nx
        decimation_factor = int(min(decimation_y, decimation_x))
        return decimation_factor

    def get_decimation_from_full_canvas_rect(self, canvas_rect):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        self.get_decimation_from_full_image_rect(full_image_rect)

    def set_decimation_from_full_image_rect(self, full_image_rect):
        dec_factor = self.get_decimation_from_full_image_rect(full_image_rect)
        self.image_decimation_factor = dec_factor

    def canvas_rect_to_full_image_rect(self,
                                       canvas_rect,  # type: (int, int, int, int)
                                       ):
        y1 = canvas_rect[0]
        x1 = canvas_rect[1]
        y2 = canvas_rect[2]
        x2 = canvas_rect[3]
        image_x1 = x1 * self.image_decimation_factor + self.canvas_full_image_upper_left_yx[1]
        image_x2 = x2 * self.image_decimation_factor + self.canvas_full_image_upper_left_yx[1]
        image_y1 = y1 * self.image_decimation_factor + self.canvas_full_image_upper_left_yx[0]
        image_y2 = y2 * self.image_decimation_factor + self.canvas_full_image_upper_left_yx[0]
        return image_y1, image_x1, image_y2, image_x2