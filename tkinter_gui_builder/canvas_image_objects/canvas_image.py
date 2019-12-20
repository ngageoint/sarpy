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

    @abc.abstractmethod
    def init_from_fname(self,
                        fname,          # type: str
                        ):
        pass

    @abc.abstractmethod
    def get_canvas_display_image_from_full_image_subsection(self,
                                                            full_image_ul_y,
                                                            full_image_ul_x,
                                                            full_image_br_y,
                                                            full_image_br_x):
        pass

    def canvas_coords_to_full_image_coords(self,
                                           canvas_x,                # type: ndarray
                                           canvas_y,                # type: ndarray
                                           ):
        x = canvas_x * self.image_decimation_factor + self.canvas_full_image_upper_left_yx[1]
        y = canvas_y * self.image_decimation_factor + self.canvas_full_image_upper_left_yx[0]

    def update_canvas_display_image_from_full_image_subsection(self,
                                                               full_image_ul_y,
                                                               full_image_ul_x,
                                                               full_image_br_y,
                                                               full_image_br_x):
        im_data = self.get_canvas_display_image_from_full_image_subsection(full_image_ul_y,
                                                                           full_image_ul_x,
                                                                           full_image_br_y,
                                                                           full_image_br_x)
        self.update_canvas_display_image(im_data)
        self.canvas_full_image_upper_left_yx = (full_image_ul_y, full_image_ul_x)

    def update_canvas_display_image_from_full_image(self):
        ul_y = 0
        ul_x = 0
        br_y = self.full_image_ny
        br_x = self.full_image_nx
        self.update_canvas_display_image_from_full_image_subsection(ul_y, ul_x, br_y, br_x)

    def update_canvas_display_image(self,
                                    image_data,                 # type: ndarray
                                    decimation_factor,          # type: int
                                    ):
        self.canvas_display_image = image_data
        self.image_decimation_factor = decimation_factor
