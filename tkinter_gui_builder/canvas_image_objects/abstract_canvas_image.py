from numpy import ndarray
from six import add_metaclass
import abc
from typing import Union
import PIL.Image
import numpy as np


@add_metaclass(abc.ABCMeta)
class AbstractCanvasImage:
    canvas_decimated_image = None  # type: ndarray
    display_image_scaled_to_fit = None  # type: ndarray
    fname = None  # type: str
    true_decimation_factor = 1  # type: int
    display_rescaling_factor = 1  # type: float
    full_image_nx = None  # type: int
    full_image_ny = None  # type: int
    canvas_full_image_upper_left_yx = (0, 0)  # type: (int, int)
    canvas_ny = None
    canvas_nx = None
    scale_to_fit_canvas = False

    @abc.abstractmethod
    def get_true_decimated_image_data_in_full_image_rect(self,
                                                         full_image_rect,  # type: (int, int, int, int)
                                                         decimation,  # type: int
                                                         ):
        pass

    @abc.abstractmethod
    def init_from_fname_and_canvas_size(self,
                                        fname,  # type: str
                                        canvas_ny,      # type: int
                                        canvas_nx,      # type: int
                                        scale_to_fit_canvas=False,        # type: bool
                                        ):
        """
        All subclassed methods of this should set an image reader object, self.canvas_ny, and self.canvas_nx, then call this
        using super()
        :param fname:
        :param canvas_ny:
        :param canvas_nx:
        :param scale_to_fit_canvas:
        :return:
        """
        pass

    def get_scaled_display_data(self, decimated_image):
        scale_factor = self.compute_display_scale_factor(decimated_image)
        new_nx = int(decimated_image.shape[1] * scale_factor)
        new_ny = int(decimated_image.shape[0] * scale_factor)
        if new_nx > self.canvas_nx:
            new_nx = self.canvas_nx
        if new_ny > self.canvas_ny:
            new_ny = self.canvas_ny
        pil_image = PIL.Image.fromarray(decimated_image)
        display_image = pil_image.resize((new_nx, new_ny))
        return np.array(display_image)

    def compute_display_scale_factor(self, decimated_image):
        decimated_image_nx = decimated_image.shape[1]
        decimated_image_ny = decimated_image.shape[0]
        scale_factor_1 = self.canvas_nx / decimated_image_nx
        scale_factor_2 = self.canvas_ny / decimated_image_ny
        scale_factor = np.min((scale_factor_1, scale_factor_2))
        return scale_factor

    def get_true_decimated_image_data_in_canvas_rect(self,
                                                     canvas_rect,  # type: (int, int, int, int)
                                                     decimation=None,  # type: int
                                                     ):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        if decimation is None:
            decimation = self.get_true_decimation_from_canvas_rect(canvas_rect)
        return self.get_true_decimated_image_data_in_full_image_rect(full_image_rect, decimation)

    def update_canvas_display_image_from_full_image(self):
        full_image_rect = (0, 0, self.full_image_ny, self.full_image_nx)
        self.update_canvas_display_image_from_full_image_rect(full_image_rect)

    def update_canvas_display_image_from_full_image_rect(self, full_image_rect):
        self.set_decimation_from_full_image_rect(full_image_rect)
        true_decimated_im_data = self.get_true_decimated_image_data_in_full_image_rect(full_image_rect, self.true_decimation_factor)
        self.update_canvas_display_from_numpy_array(true_decimated_im_data)
        self.canvas_full_image_upper_left_yx = (full_image_rect[0], full_image_rect[1])

    def update_canvas_display_image_from_canvas_rect(self, canvas_rect):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        self.update_canvas_display_image_from_full_image_rect(full_image_rect)

    def update_canvas_display_from_numpy_array(self,
                                               image_data,  # type: ndarray
                                               ):
        self.canvas_decimated_image = image_data
        if self.scale_to_fit_canvas:
            scale_factor = self.compute_display_scale_factor(image_data)
            self.display_rescaling_factor = scale_factor
            self.display_image_scaled_to_fit = self.get_scaled_display_data(image_data)

    def get_true_decimation_factor_from_full_image_rect(self, full_image_rect):
        display_decimation_factor = self.get_display_decimation_factor_from_full_image_rect(full_image_rect)
        true_decimation_factor = int(display_decimation_factor)
        if true_decimation_factor < 1:
            true_decimation_factor = 1
        return true_decimation_factor

    def get_display_decimation_factor_from_full_image_rect(self, full_image_rect):
        ny = full_image_rect[2] - full_image_rect[0]
        nx = full_image_rect[3] - full_image_rect[1]
        decimation_y = ny / self.canvas_ny
        decimation_x = nx / self.canvas_nx
        decimation_factor = max(decimation_y, decimation_x)
        return decimation_factor

    def get_true_decimation_from_canvas_rect(self, canvas_rect):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        return self.get_true_decimation_factor_from_full_image_rect(full_image_rect)

    def get_display_decimation_from_canvas_rect(self, canvas_rect):
        full_image_rect = self.canvas_rect_to_full_image_rect(canvas_rect)
        return self.get_display_decimation_factor_from_full_image_rect(full_image_rect)

    def set_decimation_from_full_image_rect(self, full_image_rect):
        true_decimation_factor = self.get_true_decimation_factor_from_full_image_rect(full_image_rect)
        display_decimation_factor = self.get_display_decimation_factor_from_full_image_rect(full_image_rect)
        self.true_decimation_factor = true_decimation_factor
        self.display_rescaling_factor = display_decimation_factor

    def canvas_coords_to_full_image_yx(self,
                                       canvas_coords,       # type: [int]
                                       ):
        x_coords = canvas_coords[0::2]
        y_coords = canvas_coords[1::2]
        xy_coords = zip(x_coords, y_coords)
        image_yx_coords = []
        # TODO: this can be optimized for speed
        for xy in xy_coords:
            decimation_factor = self.true_decimation_factor
            if self.scale_to_fit_canvas:
                decimation_factor = decimation_factor / self.display_rescaling_factor
            image_x = xy[0] * decimation_factor + self.canvas_full_image_upper_left_yx[1]
            image_y = xy[1] * decimation_factor + self.canvas_full_image_upper_left_yx[0]
            image_yx_coords.append(image_y)
            image_yx_coords.append(image_x)
        return image_yx_coords

    def canvas_rect_to_full_image_rect(self,
                                       canvas_rect,  # type: (int, int, int, int)
                                       ):            # type: (...) ->[float]
        image_y1, image_x1 = self.canvas_coords_to_full_image_yx((canvas_rect[0], canvas_rect[1]))
        image_y2, image_x2 = self.canvas_coords_to_full_image_yx((canvas_rect[2], canvas_rect[3]))

        if image_x1 < 0:
            image_x1 = 0
        if image_y1 < 0:
            image_y1 = 0
        if image_x2 > self.full_image_nx:
            image_x2 = self.full_image_nx
        if image_y2 > self.full_image_ny:
            image_y2 = self.full_image_ny

        return image_y1, image_x1, image_y2, image_x2

    def full_image_yx_to_canvas_coords(self,
                                       full_image_yx,           # type: Union[(int, int), list]
                                       ):                       # type: (...) -> Union[(int, int), list]
        y_coords = full_image_yx[0::2]
        x_coords = full_image_yx[1::2]
        image_yx_coords = zip(y_coords, x_coords)
        canvas_xy_coords = []
        decimation_factor = self.true_decimation_factor
        if self.scale_to_fit_canvas:
            decimation_factor = decimation_factor / self.display_rescaling_factor
        for image_yx in image_yx_coords:
            canvas_x = (image_yx[1] - self.canvas_full_image_upper_left_yx[1]) / decimation_factor
            canvas_y = (image_yx[0] - self.canvas_full_image_upper_left_yx[0]) / decimation_factor
            canvas_xy_coords.append(canvas_x)
            canvas_xy_coords.append(canvas_y)
        return canvas_xy_coords
