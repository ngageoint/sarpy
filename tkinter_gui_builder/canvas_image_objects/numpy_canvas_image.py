import numpy as np
from tkinter_gui_builder.canvas_image_objects.abstract_canvas_image import AbstractCanvasImage


class NumpyCanvasDisplayImage(AbstractCanvasImage):

    def __init__(self):
        self.numpy_data = None       # type: np.ndarray

    def init_from_fname_and_canvas_size(self,
                                        numpy_data,      # type: np.ndarray
                                        canvas_ny,  # type: int
                                        canvas_nx,  # type: int
                                        scale_to_fit_canvas=True,        # type: bool
                                        ):
        print("This is a special case of image object. use 'init_from_numpy_array_and_canvas_size' instead.")
        pass

    def init_from_numpy_array_and_canvas_size(self, numpy_data, canvas_ny, canvas_nx):
        self.numpy_data = numpy_data
        numpy_dims = np.shape(numpy_data)
        self.full_image_nx = numpy_dims[1]
        self.full_image_ny = numpy_dims[0]
        self.canvas_nx = canvas_nx
        self.canvas_ny = canvas_ny
        self.update_canvas_display_image_from_full_image()

    def get_decimated_image_data_in_full_image_rect(self,
                                                    full_image_rect,  # type: (int, int, int, int)
                                                    decimation,  # type: int
                                                    ):
        if decimation < 1:
            decimation = 1
        y1, x1, y2, x2 = int(full_image_rect[0]), int(full_image_rect[1]), int(full_image_rect[2]), int(full_image_rect[3])
        rect_data = self.numpy_data[y1:y2:decimation, x1:x2:decimation]
        return rect_data
