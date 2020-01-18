import unittest

from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from tkinter_gui_builder.tests.test_utils import mouse_simulator
from tkinter_gui_builder.tests.test_utils import image_canvas_utils
import numpy as np


class ImageCanvasTests(unittest.TestCase):
    canvas_nx = 2027
    canvas_ny = 1013
    image_canvas = ImageCanvas(None)

    def test_display_image_init_w_numpy_scaled_to_fit(self):
        image_canvas = self.image_canvas
        image_data = np.zeros((self.canvas_ny, self.canvas_nx))
        image_canvas.rescale_image_to_fit_canvas = True
        image_canvas.init_with_numpy_image(image_data)
        display_image = image_canvas.variables.canvas_image_object.display_image
        display_ny, display_nx = np.shape(display_image)
        assert display_ny <= image_canvas.canvas_height
        assert display_nx <= image_canvas.canvas_width
        assert display_ny == image_canvas.canvas_height or display_nx == image_canvas.canvas_width
        print("display image is smaller or equal to the canvas size")
        print("one of the x or y dimensions of the display image matches the canvas")
        print("test passed")

    def test_display_image_init_w_numpy_not_scaled_to_fit(self):
        image_canvas = self.image_canvas
        image_data = np.zeros((self.canvas_ny, self.canvas_nx))
        image_canvas.rescale_image_to_fit_canvas = False
        image_canvas.init_with_numpy_image(image_data)
        display_image = image_canvas.variables.canvas_image_object.display_image
        display_ny, display_nx = np.shape(display_image)
        assert display_ny >= image_canvas.canvas_height or display_nx >= image_canvas.canvas_width
        print("display image is larger or equal to the canvas size")
        print("test passed")

    def test_shape_consistency_on_zoom_in_zoom_out(self):
        image_canvas = self.image_canvas
        image_data = np.zeros((self.canvas_ny, self.canvas_nx))
        image_canvas.rescale_image_to_fit_canvas = False
        image_canvas.init_with_numpy_image(image_data)
        image_canvas_utils.create_new_rect_on_image_canvas(image_canvas, 50, 50)
        rect_id = image_canvas.variables.current_shape_id
        before_zoom_image_coords = image_canvas.get_shape_image_coords(rect_id)
        before_zoom_canvas_coords = image_canvas.get_shape_canvas_coords(rect_id)
        zoom_rect = (20, 20, 100, 100)
        image_canvas.zoom_to_selection(zoom_rect, animate=False)
        after_zoom_image_coords = image_canvas.get_shape_image_coords(rect_id)
        after_zoom_canvas_coords = image_canvas.get_shape_canvas_coords(rect_id)
        assert before_zoom_canvas_coords != after_zoom_canvas_coords
        assert before_zoom_image_coords == after_zoom_image_coords
        print("canvas coords are different after zooming")
        print("image coords are the same after zooming")


if __name__ == '__main__':
    unittest.main()
