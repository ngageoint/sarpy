import unittest

from tkinter_gui_builder.panel_templates.image_canvas_panel.image_canvas import ImageCanvas
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
        print("")
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
        print("")
        print("display image is larger or equal to the canvas size")
        print("test passed")

    def test_image_in_rect_after_zoom(self):
        image_canvas = self.image_canvas
        image_data = np.random.random((self.canvas_ny, self.canvas_nx))
        image_canvas.rescale_image_to_fit_canvas = False
        image_canvas.init_with_numpy_image(image_data)
        full_image_decimation = self.image_canvas.variables.canvas_image_object.decimation_factor
        image_canvas_utils.create_new_rect_on_image_canvas(image_canvas, 50, 50)
        rect_id = image_canvas.variables.current_shape_id
        image_canvas.modify_existing_shape_using_canvas_coords(rect_id, (50, 50, 300, 200), update_pixel_coords=True)
        before_zoom_image_in_rect = image_canvas.get_image_data_in_canvas_rect_by_id(rect_id)
        zoom_rect = (20, 20, 100, 100)
        image_canvas.zoom_to_selection(zoom_rect, animate=False)
        zoomed_image_decimation = self.image_canvas.variables.canvas_image_object.decimation_factor
        after_zoom_image_in_rect = image_canvas.get_image_data_in_canvas_rect_by_id(rect_id)
        assert (after_zoom_image_in_rect == before_zoom_image_in_rect).all()
        assert full_image_decimation != zoomed_image_decimation
        print("")
        print("decimation factors at zoomed out and zoomed in levels are different.")
        print("getting image data in rect is consistent at both zoomed out and zoomed in views")
        print("test passed.")


if __name__ == '__main__':
    unittest.main()
