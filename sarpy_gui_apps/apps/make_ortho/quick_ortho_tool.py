import tkinter
from tkinter.filedialog import askopenfilename
from sarpy_gui_apps.apps.make_ortho.panels.ortho_button_panel import OrthoButtonPanel
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from sarpy_gui_apps.supporting_classes.sarpy_canvas_image import SarpyCanvasDisplayImage
from tkinter_gui_builder.panel_templates.widget_panel.widget_panel import AbstractWidgetPanel
import sarpy.geometry.point_projection as point_projection
import sarpy.geometry.geocoords as geocoords
import scipy.interpolate as interp
import numpy as np
import os


class AppVariables:
    def __init__(self):
        self.fname = "None"       # type: str
        self.remap_type = "density"


class Ortho(AbstractWidgetPanel):
    button_panel = OrthoButtonPanel         # type: TaserButtonPanel
    raw_frame_image_panel = ImageCanvas     # type: ImageCanvas
    ortho_image_panel = ImageCanvas         # type: ImageCanvas

    def __init__(self, master):
        master_frame = tkinter.Frame(master)
        AbstractWidgetPanel.__init__(self, master_frame)
        self.variables = AppVariables()

        widget_list = ["button_panel", "raw_frame_image_panel", "ortho_image_panel"]
        self.init_w_horizontal_layout(widget_list)

        # define panels widget_wrappers in master frame
        self.button_panel.set_spacing_between_buttons(0)
        self.raw_frame_image_panel.variables.canvas_image_object = SarpyCanvasDisplayImage()  # type: SarpyCanvasDisplayImage
        self.raw_frame_image_panel.set_canvas_size(500, 400)
        self.raw_frame_image_panel.rescale_image_to_fit_canvas = True
        self.ortho_image_panel.set_canvas_size(500, 400)
        self.ortho_image_panel.rescale_image_to_fit_canvas = True

        # need to pack both master frame and self, since this is the main app window.
        master_frame.pack()
        self.pack()

        # bind events to callbacks here
        self.button_panel.fname_select.on_left_mouse_click(self.callback_initialize_canvas_image)
        self.button_panel.pan.on_left_mouse_click(self.callback_set_to_pan)
        self.button_panel.display_ortho.on_left_mouse_click(self.callback_display_ortho_image)

    def callback_set_to_pan(self, event):
        self.raw_frame_image_panel.set_current_tool_to_pan()
        self.raw_frame_image_panel.hide_shape(self.raw_frame_image_panel.variables.zoom_rect_id)

    def callback_initialize_canvas_image(self, event):
        image_file_extensions = ['*.nitf', '*.NITF']
        ftypes = [
            ('image files', image_file_extensions),
            ('All files', '*'),
        ]
        new_fname = askopenfilename(initialdir=os.path.expanduser("~"), filetypes=ftypes)
        if new_fname:
            self.variables.fname = new_fname
            self.raw_frame_image_panel.init_with_fname(self.variables.fname)

    def callback_display_ortho_image(self, event):
        canvas_image_object = self.raw_frame_image_panel.variables.canvas_image_object
        display_image_data = canvas_image_object.display_image
        display_image_nx = display_image_data.shape[1]
        display_image_ny = display_image_data.shape[0]
        sicd_meta = self.raw_frame_image_panel.variables.canvas_image_object.reader_object.sicdmeta

        image_points = np.zeros((display_image_nx * display_image_ny, 2))
        canvas_coords_1d = np.zeros(2*display_image_nx*display_image_ny)
        # TODO: replace this terribleness with something faster
        counter_2 = 0
        for x in range(display_image_nx):
            for y in range(display_image_ny):
                canvas_coords_1d[counter_2] = x
                canvas_coords_1d[counter_2+1] = y
                counter_2 = counter_2 + 2

        full_image_coords = canvas_image_object.canvas_coords_to_full_image_yx(canvas_coords_1d)

        image_points[:, 0] = full_image_coords[0::2]
        image_points[:, 1] = full_image_coords[1::2]

        ground_points_ecf = point_projection.image_to_ground(image_points, sicd_meta)
        ground_points_latlon = geocoords.ecf_to_geodetic(ground_points_ecf)

        world_y_coordinates = ground_points_latlon[:, 0]
        world_x_coordinates = ground_points_latlon[:, 1]

        x = np.ravel(world_x_coordinates)
        y = np.ravel(world_y_coordinates)

        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)

        ground_x_grid, ground_y_grid = self.create_ground_grid(min_x, max_x, min_y, max_y, canvas_image_object.canvas_nx, canvas_image_object.canvas_ny)
        ground_x_grid_1d = ground_x_grid.ravel()
        ground_y_grid_1d = ground_y_grid.ravel()
        height_1d = ground_x_grid_1d * 0

        s = np.zeros((len(ground_x_grid_1d), 3))
        s[:, 0] = ground_y_grid_1d
        s[:, 1] = ground_x_grid_1d
        s[:, 2] = height_1d

        s_ecf = geocoords.geodetic_to_ecf(ground_y_grid_1d, ground_x_grid_1d, height_1d)

        s_ecf_3 = np.zeros((len(ground_x_grid_1d), 3))
        s_ecf_3[:, 0] = s_ecf[0][0]
        s_ecf_3[:, 1] = s_ecf[1][0]
        #
        # gridded_image_pixels = point_projection.ground_to_image(s_ecf, sicd_meta)
        # gridded_canvas_pixels = canvas_image_object.full_image_yx_to_canvas_coords(gridded_image_pixels)

        orthod_image = self.create_ortho(display_image_data, ground_points_latlon, canvas_image_object.canvas_ny, canvas_image_object.canvas_nx)
        self.ortho_image_panel.init_with_numpy_image(orthod_image)

    def create_ortho(self,
                     input_image_data,
                     ground_points_latlon,
                     output_ny,
                     output_nx,
                     ):  # type: (...) -> GeotiffImage

        world_y_coordinates = ground_points_latlon[:, 0]
        world_x_coordinates = ground_points_latlon[:, 1]

        x = np.ravel(world_x_coordinates)
        y = np.ravel(world_y_coordinates)
        z = np.ravel(np.transpose(input_image_data))

        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)

        ground_x_grid, ground_y_grid = self.create_ground_grid(min_x, max_x, min_y, max_y, output_nx, output_ny)

        zi = interp.griddata((x, y), z, (ground_x_grid, ground_y_grid), method='nearest')
        return zi

    @staticmethod
    def create_ground_grid(min_x,  # type: float
                           max_x,  # type: float
                           min_y,  # type: float
                           max_y,  # type: float
                           npix_x,  # type: int
                           npix_y,  # type: int
                           ):  # type: (...) -> (ndarray, ndarray)
        ground_y_arr, ground_x_arr = np.mgrid[0:npix_y, 0:npix_x]
        ground_x_arr = ground_x_arr / npix_x * (max_x - min_x)
        ground_y_arr = (ground_y_arr - npix_y) * -1
        ground_y_arr = ground_y_arr / npix_y * (max_y - min_y)
        ground_x_arr = ground_x_arr + min_x
        ground_y_arr = ground_y_arr + min_y
        x_gsd = np.abs(ground_x_arr[0, 1] - ground_x_arr[0, 0])
        y_gsd = np.abs(ground_y_arr[0, 0] - ground_y_arr[1, 0])
        return ground_x_arr + x_gsd / 2.0, ground_y_arr - y_gsd / 2.0


if __name__ == '__main__':
    root = tkinter.Tk()
    app = Ortho(root)
    root.mainloop()
