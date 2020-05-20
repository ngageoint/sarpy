import numpy as np
from sarpy.geometry import point_projection
from sarpy.geometry import geocoords
from scipy.interpolate import griddata
from tkinter_gui_builder.panel_templates.image_canvas.image_canvas import ImageCanvas
from sarpy_gui_apps.supporting_classes.complex_image_reader import ComplexImageReader


class QuickOrtho:
    def __init__(self,
                 image_canvas,      # type: ImageCanvas
                 sicd_reader,       # type: ComplexImageReader
                 ):
        self.image_canvas = image_canvas
        self.sicd_reader = sicd_reader

    def create_ortho(self,
                     output_ny,
                     output_nx,
                     ):
        input_image_data = self.image_canvas.variables.canvas_image_object.display_image
        display_image_nx = input_image_data.shape[1]
        display_image_ny = input_image_data.shape[0]

        image_points = np.zeros((display_image_nx * display_image_ny, 2))
        canvas_coords_1d = np.zeros(2 * display_image_nx * display_image_ny)

        tmp_x_vals = np.arange(0, display_image_ny)
        tmp_y_vals = np.zeros(display_image_ny)
        for x in range(display_image_nx):
            start_index = display_image_ny * 2 * x + 1
            end_index = start_index + display_image_ny * 2
            canvas_coords_1d[start_index:end_index:2] = tmp_x_vals
            canvas_coords_1d[display_image_ny * x * 2::2][0:display_image_ny] = tmp_y_vals + x

        full_image_coords = self.image_canvas.variables.canvas_image_object.canvas_coords_to_full_image_yx(canvas_coords_1d)

        image_points[:, 0] = full_image_coords[0::2]
        image_points[:, 1] = full_image_coords[1::2]

        sicd_meta = self.sicd_reader.base_reader.sicd_meta
        ground_points_ecf = point_projection.image_to_ground(image_points, sicd_meta)
        ground_points_latlon = geocoords.ecf_to_geodetic(ground_points_ecf)

        world_y_coordinates = ground_points_latlon[:, 0]
        world_x_coordinates = ground_points_latlon[:, 1]

        x = np.ravel(world_x_coordinates)
        y = np.ravel(world_y_coordinates)
        z = np.ravel(np.transpose(input_image_data))

        ground_x_grid, ground_y_grid = self._create_ground_grid(min(x), max(x), min(y), max(y), output_nx, output_ny)

        ortho_image = griddata((x, y), z, (ground_x_grid, ground_y_grid), method='nearest')

        s = np.zeros((output_nx * output_ny, 3))
        s[:, 0] = ground_y_grid.ravel()
        s[:, 1] = ground_x_grid.ravel()
        s[:, 2] = ground_points_latlon[0, 2]

        s_ecf = geocoords.geodetic_to_ecf(s)

        gridded_image_pixels = point_projection.ground_to_image(s_ecf, sicd_meta)

        full_image_coords_y = full_image_coords[0::2]
        full_image_coords_x = full_image_coords[1::2]

        mask = np.ones_like(gridded_image_pixels[0][:, 0])
        indices_1 = np.where(gridded_image_pixels[0][:, 0] < min(full_image_coords_y))
        indices_2 = np.where(gridded_image_pixels[0][:, 1] < min(full_image_coords_x))
        indices_3 = np.where(gridded_image_pixels[0][:, 0] > max(full_image_coords_y))
        indices_4 = np.where(gridded_image_pixels[0][:, 1] > max(full_image_coords_x))

        mask[indices_1] = 0
        mask[indices_2] = 0
        mask[indices_3] = 0
        mask[indices_4] = 0

        mask_2d = np.reshape(mask, (output_ny, output_nx))

        return ortho_image * mask_2d

    # TODO: This should probably go into a utility somewhere as a photogrammetry helper
    @staticmethod
    def _create_ground_grid(min_x,  # type: float
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