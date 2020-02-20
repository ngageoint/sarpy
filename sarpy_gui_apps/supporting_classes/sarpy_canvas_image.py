import sarpy.io.complex as sarpy_complex
from sarpy.io.complex.base import BaseReader
import sarpy.visualization.remap as remap
import sarpy.geometry.point_projection as point_projection
from tkinter_gui_builder.canvas_image_objects.abstract_canvas_image import AbstractCanvasImage
import sarpy.geometry.geocoords as geocoords
import scipy.interpolate as interp
import numpy as np


class SarpyCanvasDisplayImage(AbstractCanvasImage):
    def __init__(self):
        self.reader_object = None           # type: BaseReader
        self.remap_type = "density"         # type: str

    def init_from_fname_and_canvas_size(self,
                                        fname,      # type: str
                                        canvas_ny,  # type: int
                                        canvas_nx,  # type: int
                                        scale_to_fit_canvas=False,      # type: bool
                                        ):
        self.fname = fname
        self.reader_object = sarpy_complex.open(fname)
        self.full_image_nx = self.reader_object.sicd_meta.ImageData.FullImage.NumCols
        self.full_image_ny = self.reader_object.sicd_meta.ImageData.FullImage.NumRows
        self.canvas_nx = canvas_nx
        self.canvas_ny = canvas_ny
        self.scale_to_fit_canvas = scale_to_fit_canvas
        self.update_canvas_display_image_from_full_image()

    def get_decimated_image_data_in_full_image_rect(self,
                                                    full_image_rect,  # type: (int, int, int, int)
                                                    decimation,  # type: int
                                                    ):
        if decimation < 1:
            decimation = 1
        y1, x1, y2, x2 = full_image_rect[0], full_image_rect[1], full_image_rect[2], full_image_rect[3]
        cdata = self.reader_object.read_chip( (y1, y2, decimation), (x1, x2, decimation))
        decimated_image_data = self.remap_complex_data(cdata)
        return decimated_image_data

    def remap_complex_data(self,
                           complex_data,    # type: np.ndarray
                           ):
        if self.remap_type == 'density':
            pix = remap.density(complex_data)
        elif self.remap_type == 'brighter':
            pix = remap.brighter(complex_data)
        elif self.remap_type == 'darker':
            pix = remap.darker(complex_data)
        elif self.remap_type == 'highcontrast':
            pix = remap.highcontrast(complex_data)
        elif self.remap_type == 'linear':
            pix = remap.linear(complex_data)
        elif self.remap_type == 'log':
            pix = remap.log(complex_data)
        elif self.remap_type == 'pedf':
            pix = remap.pedf(complex_data)
        elif self.remap_type == 'nrl':
            pix = remap.nrl(complex_data)
        return pix

    def set_remap_type(self,
                       remap_type,          # type: str
                       ):
        self.remap_type = remap_type

    def create_ortho(self,
                     output_ny,
                     output_nx,
                     ):

        input_image_data = self.display_image
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

        full_image_coords = self.canvas_coords_to_full_image_yx(canvas_coords_1d)

        image_points[:, 0] = full_image_coords[0::2]
        image_points[:, 1] = full_image_coords[1::2]

        sicd_meta = self.reader_object.sicd_meta
        ground_points_ecf = point_projection.image_to_ground(image_points, sicd_meta)
        ground_points_latlon = geocoords.ecf_to_geodetic(ground_points_ecf)

        world_y_coordinates = ground_points_latlon[:, 0]
        world_x_coordinates = ground_points_latlon[:, 1]

        x = np.ravel(world_x_coordinates)
        y = np.ravel(world_y_coordinates)
        z = np.ravel(np.transpose(input_image_data))

        ground_x_grid, ground_y_grid = self._create_ground_grid(min(x), max(x), min(y), max(y), output_nx, output_ny)

        ortho_image = interp.griddata((x, y), z, (ground_x_grid, ground_y_grid), method='nearest')

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