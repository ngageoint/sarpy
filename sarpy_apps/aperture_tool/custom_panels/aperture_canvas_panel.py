from tkinter_gui_builder.panel_templates.basic_image_canvas_panel import BasicImageCanvasPanel
import numpy as np
import sarpy.io.complex as sarpy_complex
import sarpy.visualization.remap as remap


class TaserImageCanvasPanel(BasicImageCanvasPanel):
    def __init__(self, master):
        BasicImageCanvasPanel.__init__(self, master)
        self.decimation = 1
        self.reader_object = None
        self.remap_type = "density"
        self.cdata = None

    def set_image_from_fname(self,
                             fname,  # type: str
                             remap_type='density',      # type: str
                             ):
        self.reader_object = sarpy_complex.open(fname)
        self.read_image()

    def update_display_image(self,
                             remap_type,            # type: str
                             ):
        display_data = self.remap_complex_data(self.cdata, remap_type)
        self.set_image_from_numpy_array(display_data)

    def remap_complex_data(self,
                           complex_data,    # type: np.ndarray
                           remap_type,      # type: str
                           ):
        if remap_type == 'density':
            pix = remap.density(complex_data)
        elif remap_type == 'brighter':
            pix = remap.brighter(complex_data)
        elif remap_type == 'darker':
            pix = remap.darker(complex_data)
        elif remap_type == 'highcontrast':
            pix = remap.highcontrast(complex_data)
        elif remap_type == 'linear':
            pix = remap.linear(complex_data)
        elif remap_type == 'log':
            pix = remap.log(complex_data)
        elif remap_type == 'pedf':
            pix = remap.pedf(complex_data)
        elif remap_type == 'nrl':
            pix = remap.nrl(complex_data)
        return pix

    def read_image(self):
        nx = self.reader_object.sicdmeta.ImageData.FullImage.NumCols
        ny = self.reader_object.sicdmeta.ImageData.FullImage.NumRows

        decimation_y = ny / self.canvas_height
        decimation_x = nx / self.canvas_width

        decimation = int(min(decimation_y, decimation_x))
        self.decimation = decimation

        self.cdata = self.reader_object.read_chip[0:ny:decimation, 0:nx:decimation]
        self.update_display_image("density")

    def get_data_in_rect(self):
        print(self.rect_coords)
        real_x_min = self.rect_coords[1] * self.decimation
        real_x_max = self.rect_coords[3] * self.decimation
        real_y_min = self.rect_coords[0] * self.decimation
        real_y_max = self.rect_coords[2] * self.decimation

        nx = real_x_max - real_x_min
        ny = real_y_max - real_y_min

        decimation_y = max(ny / self.canvas_height, 1)
        decimation_x = max(nx / self.canvas_width, 1)

        decimation = int(min(decimation_y, decimation_x))
        print("selected data decimation: " + str(decimation))

        rect_data = self.reader_object.read_chip[real_y_min:real_y_max:decimation, real_x_min:real_x_max:decimation]
        return rect_data
