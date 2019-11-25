from sarpy.tkinter_gui_builder.panel_templates.basic_image_canvas_panel import BasicImageCanvasPanel
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
        self.remap_type = remap_type
        if remap_type == 'density':
            pix = remap.density(self.cdata)
        elif remap_type == 'brighter':
            pix = remap.brighter(self.cdata)
        elif remap_type == 'darker':
            pix = remap.darker(self.cdata)
        elif remap_type == 'highcontrast':
            pix = remap.highcontrast(self.cdata)
        elif remap_type == 'linear':
            pix = remap.linear(self.cdata)
        elif remap_type == 'log':
            pix = remap.log(self.cdata)
        elif remap_type == 'pedf':
            pix = remap.pedf(self.cdata)
        elif remap_type == 'nrl':
            pix = remap.nrl(self.cdata)
        self.set_image_from_numpy_array(pix)

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
        real_x_min = self.rect_coords[0] * self.decimation
        real_x_max = self.rect_coords[2] * self.decimation
        real_y_min = self.rect_coords[1] * self.decimation
        real_y_max = self.rect_coords[3] * self.decimation
        y_ul = int(self.rect_coords[0])
        x_ul = int(self.rect_coords[1])
        y_br = int(self.rect_coords[2])
        x_br = int(self.rect_coords[3])
        selected_image_data = self.image_data[y_ul: y_br, x_ul:x_br]
        return selected_image_data
