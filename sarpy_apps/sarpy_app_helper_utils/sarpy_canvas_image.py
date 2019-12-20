from numpy import ndarray
import sarpy.io.complex as sarpy_complex
from sarpy.io.complex import Reader
import sarpy.visualization.remap as remap
from tkinter_gui_builder.canvas_image_objects.canvas_image import CanvasDisplayImage
import matplotlib.pyplot as plt


class SarpyCanvasDisplayImage(CanvasDisplayImage):
    def __init__(self):
        super(CanvasDisplayImage, self).__init__()
        self.reader_object = None           # type: Reader
        self.remap_type = "density"              # type: str

    def init_from_fname_and_canvas_size(self,
                                        fname,
                                        canvas_ny,
                                        canvas_nx):
        self.reader_object = sarpy_complex.open(fname)
        self.full_image_nx = self.reader_object.sicdmeta.ImageData.FullImage.NumCols
        self.full_image_ny = self.reader_object.sicdmeta.ImageData.FullImage.NumRows
        self.canvas_ny = canvas_ny
        self.canvas_nx = canvas_nx
        full_image_rect = (0, 0, self.full_image_ny, self.full_image_nx)
        self.update_canvas_display_image_from_full_image_subsection(full_image_rect)

    def get_canvas_display_image_from_full_image_subsection(self, full_image_rect):
        decimation = self.get_decimation_from_full_image_rect(full_image_rect)
        im = self.get_image_data_in_full_image_rect(full_image_rect, decimation)
        return im

    def get_image_data_in_full_image_rect(self,
                                          full_image_rect,          # type: (int, int, int, int)
                                          decimation,               # type: int
                                          ):

        y1, x1, y2, x2 = full_image_rect[0], full_image_rect[1], full_image_rect[2], full_image_rect[3]
        cdata = self.reader_object.read_chip[y1:y2:decimation, x1:x2:decimation]
        display_data = self.remap_complex_data(cdata, self.remap_type)
        return display_data, decimation

    @staticmethod
    def remap_complex_data(complex_data,    # type: np.ndarray
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

