import sarpy.io.complex as sarpy_complex
from sarpy.io.complex.base import BaseReader
import sarpy.visualization.remap as remap
from tkinter_gui_builder.canvas_image_objects.abstract_canvas_image import AbstractCanvasImage


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
