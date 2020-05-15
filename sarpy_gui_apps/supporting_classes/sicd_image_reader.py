from tkinter_gui_builder.image_readers.image_reader import ImageReader
import sarpy.io.complex as sarpy_complex
from sarpy.io.complex.base import BaseReader
import sarpy.visualization.remap as remap


class SicdImageReader(ImageReader):
    sicd = None           # type: BaseReader
    remap_type = "density"

    def __init__(self, fname):
        self.sicd = sarpy_complex.open(fname)
        self.full_image_nx = self.sicd.sicd_meta.ImageData.FullImage.NumCols
        self.full_image_ny = self.sicd.sicd_meta.ImageData.FullImage.NumRows

    def set_reader_file(self, fname):
        self.sicd = sarpy_complex.open(fname)
        self.full_image_nx = self.sicd.sicd_meta.ImageData.FullImage.NumCols
        self.full_image_ny = self.sicd.sicd_meta.ImageData.FullImage.NumRows

    def get_image_chip(self,
                       y_start,  # type: int
                       y_end,  # type: int
                       x_start,  # type: int
                       x_end,  # type: int
                       decimation=1,  # type: int
                       ):  # type: (...) -> numpy.ndarray
        cdata = self.sicd.read_chip( (y_start, y_end, decimation), (x_start, x_end, decimation))
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