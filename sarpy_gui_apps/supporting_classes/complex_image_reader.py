from tkinter_gui_builder.image_readers.image_reader import ImageReader
import sarpy.io.complex as sarpy_complex
from sarpy.io.complex.base import BaseReader
import sarpy.visualization.remap as remap
import numpy


# TODO use properties for remap, and SICD
class ComplexImageReader(ImageReader):
    base_reader = None           # type: BaseReader
    remap_type = "density"

    def __init__(self, fname):
        self.base_reader = sarpy_complex.open(fname)
        self.full_image_nx = self.base_reader.sicd_meta.ImageData.FullImage.NumCols
        self.full_image_ny = self.base_reader.sicd_meta.ImageData.FullImage.NumRows

    def __getitem__(self, key):
        cdata = self.base_reader[key]
        decimated_image_data = self.remap_complex_data(cdata)
        return decimated_image_data

    def set_reader_file(self, fname):
        self.base_reader = sarpy_complex.open(fname)
        self.full_image_nx = self.base_reader.sicd_meta.ImageData.FullImage.NumCols
        self.full_image_ny = self.base_reader.sicd_meta.ImageData.FullImage.NumRows

    # TODO get rid of strings, make these methods
    def remap_complex_data(self,
                           complex_data,    # type: numpy.ndarray
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