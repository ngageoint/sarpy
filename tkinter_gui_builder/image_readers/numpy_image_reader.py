from tkinter_gui_builder.image_readers.image_reader import ImageReader
import numpy


class NumpyImageReader(ImageReader):
    fname = None
    full_image_nx = int
    full_image_ny = int
    numpy_image_data = None     # type: numpy.ndarray

    def __init__(self,
                 numpy_image_data,          # type: numpy.ndarray
                 ):
        self.numpy_image_data = numpy_image_data
        self.full_image_ny, self.full_image_nx = numpy_image_data.shape

    def __getitem__(self, key):
        print(key)
        return self.numpy_image_data[key]
