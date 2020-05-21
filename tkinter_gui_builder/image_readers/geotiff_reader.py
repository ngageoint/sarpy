from tkinter_gui_builder.image_readers.image_reader import ImageReader
import numpy
import gdal
import scipy.misc as scipy_misc
from PIL import Image
import matplotlib.pyplot as plt

gdal_to_numpy_data_types = {
    "Byte": numpy.uint8,
    "UInt16": numpy.uint16,
    "Int16": numpy.int16,
    "UInt32": numpy.uint32,
    "Int32": numpy.int32,
    "Float32": numpy.float32,
    "Float64": numpy.float64,
    "CInt16": numpy.complex64,
    "CInt32": numpy.complex64,
    "CFloat32": numpy.complex64,
    "CFloat64": numpy.complex64
}


class GeotiffImageReader(ImageReader):
    fname = None
    full_image_nx = int
    full_image_ny = int
    n_bands = int

    rgba_bands = [0, 1, 2]        # type: [int, int, int]
    pan_band = 0            # type: int

    is_panchromatic = True
    all_image_data = None           # type: numpy.ndarray

    _dset = None

    def __init__(self,
                 fname,          # type: str
                 ):
        self._dset = gdal.Open(fname, gdal.GA_ReadOnly)
        self.full_image_ny = self._dset.RasterYSize
        self.full_image_nx = self._dset.RasterXSize
        self.n_bands = self._dset.RasterCount
        self.n_overviews = self._dset.GetRasterBand(1).GetOverviewCount()

    def get_numpy_data_type(self):
        gdal_data_type = gdal.GetDataTypeName(self._dset.GetRasterBand(1).DataType)
        return gdal_to_numpy_data_types[gdal_data_type]

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        numpy_dtype = self.get_numpy_data_type()
        self.all_image_data = numpy.zeros((self.full_image_ny, self.full_image_nx, self.n_bands), dtype=numpy_dtype)
        for bandnum in range(self.n_bands):
            band = self._dset.GetRasterBand(bandnum + 1)
            band_data = band.ReadAsArray()
            self.all_image_data[:, :, bandnum] = band_data
        if self.all_image_data.shape[2] == 1:
            self.is_panchromatic = True
            self.all_image_data = numpy.squeeze(self.all_image_data)

    def read_all_rgb_from_disk(self):  # type: (...) -> ndarray
        numpy_arr = []
        for rgb in self.rgb_bands:
            band = self._dset.GetRasterBand(rgb + 1)
            numpy_arr.append(band.ReadAsArray())
        self.all_image_data = numpy.stack(numpy_arr, axis=2)

    def __getitem__(self, key):
        print(key)
        if self.n_overviews == 0:
            return self.all_image_data[key]
        else:

            full_image_step_y = key[0].step
            full_image_step_x = key[1].step

            min_step = min(full_image_step_y, full_image_step_x)

            full_image_start_y = key[0].start
            full_image_stop_y = key[0].stop
            full_image_start_x = key[1].start
            full_image_stop_x = key[1].stop

            overview_level = int(numpy.log2(min_step)) - 1
            overview_decimation_factor = numpy.power(2, overview_level + 1)

            overview_start_y = int(full_image_start_y / overview_decimation_factor)
            overview_stop_y = int(full_image_stop_y / overview_decimation_factor)
            overview_start_x = int(full_image_start_x / overview_decimation_factor)
            overview_stop_x = int(full_image_stop_x / overview_decimation_factor)

            overview_x_size = overview_stop_x - overview_start_x - 1
            overview_y_size = overview_stop_y - overview_start_y - 1
            d = self._dset.GetRasterBand(1).GetOverview(overview_level).ReadAsArray(overview_start_x,
                                                                                    overview_start_y,
                                                                                    overview_x_size,
                                                                                    overview_y_size)
            y_resize = int(numpy.ceil((full_image_stop_y - full_image_start_y) / full_image_step_y))
            x_resize = int(numpy.ceil((full_image_stop_x - full_image_start_x) / full_image_step_x))

            pil_image = Image.fromarray(d)
            resized_pil_image = Image.Image.resize(pil_image, (x_resize, y_resize))
            resized_numpy_image = numpy.array(resized_pil_image)
            return resized_numpy_image

    def read_all_pan_from_disk(self):  # type: (...) -> ndarray
        self.all_image_data = self._dset.GetRasterBand(self.pan_band + 1).ReadAsArray()
