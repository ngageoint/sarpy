from tkinter_gui_builder.image_readers.image_reader import ImageReader
import numpy
import gdal
from PIL import Image

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

    display_bands = [0, 1, 2]        # type: [int, int, int]
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
        self.numpy_data_type = self.get_numpy_data_type()

        if self.n_bands == 1:
            self.display_bands = [0]

    def get_numpy_data_type(self):
        gdal_data_type = gdal.GetDataTypeName(self._dset.GetRasterBand(1).DataType)
        return gdal_to_numpy_data_types[gdal_data_type]

    def read_full_image_data_from_disk(self):  # type: (...) -> ndarray
        bands = range(self.n_bands)
        return self.read_full_display_image_data_from_disk(bands)

    def read_full_display_image_data_from_disk(self,
                                               bands,  # type: []
                                               ):  # type: (...) -> ndarray
        n_bands = len(bands)
        image_data = numpy.zeros((self.full_image_ny, self.full_image_nx, n_bands),
                                  dtype=self.numpy_data_type)
        for i in range(n_bands):
            image_data[:, :, i] = self._dset.GetRasterBand(bands[i] + 1).ReadAsArray()
        if image_data.shape[2] == 1:
            image_data = numpy.squeeze(image_data)
        return image_data

    def __getitem__(self, key):
        print(key)
        if self.n_overviews == 0:
            if self.all_image_data is None:
                self.all_image_data = self.read_full_display_image_data_from_disk(self.display_bands)
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
            print("overview level")
            print(overview_level)
            overview_decimation_factor = numpy.power(2, overview_level + 1)

            overview_start_y = int(full_image_start_y / overview_decimation_factor)
            overview_stop_y = int(full_image_stop_y / overview_decimation_factor)
            overview_start_x = int(full_image_start_x / overview_decimation_factor)
            overview_stop_x = int(full_image_stop_x / overview_decimation_factor)

            overview_x_size = overview_stop_x - overview_start_x - 1
            overview_y_size = overview_stop_y - overview_start_y - 1

            n_display_bands = len(self.display_bands)
            d = numpy.zeros((overview_y_size, overview_x_size, n_display_bands), dtype=self.numpy_data_type)
            if overview_level >= 0:
                for i in range(n_display_bands):
                    d[:, :, i] = self._dset.GetRasterBand(self.display_bands[i] + 1).\
                        GetOverview(overview_level).ReadAsArray(overview_start_x,
                                                                overview_start_y,
                                                                overview_x_size,
                                                                overview_y_size)
            else:
                full_image_x_size = full_image_stop_x - full_image_start_x - 1
                full_image_y_size = full_image_stop_y - full_image_start_y - 1
                for i in range(n_display_bands):
                    d[:, :, i] = self._dset.GetRasterBand(self.display_bands[i] + 1).\
                        ReadAsArray(full_image_start_x,
                                    full_image_start_y,
                                    full_image_x_size,
                                    full_image_y_size)
            y_resize = int(numpy.ceil((full_image_stop_y - full_image_start_y) / full_image_step_y))
            x_resize = int(numpy.ceil((full_image_stop_x - full_image_start_x) / full_image_step_x))

            pil_image = Image.fromarray(d)
            resized_pil_image = Image.Image.resize(pil_image, (x_resize, y_resize))
            resized_numpy_image = numpy.array(resized_pil_image)
            return resized_numpy_image
