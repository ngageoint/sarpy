from tkinter_gui_builder.image_readers.image_reader import ImageReader
import numpy
import gdal


class GeotiffImageReader(ImageReader):
    fname = None
    full_image_nx = int
    full_image_ny = int
    n_bands = int

    rgb_bands = [0, 1, 2]        # type: [int, int, int]
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

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        # TODO: allocate this array up front, rather than append to an existing array, which is much slower
        numpy_arr = []
        for bandnum in range(self.n_bands):
            band = self._dset.GetRasterBand(bandnum + 1)
            numpy_arr.append(band.ReadAsArray())
        self.all_image_data = numpy.stack(numpy_arr, axis=2)
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
        return self.all_image_data[key]

    def read_all_pan_from_disk(self):  # type: (...) -> ndarray
        self.all_image_data = self._dset.GetRasterBand(self.pan_band + 1).ReadAsArray()
