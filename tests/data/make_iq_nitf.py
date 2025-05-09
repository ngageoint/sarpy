"""Make a simple NITF 2.1 with two bands (I/Q) stored as band interleave by block."""
import pathlib

import numpy as np
from osgeo import gdal

data = np.arange(20, dtype=np.float32).reshape((2, 5, 2)).view(np.complex64)[..., 0]
dst_filename = pathlib.Path(__file__).parent / "iq.nitf"

driver = gdal.GetDriverByName("NITF")
dst_ds = driver.Create(
    str(dst_filename),
    xsize=data.shape[1],
    ysize=data.shape[0],
    bands=2,
    eType=gdal.GDT_Float32,
    options=["ISUBCAT=I,Q"],
)
dst_ds.GetRasterBand(1).WriteArray(data.real)
dst_ds.GetRasterBand(2).WriteArray(data.imag)
# Once we're done, properly close the dataset
dst_ds = None
