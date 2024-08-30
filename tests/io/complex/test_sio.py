import numpy as np

import sarpy.io.complex.sio as sarpy_sio
from sarpy.io.complex.sicd_elements.blocks import RowColType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType, FullImageType


def test_sio_io(tmp_path):
    original_data = np.arange(13*17*2, dtype=np.float32).reshape((13, 17, 2)).view(np.complex64).squeeze()
    sicd_meta = SICDType(
            ImageData=ImageDataType(
                NumRows=original_data.shape[0],
                NumCols=original_data.shape[1],
                PixelType="RE32F_IM32F",
                FirstRow=0,
                FirstCol=0,
                FullImage=FullImageType(
                    NumRows=original_data.shape[0],
                    NumCols=original_data.shape[1]
                ),
                SCPPixel=RowColType(Row=original_data.shape[0] // 2, Col=original_data.shape[1] // 2)
            ),
    )

    output_file = tmp_path / "test.sio"
    with sarpy_sio.SIOWriter(str(output_file), sicd_meta, check_existence=False) as sio_writer:
        sio_writer.write(original_data)

    with sarpy_sio.SIOReader(str(output_file)) as sio_reader:
        read_data = sio_reader[...]
        assert np.array_equal(original_data, read_data)
