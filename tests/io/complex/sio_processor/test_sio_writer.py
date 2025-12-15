__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"
# Written on: 2025-10
#

import numpy, os, pytest, re
from numpy.testing import assert_array_equal
from pathlib import Path
import shutil
import tempfile
from unittest import TestCase

from sarpy.io.complex.sicd_elements.blocks     import RowColType
from sarpy.io.complex.sicd_elements.ImageData  import ImageDataType, FullImageType
from sarpy.io.complex.sicd_elements.SICD       import SICDType
from sarpy.io.complex.sio_processor.sio_writer import SIOWriter as SIOWriter
from sarpy.io.complex.sio_processor.sio_reader import SIOReader as SIOReader

class test_sio_writer(TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_create_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, 
                                    re.escape(
                                        "SIOWriter.__init__() missing 2 required positional arguments")):
            sio_writer = SIOWriter()

    def test_create_with_filename_only_fail(self):
        with self.assertRaisesRegex(TypeError, 
                                    re.escape(
                                        "SIOWriter.__init__() missing 1 required positional argument")):
            output_sio_writer_32 = self.tmp_dir / "SIOWriterTest_filename_only_fail.sio"
            sio_writer = SIOWriter(output_sio_writer_32)
            sio_writer.close()

    def test_create_with_param_image_data_only_fail(self):
        with self.assertRaisesRegex(TypeError, 
                                    re.escape(
                                        "SIOWriter.__init__() missing 1 required positional argument")):
            image_data = numpy.arange(13*17, dtype=numpy.float32).reshape(13, 17)
            sio_writer = SIOWriter(param_image_data=image_data)

    def test_write_with_required_params_only_success(self):
        image_data = numpy.arange(13*17, dtype=numpy.float32).reshape(13, 17)
        output_sio_writer_32 = self.tmp_dir / "SIOWriterTest_required_params_only_success.sio"
        sio_writer = SIOWriter(output_sio_writer_32, image_data)
        sio_writer.write()
        sio_writer.close()
        sio_reader = SIOReader(output_sio_writer_32)
        assert_array_equal(sio_reader._image_data, image_data)
        self.assertIsNone(sio_reader._sicdmeta)

    def test_write_with_required_params_and_sicd_meta_success(self):
        image_data = numpy.arange(13*17, dtype=numpy.float32).reshape(13, 17)
        sicd_meta_real_32 = SICDType(
            ImageData=ImageDataType(
                NumRows=image_data.shape[0],
                    NumCols=image_data.shape[1],
                    PixelType="RE32F_IM32F",
                    FirstRow=0,
                    FirstCol=0,
                    FullImage=FullImageType(
                        NumRows=image_data.shape[0],
                        NumCols=image_data.shape[1]
                    ),
                    SCPPixel=RowColType(Row=image_data.shape[0], 
                                        Col=image_data.shape[1])
            ),
        )
        output_sio_writer_32 = self.tmp_dir / "SIOWriterTest_required_params_and_sicd_meta_success.sio"
        sio_writer = SIOWriter(output_sio_writer_32, image_data, sicd_meta_real_32)
        sio_writer.write()
        sio_writer.close()
        sio_reader = SIOReader(output_sio_writer_32)
        assert_array_equal(sio_reader._image_data, image_data)
        self.assertEqual(sio_reader._sicdmeta.to_xml_string(), sicd_meta_real_32.to_xml_string())
