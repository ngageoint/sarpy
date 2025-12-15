__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"
# Written on: 2025-10
#

import pytest, os, re
import numpy
from numpy.testing import assert_array_equal
from unittest import TestCase

from sarpy.io.complex.sicd_elements.blocks import RowColType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType, FullImageType
from sarpy.io.complex.sio_processor.sio_reader import SIOReader as SIOReader

class test_sio_reader(TestCase):
    
    def test_read_float_32(self):
        input_sio_reader_32 = "./tests/io/complex/sio_processor/SIOReaderTest_32.sio"
        image_data = numpy.arange(13*17, dtype=numpy.float32).reshape(13, 17)
        example_sicd = SICDType(
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
        sio_reader = SIOReader(str(input_sio_reader_32))
        assert_array_equal(sio_reader._image_data, image_data)
        self.assertEqual(sio_reader._sicdmeta.to_xml_bytes(), 
                         example_sicd.to_xml_bytes())

    def test_read_int_16(self):
        input_sio_reader_16 = "./tests/io/complex/sio_processor/SIOReaderTest_16.sio"
        image_data = numpy.arange(13*17, dtype=numpy.int16).reshape(13, 17)
        example_sicd = SICDType(
            ImageData=ImageDataType(
                NumRows=image_data.shape[0],
                    NumCols=image_data.shape[1],
                    PixelType="RE16I_IM16I",
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
        sio_reader = SIOReader(str(input_sio_reader_16))
        assert_array_equal(sio_reader._image_data, image_data)
        self.assertEqual(sio_reader._sicdmeta.to_xml_bytes(), 
                         example_sicd.to_xml_bytes())
        
    def test_read_complex_64(self):
        input_sio_reader_complex = "./tests/io/complex/sio_processor/SIOReaderTest_complex.sio"
        image_data = numpy.arange(13*17, dtype=numpy.complex64).reshape(13, 17)
        example_sicd = SICDType(
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
        sio_reader = SIOReader(str(input_sio_reader_complex))
        assert_array_equal(sio_reader._image_data, image_data)
        self.assertEqual(sio_reader._sicdmeta.to_xml_bytes(), 
                         example_sicd.to_xml_bytes())
