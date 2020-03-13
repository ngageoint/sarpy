
import numpy
from sarpy.io.complex.sicd_elements import ImageData

from . import generic_construction_test, unittest


full_image_dict = {'NumRows': 10, 'NumCols': 10}

image_data_dict = {
    'PixelType': 'AMP8I_PHS8I',
    'AmpTable': list(numpy.arange(256, dtype=numpy.float64)),
    'NumRows': 10,
    'NumCols': 10,
    'FirstRow': 0,
    'FirstCol': 0,
    'FullImage': full_image_dict,
    'SCPPixel': {'Row': 5, 'Col': 4},
    'ValidData': [
        {'Row': 0, 'Col': 1, 'index': 1},
        {'Row': 0, 'Col': 7, 'index': 2},
        {'Row': 3, 'Col': 7, 'index': 3},
        {'Row': 3, 'Col': 1, 'index': 4},
    ],
}


class TestFullImage(unittest.TestCase):
    def test_construction(self):
        the_type = ImageData.FullImageType
        the_dict = full_image_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestImageData(unittest.TestCase):
    def test_construction(self):
        the_type = ImageData.ImageDataType
        the_dict = image_data_dict
        item1 = generic_construction_test(self, the_type, the_dict)

    def test_validity(self):
        the_type = ImageData.ImageDataType
        the_dict1 = image_data_dict.copy()
        del the_dict1['AmpTable']
        the_dict2 = image_data_dict.copy()
        the_dict2['PixelType'] = 'RE32F_IM32F'

        with self.subTest(msg='Test validity'):
            item1 = the_type.from_dict(the_dict1)
            self.assertFalse(item1.is_valid())

        with self.subTest(msg='Test validity'):
            item2 = the_type.from_dict(the_dict2)
            self.assertFalse(item2.is_valid())

        with self.subTest(msg='Limits on PixelType'):
            self.assertRaises(ValueError, ImageData.ImageDataType, PixelData='bad_value')
