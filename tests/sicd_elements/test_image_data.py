import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import ImageData

import numpy

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
        {'Row': 0, 'Col': 1, 'index': 0},
        {'Row': 3, 'Col': 1, 'index': 1},
        {'Row': 3, 'Col': 7, 'index': 2},
        {'Row': 0, 'Col': 7, 'index': 3},
    ],
}


class TestFullImage(unittest.TestCase):
    def test_construction(self):
        the_type = ImageData.FullImageType
        the_dict = full_image_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())


class TestImageData(unittest.TestCase):
    def test_construction(self):
        the_type = ImageData.ImageDataType
        the_dict = image_data_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())

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
