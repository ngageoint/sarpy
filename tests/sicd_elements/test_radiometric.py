import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import Radiometric


noise_level_dict = {'NoiseLevelType': 'ABSOLUTE', 'NoisePoly': {'Coefs': [[0.2, ], ]}}
radiometric_dict = {
    'NoiseLevel': noise_level_dict,
    'RCSSFPoly': {'Coefs': [[0, ], ]},
    'SigmaZeroSFPoly': {'Coefs': [[0, ], ]},
    'BetaZeroSFPoly': {'Coefs': [[0, ], ]},
    'GammaZeroSFPoly': {'Coefs': [[0, ], ]},
}


class TestNoiseLevel(unittest.TestCase):
    def test_construction(self):
        the_type = Radiometric.NoiseLevelType_
        the_dict = noise_level_dict
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


class TestRadiometric(unittest.TestCase):
    def test_construction(self):
        the_type = Radiometric.RadiometricType
        the_dict = radiometric_dict
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
