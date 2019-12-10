import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import PFA


st_deskew_dict = {'Applied': True, 'STDSPhasePoly': {'Coefs': [[0, ], ]}}
pfa_dict = {
    'FPN': {'X': 0, 'Y': 0, 'Z': 0},
    'IPN': {'X': 0, 'Y': 0, 'Z': 0},
    'PolarAngRefTime': 0,
    'PolarAngPoly': {'Coefs': [0, 1]},
    'SpatialFreqSFPoly': {'Coefs': [1, 2]},
    'Krg1': 0, 'Krg2': 0, 'Kaz1': 0, 'Kaz2': 0,
    'StDeskew': st_deskew_dict,
}


class TestSTDeskew(unittest.TestCase):
    def test_construction(self):
        the_type = PFA.STDeskewType
        the_dict = st_deskew_dict
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


class TestPFA(unittest.TestCase):
    def test_construction(self):
        the_type = PFA.PFAType
        the_dict = pfa_dict
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
