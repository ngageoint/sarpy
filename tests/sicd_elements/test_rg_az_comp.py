import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import RgAzComp


rg_az_comp_dict = {'AzSF': 1, 'KazPoly': {'Coefs': [0, 1]}}


class TestRgAzComp(unittest.TestCase):
    def test_construction(self):
        the_type = RgAzComp.RgAzCompType
        the_dict = rg_az_comp_dict
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
