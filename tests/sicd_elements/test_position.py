import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import Position


position_dict = {
    'ARPPoly': {'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, },
    'GRPPoly': {'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, },
    'TxAPCPoly': {'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, },
    'RcvAPC': [{'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, 'index': 0}, ],
}


class TestPosition(unittest.TestCase):
    def test_construction(self):
        the_type = Position.PositionType
        the_dict = position_dict
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
