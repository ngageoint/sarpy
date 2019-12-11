import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import Timeline


ipp_set_dict = {
    'TStart': 1,
    'TEnd': 10,
    'IPPStart': 1,
    'IPPEnd': 3,
    'IPPPoly': {'Coefs': [1, 2]},
    'index': 0,
}
timeline_dict = {
    'CollectStart': '2019-12-11T13:58:27.000000',
    'CollectDuration': 54,
    'IPP': [ipp_set_dict, ],
}


class TestIPPSet(unittest.TestCase):
    def test_construction(self):
        the_type = Timeline.IPPSetType
        the_dict = ipp_set_dict
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


class TestTimeline(unittest.TestCase):
    def test_construction(self):
        the_type = Timeline.TimelineType
        the_dict = timeline_dict
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
