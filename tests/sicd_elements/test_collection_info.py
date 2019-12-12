import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import CollectionInfo

mode_dict = {'ModeId': 'TEST_ID', 'ModeType': 'SPOTLIGHT'}

info_dict = {
    'CollectorName': 'Collector',
    'IlluminatorName': 'Illuminator',
    'CoreName': 'Core',
    'CollectType': 'MONOSTATIC',
    'RadarMode': mode_dict,
    'Classification': 'UNCLASSIFIED',
    'CountryCodes': ['some-code', 'some-other-code'],
    'Parameters': {'Name1': 'Value1', 'Name2': 'Value2'},
}


class TestRadarMode(unittest.TestCase):
    def test_construction(self):
        the_type = CollectionInfo.RadarModeType
        the_dict = mode_dict
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

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_alternate(self):
        # just verify that it doesn't raise an exception
        test = CollectionInfo.RadarModeType(ModeType='spotlight')
        # verify that it does
        self.assertRaises(ValueError, CollectionInfo.RadarModeType, ModeType='junk')


class TestCollectionInfo(unittest.TestCase):
    def test_construction(self):
        the_type = CollectionInfo.CollectionInfoType
        the_dict = info_dict
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

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_default(self):
        junk = CollectionInfo.CollectionInfoType()
        self.assertEqual(junk.Classification, 'UNCLASSIFIED')

