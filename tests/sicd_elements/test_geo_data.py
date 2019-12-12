import sys
from xml.etree import ElementTree

import numpy

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import GeoData


geo_info_dict1 = {
    'name': 'Name',
    'Descriptions': {'feature': 'some kind of words'},
    'Point': {'Lat': 0, 'Lon': 0}}
geo_info_dict2 = {
    'name': 'Name',
    'Line': [{'Lat': 0, 'Lon': 0, 'index': 0}, {'Lat': 1, 'Lon': 1, 'index': 1}]}
geo_info_dict3 = {
    'name': 'Name',
    'Polygon': [{'Lat': 0, 'Lon': 0, 'index': 0}, {'Lat': 1, 'Lon': 1, 'index': 1}, {'Lat': 0, 'Lon': 1, 'index': 2}]}

scp_dict = {
    'LLH': {'Lat': 0, 'Lon': 0, 'HAE': 100},
    'ECF': {'X': 0, 'Y': 0, 'Z': 0}
}

geo_data_dict = {
    'EarthModel': 'WGS_84',
    'SCP': scp_dict,
    'ImageCorners': [
        {'Lat': 0, 'Lon': 0, 'index': '1:FRFC'}, {'Lat': 0, 'Lon': 1, 'index': '2:FRLC'},
        {'Lat': 1, 'Lon': 1, 'index': '3:LRLC'}, {'Lat': 1, 'Lon': 0, 'index': '4:LRFC'}],
    'ValidData': [
        {'Lat': 0, 'Lon': 0, 'index': 0}, {'Lat': 1, 'Lon': 1, 'index': 1}, {'Lat': 1, 'Lon': 0, 'index': 2}, ],
    'GeoInfos': [geo_info_dict1, geo_info_dict2, geo_info_dict3],
}


class TestGeoInfo(unittest.TestCase):
    def test_construction1(self):
        the_type = GeoData.GeoInfoType
        the_dict = geo_info_dict1
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

        with self.subTest(msg='Check FeatureType'):
            self.assertEqual(item1.FeatureType, 'Point')

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_construction2(self):
        the_type = GeoData.GeoInfoType
        the_dict = geo_info_dict2
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

        with self.subTest(msg='Check FeatureType'):
            self.assertEqual(item1.FeatureType, 'Line')

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_construction3(self):
        the_type = GeoData.GeoInfoType
        the_dict = geo_info_dict3
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

        with self.subTest(msg='Check FeatureType'):
            self.assertEqual(item1.FeatureType, 'Polygon')

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_validity(self):
        bad_dict = geo_info_dict1.copy()
        bad_dict['Line'] = geo_info_dict2['Line']
        bad_dict2 = {'name': 'Name', 'Line': [{'Lat': 0, 'Lon': 0, 'index': 0}]}
        bad_dict3 = {'name': geo_info_dict2['name'], 'Polygon': geo_info_dict2['Line']}

        with self.subTest(msg='More than one feature is defined'):
            item = GeoData.GeoInfoType.from_dict(bad_dict)
            self.assertFalse(item.is_valid(), False)

        with self.subTest(msg='Line with one point'):
            self.assertRaises(ValueError, GeoData.GeoInfoType.from_dict, bad_dict2)

        with self.subTest(msg='Polygon with two points'):
            self.assertRaises(ValueError, GeoData.GeoInfoType.from_dict, bad_dict3)


class TestSCP(unittest.TestCase):
    def test_construction(self):
        the_type = GeoData.SCPType
        the_dict = scp_dict
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

    def test_derive(self):
        the_type = GeoData.SCPType

        with self.subTest(msg='Derive ECF from LLH'):
            item = the_type.from_dict({'LLH': scp_dict['LLH'], })
            self.assertIsNone(item.ECF)
            self.assertFalse(item.is_valid())
            item.derive()
            self.assertIsNotNone(item.ECF)
            self.assertTrue(item.is_valid())

        with self.subTest(msg='Derive LLH from ECF'):
            item = the_type.from_dict({'ECF': scp_dict['ECF'], })
            self.assertIsNone(item.LLH)
            self.assertFalse(item.is_valid())
            item.derive()
            self.assertIsNotNone(item.LLH)
            self.assertTrue(item.is_valid())


class TestGeoData(unittest.TestCase):
    def test_construction(self):
        the_type = GeoData.GeoDataType
        the_dict = geo_data_dict
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

        with self.subTest(msg='Image Corners setting from array'):
            test_value = numpy.array([[0, 0], [0, 2], [2, 2], [2, 0]])
            item1.ImageCorners = test_value
            self.assertTrue(numpy.all(item1.ImageCorners.get_array(dtype=numpy.float64) == test_value))

        with self.subTest(msg='Valid Data setting from array'):
            test_value = numpy.array([[0, 0], [1, 1], [1, 0]])
            item1.ValidData = test_value
            self.assertTrue(numpy.all(item1.ValidData.get_array(dtype=numpy.float64) == test_value))
