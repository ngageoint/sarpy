import numpy
from sarpy.io.complex.sicd_elements import GeoData

from . import generic_construction_test, unittest


geo_info_dict1 = {
    'name': 'Name1',
    'Descriptions': {'feature': 'some kind of words'},
    'Point': {'Lat': 0, 'Lon': 0}}
geo_info_dict2 = {
    'name': 'Name2',
    'Line': [{'Lat': 0, 'Lon': 0, 'index': 0}, {'Lat': 1, 'Lon': 1, 'index': 1}],
}
geo_info_dict3 = {
    'name': 'Name3',
    'Polygon': [{'Lat': 0, 'Lon': 0, 'index': 0}, {'Lat': 1, 'Lon': 1, 'index': 1}, {'Lat': 0, 'Lon': 1, 'index': 2}],
}
geo_info_dict4 = {
    'name': 'Name4',
    'Polygon': [{'Lat': 0, 'Lon': 0, 'index': 0}, {'Lat': 1, 'Lon': 1, 'index': 1}, {'Lat': 0, 'Lon': 1, 'index': 2}],
    'GeoInfos': [geo_info_dict1, ],
}

scp_dict = {
    'LLH': {'Lat': 0, 'Lon': 0, 'HAE': 0},
    'ECF': {'X': 6378137, 'Y': 0, 'Z': 0}
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
        item1 = generic_construction_test(self, the_type, the_dict, tag='GeoInfo')

    def test_construction2(self):
        the_type = GeoData.GeoInfoType
        the_dict = geo_info_dict2
        item2 = generic_construction_test(self, the_type, the_dict, tag='GeoInfo')

    def test_construction3(self):
        the_type = GeoData.GeoInfoType
        the_dict = geo_info_dict3
        item3 = generic_construction_test(self, the_type, the_dict, tag='GeoInfo')

    def test_construction4(self):
        the_type = GeoData.GeoInfoType
        the_dict = geo_info_dict4
        item4 = generic_construction_test(self, the_type, the_dict, tag='GeoInfo')

    def test_get_item(self):
        item4 = GeoData.GeoInfoType.from_dict(geo_info_dict4)
        item1 = item4.getGeoInfo('Name1')
        self.assertEqual(item1[0].to_dict(), geo_info_dict1)

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
        item1 = generic_construction_test(self, the_type, the_dict)

    def test_derive(self):
        the_type = GeoData.SCPType

        with self.subTest(msg='Derive ECF from LLH'):
            item = the_type.from_dict({'LLH': scp_dict['LLH'], })
            self.assertIsNotNone(item.ECF)
            self.assertTrue(item.is_valid())

        with self.subTest(msg='Derive LLH from ECF'):
            item = the_type.from_dict({'ECF': scp_dict['ECF'], })
            self.assertIsNotNone(item.LLH)
            self.assertTrue(item.is_valid())


class TestGeoData(unittest.TestCase):
    def test_construction(self):
        the_type = GeoData.GeoDataType
        the_dict = geo_data_dict
        item1 = generic_construction_test(self, the_type, the_dict, print_xml=False)

        with self.subTest(msg='Image Corners setting from array'):
            test_value = numpy.array([[0, 0], [0, 2], [2, 2], [2, 0]])
            item1.ImageCorners = test_value
            self.assertTrue(numpy.all(item1.ImageCorners.get_array(dtype=numpy.float64) == test_value))

        with self.subTest(msg='Valid Data setting from array'):
            test_value = numpy.array([[0, 0], [1, 1], [1, 0]])
            item1.ValidData = test_value
            self.assertTrue(numpy.all(item1.ValidData.get_array(dtype=numpy.float64) == test_value))

    def test_get_item(self):
        geodata = GeoData.GeoDataType.from_dict(geo_data_dict)
        self.assertEqual(geodata.getGeoInfo('Name1')[0].to_dict(), geo_info_dict1)
        self.assertEqual(geodata.getGeoInfo('Name2')[0].to_dict(), geo_info_dict2)
        self.assertEqual(geodata.getGeoInfo('Name3')[0].to_dict(), geo_info_dict3)
