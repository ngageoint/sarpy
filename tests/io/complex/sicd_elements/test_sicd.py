
from sarpy.io.complex.sicd_elements import SICD

from . import generic_construction_test, unittest

from .test_collection_info import info_dict
from .test_image_creation import image_creation_dict
from .test_image_data import image_data_dict
from .test_geo_data import geo_data_dict
from .test_grid import grid_dict
from .test_timeline import timeline_dict
from .test_position import position_dict
from .test_radar_collection import radar_collection_dict
from .test_image_formation import image_formation_dict
from .test_scp_coa import scp_coa_dict
from .test_radiometric import radiometric_dict
from .test_antenna import antenna_dict
from .test_error_statistics import error_statistics_dict
from .test_match_info import match_info_dict
from .test_rg_az_comp import rg_az_comp_dict
from .test_pfa import pfa_dict
from .test_rma import rma_dict1


sicd_dict = {
    'CollectionInfo': info_dict,
    'ImageCreation': image_creation_dict,
    'ImageData': image_data_dict,
    'GeoData': geo_data_dict,
    'Grid': grid_dict,
    'Timeline': timeline_dict,
    'Position': position_dict,
    'RadarCollection': radar_collection_dict,
    'ImageFormation': image_formation_dict,
    'SCPCOA': scp_coa_dict,
    'Radiometric': radiometric_dict,
    'Antenna': antenna_dict,
    'ErrorStatistics': error_statistics_dict,
    'MatchInfo': match_info_dict,
}


# def generic_construction_test(instance, the_type, the_dict, tag='The_Type', print_xml=False, print_json=False):
#     if not issubclass(the_type, Serializable):
#         raise TypeError('Class {} must be a subclass of Serializable'.format(the_type))
#     the_item = the_type.from_dict(the_dict)
#
#     with instance.subTest(msg='Comparing json deserialization with original'):
#         new_dict = the_item.to_dict()
#         if print_json:
#             print(json.dumps(the_dict, indent=1))
#             print(json.dumps(new_dict, indent=1))
#         instance.assertEqual(the_dict, new_dict)
#
#     with instance.subTest(msg='Test xml serialization issues'):
#         # let's serialize to xml
#         xml = the_item.to_xml_string(tag=tag)
#         if print_xml:
#             print(xml)
#         # let's deserialize from xml
#         node, xml_ns = parse_xml_from_string(xml)
#         item2 = the_type.from_node(node, xml_ns)
#         instance.assertEqual(the_item.to_dict(), item2.to_dict())
#
#     with instance.subTest(msg='Test validity'):
#         instance.assertTrue(the_item.is_valid())
#     return the_item



class TestSICD(unittest.TestCase):
    def test_construction1(self):
        the_dict = sicd_dict
        the_dict['ImageFormation']['ImageFormAlgo'] = 'OTHER'
        item1 = generic_construction_test(self, SICD.SICDType, the_dict)

        with self.subTest(msg='ImageFormType'):
            self.assertEqual(item1.ImageFormType, 'OTHER')

    def test_construction2(self):
        the_dict = sicd_dict.copy()  # this is shallowish copy.
        the_dict['RgAzComp'] = rg_az_comp_dict
        the_dict['ImageFormation']['ImageFormAlgo'] = 'RGAZCOMP'
        item1 = generic_construction_test(self, SICD.SICDType, the_dict)

        with self.subTest(msg='ImageFormType'):
            self.assertEqual(item1.ImageFormType, 'RgAzComp')

    def test_construction3(self):
        the_dict = sicd_dict.copy()  # this is shallowish copy.
        the_dict['PFA'] = pfa_dict
        the_dict['ImageFormation']['ImageFormAlgo'] = 'PFA'
        item1 = generic_construction_test(self, SICD.SICDType, the_dict)

        with self.subTest(msg='ImageFormType'):
            self.assertEqual(item1.ImageFormType, 'PFA')

    def test_construction4(self):
        the_dict = sicd_dict.copy()  # this is shallowish copy.
        the_dict['RMA'] = rma_dict1
        the_dict['ImageFormation']['ImageFormAlgo'] = 'RMA'
        item1 = generic_construction_test(self, SICD.SICDType, the_dict)

        with self.subTest(msg='ImageFormType'):
            self.assertEqual(item1.ImageFormType, 'RMA')

    def test_bad_construction(self):
        the_dict = sicd_dict.copy()
        item1 = SICD.SICDType.from_dict(the_dict)
        item1.ImageFormation.ImageFormAlgo = 'PFA'
        # SICD does not have the PFA item set, so this should warn us
        self.assertFalse(item1.is_valid())
