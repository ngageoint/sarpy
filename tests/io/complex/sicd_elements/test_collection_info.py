
from sarpy.io.complex.sicd_elements import CollectionInfo

from . import generic_construction_test, unittest


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
        item1 = generic_construction_test(self, CollectionInfo.RadarModeType, mode_dict)

    def test_alternate(self):
        # just verify that it doesn't raise an exception
        test = CollectionInfo.RadarModeType(ModeType='spotlight')
        # verify that it does
        self.assertRaises(ValueError, CollectionInfo.RadarModeType, ModeType='junk')


class TestCollectionInfo(unittest.TestCase):
    def test_construction(self):
        item1 = generic_construction_test(self, CollectionInfo.CollectionInfoType, info_dict)

    def test_default(self):
        junk = CollectionInfo.CollectionInfoType()
        self.assertEqual(junk.Classification, 'UNCLASSIFIED')
