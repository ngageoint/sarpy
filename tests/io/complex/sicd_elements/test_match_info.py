
from sarpy.io.complex.sicd_elements import MatchInfo

from . import generic_construction_test, unittest


match_collection_dict = {'CoreName': 'Name', 'MatchIndex': 2, 'Parameters': {'PName': 'PValue'}}
match_dict = {
    'TypeId': 'COHERENT', 'CurrentIndex': 2, 'MatchCollections': [match_collection_dict, ], 'NumMatchCollections': 1}
match_info_dict = {'NumMatchTypes': 1, 'MatchTypes': [match_dict, ]}


class TestMatchCollection(unittest.TestCase):
    def test_construction(self):
        the_type = MatchInfo.MatchCollectionType
        the_dict = match_collection_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestMatch(unittest.TestCase):
    def test_construction(self):
        the_type = MatchInfo.MatchType
        the_dict = match_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestMatchInfo(unittest.TestCase):
    def test_construction(self):
        the_type = MatchInfo.MatchInfoType
        the_dict = match_info_dict
        item1 = generic_construction_test(self, the_type, the_dict)
