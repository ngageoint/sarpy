
from sarpy.io.complex.sicd_elements import PFA
from . import generic_construction_test, unittest


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
        item1 = generic_construction_test(self, the_type, the_dict)


class TestPFA(unittest.TestCase):
    def test_construction(self):
        the_type = PFA.PFAType
        the_dict = pfa_dict
        item1 = generic_construction_test(self, the_type, the_dict)
