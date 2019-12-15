from .. import generic_construction_test, unittest
from sarpy.sicd_elements import RgAzComp


rg_az_comp_dict = {'AzSF': 1, 'KazPoly': {'Coefs': [0, 1]}}


class TestRgAzComp(unittest.TestCase):
    def test_construction(self):
        the_type = RgAzComp.RgAzCompType
        the_dict = rg_az_comp_dict
        item1 = generic_construction_test(self, the_type, the_dict)
