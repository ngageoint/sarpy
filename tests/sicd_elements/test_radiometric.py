from .. import generic_construction_test, unittest
from sarpy.sicd_elements import Radiometric


noise_level_dict = {'NoiseLevelType': 'ABSOLUTE', 'NoisePoly': {'Coefs': [[0.2, ], ]}}
radiometric_dict = {
    'NoiseLevel': noise_level_dict,
    'RCSSFPoly': {'Coefs': [[0, ], ]},
    'SigmaZeroSFPoly': {'Coefs': [[0, ], ]},
    'BetaZeroSFPoly': {'Coefs': [[0, ], ]},
    'GammaZeroSFPoly': {'Coefs': [[0, ], ]},
}


class TestNoiseLevel(unittest.TestCase):
    def test_construction(self):
        the_type = Radiometric.NoiseLevelType_
        the_dict = noise_level_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestRadiometric(unittest.TestCase):
    def test_construction(self):
        the_type = Radiometric.RadiometricType
        the_dict = radiometric_dict
        item1 = generic_construction_test(self, the_type, the_dict)
