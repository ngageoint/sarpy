from sarpy.io.complex.sicd_elements import SCPCOA
from . import generic_construction_test, unittest


scp_coa_dict = {
    'SCPTime': 10,
    'ARPPos': {'X': 0, 'Y': 0, 'Z': 10},
    'ARPVel': {'X': 1, 'Y': 1, 'Z': 1},
    'ARPAcc': {'X': 0, 'Y': 0, 'Z': 0},
    'SideOfTrack': 'L',
    'SlantRange': 10,
    'GroundRange': 10,
    'DopplerConeAng': 10,
    'GrazeAng': 45,
    'IncidenceAng': 30,
    'TwistAng': 30,
    'SlopeAng': 30,
    'AzimAng': 45,
    'LayoverAng': 10,
}


class TestSCPCOA(unittest.TestCase):
    def test_construction(self):
        the_type = SCPCOA.SCPCOAType
        the_dict = scp_coa_dict
        item1 = generic_construction_test(self, the_type, the_dict)
