
from sarpy.io.complex.sicd_elements import Position

from . import generic_construction_test, unittest


position_dict = {
    'ARPPoly': {'X': {'Coefs': [0, 0, 0]}, 'Y': {'Coefs': [0, 0, 0]}, 'Z': {'Coefs': [0, 0, 0]}, },
    'GRPPoly': {'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, },
    'TxAPCPoly': {'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, },
    'RcvAPC': [{'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, 'index': 0}, ],
}


class TestPosition(unittest.TestCase):
    def test_construction(self):
        the_type = Position.PositionType
        the_dict = position_dict
        item1 = generic_construction_test(self, the_type, the_dict)
