
from sarpy.io.complex.sicd_elements import Grid

from . import generic_construction_test, unittest


wgt_type_dict = {
    'WindowName': 'UNKNOWN',
    'Parameters': {'Name': '0.1'},
}

dir_param_dict = {
    'UVectECF': {'X': 1, 'Y': 0, 'Z': 0},
    'SS': 1,
    'ImpRespWid': 1,
    'Sgn': -1,
    'ImpRespBW': 1,
    'KCtr': 0.5,
    'DeltaK1': -0.5,
    'DeltaK2': 0.5,
    'DeltaKCOAPoly': {'Coefs': [[0, ], ]},
    'WgtType': wgt_type_dict,
    'WgtFunct': [0.2, 0.2, 0.2, 0.2, 0.2],
}

grid_dict = {
    'ImagePlane': 'SLANT',
    'Type': 'RGAZIM',
    'TimeCOAPoly': {'Coefs': [[0, ], ]},
    'Row': dir_param_dict,
    'Col': dir_param_dict
}


class TestWgtType(unittest.TestCase):
    def test_construction(self):
        the_type = Grid.WgtTypeType
        the_dict = wgt_type_dict
        item1 = generic_construction_test(self, the_type, the_dict)


class TestDirParam(unittest.TestCase):
    def test_construction(self):
        the_type = Grid.DirParamType
        the_dict = dir_param_dict
        item1 = generic_construction_test(self, the_type, the_dict)

    # TODO: test define_weight_function() in a sensible way
    #       test define_response_widths()
    #       test estimate_deltak()


class TestGrid(unittest.TestCase):
    def test_construction(self):
        the_type = Grid.GridType
        the_dict = grid_dict
        item1 = generic_construction_test(self, the_type, the_dict)

    # TODO: all the derived things. This will be painful.

