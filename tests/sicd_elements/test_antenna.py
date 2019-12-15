import numpy

from .. import generic_construction_test, unittest

from sarpy.sicd_elements import Antenna

eb_dict = {
    'DCXPoly': {'Coefs': [0, 1]},
    'DCYPoly': {'Coefs': [1, -1]}}

antenna_type_dict = {
    'XAxisPoly': {'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, },
    'YAxisPoly': {'X': {'Coefs': [0, ]}, 'Y': {'Coefs': [0, ]}, 'Z': {'Coefs': [0, ]}, },
    'FreqZero': 0,
    'EB': eb_dict,
    'Array': {'GainPoly': {'Coefs': [[0, ], ]}, 'PhasePoly': {'Coefs': [[0, ], ]}},
    'Elem': {'GainPoly': {'Coefs': [[0, ], ]}, 'PhasePoly': {'Coefs': [[0, ], ]}},
    'GainBSPoly': {'Coefs': [0, ]},
    'EBFreqShift': False,
    'MLFreqDilation': False
}
antenna_dict = {'Tx': antenna_type_dict, 'Rcv': antenna_type_dict, 'TwoWay': antenna_type_dict}


class TestEB(unittest.TestCase):
    def test_construction(self):
        item1 = generic_construction_test(self, Antenna.EBType, eb_dict)

        item2 = Antenna.EBType(DCXPoly=[0, 1], DCYPoly=[1, -1])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

    def test_eval(self):
        item = Antenna.EBType(DCXPoly=[0, 1], DCYPoly=[1, -1])
        self.assertTrue(numpy.all(item(0) == numpy.array([0, 1])))


class TestAntParam(unittest.TestCase):
    def test_construction(self):
        item1 = generic_construction_test(self, Antenna.AntParamType, antenna_type_dict)


class TestAntenna(unittest.TestCase):
    def test_construction(self):
        item1 = generic_construction_test(self, Antenna.AntennaType, antenna_dict)
