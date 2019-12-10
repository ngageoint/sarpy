import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

import numpy

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


class TestEB(unittest.TestCase):
    def test_construction(self):
        the_dict = eb_dict

        item1 = Antenna.EBType.from_dict(the_dict)
        item2 = Antenna.EBType(DCXPoly=[0, 1], DCYPoly=[1, -1])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'EBType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = Antenna.EBType.from_node(node)
            self.assertEqual(item1.to_dict(), item3.to_dict())

    def test_eval(self):
        item = Antenna.EBType(DCXPoly=[0, 1], DCYPoly=[1, -1])
        self.assertTrue(numpy.all(item(0) == numpy.array([0, 1])))


class TestAntParam(unittest.TestCase):
    def test_construction(self):
        the_type = Antenna.AntParamType
        the_dict = antenna_type_dict
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestAntenna(unittest.TestCase):
    def test_construction(self):
        the_type = Antenna.AntennaType
        the_dict = {'Tx': antenna_type_dict, 'Rcv': antenna_type_dict, 'TwoWay': antenna_type_dict}
        item1 = the_type.from_dict(the_dict)

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'The_Type')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item2 = the_type.from_node(node)
            self.assertEqual(item1.to_dict(), item2.to_dict())
