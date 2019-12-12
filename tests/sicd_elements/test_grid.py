import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import Grid


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
    'DeltaK1': 0,
    'DeltaK2': 0,
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

        with self.subTest(msg='get_parameter_value tests'):
            self.assertEqual(item1.get_parameter_value('Name'), '0.1')
            self.assertEqual(item1.get_parameter_value('Junk', default='default'), 'default')


class TestDirParam(unittest.TestCase):
    def test_construction(self):
        the_type = Grid.DirParamType
        the_dict = dir_param_dict
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

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())

    # TODO: test define_weight_function() in a sensible way
    #       test define_response_widths()
    #       test estimate_deltak()


class TestGrid(unittest.TestCase):
    def test_construction(self):
        the_type = Grid.GridType
        the_dict = grid_dict
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

        with self.subTest(msg='Test validity'):
            self.assertTrue(item1.is_valid())

    # TODO: all the derived things. This will be painful.

