import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

import numpy

from sarpy.sicd_elements import blocks


class TestParameter(unittest.TestCase):
    def setUp(self):
        self.dict = {'name': 'Name', 'value': 'Value'}
        self.xml = '<Parameter name="Name">Value</Parameter>'
        self.node = ElementTree.fromstring(self.xml)

    def test_construction(self):
        item1 = blocks.ParameterType.from_dict(self.dict)
        item2 = blocks.ParameterType.from_node(node=self.node)

        # NB: I haven't implemented __eq__, so no instances will ever be equal.
        with self.subTest(msg='Comparing from dict construction with xml construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())
        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(self.dict, new_dict)
        with self.subTest(msg="Comparing xml serialization with original"):
            etree = ElementTree.ElementTree()
            new_xml = ElementTree.tostring(item1.to_node(etree, 'Parameter')).decode('utf-8')
            self.assertEqual(self.xml, new_xml)
        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())


class TestXYZType(unittest.TestCase):
    def test_construction(self):
        the_dict = {'X': 1, 'Y': 2, 'Z': 3}
        item1 = blocks.XYZType.from_dict(the_dict)
        item2 = blocks.XYZType(coords=(1, 2, 3))

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'XYZType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.XYZType.from_node(node)
            # let's verify that things are sufficiently equal
            tolerance = 1e-8
            equals = numpy.all(numpy.absolute(item1.get_array() - item3.get_array()) <= tolerance)
            self.assertTrue(equals, msg='xml - {}'.format(xml))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_methods(self):
        array = numpy.array([0.0, 1.0, 2.0])
        item = blocks.XYZType(coords=array)
        with self.subTest(msg='get_array test'):
            array2 = item.get_array()
            self.assertTrue(numpy.all(array == array2), msg='{} != {}'.format(array, array2))


class TestLatLon(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2}
        item1 = blocks.LatLonType.from_dict(the_dict)
        item2 = blocks.LatLonType(coords=[1, 2])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'LatLonType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.LatLonType.from_node(node)
            # let's verify that things are sufficiently equal
            tolerance = 1e-8
            equals = numpy.all(numpy.absolute(item1.get_array() - item3.get_array()) <= tolerance)
            self.assertTrue(equals, msg='xml - {}'.format(xml))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())


class TestLatLonRestriction(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 91, 'Lon': -182}
        item1 = blocks.LatLonRestrictionType.from_dict(the_dict)
        item2 = blocks.LatLonRestrictionType(coords=[91, -182])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            proper_dict = {'Lat': -89, 'Lon': 178}
            new_dict = item1.to_dict()
            self.assertEqual(proper_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'LatLonRestrictionType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.LatLonRestrictionType.from_node(node)
            # let's verify that things are sufficiently equal
            tolerance = 1e-8
            equals = numpy.all(numpy.absolute(item1.get_array() - item3.get_array()) <= tolerance)
            self.assertTrue(equals, msg='xml - {}'.format(xml))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())


class TestLatLonArrayElement(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2, 'index': 3}
        item1 = blocks.LatLonArrayElementType.from_dict(the_dict)
        item2 = blocks.LatLonArrayElementType(coords=[1, 2], index=3)

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'LatLonArrayElementType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.LatLonArrayElementType.from_node(node)
            # let's verify that things are sufficiently equal
            tolerance = 1e-8
            equals = numpy.all(numpy.absolute(item1.get_array() - item3.get_array()) <= tolerance) \
                and (item1.index == item3.index)
            self.assertTrue(equals, msg='xml - {}'.format(xml))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())


class TestLatLonHAE(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2, 'HAE': 3}
        item1 = blocks.LatLonHAEType.from_dict(the_dict)
        item2 = blocks.LatLonHAEType(coords=[1, 2, 3])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'LatLonHAEType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.LatLonHAEType.from_node(node)
            # let's verify that things are sufficiently equal
            tolerance = 1e-8
            equals = numpy.all(numpy.absolute(item1.get_array() - item3.get_array()) <= tolerance)
            self.assertTrue(equals, msg='xml - {}'.format(xml))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())


class TestLatLonHAERestriction(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 91, 'Lon': -182, 'HAE': 3}
        item1 = blocks.LatLonHAERestrictionType.from_dict(the_dict)
        item2 = blocks.LatLonHAERestrictionType(coords=[91, -182, 3])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            proper_dict = {'Lat': -89, 'Lon': 178, 'HAE': 3}
            new_dict = item1.to_dict()
            self.assertEqual(proper_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'LatLonHAERestrictionType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.LatLonHAERestrictionType.from_node(node)
            # let's verify that things are sufficiently equal
            tolerance = 1e-8
            equals = numpy.all(numpy.absolute(item1.get_array() - item3.get_array()) <= tolerance)
            self.assertTrue(equals, msg='xml - {}'.format(xml))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())


class TestLatLonCorner(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2, 'index': 3}
        item1 = blocks.LatLonCornerType.from_dict(the_dict)
        item2 = blocks.LatLonCornerType(coords=[1, 2], index=3)

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'LatLonCornerType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.LatLonCornerType.from_node(node)
            # let's verify that things are sufficiently equal
            tolerance = 1e-8
            equals = numpy.all(numpy.absolute(item1.get_array() - item3.get_array()) <= tolerance) \
                and (item1.index == item3.index)
            self.assertTrue(equals, msg='xml - {}'.format(xml))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_index_validation(self):
        self.assertRaises(ValueError, blocks.LatLonCornerType, Lat=1, Lon=2, index=5)
        self.assertRaises(ValueError, blocks.LatLonCornerType, Lat=1, Lon=2, index=0)


class TestLatLonCornerString(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2, 'index': '1:FRFC'}
        item1 = blocks.LatLonCornerStringType.from_dict(the_dict)
        item2 = blocks.LatLonCornerStringType(coords=[1, 2], index='1:FRFC')

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'LatLonCornerStringType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.LatLonCornerStringType.from_node(node)
            # let's verify that things are sufficiently equal
            tolerance = 1e-8
            equals = numpy.all(numpy.absolute(item1.get_array() - item3.get_array()) <= tolerance) \
                and (item1.index == item3.index)
            self.assertTrue(equals, msg='xml - {}'.format(xml))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_index_validation(self):
        self.assertRaises(ValueError, blocks.LatLonCornerStringType, Lat=1, Lon=2, index='junk')


class TestRowCol(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Row': 1, 'Col': 2}
        item1 = blocks.RowColType.from_dict(the_dict)
        item2 = blocks.RowColType(coords=[1, 2])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'RowColType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.RowColType.from_node(node)
            self.assertEqual(item1.to_dict(), item3.to_dict())

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())


class TestRowColArrayElement(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Row': 1, 'Col': 2, 'index': 3}
        item1 = blocks.RowColArrayElement.from_dict(the_dict)
        item2 = blocks.RowColArrayElement(coords=[1, 2], index=3)

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'RowColArrayElement')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.RowColArrayElement.from_node(node)
            self.assertEqual(item1.to_dict(), item3.to_dict())

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())


class TestPoly1D(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Coefs': [0, 1, 2]}
        item1 = blocks.Poly1DType.from_dict(the_dict)
        item2 = blocks.Poly1DType(Coefs=[0, 1, 2])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'Poly1DType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.Poly1DType.from_node(node)
            self.assertEqual(item1.to_dict(), item3.to_dict())

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_eval(self):
        item = blocks.Poly1DType(Coefs=[0, 1, 2])
        self.assertEqual(item(1), 3)

    def test_derivative(self):
        item = blocks.Poly1DType(Coefs=[0, 1, 2])
        dcoef = numpy.array([1, 4])
        calc_dcoefs = item.derivative(der_order=1, return_poly=False)
        self.assertEqual(item.derivative_eval(1, 1), 5)
        self.assertTrue(numpy.all(dcoef == calc_dcoefs), '{}\n{}'.format(dcoef, calc_dcoefs))
        item2 = item.derivative(der_order=2, return_poly=True)
        self.assertTrue(numpy.all(item2.Coefs == numpy.array([4, ])))

    def test_shift(self):
        array1 = numpy.array([1, 2, 1], dtype=numpy.float64)
        array2 = numpy.array([0, 0, 1], dtype=numpy.float64)
        array3 = numpy.array([1, 4, 4], dtype=numpy.float64)
        array4 = array2*array3
        with self.subTest(msg='Testing polynomial shift'):
            item = blocks.Poly1DType(Coefs=array1)
            shift = item.shift(1, alpha=1, return_poly=False)
            self.assertTrue(
                numpy.all(shift == array2), msg='calculated {}\nexpected {}'.format(shift, array2))
        with self.subTest(msg="Testing polynomial scale"):
            item = blocks.Poly1DType(Coefs=array1)
            scale = item.shift(0, alpha=2, return_poly=False)
            self.assertTrue(
                numpy.all(scale == array3), msg='calculated {}\nexpected {}'.format(scale, array3))
        with self.subTest(msg="Shift and scale"):
            item = blocks.Poly1DType(Coefs=array1)
            shift_scale = item.shift(1, alpha=2, return_poly=False)
            self.assertTrue(
                numpy.all(shift_scale == array4), msg='calculated {}\nexpected {}'.format(shift_scale, array4))


class TestPoly2D(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Coefs': [[0, 0, 0], [0, 1, 2]]}
        item1 = blocks.Poly2DType.from_dict(the_dict)
        item2 = blocks.Poly2DType(Coefs=[[0, 0, 0], [0, 1, 2]])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'Poly2DType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.Poly2DType.from_node(node)
            self.assertEqual(item1.to_dict(), item3.to_dict())

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_eval(self):
        item = blocks.Poly2DType(Coefs=[[0, 0, 0], [0, 1, 2]])
        self.assertEqual(item(1, 1), 3)


class TestXYZPoly(unittest.TestCase):
    def test_construction(self):
        the_dict = {'X': {'Coefs': [0, 1, 2]}, 'Y': {'Coefs': [0, 2, 4]}, 'Z': {'Coefs': [0, 3, 6]}}
        item1 = blocks.XYZPolyType.from_dict(the_dict)
        item2 = blocks.XYZPolyType(X=[0, 1, 2], Y=[0, 2, 4], Z=[0, 3, 6])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'XYZPolyType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.XYZPolyType.from_node(node)
            self.assertEqual(item1.to_dict(), item3.to_dict())

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_eval(self):
        item = blocks.XYZPolyType(X=[0, 1, 2], Y=[0, 2, 4], Z=[0, 3, 6])
        self.assertTrue(numpy.all(item(1) == numpy.array([3, 6, 9])))

    def test_derivative(self):
        item = blocks.XYZPolyType(X=[0, 1, 2], Y=[0, 2, 4], Z=[0, 3, 6])
        dcoef = [numpy.array([1, 4]), 2*numpy.array([1, 4]), 3*numpy.array([1, 4])]
        calc_dcoefs = item.derivative(der_order=1, return_poly=False)
        self.assertTrue(numpy.all(item.derivative_eval(1, 1) == numpy.array([5, 10, 15])))
        self.assertTrue(numpy.all(dcoef[0] == calc_dcoefs[0]) and
                        numpy.all(dcoef[1] == calc_dcoefs[1]) and
                        numpy.all(dcoef[2] == calc_dcoefs[2]), '{}\n{}'.format(dcoef, calc_dcoefs))
        item2 = item.derivative(der_order=2, return_poly=True)
        self.assertTrue(numpy.all(item2.X.Coefs == numpy.array([4, ])) and
                        numpy.all(item2.Y.Coefs == numpy.array([8, ])) and
                        numpy.all(item2.Z.Coefs == numpy.array([12, ]))
                        )


class TestGainPhasePoly(unittest.TestCase):
    def test_construction(self):
        the_dict = {'GainPoly': {'Coefs': [[1, ], ]}, 'PhasePoly': {'Coefs': [[2, ], ]}}
        item1 = blocks.GainPhasePolyType.from_dict(the_dict)
        item2 = blocks.GainPhasePolyType(GainPoly=[[1, ], ], PhasePoly=[[2, ], ])

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'GainPhasePolyType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.GainPhasePolyType.from_node(node)
            self.assertEqual(item1.to_dict(), item3.to_dict())

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())

    def test_eval(self):
        item = blocks.GainPhasePolyType(GainPoly=[[1, ], ], PhasePoly=[[2, ], ])
        self.assertTrue(numpy.all(item(1, 1) == numpy.array([1, 2])))


class TestErrorDecorrFunc(unittest.TestCase):
    def test_construction(self):
        the_dict = {'CorrCoefZero': 0, 'DecorrRate': 0.5}
        item1 = blocks.ErrorDecorrFuncType.from_dict(the_dict)
        item2 = blocks.ErrorDecorrFuncType(CorrCoefZero=0.0, DecorrRate=0.5)

        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = item1.to_dict()
            self.assertEqual(the_dict, new_dict)

        with self.subTest(msg='Test xml serialization issues'):
            # let's serialize to xml
            etree = ElementTree.ElementTree()
            xml = ElementTree.tostring(item1.to_node(etree, 'ErrorDecorrFuncType')).decode('utf-8')
            # let's deserialize from xml
            node = ElementTree.fromstring(xml)
            item3 = blocks.ErrorDecorrFuncType.from_node(node)
            self.assertTrue(
                abs(item1.CorrCoefZero - item3.CorrCoefZero) < 1e-8 and
                abs(item1.DecorrRate - item3.DecorrRate) < 1e-8, 'item1.CorrCoefZero = {}\n'
                                                                 'item1.DecorrRate = {}\n'
                                                                 'item3.CorrCoefZero = {}\n'
                                                                 'item3.DecorrRate = {}\n'.format(
                    item1.CorrCoefZero, item1.DecorrRate, item3.CorrCoefZero, item3.DecorrRate))

        with self.subTest(msg="validity check"):
            self.assertTrue(item1.is_valid())
