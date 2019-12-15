from xml.etree import ElementTree
import numpy

from .. import generic_construction_test, unittest

from sarpy.sicd_elements import blocks


def generic_arrayable_construction_test(instance, the_type, the_dict, array):
    the_item = the_type.from_dict(the_dict)

    with instance.subTest(msg='Comparing json deserialization with original'):
        new_dict = the_item.to_dict()
        instance.assertEqual(the_dict, new_dict)

    with instance.subTest(msg='Test xml serialization issues'):
        # let's serialize to xml
        etree = ElementTree.ElementTree()
        xml = ElementTree.tostring(the_item.to_node(etree, 'The_Type')).decode('utf-8')
        # let's deserialize from xml
        node = ElementTree.fromstring(xml)
        item2 = the_type.from_node(node)
        # let's verify that things are sufficiently equal
        tolerance = 1e-8
        equals = numpy.all(numpy.absolute(the_item.get_array(dtype=numpy.float64) - item2.get_array(dtype=numpy.float64)) <= tolerance)
        instance.assertTrue(equals, msg='xml - {}'.format(xml))

    with instance.subTest(msg='Test validity'):
        instance.assertTrue(the_item.is_valid())

    item2 = the_type.from_array(array)
    with instance.subTest(msg='get_array test'):
        array2 = item2.get_array(dtype=numpy.float64)
        instance.assertTrue(numpy.all(array == array2), msg='{} != {}'.format(array, array2))
    return the_item


class TestXYZType(unittest.TestCase):
    def test_construction(self):
        the_dict = {'X': 1, 'Y': 2, 'Z': 3}
        item1 = generic_arrayable_construction_test(self, blocks.XYZType, the_dict, numpy.array([0.0, 1.0, 2.0]))

        item2 = blocks.XYZType.from_array((1, 2, 3))
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestLatLon(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2}
        item1 = generic_arrayable_construction_test(self, blocks.LatLonType, the_dict, numpy.array([1, 2]))

        item2 = blocks.LatLonType.from_array([1, 2])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestLatLonRestriction(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': -89, 'Lon': 178}
        item1 = generic_arrayable_construction_test(self, blocks.LatLonRestrictionType, the_dict, numpy.array([-89, 178]))

        item2 = blocks.LatLonRestrictionType.from_array([91, -182])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestLatLonArrayElement(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2, 'index': 3}
        item1 = generic_arrayable_construction_test(self, blocks.LatLonArrayElementType, the_dict, numpy.array([1, 2]))

        item2 = blocks.LatLonArrayElementType.from_array([1, 2], index=3)
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestLatLonHAE(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2, 'HAE': 3}
        item1 = generic_arrayable_construction_test(self, blocks.LatLonHAEType, the_dict, numpy.array([1, 2, 3]))

        item2 = blocks.LatLonHAEType.from_array([1, 2, 3])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestLatLonHAERestriction(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': -89, 'Lon': 178, 'HAE': 3}
        item1 = generic_arrayable_construction_test(self, blocks.LatLonHAERestrictionType, the_dict, numpy.array([-89, 178, 3]))

        item2 = blocks.LatLonHAERestrictionType.from_array([91, -182, 3])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestLatLonCorner(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2, 'index': 3}
        item1 = generic_arrayable_construction_test(self, blocks.LatLonCornerType, the_dict, numpy.array([1, 2]))

        item2 = blocks.LatLonCornerType.from_array([1, 2], index=3)
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestLatLonCornerString(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Lat': 1, 'Lon': 2, 'index': '1:FRFC'}
        item1 = generic_arrayable_construction_test(self, blocks.LatLonCornerStringType, the_dict, numpy.array([1, 2]))

        item2 = blocks.LatLonCornerStringType.from_array([1, 2], index='1:FRFC')
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestRowCol(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Row': 1, 'Col': 2}
        item1 = generic_arrayable_construction_test(self, blocks.RowColType, the_dict, numpy.array([1, 2], dtype=numpy.int64))

        item2 = blocks.RowColType.from_array([1, 2])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestRowColArrayElement(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Row': 1, 'Col': 2, 'index': 3}
        item1 = generic_arrayable_construction_test(self, blocks.RowColArrayElement, the_dict, numpy.array([1, 2], dtype=numpy.int64))

        item2 = blocks.RowColArrayElement.from_array([1, 2], index=3)
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())


class TestPoly1D(unittest.TestCase):
    def test_construction(self):
        the_dict = {'Coefs': [0, 1, 2]}
        item1 = generic_arrayable_construction_test(self, blocks.Poly1DType, the_dict, numpy.array([1, 2, 3]))

        item2 = blocks.Poly1DType(Coefs=[0, 1, 2])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

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
        item1 = generic_arrayable_construction_test(
            self, blocks.Poly2DType, the_dict, numpy.array([[0, 0, 0], [0, 1, 2]]))

        item2 = blocks.Poly2DType(Coefs=[[0, 0, 0], [0, 1, 2]])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

    def test_eval(self):
        item = blocks.Poly2DType(Coefs=[[0, 0, 0], [0, 1, 2]])
        self.assertEqual(item(1, 1), 3)


class TestXYZPoly(unittest.TestCase):
    def test_construction(self):
        the_dict = {'X': {'Coefs': [0, 1, 2]}, 'Y': {'Coefs': [0, 2, 4]}, 'Z': {'Coefs': [0, 3, 6]}}
        item1 = generic_arrayable_construction_test(
            self, blocks.XYZPolyType, the_dict, numpy.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]]))

        item2 = blocks.XYZPolyType(X=[0, 1, 2], Y=[0, 2, 4], Z=[0, 3, 6])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

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
        item1 = generic_construction_test(self, blocks.GainPhasePolyType, the_dict)

        item2 = blocks.GainPhasePolyType(GainPoly=[[1, ], ], PhasePoly=[[2, ], ])
        with self.subTest(msg='Comparing from dict construction with alternate construction'):
            self.assertEqual(item1.to_dict(), item2.to_dict())

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
