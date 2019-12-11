import sys
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest

from sarpy.sicd_elements import RMA


rm_ref_dict = {'PosRef': {'X': 0, 'Y': 0, 'Z': 0}, 'VelRef': {'X': 1, 'Y': 1, 'Z': 1}, 'DopConeAngRef': 45}
inca_dict = {
    'TimeCAPoly': {'Coefs': [0, 1]},
    'R_CA_SCP': 1,
    'FreqZero': 10,
    'DRateSFPoly': {'Coefs': [[1, ], ]},
    'DopCentroidPoly': {'Coefs': [[1, ], ]},
    'DopCentroidCOA': True
}
rma_dict1 = {'RMAlgoType': 'OMEGA_K', 'RMAT': rm_ref_dict, 'ImageType': 'RMAT'}
rma_dict2 = {'RMAlgoType': 'OMEGA_K', 'RMCR': rm_ref_dict, 'ImageType': 'RMCR'}
rma_dict3 = {'RMAlgoType': 'OMEGA_K', 'INCA': inca_dict, 'ImageType': 'INCA'}


class TestRMRef(unittest.TestCase):
    def test_construction(self):
        the_type = RMA.RMRefType
        the_dict = rm_ref_dict
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


class TestINCA(unittest.TestCase):
    def test_construction(self):
        the_type = RMA.INCAType
        the_dict = inca_dict
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


class TestRMA(unittest.TestCase):
    def test_construction1(self):
        the_type = RMA.RMAType
        the_dict = rma_dict1
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

        with self.subTest(msg='ImageType'):
            self.assertEqual(item1.ImageType, 'RMAT')

    def test_construction2(self):
        the_type = RMA.RMAType
        the_dict = rma_dict2
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

        with self.subTest(msg='ImageType'):
            self.assertEqual(item1.ImageType, 'RMCR')

    def test_construction3(self):
        the_type = RMA.RMAType
        the_dict = rma_dict3
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

        with self.subTest(msg='ImageType'):
            self.assertEqual(item1.ImageType, 'INCA')

    def test_bad_construction1(self):
        the_dict = rma_dict1.copy()
        the_dict['RMCR'] = rma_dict2['RMCR']
        item1 = RMA.RMAType.from_dict(the_dict)
        self.assertFalse(item1.is_valid())

    def test_bad_construction2(self):
        the_dict = rma_dict1.copy()
        del the_dict['RMAT']
        item1 = RMA.RMAType.from_dict(the_dict)
        self.assertIsNone(item1.ImageType)
        self.assertFalse(item1.is_valid())
