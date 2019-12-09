import sys
from sarpy.sicd_elements import blocks
from xml.etree import ElementTree

if sys.version_info[0] < 3:
    # so we can use subtests, which is pretty handy
    import unittest2 as unittest
else:
    import unittest


class TestParameter(unittest.TestCase):
    def setUp(self):
        self.test_dict = {'name': 'Name', 'value': 'Value'}
        self.xml = '<Parameter name="Name">Value</Parameter>'
        self.etree = ElementTree.fromstring(self.xml)

    def test_construction(self):
        param1 = blocks.ParameterType.from_dict(self.test_dict)
        param2 = blocks.ParameterType.from_node(node=self.etree)

        with self.subTest(msg='Comparing from dict construction with xml construction'):
            self.assertEqual(param1.to_dict(), param2.to_dict())
        with self.subTest(msg='Comparing json deserialization with original'):
            new_dict = param1.to_dict()
            self.assertEqual(self.test_dict, new_dict)
        with self.subTest(msg="Comparing xml serialization with original"):
            etree = ElementTree.ElementTree()
            new_xml = ElementTree.tostring(param1.to_node(etree, 'Parameter')).decode('utf-8')
            self.assertEqual(self.xml, new_xml)
