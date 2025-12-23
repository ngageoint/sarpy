__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"

from collections import OrderedDict
from datetime import datetime, date
import logging
import numpy as np
import re
import types
import unittest
import xml.etree.ElementTree as ET

import sarpy.io.xml.base as base

# Helper for XML test data path
XML_PATH = 'tests/io/xml/'

def get_tree_and_root(filename):
    tree = ET.parse(XML_PATH + filename)
    return tree, tree.getroot()

def get_actor_tree_and_ns():
    return base.parse_xml_from_file(XML_PATH + 'actor_test_data.xml')


# ********************
# get_node_value tests
# ********************
class TestGetNodeValue(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_get_node_value_success_with_text(self):
        self.assertEqual(base.get_node_value(self.root[0][1]), '2008')

    def test_get_node_value_success_none(self):
        self.assertIsNone(base.get_node_value(self.root[0]))

    def test_get_node_value_success_empty(self):
        self.assertIsNone(base.get_node_value(self.root[0][3]))

# ********************
# create_new_node tests
# ********************
class TestCreateNewNode(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_create_new_node_no_parent_success(self):
        new_node_tag = "country"
        len_initial_root = len(self.root)
        new_node = base.create_new_node(self.tree, new_node_tag)
        self.assertEqual(len(self.root), len_initial_root + 1)
        self.assertEqual(self.root[-1].tag, new_node_tag)
        self.assertIs(self.root[-1], new_node)

    def test_create_new_node_empty_tree_success(self):
        new_tree = ET.ElementTree()
        self.assertIsNone(new_tree.getroot())
        new_node = base.create_new_node(new_tree, "country")
        self.assertEqual(new_tree.getroot(), new_node)

    def test_create_new_node_with_parent_success(self):
        new_node_tag = "ocean"
        self.assertEqual(len(self.root[1]), 5)
        new_node = base.create_new_node(self.tree, new_node_tag, self.root[1])
        self.assertEqual(len(self.root[1]), 6)
        self.assertEqual(self.root[1][5].tag, new_node_tag)
        self.assertIs(self.root[1][5], new_node)


# ********************
# create_text_node tests
# ********************
class TestCreateTextNode(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_create_text_node_no_parent_success(self):
        new_node_tag = "country"
        new_node_value = "Costa Rica"
        len_initial_root = len(self.root)
        new_node = base.create_text_node(self.tree, new_node_tag, new_node_value)
        self.assertEqual(len(self.root), len_initial_root + 1)
        self.assertEqual(self.root[-1].tag, new_node_tag)
        self.assertEqual(self.root[-1].text, new_node_value)
        self.assertIs(self.root[-1], new_node)

    def test_create_text_node_with_parent_success(self):
        new_node_tag = "ocean"
        new_node_value = "Pacific"
        self.assertEqual(len(self.root[2]), 6)
        new_node = base.create_text_node(self.tree, new_node_tag, 
                                         new_node_value, self.root[2])
        self.assertEqual(len(self.root[2]), 7)
        self.assertEqual(self.root[2][6].tag, new_node_tag)
        self.assertEqual(self.root[2][6].text, new_node_value)
        self.assertIs(self.root[2][6], new_node)


# ********************
# find_first_child tests
# ********************
class TestFindFirstChild(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_find_first_child_no_optional_params_success(self):
        found_node = base.find_first_child(self.root, "country")
        self.assertIsNotNone(found_node)
        self.assertEqual(found_node.attrib, self.root[0].attrib)

    def test_find_first_child_namespace_params_success(self):
        found_node = base.find_first_child(self.actor_root, "actor", 
                                           self.actor_ns_dict)
        self.assertIsNotNone(found_node)
        self.assertEqual(found_node.attrib, self.actor_root[0].attrib)

    def test_find_first_child_namespace_nskey_params_success(self):
        found_actor_node = base.find_first_child(self.actor_root, "actor", 
                                                 self.actor_ns_dict)
        found_node = base.find_first_child(found_actor_node, "character", 
                                           self.actor_ns_dict, "fictional")
        self.assertIsNotNone(found_node)
        self.assertEqual(found_node.attrib, self.actor_root[0].attrib)

# ********************
# find_children tests
# ********************
class TestFindChildren(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_find_children_no_optional_params_success(self):
        found_nodes = base.find_children(self.root, "country")
        self.assertEqual(found_nodes, self.root.findall("country"))

    def test_find_children_namespace_params_success(self):
        found_node = base.find_children(self.actor_root, "actor", 
                                        self.actor_ns_dict)
        self.assertEqual(found_node, 
                         self.actor_root.findall('actor', self.actor_ns_dict))

    def test_find_children_namespace_nskey_params_success(self):
        found_actor_node = base.find_first_child(self.actor_root, "actor", 
                                                 self.actor_ns_dict)
        found_nodes = base.find_children(found_actor_node, "character", 
                                         self.actor_ns_dict, "fictional")
        self.assertEqual(
            found_nodes,
            found_actor_node.findall('fictional:character', self.actor_ns_dict)
        )

# ********************
# parse_xml_from_string tests
# ********************
class TestParseXmlFromString(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_tree = ET.parse(XML_PATH + 'actor_test_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_parse_xml_from_string_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        root_node, ns_dict = base.parse_xml_from_string(xml_string)
        self.assertEqual(root_node.attrib, self.root.attrib)
        self.assertIsNone(ns_dict)

    def test_parse_xml_from_string_bytes_success(self):
        xml_string = ET.tostring(self.root, encoding='utf-8', method='xml')
        root_node, ns_dict = base.parse_xml_from_string(xml_string)
        self.assertEqual(root_node.attrib, self.root.attrib)
        self.assertIsNone(ns_dict)

    def test_parse_xml_from_string_with_namespace(self):
        xml = '<root xmlns="urn:test"><child/></root>'
        root_node, ns_dict = base.parse_xml_from_string(xml)
        self.assertEqual(root_node.tag, '{urn:test}root')
        self.assertIn('default', ns_dict)
        self.assertEqual(ns_dict['default'], 'urn:test')

    def test_parse_xml_from_string_invalid_xml(self):
        invalid_xml = "<root><unclosed></root>"
        with self.assertRaises(ET.ParseError):
            base.parse_xml_from_string(invalid_xml)

    def test_parse_xml_from_string_namespace_match_none(self):
        xml = ET.tostring(self.actor_root, encoding='utf-8', method='xml')
        original_match = re.match
        re.match = lambda pattern, string: None
        with self.assertRaisesRegex(ValueError, r"Trouble finding the " + \
                                    "default namespace for tag " + \
                                    "\{http:\/\/people.example.com\}actors$"):
            base.parse_xml_from_string(xml)
        re.match = original_match

    def test_parse_xml_from_string_namespace_match_not_none(self):
        xml = '''
            <Dummy:table xmlns:Dummy='http://www.example.com/schema'>
                <Dummy:child>value</Dummy:child>
            </Dummy:table>
        '''
        root_node, xml_ns = base.parse_xml_from_string(xml)
        self.assertTrue(root_node.tag.endswith('table'))
        self.assertIn('default', xml_ns)
        self.assertEqual(xml_ns['default'], 'http://www.example.com/schema')
    
# ********************
# parse_xml_from_file tests
# ********************
class TestParseXmlFromFile(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_parse_xml_from_file_success(self):
        test_root, test_ns_dict = base.parse_xml_from_file(XML_PATH + 
                                                           'country_data.xml')
        self.assertEqual(test_root.attrib, self.root.attrib)
        
# ********************
# validate_xml_from_string tests
# ********************
class TestValidateXmlFromString(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_validate_xml_from_string_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        xsd_path = XML_PATH + 'country.xsd'
        self.assertTrue(base.validate_xml_from_string(xml_string, xsd_path))

    def test_validate_xml_from_string_with_logger_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        xsd_path = XML_PATH + 'country.xsd'
        self.assertTrue(base.validate_xml_from_string(xml_string, xsd_path, 
                                                      base.logger))

    def test_validate_xml_from_string_not_valid_output_logger_none(self):
        class DummyEntry:
            line = 10
            message = "Invalid element"
        class DummySchema:
            def validate(self, doc): return False
            @property
            def error_log(self): return [DummyEntry()]
        class DummyDoc: pass
        original_etree = base.etree
        base.etree = types.SimpleNamespace(
            fromstring=lambda x: DummyDoc(),
            XMLSchema=lambda file: DummySchema()
        )
        from unittest.mock import patch
        with patch.object(base, "logger") as mock_logger:
            result = base.validate_xml_from_string(b"<root></root>", 
                                                   "fake.xsd", 
                                                   output_logger=None)
            self.assertFalse(result)
            self.assertTrue(mock_logger.error.called)
            self.assertIn("XML validation error on line", 
                          str(mock_logger.error.call_args[0][0]))
        base.etree = original_etree

    def test_validate_xml_from_string_not_valid_output_logger_not_none(self):
        class DummyEntry:
            line = 10
            message = "Invalid element"
        class DummySchema:
            def validate(self, doc): return False
            @property
            def error_log(self): return [DummyEntry()]
        class DummyDoc: pass
        original_etree = base.etree
        base.etree = types.SimpleNamespace(
            fromstring=lambda x: DummyDoc(),
            XMLSchema=lambda file: DummySchema()
        )
        class DummyLogger:
            def __init__(self): self.logged = []
            def error(self, msg): self.logged.append(msg)
        dummy_logger = DummyLogger()
        result = base.validate_xml_from_string(b"<root></root>", "fake.xsd", 
                                               output_logger=dummy_logger)
        self.assertFalse(result)
        self.assertIn("XML validation error on line", dummy_logger.logged[0])
        base.etree = original_etree

# ********************
# validate_xml_from_file tests
# ********************
class TestValidateXmlFromFile(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_validate_xml_from_file_success(self):
        xml_path = XML_PATH + 'country_data.xml'
        xsd_path = XML_PATH + 'country.xsd'
        self.assertTrue(base.validate_xml_from_file(xml_path, xsd_path))

    def test_validate_xml_from_file_with_logger_success(self):
        xml_path = XML_PATH + 'country_data.xml'
        xsd_path = XML_PATH + 'country.xsd'
        self.assertTrue(base.validate_xml_from_file(xml_path, xsd_path, 
                                                    base.logger))
        
# ********************
# parse_str tests
# ********************
class TestParseStr(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_parse_str_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_str\(\) missing 3 " + \
                                    "required positional arguments: 'value', " + \
                                    "'name', and 'instance'$"):
            base.parse_str()

    def test_parse_str_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_str\(\) missing 2 " + \
                                    "required positional arguments: 'name' " + \
                                    "and 'instance'$"):
            base.parse_str("Test")

    def test_parse_str_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_str\(\) missing 1 " + \
                                    "required positional argument: 'instance'$"):
            base.parse_str("Test", "Bob")

    def test_parse_str_value_param_is_string_success(self):
        self.assertEqual(base.parse_str("Test", "Bob", "base"), "Test")

    def test_parse_str_value_param_is_None_success(self):
        self.assertIsNone(base.parse_str(None, "Bob", "base"))

    def test_parse_str_value_param_is_xml_with_value_success(self):
        self.assertEqual(base.parse_str(self.root[0][2], "text", "base"), 
                         self.root[0][2].text)

    def test_parse_str_value_param_is_xml_empty_value_success(self):
        self.assertEqual(base.parse_str(self.root[0], "text", "base"), 
                         self.root[0].text.strip())

    def test_parse_str_bad_value_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"field Bob of class str " + \
                                    "requires a string value."):
            base.parse_str(1, "Bob", "base")

# ********************
# parse_bool tests
# ********************
class TestParseBool(unittest.TestCase):
    def setUp(self):
        self.tree, self.root = get_tree_and_root('country_data.xml')
        self.actor_root, self.actor_ns_dict = get_actor_tree_and_ns()

    def test_parse_bool_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_bool\(\) missing 3 " + \
                                    "required positional arguments: 'value', " + \
                                    "'name', and 'instance'$"):
            base.parse_bool()

    def test_parse_bool_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_bool\(\) missing 2 " + \
                                    "required positional arguments: 'name' " + \
                                    "and 'instance'$"):
            base.parse_bool("Test")

    def test_parse_bool_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_bool\(\) missing 1 " + \
                                    "required positional argument: 'instance'$"):
            base.parse_bool("Test", "Bob")

    def test_parse_bool_value_param_is_None_success(self):
        self.assertIsNone(base.parse_bool(None, "Bob", "base"))

    def test_parse_bool_value_param_is_bool_success(self):
        self.assertTrue(base.parse_bool(True, "Bob", "base"))

    def test_parse_bool_value_param_is_int_success(self):
        self.assertTrue(base.parse_bool(1, "Bob", "base"))

    def test_parse_bool_value_param_is_np_bool_success(self):
        arr_bool = np.array([True, False, True, False], dtype=bool)
        self.assertTrue(base.parse_bool(arr_bool[0], "Bob", "base"))

    def test_parse_bool_value_param_is_xml_success(self):
        self.assertTrue(base.parse_bool(self.root[0][0], "Bob", "base"))

    def test_parse_bool_value_param_is_string_true_success(self):
        self.assertTrue(base.parse_bool('trUe', "Bob", "base"))

    def test_parse_bool_value_param_is_string_1_success(self):
        self.assertTrue(base.parse_bool('1', "Bob", "base"))

    def test_parse_bool_value_param_is_string_false_success(self):
        self.assertFalse(base.parse_bool('FALSE', "Bob", "base"))

    def test_parse_bool_value_param_is_string_0_success(self):
        self.assertFalse(base.parse_bool('0', "Bob", "base"))

    def test_parse_bool_value_param_is_float_fail(self):
        with self.assertRaisesRegex(ValueError, r"Boolean field Bob of class " + \
                                    "str cannot assign from type <class " + \
                                    "'float'>."):
            base.parse_bool(3.5, "Bob", "base")

    def test_parse_bool_parse_string_invalid_value(self):
        class Dummy: pass
        with self.assertRaisesRegex(ValueError, "Boolean field field of " + \
                                    "class Dummy cannot assign from string " + \
                                    "value maybe."):
            base.parse_bool("maybe", "field", Dummy())

# ********************
# parse_int tests
# ********************
class TestParseInt(unittest.TestCase):
    def setUp(self):
        self.tree = ET.parse('tests/io/xml/country_data.xml')
        self.actor_tree = ET.parse('tests/io/xml/actor_test_data.xml')
        self.root = self.tree.getroot()
        # For xml ns is an abbreviation for name space
        self.actor_root, self.actor_ns_dict = \
            base.parse_xml_from_file('tests/io/xml/actor_test_data.xml') 

    def test_parse_int_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_int\(\) missing 3 " + \
                                    "required positional arguments: 'value', " + \
                                    "'name', and 'instance'$"):
            base.parse_int()
        
    def test_parse_int_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_int\(\) missing 2 " + \
                                    "required positional arguments: 'name' " + \
                                    "and 'instance'$"):
            base.parse_int("Test")

    def test_parse_int_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_int\(\) missing 1 " + \
                                    "required positional argument: 'instance'$"):
            base.parse_int("Test", "Bob")

    def test_parse_int_value_param_is_None_success(self):
        self.assertIsNone(base.parse_int(None, "Bob", "base"))

    def test_parse_int_value_param_is_int_success(self):
        self.assertEqual(base.parse_int(1, "Bob", "base"), 1)

    def test_parse_int_value_param_is_xml_success(self):
        self.assertEqual(base.parse_int(self.root[0][0], "Bob", "base"), 1)

    def test_parse_int_value_param_is_string_1_success(self):
        self.assertEqual(base.parse_int('1', "Bob", "base"), 1)

    def test_parse_int_value_param_is_string_non_int_success(self):
        with self.assertRaisesRegex(ValueError, r"invalid literal for " + \
                                    "int\(\) with base 10: 'Bob'"):
            assert(base.parse_int('Bob', "Bob", "base") == 1)

    def test_parse_int_value_param_is_list_non_int_success(self):
        with self.assertRaisesRegex(TypeError, r"int\(\) argument must be a " + \
                                    "string, a bytes-like object or a real " + \
                                    "number, not 'list'"):
            assert(base.parse_int([3.5], "Bob", "base") == 1)

# ********************
# parse_float tests
# ********************
class TestParseFloat(unittest.TestCase):
    def setUp(self):
        self.tree = ET.parse('tests/io/xml/country_data.xml')
        self.actor_tree = ET.parse('tests/io/xml/actor_test_data.xml')
        self.root = self.tree.getroot()
        # For xml ns is an abbreviation for name space
        self.actor_root, self.actor_ns_dict = \
            base.parse_xml_from_file('tests/io/xml/actor_test_data.xml') 

    def test_parse_float_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_float\(\) missing 3 " + \
                                    "required positional arguments: 'value', " + \
                                    "'name', and 'instance'$"):
            base.parse_float()
        
    def test_parse_float_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_float\(\) missing 2 " + \
                                    "required positional arguments: 'name' " + \
                                    "and 'instance'$"):
            base.parse_float("Test")

    def test_parse_float_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_float\(\) missing 1 " + \
                                    "required positional argument: 'instance'$"):
            base.parse_float("Test", "Bob")

    def test_parse_float_value_param_is_None_success(self):
        self.assertIsNone(base.parse_float(None, "Bob", "base"))

    def test_parse_float_value_param_is_float_success(self):
        self.assertEqual(base.parse_float(1.5, "Bob", "base"), 1.5)

    def test_parse_float_value_param_is_xml_success(self):
        self.assertEqual(base.parse_float(self.root[0][0], "Bob", "base"), 1.0)

    def test_parse_float_value_param_is_string_1dot5_success(self):
        self.assertEqual(base.parse_float('1.5', "Bob", "base"), 1.5)

    def test_parse_float_value_param_is_string_non_int_success(self):
        with self.assertRaisesRegex(ValueError, r"could not convert string " + \
                                    "to float: 'Bob'"):
            base.parse_float('Bob', "Bob", "base")

    def test_parse_float_value_param_is_list_non_int_success(self):
        with self.assertRaisesRegex(TypeError, r"float\(\) argument must be " + \
                                    "a string or a real number, not 'list'"):
            base.parse_float([3.5], "Bob", "base")

# ********************
# parse_complex tests
# ********************

class TestParseComplex(unittest.TestCase):
    def setUp(self):
        self.tree = ET.parse('tests/io/xml/country_data.xml')
        self.actor_tree = ET.parse('tests/io/xml/actor_test_data.xml')
        self.root = self.tree.getroot()
        # For xml ns is an abbreviation for name space
        self.actor_root, self.actor_ns_dict = \
            base.parse_xml_from_file('tests/io/xml/actor_test_data.xml') 

    def test_parse_complex_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_complex\(\) missing " + \
                                    "3 required positional arguments: 'value'," + \
                                    " 'name', and 'instance'$"):
            base.parse_complex()
        
    def test_parse_complex_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_complex\(\) missing 2 " + \
                                    "required positional arguments: 'name' " + \
                                    "and 'instance'$"):
            base.parse_complex("Test")

    def test_parse_complex_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_complex\(\) missing 1 " + \
                                    "required positional argument: 'instance'$"):
            base.parse_complex("Test", "Bob")

    def test_parse_complex_value_param_is_None_success(self):
        self.assertIsNone(base.parse_complex(None, "Bob", "base"))

    def test_parse_complex_value_param_is_complex_success(self):
        test_complex = 3 + 2j
        self.assertEqual(base.parse_complex(test_complex, "Bob", "base"), 
                         test_complex)

    def test_parse_complex_value_param_is_xml_success(self):
        test_complex = 3 + 2j
        self.assertEqual(base.parse_complex(self.root[0][4], "Bob", "base"), 
                         test_complex)
        
    def test_parse_complex_with_child_xml_ns_key(self):
        class DummyInstance:
            _xml_ns = {'ns1': 'urn:ns1'}
            _child_xml_ns_key = {'field': 'ns1'}
            __class__ = type('DummyInstance', (), {})
        # XML with Real and Imag in ns1 namespace
        xml = """
        <Complex xmlns:ns1="urn:ns1">
            <ns1:Real>3</ns1:Real>
            <ns1:Imag>2</ns1:Imag>
        </Complex>
        """
        elem = ET.fromstring(xml)
        result = base.parse_complex(elem, 'field', DummyInstance())
        assert result == complex(3, 2)

    def test_parse_complex_value_param_is_xml_2_real_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"There must be exactly one " + \
                                    "Real component of a complex type node " + \
                                    "defined for field Bob of class str."):
            base.parse_complex(self.root[1][3], "Bob", "base")

    def test_parse_complex_value_param_is_xml_2_imag_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"There must be exactly one " + \
                                    "Imag component of a complex type node " + \
                                    "defined for field Bob of class str."):
            base.parse_complex(self.root[2][3], "Bob", "base")

    def test_parse_complex_value_param_is_complex_dict_1_success(self):
        test_complex = 3 + 2j
        self.assertEqual(base.parse_complex({"real":3, "imag":2}, "Bob", "base"), 
                         test_complex)

    def test_parse_complex_value_param_is_complex_dict_2_success(self):
        test_complex = 3 + 2j
        self.assertEqual(base.parse_complex({"Real":3, "Imag":2}, "Bob", "base"), 
                         test_complex)

    def test_parse_complex_value_param_is_complex_dict_3_success(self):
        test_complex = 3 + 2j
        self.assertEqual(base.parse_complex({"re":3, "im":2}, "Bob", "base"), 
                         test_complex)

    def test_parse_complex_value_param_is_complex_dict_4_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"Cannot convert dict {'not': " + \
                                    "3, 'valid': 2} to a complex number for " + \
                                    "field Bob of class str."):
            base.parse_complex({"not":3, "valid":2}, "Bob", "base")

    def test_parse_complex_value_param_is_complex_dict_5_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"Cannot convert dict {'real': " + \
                                    "None, 'imag': 2} to a complex number " + \
                                    "for field Bob of class str."):
            base.parse_complex({"real":None, "imag":2}, "Bob", "base")

    def test_parse_complex_value_param_is_complex_dict_6_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"Cannot convert dict {'real': " + \
                                    "4, 'imag': None} to a complex number for " + \
                                    "field Bob of class str."):
            base.parse_complex({"real":4, "imag":None}, "Bob", "base")

    def test_parse_complex_value_param_is_string_non_int_success(self):
        with self.assertRaisesRegex(ValueError, r"complex\(\) arg is a " + \
                                    "malformed string"):
            base.parse_complex('Bob', "Bob", "base")

    def test_parse_complex_value_param_is_list_non_int_success(self):
        with self.assertRaisesRegex(TypeError, r"complex\(\) first argument " + \
                                    "must be a string or a number, not 'list'"):
            base.parse_complex([3.5], "Bob", "base")

# ********************
# parse_datetime tests
# ********************

class ParseDatetimeDummyInstance:
    pass

class TestParseDatetime(unittest.TestCase):
    
    def test_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_datetime\(\) missing " + \
                                    "3 required positional arguments: 'value'," + \
                                    " 'name', and 'instance'$"):
            base.parse_datetime()
        
    def test_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_datetime\(\) missing 2 " + \
                                    "required positional arguments: 'name' " + \
                                    "and 'instance'$"):
            base.parse_datetime("Test")

    def test_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_datetime\(\) missing 1 " + \
                                    "required positional argument: 'instance'$"):
            base.parse_datetime("Test", "Bob")

    def test_none_returns_none(self):
        self.assertIsNone(base.parse_datetime(None, "dt", 
                                              ParseDatetimeDummyInstance()))

    def test_numpy_datetime64_pass_through(self):
        dt = np.datetime64('2023-01-01T12:00:00')
        self.assertEqual(base.parse_datetime(dt, "dt", 
                                             ParseDatetimeDummyInstance()), dt)

    def test_string_with_Z(self):
        dt_str = "2023-01-01T12:00:00Z"
        result = base.parse_datetime(dt_str, "dt", ParseDatetimeDummyInstance())
        self.assertIsInstance(result, np.datetime64)
        self.assertEqual(result, np.datetime64("2023-01-01T12:00:00"))

    def test_string_without_Z(self):
        dt_str = "2023-01-01T12:00:00"
        result = base.parse_datetime(dt_str, "dt", ParseDatetimeDummyInstance())
        self.assertIsInstance(result, np.datetime64)
        self.assertEqual(result, np.datetime64("2023-01-01T12:00:00"))

    def test_elementtree_element(self):
        elem = ET.Element("Test")
        elem.text = "2023-01-01T12:00:00"
        result = base.parse_datetime(elem, "dt", ParseDatetimeDummyInstance())
        self.assertIsInstance(result, np.datetime64)
        self.assertEqual(result, np.datetime64("2023-01-01T12:00:00"))

    def test_date_object(self):
        d = date(2023, 1, 1)
        result = base.parse_datetime(d, "dt", ParseDatetimeDummyInstance())
        self.assertIsInstance(result, np.datetime64)
        self.assertEqual(result, np.datetime64("2023-01-01"))

    def test_datetime_object(self):
        d = datetime(2023, 1, 1, 12, 0, 0)
        result = base.parse_datetime(d, "dt", ParseDatetimeDummyInstance())
        self.assertIsInstance(result, np.datetime64)
        self.assertEqual(result, np.datetime64("2023-01-01T12:00:00"))

    def test_numpy_int64(self):
        val = np.int64(1700000000)
        result = base.parse_datetime(val, "dt", ParseDatetimeDummyInstance())
        self.assertIsInstance(result, np.datetime64)
        self.assertEqual(result, numpy.datetime64('1970-01-01T00:28:20.000000'))

    def test_numpy_float64(self):
        val = np.float64(1700000000)
        result = base.parse_datetime(val, "dt", ParseDatetimeDummyInstance())
        self.assertIsInstance(result, np.datetime64)
        self.assertEqual(result, numpy.datetime64('1970-01-01T00:28:20.000000'))

    def test_int(self):
        val = 1700000000
        result = base.parse_datetime(val, "dt", ParseDatetimeDummyInstance())
        self.assertIsInstance(result, np.datetime64)
        self.assertEqual(result, numpy.datetime64('1970-01-01T00:28:20.000000'))

    def test_invalid_type_raises(self):
        with self.assertRaisesRegex(TypeError, r"Field dt for class " + \
                                    "ParseDatetimeDummyInstance expects " + \
                                    "datetime convertible input, and got " + \
                                    "<class 'list'>$"):
            base.parse_datetime([2023, 1, 1], "dt", ParseDatetimeDummyInstance())

# ********************
# parse_serializable tests
# ********************

class ParseSerializableDummyType:
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None):
        return cls(node=node, xml_ns=xml_ns, ns_key=ns_key)
    @classmethod
    def from_array(cls, arr):
        return cls(arr=arr)
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ParseSerializableDummyArrayable(base.Arrayable):
    @classmethod
    def from_array(cls, arr):
        return cls(arr)
    def __init__(self, arr):
        self.arr = arr

class ParseSerializableDummyInstance:
    _xml_ns = {'default': 'urn:test'}
    _xml_ns_key = 'default'
    _child_xml_ns_key = {'foo': 'default'}

class TestParseSerializable(unittest.TestCase):
    def setUp(self):
        self.tree = ET.parse('tests/io/xml/country_data.xml')
        self.actor_tree = ET.parse('tests/io/xml/actor_test_data.xml')
        self.root = self.tree.getroot()
        # For xml ns is an abbreviation for name space
        self.actor_root, self.actor_ns_dict = \
            base.parse_xml_from_file('tests/io/xml/actor_test_data.xml') 

    def test_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_serializable\(\) missing " + \
                                    "4 required positional arguments: 'value'," + \
                                    " 'name', 'instance', and 'the_type'$"):
            base.parse_serializable()
            
    def test_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_serializable\(\) " + \
                                    "missing 3 required positional arguments: " + \
                                    "'name', 'instance', and 'the_type'$"):
            base.parse_serializable("Test")

    def test_value_name_params_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_serializable\(\) " + \
                                    "missing 2 required positional arguments: " + \
                                    "'instance' and 'the_type'$"):
            base.parse_serializable("Test", "foo")

    def test_value_name_instance_params_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_serializable\(\) " + \
                                    "missing 1 required positional argument: " + \
                                    "'the_type'$"):
            base.parse_serializable("Test", "foo", 
                                    ParseSerializableDummyInstance())

    def test_none(self):
        self.assertIsNone(base.parse_serializable(None, 'foo', 
                                                  ParseSerializableDummyInstance(), 
                                                  ParseSerializableDummyType))

    def test_instance(self):
        obj = ParseSerializableDummyType(a=1)
        self.assertIs(base.parse_serializable(obj, 'foo', 
                                              ParseSerializableDummyInstance(), 
                                              ParseSerializableDummyType), obj)

    def test_dict(self):
        result = base.parse_serializable({'a': 1}, 'foo', 
                                         ParseSerializableDummyInstance(), 
                                         ParseSerializableDummyType)
        self.assertIsInstance(result, ParseSerializableDummyType)
        self.assertEqual(result.a, 1)

    def test_element(self):
        elem = ET.Element('Dummy')
        result = base.parse_serializable(elem, 'foo', 
                                         ParseSerializableDummyInstance(), 
                                         ParseSerializableDummyType)
        self.assertIsInstance(result, ParseSerializableDummyType)
        self.assertEqual(result.node, elem)
        self.assertEqual(result.xml_ns, ParseSerializableDummyInstance._xml_ns)
        self.assertEqual(result.ns_key, 
                         ParseSerializableDummyInstance._child_xml_ns_key['foo'])

    def test_arrayable_ndarray(self):
        arr = np.array([1, 2, 3])
        result = base.parse_serializable(arr, 'foo', 
                                         ParseSerializableDummyInstance(), 
                                         ParseSerializableDummyArrayable)
        self.assertIsInstance(result, ParseSerializableDummyArrayable)
        np.testing.assert_array_equal(result.arr, arr)

    def test_arrayable_ndarray_bad_type_fail(self):
        arr = np.array([1, 2, 3])
        with self.assertRaisesRegex(TypeError, r"Field foo of class " + \
                                    "ParseSerializableDummyInstance is of type " + \
                                    "<class 'int'> \(not a subclass of " + \
                                    "Arrayable\) and got an argument of type " + \
                                    "<class 'numpy.ndarray'>.$"):
            result = base.parse_serializable(arr, 'foo', 
                                             ParseSerializableDummyInstance(), 
                                             int)

    def test_arrayable_list(self):
        arr = [1, 2, 3]
        result = base.parse_serializable(arr, 'foo', 
                                         ParseSerializableDummyInstance(), 
                                         ParseSerializableDummyArrayable)
        self.assertIsInstance(result, ParseSerializableDummyArrayable)
        self.assertEqual(result.arr, arr)

    def test_arrayable_tuple(self):
        arr = (1, 2, 3)
        result = base.parse_serializable(arr, 'foo', 
                                         ParseSerializableDummyInstance(), 
                                         ParseSerializableDummyArrayable)
        self.assertIsInstance(result, ParseSerializableDummyArrayable)
        self.assertEqual(result.arr, arr)

    def test_non_arrayable_array(self):
        arr = [1, 2, 3]
        with self.assertRaises(TypeError):
            base.parse_serializable(arr, 'foo', 
                                    ParseSerializableDummyInstance(), 
                                    ParseSerializableDummyType)

    def test_invalid_type(self):
        with self.assertRaisesRegex(TypeError, r"Field foo of class " + \
                                    "ParseSerializableDummyInstance is " + \
                                    "expecting type <class " + \
                                    "'test_base_functions." + \
                                    "ParseSerializableDummyType'>, but got an " + \
                                    "instance of incompatible type " + \
                                    "<class 'float'>.$"):
            base.parse_serializable(123.456, 'foo', 
                                    ParseSerializableDummyInstance(), 
                                    ParseSerializableDummyType)
            
    def test_parse_serializable_without_child_xml_ns_key(self):
        class DummyType:
            @classmethod
            def from_node(cls, node, xml_ns, ns_key=None):
                # Just return a tuple for test
                return (node.tag, xml_ns, ns_key)
            @classmethod
            def from_dict(cls, d):
                return d

        class DummyInstance:
            _xml_ns = {'default': 'urn:default'}
            _xml_ns_key = 'default'
            # _child_xml_ns_key is not set

        xml = '<TestTag>value</TestTag>'
        elem = ET.fromstring(xml)
        result = base.parse_serializable(elem, 'field', DummyInstance(), 
                                         DummyType)
        self.assertEqual(result[0], 'TestTag')
        self.assertEqual(result[1], {'default': 'urn:default'})
        self.assertEqual(result[2], 'default')

# ********************
# parse_serializable_array tests
# ********************

class ParseSerializableArrayDummyArrayable(base.Arrayable):
    def __init__(self, arr):
        self.arr = np.array(arr)
    @classmethod
    def from_array(cls, arr):
        return cls(arr)
    def get_array(self, dtype=None):
        return self.arr

class ParseSerializableArrayDummySerializable:
    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None):
        return cls(node.tag)
    @classmethod
    def from_dict(cls, d):
        return cls(d['tag'])
    def __init__(self, tag):
        self.tag = tag

class ParseSerializableArrayDummyInstance:
    _xml_ns = None
    _xml_ns_key = None
    _child_xml_ns_key = {}

class TestParseSerializableArray(unittest.TestCase):
    def test_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_serializable_array\(\) " + \
                                    "missing 5 required positional arguments: " + \
                                    "'value', 'name', 'instance', " + \
                                    "'child_type', and 'child_tag'$"):
            base.parse_serializable_array()
            
    def test_value_param_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_serializable_array\(\) " + \
                                    "missing 4 required positional arguments: " + \
                                    "'name', 'instance', 'child_type', " + \
                                    "and 'child_tag'$"):
            base.parse_serializable_array('foo')

    def test_value_name_params_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_serializable_array\(\) " + \
                                    "missing 3 required positional arguments: " + \
                                    "'instance', 'child_type', and 'child_tag'$"):
            base.parse_serializable_array('foo', 'bar')

    def test_value_name_instance_params_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_serializable_array\(\) " + \
                                    "missing 2 required positional arguments: " + \
                                    "'child_type' and 'child_tag'$"):
            base.parse_serializable_array('foo', 'bar', 
                                          ParseSerializableArrayDummyInstance())

    def test_value_name_instance_child_type_params_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_serializable_array\(\) " + \
                                    "missing 1 required positional argument: " + \
                                    "'child_tag'$"):
            base.parse_serializable_array('foo', 'bar', 
                                          ParseSerializableArrayDummyInstance(),
                                          ParseSerializableArrayDummySerializable)
            
    def test_none_returns_none(self):
        self.assertIsNone(base.\
                          parse_serializable_array(None, 'test',
                                                   ParseSerializableArrayDummyInstance(),
                                                   ParseSerializableArrayDummySerializable,
                                                   'child'))

    def test_single_child_type(self):
        obj = ParseSerializableArrayDummySerializable('child')
        arr = base.parse_serializable_array(obj, 'test', 
                                            ParseSerializableArrayDummyInstance(), 
                                            ParseSerializableArrayDummySerializable, 
                                            'child')
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.size, 1)
        self.assertIs(arr[0], obj)

    def test_ndarray_of_arrayable(self):
        arr = np.array([[1, 2], [3, 4]])
        result = base.parse_serializable_array(arr, 'test', 
                                               ParseSerializableArrayDummyInstance(), 
                                               ParseSerializableArrayDummyArrayable, 
                                               'child')
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 2)
        self.assertTrue(all(
            isinstance(x, ParseSerializableArrayDummyArrayable) for x in result)
        )

    def test_ndarray_wrong_dtype(self):
        arr = np.array([1, 2, 3], dtype=int)
        with self.assertRaisesRegex(ValueError, r"Attribute test of array " + \
                                    "type functionality belonging to class " + \
                                    "ParseSerializableArrayDummyInstance got " + \
                                    "an ndarray of dtype int64,and child " + \
                                    "type is not a subclass of Arrayable.$"):
            base.parse_serializable_array(arr, 'test', 
                                          ParseSerializableArrayDummyInstance(), 
                                          ParseSerializableArrayDummySerializable, 
                                          'child')

    def test_ndarray_wrong_shape(self):
        arr = np.empty((2,2), dtype=object)
        arr[0,0] = ParseSerializableArrayDummySerializable('child')
        arr[0,1] = ParseSerializableArrayDummySerializable('child')
        arr[1,0] = ParseSerializableArrayDummySerializable('child')
        arr[1,1] = ParseSerializableArrayDummySerializable('child')
        with self.assertRaisesRegex(ValueError, r"Attribute test of array " + \
                                    "type functionality belonging to class " + \
                                    "ParseSerializableArrayDummyInstance got " + \
                                    "an ndarray of shape \(2, 2\),but requires " + \
                                    "a one dimensional array.$"):
            base.parse_serializable_array(arr, 'test', 
                                          ParseSerializableArrayDummyInstance(), 
                                          ParseSerializableArrayDummySerializable, 
                                          'child')

    def test_ndarray_wrong_type(self):
        arr = np.array([1, 2, 3], dtype=object)
        with self.assertRaisesRegex(TypeError, r"Attribute test of array type " + \
                                    "functionality belonging to class " + \
                                    "ParseSerializableArrayDummyInstance got " + \
                                    "an ndarray containing first element of " + \
                                    "incompatible type <class 'int'>.$"):
            base.parse_serializable_array(arr, 'test', 
                                          ParseSerializableArrayDummyInstance(), 
                                          ParseSerializableArrayDummySerializable, 
                                          'child')

    def test_xml_element(self):
        xml = "<parent size='2'><child/><child/></parent>"
        elem = ET.fromstring(xml)
        result = base.parse_serializable_array(elem, 'test', 
                                               ParseSerializableArrayDummyInstance(), 
                                               ParseSerializableArrayDummySerializable, 
                                               'child')
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 2)
        self.assertTrue(
            all(isinstance(x, ParseSerializableArrayDummySerializable) 
                for x in result))

    def test_xml_element_wrong_size(self):
        xml = "<parent size='3'><child/><child/></parent>"
        elem = ET.fromstring(xml)
        with self.assertRaisesRegex(ValueError, r"Attribute test of array " + \
                                    "type functionality belonging to class " + \
                                    "ParseSerializableArrayDummyInstance got " + \
                                    "a ElementTree element with size " + \
                                    "attribute 3, but has 2 child nodes " + \
                                    "with tag child.$"):
            base.parse_serializable_array(elem, 'test', 
                                          ParseSerializableArrayDummyInstance(), 
                                          ParseSerializableArrayDummySerializable, 
                                          'child')
            
    def test_parse_serializable_array_list_first_element_incompatible(self):
        class DummyChild:
            pass

        # First element is not DummyChild, not dict, not arrayable, not list/tuple/ndarray
        values = [42, 43]
        with self.assertRaisesRegex(TypeError, 'Attribute field of array type ' + \
                                    'functionality belonging to class NoneType ' + \
                                    'got a list containing first element of ' + \
                                    'incompatible type <class \'int\'>.'):
            base.parse_serializable_array(values, 'field', None, DummyChild, 
                                          'Child')

    def test_list_of_child_type(self):
        objs = [ParseSerializableArrayDummySerializable('child'), 
                ParseSerializableArrayDummySerializable('child')]
        arr = base.parse_serializable_array(objs, 'test', 
                                            ParseSerializableArrayDummyInstance(), 
                                            ParseSerializableArrayDummySerializable, 
                                            'child')
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.size, 2)
        self.assertTrue(all(isinstance(x, ParseSerializableArrayDummySerializable) 
                            for x in arr))

    def test_list_of_dict(self):
        dicts = [{'tag': 'child'}, {'tag': 'child2'}]
        arr = base.parse_serializable_array(dicts, 'test', 
                                            ParseSerializableArrayDummyInstance(), 
                                            ParseSerializableArrayDummySerializable, 
                                            'child')
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.size, 2)
        self.assertTrue(all(
            isinstance(x, ParseSerializableArrayDummySerializable) for x in arr)
            )

    def test_list_of_arrayable(self):
        arrs = [[1,2], [3,4]]
        arr = base.parse_serializable_array(arrs, 'test', 
                                            ParseSerializableArrayDummyInstance(), 
                                            ParseSerializableArrayDummyArrayable, 
                                            'child')
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.size, 2)
        self.assertTrue(all(
            isinstance(x, ParseSerializableArrayDummyArrayable) for x in arr))

    def test_list_of_incompatible_type(self):
        arrs = [1, 2]
        with self.assertRaisesRegex(TypeError, r"Attribute test of array type " + \
                                    "functionality belonging to class " + \
                                    "ParseSerializableArrayDummyInstance got " + \
                                    "a list containing first element of " + \
                                    "incompatible type <class 'int'>.$"):
            base.parse_serializable_array(arrs, 'test', 
                                          ParseSerializableArrayDummyInstance(), 
                                          ParseSerializableArrayDummySerializable, 
                                          'child')

    def test_empty_list(self):
        arr = base.parse_serializable_array([], 'test', 
                                            ParseSerializableArrayDummyInstance(), 
                                            ParseSerializableArrayDummySerializable, 
                                            'child')
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.size, 0)

    def test_parse_serializable_array_ndarray_object_1d_child_type(self):
        class DummyChild:
            pass

        # Create a 1D numpy array of dtype 'object' where the first element is DummyChild
        arr = np.array([DummyChild(), DummyChild()], dtype=object)
        result = base.parse_serializable_array(arr, 'field', None, DummyChild, 'Child')
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.dtype.name, 'object')
        self.assertEqual(len(result.shape), 1)
        self.assertTrue(isinstance(result[0], DummyChild))

    def test_parse_serializable_array_element_no_child_xml_ns_key(self):
        class DummyChild:
            @classmethod
            def from_node(cls, node, xml_ns, ns_key=None):
                # For test, return tuple of tag, ns_key, and xml_ns
                return (node.tag, ns_key, xml_ns)
        class DummyInstance:
            _xml_ns = {'default': 'urn:default'}
            _xml_ns_key = 'default'
            # _child_xml_ns_key is NOT set

        xml = """
        <Parent size="2" xmlns="urn:default">
            <Child/>
            <Child/>
        </Parent>
        """
        elem = ET.fromstring(xml)
        arr = base.parse_serializable_array(elem, 'field', DummyInstance(), 
                                            DummyChild, 'Child')
        self.assertEqual(len(arr), 2)
        self.assertEqual(arr[0][1], 'default')
        self.assertEqual(arr[0][2], {'default': 'urn:default'})

    def test_parse_serializable_array_element_size_minus_one(self):
        class DummyChild:
            @classmethod
            def from_node(cls, node, xml_ns, ns_key=None):
                # For test, return node.tag
                return node.tag
        class DummyInstance:
            _xml_ns = None
            _xml_ns_key = None

        # XML with size attribute set to -1, should use number of child nodes
        xml = """
        <Parent size="-1">
            <Child/>
            <Child/>
            <Child/>
        </Parent>
        """
        elem = ET.fromstring(xml)
        arr = base.parse_serializable_array(elem, 'field', DummyInstance(), 
                                            DummyChild, 'Child')
        assert isinstance(arr, np.ndarray)
        assert arr.size == 3
        assert all(tag == 'Child' for tag in arr)

    def test_parse_serializable_array_list_hasattr_coefs(self):
        class DummyChild:
            def __init__(self, Coefs):
                self._coefs = Coefs
            # Simulate having a 'Coefs' property
            @property
            def Coefs(self):
                return self._coefs

        arrays = [[1, 2], [3, 4]]
        arr = base.parse_serializable_array(arrays, 'field', None, DummyChild, 
                                            'Child')
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.size, 2)
        self.assertTrue(all(isinstance(x, DummyChild) for x in arr))
        self.assertEqual(arr[0].Coefs, [1, 2])
        self.assertEqual(arr[1].Coefs, [3, 4])

    def test_parse_serializable_array_list_not_arrayable_and_no_coefs(self):
        class DummyChild:
            pass  # No from_array, not subclass of Arrayable, no Coefs

        arrays = [[1, 2], [3, 4]]
        # Should raise ValueError because DummyChild is not Arrayable and has 
        # no 'Coefs'
        with self.assertRaisesRegex(ValueError, r'Attribute field of array ' + \
                                    'type functionality belonging to class ' + \
                                    'NoneType got a list containing elements ' + \
                                    'type <class \'list\'> and construction ' + \
                                    'failed.$'):
            base.parse_serializable_array(arrays, 'field', None, DummyChild, 
                                          'Child')

    def test_parse_serializable_array_invalid_type(self):
        class DummyChild:
            pass

        # value is not None, not DummyChild, not ndarray, not Element, not 
        # list/tuple
        value = 42.0  # float type
        with self.assertRaises(TypeError):
            base.parse_serializable_array(value, 'field', None, DummyChild, 
                                          'Child')

# ********************
# parse_serializable_list tests
# ********************

class ParseSerializableListDummySerializable:
    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None):
        return cls(node.tag)
    @classmethod
    def from_dict(cls, d):
        return cls(d['tag'])
    def __init__(self, tag):
        self.tag = tag

class ParseSerializableListDummyInstance:
    _xml_ns = None
    _xml_ns_key = None
    _child_xml_ns_key = {}

class TestParseSerializableList(unittest.TestCase):
    def test_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_serializable_list\(\) " + \
                                    "missing 4 required positional arguments: " + \
                                    "'value', 'name', 'instance', and " + \
                                    "'child_type'$"):
            base.parse_serializable_list()
            
    def test_value_param_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_serializable_list\(\) " + \
                                    "missing 3 required positional arguments: " + \
                                    "'name', 'instance', and 'child_type'$"):
            base.parse_serializable_list('foo')

    def test_value_name_params_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_serializable_list\(\) " + \
                                    "missing 2 required positional arguments: " + \
                                    "'instance' and 'child_type'$"):
            base.parse_serializable_list('foo', 'bar')

    def test_value_name_instance_params_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_serializable_list\(\) " + \
                                    "missing 1 required positional argument: " + \
                                    "'child_type'$"):
            base.parse_serializable_list('foo', 'bar', 
                                          ParseSerializableListDummyInstance())

    def test_none_returns_none(self):
        self.assertIsNone(base.parse_serializable_list(None, 'test', 
                                                       ParseSerializableListDummyInstance(), 
                                                       ParseSerializableListDummySerializable))

    def test_single_child_type(self):
        obj = ParseSerializableListDummySerializable('child')
        result = base.parse_serializable_list(obj, 'test', 
                                              ParseSerializableListDummyInstance(), 
                                              ParseSerializableListDummySerializable)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], obj)

    def test_xml_element(self):
        xml = "<child/>"
        elem = ET.fromstring(xml)
        result = base.parse_serializable_list(elem, 'test', 
                                              ParseSerializableListDummyInstance(), 
                                              ParseSerializableListDummySerializable)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ParseSerializableListDummySerializable)
        self.assertEqual(result[0].tag, "child")

    def test_list_of_child_type(self):
        objs = [ParseSerializableListDummySerializable('child'), 
                ParseSerializableListDummySerializable('child')]
        result = base.parse_serializable_list(objs, 'test', 
                                              ParseSerializableListDummyInstance(), 
                                              ParseSerializableListDummySerializable)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(
            isinstance(x, ParseSerializableListDummySerializable) for x in result)
            )

    def test_list_of_dict(self):
        dicts = [{'tag': 'child'}, {'tag': 'child2'}]
        result = base.parse_serializable_list(dicts, 'test', 
                                              ParseSerializableListDummyInstance(), 
                                              ParseSerializableListDummySerializable)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(
            isinstance(x, ParseSerializableListDummySerializable) 
            for x in result)
            )
        self.assertEqual(result[0].tag, 'child')
        self.assertEqual(result[1].tag, 'child2')

    def test_list_of_xml_elements(self):
        xml = "<root><child/><child/></root>"
        elem = ET.fromstring(xml)
        children = list(elem)
        result = base.parse_serializable_list(children, 'test', 
                                              ParseSerializableListDummyInstance(), 
                                              ParseSerializableListDummySerializable)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(
            isinstance(x, ParseSerializableListDummySerializable) 
            for x in result)
            )

    def test_list_of_incompatible_type(self):
        arrs = [1, 2]
        with self.assertRaisesRegex(TypeError, r"Field test of list type " + \
                                    "functionality belonging to class " + \
                                    "ParseSerializableListDummyInstance got a " + \
                                    "list containing first element of " + \
                                    "incompatible type <class 'int'>.$"):
            base.parse_serializable_list(arrs, 'test', 
                                         ParseSerializableListDummyInstance(), 
                                         ParseSerializableListDummySerializable)

    def test_empty_list(self):
        result = base.parse_serializable_list([], 'test', 
                                              ParseSerializableListDummyInstance(), 
                                              ParseSerializableListDummySerializable)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_parse_serializable_list_no_child_xml_ns_key(self):
        class DummyChild:
            @classmethod
            def from_node(cls, node, xml_ns, ns_key=None):
                # For test, return tuple of tag, ns_key, and xml_ns
                return (node.tag, ns_key, xml_ns)

        class DummyInstance:
            _xml_ns = {'default': 'urn:default'}
            _xml_ns_key = 'default'
            # _child_xml_ns_key is NOT set

        xml = """
        <Parent>
            <Child/>
            <Child/>
        </Parent>
        """
        elem = ET.fromstring(xml)
        children = list(elem)
        result = base.parse_serializable_list(children, 'field', 
                                              DummyInstance(), DummyChild)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], 'default')
        self.assertEqual(result[0][2], {'default': 'urn:default'})

    def test_parse_serializable_list_invalid_type(self):
        class DummyChild:
            pass

        class DummyInstance:
            _xml_ns = {'default': 'urn:default'}
            _xml_ns_key = 'default'

        # value is not None, not DummyChild, not ElementTree.Element, not list, 
        # not child_type
        value = np.array([42.0])  # numpy array with float type
        with self.assertRaisesRegex(TypeError, r'Field field of class ' + 
                                    'DummyInstance got incompatible type ' + 
                                    '<class \'numpy.ndarray\'>.$'):
            base.parse_serializable_list(value, 'field', DummyInstance(), 
                                         DummyChild)

# ********************
# parse_parameters_collection tests
# ********************

class ParseParametersCollectionDummyInstance:
    pass

class TestParseParametersCollection(unittest.TestCase):
    def test_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_parameters_collection\(\) " + \
                                    "missing 3 required positional arguments: " + \
                                    "'value', 'name', and 'instance'$"):
            base.parse_parameters_collection()
            
    def test_value_param_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_parameters_collection\(\) " + \
                                    "missing 2 required positional arguments: " + \
                                    "'name' and 'instance'$"):
            base.parse_parameters_collection('foo')

    def test_value_name_params_only_fail(self):
         with self.assertRaisesRegex(TypeError, r"parse_parameters_collection\(\) " + \
                                    "missing 1 required positional argument: " + \
                                    "'instance'$"):
            base.parse_parameters_collection('foo', 'bar')

    def test_none_returns_none(self):
        self.assertIsNone(base.parse_parameters_collection(None, 'params', 
                                                           ParseParametersCollectionDummyInstance()))

    def test_dict_returns_dict(self):
        d = {'a': '1', 'b': '2'}
        result = base.parse_parameters_collection(d, 'params', 
                                                  ParseParametersCollectionDummyInstance())
        self.assertIsInstance(result, dict)
        self.assertEqual(result, d)

    def test_empty_list_returns_empty_ordereddict(self):
        result = base.parse_parameters_collection([], 'params', 
                                                  ParseParametersCollectionDummyInstance())
        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(len(result), 0)

    def test_list_of_xml_elements(self):
        xml = """
        <root>
            <Parameter name="alpha">A</Parameter>
            <Parameter name="beta">B</Parameter>
        </root>
        """
        elem = ET.fromstring(xml)
        params = list(elem)
        result = base.parse_parameters_collection(params, 'params', 
                                                  ParseParametersCollectionDummyInstance())
        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(result['alpha'], 'A')
        self.assertEqual(result['beta'], 'B')

    def test_list_of_xml_elements_empty_text(self):
        xml = """
        <root>
            <Parameter name="alpha"></Parameter>
            <Parameter name="beta"> </Parameter>
        </root>
        """
        elem = ET.fromstring(xml)
        params = list(elem)
        result = base.parse_parameters_collection(params, 'params', 
                                                  ParseParametersCollectionDummyInstance())
        self.assertIsInstance(result, OrderedDict)
        self.assertIsNone(result['alpha'])
        self.assertIsNone(result['beta'])

    def test_list_of_incompatible_type_raises(self):
        with self.assertRaisesRegex(TypeError, r"Field params of list type " + \
                                    "functionality belonging to class " + \
                                    "ParseParametersCollectionDummyInstance " + \
                                    "got a list containing first element of " + \
                                    "incompatible type <class 'int'>.$"):
            base.parse_parameters_collection([1, 2], 'params', 
                                             ParseParametersCollectionDummyInstance())

    def test_incompatible_type_raises(self):
        with self.assertRaisesRegex(TypeError, r"Field params of class " + \
                                    "ParseParametersCollectionDummyInstance " + \
                                    "got incompatible type <class 'str'>.$"):
            base.parse_parameters_collection("not_a_list_or_dict", 'params', 
                                             ParseParametersCollectionDummyInstance())

