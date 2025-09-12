__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"

import numpy as np
import unittest
import xml.etree.ElementTree as ET

import sarpy.io.xml.base as base

class Test_base_functions(unittest.TestCase):
    def setUp(self):
        self.tree = ET.parse('tests/io/xml/country_data.xml')
        self.actor_tree = ET.parse('tests/io/xml/actor_test_data.xml')
        # self.root, self.country_ns_dict = base.parse_xml_from_file('tests/io/xml/country_data.xml') #self.tree.getroot()
        self.root = self.tree.getroot()
        self.actor_root, self.actor_ns_dict = base.parse_xml_from_file('tests/io/xml/actor_test_data.xml') # self.actor_tree.getroot()

    # ********************
    # get_node_value tests
    # ********************
    def test_get_node_value_success_with_text(self):
        branch = base.get_node_value(self.root[0][1])
        assert(branch == '2008')

    def test_get_node_value_success_none(self):
        branch = base.get_node_value(self.root[0])
        assert(branch == None)

    def test_get_node_value_success_empty(self):
        branch = base.get_node_value(self.root[0][3])
        assert(branch == None)

    # ********************
    # create_new_node tests
    # ********************
    def test_create_new_node_no_parent_success(self):
        new_node_tag = "country"
        if(len(self.root) == 3):
            new_node = base.create_new_node(self.tree, new_node_tag)
        else:
            assert(False)
        assert(len(self.root) == 4)

    def test_create_new_node_with_parent_success(self):
        new_node_tag = "ocean"
        if(len(self.root[1]) == 5):
            new_node = base.create_new_node(self.tree, new_node_tag, self.root[1])
        else:
            assert(False)
        for child in self.root[1]:
            print(child.tag, child.attrib)
        assert(self.root[1][5].tag == "ocean")

    # ********************
    # create_text_node tests
    # ********************
    def test_create_text_node_no_parent_success(self):
        new_node_tag = "country"
        new_node_value = "Costa Rica"
        if(len(self.root) == 3):
            new_node = base.create_text_node(self.tree, new_node_tag, new_node_value)
        else:
            assert(False)
        assert(self.root[3].text == "Costa Rica")

    def test_create_text_node_with_parent_success(self):
        new_node_tag = "ocean"
        new_node_value = "Pacific"
        if(len(self.root[2]) == 6):
            new_node = base.create_text_node(self.tree, new_node_tag, new_node_value, self.root[2])
        else:
            assert(False)
        for child in self.root[2]:
            print(child.tag, child.attrib)
        assert(self.root[2][6].text == "Pacific")

    # ********************
    # find_first_child tests
    # ********************
    def test_find_first_child_no_optional_params_success(self):
        found_node = base.find_first_child(self.root, "country")
        assert(found_node.attrib == self.root[0].attrib)

    def test_find_first_child_namespace_params_success(self):
        found_node = base.find_first_child(self.actor_root, "actor", self.actor_ns_dict)
        assert(found_node.attrib == self.actor_root[0].attrib)

    def test_find_first_child_namespace_nskey_params_success(self):
        found_actor_node = base.find_first_child(self.actor_root, "actor", self.actor_ns_dict)
        found_node = base.find_first_child(found_actor_node, "character", self.actor_ns_dict, "fictional")
        assert(found_node.attrib == self.actor_root[0].attrib)

    # ********************
    # find_children tests
    # ********************
    def test_find_children_no_optional_params_success(self):
        found_nodes = base.find_children(self.root, "country")
        assert(found_nodes == self.root.findall("country"))

    def test_find_children_namespace_params_success(self):
        found_node = base.find_children(self.actor_root, "actor", self.actor_ns_dict)
        assert(found_node == self.actor_root.findall('actor', self.actor_ns_dict))

    def test_find_children_namespace_nskey_params_success(self):
        found_actor_node = base.find_first_child(self.actor_root, "actor", self.actor_ns_dict)
        found_node = base.find_children(found_actor_node, "character", self.actor_ns_dict, "fictional")
        assert(found_node == found_actor_node.findall('fictional:character', self.actor_ns_dict))

    # ********************
    # parse_xml_from_string tests
    # ********************
    def test_parse_xml_from_string_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        root_node, ns_dict = base.parse_xml_from_string(xml_string)
        assert(root_node.attrib == self.root.attrib)
    
    # ********************
    # parse_xml_from_file tests
    # ********************
    def test_parse_xml_from_file_success(self):
        test_root, test_ns_dict = base.parse_xml_from_file('tests/io/xml/country_data.xml')
        assert(test_root.attrib == self.root.attrib)
        
    # ********************
    # validate_xml_from_string tests
    # ********************
    def test_validate_xml_from_string_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        xsd_path = 'tests/io/xml/country.xsd'
        assert(base.validate_xml_from_string(xml_string, xsd_path))
        
    def test_validate_xml_from_string_with_logger_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        xsd_path = 'tests/io/xml/country.xsd'
        assert(base.validate_xml_from_string(xml_string, xsd_path, base.logger))
        
    # ********************
    # validate_xml_from_file tests
    # ********************
    def test_validate_xml_from_file_success(self):
        xml_path = 'tests/io/xml/country_data.xml'
        xsd_path = 'tests/io/xml/country.xsd'
        assert(base.validate_xml_from_file(xml_path, xsd_path))
        
    def test_validate_xml_from_file_with_logger_success(self):
        xml_path = 'tests/io/xml/country_data.xml'
        xsd_path = 'tests/io/xml/country.xsd'
        assert(base.validate_xml_from_file(xml_path, xsd_path, base.logger))
        
    # ********************
    # parse_str tests
    # ********************
    def test_parse_str_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_str\(\) missing 3 required positional arguments: 'value', 'name', and 'instance'$"):
            base.parse_str()
        
    def test_parse_str_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_str\(\) missing 2 required positional arguments: 'name' and 'instance'$"):
            base.parse_str("Test")

    def test_parse_str_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_str\(\) missing 1 required positional argument: 'instance'$"):
            base.parse_str("Test", "Bob")

    def test_parse_str_value_param_is_string_success(self):
        assert(base.parse_str("Test", "Bob", "base") == "Test")

    def test_parse_str_value_param_is_None_success(self):
        assert(base.parse_str(None, "Bob", "base") == None)

    def test_parse_str_value_param_is_xml_with_value_success(self):
        assert(base.parse_str(self.root[0][2], "text", "base") == 
               self.root[0][2].text)
        
    def test_parse_str_value_param_is_xml_empty_value_success(self):
        assert(base.parse_str(self.root[0], "text", "base") == 
               self.root[0].text.strip())
        
    def test_parse_str_bad_value_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"field Bob of class str requires a string value."):
            base.parse_str(1, "Bob", "base")

    # ********************
    # parse_bool tests
    # ********************
    def test_parse_bool_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_bool\(\) missing 3 required positional arguments: 'value', 'name', and 'instance'$"):
            base.parse_bool()
        
    def test_parse_bool_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_bool\(\) missing 2 required positional arguments: 'name' and 'instance'$"):
            base.parse_bool("Test")

    def test_parse_bool_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_bool\(\) missing 1 required positional argument: 'instance'$"):
            base.parse_bool("Test", "Bob")

    def test_parse_bool_value_param_is_None_success(self):
        assert(base.parse_bool(None, "Bob", "base") == None)

    def test_parse_bool_value_param_is_bool_success(self):
        assert(base.parse_bool(True, "Bob", "base") == True)

    def test_parse_bool_value_param_is_int_success(self):
        assert(base.parse_bool(1, "Bob", "base") == True)

    def test_parse_bool_value_param_is_np_bool_success(self):
        arr_bool = np.array([True, False, True, False], dtype=bool)
        assert(base.parse_bool(arr_bool[0], "Bob", "base") == True)
        
    def test_parse_bool_value_param_is_xml_success(self):
        assert(base.parse_bool(self.root[0][0], "Bob", "base") == True)

    def test_parse_bool_value_param_is_string_true_success(self):
        assert(base.parse_bool('trUe', "Bob", "base") == True)

    def test_parse_bool_value_param_is_string_1_success(self):
        assert(base.parse_bool('1', "Bob", "base") == True)

    def test_parse_bool_value_param_is_string_false_success(self):
        assert(base.parse_bool('FALSE', "Bob", "base") == False)

    def test_parse_bool_value_param_is_string_0_success(self):
        assert(base.parse_bool('0', "Bob", "base") == False)

    def test_parse_bool_value_param_is_float_fail(self):
        with self.assertRaisesRegex(ValueError, r"Boolean field Bob of class str cannot assign from type <class 'float'>."):
            assert(base.parse_bool(3.5, "Bob", "base") == False)

    # ********************
    # parse_int tests
    # ********************
    def test_parse_int_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_int\(\) missing 3 required positional arguments: 'value', 'name', and 'instance'$"):
            base.parse_int()
        
    def test_parse_int_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_int\(\) missing 2 required positional arguments: 'name' and 'instance'$"):
            base.parse_int("Test")

    def test_parse_int_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_int\(\) missing 1 required positional argument: 'instance'$"):
            base.parse_int("Test", "Bob")

    def test_parse_int_value_param_is_None_success(self):
        assert(base.parse_int(None, "Bob", "base") == None)

    def test_parse_int_value_param_is_int_success(self):
        assert(base.parse_int(1, "Bob", "base") == 1)

    def test_parse_int_value_param_is_xml_success(self):
        assert(base.parse_int(self.root[0][0], "Bob", "base") == 1)

    def test_parse_int_value_param_is_string_1_success(self):
        assert(base.parse_int('1', "Bob", "base") == 1)

    def test_parse_int_value_param_is_string_non_int_success(self):
        with self.assertRaisesRegex(ValueError, r"invalid literal for int\(\) with base 10: 'Bob'"):
            assert(base.parse_int('Bob', "Bob", "base") == 1)

    def test_parse_int_value_param_is_list_non_int_success(self):
        with self.assertRaisesRegex(TypeError, r"int\(\) argument must be a string, a bytes-like object or a real number, not 'list'"):
            assert(base.parse_int([3.5], "Bob", "base") == 1)

    # ********************
    # parse_float tests
    # ********************
    def test_parse_float_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_float\(\) missing 3 required positional arguments: 'value', 'name', and 'instance'$"):
            base.parse_float()
        
    def test_parse_float_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_float\(\) missing 2 required positional arguments: 'name' and 'instance'$"):
            base.parse_float("Test")

    def test_parse_float_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_float\(\) missing 1 required positional argument: 'instance'$"):
            base.parse_float("Test", "Bob")

    def test_parse_float_value_param_is_None_success(self):
        assert(base.parse_float(None, "Bob", "base") == None)

    def test_parse_float_value_param_is_float_success(self):
        assert(base.parse_float(1.5, "Bob", "base") == 1.5)

    def test_parse_float_value_param_is_xml_success(self):
        assert(base.parse_float(self.root[0][0], "Bob", "base") == 1.0)

    def test_parse_float_value_param_is_string_1dot5_success(self):
        assert(base.parse_float('1.5', "Bob", "base") == 1.5)

    def test_parse_float_value_param_is_string_non_int_success(self):
        with self.assertRaisesRegex(ValueError, r"could not convert string to float: 'Bob'"):
            assert(base.parse_float('Bob', "Bob", "base") == 1)

    def test_parse_float_value_param_is_list_non_int_success(self):
        with self.assertRaisesRegex(TypeError, r"float\(\) argument must be a string or a real number, not 'list'"):
            assert(base.parse_float([3.5], "Bob", "base") == 1)

    # ********************
    # parse_complex tests
    # ********************
    def test_parse_complex_no_params_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_complex\(\) missing 3 required positional arguments: 'value', 'name', and 'instance'$"):
            base.parse_complex()
        
    def test_parse_complex_value_param_only_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_complex\(\) missing 2 required positional arguments: 'name' and 'instance'$"):
            base.parse_complex("Test")

    def test_parse_complex_missing_instance_param_fail(self):
        with self.assertRaisesRegex(TypeError, r"parse_complex\(\) missing 1 required positional argument: 'instance'$"):
            base.parse_complex("Test", "Bob")

    def test_parse_complex_value_param_is_None_success(self):
        assert(base.parse_complex(None, "Bob", "base") == None)

    def test_parse_complex_value_param_is_complex_success(self):
        test_complex = 3 + 2j
        assert(base.parse_complex(test_complex, "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_xml_success(self):
        test_complex = 3 + 2j
        assert(base.parse_complex(self.root[0][4], "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_xml_2_real_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"There must be exactly one Real component of a complex type node defined for field Bob of class str."):
            assert(base.parse_complex(self.root[1][3], "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_xml_2_imag_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"There must be exactly one Imag component of a complex type node defined for field Bob of class str."):
            assert(base.parse_complex(self.root[2][3], "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_complex_dict_1_success(self):
        test_complex = 3 + 2j
        assert(base.parse_complex({"real":3, "imag":2}, "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_complex_dict_2_success(self):
        test_complex = 3 + 2j
        assert(base.parse_complex({"Real":3, "Imag":2}, "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_complex_dict_3_success(self):
        test_complex = 3 + 2j
        assert(base.parse_complex({"re":3, "im":2}, "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_complex_dict_4_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"Cannot convert dict {'not': 3, 'valid': 2} to a complex number for field Bob of class str."):
            assert(base.parse_complex({"not":3, "valid":2}, "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_complex_dict_5_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"Cannot convert dict {'real': None, 'imag': 2} to a complex number for field Bob of class str."):
            assert(base.parse_complex({"real":None, "imag":2}, "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_complex_dict_6_fail(self):
        test_complex = 3 + 2j
        with self.assertRaisesRegex(ValueError, r"Cannot convert dict {'real': 4, 'imag': None} to a complex number for field Bob of class str."):
            assert(base.parse_complex({"real":4, "imag":None}, "Bob", "base") == test_complex)

    def test_parse_complex_value_param_is_string_non_int_success(self):
        with self.assertRaisesRegex(ValueError, r"complex\(\) arg is a malformed string"):
            assert(base.parse_complex('Bob', "Bob", "base") == 1)

    def test_parse_complex_value_param_is_list_non_int_success(self):
        with self.assertRaisesRegex(TypeError, r"complex\(\) first argument must be a string or a number, not 'list'"):
            assert(base.parse_complex([3.5], "Bob", "base") == 1)