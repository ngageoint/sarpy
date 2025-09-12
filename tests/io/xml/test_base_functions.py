__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"

import xml.etree.ElementTree as ET
import unittest

import sarpy.io.xml.base as base

class Test_base_functions(unittest.TestCase):
    def setUp(self):
        self.tree = ET.parse('tests/io/xml/country_data.xml')
        self.actor_tree = ET.parse('tests/io/xml/actor_test_data.xml')
        # self.root, self.country_ns_dict = base.parse_xml_from_file('tests/io/xml/country_data.xml') #self.tree.getroot()
        self.root = self.tree.getroot()
        self.actor_root, self.actor_ns_dict = base.parse_xml_from_file('tests/io/xml/actor_test_data.xml') # self.actor_tree.getroot()

    def test_get_node_value_success_with_text(self):
        branch = base.get_node_value(self.root[0][1])
        assert(branch == '2008')

    def test_get_node_value_success_none(self):
        branch = base.get_node_value(self.root[0])
        assert(branch == None)

    def test_get_node_value_success_empty(self):
        branch = base.get_node_value(self.root[0][3])
        assert(branch == None)

    def test_create_new_node_no_parent_success(self):
        new_node_tag = "country"
        if(len(self.root) == 3):
            new_node = base.create_new_node(self.tree, new_node_tag)
        else:
            assert(False)
        assert(len(self.root) == 4)

    def test_create_new_node_with_parent_success(self):
        new_node_tag = "ocean"
        if(len(self.root[1]) == 4):
            new_node = base.create_new_node(self.tree, new_node_tag, self.root[1])
        else:
            assert(False)
        for child in self.root[1]:
            print(child.tag, child.attrib)
        assert(self.root[1][4].tag == "ocean")

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
        if(len(self.root[2]) == 5):
            new_node = base.create_text_node(self.tree, new_node_tag, new_node_value, self.root[2])
        else:
            assert(False)
        for child in self.root[2]:
            print(child.tag, child.attrib)
        assert(self.root[2][5].text == "Pacific")

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

    def test_parse_xml_from_string_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        root_node, ns_dict = base.parse_xml_from_string(xml_string)
        assert(root_node.attrib == self.root.attrib)
    
    def test_parse_xml_from_file_success(self):
        test_root, test_ns_dict = base.parse_xml_from_file('tests/io/xml/country_data.xml')
        assert(test_root.attrib == self.root.attrib)
        
    def test_validate_xml_from_string_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        xsd_path = 'tests/io/xml/country.xsd'
        assert(base.validate_xml_from_string(xml_string, xsd_path))
        
    def test_validate_xml_from_string_with_logger_success(self):
        xml_string = ET.tostring(self.root, encoding='unicode', method='xml')
        xsd_path = 'tests/io/xml/country.xsd'
        assert(base.validate_xml_from_string(xml_string, xsd_path, base.logger))
        
    def test_validate_xml_from_file_success(self):
        xml_path = 'tests/io/xml/country_data.xml'
        xsd_path = 'tests/io/xml/country.xsd'
        assert(base.validate_xml_from_file(xml_path, xsd_path))
        
    def test_validate_xml_from_file_with_logger_success(self):
        xml_path = 'tests/io/xml/country_data.xml'
        xsd_path = 'tests/io/xml/country.xsd'
        assert(base.validate_xml_from_file(xml_path, xsd_path, base.logger))
        
    def test_parse_str_no_params_fail(self):
        base.parse_str()
        