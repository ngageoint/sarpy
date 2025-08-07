__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"

import xml.etree.ElementTree as ET
import unittest

import sarpy.io.xml.base as base

class Test_base_functions(unittest.TestCase):
    def setUp(self):
        self.tree = ET.parse('tests/io/xml/country_data.xml')
        self.root = self.tree.getroot()

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