import unittest
from xml.etree import ElementTree
from collections import OrderedDict

from sarpy.io.xml.base import ParametersCollection

class TestParametersCollection(unittest.TestCase):
    def test_init_with_dict(self):
        params = {'alpha': 'A', 'beta': 'B'}
        pc = ParametersCollection(params, name='Test')
        self.assertEqual(pc['alpha'], 'A')
        self.assertEqual(pc['beta'], 'B')
        self.assertIsInstance(pc._dict, dict)

    def test_init_with_ordereddict(self):
        params = OrderedDict([('alpha', 'A'), ('beta', 'B')])
        pc = ParametersCollection(params, name='Test')
        self.assertEqual(pc['alpha'], 'A')
        self.assertEqual(pc['beta'], 'B')
        self.assertIsInstance(pc._dict, OrderedDict)

    def test_getitem_setitem(self):
        pc = ParametersCollection({'alpha': 'A'}, name='Test')
        self.assertEqual(pc['alpha'], 'A')
        pc['beta'] = 'B'
        self.assertEqual(pc['beta'], 'B')

    def test_delitem(self):
        pc = ParametersCollection({'alpha': 'A', 'beta': 'B'}, name='Test')
        del pc['alpha']
        self.assertIsNone(pc.get('alpha'))

    def test_to_dict(self):
        pc = ParametersCollection({'alpha': 'A', 'beta': 'B'}, name='Test')
        d = pc.to_dict()
        self.assertEqual(d, {'alpha': 'A', 'beta': 'B'})

    def test_get_collection(self):
        pc = ParametersCollection({'alpha': 'A', 'beta': 'B'}, name='Test')
        d = pc.get_collection()
        self.assertEqual(d, {'alpha': 'A', 'beta': 'B'})

    def test_to_node(self):
        pc = ParametersCollection({'alpha': 'A', 'beta': 'B'}, name='Test')
        doc = ElementTree.ElementTree()
        node = pc.to_node(doc)
        self.assertEqual(node.tag, 'Parameters')
        children = list(node)
        self.assertEqual(len(children), 2)
        self.assertEqual(children[0].tag, 'Parameters')
        self.assertEqual(children[0].attrib['name'], 'alpha')
        self.assertEqual(children[0].text, 'A')

    def test_repr(self):
        pc = ParametersCollection({'alpha': 'A'}, name='Test')
        self.assertIn('ParametersCollection', repr(pc))

    def test_parameterscollection_init_name_is_none(self):
        # Should raise ValueError because name is None
        with self.assertRaisesRegex(ValueError, 'The name parameter is required.'):
            ParametersCollection(collection={'a': 'A'}, name=None)
        
    def test_parameterscollection_init_name_not_str(self):
        # Should raise TypeError because name is not a string
        with self.assertRaisesRegex(TypeError, 'The name parameter is ' + \
                                    'required to be an instance of str'):
            ParametersCollection(collection={'a': 'A'}, name=123)
        
    def test_parameterscollection_init_child_tag_is_none(self):
        # Should raise ValueError because child_tag is None
        with self.assertRaisesRegex(ValueError, 'The child_tag parameter is ' + \
                                    'required.'):
            ParametersCollection(collection={'a': 'A'}, name='params', 
                                 child_tag=None)
        
    def test_parameterscollection_init_child_tag_not_str(self):
        # Should raise TypeError because child_tag is not a string
        with self.assertRaisesRegex(TypeError, 'The child_tag parameter is ' + \
                                    'required to be an instance of str'):
            ParametersCollection(collection={'a': 'A'}, name='params', child_tag=123)
        
    def test_parameterscollection_getitem_dict_is_none(self):
        pc = ParametersCollection(collection=None, name='params')
        # _dict is None, so __getitem__ should raise KeyError
        with self.assertRaisesRegex(KeyError, 'Dictionary does not contain key'):
            _ = pc['missing']
        
    def test_parameterscollection_setitem_name_not_str(self):
        pc = ParametersCollection(collection=None, name='params')
        # Should raise ValueError because name is not a string
        with self.assertRaisesRegex(ValueError, 'Parameter name must be of ' + \
                                    'type str'):
            pc[123] = "value"
        
    def test_parameterscollection_setitem_value_not_str(self):
        pc = ParametersCollection(collection=None, name='params')
        # Should raise ValueError because value is not a string
        with self.assertRaisesRegex(ValueError, 'Parameter name must be of ' + \
                                    'type str'):
            pc['param1'] = 123  # value is not a string
        
    def test_parameterscollection_setitem_dict_is_none(self):
        pc = ParametersCollection(collection=None, name='params')
        # _dict is None, so __setitem__ should initialize it as an OrderedDict
        pc['param1'] = 'value1'
        self.assertTrue(isinstance(pc._dict, OrderedDict))
        self.assertEqual(pc['param1'], 'value1')

    def test_parameterscollection_get_dict_is_none(self):
        pc = ParametersCollection(collection=None, name='params')
        # _dict is None, so get should return the default value
        self.assertEqual(pc.get('missing', default='default_value'),
                         'default_value')
        self.assertIsNone(pc.get('missing'))

    def test_parameterscollection_to_node_dict_is_none(self):
        pc = ParametersCollection(collection=None, name='params')
        doc = ElementTree.ElementTree()
        # Should return None since _dict is None
        result = pc.to_node(doc)
        self.assertIsNone(result)