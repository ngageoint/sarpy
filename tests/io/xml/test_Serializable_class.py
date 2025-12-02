import unittest
from collections import OrderedDict
from datetime import datetime, date
from xml.etree import ElementTree
import numpy as np

from sarpy.io.xml.base import ParametersCollection, Serializable, SerializableArray

class DummySerializable(Serializable):
    _fields = ('a', 'b', 'c')
    _required = ('a',)
    _tag_override = {'b': 'bee'}
    _set_as_attribute = ('c',)
    _choice = (
        {'required': True, 'collection': ('a', 'b')},
    )
    _child_xml_ns_key = {'b': 'ns1'}
    _collections_tags = {
        'arr': {'array': True, 'child_tag': 'item'},
        'lst': {'array': False, 'child_tag': 'item'}
    }
    def __init__(self, a=None, b=None, c=None, arr=None, lst=None, **kwargs):
        self.a = a
        self.b = b
        self.c = c

class DummySerializable3(Serializable):
    _fields   = ('a', 'b', 'c')
    _required = ()
    _choice   = [{'required': True, 'collection': ('a', 'b')}]

class DummySerializableTestList(Serializable):
    _fields           = ('lst',)
    _collections_tags = {'lst': {'array': False, 'child_tag': 'Item'}}
    _tag_override     = {}
    _set_as_attribute = ()
    _child_xml_ns_key = {}

    def __init__(self, lst=None):
        self.lst      = lst

class DummySerializableArrayTest(Serializable):
    _fields           = ('arr',)
    _collections_tags = {'arr': {'array': True, 'child_tag': 'Item'}}
    _tag_override     = {}
    _set_as_attribute = ()
    _child_xml_ns_key = {}

    def __init__(self, arr=None):
        self.arr = arr

class DummySerializableDictTest(Serializable):
    _fields = ('lst',)
    _collections_tags = {'lst': {'array': False, 'child_tag': 'Item'}}
    _tag_override = {}
    _set_as_attribute = ()
    _child_xml_ns_key = {}

    def __init__(self, lst=None):
        self.lst = lst

class DummySerializableTestToXMLBytes(Serializable):
            _fields           = ('a',)
            _collections_tags = {}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, a=None):
                self.a = a

class TestSerializable(unittest.TestCase):
    def test_init_and_fields(self):
        obj = DummySerializable(a=1, b=2, c=3)
        self.assertEqual((obj.a, obj.b, obj.c), (1, 2, 3))

    def test_str_and_repr(self):
        obj = DummySerializable(a=1, b=2, c=3)
        self.assertIn('DummySerializable', str(obj))
        self.assertIn('DummySerializable', repr(obj))

    def test_setattr_warning(self):
        obj = DummySerializable(a=1)
        with self.assertLogs('sarpy.io.xml.base', level='WARNING'):
            obj.unexpected = 5

    def test_getstate_setstate(self):
        obj     = DummySerializable(a=1, b=2, c=3)
        state   = obj.__getstate__()
        new_obj = DummySerializable.__new__(DummySerializable)
        new_obj.__setstate__(state)
        self.assertEqual((new_obj.a, new_obj.b, new_obj.c), (1, 2, 3))

    def test_set_numeric_format_and_get_formatter(self):
        obj = DummySerializable(a=1)
        obj.set_numeric_format('a', '.2f')
        fmt = obj._get_formatter('a')
        self.assertEqual(fmt(3.14159), '3.14')

    def test_set_numeric_format_fail(self):
        obj = DummySerializable(a=1)
        with self.assertRaisesRegex(
            ValueError, 
            r"attribute z is not permitted for class DummySerializable$"
        ):
            obj.set_numeric_format('z', '.2f')
        
    def test_log_validity_methods(self):
        obj = DummySerializable(a=1)
        with self.assertLogs('validation', level='ERROR'):
            obj.log_validity_error("error")
        with self.assertLogs('validation', level='WARNING'):
            obj.log_validity_warning("warning")
        with self.assertLogs('validation', level='INFO'):
            obj.log_validity_info("info")

    def test_basic_validity_check(self):
        self.assertTrue(DummySerializable(a=1)._basic_validity_check())
        self.assertFalse(DummySerializable()._basic_validity_check())

    def test_recursive_validity_check(self):
        self.assertTrue(DummySerializable(a=1)._recursive_validity_check())
        self.assertTrue(DummySerializable()._recursive_validity_check())

    def test_is_valid(self):
        self.assertTrue(DummySerializable(a=1).is_valid())
        self.assertFalse(DummySerializable().is_valid())

    def test_from_dict(self):
        d   = {'a': 1, 'b': 2, 'c': 3}
        obj = DummySerializable.from_dict(d)
        self.assertEqual((obj.a, obj.b, obj.c), (1, 2, 3))

    def test_to_dict(self):
        obj = DummySerializable(a=1, b=2, c=3)
        d   = obj.to_dict()
        self.assertEqual((d['a'], d['b'], d['c']), (1, 2, 3))

    def test_copy(self):
        obj  = DummySerializable(a=1, b=2, c=3)
        obj2 = obj.copy()
        self.assertEqual((obj2.a, obj2.b, obj2.c), (1, 2, 3))
        self.assertIsInstance(obj2, DummySerializable)

    def test_to_xml_bytes_and_string(self):
        obj       = DummySerializable(a=1, b=2, c=3)
        xml_bytes = obj.to_xml_bytes(tag='DummySerializable')
        xml_str   = obj.to_xml_string(tag='DummySerializable')
        self.assertIsInstance(xml_bytes, bytes)
        self.assertIsInstance(xml_str, str)
        self.assertIn('DummySerializable', xml_str)

    def test_from_node_with_tag_override_and_namespace(self):
        xml = """
            <root c='3' xmlns='urn:ns1'>
                <a>1</a>
                <bee>2</bee>
            </root>
            """
        xml_ns = {'ns1': 'urn:ns1', 'default': 'urn:ns1'}
        node   = ElementTree.fromstring(xml)
        obj    = DummySerializable.from_node(node, xml_ns)
        self.assertEqual(obj.a.text, "1")
        self.assertEqual(obj.b.text, "2")
        self.assertEqual(obj.c,      "3")

    def test_from_node_missing_required_field(self):
        xml  = '<root b="2" c="3"/>'
        node = ElementTree.fromstring(xml)
        with self.assertRaisesRegex(
            ValueError, 
            r"Attribute b in class <class " + \
            "'test_Serializable_class.DummySerializable'> expects a xml " + \
            "namespace entry of ns1, but xml_ns is None.$"
        ):
            DummySerializable.from_node(node, xml_ns=None)
    
    def test_serializable_init_with_xml_ns(self):
        class DummySerializable2(Serializable):
            _fields = ('a', 'b')
        xml_ns = {'default': 'urn:default'}
        obj    = DummySerializable2(a=1, b=2, _xml_ns=xml_ns)
        self.assertTrue(hasattr(obj, '_xml_ns'))
        self.assertEqual(obj._xml_ns, xml_ns)
        self.assertEqual((obj.a, obj.b), (1, 2))
        
    def test_repr_contains_class_name(self):
        self.assertIn('DummySerializable', repr(DummySerializable(a=1)))

    def test_to_node(self):
        obj  = DummySerializable(a=1, b=2, c=3)
        doc  = ElementTree.ElementTree()
        node = obj.to_node(doc, tag='DummySerializable')
        self.assertEqual(node.tag,             'DummySerializable')
        self.assertEqual(node.attrib.get('c'), '3')
        tags = [child.tag for child in node]
        self.assertIn('a',       tags)
        self.assertIn('ns1:bee', tags)
        a_node = next(child for child in node if child.tag == 'a')
        b_node = next(child for child in node if child.tag == 'ns1:bee')
        self.assertEqual(a_node.text, '1.00')
        self.assertEqual(b_node.text, '2')

    def test_serializable_init_sets_fields(self):
        class DummySerializable2(Serializable):
            _fields = ('a', 'b', 'c')
        obj = DummySerializable2(a=1, b=2, c=3)
        self.assertEqual((obj.a, obj.b, obj.c), (1, 2, 3))

    def test_serializable_init_missing_fields_are_none(self):
        class DummySerializable2(Serializable):
            _fields = ('a', 'b', 'c')
        obj = DummySerializable2(a=1)
        self.assertEqual(obj.a, 1)
        self.assertIsNone(obj.b)
        self.assertIsNone(obj.c)

    def test_serializable_init_unexpected_args_raises(self):
        class DummySerializable(Serializable):
            _fields = ('a', 'b')
        # Unexpected argument should raise ValueError
        with self.assertRaisesRegex(ValueError, r'Received unexpected ' + \
                                    'construction argument \[\'x\'\] for ' + \
                                    'attribute collection \(\'a\', \'b\'\)$'):
            DummySerializable(a=1, b=2, x=5)

    def test_serializable_get_formatter_callable(self):
        class DummySerializable2(Serializable):
            _fields = ('a',)
        obj = DummySerializable2(a=1)
        obj._numeric_format['a'] = lambda x: f"Value:{x}"
        fmt = obj._get_formatter('a')
        self.assertTrue(callable(fmt))
        self.assertEqual(fmt(5), "Value:5")

    def test_serializable_is_valid_not_recursive(self):
        class DummySerializable2(Serializable):
            _fields   = ('a', 'b')
            _required = ('a',)
        obj  = DummySerializable2(a=None, b=2)
        obj2 = DummySerializable2(a=1,    b=2)
        self.assertFalse(obj.is_valid(recursive=True))
        self.assertTrue(obj2.is_valid(recursive=True))

    def test_basic_validity_check_choice_multiple_present(self):
        obj = DummySerializable3(a=1, b=2, c=3)
        with self.assertLogs('validation', level='ERROR') as cm:
            valid = obj._basic_validity_check()
        self.assertFalse(valid)
        self.assertRegex(cm.output[0], r'Exactly one of the attributes ' + \
                         '\(\'a\', \'b\'\) should be set, but multiple ' + \
                         '\(\[\'a\', \'b\'\]\) are set$')

    def test_basic_validity_check_choice_required_none_present(self):
        obj = DummySerializable3(a=None, b=None, c=3)
        with self.assertLogs('validation', level='ERROR') as cm:
            valid = obj._basic_validity_check()
        self.assertFalse(valid)
        self.assertRegex(cm.output[0], r'Exactly one of the attributes ' + \
                         '\(\'a\', \'b\'\) should be set, but none are set$')
        
    def test_recursive_validity_check_check_item(self):
        class DummyChild(Serializable):
            _fields = ('x',)
            def __init__(self, x=None):
                self.x = x
            def is_valid(self, recursive=True, stack=False):
                return True
        class DummyParent(Serializable):
            _fields = ('child',)
            def __init__(self, child=None):
                self.child = child
        child  = DummyChild(x=5)
        parent = DummyParent(child=child)
        self.assertTrue(parent._recursive_validity_check(stack=False))
        class InvalidChild(Serializable):
            _fields = ('x',)
            def __init__(self, x=None):
                self.x = x
            def is_valid(self, recursive=True, stack=False):
                return False
        invalid_child  = InvalidChild(x=5)
        parent_invalid = DummyParent(child=invalid_child)
        self.assertFalse(parent_invalid._recursive_validity_check(stack=False))

    def test_recursive_validity_check_check_item_not_serializable(self):
        class DummySerializable2(Serializable):
            _fields = ('a', 'b')
            def __init__(self, a=None, b=None):
                self.a = a
                self.b = b
        obj = DummySerializable2(a=1, b=2)
        self.assertTrue(obj._recursive_validity_check(stack=False))
        class DummyWithList(Serializable):
            _fields = ('lst',)
            def __init__(self, lst=None):
                self.lst = lst
        obj2 = DummyWithList(lst=[1, 2, 3])
        self.assertTrue(obj2._recursive_validity_check(stack=False))

    def test_serializable_from_node_handle_attribute_the_xml_ns_key_none(self):
        class DummySerializable2(Serializable):
            _fields           = ('a',)
            _set_as_attribute = ('a',)
            _tag_override     = {}
            _child_xml_ns_key = {}
        xml    = '<root a="value"/>'
        node   = ElementTree.fromstring(xml)
        xml_ns = None
        obj    = DummySerializable2.from_node(node, xml_ns)
        self.assertEqual(obj.a, "value")

    def test_serializable_from_node_handle_attribute_the_xml_ns_key_not_none(self):
        class DummySerializable2(Serializable):
            _fields           = ('a',)
            _set_as_attribute = ('a',)
            _tag_override     = {}
            _child_xml_ns_key = {'a': 'ns1'}
        xml    = '<root ns1:a="value" xmlns:ns1="urn:ns1"/>'
        node   = ElementTree.fromstring(xml)
        xml_ns = {'ns1': 'urn:ns1'}
        obj    = DummySerializable2.from_node(node, xml_ns)
        self.assertEqual(obj.a, "value")

    def test_serializable_from_node_handle_list_cnodes_gt_zero(self):
        class DummySerializable2(Serializable):
            _fields           = ('a',)
            _collections_tags = {'a': {'array': False, 'child_tag': 'Child'}}
            _tag_override     = {}
            _child_xml_ns_key = {}
        xml = """
        <root>
            <Child>one</Child>
            <Child>two</Child>
        </root>
        """
        node   = ElementTree.fromstring(xml)
        xml_ns = None
        obj    = DummySerializable2.from_node(node, xml_ns)
        self.assertIsInstance(obj.a, list)
        self.assertEqual(len(obj.a),    2)
        self.assertEqual(obj.a[0].tag,  'Child')
        self.assertEqual(obj.a[0].text, 'one')
        self.assertEqual(obj.a[1].tag,  'Child')
        self.assertEqual(obj.a[1].text, 'two')

    def test_serializable_from_node_kwargs_not_dict(self):
        class DummySerializable2(Serializable):
            _fields           = ('a',)
            _tag_override     = {}
            _child_xml_ns_key = {}
        xml    = '<root a="value"/>'
        node   = ElementTree.fromstring(xml)
        xml_ns = None
        with self.assertRaisesRegex(ValueError, r'Named input argument ' + \
                                    'kwargs .* must be dictionary instance$'):
            DummySerializable2.from_node(node, xml_ns, kwargs=[])

    def test_serializable_from_node_attribute_in_kwargs(self):
        class DummySerializable2(Serializable):
            _fields           = ('a', 'b')
            _tag_override     = {}
            _child_xml_ns_key = {}
        xml    = '<root a="value"/>'
        node   = ElementTree.fromstring(xml)
        xml_ns = None
        obj    = DummySerializable2.from_node(node, xml_ns, kwargs={'a': 'preset'})
        self.assertEqual(obj.a, 'preset')
        self.assertIsNone(obj.b)

    def test_serializable_from_node_xml_ns_key_not_in_xml_ns(self):
        class DummySerializable2(Serializable):
            _fields           = ('a',)
            _tag_override     = {}
            _child_xml_ns_key = {'a': 'ns1'}
        xml    = '<root ns1="value" xmlns:ns2="urn:ns2"/>'
        node   = ElementTree.fromstring(xml)
        xml_ns = {'ns2': 'urn:ns2'}
        with self.assertRaisesRegex(ValueError, r'Attribute a in class ' + \
                                    '.* expects a xml namespace entry of ' + \
                                    'ns1, but xml_ns does not contain this ' + \
                                    'key.$'):
            DummySerializable2.from_node(node, xml_ns)

    def test_serializable_from_node_attribute_not_in_kwargs_and_not_in_child_xml_ns_key(self):
        class DummySerializable2(Serializable):
            _fields           = ('a',)
            _tag_override     = {}
            _child_xml_ns_key = {}
        xml = """
        <root>
            <a>value</a>
        </root>
        """
        node   = ElementTree.fromstring(xml)
        xml_ns = None
        obj    = DummySerializable2.from_node(node, xml_ns)
        self.assertIsInstance(obj.a, ElementTree.Element)
        self.assertEqual(obj.a.tag,  'a')
        self.assertEqual(obj.a.text, 'value')

    def test_serializable_from_node_collections_tags_array_true(self):
        class DummySerializable2(Serializable):
            _fields           = ('a',)
            _collections_tags = {'a': {'array': True, 'child_tag': 'Child'}}
            _tag_override     = {}
            _child_xml_ns_key = {}
        xml = """
        <root>
            <a>
                <Child>one</Child>
                <Child>two</Child>
            </a>
        </root>
        """
        node     = ElementTree.fromstring(xml)
        xml_ns   = None
        obj      = DummySerializable2.from_node(node, xml_ns)
        children = list(obj.a)
        self.assertIsInstance(obj.a,       ElementTree.Element)
        self.assertEqual(len(children),    2)
        self.assertEqual(children[0].tag,  'Child')
        self.assertEqual(children[0].text, 'one')
        self.assertEqual(children[1].tag,  'Child')
        self.assertEqual(children[1].text, 'two')

    def test_serializable_to_node_serialize_attribute_with_namespace(self):
        class DummySerializable(Serializable):
            _fields           = ('a',)
            _set_as_attribute = ('a',)
            _tag_override     = {}
            _child_xml_ns_key = {'a': 'ns1'}

        obj  = DummySerializable(a=123)
        doc  = ElementTree.ElementTree()
        node = obj.to_node(doc, tag='TestTag', ns_key='ns1')
        # Should serialize attribute with namespace prefix
        self.assertEqual(node.attrib.get('ns1:a'), 'Value:123')
        self.assertEqual(node.tag, 'ns1:TestTag')

    def test_serializable_to_node_serialize_array(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {'arr': {'array': True, 'child_tag': 'Item'}}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr      = arr

        obj  = DummySerializable(arr=np.array([1.1, 2.2, 3.3], dtype=np.float64))
        doc  = ElementTree.ElementTree()
        node = obj.to_node(doc, tag='TestTag')
        # Should create a node with child 'arr' containing three 'Item' children
        arr_node = node.find('arr')
        self.assertIsNotNone(arr_node)
        self.assertEqual(arr_node.attrib['size'], '3')
        items = list(arr_node)
        self.assertEqual([item.text for item in items], ['1.1', '2.2', '3.3'])

    def test_serializable_to_node_serialize_array_shape_not_1(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {'arr': {'array': True, 'child_tag': 'Item'}}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr      = arr

        # arr is a 2D numpy array, which should trigger ValueError in serialize_array
        obj = DummySerializable(arr=np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64))
        doc = ElementTree.ElementTree()
        with self.assertRaisesRegex(ValueError, r'The value associated with ' + \
                                    'attribute arr is an instance of class ' + \
                                    'DummySerializable, if None, is required ' + \
                                    'to bea one-dimensional numpy.ndarray, ' + \
                                    'but it has shape \(2, 2\)$'):
            obj.to_node(doc, tag='TestTag')

    def test_serializable_to_node_serialize_array_val_size_zero(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {'arr': {'array': True, 'child_tag': 'Item'}}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr      = arr

        # arr is an empty numpy array
        obj  = DummySerializable(arr=np.array([], dtype=np.float64))
        doc  = ElementTree.ElementTree()
        node = obj.to_node(doc, tag='TestTag')
        arr_node = node.find('arr')
        items = [child for child in node if child.tag == 'Item']
        # Should not create any 'arr' or 'Item' children
        self.assertIsNotNone(node)
        self.assertIsNone(arr_node)
        self.assertEqual(len(items), 0)

    def test_serializable_to_node_serialize_array_float64_with_namespace(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {'arr': {'array': True, 'child_tag': 'Item'}}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr      = arr

        obj      = DummySerializable(arr=np.array([1.1, 2.2, 3.3], 
                                                  dtype=np.float64))
        doc      = ElementTree.ElementTree()
        # Use a namespace key
        node     = obj.to_node(doc, tag='TestTag', ns_key='ns1')
        arr_node = node.find('ns1:arr')
        items = list(arr_node)
        self.assertIsNotNone(arr_node)
        self.assertEqual(arr_node.attrib['size'],       '3')
        self.assertEqual(len(items),                    3)
        self.assertEqual([item.text for item in items], ['1.1', '2.2', '3.3'])
        self.assertTrue(all(item.tag == 'Item' for item in items))

    def test_serializable_to_node_serialize_array_dtype_not_float64(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {'arr': {'array': True, 'child_tag': 'Item'}}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr      = arr

        # arr is a numpy array of dtype 'int32', which should trigger ValueError in serialize_array
        obj = DummySerializable(arr=np.array([1, 2, 3], dtype=np.int32))
        doc = ElementTree.ElementTree()
        with self.assertRaisesRegex(ValueError, r'The value associated with ' + \
                                    'attribute arr is an instance of class ' + \
                                    'DummySerializable, if None, is required ' + \
                                    'to be a numpy.ndarray of dtype float64 or ' + \
                                    'object, but it has dtype int32$'):
            obj.to_node(doc, tag='TestTag')

    def test_serializable_to_node_serialize_list(self):
        obj   = DummySerializableTestList(lst=['foo', 'bar', 'baz'])
        doc   = ElementTree.ElementTree()
        node  = obj.to_node(doc, tag='TestTag')
        # Should create Item children directly under the node
        items = [child for child in node if child.tag == 'Item']
        self.assertEqual(len(items), 3)
        self.assertEqual([item.text for item in items], ['foo', 'bar', 'baz'])
        
    def test_serializable_to_node_serialize_list_len_zero(self):
        obj   = DummySerializableTestList(lst=[])
        doc   = ElementTree.ElementTree()
        node  = obj.to_node(doc, tag='TestTag')
        # Should not create any 'Item' children since the list is empty
        items = [child for child in node if child.tag == 'Item']
        self.assertEqual(len(items), 0)

    def test_serializable_to_node_serialize_plain(self):
        class DummyChild(Serializable):
            _fields = ('x',)
            def __init__(self, x=None):
                self.x = x

        class DummySerializable(Serializable):
            _fields = (
                'int_field', 'float_field', 'str_field', 'bool_field', 
                'datetime_field', 'date_field', 'complex_field', 
                'serializable_field', 'serializable_array_field',
                'parameters_collection_field', 'exclude_field'
            )
            _collections_tags = {
                'serializable_array_field': {'array': True, 'child_tag': 'Item'}
            }
            def __init__(self, **kwargs):
                for k in self._fields:
                    setattr(self, k, kwargs.get(k, None))

        # Prepare values for each type handled by serialize_plain
        int_val                   = 42
        float_val                 = 3.14
        str_val                   = "hello"
        bool_val                  = True
        dt_val                    = np.datetime64('2023-01-01T12:00')
        date_val                  = date(2023, 1, 1)
        complex_val               = complex(1, 2)
        serializable_val          = DummyChild(x=99)
        serializable_array_val    = np.array([3.14, 42.314])
        parameters_collection_val = ParametersCollection({'a': 'A'}, name='params')

        obj = DummySerializable(
            int_field                   = int_val,
            float_field                 = float_val,
            str_field                   = str_val,
            bool_field                  = bool_val,
            datetime_field              = dt_val,
            date_field                  = date_val,
            complex_field               = complex_val,
            serializable_field          = serializable_val,
            serializable_array_field    = serializable_array_val,
            parameters_collection_field = parameters_collection_val
        )

        doc  = ElementTree.ElementTree()
        node = obj.to_node(doc, tag='TestTag', exclude={'exclude_field'})
        cnode = node.find('complex_field')
        items = node.findall('serializable_array_field')
        params = node.find('Parameters')
        
        # Check that each field is serialized as expected
        self.assertEqual(node.find('int_field').text,   str(int_val))
        self.assertEqual(node.find('float_field').text, str(float_val))
        self.assertEqual(node.find('str_field').text,   str_val)
        self.assertEqual(node.find('bool_field').text,  'true')
        self.assertEqual(node.find('date_field').text,  date_val.isoformat())
        self.assertTrue(node.find('datetime_field').text.endswith('Z'))
        # Complex field should have Real and Imag children
        self.assertIsNotNone(cnode)
        self.assertEqual(cnode.find('Real').text, str(complex_val.real))
        self.assertEqual(cnode.find('Imag').text, str(complex_val.imag))
        # Serializable field
        self.assertIsNotNone(node.find('serializable_field'))
        # SerializableArray field
        self.assertEqual(len(items[0]), 2)
        # ParametersCollection field
        self.assertIsNotNone(params)
        self.assertEqual(params[0].attrib['name'], 'a')

    def test_serializable_to_node_serialize_plain_complex_with_namespace(self):
        class DummySerializable(Serializable):
            _fields           = ('cplx',)
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {'cplx': 'ns1'}

            def __init__(self, cplx=None):
                self.cplx = cplx

        obj  = DummySerializable(cplx=complex(1.5, -2.5))
        doc  = ElementTree.ElementTree()
        node = obj.to_node(doc, tag='TestTag', ns_key='ns1')
        # Should create a node with tag 'ns1:cplx' and children 'ns1:Real', 'ns1:Imag'
        cplx_node = node.find('ns1:cplx')
        self.assertIsNotNone(cplx_node)
        real_node = cplx_node.find('ns1:Real')
        imag_node = cplx_node.find('ns1:Imag')
        self.assertIsNotNone(real_node)
        self.assertIsNotNone(imag_node)
        self.assertEqual(real_node.text, '1.5')
        self.assertEqual(imag_node.text, '-2.5')

    def test_serializable_to_node_check_validity_true(self):
        class DummySerializable(Serializable):
            _fields           = ('a',)
            _required         = ('a',)
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, a=None):
                self.a = a

        obj = DummySerializable(a=None)  # Missing required field, so not valid
        doc = ElementTree.ElementTree()
        # Should log a warning and still return a node (since strict=False by default)
        with self.assertLogs('validation', level='WARNING') as cm:
            node = obj.to_node(doc, tag='TestTag', check_validity=True)
        self.assertEqual(node.tag, 'TestTag')
        self.assertRegex(cm.output[0],'ERROR:validation:DummySerializable: '
                         'Missing required attribute a')

        # Now test with strict=True, should raise ValueError
        with self.assertRaisesRegex(ValueError, 'is not valid'):
            obj.to_node(doc, tag='TestTag', check_validity=True, strict=True)

    def test_serializable_to_node_xml_ns_key_default(self):
        class DummySerializable(Serializable):
            _fields           = ('a',)
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, a=None):
                self.a = a

        obj = DummySerializable(a="value")
        doc = ElementTree.ElementTree()
        # Pass ns_key='default', which should be treated as None
        node   = obj.to_node(doc, tag='TestTag', ns_key='default')
        a_node = node.find('a')
        self.assertIsNotNone(a_node)
        self.assertEqual(a_node.text, 'value')

    def test_serializable_to_node_array_tag_is_none(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {}  # No entry for 'arr'
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr = arr

        obj = DummySerializable(arr=np.array([1, 2, 3], dtype=np.float64))
        doc = ElementTree.ElementTree()
        # Should raise AttributeError because _collections_tags has no entry for 'arr'
        with self.assertRaisesRegex(AttributeError, r'The value associated ' + \
                                    'with attribute arr in an instance of ' + \
                                    'class DummySerializable is of type ' + \
                                    '\<class \'numpy.ndarray\'\>, but nothing ' + \
                                    'is populated in the _collection_tags ' + \
                                    'dictionary.$'):
            obj.to_node(doc, tag='TestTag')
        
    def test_serializable_to_node_child_tag_is_none(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {'arr': {'array': True}}  # child_tag is missing
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr = arr

        obj = DummySerializable(arr=np.array([1, 2, 3], dtype=np.float64))
        doc = ElementTree.ElementTree()
        # Should raise AttributeError because child_tag is None in _collections_tags
        with self.assertRaisesRegex(AttributeError, r'The value associated ' + \
                                    'with attribute arr in an instance of ' + \
                                    'class DummySerializable is of type ' + \
                                    '\<class \'numpy.ndarray\'\>, but ' + \
                                    '\`child_tag\` is not populated in ' + \
                                    'the _collection_tags dictionary.$'):
            obj.to_node(doc, tag='TestTag')

    def test_serializable_to_dict_serialize_array(self):
        obj = DummySerializableArrayTest(arr=np.array([1.1, 2.2, 3.3], dtype=np.float64))
        d   = obj.to_dict()
        self.assertTrue('arr' in d)
        self.assertTrue(isinstance(d['arr'], list))
        self.assertEqual(d['arr'], [1.1, 2.2, 3.3])

    def test_serializable_to_dict_serialize_array_shape_not_1(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {'arr': {'array': True, 'child_tag': 'Item'}}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr = arr

        # arr is a 2D numpy array, which should trigger ValueError in serialize_array
        obj = DummySerializable(arr=np.array([[1, 2], [3, 4]], dtype=np.float64))
        with self.assertRaisesRegex(ValueError, r'The value associated with ' + \
                                    'attribute arr is an instance of class ' + \
                                    'DummySerializable, if None, is required ' + \
                                    'to bea one-dimensional numpy.ndarray, ' + \
                                    'but it has shape \(2, 2\)$'):
            obj.to_dict()

    def test_serializable_to_dict_serialize_array_val_size_zero(self):
        # arr is an empty numpy array
        obj = DummySerializableArrayTest(arr=np.array([], dtype=np.float64))
        d   = obj.to_dict()
        self.assertTrue('arr' in d)
        self.assertTrue(isinstance(d['arr'], list))
        self.assertEqual(d['arr'], [])

    def test_serializable_to_dict_serialize_array_dtype_not_float64(self):
        # arr is a numpy array of dtype 'int32', which should trigger ValueError in serialize_array
        obj = DummySerializableArrayTest(arr=np.array([1, 2, 3], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, r'The value associated with ' + \
                                    'attribute arr is an instance of class ' + \
                                    'DummySerializableArrayTest. This is expected to ' + \
                                    'be a numpy.ndarray of dtype float64, but ' + \
                                    'it has dtype int32$'):
            obj.to_dict()

    def test_serializable_to_dict_serialize_list(self):
        obj = DummySerializableDictTest(lst=['foo', 'bar', 'baz'])
        d = obj.to_dict()
        self.assertTrue('lst' in d)
        self.assertTrue(isinstance(d['lst'], list))
        self.assertEqual(d['lst'], ['foo', 'bar', 'baz'])

    def test_serializable_to_dict_serialize_list_len_zero(self):
        obj = DummySerializableDictTest(lst=[])
        d = obj.to_dict()
        self.assertIn('lst', d)
        self.assertIsInstance(d['lst'], list)
        self.assertEqual(d['lst'], [])

    def test_serializable_to_dict_serialize_plain_various_types(self):
        class DummyChild(Serializable):
            _fields = ('x',)
            def __init__(self, x=None):
                self.x = x

        class DummyArray(SerializableArray):
            def __init__(self):
                arr = np.array([DummyChild(x=1), DummyChild(x=2)], dtype=object)
                super().__init__(coords=arr, name='arr', child_tag='x', child_type=DummyChild)

            def to_json_list(self, check_validity=False, strict=False):
                return [{'x': 1}, {'x': 2}]

        class DummySerializable(Serializable):
            _fields = (
                'serializable_field', 'serializable_array_field', 'parameters_collection_field',
                'datetime_field', 'complex_field', 'date_field', 'dt_field'
            )
            _collections_tags = {
                'serializable_array_field': {'array': True, 'child_tag': 'x'}
            }
            def __init__(self, **kwargs):
                for k in self._fields:
                    setattr(self, k, kwargs.get(k, None))

        serializable_val          = DummyChild(x=99)
        serializable_array_val    = DummyArray()
        parameters_collection_val = ParametersCollection({'a': 'A'}, name='params')
        datetime_val              = np.datetime64('2023-01-01T12:00')
        complex_val               = complex(1, 2)
        date_val                  = date(2023, 1, 1)
        dt_val                    = datetime(2023, 1, 1, 12, 34, 56)

        obj = DummySerializable(
            serializable_field          = serializable_val,
            serializable_array_field    = serializable_array_val,
            parameters_collection_field = parameters_collection_val,
            datetime_field              = datetime_val,
            complex_field               = complex_val,
            date_field                  = date_val,
            dt_field                    = dt_val
        )

        d = obj.to_dict()
        self.assertTrue(isinstance(d['serializable_field'], dict))
        self.assertEqual(d['serializable_field']['x'], 99)
        self.assertTrue(isinstance(d['serializable_array_field'], list))
        self.assertEqual(d['serializable_array_field'], [{'x': 1}, {'x': 2}])
        self.assertTrue(isinstance(d['parameters_collection_field'], dict))
        self.assertEqual(d['parameters_collection_field']['a'], 'A')
        self.assertTrue(isinstance(d['datetime_field'], str) and 
                        d['datetime_field'].endswith('Z'))
        self.assertTrue(isinstance(d['complex_field'], dict))
        self.assertEqual(d['complex_field']['Real'], 1)
        self.assertEqual(d['complex_field']['Imag'], 2)
        self.assertEqual(d['date_field'], date_val.isoformat())
        self.assertEqual(d['dt_field'], dt_val.isoformat(sep='T'))

    def test_serializable_to_dict_serialize_plain_fallback(self):
        class DummySerializable(Serializable):
            _fields           = ('custom',)
            _collections_tags = {}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, custom=None):
                self.custom   = custom

        class CustomType:
            pass

        obj = DummySerializable(custom=CustomType())
        with self.assertRaisesRegex(ValueError, r'An entry for class ' + \
                                    'DummySerializable using tag custom is ' + \
                                    'of type \<class \'test_Serializable_class.' + \
                                    'TestSerializable.' + \
                                    'test_serializable_to_dict_serialize_plain_fallback.' + \
                                    '\<locals\>.CustomType\'\>, and ' + \
                                    'serialization has not been implemented$'):
            obj.to_dict()

    def test_serializable_to_dict_check_validity(self):
        class DummySerializable(Serializable):
            _fields           = ('a',)
            _required         = ('a',)
            _collections_tags = {}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, a=None):
                self.a = a

        # Invalid object (missing required field)
        obj = DummySerializable(a=None)
        # Should log a warning and still return a dict (since strict=False by default)
        with self.assertLogs('validation', level='WARNING') as cm:
            d = obj.to_dict(check_validity=True)
        self.assertIn('ERROR:validation:DummySerializable: Missing required ' \
        'attribute a', cm.output)
        self.assertTrue(isinstance(d, dict))

        # Now test with strict=True, should raise ValueError
        with self.assertRaisesRegex(ValueError, r'DummySerializable is not ' + \
                                    'valid,\n\tand cannot be SAFELY ' + \
                                    'serialized to a dictionary valid in the ' + \
                                    'SICD standard.$'):
            obj.to_dict(check_validity=True, strict=True)

    def test_serializable_to_dict_attribute_in_exclude(self):
        class DummySerializable(Serializable):
            _fields           = ('a', 'b')
            _collections_tags = {}
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, a=None, b=None):
                self.a = a
                self.b = b

        obj = DummySerializable(a=1, b=2)
        # Exclude 'b' from serialization
        d = obj.to_dict(exclude=('b',))
        self.assertIn('a', d)
        self.assertNotIn('b', d)
        self.assertEqual(d['a'], 1)

    def test_serializable_to_dict_array_tag_is_none(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {}  # No entry for 'arr'
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr = arr

        obj = DummySerializable(arr=np.array([1, 2, 3], dtype=np.float64))
        with self.assertRaisesRegex(AttributeError, r'The value associated ' + \
                                    'with attribute arr in an instance of ' + \
                                    'class DummySerializable is of type ' + \
                                    '\<class \'numpy.ndarray\'\>, but ' + \
                                    'nothing is populated in ' + \
                                    'the _collection_tags dictionary.$'):
            obj.to_dict()
        
    def test_serializable_to_dict_child_tag_is_none(self):
        class DummySerializable(Serializable):
            _fields           = ('arr',)
            _collections_tags = {'arr': {'array': True}}  # child_tag is missing
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

            def __init__(self, arr=None):
                self.arr = arr

        obj = DummySerializable(arr=np.array([1, 2, 3], dtype=np.float64))
        with self.assertRaisesRegex(AttributeError, r'The value associated ' + \
                                    'with attribute arr in an instance of ' + \
                                    'class DummySerializable is of type ' + \
                                    '\<class \'numpy.ndarray\'\>, but ' + \
                                    '`child_tag` is not populated in ' + \
                                    'the _collection_tags dictionary.$'):
            obj.to_dict()

    def test_serializable_to_xml_bytes_tag_is_none(self):
        obj       = DummySerializableTestToXMLBytes(a="value")
        # tag is None, so should use class name as tag
        xml_bytes = obj.to_xml_bytes()
        xml_str   = xml_bytes.decode('utf-8')
        self.assertIn('<DummySerializable', xml_str)
        self.assertIn('<a>value</a>',       xml_str) 
        self.assertIn('a="value"',          xml_str)

    def test_serializable_to_xml_bytes_tag_is_none(self):
        obj       = DummySerializableTestToXMLBytes(a="value")
        # tag is None, so should use class name as tag
        xml_bytes = obj.to_xml_bytes()
        xml_str   = xml_bytes.decode('utf-8')
        self.assertIn('<DummySerializableTestToXMLBytes', xml_str)
        self.assertIn('<a>value</a>', xml_str)

    def test_serializable_to_xml_bytes_urn_is_string(self):
        obj       = DummySerializableTestToXMLBytes(a="value")
        urn       = "http://www.example.com/schema"
        xml_bytes = obj.to_xml_bytes(urn=urn, tag="TestTag")
        xml_str   = xml_bytes.decode('utf-8')
        self.assertIn('xmlns="http://www.example.com/schema"', xml_str)
        self.assertIn('<TestTag', xml_str)
        self.assertIn('<a>value</a>', xml_str) 
        
    def test_serializable_to_xml_bytes_urn_is_dict(self):
        obj       = DummySerializableTestToXMLBytes(a="value")
        urn       = {"xmlns:ns1": "http://www.example.com/ns1", "xmlns:ns2": "http://www.example.com/ns2"}
        xml_bytes = obj.to_xml_bytes(urn=urn, tag="TestTag")
        xml_str   = xml_bytes.decode('utf-8')
        self.assertIn('xmlns:ns1="http://www.example.com/ns1"', xml_str)
        self.assertIn('xmlns:ns2="http://www.example.com/ns2"', xml_str)
        self.assertIn('<TestTag', xml_str)
        self.assertIn('<a>value</a>', xml_str)
        
    def test_serializable_to_xml_bytes_urn_invalid_type(self):
        obj = DummySerializableTestToXMLBytes(a="value")
        # urn is an int, which is not supported
        with self.assertRaisesRegex(TypeError, r'Expected string or ' + \
                                    'dictionary of string for urn, got type ' + \
                                    '\<class \'int\'\>$'):
            obj.to_xml_bytes(urn=12345, tag="TestTag")

    def test_serializable_recursive_validity_check_stack_logs_error(self):
        class DummyChild(Serializable):
            _fields = ('x',)
            def __init__(self, x=None):
                self.x = x
            def is_valid(self, recursive=True, stack=False):
                return False  # Always invalid

        class DummySerializable(Serializable):
            _fields = ('child',)
            def __init__(self, child=None):
                self.child = child

        obj = DummySerializable(child=DummyChild(x=1))
        # Should log an error for the invalid child when stack=True
        with self.assertLogs('validation', level='ERROR') as cm:
            result = obj._recursive_validity_check(stack=True)
        self.assertFalse(result)
        self.assertIn("Issue discovered with attribute child",
                      cm.output[0])
        
    def test_serializable_from_node_empty_node_warns(self):
        class DummySerializable(Serializable):
            _fields           = ('a',)
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

        # Create an empty XML node (no children, no attributes)
        node   = ElementTree.Element('root')
        xml_ns = None

        with self.assertLogs('sarpy.io.xml.base', level='WARNING') as cm:
            DummySerializable.from_node(node, xml_ns)
        self.assertIn("There are no children or attributes associated",
                      cm.output[0])
        
    def test_serializable_from_node_collections_tags_child_tag_none(self):
        class DummySerializable(Serializable):
            _fields           = ('a',)
            _collections_tags = {'a': {'array': False}}  # child_tag is missing
            _tag_override     = {}
            _set_as_attribute = ()
            _child_xml_ns_key = {}

        xml = """
        <root>
            <a>value</a>
        </root>
        """
        node   = ElementTree.fromstring(xml)
        xml_ns = None
        # Should raise ValueError because child_tag is None in _collections_tags
        with self.assertRaisesRegex(ValueError, r'Attribute a in class ' + \
                                    '\<class ' + \
                                    '\'test_Serializable_class.' + \
                                    'TestSerializable.' + \
                                    'test_serializable_from_node_collections_tags_child_tag_none.' + \
                                    '\<locals\>.DummySerializable\'\> is ' + \
                                    'listed in the _collections_tags ' + \
                                    'dictionary, but the `child_tag` value is ' + \
                                    'either not populated or None.'):
            DummySerializable.from_node(node, xml_ns)
        