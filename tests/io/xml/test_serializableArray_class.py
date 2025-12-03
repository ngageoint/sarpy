import unittest
import numpy as np
from xml.etree import ElementTree

from sarpy.io.xml.base import Serializable, SerializableArray

class DummySerializable(Serializable):
    _fields   = ('x',)
    _required = ('x',)
    def __init__(self, x=None, **kwargs):
        super().__init__(x=x, **kwargs)
    @classmethod
    def from_array(cls, arr):
        return cls(x=arr[0])
    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        # For testing, just get text from node
        return cls(x=node.text if node.text else None)
    @classmethod
    def from_dict(cls, d):
        return cls(x=d.get('x', None))
    def get_array(self, dtype=None):
        return np.array([self.x], dtype=dtype if dtype else object)
    def to_dict(self, **kwargs):
        return {'x': self.x}

class DummyChild(Serializable):
    _fields = ('x',)
    def __init__(self, x=None):
        self.x = x

class TestSerializableArray(unittest.TestCase):
    def setUp(self):
        self.child_tag  = 'Dummy'
        self.child_type = DummySerializable

    def test_init_and_len(self):
        arr = [DummySerializable(x=1), DummySerializable(x=2)]
        sa  = SerializableArray(arr, 'test', self.child_tag, self.child_type)
        self.assertEqual(len(sa), 2)
        self.assertEqual(sa.size, 2)

    def test_set_array_and_get_array(self):
        arr    = [DummySerializable(x=1), DummySerializable(x=2)]
        sa     = SerializableArray(arr, 'test', self.child_tag, self.child_type)
        arr_obj   = sa.get_array()
        arr_float = sa.get_array(dtype=np.float64)
        self.assertEqual(arr_obj.size, 2)
        self.assertIsInstance(arr_obj,   np.ndarray)
        self.assertIsInstance(arr_float, np.ndarray)

    def test_set_array_none(self):
        sa = SerializableArray(None, 'test', self.child_tag, self.child_type)
        self.assertEqual(sa.size, 0)
        self.assertIsNone(sa._array)

    def test_minimum_maximum_length(self):
        arr = [DummySerializable(x=1)]
        sa  = SerializableArray(
            coords=arr, 
            name='test', 
            child_tag=self.child_tag, 
            child_type=self.child_type, 
            minimum_length=1, 
            maximum_length=2)
        self.assertEqual(sa.size, 1)
        with self.assertRaisesRegex(ValueError, 'Field test is required to ' + \
                                    'be an array with 1 <= length <= ' + \
                                    '4294967296, and input of length 0 was ' + \
                                    'received'):
            SerializableArray(
                coords=[], 
                name='test', 
                child_tag=self.child_tag, 
                child_type=self.child_type, 
                minimum_length=1)
        with self.assertRaisesRegex(ValueError, 'Field test is required to ' + \
                                    'be an array with 0 <= length <= 2, and ' + \
                                    'input of length 3 was received'):
            SerializableArray(
                coords=[DummySerializable(x=1)]*3, 
                name='test', 
                child_tag=self.child_tag, 
                child_type=self.child_type, 
                maximum_length=2)

    def test_setitem_and_getitem(self):
        arr = [DummySerializable(x=1), DummySerializable(x=2)]
        sa  = SerializableArray(arr, 'test', self.child_tag, self.child_type)
        # Test __setitem__
        sa[0] = DummySerializable(x=10)
        # Test __getitem
        self.assertEqual(sa[0].x, 10)
        with self.assertRaisesRegex(TypeError, 'Elements of test must be ' + \
                                    'of type ' + \
                                    '<class \'test_serializableArray_class.DummySerializable\'>' + \
                                    ', not None'):
            sa[1] = None

    def test_is_valid(self):
        arr = [DummySerializable(x=1), DummySerializable(x=2)]
        sa  = SerializableArray(arr, 'test', self.child_tag, self.child_type)
        sa_empty = SerializableArray(None, 'test', self.child_tag, 
                                     self.child_type)
        self.assertTrue(sa.is_valid())
        self.assertFalse(sa_empty.is_valid())

    def test_to_json_list(self):
        arr = [DummySerializable(x=1), DummySerializable(x=2)]
        sa  = SerializableArray(arr, 'test', self.child_tag, self.child_type)
        json_list = sa.to_json_list()
        self.assertEqual(json_list, [{'x': 1}, {'x': 2}])

    def test_to_node(self):
        arr  = [DummySerializable(x='foo'), DummySerializable(x='bar')]
        sa   = SerializableArray(arr, 'test', self.child_tag, self.child_type)
        doc  = ElementTree.ElementTree()
        node = sa.to_node(doc, tag='TestArray')
        children = list(node)
        self.assertEqual(node.tag,            'TestArray')
        self.assertEqual(node.attrib['size'], '2')
        self.assertEqual(len(children),       2)
        self.assertEqual(children[0].tag,     self.child_tag)
        # 'x' is the first element of the first element of the children list
        self.assertEqual(children[0][0].text, 'foo')

    def test_from_node(self):
        xml = """
        <TestArray size="2">
            <Dummy>foo</Dummy>
            <Dummy>bar</Dummy>
        </TestArray>
        """
        node = ElementTree.fromstring(xml)
        sa   = SerializableArray.from_node(node, name='test', 
                                         child_tag=self.child_tag, 
                                         child_type=self.child_type)
        self.assertEqual(sa.size, 2)
        self.assertEqual(sa[0].x, 'foo')
        self.assertEqual(sa[1].x, 'bar')

    def test_serializablearray_init_name_is_none(self):
        # Should raise ValueError because name is None
        with self.assertRaisesRegex(ValueError, 'The name parameter is required.'):
            SerializableArray([DummyChild(x=1)], None, 'Child', DummyChild)
        
    def test_serializablearray_init_name_not_str(self):
        # Should raise TypeError because name is not a string
        with self.assertRaisesRegex(TypeError, 'The name parameter is ' + \
                                    'required to be an instance of str'):
            SerializableArray([DummyChild(x=1)], 123, 'Child', DummyChild)
        
    def test_serializablearray_init_child_tag_is_none(self):
        # Should raise ValueError because child_tag is None
        with self.assertRaisesRegex(ValueError, 'The child_tag parameter is ' + \
                                    'required.'):
            SerializableArray([DummyChild(x=1)], "TestArray", None, DummyChild)

    def test_serializablearray_init_child_tag_not_str(self):
        # Should raise TypeError because child_tag is not a string
        with self.assertRaisesRegex(TypeError, 'The child_tag parameter is ' + \
                                    'required to be an instance of str'):
            SerializableArray([DummyChild(x=1)], "TestArray", 123, DummyChild)
        
    def test_serializablearray_init_child_type_is_none(self):
        # Should raise ValueError because child_type is None
        with self.assertRaisesRegex(ValueError, 'The child_type parameter is ' + \
                                    'required.'):
            SerializableArray([DummyChild(x=1)], "TestArray", "Child", None)
        
    def test_serializablearray_init_child_type_not_subclass_of_serializable(self):
        class NotSerializable:
            pass

        # Should raise TypeError because child_type is not a subclass of Serializable
        with self.assertRaisesRegex(TypeError, 'The child_type is required to be ' + \
                               'a subclass of Serializable.'):
            SerializableArray([NotSerializable()], "TestArray", "Child", 
                              NotSerializable)
    
    def test_serializablearray_log_validity_warning(self):
        arr = SerializableArray([DummyChild(x=1)], "TestArray", "Child", 
                                DummyChild)
        with self.assertLogs('validation', level='WARNING') as cm:
            arr.log_validity_warning("This is a test warning")
        self.assertIn("SerializableArray:TestArray This is a test warning",
                      cm.output[0])
        
    def test_serializablearray_log_validity_info(self):
        arr = SerializableArray([DummyChild(x=1)], "TestArray", "Child", 
                                DummyChild)
        with self.assertLogs('validation', level='INFO') as cm:
            arr.log_validity_info("This is a test info message")
        self.assertIn("SerializableArray:TestArray This is a test info message",
                      cm.output[0])
        
    def test_serializablearray_is_valid_array_not_none_not_recursive(self):
        arr = SerializableArray([DummyChild(x=1), DummyChild(x=2)], "TestArray", 
                                "Child", DummyChild)
        # _array is not None, and not recursive, so should return True
        self.assertTrue(arr.is_valid(recursive=True))

    def test_serializablearray_is_valid_stack_logs_error(self):
        class DummyChild(Serializable):
            _fields = ('x',)
            def __init__(self, x=None):
                self.x = x
            def is_valid(self, recursive=True, stack=False):
                return False  # Always invalid

        arr = SerializableArray([DummyChild(x=1), DummyChild(x=2)], "TestArray", 
                                "Child", DummyChild)
        # Should log an error for each invalid entry when stack=True
        with self.assertLogs('validation', level='ERROR') as cm:
            result = arr.is_valid(recursive=True, stack=True)
        self.assertFalse(result)
        self.assertIn("Issue discovered with entry", cm.output[0])

    def test_serializablearray_get_array_non_object_dtype_fail(self):
        class DummyChild(Serializable):
            _fields = ('x',)
            def __init__(self, x=None):
                self.x = x
            def get_array(self, dtype=np.float64, **kwargs):
                # Return a value that cannot be cast to float64 (e.g., a string)
                return "not_a_float"

        arr = SerializableArray([DummyChild(x=1), DummyChild(x=2)], "TestArray", 
                                "Child", DummyChild)
        # Should return None because conversion to float64 will fail
        result = arr.get_array(dtype=np.float64)
        self.assertIsNone(result)

    def test_serializablearray_check_indices_not_set_index(self):
        class DummySerializableArray(SerializableArray):
            _set_index = False  # Override to disable index setting

        arr = DummySerializableArray([DummyChild(x=1), DummyChild(x=2)], 
                                     "TestArray", "Child", DummyChild)
        # Should not raise or set any index attribute
        for entry in arr._array:
            self.assertFalse(hasattr(entry, 'index'))

    def test_serializablearray_check_indices_setattr_fail(self):
        class DummyChild(Serializable):
            _fields = ('x',)
            def __init__(self, x=None):
                self.x = x
            # Simulate failure on setattr
            @property
            def index(self):
                return 0
            @index.setter
            def index(self, value):
                raise AttributeError("Cannot set index")

        arr = SerializableArray([DummyChild(x=1), DummyChild(x=2)], "TestArray", 
                                "Child", DummyChild)
        # Should not raise, even though setting index fails
        arr._check_indices()
        # The index property should remain unchanged (always 0)
        for entry in arr._array:
            self.assertEqual(entry.index, 0)

    def test_serializablearray_to_node_size_zero(self):
        arr = SerializableArray([], "TestArray", "Child", DummyChild)
        doc = ElementTree.ElementTree()
        # Should return None since size == 0
        result = arr.to_node(doc, tag="TestArray")
        self.assertIsNone(result)

    def test_serializablearray_to_node_ns_key_not_none(self):
        arr  = SerializableArray([DummyChild(x=1), DummyChild(x=2)], 
                                "TestArray", "Child", DummyChild)
        doc  = ElementTree.ElementTree()
        # Use a namespace key
        node = arr.to_node(doc, tag="TestArray", ns_key="ns1")
        children = list(node)
        self.assertEqual(node.tag,      "ns1:TestArray")
        # Should have two children with tag "Child"
        self.assertEqual(len(children), 2)
        for child in children:
            self.assertEqual(child.tag, "ns1:Child")

    def test_serializablearray_to_json_list_size_zero(self):
        arr = SerializableArray([], "TestArray", "Child", DummyChild)
        # Should return an empty list since size == 0
        result = arr.to_json_list()
        self.assertEqual(result, [])