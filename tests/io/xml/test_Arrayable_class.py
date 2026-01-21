__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"

# Written using Copilot

import unittest
import numpy as np

from sarpy.io.xml.base import Arrayable

class DummyArrayable(Arrayable):
    def __init__(self, arr):
        self.arr = np.array(arr)
    @classmethod
    def from_array(cls, arr):
        return cls(arr)
    def get_array(self, dtype=None):
        if dtype is not None:
            return self.arr.astype(dtype)
        return self.arr

class TestArrayable(unittest.TestCase):
    def test_from_array_and_get_array(self):
        arr = [1, 2, 3]
        obj = DummyArrayable.from_array(arr)
        self.assertTrue(np.array_equal(obj.get_array(), np.array(arr)))

    def test_get_array_with_dtype(self):
        arr = [1, 2, 3]
        obj = DummyArrayable.from_array(arr)
        self.assertEqual(obj.get_array(dtype=np.float32).dtype, np.float32)

    def test_get_item(self):
        arr = [1, 2, 3]
        obj = DummyArrayable.from_array(arr)
        self.assertEqual(obj.__getitem__(1), 2)

    def test_repr(self):
        arr = [1, 2, 3]
        obj = DummyArrayable(arr)
        self.assertIn('DummyArrayable', repr(obj))

    def test_arrayable_from_array_not_implemented(self):
        class DummyArrayable(Arrayable):
            pass

        with self.assertRaises(NotImplementedError):
            DummyArrayable.from_array([1, 2, 3])
        dummy = DummyArrayable()
        with self.assertRaises(NotImplementedError):
            dummy.get_array()
