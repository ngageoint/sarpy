import numpy as np
import sarpy.io.phase_history.cphd1_elements.utils

from tests import unittest


class TestCphd1Utils(unittest.TestCase):
    def test_binary_format_to_dtype(self):
        self.assertEqual(sarpy.io.phase_history.cphd1_elements.utils.binary_format_string_to_dtype('I1'), np.int8)
        dt = np.dtype([('a', '>i8'),
                       ('b', '>f8'),
                       ('c', '>f8')])
        self.assertEqual(sarpy.io.phase_history.cphd1_elements.utils.binary_format_string_to_dtype('a=I8;b=F8;c=F8;'),  dt)
