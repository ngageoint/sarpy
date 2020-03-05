
from . import unittest

from sarpy.io.nitf.tres.registration import find_tre


class TestTreRegistry(unittest.TestCase):
    def test_find_tre(self):
        the_tre = find_tre('AFTCA')
        class_name = 'AFTCA'
        self.assertEqual(the_tre.__name__, class_name)
