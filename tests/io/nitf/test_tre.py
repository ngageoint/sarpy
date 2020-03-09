
from . import unittest

from sarpy.io.nitf.tres.registration import find_tre
from sarpy.io.nitf.tres.unclass.AFTCA import AFTCA


class TestTreRegistry(unittest.TestCase):
    def test_find_tre(self):
        the_tre = find_tre('AFTCA')
        self.assertEqual(the_tre, AFTCA)
