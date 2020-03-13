
from . import unittest

from sarpy.io.nitf.tres.registration import find_tre
from sarpy.io.nitf.tres.unclass.ACFTA import ACFTA


class TestTreRegistry(unittest.TestCase):
    def test_find_tre(self):
        the_tre = find_tre('ACFTA')
        self.assertEqual(the_tre, ACFTA)
