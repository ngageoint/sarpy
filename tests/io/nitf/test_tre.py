
from . import unittest

from sarpy.io.general.nitf_elements.tres.registration import find_tre
from sarpy.io.general.nitf_elements.tres.unclass.ACFTA import ACFTA


class TestTreRegistry(unittest.TestCase):
    def test_find_tre(self):
        the_tre = find_tre('ACFTA')
        self.assertEqual(the_tre, ACFTA)
