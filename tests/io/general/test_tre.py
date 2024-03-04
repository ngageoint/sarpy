from sarpy.io.general.nitf_elements.tres.registration import find_tre
from sarpy.io.general.nitf_elements.tres.unclass.ACFTA import ACFTA
import unittest


class TestTreRegistry(unittest.TestCase):
    def test_find_tre(self):
        the_tre = find_tre('ACFTA')
        self.assertEqual(the_tre, ACFTA)


def test_matesa(tests_path):
    example = find_tre('MATESA').from_bytes((tests_path / 'data/example_matesa_tre.bin').read_bytes(), 0)
    assert example.DATA.GROUPs[-1].MATEs[-1].MATE_ID == 'EO1H1680372005097110PZ.MET'
