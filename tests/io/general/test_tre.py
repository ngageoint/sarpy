import math
import pathlib
import unittest

import pytest

import sarpy.io.general.nitf_elements.tres.registration
from sarpy.io.general.nitf_elements.tres.registration import find_tre
from sarpy.io.general.nitf_elements.tres.unclass.ACFTA import ACFTA


class TestTreRegistry(unittest.TestCase):
    def test_find_tre(self):
        the_tre = find_tre('ACFTA')
        self.assertEqual(the_tre, ACFTA)


def _check_tre(tre_id, tre_bytes):
    tre_obj = find_tre(tre_id).from_bytes(tre_bytes, 0)
    assert tre_obj.to_bytes() == tre_bytes
    assert isinstance(tre_obj.DATA.to_dict(), dict)
    return tre_obj


def test_matesa(tests_path):
    tre_bytes = (tests_path / 'data/example_matesa_tre.bin').read_bytes()
    example = _check_tre("MATESA", tre_bytes)
    assert example.DATA.GROUPs[-1].MATEs[-1].MATE_ID == 'EO1H1680372005097110PZ.MET'


def test_bandsb(tests_path):
    tre_bytes = (tests_path / 'data/example_bandsb_tre.bin').read_bytes()
    example = _check_tre("BANDSB", tre_bytes)
    assert example.DATA.COUNT == 172
    assert example.DATA.RADIOMETRIC_QUANTITY == 'RADIANCE'
    assert example.DATA.RADIOMETRIC_QUANTITY_UNIT == 'S'
    assert example.DATA.SCALE_FACTOR == 80.0
    assert example.DATA.ADDITIVE_FACTOR == 0.0
    assert float(example.DATA.ROW_GSD) == 30.49
    assert example.DATA.ROW_GSD_UNIT == 'M'
    assert float(example.DATA.COL_GSD) == 29.96
    assert example.DATA.COL_GSD_UNIT == 'M'
    assert example.DATA.SPT_RESP_ROW == '-'*7
    assert example.DATA.SPT_RESP_UNIT_ROW == 'M'
    assert example.DATA.SPT_RESP_COL == '-'*7
    assert example.DATA.SPT_RESP_UNIT_COL == 'M'
    assert example.DATA.RADIOMETRIC_ADJUSTMENT_SURFACE == 'DETECTOR'
    assert math.isnan(example.DATA.ATMOSPHERIC_ADJUSTMENT_ALTITUDE)
    assert len(example.DATA.PARAMETERs) == 172
    band0 = example.DATA.PARAMETERs[0]
    assert band0.BAD_BAND == 0
    assert float(band0.CWAVE) == 0.85192
    assert float(band0.FWHM) == 0.01105
    band127 = example.DATA.PARAMETERs[127]
    assert band127.BAD_BAND == 1
    assert float(band127.CWAVE) == 2.13324
    assert float(band127.FWHM) == 0.01073
    band171 = example.DATA.PARAMETERs[171]
    assert band171.BAD_BAND == 0
    assert float(band171.CWAVE) == 2.57708
    assert float(band171.FWHM) == 0.01041


@pytest.fixture
def load_plugin(monkeypatch):
    with monkeypatch.context() as mp:
        mock_dir = pathlib.Path(__file__).parents[2] / 'mock_site-packages'
        mp.syspath_prepend(mock_dir)
        sarpy.io.general.nitf_elements.tres.registration._parsed_package = False
        sarpy.io.general.nitf_elements.tres.registration._TRE_Registry = {}
        yield
    sarpy.io.general.nitf_elements.tres.registration._parsed_package = False
    sarpy.io.general.nitf_elements.tres.registration._TRE_Registry = {}
    assert find_tre('SPLUGA') is None


def test_plugin_tres(load_plugin):
    assert find_tre('SPLUGA') is not None
