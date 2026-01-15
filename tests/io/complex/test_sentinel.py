import pathlib
import pytest
import tests
import sarpy.io.complex.sentinel
def test_isa():
    filename = '<sar data>/sarpy_test/sentinel/S1A_IW_SLC__1SDH_20200201T141253_20200201T141321_031060_03917E_BAA7.SAFE/'
    reader = sarpy.io.complex.sentinel.is_a( filename )
    assert isinstance(reader, sarpy.io.complex.sentinel.SentinelReader)
