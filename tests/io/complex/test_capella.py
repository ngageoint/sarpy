import pathlib
import pytest
import tests
import sarpy.io.complex.capella

def test_isa():

    filename = '<sar data>/sarpy_test/capella/CAPELLA_C03_SP_SLC_HH_20210313173209_20210313173212.tif'
    reader = sarpy.io.complex.capella.is_a( filename )
    print( "reader: {} ".format( reader ))
    assert isinstance( reader, sarpy.io.complex.capella.CapellaReader )
