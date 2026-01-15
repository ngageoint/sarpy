import pathlib
import pytest
import tests
import sarpy.io.complex.tsx
def test_isa():

   filename = '<sar data> /sarpy_test/TSX/SO_000009564_0001_1/TSX1_SAR__SSC______HS_S_SRA_20090212T204239_20090212T204240/'
   reader = sarpy.io.complex.tsx.is_a( filename )
   assert isinstance(reader, sarpy.io.complex.tsx.TSXReader )
