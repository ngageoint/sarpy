import os
import json
import tempfile
import unittest
import numpy as np
import pytest

from sarpy.io.complex.converter import conversion_utility
from sarpy.io.complex.sicd import SICDReader, AmpLookupFunction
from sarpy.io.complex.sicd_schema import get_schema_path, get_default_version_string
from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.nitf import NITFReader

from tests import parse_file_entry

try:
    from lxml import etree
except ImportError:
    etree = None


complex_file_types = {}
this_loc = os.path.abspath(__file__)
file_reference = os.path.join(os.path.split(this_loc)[0], 'complex_file_types.json')  # specifies file locations
if os.path.isfile(file_reference):
    with open(file_reference, 'r') as fi:
        the_files = json.load(fi)
        for the_type in the_files:
            valid_entries = []
            for entry in the_files[the_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            complex_file_types[the_type] = valid_entries

sicd_files = complex_file_types.get('SICD', [])

the_version = get_default_version_string()
the_schema = get_schema_path(the_version)


class TestSICDWriting(unittest.TestCase):

    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def test_sicd_creation(self):
        for fil in sicd_files:
            reader = SICDReader(fil)

            # check that sicd structure serializes according to the schema
            if etree is not None:
                sicd = reader.get_sicds_as_tuple()[0]
                xml_doc = etree.fromstring(sicd.to_xml_bytes())
                xml_schema = etree.XMLSchema(file=the_schema)
                with self.subTest(msg='validate xml produced from sicd structure'):
                    self.assertTrue(xml_schema.validate(xml_doc),
                                    msg='SICD structure serialized from file {} is '
                                        'not valid versus schema {}'.format(fil, the_schema))

            with self.subTest(msg='Test conversion (recreation) of the sicd file {}'.format(fil)):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    conversion_utility(reader, tmpdirname)
                    new_filename = os.path.join(tmpdirname, os.listdir(tmpdirname)[0])
                    with SICDReader(new_filename) as reader2:
                        self.assertEqual(os.stat(new_filename).st_size, reader2.nitf_details.nitf_header.FL)

            with self.subTest(msg='Test writing a single row of the sicd file {}'.format(fil)):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    conversion_utility(reader, tmpdirname, row_limits=(0, 1))

class DummySICDMeta:
    class ImageDataType:
        def __init__(self, pixel_type, amp_table=None):
            self.PixelType = pixel_type
            self.AmpTable = amp_table

    def __init__(self, pixel_type='RE32F_IM32F', amp_table=None):
        self.ImageData = self.ImageDataType(pixel_type, amp_table)

@pytest.mark.parametrize(
    "raw_dtype,complex_order,band_dimension,pixel_type,amp_table,expected_type",
    [
        (np.dtype('float32'), 'IQ', 2, 'RE32F_IM32F', None, ComplexFormatFunction),
        (np.dtype('int16'), 'IQ', 2, 'RE16I_IM16I', None, ComplexFormatFunction),
        (np.dtype('uint8'), 'MP', 2, 'AMP8I_PHS8I', np.linspace(0, 1, 256, dtype=np.float32), AmpLookupFunction),
    ]
)
def test_sicdreader_get_format_function(raw_dtype, complex_order, band_dimension, pixel_type, amp_table, expected_type):
    class DummyReader(SICDReader):
        def __init__(self, sicd_meta):
            # Avoid full parent init
            self._sicd_meta = sicd_meta

        @property
        def sicd_meta(self):
            return self._sicd_meta

    reader = DummyReader(DummySICDMeta(pixel_type, amp_table))
    func = reader.get_format_function(raw_dtype, complex_order, None, band_dimension)
    assert isinstance(func, expected_type)

def test_sicdreader_get_format_function_invalid_mp():
    class DummyReader(SICDReader):
        def __init__(self, sicd_meta):
            self._sicd_meta = sicd_meta

        @property
        def sicd_meta(self):
            return self._sicd_meta

    # Should raise ValueError for unsupported MP type
    reader = DummyReader(DummySICDMeta('AMP8I_PHS8I', np.linspace(0, 1, 256, dtype=np.float32)))
    with pytest.raises(ValueError):
        reader.get_format_function(np.dtype('uint16'), 'MP', None, 2)

def test_sicdreader_get_format_function_fallback():
    # Should fallback to NITFReader.get_format_function and return None
    class DummyReader(SICDReader):
        def __init__(self, sicd_meta):
            self._sicd_meta = sicd_meta

        @property
        def sicd_meta(self):
            return self._sicd_meta

    reader = DummyReader(DummySICDMeta('RE32F_IM32F'))
    func = reader.get_format_function(np.dtype('float32'), None, None, 2)
    assert func is None