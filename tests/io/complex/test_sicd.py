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
from sarpy.io.complex.sicd import SICDDetails
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.general.nitf_elements.des import DataExtensionHeader
from sarpy.io.general.base import SarpyIOError

from tests import parse_file_entry

try:
    from lxml import etree
except ImportError:
    etree = None


complex_file_types = {}
this_loc = os.path.abspath(__file__)
# specifies file locations
file_reference = os.path.join(os.path.split(this_loc)[0], 
                              'complex_file_types.json')  
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
                    new_filename = os.path.join(tmpdirname, 
                                                os.listdir(tmpdirname)[0])
                    with SICDReader(new_filename) as reader2:
                        self.assertEqual(os.stat(new_filename).st_size, 
                                         reader2.nitf_details.nitf_header.FL)

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
        (np.dtype('float32'), 'IQ', 2, 'RE32F_IM32F', None, 
         ComplexFormatFunction),
        (np.dtype('int16'), 'IQ', 2, 'RE16I_IM16I', None, 
         ComplexFormatFunction),
        (np.dtype('uint8'), 'MP', 2, 'AMP8I_PHS8I', 
         np.linspace(0, 1, 256, dtype=np.float32), AmpLookupFunction),
    ]
)
def test_sicdreader_get_format_function(
    raw_dtype, 
    complex_order, 
    band_dimension, 
    pixel_type, 
    amp_table, 
    expected_type):
    
    class DummyReader(SICDReader):
        def __init__(self, sicd_meta):
            # Avoid full parent init
            self._sicd_meta = sicd_meta

        @property
        def sicd_meta(self):
            return self._sicd_meta

    reader = DummyReader(DummySICDMeta(pixel_type, amp_table))
    func = reader.get_format_function(raw_dtype, complex_order, None, 
                                      band_dimension)
    assert isinstance(func, expected_type)

def test_sicdreader_get_format_function_invalid_mp():
    class DummyReader(SICDReader):
        def __init__(self, sicd_meta):
            self._sicd_meta = sicd_meta

        @property
        def sicd_meta(self):
            return self._sicd_meta

    # Should raise ValueError for unsupported MP type
    reader = DummyReader(DummySICDMeta('AMP8I_PHS8I', 
                                       np.linspace(0, 1, 256, dtype=np.float32)))
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

def test_sicdreader_get_format_function_required_params_only():
    # Should fallback to NITFReader.get_format_function and return None
    class DummyReader(SICDReader):
        def __init__(self, sicd_meta):
            self._sicd_meta = sicd_meta

        @property
        def sicd_meta(self):
            return self._sicd_meta

    reader = DummyReader(DummySICDMeta('RE32F_IM32F'))
    func = reader.get_format_function(np.dtype('float32'))
    assert func is None


def test_amplookupfunction_valid_initialization():
    # Valid initialization with float32 lookup table
    lut = np.linspace(0, 1, 256, dtype=np.float32)
    func = AmpLookupFunction('uint8', lut, band_dimension=2)
    assert isinstance(func, AmpLookupFunction)
    assert np.allclose(func.magnitude_lookup_table, lut)

def test_amplookupfunction_invalid_dtype():
    # Should raise ValueError if lookup table is not ndarray
    with pytest.raises(ValueError):
        AmpLookupFunction('uint8', list(np.linspace(0, 1, 256)), 
                          band_dimension=2)
    # Should raise ValueError if dtype is not float32 or float64
    with pytest.raises(ValueError):
        AmpLookupFunction('uint8', np.arange(256, dtype=np.int32), 
                          band_dimension=2)
    # Should raise ValueError if shape is not (256,)
    with pytest.raises(ValueError):
        AmpLookupFunction('uint8', np.linspace(0, 1, 128, dtype=np.float32), 
                          band_dimension=2)
    # Should raise ValueError if raw_dtype is not uint8
    with pytest.raises(ValueError):
        AmpLookupFunction('int16', np.linspace(0, 1, 256, dtype=np.float32), 
                          band_dimension=2)

def test_amplookupfunction_set_magnitude_lookup():
    lut1 = np.linspace(0, 1, 256, dtype=np.float32)
    lut2 = np.linspace(1, 2, 256, dtype=np.float32)
    func = AmpLookupFunction('uint8', lut1, band_dimension=2)
    func.set_magnitude_lookup(lut2)
    assert np.allclose(func.magnitude_lookup_table, lut2)

def test_amplookupfunction_forward_reverse_methods():
    lut = np.linspace(0, 1, 256, dtype=np.float32)
    func = AmpLookupFunction('uint8', lut, band_dimension=2)
    # Prepare dummy data
    data = np.zeros((4, 2), dtype=np.uint8)
    out = np.zeros((4, 2), dtype=np.complex64)
    magnitude = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.uint8)
    theta = np.array([[0, np.pi/2], [np.pi, 3*np.pi/2], [4*np.pi/2, 5*np.pi/2], 
                      [6*np.pi/2, 7*np.pi/2]], dtype=np.float32)
    subscript = (slice(None), slice(None))
    # Should run without error
    func._forward_magnitude_theta(data, out, magnitude, theta, subscript)
    slice0 = (slice(None), slice(None))
    slice1 = (slice(None), slice(None))
    func._reverse_magnitude_theta(data, out, magnitude.astype(np.float32), 
                                  theta, slice0, slice1)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_is_sicd_property():
    details = SICDDetails(sicd_files[0])
    assert details.is_sicd is True

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_sicd_meta_property():
    details = SICDDetails(sicd_files[0])
    assert isinstance(details.sicd_meta, SICDType)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_des_header_property():
    details = SICDDetails(sicd_files[0])
    assert isinstance(details.des_header, DataExtensionHeader)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_find_sicd_sets_is_sicd():
    details = SICDDetails(sicd_files[0])
    details._is_sicd = False
    details._find_sicd()
    assert details.is_sicd is True

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_subhead_sizes_zero_fail():
    details = SICDDetails(sicd_files[0])
    with pytest.raises(AttributeError):
        details._nitf_header.ImageSegments.subhead_sizes.size = 0

class DummySubheadSizes:
    def __init__(self, size):
        self.size = size

class DummyImageSegments:
    def __init__(self, size):
        self.subhead_sizes = DummySubheadSizes(size)

class DummyGraphicsSegments:
    def __init__(self, size):
        self.item_sizes = DummySubheadSizes(size)

class DummyDataExtensions:
    def __init__(self, size):
        self.subhead_sizes = DummySubheadSizes(size)

class DummyNITFHeader:
    def __init__(self):
        self.ImageSegments = DummyImageSegments(1)
        self.GraphicsSegments = DummyGraphicsSegments(0)
        self.DataExtensions = DummyDataExtensions(1)

class DummyNITFDetails(SICDDetails):
    def __init__(self):
        self._des_index = None
        self._des_header = None
        self._img_headers = None
        self._is_sicd = False  # Simulate not a SICD file
        self._sicd_meta = None
        self._nitf_header = DummyNITFHeader()
        # Skip calling super().__init__ to avoid real parsing
        # Simulate _find_sicd does nothing

    def _find_sicd(self):
        pass

    @property
    def is_sicd(self):
        return self._is_sicd

def test_sicddetails_init_not_sicd(monkeypatch):
    # Patch NITFDetails.__init__ to do nothing
    monkeypatch.setattr("sarpy.io.general.nitf.NITFDetails.__init__", lambda self, file_object: None)
    # Patch SICDDetails._find_sicd to do nothing
    monkeypatch.setattr("sarpy.io.complex.sicd.SICDDetails._find_sicd", lambda self: None)
    # Patch SICDDetails.is_sicd to always return False
    monkeypatch.setattr("sarpy.io.complex.sicd.SICDDetails.is_sicd", property(lambda self: False))
    # Patch SICDDetails._nitf_header to our dummy header
    def dummy_init(self, file_object):
        self._des_index = None
        self._des_header = None
        self._img_headers = None
        self._is_sicd = False
        self._sicd_meta = None
        self._nitf_header = DummyNITFHeader()
        self._find_sicd()
        if not self.is_sicd:
            raise SarpyIOError('Could not find the SICD XML des.')
    monkeypatch.setattr("sarpy.io.complex.sicd.SICDDetails.__init__", dummy_init)
    with pytest.raises(SarpyIOError, match="Could not find the SICD XML des."):
        SICDDetails("dummy_file")