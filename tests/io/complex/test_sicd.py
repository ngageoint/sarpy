from xml.etree   import ElementTree
from io          import StringIO
import os, re
import json
import tempfile
import unittest
from unittest import TestCase
import numpy as np
import datetime
import pytest
from pytest import fixture

from sarpy.__about__ import __title__, __version__
from sarpy.io.complex.converter import conversion_utility
from sarpy.io.complex.sicd import SICDReader, AmpLookupFunction
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType
from sarpy.io.complex.sicd_schema import get_schema_path, get_default_version_string
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType, FullImageType
from sarpy.io.complex.sicd_elements.blocks     import RowColType
from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.nitf import NITFReader
from sarpy.io.complex.sicd import SICDDetails, is_a, validate_sicd_for_writing, extract_clas
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.general.nitf_elements.des import DataExtensionHeader
from sarpy.io.general.base import SarpyIOError
from sarpy.io.xml.base import parse_xml_from_string
from sarpy.io.xml.descriptors import StringEnumDescriptor
from sarpy.io.complex.sicd import create_security_tags_from_sicd
from sarpy.io.general.nitf_elements.security import NITFSecurityTags

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

sicd_meta_data = SICDType(
    ImageData=ImageDataType(
        NumRows=1,
        NumCols=2,
        PixelType="RE32F_IM32F",
        FirstRow=0,
        FirstCol=0,
        FullImage=FullImageType(
            NumRows=1,
            NumCols=2
        ),
        SCPPixel=RowColType(Row=1, 
                            Col=2)
    ),
)
    
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

def test_amplookupfunction_set_magnitude_lookup_raw_dtype_not_uint8():
    lut = np.linspace(0, 1, 256, dtype=np.float32)
    # Should raise ValueError when raw_dtype is not 'uint8'
    with pytest.raises(ValueError, match="A magnitude lookup table has been supplied,\n\tbut the raw datatype is not `uint8`."):
        func = AmpLookupFunction('uint16', lut, band_dimension=2)

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
def test_sicddetails_find_sicd_sets_is_sicd_no_des_subheader_offsets(monkeypatch):
    details = SICDDetails(sicd_files[0])
    details.des_subheader_offsets = None
    details._find_sicd()
    assert details.is_sicd is False
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_subhead_sizes_zero_fail():
    details = SICDDetails(sicd_files[0])
    with pytest.raises(AttributeError):
        details._nitf_header.ImageSegments.subhead_sizes.size = 0

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_init_not_sicd(monkeypatch):
    monkeypatch.setattr("sarpy.io.complex.sicd.SICDDetails.is_sicd", property(lambda self: False))
    with pytest.raises(SarpyIOError, match="Could not find the SICD XML des."):
        details = SICDDetails(sicd_files[0])

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_init_des_subheader_offsets_none(monkeypatch):
    details = SICDDetails(sicd_files[0])
    details.des_subheader_offsets = None
    details._find_sicd()
    assert details.is_sicd is False

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_init_des_subheader_offsets_size_zero(monkeypatch):
    details = SICDDetails(sicd_files[0])
    details.des_subheader_offsets = np.empty(shape=(0))
    details._find_sicd()
    assert details.is_sicd is False

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_find_sicd_dexml_data_content_sicd(monkeypatch):
    details = SICDDetails(sicd_files[0])
    details._find_sicd()
    assert details.is_sicd is True
    assert isinstance(details.sicd_meta, SICDType)
    assert details._des_index == 0
    assert details._des_header is not None

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_sicddetails_find_sicd_desidd_xml(monkeypatch):
    class DummySICDDetails(SICDDetails):
        def get_des_subheader_bytes(self, index: int):
            return b'DESIDD_XML'
    with pytest.raises(SarpyIOError, match="Could not find the SICD XML des."):
        details = DummySICDDetails(sicd_files[0])

def test_sicddetails_find_sicd_dexml_data_content_sidd(monkeypatch):
    """
    Test SICDDetails._find_sicd when subhead_bytes.startswith(b'DEXML_DATA_CONTENT') is True
    and 'SIDD' is in root_node.tag, should set _is_sicd to False and break.
    """
    class DummySICDDetails(SICDDetails):
        def __init__(self):
            # Avoid parent init
            self._des_index = None
            self._des_header = None
            self._img_headers = None
            self._is_sicd = False
            self._sicd_meta = None
            self.des_subheader_offsets = type('DummyOffsets', (), {'size': 1})()
            self._called = False

        def get_des_subheader_bytes(self, index):
            return b'DEXML_DATA_CONTENT         01UUS                                                                                                                                                                    077399999XML     2021-11-30T20:59:38Z                                        SICD Volume 1 Design & Implementation Description Document  1.1       2014-09-30T00:00:00Zurn:SICD:1.1.0                                                                                                          +35.05301449-106.59263183+35.05542923-106.59424948+35.05604151-106.59253016+35.05362675-106.59091255+35.05301449-106.59263183                                                                                                                                                                                                                                                                                                                                                                             '

        def get_des_bytes(self, index):
            return b"<SIDD></SIDD>"

    # Patch parse_xml_from_string to return a dummy root_node with 'SIDD' in tag
    def dummy_parse_xml_from_string(xml_bytes):
        class DummyRoot:
            tag = 'SIDD'
        return DummyRoot(), None

    monkeypatch.setattr("sarpy.io.complex.sicd.parse_xml_from_string", dummy_parse_xml_from_string)

    details = DummySICDDetails()
    details._find_sicd()
    assert details._is_sicd is False
    assert details._des_index is None
    assert details._des_header is None
    assert details._sicd_meta is None

def test_sicddetails_find_sicd_dexml_data_content_sicd_xmlns_none(monkeypatch):
    """
    Test SICDDetails._find_sicd when subhead_bytes.startswith(b'DEXML_DATA_CONTENT') is True,
    'SICD' in root_node.tag is True, and xml_ns is None.
    Should set _is_sicd True, _des_index and _des_header set, and _sicd_meta from_node called with ns_key=None.
    """
    # Dummy SICDType for from_node
    class DummySICDType:
        called = {}
        @classmethod
        def from_node(cls, root_node, xml_ns, ns_key=None):
            cls.called = {'root_node': root_node, 'xml_ns': xml_ns, 'ns_key': ns_key}
            return 'dummy_sicd_meta'
        def derive(self): pass

    # Patch SICDType in module under test
    monkeypatch.setattr("sarpy.io.complex.sicd.SICDType", DummySICDType)

    # Dummy parse_xml_from_string returns root_node with tag 'SICD' and xml_ns None
    class DummyRoot:
        tag = 'SICD'
    def dummy_parse_xml_from_string(xml_bytes):
        return DummyRoot(), None
    monkeypatch.setattr("sarpy.io.complex.sicd.parse_xml_from_string", dummy_parse_xml_from_string)

    # Dummy DataExtensionHeader
    class DummyDataExtensionHeader:
        @staticmethod
        def from_bytes(subhead_bytes, start=0):
            return 'dummy_des_header'
    monkeypatch.setattr("sarpy.io.complex.sicd.DataExtensionHeader", DummyDataExtensionHeader)

    # Dummy SICDDetails with one DES subheader offset
    class DummySICDDetails(SICDDetails):
        def __init__(self):
            self._des_index = None
            self._des_header = None
            self._img_headers = None
            self._is_sicd = False
            self._sicd_meta = None
            self.des_subheader_offsets = type('DummyOffsets', (), {'size': 1})()
        def get_des_subheader_bytes(self, index):
            return b'DEXML_DATA_CONTENT         01UUS                                                                                                                                                                    077399999XML     2021-11-30T20:59:38Z                                        SICD Volume 1 Design & Implementation Description Document  1.1       2014-09-30T00:00:00Zurn:SICD:1.1.0                                                                                                          +35.05301449-106.59263183+35.05542923-106.59424948+35.05604151-106.59253016+35.05362675-106.59091255+35.05301449-106.59263183                                                                                                                                                                                                                                                                                                                                                                             '
        def get_des_bytes(self, index):
            return b"<SICD></SICD>"

    details = DummySICDDetails()
    details._find_sicd()
    assert details._is_sicd is True
    assert details._des_index == 0
    assert details._des_header == 'dummy_des_header'
    assert details._sicd_meta == 'dummy_sicd_meta'
    assert DummySICDType.called['ns_key'] is None
    assert DummySICDType.called['xml_ns'] is None
    assert DummySICDType.called['root_node'].tag == 'SICD'

def test_sicddetails_find_sicd_dexml_data_content_parse_xml_fails(monkeypatch):
    """
    Test SICDDetails._find_sicd when subhead_bytes.startswith(b'DEXML_DATA_CONTENT') is True
    and parse_xml_from_string raises an Exception, should continue loop and not set _is_sicd.
    """
    # Dummy DataExtensionHeader
    class DummyDataExtensionHeader:
        @staticmethod
        def from_bytes(subhead_bytes, start=0):
            return 'dummy_des_header'

    # Patch DataExtensionHeader in module under test
    monkeypatch.setattr("sarpy.io.complex.sicd.DataExtensionHeader", DummyDataExtensionHeader)

    # Patch parse_xml_from_string to raise Exception
    def dummy_parse_xml_from_string(xml_bytes):
        raise Exception("parse_xml failed")

    monkeypatch.setattr("sarpy.io.complex.sicd.parse_xml_from_string", dummy_parse_xml_from_string)

    # Dummy SICDDetails with one DES subheader offset
    class DummySICDDetails(SICDDetails):
        def __init__(self):
            self._des_index = None
            self._des_header = None
            self._img_headers = None
            self._is_sicd = False
            self._sicd_meta = None
            self.des_subheader_offsets = type('DummyOffsets', (), {'size': 1})()
        def get_des_subheader_bytes(self, index):
            return b'DEXML_DATA_CONTENT         01UUS'
        def get_des_bytes(self, index):
            return b"<SICD></SICD>"

    details = DummySICDDetails()
    details._find_sicd()
    assert details._is_sicd is False
    assert details._des_index is None
    assert details._des_header is None
    assert details._sicd_meta is None

def test_sicddetails_find_sicd_desicd_xml(monkeypatch):
    """
    Test SICDDetails._find_sicd when subhead_bytes.startswith(b'DESICD_XML') is True,
    and 'SICD' is in root_node.tag, should set _is_sicd True, _des_index set, _des_header None,
    and _sicd_meta from_node called with ns_key=None or 'default'.
    """
    # Dummy SICDType for from_node
    class DummySICDType:
        called = {}
        @classmethod
        def from_node(cls, root_node, xml_ns, ns_key=None):
            cls.called = {'root_node': root_node, 'xml_ns': xml_ns, 'ns_key': ns_key}
            return 'dummy_sicd_meta'
        def derive(self): pass

    # Patch SICDType in module under test
    monkeypatch.setattr("sarpy.io.complex.sicd.SICDType", DummySICDType)

    # Dummy parse_xml_from_string returns root_node with tag 'SICD' and xml_ns None
    class DummyRoot:
        tag = 'SICD'
    def dummy_parse_xml_from_string(xml_bytes):
        return DummyRoot(), None
    monkeypatch.setattr("sarpy.io.complex.sicd.parse_xml_from_string", dummy_parse_xml_from_string)

    # Dummy SICDDetails with one DES subheader offset
    class DummySICDDetails(SICDDetails):
        def __init__(self):
            self._des_index = None
            self._des_header = None
            self._img_headers = None
            self._is_sicd = False
            self._sicd_meta = None
            self.des_subheader_offsets = type('DummyOffsets', (), {'size': 1})()
        def get_des_subheader_bytes(self, index):
            return b'DESICD_XML'
        def get_des_bytes(self, index):
            return b"<SICD></SICD>"

    details = DummySICDDetails()
    details._find_sicd()
    assert details._is_sicd is True
    assert details._des_index == 0
    assert details._des_header is None
    assert details._sicd_meta == 'dummy_sicd_meta'
    assert DummySICDType.called['ns_key'] is None
    assert DummySICDType.called['root_node'].tag == 'SICD'

def test_sicddetails_find_sicd_desicd_xml_parse_xml_fails(monkeypatch):
    """
    Test SICDDetails._find_sicd when subhead_bytes.startswith(b'DESICD_XML') is True
    and parse_xml_from_string raises an Exception, should continue loop and not set _is_sicd.
    """
    # Patch parse_xml_from_string to raise Exception
    def dummy_parse_xml_from_string(xml_bytes):
        raise Exception("parse_xml failed")
    monkeypatch.setattr("sarpy.io.complex.sicd.parse_xml_from_string", dummy_parse_xml_from_string)

    # Dummy SICDDetails with one DES subheader offset
    class DummySICDDetails(SICDDetails):
        def __init__(self):
            self._des_index = None
            self._des_header = None
            self._img_headers = None
            self._is_sicd = False
            self._sicd_meta = None
            self.des_subheader_offsets = type('DummyOffsets', (), {'size': 1})()
        def get_des_subheader_bytes(self, index):
            return b'DESICD_XML'
        def get_des_bytes(self, index):
            return b"<SICD></SICD>"

    details = DummySICDDetails()
    details._find_sicd()
    assert details._is_sicd is False
    assert details._des_index is None
    assert details._des_header is None
    assert details._sicd_meta is None

def test_sicddetails_find_sicd_desicd_xml_xmlns_none_false(monkeypatch):
    """
    Test SICDDetails._find_sicd when subhead_bytes.startswith(b'DESICD_XML') is True,
    'SICD' in root_node.tag is True, and xml_ns is not None.
    Should set _is_sicd True, _des_index set, _des_header None,
    and _sicd_meta from_node called with ns_key='default'.
    """
    # Dummy SICDType for from_node
    class DummySICDType:
        called = {}
        @classmethod
        def from_node(cls, root_node, xml_ns, ns_key=None):
            cls.called = {'root_node': root_node, 'xml_ns': xml_ns, 'ns_key': ns_key}
            return 'dummy_sicd_meta'
        def derive(self): pass

    # Patch SICDType in module under test
    monkeypatch.setattr("sarpy.io.complex.sicd.SICDType", DummySICDType)

    # Dummy parse_xml_from_string returns root_node with tag 'SICD' and xml_ns not None
    class DummyRoot:
        tag = 'SICD'
    def dummy_parse_xml_from_string(xml_bytes):
        return DummyRoot(), {'dummy': 'ns'}
    monkeypatch.setattr("sarpy.io.complex.sicd.parse_xml_from_string", dummy_parse_xml_from_string)

    # Dummy SICDDetails with one DES subheader offset
    class DummySICDDetails(SICDDetails):
        def __init__(self):
            self._des_index = None
            self._des_header = None
            self._img_headers = None
            self._is_sicd = False
            self._sicd_meta = None
            self.des_subheader_offsets = type('DummyOffsets', (), {'size': 1})()
        def get_des_subheader_bytes(self, index):
            return b'DESICD_XML'
        def get_des_bytes(self, index):
            return b"<SICD></SICD>"

    details = DummySICDDetails()
    details._find_sicd()
    assert details._is_sicd is True
    assert details._des_index == 0
    assert details._des_header is None
    assert details._sicd_meta == 'dummy_sicd_meta'
    assert DummySICDType.called['ns_key'] == 'default'
    assert DummySICDType.called['xml_ns'] == {'dummy': 'ns'}
    assert DummySICDType.called['root_node'].tag == 'SICD'

def test_sicdreader_init_raises_typeerror_for_invalid_nitf_details():
    # Pass an object that is not a string, file-like, or SICDDetails
    class NotSICDDetails:
        pass

    with pytest.raises(TypeError, match="The input argument for SICDReader must be a filename, file-like object, or SICDDetails object."):
        SICDReader(NotSICDDetails())

def test_sicdreader_get_nitf_dict_returns_expected_keys(monkeypatch):
    # Prepare dummy nitf_header and img_headers
    class DummySecurity:
        CLAS = "U"
        CODE = "US"
        # Fill all fields in NITFSecurityTags._ordering with non-empty values
        def __getattr__(self, name):
            return "VAL"

    class DummyNITFHeader:
        Security = DummySecurity()
        OSTAID = "OSTAID_VAL"
        FTITLE = "FTITLE_VAL"

    class DummyImgHeader:
        ISORCE = "ISORCE_VAL"
        IID2 = "IID2_VAL"

    class DummyNitfDetails:
        nitf_header = DummyNITFHeader()
        img_headers = [DummyImgHeader()]

    class DummySICDReader(SICDReader):
        def __init__(self):
            self._nitf_details = DummyNitfDetails()
            self._sicd_meta = None

        @property
        def nitf_details(self):
            return self._nitf_details

    reader = DummySICDReader()
    result = reader.get_nitf_dict()
    assert "Security" in result
    assert "OSTAID" in result
    assert "FTITLE" in result
    assert "ISORCE" in result
    assert "IID2" in result
    # Security dict should contain all fields from NITFSecurityTags._ordering
    from sarpy.io.general.nitf_elements.security import NITFSecurityTags
    for field in NITFSecurityTags._ordering:
        assert field in result["Security"]

def test_sicdreader_get_nitf_dict_security_empty(monkeypatch):
    # Security fields are all empty, so Security key should not be present
    class DummySecurity:
        def __getattr__(self, name):
            return ""

    class DummyNITFHeader:
        Security = DummySecurity()
        OSTAID = "OSTAID_VAL"
        FTITLE = "FTITLE_VAL"

    class DummyImgHeader:
        ISORCE = "ISORCE_VAL"
        IID2 = "IID2_VAL"

    class DummyNitfDetails:
        nitf_header = DummyNITFHeader()
        img_headers = [DummyImgHeader()]

    class DummySICDReader(SICDReader):
        def __init__(self):
            self._nitf_details = DummyNitfDetails()
            self._sicd_meta = None

        @property
        def nitf_details(self):
            return self._nitf_details

    reader = DummySICDReader()
    result = reader.get_nitf_dict()
    assert "Security" not in result
    assert result["OSTAID"] == "OSTAID_VAL"
    assert result["FTITLE"] == "FTITLE_VAL"
    assert result["ISORCE"] == "ISORCE_VAL"
    assert result["IID2"] == "IID2_VAL"

def test_sicdreader_get_nitf_dict_multiple_img_headers(monkeypatch):
    # Only the first img_headers[0] should be used for ISORCE and IID2
    class DummySecurity:
        CLAS = "U"
        CODE = "US"
        def __getattr__(self, name):
            return "VAL"

    class DummyNITFHeader:
        Security = DummySecurity()
        OSTAID = "OSTAID_VAL"
        FTITLE = "FTITLE_VAL"

    class DummyImgHeader:
        def __init__(self, isorce, iid2):
            self.ISORCE = isorce
            self.IID2 = iid2

    class DummyNitfDetails:
        nitf_header = DummyNITFHeader()
        img_headers = [
            DummyImgHeader("ISORCE_1", "IID2_1"),
            DummyImgHeader("ISORCE_2", "IID2_2"),
        ]

    class DummySICDReader(SICDReader):
        def __init__(self):
            self._nitf_details = DummyNitfDetails()
            self._sicd_meta = None

        @property
        def nitf_details(self):
            return self._nitf_details

    reader = DummySICDReader()
    result = reader.get_nitf_dict()
    assert result["ISORCE"] == "ISORCE_1"
    assert result["IID2"] == "IID2_1"

def test_sicdreader_get_nitf_dict_multiple_img_headers(monkeypatch):
    # Only the first img_headers[0] should be used for ISORCE and IID2
    class DummySecurity:
        CLAS = "U"
        CODE = "US"
        def __getattr__(self, name):
            return "VAL"

    class DummyNITFHeader:
        Security = DummySecurity()
        OSTAID = "OSTAID_VAL"
        FTITLE = "FTITLE_VAL"

    class DummyImgHeader:
        def __init__(self, isorce, iid2):
            self.ISORCE = isorce
            self.IID2 = iid2

    class DummyNitfDetails:
        nitf_header = DummyNITFHeader()
        img_headers = [
            DummyImgHeader("ISORCE_1", "IID2_1"),
            DummyImgHeader("ISORCE_2", "IID2_2"),
        ]

    class DummySICDReader(SICDReader):
        def __init__(self):
            self._nitf_details = DummyNitfDetails()
            self._sicd_meta = None

        @property
        def nitf_details(self):
            return self._nitf_details

    reader = DummySICDReader()
    result = reader.get_nitf_dict()
    assert result["ISORCE"] == "ISORCE_1"
    assert result["IID2"] == "IID2_1"

    def test_populate_nitf_information_into_sicd(monkeypatch):
        # Prepare dummy SICDMeta and nitf_details
        class DummySICDMeta:
            def __init__(self):
                self.NITF = {}

        class DummySecurity:
            CLAS = "U"
            CODE = "US"
            def __getattr__(self, name):
                return "VAL"

        class DummyNITFHeader:
            Security = DummySecurity()
            OSTAID = "OSTAID_VAL"
            FTITLE = "FTITLE_VAL"

        class DummyImgHeader:
            ISORCE = "ISORCE_VAL"
            IID2 = "IID2_VAL"

        class DummyNitfDetails:
            nitf_header = DummyNITFHeader()
            img_headers = [DummyImgHeader()]

        class DummySICDReader(SICDReader):
            def __init__(self):
                self._nitf_details = DummyNitfDetails()
                self._sicd_meta = DummySICDMeta()

            @property
            def nitf_details(self):
                return self._nitf_details

            @property
            def sicd_meta(self):
                return self._sicd_meta

        reader = DummySICDReader()
        reader.populate_nitf_information_into_sicd()
        # Check that NITF dict is populated correctly
        assert "Security" in reader.sicd_meta.NITF
        assert reader.sicd_meta.NITF["OSTAID"] == "OSTAID_VAL"
        assert reader.sicd_meta.NITF["FTITLE"] == "FTITLE_VAL"
        assert reader.sicd_meta.NITF["ISORCE"] == "ISORCE_VAL"
        assert reader.sicd_meta.NITF["IID2"] == "IID2_VAL"

def test_populate_nitf_information_into_sicd_overwrites_existing(monkeypatch):
    # If NITF already has values, they should be overwritten
    class DummySICDMeta:
        def __init__(self):
            self.NITF = {"OSTAID": "OLD", "FTITLE": "OLD", "ISORCE": "OLD", "IID2": "OLD"}

    class DummySecurity:
        CLAS = "U"
        CODE = "US"
        def __getattr__(self, name):
            return "VAL"

    class DummyNITFHeader:
        Security = DummySecurity()
        OSTAID = "NEW_OSTAID"
        FTITLE = "NEW_FTITLE"

    class DummyImgHeader:
        ISORCE = "NEW_ISORCE"
        IID2 = "NEW_IID2"

    class DummyNitfDetails:
        nitf_header = DummyNITFHeader()
        img_headers = [DummyImgHeader()]

    class DummySICDReader(SICDReader):
        def __init__(self):
            self._nitf_details = DummyNitfDetails()
            self._sicd_meta = DummySICDMeta()

        @property
        def nitf_details(self):
            return self._nitf_details

        @property
        def sicd_meta(self):
            return self._sicd_meta

    reader = DummySICDReader()
    reader.populate_nitf_information_into_sicd()
    assert reader.sicd_meta.NITF["OSTAID"] == "NEW_OSTAID"
    assert reader.sicd_meta.NITF["FTITLE"] == "NEW_FTITLE"
    assert reader.sicd_meta.NITF["ISORCE"] == "NEW_ISORCE"
    assert reader.sicd_meta.NITF["IID2"] == "NEW_IID2"

def test_sicdreader_depopulate_nitf_information_clears_nitf(monkeypatch):
    # Dummy SICDMeta with NITF dict
    class DummySICDMeta:
        def __init__(self):
            self.NITF = {"Security": {"CLAS": "U"}, "OSTAID": "OSTAID_VAL"}

    class DummyNitfDetails:
        nitf_header = None
        img_headers = []

    class DummySICDReader(SICDReader):
        def __init__(self):
            self._sicd_meta = DummySICDMeta()
            self._nitf_details = DummyNitfDetails()

        @property
        def sicd_meta(self):
            return self._sicd_meta

        @property
        def nitf_details(self):
            return self._nitf_details

    reader = DummySICDReader()
    # Ensure NITF dict is initially populated
    assert reader.sicd_meta.NITF != {}
    reader.depopulate_nitf_information()
    assert reader.sicd_meta.NITF == {}

def test_sicdreader_depopulate_nitf_information_idempotent(monkeypatch):
    # Dummy SICDMeta with empty NITF dict
    class DummySICDMeta:
        def __init__(self):
            self.NITF = {}

    class DummyNitfDetails:
        nitf_header = None
        img_headers = []

    class DummySICDReader(SICDReader):
        def __init__(self):
            self._sicd_meta = DummySICDMeta()
            self._nitf_details = DummyNitfDetails()

        @property
        def sicd_meta(self):
            return self._sicd_meta

        @property
        def nitf_details(self):
            return self._nitf_details

    reader = DummySICDReader()
    # NITF dict is already empty
    assert reader.sicd_meta.NITF == {}
    reader.depopulate_nitf_information()
    assert reader.sicd_meta.NITF == {}

def test_sicdreader_get_format_function_amp8i_phs8i_pixeltype_missing_amptable():
    # Dummy SICDMeta with PixelType 'AMP8I_PHS8I' but AmpTable is None
    class DummyImageData:
        def __init__(self):
            self.PixelType = 'AMP8I_PHS8I'
            self.AmpTable = None

    class DummySICDMeta:
        def __init__(self):
            self.ImageData = DummyImageData()

    class DummyReader(SICDReader):
        def __init__(self, sicd_meta):
            self._sicd_meta = sicd_meta

        @property
        def sicd_meta(self):
            return self._sicd_meta

    reader = DummyReader(DummySICDMeta())
    # Should raise ValueError('Expected AMP8I_PHS8I')
    with pytest.raises(ValueError, match="Expected AMP8I_PHS8I"):
        reader.get_format_function(np.dtype('uint8'), complex_order='MP', band_dimension=2)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_check_image_segment_for_compliance_true(monkeypatch):
    reader = SICDReader(sicd_files[0])
    reader.nitf_details.img_headers[0].NBPP = 9
    result = SICDReader._check_image_segment_for_compliance(reader, 0, reader.nitf_details.img_headers[0])
    assert result is False

class SicdTest(TestCase):
    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def test_check_image_segment_for_compliance_bad_complex_order(self):
        reader = SICDReader(sicd_files[0])
        reader.nitf_details.img_headers[0].Bands[1].ISUBCAT = 'Z'
        with self.assertLogs() as captured:
            result = SICDReader._check_image_segment_for_compliance(reader, 0, 
                                                                    reader.nitf_details.img_headers[0])
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), 
                         "Image segment at index 0 is not of appropriate type for a SICD Image Segment")
        self.assertFalse(result)

    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def test_check_image_segment_for_pixel_type_32(self):
        reader = SICDReader(sicd_files[0])
        reader.nitf_details.img_headers[0].Bands[0].ISUBCAT = 'M'
        reader.nitf_details.img_headers[0].Bands[1].ISUBCAT = 'P'
        with self.assertLogs() as captured:
            result = SICDReader._check_image_segment_for_compliance(reader, 0, 
                                                                    reader.nitf_details.img_headers[0])
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), 
                         "Image segment at index 0 required to be compatible\n\t"
                         "with PIXEL_TYPE RE32F_IM32F")
        self.assertFalse(result)

    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def test_check_image_segment_for_pixel_type_16(self):
        reader = SICDReader(sicd_files[0])
        reader.sicd_meta.ImageData.PixelType = 'RE16I_IM16I'
        reader.nitf_details.img_headers[0].Bands[0].ISUBCAT = 'M'
        reader.nitf_details.img_headers[0].Bands[1].ISUBCAT = 'P'
        with self.assertLogs() as captured:
            result = SICDReader._check_image_segment_for_compliance(reader, 0, 
                                                                    reader.nitf_details.img_headers[0])
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), 
                         "Image segment at index 0 required to be compatible\n\t"
                         "with PIXEL_TYPE RE16I_IM16I")
        self.assertFalse(result)

    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def test_check_image_segment_for_pixel_type_u8(self):
        reader = SICDReader(sicd_files[0])
        reader.sicd_meta.ImageData.PixelType = 'AMP8I_PHS8I'
        reader.nitf_details.img_headers[0].Bands[0].ISUBCAT = 'I'
        reader.nitf_details.img_headers[0].Bands[1].ISUBCAT = 'Q'
        with self.assertLogs() as captured:
            result = SICDReader._check_image_segment_for_compliance(reader, 0, 
                                                                    reader.nitf_details.img_headers[0])
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), 
                         "Image segment at index 0 required to be compatible\n\t"
                         "with PIXEL_TYPE AMP8I_PHS8I")
        self.assertFalse(result)

def test_is_a_none():
    result = is_a("complex_file_types.json")
    assert result is None

def test_validate_sicd_for_writing_not_sicdtype():
    with pytest.raises(ValueError, 
                       match="sicd_meta is required to be an instance of SICDType, got <class 'int'>"):
        validate_sicd_for_writing(1234)

def test_validate_sicd_for_writing_imagedata_numcols_none():
    profile = '{} {}'.format(__title__, __version__)
    # use naive datetime because numpy warns about parsing timezone aware
    # now = np.datetime64(datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None))
    sicd_meta_data.ImageCreation = ImageCreationType(
            Application=profile,
            DateTime=None,
            Profile=profile)
    validate_sicd_for_writing(sicd_meta_data)

def test_extract_clas_no_collectioninfo():
    assert extract_clas(sicd_meta_data) == 'U'

def test_extract_clas_empty_collectioninfo():
    localCollectionInfo = CollectionInfoType()
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'U'

def test_extract_clas_collectioninfo_classification_unclass():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'UNCLASS'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'U'

def test_extract_clas_collectioninfo_classification_u():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'U'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'U'

def test_extract_clas_collectioninfo_classification_confidential():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'CONFIDENTIAL'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'C'

def test_extract_clas_collectioninfo_classification_c1():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'C'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'C'

def test_extract_clas_collectioninfo_classification_c2():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'C/'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'C'

def test_extract_clas_collectioninfo_classification_top():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'TOP SECRET'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'T'

def test_extract_clas_collectioninfo_classification_ts1():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'TS'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'T'

def test_extract_clas_collectioninfo_classification_ts2():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'TS/'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'T'

def test_extract_clas_collectioninfo_classification_secret():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'SECRET'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'S'

def test_extract_clas_collectioninfo_classification_s1():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'S'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'S'

def test_extract_clas_collectioninfo_classification_s2():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'S/'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'S'

def test_extract_clas_collectioninfo_classification_fouo():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'FOUO'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'R'

def test_extract_clas_collectioninfo_classification_restricted():
    localCollectionInfo = CollectionInfoType()
    localCollectionInfo.Classification = 'RESTRICTED'
    localsicd_meta_data = sicd_meta_data
    localsicd_meta_data.CollectionInfo = localCollectionInfo
    assert extract_clas(localsicd_meta_data) == 'R'

class ClassificaitonTest(TestCase):
    def test_extract_clas_collectioninfo_classification_restricted(self):
        localCollectionInfo = CollectionInfoType()
        localCollectionInfo.Classification = 'ZONED'
        localsicd_meta_data = sicd_meta_data
        localsicd_meta_data.CollectionInfo = localCollectionInfo
        with self.assertLogs() as captured:
            assert extract_clas(localsicd_meta_data) == 'U'
            self.assertEqual(len(captured.records), 1)
            self.assertEqual(captured.records[0].getMessage(), 
                            'Unclear how to extract CLAS for classification string ZONED.\n\t'
                            'Should be set appropriately.')
        
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_security_tags_from_sicd_get_basic_args_fld_in_sec_tags(monkeypatch):
    """
    Test create_security_tags_from_sicd's get_basic_args when fld in sec_tags is True.
    """
    reader = SICDReader(sicd_files[0])
    sicd_meta = reader.sicd_meta
    tags = create_security_tags_from_sicd(sicd_meta)
    # All fields from NITFSecurityTags._ordering should be present and match
    for fld in tags._ordering:
        assert fld in NITFSecurityTags._ordering