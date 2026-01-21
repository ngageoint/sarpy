#
# Copyright 2024 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import filecmp

import numpy as np
import pytest

import sarpy.io.general.nitf
from sarpy.io.general.nitf import _get_format_function, NITFReader, NITFWriter
from sarpy.io.general.format_function import ComplexFormatFunction, SingleLUTFormatFunction


def test_iq_band_interleaved_by_block(tests_path, tmp_path):
    in_nitf = tests_path / "data/iq.nitf"
    with sarpy.io.general.nitf.NITFReader(str(in_nitf)) as reader:
        data = reader.read()
        data_raw = reader.read_raw()
        assert data_raw.ndim == data.ndim + 1
        assert data_raw.size == data.size * 2
        assert np.iscomplexobj(data)

        data_offset = reader.nitf_details.img_segment_offsets[0]
        manual_bytes = in_nitf.read_bytes()[data_offset : data_offset + data.nbytes]
        manual_data_raw = np.frombuffer(manual_bytes, dtype=data_raw.dtype).reshape(
            data_raw.shape
        )
        manual_data = manual_data_raw[0] + 1j * manual_data_raw[1]  # interleaved by block
        assert np.array_equal(data, manual_data)

        out_nitf = tmp_path / "out.nitf"
        writer_details = sarpy.io.general.nitf.NITFWritingDetails(
            reader.nitf_details.nitf_header,
            (sarpy.io.general.nitf.ImageSubheaderManager(reader.get_image_header(0)),),
            reader.image_segment_collections,
        )
        with sarpy.io.general.nitf.NITFWriter(
            str(out_nitf), writing_details=writer_details
        ) as writer:
            writer.write(data)
        assert filecmp.cmp(in_nitf, out_nitf, shallow=False)


def test_write_filehandle(tests_path, tmp_path):
    in_nitf = tests_path / "data/iq.nitf"
    with sarpy.io.general.nitf.NITFReader(str(in_nitf)) as reader:
        data = reader.read()
        writer_details = sarpy.io.general.nitf.NITFWritingDetails(
            reader.nitf_details.nitf_header,
            (sarpy.io.general.nitf.ImageSubheaderManager(reader.get_image_header(0)),),
            reader.image_segment_collections,
        )

    out_nitf = tmp_path / 'output.nitf'
    with out_nitf.open('wb') as fd:
        with sarpy.io.general.nitf.NITFWriter(
            fd, writing_details=writer_details
        ) as writer:
            writer.write(data)

        assert not fd.closed
    assert filecmp.cmp(in_nitf, out_nitf, shallow=False)

def test_in_memory_write(tests_path, tmp_path):
    in_nitf_mem = tests_path / "data/iq.nitf"
    with sarpy.io.general.nitf.NITFReader(str(in_nitf_mem)) as reader_mem:
        data_mem = reader_mem.read()
        writer_details_mem = sarpy.io.general.nitf.NITFWritingDetails(
            reader_mem.nitf_details.nitf_header,
            (sarpy.io.general.nitf.ImageSubheaderManager(reader_mem.get_image_header(0)),),
            reader_mem.image_segment_collections,
        )
    
    out_nitf_mem = tmp_path / 'output_memory.nitf'
    with out_nitf_mem.open('wb') as fd_mem:
        with sarpy.io.general.nitf.NITFWriter(
            fd_mem, writing_details=writer_details_mem, in_memory=True
        ) as writer_mem:
            writer_mem.write(data_mem)

        assert not fd_mem.closed
    assert filecmp.cmp(in_nitf_mem, out_nitf_mem, shallow=False)

def test_get_format_function_none():
    # No complex_order, no lut
    dtype = np.dtype('>u2')
    result = _get_format_function(dtype, None, None, 2)
    assert result is None

def test_get_format_function_complex(monkeypatch):
    dtype = np.dtype('>i2')
    # Provide a complex_order
    result = _get_format_function(dtype, 'IQ', None, 1)
    assert isinstance(result, ComplexFormatFunction)
    assert result.order == 'IQ'
    assert result._raw_dtype == dtype
    assert result._band_dimension == 1

def test_get_format_function_lut(monkeypatch):
    dtype = np.dtype('>u1')
    lut = np.arange(256, dtype=np.uint8)
    result = _get_format_function(dtype, None, lut, 2)
    assert isinstance(result, SingleLUTFormatFunction)
    assert np.array_equal(result._lookup_table, lut)

def test_get_format_function_complex_and_lut(monkeypatch):
    dtype = np.dtype('>i2')
    lut = np.arange(256, dtype=np.uint8)
    # If both complex_order and lut are provided, complex_order takes precedence
    result = _get_format_function(dtype, 'QI', lut, 0)
    assert isinstance(result, ComplexFormatFunction)
    assert result.order == 'QI'

class DummyReader(NITFReader):
    def __init__(self):
        # Avoid calling parent __init__
        pass

@pytest.mark.parametrize(
    "raw_dtype,complex_order,lut,band_dimension,expected_type",
    [
        (np.dtype('>i2'), 'IQ', None, 2, ComplexFormatFunction),
        (np.dtype('>u1'), None, np.arange(256, dtype=np.uint8), 2, SingleLUTFormatFunction),
        (np.dtype('>f4'), None, None, 2, type(None)),
        (np.dtype('>i2'), 'IQ', np.arange(256, dtype=np.uint8), 2, ComplexFormatFunction),
    ]
)
def test__get_format_function(raw_dtype, complex_order, lut, band_dimension, expected_type):
    func = _get_format_function(raw_dtype, complex_order, lut, band_dimension)
    if expected_type is type(None):
        assert func is None
    else:
        assert isinstance(func, expected_type)

@pytest.mark.parametrize(
    "raw_dtype,complex_order,lut,band_dimension,expected_type",
    [
        (np.dtype('>i2'), 'IQ', None, 2, ComplexFormatFunction),
        (np.dtype('>u1'), None, np.arange(256, dtype=np.uint8), 2, SingleLUTFormatFunction),
        (np.dtype('>f4'), None, None, 2, type(None)),
        (np.dtype('>i2'), 'IQ', np.arange(256, dtype=np.uint8), 2, ComplexFormatFunction),
    ]
)
def test_nitfreader_get_format_function(raw_dtype, complex_order, lut, band_dimension, expected_type):
    reader = DummyReader()
    func = reader.get_format_function(raw_dtype, complex_order, lut, band_dimension)
    if expected_type is type(None):
        assert func is None
    else:
        assert isinstance(func, expected_type)

class DummyWriter(NITFWriter):
    def __init__(self):
        # Avoid calling parent __init__
        pass

@pytest.mark.parametrize(
    "raw_dtype,complex_order,lut,band_dimension,expected_type",
    [
        (np.dtype('>i2'), 'IQ', None, 2, ComplexFormatFunction),
        (np.dtype('>u1'), None, np.arange(256, dtype=np.uint8), 2, SingleLUTFormatFunction),
        (np.dtype('>f4'), None, None, 2, type(None)),
        (np.dtype('>i2'), 'IQ', np.arange(256, dtype=np.uint8), 2, ComplexFormatFunction),
    ]
)
def test_nitfwriter_get_format_function(raw_dtype, complex_order, lut, band_dimension, expected_type):
    writer = DummyWriter()
    func = writer.get_format_function(raw_dtype, complex_order, lut, band_dimension)
    if expected_type is type(None):
        assert func is None
    else:
        assert isinstance(func, expected_type)