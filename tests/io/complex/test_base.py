# Written by: Tex Peterson
# Written on: 2026-01-06
# Purpose: The sarpy/io/complex/base.py file is tested in multiple other test
#           files. This file adds tests to cover the lines of code that are not
#           tested elsewhere.

import numpy
import pytest
from sarpy.io.complex.base import SICDTypeReader, FlatSICDReader, SubsetSICDReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.ImageData  import ImageDataType, FullImageType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType
from sarpy.io.complex.sicd_elements.blocks     import RowColType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.general.data_segment import NumpyArraySegment

# Mock BaseReader to avoid side effects
class DummyBaseReader:
    def __init__(self, *args, **kwargs):
        pass

class DummySICD(SICDType):
    def __init__(self, val):
        self.val = val
    def __eq__(self, other):
        return self.val == other.val

def always_false_match(a, b):
    return False

def always_true_match(a, b):
    return True

# Should not raise, _check_sizes will pass since sizes match
class DummyFlatReader(DummyBaseReader):
    def __init__(self, *args, **kwargs): pass
    
image_data = numpy.arange(13*17, dtype=numpy.complex64).reshape(13, 17)
TimelineType_obj = TimelineType()
CollectionInfo_obj= CollectionInfoType()
sicd_meta = SICDType(
        ImageData=ImageDataType(
            NumRows=image_data.shape[0],
            NumCols=image_data.shape[1],
            PixelType="RE32F_IM32F",
            FirstRow=0,
            FirstCol=0,
            FullImage=FullImageType(
                NumRows=image_data.shape[0],
                NumCols=image_data.shape[1]
            ),
            SCPPixel=RowColType(Row=image_data.shape[0], 
                                Col=image_data.shape[1])
        ),
        Timeline=TimelineType_obj,
        CollectionInfo=CollectionInfo_obj
    )


def test_sicdtypereader_init_sicd_meta_none(monkeypatch):
    monkeypatch.setattr("sarpy.io.general.base.BaseReader", DummyBaseReader)
    # Should not raise, _sicd_meta should be None
    reader = SICDTypeReader(data_segment=None, sicd_meta=None)
    assert reader._sicd_meta is None
    assert reader.sicd_meta is None

def test_sicdtypereader_init_sicd_meta_list_with_non_sicdtype(monkeypatch):
    monkeypatch.setattr("sarpy.io.general.base.BaseReader", DummyBaseReader)

    # Use a dummy object that is not SICDType
    not_sicd = object()
    with pytest.raises(TypeError) as excinfo:
        SICDTypeReader(data_segment=None, sicd_meta=[not_sicd])
    assert "all elements are required to be instances of SICDType" in str(excinfo.value)


def test_sicdtypereader_check_sizes_data_size_mismatch(monkeypatch):
    monkeypatch.setattr("sarpy.io.general.base.BaseReader", DummyBaseReader)

    # Create a SICDTypeReader instance
    reader = SICDTypeReader(data_segment=None, sicd_meta=sicd_meta)

    # Patch get_data_size_as_tuple and get_sicds_as_tuple to simulate mismatch
    reader.get_data_size_as_tuple = lambda: [(5, 5)]  # Wrong size
    reader.get_sicds_as_tuple = lambda: [sicd_meta]

    with pytest.raises(ValueError) as excinfo:
        reader._check_sizes()
    assert "data segment at index 0 has data size (5, 5)" in str(excinfo.value)

def test_get_sicds_as_tuple_none(monkeypatch):
    monkeypatch.setattr("sarpy.io.general.base.BaseReader", DummyBaseReader)
    reader = SICDTypeReader(data_segment=None, sicd_meta=None)
    assert reader.get_sicds_as_tuple() is None

# ***


def test_get_sicd_partitions_matched_skips(monkeypatch):
    monkeypatch.setattr("sarpy.io.general.base.BaseReader", DummyBaseReader)

    # Create a SICDTypeReader with 3 dummy SICDs
    sicds = [DummySICD(1), DummySICD(2), DummySICD(3)]
    reader = SICDTypeReader(data_segment=None, sicd_meta=sicds)

    # Patch get_sicds_as_tuple to return our dummy SICDs
    reader.get_sicds_as_tuple = lambda: tuple(sicds)

    # Patch numpy.zeros to pre-mark the first index as matched
    orig_zeros = numpy.zeros
    def fake_zeros(shape, dtype=None):
        arr = orig_zeros(shape, dtype=dtype)
        arr[0] = True  # Mark first as already matched
        return arr
    monkeypatch.setattr("numpy.zeros", fake_zeros)

    # Should skip index 0 and only process 1 and 2
    partitions = reader.get_sicd_partitions(match_function=always_false_match)
    assert (0,) not in partitions  # index 0 was skipped
    assert (1,) in partitions and (2,) in partitions


def test_get_sicd_partitions_all_matched(monkeypatch):
    monkeypatch.setattr("sarpy.io.general.base.BaseReader", DummyBaseReader)

    # Create a SICDTypeReader with 3 dummy SICDs
    sicds = [DummySICD(1), DummySICD(2), DummySICD(3)]
    reader = SICDTypeReader(data_segment=None, sicd_meta=sicds)

    # Patch get_sicds_as_tuple to return our dummy SICDs
    reader.get_sicds_as_tuple = lambda: tuple(sicds)

    # All SICDs should be matched together since match_function always returns True
    partitions = reader.get_sicd_partitions(match_function=always_true_match)
    assert partitions == ((0, 1, 2),)

def test_flat_sicd_reader(monkeypatch):
    reader = FlatSICDReader(
        sicd_meta=sicd_meta,
        underlying_array=image_data,
        formatted_dtype=numpy.float32,
        formatted_shape=image_data.shape,
        reverse_axes=None,
        transpose_axes=None,
        format_function=None,
        close_segments=True
    )
    assert reader.sicd_meta == sicd_meta
    assert reader.get_sicds_as_tuple() == (sicd_meta,)

def test_flat_sicd_reader_write_to_file(tmp_path):
    reader = FlatSICDReader(
        sicd_meta=sicd_meta,
        underlying_array=image_data,
        formatted_dtype=numpy.float32,
        formatted_shape=image_data.shape,
        reverse_axes=None,
        transpose_axes=None,
        format_function=None,
        close_segments=True
    )

    # Write to file
    out_file = tmp_path / "test_output.npy"
    reader.write_to_file(str(out_file))

    # Check file exists and contents match
    loaded = SICDReader(str(out_file))
    read_array = loaded.data_segment[:].copy()
    numpy.testing.assert_array_equal(read_array, image_data)

def test_flat_sicd_reader_write_to_file_fail(tmp_path):
    reader = FlatSICDReader(
        sicd_meta=sicd_meta,
        underlying_array=image_data,
        formatted_dtype=numpy.float32,
        formatted_shape=image_data.shape,
        reverse_axes=None,
        transpose_axes=None,
        format_function=None,
        close_segments=True
    )

    # Write to file
    out_file = 1234
    with pytest.raises(TypeError) as excinfo:
        reader.write_to_file(out_file)
    assert "output_file is expected to a be a string, got type <class 'int'>" in str(excinfo.value)
    

def test_subset_sicd_reader(monkeypatch):
    data_segment = NumpyArraySegment(underlying_array=image_data)
    reader = SICDTypeReader(data_segment=data_segment, sicd_meta=sicd_meta)
    subset_reader = SubsetSICDReader(reader, (1, 5), (2, 6), index=0, close_parent=True)
    assert isinstance(subset_reader, SubsetSICDReader)
    assert subset_reader.sicd_meta.ImageData.PixelType == sicd_meta.ImageData.PixelType
    assert subset_reader.sicd_meta.ImageData.FullImage.NumRows == sicd_meta.ImageData.FullImage.NumRows
    assert subset_reader.sicd_meta.ImageData.FullImage.NumCols == sicd_meta.ImageData.FullImage.NumCols
    assert subset_reader.sicd_meta.ImageData.SCPPixel.Row == sicd_meta.ImageData.SCPPixel.Row
    assert subset_reader.sicd_meta.ImageData.SCPPixel.Col == sicd_meta.ImageData.SCPPixel.Col
    assert isinstance(subset_reader._sicd_meta, SICDType)
    assert subset_reader.file_name is None