import unittest

import numpy

from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.data_segment import NumpyArraySegment, SubsetSegment, \
    BandAggregateSegment, BlockAggregateSegment, FileReadDataSegment
from io import BytesIO


class TestNumpyArraySegment(unittest.TestCase):
    def test_basic_read(self):
        data = numpy.reshape(numpy.arange(24, dtype='int16'), (3, 4, 2))
        complex_data = numpy.empty((3, 4), dtype='complex64')
        complex_data.real = data[:, :, 0]
        complex_data.imag = data[:, :, 1]

        data_segment = NumpyArraySegment(
            data, formatted_dtype='complex64', formatted_shape=(3, 4),
            format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2),
            mode='r')

        with self.subTest(msg='read_raw full'):
            test_data = data_segment.read_raw(None)
            self.assertTrue(numpy.all(data == test_data))

        with self.subTest(msg='read_raw subscript'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment.read_raw(subscript)
            self.assertTrue(numpy.all(data[subscript] == test_data))

        with self.subTest(msg='read_raw index with squeeze'):
            test_data = data_segment.read_raw((0, 1, 1), squeeze=True)
            self.assertTrue(test_data.ndim == 0, msg='{}'.format(test_data))
            self.assertTrue(data[0, 1, 1] == test_data)

        with self.subTest(msg='read_raw index without squeeze'):
            test_data = data_segment.read_raw((0, 1, 1), squeeze=False)
            self.assertTrue(test_data.ndim == 3)
            self.assertTrue(data[0, 1, 1] == test_data)

        with self.subTest(msg='read full'):
            test_data = data_segment.read(None)
            self.assertTrue(numpy.all(complex_data == test_data))

        with self.subTest(msg='read subscript'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment.read(subscript)
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

        with self.subTest(msg='read subscript with ellipsis'):
            subscript = (..., slice(1, 3, 1))
            test_data = data_segment.read(subscript)
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

        with self.subTest(msg='read index with squeeze'):
            test_data = data_segment.read((0, 1), squeeze=True)
            self.assertTrue(test_data.ndim == 0)
            self.assertTrue(complex_data[0, 1] == test_data)

        with self.subTest(msg='read index without squeeze'):
            test_data = data_segment.read((0, 1), squeeze=False)
            self.assertTrue(test_data.ndim == 2)
            self.assertTrue(complex_data[0, 1] == test_data)

        with self.subTest(msg='read using __getitem__ subscript'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment[0:2, 1:3]
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

        with self.subTest(msg='read using __getitem__ and specifiying raw'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment[0:2, 1:3, 'raw']
            self.assertTrue(numpy.all(data[subscript] == test_data))

        with self.assertRaises(ValueError, msg='write_raw attempt'):
            data_segment.write_raw(data, start_indices=0)

        with self.assertRaises(ValueError, msg='write attempt'):
            data_segment.write(complex_data, start_indices=0)

        with self.subTest(msg='close functionality test'):
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='read access when closed'):
            _ = data_segment.read(None)

        with self.assertRaises(ValueError, msg='read_raw access when closed'):
            _ = data_segment.read_raw(None)

    def test_read_with_symmetry(self):
        data = numpy.reshape(numpy.arange(24, dtype='int16'), (3, 4, 2))

        complex_data = numpy.empty((3, 4), dtype='complex64')
        complex_data.real = data[:, :, 0]
        complex_data.imag = data[:, :, 1]
        complex_data = numpy.transpose(complex_data)

        data_segment = NumpyArraySegment(
            data, formatted_dtype='complex64', formatted_shape=(4, 3),
            transpose_axes=(1, 0, 2),
            format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2),
            mode='r')

        with self.subTest(msg='read full'):
            test_data = data_segment.read(None)
            self.assertTrue(numpy.all(complex_data == test_data))

        with self.subTest(msg='read subscript'):
            subscript = (slice(1, 3, 1), slice(0, 2, 1))
            test_data = data_segment.read(subscript)
            self.assertTrue(numpy.all(test_data == complex_data[subscript]))

    def test_basic_write(self):
        data = numpy.reshape(numpy.arange(24, dtype='int16'), (3, 4, 2))
        complex_data = numpy.empty((3, 4), dtype='complex64')
        complex_data.real = data[:, :, 0]
        complex_data.imag = data[:, :, 1]

        with self.subTest(msg='write_raw'):
            empty = numpy.empty((3, 4, 2), dtype='int16')
            data_segment = NumpyArraySegment(
                empty, formatted_dtype='complex64', formatted_shape=(3, 4),
                format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2),
                mode='w')

            data_segment.write_raw(data, start_indices=0)
            self.assertTrue(numpy.all(empty == data))

        with self.subTest(msg='write'):
            empty = numpy.empty((3, 4, 2), dtype='int16')
            data_segment = NumpyArraySegment(
                empty, formatted_dtype='complex64', formatted_shape=(3, 4),
                format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2),
                mode='w')

            data_segment.write(complex_data, start_indices=0)
            self.assertTrue(numpy.all(empty == data))

        with self.assertRaises(ValueError, msg='read_raw attempt'):
            _ = data_segment.read_raw(0)

        with self.assertRaises(ValueError, msg='read attempt'):
            _ = data_segment.read(0)

        with self.subTest(msg='close'):
            empty = numpy.empty((3, 4, 2), dtype='int16')
            data_segment = NumpyArraySegment(
                empty, formatted_dtype='complex64', formatted_shape=(3, 4),
                format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2),
                mode='w')
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='write access when closed'):
            data_segment.write(complex_data)

        with self.assertRaises(ValueError, msg='write_raw access when closed'):
            data_segment.write(complex_data)


class TestSubsetSegment(unittest.TestCase):
    def test_read(self):
        data = numpy.reshape(numpy.arange(24, dtype='int16'), (6, 4))

        subset_def = (slice(2, 5, 1), slice(2, 4, 1))
        parent_segment = NumpyArraySegment(data, mode='r')
        data_segment = SubsetSegment(parent_segment, subset_def, 'raw')

        with self.subTest(msg='full read'):
            test_data = data_segment[:]
            self.assertTrue(numpy.all(data[subset_def] == test_data))

        with self.subTest(msg='partial read'):
            test_data = data_segment[1:3]
            self.assertTrue(numpy.all(data[(slice(3, 5, 1), slice(2, 4, 1))] == test_data))

        with self.subTest(msg='close functionality test'):
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='read_raw access when closed'):
            _ = data_segment.read_raw(None)

        with self.assertRaises(ValueError, msg='read access when closed'):
            _ = data_segment.read(None)

    def test_write(self):
        data = numpy.zeros((6, 4), dtype='int16')

        subset_def = (slice(2, 5, 1), slice(2, 4, 1))
        parent_segment = NumpyArraySegment(data, mode='w')
        data_segment = SubsetSegment(parent_segment, subset_def, 'raw')

        test_data = numpy.reshape(numpy.arange(6, dtype='int16'), (3, 2))

        with self.subTest(msg='subset write'):
            data_segment.write(test_data, start_indices=0)
            self.assertTrue(numpy.all(data[subset_def] == test_data))

        with self.subTest(msg='close functionality test'):
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='write access when closed'):
            data_segment.write(test_data)

        with self.assertRaises(ValueError, msg='write_raw access when closed'):
            data_segment.write(test_data)


class TestBandAggregateSegment(unittest.TestCase):
    def test_read(self):
        data0 = numpy.reshape(numpy.arange(12, dtype='uint8'), (3, 4))
        data1 = numpy.reshape(numpy.arange(12, 24, dtype='uint8'), (3, 4))
        ds0 = NumpyArraySegment(data0, mode='r')
        ds1 = NumpyArraySegment(data1, mode='r')
        data_segment = BandAggregateSegment((ds0, ds1), 2)

        with self.subTest(msg='direct band comparison'):
            test_data = data_segment[:]
            self.assertTrue(numpy.all(test_data[..., 0] == data0))
            self.assertTrue(numpy.all(test_data[..., 1] == data1))

        with self.subTest(msg='reading band comparison'):
            self.assertTrue(numpy.all(data_segment[..., 0] == data0))
            self.assertTrue(numpy.all(data_segment[..., 1] == data1))

        with self.subTest(msg='section reading'):
            subset = (slice(2, 3, 1), slice(1, 3, 1))
            test_data = data_segment.read(subset)
            self.assertTrue(numpy.all(test_data[..., 0] == data0[subset]))
            self.assertTrue(numpy.all(test_data[..., 1] == data1[subset]))

        with self.subTest(msg='close functionality test'):
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='read_raw access when closed'):
            _ = data_segment.read_raw(None)

        with self.assertRaises(ValueError, msg='read access when closed'):
            _ = data_segment.read(None)

    def test_write(self):
        data0 = numpy.empty((3, 4), dtype='uint16')
        data1 = numpy.empty((3, 4), dtype='uint16')
        ds0 = NumpyArraySegment(data0, mode='w')
        ds1 = NumpyArraySegment(data1, mode='w')
        data_segment = BandAggregateSegment((ds0, ds1), 2)

        with self.subTest(msg='writing check'):
            test_data = numpy.reshape(numpy.arange(24, dtype='uint16'), (3, 4, 2))
            data_segment.write(test_data)
            self.assertTrue(numpy.all(test_data[..., 0] == data0))
            self.assertTrue(numpy.all(test_data[..., 1] == data1))

        with self.subTest(msg='close functionality test'):
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='write access when closed'):
            data_segment.write(test_data)

        with self.assertRaises(ValueError, msg='write_raw access when closed'):
            data_segment.write(test_data)


class TestBlockAggregateSegment(unittest.TestCase):
    def test_read(self):
        data0 = numpy.reshape(numpy.arange(6, dtype='int16'), (3, 2))
        data1 = numpy.reshape(numpy.arange(6, 12, dtype='int16'), (3, 2))
        ds0 = NumpyArraySegment(data0, mode='r')
        ds1 = NumpyArraySegment(data1, mode='r')
        data_segment = BlockAggregateSegment(
            (ds0, ds1), (
                (slice(0, 3, 1), slice(0, 2, 1)),
                (slice(0, 3, 1), slice(2, 4, 1))),
            'raw', 0, (3, 4), 'int16', (3, 4))

        with self.subTest(msg='read'):
            test_data = data_segment[:]
            self.assertTrue(numpy.all(data0 == test_data[:, :2]))
            self.assertTrue(numpy.all(data1 == test_data[:, 2:]))

        with self.subTest(msg='close functionality test'):
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='read_raw access when closed'):
            _ = data_segment.read_raw(None)

        with self.assertRaises(ValueError, msg='read access when closed'):
            _ = data_segment.read(None)

    def test_write(self):
        data0 = numpy.empty((3, 2), dtype='int16')
        data1 = numpy.empty((3, 2), dtype='int16')
        ds0 = NumpyArraySegment(data0, mode='w')
        ds1 = NumpyArraySegment(data1, mode='w')
        data_segment = BlockAggregateSegment(
            (ds0, ds1), (
                (slice(0, 3, 1), slice(0, 2, 1)),
                (slice(0, 3, 1), slice(2, 4, 1))),
            'raw', 0, (3, 4), 'int16', (3, 4))

        test_data = numpy.reshape(numpy.arange(12, dtype='int16'), (3, 4))
        with self.subTest(msg='write'):
            data_segment.write(test_data, start_indices=0)
            self.assertTrue(numpy.all(test_data[:, :2] == data0))
            self.assertTrue(numpy.all(test_data[:, 2:] == data1))

        with self.subTest(msg='close functionality test'):
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='write access when closed'):
            data_segment.write(test_data)

        with self.assertRaises(ValueError, msg='write_raw access when closed'):
            data_segment.write(test_data)


class TestFileReadSegment(unittest.TestCase):
    def test_read(self):
        data = numpy.reshape(numpy.arange(24, dtype='int16'), (3, 4, 2))
        complex_data = numpy.empty((3, 4), dtype='complex64')
        complex_data.real = data[:, :, 0]
        complex_data.imag = data[:, :, 1]

        file_object = BytesIO(data.tobytes())
        data_segment = FileReadDataSegment(
            file_object, 0, 'int16', (3, 4, 2), 'complex64', (3, 4),
            format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2))

        with self.subTest(msg='read_raw full'):
            test_data = data_segment.read_raw(None)
            self.assertTrue(numpy.all(data == test_data))

        with self.subTest(msg='read_raw subscript'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment.read_raw(subscript)
            self.assertTrue(numpy.all(data[subscript] == test_data))

        with self.subTest(msg='read_raw index with squeeze'):
            test_data = data_segment.read_raw((0, 1, 1), squeeze=True)
            self.assertTrue(test_data.ndim == 0, msg='{}'.format(test_data))
            self.assertTrue(data[0, 1, 1] == test_data)

        with self.subTest(msg='read_raw index without squeeze'):
            test_data = data_segment.read_raw((0, 1, 1), squeeze=False)
            self.assertTrue(test_data.ndim == 3)
            self.assertTrue(data[0, 1, 1] == test_data)

        with self.subTest(msg='read full'):
            test_data = data_segment.read(None)
            self.assertTrue(numpy.all(complex_data == test_data))

        with self.subTest(msg='read subscript'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment.read(subscript)
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

        with self.subTest(msg='read subscript with ellipsis'):
            subscript = (..., slice(1, 3, 1))
            test_data = data_segment.read(subscript)
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

        with self.subTest(msg='read index with squeeze'):
            test_data = data_segment.read((0, 1), squeeze=True)
            self.assertTrue(test_data.ndim == 0)
            self.assertTrue(complex_data[0, 1] == test_data)

        with self.subTest(msg='read index without squeeze'):
            test_data = data_segment.read((0, 1), squeeze=False)
            self.assertTrue(test_data.ndim == 2)
            self.assertTrue(complex_data[0, 1] == test_data)

        with self.subTest(msg='read using __getitem__ subscript'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment[0:2, 1:3]
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

        with self.subTest(msg='read using __getitem__ and specifiying raw'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment[0:2, 1:3, 'raw']
            self.assertTrue(numpy.all(data[subscript] == test_data))

        with self.assertRaises(ValueError, msg='write_raw attempt'):
            data_segment.write_raw(data, start_indices=0)

        with self.assertRaises(ValueError, msg='write attempt'):
            data_segment.write(complex_data, start_indices=0)

        with self.subTest(msg='close functionality test'):
            self.assertFalse(data_segment.closed)
            data_segment.close()
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='read access when closed'):
            _ = data_segment.read(None)

        with self.assertRaises(ValueError, msg='read_raw access when closed'):
            _ = data_segment.read_raw(None)
