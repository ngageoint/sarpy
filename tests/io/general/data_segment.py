import unittest

import numpy

from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.data_segment import NumpyArraySegment, SubsetSegment, \
    BandAggregateSegment, BlockAggregateSegment


class TestNumpyArray(unittest.TestCase):
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

        with self.subTest(msg='read using __getitem__ full'):
            test_data = data_segment[:]
            self.assertTrue(numpy.all(complex_data == test_data))

        with self.subTest(msg='read using __getitem__ subscript'):
            subscript = (slice(0, 2, 1), slice(1, 3, 1))
            test_data = data_segment[0:2, 1:3]
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

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


class TestSubset(unittest.TestCase):
    pass


class TestBandAggregate(unittest.TestCase):
    pass


class TestBlockAggregate(unittest.TestCase):
    pass
