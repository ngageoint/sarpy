import unittest

import numpy

from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.data_segment import NumpyArraySegment
from sarpy.io.general.base import BaseReader, BaseWriter


class TestBaseReader(unittest.TestCase):
    def test_read(self):
        data = numpy.reshape(numpy.arange(24, dtype='int16'), (3, 4, 2))
        complex_data = numpy.empty((3, 4), dtype='complex64')
        complex_data.real = data[:, :, 0]
        complex_data.imag = data[:, :, 1]

        data_segment = NumpyArraySegment(
            data, formatted_dtype='complex64', formatted_shape=(3, 4),
            format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2),
            mode='r')

        with self.subTest(msg='blank initialization'):
            reader = BaseReader(None)

        with self.subTest(msg='reinitialization with segment'):
            BaseReader.__init__(reader, data_segment)

        with self.assertRaises(ValueError, msg='repeated initialization with segment'):
            BaseReader.__init__(reader, data_segment)

        with self.subTest(msg='full read'):
            test_data = reader.read(index=0)
            self.assertTrue(numpy.all(complex_data == test_data))

        with self.subTest(msg='full read_raw'):
            test_data = reader.read_raw(index=0)
            self.assertTrue(numpy.all(data == test_data))

        with self.subTest(msg='full __getitem__ read'):
            test_data = reader[:]
            self.assertTrue(numpy.all(complex_data == test_data))

        with self.subTest(msg='full __getitem__ raw read'):
            test_data = reader[:, 'raw']
            self.assertTrue(numpy.all(data == test_data))

        with self.subTest(msg='partial read'):
            subscript = (slice(1, 2, 1), slice(2, 4, 1))
            test_data = reader.read(*subscript)
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

        with self.subTest(msg='partial __getitem__ read'):
            subscript = (slice(1, 2, 1), slice(2, 4, 1))
            test_data = reader[1:2, 2:4]
            self.assertTrue(numpy.all(complex_data[subscript] == test_data))

        with self.subTest(msg='partial raw read'):
            subscript = (slice(1, 2, 1), slice(2, 4, 1))
            test_data = reader.read_raw(*subscript)
            self.assertTrue(numpy.all(data[subscript] == test_data))

        with self.subTest(msg='partial __getitem__ raw read'):
            subscript = (slice(1, 2, 1), slice(2, 4, 1))
            test_data = reader[1:2, 2:4, 'raw']
            self.assertTrue(numpy.all(data[subscript] == test_data))

        with self.subTest(msg='close functionality test'):
            self.assertFalse(reader.closed)
            the_segment = reader.data_segment  # NB: it will be flushed from the reader
            self.assertFalse(the_segment.closed)
            reader.close()
            self.assertTrue(reader.closed)
            self.assertTrue(reader.data_segment is None)
            self.assertTrue(the_segment.closed)

        with self.assertRaises(ValueError, msg='read access when closed'):
            _ = reader.read()

        with self.assertRaises(ValueError, msg='read_raw access when closed'):
            _ = reader.read_raw()

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
        reader = BaseReader(data_segment)

        with self.subTest(msg='read full'):
            test_data = reader.read()
            self.assertTrue(numpy.all(complex_data == test_data))

        with self.subTest(msg='read subscript'):
            subscript = (slice(1, 3, 1), slice(0, 2, 1))
            test_data = reader.read(*subscript)
            self.assertTrue(numpy.all(test_data == complex_data[subscript]))


class TestBaseWriter(unittest.TestCase):
    def test_write(self):
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
            with BaseWriter(data_segment) as writer:
                writer.write_raw(data, start_indices=0)
            self.assertTrue(numpy.all(empty == data))

        with self.subTest(msg='write'):
            empty = numpy.empty((3, 4, 2), dtype='int16')
            data_segment = NumpyArraySegment(
                empty, formatted_dtype='complex64', formatted_shape=(3, 4),
                format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2),
                mode='w')
            with BaseWriter(data_segment) as writer:
                writer.write(complex_data, start_indices=0)
            self.assertTrue(numpy.all(empty == data))

        with self.subTest(msg='close'):
            empty = numpy.empty((3, 4, 2), dtype='int16')
            data_segment = NumpyArraySegment(
                empty, formatted_dtype='complex64', formatted_shape=(3, 4),
                format_function=ComplexFormatFunction('int16', 'IQ', band_dimension=2),
                mode='w')
            writer = BaseWriter(data_segment)

            self.assertFalse(writer.closed)
            self.assertFalse(writer.data_segment[0].closed)
            writer.close()
            self.assertTrue(writer.closed)
            self.assertTrue(writer.data_segment is None)
            self.assertTrue(data_segment.closed)

        with self.assertRaises(ValueError, msg='write access when closed'):
            data_segment.write(complex_data)

        with self.assertRaises(ValueError, msg='write_raw access when closed'):
            data_segment.write(complex_data)
