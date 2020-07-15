import os
import time
import logging
import numpy

from . import unittest

from sarpy.io.complex.sentinel import SentinelReader
from sarpy.deprecated.io.complex.sentinel import Reader as DepReader


def generic_sen_test(instance, test_root, test_file):
    assert isinstance(instance, unittest.TestCase)

    with instance.subTest(msg='establish sen reader for file {}'.format(test_root)):
        reader = SentinelReader(test_root)

    with instance.subTest(msg='Verify is_sicd_type for file {}'.format(test_root)):
        instance.assertTrue(reader.is_sicd_type, msg='Verifying is_sicd_type is True')

    with instance.subTest(msg='Verify image size between reader and sicd for file {}'.format(test_root)):
        data_sizes = reader.get_data_size_as_tuple()
        sicds = reader.get_sicds_as_tuple()
        instance.assertEqual(data_sizes[0][0], sicds[0].ImageData.NumRows, msg='data_size[0] and NumRows do not agree')
        instance.assertEqual(data_sizes[0][1], sicds[0].ImageData.NumCols, msg='data_size[1] and NumCols do not agree')

    with instance.subTest(msg='establish deprecated sen reader for file {}'.format(test_root)):
        dep_reader = DepReader(test_file)

    for i in range(len(reader.sicd_meta)):
        data_new = reader[:5, :5, i]
        data_dep = dep_reader.read_chip[i](numpy.array((0, 5, 1), dtype=numpy.int64), numpy.array((0, 5, 1), dtype=numpy.int64))
        comp = numpy.abs(data_new - data_dep)
        same = numpy.all(comp < 1e-10)
        with instance.subTest(msg='sen fetch test for file {}'.format(test_root)):
            instance.assertTrue(same, msg='index {} fetch comparison failed'.format(i))
            if not same:
                logging.error('index = {}\nnew data = {}\nold data = {}'.format(i, data_new, data_dep))


class TestSentinelReader(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.sen_root = os.path.expanduser('~/Desktop/sarpy_testing/sentinel/')

    def test_reader(self):
        tested = 0
        for fil in [
                'S1A_IW_SLC__1SDH_20200201T141253_20200201T141321_031060_03917E_BAA7.SAFE', ]:
            test_root = os.path.join(self.sen_root, fil)
            test_file = os.path.join(test_root, 'manifest.safe')

            if os.path.exists(test_file):
                tested += 1
                generic_sen_test(self, test_root, test_file)
            else:
                logging.info('No file {} found'.format(test_file))

        self.assertTrue(tested > 0, msg="No files for testing found")
