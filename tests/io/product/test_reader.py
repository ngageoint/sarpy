import logging
import os
import json

from sarpy.io.product.converter import open_product
from sarpy.io.product.sidd import SIDDReader

from tests import unittest, parse_file_entry


product_file_types = {}

this_loc = os.path.abspath(__file__)
file_reference = os.path.join(os.path.split(this_loc)[0], 'product_file_types.json')  # specifies file locations
if os.path.isfile(file_reference):
    with open(file_reference, 'r') as fi:
        the_files = json.load(fi)
        for the_type in the_files:
            valid_entries = []
            for entry in the_files[the_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            product_file_types[the_type] = valid_entries


def generic_reader_test(instance, test_file, reader_type_string, reader_type):
    assert isinstance(instance, unittest.TestCase)

    reader = None
    with instance.subTest(msg='establish reader for type {} and file {}'.format(reader_type_string, test_file)):
        reader = open_product(test_file)
        instance.assertTrue(reader is not None, msg='Returned None, so opening failed.')

    if reader is None:
        return  # remaining tests make no sense

    assert isinstance(reader, SIDDReader)
    with instance.subTest(msg='Reader for type {} should be appropriate reader'):
        instance.assertTrue(isinstance(reader, reader_type), msg='Returned reader should be of type {}'.format(reader_type))

    if not isinstance(reader, reader_type):
        return  # remaining tests might be misleading

    with instance.subTest(msg='Verify reader_type for type {} and file {}'.format(reader_type_string, test_file)):
        instance.assertEqual(reader.reader_type, "SIDD", msg='reader.reader_type should be "SIDD"')

    with instance.subTest(msg='Fetch data_sizes and sidds for type {} and file {}'.format(reader_type_string, test_file)):
        data_sizes = reader.get_data_size_as_tuple()
        sidds = reader.sidd_meta

    for i, (data_size, sidd) in enumerate(zip(data_sizes, sidds)):
        with instance.subTest(msg='Verify image size for sidd index {} in reader '
                                  'of type {} for file {}'.format(i, reader_type_string, test_file)):
            instance.assertEqual(data_size[0], sidd.Measurement.PixelFootprint.Row, msg='data_size[0] and Row do not agree')
            instance.assertEqual(data_size[1], sidd.Measurement.PixelFootprint.Col, msg='data_size[1] and Col do not agree')

        with instance.subTest(msg='Basic fetch test for sidd index {} in reader '
                                  'of type {} for file {}'.format(i, reader_type_string, test_file)):
            instance.assertEqual(reader[:2, :2, i].shape[:2], (2, 2), msg='upper left fetch')
            instance.assertEqual(reader[-2:, :2, i].shape[:2], (2, 2), msg='lower left fetch')
            instance.assertEqual(reader[-2:, -2:, i].shape[:2], (2, 2), msg='lower right fetch')
            instance.assertEqual(reader[:2, -2:, i].shape[:2], (2, 2), msg='upper right fetch')

        with instance.subTest(msg='Verify fetching complete row(s) have correct size '
                                  'for sidd index {} in reader of type {} and file {}'.format(i, reader_type_string, test_file)):
            test_data = reader[:, :2, i]
            instance.assertEqual(test_data.shape[:2], (data_size[0], 2), msg='Complete row fetch size mismatch')

        with instance.subTest(msg='Verify fetching complete columns(s) have correct size '
                                  'for sidd index {} in reader of type {} file {}'.format(i, reader_type_string, test_file)):
            test_data = reader[:2, :, i]
            instance.assertEqual(test_data.shape[:2], (2, data_size[1]), msg='Complete row fetch size mismatch')

        with instance.subTest(msg='Validity of sidd at index {} in reader of '
                                  'type {} for file {}'.format(i, reader_type_string, test_file)):
            if not sidd.is_valid(recursive=True, stack=False):
                logging.warning('sidd at index {} in reader of type {} for file {} not valid'.format(i, reader_type_string, test_file))
    del reader


# NB: I'm splitting these tests to ensure interpretable names - each reader has it's own test.


class TestSIDD(unittest.TestCase):
    @unittest.skipIf(len(product_file_types.get('SIDD', [])) == 0, 'No SIDD files specified or found')
    def test_sidd_reader(self):
        for test_file in product_file_types['SIDD']:
            generic_reader_test(self, test_file, 'SIDD', SIDDReader)
