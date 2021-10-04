import logging
import os
import json
import tempfile

import numpy.testing
from sarpy.io.received.crsd import CRSDReader, CRSDReader1_0, CRSDWriter1_0
from sarpy.io.received.converter import open_received
from sarpy.io.received.crsd_schema import get_schema_path

from tests import unittest, parse_file_entry

DEFAULT_SCHEMA = get_schema_path(version='1.0.0')

crsd_file_types = {}

this_loc = os.path.abspath(__file__)
file_reference = os.path.join(os.path.split(this_loc)[0], 'crsd_file_types.json')  # specifies file locations
if os.path.isfile(file_reference):
    with open(file_reference, 'r') as fi:
        the_files = json.load(fi)
        for the_type in the_files:
            valid_entries = []
            for entry in the_files[the_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            crsd_file_types[the_type] = valid_entries


def generic_io_test(instance, test_file, reader_type_string, reader_type):
    assert isinstance(instance, unittest.TestCase)

    reader = None
    with instance.subTest(msg='establish reader for type {} and file {}'.format(reader_type_string, test_file)):
        reader = open_received(test_file)
        instance.assertTrue(reader is not None, msg='Returned None, so opening failed.')

    if reader is None:
        return  # remaining tests make no sense

    assert isinstance(reader, CRSDReader)
    with instance.subTest(msg='Reader for type {} should be appropriate reader'):
        instance.assertTrue(isinstance(reader, reader_type), msg='Returned reader should be of type {}'.format(reader_type))

    if not isinstance(reader, reader_type):
        return  # remaining tests might be misleading

    with instance.subTest(msg='Verify reader_type for type {} and file {}'.format(reader_type_string, test_file)):
        instance.assertEqual(reader.reader_type, "CRSD", msg='reader.reader_type should be "CRSD"')

    with instance.subTest(msg='Validity of crsd in reader of '
                              'type {} for file {}'.format(reader_type_string, test_file)):
        if not reader.crsd_meta.is_valid(recursive=True, stack=False):
            logging.warning(
                'crsd in reader of type {} for file {} not valid'.format(reader_type_string, test_file))

    with instance.subTest(msg='Fetch data_sizes and sidds for type {} and file {}'.format(reader_type_string, test_file)):
        data_sizes = reader.get_data_size_as_tuple()
        if isinstance(reader, CRSDReader1_0):
            elements = reader.crsd_meta.Data.Channels
        else:
            raise TypeError('Got unhandled reader type {}'.format(type(reader)))

    for i, (data_size, element) in enumerate(zip(data_sizes, elements)):
        with instance.subTest(msg='Verify image size for sidd index {} in reader '
                                  'of type {} for file {}'.format(i, reader_type_string, test_file)):
            instance.assertEqual(data_size[0], element.NumVectors, msg='data_size[0] and NumVectors do not agree')
            instance.assertEqual(data_size[1], element.NumSamples, msg='data_size[1] and NumSamples do not agree')

        with instance.subTest(msg='Basic fetch test for crsd index {} in reader '
                                  'of type {} for file {}'.format(i, reader_type_string, test_file)):
            instance.assertEqual(reader[:2, :2, i].shape[:2], (2, 2), msg='upper left fetch')
            instance.assertEqual(reader[-2:, :2, i].shape[:2], (2, 2), msg='lower left fetch')
            instance.assertEqual(reader[-2:, -2:, i].shape[:2], (2, 2), msg='lower right fetch')
            instance.assertEqual(reader[:2, -2:, i].shape[:2], (2, 2), msg='upper right fetch')

        with instance.subTest(msg='Verify fetching complete row(s) have correct size '
                                  'for crsd index {} in reader of type {} and file {}'.format(i, reader_type_string, test_file)):
            test_data = reader[:, :2, i]
            instance.assertEqual(test_data.shape[:2], (data_size[0], 2), msg='Complete row fetch size mismatch')

        with instance.subTest(msg='Verify fetching complete columns(s) have correct size '
                                  'for crsd index {} in reader of type {} file {}'.format(i, reader_type_string, test_file)):
            test_data = reader[:2, :, i]
            instance.assertEqual(test_data.shape[:2], (2, data_size[1]), msg='Complete row fetch size mismatch')

        with instance.subTest(msg='Verify fetching entire pvp data has correct size for crsd '
                                  'index {} in reader of type {} file {}'.format(i, reader_type_string, test_file)):
            test_pvp = reader.read_pvp_variable('TxTime', i, the_range=None)
            instance.assertEqual(test_pvp.shape, (data_size[0], ), msg='Unexpected pvp total fetch size')

        with instance.subTest(msg='Verify fetching pvp data for slice has correct size for crsd '
                                  'index {} in reader of type {} file {}'.format(i, reader_type_string, test_file)):
            test_pvp = reader.read_pvp_variable('TxTime', i, the_range=(0, 10, 2))
            instance.assertEqual(test_pvp.shape, (5, ), msg='Unexpected pvp strided slice fetch size')

    if isinstance(reader, CRSDReader1_0):
        generic_writer_test(reader)

    del reader


def generic_writer_test(crsd_reader):
    with tempfile.NamedTemporaryFile() as written_crsd:

        read_support = crsd_reader.read_support_block()
        read_pvp = crsd_reader.read_pvp_block()
        read_signal = crsd_reader.read_signal_block()

        # write the crsd file
        with CRSDWriter1_0(written_crsd.name, crsd_reader.crsd_meta, check_existence=False) as writer:
            writer.write_file(read_pvp, read_signal, read_support)

        # reread the newly written data
        rereader = CRSDReader(written_crsd.name)
        reread_support = rereader.read_support_block()
        reread_pvp = rereader.read_pvp_block()
        reread_signal = rereader.read_signal_block()

        # byte compare that the original data and re-read data are identical
        numpy.testing.assert_equal(read_support, reread_support)
        numpy.testing.assert_equal(read_pvp, reread_pvp)
        numpy.testing.assert_equal(read_signal, reread_signal)


class TestCRSD(unittest.TestCase):
    @unittest.skipIf(len(crsd_file_types.get('CRSD', [])) == 0, 'No CRSD files specified or found')
    def test_crsd_io(self):
        for test_file in crsd_file_types['CRSD']:
            generic_io_test(self, test_file, 'CRSD', CRSDReader)
