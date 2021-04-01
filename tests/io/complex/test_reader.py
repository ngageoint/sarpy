import logging
import os
import json

from sarpy.io.complex.converter import open_complex
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.complex.radarsat import RadarSatReader
from sarpy.io.complex.sentinel import SentinelReader
from sarpy.io.complex.tsx import TSXReader
from sarpy.io.complex.csk import CSKReader
from sarpy.io.complex.iceye import ICEYEReader
from sarpy.io.complex.palsar2 import PALSARReader

from tests import unittest, parse_file_entry


complex_file_types = {}

this_loc = os.path.abspath(__file__)
file_reference = os.path.join(os.path.split(this_loc)[0], 'complex_file_types.json')  # specifies file locations
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


def generic_reader_test(instance, test_file, reader_type_string, reader_type):
    assert isinstance(instance, unittest.TestCase)

    reader = None
    with instance.subTest(msg='establish reader for type {} and file {}'.format(reader_type_string, test_file)):
        reader = open_complex(test_file)
        instance.assertTrue(reader is not None, msg='Returned None, so opening failed.')

    if reader is None:
        return  # remaining tests make no sense

    with instance.subTest(msg='Reader for type {} should be appropriate reader'):
        instance.assertTrue(isinstance(reader, reader_type), msg='Returned reader should be of type {}'.format(reader_type))

    if not isinstance(reader, reader_type):
        return  # remaining tests might be misleading

    with instance.subTest(msg='Verify reader_type for type {} and file {}'.format(reader_type_string, test_file)):
        instance.assertEqual(reader.reader_type, "SICD", msg='reader.reader_type should be "SICD""')

    with instance.subTest(msg='Fetch data_sizes and sicds for type {} and file {}'.format(reader_type_string, test_file)):
        data_sizes = reader.get_data_size_as_tuple()
        sicds = reader.get_sicds_as_tuple()

    for i, (data_size, sicd) in enumerate(zip(data_sizes, sicds)):
        with instance.subTest(msg='Verify image size for sicd index {} in reader '
                                  'of type {} for file {}'.format(i, reader_type_string, test_file)):
            instance.assertEqual(data_size[0], sicd.ImageData.NumRows, msg='data_size[0] and NumRows do not agree')
            instance.assertEqual(data_size[1], sicd.ImageData.NumCols, msg='data_size[1] and NumCols do not agree')

        with instance.subTest(msg='Basic fetch test for sicd index {} in reader '
                                  'of type {} for file {}'.format(i, reader_type_string, test_file)):
            instance.assertEqual(reader[:2, :2, i].shape, (2, 2), msg='upper left fetch')
            instance.assertEqual(reader[-2:, :2, i].shape, (2, 2), msg='lower left fetch')
            instance.assertEqual(reader[-2:, -2:, i].shape, (2, 2), msg='lower right fetch')
            instance.assertEqual(reader[:2, -2:, i].shape, (2, 2), msg='upper right fetch')

        with instance.subTest(msg='Verify fetching complete row(s) have correct size '
                                  'for sicd index {} in reader of type {} and file {}'.format(i, reader_type_string, test_file)):
            test_data = reader[:, :2, i]
            instance.assertEqual(test_data.shape, (data_size[0], 2), msg='Complete row fetch size mismatch')

        with instance.subTest(msg='Verify fetching complete columns(s) have correct size '
                                  'for sicd index {} in reader of type {} file {}'.format(i, reader_type_string, test_file)):
            test_data = reader[:2, :, i]
            instance.assertEqual(test_data.shape, (2, data_size[1]), msg='Complete row fetch size mismatch')

        with instance.subTest(msg='Validity of sicd at index {} in reader of '
                                  'type {} for file {}'.format(i, reader_type_string, test_file)):
            if not sicd.is_valid(recursive=True, stack=False):
                logging.warning('sicd at index {} in reader of type {} for file {} not valid'.format(i, reader_type_string, test_file))
    del reader


# NB: I'm splitting these tests to ensure interpretable names - each reader has it's own test.


class TestSICD(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('SICD', [])) == 0, 'No SICD files specified or found')
    def test_sicd_reader(self):
        for test_file in complex_file_types['SICD']:
            generic_reader_test(self, test_file, 'SICD', SICDReader)


class TestRCM(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('RCM', [])) == 0, 'No RCM files specified or found')
    def test_rcm_reader(self):
        for test_file in complex_file_types['RCM']:
            generic_reader_test(self, test_file, 'RCM', RadarSatReader)


class TestRCM_NITF(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('RCM_NITF', [])) == 0, 'No RCM_NITF files specified or found')
    def test_rcm_nitf_reader(self):
        for test_file in complex_file_types['RCM_NITF']:
            generic_reader_test(self, test_file, 'RCM_NITF', RadarSatReader)


class TestRS2(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('RadarSat-2', [])) == 0, 'No RadarSat-2 files specified or found')
    def test_rs2_reader(self):
        for test_file in complex_file_types['RadarSat-2']:
            generic_reader_test(self, test_file, 'RadarSat-2', RadarSatReader)


class TestSentinel(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('Sentinel-1', [])) == 0, 'No Sentinel-1 files specified or found')
    def test_sentinel_reader(self):
        for test_file in complex_file_types['Sentinel-1']:
            generic_reader_test(self, test_file, 'Sentinel-1', SentinelReader)


class TestTerraSAR(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('TerraSAR-X', [])) == 0, 'No TerraSAR-X files specified or found')
    def test_terrasarx_reader(self):
        for test_file in complex_file_types['TerraSAR-X']:
            generic_reader_test(self, test_file, 'TerraSAR-X', TSXReader)


class TestCosmoSkymed(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('CosmoSkymed', [])) == 0, 'No CosmoSkymed files specified or found')
    def test_csk_reader(self):
        for test_file in complex_file_types['CosmoSkymed']:
            generic_reader_test(self, test_file, 'CosmoSkymed', CSKReader)


class TestKompSat5(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('KompSat-5', [])) == 0, 'No KompSat-5 files specified or found')
    def test_kompsat_reader(self):
        for test_file in complex_file_types['KompSat-5']:
            generic_reader_test(self, test_file, 'KompSat-5', CSKReader)


class TestICEYE(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('ICEYE', [])) == 0, 'No ICEYE files specified or found')
    def test_iceye_reader(self):
        for test_file in complex_file_types['ICEYE']:
            generic_reader_test(self, test_file, 'ICEYE', ICEYEReader)


class TestPALSAR(unittest.TestCase):
    @unittest.skipIf(len(complex_file_types.get('PALSAR', [])) == 0, 'No PALSAR files specified or found')
    def test_palsar_reader(self):
        for test_file in complex_file_types['PALSAR']:
            generic_reader_test(self, test_file, 'PALSAR', PALSARReader)
