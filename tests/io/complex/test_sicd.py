import os
import time
import logging

from . import unittest

from sarpy.io.complex.sicd import SICDDetails, SICDReader
from sarpy.io.complex.converter import open_complex


def generic_sicd_check(instance, test_file):
    assert isinstance(instance, unittest.TestCase)

    # Check the basics for the sicd details object
    with instance.subTest(msg='parse details'):
        start = time.time()
        test_details = SICDDetails(test_file)
        # how long does it take to unpack test_details?
        logging.info('unpacked sicd details in {}'.format(time.time() - start))

    with instance.subTest(msg='is_sicd'):
        # does it register as a sicd?
        instance.assertTrue(test_details.is_sicd)

    with instance.subTest(msg='meta data fetch'):
        # how does the sicd_meta look?
        logging.debug(test_details.sicd_meta)

    with instance.subTest(msg='image headers fetch'):
        # how do the image headers look?
        for i, entry in enumerate(test_details.img_headers):
            logging.debug('image header {}\n{}'.format(i, entry))

    # Can we construct a reader from a file name?
    with instance.subTest(msg="SICD reader from file name"):
        start = time.time()
        reader = SICDReader(test_file)
        # how long does it take to unpack details and load file?
        logging.info('unpacked sicd details from file name and loaded sicd file in {}'.format(time.time() - start))

    # can we construct a sicd reader from a reader object?
    with instance.subTest(msg="SICD reader from sicd details"):
        start = time.time()
        reader = SICDReader(test_details)
        # how long does it take to unpack details and load file?
        logging.info('loaded sicd file from sicd details in {}'.format(time.time() - start))

    # can we use the open_complex method?
    with instance.subTest(msg="SICD reader from open_complex"):
        start = time.time()
        reader = open_complex(test_file)
        # how long does it take to unpack details and load file?
        logging.info('Used open_complex to unpack sicd details from file name and loaded sicd file in {}'.format(time.time() - start))

    with instance.subTest(msg="data_size access"):
        # how about the sizes?
        logging.debug('data_size = {}'.format(reader.data_size))

    with instance.subTest(msg="sicd meta access"):
        # how about the metadata?
        logging.debug('meta-data = {}'.format(reader.sicd_meta))

    with instance.subTest(msg="data fetch"):
        # how about fetching some data?
        start = time.time()
        rows = min(500, reader.data_size[0][0])
        cols = min(500, reader.data_size[0][1])
        data = reader[:rows, :cols]
        logging.info('data shape = {}, fetched in {}'.format(data.shape, time.time() - start))


class TestSICDReader(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.test_root = os.path.expanduser(os.path.join('~', 'Desktop', 'sarpy_testing', 'sicd'))

    def test_reader(self):
        tested = 0
        for fil in [
                'sicd_example_1_PFA_RE32F_IM32F_HH.nitf',
                'sicd_example_RMA_RGZERO_RE16I_IM16I.nitf',
                'sicd_example_RMA_RGZERO_RE32F_IM32F.nitf',
                'sicd_example_RMA_RGZERO_RE32F_IM32F_cropped_multiple_image_segments_v1.2.nitf']:
            test_file = os.path.join(self.test_root, fil)
            if os.path.exists(test_file):
                tested += 1
                generic_sicd_check(self, test_file)
            else:
                logging.info('No file {} found'.format(test_file))

        self.assertTrue(tested > 0, msg="No files for testing found")
