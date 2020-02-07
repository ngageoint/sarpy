import os
import time
import logging

from sarpy.io.nitf_headers import NITFDetails

from . import unittest


def generic_nitf_header_test(instance, test_file):
    assert isinstance(instance, unittest.TestCase)

    # can we parse it at all? how long does it take?
    with instance.subTest(msg="header parsing"):
        start = time.time()
        details = NITFDetails(test_file)
        # how long does it take?
        logging.info('unpacked nitf details in {}'.format(time.time() - start))
        # how does it look?
        logging.debug(details.nitf_header)

    # is the output as long as it should be?
    with instance.subTest(msg="header length match"):
        header_string = details.nitf_header.to_string()
        equality = (len(header_string) == details.nitf_header.HL)
        if not equality:
            logging.error(
                'len(produced header) = {}, nitf_header.HL = {}'.format(len(header_string),
                                                                        details.nitf_header.HL))
        instance.assertTrue(equality)

    # is the output what it should be?
    with instance.subTest(msg="header content match"):
        with open(test_file, 'rb') as fi:
            file_header = fi.read(details.nitf_header.HL)

        equality = (file_header == header_string)
        if not equality:
            chunk_size = 80
            start_chunk = 0
            while start_chunk < len(header_string):
                end_chunk = min(start_chunk + chunk_size, len(header_string))
                logging.error('real[{}:{}] = {}'.format(
                    start_chunk, end_chunk, file_header[start_chunk:end_chunk]))
                logging.error('prod[{}:{}] = {}'.format(
                    start_chunk, end_chunk, header_string[start_chunk:end_chunk]))
                start_chunk = end_chunk
        instance.assertTrue(equality)


class TestNITFHeader(unittest.TestCase):
    @classmethod
    def setUp(cls):
        # todo: fix this up
        cls.test_root = os.path.expanduser(os.path.join('~', 'Desktop', 'sarpy_testing', 'sicd'))

    def test_nitf_header(self):
        tested = 0
        for fil in [
                'sicd_example_RMA_RGZERO_RE16I_IM16I.nitf',
                'sicd_example_RMA_RGZERO_RE32F_IM32F.nitf',
                'sicd_example_RMA_RGZERO_RE32F_IM32F_cropped_multiple_image_segments_v1.2.nitf']:
            test_file = os.path.join(self.test_root, fil)
            if os.path.exists(test_file):
                tested += 1
                generic_nitf_header_test(self, test_file)
            else:
                logging.info('No file {} found'.format(test_file))

        self.assertTrue(tested > 0, msg="No files for testing found")
