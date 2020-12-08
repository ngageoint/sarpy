import os
import time
import logging
import json

from sarpy.io.general.nitf import NITFDetails
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader
from sarpy.io.general.nitf_elements.text import TextSegmentHeader
from sarpy.io.general.nitf_elements.graphics import GraphicsSegmentHeader
from sarpy.io.general.nitf_elements.des import DataExtensionHeader

from tests import unittest, parse_file_entry

no_files = False
test_files = []
this_loc = os.path.abspath(__file__)
file_reference = os.path.join(os.path.split(this_loc)[0], 'nitf_headers.json')  # specifies file locations

if os.path.isfile(file_reference):
    with open(file_reference, 'r') as fi:
        for entry in json.load(fi):
            the_file = parse_file_entry(entry)
            if the_file is not None:
                test_files.append(the_file)
    if len(test_files) == 0:
        logging.warning('No files have been identified for the nitf header tests.')
        no_files = True
else:
    logging.error('Can not find the nitf_header.json file identifying nitf header tests.')
    no_files = True


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
        header_string = details.nitf_header.to_bytes()
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

    # is each image subheader working?
    if details.img_segment_offsets is not None:
        for i in range(details.img_segment_offsets.size):
            with instance.subTest('image subheader {} match'.format(i)):
                img_bytes = details.get_image_subheader_bytes(i)
                img_sub = ImageSegmentHeader.from_bytes(img_bytes, start=0)
                instance.assertEqual(
                    len(img_bytes), img_sub.get_bytes_length(), msg='image subheader as long as expected')
                instance.assertEqual(
                    img_bytes, img_sub.to_bytes(), msg='image subheader serializes and deserializes as expected')

    # is each text segment working?
    if details.text_segment_offsets is not None:
        for i in range(details.text_segment_offsets.size):
            with instance.subTest('text subheader {} match'.format(i)):
                txt_bytes = details.get_text_subheader_bytes(i)
                txt_sub = TextSegmentHeader.from_bytes(txt_bytes, start=0)
                instance.assertEqual(
                    len(txt_bytes), txt_sub.get_bytes_length(), msg='text subheader as long as expected')
                instance.assertEqual(
                    txt_bytes, txt_sub.to_bytes(), msg='text subheader serializes and deserializes as expected')

    # is each graphics segment working?
    if details.graphics_segment_offsets is not None:
        for i in range(details.graphics_segment_offsets.size):
            with instance.subTest('graphics subheader {} match'.format(i)):
                graphics_bytes = details.get_graphics_subheader_bytes(i)
                graphics_sub = GraphicsSegmentHeader.from_bytes(graphics_bytes, start=0)
                instance.assertEqual(
                    len(graphics_bytes), graphics_sub.get_bytes_length(), msg='graphics subheader as long as expected')
                instance.assertEqual(
                    graphics_bytes, graphics_sub.to_bytes(), msg='graphics subheader serializes and deserializes as expected')


    # is each data extenson subheader working?
    if details.des_segment_offsets is not None:
        for i in range(details.des_segment_offsets.size):
            with instance.subTest('des subheader {} match'.format(i)):
                des_bytes = details.get_des_subheader_bytes(i)
                des_sub = DataExtensionHeader.from_bytes(des_bytes, start=0)
                instance.assertEqual(
                    len(des_bytes), des_sub.get_bytes_length(), msg='des subheader as long as expected')
                instance.assertEqual(
                    des_bytes, des_sub.to_bytes(), msg='des subheader serializes and deserializes as expected')


class TestNITFHeader(unittest.TestCase):

    @unittest.skipIf(no_files, 'No nitf files identified for testing')
    def test_nitf_header(self):
        for test_file in test_files:
            generic_nitf_header_test(self, test_file)
