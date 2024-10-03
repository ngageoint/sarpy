import collections
import os
import pathlib
import tempfile
import shutil
import unittest

import numpy as np

from sarpy.io.complex.sicd import SICDReader
from sarpy.io.product.sidd import SIDDReader
from sarpy.io.product.sidd_schema import get_schema_path
import sarpy.io.product.sidd
from sarpy.processing.sidd.sidd_product_creation import create_detected_image_sidd, create_dynamic_image_sidd, create_csi_sidd
from sarpy.processing.ortho_rectify import NearestNeighborMethod

import tests

try:
    from lxml import etree
except ImportError:
    etree = None


product_file_types = tests.find_test_data_files(pathlib.Path(__file__).parent / 'product_file_types.json')
sicd_files = product_file_types.get('SICD', [])


def check_versus_schema(input_nitf, the_schema):
    reader = SIDDReader(input_nitf)
    sidd_bytes = reader.nitf_details.get_des_bytes(0)
    xml_doc = etree.fromstring(sidd_bytes)
    xml_schema = etree.XMLSchema(file=the_schema)
    return xml_schema.validate(xml_doc)


class TestSIDDWriting(unittest.TestCase):

    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def test_sidd_creation(self):
        for fil in sicd_files:
            reader = SICDReader(fil)
            ortho_helper = NearestNeighborMethod(reader)

            # create a temp directory
            temp_directory = tempfile.mkdtemp()
            sidd_files = []

            # create a basic sidd detected image
            with self.subTest(msg='Create version 1 detected image for file {}'.format(fil)):
                create_detected_image_sidd(
                    ortho_helper, temp_directory, output_file='di_1.nitf', version=1)
                sidd_files.append('di_1.nitf')
            with self.subTest(msg='Create version 2 detected image for file {}'.format(fil)):
                create_detected_image_sidd(
                    ortho_helper, temp_directory, output_file='di_2.nitf', version=2)
                sidd_files.append('di_2.nitf')
            with self.subTest(msg='Create version 3 detected image for file {}'.format(fil)):
                create_detected_image_sidd(
                    ortho_helper, temp_directory, output_file='di_3.nitf', version=3)
                sidd_files.append('di_3.nitf')

            # create a csi image
            with self.subTest(msg='Create version 1 csi for file {}'.format(fil)):
                create_csi_sidd(
                    ortho_helper, temp_directory, output_file='csi_1.nitf', version=1)
                sidd_files.append('csi_1.nitf')
            with self.subTest(msg='Create version 2 csi for file {}'.format(fil)):
                create_csi_sidd(
                    ortho_helper, temp_directory, output_file='csi_2.nitf', version=2)
                sidd_files.append('csi_2.nitf')
            with self.subTest(msg='Create version 3 csi for file {}'.format(fil)):
                create_csi_sidd(
                    ortho_helper, temp_directory, output_file='csi_3.nitf', version=3)
                sidd_files.append('csi_3.nitf')

            # create a dynamic image
            with self.subTest(msg='Create version 1 subaperture stack for file {}'.format(fil)):
                create_dynamic_image_sidd(
                    ortho_helper, temp_directory, output_file='sast_1.nitf', version=1, frame_count=3)
                sidd_files.append('sast_1.nitf')
            with self.subTest(msg='Create version 2 subaperture stack for file {}'.format(fil)):
                create_dynamic_image_sidd(
                    ortho_helper, temp_directory, output_file='sast_2.nitf', version=2, frame_count=3)
                sidd_files.append('sast_2.nitf')
            with self.subTest(msg='Create version 3 subaperture stack for file {}'.format(fil)):
                create_dynamic_image_sidd(
                    ortho_helper, temp_directory, output_file='sast_3.nitf', version=3, frame_count=3)
                sidd_files.append('sast_3.nitf')

            # check that each sidd structure serialized according to the schema
            if etree is not None:
                for vers in [1, 2, 3]:
                    schema = get_schema_path('urn:SIDD:{}.0.0'.format(vers))
                    the_fil = 'di_{}.nitf'.format(vers)
                    if the_fil in sidd_files:
                        self.assertTrue(
                            check_versus_schema(os.path.join(temp_directory, the_fil), schema),
                            'Detected image version {} structure not valid versus schema {}'.format(vers, schema))

                    the_fil = 'csi_{}.nitf'.format(vers)
                    if the_fil in sidd_files:
                        self.assertTrue(
                            check_versus_schema(os.path.join(temp_directory, the_fil), schema),
                            'csi version {} structure not valid versus schema {}'.format(vers, schema))

                    the_fil = 'sast_{}.nitf'.format(vers)
                    if the_fil in sidd_files:
                        self.assertTrue(
                            check_versus_schema(os.path.join(temp_directory, the_fil), schema),
                            'Dynamic image version {} structure not valid versus schema {}'.format(vers, schema))

            # clean up the temporary directory
            shutil.rmtree(temp_directory)

    def test_nitf_multi_image_multi_segment(self):
        """From Figure 2.5-6 SIDD 1.0 Multiple Input Image - Multiple Product Images Requiring Segmentation"""
        sidd_xml = pathlib.Path(__file__).parents[2]/ 'data/example.sidd.xml'
        sidd_meta = sarpy.io.product.sidd.SIDDType2.from_xml_file(sidd_xml)
        assert sidd_meta.Display.PixelType == 'MONO8I'

        # Tweak SIDD size to force three image segments
        li_max = 9_999_999_998
        iloc_max = 99_999
        num_cols = li_max // (2 * iloc_max)  # set num_cols so that row limit is iloc_max
        last_rows = 24
        num_rows = iloc_max * 2 + last_rows
        sidd_meta.Measurement.PixelFootprint.Row = num_rows
        sidd_meta.Measurement.PixelFootprint.Col = num_cols
        sidd_writing_details = sarpy.io.product.sidd.SIDDWritingDetails([sidd_meta, sidd_meta], None)

        # SIDD segmentation algorithm (2.4.2.1 in 1.0/2.0/3.0) would lead to overlaps of the last partial
        # image segment due to ILOC. This implements a scheme similar to SICD wherein "RRRRR" of ILOC matches
        # the NROWs in the previous segment.
        ImHdr = collections.namedtuple('ImHdr', ['IID1', 'IDLVL', 'IALVL', 'ILOC', 'NROWS', 'NCOLS'])
        expected_imhdrs = [
            ImHdr(IID1='SIDD001001', IDLVL=1, IALVL=0, ILOC='0'*10, NROWS=iloc_max, NCOLS=num_cols),
            ImHdr(IID1='SIDD001002', IDLVL=2, IALVL=1, ILOC=f'{iloc_max:05d}{0:05d}', NROWS=iloc_max, NCOLS=num_cols),
            ImHdr(IID1='SIDD001003', IDLVL=3, IALVL=2, ILOC=f'{iloc_max:05d}{0:05d}', NROWS=last_rows, NCOLS=num_cols),
            ImHdr(IID1='SIDD002001', IDLVL=4, IALVL=0, ILOC='0'*10, NROWS=iloc_max, NCOLS=num_cols),
            ImHdr(IID1='SIDD002002', IDLVL=5, IALVL=4, ILOC=f'{iloc_max:05d}{0:05d}', NROWS=iloc_max, NCOLS=num_cols),
            ImHdr(IID1='SIDD002003', IDLVL=6, IALVL=5, ILOC=f'{iloc_max:05d}{0:05d}', NROWS=last_rows, NCOLS=num_cols),
        ]

        actual_imhdrs = [
            ImHdr(IID1=im.subheader.IID1,
                  IDLVL=im.subheader.IDLVL,
                  IALVL=im.subheader.IALVL,
                  ILOC=im.subheader.ILOC,
                  NROWS=im.subheader.NROWS,
                  NCOLS=im.subheader.NCOLS)
            for im in sidd_writing_details.image_managers
        ]
        assert expected_imhdrs == actual_imhdrs


class TestSIDDOptionalFields(unittest.TestCase):
    def setUp(self):
        if not sicd_files:
            return
        sicd_filename = sicd_files[0]
        self.temp_directory = tempfile.mkdtemp()

        reader = SICDReader(sicd_filename)
        ortho_helper = NearestNeighborMethod(reader)

        self.sidd_filename = 'di.nitf'
        create_detected_image_sidd(
            ortho_helper, self.temp_directory, output_file=self.sidd_filename, version=3)

        self.schema = get_schema_path('urn:SIDD:3.0.0')

    def is_instance_valid(self, instance_bytes):
        xml_doc = etree.fromstring(instance_bytes)
        xml_schema = etree.XMLSchema(file=self.schema)
        result = xml_schema.validate(xml_doc)
        if not result:
            print(xml_schema.error_log)
        return result

    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def tearDown(self):
        if not sicd_files:
            return
        shutil.rmtree(self.temp_directory)
