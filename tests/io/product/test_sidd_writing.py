import logging
import os
import json
import tempfile
import shutil

from sarpy.io.complex.sicd import SICDReader
import sarpy.io.product.sidd
from sarpy.io.product.sidd import SIDDReader
from sarpy.io.product.sidd_product_creation import create_detected_image_sidd, create_dynamic_image_sidd, create_csi_sidd
from sarpy.processing.ortho_rectify import NearestNeighborMethod

from tests import unittest, parse_file_entry

try:
    from lxml import etree
except ImportError:
    etree = None


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

sicd_files = product_file_types.get('SICD', [])

schemas = {
    1: os.path.join(
        os.path.split(sarpy.io.product.sidd.__file__)[0],
        'sidd_schema',
        'SIDD_schema_V1.0.0_2011_08_31.xsd'),
    2: os.path.join(
        os.path.split(sarpy.io.product.sidd.__file__)[0],
        'sidd_schema',
        'SIDD_schema_V2.0.0_2019_05_31.xsd')}


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

            # create a csi image
            with self.subTest(msg='Create version 1 csi for file {}'.format(fil)):
                create_csi_sidd(
                    ortho_helper, temp_directory, output_file='csi_1.nitf', version=1)
                sidd_files.append('csi_1.nitf')
            with self.subTest(msg='Create version 2 csi for file {}'.format(fil)):
                create_csi_sidd(
                    ortho_helper, temp_directory, output_file='csi_2.nitf', version=2)
                sidd_files.append('csi_2.nitf')

            # create a dynamic image
            with self.subTest(msg='Create version 1 subaperture stack for file {}'.format(fil)):
                create_dynamic_image_sidd(
                    ortho_helper, temp_directory, output_file='sast_1.nitf', version=1, frame_count=3)
                sidd_files.append('sast_1.nitf')
            with self.subTest(msg='Create version 2 subaperture stack for file {}'.format(fil)):
                create_dynamic_image_sidd(
                    ortho_helper, temp_directory, output_file='sast_2.nitf', version=2, frame_count=3)
                sidd_files.append('sast_2.nitf')

            # check that each sidd structure serialized according to the schema
            if etree is not None:
                for vers in [1, 2]:
                    the_fil = 'di_{}.nitf'.format(vers)
                    if the_fil in sidd_files:
                        if not check_versus_schema(os.path.join(temp_directory, the_fil), schemas[vers]):
                            logging.warning('Detected image version {} structure not valid versus schema {}'.format(vers, schemas[vers]))

                    the_fil = 'csi_{}.nitf'.format(vers)
                    if the_fil in sidd_files:
                        if not check_versus_schema(os.path.join(temp_directory, the_fil), schemas[vers]):
                            logging.warning('csi version {} structure not valid versus schema {}'.format(vers, schemas[vers]))

                    the_fil = 'sast_{}.nitf'.format(vers)
                    if the_fil in sidd_files:
                        if not check_versus_schema(os.path.join(temp_directory, the_fil), schemas[vers]):
                            logging.warning('Dynamic image version {} structure not valid versus schema {}'.format(vers, schemas[vers]))

            # clean up the temporary directory
            shutil.rmtree(temp_directory)
