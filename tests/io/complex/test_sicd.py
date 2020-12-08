import os
import json
import tempfile
import shutil

from sarpy.io.complex.converter import open_complex, conversion_utility
import sarpy.io.complex.sicd
from sarpy.io.complex.sicd import SICDReader


from tests import unittest, parse_file_entry

try:
    from lxml import etree
except ImportError:
    etree = None


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

sicd_files = complex_file_types.get('SICD', [])

the_schema = os.path.join(
    os.path.split(sarpy.io.complex.sicd.__file__)[0],
    'sicd_schema',
    'SICD_schema_V1_2_1_2018_12_13.xsd')


class TestSICDWriting(unittest.TestCase):

    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def test_sicd_creation(self):
        for fil in sicd_files:
            reader = SICDReader(fil)

            # check that sicd structure serializes according to the schema
            if etree is not None:
                sicd = reader.get_sicds_as_tuple()[0]
                xml_doc = etree.fromstring(sicd.to_xml_bytes())
                xml_schema = etree.XMLSchema(file=the_schema)
                with self.subTest(msg='validate xml produced from sicd structure'):
                    self.assertTrue(xml_schema.validate(xml_doc),
                                    msg='SICD structure serialized from file {} is '
                                        'not valid versus schema {}'.format(fil, the_schema))

            # create a temp directory
            temp_directory = tempfile.mkdtemp()

            with self.subTest(msg='Test conversion (recreation) of the sicd file {}'.format(fil)):
                conversion_utility(reader, temp_directory)

            # clean up the temporary directory
            shutil.rmtree(temp_directory)
