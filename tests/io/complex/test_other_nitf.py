import os
import json
import tempfile
import unittest

from sarpy.io.complex.converter import conversion_utility
from sarpy.io.complex.other_nitf import ComplexNITFDetails, ComplexNITFReader
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.complex.sicd_schema import get_schema_path, get_default_version_string


from tests import parse_file_entry

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

the_version = get_default_version_string()
the_schema = get_schema_path(the_version)


class TestSICDWriting(unittest.TestCase):

    @unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
    def test_sicd_creation(self):
        for fil in sicd_files:
            details = ComplexNITFDetails(fil)
            reader = ComplexNITFReader(details)

            with self.subTest(msg='Test conversion (recreation) of the sicd file {}'.format(fil)):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    conversion_utility(reader, tmpdirname)
                    new_filename = os.path.join(tmpdirname, os.listdir(tmpdirname)[0])
                    reader2 = SICDReader(new_filename)
                    self.assertEqual(os.stat(new_filename).st_size, reader2.nitf_details.nitf_header.FL)

            with self.subTest(msg='Test writing a single row of the sicd file {}'.format(fil)):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    conversion_utility(reader, tmpdirname, row_limits=(0, 1))
