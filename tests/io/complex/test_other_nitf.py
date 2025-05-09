import os
import json

import pytest

from sarpy.io.complex import other_nitf

from tests import parse_file_entry


complex_file_types = {}
this_loc = os.path.abspath(__file__)
file_reference = os.path.join(
    os.path.split(this_loc)[0], "complex_file_types.json"
)  # specifies file locations
if os.path.isfile(file_reference):
    with open(file_reference, "r") as fi:
        the_files = json.load(fi)
        for the_type in the_files:
            valid_entries = []
            for entry in the_files[the_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            complex_file_types[the_type] = valid_entries

sicd_files = complex_file_types.get("SICD", [])


@pytest.mark.parametrize("sicd_file", sicd_files)
def test_read_sicd_with_complex_reader(sicd_file):
    details = other_nitf.ComplexNITFDetails(sicd_file)
    assert other_nitf.ComplexNITFReader(details) is not None
