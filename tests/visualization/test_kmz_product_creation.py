#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import json
import pathlib
import xml.etree.ElementTree
import zipfile

import pytest

import sarpy.io.complex.converter
import sarpy.visualization.kmz_product_creation as complex_kmz

from tests import parse_file_entry


complex_file_types = {}

file_reference = (
    pathlib.Path(__file__).parents[1] / "io/complex/complex_file_types.json"
)
if file_reference.is_file():
    the_files = json.loads(file_reference.read_text())
    for the_type in the_files:
        valid_entries = []
        for entry in the_files[the_type]:
            the_file = parse_file_entry(entry)
            if the_file is not None:
                valid_entries.append(the_file)
        complex_file_types[the_type] = valid_entries


@pytest.fixture(scope="module")
def sicd_file():
    for file in complex_file_types.get("SICD", []):
        if pathlib.Path(file).name == "sicd_example_1_PFA_RE32F_IM32F_HH.nitf":
            return file
    pytest.skip("sicd test file not found")


@pytest.mark.parametrize(
    "inc_antenna",
    (
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True, reason="Example SICD has nonsensical antenna parameters"
            ),
        ),
    ),
)
def test_create_kmz(sicd_file, tmp_path, inc_antenna):
    reader = sarpy.io.complex.converter.open_complex(sicd_file)
    file_stem = "the_file_stem"
    complex_kmz.create_kmz_view(
        reader,
        tmp_path,
        file_stem=file_stem,
        inc_image_corners=True,
        inc_valid_data=True,
        inc_scp=True,
        inc_collection_wedge=True,
        inc_antenna=inc_antenna,
    )

    assert len(list(tmp_path.glob("**/*"))) == 1
    produced_file = next(tmp_path.glob(file_stem + "*.kmz"))

    with zipfile.ZipFile(produced_file, "r") as kmz:
        assert "doc.kml" in kmz.namelist()
        with kmz.open("doc.kml") as kml_fd:
            tree = xml.etree.ElementTree.parse(kml_fd)
            assert tree.getroot().tag == "{http://www.opengis.net/kml/2.2}kml"
