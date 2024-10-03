#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import os
import pathlib

import pytest

import sarpy.consistency.sicd_consistency as sc
from tests import find_test_data_files

TEST_FILE_PATHS = {}
TEST_FILE_ROOT = os.environ.get('SARPY_TEST_PATH', None)
if TEST_FILE_ROOT is not None:
    TEST_FILE_PATHS = find_test_data_files(pathlib.Path(__file__).parent / 'consistency_file_types.json')


@pytest.fixture(scope="module")
def bad_sicd():
    file_path = TEST_FILE_PATHS.get("inconsistent_sicd", None)
    if file_path in [None, []]:
        pytest.skip("simple_bad sicd test file not found")
    else:
        return file_path[0]


@pytest.fixture(scope="module")
def good_sicd_xml():
    return str(pathlib.Path(__file__).parent / "../data/example.sicd.xml")


def test_main(bad_sicd, good_sicd_xml):
    assert not sc.main([str(bad_sicd)])
    assert sc.main([good_sicd_xml])


def test_check_file(bad_sicd, good_sicd_xml):
    assert not sc.check_file(bad_sicd)
    assert sc.check_file(good_sicd_xml)

    with pytest.raises(ValueError, match="Got string input, but it is not a valid path"):
        sc.check_file("bad_string")


def test_evaluate_xml_versus_schema(good_sicd_xml, caplog):
    valid_urn = "urn:SICD:1.2.1"

    with open(good_sicd_xml, "rb") as fi:
        sicd_xml = fi.read()

    assert sc.evaluate_xml_versus_schema(sicd_xml, valid_urn)

    invalid_urn = "urn:SICD:1.2.3"
    assert not sc.evaluate_xml_versus_schema(sicd_xml, invalid_urn)
    assert "SICD: Failed getting the schema for urn" in caplog.text
    caplog.clear()
