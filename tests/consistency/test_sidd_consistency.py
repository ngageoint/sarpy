#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import os
import pathlib
import shutil
import tempfile

from lxml import etree
import pytest

import sarpy.consistency.sidd_consistency as sc
import sarpy.utils.create_product
from tests import find_test_data_files

TEST_FILE_PATHS = {}
TEST_FILE_ROOT = os.environ.get('SARPY_TEST_PATH', None)
if TEST_FILE_ROOT is not None:
    TEST_FILE_PATHS = find_test_data_files(pathlib.Path(__file__).parent / 'consistency_file_types.json')


@pytest.fixture(scope="module")
def tmpdir():
    dirname = tempfile.mkdtemp()
    yield dirname
    shutil.rmtree(dirname)


@pytest.fixture(scope="module")
def good_sidd(tmpdir):
    file_path = TEST_FILE_PATHS.get("inconsistent_sicd", None)
    if file_path in [None, []]:
        pytest.skip("sicd test file not found")
    else:
        sidd_dir_path = pathlib.Path(tmpdir)

        args = [str(file_path[0]), str(sidd_dir_path),
                "--type", "detected", "--method", "nearest", "--version", "3"]

        sarpy.utils.create_product.main(args)

        sidd_files = [x for x in sidd_dir_path.iterdir()]
        yield str(sidd_files[0])


@pytest.fixture(scope="module")
def good_sidd_xml_path():
    return str(pathlib.Path(__file__).parent / "../data/example.sidd.xml")


@pytest.fixture(scope='module')
def good_sidd_xml_str(good_sidd_xml_path):
    with open(good_sidd_xml_path, 'rb') as fi:
        return fi.read()


def _copy_xml(elem):
    return etree.fromstring(etree.tostring(elem))


def test_main(good_sidd, good_sidd_xml_path):
    assert sc.main([good_sidd])
    assert sc.main([good_sidd_xml_path])


def test_check_file(good_sidd, good_sidd_xml_path):
    assert sc.check_file(good_sidd)
    assert sc.check_file(good_sidd_xml_path)

    with pytest.raises(ValueError, match="Got string input, but it is not a valid path"):
        sc.check_file("bad_string")


def test_evaluate_xml_versus_schema(good_sidd_xml_str, caplog):
    valid_urn = "urn:SIDD:2.0.0"

    assert sc.evaluate_xml_versus_schema(good_sidd_xml_str, valid_urn)

    invalid_urn = "urn:SIDD:2.2.2"
    assert not sc.evaluate_xml_versus_schema(good_sidd_xml_str, invalid_urn)
    assert "SIDD: Failed finding the schema for urn" in caplog.text
    caplog.clear()


def test_invalid_xml_pixeltype(good_sidd_xml_str, tmpdir):
    good_xml_root = etree.fromstring(good_sidd_xml_str)
    bad_xml_root = _copy_xml(good_xml_root)
    bad_xml_root.find('./{urn:SIDD:2.0.0}Display/{urn:SIDD:2.0.0}PixelType').text += '-make-bad'

    bad_xml_str = etree.tostring(bad_xml_root)

    bad_xml_file_path = os.path.join(tmpdir, 'sidd.xml')
    with open(bad_xml_file_path, 'wb') as fid:
        fid.write(bad_xml_str)

    assert not sc.main([bad_xml_file_path])