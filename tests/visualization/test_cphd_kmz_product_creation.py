import os
import xml.etree.ElementTree
import zipfile

import pytest

import sarpy.io.phase_history
import sarpy.visualization.cphd_kmz_product_creation as cphd_kmz


TEST_FILE_NAMES = {
    'simple': 'spotlight_example.cphd',
}

TEST_FILE_PATHS = {}
TEST_FILE_ROOT = os.environ.get('SARPY_TEST_PATH', None)
if TEST_FILE_ROOT is not None:
    for name_key, path_value in TEST_FILE_NAMES.items():
        the_file = os.path.join(TEST_FILE_ROOT, 'cphd', path_value)
        if os.path.isfile(the_file):
            TEST_FILE_PATHS[name_key] = the_file


@pytest.fixture(scope='module')
def cphd_file():
    file_path = TEST_FILE_PATHS.get('simple', None)
    if file_path is None:
        pytest.skip('simple cphd test file not found')
    else:
        return file_path


def test_create_kmz(cphd_file, tmp_path):
    reader = sarpy.io.phase_history.open(cphd_file)
    file_stem = 'the_file_stem'
    cphd_kmz.cphd_create_kmz_view(reader, tmp_path, file_stem=file_stem)

    assert len(list(tmp_path.glob('**/*'))) == 1
    produced_file = next(tmp_path.glob(file_stem + '*.kmz'))

    with zipfile.ZipFile(produced_file, 'r') as kmz:
        assert set(kmz.namelist()) == {'doc.kml'}
        with kmz.open('doc.kml') as kml_fd:
            tree = xml.etree.ElementTree.parse(kml_fd)
            assert tree.getroot().tag == '{http://www.opengis.net/kml/2.2}kml'

