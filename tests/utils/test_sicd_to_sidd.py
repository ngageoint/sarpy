import logging
import pathlib
import tempfile

import pytest

from tests import find_test_data_files
import sarpy.utils.sicd_to_sidd

sicd_files = find_test_data_files(pathlib.Path(__file__).parent / 'utils_file_types.json').get('SICD', [])

logging.basicConfig(level=logging.WARNING)


def test_sicd_to_sidd_help(capsys):
    with pytest.raises(SystemExit):
        sarpy.utils.sicd_to_sidd.main(['--help'])

    captured = capsys.readouterr()

    assert captured.err == ''
    assert captured.out.startswith('usage:')


@pytest.mark.parametrize("sicd_file", sicd_files)
def test_sicd_to_sidd_without_taper(sicd_file):
    with tempfile.TemporaryDirectory() as tempdir:
        sidd_dir_path = pathlib.Path(tempdir)

        args = [str(sicd_file), str(sidd_dir_path),
                "--type", "detected", "--method", "nearest", "--version", "3"]

        sarpy.utils.sicd_to_sidd.main(args)

        output_sidd_files = list(sidd_dir_path.iterdir())
        assert len(output_sidd_files) == 1


@pytest.mark.parametrize("sicd_file", sicd_files)
def test_sicd_to_sidd_with_taper(sicd_file):
    with tempfile.TemporaryDirectory() as tempdir:
        sidd_dir_path = pathlib.Path(tempdir)

        args = [str(sicd_file), str(sidd_dir_path),
                "--window", "taylor", "--pars", "5", "-35",
                "--type", "detected", "--method", "nearest", "--version", "3"]

        sarpy.utils.sicd_to_sidd.main(args)

        output_sidd_files = list(sidd_dir_path.iterdir())
        assert len(output_sidd_files) == 1
