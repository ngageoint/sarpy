import logging
import pathlib
import tempfile

import pytest

from tests import find_test_data_files
import sarpy.utils.sicd_sidelobe_control

sicd_files = find_test_data_files(pathlib.Path(__file__).parent / 'utils_file_types.json').get('SICD', [])

logging.basicConfig(level=logging.WARNING)


def test_sicd_sidelobe_control_help(capsys):
    with pytest.raises(SystemExit):
        sarpy.utils.sicd_sidelobe_control.main(['--help'])

    captured = capsys.readouterr()

    assert captured.err == ''
    assert captured.out.startswith('usage:')


@pytest.mark.parametrize("sicd_in_file", sicd_files)
def test_sicd_sidelobe_control_pars(sicd_in_file):
    with tempfile.TemporaryDirectory() as tempdir:
        sicd_out_file = pathlib.Path(tempdir) / "sicd_taper.nitf"

        args = [str(sicd_in_file), str(sicd_out_file), "--window", "taylor", "--pars", "5", "-35"]

        sarpy.utils.sicd_sidelobe_control.main(args)

        assert sicd_out_file.exists()
