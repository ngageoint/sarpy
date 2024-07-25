import json
import os
import pathlib

import pytest

from sarpy.io.DEM.DTED import DTEDInterpolator
from sarpy.io.DEM.geoid import GeoidHeight

SRC_FILE_PATH = pathlib.Path(__file__).parent
sarpy_test_path = os.environ.get("SARPY_TEST_PATH", ".")
geoid_json_path = SRC_FILE_PATH / "geoid.json"

geoid_file_info = json.loads(geoid_json_path.read_text())
egm96_file = sarpy_test_path / pathlib.Path(geoid_file_info["geoid_files"][0]["path"])


@pytest.mark.skipif(not egm96_file.exists(), reason="EGM 96 data does not exist")
def test_interpolator_no_readers():
    llb = [10.0, 20.0, 10.5, 20.5]
    geoid = GeoidHeight(egm96_file)
    dtedinterp = DTEDInterpolator([], geoid_file=geoid, lat_lon_box=llb)

    assert dtedinterp.get_max_geoid(llb) == 0
    assert dtedinterp.get_max_hae(llb) == geoid(10, 10.5)
