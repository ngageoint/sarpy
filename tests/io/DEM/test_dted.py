import pathlib

import pytest

import sarpy.io.DEM.DTED as sarpy_dted
from sarpy.io.DEM.geoid import GeoidHeight

import tests


test_data = tests.find_test_data_files(pathlib.Path(__file__).parent / "geoid.json")
egm96_file = test_data["geoid_files"][0] if test_data["geoid_files"] else None


@pytest.mark.skipif(egm96_file is None, reason="EGM 96 data does not exist")
def test_interpolator_no_readers():
    llb = [10.0, 20.0, 10.5, 20.5]
    geoid = GeoidHeight(egm96_file)
    dtedinterp = sarpy_dted.DTEDInterpolator([], geoid_file=geoid, lat_lon_box=llb)

    assert dtedinterp.get_max_geoid(llb) == 0
    assert dtedinterp.get_max_hae(llb) == geoid(10, 10.5)


@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][0])

    # From entity ID: SRTM3S04W061V1, date updated: 2013-04-17T12:16:47-05
    # Acquired from https://earthexplorer.usgs.gov/ on 2024-08-21
    known_values = {
        (1000, 800): -32767,  # null
        (1000, 799): 7,
        (3, 841): -5,
    }
    for index, expected_value in known_values.items():
        assert dted_reader[index] == expected_value
