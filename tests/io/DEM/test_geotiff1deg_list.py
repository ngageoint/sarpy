"""
These test functions will exercise the GeoTIFF1DegList class which contains methods used to determine
which GeoTIFF files are needed to cover a specified geodetic bounding box.  These tests create dummy,
temporary, DEM files on-the-fly, so there is no need to provide any actual DEM files.

"""
import logging
import pathlib
import tempfile

import pytest

from sarpy.io.DEM.geotiff1deg import GeoTIFF1DegList

# SW corner (degrees) of valid DEM pixels
MIN_LAT = -3
MIN_LON = -2

# NE corner (degrees) of valid DEM pixels
MAX_LAT = 2
MAX_LON = 4

logging.basicConfig(level=logging.WARNING)

dataset_ff = {
    "high_res": ["tdt_{ns:1s}{abslat:02}{ew:1s}{abslon:03}_{ver:2s}",
                 "DEM",
                 "TDT_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_{ver:2s}_DEM.tif"],
    "low_res": ["TDM1_DEM__30_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_V{ver:2s}_C",
                "DEM",
                "TDM1_DEM__30_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_DEM.tif"
                ]
}


@pytest.fixture(scope='module')
def dem_file_path():
    """
    Create a directory of empty files that satisfy the DEM "high_res" naming convention.

    """
    ver_choice = ('01', '02')

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        for lat in range(MIN_LAT, MAX_LAT):
            ns = 's' if lat < 0 else 'n'

            for lon in range(MIN_LON, MAX_LON):
                ew = 'w' if lon < 0 else 'e'

                # Make files that span the prime meridian and the equator
                stem = f"tdt_{ns}{abs(lat):02}{ew}{abs(lon):03}_{ver_choice[0]}"
                filename = temp_path / f"{stem}" / "DEM" / f"{stem.upper()}_DEM.tif"
                filename.parent.mkdir(parents=True, exist_ok=True)
                filename.touch()

                # Make files that span the antimeridian and the equator
                lon2 = lon + 180
                lon2 = (lon2 + 180) % 360 - 180
                ew = 'w' if lon2 < 0 else 'e'

                stem = f"tdt_{ns}{abs(lat):02}{ew}{abs(lon2):03}_{ver_choice[1]}"
                filename = temp_path / f"{stem}" / "DEM" / f"{stem.upper()}_DEM.tif"
                filename.parent.mkdir(parents=True, exist_ok=True)
                filename.touch()

        yield temp_path


def test_filename_from_lat_lon():
    filename_format = "Test_{lat:02d}_{lon:03d}_{abslat:02d}_{abslon:03d}_{ns:1s}{NS:1s}_{ew:1s}{EW:1s}_{ver:2s}"
    obj = GeoTIFF1DegList('/tmp', filename_format=filename_format)
    filename = obj.filename_from_lat_lon(-1, -2, 'V1')
    assert filename == "/tmp/Test_-1_-02_01_002_sS_wW_V1"

    obj = GeoTIFF1DegList('/tmp', filename_format=filename_format+'_{bad}')
    filename = obj.filename_from_lat_lon(-1, -2, 'V1')
    assert filename == "/tmp/Test_-1_-02_01_002_sS_wW_V1_{bad}"

    filename_format = "Test_{lat:02d}_{lon:03d}_{abslat:02d}_{abslon:03d}_{ns}{NS}_{ew}{EW}_{ver:2s}"
    obj = GeoTIFF1DegList('/tmp', filename_format=filename_format)
    filename = obj.filename_from_lat_lon(-1, -2, 'V1')
    assert filename == "/tmp/Test_-1_-02_01_002_sS_wW_V1"


def test_find_dem_files(dem_file_path):
    obj = GeoTIFF1DegList(str(dem_file_path), dataset_ff["high_res"])

    filenames = obj.find_dem_files(MIN_LAT - 1, MIN_LON - 1)
    assert len(filenames) == 0

    filenames = obj.find_dem_files(MIN_LAT, MIN_LON)
    expected_filename = obj.filename_from_lat_lon(MIN_LAT, MIN_LON, '01')
    assert len(filenames) == 1 and filenames[0].endswith(expected_filename)

    filenames = obj.find_dem_files(MIN_LAT + 0.5, MIN_LON + 0.5)
    assert len(filenames) == 1 and filenames[0].endswith(expected_filename)

    filenames = obj.find_dem_files(MAX_LAT, MAX_LON)
    expected_filename = obj.filename_from_lat_lon(MAX_LAT-1, MAX_LON-1, '01')
    assert len(filenames) == 1 and filenames[0].endswith(expected_filename)

    filenames = obj.find_dem_files(MIN_LAT+1, MIN_LON)
    assert len(filenames) == 2

    filenames = obj.find_dem_files(MIN_LAT, MIN_LON+1)
    assert len(filenames) == 2

    filenames = obj.find_dem_files(MIN_LAT+1, MIN_LON + 1)
    assert len(filenames) == 4

    # Zero files because the low_res dataset does not exist in dem_file_path
    obj = GeoTIFF1DegList(str(dem_file_path), dataset_ff["low_res"])
    filenames = obj.find_dem_files(MIN_LAT, MIN_LON)
    assert len(filenames) == 0


def test_file_list(dem_file_path):
    obj = GeoTIFF1DegList(str(dem_file_path), dataset_ff["high_res"])

    # Zero files
    lat_lon_box = [MIN_LAT - 10, MIN_LAT - 9, MIN_LON - 10, MIN_LON - 9]
    filenames = obj.get_file_list(lat_lon_box)
    assert len(filenames) == 0

    # Single file
    lat_lon_box = [MIN_LAT + 0.3, MIN_LAT + 0.6, MIN_LON + 0.2, MIN_LON + 0.5]
    filenames = obj.get_file_list(lat_lon_box)
    assert len(filenames) == 1

    # All files near the prime meridian
    lat_lon_box = [MIN_LAT + 0.3, MAX_LAT - 0.5, MIN_LON + 0.2, MAX_LON - 0.5]
    filenames = obj.get_file_list(lat_lon_box)
    assert len(filenames) == 30

    # All files near the Antimeridian
    lat_lon_box = [MIN_LAT + 0.3, MAX_LAT - 0.5, 180 + MIN_LON + 0.2, MAX_LON - 180 - 0.5]
    filenames = obj.get_file_list(lat_lon_box)
    assert len(filenames) == 30

    # 360 degrees of longitude
    lat_lon_box = [MIN_LAT + 0.3, MAX_LAT - 0.5, 0.1, -0.1]
    filenames = obj.get_file_list(lat_lon_box)
    assert len(filenames) == 60


def test_exceptions(dem_file_path, caplog):
    with pytest.raises(ValueError, match="^The top level directory \\(/bad_dir\\) does not exist\\."):
        GeoTIFF1DegList("/bad_dir", dataset_ff["high_res"])

    obj = GeoTIFF1DegList(str(dem_file_path), dataset_ff["high_res"])
    with pytest.raises(ValueError) as info:
        obj.find_dem_files(999, 999)

    msgs = str(info.value).split('\n')
    assert len(msgs) == 2
    assert info.match("The latitude value must be between \\[-90, \\+90\\]")
    assert info.match("The longitude value must be between \\[-180, \\+180\\)")

    with pytest.raises(ValueError) as info:
        obj.get_file_list([999, 999, 999, 999])

    msgs = str(info.value).split('\n')
    assert len(msgs) == 4
    assert info.match("The minimum latitude value must be between \\[-90, \\+90\\]")
    assert info.match("The maximum latitude value must be between \\[-90, \\+90\\]")
    assert info.match("The minimum longitude value must be between \\[-180, \\+180\\)")
    assert info.match("The maximum longitude value must be between \\[-180, \\+180\\)")

    caplog.set_level(logging.WARNING)
    obj.get_file_list([45.1, 45.3, 90.1, 90.3])
    assert caplog.text.startswith("WARNING  sarpy.io.DEM.geotiff1deg:geotiff1deg.py")
    assert "Missing expected DEM file for tile with lower left lat/lon corner (45.0, 90.0)" in caplog.text

    obj = GeoTIFF1DegList(str(dem_file_path), dataset_ff["high_res"], missing_error=True)
    with pytest.raises(ValueError,
                       match="^Missing expected DEM file for tile with lower left lat/lon corner \\(45.0, 90.0\\)"):
        obj.get_file_list([45.1, 45.3, 90.1, 90.3])
