"""
These test functions will exercise the GeoTIFF1DegList class which contains methods used to determine
which GeoTIFF files are needed to cover a specified geodetic bounding box.  These tests create dummy,
temporary, DEM files on-the-fly, so there is no need to provide any actual DEM files.

"""
import logging
import pathlib
import re
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


def infer_filename_format(root_dir_path):
    """
    This is a helper function used to generate a dem_filename_pattern string without explicit
    knowledge of the DEM filenames.  It assumes the Lat/Lon is encoded in the filename using
    a string like his: {NS}{abslat:02}{EW}{abslon:03}.
    """
    lat_lon_regex = '(n|s)([0-8][0-9])(e|w)((0[0-9][0-9])|(1[0-7][0-9])|180)'
    lat_lon_munge = [{'fmt_str': '{ns}{abslat:02}{ew}{abslon:03}', 'regex': lat_lon_regex},
                     {'fmt_str': '{NS}{abslat:02}{EW}{abslon:03}', 'regex': lat_lon_regex.upper()}]

    filename_formats = []

    tiff_filenames = [str(f) for f in root_dir_path.glob("**/*DEM.tif")]

    if len(tiff_filenames) == 0:
        raise FileNotFoundError(f"Could not find any TIFF files in ({str(root_dir_path)}).")

    for tiff_filename in tiff_filenames:
        munged_filename = tiff_filename
        for munge in lat_lon_munge:
            munged_filename = re.sub(munge['regex'], munge['fmt_str'], munged_filename)

        if munged_filename == tiff_filename:
            raise ValueError(f"Could not find a Lat/Lon substring in filename ({tiff_filename}).")

        if munged_filename not in filename_formats:
            filename_formats.append(munged_filename)

    fmt_ref = list(filename_formats[0])
    for tst_str in filename_formats[1:]:
        fmt_tst = list(tst_str)
        if len(fmt_ref) != len(fmt_tst):
            raise ValueError('Format string lengths do not match')
        for i, (c0, c1) in enumerate(zip(fmt_ref, fmt_tst)):
            if c0 != c1:
                fmt_ref[i] = '?'

    filename_format = ''.join(fmt_ref)
    return filename_format


@pytest.fixture(scope='module')
def dem_file_path():
    """
    Create a directory of empty files that satisfy the DEM naming convention.
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
    obj = GeoTIFF1DegList('dummy_format')

    # Test fully specified filename_format
    filename_format = "Test_{lat:02d}_{lon:03d}_{abslat:02d}_{abslon:03d}_{ns:1s}{NS:1s}_{ew:1s}{EW:1s}"
    filename = obj.filename_from_lat_lon(-1, -2, filename_format)
    assert filename == "Test_-1_-02_01_002_sS_wW"

    # Test fully specified filename_format with an extra {*}
    filename_format = "Test_{lat:02d}_{lon:03d}_{abslat:02d}_{abslon:03d}_{ns:1s}{NS:1s}_{ew:1s}{EW:1s}_{bad}"
    filename = obj.filename_from_lat_lon(-1, -2, filename_format)
    assert filename == "Test_-1_-02_01_002_sS_wW_{bad}"

    # Test filename_format where field width is omitted for single character fields
    filename_format = "Test_{lat:02d}_{lon:03d}_{abslat:02d}_{abslon:03d}_{ns}{NS}_{ew}{EW}"
    filename = obj.filename_from_lat_lon(-1, -2, filename_format)
    assert filename == "Test_-1_-02_01_002_sS_wW"


def test_find_dem_files(dem_file_path):
    filename_format = infer_filename_format(dem_file_path)

    obj = GeoTIFF1DegList(filename_format)

    filenames = obj.find_dem_files(MIN_LAT - 1, MIN_LON - 1)
    assert len(filenames) == 0

    filenames = obj.find_dem_files(MIN_LAT, MIN_LON)
    expected_filename = obj.filename_from_lat_lon(MIN_LAT, MIN_LON, filename_format).replace('?', '1')
    assert len(filenames) == 1 and filenames[0].endswith(expected_filename)

    filenames = obj.find_dem_files(MIN_LAT + 0.5, MIN_LON + 0.5)
    assert len(filenames) == 1 and filenames[0] == expected_filename

    filenames = obj.find_dem_files(MAX_LAT, MAX_LON)
    expected_filename = obj.filename_from_lat_lon(MAX_LAT-1, MAX_LON-1, filename_format).replace('?', '1')
    assert len(filenames) == 1 and filenames[0] == expected_filename

    filenames = obj.find_dem_files(MIN_LAT+1, MIN_LON)
    assert len(filenames) == 2

    filenames = obj.find_dem_files(MIN_LAT, MIN_LON+1)
    assert len(filenames) == 2

    filenames = obj.find_dem_files(MIN_LAT+1, MIN_LON + 1)
    assert len(filenames) == 4


def test_file_list(dem_file_path):
    filename_format = infer_filename_format(dem_file_path)

    obj = GeoTIFF1DegList(filename_format)

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
    filename_format = infer_filename_format(dem_file_path)

    obj = GeoTIFF1DegList(filename_format)
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

    obj = GeoTIFF1DegList(filename_format, missing_error=True)
    with pytest.raises(ValueError,
                       match="^Missing expected DEM file for tile with lower left lat/lon corner \\(45.0, 90.0\\)"):
        obj.get_file_list([45.1, 45.3, 90.1, 90.3])
