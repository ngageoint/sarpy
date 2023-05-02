"""
These test functions will exercise the GeoTIFF1DegInterpolator class which reads DEM data files and interpolates
data points as needed.  Most of these tests use fabricated data, so the GeoTIFF1DegInterpolator class is moderately
well tests without providing external DEM data.  However, when available, real DEM data is used for some tests.
The location of the real DEM data files and real Geoid data files are defined by the parameters:

    GEOTIFF_ROOT_PATH        - A pathlib.Path to the root directory containing DEM data GeoTIFF files

    Low resolution GeoTIFF data files can be downloaded from here:
        https://download.geoservice.dlr.de/TDM90/

    High resolution GeoTIFF data files are typically restricted, but more information can be found here:
        https://data.europa.eu/data/datasets/5eecdf4c-de57-4624-99e9-60086b032aea?locale=en

    The Geoid model files are available in either ZIP of BZ2 format from here:
        https://sourceforge.net/projects/geographiclib/files/geoids-distrib/

If real DEM files and/or real Geoid files are not available then tests that require these files will be skipped.

"""
import json
import os
import pathlib
import re
import tempfile

import numpy as np
from PIL import Image
import pytest

from sarpy.io.DEM.geotiff1deg import GeoTIFF1DegInterpolator

SRC_FILE_PATH = pathlib.Path(__file__).parent
parent_path = os.environ.get('SARPY_TEST_PATH', None)
if parent_path == 'NONE':
    parent_path = None
if parent_path is None:
    GEOTIFF_ROOT_PATH = SRC_FILE_PATH / "dem_data"
if parent_path is not None:
    parent_path = pathlib.Path(os.path.expanduser(parent_path))
    GEOTIFF_ROOT_PATH = pathlib.Path(parent_path, "dem")


# The geoid.json file, used by test_geoid, is reused here to define the location of the geoid files.
geoid_json_path = SRC_FILE_PATH / "geoid.json"
geoid_file_info = json.loads(geoid_json_path.read_text())
geoid_path_type = geoid_file_info['geoid_files'][-1]['path_type']
geoid_path_sufx = geoid_file_info['geoid_files'][-1]['path']
if parent_path is None:
    GEOID_FILE_PATH = SRC_FILE_PATH / geoid_path_sufx if geoid_path_type == 'relative' else pathlib.Path(geoid_path_sufx)
elif parent_path:
    GEOID_FILE_PATH = parent_path / geoid_path_sufx if geoid_path_type == 'relative' else pathlib.Path(
        geoid_path_sufx)
NUM_LATS_DUMMY = 201
NUM_LONS_DUMMY = 101


def lat_lon_to_dummy_height(lat, lon):
    return lat + 1000 * lon


def lat_lon_from_filename(filename):
    m = re.search('(N|S)([0-8][0-9])(E|W)((0[0-9][0-9])|(1[0-7][0-9])|180)', filename.upper())
    if m is None:
        raise ValueError(f"Could not find a Lat/Lon substring in filename ({filename}).")

    lat_sgn = 1 if m.group(1) == 'N' else -1
    lat_abs = int(m.group(2))
    lon_sgn = 1 if m.group(3) == 'E' else -1
    lon_abs = int(m.group(4))

    return lat_sgn * lat_abs, lon_sgn * lon_abs


def infer_filename_format(root_dir_path):
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


def dummy_pil_image_open(filename, ref_surface):
    """
    This function is intended to be a monkeypatch target for PIL.Image.open().  It returns a PIL.Image
    object of dummy DEM values.  The tile's SW corner Lat/Lon value is extracted from the filename.
    The DEM height values are a linear function of the Lat/Lon value so that interpolated
    values of the DEM are predictable and easily compared to the expected result.  The filename
    is assumed to be in "high_res" format.
    """
    lat, lon = lat_lon_from_filename(str(filename))

    lat_values = np.linspace(lat+1, lat, NUM_LATS_DUMMY)    # Increasing axis-0 index is decreasing latitude
    lon_values = np.linspace(lon, lon+1, NUM_LONS_DUMMY)    # Increasing axis-1 index is increasing longitude
    lon_mat, lat_mat = np.meshgrid(lon_values, lat_values)
    heights = lat_lon_to_dummy_height(lat_mat, lon_mat)

    im = Image.fromarray(heights.astype(np.float64))
    im.tag = {256: (NUM_LONS_DUMMY,),             # ImageWidth
              257: (NUM_LATS_DUMMY,),             # ImageLength
              34737: (f"Dummy: {ref_surface}",)}  # GeoAsciiParamsTag

    return im


@pytest.fixture(scope='module')
def dummy_dem_file_path_high_res():
    dataset = 'high_res'
    filename_format = ["tdt_{ns}{abslat:02}{ew}{abslon:03}_{ver:2s}", "DEM",
                       "TDT_{NS}{abslat:02}{EW}{abslon:03}_{ver:2s}_DEM.tif"]
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = pathlib.Path(temp_dir) / dataset
        dummy_dem_file_path(root_path, filename_format)
        yield root_path


@pytest.fixture(scope='module')
def dummy_dem_file_path_low_res():
    dataset = 'low_res'
    filename_format = ["TDM1_DEM__30_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_V{ver:2s}_C", "DEM",
                       "TDM1_DEM__30_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_DEM.tif"]
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = pathlib.Path(temp_dir) / dataset
        dummy_dem_file_path(root_path, filename_format)
        yield root_path


def dummy_dem_file_path(root_path, filename_format):
    """
    Create a directory of empty files that satisfy the DEM naming convention.
    """
    min_lat = -1
    max_lat = +2
    min_lon = -1
    max_lon = +2
    ver_choice = ('01', '02')

    for lat in range(min_lat, max_lat):
        for lon in range(min_lon, max_lon):
            anti_lon = (lon + 360) % 360 - 180
            # Make empty files that span the prime meridian and the equator,
            # then make more empty files that span the anti-meridian and the equator.
            for xlon, ver in zip([lon, anti_lon], ver_choice):
                pars = {"abslat": int(abs(np.floor(lat))), "abslon": int(abs(np.floor(xlon))),
                        "ns": 's' if lat < 0 else 'n', "ew": 'w' if xlon < 0 else 'e',
                        "NS": 'S' if lat < 0 else 'N', "EW": 'W' if xlon < 0 else 'E', "ver": ver}
                filename = root_path / os.path.join(*filename_format).format_map(pars)
                filename.parent.mkdir(parents=True, exist_ok=True)
                filename.touch()


def lat_lon_bounds(root_dir_path):
    """
    Determine the min/max lat/lon covered by DEM files in a specified directory.
    """
    filenames = list(root_dir_path.glob("**/*DEM.tif"))
    lats = np.zeros(len(filenames))
    lons = np.zeros(len(filenames))
    for i, filename in enumerate(filenames):
        lats[i], lons[i] = lat_lon_from_filename(str(filename))

    sw_lat = np.min(lats)
    ne_lat = np.max(lats) + 1
    sw_lon = np.min(lons)
    ne_lon = np.max(lons) + 1

    return [sw_lat, ne_lat, sw_lon, ne_lon]


def test_setter_getter(dummy_dem_file_path_high_res):
    obj = GeoTIFF1DegInterpolator('/dummy', interp_method="smarty")
    assert obj.interp_method == "smarty"
    obj.interp_method = "dummy"
    assert obj.interp_method == "dummy"


@pytest.mark.parametrize("dataset", [
    pytest.param("high_res", marks=pytest.mark.skipif(not (GEOTIFF_ROOT_PATH / "high_res").exists(),
                                                      reason="GeoTIFF test data does not exist.")),
    pytest.param("low_res", marks=pytest.mark.skipif(not (GEOTIFF_ROOT_PATH / "low_res").exists(),
                                                     reason="GeoTIFF test data does not exist."))])
def test_get_elevation_native(dataset):
    root_dir_path = GEOTIFF_ROOT_PATH / dataset
    sw_lat, ne_lat, sw_lon, ne_lon = lat_lon_bounds(root_dir_path)
    filename_format = infer_filename_format(root_dir_path)

    obj = GeoTIFF1DegInterpolator(filename_format)

    num_points = 8
    d_offset = 1 / 18111    # Offset used to avoid exact Lat/Lon samples which occur ever 1/9000

    # All test points are outside valid DEM area
    lats = np.linspace(sw_lat - 0.9, sw_lat - 0.1, num_points)
    lons = np.linspace(sw_lon - 0.9, sw_lon - 0.1, num_points)
    hgts = obj.get_elevation_native(lats, lons)
    assert np.all(np.equal(hgts, np.zeros(num_points)))

    # All test points are inside valid DEM area and are inside a single tile
    lats = np.linspace(sw_lat + d_offset, sw_lat + (1 - d_offset), num_points)
    lons = np.linspace(sw_lon + d_offset, sw_lon + (1 - d_offset), num_points)
    hgts = obj.get_elevation_native(lats, lons)
    assert not np.all(np.equal(hgts, np.zeros(num_points)))

    # Test all point are inside valid DEM area and span several tiles
    lats = np.linspace(sw_lat + d_offset, ne_lat - d_offset, num_points)
    lons = np.linspace(sw_lon + d_offset, ne_lon - d_offset, num_points)
    hght = obj.get_elevation_native(lats, lons)
    assert not np.all(np.equal(hght, np.zeros(num_points)))

    # Test scale lat / lon arguments
    hght = obj.get_elevation_native(lats[0], lons[0])
    assert not np.all(np.equal(hght, np.zeros(1)))


def test_get_elevation_native_dummy(dummy_dem_file_path_high_res, monkeypatch):
    monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "EGM2008"))

    sw_lat = -1
    ne_lat = 2
    sw_lon = -1
    ne_lon = 2

    filename_format = infer_filename_format(dummy_dem_file_path_high_res)
    obj = GeoTIFF1DegInterpolator(filename_format)

    num_points = 8
    d_offset = 1 / 18000

    lats = np.linspace(sw_lat + d_offset, ne_lat - d_offset, num_points)
    lons = np.linspace(sw_lon + d_offset, ne_lon/2 - d_offset, num_points)
    hght = obj.get_elevation_native(lats, lons)
    expected_hght = lat_lon_to_dummy_height(lats, lons)
    assert np.allclose(hght, expected_hght)


def test_get_min_max_native_dummy(dummy_dem_file_path_high_res, monkeypatch):
    monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "EGM2008"))

    sw_lat = -1
    ne_lat = 2
    sw_lon = -1
    ne_lon = 2

    lat_ss = 1 / (NUM_LATS_DUMMY - 1)
    lon_ss = 1 / (NUM_LONS_DUMMY - 1)

    def assert_result_good(pars, box):
        def lt_or_close(lower, upper):
            return lower < upper or np.isclose(lower, upper)

        assert pars['box'] == box
        assert pars['ref_surface'] == 'geoid'

        assert lt_or_close(box[0], pars['min']['lat']) and lt_or_close(pars['min']['lat'], box[0] + lat_ss)
        assert lt_or_close(box[2], pars['min']['lon']) and lt_or_close(pars['min']['lon'], box[2] + lon_ss)
        assert (lt_or_close(lat_lon_to_dummy_height(box[0], box[2]), pars['min']['height']) and
                lt_or_close(pars['min']['height'], lat_lon_to_dummy_height(box[0] + lat_ss, box[2] + lon_ss)))

        assert lt_or_close(box[1] - lat_ss, pars['max']['lat']) and lt_or_close(pars['max']['lat'], box[1])
        assert lt_or_close(box[3] - lon_ss, pars['max']['lon']) and lt_or_close(pars['max']['lon'], box[3])
        assert (lt_or_close(lat_lon_to_dummy_height(box[1] - lat_ss, box[3] - lon_ss), pars['max']['height']) and
                lt_or_close(pars['max']['height'], lat_lon_to_dummy_height(box[1], box[3] + lon_ss)))

    filename_format = infer_filename_format(dummy_dem_file_path_high_res)
    obj = GeoTIFF1DegInterpolator(filename_format)

    # Test bounding box outside the DEM tiles
    lat_lon_bounding_box = [sw_lat-10, ne_lat - 10, sw_lon - 10, ne_lon - 10]
    result = obj.get_min_max_native(lat_lon_bounding_box)
    expected_result = {'box': lat_lon_bounding_box,
                       'ref_surface': 'geoid',
                       'min': {'lat': lat_lon_bounding_box[0], 'lon': lat_lon_bounding_box[2], 'height': 0.0},
                       'max': {'lat': lat_lon_bounding_box[0], 'lon': lat_lon_bounding_box[2], 'height': 0.0}}
    assert result == expected_result

    # All test points are inside valid DEM area and are inside a single tile
    lat_min = sw_lat + 0.111111
    lat_max = sw_lat + 0.811111
    lon_min = sw_lon + 0.211111
    lon_max = sw_lon + 0.711111
    result = obj.get_min_max_native([lat_min, lat_max, lon_min, lon_max])
    assert_result_good(result, [lat_min, lat_max, lon_min, lon_max])

    # Test all point are inside valid DEM area and span several tiles
    lat_min = sw_lat + 0.111
    lat_max = sw_lat + 1.811
    lon_min = sw_lon + 0.211
    lon_max = sw_lon + 1.711
    result = obj.get_min_max_native([lat_min, lat_max, lon_min, lon_max])
    assert_result_good(result, [lat_min, lat_max, lon_min, lon_max])

    # Exercise the bounding_box_cache
    result = obj.get_min_max_native([lat_min, lat_max, lon_min, lon_max])
    assert_result_good(result, [lat_min, lat_max, lon_min, lon_max])


@pytest.mark.parametrize("ref_surface", [
    pytest.param("egm2008", marks=pytest.mark.skipif(not GEOID_FILE_PATH.exists(), reason="Geoid data does not exist")),
    pytest.param("wgs84", marks=pytest.mark.skipif(not GEOID_FILE_PATH.exists(), reason="Geoid data does not exist"))])
def test_get_elevation_hae_geoid(ref_surface, dummy_dem_file_path_high_res, monkeypatch):
    if ref_surface == "egm2008":
        monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "EGM2008"))
    else:
        monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "WGS84"))

    sw_lat = -1
    ne_lat = 2
    sw_lon = -1
    ne_lon = 2

    filename_format = infer_filename_format(dummy_dem_file_path_high_res)
    obj = GeoTIFF1DegInterpolator(filename_format, str(GEOID_FILE_PATH.parent.parent))

    num_points = 8
    d_offset = 1 / 18000

    lats = np.linspace(sw_lat + d_offset, ne_lat - d_offset, num_points)
    lons = np.linspace(sw_lon + d_offset, ne_lon/2 - d_offset, num_points)
    hght_wgs84 = obj.get_elevation_hae(lats, lons)
    hght_geoid = obj.get_elevation_geoid(lats, lons)
    assert np.all(np.abs(hght_wgs84 - hght_geoid) > 0)

    min_hae = obj.get_min_hae([lats[0], lats[-1], lons[0], lons[-1]])
    max_hae = obj.get_max_hae([lats[0], lats[-1], lons[0], lons[-1]])
    min_geoid = obj.get_min_geoid([lats[0], lats[-1], lons[0], lons[-1]])
    max_geoid = obj.get_max_geoid([lats[0], lats[-1], lons[0], lons[-1]])
    assert np.all(np.abs(min_hae - min_geoid))
    assert np.all(np.abs(max_hae - max_geoid))

    obj2 = GeoTIFF1DegInterpolator(filename_format, str(GEOID_FILE_PATH))
    hght_geoid2 = obj2.get_elevation_geoid(lats, lons)
    assert np.all(hght_geoid2 == hght_geoid)


def test_exceptions(dummy_dem_file_path_high_res, monkeypatch):
    filename_format = infer_filename_format(dummy_dem_file_path_high_res)
    obj = GeoTIFF1DegInterpolator(filename_format)
    with pytest.raises(ValueError, match="^The lat and lon arrays are not the same shape\\."):
        obj.get_elevation_native([1, 2, 3], [1, 2])

    monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "WGS84"))
    obj = GeoTIFF1DegInterpolator(filename_format)
    with pytest.raises(ValueError,
                       match="^The geoid_dir parameter was not defined so geoid calculations are disabled\\."):
        obj.get_elevation_geoid(1, 1)

    monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "EGM2008"))
    obj = GeoTIFF1DegInterpolator(filename_format)
    with pytest.raises(ValueError,
                       match="^The geoid_dir parameter was not defined so geoid calculations are disabled\\."):
        obj.get_elevation_hae(1, 1)

    monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "Unknown"))
    obj = GeoTIFF1DegInterpolator(filename_format)
    with pytest.raises(ValueError, match="^The reference surface is Unknown, which is not supported"):
        obj.get_elevation_geoid(1, 1)
    with pytest.raises(ValueError, match="^The reference surface is Unknown, which is not supported"):
        obj.get_elevation_hae(1, 1)
