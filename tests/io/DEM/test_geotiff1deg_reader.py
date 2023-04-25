"""
These test functions will exercise the GeoTIFF1DegInterpolator class which reads DEM data files and interpolates
data points as needed.  Most of these tests use fabricated data, so the GeoTIFF1DegInterpolator class is moderately
well tests without providing external DEM data.  However, when available, real DEM data is used for some tests.
The location of the real DEM data files and real Geoid data files are defined by the parameters:

    GEOTIFF_ROOT_PATH        - A pathlib.Path to the root directory containing DEM data files in GeoTIFF
    GEOID_ROOT_PATH          - A pathlib.Path to the root directory containing Geoid data files in PGM format
    FILENAME_FORMAT_HIGH_RES - A format string for a high-res DEM filenames relative to GEOTIFF_ROOT_PATH
    FILENAME_FORMAT_LOW_RES  - A format string for a low-res DEM filenames relative to GEOTIFF_ROOT_PATH

    Low resolution GeoTIFF data files can be downloaded from here:
        https://download.geoservice.dlr.de/TDM90/

    High resolution GeoTIFF data files are typically restricted, but more information can be found here:
        https://data.europa.eu/data/datasets/5eecdf4c-de57-4624-99e9-60086b032aea?locale=en

    The Geoid model files are available in either ZIP of BZ2 format from here:
        https://sourceforge.net/projects/geographiclib/files/geoids-distrib/

If real DEM files and/or real Geoid files are not available then tests that require these files will be skipped.

"""
import os
import pathlib
import re
import tempfile

import numpy as np
from PIL import Image
import pytest

from sarpy.io.DEM.geotiff1deg import GeoTIFF1DegInterpolator

GEOTIFF_ROOT_PATH = pathlib.Path(__file__).parent / "dem_data"
GEOID_ROOT_PATH = pathlib.Path(__file__).parent / "dem"

FILENAME_FORMAT_HIGH_RES = ["tdt_{ns:1s}{abslat:02}{ew:1s}{abslon:03}_{ver:2s}", "DEM",
                            "TDT_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_{ver:2s}_DEM.tif"]
FILENAME_FORMAT_LOW_RES = ["TDM1_DEM__30_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_V{ver:2s}_C", "DEM",
                           "TDM1_DEM__30_{NS:1s}{abslat:02}{EW:1s}{abslon:03}_DEM.tif"]
FILENAME_FORMAT_DICT = {"high_res": FILENAME_FORMAT_HIGH_RES, "low_res": FILENAME_FORMAT_LOW_RES}

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
    filename_format = FILENAME_FORMAT_HIGH_RES
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = pathlib.Path(temp_dir) / dataset
        dummy_dem_file_path(root_path, filename_format)
        yield root_path


@pytest.fixture(scope='module')
def dummy_dem_file_path_low_res():
    dataset = 'low_res'
    filename_format = FILENAME_FORMAT_LOW_RES
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
            # then make more empty files that span the antimeridian and the equator.
            for xlon, ver in zip([lon, anti_lon], ver_choice):
                pars = {"abslat": int(abs(np.floor(lat))), "abslon": int(abs(np.floor(xlon))),
                        "ns": 's' if lat < 0 else 'n', "ew": 'w' if xlon < 0 else 'e',
                        "NS": 'S' if lat < 0 else 'N', "EW": 'W' if xlon < 0 else 'E', "ver": ver}
                filename = root_path / os.path.join(*filename_format).format_map(pars)
                filename.parent.mkdir(parents=True, exist_ok=True)
                filename.touch()


def lat_lon_bounds(root_dir):
    """
    Determine the min/max lat/lon covered by DEM files in a specified directory.
    """
    filenames = list(pathlib.Path(root_dir).glob("**/*DEM.tif"))
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
    obj = GeoTIFF1DegInterpolator(str(dummy_dem_file_path_high_res), FILENAME_FORMAT_HIGH_RES)
    obj.interp_method = "dummy"
    assert obj.interp_method == "dummy"


@pytest.mark.parametrize("dataset", [
    pytest.param("high_res", marks=pytest.mark.skipif(not (GEOTIFF_ROOT_PATH / "high_res").exists(),
                                                      reason="GeoTIFF test data does not exist.")),
    pytest.param("low_res", marks=pytest.mark.skipif(not (GEOTIFF_ROOT_PATH / "low_res").exists(),
                                                     reason="GeoTIFF test data does not exist."))])
def test_get_elevation_native(dataset):
    root_dir = str(GEOTIFF_ROOT_PATH / dataset)
    sw_lat, ne_lat, sw_lon, ne_lon = lat_lon_bounds(root_dir)

    obj = GeoTIFF1DegInterpolator(root_dir, FILENAME_FORMAT_DICT[dataset])

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

    obj = GeoTIFF1DegInterpolator(str(dummy_dem_file_path_high_res), FILENAME_FORMAT_HIGH_RES)

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

    obj = GeoTIFF1DegInterpolator(str(dummy_dem_file_path_high_res), FILENAME_FORMAT_HIGH_RES, "EGM2008")

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
    pytest.param("egm2008", marks=pytest.mark.skipif(not (GEOID_ROOT_PATH / "geoid" / 'egm2008-1.pgm').exists(),
                                                     reason="Geoid data does not exist.")),
    pytest.param("wgs84", marks=pytest.mark.skipif(not (GEOID_ROOT_PATH / "geoid" / 'egm2008-1.pgm').exists(),
                                                   reason="Geoid data does not exist."))])
def test_get_elevation_hae_geoid(ref_surface, dummy_dem_file_path_high_res, monkeypatch):
    if ref_surface == "egm2008":
        monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "EGM2008"))
    else:
        monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "WGS84"))

    sw_lat = -1
    ne_lat = 2
    sw_lon = -1
    ne_lon = 2

    obj = GeoTIFF1DegInterpolator(str(dummy_dem_file_path_high_res), FILENAME_FORMAT_HIGH_RES, str(GEOID_ROOT_PATH))

    num_points = 8
    d_offset = 1 / 18000

    lats = np.linspace(sw_lat + d_offset, ne_lat - d_offset, num_points)
    lons = np.linspace(sw_lon + d_offset, ne_lon/2 - d_offset, num_points)
    hght1 = obj.get_elevation_hae(lats, lons)
    hght2 = obj.get_elevation_geoid(lats, lons)
    assert np.all(np.abs(hght1 - hght2) > 0)

    min_hae = obj.get_min_hae([lats[0], lats[-1], lons[0], lons[-1]])
    max_hae = obj.get_max_hae([lats[0], lats[-1], lons[0], lons[-1]])
    min_geoid = obj.get_min_geoid([lats[0], lats[-1], lons[0], lons[-1]])
    max_geoid = obj.get_max_geoid([lats[0], lats[-1], lons[0], lons[-1]])
    assert np.all(np.abs(min_hae - min_geoid))
    assert np.all(np.abs(max_hae - max_geoid))


def test_exceptions(dummy_dem_file_path_high_res, monkeypatch):
    obj = GeoTIFF1DegInterpolator(str(dummy_dem_file_path_high_res), FILENAME_FORMAT_HIGH_RES)
    with pytest.raises(ValueError, match="^The lat and lon arrays are not the same shape\\."):
        obj.get_elevation_native([1, 2, 3], [1, 2])

    monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "WGS84"))
    obj = GeoTIFF1DegInterpolator(str(dummy_dem_file_path_high_res), FILENAME_FORMAT_HIGH_RES)
    with pytest.raises(ValueError,
                       match="^The geoid_dir parameter was not defined so geoid calculations are disabled\\."):
        obj.get_elevation_geoid(1, 1)

    monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "EGM2008"))
    obj = GeoTIFF1DegInterpolator(str(dummy_dem_file_path_high_res), FILENAME_FORMAT_HIGH_RES)
    with pytest.raises(ValueError,
                       match="^The geoid_dir parameter was not defined so geoid calculations are disabled\\."):
        obj.get_elevation_hae(1, 1)

    monkeypatch.setattr(Image, 'open', lambda filename: dummy_pil_image_open(filename, "Unknown"))
    obj = GeoTIFF1DegInterpolator(str(dummy_dem_file_path_high_res), FILENAME_FORMAT_HIGH_RES)
    with pytest.raises(ValueError, match="^The reference surface is Unknown, which is not supported"):
        obj.get_elevation_geoid(1, 1)
    with pytest.raises(ValueError, match="^The reference surface is Unknown, which is not supported"):
        obj.get_elevation_hae(1, 1)
