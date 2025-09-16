import pathlib

import pytest

import sarpy.io.DEM.DTED as sarpy_dted
from sarpy.io.DEM.geoid import GeoidHeight

import tests
# Note
# set this for your storage of dted and egm files
# export SARPY_TEST_PATH=<your dem stuff path>

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

@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader_south_west():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][0]) # belive wants the s file
 
    # From entity ID: SRTM3S04W061V1, date updated: 2013-04-17T12:16:47-05
    # Acquired from https://earthexplorer.usgs.gov/ on 2024-08-21
    # to follow along in qgis
    # know_value is one of known_values index
    # qgis row = 1200 - known_value[ 1 ]   # dted1 data in 1200 blocks
    # qgis col =  known_value[ 0 ]
    known_values = {
        (1000, 800): -32767,  # null 
        (1000, 799):  7,
        (3, 841):    -5,
        (1004, 797):  7,     # a value among the voids displayed in qgis
    }
    for index, expected_value in known_values.items():
        assert dted_reader[index] == expected_value
 
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader_north_west():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][1]) # belive wants the northern  file
 
    # From entity ID: SRTM3N33W119V1, date updated: 2013-04-17T12:16:47-05
    # Acquired from https://earthexplorer.usgs.gov/ on 2024-08-21
    # to follow along in qgis
    # know_value is one of known_values index
    # qgis row = 1200 - known_value[ 1 ]   # dted1 data in 1200 blocks
    # qgis col =  known_value[ 0 ]
    known_values = {
        (812, 927): -32767,  # null 
        (813, 927): -32767,  # null 
        (811, 927):  79,     # a value among the voids displayed in qgi
        (813, 926):  110     # a value among the voids displayed in qgi
    }
    for index, expected_value in known_values.items():
        assert dted_reader[index] == expected_value
 
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader_north_east():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][3]) # belive wants the northern  file in Nepal
 
    # From entity ID: SRTM3N27E084V1, date updated: 2005-02-01 00:00:00-06
    # Acquired from https://earthexplorer.usgs.gov/ on 2024-08-21
    # to follow along in qgis
    # known_value is one of known_values/index
    # qgis row = 1200 - known_value[ 1 ]   # dted1 data in 1200 blocks
    # qgis col =  known_value[ 0 ]
    known_values = {
        (927, 681): -32767,  # null /void  Nepal : 27.56751, 84.77241   [ lat/lon ]
        (928, 681):  830,    # a value east the void displayed in qgis
        (927, 680):  756     # a value south the void displayed in qgis
    }
    for index, expected_value in known_values.items():
        assert dted_reader[index] == expected_value

@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader_south_east():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][4]) # belive wants the Austrial
 
    # From entity ID:  SRTM3S36E149V1, date updated: 2005-02-01 00:00:00-06  Austrialia south west of Sydney
    # Acquired from https://earthexplorer.usgs.gov/ on 2025-08-28
    # to follow along in qgis
    # know_value is one of known_values index
    # qgis row = 1200 - known_value[ 1 ]   # dted1 data in 1200 blocks
    # qgis col =  known_value[ 0 ]
    known_values = {
        (547,  649): -32767,  # null near  -35.45881, 149.45613
        (547, 648):  752,     # a value south of void displayable via QGIS
        (546, 649):  756,     # a value west of void displayable via QGIS
    }
    for index, expected_value in known_values.items():
        assert dted_reader[index] == expected_value
 
 
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader_get_elevation_northern_pt():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][1]) # the northern  file
    llbx =  [ 33.3748, -118.4187 ]
    assert dted_reader.get_elevation( llbx[ 0], llbx[ 1 ]) ==  pytest.approx( 588.23, abs=0.01 )
 

@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader_get_elevation_northern_box():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][1]) # the northern  file
 
    lats = [ 33.3748,    33.405 ]
    lons = [ -118.4187, -118.4027 ]
    assert dted_reader.get_elevation( lats, lons ) ==  pytest.approx( [ 588.2368, 468.36  ] , abs=0.01 )
 
 
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader_get_elevation_southern_pt ():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][2]) # second souther file
    llbx = [ -1.0, -70.0 ] # lat long,
    assert dted_reader.get_elevation( llbx[ 0], llbx[ 1 ]) ==  pytest.approx( 145.0, abs=0.01 )
 
  
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_reader_get_elevation_southern_box ():
    dted_reader = sarpy_dted.DTEDReader(test_data["dted_with_null"][2]) # second southern file
 
    lats = [ -0.92, -0.90 ]
    lons = [ -69.9, -69.8 ]
    assert dted_reader.get_elevation( lats, lons ) ==  pytest.approx( [ 81.0, 111.0  ] , abs=0.1 )
 
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_interpolator_get_elevation_hae_north_west():
    ll = [ 33.3748, -118.4187 ]  # catinlia island off California coast
    geoid = GeoidHeight(egm96_file)
    files = test_data["dted_with_null"][1]  #  dem/dted/n33_w119_3arc_v1.dt1
    dem_interpolator = sarpy_dted.DTEDInterpolator(files=files, geoid_file=geoid, lat_lon_box=ll)
    assert dem_interpolator.get_elevation_hae(ll[0], ll[1]) == pytest.approx( 551.87, abs=0.01 )
 
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_interpolator_get_elevation_hae_north_east():
    ll = [ 27.57071, 84.77881 ] # Nepal  Near void used above
    geoid = GeoidHeight(egm96_file)
    files = test_data["dted_with_null"][3]  #  dem/dted/n27_e084_3arc_v1.dt1`
    dem_interpolator = sarpy_dted.DTEDInterpolator(files=files, geoid_file=geoid, lat_lon_box=ll)
    assert dem_interpolator.get_elevation_hae(ll[0], ll[1]) == pytest.approx( 954.70, abs=0.01 )

 
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_interpolator_get_elevation_hae_south_west():
    ll = [-1, -70]  # to get this point on -1,-70 tile with a tighter tolerance  From kjurka  Apr 29, 2025 github issue #587
    geoid = GeoidHeight(egm96_file)
    files = test_data["dted_with_null"][2]  #  dem/dted/s01_w070_3arc_v1.dt1'
    dem_interpolator = sarpy_dted.DTEDInterpolator(files=files, geoid_file=geoid, lat_lon_box=ll)
    assert dem_interpolator.get_elevation_hae(ll[0], ll[1]) ==  pytest.approx( 159.98, abs=0.01 )
 
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_interpolator_get_elevation_hae_south_west():
    ll = [ -35.4237, 149.5331 ]  # Austrialia, south west of Sydney, this point is north east of the void used above in the reader test
    geoid = GeoidHeight(egm96_file)
    files = test_data["dted_with_null"][4]  #  dem/dted/s36_e149_3arc_v1.dt1'
    dem_interpolator = sarpy_dted.DTEDInterpolator(files=files, geoid_file=geoid, lat_lon_box=ll)
    assert dem_interpolator.get_elevation_hae(ll[0], ll[1]) ==  pytest.approx( 1283.93, abs=0.01 )


@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_dted_interpolator_get_elevation_hae_south_east_cross_equator():
    ll = [ -0.01, -70.0 ] # lat long,
    geoid = GeoidHeight(egm96_file)
    files = test_data["dted_with_null"][2] 
    dem_interpolator = sarpy_dted.DTEDInterpolator.from_reference_point( ll, files, geoid_file=geoid, pad_value=1.0 )
    assert dem_interpolator.get_elevation_hae(ll[0], ll[1]) ==  pytest.approx( 13.53, abs=0.01 )
