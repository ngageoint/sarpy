import os  # adding stuff for cross platform
import sys
import pathlib

import pytest

from sarpy.utils.dted_check_voids import check_for_voids

import sarpy.io.DEM.DTED as sarpy_dted

import tests


# SARPY_TEST_PATH should be set first
test_data = tests.find_test_data_files(pathlib.Path(__file__).parent / "geoid.json")

parent_path = os.environ.get('SARPY_TEST_PATH', None)

@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_check_for_voids_by_file_true():
    quickCheck = check_for_voids( test_data['dted_with_null'][4] ) 
    # just grab basename off filepath and rebuild the result dict
    # this will help with the cross platoform and personal preference 
    newResults = {}
    for key, val in quickCheck.items():
        newResults[ os.path.basename( key ) ] = val #  resutls[ basename/dted filename ] = <bool check>
    
    answer = { "s36_e149_3arc_v1.dt1": "True"}
    assert newResults == answer

    #
    # Do test again but return cells that have void data.
    quickCheck = check_for_voids( test_data['dted_with_null'][4], True)
    # just grab basename off filepath and rebuild the result dict
    # this will help with the cross platoform and personal preference 
    newResults = {}
    for key, val in quickCheck.items():
        newResults[ os.path.basename( key ) ] = val #  resutls[ basename/dted filename ] = <bool check>

    fullAnswer   = {"s36_e149_3arc_v1.dt1": {'indices': [(96, 97, 97, 97, 98, 98, 98, 99, 183, 183, 547, 547, 548, 548, 549, 549, 823, 823, 823, 1103, 1103, 1122, 1122, 1122, 1125, 1125, 1125, 1173, 1173, 1174), (657, 657, 658, 659, 657, 658, 659, 658, 224, 225, 653, 654, 653, 654, 653, 654, 45, 46, 47, 412, 413, 463, 464, 465, 463, 464, 465, 1126, 1127, 1127)], 'has_voids': 'True'}}
    assert newResults == fullAnswer


@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_check_for_voids_by_list_of_files_true():
    test_data      = tests.find_test_data_files(pathlib.Path(__file__).parent / "geoid.json")
    quickCheck = check_for_voids(   [ test_data['dted_with_null'][1], test_data['dted_with_null'][4] ] )
    # just grab basename off filepath and rebuild the result dict
    # this will help with the cross platoform and personal preference 
    newResults = {}
    for key, val in quickCheck.items():
        newResults[ os.path.basename( key ) ] = val #  resutls[ basename/dted filename ] = <bool check>
        
    quickAnswer = {
        "n33_w119_3arc_v1.dt1": "True",
        "s36_e149_3arc_v1.dt1": "True"
    }
    assert newResults == quickAnswer


@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_check_for_voids_by_dir_true():
    quickCheck = check_for_voids(  os.path.join( parent_path, "dem", "dted" )) 
    # just grab basename off filepath and rebuild the result dict
    # this will help with the cross platoform and personal preference 
    newResults = {}
    for key, val in quickCheck.items():
        newResults[ os.path.basename( key ) ] = val #  resutls[ basename/dted filename ] = <bool check>
        
    quickAnswer = {"n27_e084_3arc_v1.dt1": 'True', "n33_w119_3arc_v1.dt1": 'True', "n38_w078_3arc_v1.dt1": 'True', "n38_w078_3arc_v2.dt1": 'False', "s01_w070_3arc_v1.dt1": 'True', "s04_w061_3arc_v1.dt1": 'True', "s36_e149_3arc_v1.dt1": 'True'}
    # this is a set subset test
    # the dted dir can have subdirectories and/or other files in it
    #  we just want to verify our known files
    assert newResults.items() >= quickAnswer.items()

    
    
@pytest.mark.skipif(not test_data["dted_with_null"], reason="DTED with null data does not exist")
def test_check_for_voids_by_file_false():
    quickCheck = check_for_voids(  test_data['dted_with_null'][5] )
    # just grab basename off filepath and rebuild the result dict
    # this will help with the cross platoform and personal preference 
    newResults = {}
    for key, val in quickCheck.items():
        newResults[ os.path.basename( key ) ] = val #  resutls[ basename/dted filename ] = <bool check>
           
    quickAnswer = {"n38_w078_3arc_v2.dt1": 'False'}
    assert newResults == quickAnswer    
    
