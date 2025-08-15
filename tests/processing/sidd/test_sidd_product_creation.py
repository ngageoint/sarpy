import json
import os
import pytest
import unittest
from tests import parse_file_entry

from sarpy.io.complex.converter          import conversion_utility, open_complex
from sarpy.processing.ortho_rectify import NearestNeighborMethod
from sarpy.processing.sidd.sidd_product_creation import \
    create_detected_image_sidd, create_csi_sidd, create_dynamic_image_sidd
import sarpy.visualization.remap as remap

complex_file_types = {}
this_loc           = os.path.abspath(__file__)
# specifies file locations
file_reference     = os.path.join(os.path.split(this_loc)[0], \
                                  'complex_file_types.json')  
if os.path.isfile(file_reference):
    with open(file_reference, 'r') as local_file:
        test_files_list = json.load(local_file)
        for test_files_type in test_files_list:
            valid_entries = []
            for entry in test_files_list[test_files_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            complex_file_types[test_files_type] = valid_entries

sicd_files = complex_file_types.get('SICD', [])

def get_test_reader():
    input_file       = sicd_files[0]
    reader           = open_complex(input_file)
    return reader


@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_detected_image_sidd_required_params_only_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    test_sidd = create_detected_image_sidd(ortho_helper, output_directory)
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_detected_image_sidd_required_params_and_output_file_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    test_sidd = create_detected_image_sidd(ortho_helper, output_directory, output_file)
    
def test_create_detected_image_sidd_remap_function_fail(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    local_remap_function = 'bob'
    with pytest.raises(TypeError, 
                       match="remap_function must be an instance of " + \
                        "MonochromaticRemap"):
          test_sidd = create_detected_image_sidd(ortho_helper, output_directory, \
                                           output_file, \
                                            remap_function=local_remap_function)
    
def test_create_detected_image_sidd_remap_function_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    local_remap_function = remap.get_registered_remap('nrl')
    test_sidd = create_detected_image_sidd(ortho_helper, output_directory, \
                                           output_file, \
                                            remap_function=local_remap_function)
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_csi_sidd_required_params_only_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    test_sidd = create_csi_sidd(ortho_helper, output_directory)
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_csi_sidd_required_params_and_output_file_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    test_sidd = create_csi_sidd(ortho_helper, output_directory, output_file)
    
def test_create_csi_sidd_remap_function_fail(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    local_remap_function = 'bob'
    with pytest.raises(TypeError, 
                       match="remap_function must be an instance of " + \
                        "MonochromaticRemap"):
          test_sidd = create_csi_sidd(ortho_helper, output_directory, \
                                           output_file, \
                                            remap_function=local_remap_function)
    
def test_create_csi_sidd_sidd_remap_function_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    local_remap_function = remap.get_registered_remap('nrl')
    test_sidd = create_csi_sidd(ortho_helper, output_directory, \
                                           output_file, \
                                            remap_function=local_remap_function)

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_dynamic_image_sidd_required_params_only_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    test_sidd = create_dynamic_image_sidd(ortho_helper, output_directory)
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_dynamic_image_sidd_required_params_and_output_file_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    test_sidd = create_dynamic_image_sidd(ortho_helper, output_directory, output_file)
    
def test_create_dynamic_image_sidd_remap_function_fail(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    local_remap_function = 'bob'
    with pytest.raises(TypeError, 
                       match="remap_function must be an instance of " + \
                        "MonochromaticRemap"):
          test_sidd = create_dynamic_image_sidd(ortho_helper, output_directory, \
                                           output_file, \
                                            remap_function=local_remap_function)
    
def test_create_dynamic_image_sidd_remap_function_success(tmp_path):
    local_reader = get_test_reader()
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    local_remap_function = remap.get_registered_remap('nrl')
    test_sidd = create_dynamic_image_sidd(ortho_helper, output_directory, \
                                           output_file, \
                                            remap_function=local_remap_function)
