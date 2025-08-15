import json
import os
import pytest
import unittest
from tests import parse_file_entry

from sarpy.io.complex.converter          import conversion_utility, open_complex
from sarpy.processing.ortho_rectify import NearestNeighborMethod
from sarpy.processing.ortho_rectify.base import FullResolutionFetcher, OrthorectificationIterator
from sarpy.processing.sidd.sidd_structure_creation import \
    create_sidd_structure_v3, create_sidd_structure_v2, create_sidd_structure
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

class setup_test():
    def __init__(self):
        self.input_file       = sicd_files[0]
        self.reader           = open_complex(self.input_file)
        self.ortho_helper = NearestNeighborMethod(self.reader, index=0)
        self.calculator = FullResolutionFetcher(
            self.ortho_helper.reader, dimension=0, index=self.ortho_helper.index, 
            block_size=10)
        self.ortho_iterator = OrthorectificationIterator(
            self.ortho_helper, calculator=self.calculator, bounds=None,
            remap_function=None, recalc_remap_globals=False)


@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_v3_required_params_only_success(tmp_path):
    local_setup = setup_test()    
    test_sidd = create_sidd_structure_v3(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I')
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_v3_with_remap_param_success(tmp_path):
    local_setup = setup_test()    
    test_sidd = create_sidd_structure_v3(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I',
                                         remap_function=remap.get_registered_remap('nrl'))

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_v3_with_remap_param_fail(tmp_path):
    local_setup = setup_test()    
    with pytest.raises(TypeError, 
                       match="Input 'remap_function' must be a remap.RemapFunction."):
        test_sidd = create_sidd_structure_v3(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I',
                                         remap_function='bob')


@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_v2_required_params_only_success(tmp_path):
    local_setup = setup_test()    
    test_sidd = create_sidd_structure_v2(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I')
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_v2_with_remap_param_success(tmp_path):
    local_setup = setup_test()    
    test_sidd = create_sidd_structure_v2(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I',
                                         remap_function=remap.get_registered_remap('nrl'))

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_v2_with_remap_param_fail(tmp_path):
    local_setup = setup_test()    
    with pytest.raises(TypeError, 
                       match="Input 'remap_function' must be a remap.RemapFunction."):
        test_sidd = create_sidd_structure_v2(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I',
                                         remap_function='bob')


@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_required_params_only_success(tmp_path):
    local_setup = setup_test()    
    test_sidd = create_sidd_structure(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I')
    
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_with_remap_param_success(tmp_path):
    local_setup = setup_test()    
    test_sidd = create_sidd_structure(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I',
                                         remap_function=remap.get_registered_remap('nrl'))

@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_sidd_structure_with_remap_param_fail(tmp_path):
    local_setup = setup_test()    
    with pytest.raises(TypeError, 
                       match="Input 'remap_function' must be a remap.RemapFunction."):
        test_sidd = create_sidd_structure(local_setup.ortho_helper, 
                                         bounds=local_setup.ortho_iterator.ortho_bounds,
                                         product_class='Detected Image',
                                         pixel_type='MONO8I',
                                         remap_function='bob')
