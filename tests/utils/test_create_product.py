import json
import os
import pytest
import unittest
from tests import parse_file_entry

from sarpy.io.complex.converter          import conversion_utility, open_complex
from sarpy.processing.ortho_rectify import NearestNeighborMethod
from sarpy.processing.ortho_rectify.base import FullResolutionFetcher, OrthorectificationIterator
from sarpy.utils.create_product import main as run_main
import sarpy.visualization.remap as remap


utils_file_types = {}
this_loc           = os.path.abspath(__file__)
# specifies file locations
file_reference     = os.path.join(os.path.split(this_loc)[0], \
                                  'utils_file_types.json')  
if os.path.isfile(file_reference):
    with open(file_reference, 'r') as local_file:
        test_files_list = json.load(local_file)
        for test_files_type in test_files_list:
            valid_entries = []
            for entry in test_files_list[test_files_type]:
                the_file = parse_file_entry(entry)
                if the_file is not None:
                    valid_entries.append(the_file)
            utils_file_types[test_files_type] = valid_entries

sicd_files = utils_file_types.get('SICD', [])

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
def test_create_product_required_params_only_success(tmp_path):
    local_setup = setup_test()    
    test_args = [str(local_setup.input_file), str(tmp_path)]
    test_sidd = run_main(test_args)
                                         
@unittest.skipIf(len(sicd_files) == 0, 'No sicd files found')
def test_create_product_with_bit_depth_success(tmp_path):
    local_setup = setup_test()    
    test_args = [str(local_setup.input_file), str(tmp_path), '-b', '16']
    test_sidd = run_main(test_args)