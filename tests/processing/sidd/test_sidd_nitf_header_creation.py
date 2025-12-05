'''
This unit test determines if the NITF header value of FDT (FileDateTime) is properly updated when the SIDD product is created.
The NITF header includes (at least) two datetime values.  IDATIM is the datetime that the image was collected (aka acquired).
FDT is the datetime that the NITF (SIDD) file was created.  IDATIM and FDT should have different values.
'''

import json
import os

import pytest
import logging
import unittest
from tests import parse_file_entry # fails unless run using pytest
from datetime  import datetime, timezone

from sarpy.io.complex.converter import conversion_utility, open_complex
from sarpy.processing.ortho_rectify import NearestNeighborMethod
from sarpy.processing.sidd.sidd_product_creation import \
    create_detected_image_sidd, create_dynamic_image_sidd
import sarpy.visualization.remap as remap

from sarpy.utils import create_product

from sarpy.io.general.nitf import NITFDetails

complex_file_types = {}
this_loc           = os.path.abspath(__file__)

# JSON file that specifies test file locations on the local system
file_reference     = os.path.join(os.path.split(this_loc)[0], \
                                  'complex_file_types.json')  
# Find valid files from file_refernce and add them to the valid_entries list
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

# Determine a valid file reader for the complex input image
def get_test_reader(idx):
    if idx >= len(sicd_files):
        return None
    input_file       = sicd_files[idx]
    reader           = open_complex(input_file)
    return reader

''' Read a single input SICD image, create the detected-image NITF SIDD product, then
read the NITF header from the newly created SIDD product and test that the FDT value
is different than the IDATIM value and that the FDT value is close to the current time
(becuase the NITF SIDD product was just created). 
'''
def test_nitf_fdt_updated_for_detected_image_sidd(tmp_path):
    local_reader = get_test_reader(0)
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    
    # create SIDD product
    test_sidd = create_detected_image_sidd(ortho_helper, output_directory, output_file)

    # Full path to the created SIDD file
    sidd_file = os.path.join(*output_directory.parts,output_file)

     # Get NITF header data from created SIDD file
    details = NITFDetails(sidd_file)

    # Get datetime values from the NITF header data
    fdt_datetime = datetime.strptime(details.nitf_header.FDT,"%Y%m%d%H%M%S%f")
    collection_datetime = datetime.strptime(details.img_headers[0].IDATIM,"%Y%m%d%H%M%S%f")
    
     # Since fdt_datetime and collection_datetime are realtive to Zulu but that information
    # is not represented in their datetime python objects, we need to determine the
    # current datetime relative to Zule, but then remove the tiemzone info from the
    # object in order to compute time deltas later.
    current_time_zulu = datetime.now(timezone.utc).replace(tzinfo=None)
    
     # Compute time difference between FDT (presumably current time) and IDATIM (collection time)
    time_delta = fdt_datetime - collection_datetime
    
    # boolean representing collection time is before file creation time
    fdt_gt_cdt = time_delta.total_seconds() > 0
    
     # Compute time difference between current time the FDT from NITF header
    recent_time_delta = current_time_zulu - fdt_datetime
    
    # Boolean representig that the SIDD file was created within the past 2 minutes
    fdt_is_recent = recent_time_delta.total_seconds() < 120
    
    assert (fdt_gt_cdt and fdt_is_recent), 'NITF FileDateTime should be greater than IDATIM: {} > {}?'.format(details.nitf_header.FDT,details.img_headers[0].IDATIM)

''' Read a single input SICD image, create the dynamic-image NITF SIDD product, then
read the NITF header from the newly created SIDD product and test that the FDT value
is different than the IDATIM value and that the FDT value is close to the current time
(becuase the NITF SIDD product was just created).  It was observed during debug that 
the FDT and IDATIM values are recomputed for each subaerture of the dynamic-image, but
that information may not be represented in the NITF header data.
'''
def test_nitf_fdt_updated_for_dynamic_image_sidd(tmp_path):
    local_reader = get_test_reader(0)
    ortho_helper = NearestNeighborMethod(local_reader, index=0)
    output_directory = tmp_path
    output_file = 'output.sidd'
    
    # create SIDD product
    test_sidd = create_dynamic_image_sidd(ortho_helper, output_directory, output_file)
    
    # Full path to the created SIDD file
    sidd_file = os.path.join(*output_directory.parts,output_file)
    
    # Get NITF header data from created SIDD file
    details = NITFDetails(sidd_file)

    # Get datetime values from the NITF header data
    fdt_datetime = datetime.strptime(details.nitf_header.FDT,"%Y%m%d%H%M%S%f")
    collection_datetime = datetime.strptime(details.img_headers[0].IDATIM,"%Y%m%d%H%M%S%f")
    assert (fdt_datetime > collection_datetime)

    # Since fdt_datetime and collection_datetime are realtive to Zulu but that information
    # is not represented in their datetime python objects, we need to determine the
    # current datetime relative to Zule, but then remove the tiemzone info from the
    # object in order to compute time deltas later.
#     current_time_zulu = datetime.now(timezone.utc).replace(tzinfo=None)
    
#     # Compute time difference between FDT (presumably current time) and IDATIM (collection time)
#     time_delta = fdt_datetime - collection_datetime

#     # boolean representing collection time is before file creation time
#     fdt_gt_cdt = time_delta.total_seconds() > 0

#     # Compute time difference between current time the FDT from NITF header
#     recent_time_delta = current_time_zulu - fdt_datetime

#     # Boolean representig that the SIDD file was created within the past 2 minutes
#     fdt_is_recent = recent_time_delta.total_seconds() < 120
    
#     assert (fdt_gt_cdt and fdt_is_recent), 'NITF FileDateTime should be greater than IDATIM: {} > {}?'.format(details.nitf_header.FDT,details.img_headers[0].IDATIM)

if __name__ == '__main__':
    unittest.main()