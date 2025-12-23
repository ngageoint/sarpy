"""
Test a dted file for voids.
It can check a file, list of files, or a complete path to a directory
return  list/file with { <dtedfilename} :[ (lat/lon void),]
or  a boolean on yes this file has a void
and a note on how to see/check voids in QGIS
"""

__classification__ = "UNCLASSIFIED"
__author__ = "John O'Neill"

import os
import sarpy.io.DEM.DTED as sarpy_dted

import numpy as np
import json

def check_for_voids(dtedFilePath, return_index=False ):
    
    """
        Method checks for void data in DTED files.
        By spec a void-ed dted cell has a value of 65535, which is all bits lit for 16-bit data.
        There does a exist dted data that has been cleaned up with voids removed.
        
        Parameters
        ----------
            dtedFilePath: string or list
                string of a filename or full path to a directory
                list would be a list of strings of filename
            return_index : Bool
                true if user wants wants python row/cols of void data.
                
        Returns
        --------
        if return_index == False
            dict where key : value where key is dted filename and value is True if there is any void data in file
        if return_index == True
        dict where key : value where key is dted filename and value is a dict with two more key: 'has_voids' with value bool True void in data 
            and 'indices' which list of rows and cols where data is void
            
        ## this is probably over kill, but I created this code to test and prove my work so figured i would pass it along 

    """
    
    voidResults = dict()
    #
    # allow filename, list of filenames , or dir
    if isinstance( dtedFilePath, list):
        files = dtedFilePath
    elif os.path.isdir( dtedFilePath ):
        files = []
        for root, _, tmpFiles in os.walk( dtedFilePath ):
            for filename in tmpFiles:
                if filename.endswith(".dt1"):  # check for ".dt1" extension
                    file_path = os.path.join(root, filename)
                    files.append( file_path )        
        
    elif isinstance( dtedFilePath, str):
        files = [dtedFilePath, ]
        

    result_set = {}
    for dted_file in files:
            
        dted_reader = sarpy_dted.DTEDReader( dted_file )

        value_to_check = 65535  # 16 bit value that defines a void cell value in dted data as per spec.
        is_present     = str( np.isin( value_to_check, dted_reader._mem_map ).any())
        
        if return_index == True:
            
            # one can display dted data in QGIS
            # and can display individual cell values with the QGIS plugin Value Tool
            #  see DTEDReader test_dted_reader in tests/io/DEM/test_dted.py
            # to follow along in qgis
            # qgis_row = 1200 -  known_value[ 1 ] # dted1 data is in 1200 blocks
            # qgis_col = known_value[ 0 ]
            # 
            # Create a boolean mask
            mask = (value_to_check == dted_reader._mem_map)      

            # Get the indices where the condition is True
            indices = np.where(mask)

            # break up indices for output via dict or json
            index0 = indices[0].tolist()
            index1 = indices[1].tolist()

            result_set[ dted_file ]               = {}
            result_set[ dted_file ][ 'indices']   = list(map( tuple, (index0, index1)))
            result_set[ dted_file ][ 'has_voids'] = is_present
        
        else:
            result_set[ dted_file ] = is_present

    return result_set
