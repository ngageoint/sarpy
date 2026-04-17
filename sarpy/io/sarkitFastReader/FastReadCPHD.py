# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:29:50 2026

@author: Daniel Haverporth, NGA GEOINT Innovation and Research

Description:  This code create SARKit Fast Readers for CPHD files.

Inputs:
    FileName: str full path to file
    
    Optional:  metaformat: 'wrapped', 'helper', or 'xmltree'
               defulats to wrapped, see sarkit documentation for details.
               
               preload:  Bool, defaults to False, If true the data is preloaded
                         into memory.
                         
Attributes:
    metadata:  CPHD metadata format in user selected metaformat
    
    reader:  Actual sarkit reader, see sarkit documentation for details
    
    metaformat:  Captures the metaformat that the user selected
    
    signal:  If preload selected then image signal data is stored here.

"""

import sarkit
from sarkit import cphd as skcphd

class CPHD_Fast_Reader:
    
    def __init__(self, fname, metaformat = 'wrapped', preload = False):
        with open(fname, 'rb') as f, skcphd.Reader(f) as reader:
            # Create metadata attribute with SARKit metadata options
            
            if metaformat == 'wrapped':
                Meta = skcphd.ElementWrapper(reader.metadata.xmltree.getroot())
            elif metaformat == 'helper':
                Meta = skcphd.XmlHelper(reader.metadata.xmltree)
            else:
                Meta = reader.metadata.xmltree
                metaformat = 'xmltree'
            
            ch_id = reader.metadata.xmltree.findtext("{*}Data/{*}Channel/{*}Identifier")
            print("channel IDs:  ", ch_id) 
            if preload:
                image = reader.read_signal(ch_id)
                self.signal = image
            else:
                self.signal = None
            
            self.metadata = Meta
            self.reader = reader
            self.metaformat = metaformat