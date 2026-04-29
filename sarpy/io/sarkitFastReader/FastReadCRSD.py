# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:48:49 2026

@author: Daniel Haverporth, NGA GEOINT Innovation and Research

Description:  This code reads CRSD 1.0 files.  A list of the channel IDs is
              displayed.
              
Inputs:  
    FileName:  str full path to file
    
    Optional:  metaformat: 'wrapped', 'helper', or 'xmltree'
               defaults to wrapped, see sarkit documentation for details
               
               preload:  Bool, defaults to False, If true the image is preloaded
                         into memory.
                         
Attributes: 
    metadata:  Captures the metaformat that the user selected.
    
    reader:    Actual sarkit reader, see sarkit documentation for details
    
    metaformat: Captures the metaformat that the user selected
    
    signal:  If preload is true, signal data will be stored here.
"""

import sarkit
from sarkit import crsd as skcrsd

class CRSD_Fast_Reader:
    
    def __init__(self, fname, metaformat = 'wrapped', preload = False):
        self.f = open(fname, 'rb') 
        reader = skcrsd.Reader(self.f)
        # Create metadata attribute with SARKit metadata options
            
        if metaformat == 'wrapped':
            Meta = skcrsd.ElementWrapper(reader.metadata.xmltree.getroot())
        elif metaformat == 'helper':
            Meta = skcrsd.XmlHelper(reader.metadata.xmltree)
        else:
            Meta = reader.metadata.xmltree
            metaformat = 'xmltree'
            
        ch_id = reader.metadata.xmltree.findtext("{*}Data/{*}Receive/{*}Channel/{*}ChId")
        print("channel IDs:  ", ch_id) 
        if preload:
            image = reader.read_signal(ch_id)
            self.signal = image
        else:
            self.signal = None
            
        self.metadata = Meta
        self.reader = reader
        self.metaformat = metaformat