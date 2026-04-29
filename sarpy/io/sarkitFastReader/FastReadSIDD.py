# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:15:04 2026
author: Daniel Haverporth, NGA GEOINT Innovation and Research

Inputs:
    Filename:  str to include entire file path
    
    Optional:
    metaformat:  options: 'wrapped', 'helper', or 'xmltree'
    default if 'wrapped' which is a python dictionary format
    
    Preload: Defaults to False and preloads the image in memory
    
    display: Defaults to False and will display the image using MATPLOTLIB.
             Preload must be turned on to use this function
             
Attributes:
    metadata:  Returns XML metadata of first image segment.  Metadata if formed
               According to metaformat
               
    reader:    actual sarkit reader, see sarkit documentation for details
    
    image:    Only included if preload is True.  Includes the image as a numpy
              array preloaded in memory.
"""

import sarkit
from sarkit import sidd as sksidd



class SIDD_Fast_Reader:
    
    def __init__(self, fname, metaformat = 'wrapped', preload = False, display= False):
        with open(fname, 'rb') as f, sksidd.NitfReader(f) as reader:
            
            # Create metadata attribute with SARKit metadata options
            
            
            if metaformat == 'wrapped':
                Meta = sksidd.ElementWrapper(reader.metadata.images[0].xmltree.getroot())
            elif metaformat == 'helper':
                Meta = sksidd.XmlHelper(reader.metadata.images[0])
            else:
                Meta = reader.metadata.images[0]
                metaformat = 'xmltree'
                
            self.metadata = Meta
            self.reader = reader
                
            if preload:
                image = reader.read_image(0)
                self.image = image
            else:
                self.image = None
                
            if display and preload:
                import matplotlib.pyplot as plt
                plt.imshow(image, cmap='gray')