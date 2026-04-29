# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:26:36 2026

@author: Daniel Haverporth, NGA GEOINT Innovation and Research

Desription:  This code creates SARKit Fast Readers for SICD for fast SICD 
Reading.

Inputs: 
    FileName:  str full path to file

    Optional:  metaformat: 'wrapped', 'helper', or 'xmltree'
               defaults to wrapped, see sarrkit documentation for details.

               preload:  Bool, defaults to False.  If true the image is 
                         preloaded into memory
Attributes:
    metadata:  Returns XML metadata of SICD File.  Metadata if formed
               According to metaformat.
               
    reader:  actual sarkit reader, see sarkit documentation for details
    
    metaformat: captures the metaformat that the user selected
    
    preload:  Bool defaults to False.  If true the SICD image is loaded into 
              memory.
              
Methods:
    image_to_llh(row, col):
        returns a Lat, Long, HAE coordinate for a given pixel defined by row
        and column.  Multiple points can be entered if row and col are numpy
        arrays.  Ex. row = [50,100], col = [50, 100]
        
    llh_to_image(lat,lon):
        returns a pixel coordinate for a given lat, lon.  Multiple points can
        be entered if lat and lon are numpy arrays.
        
    Remap(chip=None, display=False):
        This function can only be used if the preload option is turned on.
        This funciton will produce a remapped image using the Density remap
        algorithm.  If display is True, the remapped image will be displayed
        using MATPLOTLIB.
    
"""
import sarkit
import sarkit.sicd as sksicd
import numpy as np
from sarpy.visualization import remap

class SICD_Fast_Reader:
    
    def __init__(self, fname, metaformat = 'wrapped', preload = False, chip = False):
        self.f = open(fname, 'rb')
        reader = sksicd.NitfReader(self.f) 
        # Create metadata attribute with SARKit metadata options
        if metaformat == 'wrapped':
            Meta = sksicd.ElementWrapper(reader.metadata.xmltree.getroot())
        elif metaformat == 'helper':
            Meta = sksicd.XmlHelper(reader.metadata.xmltree)
        else:
            Meta = reader.metadata.xmltree
            metaformat = 'xmltree'
                
        if preload:
            self.complex_image = reader.read_image()
        elif chip and len(chip) == 4: #[rowStart, ColStart, rowEnd, colEnd]
            self.complex_image = reader.read_sub_image(chip[0], chip[1], chip[2], chip[3])[0]
            preload = True
        else:
            self.complex_image = None
            
        
                
        self.metadata = Meta
        self.reader = reader
        self.metaformat = metaformat
            
        self.Preload = preload
            
    def image_to_llh(self, row, col):
        Meta = self.metadata
        
        scp_pixel = Meta['ImageData']['SCPPixel']
        row_ss = Meta['Grid']['Row']['SS']
        col_ss = Meta['Grid']['Col']['SS']
        scp_ecf = Meta['GeoData']['SCP']['ECF']
        
        # If row and col are entered as ndarray
        if isinstance(row, np.ndarray):
          if len(row) != len(col):
            raise ValueError('row and column not the same length')
            
          icp_llh = np.zeros([len(row), 3])    
          for x in range(len(row)):
            
        
            image_grid_locations = (
               np.array([row[x], col[x]])
             - scp_pixel
            ) * [row_ss, col_ss]
            icp_ecef, _, _ = sksicd.image_to_ground_plane(
              self.reader.metadata.xmltree,
              image_grid_locations,
              scp_ecf,
              sarkit.wgs84.up(sarkit.wgs84.cartesian_to_geodetic(scp_ecf)),
            )
            icp_llh[x] = sarkit.wgs84.cartesian_to_geodetic(icp_ecef)
            
          return(icp_llh)
        else: #row and col are integers
          image_grid_locations = (
              np.array([row, col])
              - scp_pixel
          ) * [row_ss, col_ss]
          icp_ecef, _, _ = sksicd.image_to_ground_plane(
              self.reader.metadata.xmltree,
              image_grid_locations,
              scp_ecf,
              sarkit.wgs84.up(sarkit.wgs84.cartesian_to_geodetic(scp_ecf)),
          )
          icp_llh = sarkit.wgs84.cartesian_to_geodetic(icp_ecef)
          return(icp_llh)
      
    def llh_to_image(self, lat, lon):
        Meta = self.metadata
        HAE = Meta['GeoData']['SCP']['LLH'][2] #use SCP HAE as reference
        scp_pixel = Meta['ImageData']['SCPPixel']
        row_ss = Meta['Grid']['Row']['SS']
        col_ss = Meta['Grid']['Col']['SS']
  
        
        if isinstance(lat,np.ndarray):
            if len(lat) != len(lon):
                raise ValueError('lat and lon not the same length')
                
            image_coords = np.zeros([len(lat), 2])
            for x in range(len(lat)):
              scene_points = sarkit.wgs84.geodetic_to_cartesian(np.array([lat[x], lon[x], HAE]))
              image_grid = sksicd.scene_to_image(self.reader.metadata.xmltree, scene_points)
              iRow = round(image_grid[0][0] / row_ss + scp_pixel[0])
              iCol = round(image_grid[0][1] / col_ss + scp_pixel[1])
              image_coords[x] = np.array([iRow, iCol ])
        else: #if single integer
         scene_points = sarkit.wgs84.geodetic_to_cartesian(np.array([lat, lon, HAE]))
         image_grid = sksicd.scene_to_image(self.reader.metadata.xmltree, scene_points)
         iRow = round(image_grid[0][0] / row_ss + scp_pixel[0])
         iCol = round(image_grid[0][1] / col_ss + scp_pixel[1])
         #delta_scp_row = uRow * image_grid[0][0]
         #delta_scp_col = uCol * image_grid[0][1]

         image_coords = np.array([iRow, iCol])
        return(image_coords)
    
    def Remap(self, display = False):
        if not self.Preload:
            raise ValueError('preload option must be turned on to use Remap function')
        #remapClass = getattr(remap, alg)
        remapClass = remap.Density()
        cmplx = self.complex_image
        print(cmplx)
        image = remapClass(cmplx)            
        if display:
           import matplotlib.pyplot as plt
           plt.imshow(image, cmap='gray')
           
        return(image) 
        