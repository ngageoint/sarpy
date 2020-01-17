"""
MAKE NICE LOOKING PLOTS OF THE DEM
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_dem(DEM, coordinates=[], log_elev = False, contour=True, threeD=False, imshow=True, **kwargs):
    '''
    Make a plot/image of the current DEM being used
    '''
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees')
    lats, lons = np.meshgrid(DEM.lats_1D,DEM.lons_1D)  # Create meshgrid for plotting
    elevations = np.ones(DEM.dem.shape)  # Make default value 0.01 m for plotting purposes
    valid = (DEM.dem > -32768.0) # Collect all bad points
    elevations[valid] = DEM.dem[valid]

    if log_elev:
        elevations = np.log(elevations)

    if 'colormap' in kwargs:  # Try to read user colormap
        try:
            plt.set_cmap(kwargs['colormap'])
        except:
            print('Could not find {} colormap.'.format(kwargs['colormap']))
    else:
        plt.set_cmap('gist_earth')  # Default cmap

    if threeD:
        print('3D plot not implemented yet.')

    if contour:
        plt.figure(1)  # Create plot
        plt.contourf(lons,lats,elevations)
        if coordinates != []:
            plt.scatter(coordinates[:,1],coordinates[:,0],c='k',s=24)
        plt.show()

