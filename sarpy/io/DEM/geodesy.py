import numpy as np

def calculateEarthRadius(lat_deg):
    """
    IN RADIANS!!!
    """
    lat_rad = np.deg2rad(lat_deg)
    major = 6378137.0  # semi-major axis of the earth
    minor = 6356752.3142  # semi-minor axis of the earth

    radius = np.sqrt((((major**2)*np.cos(lat_rad))**2 + ((minor**2)*np.sin(lat_rad))**2)/
                      ((major*np.cos(lat_rad))**2 + (minor*np.sin(lat_rad))**2))  # defines the radius of the earth at a specific point

    return radius
