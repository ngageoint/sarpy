import numpy as np

__classification__ = "UNCLASSIFIED"


def calculateEarthRadius(lat_deg):
    """
    Get the radius of the earth at the given latitude.
    :param lat_deg: degrees latitude
    :return: radius of the earth at the given latitude
    """

    lat_rad = np.deg2rad(lat_deg)
    major = 6378137.0  # semi-major axis of the earth
    minor = 6356752.3142  # semi-minor axis of the earth

    return np.sqrt(
        ((major*major*np.cos(lat_rad))**2 + (minor*minor*np.sin(lat_rad))**2) /
        ((major*np.cos(lat_rad))**2 + (minor*np.sin(lat_rad))**2))
