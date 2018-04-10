'''This module contains coordinate transformations on the WGS 84 ellipsoid.'''

import numpy as np

__classification__ = "UNCLASSIFIED"
__email__ = "Wade.C.Schwartzkopf.ctr@nga.mil"

# WGS 84 defining parameters
a = 6378137.0           # Semi-major radius (m)
f = 1/298.257223563     # Flattening
GM = 3986004.418E8      # Earth's gravitational constant (including atmosphere)
w = 7292115.1467E-11    # Angular velocity (radians/second), not including precession
# WGS 84 derived geometric constants
b = a - f*a               # 6356752.3142, Semi-minor radius (m)
e2 = ((a*a)-(b*b))/(a*a)  # 6.69437999014E-3, First eccentricity squared


def ecf_to_geodetic(x, y=None, z=None):
    '''Convert ECF (Earth Centered Fixed) coordinates to geodetic latitude,
    longitude, and altitude.

    USAGE:
       pos_lla = ecf_to_geodetic(pos_ecf)
       [lat, lon, alt] = ecf_to_geodetic(pos_ecf_x, pos_ecf_y, pos_ecf_z)

    INPUTS:
       pos_ecf - required : ecf x, y, z coordinates                      [m, m, m]

    OUTPUTS:
       pos_lla - required : geodetic latitude, longitude, and altitude   [deg, deg, m]

    NOTES:
       Zhu, J. Conversion of Earth-centered, Earth-fixed coordinates to
       geodetic coordinates. IEEE Transactions on Aerospace and Electronic
       Systems, 30, 3 (July 1994), 957-962.

    VERSION:
       1.0
         - Sean Hatch 20070911
         - initial version
       1.1
         - Wade Schwartzkopf 20130708
         - vectorized and componentwise data handling
       1.2
         - Nick Tobin 20140625, nickolas.w.tobin@nga.ic.gov, NGA/IIG
         - translation to Python
       1.3
         - Wade Schwartzkopf 20161012
         - wrapped into geocoords module
    '''

    x, y, z, componentwise = _normalize_3dinputs(x, y, z)

    # calculate derived constants
    e4 = e2 * e2
    ome2 = 1.0 - e2
    a2 = a * a
    b2 = b * b
    e_b2 = (a2 - b2) / b2

    # calculate intermediates
    z2 = z * z
    r2 = (x * x) + (y * y)
    r = np.sqrt(r2)

    # Check for invalid solution
    valid = ((a * r) * (a * r) + (b * z) * (b * z) > (a2 - b2) * (a2 - b2))
    # Default values for invalid solutions
    lon = np.empty(x.shape) * np.nan
    lat = np.empty(x.shape) * np.nan
    alt = np.empty(x.shape) * np.nan

    # calculate longitude
    lon[valid] = np.rad2deg(np.arctan2(y[valid], x[valid]))

    # calculate intermediates
    f_ = 54.0 * b2 * z2  # not the WGS 84 flattening parameter
    g = r2 + ome2 * z2 - e2 * (a2 - b2)
    c = e4 * f_ * r2 / (g * g * g)
    s = (1.0 + c + np.sqrt(c * c + 2 * c)) ** (1. / 3.)
    templ = s + 1.0 / s + 1.0
    p = f_ / (3.0 * templ * templ * g * g)
    q = np.sqrt(1.0 + 2.0 * e4 * p)
    r0 = -p * e2 * r / (1.0 + q) + \
        np.sqrt(abs(0.5 * a2 * (1.0 + 1.0 / q) -
                    p * ome2 * z2 / (q * (1.0 + q)) - 0.5 * p * r2))
    temp2 = r - e2 * r0
    temp22 = temp2 * temp2
    u = np.sqrt(temp22 + z2)
    v = np.sqrt(temp22 + ome2 * z2)
    z0 = b2 * z / (a * v)

    # calculate latitude
    lat[valid] = np.rad2deg(np.arctan2(z[valid] + e_b2 * z0[valid], r[valid]))

    # calculate altitude
    alt[valid] = u[valid] * (1.0 - b2 / (a * v[valid]))
    if componentwise:
        return lat, lon, alt
    else:
        return np.column_stack((lat, lon, alt))


def geodetic_to_ecf(lat, lon=None, alt=None):
    '''Convert geodetic coordinates to ECF

    Convert geodetic latitude, longitude, and altitude to ECF (Earth
    Centered Fixed) coordinates.

    USAGE:
       pos_ecf = geodetic_to_ecf(pos_lla)
       [pos_ecf_x, pos_ecf_y, pos_ecf_z] = geodetic_to_ecf(lat, lon, alt)

    INPUTS:
       pos_lla - required : geodetic latitude, longitude, and altitude   [deg, deg, m]

    OUTPUTS:
       pos_ecf - required : ecf x, y, z coordinates                      [m, m, m]

    NOTES:
       Zhu, J. Conversion of Earth-centered, Earth-fixed coordinates to
       geodetic coordinates. IEEE Transactions on Aerospace and Electronic
       Systems, 30, 3 (July 1994), 957-962.

    VERSION:
       1.0
         - Sean Hatch 20070911
         - initial version
       1.1
         - Wade Schwartzkopf 20130708
         - vectorized and componentwise data handling
       1.2
         - Clayton Williams 20160511
         - translation to Python
       1.3
         - Wade Schwartzkopf 20161012
         - wrapped into geocoords module
    '''

    lat, lon, alt, componentwise = _normalize_3dinputs(lat, lon, alt)

    # calculate distance to surface of ellipsoid
    r = a / np.sqrt(1.0 - e2 * np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(lat)))

    # calculate coordinates
    x = (r + alt) * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = (r + alt) * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = (r + alt - e2 * r) * np.sin(np.deg2rad(lat))

    if componentwise:
        return x, y, z
    else:
        return np.column_stack((x, y, z))


def wgs_84_norm(x, y=None, z=None):
    """This function computes the normal vector to the WGS_84 ellipsoid at a given point in ECF
    space"""

    x, y, z, componentwise = _normalize_3dinputs(x, y, z)

    # Calculate normal vector
    x = x/(a**2)
    y = y/(a**2)
    z = z/(b**2)
    mag = np.sqrt(x**2 + y**2 + z**2)
    x = x/mag
    y = y/mag
    z = z/mag

    if componentwise:
        return x, y, z
    else:
        return np.column_stack((x, y, z))


def ric_ecf_mat(rarp, varp, frame_type):
    """Compute ECF transformation matrix for RIC frame"""

    if frame_type == 'eci':  # RIC_ECI frame
        vi = varp + np.cross([0, 0, w], rarp)
    elif frame_type == 'ecf':  # RIC_ECF frame
        vi = varp

    r = rarp/np.sqrt(np.sum(np.power(rarp, 2)))
    c = np.cross(rarp, vi)
    c = c/np.sqrt(np.sum(np.power(c, 2)))
    i = np.cross(c, r)

    return np.matrix([r, i, c])


def _normalize_3dinputs(x, y, z):
    """Allow for a variety of different input types, but convert them all to componentwise
    numpy arrays that the functions in this module assume."""

    # Handle different forms in input arguments
    x = np.atleast_2d(x)  # Assure a numpy array for componentwise or array versions
    componentwise = False
    if y is not None and z is not None:
        # Componentwise inputs, separate arguments for X,Y,Z
        componentwise = True
        y = np.atleast_2d(y)
        z = np.atleast_2d(z)
    elif x.ndim == 2 and x.shape[1] == 3:  # Array of 3-element vectors
        y = x[:, 1]
        z = x[:, 2]
        x = x[:, 0]
    else:
        raise ValueError()  # Must be right type if np.array(x) worked above

    return x, y, z, componentwise
