# -*- coding: utf-8 -*-
"""
Provides coordinate transforms for WGS-84 and ECF coordinate systems
"""

import numpy

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Wade Schwartzkopf")

#####
# WGS-84 parameters and related derived parameters
_A = 6378137.0            # Semi-major radius (m)
_F = 1/298.257223563      # Flattening
_GM = 3986004.418E8       # Earth's gravitational constant (including atmosphere)
_W = 7292115.1467E-11     # Angular velocity (radians/second), not including precession
_B = _A - _F*_A           # 6356752.3142, Semi-minor radius (m)
_A2 = _A*_A
_B2 = _B*_B
_E2 = (_A2-_B2)/_A2  # 6.69437999014E-3, First eccentricity squared
_E4 = _E2*_E2
_OME2 = 1.0 - _E2
_EB2 = (_A2 - _B2)/_B2


def _validate(arr):
    if not isinstance(arr, numpy.ndarray):
        arr = numpy.array(arr, dtype=numpy.float64)

    if arr.shape[-1] != 3:
        raise ValueError(
            'The input argument should represent geographical coordinates, so the final dimension should have size 3. '
            'Got shape {}.'.format(arr.shape))
    orig_shape = arr.shape
    arr = numpy.reshape(arr, (-1, 3))  # this is just a view
    return arr, orig_shape


def ecf_to_geodetic(ecf, ordering='latlong'):
    """
    Converts ECF (Earth Centered Fixed) coordinates to WGS-84 coordinates.

    Parameters
    ----------
    ecf : numpy.ndarray|list|tuple
    ordering : str
        If 'longlat', then the return will be `[longitude, latitude, hae]`.
        Otherwise, the return will be `[latitude, longitude, hae]`.

    Returns
    -------
    numpy.ndarray
        The WGS-84 coordinates, of the same shape as `ecf`.
    """

    ecf, orig_shape = _validate(ecf)

    x = ecf[:, 0]
    y = ecf[:, 1]
    z = ecf[:, 2]

    llh = numpy.full(ecf.shape, numpy.nan, dtype=numpy.float64)

    r = numpy.sqrt((x * x) + (y * y))

    # Check for invalid solution
    valid = ((_A*r)*(_A*r) + (_B*z)*(_B*z) > (_A2 - _B2)*(_A2 - _B2))

    # calculate intermediates
    F = 54.0*_B2*z*z  # not the WGS 84 flattening parameter
    G = r*r + _OME2*z*z - _E2*(_A2 - _B2)
    C = _E4*F*r*r/(G*G*G)
    S = (1.0 + C + numpy.sqrt(C*C + 2*C))**(1./3)
    P = F/(3.0*(G*(S + 1.0/S + 1.0))**2)
    Q = numpy.sqrt(1.0 + 2.0*_E4*P)
    R0 = -P*_E2*r/(1.0 + Q) + numpy.sqrt(numpy.abs(0.5*_A2*(1.0 + 1/Q) - P*_OME2*z*z/(Q*(1.0 + Q)) - 0.5*P*r*r))
    T = r - _E2*R0
    U = numpy.sqrt(T*T + z*z)
    V = numpy.sqrt(T*T + _OME2*z*z)
    z0 = _B2*z/(_A*V)

    # account for ordering
    if ordering.lower() == 'longlat':
        inds = [0, 1, 2]
    else:
        inds = [1, 0, 2]
    # calculate longitude
    llh[valid, inds[0]] = numpy.rad2deg(numpy.arctan2(y[valid], x[valid]))
    # calculate latitude
    llh[valid, inds[1]] = numpy.rad2deg(numpy.arctan2(z[valid] + _EB2*z0[valid], r[valid]))
    # calculate altitude
    llh[valid, inds[2]] = U[valid]*(1.0 - _B2/(_A*V[valid]))
    return numpy.reshape(llh, orig_shape)


def geodetic_to_ecf(llh, ordering='latlong'):
    """
    Converts WGS-84 coordinates to ECF (Earth Centered Fixed).

    Parameters
    ----------
    llh : numpy.ndarray|list|tuple
    ordering : str
        If 'longlat', then the input is `[longitude, latitude, hae]`.
        Otherwise, the input is `[latitude, longitude, hae]`.

    Returns
    -------
    numpy.ndarray
        The ECF coordinates, of the same shape as `llh`.
    """

    llh, orig_shape = _validate(llh)

    # account for ordering
    if ordering.lower() == 'longlat':
        inds = [0, 1, 2]
    else:
        inds = [1, 0, 2]
    lon = llh[:, inds[0]]
    lat = llh[:, inds[1]]
    alt = llh[:, inds[2]]

    out = numpy.full(llh.shape, numpy.nan, dtype=numpy.float64)
    # calculate distance to surface of ellipsoid
    r = _A / numpy.sqrt(1.0 - _E2*numpy.sin(numpy.deg2rad(lat)) * numpy.sin(numpy.deg2rad(lat)))

    # calculate coordinates
    out[:, 0] = (r + alt)*numpy.cos(numpy.deg2rad(lat))*numpy.cos(numpy.deg2rad(lon))
    out[:, 1] = (r + alt)*numpy.cos(numpy.deg2rad(lat))*numpy.sin(numpy.deg2rad(lon))
    out[:, 2] = (r + alt - _E2*r)*numpy.sin(numpy.deg2rad(lat))
    return numpy.reshape(out, orig_shape)


def wgs_84_norm(ecf):
    """
    Calculates the normal vector to the WGS_84 ellipsoid at the given ECF coordinates.

    Parameters
    ----------
    ecf : numpy.ndarray|list|tuple

    Returns
    -------
    numpy.ndarray
        The normal vector, of the same shape as `ecf`.
    """

    ecf, orig_shape = _validate(ecf)
    out = numpy.copy(ecf)/numpy.array([_A2, _A2, _B2], dtype=numpy.float64)
    out = out/(numpy.linalg.norm(out, axis=1)[:, numpy.newaxis])
    return numpy.reshape(out, orig_shape)
