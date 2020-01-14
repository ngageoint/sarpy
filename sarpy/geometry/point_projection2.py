"""
Functions to map between the pixel grid in the image space and geolocated points in 3D space.
"""

import os
from typing import Tuple

import numpy

from . import geocoords
from ..sicd_elements.SICD import SICDType


__classification__ = "UNCLASSIFIED"


def ground_to_image(coords, sicd, delta_gp_max=None, delta_arp=None, delta_varp=None,
                    range_bias=0, adj_params_frame='ECF', max_iterations=10, block_size=10000):
    """
    Transforms a 3D ECF point to pixel (row, column) coordinates. This is implemented in accordance with the SICD Image
    Projections Description Document: http://www.gwg.nga.mil/ntb/baseline/docs/SICD/index.html

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
        ECF coordinate to map to scene coordinates, of size `N x 3`.
    sicd : SICDType
        SICD meta data structure.
    delta_gp_max : float|None
        Ground plane displacement tol (m). Defaults to 0.1*pixel.
    delta_arp : None|numpy.ndarray|list|tuple
        ARP position adjustable parameter (ECF, m). Defaults to 0 in each dimension.
    delta_varp : None|numpy.ndarray|list|tuple
        VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each dimension.
    range_bias : float|int
        Range bias adjustable parameter (m). Defaults to 0.
    adj_params_frame : str
        One of ['ECF', 'RIC_ECF', 'RIC_ECI']. This specifies the coordinate frame used for expressing
        `delta_arp` and `delta_varp`.
    max_iterations : int
        maximum number of iterations to perform
    block_size : int
        size of blocks of coordinates to transform at a time

    Returns
    -------
    Tuple[numpy.ndarray, float, int]
        * `image_points` - the determined image point array, of size `N x 2`. Following SICD convention,
           the upper-left pixel is [0, 0].
        * `delta_gpn` - residual ground plane displacement (m).
        * `iterations` - the number of iterations performed.
    """

    def iterate(coord_block):
        im_pts = numpy.zeros((coord_block.shape[0], 2), dtype=numpy.float64)
        del_gp = numpy.zeros((coord_block.shape[0], ), dtype=numpy.float64)
        t_iters = numpy.zeros((coord_block.shape[0], ), dtype=numpy.int16)
        it_here = numpy.ones((coord_block.shape[0], ), dtype=numpy.bool)
        uGPN = (coord_block.T/numpy.linalg.norm(coord_block, axis=1)).T
        cont = True
        iteration = 0
        while cont:
            # project ground plane to image plane iteration
            iteration += 1
            g_n = coord_block[it_here, :]
            dist_n = numpy.dot(SCP - g_n, uIPN)/sf
            i_n = g_n + numpy.outer(dist_n, uProj)
            delta_ipp = i_n - SCP  # N x 3
            ip_iter = numpy.dot(numpy.dot(delta_ipp, row_col_transform), ipp_transform)
            im_pts[it_here, 0] = ip_iter[:, 0]/row_ss + SCP_Pixel[0]
            im_pts[it_here, 1] = ip_iter[:, 1]/col_ss + SCP_Pixel[1]
            # transform to ground plane containing the scene points
            # TODO: refactor this to use the appropriate direct method
            gnd_pln = image_to_ground(
                im_pts[it_here, :], sicd, projection_type='plane', gref=g_n, ugpn=uGPN[it_here, :],
                delta_arp=delta_arp, delta_varp=delta_varp, range_bias=range_bias,
                adj_params_frame=adj_params_frame)
            # compute displacement
            disp_gpn_pln = numpy.linalg.norm(g_n - gnd_pln, axis=1)
            del_gp[it_here] = disp_gpn_pln
            # update the iterations
            t_iters[it_here] = iteration
            # where are we finished?
            must_continue = (disp_gpn_pln > delta_gp_max)
            it_here[it_here] = must_continue
            # should we continue?
            if not numpy.any(must_continue) or (iteration >= max_iterations):
                cont = False
        return im_pts, del_gp, t_iters

    if not isinstance(coords, numpy.ndarray):
        coords = numpy.array(coords, dtype=numpy.float64)
    orig_shape = coords.shape

    if coords.shape[-1] != 3:
        raise ValueError(
            'The coords array must represent an array of points in ECF coordinates, '
            'so the final dimension of coords must have length 3. Have coords.shape = {}'.format(coords.shape))

    if delta_arp is None:
        delta_arp = numpy.array([0, 0, 0], dtype=numpy.float64)
    if not isinstance(delta_arp, numpy.ndarray):
        delta_arp = numpy.array(delta_arp, dtype=numpy.float64)
    if delta_arp.shape != (3, ):
        raise ValueError('delta_arp must have shape (3, ). Got {}'.format(delta_arp.shape))

    if delta_varp is None:
        delta_varp = numpy.array([0, 0, 0], dtype=numpy.float64)
    if not isinstance(delta_varp, numpy.ndarray):
        delta_varp = numpy.array(delta_varp, dtype=numpy.float64)
    if delta_varp.shape != (3, ):
        raise ValueError('delta_varp must have shape (3, ). Got {}'.format(delta_varp.shape))

    row_ss = sicd.Grid.Row.SS
    col_ss = sicd.Grid.Col.SS
    if delta_gp_max is None:
        delta_gp_max = 0.1*numpy.sqrt(row_ss*row_ss + col_ss*col_ss)

    # establishing the basic projection components
    SCP_Pixel = sicd.ImageData.SCPPixel.get_array()
    uRow = sicd.Grid.Row.UVectECF.get_array()  # unit normal in row direction
    uCol = sicd.Grid.Col.UVectECF.get_array()  # unit normal in column direction
    uIPN = numpy.cross(uRow, uCol)  # image plane unit normal
    uIPN /= numpy.linalg.norm(uIPN)  # NB: uRow/uCol may not be perpendicular
    cos_theta = numpy.dot(uRow, uCol)
    sin_theta = numpy.sqrt(1 - cos_theta*cos_theta)
    ipp_transform = numpy.array([[1, -cos_theta], [-cos_theta, 1]], dtype=numpy.float64)/(sin_theta*sin_theta)
    row_col_transform = numpy.zeros((3, 2), dtype=numpy.float64)
    row_col_transform[:, 0] = uRow
    row_col_transform[:, 1] = uCol

    SCP = sicd.GeoData.SCP.ECF.get_array()
    ARP_SCP_COA = sicd.SCPCOA.ARPPos.get_array()
    VARP_SCP_COA = sicd.SCPCOA.ARPVel.get_array()
    uSPN = sicd.SCPCOA.look*numpy.cross(VARP_SCP_COA, SCP-ARP_SCP_COA)
    uSPN /= numpy.linalg.norm(uSPN)
    # uSPN - defined in section 3.1 as normal to instantaneous slant plane that contains SCP at SCP COA is
    # tangent to R/Rdot contour at SCP. Points away from center of Earth. Use look to establish sign.
    sf = numpy.dot(uSPN, uIPN)  # scale factor
    uProj = uSPN  # the projection direction (this is just shorthand for consistent notation)

    # prepare the work space
    coords_view = numpy.reshape(coords, (-1, 3))  # possibly or make 2-d flatten
    num_points = coords_view.shape[0]
    if num_points <= block_size:
        image_points, delta_gpn, iters = iterate(coords_view)
    else:
        image_points = numpy.zeros((num_points, 2), dtype=numpy.float64)
        delta_gpn = numpy.zeros((num_points, ), dtype=numpy.float64)
        iters = numpy.zeros((num_points, ), dtype=numpy.int16)

        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block+block_size, num_points)
            c_block = coords_view[start_block:end_block, :]
            image_points[start_block:end_block, :], \
            delta_gpn[start_block:end_block], \
            iters[start_block:end_block] = iterate(c_block)
            start_block = end_block

    if len(orig_shape) > 1:
        image_points = numpy.reshape(image_points, orig_shape[:-1]+(2, ))
        delta_gpn = numpy.reshape(delta_gpn, orig_shape[:-1])
        iters = numpy.reshape(iters, orig_shape[:-1])
    return image_points, delta_gpn, iters


def ground_to_image_geo(coords, *args, **kwargs):
    """

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
        Lat/Lon/HAE coordinate to map to scene coordinates, of size `N x 3`.
    args : list
    kwargs : dict

    Returns
    -------
    Tuple[numpy.ndarray, float, int]
        * `image_points` - the determined image point array, of size `N x 2`. Following SICD convention,
           the upper-left pixel is [0, 0].
        * `delta_gpn` - residual ground plane displacement (m).
        * `iterations` - the number of iterations performed.

    """

    return ground_to_image(geocoords.geodetic_to_ecf(coords), *args, **kwargs)


def image_to_ground(im_points, sicd, projection_type='HAE',
                    gref=None, ugpn=None, hae0=None, delta_hae_max=1, hae_nlim=3,
                    delta_arp=None, delta_varp=None, range_bias=0,
                    adj_params_frame='ECF', dem=None, dem_type='SRTM2F', geoid_path=None,
                    del_DISTrrc=10, del_HDlim=0.001):
    """

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        (row, column) coordinates of N points in image (or subimage if FirstRow/FirstCol are nonzero).
        Following SICD convention, the upper-left pixel is [0, 0].
    sicd : SICDType
            SICD meta data structure.
    projection_type : str
        One of ['PLANE', 'HAE', 'DEM'].
    gref : None|numpy.ndarray|list|tuple
        Ground plane reference point ECF coordinates (m). The default is the SCP, and only used
        if `projection_type` is `'PLANE'`.
    ugpn : None|numpy.ndarray|list|tuple
        Ground plane unit normal vector.  The default is the tangent to the surface of constant
        geodetic height above the WGS-84 reference ellipsoid passing through `GREF`. Only used
        if `projection_type` is `'PLANE'`.
    hae0 : None|float|int
        Surface height (m) above the WGS-84 reference ellipsoid for projection point. Only used if
        `projection_type` is `'HAE'`.
    delta_hae_max : None|float|int
        Height threshold for convergence of iterative constant HAE computation (m). Only used if
        `projection_type` is `'HAE'`.
    hae_nlim : int
        Maximum number of iterations allowed for constant hae computation. Only used if
        `projection_type` is `'HAE'`.
    delta_arp : None|numpy.ndarray|list|tuple
        ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
    delta_varp : None|numpy.ndarray|list|tuple
        VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
    range_bias : float|int
        Range bias adjustable parameter (m), defaults to 0.
    adj_params_frame : str
        One of ['ECF', 'RIC_ECF', 'RIC_ECI'], specifying the coordinate frame used for
        expressing `delta_arp` and `delta_varp` parameters.
    dem : str
        SRTM pathname or structure with lats/lons/elevations fields where are all are arrays of same size
        and elevation is height above WGS-84 ellipsoid. Only valid if `projection_type` is `'DEM'`.
    dem_type : str
        One of ['DTED1', 'DTED2', 'SRTM1', 'SRTM2', 'SRTM2F'], specifying the DEM type.
    geoid_path : str
        Parent path to EGM2008 file(s).
    del_DISTrrc : float|int
        Maximum distance between adjacent points along the R/Rdot contour. Only valid if
        `projection_type` is `'DEM'`.
    del_HDlim : Height difference threshold for determining if a point on the R/Rdot contour is on the DEM
        surface (m). Only valid if `projection_type` is `'DEM'`.

    Returns
    -------
    numpy.ndarray
        ECF Ground Points along the R/Rdot contour
    """

    # TODO: break this into child methods for projection type
    #   associate the appropriate arguments with each projection type

    # subsequently break each of those into appropriate "child" methods for better calling by
    # ground_to_image() in particular.

    pass


def image_to_ground_geo(*args, **kwargs):
    # TODO: docstring
    return geocoords.ecf_to_geodetic(image_to_ground(*args, **kwargs))

