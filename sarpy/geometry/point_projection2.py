"""
Functions to map between the pixel grid in the image space and geolocated points in 3D space.
"""

import os
import logging
from typing import Tuple

import numpy

from . import geocoords
from ..sicd_elements.SICD import SICDType
from ..sicd_elements.blocks import Poly2DType

__classification__ = "UNCLASSIFIED"


def ground_to_image(coords, sicd, delta_gp_max=None, max_iterations=10, block_size=50000, **kwargs):
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
    max_iterations : int
        maximum number of iterations to perform
    block_size : int|None
        size of blocks of coordinates to transform at a time
    kwargs : dict
        keywords arguments passed through to :func:`image_to_ground_plane`.

    Returns
    -------
    Tuple[numpy.ndarray, float, int]
        * `image_points` - the determined image point array, of size `N x 2`. Following SICD convention,
           the upper-left pixel is [0, 0].
        * `delta_gpn` - residual ground plane displacement (m).
        * `iterations` - the number of iterations performed.
    """

    if not isinstance(coords, numpy.ndarray):
        coords = numpy.array(coords, dtype=numpy.float64)

    if coords.shape[-1] != 3:
        raise ValueError(
            'The coords array must represent an array of points in ECF coordinates, '
            'so the final dimension of coords must have length 3. Have coords.shape = {}'.format(coords.shape))

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

    # prepare the work space
    orig_shape = coords.shape
    coords_view = numpy.reshape(coords, (-1, 3))  # possibly or make 2-d flatten
    num_points = coords_view.shape[0]
    if block_size is None or num_points <= block_size:
        image_points, delta_gpn, iters = _ground_to_image(
            coords_view, sicd, SCP, SCP_Pixel, uIPN, sf, row_ss, col_ss, uSPN,
            row_col_transform, ipp_transform, delta_gp_max, max_iterations, **kwargs)
    else:
        image_points = numpy.zeros((num_points, 2), dtype=numpy.float64)
        delta_gpn = numpy.zeros((num_points, ), dtype=numpy.float64)
        iters = numpy.zeros((num_points, ), dtype=numpy.int16)

        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block+block_size, num_points)
            image_points[start_block:end_block, :], delta_gpn[start_block:end_block], \
                iters[start_block:end_block] = _ground_to_image(
                    coords_view[start_block:end_block, :], sicd, SCP, SCP_Pixel, uIPN, sf, row_ss, col_ss, uSPN,
                    row_col_transform, ipp_transform, delta_gp_max, max_iterations, **kwargs)
            start_block = end_block

    if len(orig_shape) > 1:
        image_points = numpy.reshape(image_points, orig_shape[:-1]+(2, ))
        delta_gpn = numpy.reshape(delta_gpn, orig_shape[:-1])
        iters = numpy.reshape(iters, orig_shape[:-1])
    return image_points, delta_gpn, iters


def _ground_to_image(coords, sicd,
                     SCP, SCP_Pixel, uIPN, sf, row_ss, col_ss, uProj, row_col_transform, ipp_transform,
                     delta_gp_max, max_iterations, **kwargs):
    # TODO: docstring - after this is finalized
    im_pts = numpy.zeros((coords.shape[0], 2), dtype=numpy.float64)
    del_gp = numpy.zeros((coords.shape[0],), dtype=numpy.float64)
    t_iters = numpy.zeros((coords.shape[0],), dtype=numpy.int16)
    it_here = numpy.ones((coords.shape[0],), dtype=numpy.bool)
    uGPN = (coords.T / numpy.linalg.norm(coords, axis=1)).T
    cont = True
    iteration = 0
    while cont:
        # project ground plane to image plane iteration
        iteration += 1
        g_n = coords[it_here, :]
        dist_n = numpy.dot(SCP - g_n, uIPN) / sf
        i_n = g_n + numpy.outer(dist_n, uProj)
        delta_ipp = i_n - SCP  # N x 3
        ip_iter = numpy.dot(numpy.dot(delta_ipp, row_col_transform), ipp_transform)
        im_pts[it_here, 0] = ip_iter[:, 0] / row_ss + SCP_Pixel[0]
        im_pts[it_here, 1] = ip_iter[:, 1] / col_ss + SCP_Pixel[1]
        # transform to ground plane containing the scene points
        # TODO: refactor this to use the appropriate direct method
        #   _image_to_ground_plane()
        gnd_pln = image_to_ground(
            im_pts[it_here, :], sicd, projection_type='PLANE', gref=g_n, ugpn=uGPN[it_here, :], **kwargs)
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


def ground_to_image_geo(coords, sicd, **kwargs):
    """

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
        Lat/Lon/HAE coordinate to map to scene coordinates, of size `N x 3`.
    sicd : SICDType
        SICD meta data structure.
    kwargs : dict
        See the key word arguments of :func:`ground_to_image`
    Returns
    -------
    Tuple[numpy.ndarray, float, int]
        * `image_points` - the determined image point array, of size `N x 2`. Following SICD convention,
           the upper-left pixel is [0, 0].
        * `delta_gpn` - residual ground plane displacement (m).
        * `iterations` - the number of iterations performed.

    """

    return ground_to_image(geocoords.geodetic_to_ecf(coords), sicd, **kwargs)


############
# Items for Image-To-Ground projections

def _validate_im_points(im_points):
    if not isinstance(im_points, numpy.ndarray):
        im_points = numpy.array(im_points, dtype=numpy.float64)

    if im_points.shape[-1] != 2:
        raise ValueError(
            'The im_points array must represent an array of points in pixel coordinates, '
            'so the final dimension of im_points must have length 2. Have im_points.shape = {}'.format(im_points.shape))
    return im_points


def _ric_ecf_mat(rarp, varp, frame_type):
    """
    Computes the ECF transformation matrix for RIC frame.

    Parameters
    ----------
    rarp : numpy.ndarray
    varp : numpy.ndarray
    frame_type : str
        the final three characters should be one of ['ECI', 'ECF']

    Returns
    -------
    numpy.ndarray
        the RIC transform matrix (array)
    """

    # Angular velocity of earth in radians/second, not including precession
    w = 7292115.1467E-11
    typ = frame_type.upper()[-3:]
    vi = varp if typ == 'ECF' else varp + numpy.cross([0, 0, w], rarp)

    r = rarp/numpy.linalg.norm(rarp)
    c = numpy.cross(r, vi)
    c /= numpy.linalg.norm(c)  # NB: perpendicular to r
    i = numpy.cross(c, r)
    # this is the cross of two perpendicular normal vectors, so normal
    return numpy.array([r, i, c], dtype=numpy.float64)


def _validate_basic(delta_arp=None, delta_varp=None, range_bias=0):
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
    if range_bias is None:
        range_bias = 0.0
    else:
        range_bias = float(range_bias)
    return delta_arp, delta_varp, range_bias


def _apply_adjustment(ARP_SCP_COA, VARP_SCP_COA,
                      adj_params_frame, r_tgt_coa, arp_coa, varp_coa, delta_arp, delta_varp, range_bias):
    if adj_params_frame in ['RIC_ECI', 'RIC_ECF']:
        # Translate from RIC frame to ECF frame
        # Use the RIC frame at SCP COA time, not at COA time for im_points
        T_ECEF_RIC = _ric_ecf_mat(ARP_SCP_COA, VARP_SCP_COA, adj_params_frame)
        # NB: transpose since we are multiplying on the right for dimensionality
        delta_arp = delta_arp.dot(T_ECEF_RIC.T)
        delta_varp = delta_varp.dot(T_ECEF_RIC.T)
    return arp_coa+delta_arp, varp_coa+delta_varp, r_tgt_coa+range_bias


def _coa_projection_set(im_points, sicd, SCP, row_ss, col_ss):
    """
    Computes the set of fundamental parameters for projecting a pixel down to the ground.

    Parameters
    ----------
    im_points : numpy.ndarray
    sicd : SICDType

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        * `r_tgt_coa` - range to the ARP at COA
        * `r_dot_tgt_coa` - range rate relative to the ARP at COA
        * `arp_coa` - aperture reference position at t_coa
        * `varp_coa` - velocity at t_coa
        * `t_coa` - center of aperture time since CDP start for input ip
    """

    # convert pixels coordinates to meters from SCP
    row = (im_points[:, 0] + sicd.ImageData.FirstRow - sicd.ImageData.SCPPixel.Row)*row_ss
    col = (im_points[:, 1] + sicd.ImageData.FirstCol - sicd.ImageData.SCPPixel.Col)*col_ss

    # compute target pixel time
    time_coa_poly = sicd.Grid.TimeCOAPoly
    # fall back to (bad?) approximation if TimeCOAPoly is not populated
    if time_coa_poly is None:
        time_coa_poly = Poly2DType(Coefs=[[sicd.Timeline.CollectDuration/2, ], ])
        logging.warning('Using (constant) approximation to TimeCOAPoly, which may result in poor projection results.')

    t_coa = time_coa_poly(row, col)
    # calculate aperture reference position and velocity at target time
    arp_coa = sicd.Position.ARPPoly(t_coa)
    varp_coa = sicd.Position.ARPPoly.derivative_eval(t_coa, der_order=1)

    # image grid to R/Rdot
    if sicd.Grid.Type == 'RGAZIM':
        # Compute range and range rate to the SCP at target pixel COA time
        ARP_minus_SCP = arp_coa - SCP
        rSCPTgtCoa = numpy.linalg.norm(ARP_minus_SCP, axis=-1)
        rDotSCPTgtCoa = numpy.sum(varp_coa*ARP_minus_SCP, axis=-1)/rSCPTgtCoa
        # This computation is dependent on Grid type and image formation algorithm.
        if sicd.ImageFormation.ImageFormAlgo == 'PFA':
            # Compute polar angle theta and its derivative wrt time at the target pixel COA
            thetaTgtCoa = sicd.PFA.PolarAngPoly(t_coa)
            dThetaDtTgtCoa = sicd.PFA.PolarAngPoly.derivative_eval(t_coa, der_order=1)
            # Compute polar aperture scale factor (KSF) and derivative wrt polar angle
            ksfTgtCoa = sicd.PFA.SpatialFreqSFPoly(thetaTgtCoa)
            dKsfDThetaTgtCoa = sicd.PFA.SpatialFreqSFPoly.derivative_eval(thetaTgtCoa, der_order=1)
            # Compute spatial frequency domain phase slopes in Ka and Kc directions
            # NB: sign for the phase may be ignored as it is cancelled in a subsequent computation.
            dPhiDKaTgtCoa = row*numpy.cos(thetaTgtCoa) + col*numpy.sin(thetaTgtCoa)
            dPhiDKcTgtCoa = -row*numpy.sin(thetaTgtCoa) + col*numpy.cos(thetaTgtCoa)
            # Compute range relative to SCP
            deltaRTgtCoa = ksfTgtCoa*dPhiDKaTgtCoa
            # Compute derivative of range relative to SCP wrt polar angle.
            # Scale by derivative of polar angle wrt time.
            dDeltaRDThetaTgtCoa = dKsfDThetaTgtCoa*dPhiDKaTgtCoa + ksfTgtCoa*dPhiDKcTgtCoa
            deltaRDotTgtCoa = dDeltaRDThetaTgtCoa*dThetaDtTgtCoa
        elif sicd.ImageFormation.ImageFormAlgo == 'RGAZCOMP':
            deltaRTgtCoa = row
            deltaRDotTgtCoa = -numpy.linalg.norm(varp_coa, axis=-1)*sicd.RgAzComp.AzSF*col
        else:
            raise ValueError('Unhandled Image Formation Algorithm {}'.format(sicd.ImageFormation.ImageFormAlgo))
        r_tgt_coa = rSCPTgtCoa + deltaRTgtCoa
        r_dot_tgt_coa = rDotSCPTgtCoa + deltaRDotTgtCoa
    elif sicd.Grid.Type == 'RGZERO':
        # compute range/time of closest approach
        R_CA_TGT = sicd.RMA.INCA.R_CA_SCP + row  # Range at closest approach
        t_CA_TGT = sicd.RMA.INCA.TimeCAPoly(col)  # Time of closest approach
        # Compute ARP velocity magnitude (actually squared, since that's how it's used) at t_CA_TGT
        VEL2_CA_TGT = numpy.sum(sicd.Position.ARPPoly.derivative_eval(t_CA_TGT, der_order=1)**2, axis=-1)
        # Compute the Doppler Rate Scale Factor for image Grid location
        DRSF_TGT = sicd.RMA.INCA.DRateSFPoly(row, col)
        # Difference between COA time and CA time
        dt_COA_TGT = t_coa - t_CA_TGT
        r_tgt_coa = numpy.sqrt(R_CA_TGT*R_CA_TGT + DRSF_TGT*VEL2_CA_TGT*dt_COA_TGT*dt_COA_TGT)
        r_dot_tgt_coa = (DRSF_TGT/r_tgt_coa)*VEL2_CA_TGT*dt_COA_TGT
    elif sicd.Grid.Type in ('XRGYCR', 'XCTYAT', 'PLANE'):
        uRow = sicd.Grid.Row.UVectECF.get_array()
        uCol = sicd.Grid.Col.UVectECF.get_array()
        ARP_minus_IPP = arp_coa - (SCP + numpy.outer(row, uRow) + numpy.outer(col, uCol))
        r_tgt_coa = numpy.linalg.norm(ARP_minus_IPP, axis=-1)
        r_dot_tgt_coa = numpy.sum(varp_coa * ARP_minus_IPP, axis=-1)/r_tgt_coa
    else:
        raise ValueError('Unhandled Grid type {}'.format(sicd.Grid.Type))
    return r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, t_coa


def image_to_ground(im_points, sicd, block_size=50000, projection_type='HAE',
                    delta_arp=None, delta_varp=None, range_bias=0,
                    adj_params_frame='ECF', **kwargs):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        (row, column) coordinates of N points in image (or subimage if FirstRow/FirstCol are nonzero).
        Following SICD convention, the upper-left pixel is [0, 0].
    sicd : SICDType
            SICD meta data structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    projection_type : str
        One of ['PLANE', 'HAE', 'DEM'].
    delta_arp : None|numpy.ndarray|list|tuple
        ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
    delta_varp : None|numpy.ndarray|list|tuple
        VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
    range_bias : float|int
        Range bias adjustable parameter (m), defaults to 0.
    adj_params_frame : str
        One of ['ECF', 'RIC_ECF', 'RIC_ECI'], specifying the coordinate frame used for
        expressing `delta_arp` and `delta_varp` parameters.
    kwargs : dict
        keyword arguments relevant for the given projection type. See image_to_ground_plane/hae/dem methods.

    Returns
    -------
    numpy.ndarray
        ECF Ground Points along the R/Rdot contour
    """

    p_type = projection_type.upper()
    if p_type == 'PLANE':
        return image_to_ground_plane(
            im_points, sicd, block_size=block_size,
            delta_arp=delta_arp, delta_varp=delta_varp, range_bias=range_bias,
            adj_params_frame=adj_params_frame, **kwargs)
    elif p_type == 'HAE':
        return image_to_ground_hae(
            im_points, sicd, block_size=block_size,
            delta_arp=delta_arp, delta_varp=delta_varp, range_bias=range_bias,
            adj_params_frame=adj_params_frame, **kwargs)
    elif p_type == 'DEM':
        return image_to_ground_dem(
            im_points, sicd, block_size=block_size,
            delta_arp=delta_arp, delta_varp=delta_varp, range_bias=range_bias,
            adj_params_frame=adj_params_frame, **kwargs)
    else:
        raise ValueError('Got unrecognized projection type {}'.format(projection_type))


def image_to_ground_geo(im_points, sicd, **kwargs):
    # TODO: docstring
    return geocoords.ecf_to_geodetic(image_to_ground(im_points, sicd, **kwargs))


#####
# Projection type is PLANE

def _image_to_ground_plane_perform(
        r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, uZ):
    # Solve for the intersection of a R/Rdot contour and a ground plane.
    arpZ = numpy.sum((arp_coa - gref)*uZ, axis=-1)
    arpZ[arpZ > r_tgt_coa] = numpy.nan
    # ARP ground plane nadir
    aGPN = arp_coa - numpy.outer(arpZ, uZ)
    # Compute ground plane distance (gd) from ARP nadir to circle of const range
    gd = numpy.sqrt(r_tgt_coa*r_tgt_coa - arpZ*arpZ)
    # Compute sine and cosine of grazing angle
    cosGraz = gd/r_tgt_coa
    sinGraz = arpZ/r_tgt_coa

    # Velocity components normal to ground plane and parallel to ground plane.
    vMag = numpy.linalg.norm(varp_coa, axis=-1)
    vZ = numpy.dot(varp_coa, uZ)
    vX = numpy.sqrt(vMag*vMag - vZ*vZ)  # Note: For Vx = 0, no Solution
    # Orient X such that Vx > 0 and compute unit vectors uX and uY
    uX = ((varp_coa - numpy.outer(vZ, uZ)).T/vX).T
    uY = numpy.cross(uZ, uX)
    # Compute cosine of azimuth angle to ground plane point
    cosAz = (-r_dot_tgt_coa+vZ*sinGraz) / (vX * cosGraz)
    cosAz[numpy.abs(cosAz) > 1] = numpy.nan  # R/Rdot combination not possible in given plane

    # Compute sine of azimuth angle. Use LOOK to establish sign.
    look = numpy.sign(numpy.dot(numpy.cross(arp_coa-gref, varp_coa), uZ))
    sinAz = look * numpy.sqrt(1-cosAz*cosAz)

    # Compute Ground Plane Point in ground plane and along the R/Rdot contour
    return aGPN + numpy.outer(gd*cosAz, uX) + numpy.outer(gd*sinAz, uY)


def _image_to_ground_plane(im_points, sicd, delta_arp, delta_varp, range_bias, adj_params_frame,
                           SCP, row_ss, col_ss, ARP_SCP_COA, VARP_SCP_COA, gref, uZ):
    # get (image formation specific) projection parameters
    r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, t_coa = _coa_projection_set(im_points, sicd, SCP, row_ss, col_ss)

    # Apply adjustable parameters if necessary
    arp_coa, varp_coa, r_tgt_coa = _apply_adjustment(
        ARP_SCP_COA, VARP_SCP_COA, adj_params_frame,
        r_tgt_coa, arp_coa, varp_coa, delta_arp, delta_varp, range_bias)

    return _image_to_ground_plane_perform(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, uZ)


def image_to_ground_plane(
        im_points, sicd, block_size=50000,
        delta_arp=None, delta_varp=None, range_bias=None, adj_params_frame='ECF',
        gref=None, ugpn=None):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    sicd : SICDType
        the SICD metadata structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    delta_arp : None|numpy.ndarray|list|tuple
        ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
    delta_varp : None|numpy.ndarray|list|tuple
        VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
    range_bias : float|int
        Range bias adjustable parameter (m), defaults to 0.
    adj_params_frame : str
        One of ['ECF', 'RIC_ECF', 'RIC_ECI'], specifying the coordinate frame used for
        expressing `delta_arp` and `delta_varp` parameters.
    gref : None|numpy.ndarray|list|tuple
        Ground plane reference point ECF coordinates (m). The default is the SCP
    ugpn : None|numpy.ndarray|list|tuple
        Vector normal to the plane to which we are projecting.

    Returns
    -------
    numpy.ndarray
        ECF Ground Points along the R/Rdot contour
    """

    if gref is None:
        gref = sicd.GeoData.SCP.ECF.get_array()
    if ugpn is None:
        ugpn = sicd.PFA.FPN.get_array() if sicd.ImageFormation.ImageFormAlgo == 'PFA' \
            else geocoords.wgs_84_norm(gref)
    if len(ugpn.shape) == 2:
        ugpn = numpy.reshape(ugpn, (3, ))
    uZ = ugpn/numpy.linalg.norm(ugpn)

    im_points = _validate_im_points(im_points)

    ARP_SCP_COA = sicd.SCPCOA.ARPPos.get_array()
    VARP_SCP_COA = sicd.SCPCOA.ARPVel.get_array()
    SCP = sicd.GeoData.SCP.ECF.get_array()
    row_ss = sicd.Grid.Row.SS
    col_ss = sicd.Grid.Col.SS

    delta_arp, delta_varp, range_bias = _validate_basic(delta_arp, delta_varp, range_bias)

    # prepare workspace
    orig_shape = im_points.shape
    im_points_view = numpy.reshape(im_points, (-1, 2))  # possibly or make 2-d flatten
    num_points = im_points_view.shape[0]
    if block_size is None or num_points <= block_size:
        coords = _image_to_ground_plane(
            im_points_view, sicd, delta_arp, delta_varp, range_bias, adj_params_frame,
            SCP, row_ss, col_ss, ARP_SCP_COA, VARP_SCP_COA, gref, uZ)
    else:
        coords = numpy.zeros((num_points, 3), dtype=numpy.float64)
        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block + block_size, num_points)
            coords[start_block:end_block, :] = _image_to_ground_plane(
                im_points_view[start_block:end_block], sicd, delta_arp, delta_varp, range_bias, adj_params_frame,
                SCP, row_ss, col_ss, ARP_SCP_COA, VARP_SCP_COA, gref, uZ)
            start_block = end_block

    if len(orig_shape) > 1:
        coords = numpy.reshape(coords, orig_shape[:-1] + (3,))
    return coords


#####
# Projection type is HAE

def _image_to_ground_hae(
        im_points, sicd, delta_arp, delta_varp, range_bias, adj_params_frame,
        SCP, row_ss, col_ss, ARP_SCP_COA, VARP_SCP_COA, hae0, delta_hae_max, hae_nlim, scp_hae):

    # get (image formation specific) projection parameters
    r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, t_coa = _coa_projection_set(im_points, sicd, SCP, row_ss, col_ss)

    # Apply adjustable parameters if necessary
    arp_coa, varp_coa, r_tgt_coa = _apply_adjustment(
        ARP_SCP_COA, VARP_SCP_COA, adj_params_frame,
        r_tgt_coa, arp_coa, varp_coa, delta_arp, delta_varp, range_bias)

    # Compute the geodetic ground plane normal at the SCP.
    ugpn = geocoords.wgs_84_norm(SCP)
    look = numpy.sign(numpy.dot(numpy.cross(arp_coa, varp_coa), SCP-arp_coa))
    gref = SCP - (scp_hae - hae0)*ugpn
    # iteration variables
    gpp = None
    delta_hae = None
    cont = True
    iters = 0
    while cont:
        iters += 1
        # Compute the precise projection along the R/Rdot contour to Ground Plane n.
        gpp = _image_to_ground_plane_perform(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, ugpn)
        # Compute the unit vector in the increasing height direction
        ugpn = geocoords.wgs_84_norm(gpp)
        # check our hae value versus hae0
        gpp_llh = geocoords.ecf_to_geodetic(gpp)
        delta_hae = gpp_llh[:, 2] - hae0
        gref = gpp - numpy.outer(delta_hae, ugpn)
        # should we stop our iteration?
        cont = numpy.all(numpy.abs(delta_hae) > delta_hae_max) and (iters <= hae_nlim)
    # TODO: what if the above is bad?
    # Compute the unit slant plane normal vector, uspn, that is tangent to the R/Rdot contour at point gpp
    spn = (numpy.cross(varp_coa, (gpp - arp_coa)).T*look).T
    uspn = (spn.T/numpy.linalg.norm(spn, axis=-1)).T
    # For the final straight line projection, project from point gpp along
    # the slant plane normal (as opposed to the ground plane normal that was
    # used in the iteration) to point slp.
    sf = numpy.sum(ugpn*uspn, axis=-1)
    slp = gpp - (uspn.T*delta_hae/sf).T
    # Assign surface point SPP position by adjusting the HAE to be on the
    # HAE0 surface.
    spp_llh = geocoords.ecf_to_geodetic(slp)
    spp_llh[:, 2] = hae0
    spp = geocoords.geodetic_to_ecf(spp_llh)
    return spp


def image_to_ground_hae(im_points, sicd, block_size=50000,
                        delta_arp=None, delta_varp=None, range_bias=None, adj_params_frame='ECF',
                        hae0=None, delta_hae_max=None, hae_nlim=None):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    sicd : SICDType
        the SICD metadata structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    delta_arp : None|numpy.ndarray|list|tuple
        ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
    delta_varp : None|numpy.ndarray|list|tuple
        VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
    range_bias : float|int
        Range bias adjustable parameter (m), defaults to 0.
    adj_params_frame : str
        One of ['ECF', 'RIC_ECF', 'RIC_ECI'], specifying the coordinate frame used for
        expressing `delta_arp` and `delta_varp` parameters.
    hae0 : None|float|int
        Surface height (m) above the WGS-84 reference ellipsoid for projection point.
    delta_hae_max : None|float|int
        Height threshold for convergence of iterative constant HAE computation (m). Defaults to 1.
    hae_nlim : int
        Maximum number of iterations allowed for constant hae computation. Defaults to 5.

    Returns
    -------
    numpy.ndarray
        ECF Ground Points along the R/Rdot contour
    """

    scp_hae = sicd.GeoData.SCP.LLH.HAE
    if hae0 is None:
        hae0 = scp_hae

    if delta_hae_max is None:
        delta_hae_max = 1.0  # TODO: is this the best value?
    delta_hae_max = float(delta_hae_max)
    if delta_hae_max <= 1e-4:
        raise ValueError('delta_hae_max must be at least 1e-4. Got {0:8f}'.format(delta_hae_max))
    if hae_nlim is None:
        hae_nlim = 5  # TODO: is this the best value?
    hae_nlim = int(hae_nlim)
    if hae_nlim <= 0:
        raise ValueError('hae_nlim must be a positive integer. Got {}'.format(hae_nlim))

    im_points = _validate_im_points(im_points)

    ARP_SCP_COA = sicd.SCPCOA.ARPPos.get_array()
    VARP_SCP_COA = sicd.SCPCOA.ARPVel.get_array()
    SCP = sicd.GeoData.SCP.ECF.get_array()
    row_ss = sicd.Grid.Row.SS
    col_ss = sicd.Grid.Col.SS

    delta_arp, delta_varp, range_bias = _validate_basic(delta_arp, delta_varp, range_bias)

    # prepare workspace
    orig_shape = im_points.shape
    im_points_view = numpy.reshape(im_points, (-1, 2))  # possibly or make 2-d flatten
    num_points = im_points_view.shape[0]
    if block_size is None or num_points <= block_size:
        coords = _image_to_ground_hae(
            im_points_view, sicd, delta_arp, delta_varp, range_bias, adj_params_frame,
            SCP, row_ss, col_ss, ARP_SCP_COA, VARP_SCP_COA, hae0, delta_hae_max, hae_nlim, scp_hae)
    else:
        coords = numpy.zeros((num_points, 3), dtype=numpy.float64)
        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block + block_size, num_points)
            coords[start_block:end_block, :] = _image_to_ground_hae(
                im_points_view[start_block:end_block], sicd, delta_arp, delta_varp, range_bias, adj_params_frame,
                SCP, row_ss, col_ss, ARP_SCP_COA, VARP_SCP_COA, hae0, delta_hae_max, hae_nlim, scp_hae)
            start_block = end_block

    if len(orig_shape) > 1:
        coords = numpy.reshape(coords, orig_shape[:-1] + (3,))
    return coords


#####
# Projection type is HAE

def image_to_ground_dem(im_points, sicd, block_size=50000,
                        delta_arp=None, delta_varp=None, range_bias=None, adj_params_frame='ECF',
                        dem=None, dem_type='SRTM2F', geoid_path=None, del_DISTrrc=10, del_HDlim=0.001):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    sicd : SICDType
        the SICD metadata structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be transformed as a single block if `None`.
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

    raise NotImplementedError
