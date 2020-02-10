'''This module contains the functions to map between the pixel grid in the
image space and geolocated points in 3D space.'''
import os


# SarPy imports
from . import geocoords as gc
# Python standard library imports
import copy
import warnings
# External dependencies
import numpy as np
from numpy.polynomial import polynomial as poly
from ..io.DEM import geoid, DEM

__classification__ = "UNCLASSIFIED"
__email__ = "Wade.C.Schwartzkopf.ctr@nga.mil"


def ground_to_image(s, sicd_meta, delta_gp_max=None,
                    # Adjustable parameters
                    delta_arp=[0, 0, 0], delta_varp=[0, 0, 0],
                    range_bias=0, adj_params_frame='ECF'):
    '''Transforms a 3D ECF point to pixel row, column
    This function implements the SICD Image Projections Description Document:
    http://www.gwg.nga.mil/ntb/baseline/docs/SICD/index.html

    [ip, deltaGPn, counter] = ground_to_image(s, sicd_meta, ...)

    Inputs:
        s            - [Nx3] N scene points in ECF coordinates
        sicd_meta    - SICD meta data structure
        delta_gp_max - Ground plane displacement tol (m), default = quarter pixel
        delta_arp    - ARP position adjustable parameter (ECF, m).  Default 0.
        delta_varp   - VARP position adjustable parameter (ECF, m/s).  Default 0.
        range_bias   - Range bias adjustable parameter (m).  Default 0.
        adj_params_frame - Coordinate frame used for expressing delta_arp
                         and delta_varp adjustable parameters.  Allowed
                         values: 'ECF', 'RIC_ECF', 'RIC_ECI'. Default ECF.

     Outputs:
        ip           - [2xN] (row, column) coordinates of N points in image
                       (or subimage if FirstRow/FirstCol are nonzero).
                       Zero-based, following SICD (and Python) convention; that
                       is upper-left pixel is [0, 0].
        deltaGPn     - Residual ground plane displacement (m)
        counter      - # iterations required

     Contributors: Thomas McDowall, Harris Corporation
                   Wade Schwartzkopf, NGA/R
                   Clayton Williams, NRL
    '''

    # Extract the relevant SICD info
    scp_row = sicd_meta.ImageData.SCPPixel.Row
    scp_col = sicd_meta.ImageData.SCPPixel.Col
    FirstRow = sicd_meta.ImageData.FirstRow
    FirstCol = sicd_meta.ImageData.FirstCol
    # We override this so that imageToGround will work in the coordinate
    # space of full image. We will add it back in at the end.
    sicd_meta.ImageData.FirstRow = 0
    sicd_meta.ImageData.FirstCol = 0
    row_ss = sicd_meta.Grid.Row.SS
    col_ss = sicd_meta.Grid.Col.SS
    if delta_gp_max is None:
        delta_gp_max = 0.25 * np.sqrt(row_ss*row_ss + col_ss*col_ss)
    uRow = np.array([sicd_meta.Grid.Row.UVectECF.X,
                     sicd_meta.Grid.Row.UVectECF.Y,
                     sicd_meta.Grid.Row.UVectECF.Z])
    uCol = np.array([sicd_meta.Grid.Col.UVectECF.X,
                     sicd_meta.Grid.Col.UVectECF.Y,
                     sicd_meta.Grid.Col.UVectECF.Z])
    SCP = np.array([sicd_meta.GeoData.SCP.ECF.X,
                    sicd_meta.GeoData.SCP.ECF.Y,
                    sicd_meta.GeoData.SCP.ECF.Z])
    arpSCPCOA = np.array([sicd_meta.SCPCOA.ARPPos.X,
                          sicd_meta.SCPCOA.ARPPos.Y,
                          sicd_meta.SCPCOA.ARPPos.Z])
    varpSCPCOA = np.array([sicd_meta.SCPCOA.ARPVel.X,
                           sicd_meta.SCPCOA.ARPVel.Y,
                           sicd_meta.SCPCOA.ARPVel.Z])

    # 3.1 SCP Projection Equations
    # Normal to instantaneous slant plane that contains SCP at SCP COA is
    # tangent to R/Rdot contour at SCP. Points away from center of Earth. Use
    # LOOK to establish sign.
    spnSCPCOA = np.cross(varpSCPCOA, SCP - arpSCPCOA)
    if sicd_meta.SCPCOA.SideOfTrack == 'R':
        spnSCPCOA = -1*spnSCPCOA
    uSpn = spnSCPCOA/np.sqrt(np.sum(spnSCPCOA*spnSCPCOA))

    # 6.1 Scene To Image
    # 1) Spherical earth ground plane normal-- exact orientation of plane is not
    # critical, as long as it contains point(s)
    s = np.atleast_2d(s)
    # linalg.norm does not work with axis parameter in earlier versions of numpy (< 1.8.0)
    # uGpn = s / np.linalg.norm(s, ord=2, axis=-1, keepdims=True)
    # keepdims was not a valid option in np.sum until numpy version 1.7
    uGpn = s / np.sqrt(np.sum(np.power(s, 2), axis=-1, keepdims=True))

    # 2) Ground plane points are projected along straight lines to the image plane. The
    # GP to IP direction is along the SCP COA slant plane normal. Also, compute
    # image plane unit normal, uIPN. Compute projection scale factor SF.
    uProj = uSpn
    ipn = np.cross(uRow, uCol)  # should match SICD.PFA.IPN for PFA data
    uIpn = ipn/np.sqrt(np.sum(ipn*ipn))  # pointing away from center of earth
    sf = np.dot(uProj, uIpn)

    # 2.4 - Image Plane parameters
    # The following section is for ground to image with non-orthogonal axes.
    # theta col is angle between uRow and uCol if not 0.
    cosThetaCol = np.dot(uRow, uCol)
    sinThetaCol = np.sqrt(1-(cosThetaCol*cosThetaCol))
    gI = 1.0/(sinThetaCol*sinThetaCol) * np.array([[1, -cosThetaCol], [-cosThetaCol, 1]])

    # Iterate the ground to image transform
    g_n = s.copy()
    # New version of numpy allow for a full method, but this works on older
    # versions as well.
    to_iter = np.empty(s.shape[0], dtype=bool)
    to_iter.fill(True)
    counter = np.zeros(s.shape[0])
    ip = np.empty([s.shape[0], 2]) * np.nan
    deltaPn = np.empty([s.shape[0], 3]) * np.nan
    deltaGPn = np.empty(s.shape[0]) * np.nan
    while any(counter < 5):
        # 4) Project ground plane point g_n to image plane point i_n. The
        # projection distance is dist_n. Compute image coordinates xrow and ycol.
        dist_n = (1.0/sf) * np.dot(SCP - g_n, uIpn)
        i_n = g_n + np.outer(dist_n, uProj)
        # Back to section 2.4.  Convert from image plane point (IPP) to row/col
        deltaIpp = i_n - SCP
        ipIter = np.dot(gI, np.vstack((np.dot(deltaIpp, uRow), np.dot(deltaIpp, uCol))))
        xrow = ipIter[0]
        ycol = ipIter[1]
        irow = xrow / row_ss
        icol = ycol / col_ss
        row = irow + scp_row
        col = icol + scp_col
        ip[to_iter, :] = np.column_stack((row, col))
        # 5) Transform to ground plane containing the scene point(s)
        pN = image_to_ground(ip[to_iter, :], sicd_meta, projection_type='plane',
                             gref=s[to_iter, :], ugpn=uGpn[to_iter, :],
                             # Pass adjustable parameters
                             delta_arp=delta_arp, delta_varp=delta_varp,
                             range_bias=range_bias, adj_params_frame=adj_params_frame)
        # 6) Compute displacement between plane points (Pn) and scene points (s)
        deltaPn[to_iter, :] = s[to_iter, :] - pN
        deltaGPn[to_iter] = np.sqrt(np.sum(np.power(deltaPn[to_iter, :], 2), -1))
        # Need to iterate further on these points
        old_iter = to_iter
        to_iter = deltaGPn > delta_gp_max
        if any(to_iter):
            g_n = g_n[to_iter[old_iter], :] + deltaPn[to_iter, :]
            counter[to_iter] = counter[to_iter] + 1
        else:
            break
    if any(counter == 5):
        counter[counter == 5] = counter[counter == 5] - 1
        warnings.warn('ground_to_image computation did not converge.')

    # Make adjustments for partial vs full image
    ip[:, 0] = ip[:, 0] - FirstRow
    ip[:, 1] = ip[:, 1] - FirstCol
    # Since it is passed in by reference, we might need to reset sicd_meta
    sicd_meta.ImageData.FirstRow = FirstRow
    sicd_meta.ImageData.FirstCol = FirstCol

    return ip, deltaGPn, counter+1


def ground_to_image_geo(s, *args, **kwargs):
    """This doesn't really do anything but is defined as a convenience
    function since this is often how it is called."""
    return ground_to_image(gc.geodetic_to_ecf(s), *args, **kwargs)


def image_to_ground(im_points, sicd_meta, projection_type='hae',  # CSM default is hae
                    # Required for projection_type='plane'
                    gref=None, ugpn=None,
                    # Required for projection_type='hae'
                    hae0=None, delta_hae_max=1, hae_nlim=3,
                    # Adjustable parameters
                    delta_arp=[0, 0, 0], delta_varp=[0, 0, 0], range_bias=0,
                    adj_params_frame='ECF', dem=None, dem_type='SRTM2F', geoid_path=None,
                    del_DISTrrc=10, del_HDlim=.001):
    '''Transforms pixel row, col to ground ECF coordinate
     This function implements the SICD Image Projections Description Document:
     http://www.gwg.nga.mil/ntb/baseline/docs/SICD/index.html

     gpos = image_to_ground(im_points, sicd_meta, ...)

     Inputs:
        im_points   - [Nx2] (row, column) coordinates of N points in image (or
                      subimage if FirstRow/FirstCol are nonzero).  Zero-based,
                      following SICD (and Python) convention; that is, upper-left
                      pixel is [0, 0].
        sicd_meta    - SICD meta data structure
        projection_type - 'plane', 'hae', 'dem'.  Default is hae.
        gref              Ground plane reference point ECF coordinates (m).
                          Default is SCP.  Only valid if projection_type is
                          'plane'.
        ugpn            - Ground plane unit normal vector.  Default is
                          tangent to the surface of constant geodetic
                          height above the WGS-84 reference ellipsoid
                          passing through GREF.  Only valid if
                          projection_type is 'plane'.
        hae0            - Surface height (m) above the WGS-84 reference
                          ellipsoid for projection point.  Only valid if
                          projection_type is 'hae'.
        delta_hae_max   - Height threshold for convergence of iterative
                          constant HAE computation (m).  Default 1 meter.
                          Only valid if projection_type is 'hae'.
        hae_nlim        - Maximum number of iterations allowed for constant
                          hae computation.  Default 3.  Only valid if
                          projection_type is 'hae'.
        delta_arp       - ARP position adjustable parameter (ECF, m).  Default 0.
        delta_varp      - VARP position adjustable parameter (ECF, m/s).  Default 0.
        range_bias      - Range bias adjustable parameter (m).  Default 0.
        adj_params_frame - Coordinate frame used for expressing delta_arp
                          and delta_varp adjustable parameters.  Allowed
                          values: 'ECF', 'RIC_ECF', 'RIC_ECI'. Default ECF.
       dem               SRTM pathname or structure with
                         lats/lons/elevations fields where are all are
                         arrays of same size and elevation is height above
                         WGS-84 ellipsoid.
                         Only valid if projection_type is 'dem'.
       dem_type        - One of 'DTED1', 'DTED2', 'SRTM1', 'SRTM2', 'SRTM2F'
                         Defaults to SRTM2f
       geoid_path      - Parent path to EGM2008 file(s)
       del_DISTrrc       Maximum distance between adjacent points along
                         the R/Rdot contour. Recommended value: 10.0 m.
                         Only valid if projection_type is 'dem'.
       del_HDlim         Height difference threshold for determining if a
                         point on the R/Rdot contour is on the DEM
                         surface (m).  Recommended value: 0.001 m.  Only
                         valid if projection_type is 'dem'.

     Outputs:
        gpos        - [Nx3] ECF Ground Points along the R/Rdot contour

     Contributors: Thomas McDowall, Harris Corporation
                   Lisa Talbot, NGA/IB
                   Rocco Corsetti, NGA/IB
                   Wade Schwartzkopf, NGA/R
                   Clayton Williams, NRL
    '''

    # Compute defaults if necessary
    if gref is None:
        gref = np.array([sicd_meta.GeoData.SCP.ECF.X,
                         sicd_meta.GeoData.SCP.ECF.Y,
                         sicd_meta.GeoData.SCP.ECF.Z])
    if ugpn is None:
        if sicd_meta.ImageFormation.ImageFormAlgo == 'PFA':
            ugpn = np.array([sicd_meta.PFA.FPN.X,
                             sicd_meta.PFA.FPN.Y,
                             sicd_meta.PFA.FPN.Z])
        else:
            ugpn = np.empty(3)
            ugpn[0], ugpn[1], ugpn[2] = gc.wgs_84_norm(gref[0], gref[1], gref[2])

    # Chapter 4: Compute projection parameters
    try:
        # Computation of r/rdot is specific to image formation type
        [r, rDot, arp_coa, varp_coa, t_coa] = coa_projection_set(sicd_meta, im_points)
    # If full metadata not available, try to make the approximation of uniformly sampled Grid
    except Exception:
        # Use a copy to avoid changing original reference.
        sicd_approx = copy.deepcopy(sicd_meta)
        sicd_approx.Grid.Type = 'PLANE'
        # Another approximation that may have to be made
        if (not hasattr(sicd_approx.Grid, 'TimeCOAPoly') and
           hasattr(sicd_approx, 'Timeline') and
           hasattr(sicd_approx.Timeline, 'CollectDuration')):
            sicd_approx.Grid.TimeCOAPoly = sicd_approx.Timeline.CollectDuration/2
        [r, rDot, arp_coa, varp_coa, t_coa] = coa_projection_set(sicd_approx, im_points)
        warnings.warn('Unable to compute precise position due to incomplete metadata.  ' +
                      'Resorting to approximation.')
    # After r/rdot is computed, the rest is generic to all SAR.

    # Apply adjustable parameters if necessary
    if adj_params_frame == 'ECF':
        # No transformation necessary
        pass
    else:  # Translate from RIC frame to ECF frame
        # Use the RIC frame at SCP COA time, not at COA time for im_points
        ARP_SCP_COA = np.array([sicd_meta.SCPCOA.ARPPos.X,
                                sicd_meta.SCPCOA.ARPPos.Y,
                                sicd_meta.SCPCOA.ARPPos.Z])
        VARP_SCP_COA = np.array([sicd_meta.SCPCOA.ARPVel.X,
                                 sicd_meta.SCPCOA.ARPVel.Y,
                                 sicd_meta.SCPCOA.ARPVel.Z])
        if adj_params_frame == 'RIC_ECI':
            T_ECEF_RIC = gc.ric_ecf_mat(ARP_SCP_COA, VARP_SCP_COA, 'eci')
        elif adj_params_frame == 'RIC_ECF':
            T_ECEF_RIC = gc.ric_ecf_mat(ARP_SCP_COA, VARP_SCP_COA, 'ecf')
        delta_arp = np.array(delta_arp * T_ECEF_RIC)  # Switch from matrix back to array
        delta_varp = np.array(delta_varp * T_ECEF_RIC)
    arp_coa = arp_coa + delta_arp
    varp_coa = varp_coa + delta_varp
    r = r + range_bias

    # Perform actual projection
    if projection_type.lower() == 'plane':
        gpos = projection_set_to_plane(r, rDot, arp_coa, varp_coa, gref, ugpn)
    elif projection_type.lower() == 'hae':
        scp = np.array([sicd_meta.GeoData.SCP.ECF.X,
                        sicd_meta.GeoData.SCP.ECF.Y,
                        sicd_meta.GeoData.SCP.ECF.Z])
        gpos = projection_set_to_hae(r, rDot, arp_coa, varp_coa,
                                     scp, hae0, delta_hae_max, hae_nlim)
    elif projection_type.lower() == 'dem':
        scp = np.array([sicd_meta.GeoData.SCP.ECF.X,
                        sicd_meta.GeoData.SCP.ECF.Y,
                        sicd_meta.GeoData.SCP.ECF.Z])
        gpos = projection_set_to_dem(r, rDot, arp_coa, varp_coa, scp, delta_hae_max, hae_nlim, dem,
                                     dem_type, geoid_path, del_DISTrrc, del_HDlim)

    else:
        raise ValueError('Unrecognized projection type.')

    return gpos


def image_to_ground_geo(*args, **kwargs):
    """This doesn't really do anything but is defined as a convenience
    function since this is often how it is called."""
    return gc.ecf_to_geodetic(image_to_ground(*args, **kwargs))


def projection_set_to_plane(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, gpn):
    ''' Transforms pixel row, col to ground plane ECF coordinate
     via algorithm in SICD Image Projections document.

     GPP = projection_set_to_plane(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, gpn)

     Inputs:
        r_tgt_coa     - [1xN] range to the ARP at COA
        r_dot_tgt_coa - [1xN] range rate relative to the ARP at COA
        arp_coa       - [Nx3] aperture reference position at t_coa
        varp_coa      - [Nx3] velocity at t_coa
        gref          - [3x1] reference point in the plane to which we are projecting
        gpn           - [3x1] vector normal to the plane to which we are projecting

     Outputs:
        GPP        - [3xN] ECF Ground Plane Point along the R/Rdot contour

     Contributors: Thomas McDowall, Harris Corporation
                   Wade Schwartzkopf, NGA/R
                   Clayton Williams, NRL
    '''

    # Ground plane description could be of single plane for all points
    # (probably the more typical case) or a plane for each point.

    # 5. Precise R/Rdot to Ground Plane Projection
    # Solve for the intersection of a R/Rdot contour and a ground plane.
    uZ = gpn / np.sqrt(np.sum(np.power(gpn, 2), axis=-1, keepdims=True))

    # ARP distance from plane
    arpZ = np.sum((arp_coa - gref) * uZ, axis=-1)
    arpZ[arpZ > r_tgt_coa] = np.nan

    # ARP ground plane nadir
    aGPN = arp_coa - arpZ[:, np.newaxis] * uZ

    # Compute ground plane distance (gd) from ARP nadir to circle of const range
    gd = np.sqrt(r_tgt_coa*r_tgt_coa - arpZ*arpZ)

    # Compute sine and cosine of grazing angle
    cosGraz = gd / r_tgt_coa
    sinGraz = arpZ / r_tgt_coa

    # Velocity components normal to ground plane and parallel to ground plane.
    vMag = np.sqrt(np.sum(np.power(varp_coa, 2), axis=-1))
    vZ = np.sum(varp_coa * uZ, axis=-1)
    vX = np.sqrt(vMag*vMag - vZ*vZ)  # Note: For Vx = 0, no Solution

    # Orient X such that Vx > 0 and compute unit vectors uX and uY
    uX = 1/vX[:, np.newaxis] * (varp_coa - vZ[:, np.newaxis] * uZ)
    uY = np.cross(uZ, uX)

    # Compute cosine of azimuth angle to ground plane point
    cosAz = (-r_dot_tgt_coa+vZ*sinGraz) / (vX * cosGraz)
    cosAz[(cosAz > 1) | (cosAz < -1)] = np.nan  # R/Rdot combination not possible in given plane
    # Don't use to get AZ. Sign may be adjusted in next step.

    # Compute sine of azimuth angle. Use LOOK to establish sign.
    look = np.sign(np.sum(gpn * np.cross(arp_coa-gref, varp_coa), axis=-1))[0]
    sinAz = look * np.sqrt(1-cosAz*cosAz)

    # Compute Ground Plane Point in ground plane and along the R/Rdot contour
    gPP = aGPN + ((gd*cosAz)[:, np.newaxis]*uX) + ((gd*sinAz)[:, np.newaxis]*uY)  # in ECF

    return gPP


def projection_set_to_hae(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa,
                          scp, hae0=None, delta_hae_max=1, nlim=3):
    ''' Transforms pixel row, col to constant height above the ellipsoid
    via algorithm in SICD Image Projections document.

     spp = projection_set_to_hae(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa,
                                 scp, hae0, delta_hae_max, nlim)

     Inputs:
        r_tgt_coa     - [1xN] range to the ARP at COA
        r_dot_tgt_coa - [1xN] range rate relative to the ARP at COA
        arp_coa       - [Nx3] aperture reference position at t_coa
        varp_coa      - [Nx3] velocity at t_coa
        scp           - [3] scene center point (ECF meters)
        hae0          - surface height (m) above the WGS-84 reference ellipsoid
                        for projection point spp
        delta_hae_max - height threshold for convergence of iterative
                        projection sequence.
        nlim          - maximum number of iterations allowed.

     Outputs:
        spp           - [Nx3] Surface Projection Point position on the hae0
                              surface and along the R/Rdot contour

     Contributors: Rocco Corsetti, NGA/IB
                   Wade Schwartzkopf, NGA/R
    '''

    # 9. Precise R/Rdot To Constant HAE Surface Projection
    iters = 0
    delta_hae = np.inf
    # (1) Compute the geodetic ground plane normal at the SCP.
    ugpn = gc.wgs_84_norm(scp)
    look = np.sign(np.sum(np.cross(arp_coa, varp_coa) * (scp - arp_coa), axis=-1))
    gpp_llh = gc.ecf_to_geodetic(scp)
    if hae0 is None:
        hae0 = gpp_llh[:, 2]
    # gref = scp + ((hae0 - gpp_llh[:, 2]) * ugpn)
    gref = scp - (gpp_llh[:, 2] - hae0) * ugpn
    while np.all(np.abs(delta_hae) > delta_hae_max) and (iters <= nlim):
        # (2) Compute the precise projection along the R/Rdot contour to Ground Plane n.
        gpp = projection_set_to_plane(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, ugpn)
        # (3) Compute the unit vector in the increasing height direction
        ugpn = gc.wgs_84_norm(gpp)
        gpp_llh = gc.ecf_to_geodetic(gpp)
        delta_hae = gpp_llh[:, 2] - hae0
        gref = gpp - (delta_hae[:, np.newaxis] * ugpn)
        iters = iters + 1
        # (4) Test for delta_hae_MAX and NLIM
    # (5) Compute the unit slant plane normal vector, uspn, that is tangent to
    # the R/Rdot contour at point gpp
    spn = look[:, np.newaxis] * np.cross(varp_coa, (gpp - arp_coa))
    uspn = spn/np.sqrt(np.sum(np.power(spn, 2), axis=-1, keepdims=True))
    # (6) For the final straight line projection, project from point gpp along
    # the slant plane normal (as opposed to the ground plane normal that was
    # used in the iteration) to point slp.
    sf = np.sum(ugpn * uspn, axis=-1)
    slp = gpp - ((delta_hae / sf)[:, np.newaxis] * uspn)
    # (7) Assign surface point SPP position by adjusting the HAE to be on the
    # HAE0 surface.
    spp_llh = gc.ecf_to_geodetic(slp)
    spp_llh[:, 2] = hae0
    spp = gc.geodetic_to_ecf(spp_llh)
    return spp


def projection_set_to_dem(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, scp, delta_hae_max,
                          nlim, dem, dem_type, geoid_path, del_DISTrrc=10, del_HDlim=.001):
    ''' Transforms pixel row, col to geocentric value projected to dem
    via algorithm in SICD Image Projections document.

     gpp = projection_set_to_dem(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa,
                                 scp, delta_hae_max, nlim, dem, del_DISTrrc = 10,
                                 del_HDlim = .001):
     Inputs:
        r_tgt_coa     - [N] range to the ARP at COA
        r_dot_tgt_coa - [N] range rate relative to the ARP at COA
        arp_coa       - [Nx3] aperture reference position at t_coa
        varp_coa      - [Nx3] velocity at t_coa
        scp           - [3] scene center point (ECF meters)
        delta_hae_max - height threshold for convergence of iterative
                        projection sequence.
        nlim          - maximum number of iterations allowed.
        dem              DTED/SRTM pathname or structure with
                         lats/lons/elevations fields where are all are
                         arrays of same size and elevation is height above
                         WGS-84 ellipsoid.
                         Only valid if projection_type is 'dem'.
        dem_type      - One of 'DTED1', 'DTED2', 'SRTM1', 'SRTM2', 'SRTM2F'
                        Defaults to SRTM2f
        geoid_path    - Parent path to EGM2008 file(s)
        del_DISTrrc       Maximum distance between adjacent points along
                         the R/Rdot contour. Recommended value: 10.0 m.
                         Only valid if projection_type is 'dem'.
        del_HDlim         Height difference threshold for determining if a
                         point on the R/Rdot contour is on the DEM
                         surface (m).  Recommended value: 0.001 m.  Only
                         valid if projection_type is 'dem'.

     Outputs:
        gpp           - [Nx3] ECF Ground Points along the R/Rdot contour
    '''
    ugpn = gc.wgs_84_norm(scp)
    look = np.sign(np.sum(ugpn*np.cross(arp_coa - scp, varp_coa, axis=1), axis=1))[0]

    if(len(dem) > 0 and os.path.exists(dem)):
        # Load DEM data for 5km buffer around points roughly projected to
        # height above ellipsoid provided in SICD SCP.
        gpos = gc.ecf_to_geodetic(projection_set_to_plane(r_tgt_coa, r_dot_tgt_coa, arp_coa,
                                                          varp_coa, scp, ugpn))
        LL = np.array([gpos[:, 0].min(), gpos[:, 1].min()]) - (
                      5 * np.array([1, 1/np.cos(gpos[:, 0].min() * (np.pi / 180.))]) / 111.111)
        UR = np.array([gpos[:, 0].max(), gpos[:, 1].max()]) + (
                      5*np.array([1, 1/np.cos(gpos[:, 0].max()*(np.pi / 180.))])/111.111)
        coord = [[LL[0], LL[1]], [UR[0], UR[1]]]
        evaldem = DEM.DEM(coordinates=coord, masterpath=dem,
                          dem_type='SRTM2F', log_level="WARNING")
        withinLat = (evaldem.lats_1D > LL[0]) & (evaldem.lats_1D < UR[0])
        withinLon = (evaldem.lons_1D > LL[1]) & (evaldem.lons_1D < UR[1])
        latindx = np.where(withinLat)[0]
        lonindx = np.where(withinLon)[0]
        aoicoords = np.empty((np.size(latindx)*np.size(lonindx), ), dtype=object)
        for i, v in enumerate(aoicoords):
            aoicoords[i] = [v, i]
        geoid_file = geoid_path + os.path.sep + 'egm2008-1.pgm'
        eval_egm = geoid.GeoidHeight(name=geoid_file)
        indx = 0
        egm_elevlist = np.empty(np.size(latindx)*np.size(lonindx))
        aoielevs = np.empty(np.size(latindx)*np.size(lonindx))
        for lat_indx in range(0, latindx.size):
            for lon_indx in range(0, lonindx.size):
                local_lat = evaldem.lats_1D[latindx[lat_indx]]
                local_lon = evaldem.lons_1D[lonindx[lon_indx]]
                egm_elevlist[indx] = eval_egm.get(local_lat, local_lon, cubic=True)
                aoicoords[indx] = [local_lat, local_lon]
                indx = indx + 1
        coordlist = aoicoords.tolist()
        aoielevs_dem = evaldem.elevate(coord=coordlist, method='nearest')
        for indx_elev in range(0, aoielevs.size):
            aoielevs[indx_elev] = aoielevs_dem[indx_elev] + egm_elevlist[indx_elev]

    numPoints = r_tgt_coa.size
    gpp = np.zeros([numPoints, 3])
    for i in range(numPoints):
        # Step 1 - Compute the center point and the radius of the R/RDot projection contour, Rrrc
        vMag = np.linalg.norm(varp_coa[i, ])
        uVel = varp_coa[i, ]/vMag
        cos_dca = -1*r_dot_tgt_coa[i]/vMag
        sin_dca = np.sqrt(1-cos_dca*cos_dca)
        ctr = arp_coa[i, ]+r_tgt_coa[i]*cos_dca*uVel
        Rrrc = r_tgt_coa[i]*sin_dca
        # Step 2 - Compute unit vectors uRRX and uRRY
        dec_arp = np.linalg.norm(arp_coa[i, ])
        uUP = arp_coa[i, ]/dec_arp
        RRY = np.cross(uUP, uVel)
        uRRY = RRY/np.linalg.norm(RRY)
        uRRX = np.cross(uRRY, uVel)
        # Step 3 - Project R/Rdot contour to constant height HAEmax
        Aa = projection_set_to_hae(r_tgt_coa[[i]], r_dot_tgt_coa[[i]], arp_coa[[i], :],
                                   varp_coa[[i], :], scp, aoielevs.max(), delta_hae_max, nlim)
        cos_CAa = np.dot(Aa-ctr, uRRX)/Rrrc
        # Step 4 - Project R/Rdot contour to constant height HAEmin
        Ab = projection_set_to_hae(r_tgt_coa[[i]], r_dot_tgt_coa[[i]], arp_coa[[i], :],
                                   varp_coa[[i], :], scp, aoielevs.min(), delta_hae_max, nlim)
        cos_CAb = np.dot(Ab-ctr, uRRX)/Rrrc
        sin_CAb = look*np.sqrt(1-cos_CAb*cos_CAb)
        # Step 5 - Compute the step size for points along R/Rdot contour
        del_cos_rrc = del_DISTrrc*(1/Rrrc)*np.abs(sin_CAb)
        del_cos_dem = del_DISTrrc*(1/Rrrc)*(np.abs(sin_CAb)/cos_CAb)
        del_cos_CA = -1*np.minimum(del_cos_rrc, del_cos_dem)
        # Step 6 - Compute Number of Points Along R/RDot contour
        npts = np.floor(((cos_CAa - cos_CAb)/del_cos_CA))+2
        # Step 7 - Compute the set of NPTS along R/RDot contour
        cos_CAn = cos_CAb + np.arange(npts)*del_cos_CA
        sin_CAn = look*np.sqrt(1-cos_CAn*cos_CAn)
        P = ctr + Rrrc * (np.outer(cos_CAn, uRRX) + np.outer(sin_CAn, uRRY))
        # Step 8 & 9 - For Each Point convert from ECF to DEM coordinates and compute Delta Height
        # elevate handles:
        llh = gc.ecf_to_geodetic(P)
        egm_elevfinlist = np.empty(llh.shape[0])
        for indx in range(llh.shape[0]):
            egm_elevfinlist[indx] = eval_egm.get(llh[indx, 0], llh[indx, 1], cubic=True)
        hgts = evaldem.elevate(coord=llh[:, 0:2])
        del_h = llh[:, 2] - (hgts + egm_elevfinlist)
        # Step 10 - Solve for the points that are on the DEM in increasing height
        #     (we may just take one for simplicity)
        #     *Currently only finds first point (lowest WGS-84 HAE)
        #     *Finding all solutions would require inspecting all zero crossings
        close_enough_vec = np.nonzero(np.abs(del_h) < del_HDlim)[0]
        if(len(close_enough_vec) > 0):
            gpp[i, :] = P[close_enough_vec[0], :]
        else:
            zero_cross = np.where(del_h[:-1] * del_h[1:] < 1)[0]
            if(zero_cross.size == 0):
                gpp[i, :] = np.nan
            else:
                zero_cross = zero_cross[0]
                frac = del_h[zero_cross] / (del_h[zero_cross] - del_h[zero_cross+1])
                cos_CA_S = cos_CAb+(zero_cross+frac)*del_cos_CA
                sin_CA_S = look*np.sqrt(1-cos_CA_S*cos_CA_S)
                gpp[i, :] = ctr + Rrrc*(cos_CA_S*uRRX+sin_CA_S*uRRY)
    return gpp


def coa_projection_set(sicd_meta, grow, gcol=None):
    '''Computes the set of fundamental parameters for projecting a pixel down to the ground

     [R_TGT_COA, Rdot_TGT_COA, ARP_COA, VARP_COA, t_coa] =
        coa_projection_set(sicd_meta, ip)
     [R_TGT_COA, Rdot_TGT_COA, ARP_COA, VARP_COA, t_coa] =
        coa_projection_set(sicd_meta, grow, gcol)

     Inputs:
        sicd_meta    - SICD meta data structure (see NGA SAR Toolbox read_sicd_meta)
        ip          - [Nx2] (row, column) coordinates of N points in image (or
                      subimage if FirstRow/FirstCol are nonzero).  Zero-based,
                      following SICD (and Python) convention; that is, upper-left
                      pixel is [0, 0].

     Outputs:
        R_TGT_COA    - range to the ARP at COA
        Rdot_TGT_COA - range rate relative to the ARP at COA
        ARP_COA      - aperture reference position at t_coa
        VARP_COA     - velocity at t_coa
        t_coa         - center of aperture time since CDP start for input ip

     Contributors: Thomas McDowall, Harris Corporation
                   Rocco Corsetti, NGA/IB
                   Lisa Talbot, NGA/IB
                   Wade Schwartzkopf, NGA/R
                   Clayton Williams, NRL
                   Daniel Haverporth, NGA/R
    '''

    # Global row and column can be passed as a single [Nx2] array or componentwise
    grow = np.atleast_2d(grow)
    if gcol is not None:
        # Componentwise inputs, separate arguments for X,Y,Z
        gcol = np.atleast_2d(gcol)
    elif grow.ndim == 2 and grow.shape[1] == 2:  # Array of 2-element vectors
        gcol = grow[:, 1]
        grow = grow[:, 0]
    else:
        raise ValueError()  # Must be right type if np.array(x) worked above

    # Making this 2D allow us to broadcast across arrays later
    SCP = np.atleast_2d([sicd_meta.GeoData.SCP.ECF.X,
                         sicd_meta.GeoData.SCP.ECF.Y,
                         sicd_meta.GeoData.SCP.ECF.Z])

    # Convert from global index to SCP centered
    irow = sicd_meta.ImageData.FirstRow + grow - sicd_meta.ImageData.SCPPixel.Row
    icol = sicd_meta.ImageData.FirstCol + gcol - sicd_meta.ImageData.SCPPixel.Col
    # Convert from pixels-from-SCP to meters-from-SCP
    xrow = irow * sicd_meta.Grid.Row.SS
    ycol = icol * sicd_meta.Grid.Col.SS

    # 2.5 Image Grid to COA Parameters
    # Compute target pixel time
    t_coa = poly.polyval2d(xrow, ycol, sicd_meta.Grid.TimeCOAPoly)
    # Calculate aperture reference position and velocity at t_coa
    cARPx = sicd_meta.Position.ARPPoly.X
    cARPy = sicd_meta.Position.ARPPoly.Y
    cARPz = sicd_meta.Position.ARPPoly.Z
    arp_coa = np.atleast_2d([poly.polyval(t_coa, cARPx),
                             poly.polyval(t_coa, cARPy),
                             poly.polyval(t_coa, cARPz)]).transpose()
    varp_coa = np.atleast_2d([poly.polyval(t_coa, poly.polyder(cARPx)),
                              poly.polyval(t_coa, poly.polyder(cARPy)),
                              poly.polyval(t_coa, poly.polyder(cARPz))]).transpose()

    # 4 Image Grid to R/Rdot
    if sicd_meta.Grid.Type == 'RGAZIM':
        # SICD spec changes variable names
        rg = xrow
        az = ycol

        # Compute range and range rate to the SCP at target pixel COA time
        arpMinusSCP = arp_coa - SCP
        rSCPTgtCoa = np.sqrt(np.sum(np.power(arpMinusSCP, 2), axis=-1))

        rDotSCPTgtCoa = (1/rSCPTgtCoa) * np.sum(varp_coa * arpMinusSCP, axis=-1)

        # 4.1 Image Grid to R/Rdot for RGAZIM and PFA
        # This computation is dependent on Grid type and image formation algorithm.
        if sicd_meta.ImageFormation.ImageFormAlgo == 'PFA':
            # PFA specific SICD metadata
            cPA = sicd_meta.PFA.PolarAngPoly
            cKSF = sicd_meta.PFA.SpatialFreqSFPoly

            # Compute polar angle theta and its derivative wrt time at the target pixel COA
            thetaTgtCoa = poly.polyval(t_coa, cPA)  # Time of closest approach
            dThetaDtTgtCoa = poly.polyval(t_coa, poly.polyder(cPA))
            # Compute polar aperture scale factor (KSF) and derivative wrt polar angle
            ksfTgtCoa = poly.polyval(thetaTgtCoa, cKSF)  # Time of closest approach
            dKsfDThetaTgtCoa = poly.polyval(thetaTgtCoa, poly.polyder(cKSF))

            # Compute spatial frequency domain phase slopes in Ka and Kc directions
            # Note: sign for the phase may be ignored as it is cancelled in a
            # subsequent computation.
            dPhiDKaTgtCoa = rg * np.cos(thetaTgtCoa) + az * np.sin(thetaTgtCoa)
            dPhiDKcTgtCoa = -rg * np.sin(thetaTgtCoa) + az * np.cos(thetaTgtCoa)
            # Compute range relative to SCP
            deltaRTgtCoa = ksfTgtCoa * dPhiDKaTgtCoa
            # Compute derivative of range relative to SCP wrt polar angle.
            # Scale by derivative of polar angle wrt time.
            dDeltaRDThetaTgtCoa = dKsfDThetaTgtCoa * dPhiDKaTgtCoa + ksfTgtCoa * dPhiDKcTgtCoa
            deltaRDotTgtCoa = dDeltaRDThetaTgtCoa * dThetaDtTgtCoa
        # 4.2 Image Grid to R/Rdot for RGAZIM and RGAZCOMP
        elif sicd_meta.ImageFormation.ImageFormAlgo == 'RGAZCOMP':
            AzSF = sicd_meta.RgAzComp.AzSF  # RGAZCOMP-specific SICD metadata
            deltaRTgtCoa = rg
            deltaRDotTgtCoa = - np.sqrt(np.sum(np.power(varp_coa, 2), axis=-1)) * AzSF * az
        else:
            raise ValueError('Unrecognized Grid/IFP type.')

        # Compute the range and range rate relative to the ARP at COA. The
        # projection to 3D scene point for Grid location (rg, az) is along this
        # R/Rdot contour.
        r_tgt_coa = rSCPTgtCoa + deltaRTgtCoa
        r_dot_tgt_coa = rDotSCPTgtCoa + deltaRDotTgtCoa
    # 4.3 Image Grid to R/Rdot for RGZERO
    elif sicd_meta.Grid.Type == 'RGZERO':
        # SICD spec changes variable names
        rg = xrow
        az = ycol

        # Compute range at closest approach and time of closest approach
        R_CA_TGT = sicd_meta.RMA.INCA.R_CA_SCP + rg  # Range at closest approach
        t_CA_TGT = poly.polyval(az, sicd_meta.RMA.INCA.TimeCAPoly)  # Time of closest approach
        # Compute ARP velocity magnitude at t_CA_TGT
        VARP_CA_TGT = np.atleast_2d([poly.polyval(t_CA_TGT, poly.polyder(cARPx)),
                                     poly.polyval(t_CA_TGT, poly.polyder(cARPy)),
                                     poly.polyval(t_CA_TGT, poly.polyder(cARPz))]).transpose()
        VM_CA_TGT = np.sqrt(np.sum(np.power(VARP_CA_TGT, 2), 1))
        # Compute the Doppler Rate Scale Factor for image Grid location
        DRSF_TGT = poly.polyval2d(rg, az, sicd_meta.RMA.INCA.DRateSFPoly)
        dt_COA_TGT = t_coa - t_CA_TGT  # Difference between COA time and CA time
        # Now compute range/range rate
        r_tgt_coa = np.sqrt(np.power(R_CA_TGT, 2) +
                            np.multiply(np.multiply(DRSF_TGT, np.power(VM_CA_TGT, 2)),
                                        np.power(dt_COA_TGT, 2)))
        r_dot_tgt_coa = (DRSF_TGT/r_tgt_coa) * np.power(VM_CA_TGT, 2)*dt_COA_TGT
    # 4.4-4.6 Image Grid to R/Rdot for uniformly spaced Grids
    elif sicd_meta.Grid.Type in ['XRGYCR', 'XCTYAT', 'PLANE']:
        # These Grids are all uniformly spaced locations in the image plane.
        # They can all be treated the same.
        # SICD metadata
        uRow = np.array([sicd_meta.Grid.Row.UVectECF.X,
                         sicd_meta.Grid.Row.UVectECF.Y,
                         sicd_meta.Grid.Row.UVectECF.Z])
        uCol = np.array([sicd_meta.Grid.Col.UVectECF.X,
                         sicd_meta.Grid.Col.UVectECF.Y,
                         sicd_meta.Grid.Col.UVectECF.Z])
        # Image plane point
        IPP = SCP + (xrow[:, np.newaxis]*uRow) + (ycol[:, np.newaxis]*uCol)
        # Compute R/Rdot
        ARPminusIPP = arp_coa - IPP
        r_tgt_coa = np.sqrt(np.sum(np.power(ARPminusIPP, 2), axis=-1))
        r_dot_tgt_coa = (1/r_tgt_coa) * np.sum(varp_coa * ARPminusIPP, axis=-1)
    else:
        raise ValueError('Unrecognized Grid/image formation type.')

    return r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, t_coa
