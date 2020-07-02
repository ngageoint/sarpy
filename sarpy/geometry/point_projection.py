"""
Functions to map between the coordinates in image pixel space and geographical coordinates.
"""

import logging
from typing import Tuple
from types import MethodType  # for binding a method dynamically to a class

import numpy

from . import geocoords
from ..io.complex.sicd_elements.blocks import Poly2DType, XYZPolyType
from ..io.DEM.DEM import DTEDList, GeoidHeight, DTEDInterpolator


__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Wade Schwartzkopf")


#############
# Ground-to-Image (aka Scene-to-Image) projection.

# noinspection PyUnusedLocal
def _validate_coords(coords, sicd):
    if not isinstance(coords, numpy.ndarray):
        coords = numpy.array(coords, dtype=numpy.float64)

    orig_shape = coords.shape

    if len(orig_shape) == 1:
        coords = numpy.reshape(coords, (1, -1))
    if coords.shape[-1] != 3:
        raise ValueError(
            'The coords array must represent an array of points in ECF coordinates, '
            'so the final dimension of coords must have length 3. Have coords.shape = {}'.format(coords.shape))

    return coords, orig_shape


def _ground_to_image(coords, coa_proj, uGPN,
                     SCP, SCP_Pixel, uIPN, sf, row_ss, col_ss, uProj,
                     row_col_transform, ipp_transform, delta_gp_max, max_iterations):
    """
    Basic level helper function.

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
    coa_proj : COAProjection
    uGPN : numpy.ndarray
    SCP : numpy.ndarray
    SCP_Pixel : numpy.ndarray
    uIPN : numpy.ndarray
    sf : float
    row_ss : float
    col_ss : float
    uProj : numpy.ndarray
    row_col_transform : numpy.ndarray
    ipp_transform : numpy.ndarray
    delta_gp_max : float
    max_iterations : int

    Returns
    -------
    Tuple[numpy.ndarray, float, int]
        * `image_points` - the determined image point array, of size `N x 2`. Following SICD convention,
           the upper-left pixel is [0, 0].
        * `delta_gpn` - residual ground plane displacement (m).
        * `iterations` - the number of iterations performed.
    """
    g_n = coords.copy()
    im_points = numpy.zeros((coords.shape[0], 2), dtype=numpy.float64)
    delta_gpn = numpy.zeros((coords.shape[0],), dtype=numpy.float64)
    cont = True
    iteration = 0

    matrix_transform = numpy.dot(row_col_transform, ipp_transform)
    # (3 x 2)*(2 x 2) = (3 x 2)

    while cont:
        # project ground plane to image plane iteration
        iteration += 1
        dist_n = numpy.dot(SCP - g_n, uIPN)/sf  # (N, )
        i_n = g_n + numpy.outer(dist_n, uProj)  # (N, 3)
        delta_ipp = i_n - SCP  # (N, 3)
        ip_iter = numpy.dot(delta_ipp, matrix_transform)  # (N, 2)
        im_points[:, 0] = ip_iter[:, 0]/row_ss + SCP_Pixel[0]
        im_points[:, 1] = ip_iter[:, 1]/col_ss + SCP_Pixel[1]
        # transform to ground plane containing the scene points and check how it compares
        p_n = _image_to_ground_plane(im_points, coa_proj, g_n, uGPN)
        # compute displacement between scene point and this new projected point
        diff_n = coords - p_n
        disp_pn = numpy.linalg.norm(diff_n, axis=1)
        # should we continue iterating?
        cont = numpy.any(disp_pn > delta_gp_max) or (iteration <= max_iterations)
        if cont:
            g_n += diff_n

    return im_points, delta_gpn, iteration


def ground_to_image(coords, sicd, delta_gp_max=None, max_iterations=10, block_size=50000,
                    delta_arp=None, delta_varp=None, range_bias=None, adj_params_frame='ECF',
                    use_sicd_coa=True):
    """
    Transforms a 3D ECF point to pixel (row/column) coordinates. This is
    implemented in accordance with the SICD Image Projections Description Document.
    **Really Scene-To-Image projection.**"

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
        ECF coordinate to map to scene coordinates, of size `N x 3`.
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        SICD meta data structure.
    delta_gp_max : float|None
        Ground plane displacement tol (m). Defaults to 0.1*pixel.
    max_iterations : int
        maximum number of iterations to perform
    block_size : int|None
        size of blocks of coordinates to transform at a time
    delta_arp : None|numpy.ndarray|list|tuple
        ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
    delta_varp : None|numpy.ndarray|list|tuple
        VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
    range_bias : float|int
        Range bias adjustable parameter (m), defaults to 0.
    adj_params_frame : str
        One of ['ECF', 'RIC_ECF', 'RIC_ECI'], specifying the coordinate frame used for
        expressing `delta_arp` and `delta_varp` parameters.
    use_sicd_coa : bool
        If sicd.coa_projection is populated, use that one **ignoring the COAProjection parameters.**

    Returns
    -------
    Tuple[numpy.ndarray, float, int]
        * `image_points` - the determined image point array, of size `N x 2`. Following
          the SICD convention, he upper-left pixel is [0, 0].
        * `delta_gpn` - residual ground plane displacement (m).
        * `iterations` - the number of iterations performed.
    """

    coords, orig_shape = _validate_coords(coords, sicd)

    row_ss = sicd.Grid.Row.SS
    col_ss = sicd.Grid.Col.SS
    pixel_size = numpy.sqrt(row_ss*row_ss + col_ss*col_ss)
    if delta_gp_max is None:
        delta_gp_max = 0.1*pixel_size
    delta_gp_max = float(delta_gp_max)
    if delta_gp_max < 0.01*pixel_size:
        delta_gp_max = 0.01*pixel_size
        logging.warning('delta_gp_max was less than 0.01*pixel_size, '
                        'and has been reset to {}'.format(delta_gp_max))

    if use_sicd_coa and sicd.coa_projection is not None:
        coa_proj = sicd.coa_projection
    else:
        coa_proj = COAProjection(sicd, delta_arp, delta_varp, range_bias, adj_params_frame)

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
    uGPN = sicd.PFA.FPN.get_array() if sicd.ImageFormation.ImageFormAlgo == 'PFA' \
        else geocoords.wgs_84_norm(SCP)
    ARP_SCP_COA = sicd.SCPCOA.ARPPos.get_array()
    VARP_SCP_COA = sicd.SCPCOA.ARPVel.get_array()
    uSPN = sicd.SCPCOA.look*numpy.cross(VARP_SCP_COA, SCP-ARP_SCP_COA)
    uSPN /= numpy.linalg.norm(uSPN)
    # uSPN - defined in section 3.1 as normal to instantaneous slant plane that contains SCP at SCP COA is
    # tangent to R/Rdot contour at SCP. Points away from center of Earth. Use look to establish sign.
    sf = float(numpy.dot(uSPN, uIPN))  # scale factor

    # prepare the work space
    coords_view = numpy.reshape(coords, (-1, 3))  # possibly or make 2-d flatten
    num_points = coords_view.shape[0]
    if block_size is None or num_points <= block_size:
        image_points, delta_gpn, iters = _ground_to_image(
            coords_view, coa_proj, uGPN,
            SCP, SCP_Pixel, uIPN, sf, row_ss, col_ss, uSPN,
            row_col_transform, ipp_transform, delta_gp_max, max_iterations)
        iters = numpy.full((num_points, ), iters)
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
                    coords_view[start_block:end_block, :], coa_proj, uGPN,
                    SCP, SCP_Pixel, uIPN, sf, row_ss, col_ss, uSPN,
                    row_col_transform, ipp_transform, delta_gp_max, max_iterations)
            start_block = end_block

    if len(orig_shape) == 1:
        image_points = numpy.reshape(image_points, (-1,))
    elif len(orig_shape) > 1:
        image_points = numpy.reshape(image_points, orig_shape[:-1]+(2, ))
        delta_gpn = numpy.reshape(delta_gpn, orig_shape[:-1])
        iters = numpy.reshape(iters, orig_shape[:-1])
    return image_points, delta_gpn, iters


def ground_to_image_geo(coords, sicd, ordering='latlong', **kwargs):
    """
    Transforms a 3D Lat/Lon/HAE point to pixel (row/column) coordinates.
    This is implemented in accordance with the SICD Image Projections Description Document.

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
        Lat/Lon/HAE coordinate to map to scene coordinates, of size `N x 3`.
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        SICD meta data structure.
    ordering : str
        If 'longlat', then the input is `[longitude, latitude, hae]`.
        Otherwise, the input is `[latitude, longitude, hae]`. Passed through
        to :func:`sarpy.geometry.geocoords.geodetic_to_ecf`.
    kwargs
        See the key word arguments of :func:`ground_to_image`

    Returns
    -------
    Tuple[numpy.ndarray, float, int]
        * `image_points` - the determined image point array, of size `N x 2`. Following SICD convention,
           the upper-left pixel is [0, 0].
        * `delta_gpn` - residual ground plane displacement (m).
        * `iterations` - the number of iterations performed.
    """

    return ground_to_image(geocoords.geodetic_to_ecf(coords, ordering=ordering), sicd, **kwargs)


############
# Image-To-Ground projections

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


class COAProjection(object):
    """
    The COA projection object - provide common projection functionality for all Image-to-R/Rdot projection.
    """

    def __init__(self, sicd, delta_arp=None, delta_varp=None, range_bias=None, adj_params_frame='ECF'):
        """

        Parameters
        ----------
        sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
            The SICD metadata structure.
        delta_arp : None|numpy.ndarray|list|tuple
            ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
        delta_varp : None|numpy.ndarray|list|tuple
            VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
        range_bias : float|int
            Range bias adjustable parameter (m), defaults to 0.
        adj_params_frame : str
            One of ['ECF', 'RIC_ECF', 'RIC_ECI'], specifying the coordinate frame used for
            expressing `delta_arp` and `delta_varp` parameters.
        """

        if not sicd.can_project_coordinates():
            raise ValueError('Insufficient metadata populated to formulate projection.')

        time_coa_poly = sicd.Grid.TimeCOAPoly
        # fall back to approximation if TimeCOAPoly is not populated
        if time_coa_poly is None:
            time_coa_poly = Poly2DType(Coefs=[[sicd.Timeline.CollectDuration/2, ], ])
            logging.warning(
                'Using (constant) approximation to TimeCOAPoly, which may result in poor projection results.')
        self.time_coa_poly = time_coa_poly  # type: Poly2DType

        self.arp_poly = sicd.Position.ARPPoly  # type: XYZPolyType
        self.varp_poly = self.arp_poly.derivative(der_order=1, return_poly=True)  # type: XYZPolyType

        self.row_ss = sicd.Grid.Row.SS  # type: float
        self.col_ss = sicd.Grid.Col.SS  # type: float
        self.first_row = sicd.ImageData.FirstRow  # type: int
        self.first_col = sicd.ImageData.FirstCol  # type: int
        self.scp_row = sicd.ImageData.SCPPixel.Row  # type: int
        self.scp_col = sicd.ImageData.SCPPixel.Col  # type: int

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

        if adj_params_frame in ['RIC_ECI', 'RIC_ECF']:
            if sicd.SCPCOA.ARPPos is None or sicd.SCPCOA.ARPVel is None:
                raise ValueError(
                    'The adj_params_frame is of RIC type, but one of SCPCOA.ARPPos or '
                    'SCPCOA.ARPVel is not populated.')
            ARP_SCP_COA = sicd.SCPCOA.ARPPos.get_array()
            VARP_SCP_COA = sicd.SCPCOA.ARPVel.get_array()
            ric_matrix = _ric_ecf_mat(ARP_SCP_COA, VARP_SCP_COA, adj_params_frame)
            delta_arp = ric_matrix.dot(delta_arp)
            delta_varp = ric_matrix.dot(delta_varp)
        self.delta_arp = delta_arp  # type: numpy.ndarray
        self.delta_varp = delta_varp  # type: numpy.ndarray

        if range_bias is None:
            range_bias = 0.0
        else:
            range_bias = float(range_bias)
        self.range_bias = range_bias  # type: float
        # bind the method specific intermediate projection method
        self._method_proj = MethodType(_get_type_specific_projection(sicd), self)

    def _init_proj(self, im_points):
        """

        Parameters
        ----------
        im_points : numpy.ndarray

        Returns
        -------
        Tuple[numpy.ndarray,...]
        """

        row_meters = (im_points[:, 0] + self.first_row - self.scp_row)*self.row_ss
        col_meters = (im_points[:, 1] + self.first_col - self.scp_col)*self.col_ss
        t_coa = self.time_coa_poly(row_meters, col_meters)
        # calculate aperture reference position and velocity at target time
        arp_coa = self.arp_poly(t_coa)
        varp_coa = self.varp_poly(t_coa)
        return row_meters, col_meters, t_coa, arp_coa, varp_coa

    def projection(self, im_points):
        """
        Perform the projection from image coordinates to R/Rdot coordinates.

        Parameters
        ----------
        im_points : numpy.ndarray
            This array of image point coordinates, **expected to have shape (N, 2)**.

        Returns
        -------
        Tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray,numpy.ndarray,numpy.ndarray]
            * `r_tgt_coa` - range to the ARP at COA
            * `r_dot_tgt_coa` - range rate relative to the ARP at COA
            * `t_coa` - center of aperture time since CDP start for input ip
            * `arp_coa` - aperture reference position at t_coa
            * `varp_coa` - velocity at t_coa
        """

        row_meters, col_meters, t_coa, arp_coa, varp_coa = self._init_proj(im_points)
        r_tgt_coa, r_dot_tgt_coa = self._method_proj(row_meters, col_meters, t_coa, arp_coa, varp_coa)
        # adjust parameters
        arp_coa += self.delta_arp
        varp_coa += self.delta_varp
        r_tgt_coa += self.range_bias
        return r_tgt_coa, r_dot_tgt_coa, t_coa, arp_coa, varp_coa


def _get_type_specific_projection(sicd):
    """
    Gets an intermediate method specific projection method with six required
    calling arguments (self, row_meters, col_meters, t_coa, arp_coa, varp_coa).

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    callable
    """
    # triple-nested function - it was conceptually clearest...

    def pfa_projection():
        SCP = sicd.GeoData.SCP.ECF.get_array()
        pfa = sicd.PFA
        polar_ang_poly = pfa.PolarAngPoly
        spatial_freq_sf_poly = pfa.SpatialFreqSFPoly
        polar_ang_poly_der = polar_ang_poly.derivative(der_order=1, return_poly=True)
        spatial_freq_sf_poly_der = spatial_freq_sf_poly.derivative(der_order=1, return_poly=True)

        polar_ang_poly_der = polar_ang_poly.derivative(der_order=1, return_poly=True)
        spatial_freq_sf_poly_der = spatial_freq_sf_poly.derivative(der_order=1, return_poly=True)

        # noinspection PyUnusedLocal, PyIncorrectDocstring
        def method_projection(instance, row_meters, col_meters, t_coa, arp_coa, varp_coa):
            """
            PFA specific intermediate projection.

            Parameters
            ----------
            row_meters : numpy.ndarray
            col_meters : numpy.ndarray
            t_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            ARP_minus_SCP = arp_coa - SCP
            rSCPTgtCoa = numpy.linalg.norm(ARP_minus_SCP, axis=-1)
            rDotSCPTgtCoa = numpy.sum(varp_coa * ARP_minus_SCP, axis=-1) / rSCPTgtCoa

            thetaTgtCoa = polar_ang_poly(t_coa)
            dThetaDtTgtCoa = polar_ang_poly_der(t_coa)
            # Compute polar aperture scale factor (KSF) and derivative wrt polar angle
            ksfTgtCoa = spatial_freq_sf_poly(thetaTgtCoa)
            dKsfDThetaTgtCoa = spatial_freq_sf_poly_der(thetaTgtCoa)
            # Compute spatial frequency domain phase slopes in Ka and Kc directions
            # NB: sign for the phase may be ignored as it is cancelled in a subsequent computation.
            dPhiDKaTgtCoa = row_meters * numpy.cos(thetaTgtCoa) + col_meters * numpy.sin(thetaTgtCoa)
            dPhiDKcTgtCoa = -row_meters * numpy.sin(thetaTgtCoa) + col_meters * numpy.cos(thetaTgtCoa)
            # Compute range relative to SCP
            deltaRTgtCoa = ksfTgtCoa * dPhiDKaTgtCoa
            # Compute derivative of range relative to SCP wrt polar angle.
            # Scale by derivative of polar angle wrt time.
            dDeltaRDThetaTgtCoa = dKsfDThetaTgtCoa * dPhiDKaTgtCoa + ksfTgtCoa * dPhiDKcTgtCoa
            deltaRDotTgtCoa = dDeltaRDThetaTgtCoa * dThetaDtTgtCoa
            return rSCPTgtCoa + deltaRTgtCoa, rDotSCPTgtCoa + deltaRDotTgtCoa

        return method_projection

    def rgazcomp_projection():
        SCP = sicd.GeoData.SCP.ECF.get_array()
        az_sf = sicd.RgAzComp.AzSF

        # noinspection PyUnusedLocal, PyIncorrectDocstring
        def method_projection(instance, row_meters, col_meters, t_coa, arp_coa, varp_coa):
            """
            RgAzComp specific intermediate projection.

            Parameters
            ----------
            row_meters : numpy.ndarray
            col_meters : numpy.ndarray
            t_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            ARP_minus_SCP = arp_coa - SCP
            rSCPTgtCoa = numpy.linalg.norm(ARP_minus_SCP, axis=-1)
            rDotSCPTgtCoa = numpy.sum(varp_coa*ARP_minus_SCP, axis=-1)/rSCPTgtCoa
            deltaRTgtCoa = row_meters
            deltaRDotTgtCoa = -numpy.linalg.norm(varp_coa, axis=-1)*az_sf*col_meters
            return rSCPTgtCoa + deltaRTgtCoa, rDotSCPTgtCoa + deltaRDotTgtCoa

        return method_projection

    def inca_projection():
        inca = sicd.RMA.INCA
        r_ca_scp = inca.R_CA_SCP
        time_ca_poly = inca.TimeCAPoly
        drate_sf_poly = inca.DRateSFPoly

        # noinspection PyUnusedLocal, PyIncorrectDocstring
        def method_projection(instance, row_meters, col_meters, t_coa, arp_coa, varp_coa):
            """
            INCA specific intermediate projection.

            Parameters
            ----------
            row_meters : numpy.ndarray
            col_meters : numpy.ndarray
            t_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            # compute range/time of closest approach
            R_CA_TGT = r_ca_scp + row_meters  # Range at closest approach
            t_CA_TGT = time_ca_poly(col_meters)  # Time of closest approach
            # Compute ARP velocity magnitude (actually squared, since that's how it's used) at t_CA_TGT
            VEL2_CA_TGT = numpy.sum(instance.varp_poly(t_CA_TGT)**2, axis=-1)
            # Compute the Doppler Rate Scale Factor for image Grid location
            DRSF_TGT = drate_sf_poly(row_meters, col_meters)
            # Difference between COA time and CA time
            dt_COA_TGT = t_coa - t_CA_TGT
            r_tgt_coa = numpy.sqrt(R_CA_TGT*R_CA_TGT + DRSF_TGT*VEL2_CA_TGT*dt_COA_TGT*dt_COA_TGT)
            r_dot_tgt_coa = (DRSF_TGT/r_tgt_coa)*VEL2_CA_TGT*dt_COA_TGT
            return r_tgt_coa, r_dot_tgt_coa

        return method_projection

    def plane_projection():
        SCP = sicd.GeoData.SCP.ECF.get_array()
        uRow = sicd.Grid.Row.UVectECF.get_array()
        uCol = sicd.Grid.Row.UVectECF.get_array()

        # noinspection PyUnusedLocal, PyIncorrectDocstring
        def method_projection(instance, row_meters, col_meters, t_coa, arp_coa, varp_coa):
            """
            Plane specific intermediate projection.

            Parameters
            ----------
            row_meters : numpy.ndarray
            col_meters : numpy.ndarray
            t_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            ARP_minus_IPP = arp_coa - (SCP + numpy.outer(row_meters, uRow) + numpy.outer(col_meters, uCol))
            r_tgt_coa = numpy.linalg.norm(ARP_minus_IPP, axis=-1)
            r_dot_tgt_coa = numpy.sum(varp_coa * ARP_minus_IPP, axis=-1)/r_tgt_coa
            return r_tgt_coa, r_dot_tgt_coa
        return method_projection

    # NB: sicd.can_project_coordinates() has been called, so all required attributes
    #   must be populated
    if sicd.Grid.Type == 'RGAZIM':
        if sicd.ImageFormation.ImageFormAlgo == 'PFA':
            return pfa_projection()
        elif sicd.ImageFormation.ImageFormAlgo == 'RGAZCOMP':
            return rgazcomp_projection()
    elif sicd.Grid.Type == 'RGZERO':
        return inca_projection()
    elif sicd.Grid.Type in ['XRGYCR', 'XCTYAT', 'PLANE']:
        return plane_projection()
    else:
        # NB: this will have been noted by sicd.can_project_coordinates(), but is
        #   here for completeness
        raise ValueError('Unhandled Grid.Type'.format(sicd.Grid.Type))


def _validate_im_points(im_points, sicd):
    """

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    numpy.ndarray
    """

    if im_points is None:
        raise ValueError('The argument cannot be None')

    if not isinstance(im_points, numpy.ndarray):
        im_points = numpy.array(im_points, dtype=numpy.float64)

    orig_shape = im_points.shape

    if len(im_points.shape) == 1:
        im_points = numpy.reshape(im_points, (1, -1))
    if im_points.shape[-1] != 2:
        raise ValueError(
            'The im_points array must represent an array of points in pixel coordinates, '
            'so the final dimension of im_points must have length 2. Have im_points.shape = {}'.format(im_points.shape))

    # check to ensure that the entries of im_points are not ridiculous
    rows = sicd.ImageData.NumRows
    cols = sicd.ImageData.NumCols
    row_bounds = (-rows/2, 3*rows/2)
    col_bounds = (-cols/2, 3*cols/2)
    if numpy.any(
            (im_points[:, 0] < row_bounds[0]) | (im_points[:, 0] > row_bounds[1]) |
            (im_points[:, 1] < col_bounds[0]) | (im_points[:, 1] > col_bounds[1])):
        raise ValueError(
            'The sicd has {} rows and {} cols. image_to_ground projection effort '
            'requires row coordinates in the range {} and column coordinates '
            'in the range {}'.format(rows, cols, row_bounds, col_bounds))
    return im_points, orig_shape


def image_to_ground(im_points, sicd, block_size=50000, projection_type='HAE', use_sicd_coa=True, **kwargs):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        (row, column) coordinates of N points in image (or subimage if FirstRow/FirstCol are nonzero).
        Following SICD convention, the upper-left pixel is [0, 0].
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        SICD meta data structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    projection_type : str
        One of ['PLANE', 'HAE', 'DEM'].
    use_sicd_coa : bool
        If sicd.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    kwargs
        keyword arguments relevant for the given projection type. See image_to_ground_plane/hae/dem methods.

    Returns
    -------
    numpy.ndarray
        Physical coordinates (in ECF) corresponding input image coordinates. The interpretation
        or meaning of the physical coordinates depends on `projection_type` chosen.
    """

    p_type = projection_type.upper()
    if p_type == 'PLANE':
        return image_to_ground_plane(im_points, sicd, block_size=block_size, use_sicd_coa=use_sicd_coa, **kwargs)
    elif p_type == 'HAE':
        return image_to_ground_hae(im_points, sicd, block_size=block_size, use_sicd_coa=use_sicd_coa, **kwargs)
    elif p_type == 'DEM':
        return image_to_ground_dem(im_points, sicd, block_size=block_size, use_sicd_coa=use_sicd_coa, **kwargs)
    else:
        raise ValueError('Got unrecognized projection type {}'.format(projection_type))


def image_to_ground_geo(im_points, sicd, ordering='latlong', block_size=50000, projection_type='HAE', use_sicd_coa=True, **kwargs):
    """
    Transforms image coordinates to ground plane Lat/Lon/HAE coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        (row, column) coordinates of N points in image (or subimage if FirstRow/FirstCol are nonzero).
        Following SICD convention, the upper-left pixel is [0, 0].
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        SICD meta data structure.
    ordering : str
        Determines whether return is ordered as `[lat, long, hae]` or `[long, lat, hae]`.
        Passed through to :func:`sarpy.geometry.geocoords.ecf_to_geodetic`.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    projection_type : str
        One of ['PLANE', 'HAE', 'DEM'].
    use_sicd_coa : bool
        If sicd.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    kwargs
        See the keyword arguments in :func:`image_to_ground`.

    Returns
    -------
    numpy.ndarray
        Ground Plane Point (in Lat/Lon/HAE coordinates) along the R/Rdot contour.
    """

    return geocoords.ecf_to_geodetic(image_to_ground(
        im_points, sicd, block_size=block_size, projection_type=projection_type, use_sicd_coa=use_sicd_coa, **kwargs),
        ordering=ordering)


#####
# Image-to-Ground Plane

def _image_to_ground_plane_perform(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, uZ):
    """

    Parameters
    ----------
    r_tgt_coa : numnpy.ndarray
    r_dot_tgt_coa : numnpy.ndarray
    arp_coa : numnpy.ndarray
    varp_coa : numnpy.ndarray
    gref : numnpy.ndarray
    uZ : numnpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

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
    return aGPN + (uX.T*gd*cosAz + uY.T*gd*sinAz).T


def _image_to_ground_plane(im_points, coa_projection, gref, uZ):
    """

    Parameters
    ----------
    im_points : numpy.ndarray
    coa_projection : COAProjection
    gref : numpy.ndarray
    uZ : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    r_tgt_coa, r_dot_tgt_coa, t_coa, arp_coa, varp_coa = coa_projection.projection(im_points)
    return _image_to_ground_plane_perform(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, uZ)


def image_to_ground_plane(im_points, sicd, block_size=50000, gref=None, ugpn=None, use_sicd_coa=True, **coa_args):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        the SICD metadata structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    gref : None|numpy.ndarray|list|tuple
        Ground plane reference point ECF coordinates (m). The default is the SCP
    ugpn : None|numpy.ndarray|list|tuple
        Vector normal to the plane to which we are projecting.
    use_sicd_coa : bool
        If sicd.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    coa_args
        keyword arguments for COAProjection constructor.

    Returns
    -------
    numpy.ndarray
        Ground Plane Point (in ECF coordinates) corresponding to the input image coordinates.
    """

    # method parameter validation
    if gref is None:
        gref = sicd.GeoData.SCP.ECF.get_array()
    if ugpn is None:
        ugpn = sicd.PFA.FPN.get_array() if sicd.ImageFormation.ImageFormAlgo == 'PFA' \
            else geocoords.wgs_84_norm(gref)
    if len(ugpn.shape) == 2:
        ugpn = numpy.reshape(ugpn, (3, ))
    uZ = ugpn/numpy.linalg.norm(ugpn)

    # coa projection creation
    im_points, orig_shape = _validate_im_points(im_points, sicd)
    if use_sicd_coa and sicd.coa_projection is not None:
        coa_proj = sicd.coa_projection
    else:
        coa_proj = COAProjection(sicd, **coa_args)

    # prepare workspace
    im_points_view = numpy.reshape(im_points, (-1, 2))  # possibly or make 2-d flatten
    num_points = im_points_view.shape[0]
    if block_size is None or num_points <= block_size:
        coords = _image_to_ground_plane(im_points_view, coa_proj, gref, uZ)
    else:
        coords = numpy.zeros((num_points, 3), dtype=numpy.float64)
        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block + block_size, num_points)
            coords[start_block:end_block, :] = _image_to_ground_plane(
                im_points_view[start_block:end_block], coa_proj, gref, uZ)
            start_block = end_block

    if len(orig_shape) == 1:
        coords = numpy.reshape(coords, (-1, ))
    elif len(orig_shape) > 1:
        coords = numpy.reshape(coords, orig_shape[:-1] + (3,))
    return coords


#####
# Image-to-HAE

def _image_to_ground_hae_perform(
        r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, SCP, ugpn,
        hae0, delta_hae_max, hae_nlim, scp_hae):
    """
    Intermediate helper method.

    Parameters
    ----------
    r_tgt_coa : numpy.ndarray
    r_dot_tgt_coa : numpy.ndarray
    arp_coa : numpy.ndarray
    varp_coa : numpy.ndarray
    SCP : numpy.ndarray
    ugpn : numpy.ndarray
    hae0 : float
    delta_hae_max : float
    hae_nlim : int
    scp_hae : float

    Returns
    -------
    numpy.ndarray
    """

    # Compute the geodetic ground plane normal at the SCP.
    look = numpy.sign(numpy.sum(numpy.cross(arp_coa, varp_coa)*(SCP-arp_coa), axis=1))
    gref = SCP - (scp_hae - hae0)*ugpn
    # iteration variables
    gpp = None
    delta_hae = None
    cont = True
    iters = 0
    while cont:
        iters += 1
        # Compute the precise projection along the R/Rdot contour to Ground Plane.
        gpp = _image_to_ground_plane_perform(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, ugpn)
        # check our hae value versus hae0
        gpp_llh = geocoords.ecf_to_geodetic(gpp)
        delta_hae = gpp_llh[:, 2] - hae0
        abs_delta_hae = numpy.abs(delta_hae)
        # should we stop our iteration?
        cont = numpy.all(abs_delta_hae > delta_hae_max) and (iters <= hae_nlim)
        if cont:
            delta_hae_min = delta_hae[numpy.argmin(abs_delta_hae)]
            gref -= delta_hae_min*ugpn
    # Compute the unit slant plane normal vector, uspn, that is tangent to the R/Rdot contour at point gpp
    uspn = (numpy.cross(varp_coa, (gpp - arp_coa)).T*look).T
    uspn = (uspn.T/numpy.linalg.norm(uspn, axis=-1)).T
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


def _image_to_ground_hae(im_points, coa_projection, hae0, delta_hae_max, hae_nlim, scp_hae, SCP):
    """
    Intermediate helper function for projection.

    Parameters
    ----------
    im_points : numpy.ndarray
        the image coordinate array
    coa_projection : COAProjection
    hae0 : float
    delta_hae_max : float
    hae_nlim : int
    scp_hae : float
    SCP : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    # get (image formation specific) projection parameters
    r_tgt_coa, r_dot_tgt_coa, t_coa, arp_coa, varp_coa = coa_projection.projection(im_points)
    ugpn = geocoords.wgs_84_norm(SCP)
    return _image_to_ground_hae_perform(
        r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, SCP, ugpn,
        hae0, delta_hae_max, hae_nlim, scp_hae)


def image_to_ground_hae(im_points, sicd, block_size=50000,
                        hae0=None, delta_hae_max=None, hae_nlim=None, use_sicd_coa=True, **coa_args):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        the SICD metadata structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    hae0 : None|float|int
        Surface height (m) above the WGS-84 reference ellipsoid for projection point.
        Defaults to HAE at the SCP.
    delta_hae_max : None|float|int
        Height threshold for convergence of iterative constant HAE computation (m). Defaults to 1.
    hae_nlim : int
        Maximum number of iterations allowed for constant hae computation. Defaults to 5.
    use_sicd_coa : bool
        If sicd.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    coa_args
        keyword arguments for COAProjection constructor.

    Returns
    -------
    numpy.ndarray
        Ground Plane Point (in ECF coordinates) with target hae corresponding to
        the input image coordinates.
    """

    # method parameter validation
    SCP = sicd.GeoData.SCP.ECF.get_array()
    scp_hae = sicd.GeoData.SCP.LLH.HAE
    if hae0 is None:
        hae0 = scp_hae

    if delta_hae_max is None:
        delta_hae_max = 1.0
    delta_hae_max = float(delta_hae_max)
    if delta_hae_max <= 1e-2:
        raise ValueError('delta_hae_max must be at least 1e-2 (1 cm). Got {0:8f}'.format(delta_hae_max))
    if hae_nlim is None:
        hae_nlim = 5
    hae_nlim = int(hae_nlim)
    if hae_nlim <= 0:
        raise ValueError('hae_nlim must be a positive integer. Got {}'.format(hae_nlim))

    # coa projection creation
    im_points, orig_shape = _validate_im_points(im_points, sicd)
    if use_sicd_coa and sicd.coa_projection is not None:
        coa_proj = sicd.coa_projection
    else:
        coa_proj = COAProjection(sicd, **coa_args)

    # prepare workspace
    im_points_view = numpy.reshape(im_points, (-1, 2))  # possibly or make 2-d flatten
    num_points = im_points_view.shape[0]
    if block_size is None or num_points <= block_size:
        coords = _image_to_ground_hae(im_points_view, coa_proj, hae0, delta_hae_max, hae_nlim, scp_hae, SCP)
    else:
        coords = numpy.zeros((num_points, 3), dtype=numpy.float64)
        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block + block_size, num_points)
            coords[start_block:end_block, :] = _image_to_ground_hae(
                im_points_view[start_block:end_block], coa_proj, hae0, delta_hae_max, hae_nlim, scp_hae, SCP)
            start_block = end_block

    if len(orig_shape) == 1:
        coords = numpy.reshape(coords, (-1,))
    elif len(orig_shape) > 1:
        coords = numpy.reshape(coords, orig_shape[:-1] + (3,))
    return coords


#####
# Image-to-DEM

def _image_to_ground_dem(
        im_points, coa_projection, dem_interpolator, min_dem, max_dem, horizontal_step_size, scp_hae, SCP):
    """

    Parameters
    ----------
    im_points : numpy.ndarray
    coa_projection : COAProjection
    dem_interpolator : DTEDInterpolator
    min_dem : float
    max_dem : float
    horizontal_step_size : float|int
    scp_hae: float
    SCP : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    # get (image formation specific) projection parameters
    r_tgt_coa, r_dot_tgt_coa, t_coa, arp_coa, varp_coa = coa_projection.projection(im_points)
    ugpn = geocoords.wgs_84_norm(SCP)
    delta_hae_max = 1
    hae_nlim = 5

    # if max_dem - min_dem is sufficiently small, then just do the simplest thing
    if max_dem - min_dem < 1:
        return _image_to_ground_hae_perform(
            r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, SCP, ugpn, max_dem,
            delta_hae_max, hae_nlim, scp_hae)
    # get projection to hae at high/low points
    coords_high = _image_to_ground_hae_perform(
        r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, SCP, ugpn, max_dem,
        delta_hae_max, hae_nlim, scp_hae)
    coords_low = _image_to_ground_hae_perform(
        r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, SCP, ugpn, min_dem,
        delta_hae_max, hae_nlim, scp_hae)
    ecf_diffs = coords_low - coords_high
    dists = numpy.linalg.norm(ecf_diffs, axis=1)
    # NB: the proper projection point will be the HIGHEST point
    #   on the DEM along the straight line between the high and low point
    sin_ang = (max_dem - min_dem)/numpy.min(dists)
    cos_ang = numpy.sqrt(1 - sin_ang*sin_ang)
    num_pts = numpy.max(dists)/(cos_ang*horizontal_step_size)
    step = numpy.linspace(0., 1., num_pts, dtype=numpy.float64)
    # construct our lat lon space of lines
    llh_high = geocoords.ecf_to_geodetic(coords_high)
    llh_low = geocoords.ecf_to_geodetic(coords_low)
    # I'm drawing these lines in lat/lon space, because this should be incredibly local
    diffs = llh_low - llh_high
    elevations = numpy.linspace(max_dem, min_dem, num_pts, dtype=numpy.float64)
    # construct the space of points connecting high to low of shape (N, 2, num_pts)
    lat_lon_space = llh_low[:, :2] + numpy.multiply.outer(diffs[:, :2], step)  # NB: this is a numpy.ufunc trick
    # determine the ground hae elevation at these points according to the dem interpolator
    # NB: lat_lon_elevations is shape (N, num_pts)
    lat_lon_elevation = dem_interpolator.get_elevation_hae(lat_lon_space[:, 0, :], lat_lon_space[:, 1, :], block_size=50000)
    del lat_lon_space  # we can free this up, since it's potentially large
    bad_values = numpy.isnan(lat_lon_elevation)
    if numpy.any(bad_values):
        lat_lon_elevation[bad_values] = scp_hae
    # adjust by the hae, to find the diff between our line in elevation
    lat_lon_elevation -= elevations
    # these elevations should be guaranteed to start positive because we used to
    #   total bounds for the DEM values
    # we find the "first" (in high to low order) element where the elevation is close enough to negative
    # NB: this is shape (N, )
    indices = numpy.argmax(lat_lon_elevation < 0.5, axis=1)
    # linearly interpolate to find the best guess for 0 crossing.
    prev_indices = indices - 1
    if numpy.any(prev_indices < 0):
        raise ValueError("The first negative entry should have occurred at a strictly positive index")
    d1 = lat_lon_elevation[:, indices]
    d0 = lat_lon_elevation[:, prev_indices]
    frac_indices = indices + (d1/(d0 - d1))
    return coords_high + ((frac_indices/(num_pts - 1))*ecf_diffs.T).T


def image_to_ground_dem(im_points, sicd, block_size=50000,
                        dted_list=None, dem_type='SRTM2F', geoid_file=None,
                        horizontal_step_size=10, use_sicd_coa=True, **coa_args):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
        the SICD metadata structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be transformed as a single block if `None`.
    dted_list : None|str|DTEDList|DTEDInterpolator
    dem_type : str
        One of ['DTED1', 'DTED2', 'SRTM1', 'SRTM2', 'SRTM2F'], specifying the DEM type.
    geoid_file : None|str|GeoidHeight
    horizontal_step_size : None|float|int
        Maximum distance between adjacent points along the R/Rdot contour.
    use_sicd_coa : bool
        If sicd.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    coa_args
        keyword arguments for COAProjection constructor.

    Returns
    -------
    numpy.ndarray
        Physical coordinates (in ECF coordinates) with corresponding to the input image
        coordinates, assuming detected features actually correspond to the DEM.
    """

    # coa projection creation
    im_points, orig_shape = _validate_im_points(im_points, sicd)
    if use_sicd_coa and sicd.coa_projection is not None:
        coa_proj = sicd.coa_projection
    else:
        coa_proj = COAProjection(sicd, **coa_args)

    if isinstance(dted_list, str):
        dted_list = DTEDList(dted_list)

    scp = sicd.GeoData.SCP.LLH.get_array()
    if isinstance(dted_list, DTEDList):
        # find sensible bounds for the DEMs that we need to load up
        t_lats = numpy.array([scp[0]-0.1, scp[0], scp[0] + 0.1], dtype=numpy.float64)
        lon_diff = min(10., abs(10.0/(112*numpy.sin(numpy.rad2deg(scp[0])))))
        t_lons = numpy.arange(scp[1]-lon_diff, scp[1]+lon_diff+1, lon_diff, dtype=numpy.float64)
        t_lats[t_lats > 90] = 90.0
        t_lats[t_lats < -90] = -90.0
        t_lons[t_lons > 180] -= 360
        t_lons[t_lons < -180] += 360
        lats, lons = numpy.meshgrid(t_lats, t_lons)
        dem_interpolator = DTEDInterpolator.from_coords_and_list(lats, lons, dted_list, dem_type, geoid_file=geoid_file)
    elif isinstance(dted_list, DTEDInterpolator):
        dem_interpolator = dted_list
    else:
        raise ValueError(
            'dted_list is expected to be a string suitable for constructing a DTEDList, '
            'an instance of a DTEDList suitable for constructing a DTEDInterpolator, '
            'or DTEDInterpolator instance. Got {}'.format(type(dted_list)))
    # determine max/min hae in the DEM
    # not the ellipsoid
    scp_geoid = dem_interpolator.geoid.get(scp[0], scp[1])
    # remember that min/max in a DTED is relative to the geoid, not hae
    min_dem = dem_interpolator.get_min_dem() + scp_geoid + 10
    max_dem = dem_interpolator.get_max_dem() + scp_geoid - 10

    # prepare workspace
    im_points_view = numpy.reshape(im_points, (-1, 2))  # possibly or make 2-d flatten
    num_points = im_points_view.shape[0]
    if block_size is None or num_points <= block_size:
        coords = _image_to_ground_dem(
            im_points_view, coa_proj, dem_interpolator, min_dem, max_dem, horizontal_step_size,
            scp[2], sicd.GeoData.SCP.ECF.get_array())
    else:
        coords = numpy.zeros((num_points, 3), dtype=numpy.float64)
        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block + block_size, num_points)
            coords[start_block:end_block, :] = _image_to_ground_dem(
                im_points_view[start_block:end_block], coa_proj, dem_interpolator,
                min_dem, max_dem, horizontal_step_size, scp[2], sicd.GeoData.SCP.ECF.get_array())
            start_block = end_block

    if len(orig_shape) == 1:
        coords = numpy.reshape(coords, (-1,))
    elif len(orig_shape) > 1:
        coords = numpy.reshape(coords, orig_shape[:-1] + (3,))
    return coords
