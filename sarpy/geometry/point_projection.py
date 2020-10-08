"""
Functions to map between the coordinates in image pixel space and geographical
coordinates.
"""

import logging
from typing import Tuple
from types import MethodType  # for binding a method dynamically to a class

import numpy

from sarpy.compliance import string_types, int_func
from sarpy.geometry.geocoords import ecf_to_geodetic, geodetic_to_ecf, wgs_84_norm
from sarpy.io.complex.sicd_elements.blocks import Poly2DType, XYZPolyType
from sarpy.io.DEM.DEM import DEMInterpolator
from sarpy.io.DEM.DTED import DTEDList, DTEDInterpolator

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Wade Schwartzkopf")


#############
# COA Projection definition

def _validate_adj_param(value, name):
    """
    Validate the aperture adjustment vector parameters.

    Parameters
    ----------
    value : None|numpy.ndarray|list|tuple
    name : str

    Returns
    -------
    numpy.ndarray
    """

    if value is None:
        value = numpy.array([0, 0, 0], dtype='float64')
    if not isinstance(value, numpy.ndarray):
        value = numpy.array(value, dtype='float64')
    if value.shape != (3,):
        raise ValueError('{} must have shape (3, ). Got {}'.format(name, value.shape))
    return value


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
    return numpy.array([r, i, c], dtype='float64')


def _get_sicd_type_specific_projection(sicd):
    """
    Gets an intermediate method specific projection method with six required
    calling arguments (self, row_transform, col_transform, time_coa, arp_coa, varp_coa).

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    callable
    """

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
        def method_projection(instance, row_transform, col_transform, time_coa, arp_coa, varp_coa):
            """
            PFA specific intermediate projection.

            Parameters
            ----------
            row_transform : numpy.ndarray
            col_transform : numpy.ndarray
            time_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            ARP_minus_SCP = arp_coa - SCP
            rSCPTgtCoa = numpy.linalg.norm(ARP_minus_SCP, axis=-1)
            rDotSCPTgtCoa = numpy.sum(varp_coa * ARP_minus_SCP, axis=-1) / rSCPTgtCoa

            thetaTgtCoa = polar_ang_poly(time_coa)
            dThetaDtTgtCoa = polar_ang_poly_der(time_coa)
            # Compute polar aperture scale factor (KSF) and derivative wrt polar angle
            ksfTgtCoa = spatial_freq_sf_poly(thetaTgtCoa)
            dKsfDThetaTgtCoa = spatial_freq_sf_poly_der(thetaTgtCoa)
            # Compute spatial frequency domain phase slopes in Ka and Kc directions
            # NB: sign for the phase may be ignored as it is cancelled in a subsequent computation.
            dPhiDKaTgtCoa = row_transform * numpy.cos(thetaTgtCoa) + col_transform * numpy.sin(thetaTgtCoa)
            dPhiDKcTgtCoa = -row_transform * numpy.sin(thetaTgtCoa) + col_transform * numpy.cos(thetaTgtCoa)
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
        def method_projection(instance, row_transform, col_transform, time_coa, arp_coa, varp_coa):
            """
            RgAzComp specific intermediate projection.

            Parameters
            ----------
            row_transform : numpy.ndarray
            col_transform : numpy.ndarray
            time_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            ARP_minus_SCP = arp_coa - SCP
            rSCPTgtCoa = numpy.linalg.norm(ARP_minus_SCP, axis=-1)
            rDotSCPTgtCoa = numpy.sum(varp_coa*ARP_minus_SCP, axis=-1)/rSCPTgtCoa
            deltaRTgtCoa = row_transform
            deltaRDotTgtCoa = -numpy.linalg.norm(varp_coa, axis=-1)*az_sf*col_transform
            return rSCPTgtCoa + deltaRTgtCoa, rDotSCPTgtCoa + deltaRDotTgtCoa

        return method_projection

    def inca_projection():
        inca = sicd.RMA.INCA
        r_ca_scp = inca.R_CA_SCP
        time_ca_poly = inca.TimeCAPoly
        drate_sf_poly = inca.DRateSFPoly

        # noinspection PyUnusedLocal, PyIncorrectDocstring
        def method_projection(instance, row_transform, col_transform, time_coa, arp_coa, varp_coa):
            """
            INCA specific intermediate projection.

            Parameters
            ----------
            row_transform : numpy.ndarray
            col_transform : numpy.ndarray
            time_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            # compute range/time of closest approach
            R_CA_TGT = r_ca_scp + row_transform  # Range at closest approach
            t_CA_TGT = time_ca_poly(col_transform)  # Time of closest approach
            # Compute ARP velocity magnitude (actually squared, since that's how it's used) at t_CA_TGT
            # noinspection PyProtectedMember
            VEL2_CA_TGT = numpy.sum(instance._varp_poly(t_CA_TGT)**2, axis=-1)
            # Compute the Doppler Rate Scale Factor for image Grid location
            DRSF_TGT = drate_sf_poly(row_transform, col_transform)
            # Difference between COA time and CA time
            dt_COA_TGT = time_coa - t_CA_TGT
            r_tgt_coa = numpy.sqrt(R_CA_TGT*R_CA_TGT + DRSF_TGT*VEL2_CA_TGT*dt_COA_TGT*dt_COA_TGT)
            r_dot_tgt_coa = (DRSF_TGT/r_tgt_coa)*VEL2_CA_TGT*dt_COA_TGT
            return r_tgt_coa, r_dot_tgt_coa

        return method_projection

    def plane_projection():
        SCP = sicd.GeoData.SCP.ECF.get_array()
        uRow = sicd.Grid.Row.UVectECF.get_array()
        uCol = sicd.Grid.Row.UVectECF.get_array()

        # noinspection PyUnusedLocal, PyIncorrectDocstring
        def method_projection(instance, row_transform, col_transform, time_coa, arp_coa, varp_coa):
            """
            Plane specific intermediate projection.

            Parameters
            ----------
            row_transform : numpy.ndarray
            col_transform : numpy.ndarray
            time_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            ARP_minus_IPP = arp_coa - (SCP + numpy.outer(row_transform, uRow) + numpy.outer(col_transform, uCol))
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


def _get_sicd_adjustment_params(sicd, delta_arp, delta_varp, adj_params_frame):
    """
    Gets the SICD adjustment params.

    Parameters
    ----------
    sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    delta_arp : None|numpy.ndarray|list|tuple
    delta_varp : None|numpy.ndarray|list|tuple
    adj_params_frame : str

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
    """

    delta_arp = _validate_adj_param(delta_arp, 'delta_arp')
    delta_varp = _validate_adj_param(delta_varp, 'delta_varp')
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
    return delta_arp, delta_varp


def _get_sidd_type_projection(sidd):
    """
    Gets an intermediate method specific projection method with six required
    calling arguments (self, row_transform, col_transform, time_coa, arp_coa, varp_coa).

    Parameters
    ----------
    sidd : sarpy.io.product.sidd1_elements.SIDD.SIDDType1|sarpy.io.product.sidd2_elements.SIDD.SIDDType2

    Returns
    -------
    (Poly2DType, callable)
    """

    from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
    from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1

    def pgp(the_sidd):
        """

        Parameters
        ----------
        the_sidd : SIDDType2|SIDDType1

        Returns
        -------
        callable
        """
        plane_proj = the_sidd.Measurement.PlaneProjection

        SRP = plane_proj.ReferencePoint.ECEF.get_array()
        SRP_row = plane_proj.ReferencePoint.Point.Row
        SRP_col = plane_proj.ReferencePoint.Point.Col
        row_vector = plane_proj.ProductPlane.RowUnitVector.get_array()*plane_proj.SampleSpacing.Row
        col_vector = plane_proj.ProductPlane.ColUnitVector.get_array()*plane_proj.SampleSpacing.Col

        # noinspection PyUnusedLocal, PyIncorrectDocstring
        def method_projection(instance, row_transform, col_transform, time_coa, arp_coa, varp_coa):
            """
            Plane specific intermediate projection.

            Parameters
            ----------
            row_transform : numpy.ndarray
            col_transform : numpy.ndarray
            time_coa : numpy.ndarray
            arp_coa : numpy.ndarray
            varp_coa : numpy.ndarray

            Returns
            -------
            Tuple[numpy.ndarray, numpy.ndarray]
            """

            ARP_minus_IPP = arp_coa - \
                            (SRP + numpy.outer(row_transform - SRP_row, row_vector) +
                             numpy.outer(col_transform - SRP_col, col_vector))
            r_tgt_coa = numpy.linalg.norm(ARP_minus_IPP, axis=-1)
            r_dot_tgt_coa = numpy.sum(varp_coa * ARP_minus_IPP, axis=-1)/r_tgt_coa
            return r_tgt_coa, r_dot_tgt_coa
        return plane_proj.TimeCOAPoly, method_projection

    if not isinstance(sidd, (SIDDType2, SIDDType1)):
        raise TypeError('Got unhandled type {}'.format(type(sidd)))

    if sidd.Measurement.PlaneProjection is not None:
        return pgp(sidd)
    else:
        raise ValueError('Currently the only supported projection is PlaneProjection.')


def _get_sidd_adjustment_params(sidd, delta_arp, delta_varp, adj_params_frame):
    """
    Get the SIDD adjustment parameters.

    Parameters
    ----------
    sidd : sarpy.io.product.sidd1_elements.SIDD.SIDDType1|sarpy.io.product.sidd2_elements.SIDD.SIDDType2
    delta_arp : None|numpy.ndarray|list|tuple
    delta_varp : None|numpy.ndarray|list|tuple
    adj_params_frame : str

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
    """

    from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
    from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1

    if not isinstance(sidd, (SIDDType2, SIDDType1)):
        raise TypeError('Got sidd of unhandled type {}'.format(type(sidd)))

    delta_arp = _validate_adj_param(delta_arp, 'delta_arp')
    delta_varp = _validate_adj_param(delta_varp, 'delta_varp')

    if adj_params_frame in ['RIC_ECI', 'RIC_ECF']:
        arp_pos_poly = sidd.Measurement.ARPPoly
        arp_vel_poly = arp_pos_poly.derivative(der_order=1, return_poly=True)
        if sidd.Measurement.PlaneProjection is not None:
            srp_row = sidd.Measurement.PlaneProjection.ReferencePoint.Point.Row
            srp_col = sidd.Measurement.PlaneProjection.ReferencePoint.Point.Col
            srp_coa_time = sidd.Measurement.PlaneProjection.TimeCOAPoly(srp_row, srp_col)
            srp_pos = arp_pos_poly(srp_coa_time)
            srp_vel = arp_vel_poly(srp_coa_time)
            ric_matrix = _ric_ecf_mat(srp_pos, srp_vel, adj_params_frame)
            delta_arp = ric_matrix.dot(delta_arp)
            delta_varp = ric_matrix.dot(delta_varp)
        else:
            raise ValueError('Got unhandled projection type {}'.format(sidd.Measurement.ProjectionType))
    return delta_arp, delta_varp


class COAProjection(object):
    """
    The Center of Aperture projection object, which provides common projection
    functionality for all image to R/Rdot projection.
    """

    __slots__ = (
        '_time_coa_poly', '_arp_poly', '_varp_poly', '_method_proj',
        '_row_shift', '_row_mult', '_col_shift', '_col_mult',
        '_delta_arp', '_delta_varp', '_range_bias',)

    def __init__(self, time_coa_poly, arp_poly, method_projection,
                 row_shift=0, row_mult=1, col_shift=0, col_mult=1,
                 delta_arp=None, delta_varp=None, range_bias=None):
        """

        Parameters
        ----------
        time_coa_poly : Poly2DType
            The time center of aperture polynomial.
        arp_poly : XYZPolyType
            The aperture position polynomial.
        method_projection : callable
            The method specific projection for performing the projection from image
            coordinates to R/Rdot space. The call signature is expected to be
            `method_projection(instance, row_transform, col_transform, time_coa, arp_coa, varp_coa)`,
            where `row_transform = row_mult*(row - row_shift)`,
            `col_transform = col_mult*(col - col_shift)`,
            `time_coa = time_coa_poly(row_transform, col_transform)`,
            `arp_coa = arp_poly(time_coa)`, and `varp_coa = varp_poly(time_coa)`.
        row_shift : int|float
            The shift part of the affine row transformation for plugging into the
            time coa polynomial.
        row_mult : int|float
            The multiple part of the affine row transformation for plugging into
            the time coa polynomial.
        col_shift : int|float
            The shift part of the affine column transformation for plugging into
            the time coa polynomial.
        col_mult : int|float
            The multiple part of the affine column transformation for plugging into
            the time coa polynomial.
        delta_arp : None|numpy.ndarray|list|tuple
            ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
        delta_varp : None|numpy.ndarray|list|tuple
            VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
        range_bias : float|int
            Range bias adjustable parameter (m), defaults to 0.
        """

        if not isinstance(time_coa_poly, Poly2DType):
            raise TypeError('time_coa_poly must be a Poly2DType instance.')
        self._time_coa_poly = time_coa_poly

        if not isinstance(arp_poly, XYZPolyType):
            raise TypeError('arp_poly must be an XYZPolyType instance.')
        self._arp_poly = arp_poly
        self._varp_poly = self._arp_poly.derivative(der_order=1, return_poly=True)  # type: XYZPolyType

        if not callable(method_projection):
            raise TypeError('method_projection must be callable.')
        self._method_proj = MethodType(method_projection, self)
        # affine transform parameters
        self._row_shift = float(row_shift)
        self._row_mult = float(row_mult)
        self._col_shift = float(col_shift)
        self._col_mult = float(col_mult)
        # aperture location adjustment parameters
        self._delta_arp = _validate_adj_param(delta_arp, 'delta_arp')
        self._delta_varp = _validate_adj_param(delta_varp, 'delta_varp')
        self._range_bias = 0.0 if range_bias is None else float(range_bias) # type: float

    @classmethod
    def from_sicd(cls, sicd, delta_arp=None, delta_varp=None, range_bias=None, adj_params_frame='ECF'):
        """
        Construct from a SICD structure.

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
            One of `('ECF', 'RIC_ECI', 'RIC_ECF')`.

        Returns
        -------
        COAProjection
        """

        if not sicd.can_project_coordinates():
            raise ValueError('Insufficient metadata populated to formulate projection.')

        time_coa_poly = sicd.Grid.TimeCOAPoly
        # fall back to approximation if TimeCOAPoly is not populated
        if time_coa_poly is None:
            time_coa_poly = Poly2DType(Coefs=[[sicd.Timeline.CollectDuration/2, ], ])
            logging.warning(
                'Using (constant) approximation to TimeCOAPoly, which may result in poor projection results.')
        arp_poly = sicd.Position.ARPPoly
        # transform parameters
        row_mult = sicd.Grid.Row.SS
        row_shift = sicd.ImageData.SCPPixel.Row - sicd.ImageData.FirstRow
        col_mult = sicd.Grid.Col.SS
        col_shift = sicd.ImageData.SCPPixel.Col - sicd.ImageData.FirstCol
        # location adjustment parameters
        delta_arp, delta_varp = _get_sicd_adjustment_params(sicd, delta_arp, delta_varp, adj_params_frame)
        return cls(time_coa_poly, arp_poly, _get_sicd_type_specific_projection(sicd),
                   row_shift=row_shift, row_mult=row_mult, col_shift=col_shift, col_mult=col_mult,
                   delta_arp=delta_arp, delta_varp=delta_varp, range_bias=range_bias)

    @classmethod
    def from_sidd(cls, sidd, delta_arp=None, delta_varp=None, range_bias=None, adj_params_frame='ECF'):
        """
        Construct from the SIDD structure.

        Parameters
        ----------
        sidd : sarpy.io.product.sidd1_elements.SIDD.SIDDType1|sarpy.io.product.sidd2_elements.SIDD.SIDDType2
        delta_arp : None|numpy.ndarray|list|tuple
            ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
        delta_varp : None|numpy.ndarray|list|tuple
            VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
        range_bias : float|int
            Range bias adjustable parameter (m), defaults to 0.
        adj_params_frame : str
            One of `('ECF', 'RIC_ECI', 'RIC_ECF')`.

        Returns
        -------
        COAProjection
        """

        time_coa_poly, method_projection = _get_sidd_type_projection(sidd)
        arp_poly = sidd.Measurement.ARPPoly
        delta_arp, delta_varp = _get_sidd_adjustment_params(
            sidd, delta_arp, delta_varp, adj_params_frame)
        return cls(time_coa_poly, arp_poly, method_projection,
                   row_shift=0, row_mult=1, col_shift=0, col_mult=1,
                   delta_arp=delta_arp, delta_varp=delta_varp, range_bias=range_bias)

    def _init_proj(self, im_points):
        """

        Parameters
        ----------
        im_points : numpy.ndarray

        Returns
        -------
        Tuple[numpy.ndarray,...]
        """

        row_transform = (im_points[:, 0] - self._row_shift)*self._row_mult
        col_transform = (im_points[:, 1] - self._col_shift)*self._col_mult
        time_coa = self._time_coa_poly(row_transform, col_transform)
        # calculate aperture reference position and velocity at target time
        arp_coa = self._arp_poly(time_coa)
        varp_coa = self._varp_poly(time_coa)
        return row_transform, col_transform, time_coa, arp_coa, varp_coa

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
            * `time_coa` - center of aperture time since CDP start for input ip
            * `arp_coa` - aperture reference position at time_coa
            * `varp_coa` - velocity at time_coa
        """

        row_transform, col_transform, time_coa, arp_coa, varp_coa = self._init_proj(im_points)
        r_tgt_coa, r_dot_tgt_coa = self._method_proj(row_transform, col_transform, time_coa, arp_coa, varp_coa)
        # adjust parameters
        arp_coa += self._delta_arp
        varp_coa += self._delta_varp
        r_tgt_coa += self._range_bias
        return r_tgt_coa, r_dot_tgt_coa, time_coa, arp_coa, varp_coa


def _get_coa_projection(structure, use_structure_coa, **coa_args):
    """

    Parameters
    ----------
    structure
    use_structure_coa : bool
    coa_args

    Returns
    -------
    COAProjection
    """

    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
    from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1

    if use_structure_coa and structure.coa_projection is not None:
        return structure.coa_projection
    elif isinstance(structure, SICDType):
        return COAProjection.from_sicd(structure, **coa_args)
    elif isinstance(structure, (SIDDType2, SIDDType1)):
        return COAProjection.from_sidd(structure, **coa_args)
    else:
        raise ValueError('Got unhandled type {}'.format(type(structure)))


###############
# General helper methods for extracting params from the sicd or sidd

def _get_reference_point(structure):
    """
    Gets the reference point in ECF coordinates.

    Parameters
    ----------
    structure

    Returns
    -------
    numpy.ndarray
    """

    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
    from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1

    if isinstance(structure, SICDType):
        return structure.GeoData.SCP.ECF.get_array(dtype='float64')
    elif isinstance(structure, (SIDDType2, SIDDType1)):
        proj_type = structure.Measurement.ProjectionType
        if proj_type != 'PlaneProjection':
            raise ValueError('Got unsupported projection type {}'.format(proj_type))
        return structure.Measurement.PlaneProjection.ReferencePoint.ECEF.get_array(dtype='float64')
    else:
        raise TypeError('Got unhandled type {}'.format(type(structure)))


def _get_outward_norm(structure, gref):
    """
    Gets the default outward unit norm.

    Parameters
    ----------
    structure

    Returns
    -------
    numpy.ndarray
    """

    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
    from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1

    if isinstance(structure, SICDType):
        if structure.ImageFormation.ImageFormAlgo == 'PFA':
            return structure.PFA.FPN.get_array()
        else:
            return wgs_84_norm(gref)
    elif isinstance(structure, (SIDDType2, SIDDType1)):
        proj_type = structure.Measurement.ProjectionType
        if proj_type != 'PlaneProjection':
            raise ValueError('Got unsupported projection type {}'.format(proj_type))
        the_proj = structure.Measurement.PlaneProjection
        # image plane details
        uRow = the_proj.ProductPlane.RowUnitVector.get_array(dtype='float64')
        uCol = the_proj.ProductPlane.ColUnitVector.get_array(dtype='float64')
        # outward unit norm for plane
        uGPN = numpy.cross(uRow, uCol)
        uGPN /= numpy.linalg.norm(uGPN)
        if numpy.dot(uGPN, gref) < 0:
            uGPN *= -1
        return uGPN
    else:
        raise TypeError('Got unhandled type {}'.format(type(structure)))


def _extract_plane_params(structure):
    """
    Extract the required parameters for projection from ground to plane for a SICD.

    Parameters
    ----------
    structure : sarpy.io.complex.sicd_elements.SICD.SICDType|sarpy.io.product.sidd2_elements.SIDD.SIDDType|sarpy.io.product.sidd1_elements.SIDD.SIDDType

    Returns
    -------

    """

    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
    from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1

    if isinstance(structure, SICDType):
        # reference point for the plane
        ref_point = structure.GeoData.SCP.ECF.get_array()
        ref_pixel = structure.ImageData.SCPPixel.get_array()

        # pixel spacing
        row_ss = structure.Grid.Row.SS
        col_ss = structure.Grid.Col.SS

        # image plane details
        uRow = structure.Grid.Row.UVectECF.get_array()  # unit normal in row direction
        uCol = structure.Grid.Col.UVectECF.get_array()  # unit normal in column direction

        # outward unit norm
        uGPN = structure.PFA.FPN.get_array() if structure.ImageFormation.ImageFormAlgo == 'PFA' \
            else wgs_84_norm(ref_point)

        # uSPN - defined in section 3.1 as normal to instantaneous slant plane that contains SCP at SCP COA is
        # tangent to R/Rdot contour at SCP. Points away from center of Earth. Use look to establish sign.
        ARP_SCP_COA = structure.SCPCOA.ARPPos.get_array()
        VARP_SCP_COA = structure.SCPCOA.ARPVel.get_array()
        uSPN = structure.SCPCOA.look*numpy.cross(VARP_SCP_COA, ref_point - ARP_SCP_COA)
        uSPN /= numpy.linalg.norm(uSPN)

        return ref_point, ref_pixel, row_ss, col_ss, uRow, uCol, uGPN, uSPN
    elif isinstance(structure, (SIDDType1, SIDDType2)):
        proj_type = structure.Measurement.ProjectionType
        if proj_type != 'PlaneProjection':
            raise ValueError('Got unsupported projection type {}'.format(proj_type))

        the_proj = structure.Measurement.PlaneProjection

        # reference point for the plane
        ref_point = the_proj.ReferencePoint.ECEF.get_array(dtype='float64')
        ref_pixel = the_proj.ReferencePoint.Point.get_array(dtype='float64')
        # pixel spacing
        row_ss = the_proj.SampleSpacing.Row
        col_ss = the_proj.SampleSpacing.Col
        # image plane details
        uRow = the_proj.ProductPlane.RowUnitVector.get_array(dtype='float64')
        uCol = the_proj.ProductPlane.ColUnitVector.get_array(dtype='float64')
        # outward unit norm for plane
        uGPN = numpy.cross(uRow, uCol)
        uGPN /= numpy.linalg.norm(uGPN)
        if numpy.dot(uGPN, ref_point) < 0:
            uGPN *= -1

        # slant plane is identical to outward unit norm
        return ref_point, ref_pixel, row_ss, col_ss, uRow, uCol, uGPN, uGPN
    else:
        raise TypeError('Got structure unsupported type {}'.format(type(structure)))


#############
# Ground-to-Image (aka Scene-to-Image) projection.

def _validate_coords(coords):
    if not isinstance(coords, numpy.ndarray):
        coords = numpy.array(coords, dtype='float64')

    orig_shape = coords.shape

    if len(orig_shape) == 1:
        coords = numpy.reshape(coords, (1, -1))
    if coords.shape[-1] != 3:
        raise ValueError(
            'The coords array must represent an array of points in ECF coordinates, '
            'so the final dimension of coords must have length 3. Have coords.shape = {}'.format(coords.shape))

    return coords, orig_shape


def _ground_to_image(coords, coa_proj, uGPN,
                     ref_point, ref_pixel, uIPN, sf, row_ss, col_ss, uProj,
                     row_col_transform, ipp_transform, tolerance, max_iterations):
    """
    Basic level helper function.

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
    coa_proj : COAProjection
    uGPN : numpy.ndarray
    ref_point : numpy.ndarray
    ref_pixel : numpy.ndarray
    uIPN : numpy.ndarray
    sf : float
    row_ss : float
    col_ss : float
    uProj : numpy.ndarray
    row_col_transform : numpy.ndarray
    ipp_transform : numpy.ndarray
    tolerance : float
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
    im_points = numpy.zeros((coords.shape[0], 2), dtype='float64')
    delta_gpn = numpy.zeros((coords.shape[0],), dtype='float64')
    cont = True
    iteration = 0

    matrix_transform = numpy.dot(row_col_transform, ipp_transform)
    # (3 x 2)*(2 x 2) = (3 x 2)

    while cont:
        # project ground plane to image plane iteration
        iteration += 1
        dist_n = numpy.dot(ref_point - g_n, uIPN)/sf  # (N, )
        i_n = g_n + numpy.outer(dist_n, uProj)  # (N, 3)
        delta_ipp = i_n - ref_point  # (N, 3)
        ip_iter = numpy.dot(delta_ipp, matrix_transform)  # (N, 2)
        im_points[:, 0] = ip_iter[:, 0]/row_ss + ref_pixel[0]
        im_points[:, 1] = ip_iter[:, 1]/col_ss + ref_pixel[1]
        # transform to ground plane containing the scene points and check how it compares
        p_n = _image_to_ground_plane(im_points, coa_proj, g_n, uGPN)
        # compute displacement between scene point and this new projected point
        diff_n = coords - p_n
        delta_gpn[:] = numpy.linalg.norm(diff_n, axis=1)
        g_n += diff_n

        # should we continue iterating?
        cont = numpy.any(delta_gpn > tolerance) and (iteration < max_iterations)

    return im_points, delta_gpn, iteration


def ground_to_image(coords, structure, tolerance=1e-2, max_iterations=10, block_size=50000,
                    use_structure_coa=True, **coa_args):
    """
    Transforms a 3D ECF point to pixel (row/column) coordinates. This is
    implemented in accordance with the SICD Image Projections Description Document.
    **Really Scene-To-Image projection.**"

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
        ECF coordinate to map to scene coordinates, of size `N x 3`.
    structure : sarpy.io.complex.sicd_elements.SICD.SICDType|sarpy.io.product.sidd2_elements.SIDD.SIDDType|sarpy.io.product.sidd1_elements.SIDD.SIDDType
        The SICD or SIDD data structure.
    tolerance : float|int
        Ground plane displacement tol (m).
    max_iterations : int
        maximum number of iterations to perform
    block_size : int|None
        size of blocks of coordinates to transform at a time
    use_structure_coa : bool
        If sicd.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    coa_args
        The keyword arguments from the COAProjection.from_sicd class method.

    Returns
    -------
    Tuple[numpy.ndarray, float, int]
        * `image_points` - the determined image point array, of size `N x 2`. Following
          the SICD convention, he upper-left pixel is [0, 0].
        * `delta_gpn` - residual ground plane displacement (m).
        * `iterations` - the number of iterations performed.
    """

    coords, orig_shape = _validate_coords(coords)
    coa_proj = _get_coa_projection(structure, use_structure_coa, **coa_args)

    ref_point, ref_pixel, row_ss, col_ss, uRow, uCol, \
    uGPN, uSPN = _extract_plane_params(structure)

    uIPN = numpy.cross(uRow, uCol)  # NB: only outward pointing if Row/Col are right handed system
    uIPN /= numpy.linalg.norm(uIPN)  # NB: uRow/uCol may not be perpendicular

    cos_theta = numpy.dot(uRow, uCol)
    sin_theta = numpy.sqrt(1 - cos_theta*cos_theta)
    ipp_transform = numpy.array(
        [[1, -cos_theta], [-cos_theta, 1]], dtype='float64')/(sin_theta*sin_theta)
    row_col_transform = numpy.zeros((3, 2), dtype='float64')
    row_col_transform[:, 0] = uRow
    row_col_transform[:, 1] = uCol
    sf = float(numpy.dot(uSPN, uIPN))  # scale factor

    tolerance = float(tolerance)
    if tolerance < 1e-12:
        logging.warning(
            'minimum allowed tolerance is 1e-12 meters, resetting from {}'.format(tolerance))
        tolerance = 1e-12

    # prepare the work space
    coords_view = numpy.reshape(coords, (-1, 3))  # possibly or make 2-d flatten
    num_points = coords_view.shape[0]
    if block_size is None or num_points <= block_size:
        image_points, delta_gpn, iters = _ground_to_image(
            coords_view, coa_proj, uGPN,
            ref_point, ref_pixel, uIPN, sf, row_ss, col_ss, uSPN,
            row_col_transform, ipp_transform, tolerance, max_iterations)
        iters = numpy.full((num_points, ), iters)
    else:
        image_points = numpy.zeros((num_points, 2), dtype='float64')
        delta_gpn = numpy.zeros((num_points, ), dtype='float64')
        iters = numpy.zeros((num_points, ), dtype='int16')

        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block+block_size, num_points)
            image_points[start_block:end_block, :], delta_gpn[start_block:end_block], \
                iters[start_block:end_block] = _ground_to_image(
                    coords_view[start_block:end_block, :], coa_proj, uGPN,
                    ref_point, ref_pixel, uIPN, sf, row_ss, col_ss, uSPN,
                    row_col_transform, ipp_transform, tolerance, max_iterations)
            start_block = end_block

    if len(orig_shape) == 1:
        image_points = numpy.reshape(image_points, (-1,))
    elif len(orig_shape) > 1:
        image_points = numpy.reshape(image_points, orig_shape[:-1]+(2, ))
        delta_gpn = numpy.reshape(delta_gpn, orig_shape[:-1])
        iters = numpy.reshape(iters, orig_shape[:-1])
    return image_points, delta_gpn, iters


def ground_to_image_geo(coords, structure, ordering='latlong', **kwargs):
    """
    Transforms a 3D Lat/Lon/HAE point to pixel (row/column) coordinates.
    This is implemented in accordance with the SICD Image Projections Description Document.

    Parameters
    ----------
    coords : numpy.ndarray|tuple|list
        Lat/Lon/HAE coordinate to map to scene coordinates, of size `N x 3`.
    structure : sarpy.io.complex.sicd_elements.SICD.SICDType|sarpy.io.product.sidd2_elements.SIDD.SIDDType|sarpy.io.product.sidd1_elements.SIDD.SIDDType
        The SICD or SIDD structure.
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

    return ground_to_image(geodetic_to_ecf(coords, ordering=ordering), structure, **kwargs)


############
# Image-To-Ground projections

def _validate_im_points(im_points):
    """

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple

    Returns
    -------
    numpy.ndarray
    """

    if im_points is None:
        raise ValueError('The argument cannot be None')

    if not isinstance(im_points, numpy.ndarray):
        im_points = numpy.array(im_points, dtype='float64')

    orig_shape = im_points.shape

    if len(im_points.shape) == 1:
        im_points = numpy.reshape(im_points, (1, -1))
    if im_points.shape[-1] != 2:
        raise ValueError(
            'The im_points array must represent an array of points in pixel coordinates, '
            'so the final dimension of im_points must have length 2. '
            'Have im_points.shape = {}'.format(im_points.shape))
    return im_points, orig_shape


def image_to_ground(
        im_points, structure, block_size=50000, projection_type='HAE',
        use_structure_coa=True, **kwargs):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        (row, column) coordinates of N points in image (or subimage if FirstRow/FirstCol are nonzero).
        Following SICD convention, the upper-left pixel is [0, 0].
    structure : sarpy.io.complex.sicd_elements.SICD.SICDType|sarpy.io.product.sidd2_elements.SIDD.SIDDType|sarpy.io.product.sidd1_elements.SIDD.SIDDType
        The SICD or SIDD structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    projection_type : str
        One of ['PLANE', 'HAE', 'DEM'].
    use_structure_coa : bool
        If structure.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
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
        return image_to_ground_plane(
            im_points, structure, block_size=block_size, use_structure_coa=use_structure_coa, **kwargs)
    elif p_type == 'HAE':
        return image_to_ground_hae(
            im_points, structure, block_size=block_size, use_structure_coa=use_structure_coa, **kwargs)
    elif p_type == 'DEM':
        return image_to_ground_dem(
            im_points, structure, block_size=block_size, use_structure_coa=use_structure_coa, **kwargs)
    else:
        raise ValueError('Got unrecognized projection type {}'.format(projection_type))


def image_to_ground_geo(
        im_points, structure, ordering='latlong', block_size=50000, projection_type='HAE',
        use_structure_coa=True, **kwargs):
    """
    Transforms image coordinates to ground plane Lat/Lon/HAE coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        (row, column) coordinates of N points in image (or subimage if FirstRow/FirstCol are nonzero).
        Following SICD convention, the upper-left pixel is [0, 0].
    structure : sarpy.io.complex.sicd_elements.SICD.SICDType|sarpy.io.product.sidd2_elements.SIDD.SIDDType|sarpy.io.product.sidd1_elements.SIDD.SIDDType
        The SICD or SIDD structure.
    ordering : str
        Determines whether return is ordered as `[lat, long, hae]` or `[long, lat, hae]`.
        Passed through to :func:`sarpy.geometry.geocoords.ecf_to_geodetic`.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    projection_type : str
        One of ['PLANE', 'HAE', 'DEM'].
    use_structure_coa : bool
        If structure.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    kwargs
        See the keyword arguments in :func:`image_to_ground`.

    Returns
    -------
    numpy.ndarray
        Ground Plane Point (in Lat/Lon/HAE coordinates) along the R/Rdot contour.
    """

    return ecf_to_geodetic(
        image_to_ground(
            im_points, structure, block_size=block_size, projection_type=projection_type,
            use_structure_coa=use_structure_coa, **kwargs),
        ordering=ordering)


#####
# Image-to-Ground Plane

def _image_to_ground_plane_perform(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, uZ):
    """

    Parameters
    ----------
    r_tgt_coa : numpy.ndarray
    r_dot_tgt_coa : numpy.ndarray
    arp_coa : numpy.ndarray
    varp_coa : numpy.ndarray
    gref : numpy.ndarray
    uZ : numpy.ndarray

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
    uX = (varp_coa - numpy.outer(vZ, uZ))/vX[:, numpy.newaxis]
    uY = numpy.cross(uZ, uX)
    # Compute cosine of azimuth angle to ground plane point
    cosAz = (-r_dot_tgt_coa+vZ*sinGraz) / (vX * cosGraz)
    cosAz[numpy.abs(cosAz) > 1] = numpy.nan  # R/Rdot combination not possible in given plane

    # Compute sine of azimuth angle. Use LOOK to establish sign.
    look = numpy.sign(numpy.dot(numpy.cross(arp_coa-gref, varp_coa), uZ))
    sinAz = look*numpy.sqrt(1-cosAz*cosAz)

    # Compute Ground Plane Point in ground plane and along the R/Rdot contour
    return aGPN + uX*(gd*cosAz)[:, numpy.newaxis] + uY*(gd*sinAz)[:, numpy.newaxis]


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

    r_tgt_coa, r_dot_tgt_coa, time_coa, arp_coa, varp_coa = coa_projection.projection(im_points)
    values = _image_to_ground_plane_perform(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, gref, uZ)
    return values


def image_to_ground_plane(
        im_points, structure, block_size=50000, gref=None, ugpn=None,
        use_structure_coa=True, **coa_args):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    structure : sarpy.io.complex.sicd_elements.SICD.SICDType|sarpy.io.product.sidd2_elements.SIDD.SIDDType|sarpy.io.product.sidd1_elements.SIDD.SIDDType
        The SICD or SIDD structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    gref : None|numpy.ndarray|list|tuple
        Ground plane reference point ECF coordinates (m). The default is the SCP or Reference Point.
    ugpn : None|numpy.ndarray|list|tuple
        Vector normal to the plane to which we are projecting.
    use_structure_coa : bool
        If structure.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    coa_args
        keyword arguments for COAProjection.from_sicd class method.

    Returns
    -------
    numpy.ndarray
        Ground Plane Point (in ECF coordinates) corresponding to the input image coordinates.
    """

    im_points, orig_shape = _validate_im_points(im_points)
    coa_proj = _get_coa_projection(structure, use_structure_coa, **coa_args)

    # method parameter validation
    if gref is None:
        gref = _get_reference_point(structure)
    if not isinstance(gref, numpy.ndarray):
        gref = numpy.array(gref, dtype='float64')
    if gref.size != 3:
        raise ValueError('gref must have three elements.')
    if gref.ndim != 1:
        gref = numpy.reshape(gref, (3, ))

    if ugpn is None:
        ugpn = _get_outward_norm(structure, gref)
    if not isinstance(ugpn, numpy.ndarray):
        ugpn = numpy.array(ugpn, dtype='float64')
    if ugpn.size != 3:
        raise ValueError('ugpn must have three elements.')
    if ugpn.ndim != 1:
        ugpn = numpy.reshape(ugpn, (3, ))
    uZ = ugpn/numpy.linalg.norm(ugpn)

    # prepare workspace
    im_points_view = numpy.reshape(im_points, (-1, 2))  # possibly or make 2-d flatten
    num_points = im_points_view.shape[0]
    if block_size is None or num_points <= block_size:
        coords = _image_to_ground_plane(im_points_view, coa_proj, gref, uZ)
    else:
        coords = numpy.zeros((num_points, 3), dtype='float64')
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
        r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, ref_point, ugpn,
        hae0, tolerance, max_iterations, ref_hae):
    """
    Intermediate helper method.

    Parameters
    ----------
    r_tgt_coa : numpy.ndarray
    r_dot_tgt_coa : numpy.ndarray
    arp_coa : numpy.ndarray
    varp_coa : numpy.ndarray
    ref_point : numpy.ndarray
    ugpn : numpy.ndarray
    hae0 : float
    tolerance : float
    max_iterations : int
    ref_hae : float

    Returns
    -------
    numpy.ndarray
    """

    # Compute the geodetic ground plane normal at the ref_point.
    look = numpy.sign(numpy.sum(numpy.cross(arp_coa, varp_coa)*(ref_point - arp_coa), axis=1))
    gref = ref_point - (ref_hae - hae0)*ugpn
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
        gpp_llh = ecf_to_geodetic(gpp)
        delta_hae = gpp_llh[:, 2] - hae0
        max_abs_delta_hae = numpy.max(numpy.abs(delta_hae))
        gref = gpp - (delta_hae[:, numpy.newaxis] * ugpn)
        # should we stop our iteration?
        cont = (max_abs_delta_hae > tolerance) and (iters < max_iterations)

    # Compute the unit slant plane normal vector, uspn, that is tangent to the R/Rdot contour at point gpp
    uspn = numpy.cross(varp_coa, (gpp - arp_coa))*look[:, numpy.newaxis]
    uspn /= numpy.linalg.norm(uspn, axis=-1)[:, numpy.newaxis]
    # For the final straight line projection, project from point gpp along
    # the slant plane normal (as opposed to the ground plane normal that was
    # used in the iteration) to point slp.
    sf = numpy.sum(ugpn*uspn, axis=-1)
    slp = gpp - uspn*(delta_hae/sf)[:, numpy.newaxis]
    # Assign surface point SPP position by adjusting the HAE to be on the
    # HAE0 surface.
    spp_llh = ecf_to_geodetic(slp)
    spp_llh[:, 2] = hae0
    spp = geodetic_to_ecf(spp_llh)
    return spp


def _image_to_ground_hae(
        im_points, coa_projection, hae0, tolerance, max_iterations, ref_hae, ref_point):
    """
    Intermediate helper function for projection.

    Parameters
    ----------
    im_points : numpy.ndarray
        the image coordinate array
    coa_projection : COAProjection
    hae0 : float
    tolerance : float
    max_iterations : int
    ref_hae : float
    ref_point : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    # get (image formation specific) projection parameters
    r_tgt_coa, r_dot_tgt_coa, time_coa, arp_coa, varp_coa = coa_projection.projection(im_points)
    ugpn = wgs_84_norm(ref_point)
    return _image_to_ground_hae_perform(
        r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, ref_point, ugpn,
        hae0, tolerance, max_iterations, ref_hae)


def image_to_ground_hae(im_points, structure, block_size=50000,
                        hae0=None, tolerance=1e-3, max_iterations=10, use_structure_coa=True, **coa_args):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    structure : sarpy.io.complex.sicd_elements.SICD.SICDType|sarpy.io.product.sidd2_elements.SIDD.SIDDType|sarpy.io.product.sidd1_elements.SIDD.SIDDType
        The SICD or SIDD structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array will be
        transformed as a single block if `None`.
    hae0 : None|float|int
        Surface height (m) above the WGS-84 reference ellipsoid for projection point.
        Defaults to HAE at the SCP or Reference Point.
    tolerance : float|int
        Height threshold for convergence of iterative constant HAE computation (m).
    max_iterations : int
        Maximum number of iterations allowed for constant hae computation.
    use_structure_coa : bool
        If structure.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    coa_args
        keyword arguments for COAProjection.from_sicd class method.

    Returns
    -------
    numpy.ndarray
        Ground Plane Point (in ECF coordinates) with target hae corresponding to
        the input image coordinates.
    """

    # coa projection creation
    im_points, orig_shape = _validate_im_points(im_points)
    coa_proj = _get_coa_projection(structure, use_structure_coa, **coa_args)

    tolerance = float(tolerance)
    if tolerance < 1e-12:
        logging.warning(
            'minimum allowed tolerance is 1e-12, resetting from {0:8f}'.format(tolerance))
        tolerance = 1e-12

    max_iterations = int(max_iterations)
    if max_iterations < 1:
        logging.error('max_iterations must be a positive integer, resetting to 1 from {}'.format(max_iterations))
        max_iterations = 1
    if max_iterations > 100:
        logging.error('maximum allowed max_iterations is 100, resetting from {}'.format(max_iterations))
        max_iterations = 100

    # method parameter validation
    ref_point = _get_reference_point(structure)
    ref_llh = ecf_to_geodetic(ref_point)
    ref_hae = float(ref_llh[2])
    if hae0 is None:
        hae0 = ref_hae

    # prepare workspace
    im_points_view = numpy.reshape(im_points, (-1, 2))  # possibly or make 2-d flatten
    num_points = im_points_view.shape[0]
    if block_size is None or num_points <= block_size:
        coords = _image_to_ground_hae(im_points_view, coa_proj, hae0, tolerance, max_iterations, ref_hae, ref_point)
    else:
        coords = numpy.zeros((num_points, 3), dtype='float64')
        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block + block_size, num_points)
            coords[start_block:end_block, :] = _image_to_ground_hae(
                im_points_view[start_block:end_block], coa_proj, hae0, tolerance, max_iterations, ref_hae, ref_point)
            start_block = end_block

    if len(orig_shape) == 1:
        coords = numpy.reshape(coords, (-1,))
    elif len(orig_shape) > 1:
        coords = numpy.reshape(coords, orig_shape[:-1] + (3,))
    return coords


#####
# Image-to-DEM

def _do_dem_iteration(previous_ecf, previous_diff, this_ecf, this_diff):
    mask = (this_diff < 0)
    if numpy.any(mask):
        d0 = (previous_diff[mask])
        d1 = numpy.abs(this_diff[mask])
        return mask, (d1[:, numpy.newaxis]*previous_ecf[mask] + d0[:, numpy.newaxis]*this_ecf[mask])/((d0+d1)[:, numpy.newaxis])
    else:
        return None

def _image_to_ground_dem(
        im_points, coa_projection, dem_interpolator, min_dem, max_dem,
        vertical_step_size, ref_hae, ref_point):
    """

    Parameters
    ----------
    im_points : numpy.ndarray
    coa_projection : COAProjection
    dem_interpolator : DEMInterpolator
    min_dem : float
    max_dem : float
    vertical_step_size : float|int
    ref_hae: float
    ref_point : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    # get (image formation specific) projection parameters
    r_tgt_coa, r_dot_tgt_coa, time_coa, arp_coa, varp_coa = coa_projection.projection(im_points)
    ugpn = wgs_84_norm(ref_point)
    tolerance = 1e-3
    max_iterations = 10

    # if max_dem - min_dem is sufficiently small, then pretend it's flat
    if max_dem - min_dem < vertical_step_size:
        return _image_to_ground_hae_perform(
            r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, ref_point, ugpn, max_dem,
            tolerance, max_iterations, ref_hae)
    # set up workspace
    out = numpy.zeros((im_points.shape[0], 3), dtype='float64')
    cont_mask = numpy.ones((im_points.shape[0], ), dtype='bool')
    cont = True
    this_hae = max_dem
    previous_coords = _image_to_ground_hae_perform(
        r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, ref_point, ugpn, max_dem,
        tolerance, max_iterations, ref_hae)
    previous_llh = ecf_to_geodetic(previous_coords)
    previous_diff = previous_llh[:, 2] - dem_interpolator.get_elevation_hae(
        previous_llh[:, 0], previous_llh[:, 1])

    while cont:
        this_hae -= vertical_step_size
        this_coords = _image_to_ground_hae_perform(
            r_tgt_coa[cont_mask], r_dot_tgt_coa[cont_mask], arp_coa[cont_mask], varp_coa[cont_mask],
            ref_point, ugpn, this_hae, tolerance, max_iterations, ref_hae)
        this_llh = ecf_to_geodetic(this_coords)
        this_diff = this_llh[:, 2] - dem_interpolator.get_elevation_hae(this_llh[:, 0], this_llh[:, 1])
        result = _do_dem_iteration(previous_coords, previous_diff, this_coords, this_diff)
        if result is not None:
            this_mask, this_result = result
            temp_mask = numpy.zeros((im_points.shape[0], ), dtype='bool')
            temp_mask[cont_mask] = this_mask
            out[temp_mask, :] = this_result
            cont_mask[temp_mask] = False
            cont = numpy.any(cont_mask)
            if cont:
                previous_coords = this_coords[~this_mask, :]
                previous_diff = this_diff[~this_mask]
        else:
            previous_coords = this_coords
            previous_diff = this_diff
    return out


def _image_to_ground_dem_block(
        im_points, coa_projection, dem_interpolator, horizontal_step, lat_lon_box, block_size,
        lat_pad, lon_pad):
    """

    Parameters
    ----------
    im_points : numpy.ndarray
    coa_projection : COAProjection
    dem_interpolator : DEMInterpolator
    horizontal_step : float
    lat_lon_box : numpy.ndarray
    block_size : int|None
    lat_pad : float
    lon_pad : float

    Returns
    -------
    numpy.ndarray
    """

    # determine reference point
    ref_lat = 0.5*(lat_lon_box[0] + lat_lon_box[1])
    ref_lon = 0.5*(lat_lon_box[2] + lat_lon_box[3])
    ref_hae = float(dem_interpolator.get_elevation_hae(ref_lat, ref_lon))
    ref_ecf = geodetic_to_ecf([ref_lat, ref_lon, ref_hae])

    # determine max/min hae in the DEM region
    padded_box = numpy.array([
        max(-90, lat_lon_box[0] - 0.5*lat_pad), min(lat_lon_box[1] + 0.5*lat_pad, 90),
        max(-180, lat_lon_box[2] - 0.5*lon_pad), min(lat_lon_box[3] + 0.5*lon_pad, 180)], dtype='float64')
    min_dem = dem_interpolator.get_min_hae(padded_box) - 10
    max_dem = dem_interpolator.get_max_hae(padded_box) + 10

    # prepare workspace
    num_points = im_points.shape[0]
    if block_size is None or num_points <= block_size:
        coords = _image_to_ground_dem(
            im_points, coa_projection, dem_interpolator, min_dem, max_dem,
            horizontal_step, ref_hae, ref_ecf)
    else:
        coords = numpy.zeros((num_points, 3), dtype='float64')
        # proceed with block processing
        start_block = 0
        while start_block < num_points:
            end_block = min(start_block + block_size, num_points)
            coords[start_block:end_block, :] = _image_to_ground_dem(
                im_points[start_block:end_block, :], coa_projection, dem_interpolator,
                min_dem, max_dem, horizontal_step, ref_hae, ref_ecf)
            start_block = end_block
    return coords


def image_to_ground_dem(
        im_points, structure, block_size=50000, dem_interpolator=None,
        dem_type=None, geoid_file=None, pad_value=0.2,
        vertical_step_size=10, use_structure_coa=True, **coa_args):
    """
    Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
    described in SICD Image Projections document.

    Parameters
    ----------
    im_points : numpy.ndarray|list|tuple
        the image coordinate array
    structure : sarpy.io.complex.sicd_elements.SICD.SICDType|sarpy.io.product.sidd2_elements.SIDD.SIDDType|sarpy.io.product.sidd1_elements.SIDD.SIDDType
        The SICD or SIDD structure.
    block_size : None|int
        Size of blocks of coordinates to transform at a time. The entire array
        will be transformed as a single block if `None`.
    dem_interpolator : str|DEMInterpolator
        The DEMInterpolator. If this is a string, then a DTEDInterpolator will be
        constructed assuming that this is the DTED root search directory.
    dem_type : None|str|List[str]
        The DEM type or list of DEM types in order of priority. Only used if
        `dem_interpolator` is the search path.
    geoid_file : None|str|GeoidHeight
        The `GeoidHeight` object, an egm file name, or root directory containing
        one of the egm files in the sub-directory "geoid". If `None`, then default
        to the root directory of `dted_list`. Only used if `dem_interpolator` is
        the search path.
    pad_value : float
        The degree value to pad by for the dem interpolator. Only used if
        `dem_interpolator` is the search path.
    vertical_step_size : float|int
        Sampling along HAE altitude at the given resolution in meters. Bounds of
        `[0.1, 100]` will be enforced by replacement.
    use_structure_coa : bool
        If structure.coa_projection is populated, use that one **ignoring the COAProjection parameters.**
    coa_args
        keyword arguments for COAProjection.from_sicd class method.

    Returns
    -------
    numpy.ndarray
        Physical coordinates (in ECF coordinates) with corresponding to the input image
        coordinates, assuming detected features actually correspond to the DEM.
    """

    def append_grid_elements(this_lon_min, this_lon_max, the_list):
        lat_start = lat_min
        while lat_start < lat_max:
            lon_start = this_lon_min
            lat_end = min(lat_start + lat_grid_size, lat_max)
            while lon_start < this_lon_max:
                lon_end = min(lon_start + lon_grid_size, this_lon_max)
                the_list.append((lat_start, lat_end, lon_start, lon_end))
                lon_start = lon_end
            lat_start = lat_end

    # coa projection creation
    im_points, orig_shape = _validate_im_points(im_points)
    coa_proj = _get_coa_projection(structure, use_structure_coa, **coa_args)
    vertical_step_size = float(vertical_step_size)
    if vertical_step_size < 0.1:
        vertical_step_size = 0.1
    if vertical_step_size > 100:
        vertical_step_size = 100

    # reference point extraction
    ref_ecf = _get_reference_point(structure)
    ref_llh = ecf_to_geodetic(ref_ecf)
    ref_hae = ref_llh[2]
    # subgrid size definition
    lat_grid_size = 0.03
    lon_grid_size = min(10, lat_grid_size/numpy.sin(numpy.deg2rad(ref_llh[0])))

    # validate the dem_interpolator
    if dem_interpolator is None:
        raise ValueError('dem_interpolator is None, this is unhandled.')
    if isinstance(dem_interpolator, string_types):
        dted_list = DTEDList(dem_interpolator)
        dem_interpolator = DTEDInterpolator.from_reference_point(
            ref_llh, dted_list, dem_type=dem_type, geoid_file=geoid_file, pad_value=pad_value)
    if not isinstance(dem_interpolator, DEMInterpolator):
        raise TypeError('dem_interpolator is of unsupported type {}'.format(type(dem_interpolator)))

    # perform a projection to reference point hae for approximate lat/lon values
    im_points_view = numpy.reshape(im_points, (-1, 2))  # possibly or make 2-d flatten
    r_tgt_coa, r_dot_tgt_coa, time_coa, arp_coa, varp_coa = coa_proj.projection(im_points_view)
    ugpn = wgs_84_norm(ref_ecf)
    tolerance = 1e-3
    max_iterations = 10
    llh_rough = ecf_to_geodetic(_image_to_ground_hae_perform(
            r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa, ref_ecf, ugpn, ref_hae,
            tolerance, max_iterations, ref_hae))

    # segment into lat/lon grid of small size for more efficient dem lookup
    lat_min = numpy.min(llh_rough[:, 0])
    lat_max = numpy.max(llh_rough[:, 0])
    lon_min = numpy.min(llh_rough[:, 1])
    lon_max = numpy.max(llh_rough[:, 1])
    lat_lon_grids = []
    if (lon_min < -90) and (lon_max > 90):
        # there is a -180/180 crossing
        append_grid_elements(numpy.min(llh_rough[(llh_rough[:, 1] > 0), 1]), 180, lat_lon_grids)
        append_grid_elements(-180, numpy.max(llh_rough[(llh_rough[:, 1] < 0), 1]), lat_lon_grids)
    else:
        append_grid_elements(lon_min, lon_max, lat_lon_grids)

    if len(lat_lon_grids) == 1:
        return _image_to_ground_dem_block(
            im_points, coa_proj, dem_interpolator, vertical_step_size,
            lat_lon_grids[0], block_size, lat_grid_size, lon_grid_size)
    else:
        num_points = im_points_view.shape[0]
        coords = numpy.zeros((num_points, 3), dtype='float64')
        for entry in lat_lon_grids:
            mask = ((llh_rough[:, 0] >= entry[0]) & (llh_rough[:, 0] <= entry[1]) &
                    (llh_rough[:, 1] >= entry[2]) & (llh_rough[:, 1] <= entry[3]))
            if numpy.any(mask):
                coords[mask, :] = _image_to_ground_dem_block(
                    im_points[mask, :], coa_proj, dem_interpolator, vertical_step_size,
                    entry, block_size, lat_grid_size, lon_grid_size)
    if len(orig_shape) == 1:
        coords = numpy.reshape(coords, (-1,))
    elif len(orig_shape) > 1:
        coords = numpy.reshape(coords, orig_shape[:-1] + (3,))
    return coords
