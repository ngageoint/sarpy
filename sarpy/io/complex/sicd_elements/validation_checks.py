"""
The detailed and involved validity checks for the sicd structure.

Note: These checks were originally implemented in the SICD component objects,
but separating this implementation is probably less confusing in the long run.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import numpy
from scipy.constants import speed_of_light
from sarpy.geometry import geocoords
from sarpy.geometry.geometry_elements import LinearRing


##############
# RgAzComp image formation parameter checks

def _rgazcomp_check_kaz_poly(RgAzComp, Timeline, Grid, SCPCOA, look, ARP_Vel):
    """
    Check the KAZ polynomial value.

    Parameters
    ----------
    RgAzComp : sarpy.io.complex.sicd_elements.RgAzComp.RgAzCompType
    Timeline : sarpy.io.complex.sicd_elements.Timeline.TimelineType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType
    SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
    look : int
    ARP_Vel : numpy.ndarray

    Returns
    -------
    bool
    """

    cond = True
    if Timeline.IPP is not None and len(Timeline.IPP) == 1:
        try:
            st_rate_coa = Timeline.IPP[0].IPPPoly.derivative_eval(SCPCOA.SCPTime, der_order=1)
            if Grid.Row.DeltaKCOAPoly is not None:
                krg_coa = Grid.Row.KCtr + Grid.Row.DeltaKCOAPoly.get_array(dtype='float64')
            else:
                krg_coa = Grid.Row.KCtr
            delta_kaz_per_deltav = look * krg_coa * numpy.linalg.norm(ARP_Vel) * numpy.sin(
                numpy.deg2rad(SCPCOA.DopplerConeAng)) / \
                                   (SCPCOA.SlantRange * st_rate_coa)
            if isinstance(delta_kaz_per_deltav, numpy.ndarray):
                derived_kaz_poly = delta_kaz_per_deltav.dot(Timeline.IPP[0].IPPPoly.get_array(dtype='float64'))
            else:
                derived_kaz_poly = delta_kaz_per_deltav * Timeline.IPP[0].IPPPoly.get_array(dtype='float64')

            kaz_populated = RgAzComp.KazPoly.get_array(dtype='float64')
            if numpy.linalg.norm(kaz_populated - derived_kaz_poly) > 1e-3:
                RgAzComp.log_validity_error(
                    'Timeline.IPP has one element, the RgAzComp.KazPoly populated as\n\t{}\n\t'
                    'but the expected value is\n\t{}'.format(kaz_populated, derived_kaz_poly))
                cond = False
        except (AttributeError, ValueError, TypeError):
            pass
    return cond


def _rgazcomp_check_row_deltakcoa(RgAzComp, Grid, RadarCollection, ImageFormation):
    """

    Parameters
    ----------
    RgAzComp : sarpy.io.complex.sicd_elements.RgAzComp.RgAzCompType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType
    RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
    ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

    Returns
    -------
    bool
    """

    if Grid.Row.DeltaKCOAPoly is None:
        return True

    cond = True
    row_deltakcoa = Grid.Row.DeltaKCOAPoly.get_array(dtype='float64')
    if numpy.any(row_deltakcoa != row_deltakcoa[0, 0]):
        RgAzComp.log_validity_error(
            'Grid.Row.DeltaKCOAPoly is defined, '
            'but all entries are not constant\n\t{}'.format(row_deltakcoa))
        cond = False

    if RadarCollection.RefFreqIndex is None:
        try:
            fc_proc = ImageFormation.TxFrequencyProc.center_frequency
            k_f_c = fc_proc*2/speed_of_light
            if row_deltakcoa.shape == (1, 1):
                if abs(Grid.Row.KCtr - (k_f_c - row_deltakcoa[0, 0])) > 1e-6:
                    RgAzComp.log_validity_error(
                        'the Grid.Row.DeltaCOAPoly is scalar, '
                        'and not in agreement with Grid.Row.KCtr and center frequency')
                    cond = False
            else:
                if abs(Grid.Row.KCtr - k_f_c) > 1e-6:
                    RgAzComp.log_validity_error(
                        'the Grid.Row.DeltaCOAPoly is not scalar, '
                        'and Grid.Row.KCtr not in agreement with center frequency')
                    cond = False
        except (AttributeError, ValueError, TypeError):
            pass
    return cond


def _rgazcomp_check_col_deltacoa(RgAzComp, Grid):
    """

    Parameters
    ----------
    RgAzComp : sarpy.io.complex.sicd_elements.RgAzComp.RgAzCompType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType

    Returns
    -------
    bool
    """

    cond = True
    if Grid.Col.DeltaKCOAPoly is None:
        if Grid.Col.KCtr != 0:
            RgAzComp.log_validity_error(
                'the Grid.Col.DeltaKCOAPoly is not defined, '
                'and Grid.Col.KCtr is non-zero.')
            cond = False
    else:
        col_deltakcoa = Grid.Col.DeltaKCOAPoly.get_array(dtype='float64')
        if numpy.any(col_deltakcoa != col_deltakcoa[0, 0]):
            RgAzComp.log_validity_error(
                'the Grid.Col.DeltaKCOAPoly is defined, '
                'but all entries are not constant\n\t{}'.format(col_deltakcoa))
            cond = False

        if col_deltakcoa.shape == (1, 1) and abs(Grid.Col.KCtr + col_deltakcoa[0, 0]) > 1e-6:
            RgAzComp.log_validity_error(
                'the Grid.Col.DeltaCOAPoly is scalar, '
                'and not in agreement with Grid.Col.KCtr')
            cond = False
    return cond


def _rgazcomp_checks(the_sicd):
    """
    Perform the RgAzComp structure validation checks.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    RgAzComp = the_sicd.RgAzComp
    Grid = the_sicd.Grid
    GeoData = the_sicd.GeoData
    SCPCOA = the_sicd.SCPCOA
    cond = True

    if Grid.ImagePlane != 'SLANT':
        the_sicd.log_validity_error(
            'The image formation algorithm is RGAZCOMP,\n\t'
            'and Grid.ImagePlane is populated as "{}",\n\t'
            'but should be "SLANT"'.format(Grid.ImagePlane))
        cond = False
    if Grid.Type != 'RGAZIM':
        the_sicd.log_validity_error(
            'The image formation algorithm is RGAZCOMP, and Grid.Type is populated as "{}",\n\t'
            'but should be "RGAZIM"'.format(Grid.Type))
        cond = False

    try:
        SCP = GeoData.SCP.ECF.get_array(dtype='float64')
        row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
        col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')
        ARP_Pos = SCPCOA.ARPPos.get_array(dtype='float64')
        ARP_Vel = SCPCOA.ARPVel.get_array(dtype='float64')
        uRG = SCP - ARP_Pos
        uRG /= numpy.linalg.norm(uRG)
        left = numpy.cross(ARP_Pos / numpy.linalg.norm(ARP_Pos), ARP_Vel / numpy.linalg.norm(ARP_Vel))
        look = numpy.sign(numpy.dot(left, uRG))
        Spn = -look * numpy.cross(uRG, ARP_Vel)
        uSpn = Spn / numpy.linalg.norm(Spn)
        derived_col_vec = numpy.cross(uSpn, uRG)
    except (AttributeError, ValueError, TypeError):
        return cond

    derived_AzSF = -look * numpy.sin(numpy.deg2rad(SCPCOA.DopplerConeAng)) / SCPCOA.SlantRange
    if abs(RgAzComp.AzSF - derived_AzSF) > 1e-6:
        RgAzComp.log_validity_error(
            'AzSF is populated as {}, '
            'but expected value is {}'.format(RgAzComp.AzSF, derived_AzSF))
        cond = False

    if numpy.linalg.norm(uRG - row_uvect) > 1e-3:
        RgAzComp.log_validity_error(
            'Grid.Row.UVectECF is populated as \n\t{}\n\t'
            'which should agree with the unit range vector\n\t{}'.format(row_uvect, uRG))
        cond = False
    if numpy.linalg.norm(derived_col_vec - col_uvect) > 1e-3:
        RgAzComp.log_validity_error(
            'Grid.Col.UVectECF is populated as \n\t{}\n\t'
            'which should agree with derived vector\n\t{}'.format(col_uvect, derived_col_vec))
        cond = False

    cond &= _rgazcomp_check_kaz_poly(RgAzComp, the_sicd.Timeline, Grid, SCPCOA, look, ARP_Vel)
    cond &= _rgazcomp_check_row_deltakcoa(RgAzComp, Grid, the_sicd.RadarCollection, the_sicd.ImageFormation)
    cond &= _rgazcomp_check_col_deltacoa(RgAzComp, Grid)
    return cond


##############
# PFA image formation parameter checks

def _pfa_check_kaz_krg(PFA, Grid):
    """
    Check the validity of the Kaz and Krg values.

    Parameters
    ----------
    PFA : sarpy.io.complex.sicd_elements.PFA.PFAType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType

    Returns
    -------
    bool
    """

    cond = True
    if PFA.STDeskew is None or not PFA.STDeskew.Applied:
        try:
            if PFA.Kaz2 - Grid.Col.KCtr > 0.5/Grid.Col.SS + 1e-10:
                PFA.log_validity_error(
                    'PFA.Kaz2 - Grid.Col.KCtr ({}) > 0.5/Grid.Col.SS ({})'.format(
                        PFA.Kaz2 - Grid.Col.KCtr, 0.5/Grid.Col.SS))
                cond = False
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            if PFA.Kaz1 - Grid.Col.KCtr < -0.5/Grid.Col.SS - 1e-10:
                PFA.log_validity_error(
                    'PFA.Kaz1 - Grid.Col.KCtr ({}) < -0.5/Grid.Col.SS ({})'.format(
                        PFA.Kaz1 - Grid.Col.KCtr, -0.5/Grid.Col.SS))
                cond = False
        except (AttributeError, TypeError, ValueError):
            pass

    try:
        if PFA.Krg2 - Grid.Row.KCtr > 0.5/Grid.Row.SS + 1e-10:
            PFA.log_validity_error(
                'PFA.Krg2 - Grid.Row.KCtr ({}) > 0.5/Grid.Row.SS ({})'.format(
                    PFA.Krg2 - Grid.Row.KCtr, 0.5/Grid.Row.SS))
            cond = False
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        if PFA.Krg1 - Grid.Row.KCtr < -0.5/Grid.Row.SS - 1e-10:
            PFA.log_validity_error(
                'PFA.Krg1 - Grid.Row.KCtr ({}) < -0.5/Grid.Row.SS ({})'.format(
                    PFA.Krg1 - Grid.Row.KCtr, -0.5/Grid.Row.SS))
            cond = False
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        if Grid.Row.ImpRespBW > (PFA.Krg2 - PFA.Krg1) + 1e-10:
            PFA.log_validity_error(
                'Grid.Row.ImpRespBW ({}) > PFA.Krg2 - PFA.Krg1 ({})'.format(
                    Grid.Row.ImpRespBW, PFA.Krg2 - PFA.Krg1))
            cond = False
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        if abs(Grid.Col.KCtr) > 1e-5 and abs(Grid.Col.KCtr - 0.5*(PFA.Kaz2 +PFA.Kaz1)) > 1e-5:
            PFA.log_validity_error(
                'Grid.Col.KCtr ({}) not within 1e-5 of 0.5*(PFA.Kaz2 + PFA.Kaz1) ({})'.format(
                    Grid.Col.KCtr, 0.5*(PFA.Kaz2 + PFA.Kaz1)))
            cond = False
    except (AttributeError, TypeError, ValueError):
        pass

    return cond


def _pfa_check_polys(PFA, Position, Timeline, SCP):
    """

    Parameters
    ----------
    PFA : sarpy.io.complex.sicd_elements.PFA.PFAType
    Position : sarpy.io.complex.sicd_elements.Position.PositionType
    Timeline : sarpy.io.complex.sicd_elements.Timeline.TimelineType
    SCP : numpy.ndarray

    Returns
    -------
    bool
    """

    cond = True
    num_samples = max(PFA.PolarAngPoly.Coefs.size, 40)
    times = numpy.linspace(0, Timeline.CollectDuration, num_samples)

    k_a, k_sf = PFA.pfa_polar_coords(Position, SCP, times)
    if k_a is None:
        return True
    # check for agreement with k_a and k_sf derived from the polynomials
    k_a_derived = PFA.PolarAngPoly(times)
    k_sf_derived = PFA.SpatialFreqSFPoly(k_a)
    k_a_diff = numpy.amax(numpy.abs(k_a_derived - k_a))
    k_sf_diff = numpy.amax(numpy.abs(k_sf_derived - k_sf))
    if k_a_diff > 5e-3:
        PFA.log_validity_error(
            'the PolarAngPoly evaluated values do not agree with actual calculated values')
        cond = False
    if k_sf_diff > 5e-3:
        PFA.log_validity_error(
            'the SpatialFreqSFPoly evaluated values do not agree with actual calculated values')
        cond = False
    return cond


def _pfa_check_uvects(PFA, Position, Grid, SCP):
    """

    Parameters
    ----------
    PFA : sarpy.io.complex.sicd_elements.PFA.PFAType
    Position : sarpy.io.complex.sicd_elements.Position.PositionType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType
    SCP : numpy.ndarray

    Returns
    -------
    bool
    """

    if PFA.IPN is None or PFA.FPN is None:
        return True

    cond = True
    ipn = PFA.IPN.get_array(dtype='float64')
    fpn = PFA.FPN.get_array(dtype='float64')
    row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
    col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')

    pol_ref_point = Position.ARPPoly(PFA.PolarAngRefTime)
    offset = (SCP - pol_ref_point).dot(ipn)/(fpn.dot(ipn))
    ref_position_ipn = pol_ref_point + offset*fpn
    slant_range = ref_position_ipn - SCP
    u_slant_range = slant_range/numpy.linalg.norm(slant_range)
    derived_row_vector = -u_slant_range
    if numpy.linalg.norm(derived_row_vector - row_uvect) > 1e-3:
        PFA.log_validity_error(
            'the Grid.Row.UVectECF ({}) is not in good agreement with\n\t'
            'the expected value derived from PFA parameters ({})'.format(row_uvect, derived_row_vector))
        cond = False

    derived_col_vector = numpy.cross(ipn, derived_row_vector)
    if numpy.linalg.norm(derived_col_vector - col_uvect) > 1e-3:
        PFA.log_validity_error(
            'the Grid.Col.UVectECF ({}) is not in good agreement with\n\t'
            'the expected value derived from the PFA parameters ({})'.format(col_uvect, derived_col_vector))
        cond = False

    return cond


def _pfa_check_stdeskew(PFA, Grid):
    """

    Parameters
    ----------
    PFA : sarpy.io.complex.sicd_elements.PFA.PFAType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType

    Returns
    -------
    bool
    """

    if PFA.STDeskew is None or not PFA.STDeskew.Applied:
        return True
    cond = True
    if Grid.TimeCOAPoly is not None:
        timecoa_poly = Grid.TimeCOAPoly.get_array(dtype='float64')
        if timecoa_poly.shape == (1, 1) or numpy.all(timecoa_poly.flatten()[1:] < 1e-6):
            PFA.log_validity_error(
                'PFA.STDeskew.Applied is True, and the Grid.TimeCOAPoly is essentially constant.')
            cond = False

    # the Row DeltaKCOAPoly and STDSPhasePoly should be essentially identical
    if Grid.Row is not None and Grid.Row.DeltaKCOAPoly is not None and \
            PFA.STDeskew.STDSPhasePoly is not None:
        stds_phase_poly = PFA.STDeskew.STDSPhasePoly.get_array(dtype='float64')
        delta_kcoa = Grid.Row.DeltaKCOAPoly.get_array(dtype='float64')
        rows = max(stds_phase_poly.shape[0], delta_kcoa.shape[0])
        cols = max(stds_phase_poly.shape[1], delta_kcoa.shape[1])
        exp_stds_phase_poly = numpy.zeros((rows, cols), dtype='float64')
        exp_delta_kcoa = numpy.zeros((rows, cols), dtype='float64')
        exp_stds_phase_poly[:stds_phase_poly.shape[0], :stds_phase_poly.shape[1]] = stds_phase_poly
        exp_delta_kcoa[:delta_kcoa.shape[0], :delta_kcoa.shape[1]] = delta_kcoa

        if numpy.max(numpy.abs(exp_delta_kcoa - exp_stds_phase_poly)) > 1e-6:
            PFA.log_validity_warning(
                'PFA.STDeskew.Applied is True,\n\t'
                'and the Grid.Row.DeltaKCOAPoly ({}) and PFA.STDeskew.STDSPhasePoly ({})\n\t'
                'are not in good agreement.'.format(delta_kcoa, stds_phase_poly))
            cond = False

    return cond


def _pfa_check_kctr(PFA, RadarCollection, ImageFormation, Grid):
    """

    Parameters
    ----------
    PFA : sarpy.io.complex.sicd_elements.PFA.PFAType
    RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
    ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType

    Returns
    -------
    bool
    """

    if RadarCollection.RefFreqIndex is not None:
        return True

    cond = True
    try:
        center_freq = ImageFormation.TxFrequencyProc.center_frequency
        kap_ctr = center_freq*PFA.SpatialFreqSFPoly.Coefs[0]*2/speed_of_light
        theta = numpy.arctan(0.5*Grid.Col.ImpRespBW/Grid.Row.KCtr)  # aperture angle
        kctr_total = max(1e-2, 1 - numpy.cos(theta))  # difference between Krg and Kap
        if abs(Grid.Row.KCtr/kap_ctr - 1) > kctr_total:
            PFA.log_validity_error(
                'the Grid.Row.KCtr value ({}) is not in keeping with\n\t'
                'the expected derived from PFA parameters ({})'.format(Grid.Row.KCtr, kap_ctr))
            cond = False
    except (AttributeError, ValueError, TypeError):
        pass

    return cond


def _pfa_check_image_plane(PFA, Grid, SCPCOA, SCP):
    """

    Parameters
    ----------
    PFA : sarpy.io.complex.sicd_elements.PFA.PFAType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType
    SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
    SCP : numpy.ndarray

    Returns
    -------
    bool
    """

    if PFA.IPN is None or PFA.FPN is None:
        return True

    cond = True
    ipn = PFA.IPN.get_array(dtype='float64')
    fpn = PFA.FPN.get_array(dtype='float64')
    ETP = geocoords.wgs_84_norm(SCP)

    if Grid.ImagePlane == 'SLANT':
        try:
            ARP_Pos = SCPCOA.ARPPos.get_array(dtype='float64')
            ARP_Vel = SCPCOA.ARPVel.get_array(dtype='float64')
            uRG = SCP - ARP_Pos
            uRG /= numpy.linalg.norm(uRG)
            left = numpy.cross(ARP_Pos/numpy.linalg.norm(ARP_Pos), ARP_Vel/numpy.linalg.norm(ARP_Vel))
            look = numpy.sign(numpy.dot(left, uRG))
            Spn = -look*numpy.cross(uRG, ARP_Vel)
            uSpn = Spn/numpy.linalg.norm(Spn)

            if numpy.arccos(ipn.dot(uSpn)) > numpy.deg2rad(1):
                PFA.log_validity_error(
                    'the Grid.ImagePlane is "SLANT",\n\t'
                    'but COA slant plane and provided IPN are not within one degree of each other')
                cond = False
        except (AttributeError, ValueError, TypeError):
            pass
    elif Grid.ImagePlane == 'GROUND':
        if numpy.arccos(ipn.dot(ETP)) > numpy.deg2rad(3):
            PFA.log_validity_error(
                'the Grid.ImagePlane is "Ground",\n\t'
                'but the Earth Tangent Plane at SCP and provided IPN\n\t'
                'are not within three degrees of each other.')
            cond = False

    # verify that fpn points outwards
    if fpn.dot(SCP) < 0:
        PFA.log_validity_error(
            'the focus plane unit normal does not appear to be outward pointing')
        cond = False
    # check agreement between focus plane and ground plane
    if numpy.arccos(fpn.dot(ETP)) > numpy.deg2rad(3):
        PFA.log_validity_warning(
            'the focus plane unit normal is not within three degrees of the earth Tangent Plane')
    return cond


def _pfa_check_polar_angle_consistency(PFA, CollectionInfo, ImageFormation):
    """

    Parameters
    ----------
    PFA : sarpy.io.complex.sicd_elements.PFA.PFAType
    CollectionInfo : sarpy.io.complex.sicd_elements.CollectionInfo.CollectionInfoType
    ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

    Returns
    -------
    bool
    """

    if CollectionInfo.RadarMode is None or CollectionInfo.RadarMode.ModeType != 'SPOTLIGHT':
        return True

    cond = True
    if PFA.Kaz1 is not None and PFA.Kaz2 is not None and PFA.Krg1 is not None:
        polar_angle_bounds = numpy.sort(PFA.PolarAngPoly(numpy.array([ImageFormation.TStartProc, ImageFormation.TEndProc], dtype='float64')))
        derived_pol_angle_bounds = numpy.arctan(numpy.array([PFA.Kaz1, PFA.Kaz2], dtype='float64')/PFA.Krg1)
        pol_angle_bounds_diff = numpy.rad2deg(numpy.amax(numpy.abs(polar_angle_bounds - derived_pol_angle_bounds)))
        if pol_angle_bounds_diff > 1e-2:
            PFA.log_validity_warning(
                'the derived polar angle bounds ({})\n\t'
                'are not consistent with the provided ImageFormation processing times\n\t'
                '(expected bounds {}).'.format(polar_angle_bounds, derived_pol_angle_bounds))
            cond = False
    return cond


def _pfa_checks(the_sicd):
    """
    Perform the PFA structure validation checks.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    PFA = the_sicd.PFA
    Grid = the_sicd.Grid
    SCPCOA = the_sicd.SCPCOA

    cond = True
    if Grid.Type != 'RGAZIM':
        Grid.log_validity_warning(
            'The image formation algorithm is PFA,\n\t'
            'and Grid.Type is populated as "{}",\n\t'
            'but should be "RGAZIM"'.format(Grid.Type))
        cond = False
    if abs(PFA.PolarAngRefTime - SCPCOA.SCPTime) > 1e-6:
        PFA.log_validity_warning(
            'the PFA.PolarAngRefTime ({})\n\t'
            'does not agree with the SCPCOA.SCPTime ({})'.format(PFA.PolarAngRefTime, SCPCOA.SCPTime))
        cond = False
    cond &= _pfa_check_kaz_krg(PFA, Grid)
    cond &= _pfa_check_stdeskew(PFA, Grid)
    cond &= _pfa_check_kctr(PFA, the_sicd.RadarCollection, the_sicd.ImageFormation, Grid)

    try:
        SCP = the_sicd.GeoData.SCP.ECF.get_array(dtype='float64')
    except (AttributeError, ValueError, TypeError):
        return cond

    cond &= _pfa_check_polys(PFA, the_sicd.Position, the_sicd.Timeline, SCP)
    cond &= _pfa_check_uvects(PFA, the_sicd.Position, Grid, SCP)
    cond &= _pfa_check_image_plane(PFA, Grid, SCPCOA, SCP)
    cond &= _pfa_check_polar_angle_consistency(PFA, the_sicd.CollectionInfo, the_sicd.ImageFormation)
    return cond


##############
# PFA image formation parameter checks

def _rma_check_rmat(RMA, Grid, GeoData, RadarCollection, ImageFormation):
    """

    Parameters
    ----------
    RMA : sarpy.io.complex.sicd_elements.RMA.RMAType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType
    GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
    RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
    ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

    Returns
    -------
    bool
    """

    cond = True
    RMAT = RMA.RMAT

    if Grid.Type != 'XCTYAT':
        Grid.log_validity_error(
            'The image formation algorithm is RMA/RMAT, which should yield '
            'Grid.Type == "XCTYAT", but Grid.Type is populated as "{}"'.format(Grid.Type))
        cond = False

    try:
        SCP = GeoData.SCP.ECF.get_array(dtype='float64')
        row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
        col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')
        position_ref = RMAT.PosRef.get_array(dtype='float64')
        velocity_ref = RMAT.VelRef.get_array(dtype='float64')
        LOS = (SCP - position_ref)
        uLOS = LOS/numpy.linalg.norm(LOS)
        left = numpy.cross(
            position_ref/numpy.linalg.norm(position_ref),
            velocity_ref/numpy.linalg.norm(velocity_ref))
        look = numpy.sign(left.dot(uLOS))
        uYAT = -look*velocity_ref/numpy.linalg.norm(velocity_ref)
        uSPN = numpy.cross(uLOS, uYAT)
        uSPN /= numpy.linalg.norm(uSPN)
        uXCT = numpy.cross(uYAT, uSPN)
    except (AttributeError, ValueError, TypeError):
        return cond

    if numpy.linalg.norm(row_uvect - uXCT) > 1e-3:
        RMAT.log_validity_error(
            'the Grid.Row.UVectECF is populated as {},\n\t'
            'but derived from the RMAT parameters is expected to be {}'.format(row_uvect, uXCT))
        cond = False
    if numpy.linalg.norm(col_uvect - uYAT) > 1e-3:
        RMAT.log_validity_error(
            'the Grid.Col.UVectECF is populated as {},\n\t'
            'but derived from the RMAT parameters is expected to be {}'.format(col_uvect, uYAT))
        cond = False
    exp_doppler_cone = numpy.rad2deg(numpy.arccos(uLOS.dot(velocity_ref/numpy.linalg.norm(velocity_ref))))
    if abs(exp_doppler_cone - RMAT.DopConeAngRef) > 1e-6:
        RMAT.log_validity_error(
            'the RMAT.DopConeAngRef is populated as {},\n\t'
            'but derived from the RMAT parameters is expected to be {}'.format(RMAT.DopConeAngRef, exp_doppler_cone))
        cond = False

    if RadarCollection.RefFreqIndex is None:
        center_freq = ImageFormation.TxFrequencyProc.center_frequency
        k_f_c = center_freq*2/speed_of_light
        exp_row_kctr = k_f_c*numpy.sin(numpy.deg2rad(RMAT.DopConeAngRef))
        exp_col_kctr = k_f_c*numpy.cos(numpy.deg2rad(RMAT.DopConeAngRef))
        try:
            if abs(exp_row_kctr/Grid.Row.KCtr - 1) > 1e-3:
                RMAT.log_validity_warning(
                    'the Grid.Row.KCtr is populated as {},\n\t'
                    'and derived from the RMAT parameters is expected to be {}'.format(Grid.Row.KCtr, exp_row_kctr))
                cond = False
            if abs(exp_col_kctr/Grid.Col.KCtr - 1) > 1e-3:
                RMAT.log_validity_warning(
                    'the Grid.Col.KCtr is populated as {},\n\t'
                    'and derived from the RMAT parameters is expected to be {}'.format(Grid.Col.KCtr, exp_col_kctr))
                cond = False
        except (AttributeError, ValueError, TypeError):
            pass
    return cond


def _rma_check_rmcr(RMA, Grid, GeoData, RadarCollection, ImageFormation):
    """

    Parameters
    ----------
    RMA : sarpy.io.complex.sicd_elements.RMA.RMAType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType
    GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
    RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
    ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

    Returns
    -------
    bool
    """

    cond = True
    RMCR = RMA.RMCR

    if Grid.Type != 'XRGYCR':
        Grid.log_validity_error(
            'The image formation algorithm is RMA/RMCR, which should yield '
            'Grid.Type == "XRGYCR", but Grid.Type is populated as "{}"'.format(Grid.Type))
        cond = False

    try:
        SCP = GeoData.SCP.ECF.get_array(dtype='float64')
        row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
        col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')
        position_ref = RMCR.PosRef.get_array(dtype='float64')
        velocity_ref = RMCR.VelRef.get_array(dtype='float64')
        uXRG = SCP - position_ref
        uXRG /= numpy.linalg.norm(uXRG)
        left = numpy.cross(
            position_ref/numpy.linalg.norm(position_ref),
            velocity_ref/numpy.linalg.norm(velocity_ref))
        look = numpy.sign(left.dot(uXRG))
        uSPN = look*numpy.cross(velocity_ref/numpy.linalg.norm(velocity_ref), uXRG)
        uSPN /= numpy.linalg.norm(uSPN)
        uYCR = numpy.cross(uSPN, uXRG)
    except (AttributeError, ValueError, TypeError):
        return cond

    if numpy.linalg.norm(row_uvect - uXRG) > 1e-3:
        RMCR.log_validity_error(
            'the Grid.Row.UVectECF is populated as {},\n\t'
            'but derived from the RMCR parameters is expected to be {}'.format(row_uvect, uXRG))
        cond = False
    if numpy.linalg.norm(col_uvect - uYCR) > 1e-3:
        RMCR.log_validity_error(
            'the Grid.Col.UVectECF is populated as {},\n\t'
            'but derived from the RMCR parameters is expected to be {}'.format(col_uvect, uYCR))
        cond = False
    exp_doppler_cone = numpy.rad2deg(numpy.arccos(uXRG.dot(velocity_ref/numpy.linalg.norm(velocity_ref))))
    if abs(exp_doppler_cone - RMCR.DopConeAngRef) > 1e-6:
        RMCR.log_validity_error(
            'the RMCR.DopConeAngRef is populated as {},\n\t'
            'but derived from the RMCR parameters is expected to be {}'.format(RMCR.DopConeAngRef, exp_doppler_cone))
        cond = False
    if abs(Grid.Col.KCtr) > 1e-6:
        Grid.log_validity_error(
            'The image formation algorithm is RMA/RMCR,\n\t'
            'but Grid.Col.KCtr is non-zero ({}).'.format(Grid.Col.KCtr))
        cond = False

    if RadarCollection.RefFreqIndex is None:
        center_freq = ImageFormation.TxFrequencyProc.center_frequency
        k_f_c = center_freq*2/speed_of_light
        try:
            if abs(k_f_c/Grid.Row.KCtr - 1) > 1e-3:
                RMCR.log_validity_warning(
                    'the Grid.Row.KCtr is populated as {},\n\t'
                    'and derived from the RMCR parameters is expected to be {}'.format(Grid.Row.KCtr, k_f_c))
                cond = False
        except (AttributeError, ValueError, TypeError):
            pass
    return cond


def _rma_check_inca(RMA, Grid, GeoData, RadarCollection, CollectionInfo, Position):
    """

    Parameters
    ----------
    RMA : sarpy.io.complex.sicd_elements.RMA.RMAType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType
    GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
    RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
    CollectionInfo : sarpy.io.complex.sicd_elements.CollectionInfo.CollectionInfoType
    Position :  sarpy.io.complex.sicd_elements.Position.PositionType

    Returns
    -------
    bool
    """

    cond = True
    INCA = RMA.INCA

    if Grid.Type != 'RGZERO':
        Grid.log_validity_warning(
            'The image formation algorithm is RMA/INCA, which should yield '
            'Grid.Type == "RGZERO", but Grid.Type is populated as "{}"'.format(Grid.Type))
        cond = False

    if CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT':
        if INCA.DopCentroidPoly is not None:
            INCA.log_validity_error(
                'the CollectionInfo.RadarMode.ModeType == "SPOTLIGHT",\n\t'
                'and INCA.DopCentroidPoly is populated.')
            cond = False
        if INCA.DopCentroidCOA is True:
            INCA.log_validity_error(
                'the CollectionInfo.RadarMode.ModeType == "SPOTLIGHT",\n\t'
                'and INCA.DopCentroidCOA == True.')
            cond = False
    else:
        if INCA.DopCentroidPoly is None:
            INCA.log_validity_error(
                'the CollectionInfo.RadarMode.ModeType == "{}",\n\t'
                'and INCA.DopCentroidPoly is not populated.'.format(CollectionInfo.RadarMode.ModeType))
            cond = False
        if INCA.DopCentroidCOA is not True:
            INCA.log_validity_error(
                'the CollectionInfo.RadarMode.ModeType == "{}",\n\t'
                'and INCA.DopCentroidCOA is not True.'.format(CollectionInfo.RadarMode.ModeType))
            cond = False
        if Grid.Col.DeltaKCOAPoly is not None and INCA.DopCentroidPoly is not None:
            col_deltakcoa = Grid.Col.DeltaKCOAPoly.get_array(dtype='float64')
            dop_centroid = INCA.DopCentroidPoly.get_array(dtype='float64')
            rows = max(col_deltakcoa.shape[0], dop_centroid.shape[0])
            cols = max(col_deltakcoa.shape[1], dop_centroid.shape[1])
            exp_deltakcoa1 = numpy.zeros((rows, cols), dtype='float64')
            exp_deltakcoa2 = numpy.zeros((rows, cols), dtype='float64')
            exp_deltakcoa1[:col_deltakcoa.shape[0], :col_deltakcoa.shape[1]] = col_deltakcoa
            exp_deltakcoa2[:dop_centroid.shape[0], :dop_centroid.shape[1]] = dop_centroid*INCA.TimeCAPoly[1]
            if numpy.max(numpy.abs(exp_deltakcoa1 - exp_deltakcoa2)) > 1e-6:
                INCA.log_validity_error(
                    'the Grid.Col.DeltaKCOAPoly ({}),\n\t'
                    'INCA.DopCentroidPoly ({}), and INCA.TimeCAPoly ({}) '
                    'are inconsistent.'.format(col_deltakcoa,
                                               dop_centroid,
                                               INCA.TimeCAPoly.get_array(dtype='float64')))
                cond = False

    center_freq = RadarCollection.TxFrequency.center_frequency
    if abs(center_freq/INCA.FreqZero - 1) > 1e-5:
        INCA.log_validity_error(
            'the INCA.FreqZero ({}) should typically agree with center '
            'transmit frequency ({})'.format(INCA.FreqZero, center_freq))
        cond = False

    if abs(Grid.Col.KCtr) > 1e-8:
        Grid.log_validity_error(
            'The image formation algorithm is RMA/INCA, but Grid.Col.KCtr is '
            'non-zero ({})'.format(Grid.Col.KCtr))
        cond = False

    if RadarCollection.RefFreqIndex is None:
        exp_row_kctr = INCA.FreqZero*2/speed_of_light
        if abs(exp_row_kctr/Grid.Row.KCtr - 1) > 1e-8:
            INCA.log_validity_error(
                'the Grid.Row.KCtr is populated as ({}),\n\t'
                'which is not consistent with INCA.FreqZero ({})'.format(Grid.Row.KCtr, INCA.FreqZero))
            cond = False

    try:
        SCP = GeoData.SCP.ECF.get_array(dtype='float64')
        row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
        col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')
        scp_time = INCA.TimeCAPoly[0]
        ca_pos = Position.ARPPoly(scp_time)
        ca_vel = Position.ARPPoly.derivative_eval(scp_time, der_order=1)
        RG = SCP - ca_pos
        uRG = RG/numpy.linalg.norm(RG)
        left = numpy.cross(ca_pos/numpy.linalg.norm(ca_pos), ca_vel/numpy.linalg.norm(ca_vel))
        look = numpy.sign(left.dot(uRG))
        uSPN = -look*numpy.cross(uRG, ca_vel)
        uSPN /= numpy.linalg.norm(uSPN)
        uAZ = numpy.cross(uSPN, uRG)
    except (AttributeError, ValueError, TypeError):
        return cond

    if numpy.linalg.norm(row_uvect - uRG) > 1e-3:
        INCA.log_validity_error(
            'the Grid.Row.UVectECF is populated as {},\n\t'
            'but derived from the INCA parameters is expected to be {}'.format(row_uvect, uRG))
        cond = False
    if numpy.linalg.norm(col_uvect - uAZ) > 1e-3:
        INCA.log_validity_error(
            'the Col.UVectECF is populated as {},\n\t'
            'but derived from the INCA parameters is expected to be {}'.format(col_uvect, uAZ))
        cond = False

    exp_R_CA_SCP = numpy.linalg.norm(RG)
    if abs(exp_R_CA_SCP - INCA.R_CA_SCP) > 1e-2:
        INCA.log_validity_error(
            'the INCA.R_CA_SCP is populated as {},\n\t'
            'but derived from the INCA parameters is expected to be {}'.format(INCA.R_CA_SCP, exp_R_CA_SCP))
        cond = False

    drate_const = INCA.DRateSFPoly[0, 0]
    exp_drate_const = 1./abs(INCA.TimeCAPoly[1]*numpy.linalg.norm(ca_vel))
    if abs(exp_drate_const - drate_const) > 1e-3:
        INCA.log_validity_error(
            'the populated INCA.DRateSFPoly constant term ({})\n\t'
            'and expected constant term ({}) are not consistent.'.format(drate_const, exp_drate_const))
        cond = False
    return cond


def _rma_checks(the_sicd):
    """
    Perform the RMA structure validation checks.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    RMA = the_sicd.RMA

    if RMA.ImageType == 'RMAT':
        return _rma_check_rmat(RMA, the_sicd.Grid, the_sicd.GeoData, the_sicd.RadarCollection, the_sicd.ImageFormation)
    elif RMA.ImageType == 'RMCR':
        return _rma_check_rmcr(RMA, the_sicd.Grid, the_sicd.GeoData, the_sicd.RadarCollection, the_sicd.ImageFormation)
    elif RMA.ImageType == 'INCA':
        return _rma_check_inca(RMA, the_sicd.Grid, the_sicd.GeoData, the_sicd.RadarCollection, the_sicd.CollectionInfo, the_sicd.Position)
    return True


##############
# SICD checks

def _validate_scp_time(the_sicd):
    """
    Validate the SCPTime.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.SCPCOA is None or the_sicd.SCPCOA.SCPTime is None or \
            the_sicd.Grid is None or the_sicd.Grid.TimeCOAPoly is None:
        return True

    cond = True
    val1 = the_sicd.SCPCOA.SCPTime
    val2 = the_sicd.Grid.TimeCOAPoly[0, 0]
    if abs(val1 - val2) > 1e-6:
        the_sicd.log_validity_error(
            'SCPTime populated as {},\n\t'
            'and constant term of TimeCOAPoly populated as {}'.format(val1, val2))
        cond = False
    return cond


def _validate_image_form_parameters(the_sicd, alg_type):
    """
    Validate the image formation parameter specifics.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    alg_type : str

    Returns
    -------
    bool
    """

    cond = True
    if the_sicd.ImageFormation.ImageFormAlgo is None:
        the_sicd.log_validity_warning(
            'Image formation algorithm(s) `{}` populated,\n\t'
            'but ImageFormation.ImageFormAlgo was not set.\n\t'
            'ImageFormation.ImageFormAlgo has been set HERE,\n\t'
            'but the incoming structure was incorrect.'.format(alg_type))
        the_sicd.ImageFormation.ImageFormAlgo = alg_type.upper()
        cond = False
    elif the_sicd.ImageFormation.ImageFormAlgo == 'OTHER':
        the_sicd.log_validity_warning(
            'Image formation algorithm `{0:s}` populated,\n\t'
            'but ImageFormation.ImageFormAlgo populated as `OTHER`.\n\t'
            'Possibly non-applicable validity checks will '
            'be performed using the `{0:s}` object'.format(alg_type))
    elif the_sicd.ImageFormation.ImageFormAlgo != alg_type:
        the_sicd.log_validity_warning(
            'Image formation algorithm {} populated,\n\t'
            'but ImageFormation.ImageFormAlgo populated as {}.\n\t'
            'ImageFormation.ImageFormAlgo has been set properly HERE,\n\t'
            'but the incoming structure was incorrect.'.format(alg_type, the_sicd.ImageFormation.ImageFormAlgo))
        the_sicd.ImageFormation.ImageFormAlgo = alg_type.upper()
        cond = False

    # verify that the referenced received channels are consistent with radar collection
    if the_sicd.ImageFormation.RcvChanProc is not None:
        channels = the_sicd.ImageFormation.RcvChanProc.ChanIndices
        if channels is None or len(channels) < 1:
            the_sicd.ImageFormation.RcvChanProc.log_validity_error('No ChanIndex values populated')
            cond = False
        else:
            rcv_channels = the_sicd.RadarCollection.RcvChannels
            if rcv_channels is None:
                the_sicd.ImageFormation.RcvChanProc.log_validity_error(
                    'Some ChanIndex values are populated,\n\t'
                    'but no RadarCollection.RcvChannels is populated.')
                cond = False
            else:
                for i, entry in enumerate(channels):
                    if not (0 < entry <= len(rcv_channels)):
                        the_sicd.ImageFormation.RcvChanProc.log_validity_error(
                            'ChanIndex entry {} is populated as {},\n\tbut must be in '
                            'the range [1, {}]'.format(i, entry, len(rcv_channels)))
                        cond = False
    if the_sicd.Grid is None:
        return cond

    if alg_type == 'RgAzComp':
        cond &= _rgazcomp_checks(the_sicd)
    elif alg_type == 'PFA':
        cond &= _pfa_checks(the_sicd)
    elif alg_type == 'RMA':
        cond &= _rma_checks(the_sicd)
    elif the_sicd.ImageFormation.ImageFormAlgo == 'OTHER':
        the_sicd.log_validity_warning(
            'Image formation algorithm populated as "OTHER", which inherently limits SICD analysis capability')
        cond = False
    return cond


def _validate_image_formation(the_sicd):
    """
    Validate the image formation.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.ImageFormation is None:
        the_sicd.log_validity_error(
            'ImageFormation attribute is not populated, and ImageFormType is {}. This '
            'cannot be valid.'.format(the_sicd.ImageFormType))
        return False  # nothing more to be done.

    alg_types = []
    for alg in ['RgAzComp', 'PFA', 'RMA']:
        if getattr(the_sicd, alg) is not None:
            alg_types.append(alg)

    if len(alg_types) > 1:
        the_sicd.log_validity_error(
            'ImageFormation.ImageFormAlgo is set as {}, and multiple SICD image formation parameters {} are set.\n\t'
            'Only one image formation algorithm should be set, and ImageFormation.ImageFormAlgo '
            'should match.'.format(the_sicd.ImageFormation.ImageFormAlgo, alg_types))
        return False
    elif len(alg_types) == 0:
        if the_sicd.ImageFormation.ImageFormAlgo is None:
            the_sicd.log_validity_warning(
                'ImageFormation.ImageFormAlgo is not set, and there is no corresponding\n\t'
                'RgAzComp, PFA, or RMA SICD parameters set. Setting ImageFormAlgo to "OTHER".')
            the_sicd.ImageFormation.ImageFormAlgo = 'OTHER'
        elif the_sicd.ImageFormation.ImageFormAlgo == 'OTHER':
            the_sicd.log_validity_warning(
                'Image formation algorithm populated as "OTHER", which inherently limits SICD analysis capability')
        else:
            the_sicd.log_validity_error(
                'No RgAzComp, PFA, or RMA SICD parameters populated, but ImageFormation.ImageFormAlgo '
                'is set as {}.'.format(the_sicd.ImageFormation.ImageFormAlgo))
            return False
        return True
    # there is exactly one algorithm type populated
    return _validate_image_form_parameters(the_sicd, alg_types[0])


def _validate_image_segment_id(the_sicd):
    """
    Validate the image segment id.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.ImageFormation is None or the_sicd.RadarCollection is None:
        return False

    # get the segment identifier
    seg_id = the_sicd.ImageFormation.SegmentIdentifier
    # get the segment list
    try:
        seg_list = the_sicd.RadarCollection.Area.Plane.SegmentList
    except AttributeError:
        seg_list = None

    if seg_id is None:
        if seg_list is not None:
            the_sicd.log_validity_error(
                'ImageFormation.SegmentIdentifier is not populated, but\n\t'
                'RadarCollection.Area.Plane.SegmentList is populated.\n\t'
                'ImageFormation.SegmentIdentifier should be set to identify the appropriate segment.')
            return False
        return True

    if seg_list is None:
        the_sicd.log_validity_error(
            'ImageFormation.SegmentIdentifier is populated as {},\n\t'
            'but RadarCollection.Area.Plane.SegmentList is not populated.'.format(seg_id))
        return False

    # let's double check that seg_id is sensibly populated
    the_ids = [entry.Identifier for entry in seg_list]
    if seg_id not in the_ids:
        the_sicd.log_validity_error(
            'ImageFormation.SegmentIdentifier is populated as {},\n\t'
            'but this is not one of the possible identifiers in the\n\t'
            'RadarCollection.Area.Plane.SegmentList definition {}.\n\t'
            'ImageFormation.SegmentIdentifier should be set to identify the '
            'appropriate segment.'.format(seg_id, the_ids))
        return False
    return True


def _validate_spotlight_mode(the_sicd):
    """
    Validate the spotlight mode situation.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.CollectionInfo is None or the_sicd.CollectionInfo.RadarMode is None \
            or the_sicd.CollectionInfo.RadarMode.ModeType is None:
        return True

    if the_sicd.Grid is None or the_sicd.Grid.TimeCOAPoly is None:
        return True

    if the_sicd.CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT' and \
            the_sicd.Grid.TimeCOAPoly.Coefs.shape != (1, 1):
        the_sicd.log_validity_error(
            'CollectionInfo.RadarMode.ModeType is SPOTLIGHT,\n\t'
            'but the Grid.TimeCOAPoly is not scalar - {}.\n\t'
            'This cannot be valid.'.format(the_sicd.Grid.TimeCOAPoly.Coefs))
        return False
    elif the_sicd.Grid.TimeCOAPoly.Coefs.shape == (1, 1) and \
            the_sicd.CollectionInfo.RadarMode.ModeType != 'SPOTLIGHT':
        the_sicd.log_validity_warning(
            'The Grid.TimeCOAPoly is scalar,\n\t'
            'but the CollectionInfo.RadarMode.ModeType is not SPOTLIGHT - {}.\n\t'
            'This is likely not valid.'.format(the_sicd.CollectionInfo.RadarMode.ModeType))
        return True
    return True


def _validate_valid_data(the_sicd):
    """
    Check that either both ValidData fields are populated, or neither.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.ImageData is None or the_sicd.GeoData is None:
        return True

    if the_sicd.ImageData.ValidData is not None and the_sicd.GeoData.ValidData is None:
        the_sicd.log_validity_error('ValidData is populated in ImageData, but not GeoData')
        return False

    if the_sicd.GeoData.ValidData is not None and the_sicd.ImageData.ValidData is None:
        the_sicd.log_validity_error('ValidData is populated in GeoData, but not ImageData')
        return False

    return True


def _validate_polygons(the_sicd):
    """
    Checks that the polygons appear to be appropriate.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    def orientation(linear_ring, the_name):
        area = linear_ring.get_area()
        if area == 0:
            the_sicd.GeoData.log_validity_error(
                '{} encloses no area.\n\t'
                '**disregard if crosses the +/-180 boundary'.format(the_name))
            return False
        elif area < 0:
            the_sicd.GeoData.log_validity_error(
                "{} must be traversed in clockwise direction.\n\t"
                "**disregard if crosses the +/-180 boundary".format(the_name))
            return False
        return True

    if the_sicd.GeoData is None:
        return True

    if the_sicd.GeoData.ImageCorners is None:
        return True  # checked elsewhere

    image_corners = the_sicd.GeoData.ImageCorners.get_array(dtype='float64')

    if numpy.any(~numpy.isfinite(image_corners)):
        the_sicd.GeoData.log_validity_error('ImageCorners populated with some infinite or NaN values')
        return False

    value = True
    lin_ring = LinearRing(coordinates=image_corners)
    value &= orientation(lin_ring, 'ImageCorners')

    if the_sicd.GeoData.ValidData is not None:
        valid_data = the_sicd.GeoData.ValidData.get_array(dtype='float64')
        if numpy.any(~numpy.isfinite(valid_data)):
            the_sicd.GeoData.log_validity_error(
                'ValidData populated with some infinite or NaN values')
            value = False
        else:
            value &= orientation(LinearRing(coordinates=valid_data), 'ValidData')
            for i, entry in enumerate(valid_data):
                contained = lin_ring.contain_coordinates(entry[0], entry[1])
                close = (lin_ring.get_minimum_distance(entry[:2]) < 1e-7)
                if not (contained or close):
                    the_sicd.GeoData.log_validity_error(
                        'ValidData entry {} is not contained ImageCorners.\n\t'
                        '**disregard if crosses the +/-180 boundary'.format(i))
                    value = False

    if not the_sicd.can_project_coordinates():
        the_sicd.log_validity_warning(
            'This sicd does not permit coordinate projection,\n\t'
            'and image corner points can not be evaluated')
        return False

    origin_loc = the_sicd.project_image_to_ground_geo([0, 0])
    if numpy.abs(origin_loc[0] - image_corners[0, 0]) > 1e-3 or numpy.abs(origin_loc[1] - image_corners[0, 1]) > 1e-3:
        the_sicd.GeoData.log_validity_error(
            'The pixel coordinate [0, 0] projects to {},\n\t'
            'which is not in good agreement with the first corner point {}'.format(origin_loc, image_corners[0]))
        value = False
    return value


def _validate_polarization(the_sicd):
    """
    Validate the polarization.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.ImageFormation is None or the_sicd.ImageFormation.TxRcvPolarizationProc is None:
        return True

    if the_sicd.RadarCollection is None or the_sicd.RadarCollection.RcvChannels is None:
        return True

    pol_form = the_sicd.ImageFormation.TxRcvPolarizationProc
    rcv_pols = [entry.TxRcvPolarization for entry in the_sicd.RadarCollection.RcvChannels]
    if pol_form not in rcv_pols:
        the_sicd.log_validity_error(
            'ImageFormation.TxRcvPolarizationProc is populated as {},\n\t'
            'but it not one of the tx/rcv polarizations populated for '
            'the collect {}'.format(pol_form, rcv_pols))
        return False
    return True


def _check_deltak(the_sicd):
    """
    Checks the deltak parameters.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    if the_sicd.Grid is None:
        return True

    x_coords, y_coords = None, None
    try:
        valid_vertices = the_sicd.ImageData.get_valid_vertex_data()
        if valid_vertices is None:
            valid_vertices = the_sicd.ImageData.get_full_vertex_data()
        x_coords = the_sicd.Grid.Row.SS * (
                    valid_vertices[:, 0] - (the_sicd.ImageData.SCPPixel.Row - the_sicd.ImageData.FirstRow))
        y_coords = the_sicd.Grid.Col.SS * (
                    valid_vertices[:, 1] - (the_sicd.ImageData.SCPPixel.Col - the_sicd.ImageData.FirstCol))
    except (AttributeError, ValueError):
        pass
    return the_sicd.Grid.check_deltak(x_coords, y_coords)


def _check_projection(the_sicd):
    """
    Checks the projection ability.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    """

    if not the_sicd.can_project_coordinates():
        the_sicd.log_validity_warning(
            'No projection can be performed for this SICD.\n'
            'In particular, no derived products can be produced.')


def _validate_radiometric(Radiometric, Grid, SCPCOA):
    """
    Validate the radiometric parameters.

    Parameters
    ----------
    Radiometric : sarpy.io.complex.sicd_elements.Radiometric.RadiometricType
    Grid : sarpy.io.complex.sicd_elements.Grid.GridType
    SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType

    Returns
    -------
    bool
    """

    if Grid is None or Grid.Row is None or Grid.Col is None:
        return True

    cond = True
    area_sp = Grid.get_slant_plane_area()

    if Radiometric.RCSSFPoly is not None:
        rcs_sf = Radiometric.RCSSFPoly.get_array(dtype='float64')

        if Radiometric.BetaZeroSFPoly is not None:
            beta_sf = Radiometric.BetaZeroSFPoly.get_array(dtype='float64')
            if abs(rcs_sf[0, 0] / (beta_sf[0, 0] * area_sp) - 1) > 5e-2:
                Radiometric.log_validity_error(
                    'The BetaZeroSFPoly and RCSSFPoly are not consistent.')
                cond = False

        if SCPCOA is not None:
            if Radiometric.SigmaZeroSFPoly is not None:
                sigma_sf = Radiometric.SigmaZeroSFPoly.get_array(dtype='float64')
                mult = area_sp / numpy.cos(numpy.deg2rad(SCPCOA.SlopeAng))
                if (rcs_sf[0, 0] / (sigma_sf[0, 0] * mult) - 1) > 5e-2:
                    Radiometric.log_validity_error('The SigmaZeroSFPoly and RCSSFPoly are not consistent.')
                    cond = False

            if Radiometric.GammaZeroSFPoly is not None:
                gamma_sf = Radiometric.GammaZeroSFPoly.get_array(dtype='float64')
                mult = area_sp / (numpy.cos(numpy.deg2rad(SCPCOA.SlopeAng)) * numpy.sin(numpy.deg2rad(SCPCOA.GrazeAng)))
                if (rcs_sf[0, 0] / (gamma_sf[0, 0] * mult) - 1) > 5e-2:
                    Radiometric.log_validity_error('The GammaZeroSFPoly and RCSSFPoly are not consistent.')
                    cond = False

    if Radiometric.BetaZeroSFPoly is not None:
        beta_sf = Radiometric.BetaZeroSFPoly.get_array(dtype='float64')

        if SCPCOA is not None:
            if Radiometric.SigmaZeroSFPoly is not None:
                sigma_sf = Radiometric.SigmaZeroSFPoly.get_array(dtype='float64')
                mult = 1. / numpy.cos(numpy.deg2rad(SCPCOA.SlopeAng))
                if (beta_sf[0, 0] / (sigma_sf[0, 0] * mult) - 1) > 5e-2:
                    Radiometric.log_validity_error('The SigmaZeroSFPoly and BetaZeroSFPoly are not consistent.')
                    cond = False

            if Radiometric.GammaZeroSFPoly is not None:
                gamma_sf = Radiometric.GammaZeroSFPoly.get_array(dtype='float64')
                mult = 1. / (numpy.cos(numpy.deg2rad(SCPCOA.SlopeAng)) * numpy.sin(numpy.deg2rad(SCPCOA.GrazeAng)))
                if (beta_sf[0, 0] / (gamma_sf[0, 0] * mult) - 1) > 5e-2:
                    Radiometric.log_validity_error('The GammaZeroSFPoly and BetaZeroSFPoly are not consistent.')
                    cond = False
    return cond


def _check_radiometric_recommended(radiometric):
    """
    Checks the recommended fields for the radiometric object.

    Parameters
    ----------
    radiometric : sarpy.io.complex.sicd_elements.Radiometric.RadiometricType
    """

    for attribute in ['RCSSFPoly', 'BetaZeroSFPoly', 'SigmaZeroSFPoly', 'GammaZeroSFPoly']:
        value = getattr(radiometric, attribute)
        if value is None:
            radiometric.log_validity_warning(
                'No {} field provided, and associated RCS measurements '
                'will not be possible'.format(attribute))
    if radiometric.NoiseLevel is None:
        radiometric.log_validity_warning('No Radiometric.NoiseLevel provided, so noise estimates will not be possible.')
    elif radiometric.NoiseLevel.NoiseLevelType != 'ABSOLUTE':
        radiometric.log_validity_warning(
            'Radiometric.NoiseLevel provided are not ABSOLUTE, so noise estimates '
            'are not easily available.')


def _check_recommended_attributes(the_sicd):
    """
    Checks recommended attributes.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType
    """

    if the_sicd.Radiometric is None:
        the_sicd.log_validity_warning('No Radiometric data provided.')
    else:
        _check_radiometric_recommended(the_sicd.Radiometric)

    if the_sicd.Timeline is not None and the_sicd.Timeline.IPP is None:
        the_sicd.log_validity_warning(
            'No Timeline.IPP provided, so no PRF/PRI available '
            'for analysis of ambiguities.')

    if the_sicd.RadarCollection is not None and the_sicd.RadarCollection.Area is None:
        the_sicd.log_validity_info(
            'No RadarCollection.Area provided, and some tools prefer using\n\t'
            'a pre-determined output plane for consistent product definition.')

    if the_sicd.ImageData is not None and the_sicd.ImageData.ValidData is None:
        the_sicd.log_validity_info(
            'No ImageData.ValidData is defined. It is recommended to populate\n\t'
            'this data, if validity of pixels/areas is known.')

    if the_sicd.RadarCollection is not None and the_sicd.RadarCollection.RefFreqIndex is not None:
        the_sicd.log_validity_warning(
            'A reference frequency is being used. This may affect the results of\n\t'
            'this validation test, because a number tests could not be performed.')


def detailed_validation_checks(the_sicd):
    """
    Assembles the suite of detailed sicd validation checks.

    Parameters
    ----------
    the_sicd : sarpy.io.complex.sicd_elements.SICD.SICDType

    Returns
    -------
    bool
    """

    out = _validate_scp_time(the_sicd)
    out &= _validate_image_formation(the_sicd)
    out &= _validate_image_segment_id(the_sicd)
    out &= _validate_spotlight_mode(the_sicd)
    out &= _validate_valid_data(the_sicd)
    out &= _validate_polygons(the_sicd)
    out &= _validate_polarization(the_sicd)
    out &= _check_deltak(the_sicd)

    if the_sicd.SCPCOA is not None:
        out &= the_sicd.SCPCOA.check_values(the_sicd.GeoData)
    if the_sicd.Radiometric is not None:
        out &= _validate_radiometric(the_sicd.Radiometric, the_sicd.Grid, the_sicd.SCPCOA)

    _check_projection(the_sicd)
    _check_recommended_attributes(the_sicd)
    return out
