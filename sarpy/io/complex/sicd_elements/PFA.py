# -*- coding: utf-8 -*-
"""
The PFAType definition.
"""

import logging
import numpy
from numpy.linalg import norm
from scipy.constants import speed_of_light

from .base import Serializable, DEFAULT_STRICT, _BooleanDescriptor, _FloatDescriptor, \
    _SerializableDescriptor, _UnitVectorDescriptor
from .blocks import Poly1DType, Poly2DType, XYZType

from sarpy.geometry import geocoords


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class STDeskewType(Serializable):
    """
    Parameters to describe image domain ST Deskew processing.
    """

    _fields = ('Applied', 'STDSPhasePoly')
    _required = _fields
    # descriptors
    Applied = _BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Parameter indicating if slow time *(ST)* Deskew Phase function has been applied.')  # type: bool
    STDSPhasePoly = _SerializableDescriptor(
        'STDSPhasePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Slow time deskew phase function to perform the *ST/Kaz* shift. Two-dimensional phase '
                  '(cycles) polynomial function of image range coordinate *(variable 1)* and '
                  'azimuth coordinate *(variable 2)*.')  # type: Poly2DType

    def __init__(self, Applied=None, STDSPhasePoly=None, **kwargs):
        """

        Parameters
        ----------
        Applied : bool
        STDSPhasePoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Applied = Applied
        # noinspection PytypeChecker
        self.STDSPhasePoly = STDSPhasePoly
        super(STDeskewType, self).__init__(**kwargs)


class PFAType(Serializable):
    """Parameters for the Polar Formation Algorithm."""
    _fields = (
        'FPN', 'IPN', 'PolarAngRefTime', 'PolarAngPoly', 'SpatialFreqSFPoly', 'Krg1', 'Krg2', 'Kaz1', 'Kaz2',
        'STDeskew')
    _required = ('FPN', 'IPN', 'PolarAngRefTime', 'PolarAngPoly', 'SpatialFreqSFPoly', 'Krg1', 'Krg2', 'Kaz1', 'Kaz2')
    _numeric_format = {'PolarAngRefTime': '0.16G', 'Krg1': '0.16G', 'Krg2': '0.16G', 'Kaz1': '0.16G', 'Kaz2': '0.16G'}
    # descriptors
    FPN = _UnitVectorDescriptor(
        'FPN', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Focus Plane unit normal in ECF coordinates. Unit vector FPN points away from the center of '
                  'the Earth.')  # type: XYZType
    IPN = _UnitVectorDescriptor(
        'IPN', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Image Formation Plane unit normal in ECF coordinates. Unit vector IPN points away from the '
                  'center of the Earth.')  # type: XYZType
    PolarAngRefTime = _FloatDescriptor(
        'PolarAngRefTime', _required, strict=DEFAULT_STRICT,
        docstring='Polar image formation reference time *(in seconds)*. Polar Angle = 0 at the reference time. '
                  'Measured relative to collection start. *Note: Reference time is typically set equal to the SCP '
                  'COA time but may be different.*')  # type: float
    PolarAngPoly = _SerializableDescriptor(
        'PolarAngPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Polar Angle *(in radians)* as function of time '
                  'relative to Collection Start.')  # type: Poly1DType
    SpatialFreqSFPoly = _SerializableDescriptor(
        'SpatialFreqSFPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields the *Spatial Frequency Scale Factor (KSF)* as a function of Polar '
                  r'Angle. That is, :math:`Polar Angle[radians] \to KSF[dimensionless]`. Used to scale RF '
                  'frequency *(fx, Hz)* to aperture spatial frequency *(Kap, cycles/m)*. Where,'
                  r':math:`Kap = fx\cdot (2/c)\cdot KSF`, and `Kap` is the effective spatial '
                  'frequency in the polar aperture.')  # type: Poly1DType
    Krg1 = _FloatDescriptor(
        'Krg1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum *range spatial frequency (Krg)* output from the polar to rectangular '
                  'resampling.')  # type: float
    Krg2 = _FloatDescriptor(
        'Krg2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum *range spatial frequency (Krg)* output from the polar to rectangular '
                  'resampling.')  # type: float
    Kaz1 = _FloatDescriptor(
        'Kaz1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum *azimuth spatial frequency (Kaz)* output from the polar to rectangular '
                  'resampling.')  # type: float
    Kaz2 = _FloatDescriptor(
        'Kaz2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum *azimuth spatial frequency (Kaz)* output from the polar to rectangular '
                  'resampling.')  # type: float
    STDeskew = _SerializableDescriptor(
        'STDeskew', STDeskewType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters to describe image domain slow time *(ST)* Deskew processing.')  # type: STDeskewType

    def __init__(self, FPN=None, IPN=None, PolarAngRefTime=None, PolarAngPoly=None,
                 SpatialFreqSFPoly=None, Krg1=None, Krg2=None, Kaz1=None, Kaz2=None,
                 STDeskew=None, **kwargs):
        """

        Parameters
        ----------
        FPN : XYZType|numpy.ndarray|list|tuple
        IPN : XYZType|numpy.ndarray|list|tuple
        PolarAngRefTime : float
        PolarAngPoly : Poly1DType|numpy.ndarray|list|tuple
        SpatialFreqSFPoly : Poly1DType|numpy.ndarray|list|tuple
        Krg1 : float
        Krg2 : float
        Kaz1 : float
        Kaz2 : float
        STDeskew : STDeskewType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.FPN = FPN
        self.IPN = IPN
        self.PolarAngRefTime = PolarAngRefTime
        self.PolarAngPoly = PolarAngPoly
        self.SpatialFreqSFPoly = SpatialFreqSFPoly
        self.Krg1, self.Krg2 = Krg1, Krg2
        self.Kaz1, self.Kaz2 = Kaz1, Kaz2
        self.STDeskew = STDeskew
        super(PFAType, self).__init__(**kwargs)

    def pfa_polar_coords(self, Position, SCP, times):
        """
        Calculate the PFA parameters necessary for mapping phase history to polar coordinates.

        Parameters
        ----------
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        SCP : numpy.ndarray
        times : numpy.ndarray|float|int

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)|(float, float)
            `(k_a, k_sf)`, where `k_a` is polar angle, and `k_sf` is spatial
            frequency scale factor. The shape of the output array (or scalar) will
            match the shape of the `times` array (or scalar).
        """

        def project_to_image_plane(points):
            # type: (numpy.ndarray) -> numpy.ndarray
            # project into the image plane along line normal to the focus plane
            offset = (SCP - points).dot(ipn)/fpn.dot(ipn)
            if offset.ndim == 0:
                return points + offset*fpn
            else:
                return points + numpy.outer(offset, fpn)

        if self.IPN is None or self.FPN is None:
            return None, None

        ipn = self.IPN.get_array(dtype='float64')
        fpn = self.FPN.get_array(dtype='float64')
        if isinstance(times, (float, int)) or times.ndim == 0:
            o_shape = None
            times = numpy.array([times, ], dtype='float64')
        else:
            o_shape = times.shape
            times = numpy.reshape(times, (-1, ))
        positions = Position.ARPPoly(times)
        reference_position = Position.ARPPoly(self.PolarAngRefTime)
        image_plane_positions = project_to_image_plane(positions)
        image_plane_coa = project_to_image_plane(reference_position)

        # establish image plane coordinate system
        ip_x = image_plane_coa - SCP
        ip_x /= numpy.linalg.norm(ip_x)
        ip_y = numpy.cross(ip_x, ipn)

        # compute polar angle of sensor position in image plane
        ip_range = image_plane_positions - SCP
        ip_range /= numpy.linalg.norm(ip_range, axis=1)[:, numpy.newaxis]
        k_a = -numpy.arctan2(ip_range.dot(ip_y), ip_range.dot(ip_x))

        # compute the spatial frequency scale factor
        range_vectors = positions - SCP
        range_vectors /= numpy.linalg.norm(range_vectors, axis=1)[:, numpy.newaxis]
        sin_graze = range_vectors.dot(fpn)
        sin_graze_ip = ip_range.dot(fpn)
        k_sf = numpy.sqrt((1 - sin_graze*sin_graze)/(1 - sin_graze_ip*sin_graze_ip))
        if o_shape is None:
            return k_a[0], k_sf[0]
        elif len(o_shape) > 1:
            return numpy.reshape(k_a, o_shape), numpy.reshape(k_sf, o_shape)
        else:
            return k_a, k_sf

    def _derive_parameters(self, Grid, SCPCOA, GeoData):
        """
        Expected to be called from SICD parent.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.GridType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType

        Returns
        -------
        None
        """

        if self.PolarAngRefTime is None and SCPCOA.SCPTime is not None:
            self.PolarAngRefTime = SCPCOA.SCPTime

        if GeoData is not None and GeoData.SCP is not None and GeoData.SCP.ECF is not None and \
                SCPCOA.ARPPos is not None and SCPCOA.ARPVel is not None:
            SCP = GeoData.SCP.ECF.get_array()
            ETP = geocoords.wgs_84_norm(SCP)

            ARP = SCPCOA.ARPPos.get_array()
            LOS = (SCP - ARP)
            uLOS = LOS/norm(LOS)

            look = SCPCOA.look
            ARP_vel = SCPCOA.ARPVel.get_array()
            uSPZ = look*numpy.cross(ARP_vel, uLOS)
            uSPZ /= norm(uSPZ)
            if Grid is not None and Grid.ImagePlane is not None:
                if self.IPN is None:
                    if Grid.ImagePlane == 'SLANT':
                        self.IPN = XYZType.from_array(uSPZ)
                    elif Grid.ImagePlane == 'GROUND':
                        self.IPN = XYZType.from_array(ETP)
            elif self.IPN is None:
                self.IPN = XYZType.from_array(uSPZ)  # assuming slant -> most common

            if self.FPN is None:
                self.FPN = XYZType.from_array(ETP)

    def _check_kaz_krg(self, Grid):
        """
        Check the validity of the Kaz and Krg values.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType

        Returns
        -------
        bool
        """

        cond = True
        if self.STDeskew is None or not self.STDeskew.Applied:
            try:
                if self.Kaz2 - Grid.Col.KCtr > 0.5/Grid.Col.SS + 1e-10:
                    logging.error(
                        'PFA.Kaz2 - Grid.Col.KCtr ({}) > 0.5/Grid.Col.SS ({})'.format(
                            self.Kaz2 - Grid.Col.KCtr, 0.5/Grid.Col.SS))
                    cond = False
            except (AttributeError, TypeError, ValueError):
                pass

            try:
                if self.Kaz1 - Grid.Col.KCtr < -0.5/Grid.Col.SS - 1e-10:
                    logging.error(
                        'PFA.Kaz1 - Grid.Col.KCtr ({}) < -0.5/Grid.Col.SS ({})'.format(
                            self.Kaz1 - Grid.Col.KCtr, -0.5/Grid.Col.SS))
                    cond = False
            except (AttributeError, TypeError, ValueError):
                pass

        try:
            if self.Krg2 - Grid.Row.KCtr > 0.5/Grid.Row.SS + 1e-10:
                logging.error(
                    'PFA.Krg2 - Grid.Row.KCtr ({}) > 0.5/Grid.Row.SS ({})'.format(
                        self.Krg2 - Grid.Row.KCtr, 0.5/Grid.Row.SS))
                cond = False
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            if self.Krg1 - Grid.Row.KCtr < -0.5/Grid.Row.SS - 1e-10:
                logging.error(
                    'PFA.Krg1 - Grid.Row.KCtr ({}) < -0.5/Grid.Row.SS ({})'.format(
                        self.Krg1 - Grid.Row.KCtr, -0.5/Grid.Row.SS))
                cond = False
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            if Grid.Row.ImpRespBW > (self.Krg2 - self.Krg1) + 1e-10:
                logging.error(
                    'Grid.Row.ImpRespBW ({}) > PFA.Krg2 - PFA.Krg1 ({})'.format(
                        Grid.Row.ImpRespBW, self.Krg2 - self.Krg1))
                cond = False
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            if abs(Grid.Col.KCtr) > 1e-5 and abs(Grid.Col.KCtr - 0.5*(self.Kaz2 +self.Kaz1)) > 1e-5:
                logging.error(
                    'Grid.Col.KCtr ({}) not within 1e-5 of 0.5*(PFA.Kaz2 + PFA.Kaz1) ({})'.format(
                        Grid.Col.KCtr, 0.5*(self.Kaz2 + self.Kaz1)))
                cond = False
        except (AttributeError, TypeError, ValueError):
            pass

        return cond

    def _check_polys(self, Position, Timeline, SCP):
        """

        Parameters
        ----------
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        Timeline : sarpy.io.complex.sicd_elements.Timeline.TimelineType
        SCP : numpy.ndarray

        Returns
        -------
        bool
        """

        cond = True
        num_samples = max(self.PolarAngPoly.Coefs.size, 40)
        times = numpy.linspace(0, Timeline.CollectDuration, num_samples)

        k_a, k_sf = self.pfa_polar_coords(Position, SCP, times)
        if k_a is None:
            return True
        # check for agreement with k_a and k_sf derived from the polynomials
        k_a_derived = self.PolarAngPoly(times)
        k_sf_derived = self.SpatialFreqSFPoly(k_a)
        k_a_diff = numpy.amax(numpy.abs(k_a_derived - k_a))
        k_sf_diff = numpy.amax(numpy.abs(k_sf_derived - k_sf))
        if k_a_diff > 5e-3:
            logging.error(
                'The image formation algorithm is PFA, and the PolarAngPoly evaluated values do not '
                'agree with actual calculated values')
            cond = False
        if k_sf_diff > 5e-3:
            logging.error(
                'The image formation algorithm is PFA, and the SpatialFreqSFPoly evaluated values do not '
                'agree with actual calculated values')
            cond = False
        return cond

    def _check_uvects(self, Position, Grid, SCP):
        """

        Parameters
        ----------
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        SCP : numpy.ndarray

        Returns
        -------
        bool
        """

        if self.IPN is None or self.FPN is None:
            return True

        cond = True
        ipn = self.IPN.get_array(dtype='float64')
        fpn = self.FPN.get_array(dtype='float64')
        row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
        col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')

        pol_ref_point = Position.ARPPoly(self.PolarAngRefTime)
        offset = (SCP - pol_ref_point).dot(ipn)/(fpn.dot(ipn))
        ref_position_ipn = pol_ref_point + offset*fpn
        slant_range = ref_position_ipn - SCP
        u_slant_range = slant_range/numpy.linalg.norm(slant_range)
        derived_row_vector = -u_slant_range
        if numpy.linalg.norm(derived_row_vector - row_uvect) > 1e-3:
            logging.error(
                'The image formation algorithm is PFA, and the Grid.Row.UVectECF ({}) '
                'is not in good agreement with the expected value ({})'.format(row_uvect, derived_row_vector))
            cond = False

        derived_col_vector = numpy.cross(ipn, derived_row_vector)
        if numpy.linalg.norm(derived_col_vector - col_uvect) > 1e-3:
            logging.error(
                'The image formation algorithm is PFA, and the Grid.Col.UVectECF ({}) '
                'is not in good agreement with the expected value ({})'.format(col_uvect, derived_col_vector))
            cond = False

        return cond

    def _check_stdeskew(self, Grid):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType

        Returns
        -------
        bool
        """

        if self.STDeskew is None or not self.STDeskew.Applied:
            return True
        cond = True
        if Grid.TimeCOAPoly is not None:
            timecoa_poly = Grid.TimeCOAPoly.get_array(dtype='float64')
            if timecoa_poly.shape == (1, 1) or numpy.all(timecoa_poly.flatten()[1:] < 1e-6):
                logging.error(
                    'The image formation algorithm is PFA, PFA.STDeskew.Applied is True, and '
                    'the Grid.TimeCOAPoly is essentially constant.')
                cond = False

        # the Row DeltaKCOAPoly and STDSPhasePoly should be essentially identical
        if Grid.Row is not None and Grid.Row.DeltaKCOAPoly is not None and \
                self.STDeskew.STDSPhasePoly is not None:
            stds_phase_poly = self.STDeskew.STDSPhasePoly.get_array(dtype='float64')
            delta_kcoa = Grid.Row.DeltaKCOAPoly.get_array(dtype='float64')
            rows = max(stds_phase_poly.shape[0], delta_kcoa.shape[0])
            cols = max(stds_phase_poly.shape[1], delta_kcoa.shape[1])
            exp_stds_phase_poly = numpy.zeros((rows, cols), dtype='float64')
            exp_delta_kcoa = numpy.zeros((rows, cols), dtype='float64')
            exp_stds_phase_poly[:stds_phase_poly.shape[0], :stds_phase_poly.shape[1]] = stds_phase_poly
            exp_delta_kcoa[:delta_kcoa.shape[0], :delta_kcoa.shape[1]] = delta_kcoa

            if numpy.max(numpy.abs(exp_delta_kcoa - exp_stds_phase_poly)) > 1e-6:
                logging.error(
                    'The image formation algorithm is PFA, PFA.STDeskew.Applied is True, and '
                    'the Grid.Row.DeltaKCOAPoly ({}) and PFA.STDeskew.STDSPhasePoly ({}) are '
                    'not in good agreement.'.format(delta_kcoa, stds_phase_poly))
                cond = False

        return cond

    def _check_kctr(self, RadarCollection, ImageFormation, Grid):
        """

        Parameters
        ----------
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
            kap_ctr = center_freq*self.SpatialFreqSFPoly.Coefs[0]*2/speed_of_light
            theta = numpy.arctan(0.5*Grid.Col.ImpRespBW/Grid.Row.KCtr)  # aperture angle
            kctr_total = max(1e-2, 1 - numpy.cos(theta))  # difference between Krg and Kap
            if abs(Grid.Row.KCtr/kap_ctr - 1) > kctr_total:
                logging.error(
                    'The image formation algorithm is PFA, and the Row.KCtr value ({}) '
                    'is not in keeping with expected ({})'.format(Grid.Row.KCtr, kap_ctr))
                cond = False
        except (AttributeError, ValueError, TypeError):
            pass

        return cond

    def _check_image_plane(self, Grid, SCPCOA, SCP):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
        SCP : numpy.ndarray

        Returns
        -------
        bool
        """

        if self.IPN is None or self.FPN is None:
            return True

        cond = True
        ipn = self.IPN.get_array(dtype='float64')
        fpn = self.FPN.get_array(dtype='float64')
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
                    logging.error(
                        'The image formation algorithm is PFA, the Grid.ImagePlane is "SLANT", '
                        'but COA slant plane and provided ipn are not within one degree of each other.')
                    cond = False
            except (AttributeError, ValueError, TypeError):
                pass
        elif Grid.ImagePlane == 'GROUND':
            if numpy.arccos(ipn.dot(ETP)) > numpy.deg2rad(3):
                logging.error(
                    'The image formation algorithm is PFA, the Grid.ImagePlane is "Ground", '
                    'but the Earth Tangent Plane at SCP and provided ipn are not within three '
                    'degrees of each other.')
                cond = False

        # verify that fpn points outwards
        if fpn.dot(SCP) < 0:
            logging.error(
                'The image formation algorithm is PFA, and the focus plane unit normal does not '
                'appear to be outward pointing')
            cond = False
        # check agreement between focus plane and ground plane
        if numpy.arccos(fpn.dot(ETP)) > numpy.deg2rad(3):
            logging.warning(
                'The image formation algorithm is PFA, and the focus plane unit normal is not within '
                'three degrees of the earth Tangent Plane.')
        return cond

    def _check_polar_angle_consistency(self, CollectionInfo, ImageFormation):
        """

        Parameters
        ----------
        CollectionInfo : sarpy.io.complex.sicd_elements.CollectionInfo.CollectionInfoType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

        Returns
        -------
        bool
        """

        if CollectionInfo.RadarMode is None or CollectionInfo.RadarMode.ModeType != 'SPOTLIGHT':
            return True

        cond = True
        polar_angle_bounds = self.PolarAngPoly(numpy.array(sorted([ImageFormation.TStartProc, ImageFormation.TEndProc]), dtype='float64'))
        derived_pol_angle_bounds = numpy.arctan(numpy.array([self.Kaz1, self.Kaz2], dtype='float64')/self.Krg1)
        pol_angle_bounds_diff = numpy.amax(numpy.abs(polar_angle_bounds - derived_pol_angle_bounds))
        if pol_angle_bounds_diff > 1e-2:
            logging.error(
                'The image formation algorithm is PFA, the derived polar angle bounds ({}) are not consistent '
                'with the provided ImageFormation processing times (expected bounds {}).'.format(polar_angle_bounds, derived_pol_angle_bounds))
            cond = False
        return cond

    def check_parameters(self, Grid, SCPCOA, GeoData, Position, Timeline, RadarCollection,
                         ImageFormation, CollectionInfo):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        Timeline : sarpy.io.complex.sicd_elements.Timeline.TimelineType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType
        CollectionInfo : sarpy.io.complex.sicd_elements.CollectionInfo.CollectionInfoType

        Returns
        -------
        bool
        """

        cond = True
        if Grid.Type != 'RGAZIM':
            logging.error(
                'The image formation algorithm is PFA, and Grid.Type is populated as "{}", '
                'but should be "RGAZIM"'.format(Grid.Type))
            cond = False
        if abs(self.PolarAngRefTime - SCPCOA.SCPTime) > 1e-6:
            logging.warning(
                'The image formation algorithm is PFA, and the PFA.PolarAngRefTime ({})'
                'does not agree with the SCPCOA.SCPTime ({})'.format(self.PolarAngRefTime, SCPCOA.SCPTime))
            cond = False
        cond &= self._check_kaz_krg(Grid)
        cond &= self._check_stdeskew(Grid)
        cond &= self._check_kctr(RadarCollection, ImageFormation, Grid)

        try:
            SCP = GeoData.SCP.ECF.get_array(dtype='float64')
        except (AttributeError, ValueError, TypeError):
            return cond

        cond &= self._check_polys(Position, Timeline, SCP)
        cond &= self._check_uvects(Position, Grid, SCP)
        cond &= self._check_image_plane(Grid, SCPCOA, SCP)
        cond &= self._check_polar_angle_consistency(CollectionInfo, ImageFormation)
        return cond

    def _check_polar_ang_ref(self):
        """
        Checks the polar angle origin makes sense.

        Returns
        -------
        bool
        """

        if self.PolarAngPoly is None or self.PolarAngRefTime is None:
            return True

        cond = True
        polar_angle_ref = self.PolarAngPoly(self.PolarAngRefTime)
        if abs(polar_angle_ref) > 1e-4:
            logging.error(
                'The PolarAngPoly evaluated at PolarAngRefTime yields {}, which should be 0'.format(polar_angle_ref))
            cond = False
        return cond

    def _basic_validity_check(self):
        condition = super(PFAType, self)._basic_validity_check()
        condition &= self._check_polar_ang_ref()
        return condition
