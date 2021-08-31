"""
The PFAType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import numpy
from numpy.linalg import norm
from numpy.polynomial import polynomial

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import BooleanDescriptor, FloatDescriptor, \
    SerializableDescriptor, UnitVectorDescriptor

from .base import DEFAULT_STRICT
from .blocks import Poly1DType, Poly2DType, XYZType

from sarpy.geometry import geocoords


class STDeskewType(Serializable):
    """
    Parameters to describe image domain ST Deskew processing.
    """

    _fields = ('Applied', 'STDSPhasePoly')
    _required = _fields
    # descriptors
    Applied = BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Parameter indicating if slow time *(ST)* Deskew Phase function has been applied.')  # type: bool
    STDSPhasePoly = SerializableDescriptor(
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
    FPN = UnitVectorDescriptor(
        'FPN', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Focus Plane unit normal in ECF coordinates. Unit vector FPN points away from the center of '
                  'the Earth.')  # type: XYZType
    IPN = UnitVectorDescriptor(
        'IPN', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Image Formation Plane unit normal in ECF coordinates. Unit vector IPN points away from the '
                  'center of the Earth.')  # type: XYZType
    PolarAngRefTime = FloatDescriptor(
        'PolarAngRefTime', _required, strict=DEFAULT_STRICT,
        docstring='Polar image formation reference time *(in seconds)*. Polar Angle = 0 at the reference time. '
                  'Measured relative to collection start. *Note: Reference time is typically set equal to the SCP '
                  'COA time but may be different.*')  # type: float
    PolarAngPoly = SerializableDescriptor(
        'PolarAngPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Polar Angle *(in radians)* as function of time '
                  'relative to Collection Start.')  # type: Poly1DType
    SpatialFreqSFPoly = SerializableDescriptor(
        'SpatialFreqSFPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields the *Spatial Frequency Scale Factor (KSF)* as a function of Polar '
                  r'Angle. That is, :math:`Polar Angle[radians] \to KSF[dimensionless]`. Used to scale RF '
                  'frequency *(fx, Hz)* to aperture spatial frequency *(Kap, cycles/m)*. Where,'
                  r':math:`Kap = fx\cdot (2/c)\cdot KSF`, and `Kap` is the effective spatial '
                  'frequency in the polar aperture.')  # type: Poly1DType
    Krg1 = FloatDescriptor(
        'Krg1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum *range spatial frequency (Krg)* output from the polar to rectangular '
                  'resampling.')  # type: float
    Krg2 = FloatDescriptor(
        'Krg2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum *range spatial frequency (Krg)* output from the polar to rectangular '
                  'resampling.')  # type: float
    Kaz1 = FloatDescriptor(
        'Kaz1', _required, strict=DEFAULT_STRICT,
        docstring='Minimum *azimuth spatial frequency (Kaz)* output from the polar to rectangular '
                  'resampling.')  # type: float
    Kaz2 = FloatDescriptor(
        'Kaz2', _required, strict=DEFAULT_STRICT,
        docstring='Maximum *azimuth spatial frequency (Kaz)* output from the polar to rectangular '
                  'resampling.')  # type: float
    STDeskew = SerializableDescriptor(
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

    def _derive_parameters(self, Grid, SCPCOA, GeoData, Position, Timeline):
        """
        Expected to be called from SICD parent.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        Timeline : sarpy.io.complex.sicd_elements.Timeline.TimelineType

        Returns
        -------
        None
        """

        if self.PolarAngRefTime is None and SCPCOA.SCPTime is not None:
            self.PolarAngRefTime = SCPCOA.SCPTime

        if GeoData is None or GeoData.SCP is None or GeoData.SCP.ECF is None:
            return

        scp = GeoData.SCP.ECF.get_array()

        if SCPCOA.ARPPos is not None and SCPCOA.ARPVel is not None:
            scp = GeoData.SCP.ECF.get_array()
            etp = geocoords.wgs_84_norm(scp)

            arp = SCPCOA.ARPPos.get_array()
            los = (scp - arp)
            ulos = los/norm(los)

            look = SCPCOA.look
            arp_vel = SCPCOA.ARPVel.get_array()
            uspz = look*numpy.cross(arp_vel, ulos)
            uspz /= norm(uspz)
            if Grid is not None and Grid.ImagePlane is not None:
                if self.IPN is None:
                    if Grid.ImagePlane == 'SLANT':
                        self.IPN = XYZType.from_array(uspz)
                    elif Grid.ImagePlane == 'GROUND':
                        self.IPN = XYZType.from_array(etp)
            elif self.IPN is None:
                self.IPN = XYZType.from_array(uspz)  # assuming slant -> most common

            if self.FPN is None:
                self.FPN = XYZType.from_array(etp)

        if Position is not None and \
                Timeline is not None and Timeline.CollectDuration is not None and \
                (self.PolarAngPoly is None or self.SpatialFreqSFPoly is None):
            pol_ref_pos = Position.ARPPoly(self.PolarAngRefTime)
            # fit the PFA polynomials
            times = numpy.linspace(0, Timeline.CollectDuration, 15)
            k_a, k_sf = self.pfa_polar_coords(Position, scp, times)

            self.PolarAngPoly = Poly1DType(Coefs=polynomial.polyfit(times, k_a, 5, full=False))
            self.SpatialFreqSFPoly = Poly1DType(Coefs=polynomial.polyfit(k_a, k_sf, 5, full=False))

        if Grid is not None and Grid.Row is not None and \
                Grid.Row.KCtr is not None and Grid.Row.ImpRespBW is not None:
            if self.Krg1 is None:
                self.Krg1 = Grid.Row.KCtr - 0.5*Grid.Row.ImpRespBW
            if self.Krg2 is None:
                self.Krg2 = Grid.Row.KCtr + 0.5*Grid.Row.ImpRespBW
        if Grid is not None and Grid.Col is not None and \
                Grid.Col.KCtr is not None and Grid.Col.ImpRespBW is not None:
            if self.Kaz1 is None:
                self.Kaz1 = Grid.Col.KCtr - 0.5*Grid.Col.ImpRespBW
            if self.Kaz2 is None:
                self.Kaz2 = Grid.Col.KCtr + 0.5*Grid.Col.ImpRespBW

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
            self.log_validity_error(
                'The PolarAngPoly evaluated at PolarAngRefTime yields {}, which should be 0'.format(polar_angle_ref))
            cond = False
        return cond

    def _basic_validity_check(self):
        condition = super(PFAType, self)._basic_validity_check()
        condition &= self._check_polar_ang_ref()
        return condition
