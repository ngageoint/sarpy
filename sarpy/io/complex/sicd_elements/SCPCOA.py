"""
The SCPCOAType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging

import numpy
from numpy.linalg import norm

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import StringEnumDescriptor, FloatDescriptor, \
    SerializableDescriptor

from .base import DEFAULT_STRICT
from .blocks import XYZType

from sarpy.geometry import geocoords

logger = logging.getLogger(__name__)


class GeometryCalculator(object):
    """
    Performs the necessary SCPCOA geometry element calculations.
    """

    def __init__(self, SCP, ARPPos, ARPVel):
        """

        Parameters
        ----------
        SCP : numpy.ndarray
            The scene center point.
        ARPPos : numpy.ndarray
            The aperture position in ECEF coordinates at the SCP center of aperture time.
        ARPVel : numpy.ndarray
            The aperture velocity in ECEF coordinates at the SCP center of aperture time.
        """
        self.SCP = SCP
        self.ARP = ARPPos
        self.ARP_vel = ARPVel
        self.LOS = (self.SCP - self.ARP)
        # unit vector versions
        self.uSCP = self._make_unit(self.SCP)
        self.uARP = self._make_unit(self.ARP)
        self.uARP_vel = self._make_unit(self.ARP_vel)
        self.uLOS = self._make_unit(self.LOS)
        self.left = numpy.cross(self.uARP, self.uARP_vel)
        self.look = numpy.sign(self.left.dot(self.uLOS))
        # Earth Tangent Plane (ETP) at the SCP is the plane tangent to the surface of constant height
        # above the WGS 84 ellipsoid (HAE) that contains the SCP. The ETP is an approximation to the
        # ground plane at the SCP.
        self.ETP = geocoords.wgs_84_norm(SCP)
        # slant plane unit normal
        self.uSPZ = self._make_unit(self.look*numpy.cross(self.ARP_vel, self.uLOS))
        # perpendicular component of range vector wrt the ground plane
        self.uGPX = self._make_unit(-self.uLOS + numpy.dot(self.ETP, self.uLOS)*self.ETP)
        self.uGPY = numpy.cross(self.ETP, self.uGPX)  # already unit vector
        # perpendicular component of north wrt the ground plane
        self.uNORTH = self._make_unit(numpy.array([0, 0, 1]) - self.ETP[2]*self.ETP)
        self.uEAST = numpy.cross(self.uNORTH, self.ETP)  # already unit vector

    @staticmethod
    def _make_unit(vec):
        vec_norm = norm(vec)
        if vec_norm < 1e-6:
            logger.error(
                'The input vector to be normalized has norm {},\n\t'
                'this is likely a mistake'.format(vec_norm))
        return vec/vec_norm

    @property
    def ROV(self):
        """
        float: Range over velocity
        """
        return float(norm(self.LOS)/norm(self.ARP_vel))

    @property
    def SideOfTrack(self):
        return 'R' if self.look < 0 else 'L'

    @property
    def SlantRange(self):
        return float(norm(self.LOS))

    @property
    def GroundRange(self):
        return norm(self.SCP) * numpy.arccos(self.uSCP.dot(self.uARP))

    @property
    def DopplerConeAng(self):
        return float(numpy.rad2deg(numpy.arccos(self.uARP_vel.dot(self.uLOS))))

    @property
    def GrazeAng(self):
        return self.get_graze_and_incidence()[0]

    @property
    def IncidenceAng(self):
        return self.get_graze_and_incidence()[1]

    def get_graze_and_incidence(self):
        graze_ang = -float(numpy.rad2deg(numpy.arcsin(self.ETP.dot(self.uLOS))))
        return graze_ang, 90 - graze_ang

    @property
    def TwistAng(self):
        return float(-numpy.rad2deg(numpy.arcsin(self.uGPY.dot(self.uSPZ))))

    @property
    def SquintAngle(self):
        arp_vel_proj = self._make_unit(self.uARP_vel - self.uARP_vel.dot(self.uARP)*self.uARP)
        los_proj = self._make_unit(-self.uLOS + self.uLOS.dot(self.uARP)*self.uARP)
        return float(numpy.rad2deg(
            numpy.arctan2(numpy.cross(arp_vel_proj, los_proj).dot(self.uARP), arp_vel_proj.dot(los_proj))))

    @property
    def SlopeAng(self):
        return float(numpy.rad2deg(numpy.arccos(self.ETP.dot(self.uSPZ))))

    @property
    def AzimAng(self):
        azim_ang = numpy.rad2deg(numpy.arctan2(self.uGPX.dot(self.uEAST), self.uGPX.dot(self.uNORTH)))
        azim_ang = azim_ang if azim_ang > 0 else azim_ang + 360
        return float(azim_ang)

    @property
    def LayoverAng(self):
        return self.get_layover()[0]

    def get_layover(self):
        layover_ground = self.ETP - self.ETP.dot(self.uSPZ)*self.uSPZ
        layover_ang = numpy.rad2deg(
            numpy.arctan2(layover_ground.dot(self.uEAST), layover_ground.dot(self.uNORTH)))
        layover_ang = layover_ang if layover_ang > 0 else layover_ang + 360
        return float(layover_ang), float(norm(layover_ground))

    def get_shadow(self):
        shadow = self.ETP - self.uLOS/self.uLOS.dot(self.ETP)
        shadow_prime = shadow - self.uSPZ*(shadow.dot(self.ETP)/self.uSPZ.dot(self.ETP))
        shadow_angle = numpy.rad2deg(numpy.arctan2(shadow_prime.dot(self.uGPY), shadow_prime.dot(self.uGPX)))
        return float(shadow_angle), float(norm(shadow_prime))


class SCPCOAType(Serializable):
    """
    Center of Aperture (COA) for the Scene Center Point (SCP).
    """

    _fields = (
        'SCPTime', 'ARPPos', 'ARPVel', 'ARPAcc', 'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAng',
        'GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng')
    _required = _fields
    _numeric_format = {
        'SCPTime': '0.16G', 'SlantRange': '0.16G', 'GroundRange': '0.16G', 'DopplerConeAng': '0.16G',
        'GrazeAng': '0.16G', 'IncidenceAng': '0.16G', 'TwistAng': '0.16G', 'SlopeAng': '0.16G',
        'AzimAng': '0.16G', 'LayoverAng': '0.16G'}
    # class variables
    _SIDE_OF_TRACK_VALUES = ('L', 'R')
    # descriptors
    SCPTime = FloatDescriptor(
        'SCPTime', _required, strict=DEFAULT_STRICT,
        docstring='*Center Of Aperture time for the SCP (t_COA_SCP)*, relative to collection '
                  'start in seconds.')  # type: float
    ARPPos = SerializableDescriptor(
        'ARPPos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Aperture position at *t_COA_SCP* in ECF coordinates.')  # type: XYZType
    ARPVel = SerializableDescriptor(
        'ARPVel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP Velocity at *t_COA_SCP* in ECF coordinates.')  # type: XYZType
    ARPAcc = SerializableDescriptor(
        'ARPAcc', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP Acceleration at *t_COA_SCP* in ECF coordinates.')  # type: XYZType
    SideOfTrack = StringEnumDescriptor(
        'SideOfTrack', _SIDE_OF_TRACK_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Side of track.')  # type: str
    SlantRange = FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT,
        docstring='Slant range from the aperture to the *SCP* in meters.')  # type: float
    GroundRange = FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT,
        docstring='Ground Range from the aperture nadir to the *SCP*. Distance measured along spherical earth model '
                  'passing through the *SCP* in meters.')  # type: float
    DopplerConeAng = FloatDescriptor(
        'DopplerConeAng', _required, strict=DEFAULT_STRICT,
        docstring='The Doppler Cone Angle to SCP at *t_COA_SCP* in degrees.')  # type: float
    GrazeAng = FloatDescriptor(
        'GrazeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Grazing Angle between the SCP *Line of Sight (LOS)* and *Earth Tangent Plane (ETP)*.')  # type: float
    IncidenceAng = FloatDescriptor(
        'IncidenceAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Incidence Angle between the *LOS* and *ETP* normal.')  # type: float
    TwistAng = FloatDescriptor(
        'TwistAng', _required, strict=DEFAULT_STRICT, bounds=(-90., 90.),
        docstring='Angle between cross range in the *ETP* and cross range in the slant plane.')  # type: float
    SlopeAng = FloatDescriptor(
        'SlopeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Slope Angle from the *ETP* to the slant plane at *t_COA_SCP*.')  # type: float
    AzimAng = FloatDescriptor(
        'AzimAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.),
        docstring='Angle from north to the line from the *SCP* to the aperture nadir at *COA*. Measured '
                  'clockwise in the *ETP*.')  # type: float
    LayoverAng = FloatDescriptor(
        'LayoverAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.),
        docstring='Angle from north to the layover direction in the *ETP* at *COA*. Measured '
                  'clockwise in the *ETP*.')  # type: float

    def __init__(self, SCPTime=None, ARPPos=None, ARPVel=None, ARPAcc=None, SideOfTrack=None,
                 SlantRange=None, GroundRange=None, DopplerConeAng=None, GrazeAng=None, IncidenceAng=None,
                 TwistAng=None, SlopeAng=None, AzimAng=None, LayoverAng=None, **kwargs):
        """

        Parameters
        ----------
        SCPTime : float
        ARPPos : XYZType|numpy.ndarray|list|tuple
        ARPVel : XYZType|numpy.ndarray|list|tuple
        ARPAcc : XYZType|numpy.ndarray|list|tuple
        SideOfTrack : str
        SlantRange : float
        GroundRange : float
        DopplerConeAng : float
        GrazeAng : float
        IncidenceAng : float
        TwistAng : float
        SlopeAng : float
        AzimAng : float
        LayoverAng : float
        kwargs : dict
        """

        self._ROV = None
        self._squint = None
        self._shadow = None
        self._shadow_magnitude = None
        self._layover_magnitude = None

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SCPTime = SCPTime
        self.ARPPos, self.ARPVel, self.ARPAcc = ARPPos, ARPVel, ARPAcc
        self.SideOfTrack = SideOfTrack
        self.SlantRange, self.GroundRange = SlantRange, GroundRange
        self.DopplerConeAng, self.GrazeAng, self.IncidenceAng = DopplerConeAng, GrazeAng, IncidenceAng
        self.TwistAng, self.SlopeAng, self.AzimAng, self.LayoverAng = TwistAng, SlopeAng, AzimAng, LayoverAng
        super(SCPCOAType, self).__init__(**kwargs)

    @property
    def look(self):
        """
        int: An integer version of `SideOfTrack`:

            * None if `SideOfTrack` is not defined

            * -1 if SideOfTrack == 'R'

            * 1 if SideOftrack == 'L'
        """

        if self.SideOfTrack is None:
            return None
        else:
            return -1 if self.SideOfTrack == 'R' else 1

    @property
    def ROV(self):
        """
        float: The Ratio of Range to Velocity at Center of Aperture time.
        """

        return self._ROV

    @property
    def ThetaDot(self):
        """
        float: Derivative of Theta as a function of time at Center of Aperture time.
        """

        if self.DopplerConeAng is None or self.ROV is None:
            return None
        return float(numpy.sin(numpy.deg2rad(self.DopplerConeAng))/self.ROV)

    @property
    def MultipathGround(self):
        """
        float: The anticipated angle of multipath features on the ground in degrees.
        """
        if self.GrazeAng is None or self.TwistAng is None:
            return None
        return numpy.rad2deg(
            -numpy.arctan(numpy.tan(numpy.deg2rad(self.TwistAng))*numpy.sin(numpy.deg2rad(self.GrazeAng))))

    @property
    def Multipath(self):
        """
        float: The anticipated angle of multipath features in degrees.
        """
        if self.AzimAng is None or self.MultipathGround is None:
            return None
        return numpy.mod(self.AzimAng - 180 + self.MultipathGround, 360)

    @property
    def Shadow(self):
        """
        float: The anticipated angle of shadow features in degrees.
        """

        return self._shadow

    @property
    def ShadowMagnitude(self):
        """
        float: The anticipated relative magnitude of shadow features.
        """

        return self._shadow_magnitude

    @property
    def Squint(self):
        """
        float: The squint angle, in degrees.
        """

        return self._squint

    @property
    def LayoverMagnitude(self):
        """
        float: The anticipated relative magnitude of layover features.
        """

        return self._layover_magnitude

    def _derive_scp_time(self, Grid, overwrite=False):
        """
        Expected to be called by SICD parent.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.GridType
        overwrite : bool

        Returns
        -------
        None
        """

        if Grid is None or Grid.TimeCOAPoly is None:
            return  # nothing can be done
        if not overwrite and self.SCPTime is not None:
            return  # nothing should be done

        scp_time = Grid.TimeCOAPoly.Coefs[0, 0]
        self.SCPTime = scp_time

    def _derive_position(self, Position, overwrite=False):
        """
        Derive aperture position parameters, if necessary. Expected to be called by SICD parent.

        Parameters
        ----------
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        overwrite : bool

        Returns
        -------
        None
        """

        if Position is None or Position.ARPPoly is None or self.SCPTime is None:
            return  # nothing can be derived

        # set aperture position, velocity, and acceleration at scptime from position
        # polynomial, if necessary
        poly = Position.ARPPoly
        scptime = self.SCPTime

        if self.ARPPos is None or overwrite:
            self.ARPPos = XYZType.from_array(poly(scptime))
            self.ARPVel = XYZType.from_array(poly.derivative_eval(scptime, 1))
            self.ARPAcc = XYZType.from_array(poly.derivative_eval(scptime, 2))

    def _derive_geometry_parameters(self, GeoData, overwrite=False):
        """
        Expected to be called by SICD parent.

        Parameters
        ----------
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        overwrite : bool

        Returns
        -------
        None
        """

        if GeoData is None or GeoData.SCP is None or GeoData.SCP.ECF is None or \
                self.ARPPos is None or self.ARPVel is None:
            return  # nothing can be derived

        # construct our calculator
        calculator = GeometryCalculator(
            GeoData.SCP.ECF.get_array(), self.ARPPos.get_array(), self.ARPVel.get_array())
        # set all the values
        self._ROV = calculator.ROV
        if self.SideOfTrack is None or overwrite:
            self.SideOfTrack = calculator.SideOfTrack
        if self.SlantRange is None or overwrite:
            self.SlantRange = calculator.SlantRange
        if self.GroundRange is None or overwrite:
            self.GroundRange = calculator.GroundRange
        if self.DopplerConeAng is None or overwrite:
            self.DopplerConeAng = calculator.DopplerConeAng
        graz, inc = calculator.get_graze_and_incidence()
        if self.GrazeAng is None or overwrite:
            self.GrazeAng = graz
        if self.IncidenceAng is None or overwrite:
            self.IncidenceAng = inc
        if self.TwistAng is None or overwrite:
            self.TwistAng = calculator.TwistAng
        self._squint = calculator.SquintAngle
        if self.SlopeAng is None or overwrite:
            self.SlopeAng = calculator.SlopeAng
        if self.AzimAng is None or overwrite:
            self.AzimAng = calculator.AzimAng
        layover, self._layover_magnitude = calculator.get_layover()
        if self.LayoverAng is None or overwrite:
            self.LayoverAng = layover
        self._shadow, self._shadow_magnitude = calculator.get_shadow()

    def rederive(self, Grid, Position, GeoData):
        """
        Rederive all derived quantities.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.GridType
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType

        Returns
        -------
        None
        """

        self._derive_scp_time(Grid, overwrite=True)
        self._derive_position(Position, overwrite=True)
        self._derive_geometry_parameters(GeoData, overwrite=True)

    def check_values(self, GeoData):
        """
        Check derived values for validity.

        Parameters
        ----------
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType

        Returns
        -------
        bool
        """

        if GeoData is None or GeoData.SCP is None or GeoData.SCP.ECF is None or \
                self.ARPPos is None or self.ARPVel is None:
            return True  # nothing can be derived

        # construct our calculator
        calculator = GeometryCalculator(
            GeoData.SCP.ECF.get_array(), self.ARPPos.get_array(), self.ARPVel.get_array())

        cond = True
        if calculator.SideOfTrack != self.SideOfTrack:
            self.log_validity_error(
                'SideOfTrack is expected to be {}, and is populated as {}'.format(
                    calculator.SideOfTrack, self.SideOfTrack))
            cond = False

        for attribute in ['SlantRange', 'GroundRange']:
            val1 = getattr(self, attribute)
            val2 = getattr(calculator, attribute)
            if abs(val1/val2 - 1) > 1e-6:
                self.log_validity_error(
                    'attribute {} is expected to have value {}, but is populated as {}'.format(attribute, val1, val2))
                cond = False

        for attribute in [
                'DopplerConeAng', 'GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng']:
            val1 = getattr(self, attribute)
            val2 = getattr(calculator, attribute)
            if abs(val1 - val2) > 1e-3:
                self.log_validity_error(
                    'attribute {} is expected to have value {}, but is populated as {}'.format(attribute, val1, val2))
                cond = False
        return cond
