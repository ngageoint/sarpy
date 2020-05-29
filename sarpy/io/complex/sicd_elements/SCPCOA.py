# -*- coding: utf-8 -*-
"""
The SCPCOAType definition.
"""

import logging

import numpy
from numpy.linalg import norm

from .base import Serializable, DEFAULT_STRICT, \
    _StringEnumDescriptor, _FloatDescriptor, _SerializableDescriptor
from .blocks import XYZType

from sarpy.geometry import geocoords


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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
    SCPTime = _FloatDescriptor(
        'SCPTime', _required, strict=DEFAULT_STRICT,
        docstring='*Center Of Aperture time for the SCP (t_COA_SCP)*, relative to collection '
                  'start in seconds.')  # type: float
    ARPPos = _SerializableDescriptor(
        'ARPPos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Aperture position at *t_COA_SCP* in ECF coordinates.')  # type: XYZType
    ARPVel = _SerializableDescriptor(
        'ARPVel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP Velocity at *t_COA_SCP* in ECF coordinates.')  # type: XYZType
    ARPAcc = _SerializableDescriptor(
        'ARPAcc', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP Acceleration at *t_COA_SCP* in ECF coordinates.')  # type: XYZType
    SideOfTrack = _StringEnumDescriptor(
        'SideOfTrack', _SIDE_OF_TRACK_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Side of track.')  # type: str
    SlantRange = _FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT,
        docstring='Slant range from the aperture to the *SCP* in meters.')  # type: float
    GroundRange = _FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT,
        docstring='Ground Range from the aperture nadir to the *SCP*. Distance measured along spherical earth model '
                  'passing through the *SCP* in meters.')  # type: float
    DopplerConeAng = _FloatDescriptor(
        'DopplerConeAng', _required, strict=DEFAULT_STRICT,
        docstring='The Doppler Cone Angle to SCP at *t_COA_SCP* in degrees.')  # type: float
    GrazeAng = _FloatDescriptor(
        'GrazeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Grazing Angle between the SCP *Line of Sight (LOS)* and *Earth Tangent Plane (ETP)*.')  # type: float
    IncidenceAng = _FloatDescriptor(
        'IncidenceAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Incidence Angle between the *LOS* and *ETP* normal.')  # type: float
    TwistAng = _FloatDescriptor(
        'TwistAng', _required, strict=DEFAULT_STRICT, bounds=(-90., 90.),
        docstring='Angle between cross range in the *ETP* and cross range in the slant plane.')  # type: float
    SlopeAng = _FloatDescriptor(
        'SlopeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Slope Angle from the *ETP* to the slant plane at *t_COA_SCP*.')  # type: float
    AzimAng = _FloatDescriptor(
        'AzimAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.),
        docstring='Angle from north to the line from the *SCP* to the aperture nadir at *COA*. Measured '
                  'clockwise in the *ETP*.')  # type: float
    LayoverAng = _FloatDescriptor(
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

        return float(numpy.sin(numpy.deg2rad(self.DopplerConeAng))/self.ROV)

    @property
    def MultipathGround(self):
        """
        float: The anticipated angle of multipath features on the ground in degrees.
        """

        return numpy.rad2deg(
            -numpy.arctan(numpy.tan(numpy.deg2rad(self.TwistAng))*numpy.sin(numpy.deg2rad(self.GrazeAng))))

    @property
    def Multipath(self):
        """
        float: The anticipated angle of multipath features in degrees.
        """

        return numpy.mod(self.AzimAng - 180 + self.MultipathGround, 360)

    @property
    def Shadow(self):
        """
        float: The anticipated angle of shadow features in degrees.
        """

        return numpy.mod(self.AzimAng - 180, 360)

    def _derive_scp_time(self, Grid):
        """
        Expected to be called by SICD parent.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.GridType

        Returns
        -------
        None
        """

        if Grid is None or Grid.TimeCOAPoly is None:
            return  # nothing can be done

        scp_time = Grid.TimeCOAPoly.Coefs[0, 0]
        if self.SCPTime is None:
            self.SCPTime = scp_time
        elif abs(self.SCPTime - scp_time) > 1e-8:  # useful tolerance?
            logging.warning(
                'The SCPTime is derived from Grid.TimeCOAPoly as {}, and it is set '
                'as {}'.format(scp_time, self.SCPTime))

    def _derive_position(self, Position):
        """
        Derive aperture position parameters, if necessary. Expected to be called by SICD parent.

        Parameters
        ----------
        Position : sarpy.io.complex.sicd_elements.Position.PositionType

        Returns
        -------
        None
        """

        if Position is None or Position.ARPPoly is None or self.SCPTime is None:
            return  # nothing can be derived

        # set aperture position, velocity, and acceleration at scptime from position polynomial, if necessary
        poly = Position.ARPPoly
        scptime = self.SCPTime

        if self.ARPPos is None:
            self.ARPPos = XYZType.from_array(poly(scptime))
        if self.ARPVel is None:
            self.ARPVel = XYZType.from_array(poly.derivative_eval(scptime, 1))
        if self.ARPAcc is None:
            self.ARPAcc = XYZType.from_array(poly.derivative_eval(scptime, 2))

    def _derive_geometry_parameters(self, GeoData):
        """
        Expected to be called by SICD parent.

        Parameters
        ----------
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType

        Returns
        -------
        None
        """

        if GeoData is None or GeoData.SCP is None or GeoData.SCP.ECF is None or \
                self.ARPPos is None or self.ARPVel is None:
            return  # nothing can be derived

        SCP = GeoData.SCP.ECF.get_array()
        ARP = self.ARPPos.get_array()
        ARP_vel = self.ARPVel.get_array()
        LOS = (SCP - ARP)
        self._ROV = float(numpy.linalg.norm(LOS)/numpy.linalg.norm(ARP_vel))
        # unit vector versions
        uSCP = SCP/norm(SCP)
        uARP = ARP/norm(ARP)
        uARP_vel = ARP_vel/norm(ARP_vel)
        uLOS = LOS/norm(LOS)
        # cross product junk
        left = numpy.cross(uARP, uARP_vel)
        look = numpy.sign(numpy.dot(left, uLOS))

        sot = 'R' if look < 0 else 'L'
        if self.SideOfTrack is None:
            self.SideOfTrack = sot
        elif self.SideOfTrack != sot:
            logging.error(
                'In SCPCOAType, the derived value for SideOfTrack is {} and the set '
                'value is {}'.format(sot, self.SideOfTrack))

        slant_range = numpy.linalg.norm(LOS)
        if self.SlantRange is None:
            self.SlantRange = slant_range
        elif abs(self.SlantRange - slant_range) > 1e-5:  # sensible tolerance?
            logging.error('In SCPCOAType, the derived value for SlantRange is {} and the set '
                          'value is {}'.format(slant_range, self.SlantRange))

        ground_range = norm(SCP)*numpy.arccos(numpy.dot(uSCP, uARP))
        if self.GroundRange is None:
            self.GroundRange = ground_range
        elif abs(self.GroundRange - ground_range) > 1e-5:  # sensible tolerance?
            logging.error('In SCPCOAType, the derived value for GroundRange is {} and the set '
                          'value is {}'.format(ground_range, self.GroundRange))

        doppler_cone = numpy.rad2deg(numpy.arccos(numpy.dot(uARP_vel, uLOS)))
        if self.DopplerConeAng is None:
            self.DopplerConeAng = doppler_cone
        elif abs(self.DopplerConeAng - doppler_cone) > 1e-5:  # sensible tolerance?
            logging.error('In SCPCOAType, the derived value for DopplerConeAng is {} and the set '
                          'value is {}'.format(doppler_cone, self.DopplerConeAng))

        # Earth Tangent Plane (ETP) at the SCP is the plane tangent to the surface of constant height
        # above the WGS 84 ellipsoid (HAE) that contains the SCP. The ETP is an approximation to the
        # ground plane at the SCP.
        ETP = geocoords.wgs_84_norm(SCP)
        graze_ang = numpy.rad2deg(numpy.arcsin(numpy.dot(ETP, -uLOS)))
        if self.GrazeAng is None:
            self.GrazeAng = graze_ang
        elif abs(self.GrazeAng - graze_ang) > 1e-5:  # sensible tolerance?
            logging.error('In SCPCOAType, the derived value for GrazeAng is {} and the set '
                          'value is {}'.format(graze_ang, self.GrazeAng))

        if self.IncidenceAng is None:
            self.IncidenceAng = 90 - self.GrazeAng

        # slant plane unit normal
        uSPZ = look*numpy.cross(ARP_vel, uLOS)
        uSPZ /= norm(uSPZ)
        # perpendicular component of range vector wrt the ground plane
        uGPX = -uLOS + numpy.dot(ETP, uLOS)*ETP
        uGPX /= norm(uGPX)
        uGPY = numpy.cross(ETP, uGPX)  # already unit vector
        twist_ang = -numpy.rad2deg(numpy.arcsin(numpy.dot(uGPY, uSPZ)))
        if self.TwistAng is None:
            self.TwistAng = twist_ang
        elif abs(self.TwistAng - twist_ang) > 1e-5:  # sensible tolerance?
            logging.error('In SCPCOAType, the derived value for TwistAng is {} and the set '
                          'value is {}'.format(twist_ang, self.TwistAng))

        slope_ang = numpy.rad2deg(numpy.arccos(numpy.dot(ETP, uSPZ)))
        if self.SlopeAng is None:
            self.SlopeAng = slope_ang
        elif abs(self.SlopeAng - slope_ang) > 1e-5:  # sensible tolerance?
            logging.error('In SCPCOAType, the derived value for SlopeAng is {} and the set '
                          'value is {}'.format(slope_ang, self.SlopeAng))

        # perpendicular component of north wrt the ground plane
        NORTH = numpy.array([0, 0, 1]) - ETP[2]*ETP
        uNORTH = NORTH/norm(NORTH)
        uEAST = numpy.cross(uNORTH, ETP)  # already unit vector
        azim_ang = numpy.rad2deg(numpy.arctan2(numpy.dot(uGPX, uEAST), numpy.dot(uGPX, uNORTH)))
        azim_ang = azim_ang if azim_ang > 0 else azim_ang + 360
        if self.AzimAng is None:
            self.AzimAng = azim_ang
        elif abs(self.AzimAng - azim_ang) > 1e-5:  # sensible tolerance?
            logging.error('In SCPCOAType, the derived value for AzimAng is {} and the set '
                          'value is {}'.format(azim_ang, self.AzimAng))

        # perpendicular component of ground plane wrt slant plane
        layover_ground = ETP - numpy.dot(ETP, uSPZ)*uSPZ
        layover_ang = numpy.rad2deg(numpy.arctan2(numpy.dot(layover_ground, uEAST), numpy.dot(layover_ground, uNORTH)))
        layover_ang = layover_ang if layover_ang > 0 else layover_ang + 360
        if self.LayoverAng is None:
            self.LayoverAng = layover_ang
        elif abs(self.LayoverAng - layover_ang) > 1e-5:  # sensible tolerance?
            logging.error('In SCPCOAType, the derived value for LayoverAng is {} and the set '
                          'value is {}'.format(layover_ang, self.LayoverAng))
