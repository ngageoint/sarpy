# -*- coding: utf-8 -*-
"""
The reference geometry parameters definition.
"""

from typing import Union

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _FloatDescriptor, \
    _StringEnumDescriptor, _SerializableDescriptor
from sarpy.io.complex.sicd_elements.blocks import XYZType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SRPType(Serializable):
    """
    The SRP position for the reference vector of the reference channel.
    """

    _fields = ('ECF', 'IAC')
    _required = _fields
    # descriptors
    ECF = _SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='SRP position in ECF coordinates.')  # type: XYZType
    IAC = _SerializableDescriptor(
        'IAC', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='SRP position in Image Area Coordinates.')  # type: XYZType

    def __init__(self, ECF=None, IAC=None, **kwargs):
        """

        Parameters
        ----------
        ECF : XYZType|numpy.ndarray|list|tuple
        IAC : XYZType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ECF = ECF
        self.IAC = IAC
        super(SRPType, self).__init__(**kwargs)


class ReferenceGeometryCore(Serializable):
    """
    The base reference geometry implementation.
    """

    _fields = (
        'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAngle',
        'GrazeAngle', 'IncidenceAngle', 'AzimuthAngle')
    _required = _fields
    _numeric_format = {
        'SlantRange': '0.16G', 'GroundRange': '0.16G', 'DopplerConeAngle': '0.16G',
        'GrazeAngle': '0.16G', 'IncidenceAngle': '0.16G', 'AzimuthAngle': '0.16G'}
    # descriptors
    SideOfTrack = _StringEnumDescriptor(
        'SideOfTrack', ('L', 'R'), _required, strict=DEFAULT_STRICT,
        docstring='Side of Track parameter for the collection.')  # type: str
    SlantRange = _FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Slant range from the ARP to the SRP.')  # type: float
    GroundRange = _FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Ground range from the ARP to the SRP.')  # type: float
    DopplerConeAngle = _FloatDescriptor(
        'DopplerConeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 180),
        docstring='Doppler Cone Angle between ARP velocity and deg SRP Line of '
                  'Sight (LOS).')  # type: float
    GrazeAngle = _FloatDescriptor(
        'GrazeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Grazing angle for the ARP to SRP LOS and the deg Earth Tangent '
                  'Plane (ETP) at the SRP.')  # type: float
    IncidenceAngle = _FloatDescriptor(
        'IncidenceAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Incidence angle for the ARP to SRP LOS and the Earth Tangent '
                  'Plane (ETP) at the SRP.')  # type: float
    AzimuthAngle = _FloatDescriptor(
        'AzimuthAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 360),
        docstring='Angle from north to the line from the SRP to the ARP ETP '
                  'Nadir (i.e. North to +GPX). Measured clockwise from North '
                  'toward East.')  # type: float

    def __init__(self, SideOfTrack=None, SlantRange=None, GroundRange=None,
                 DopplerConeAngle=None, GrazeAngle=None, IncidenceAngle=None,
                 AzimuthAngle=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SideOfTrack = SideOfTrack
        self.SlantRange = SlantRange
        self.GroundRange = GroundRange
        self.DopplerConeAngle = DopplerConeAngle
        self.GrazeAngle = GrazeAngle
        self.IncidenceAngle = IncidenceAngle
        self.AzimuthAngle = AzimuthAngle
        super(ReferenceGeometryCore, self).__init__(**kwargs)

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
        return -1 if self.SideOfTrack == 'R' else 1


class MonostaticType(ReferenceGeometryCore):
    """
    Parameters for monostatic collection.
    """

    _fields = (
        'ARPPos', 'ARPVel',
        'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAngle',
        'GrazeAngle', 'IncidenceAngle', 'AzimuthAngle',
        'TwistAngle', 'SlopeAngle', 'LayoverAngle')
    _required = _fields
    _numeric_format = {
        'SlantRange': '0.16G', 'GroundRange': '0.16G', 'DopplerConeAngle': '0.16G',
        'GrazeAngle': '0.16G', 'IncidenceAngle': '0.16G', 'AzimuthAngle': '0.16G',
        'TwistAngle': '0.16G', 'SlopeAngle': '0.16G', 'LayoverAngle': '0.16G'}
    # descriptors
    ARPPos = _SerializableDescriptor(
        'ARPPos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP position in ECF coordinates.')  # type: XYZType
    ARPVel = _SerializableDescriptor(
        'ARPVel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP velocity in ECF coordinates.')  # type: XYZType
    TwistAngle = _FloatDescriptor(
        'TwistAngle', _required, strict=DEFAULT_STRICT, bounds=(-90, 90),
        docstring='Twist angle between cross range in the ETP and cross range in '
                  'the slant plane at the SRP.')  # type: float
    SlopeAngle = _FloatDescriptor(
        'SlopeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Angle between the ETP normal (uUP) and the slant plane normal '
                  '(uSPN) at the SRP.')  # type: float
    LayoverAngle = _FloatDescriptor(
        'LayoverAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 360),
        docstring='Angle from north to the layover direction in the ETP. Measured '
                  'clockwise from +North toward +East.')  # type: float

    def __init__(self, ARPPos=None, ARPVel=None,
                 SideOfTrack=None, SlantRange=None, GroundRange=None, DopplerConeAngle=None,
                 GrazeAngle=None, IncidenceAngle=None, AzimuthAngle=None,
                 TwistAngle=None, SlopeAngle=None, LayoverAngle=None, **kwargs):
        """

        Parameters
        ----------
        ARPPos : XYZType|numpy.ndarray|list|tuple
        ARPVel : XYZType|numpy.ndarray|list|tuple
        SideOfTrack : float
        SlantRange : float
        GroundRange : float
        DopplerConeAngle : float
        GrazeAngle : float
        IncidenceAngle : float
        AzimuthAngle : float
        TwistAngle : float
        SlopeAngle : float
        LayoverAngle : float
        kwargs
        """
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ARPPos = ARPPos
        self.ARPVel = ARPVel
        self.TwistAngle = TwistAngle
        self.SlopeAngle = SlopeAngle
        self.LayoverAngle = LayoverAngle
        super(MonostaticType, self).__init__(
            SideOfTrack=SideOfTrack, SlantRange=SlantRange, GroundRange=GroundRange,
            DopplerConeAngle=DopplerConeAngle, GrazeAngle=GrazeAngle, IncidenceAngle=IncidenceAngle,
            AzimuthAngle=AzimuthAngle, **kwargs)

    @property
    def MultipathGround(self):
        """
        float: The anticipated angle of multipath features on the ground in degrees.
        """

        if self.TwistAngle is None:
            return None
        else:
            return numpy.rad2deg(
                -numpy.arctan(numpy.tan(numpy.deg2rad(self.TwistAngle))*
                              numpy.sin(numpy.deg2rad(self.GrazeAngle))))

    @property
    def Multipath(self):
        """
        float: The anticipated angle of multipath features in degrees.
        """
        if self.MultipathGround is None:
            return None
        else:
            return numpy.mod(self.AzimuthAngle - 180 + self.MultipathGround, 360)

    @property
    def Shadow(self):
        """
        float: The anticipated angle of shadow features in degrees.
        """

        return numpy.mod(self.AzimuthAngle - 180, 360)


class BistaticTxRcvType(ReferenceGeometryCore):
    """
    Parameters that describe the Transmit/Receive platforms.
    """

    _fields = (
        'Time', 'Pos', 'Vel',
        'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAngle',
        'GrazeAngle', 'IncidenceAngle', 'AzimuthAngle')
    _required = _fields
    _numeric_format = {
        'Time': '0.16G', 'SlantRange': '0.16G', 'GroundRange': '0.16G', 'DopplerConeAngle': '0.16G',
        'GrazeAngle': '0.16G', 'IncidenceAngle': '0.16G', 'AzimuthAngle': '0.16G'}
    # descriptors
    Time = _FloatDescriptor(
        'Time', _required, strict=DEFAULT_STRICT,
        docstring='The transmit or receive time for the vector.')  # type: float
    Pos = _SerializableDescriptor(
        'Pos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='APC position in ECF coordinates.')  # type: XYZType
    Vel = _SerializableDescriptor(
        'Vel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='APC velocity in ECF coordinates.')  # type: XYZType

    def __init__(self, Time=None, Pos=None, Vel=None,
                 SideOfTrack=None, SlantRange=None, GroundRange=None, DopplerConeAngle=None,
                 GrazeAngle=None, IncidenceAngle=None, AzimuthAngle=None, **kwargs):
        """

        Parameters
        ----------
        Time : float
        Pos : XYZType|numpy.ndarray|list|tuple
        Vel : XYZType|numpy.ndarray|list|tuple
        SideOfTrack : float
        SlantRange : float
        GroundRange : float
        DopplerConeAngle : float
        GrazeAngle : float
        IncidenceAngle : float
        AzimuthAngle : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Time = Time
        self.Pos = Pos
        self.Vel = Vel
        super(BistaticTxRcvType, self).__init__(
            SideOfTrack=SideOfTrack, SlantRange=SlantRange, GroundRange=GroundRange,
            DopplerConeAngle=DopplerConeAngle, GrazeAngle=GrazeAngle, IncidenceAngle=IncidenceAngle,
            AzimuthAngle=AzimuthAngle, **kwargs)


class BistaticType(Serializable):
    """

    """

    _fields = (
        'AzimuthAngle', 'AzimuthAngleRate', 'BistaticAngle', 'BistaticAngleRate',
        'GrazeAngle', 'TwistAngle', 'SlopeAngle', 'LayoverAngle',
        'TxPlatform', 'RcvPlatform')
    _required = _fields
    _numeric_format = {
        'AzimuthAngle': '0.16G', 'AzimuthAngleRate': '0.16G', 'BistaticAngle': '0.16G',
        'BistaticAngleRate': '0.16G', 'GrazeAngle': '0.16G', 'TwistAngle': '0.16G',
        'SlopeAngle': '0.16G', 'LayoverAngle': '0.16G'}

    # descriptors
    AzimuthAngle = _FloatDescriptor(
        'AzimuthAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 360),
        docstring='Angle from north to the projection of the Bistatic pointing vector '
                  '(bP) into the ETP. Measured clockwise from +North toward '
                  '+East.')  # type: float
    AzimuthAngleRate = _FloatDescriptor(
        'AzimuthAngleRate', _required, strict=DEFAULT_STRICT,
        docstring='Instantaneous rate of change of the Azimuth Angle '
                  ':math:`d(AZIM)/dt`.')  # type: float
    BistaticAngle = _FloatDescriptor(
        'BistaticAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 180),
        docstring='Bistatic angle (Beta) between unit vector from SRP to transmit APC '
                  '(uXmt) and the unit vector from the SRP to the receive '
                  'APC (uRcv).')  # type: float
    BistaticAngleRate = _FloatDescriptor(
        'BistaticAngleRate', _required, strict=DEFAULT_STRICT,
        docstring='Instantaneous rate of change of the bistatic angle '
                  ':math:`d(Beta)/dt)`.')  # type: float
    GrazeAngle = _FloatDescriptor(
        'GrazeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Angle between the bistatic pointing vector and the ETP at the '
                  'SRP.')  # type: float
    TwistAngle = _FloatDescriptor(
        'TwistAngle', _required, strict=DEFAULT_STRICT, bounds=(-90, 90),
        docstring='Angle between cross range in the ETP at the SRP and cross range '
                  'in the instantaneous plane of maximum bistatic resolution. '
                  'Note - For monostatic imaging, the plane of maximum resolution is '
                  'the instantaneous slant plane.')  # type: float
    SlopeAngle = _FloatDescriptor(
        'SlopeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Angle between the ETP normal and the normal to the instantaneous '
                  'plane of maximum bistatic resolution.')  # type: float
    LayoverAngle = _FloatDescriptor(
        'LayoverAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 360),
        docstring='Angle from north to the bistatic layover direction in the ETP. '
                  'Measured clockwise from +North toward +East.')  # type: float
    TxPlatform = _SerializableDescriptor(
        'TxPlatform', BistaticTxRcvType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the Transmit platform.')  # type: BistaticTxRcvType
    RcvPlatform = _SerializableDescriptor(
        'RcvPlatform', BistaticTxRcvType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the Receive platform.')  # type: BistaticTxRcvType

    def __init__(self, AzimuthAngle=None, AzimuthAngleRate=None, BistaticAngle=None,
                 BistaticAngleRate=None, GrazeAngle=None, TwistAngle=None,
                 SlopeAngle=None, LayoverAngle=None, TxPlatform=None,
                 RcvPlatform=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.AzimuthAngle = AzimuthAngle
        self.AzimuthAngleRate = AzimuthAngleRate
        self.BistaticAngle = BistaticAngle
        self.BistaticAngleRate = BistaticAngleRate
        self.GrazeAngle = GrazeAngle
        self.TwistAngle = TwistAngle
        self.SlopeAngle = SlopeAngle
        self.LayoverAngle = LayoverAngle
        self.TxPlatform = TxPlatform
        self.RcvPlatform = RcvPlatform
        super(BistaticType, self).__init__(**kwargs)


class ReferenceGeometryType(Serializable):
    """
    Parameters that describe the collection geometry for the reference vector
    of the reference channel.
    """

    _fields = ('SRP', 'ReferenceTime', 'SRPCODTime', 'SRPDwellTime', 'Monostatic', 'Bistatic')
    _required = ('SRP', 'ReferenceTime', 'SRPCODTime', 'SRPDwellTime')
    _choice = ({'required': True, 'collection': ('Monostatic', 'Bistatic')}, )
    _numeric_format = {'ReferenceTime': '0.16G', 'SRPCODTime': '0.16G'}
    # descriptors
    SRP = _SerializableDescriptor(
        'SRP', SRPType, _required, strict=DEFAULT_STRICT,
        docstring='The SRP position for the reference vector of the reference '
                  'channel.')  # type: SRPType
    ReferenceTime = _FloatDescriptor(
        'ReferenceTime', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Reference time for the selected reference vector, in '
                  'seconds.')  # type: float
    SRPCODTime = _FloatDescriptor(
        'SRPCODTime', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The COD Time for point on the reference surface, in '
                  'seconds.')  # type: float
    SRPDwellTime = _FloatDescriptor(
        'SRPDwellTime', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    Monostatic = _SerializableDescriptor(
        'Monostatic', MonostaticType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for monstatic collection.')  # type: Union[None, MonostaticType]
    Bistatic = _SerializableDescriptor(
        'Bistatic', BistaticType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for bistatic collection.')  # type: Union[None, BistaticType]

    def __init__(self, SRP=None, ReferenceTime=None, SRPCODTime=None, SRPDwellTime=None,
                 Monostatic=None, Bistatic=None, **kwargs):
        """

        Parameters
        ----------
        SRP : SRPType
        ReferenceTime : float
        SRPCODTime : float
        SRPDwellTime : float
        Monostatic : None|MonostaticType
        Bistatic : None|BistaticType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SRP = SRP
        self.ReferenceTime = ReferenceTime
        self.SRPCODTime = SRPCODTime
        self.SRPDwellTime = SRPDwellTime
        self.Monostatic = Monostatic
        self.Bistatic = Bistatic
        super(ReferenceGeometryType, self).__init__(**kwargs)
