"""
The reference geometry parameters definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union

import numpy

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import FloatDescriptor, StringEnumDescriptor, \
    SerializableDescriptor
from sarpy.io.complex.sicd_elements.blocks import XYZType

from .base import DEFAULT_STRICT, FLOAT_FORMAT


class SRPType(Serializable):
    """
    The SRP position for the reference vector of the reference channel.
    """

    _fields = ('ECF', 'IAC')
    _required = _fields
    # descriptors
    ECF = SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='SRP position in ECF coordinates.')  # type: XYZType
    IAC = SerializableDescriptor(
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
        'SlantRange': FLOAT_FORMAT, 'GroundRange': FLOAT_FORMAT, 'DopplerConeAngle': FLOAT_FORMAT,
        'GrazeAngle': FLOAT_FORMAT, 'IncidenceAngle': FLOAT_FORMAT, 'AzimuthAngle': FLOAT_FORMAT}
    # descriptors
    SideOfTrack = StringEnumDescriptor(
        'SideOfTrack', ('L', 'R'), _required, strict=DEFAULT_STRICT,
        docstring='Side of Track parameter for the collection.')  # type: str
    SlantRange = FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Slant range from the ARP to the SRP.')  # type: float
    GroundRange = FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Ground range from the ARP to the SRP.')  # type: float
    DopplerConeAngle = FloatDescriptor(
        'DopplerConeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 180),
        docstring='Doppler Cone Angle between ARP velocity and deg SRP Line of '
                  'Sight (LOS).')  # type: float
    GrazeAngle = FloatDescriptor(
        'GrazeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Grazing angle for the ARP to SRP LOS and the deg Earth Tangent '
                  'Plane (ETP) at the SRP.')  # type: float
    IncidenceAngle = FloatDescriptor(
        'IncidenceAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Incidence angle for the ARP to SRP LOS and the Earth Tangent '
                  'Plane (ETP) at the SRP.')  # type: float
    AzimuthAngle = FloatDescriptor(
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
        'SlantRange': FLOAT_FORMAT, 'GroundRange': FLOAT_FORMAT, 'DopplerConeAngle': FLOAT_FORMAT,
        'GrazeAngle': FLOAT_FORMAT, 'IncidenceAngle': FLOAT_FORMAT, 'AzimuthAngle': FLOAT_FORMAT,
        'TwistAngle': FLOAT_FORMAT, 'SlopeAngle': FLOAT_FORMAT, 'LayoverAngle': FLOAT_FORMAT}
    # descriptors
    ARPPos = SerializableDescriptor(
        'ARPPos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP position in ECF coordinates.')  # type: XYZType
    ARPVel = SerializableDescriptor(
        'ARPVel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP velocity in ECF coordinates.')  # type: XYZType
    TwistAngle = FloatDescriptor(
        'TwistAngle', _required, strict=DEFAULT_STRICT, bounds=(-90, 90),
        docstring='Twist angle between cross range in the ETP and cross range in '
                  'the slant plane at the SRP.')  # type: float
    SlopeAngle = FloatDescriptor(
        'SlopeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Angle between the ETP normal (uUP) and the slant plane normal '
                  '(uSPN) at the SRP.')  # type: float
    LayoverAngle = FloatDescriptor(
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
                -numpy.arctan(numpy.tan(numpy.deg2rad(self.TwistAngle)) *
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
        'Time': FLOAT_FORMAT, 'SlantRange': FLOAT_FORMAT, 'GroundRange': FLOAT_FORMAT, 'DopplerConeAngle': FLOAT_FORMAT,
        'GrazeAngle': FLOAT_FORMAT, 'IncidenceAngle': FLOAT_FORMAT, 'AzimuthAngle': FLOAT_FORMAT}
    # descriptors
    Time = FloatDescriptor(
        'Time', _required, strict=DEFAULT_STRICT,
        docstring='The transmit or receive time for the vector.')  # type: float
    Pos = SerializableDescriptor(
        'Pos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='APC position in ECF coordinates.')  # type: XYZType
    Vel = SerializableDescriptor(
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
        'AzimuthAngle': FLOAT_FORMAT, 'AzimuthAngleRate': FLOAT_FORMAT, 'BistaticAngle': FLOAT_FORMAT,
        'BistaticAngleRate': FLOAT_FORMAT, 'GrazeAngle': FLOAT_FORMAT, 'TwistAngle': FLOAT_FORMAT,
        'SlopeAngle': FLOAT_FORMAT, 'LayoverAngle': FLOAT_FORMAT}

    # descriptors
    AzimuthAngle = FloatDescriptor(
        'AzimuthAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 360),
        docstring='Angle from north to the projection of the Bistatic pointing vector '
                  '(bP) into the ETP. Measured clockwise from +North toward '
                  '+East.')  # type: float
    AzimuthAngleRate = FloatDescriptor(
        'AzimuthAngleRate', _required, strict=DEFAULT_STRICT,
        docstring='Instantaneous rate of change of the Azimuth Angle '
                  ':math:`d(AZIM)/dt`.')  # type: float
    BistaticAngle = FloatDescriptor(
        'BistaticAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 180),
        docstring='Bistatic angle (Beta) between unit vector from SRP to transmit APC '
                  '(uXmt) and the unit vector from the SRP to the receive '
                  'APC (uRcv).')  # type: float
    BistaticAngleRate = FloatDescriptor(
        'BistaticAngleRate', _required, strict=DEFAULT_STRICT,
        docstring='Instantaneous rate of change of the bistatic angle '
                  ':math:`d(Beta)/dt)`.')  # type: float
    GrazeAngle = FloatDescriptor(
        'GrazeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Angle between the bistatic pointing vector and the ETP at the '
                  'SRP.')  # type: float
    TwistAngle = FloatDescriptor(
        'TwistAngle', _required, strict=DEFAULT_STRICT, bounds=(-90, 90),
        docstring='Angle between cross range in the ETP at the SRP and cross range '
                  'in the instantaneous plane of maximum bistatic resolution. '
                  'Note - For monostatic imaging, the plane of maximum resolution is '
                  'the instantaneous slant plane.')  # type: float
    SlopeAngle = FloatDescriptor(
        'SlopeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Angle between the ETP normal and the normal to the instantaneous '
                  'plane of maximum bistatic resolution.')  # type: float
    LayoverAngle = FloatDescriptor(
        'LayoverAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 360),
        docstring='Angle from north to the bistatic layover direction in the ETP. '
                  'Measured clockwise from +North toward +East.')  # type: float
    TxPlatform = SerializableDescriptor(
        'TxPlatform', BistaticTxRcvType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the Transmit platform.')  # type: BistaticTxRcvType
    RcvPlatform = SerializableDescriptor(
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
    _numeric_format = {'ReferenceTime': FLOAT_FORMAT, 'SRPCODTime': FLOAT_FORMAT, 'SRPDwellTime': FLOAT_FORMAT}
    # descriptors
    SRP = SerializableDescriptor(
        'SRP', SRPType, _required, strict=DEFAULT_STRICT,
        docstring='The SRP position for the reference vector of the reference '
                  'channel.')  # type: SRPType
    ReferenceTime = FloatDescriptor(
        'ReferenceTime', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Reference time for the selected reference vector, in '
                  'seconds.')  # type: float
    SRPCODTime = FloatDescriptor(
        'SRPCODTime', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The COD Time for point on the reference surface, in '
                  'seconds.')  # type: float
    SRPDwellTime = FloatDescriptor(
        'SRPDwellTime', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    Monostatic = SerializableDescriptor(
        'Monostatic', MonostaticType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for monstatic collection.')  # type: Union[None, MonostaticType]
    Bistatic = SerializableDescriptor(
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
