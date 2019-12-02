"""
The SCPCOAType definition.
"""

from ._base import Serializable, DEFAULT_STRICT, \
    _StringEnumDescriptor, _FloatDescriptor, \
    _SerializableDescriptor
from ._blocks import XYZType


__classification__ = "UNCLASSIFIED"


class SCPCOAType(Serializable):
    """Center of Aperture (COA) for the Scene Center Point (SCP)."""
    _fields = (
        'SCPTime', 'ARPPos', 'ARPVel', 'ARPAcc', 'SideOfTrack', 'SlantRange', 'GroundRange', 'DopplerConeAng',
        'GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng')
    _required = _fields
    # class variables
    _SIDE_OF_TRACK_VALUES = ('L', 'R')
    # descriptors
    SCPTime = _FloatDescriptor(
        'SCPTime', _required, strict=DEFAULT_STRICT,
        docstring='Center Of Aperture time for the SCP t_COA_SCP, relative to collection '
                  'start in seconds.')  # type: float
    ARPPos = _SerializableDescriptor(
        'ARPPos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Aperture position at t_COA_SCP in ECF.')  # type: XYZType
    ARPVel = _SerializableDescriptor(
        'ARPVel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP Velocity at t_COA_SCP in ECF.')  # type: XYZType
    ARPAcc = _SerializableDescriptor(
        'ARPAcc', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='ARP Acceleration at t_COA_SCP in ECF.')  # type: XYZType
    SideOfTrack = _StringEnumDescriptor(
        'SideOfTrack', _SIDE_OF_TRACK_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Side of track.')  # type: str
    SlantRange = _FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT,
        docstring='Slant range from the ARP to the SCP in meters.')  # type: float
    GroundRange = _FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT,
        docstring='Ground Range from the ARP nadir to the SCP. Distance measured along spherical earth model '
                  'passing through the SCP in meters.')  # type: float
    DopplerConeAng = _FloatDescriptor(
        'DopplerConeAng', _required, strict=DEFAULT_STRICT,
        docstring='The Doppler Cone Angle to SCP at t_COA_SCP in degrees.')  # type: float
    GrazeAng = _FloatDescriptor(
        'GrazeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Grazing Angle between the SCP Line of Sight (LOS) and Earth Tangent Plane (ETP).')  # type: float
    IncidenceAng = _FloatDescriptor(
        'IncidenceAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Incidence Angle between the SCP LOS and ETP normal.')  # type: float
    TwistAng = _FloatDescriptor(
        'TwistAng', _required, strict=DEFAULT_STRICT, bounds=(-90., 90.),
        docstring='Angle between cross range in the ETP and cross range in the slant plane.')  # type: float
    SlopeAng = _FloatDescriptor(
        'SlopeAng', _required, strict=DEFAULT_STRICT, bounds=(0., 90.),
        docstring='Slope Angle from the ETP to the slant plane at t_COA_SCP.')  # type: float
    AzimAng = _FloatDescriptor(
        'AzimAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.),
        docstring='Angle from north to the line from the SCP to the ARP Nadir at COA. Measured '
                  'clockwise in the ETP.')  # type: float
    LayoverAng = _FloatDescriptor(
        'LayoverAng', _required, strict=DEFAULT_STRICT, bounds=(0., 360.),
        docstring='Angle from north to the layover direction in the ETP at COA. Measured '
                  'clockwise in the ETP.')  # type: float
