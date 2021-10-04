"""
The reference geometry parameters definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Michael Stewart, Valkyrie")


import numpy

from sarpy.io.xml.base import Serializable, parse_serializable
from sarpy.io.xml.descriptors import FloatDescriptor, StringEnumDescriptor, \
    SerializableDescriptor
from sarpy.io.complex.sicd_elements.blocks import XYZType, LatLonHAEType
from sarpy.geometry.geocoords import geodetic_to_ecf, ecf_to_geodetic

from .base import DEFAULT_STRICT


class CRPType(Serializable):
    """
    The CRP position for the reference vector of the reference channel.
    """

    _fields = ('ECF', 'LLH')
    _required = _fields
    _ECF = None
    _LLH = None

    def __init__(self, ECF=None, LLH=None, **kwargs):
        """
        To avoid the potential of inconsistent state, ECF and LLH are not simultaneously
        used. If ECF is provided, it is used to populate LLH. Otherwise, if LLH is provided,
        then it is used the populate ECF.

        Parameters
        ----------
        ECF : XYZType|numpy.ndarray|list|tuple
        LLH : LatLonHAEType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        if ECF is not None:
            self.ECF = ECF
        elif LLH is not None:
            self.LLH = LLH
        super(CRPType, self).__init__(**kwargs)

    @property
    def ECF(self):  # type: () -> XYZType
        """
        XYZType: The CRP Position ECF coordinates.
        """

        return self._ECF

    @ECF.setter
    def ECF(self, value):
        if value is not None:
            self._ECF = parse_serializable(value, 'ECF', self, XYZType)
            self._LLH = LatLonHAEType.from_array(ecf_to_geodetic(self._ECF.get_array()))

    @property
    def LLH(self):  # type: () -> LatLonHAEType
        """
        LatLonHAEType: The CRP Position in WGS-84 coordinates.
        """

        return self._LLH

    @LLH.setter
    def LLH(self, value):
        if value is not None:
            self._LLH = parse_serializable(value, 'LLH', self, LatLonHAEType)
            self._ECF = XYZType.from_array(geodetic_to_ecf(self._LLH.get_array(order='LAT')))


class RcvParametersType(Serializable):
    """
    The receive parameters geometry implementation.
    """

    _fields = (
        'RcvTime', 'RcvPos', 'RcvVel', 'SideOfTrack', 'SlantRange', 'GroundRange',
        'DopplerConeAngle', 'GrazeAngle', 'IncidenceAngle', 'AzimuthAngle')
    _required = _fields
    _numeric_format = {
        'SlantRange': '0.16G', 'GroundRange': '0.16G', 'DopplerConeAngle': '0.16G',
        'GrazeAngle': '0.16G', 'IncidenceAngle': '0.16G', 'AzimuthAngle': '0.16G'}
    # descriptors
    RcvTime = FloatDescriptor(
        'RcvTime', _required, strict=DEFAULT_STRICT,
        docstring='Receive time for the first sample for the reference vector.')  # type: float
    RcvPos = SerializableDescriptor(
        'RcvPos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='APC position in ECF coordinates.')  # type: XYZType
    RcvVel = SerializableDescriptor(
        'RcvVel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='APC velocity in ECF coordinates.')  # type: XYZType
    SideOfTrack = StringEnumDescriptor(
        'SideOfTrack', ('L', 'R'), _required, strict=DEFAULT_STRICT,
        docstring='Side of Track parameter for the collection.')  # type: str
    SlantRange = FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Slant range from the APC to the CRP.')  # type: float
    GroundRange = FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Ground range from the APC nadir to the CRP.')  # type: float
    DopplerConeAngle = FloatDescriptor(
        'DopplerConeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 180),
        docstring='Doppler Cone Angle between APC velocity and deg CRP Line of '
                  'Sight (LOS).')  # type: float
    GrazeAngle = FloatDescriptor(
        'GrazeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Grazing angle for the APC to CRP LOS and the Earth Tangent '
                  'Plane (ETP) at the CRP.')  # type: float
    IncidenceAngle = FloatDescriptor(
        'IncidenceAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Incidence angle for the APC to CRP LOS and the Earth Tangent '
                  'Plane (ETP) at the CRP.')  # type: float
    AzimuthAngle = FloatDescriptor(
        'AzimuthAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 360),
        docstring='Angle from north to the line from the CRP to the APC ETP '
                  'Nadir (i.e. North to +GPX). Measured clockwise from North '
                  'toward East.')  # type: float

    def __init__(self, RcvTime=None, RcvPos=None, RcvVel=None,
                 SideOfTrack=None, SlantRange=None, GroundRange=None,
                 DopplerConeAngle=None, GrazeAngle=None, IncidenceAngle=None,
                 AzimuthAngle=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RcvTime = RcvTime
        self.RcvPos = RcvPos
        self.RcvVel = RcvVel
        self.SideOfTrack = SideOfTrack
        self.SlantRange = SlantRange
        self.GroundRange = GroundRange
        self.DopplerConeAngle = DopplerConeAngle
        self.GrazeAngle = GrazeAngle
        self.IncidenceAngle = IncidenceAngle
        self.AzimuthAngle = AzimuthAngle
        super(RcvParametersType, self).__init__(**kwargs)

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


class ReferenceGeometryType(Serializable):
    """
    Parameters that describe the collection geometry for the reference vector
    of the reference channel.
    """

    _fields = ('CRP', 'RcvParameters')
    _required = _fields
    # descriptors
    CRP = SerializableDescriptor(
        'CRP', CRPType, _required, strict=DEFAULT_STRICT,
        docstring='The Collection Reference Point (CRP) used for computing the'
                  ' geometry parameters.')  # type: CRPType
    RcvParameters = SerializableDescriptor(
        'RcvParameters', RcvParametersType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters computed for the receive APC position.')  # type: RcvParametersType

    def __init__(self, CRP=None, RcvParameters=None, **kwargs):
        """

        Parameters
        ----------
        CRP : CRPType
        RcvParameters : RcvParametersType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CRP = CRP
        self.RcvParameters = RcvParameters
        super(ReferenceGeometryType, self).__init__(**kwargs)
