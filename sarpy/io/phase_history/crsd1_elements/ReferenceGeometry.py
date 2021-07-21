"""
The reference geometry parameters definition.
"""

from typing import Union

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _FloatDescriptor, \
    _StringEnumDescriptor, _SerializableDescriptor
from sarpy.io.complex.sicd_elements.blocks import XYZType, LatLonHAEType

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Michael Stewart, Valkyrie")


class CRPType(Serializable):
    """
    The CRP position for the reference vector of the reference channel.
    """

    _fields = ('ECF', 'LLH')
    _required = _fields
    # descriptors
    ECF = _SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='CRP position in ECF coordinates.')  # type: XYZType
    LLH = _SerializableDescriptor(
        'LLH', LatLonHAEType, _required, strict=DEFAULT_STRICT,
        docstring='CRP position in WGS 84 LLH Coordinates.')  # type: LatLonHAEType

    def __init__(self, ECF=None, LLH=None, **kwargs):
        """

        Parameters
        ----------
        ECF : XYZType|numpy.ndarray|list|tuple
        LLH : LatLonHAEType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ECF = ECF
        self.LLH = LLH
        super(CRPType, self).__init__(**kwargs)


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
    RcvTime = _FloatDescriptor(
        'RcvTime', _required, strict=DEFAULT_STRICT,
        docstring='Receive time for the first sample for the reference vector.')  # type: float
    RcvPos = _SerializableDescriptor(
        'RcvPos', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='APC position in ECF coordinates.')  # type: XYZType
    RcvVel = _SerializableDescriptor(
        'RcvVel', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='APC velocity in ECF coordinates.')  # type: XYZType
    SideOfTrack = _StringEnumDescriptor(
        'SideOfTrack', ('L', 'R'), _required, strict=DEFAULT_STRICT,
        docstring='Side of Track parameter for the collection.')  # type: str
    SlantRange = _FloatDescriptor(
        'SlantRange', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Slant range from the APC to the CRP.')  # type: float
    GroundRange = _FloatDescriptor(
        'GroundRange', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Ground range from the APC nadir to the CRP.')  # type: float
    DopplerConeAngle = _FloatDescriptor(
        'DopplerConeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 180),
        docstring='Doppler Cone Angle between APC velocity and deg CRP Line of '
                  'Sight (LOS).')  # type: float
    GrazeAngle = _FloatDescriptor(
        'GrazeAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Grazing angle for the APC to CRP LOS and the Earth Tangent '
                  'Plane (ETP) at the CRP.')  # type: float
    IncidenceAngle = _FloatDescriptor(
        'IncidenceAngle', _required, strict=DEFAULT_STRICT, bounds=(0, 90),
        docstring='Incidence angle for the APC to CRP LOS and the Earth Tangent '
                  'Plane (ETP) at the CRP.')  # type: float
    AzimuthAngle = _FloatDescriptor(
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
    CRP = _SerializableDescriptor(
        'CRP', CRPType, _required, strict=DEFAULT_STRICT,
        docstring='The Collection Reference Point (CRP) used for computing the'
                  ' geometry parameters.')  # type: CRPType
    RcvParameters = _SerializableDescriptor(
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
