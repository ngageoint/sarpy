# -*- coding: utf-8 -*-
"""
The SRP definition for CPHD 0.3.
"""

from typing import Union, List

from ..base import DEFAULT_STRICT
# noinspection PyProtectedMember
from ...sicd_elements.base import Serializable, _IntegerDescriptor, \
    _SerializableListDescriptor, _StringEnumDescriptor
from ...sicd_elements.blocks import XYZType, XYZPolyType


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

# TODO:
#  1.) What happens with STEPPED?
#  2.) Can SRPType be inferred?
#  3.) Can NumSRPs be inferred? Related to 1.) and 2.)


class SRPTyp(Serializable):
    """
    """

    _fields = ('SRPType', 'NumSRPs', 'FIXEDPT', 'PVTPOLY', 'PVVPOLY')
    _required = ('SRPType', 'NumSRPs')
    _collections_tags = {
        'FIXEDPT': {'array': False, 'child_tag': 'FIXEDPT'},
        'PVTPOLY': {'array': False, 'child_tag': 'PVTPOLY'},
        'PVVPOLY': {'array': False, 'child_tag': 'PVVPOLY'}}
    _choice = ({'required': False, 'collection': ('FIXEDPT', 'PVTPOLY', 'PVVPOLY')}, )
    # descriptors
    SRPType = _StringEnumDescriptor(
        'SRPType', ('FIXEDPT', 'PVTPOLY', 'PVVPOLY', 'STEPPED'), _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    NumSRPs = _IntegerDescriptor(
        'NumSRPs', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: int
    FIXEDPT = _SerializableListDescriptor(
        'FIXEDPT', XYZType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, List[XYZType]]
    PVTPOLY = _SerializableListDescriptor(
        'PVTPOLY', XYZPolyType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, List[XYZPolyType]]
    PVVPOLY = _SerializableListDescriptor(
        'PVVPOLY', XYZPolyType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, List[XYZPolyType]]

    def __init__(self, SRPType=None, NumSRPs=None, FIXEDPT=None, PVTPOLY=None, PVVPOLY=None,
                 **kwargs):
        """

        Parameters
        ----------
        SRPType : str
        NumSRPs : int
        FIXEDPT : None|List[XYZType]
        PVTPOLY : None|List[XYZPolyType]
        PVVPOLY : None|List[XYZPolyType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SRPType = SRPType
        self.NumSRPs = NumSRPs
        self.FIXEDPT = FIXEDPT
        self.PVTPOLY = PVTPOLY
        self.PVVPOLY = PVVPOLY
        super(SRPTyp, self).__init__(**kwargs)
