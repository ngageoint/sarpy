"""
The SRP definition for CPHD 0.3.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from typing import Union, List

from sarpy.io.phase_history.cphd1_elements.base import DEFAULT_STRICT
from sarpy.io.complex.sicd_elements.blocks import XYZType, XYZPolyType
from sarpy.io.xml.base import Serializable, SerializableArray, parse_str, parse_int
from sarpy.io.xml.descriptors import SerializableArrayDescriptor

logger = logging.getLogger(__name__)


class PlainArrayType(SerializableArray):
    _set_index = False
    _set_size = False


class SRPTyp(Serializable):
    """
    """

    _fields = ('SRPType', 'NumSRPs', 'FIXEDPT', 'PVTPOLY', 'PVVPOLY')
    _required = ('SRPType', 'NumSRPs')
    _collections_tags = {
        'FIXEDPT': {'array': True, 'child_tag': 'SRPPT'},
        'PVTPOLY': {'array': True, 'child_tag': 'SRPPVTPoly'},
        'PVVPOLY': {'array': True, 'child_tag': 'SRPPVVPoly'}}
    _choice = ({'required': False, 'collection': ('FIXEDPT', 'PVTPOLY', 'PVVPOLY')}, )
    # descriptors
    FIXEDPT = SerializableArrayDescriptor(
        'FIXEDPT', XYZType, _collections_tags, _required, strict=DEFAULT_STRICT, array_extension=PlainArrayType,
        docstring='')  # type: Union[None, PlainArrayType, List[XYZType]]
    PVTPOLY = SerializableArrayDescriptor(
        'PVTPOLY', XYZPolyType, _collections_tags, _required, strict=DEFAULT_STRICT, array_extension=PlainArrayType,
        docstring='')  # type: Union[None, PlainArrayType, List[XYZPolyType]]
    PVVPOLY = SerializableArrayDescriptor(
        'PVVPOLY', XYZPolyType, _collections_tags, _required, strict=DEFAULT_STRICT, array_extension=PlainArrayType,
        docstring='')  # type: Union[None, PlainArrayType, List[XYZPolyType]]

    def __init__(self, SRPType=None, NumSRPs=None, FIXEDPT=None, PVTPOLY=None, PVVPOLY=None,
                 **kwargs):
        """

        Parameters
        ----------
        SRPType : str
        NumSRPs : int
        FIXEDPT : None|PlainArrayType|List[XYZType]
        PVTPOLY : None|PlainArrayType|List[XYZPolyType]
        PVVPOLY : None|PlainArrayType|List[XYZPolyType]
        kwargs
        """

        self._SRPType = None
        self._NumSRPs = None
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.FIXEDPT = FIXEDPT
        self.PVTPOLY = PVTPOLY
        self.PVVPOLY = PVVPOLY
        self.SRPType = SRPType
        self.NumSRPs = NumSRPs
        super(SRPTyp, self).__init__(**kwargs)

    @property
    def SRPType(self):
        """
        str: The type of SRP.
        """
        if self.FIXEDPT is not None:
            return 'FIXEDPT'
        elif self.PVTPOLY is not None:
            return 'PVTPOLY'
        elif self.PVVPOLY is not None:
            return 'PVVPOLY'
        else:
            return self._SRPType

    @SRPType.setter
    def SRPType(self, value):
        if self.FIXEDPT is not None or self.PVTPOLY is not None or self.PVVPOLY is not None:
            self._SRPType = None
        else:
            value = parse_str(value, 'SRPType', self).upper()
            if value in ('FIXEDPT', 'PVTPOLY', 'PVVPOLY', 'STEPPED'):
                self._SRPType = value
            else:
                logger.warning(
                    'Got {} for the SRPType field of class SRPTyp.\n\t'
                    'It is required to be one of {}.\n\t'
                    'Setting to None, which is required to be fixed.'.format(
                        value, ('FIXEDPT', 'PVTPOLY', 'PVVPOLY', 'STEPPED')))
                self._SRPType = None

    @property
    def NumSRPs(self):
        """
        None|int: The number of SRPs.
        """

        if self.FIXEDPT is not None:
            return self.FIXEDPT.size
        elif self.PVTPOLY is not None:
            return self.PVTPOLY.size
        elif self.PVVPOLY is not None:
            return self.PVVPOLY.size
        else:
            return self._NumSRPs

    @NumSRPs.setter
    def NumSRPs(self, value):
        if self.FIXEDPT is not None or self.PVTPOLY is not None or self.PVVPOLY is not None:
            self._NumSRPs = None
        else:
            self._NumSRPs = parse_int(value, 'NumSRPs', self)
