# -*- coding: utf-8 -*-
"""
The label extension subheader definitions - applies only to NITF version 2.0
"""

from .base import NITFElement, UserHeaderType, _IntegerDescriptor, _RawDescriptor, \
    _StringDescriptor, _StringEnumDescriptor, _NITFElementDescriptor
from .security import NITFSecurityTags0

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class LabelSegmentHeader(NITFElement):
    """
    Symbol segment subheader for NITF version 2.0 - see standards document
    MIL-STD-2500A for more information.
    """

    _ordering = (
        'LA', 'LID', 'Security', 'ENCRYP', 'LFS', 'LCW', 'LCH',
        'LDLVL', 'LALVL', 'LLOC', 'LTC', 'LBC', 'UserHeader')
    _lengths = {
        'LA': 2, 'LID': 7, 'ENCRYP': 1, 'LFS': 1, 'LCW': 2, 'LCH': 2,
        'LDLVL': 3, 'LALVL': 3, 'LLOC': 10, 'LTC': 3, 'LBC': 3}
    #######
    LA = _StringEnumDescriptor(
        'LA', True, 2, {'LA', }, default_value='LA')  # type: str
    LID = _StringDescriptor('LID', True, 10)  # type: str
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags0, default_args={})  # type: NITFSecurityTags0
    ENCRYP = _StringEnumDescriptor(
        'ENCRYP', True, 1, {'0'}, default_value='0',
        docstring='Encryption.')  # type: str
    LFS = _StringDescriptor('LFS', True, 1)  # type: str
    LCW = _StringDescriptor('LCW', True, 2, default_value='00')  # type: str
    LCH = _StringDescriptor('LCH', True, 2, default_value='00')  # type: str
    LDLVL = _IntegerDescriptor(
        'LDLVL', True, 3, default_value=1)  # type: int
    LALVL = _IntegerDescriptor(
        'LALVL', True, 3, default_value=1)  # type: int
    LLOC = _StringDescriptor('LLOC', True, 10)  # type: str
    LTC = _RawDescriptor(
        'LTC', True, 3, default_value=b'\x00\x00\x00')  # type: bytes
    LBC = _RawDescriptor(
        'LBC', True, 3, default_value=b'\xff\xff\xff')  # type: bytes
    UserHeader = _NITFElementDescriptor(
        'UserHeader', True, UserHeaderType, default_args={},
        docstring='User defined header.')  # type: UserHeaderType
