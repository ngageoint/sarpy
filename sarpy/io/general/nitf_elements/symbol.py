"""
The symbol header element definition - only applies to NITF 2.0
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import struct
import numpy

from sarpy.compliance import int_func
from .base import NITFElement, UserHeaderType, _IntegerDescriptor,\
    _StringDescriptor, _StringEnumDescriptor, _NITFElementDescriptor
from .security import NITFSecurityTags0


class SymbolSegmentHeader(NITFElement):
    """
    Symbol segment subheader for NITF version 2.0 - see standards document
    MIL-STD-2500A for more information.
    """
    _ordering = (
        'SY', 'SID', 'SNAME', 'Security', 'ENCRYP', 'STYPE',
        'NLIPS', 'NPIXPL', 'NWDTH', 'NBPP', 'SDLVL', 'SALVL', 'SLOC',
        'SLOC2', 'SCOLOR', 'SNUM', 'SROT', 'DLUT', 'UserHeader')
    _lengths = {
        'SY': 2, 'SID': 10, 'SNAME': 20, 'ENCRYP': 1, 'STYPE': 1,
        'NLIPS': 4, 'NPIXPL': 4, 'NWDTH': 4, 'NBPP': 1,
        'SDLVL': 3, 'SALVL': 3, 'SLOC': 10, 'SLOC2': 10,'SCOLOR': 1,
        'SNUM': 6, 'SROT': 3}
    SY = _StringEnumDescriptor('SY', True, 2, {'SY', })  # type: str
    SID = _StringDescriptor('SID', True, 10)  # type: str
    SNAME = _StringDescriptor('SNAME', True, 20)  # type: str
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags0, default_args={})  # type: NITFSecurityTags0
    ENCRYP = _StringEnumDescriptor('ENCRYP', True, 1, {'0'})  # type: str
    STYPE = _StringDescriptor('STYPE', True, 1)  # type: str
    NLIPS = _IntegerDescriptor('NLIPS', True, 4)  # type: int
    NPIXPL = _IntegerDescriptor('NPIXPL', True, 4)  # type: int
    NWDTH = _IntegerDescriptor('NWDTH', True, 4)  # type: int
    NBPP = _IntegerDescriptor('NBPP', True, 1)  # type: int
    SDLVL = _IntegerDescriptor('SDLVL', True, 3)  # type: int
    SALVL = _IntegerDescriptor('SALVL', True, 3)  # type: int
    SLOC = _StringDescriptor('SLOC', True, 10)  # type: str
    SLOC2 = _StringDescriptor('SLOC2', True, 10)  # type: str
    SCOLOR = _StringDescriptor('SCOLOR', True, 1)  # type: str
    SNUM = _StringDescriptor('SLOC2', True, 6)  # type: str
    SROT = _IntegerDescriptor('SROT', True, 3)  # type: int
    UserHeader = _NITFElementDescriptor(
        'UserHeader', True, UserHeaderType, default_args={})  # type: UserHeaderType

    def __init__(self, **kwargs):
        self._DLUT = None
        super(SymbolSegmentHeader, self).__init__(**kwargs)

    @classmethod
    def minimum_length(cls):
        return 13

    @property
    def DLUT(self):
        """
        The Look-up Table (LUT) data.

        Returns
        -------
        None|numpy.ndarray
        """

        return self._DLUT

    @DLUT.setter
    def DLUT(self, value):
        if value is None:
            self._DLUT = None
            return

        if not isinstance(value, numpy.ndarray):
            raise TypeError('DLUT must be a numpy array')
        if value.dtype.name != 'uint8':
            raise ValueError('DLUT must be a numpy array of dtype uint8, got {}'.format(value.dtype.name))
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError('DLUT must be a two-dimensional array of shape (N, 3).')
        if value.size > 256:
            raise ValueError(
                'The number of DLUT elements must be 256 or fewer. '
                'Got DLUT shape {}'.format(value.shape))
        self._DLUT = value

    @property
    def NELUT(self):
        """
        Number of LUT Entries.

        Returns
        -------
        int
        """

        return 0 if self._DLUT is None else self._DLUT.size

    def _get_attribute_bytes(self, attribute):
        if attribute == 'DLUT':
            if self.NELUT == 0:
                out = b'000'
            else:
                out = '{0:d}'.format(self.NELUT).encode() + \
                      struct.pack('{}B'.format(self.NELUT*3), *self.DLUT.flatten())
            return out
        else:
            return super(SymbolSegmentHeader, self)._get_attribute_bytes(attribute)

    def _get_attribute_length(self, attribute):
        if attribute == 'DLUT':
            return 3 + self.NELUT*3
        else:
            return super(SymbolSegmentHeader, self)._get_attribute_length(attribute)

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'DLUT':
            loc = start
            nelut = int_func(value[loc:loc + 3])
            loc += 3
            if nelut == 0:
                fields['DLUT'] = None
            else:
                fields['DLUT'] = numpy.array(
                    struct.unpack(
                        '{}B'.format(3*nelut),
                        value[loc:loc + 3*nelut]), dtype=numpy.uint8).reshape((nelut, 3))
                loc += nelut*3
            return loc
        return super(SymbolSegmentHeader, cls)._parse_attribute(fields, attribute, value, start)
