"""
The main NITF header definitions.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging

from .base import NITFElement, UserHeaderType, _IntegerDescriptor,\
    _StringDescriptor, _StringEnumDescriptor, _NITFElementDescriptor, _RawDescriptor, \
    _ItemArrayHeaders
from .security import NITFSecurityTags, NITFSecurityTags0
from sarpy.compliance import string_types

logger = logging.getLogger(__name__)


#############
# NITF 2.1 version

class ImageSegmentsType(_ItemArrayHeaders):
    """
    This holds the image subheader and item sizes.
    """

    _subhead_len = 6
    _item_len = 10


class GraphicsSegmentsType(_ItemArrayHeaders):
    """
    This holds the graphics subheader and item sizes.
    """

    _subhead_len = 4
    _item_len = 6


class TextSegmentsType(_ItemArrayHeaders):
    """
    This holds the text subheader size and item sizes.
    """

    _subhead_len = 4
    _item_len = 5


class DataExtensionsType(_ItemArrayHeaders):
    """
    This holds the data extension subheader and item sizes.
    """

    _subhead_len = 4
    _item_len = 9


class ReservedExtensionsType(_ItemArrayHeaders):
    """
    This holds the reserved extension subheader and item sizes.
    """

    _subhead_len = 4
    _item_len = 7


class NITFHeader(NITFElement):
    """
    The main NITF file header for NITF version 2.1 - see standards document
    MIL-STD-2500C for more information.
    """

    _ordering = (
        'FHDR', 'FVER', 'CLEVEL', 'STYPE',
        'OSTAID', 'FDT', 'FTITLE', 'Security',
        'FSCOP', 'FSCPYS', 'ENCRYP', 'FBKGC',
        'ONAME', 'OPHONE', 'FL', 'HL',
        'ImageSegments', 'GraphicsSegments', 'NUMX',
        'TextSegments', 'DataExtensions', 'ReservedExtensions',
        'UserHeader', 'ExtendedHeader')
    _lengths = {
        'FHDR': 4, 'FVER': 5, 'CLEVEL': 2, 'STYPE': 4,
        'OSTAID': 10, 'FDT': 14, 'FTITLE': 80,
        'FSCOP': 5, 'FSCPYS': 5, 'ENCRYP': 1, 'FBKGC': 3,
        'ONAME': 24, 'OPHONE': 18, 'FL': 12, 'HL': 6,
        'NUMX': 3}
    CLEVEL = _IntegerDescriptor(
        'CLEVEL', True, 2, default_value=0,
        docstring='Complexity Level. This field shall contain the complexity level required to '
                  'interpret fully all components of the file. Valid entries are assigned in '
                  'accordance with complexity levels established in Table A-10.')  # type: int
    STYPE = _StringDescriptor(
        'STYPE', True, 4, default_value='BF01',
        docstring='Standard Type. Standard type or capability. A BCS-A character string `BF01` '
                  'which indicates that this file is formatted using ISO/IEC IS 12087-5. '
                  'NITF02.10 is intended to be registered as a profile of ISO/IEC IS 12087-5.')  # type: str
    OSTAID = _StringDescriptor(
        'OSTAID', True, 10, default_value='',
        docstring='Originating Station ID. This field shall contain the identification code or name of '
                  'the originating organization, system, station, or product. It shall not be '
                  'filled with BCS spaces')  # type: str
    FDT = _StringDescriptor(
        'FDT', True, 14, default_value='',
        docstring='File Date and Time. This field shall contain the time (UTC) of the files '
                  'origination in the format `YYYYMMDDhhmmss`.')  # type: str
    FTITLE = _StringDescriptor(
        'FTITLE', True, 80, default_value='',
        docstring='File Title. This field shall contain the title of the file.')  # type: str
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags, default_args={},
        docstring='The image security tags.')  # type: NITFSecurityTags
    FSCOP = _IntegerDescriptor(
        'FSCOP', True, 5, default_value=0,
        docstring='File Copy Number. This field shall contain the copy number of the file.')  # type: int
    FSCPYS = _IntegerDescriptor(
        'FSCPYS', True, 5, default_value=0,
        docstring='File Number of Copies. This field shall contain the total number of '
                  'copies of the file.')  # type: int
    ENCRYP = _StringEnumDescriptor(
        'ENCRYP', True, 1, {'0'}, default_value='0',
        docstring='Encryption.')  # type: str
    FBKGC = _RawDescriptor(
        'FBKGC', True, 3, default_value=b'\x00\x00\x00',
        docstring='File Background Color. This field shall contain the three color components of '
                  'the file background in the order Red, Green, Blue.')  # type: bytes
    ONAME = _StringDescriptor(
        'ONAME', True, 24, default_value='',
        docstring='Originator Name. This field shall contain a valid name for the operator '
                  'who originated the file.')  # type: str
    OPHONE = _StringDescriptor(
        'OPHONE', True, 18, default_value='',
        docstring='Originator Phone Number. This field shall contain a valid phone number '
                  'for the operator who originated the file.')  # type: str
    FL = _IntegerDescriptor(
        'FL', True, 12, docstring='The size in bytes of the entire file.')  # type: int
    ImageSegments = _NITFElementDescriptor(
        'ImageSegments', True, ImageSegmentsType, default_args={},
        docstring='The image segment basic information.')  # type: ImageSegmentsType
    GraphicsSegments = _NITFElementDescriptor(
        'GraphicsSegments', True, GraphicsSegmentsType, default_args={},
        docstring='The graphics segment basic information.')  # type: GraphicsSegmentsType
    TextSegments = _NITFElementDescriptor(
        'TextSegments', True, TextSegmentsType, default_args={},
        docstring='The text segment basic information.')  # type: TextSegmentsType
    DataExtensions = _NITFElementDescriptor(
        'DataExtensions', True, DataExtensionsType, default_args={},
        docstring='The data extension basic information.')  # type: DataExtensionsType
    ReservedExtensions = _NITFElementDescriptor(
        'ReservedExtensions', True, ReservedExtensionsType, default_args={},
        docstring='The reserved extension basic information.')  # type: ReservedExtensionsType
    UserHeader = _NITFElementDescriptor(
        'UserHeader', True, UserHeaderType, default_args={},
        docstring='User defined header.')  # type: UserHeaderType
    ExtendedHeader = _NITFElementDescriptor(
        'ExtendedHeader', True, UserHeaderType, default_args={},
        docstring='Extended subheader - TRE list.')  # type: UserHeaderType

    def __init__(self, **kwargs):
        self._FHDR = 'NITF'
        self._FVER = '02.10'
        self._NUMX = 0
        super(NITFHeader, self).__init__(**kwargs)

    @property
    def FHDR(self):
        """
        str: File Profile Name. This field shall contain the character string uniquely denoting
        that the file is formatted using NITF. Always `NITF`.
        """

        return self._FHDR

    @FHDR.setter
    def FHDR(self, value):
        pass

    @property
    def FVER(self):
        """
        str: File Version. This field shall contain a BCS-A character string uniquely
        denoting the version. Always `02.10`.
        """

        return self._FVER

    @FVER.setter
    def FVER(self, value):
        pass

    @property
    def NUMX(self):
        """
        int: Reserved for future use. Always :code:`0`.
        """

        return self._NUMX

    @NUMX.setter
    def NUMX(self, value):
        pass

    @property
    def HL(self):
        """
        int: The length of this header object in bytes.
        """

        return self.get_bytes_length()

    @HL.setter
    def HL(self, value):
        pass


#############
# NITF 2.0 version

class SymbolSegmentsType(_ItemArrayHeaders):
    """
    This holds the symbol subheader and item sizes.
    """

    _subhead_len = 4
    _item_len = 6


class LabelSegmentsType(_ItemArrayHeaders):
    """
    This holds the label subheader and item sizes.
    """

    _subhead_len = 4
    _item_len = 3


class NITFHeader0(NITFElement):
    """
    The main NITF file header for NITF version 2.0 - see standards document
    MIL-STD-2500A for more information.
    """

    _ordering = (
        'FHDR', 'FVER', 'CLEVEL', 'STYPE', 'OSTAID', 'FDT', 'FTITLE', 'Security',
        'FSCOP', 'FSCPYS', 'ENCRYP', 'ONAME', 'OPHONE', 'FL', 'HL',
        'ImageSegments', 'SymbolSegments', 'LabelSegments', 'TextSegments',
        'DataExtensions', 'ReservedExtensions', 'UserHeader', 'ExtendedHeader')
    _lengths = {
        'FHDR': 4, 'FVER': 5, 'CLEVEL': 2, 'STYPE': 4,
        'OSTAID': 10, 'FDT': 14, 'FTITLE': 80,
        'FSCOP': 5, 'FSCPYS': 5, 'ENCRYP': 1,
        'ONAME': 27, 'OPHONE': 18, 'FL': 12, 'HL': 6}
    CLEVEL = _IntegerDescriptor(
        'CLEVEL', True, 2, default_value=0,
        docstring='Complexity Level. This field shall contain the complexity level required to '
                  'interpret fully all components of the file. Valid entries are assigned in '
                  'accordance with complexity levels established in Table A-10.')  # type: int
    STYPE = _StringDescriptor(
        'STYPE', True, 4, default_value='BF01',
        docstring='Standard Type. Standard type or capability. A BCS-A character string `BF01` '
                  'which indicates that this file is formatted using ISO/IEC IS 12087-5. '
                  'NITF02.10 is intended to be registered as a profile of ISO/IEC IS 12087-5.')  # type: str
    OSTAID = _StringDescriptor(
        'OSTAID', True, 10, default_value='',
        docstring='Originating Station ID. This field shall contain the identification code or name of '
                  'the originating organization, system, station, or product. It shall not be '
                  'filled with BCS spaces')  # type: str
    FDT = _StringDescriptor(
        'FDT', True, 14, default_value='',
        docstring='File Date and Time. This field shall contain the time (UTC) of the files '
                  'origination in the format `YYYYMMDDhhmmss`.')  # type: str
    FTITLE = _StringDescriptor(
        'FTITLE', True, 80, default_value='',
        docstring='File Title. This field shall contain the title of the file.')  # type: str
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags0, default_args={},
        docstring='The image security tags.')  # type: NITFSecurityTags0
    FSCOP = _StringDescriptor(
        'FSCOP', True, 5, default_value='    0',
        docstring='File Copy Number. This field shall contain the copy number of the file.')  # type: str
    FSCPYS = _StringDescriptor(
        'FSCPYS', True, 5, default_value='    0',
        docstring='File Number of Copies. This field shall contain the total number of '
                  'copies of the file.')  # type: str
    ENCRYP = _StringEnumDescriptor(
        'ENCRYP', True, 1, {'0'}, default_value='0',
        docstring='Encryption.')  # type: str
    ONAME = _StringDescriptor(
        'ONAME', True, 27, default_value='',
        docstring='Originator Name. This field shall contain a valid name for the operator '
                  'who originated the file.')  # type: str
    OPHONE = _StringDescriptor(
        'OPHONE', True, 18, default_value='',
        docstring='Originator Phone Number. This field shall contain a valid phone number '
                  'for the operator who originated the file.')  # type: str
    FL = _IntegerDescriptor(
        'FL', True, 12, docstring='The size in bytes of the entire file.')  # type: int
    ImageSegments = _NITFElementDescriptor(
        'ImageSegments', True, ImageSegmentsType, default_args={},
        docstring='The image segment basic information.')  # type: ImageSegmentsType
    SymbolSegments = _NITFElementDescriptor(
        'SymbolSegments', True, SymbolSegmentsType, default_args={},
        docstring='The symbols segment basic information.')  # type: SymbolSegmentsType
    LabelSegments = _NITFElementDescriptor(
        'LabelSegments', True, LabelSegmentsType, default_args={},
        docstring='The labels segment basic information.')  # type: LabelSegmentsType
    TextSegments = _NITFElementDescriptor(
        'TextSegments', True, TextSegmentsType, default_args={},
        docstring='The text segment basic information.')  # type: TextSegmentsType
    DataExtensions = _NITFElementDescriptor(
        'DataExtensions', True, DataExtensionsType, default_args={},
        docstring='The data extension basic information.')  # type: DataExtensionsType
    ReservedExtensions = _NITFElementDescriptor(
        'ReservedExtensions', True, ReservedExtensionsType, default_args={},
        docstring='The reserved extension basic information.')  # type: ReservedExtensionsType
    UserHeader = _NITFElementDescriptor(
        'UserHeader', True, UserHeaderType, default_args={},
        docstring='User defined header.')  # type: UserHeaderType
    ExtendedHeader = _NITFElementDescriptor(
        'ExtendedHeader', True, UserHeaderType, default_args={},
        docstring='Extended subheader - TRE list.')  # type: UserHeaderType

    def __init__(self, **kwargs):
        self._FHDR = 'NITF'
        self._FVER = '02.00'
        super(NITFHeader0, self).__init__(**kwargs)

    @property
    def FHDR(self):
        """
        str: File Profile Name. This field shall contain the character string uniquely denoting
        that the file is formatted using NITF. Always `NITF`.
        """

        return self._FHDR

    @FHDR.setter
    def FHDR(self, value):
        pass

    @property
    def FVER(self):
        """
        str: File Version. This field shall contain a BCS-A character string uniquely
        denoting the version, should generally be `02.00` or `01.10`.
        """

        return self._FVER

    @FVER.setter
    def FVER(self, value):
        if isinstance(value, bytes) and not isinstance(value, string_types):
            value = value.decode('utf-8')
        if not isinstance(value, string_types):
            raise TypeError('FVER is required to be a string')
        if len(value) != 5:
            raise ValueError('FVER must have length 5')
        if value not in ['02.00', '01.10']:
            logger.warning('Got unexpected version {}, and NITF parsing is likely to fail.'.format(value))
        self._FVER = value

    @property
    def HL(self):
        """
        int: The length of this header object in bytes.
        """

        return self.get_bytes_length()

    @HL.setter
    def HL(self, value):
        pass
