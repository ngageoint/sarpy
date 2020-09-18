# -*- coding: utf-8 -*-
"""
The graphics header element definition.
"""

from .base import NITFElement, UserHeaderType, _IntegerDescriptor,\
    _StringDescriptor, _StringEnumDescriptor, _NITFElementDescriptor
from .security import NITFSecurityTags

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class GraphicsSegmentHeader(NITFElement):
    """
    Graphics segment subheader - see standards document MIL-STD-2500C for more
    information.
    """
    _ordering = (
        'SY', 'SID', 'SNAME', 'Security', 'ENCRYP', 'SFMT',
        'SSTRUCT', 'SDLVL', 'SALVL', 'SLOC', 'SBND1',
        'SCOLOR', 'SBND2', 'SRES2', 'UserHeader')
    _lengths = {
        'SY': 2, 'SID': 10, 'SNAME': 20, 'ENCRYP': 1,
        'SFMT': 1, 'SSTRUCT': 13, 'SDLVL': 3, 'SALVL': 3,
        'SLOC': 10, 'SBND1': 10, 'SCOLOR': 1, 'SBND2': 10,
        'SRES2': 2}
    SY = _StringEnumDescriptor(
        'SY', True, 2, {'SY', }, default_value='SY',
        docstring='File part type.')  # type: str
    SID = _StringDescriptor(
        'SID', True, 10, default_value='',
        docstring='Graphic Identifier. This field shall contain a valid alphanumeric identification code '
                  'associated with the graphic. The valid codes are determined by the application.')  # type: str
    SNAME = _StringDescriptor(
        'SNAME', True, 20, default_value='',
        docstring='Graphic name. This field shall contain an alphanumeric name for the graphic.')  # type: str
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags, default_args={},
        docstring='The security tags.')  # type: NITFSecurityTags
    ENCRYP = _StringEnumDescriptor(
        'ENCRYP', True, 1, {'0'}, default_value='0',
        docstring='Encryption.')  # type: str
    SFMT = _StringDescriptor(
        'SFMT', True, 1, default_value='C',
        docstring='Graphic Type. This field shall contain a valid indicator of the '
                  'representation type of the graphic.')  # type: str
    SSTRUCT = _IntegerDescriptor(
        'SSTRUCT', True, 13, default_value=0,
        docstring='Reserved for Future Use.')  # type: int
    SDLVL = _IntegerDescriptor(
        'SDLVL', True, 3, default_value=1,
        docstring='Graphic Display Level. This field shall contain a valid value that indicates '
                  'the graphic display level of the graphic relative to other displayed file '
                  'components in a composite display. The valid values are :code:`1-999`. '
                  'The display level of each displayable file component (image or graphic) '
                  'within a file shall be unique.')  # type: int
    SALVL = _IntegerDescriptor(
        'SALVL', True, 3, default_value=0,
        docstring='Graphic Attachment Level. This field shall contain a valid value '
                  'that indicates the attachment level of the graphic. Valid values for '
                  'this field are 0 and the display level value of any other '
                  'image or graphic in the file.')  # type: int
    SLOC = _IntegerDescriptor(
        'SLOC', True, 10, default_value=0,
        docstring='Graphic Location. The graphics location is specified by providing the location '
                  'of the graphic’s origin point relative to the position (location of the CCS, image, '
                  'or graphic to which it is attached. This field shall contain the graphic location '
                  'offset from the `ILOC` or `SLOC` value of the CCS, image, or graphic to which the graphic '
                  'is attached or from the origin of the CCS when the graphic is unattached (`SALVL = 0`). '
                  'A row and column value of :code:`0` indicates no offset. Positive row and column values indicate '
                  'offsets down and to the right, while negative row and column values indicate '
                  'offsets up and to the left.')  # type: int
    SBND1 = _IntegerDescriptor(
        'SBND1', True, 10, default_value=0,
        docstring='First Graphic Bound Location. This field shall contain an ordered pair of '
                  'integers defining a location in Cartesian coordinates for use with CGM graphics. It is '
                  'the upper left corner of the bounding box for the CGM graphic.')  # type: int
    SCOLOR = _StringEnumDescriptor(
        'SCOLOR', True, 1, {'C', 'M'}, default_value='M',
        docstring='Graphic Color. If `SFMT = C`, this field shall contain a :code:`C` if the CGM contains any '
                  'color pieces or an :code:`M` if it is monochrome (i.e., black, '
                  'white, or levels of grey).')  # type: str
    SBND2 = _IntegerDescriptor(
        'SBND2', True, 10, default_value=0,
        docstring='Second Graphic Bound Location. This field shall contain an ordered pair of '
                  'integers defining a location in Cartesian coordinates for use with CGM graphics. '
                  'It is the lower right corner of the bounding box for the CGM graphic.')  # type: int
    SRES2 = _IntegerDescriptor(
        'SRES2', True, 2, default_value=0,
        docstring='Reserved for Future Use.')  # type: int
    UserHeader = _NITFElementDescriptor(
        'UserHeader', True, UserHeaderType, default_args={},
        docstring='User defined header.')  # type: UserHeaderType
