# -*- coding: utf-8 -*-
"""
The text extension subheader definitions.
"""

from .base import NITFElement, UserHeaderType, _IntegerDescriptor,\
    _StringDescriptor, _StringEnumDescriptor, _NITFElementDescriptor
from .security import NITFSecurityTags


class TextSegmentHeader(NITFElement):
    """
    Text Segment Subheader
    """
    _ordering = (
        'TE', 'TEXTID', 'TXTALVL', 'TXTDT', 'TXTITL', 'Security',
        'ENCRYP', 'TXTFMT', 'UserHeader')
    _lengths = {}
    TE = _StringEnumDescriptor(
        'TE', True, 2, {'TE', }, default_value='TE',
        docstring='File part type.')  # type: str
    TEXTID = _StringDescriptor(
        'TEXTID', True, 7, default_value='',
        docstring='Text Identifier. This field shall contain a valid alphanumeric identification '
                  'code associated with the text item. The valid codes are determined '
                  'by the application.')  # type: str
    TXTALVL = _IntegerDescriptor(
        'TXTALVL', True, 3, default_value=0,
        docstring='Text Attachment Level. This field shall contain a valid value that '
                  'indicates the attachment level of the text.')  # type: int
    TXTDT = _StringDescriptor(
        'TXTDT', True, 14, default_value='',
        docstring='Text Date and Time. This field shall contain the time (UTC) of origination '
                  'of the text in the format :code:`YYYYMMDDhhmmss`')  # type: str
    TXTITL = _StringDescriptor(
        'TXTITL', True, 80, default_value='',
        docstring='Text Title. This field shall contain the title of the text item.')  # type: str
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags, default_args={},
        docstring='The security tags.')  # type: NITFSecurityTags
    ENCRYP = _StringEnumDescriptor(
        'ENCRYP', True, 1, {'0'}, default_value='0',
        docstring='Encryption.')  # type: str
    TXTFMT = _StringEnumDescriptor(
        'TXTFMT', True, 3, {'', 'MTF', 'STA', 'UT1', 'U8S'}, default_value='',
        docstring='Text Format. This field shall contain a valid three character code '
                  'indicating the format or type of text data. Valid codes are :code:`MTF` to '
                  'indicate USMTF (Refer to MIL-STD-6040 for examples of the USMTF format), '
                  ':code:`STA` to indicate BCS, :code:`UT1` to indicate ECS text formatting, and '
                  ':code:`U8S` to indicate U8S text formatting.')  # type: str
    UserHeader = _NITFElementDescriptor(
        'UserHeader', True, UserHeaderType, default_args={},
        docstring='User defined header.')  # type: UserHeaderType
