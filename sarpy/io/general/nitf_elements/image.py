# -*- coding: utf-8 -*-
"""
The image subheader definitions.
"""

import logging
import struct

import numpy

from sarpy.compliance import int_func
from .base import NITFElement, NITFLoop, UserHeaderType, _IntegerDescriptor,\
    _StringDescriptor, _StringEnumDescriptor, _NITFElementDescriptor, _parse_str
from .security import NITFSecurityTags, NITFSecurityTags0


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


#########
# NITF 2.1 version

class ImageBand(NITFElement):
    """
    Single image band, part of the image bands collection
    """
    _ordering = ('IREPBAND', 'ISUBCAT', 'IFC', 'IMFLT', 'LUTD')
    _lengths = {'IREPBAND': 2, 'ISUBCAT': 6, 'IFC': 1, 'IMFLT': 3}
    IREPBAND = _StringDescriptor(
        'IREPBAND', True, 2, default_value='',
        docstring='Representation. This field shall contain a valid indicator of the processing '
                  'required to display the nth band of the image with regard to the general image type '
                  'as recorded in the `IREP` field. The significance of each band in the image can be '
                  'derived from the combination of the `ICAT`, and `ISUBCAT` fields. Valid values of '
                  'the `IREPBAND` field depend on the value of '
                  'the `IREP` field.')  # type: str
    ISUBCAT = _StringDescriptor(
        'ISUBCAT', True, 6, default_value='',
        docstring='Subcategory. The purpose of this field is to provide the significance of the band '
                  'of the image with regard to the specific category (`ICAT` field) '
                  'of the overall image.')  # type: str
    IFC = _StringEnumDescriptor(
        'IFC', True, 1, {'N', }, default_value='N',
        docstring=' Image Filter Condition.')  # type: str
    IMFLT = _StringDescriptor(
        'IMFLT', True, 3, default_value='',
        docstring='Standard Image Filter Code. This field is reserved '
                  'for future use.')  # type: str

    def __init__(self, **kwargs):
        self._LUTD = None
        super(ImageBand, self).__init__(**kwargs)

    @classmethod
    def minimum_length(cls):
        return 13

    @property
    def LUTD(self):
        """
        The Look-up Table (LUT) data.

        Returns
        -------
        None|numpy.ndarray
        """

        return self._LUTD

    @LUTD.setter
    def LUTD(self, value):
        if value is None:
            self._LUTD = None
            return

        if not isinstance(value, numpy.ndarray):
            raise TypeError('LUTD must be a numpy array')
        if value.dtype.name != 'uint8':
            raise ValueError('LUTD must be a numpy array of dtype uint8, got {}'.format(value.dtype.name))
        if value.ndim != 2:
            raise ValueError('LUTD must be a two-dimensional array')
        if value.shape[0] > 4:
            raise ValueError(
                'The number of LUTD bands (axis 0) must be 4 or fewer. '
                'Got LUTD shape {}'.format(value.shape))
        if value.shape[1] > 65536:
            raise ValueError(
                'The number of LUTD elemnts (axis 1) must be 65536 or fewer. '
                'Got LUTD shape {}'.format(value.shape))
        self._LUTD = value

    @property
    def NLUTS(self):
        """
        Number of LUTS for the Image Band. This field shall contain the number
        of LUTs associated with the nth band of the image. LUTs are allowed
        only if the value of the `PVTYPE` field is :code:`INT` or :code:`B`.

        Returns
        -------
        int
        """

        return 0 if self._LUTD is None else self._LUTD.shape[0]

    @property
    def NELUTS(self):
        """
        Number of LUT Entries for the Image Band. This field shall contain
        the number of entries in each of the LUTs for the nth image band.

        Returns
        -------
        int
        """

        return 0 if self._LUTD is None else self._LUTD.shape[1]

    def _get_attribute_bytes(self, attribute):
        if attribute == 'LUTD':
            if self.NLUTS == 0:
                out = b'0'
            else:
                out = '{0:d}{1:05d}'.format(self.NLUTS, self.NELUTS).encode() + \
                      struct.pack('{}B'.format(self.NLUTS * self.NELUTS), *self.LUTD.flatten())
            return out
        else:
            return super(ImageBand, self)._get_attribute_bytes(attribute)

    def _get_attribute_length(self, attribute):
        if attribute == 'LUTD':
            nluts = self.NLUTS
            if nluts == 0:
                return 1
            else:
                neluts = self.NELUTS
                return 6 + nluts * neluts
        else:
            return super(ImageBand, self)._get_attribute_length(attribute)

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'LUTD':
            loc = start
            nluts = int_func(value[loc:loc + 1])
            loc += 1
            if nluts == 0:
                fields['LUTD'] = None
            else:
                neluts = int_func(value[loc:loc + 5])
                loc += 5
                siz = nluts * neluts
                lutd = numpy.array(
                    struct.unpack('{}B'.format(siz), value[loc:loc + siz]), dtype=numpy.uint8).reshape(
                    (nluts, neluts))
                fields['LUTD'] = lutd
                loc += siz
            return loc
        return super(ImageBand, cls)._parse_attribute(fields, attribute, value, start)


class ImageBands(NITFLoop):
    _child_class = ImageBand
    _count_size = 1

    @classmethod
    def _parse_count(cls, value, start):
        loc = start
        count = int_func(value[loc:loc + cls._count_size])
        loc += cls._count_size
        if count == 0:
            # (only) if there are more than 9, a longer field is used
            count = int_func(value[loc:loc + 5])
            loc += 5
        return count, loc

    def _counts_bytes(self):
        siz = len(self.values)
        if siz <= 9:
            return '{0:1d}'.format(siz).encode()
        else:
            return '0{0:05d}'.format(siz).encode()


class ImageComment(NITFElement):
    _ordering = ('COMMENT', )
    _lengths = {'COMMENT': 80}
    COMMENT = _StringDescriptor('COMMENT', True, 80, default_value='', docstring='The image comment')


class ImageComments(NITFLoop):
    _child_class = ImageComment
    _count_size = 1


class ImageSegmentHeader(NITFElement):
    """
    The image segment header - see standards document MIL-STD-2500C for more
    information.
    """

    _ordering = (
        'IM', 'IID1', 'IDATIM', 'TGTID',
        'IID2', 'Security', 'ENCRYP', 'ISORCE',
        'NROWS', 'NCOLS', 'PVTYPE', 'IREP',
        'ICAT', 'ABPP', 'PJUST', 'ICORDS',
        'IGEOLO', 'Comments', 'IC', 'COMRAT', 'Bands',
        'ISYNC', 'IMODE', 'NBPR', 'NBPC', 'NPPBH',
        'NPPBV', 'NBPP', 'IDLVL', 'IALVL',
        'ILOC', 'IMAG', 'UserHeader', 'ExtendedHeader')
    _lengths = {
        'IM': 2, 'IID1': 10, 'IDATIM': 14, 'TGTID': 17,
        'IID2': 80, 'ENCRYP': 1, 'ISORCE': 42,
        'NROWS': 8, 'NCOLS': 8, 'PVTYPE': 3, 'IREP': 8,
        'ICAT': 8, 'ABPP': 2, 'PJUST': 1, 'ICORDS': 1,
        'IGEOLO': 60, 'IC': 2, 'COMRAT': 4, 'ISYNC': 1, 'IMODE': 1,
        'NBPR': 4, 'NBPC': 4, 'NPPBH': 4, 'NPPBV': 4,
        'NBPP': 2, 'IDLVL': 3, 'IALVL': 3, 'ILOC': 10,
        'IMAG': 4, 'UDIDL': 5, 'IXSHDL': 5}
    # Descriptors
    IM = _StringEnumDescriptor(
        'IM', True, 2, {'IM', }, default_value='IM',
        docstring='File part type.')  # type: str
    IID1 = _StringDescriptor(
        'IID1', True, 10, default_value='',
        docstring='Image Identifier 1. This field shall contain a valid alphanumeric identification code '
                  'associated with the image. The valid codes are determined by '
                  'the application.')  # type: str
    IDATIM = _StringDescriptor(
        'IDATIM', True, 14, default_value='',
        docstring='Image Date and Time. This field shall contain the time (UTC) of the image '
                  'acquisition in the format :code:`YYYYMMDDhhmmss`.')  # type: str
    TGTID = _StringDescriptor(
        'TGTID', True, 17, default_value='',
        docstring='Target Identifier. This field shall contain the identification of the primary target '
                  'in the format, :code:`BBBBBBBBBBOOOOOCC`, consisting of ten characters of Basic Encyclopedia '
                  '`(BE)` identifier, followed by five characters of facility OSUFFIX, followed by the two '
                  'character country code as specified in FIPS PUB 10-4.')  # type: str
    IID2 = _StringDescriptor(
        'IID2', True, 80, default_value='',
        docstring='Image Identifier 2. This field can contain the identification of additional '
                  'information about the image.')  # type: str
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags, default_args={},
        docstring='The image security tags.')  # type: NITFSecurityTags
    ENCRYP = _StringEnumDescriptor(
        'ENCRYP', True, 1, {'0'}, default_value='0',
        docstring='Encryption.')  # type: str
    ISORCE = _StringDescriptor(
        'ISORCE', True, 42, default_value='',
        docstring='Image Source. This field shall contain a description of the source of the image. '
                  'If the source of the data is classified, then the description shall be preceded by '
                  'the classification, including codeword(s).')  # type: str
    NROWS = _IntegerDescriptor(
        'NROWS', True, 8, default_value=0,
        docstring='Number of Significant Rows in Image. This field shall contain the total number of rows '
                  'of significant pixels in the image. When the product of the values of the `NPPBV` field '
                  'and the `NBPC` field is greater than the value of the `NROWS` field '
                  r'(:math:`NPPBV \cdot NBPC > NROWS`), the rows indexed with the value of the `NROWS` field '
                  r'to (:math:`NPPBV\cdot NBPC - 1`) shall contain fill data. NOTE: Only the rows indexed '
                  '0 to the value of the `NROWS` field minus 1 of the image contain significant data. '
                  'The pixel fill values are determined by the application.')  # type: int
    NCOLS = _IntegerDescriptor(
        'NCOLS', True, 8, default_value=0,
        docstring='Number of Significant Columns in Image. This field shall contain the total number of '
                  'columns of significant pixels in the image. When the product of the values of the `NPPBH` '
                  'field and the `NBPR` field is greater than the `NCOLS` field '
                  r'(:math:`NPPBH\cdot NBPR > NCOLS`), the columns indexed with the value of the `NCOLS` field '
                  r'to (:math:`NPPBH\cdot NBPR - 1`) shall contain fill data. NOTE: Only the columns '
                  'indexed 0 to the value of the `NCOLS` field minus 1 of the image contain significant data. '
                  'The pixel fill values are determined by the application.')  # type: int
    PVTYPE = _StringEnumDescriptor(
        'PVTYPE', True, 3, {'INT', 'B', 'SI', 'R', 'C'},
        docstring='Pixel Value Type. This field shall contain an indicator of the type of computer representation '
                  'used for the value for each pixel for each band in the image. ')  # type: str
    IREP = _StringEnumDescriptor(
        'IREP', True, 8,
        {'MONO', 'RGB', 'RGB/LUT', 'MULTI', 'NODISPLY', 'NVECTOR', 'POLAR', 'VPH', 'YCbCr601'},
        default_value='NODISPLY',
        docstring='Image Representation. This field shall contain a valid indicator of the processing required '
                  'in order to display an image.')  # type: str
    ICAT = _StringDescriptor(
        'ICAT', True, 8, default_value='SAR',
        docstring='Image Category. This field shall contain a valid indicator of the specific category of image, '
                  'raster or grid data. The specific category of an IS reveals its intended use or the nature '
                  'of its collector.')  # type: str
    ABPP = _IntegerDescriptor(
        'ABPP', True, 2,
        docstring='Actual Bits-Per-Pixel Per Band. This field shall contain the number of “significant bits” for '
                  'the value in each band of each pixel without compression. Even when the image is compressed, '
                  '`ABPP` contains the number of significant bits per pixel that were present in the image '
                  'before compression. This field shall be less than or equal to Number of Bits Per Pixel '
                  '(field `NBPP`). The number of adjacent bits within each `NBPP` is '
                  'used to represent the value.')  # type: int
    PJUST = _StringEnumDescriptor(
        'PJUST', True, 1, {'L', 'R'}, default_value='R',
        docstring='Pixel Justification. When `ABPP` is not equal to `NBPP`, this field indicates whether the '
                  'significant bits are left justified (:code:`L`) or right '
                  'justified (:code:`R`).')  # type: str
    ICORDS = _StringEnumDescriptor(
        'ICORDS', True, 1, {'', 'U', 'G', 'N', 'S', 'D'}, default_value='G',
        docstring='Image Coordinate Representation. This field shall contain a valid code indicating the type '
                  'of coordinate representation used for providing an approximate location of the image in the '
                  'Image Geographic Location field (`IGEOLO`).')  # type: str
    Comments = _NITFElementDescriptor(
        'Comments', True, ImageComments, default_args={},
        docstring='The image comments.')  # type: ImageComments
    Bands = _NITFElementDescriptor(
        'Bands', True, ImageBands, default_args={},
        docstring='The image bands.')  # type: ImageBands
    ISYNC = _IntegerDescriptor(
        'ISYNC', True, 1, default_value=0,
        docstring='Image Sync code. This field is reserved for future use. ')  # type: int
    IMODE = _StringDescriptor(
        'IMODE', True, 1, default_value='P',
        docstring='Image Mode. This field shall indicate how the Image Pixels are '
                  'stored in the NITF file.')  # type: str
    NBPR = _IntegerDescriptor(
        'NBPR', True, 4, default_value=1,
        docstring='Number of Blocks Per Row. This field shall contain the number of image blocks in a row of '
                  'blocks (paragraph 5.4.2.2) in the horizontal direction. If the image consists of only a '
                  'single block, this field shall contain the value one.')  # type: int
    NBPC = _IntegerDescriptor(
        'NBPC', True, 4, default_value=1,
        docstring='Number of Blocks Per Column. This field shall contain the number of image blocks in a column '
                  'of blocks (paragraph 5.4.2.2) in the vertical direction. If the image consists of only a '
                  'single block, this field shall contain the value one.')  # type: int
    NPPBH = _IntegerDescriptor(
        'NPPBH', True, 4, default_value=0,
        docstring='Number of Pixels Per Block Horizontal. This field shall contain the number of pixels horizontally '
                  'in each block of the image. It shall be the case that the product of the values of the `NBPR` '
                  'field and the `NPPBH` field is greater than or equal to the value of the `NCOLS` field '
                  r'(:math:`NBPR\cdot NPPBH \geq NCOLS`). When NBPR is :code:`1`, setting the `NPPBH` '
                  'value to :code:`0` designates that the number of pixels horizontally is specified by the '
                  'value in NCOLS.')  # type: int
    NPPBV = _IntegerDescriptor(
        'NPPBV', True, 4, default_value=0,
        docstring='Number of Pixels Per Block Vertical. This field shall contain the number of pixels vertically '
                  'in each block of the image. It shall be the case that the product of the values of the `NBPC` '
                  'field and the `NPPBV` field is greater than or equal to the value of the `NROWS` field '
                  r'(:math:`NBPC\cdot NPPBV \geq NROWS`). When `NBPC` is :code:`1`, setting the `NPPBV` value '
                  r'to :code:`0` designates that the number of pixels vertically is specified by '
                  r'the value in `NROWS`.')  # type: int
    NBPP = _IntegerDescriptor(
        'NBPP', True, 2, default_value=0,
        docstring='Number of Bits Per Pixel Per Band.')  # type: int
    IDLVL = _IntegerDescriptor(
        'IDLVL', True, 3, default_value=0,
        docstring='Image Display Level. This field shall contain a valid value that indicates the display level of '
                  'the image relative to other displayed file components in a composite display. The valid values '
                  'are :code:`1-999`. The display level of each displayable segment (image or graphic) within a file '
                  'shall be unique.')  # type: int
    IALVL = _IntegerDescriptor(
        'IALVL', True, 3, default_value=0,
        docstring='Attachment Level. This field shall contain a valid value that indicates the attachment '
                  'level of the image.')  # type: int
    ILOC = _StringDescriptor(
        'ILOC', True, 10, default_value='',
        docstring='Image Location. The image location is the location of the first pixel of the first line of the '
                  'image. This field shall contain the image location offset from the `ILOC` or `SLOC` value '
                  'of the segment to which the image is attached or from the origin of the CCS when the image '
                  'is unattached (`IALVL` contains :code:`0`). A row or column value of :code:`0` indicates no offset. '
                  'Positive row and column values indicate offsets down and to the right while negative row and '
                  'column values indicate offsets up and to the left.')  # type: str
    IMAG = _StringDescriptor(
        'IMAG', True, 4, default_value='1.0',
        docstring='Image Magnification. This field shall contain the magnification (or reduction) factor of the '
                  'image relative to the original source image. Decimal values are used to indicate magnification, '
                  'and decimal fraction values indicate reduction. For example, :code:`2.30` indicates the original '
                  'image has been magnified by a factor of :code:`2.30`, while :code:`0.5` indicates '
                  'the original image has been reduced by a factor of 2.')  # type: str
    UserHeader = _NITFElementDescriptor(
        'UserHeader', True, UserHeaderType, default_args={},
        docstring='User defined header.')  # type: UserHeaderType
    ExtendedHeader = _NITFElementDescriptor(
        'ExtendedHeader', True, UserHeaderType, default_args={},
        docstring='Extended subheader - TRE list.')  # type: UserHeaderType

    def __init__(self, **kwargs):
        self._IC = None
        self._COMRAT = None
        self._IGEOLO = None
        super(ImageSegmentHeader, self).__init__(**kwargs)

    @property
    def IC(self):
        """
        str: Image Compression. This field shall contain a valid code indicating
        the form of compression used in representing the image data.

        Valid values for this field are, :code:`C1` to represent bi-level, :code:`C3`
        to represent JPEG, :code:`C4` to represent Vector Quantization, :code:`C5`
        to represent lossless JPEG, :code:`I1` to represent down sampled JPEG,
        and :code:`NC` to represent the image is not compressed. Also valid are
        :code:`M1, M3, M4`, and :code:`M5` for compressed images, and :code:`NM`
        for uncompressed images indicating an image that contains a block
        mask and/or a pad pixel mask. :code:`C6` and :code:`M6` are reserved values
        that will represent a future correlated multicomponent compression
        algorithm. :code:`C7` and :code:`M7` are reserved values that will represent
        a future complex SAR compression. :code:`C8` and :code:`M8` are the values
        for ISO standard compression JPEG 2000.

        The format of a mask image is identical to the format of its corresponding non-masked image
        except for the presence of an Image Data Mask at the beginning of
        the image data area. The format of the Image Data Mask is described
        in paragraph 5.4.3.2 and is shown in table A-3(A). The definitions
        of the compression schemes associated with codes :code:`C1/M1, C3/M3, C4/M4, C5/M5`
        are given, respectively, in ITU- T T.4, AMD2, MIL-STD-188-198A,
        MIL-STD- 188-199, and NGA N0106-97. :code:`C1` is found in ITU- T T.4 AMD2,
        :code:`C3` is found in MIL-STD-188-198A, :code:`C4` is found in MIL-STD-188-199,
        and :code:`C5` and :code:`I1` are found in NGA N0106-97. (NOTE: :code:`C2` (ARIDPCM) is not
        valid in NITF 2.1.) The definition of the compression scheme associated
        with codes :code:`C8/M8` is found in ISO/IEC 15444- 1:2000 (with amendments 1 and 2).
        """

        return self._IC

    @IC.setter
    def IC(self, value):
        value = _parse_str(value, 2, 'NC', 'IC', self)
        if value not in {
                'NC', 'NM', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'I1',
                'M1', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'}:
            raise ValueError('IC got invalid value {}'.format(value))
        self._IC = value
        if value in ('NC', 'NM'):
            self._COMRAT = None
        elif self._COMRAT is not None:
            self._COMRAT = '\x20'*4

    @property
    def COMRAT(self):
        """
        None|str: Compression Rate Code. If the IC field contains one of
        :code:`C1, C3, C4, C5, C8, M1, M3, M4, M5, M8, I1`, this field shall be contain
        a code indicating the compression rate for the image.

        If `IC` is :code:`NC` or :code:`NM`, then this will be set to :code:`None`.
        """

        return self._COMRAT

    @COMRAT.setter
    def COMRAT(self, value):
        value = _parse_str(value, 4, None, 'COMRAT', self)
        if value is None and self.IC not in ('NC', 'NM'):
            value = '\x20'*4
            logging.error(
                'COMRAT value is None, but IC is not in {"NC", "NM"}. '
                'This must be resolved.')
        if value is not None and self.IC in ('NC', 'NM'):
            value = None
            logging.error(
                'COMRAT value is something other than None, but IC in {"NC", "NM"}. '
                'This is invalid, and COMRAT is being set to None.')
        self._COMRAT = value

    @property
    def IGEOLO(self):
        """
        None|str: Image Geographic Location. This field, when present, shall contain
        an approximate geographic location which is not intended for analytical purposes
        (e.g., targeting, mensuration, distance calculation); it is intended to support
        general user appreciation for the image location (e.g., cataloguing). The
        representation of the image corner locations is specified in the `ICORDS` field.
        The locations of the four corners of the (significant) image data shall be given
        in image coordinate order: (0,0), (0, MaxCol), (MaxRow, MaxCol), (MaxRow, 0).
        MaxCol and MaxRow shall be determined from the values contained, respectively,
        in the `NCOLS` field and the `NROWS` field.
        """

        return self._IGEOLO

    @IGEOLO.setter
    def IGEOLO(self, value):
        value = _parse_str(value, 60, None, 'IGEOLO', self)
        if value is None and self.ICORDS.strip() != '':
            value = '\x20'*60
        if value is not None and self.ICORDS.strip() == '':
            value = None
        self._IGEOLO = value

    def _get_attribute_length(self, fld):
        if fld in ['COMRAT', 'IGEOLO']:
            if getattr(self, '_'+fld) is None:
                return 0
            else:
                return self._lengths[fld]
        else:
            return super(ImageSegmentHeader, self)._get_attribute_length(fld)

    @classmethod
    def minimum_length(cls):
        # COMRAT and IGEOLO may not be there
        return super(ImageSegmentHeader, cls).minimum_length() - 64

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'IC':
            val = value[start:start+2].decode('utf-8')
            fields['IC'] = val
            if val in ('NC', 'NM'):
                fields['COMRAT'] = None
            return start+2
        elif attribute == 'ICORDS':
            fields['ICORDS'] = value[start:start+1]
            if fields['ICORDS'] == b' ':
                fields['IGEOLO'] = None
            return start+1
        else:
            return super(ImageSegmentHeader, cls)._parse_attribute(fields, attribute, value, start)


#########
# NITF 2.0 version

class ImageSegmentHeader0(NITFElement):
    """
    The image segment header for NITF version 2.0 - see standards document
    MIL-STD-2500A for more information.
    """

    _ordering = (
        'IM', 'IID', 'IDATIM', 'TGTID',
        'ITITLE', 'Security', 'ENCRYP', 'ISORCE',
        'NROWS', 'NCOLS', 'PVTYPE', 'IREP',
        'ICAT', 'ABPP', 'PJUST', 'ICORDS',
        'IGEOLO', 'Comments', 'IC', 'COMRAT', 'Bands',
        'ISYNC', 'IMODE', 'NBPR', 'NBPC', 'NPPBH',
        'NPPBV', 'NBPP', 'IDLVL', 'IALVL',
        'ILOC', 'IMAG', 'UserHeader', 'ExtendedHeader')
    _lengths = {
        'IM': 2, 'IID': 10, 'IDATIM': 14, 'TGTID': 17,
        'ITITLE': 80, 'ENCRYP': 1, 'ISORCE': 42,
        'NROWS': 8, 'NCOLS': 8, 'PVTYPE': 3, 'IREP': 8,
        'ICAT': 8, 'ABPP': 2, 'PJUST': 1, 'ICORDS': 1,
        'IGEOLO': 60, 'IC': 2, 'COMRAT': 4, 'ISYNC': 1, 'IMODE': 1,
        'NBPR': 4, 'NBPC': 4, 'NPPBH': 4, 'NPPBV': 4,
        'NBPP': 2, 'IDLVL': 3, 'IALVL': 3, 'ILOC': 10,
        'IMAG': 4, 'UDIDL': 5, 'IXSHDL': 5}
    # Descriptors
    IM = _StringEnumDescriptor(
        'IM', True, 2, {'IM', }, default_value='IM',
        docstring='File part type.')  # type: str
    IID = _StringDescriptor(
        'IID', True, 10, default_value='',
        docstring='Image Identifier 1. This field shall contain a valid alphanumeric identification code '
                  'associated with the image. The valid codes are determined by '
                  'the application.')  # type: str
    IDATIM = _StringDescriptor(
        'IDATIM', True, 14, default_value='',
        docstring='Image Date and Time. This field shall contain the time (UTC) of the image '
                  'acquisition in the format :code:`YYYYMMDDhhmmss`.')  # type: str
    TGTID = _StringDescriptor(
        'TGTID', True, 17, default_value='',
        docstring='Target Identifier. This field shall contain the identification of the primary target '
                  'in the format, :code:`BBBBBBBBBBOOOOOCC`, consisting of ten characters of Basic Encyclopedia '
                  '`(BE)` identifier, followed by five characters of facility OSUFFIX, followed by the two '
                  'character country code as specified in FIPS PUB 10-4.')  # type: str
    ITITLE = _StringDescriptor(
        'ITITLE', True, 80, default_value='',
        docstring='Image Identifier 2. This field can contain the identification of additional '
                  'information about the image.')  # type: str
    Security = _NITFElementDescriptor(
        'Security', True, NITFSecurityTags0, default_args={},
        docstring='The image security tags.')  # type: NITFSecurityTags0
    ENCRYP = _StringEnumDescriptor(
        'ENCRYP', True, 1, {'0'}, default_value='0',
        docstring='Encryption.')  # type: str
    ISORCE = _StringDescriptor(
        'ISORCE', True, 42, default_value='',
        docstring='Image Source. This field shall contain a description of the source of the image. '
                  'If the source of the data is classified, then the description shall be preceded by '
                  'the classification, including codeword(s).')  # type: str
    NROWS = _IntegerDescriptor(
        'NROWS', True, 8, default_value=0,
        docstring='Number of Significant Rows in Image. This field shall contain the total number of rows '
                  'of significant pixels in the image. When the product of the values of the `NPPBV` field '
                  'and the `NBPC` field is greater than the value of the `NROWS` field '
                  r'(:math:`NPPBV \cdot NBPC > NROWS`), the rows indexed with the value of the `NROWS` field '
                  r'to (:math:`NPPBV\cdot NBPC - 1`) shall contain fill data. NOTE: Only the rows indexed '
                  '0 to the value of the `NROWS` field minus 1 of the image contain significant data. '
                  'The pixel fill values are determined by the application.')  # type: int
    NCOLS = _IntegerDescriptor(
        'NCOLS', True, 8, default_value=0,
        docstring='Number of Significant Columns in Image. This field shall contain the total number of '
                  'columns of significant pixels in the image. When the product of the values of the `NPPBH` '
                  'field and the `NBPR` field is greater than the `NCOLS` field '
                  r'(:math:`NPPBH\cdot NBPR > NCOLS`), the columns indexed with the value of the `NCOLS` field '
                  r'to (:math:`NPPBH\cdot NBPR - 1`) shall contain fill data. NOTE: Only the columns '
                  'indexed 0 to the value of the `NCOLS` field minus 1 of the image contain significant data. '
                  'The pixel fill values are determined by the application.')  # type: int
    PVTYPE = _StringEnumDescriptor(
        'PVTYPE', True, 3, {'INT', 'B', 'SI', 'R', 'C'},
        docstring='Pixel Value Type. This field shall contain an indicator of the type of computer representation '
                  'used for the value for each pixel for each band in the image. ')  # type: str
    IREP = _StringEnumDescriptor(
        'IREP', True, 8,
        {'MONO', 'RGB', 'RGB/LUT', 'MULTI', 'NODISPLY', 'NVECTOR', 'POLAR', 'VPH', 'YCbCr601'},
        default_value='NODISPLY',
        docstring='Image Representation. This field shall contain a valid indicator of the processing required '
                  'in order to display an image.')  # type: str
    ICAT = _StringDescriptor(
        'ICAT', True, 8, default_value='SAR',
        docstring='Image Category. This field shall contain a valid indicator of the specific category of image, '
                  'raster or grid data. The specific category of an IS reveals its intended use or the nature '
                  'of its collector.')  # type: str
    ABPP = _IntegerDescriptor(
        'ABPP', True, 2,
        docstring='Actual Bits-Per-Pixel Per Band. This field shall contain the number of “significant bits” for '
                  'the value in each band of each pixel without compression. Even when the image is compressed, '
                  '`ABPP` contains the number of significant bits per pixel that were present in the image '
                  'before compression. This field shall be less than or equal to Number of Bits Per Pixel '
                  '(field `NBPP`). The number of adjacent bits within each `NBPP` is '
                  'used to represent the value.')  # type: int
    PJUST = _StringEnumDescriptor(
        'PJUST', True, 1, {'L', 'R'}, default_value='R',
        docstring='Pixel Justification. When `ABPP` is not equal to `NBPP`, this field indicates whether the '
                  'significant bits are left justified (:code:`L`) or right '
                  'justified (:code:`R`).')  # type: str
    ICORDS = _StringEnumDescriptor(
        'ICORDS', True, 1, {'U', 'G', 'C', 'N'}, default_value='G',
        docstring='Image Coordinate Representation. This field shall contain a valid code indicating the type '
                  'of coordinate representation used for providing an approximate location of the image in the '
                  'Image Geographic Location field (`IGEOLO`).')  # type: str
    Comments = _NITFElementDescriptor(
        'Comments', True, ImageComments, default_args={},
        docstring='The image comments.')  # type: ImageComments
    Bands = _NITFElementDescriptor(
        'Bands', True, ImageBands, default_args={},
        docstring='The image bands.')  # type: ImageBands
    ISYNC = _IntegerDescriptor(
        'ISYNC', True, 1, default_value=0,
        docstring='Image Sync code. This field is reserved for future use. ')  # type: int
    IMODE = _StringDescriptor(
        'IMODE', True, 1, default_value='P',
        docstring='Image Mode. This field shall indicate how the Image Pixels are '
                  'stored in the NITF file.')  # type: str
    NBPR = _IntegerDescriptor(
        'NBPR', True, 4, default_value=1,
        docstring='Number of Blocks Per Row. This field shall contain the number of image blocks in a row of '
                  'blocks (paragraph 5.4.2.2) in the horizontal direction. If the image consists of only a '
                  'single block, this field shall contain the value one.')  # type: int
    NBPC = _IntegerDescriptor(
        'NBPC', True, 4, default_value=1,
        docstring='Number of Blocks Per Column. This field shall contain the number of image blocks in a column '
                  'of blocks (paragraph 5.4.2.2) in the vertical direction. If the image consists of only a '
                  'single block, this field shall contain the value one.')  # type: int
    NPPBH = _IntegerDescriptor(
        'NPPBH', True, 4, default_value=0,
        docstring='Number of Pixels Per Block Horizontal. This field shall contain the number of pixels horizontally '
                  'in each block of the image. It shall be the case that the product of the values of the `NBPR` '
                  'field and the `NPPBH` field is greater than or equal to the value of the `NCOLS` field '
                  r'(:math:`NBPR\cdot NPPBH \geq NCOLS`). When NBPR is :code:`1`, setting the `NPPBH` '
                  'value to :code:`0` designates that the number of pixels horizontally is specified by the '
                  'value in NCOLS.')  # type: int
    NPPBV = _IntegerDescriptor(
        'NPPBV', True, 4, default_value=0,
        docstring='Number of Pixels Per Block Vertical. This field shall contain the number of pixels vertically '
                  'in each block of the image. It shall be the case that the product of the values of the `NBPC` '
                  'field and the `NPPBV` field is greater than or equal to the value of the `NROWS` field '
                  r'(:math:`NBPC\cdot NPPBV \geq NROWS`). When `NBPC` is :code:`1`, setting the `NPPBV` value '
                  r'to :code:`0` designates that the number of pixels vertically is specified by '
                  r'the value in `NROWS`.')  # type: int
    NBPP = _IntegerDescriptor(
        'NBPP', True, 2, default_value=0,
        docstring='Number of Bits Per Pixel Per Band.')  # type: int
    IDLVL = _IntegerDescriptor(
        'IDLVL', True, 3, default_value=0,
        docstring='Image Display Level. This field shall contain a valid value that indicates the display level of '
                  'the image relative to other displayed file components in a composite display. The valid values '
                  'are :code:`1-999`. The display level of each displayable segment (image or graphic) within a file '
                  'shall be unique.')  # type: int
    IALVL = _IntegerDescriptor(
        'IALVL', True, 3, default_value=0,
        docstring='Attachment Level. This field shall contain a valid value that indicates the attachment '
                  'level of the image.')  # type: int
    ILOC = _StringDescriptor(
        'ILOC', True, 10, default_value='',
        docstring='Image Location. The image location is the location of the first pixel of the first line of the '
                  'image. This field shall contain the image location offset from the `ILOC` or `SLOC` value '
                  'of the segment to which the image is attached or from the origin of the CCS when the image '
                  'is unattached (`IALVL` contains :code:`0`). A row or column value of :code:`0` indicates no offset. '
                  'Positive row and column values indicate offsets down and to the right while negative row and '
                  'column values indicate offsets up and to the left.')  # type: str
    IMAG = _StringDescriptor(
        'IMAG', True, 4, default_value='1.0',
        docstring='Image Magnification. This field shall contain the magnification (or reduction) factor of the '
                  'image relative to the original source image. Decimal values are used to indicate magnification, '
                  'and decimal fraction values indicate reduction. For example, :code:`2.30` indicates the original '
                  'image has been magnified by a factor of :code:`2.30`, while :code:`0.5` indicates '
                  'the original image has been reduced by a factor of 2.')  # type: str
    UserHeader = _NITFElementDescriptor(
        'UserHeader', True, UserHeaderType, default_args={},
        docstring='User defined header.')  # type: UserHeaderType
    ExtendedHeader = _NITFElementDescriptor(
        'ExtendedHeader', True, UserHeaderType, default_args={},
        docstring='Extended subheader - TRE list.')  # type: UserHeaderType

    def __init__(self, **kwargs):
        self._IC = None
        self._COMRAT = None
        self._IGEOLO = None
        super(ImageSegmentHeader0, self).__init__(**kwargs)

    @property
    def IC(self):
        """
        str: Image Compression. This field shall contain a valid code indicating
        the form of compression used in representing the image data.

        Valid values for this field are, :code:`C1` to represent bi-level, :code:`C3`
        to represent JPEG, :code:`C4` to represent Vector Quantization, :code:`C5`
        to represent lossless JPEG, :code:`I1` to represent down sampled JPEG,
        and :code:`NC` to represent the image is not compressed. Also valid are
        :code:`M1, M3, M4`, and :code:`M5` for compressed images, and :code:`NM`
        for uncompressed images indicating an image that contains a block
        mask and/or a pad pixel mask. :code:`C6` and :code:`M6` are reserved values
        that will represent a future correlated multicomponent compression
        algorithm. :code:`C7` and :code:`M7` are reserved values that will represent
        a future complex SAR compression. :code:`C8` and :code:`M8` are the values
        for ISO standard compression JPEG 2000.

        The format of a mask image is identical to the format of its corresponding non-masked image
        except for the presence of an Image Data Mask at the beginning of
        the image data area. The format of the Image Data Mask is described
        in paragraph 5.4.3.2 and is shown in table A-3(A). The definitions
        of the compression schemes associated with codes :code:`C1/M1, C3/M3, C4/M4, C5/M5`
        are given, respectively, in ITU- T T.4, AMD2, MIL-STD-188-198A,
        MIL-STD- 188-199, and NGA N0106-97. :code:`C1` is found in ITU- T T.4 AMD2,
        :code:`C3` is found in MIL-STD-188-198A, :code:`C4` is found in MIL-STD-188-199,
        and :code:`C5` and :code:`I1` are found in NGA N0106-97. (NOTE: :code:`C2` (ARIDPCM) is not
        valid in NITF 2.1.) The definition of the compression scheme associated
        with codes :code:`C8/M8` is found in ISO/IEC 15444- 1:2000 (with amendments 1 and 2).
        """

        return self._IC

    @IC.setter
    def IC(self, value):
        value = _parse_str(value, 2, 'NC', 'IC', self)
        if value not in {
                'NC', 'NM', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'I1',
                'M1', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'}:
            raise ValueError('IC got invalid value {}'.format(value))
        self._IC = value
        if value in ('NC', 'NM'):
            self._COMRAT = None
        elif self._COMRAT is not None:
            self._COMRAT = '\x20'*4

    @property
    def COMRAT(self):
        """
        None|str: Compression Rate Code. If the IC field contains one of
        :code:`C1, C3, C4, C5, C8, M1, M3, M4, M5, M8, I1`, this field shall be contain
        a code indicating the compression rate for the image.

        If `IC` is :code:`NC` or :code:`NM`, then this will be set to :code:`None`.
        """

        return self._COMRAT

    @COMRAT.setter
    def COMRAT(self, value):
        value = _parse_str(value, 4, None, 'COMRAT', self)
        if value is None and self.IC not in ('NC', 'NM'):
            value = '\x20'*4
            logging.error(
                'COMRAT value is None, but IC is not in {"NC", "NM"}. '
                'This must be resolved.')
        if value is not None and self.IC in ('NC', 'NM'):
            value = None
            logging.error(
                'COMRAT value is something other than None, but IC in {"NC", "NM"}. '
                'This is invalid, and COMRAT is being set to None.')
        self._COMRAT = value

    @property
    def IGEOLO(self):
        """
        None|str: Image Geographic Location. This field, when present, shall contain
        an approximate geographic location which is not intended for analytical purposes
        (e.g., targeting, mensuration, distance calculation); it is intended to support
        general user appreciation for the image location (e.g., cataloguing). The
        representation of the image corner locations is specified in the `ICORDS` field.
        The locations of the four corners of the (significant) image data shall be given
        in image coordinate order: (0,0), (0, MaxCol), (MaxRow, MaxCol), (MaxRow, 0).
        MaxCol and MaxRow shall be determined from the values contained, respectively,
        in the `NCOLS` field and the `NROWS` field.
        """

        return self._IGEOLO

    @IGEOLO.setter
    def IGEOLO(self, value):
        value = _parse_str(value, 60, None, 'IGEOLO', self)
        if value is None and self.ICORDS.strip() != '':
            value = '\x20'*60
        if value is not None and self.ICORDS.strip() == '':
            value = None
        self._IGEOLO = value

    def _get_attribute_length(self, fld):
        if fld in ['COMRAT', 'IGEOLO']:
            if getattr(self, '_'+fld) is None:
                return 0
            else:
                return self._lengths[fld]
        else:
            return super(ImageSegmentHeader0, self)._get_attribute_length(fld)

    @classmethod
    def minimum_length(cls):
        # COMRAT and IGEOLO may not be there
        return super(ImageSegmentHeader0, cls).minimum_length() - 64

    @classmethod
    def _parse_attribute(cls, fields, attribute, value, start):
        if attribute == 'IC':
            val = value[start:start+2].decode('utf-8')
            fields['IC'] = val
            if val in ('NC', 'NM'):
                fields['COMRAT'] = None
            out = start+2
        elif attribute == 'ICORDS':
            fields['ICORDS'] = value[start:start+1]
            if fields['ICORDS'] == b'N':
                fields['IGEOLO'] = None
            out = start+1
        else:
            out = super(ImageSegmentHeader0, cls)._parse_attribute(fields, attribute, value, start)
        return out
