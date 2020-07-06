# -*- coding: utf-8 -*-

from typing import Union, Tuple
import os

import numpy

from .base import NITFElement, UserHeaderType, _IntegerDescriptor,\
    _StringDescriptor, _StringEnumDescriptor, _NITFElementDescriptor, _RawDescriptor, \
    _ItemArrayHeaders, int_func
from .security import NITFSecurityTags
from .image import ImageSegmentHeader
from .graphics import GraphicsSegmentHeader
from .text import TextSegmentHeader
from .des import DataExtensionHeader
from .res import ReservedExtensionHeader


#############
# various subheader and item information

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


############

class NITFHeader(NITFElement):
    """
    The main NITF file header - see standards document MIL-STD-2500C for more
    information.
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
        'FL', True, 12, docstring='The size in bytes of the entire file.')
    ImageSegments = _NITFElementDescriptor(
        'ImageSegments', True, ImageSegmentsType, default_args={},
        docstring='The image segment basic information.')
    GraphicsSegments = _NITFElementDescriptor(
        'GraphicsSegments', True, GraphicsSegmentsType, default_args={},
        docstring='The graphics segment basic information.')
    TextSegments = _NITFElementDescriptor(
        'TextSegments', True, TextSegmentsType, default_args={},
        docstring='The text segment basic information.')
    DataExtensions = _NITFElementDescriptor(
        'DataExtensions', True, DataExtensionsType, default_args={},
        docstring='The data extension basic information.')
    ReservedExtensions = _NITFElementDescriptor(
        'ReservedExtensions', True, ReservedExtensionsType, default_args={},
        docstring='The reserved extension basic information.')
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


#####
# A general nitf header interpreter - intended for extension

class NITFDetails(object):
    """
    This class allows for somewhat general parsing of the header information in a NITF 2.1 file.
    """

    __slots__ = (
        '_file_name', '_nitf_header',
        'img_subheader_offsets', 'img_segment_offsets',
        'graphics_subheader_offsets', 'graphics_segment_offsets',
        'text_subheader_offsets', 'text_segment_offsets',
        'des_subheader_offsets', 'des_segment_offsets',
        'res_subheader_offsets', 'res_segment_offsets')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF 2.1 file
        """

        self._file_name = file_name

        if not os.path.isfile(file_name):
            raise IOError('Path {} is not a file'.format(file_name))

        with open(file_name, mode='rb') as fi:
            # Read the first 9 bytes to verify NITF
            try:
                version_info = fi.read(9).decode('utf-8')
            except:
                raise IOError('Not a NITF 2.1 file.')

            if version_info != 'NITF02.10':
                raise IOError('Not a NITF 2.1 file.')
            # get the header length
            fi.seek(354)  # offset to first field of interest
            header_length = int_func(fi.read(6))
            # go back to the beginning of the file, and parse the whole header
            fi.seek(0)
            header_string = fi.read(header_length)
            self._nitf_header = NITFHeader.from_bytes(header_string, 0)

        curLoc = self._nitf_header.HL
        # populate image segment offset information
        curLoc, self.img_subheader_offsets, self.img_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.ImageSegments)
        # populate graphics segment offset information
        curLoc, self.graphics_subheader_offsets, self.graphics_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.GraphicsSegments)
        # populate text segment offset information
        curLoc, self.text_subheader_offsets, self.text_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.TextSegments)
        # populate data extension offset information
        curLoc, self.des_subheader_offsets, self.des_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.DataExtensions)
        # populate data extension offset information
        curLoc, self.res_subheader_offsets, self.res_segment_offsets = self._element_offsets(
            curLoc, self._nitf_header.ReservedExtensions)

    @staticmethod
    def _element_offsets(curLoc, item_array_details):
        # type: (int, _ItemArrayHeaders) -> Tuple[int, Union[None, numpy.ndarray], Union[None, numpy.ndarray]]
        subhead_sizes = item_array_details.subhead_sizes
        item_sizes = item_array_details.item_sizes
        if subhead_sizes.size == 0:
            return curLoc, None, None

        subhead_offsets = numpy.full(subhead_sizes.shape, curLoc, dtype=numpy.int64)
        subhead_offsets[1:] += numpy.cumsum(subhead_sizes[:-1]) + numpy.cumsum(item_sizes[:-1])
        item_offsets = subhead_offsets + subhead_sizes
        curLoc = item_offsets[-1] + item_sizes[-1]
        return curLoc, subhead_offsets, item_offsets

    @property
    def file_name(self):
        """str: the file name."""
        return self._file_name

    @property
    def nitf_header(self):  # type: () -> NITFHeader
        """NITFHeader: the nitf header object"""
        return self._nitf_header

    def _fetch_item(self, name, index, offsets, sizes):
        # type: (str, int, numpy.ndarray, numpy.ndarray) -> bytes
        if index >= offsets.size:
            raise IndexError(
                'There are only {0:d} {1:s}, invalid {1:s} position {2:d}'.format(
                    offsets.size, name, index))
        the_offset = offsets[index]
        the_size = sizes[index]
        with open(self._file_name, mode='rb') as fi:
            fi.seek(int_func(the_offset))
            the_item = fi.read(int_func(the_size))
        return the_item

    def get_image_subheader_bytes(self, index):
        """
        Fetches the image segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('image subheader',
                                index,
                                self.img_subheader_offsets,
                                self._nitf_header.ImageSegments.subhead_sizes)

    def parse_image_subheader(self, index):
        """
        Parse the image segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ImageSegmentHeader
        """

        ih = self.get_image_subheader_bytes(index)
        return ImageSegmentHeader.from_bytes(ih, 0)

    def get_text_subheader_bytes(self, index):
        """
        Fetches the text segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('text subheader',
                                index,
                                self.text_subheader_offsets,
                                self._nitf_header.TextSegments.subhead_sizes)

    def get_text_bytes(self, index):
        """
        Fetches the text extension segment bytes at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('text segment',
                                index,
                                self.text_segment_offsets,
                                self._nitf_header.TextSegments.item_sizes)

    def parse_text_subheader(self, index):
        """
        Parse the text segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        TextSegmentHeader
        """

        th = self.get_text_subheader_bytes(index)
        return TextSegmentHeader.from_bytes(th, 0)

    def get_graphics_subheader_bytes(self, index):
        """
        Fetches the graphics segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('graphics subheader',
                                index,
                                self.graphics_subheader_offsets,
                                self._nitf_header.GraphicsSegments.subhead_sizes)

    def get_graphics_bytes(self, index):
        """
        Fetches the graphics extension segment bytes at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('graphics segment',
                                index,
                                self.graphics_segment_offsets,
                                self._nitf_header.GraphicsSegments.item_sizes)

    def parse_graphics_subheader(self, index):
        """
        Parse the graphics segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        GraphicsSegmentHeader
        """

        gh = self.get_graphics_subheader_bytes(index)
        return GraphicsSegmentHeader.from_bytes(gh, 0)

    def get_des_subheader_bytes(self, index):
        """
        Fetches the data extension segment subheader bytes at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('des subheader',
                                index,
                                self.des_subheader_offsets,
                                self._nitf_header.DataExtensions.subhead_sizes)

    def get_des_bytes(self, index):
        """
        Fetches the data extension segment bytes at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('des',
                                index,
                                self.des_segment_offsets,
                                self._nitf_header.DataExtensions.item_sizes)

    def parse_des_subheader(self, index):
        """
        Parse the data extension segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        DataExtensionHeader
        """

        dh = self.get_des_subheader_bytes(index)
        return DataExtensionHeader.from_bytes(dh, 0)

    def get_res_subheader_bytes(self, index):
        """
        Fetches the reserved extension segment subheader bytes at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('res subheader',
                                index,
                                self.res_subheader_offsets,
                                self._nitf_header.ReservedExtensions.subhead_sizes)

    def get_res_bytes(self, index):
        """
        Fetches the reserved extension segment bytes at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item('res',
                                index,
                                self.res_segment_offsets,
                                self._nitf_header.ReservedExtensions.item_sizes)

    def parse_res_subheader(self, index):
        """
        Parse the reserved extension subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ReservedExtensionHeader
        """

        rh = self.get_res_subheader_bytes(index)
        return ReservedExtensionHeader.from_bytes(rh, 0)
