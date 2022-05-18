"""
Module laying out basic functionality for reading and writing NITF files.

**Updated extensively in version 1.3.0**.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


# TODO:
#  - update NITF reading paradigm
#  - update NITF writing paradigm


import logging
import os
from typing import Union, List, Tuple, BinaryIO, Sequence, Optional
import re
import mmap
from tempfile import mkstemp
from collections import OrderedDict
import struct
from io import BytesIO

import numpy

from sarpy.io.general.base import SarpyIOError, BaseReader, BaseWriter  #, BIPWriter
from sarpy.io.general.format_function import FormatFunction, ComplexFormatFunction, \
    SingleLUTFormatFunction
from sarpy.io.general.data_segment import DataSegment, BandAggregateSegment, \
    BlockAggregateSegment, SubsetSegment, NumpyMemmapSegment, FileReadDataSegment

# noinspection PyProtectedMember
from sarpy.io.general.nitf_elements.nitf_head import NITFHeader, NITFHeader0, \
    ImageSegmentsType, DataExtensionsType, _ItemArrayHeaders
from sarpy.io.general.nitf_elements.text import TextSegmentHeader, TextSegmentHeader0
from sarpy.io.general.nitf_elements.graphics import GraphicsSegmentHeader
from sarpy.io.general.nitf_elements.symbol import SymbolSegmentHeader
from sarpy.io.general.nitf_elements.label import LabelSegmentHeader
from sarpy.io.general.nitf_elements.res import ReservedExtensionHeader, ReservedExtensionHeader0
from sarpy.io.general.nitf_elements.security import NITFSecurityTags
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, ImageSegmentHeader0, MaskSubheader
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, DataExtensionHeader0
from sarpy.io.general.utils import is_file_like, is_nitf, is_real_file

from sarpy.io.complex.sicd_elements.blocks import LatLonType
from sarpy.geometry.geocoords import ecf_to_geodetic, geodetic_to_ecf
from sarpy.geometry.latlon import num as lat_lon_parser

# import some optional dependencies
try:
    # noinspection PyPackageRequirements
    import pyproj
except ImportError:
    pyproj = None

try:
    # noinspection PyPackageRequirements
    from PIL import PIL_Image
except ImportError:
    PIL_Image = None


logger = logging.getLogger(__name__)

_unhandled_version_text = 'Unhandled NITF version `{}`'


#####
# helper functions

def extract_image_corners(
        img_header: Union[ImageSegmentHeader, ImageSegmentHeader0]) -> Union[None, numpy.ndarray]:
    """
    Extract the image corner point array for the image segment header.

    Parameters
    ----------
    img_header : ImageSegmentHeader

    Returns
    -------
    numpy.ndarray
    """

    corner_string = img_header.IGEOLO
    # NB: there are 4 corner point string, each of length 15
    corner_strings = [corner_string[start:stop] for start, stop in zip(range(0, 59, 15), range(15, 74, 15))]

    icps = []
    # TODO: handle ICORDS == 'U', which is MGRS
    if img_header.ICORDS in ['N', 'S']:
        if pyproj is None:
            logger.error('ICORDS is {}, which requires pyproj, which was not successfully imported.')
            return None
        for entry in corner_strings:
            the_proj = pyproj.Proj(proj='utm', zone=int(entry[:2]), south=(img_header.ICORDS == 'S'), ellps='WGS84')
            lon, lat = the_proj(float(entry[2:8]), float(entry[8:]), inverse=True)
            icps.append([lon, lat])
    elif img_header.ICORDS == 'D':
        icps = [[float(corner[:7]), float(corner[7:])] for corner in corner_strings]
    elif img_header.ICORDS == 'G':
        icps = [[lat_lon_parser(corner[:7]), lat_lon_parser(corner[7:])] for corner in corner_strings]
    else:
        logger.error('Got unhandled ICORDS {}'.format(img_header.ICORDS))
        return None
    return numpy.array(icps, dtype='float64')


class NITFDetails(object):
    """
    This class allows for somewhat general parsing of the header information in
    a NITF 2.0 or 2.1 file.
    """

    __slots__ = (
        '_file_name', '_file_object', '_close_after',
        '_nitf_version', '_nitf_header', '_img_headers',

        'img_subheader_offsets', 'img_subheader_sizes',
        'img_segment_offsets', 'img_segment_sizes',

        'graphics_subheader_offsets', 'graphics_subheader_sizes',  # only 2.1
        'graphics_segment_offsets', 'graphics_segment_sizes',

        'symbol_subheader_offsets', 'symbol_subheader_sizes',  # only 2.0
        'symbol_segment_offsets', 'symbol_segment_sizes',

        'label_subheader_offsets', 'label_subheader_sizes',  # only 2.0
        'label_segment_offsets', 'label_segment_sizes',

        'text_subheader_offsets', 'text_subheader_sizes',
        'text_segment_offsets', 'text_segment_sizes',

        'des_subheader_offsets', 'des_subheader_sizes',
        'des_segment_offsets', 'des_segment_sizes',

        'res_subheader_offsets', 'res_subheader_sizes',  # only 2.1
        'res_segment_offsets', 'res_segment_sizes')

    def __init__(self, file_object: Union[str, BinaryIO]):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
            file name for a NITF file, or file like object opened in binary mode.
        """

        self._img_headers = None
        self._file_name = None
        self._file_object = None
        self._close_after = False

        if isinstance(file_object, str):
            if not os.path.isfile(file_object):
                raise SarpyIOError('Path {} is not a file'.format(file_object))
            self._file_name = file_object
            self._file_object = open(file_object, 'rb')
            self._close_after = True
        elif is_file_like(file_object):
            self._file_object = file_object
            if hasattr(file_object, 'name') and isinstance(file_object.name, str):
                self._file_name = file_object.name
            else:
                self._file_name = '<file like object>'
        else:
            raise TypeError('file_object is required to be a file like object, or string path to a file.')

        is_nitf_file, vers_string = is_nitf(self._file_object, return_version=True)
        if not is_nitf_file:
            raise SarpyIOError('Not a NITF file')
        self._nitf_version = vers_string

        if self._nitf_version not in ['02.10', '02.00']:
            raise SarpyIOError('Unsupported NITF version {} for file {}'.format(self._nitf_version, self._file_name))

        if self._nitf_version == '02.10':
            self._file_object.seek(354, os.SEEK_SET)  # offset to header length field
            header_length = int(self._file_object.read(6))
            # go back to the beginning of the file, and parse the whole header
            self._file_object.seek(0, os.SEEK_SET)
            header_string = self._file_object.read(header_length)
            self._nitf_header = NITFHeader.from_bytes(header_string, 0)
        elif self._nitf_version == '02.00':
            self._file_object.seek(280, os.SEEK_SET)  # offset to check if DEVT is defined
            # advance past security tags
            DWSG = self._file_object.read(6)
            if DWSG == b'999998':
                self._file_object.seek(40, os.SEEK_CUR)
            # seek to header length field
            self._file_object.seek(68, os.SEEK_CUR)
            header_length = int(self._file_object.read(6))
            self._file_object.seek(0, os.SEEK_SET)
            header_string = self._file_object.read(header_length)
            self._nitf_header = NITFHeader0.from_bytes(header_string, 0)
        else:
            raise ValueError(_unhandled_version_text.format(self._nitf_version))

        if self._nitf_header.get_bytes_length() != header_length:
            logger.critical(
                'Stated header length of file {} is {},\n\t'
                'while the interpreted header length is {}.\n\t'
                'This will likely be accompanied by serious parsing failures,\n\t'
                'and should be reported to the sarpy team for investigation.'.format(
                    self._file_name, header_length, self._nitf_header.get_bytes_length()))
        cur_loc = header_length
        # populate image segment offset information
        cur_loc, self.img_subheader_offsets, self.img_subheader_sizes, \
            self.img_segment_offsets, self.img_segment_sizes = self._element_offsets(
                cur_loc, self._nitf_header.ImageSegments)

        # populate graphics segment offset information - only version 2.1
        cur_loc, self.graphics_subheader_offsets, self.graphics_subheader_sizes, \
            self.graphics_segment_offsets, self.graphics_segment_sizes = self._element_offsets(
                cur_loc, getattr(self._nitf_header, 'GraphicsSegments', None))

        # populate symbol segment offset information - only version 2.0
        cur_loc, self.symbol_subheader_offsets, self.symbol_subheader_sizes, \
            self.symbol_segment_offsets, self.symbol_segment_sizes = self._element_offsets(
                cur_loc, getattr(self._nitf_header, 'SymbolsSegments', None))
        # populate label segment offset information - only version 2.0
        cur_loc, self.label_subheader_offsets, self.label_subheader_sizes, \
            self.label_segment_offsets, self.label_segment_sizes = self._element_offsets(
                cur_loc, getattr(self._nitf_header, 'LabelsSegments', None))

        # populate text segment offset information
        cur_loc, self.text_subheader_offsets, self.text_subheader_sizes, \
            self.text_segment_offsets, self.text_segment_sizes = self._element_offsets(
                cur_loc, self._nitf_header.TextSegments)
        # populate data extension offset information
        cur_loc, self.des_subheader_offsets, self.des_subheader_sizes, \
            self.des_segment_offsets, self.des_segment_sizes = self._element_offsets(
                cur_loc, self._nitf_header.DataExtensions)
        # populate data extension offset information - only version 2.1
        cur_loc, self.res_subheader_offsets, self.res_subheader_sizes, \
            self.res_segment_offsets, self.res_segment_sizes = self._element_offsets(
                cur_loc, getattr(self._nitf_header, 'ReservedExtensions', None))

    @staticmethod
    def _element_offsets(
            cur_loc: int,
            item_array_details: Union[_ItemArrayHeaders, None]
    ) -> Tuple[int, Optional[numpy.ndarray], Optional[numpy.ndarray], Optional[numpy.ndarray], Optional[numpy.ndarray]]:

        if item_array_details is None:
            return cur_loc, None, None, None, None
        subhead_sizes = item_array_details.subhead_sizes
        item_sizes = item_array_details.item_sizes
        if subhead_sizes.size == 0:
            return cur_loc, None, None, None, None

        subhead_offsets = numpy.full(subhead_sizes.shape, cur_loc, dtype=numpy.int64)
        subhead_offsets[1:] += numpy.cumsum(subhead_sizes[:-1]) + numpy.cumsum(item_sizes[:-1])
        item_offsets = subhead_offsets + subhead_sizes
        cur_loc = item_offsets[-1] + item_sizes[-1]
        return cur_loc, subhead_offsets, subhead_sizes, item_offsets, item_sizes

    @property
    def file_name(self) -> Optional[str]:
        """
        None|str: the file name, which may not be useful if the input was based
        on a file like object
        """

        return self._file_name

    @property
    def file_object(self) -> BinaryIO:
        """
        BinaryIO: The binary file object
        """

        return self._file_object

    @property
    def nitf_header(self) -> Union[NITFHeader, NITFHeader0]:
        """
        NITFHeader: the nitf header object
        """

        return self._nitf_header

    @property
    def img_headers(self) -> Union[None, List[ImageSegmentHeader], List[ImageSegmentHeader0]]:
        """
        The image segment headers.

        Returns
        -------
        None|List[ImageSegmentHeader]|List[ImageSegmentHeader0]
            Only `None` in the unlikely event that there are no image segments.
        """

        if self._img_headers is not None:
            return self._img_headers

        self._parse_img_headers()
        # noinspection PyTypeChecker
        return self._img_headers

    @property
    def nitf_version(self) -> str:
        """
        str: The NITF version number.
        """

        return self._nitf_version

    def _parse_img_headers(self) -> None:
        if self.img_segment_offsets is None or \
                self._img_headers is not None:
            return

        self._img_headers = [self.parse_image_subheader(i) for i in range(self.img_subheader_offsets.size)]

    def _fetch_item(
            self,
            name: str,
            index: int,
            offsets: numpy.ndarray,
            sizes: numpy.ndarray) -> bytes:
        if index >= offsets.size:
            raise IndexError(
                'There are only {0:d} {1:s}, invalid {1:s} position {2:d}'.format(
                    offsets.size, name, index))
        the_offset = offsets[index]
        the_size = sizes[index]
        self._file_object.seek(int(the_offset), os.SEEK_SET)
        the_item = self._file_object.read(int(the_size))
        return the_item

    def get_image_subheader_bytes(self, index: int) -> bytes:
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

    def parse_image_subheader(self, index: int) -> Union[ImageSegmentHeader, ImageSegmentHeader0]:
        """
        Parse the image segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ImageSegmentHeader|ImageSegmentHeader0
        """

        ih = self.get_image_subheader_bytes(index)
        if self.nitf_version == '02.10':
            out = ImageSegmentHeader.from_bytes(ih, 0)
        elif self.nitf_version == '02.00':
            out = ImageSegmentHeader0.from_bytes(ih, 0)
        else:
            raise ValueError(_unhandled_version_text.format(self.nitf_version))
        if out.is_masked:
            # read the mask subheader bytes
            the_offset = int(self.img_segment_offsets[index])
            self._file_object.seek(the_offset, os.SEEK_SET)
            the_size = struct.unpack('>I', self._file_object.read(4))[0]
            self._file_object.seek(the_offset, os.SEEK_SET)
            the_bytes = self._file_object.read(the_size)
            # interpret the mask subheader
            band_depth = len(out.Bands) if out.IMODE == 'S' else 1
            blocks = out.NBPR*out.NBPC
            out.mask_subheader = MaskSubheader.from_bytes(
                the_bytes, 0, band_depth=band_depth, blocks=blocks)
        return out

    def get_text_subheader_bytes(self, index: int) -> bytes:
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

    def get_text_bytes(self, index: int) -> bytes:
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

    def parse_text_subheader(self, index: int) -> Union[TextSegmentHeader, TextSegmentHeader0]:
        """
        Parse the text segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        TextSegmentHeader|TextSegmentHeader0
        """

        th = self.get_text_subheader_bytes(index)
        if self._nitf_version == '02.10':
            return TextSegmentHeader.from_bytes(th, 0)
        elif self._nitf_version == '02.00':
            return TextSegmentHeader0.from_bytes(th, 0)
        else:
            raise ValueError(_unhandled_version_text.format(self.nitf_version))

    def get_graphics_subheader_bytes(self, index: int) -> bytes:
        """
        Fetches the graphics segment subheader at the given index (only version 2.1).

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        if self._nitf_version == '02.10':
            return self._fetch_item('graphics subheader',
                                    index,
                                    self.graphics_subheader_offsets,
                                    self._nitf_header.GraphicsSegments.subhead_sizes)
        else:
            raise ValueError('Only NITF version 02.10 has graphics segments')

    def get_graphics_bytes(self, index: int) -> bytes:
        """
        Fetches the graphics extension segment bytes at the given index (only version 2.1).

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        if self._nitf_version == '02.10':
            return self._fetch_item('graphics segment',
                                    index,
                                    self.graphics_segment_offsets,
                                    self._nitf_header.GraphicsSegments.item_sizes)
        else:
            raise ValueError('Only NITF version 02.10 has graphics segments')

    def parse_graphics_subheader(self, index: int) -> GraphicsSegmentHeader:
        """
        Parse the graphics segment subheader at the given index (only version 2.1).

        Parameters
        ----------
        index : int

        Returns
        -------
        GraphicsSegmentHeader
        """

        if self._nitf_version == '02.10':
            gh = self.get_graphics_subheader_bytes(index)
            return GraphicsSegmentHeader.from_bytes(gh, 0)
        else:
            raise ValueError('Only NITF version 02.10 has graphics segments')

    def get_symbol_subheader_bytes(self, index: int) -> bytes:
        """
        Fetches the symbol segment subheader at the given index (only version 2.0).

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        if self.nitf_version == '02.00':
            return self._fetch_item('symbol subheader',
                                    index,
                                    self.symbol_subheader_offsets,
                                    self._nitf_header.GraphicsSegments.subhead_sizes)
        else:
            raise ValueError('Only NITF 02.00 has symbol elements.')

    def get_symbol_bytes(self, index: int) -> bytes:
        """
        Fetches the symbol extension segment bytes at the given index (only version 2.0).

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        if self.nitf_version == '02.00':
            return self._fetch_item('symbol segment',
                                    index,
                                    self.symbol_segment_offsets,
                                    self._nitf_header.GraphicsSegments.item_sizes)
        else:
            raise ValueError('Only NITF 02.00 has symbol elements.')

    def parse_symbol_subheader(self, index: int) -> SymbolSegmentHeader:
        """
        Parse the symbol segment subheader at the given index (only version 2.0).

        Parameters
        ----------
        index : int

        Returns
        -------
        SymbolSegmentHeader
        """

        if self.nitf_version == '02.00':
            gh = self.get_symbol_subheader_bytes(index)
            return SymbolSegmentHeader.from_bytes(gh, 0)
        else:
            raise ValueError('Only NITF 02.00 has symbol elements.')

    def get_label_subheader_bytes(self, index: int) -> bytes:
        """
        Fetches the label segment subheader at the given index (only version 2.0).

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        if self.nitf_version == '02.00':
            return self._fetch_item('label subheader',
                                    index,
                                    self.label_subheader_offsets,
                                    self._nitf_header.GraphicsSegments.subhead_sizes)
        else:
            raise ValueError('Only NITF 02.00 has label elements.')

    def get_label_bytes(self, index: int) -> bytes:
        """
        Fetches the label extension segment bytes at the given index (only version 2.0).

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        if self.nitf_version == '02.00':
            return self._fetch_item('label segment',
                                    index,
                                    self.label_segment_offsets,
                                    self._nitf_header.GraphicsSegments.item_sizes)
        else:
            raise ValueError('Only NITF 02.00 has symbol elements.')

    def parse_label_subheader(self, index: int) -> LabelSegmentHeader:
        """
        Parse the label segment subheader at the given index (only version 2.0).

        Parameters
        ----------
        index : int

        Returns
        -------
        LabelSegmentHeader
        """

        if self.nitf_version == '02.00':
            gh = self.get_label_subheader_bytes(index)
            return LabelSegmentHeader.from_bytes(gh, 0)
        else:
            raise ValueError('Only NITF 02.00 has label elements.')

    def get_des_subheader_bytes(self, index: int) -> bytes:
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

    def get_des_bytes(self, index: int) -> bytes:
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

    def parse_des_subheader(self, index: int) -> Union[DataExtensionHeader, DataExtensionHeader0]:
        """
        Parse the data extension segment subheader at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        DataExtensionHeader|DataExtensionHeader0
        """

        dh = self.get_des_subheader_bytes(index)
        if self.nitf_version == '02.10':
            return DataExtensionHeader.from_bytes(dh, 0)
        elif self.nitf_version == '02.00':
            return DataExtensionHeader0.from_bytes(dh, 0)
        else:
            raise ValueError(_unhandled_version_text.format(self.nitf_version))

    def get_res_subheader_bytes(self, index: int) -> bytes:
        """
        Fetches the reserved extension segment subheader bytes at the given index (only version 2.1).

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

    def get_res_bytes(self, index: int) -> bytes:
        """
        Fetches the reserved extension segment bytes at the given index (only version 2.1).

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

    def parse_res_subheader(self, index: int) -> Union[ReservedExtensionHeader, ReservedExtensionHeader0]:
        """
        Parse the reserved extension subheader at the given index (only version 2.1).

        Parameters
        ----------
        index : int

        Returns
        -------
        ReservedExtensionHeader|ReservedExtensionHeader0
        """

        rh = self.get_res_subheader_bytes(index)
        if self.nitf_version == '02.10':
            return ReservedExtensionHeader.from_bytes(rh, 0)
        elif self.nitf_version == '02.00':
            return ReservedExtensionHeader0.from_bytes(rh, 0)
        else:
            raise ValueError('Unhandled version {}.'.format(self.nitf_version))

    def get_headers_json(self) -> dict:
        """
        Get a json (i.e. dict) representation of the NITF header elements.

        Returns
        -------
        dict
        """

        out = OrderedDict([('header', self._nitf_header.to_json()), ])
        if self.img_subheader_offsets is not None:
            out['Image_Subheaders'] = [
                self.parse_image_subheader(i).to_json() for i in range(self.img_subheader_offsets.size)]
        if self.graphics_subheader_offsets is not None:
            out['Graphics_Subheaders'] = [
                self.parse_graphics_subheader(i).to_json() for i in range(self.graphics_subheader_offsets.size)]
        if self.symbol_subheader_offsets is not None:
            out['Symbol_Subheaders'] = [
                self.parse_symbol_subheader(i).to_json() for i in range(self.symbol_subheader_offsets.size)]
        if self.label_subheader_offsets is not None:
            out['Label_Subheaders'] = [
                self.parse_label_subheader(i).to_json() for i in range(self.label_subheader_offsets.size)]
        if self.text_subheader_offsets is not None:
            out['Text_Subheaders'] = [
                self.parse_text_subheader(i).to_json() for i in range(self.text_subheader_offsets.size)]
        if self.des_subheader_offsets is not None:
            out['DES_Subheaders'] = [
                self.parse_des_subheader(i).to_json() for i in range(self.des_subheader_offsets.size)]
        if self.res_subheader_offsets is not None:
            out['RES_Subheaders'] = [
                self.parse_res_subheader(i).to_json() for i in range(self.res_subheader_offsets.size)]
        return out

    def __del__(self):
        if self._close_after:
            self._close_after = False
            # noinspection PyBroadException
            try:
                self._file_object.close()
            except Exception:
                pass


def find_jpeg_delimiters(the_bytes: bytes) -> List[Tuple[int, int]]:
    """
    Finds regular jpeg delimiters from the image segment bytes.

    Parameters
    ----------
    the_bytes : bytes

    Returns
    -------
    List[Tuple[int, int]]

    Raises
    ------
    ValueError
        If the bytes doesn't start with the begin jpeg delimiter and end with the
        end jpeg delimiter.
    """

    start_pattern = b'\xff\xd8'
    end_pattern = b'\xff\xd9'  # these should never be used for anything else

    out = []
    next_location = 0
    while next_location < len(the_bytes):
        if the_bytes[next_location:next_location+2] != start_pattern:
            raise ValueError('The jpeg block {} does not start with the jpeg start delimiter'.format(len(out)))
        end_block = the_bytes.find(end_pattern, next_location)
        if end_block == -1:
            raise ValueError('The new jpeg block {} does not contain the jpeg end delimiter'.format(len(out)))
        next_location = end_block +2
        out.append((0, next_location))
    return out


def _get_shape(rows: int, cols: int, bands: int, band_dimension=2) -> Tuple[int, ...]:
    """
    Helper function for turning rows/cols/bands into a shape tuple.

    Parameters
    ----------
    rows: int
    cols: int
    bands: int
    band_dimension : int
        One of `{0, 1, 2}`.

    Returns
    -------
    shape_tuple : Tuple[int, ...]
        The shape tuple with band omitted if `bands=1`
    """

    if bands == 1:
        return rows, cols
    elif band_dimension == 0:
        return bands, rows, cols
    elif band_dimension == 1:
        return rows, bands, cols
    else:
        return rows, cols, bands


def _get_subscript_def(
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        raw_bands: int,
        raw_band_dimension: int) -> Tuple[slice, ...]:
    if raw_bands == 1:
        return slice(row_start, row_end, 1), slice(col_start, col_end, 1)
    elif raw_band_dimension == 0:
        return slice(0, raw_bands, 1), slice(row_start, row_end, 1), slice(col_start, col_end, 1)
    elif raw_band_dimension == 1:
        return slice(row_start, row_end, 1), slice(0, raw_bands, 1), slice(col_start, col_end, 1)
    elif raw_band_dimension == 2:
        return slice(row_start, row_end, 1), slice(col_start, col_end, 1), slice(0, raw_bands, 1)
    else:
        raise ValueError('Unhandled raw_band_dimension {}'.format(raw_band_dimension))


#####
# A general nitf reader

class NITFReader(BaseReader):
    """
    A reader implementation based around array-type image data fetching for
    NITF 2.0 or 2.1 files.

    **Significantly refactored in version 1.3.0** General NITF support is improved
    form previous version, but there remain unsupported edge cases.
    """

    _maximum_number_of_images = None
    unsupported_compressions = ('C1', 'C4', 'C6', 'C7', 'M1', 'M4', 'M6', 'M7')

    __slots__ = (
        '_nitf_details', '_unsupported_segments', '_image_segment_collections',
        '_reverse_axes', '_transpose_axes', '_make_local_copy')

    def __init__(
            self,
            nitf_details: Union[str, BinaryIO, NITFDetails],
            reader_type="OTHER",
            reverse_axes: Union[None, int, Sequence[int]] = None,
            transpose_axes: Union[None, Tuple[int, ...]] = None):
        """

        Parameters
        ----------
        nitf_details : str|BinaryIO|NITFDetails
            The NITFDetails object or path to a nitf file.
        reader_type : str
            What type of reader is this? e.g. "SICD", "SIDD", "OTHER"
        reverse_axes : None|Sequence[int]
            Any entries should be restricted to `{0, 1}`. The presence of
            `0` means to reverse the rows (in the raw sense), and the presence
            of `1` means to reverse the columns (in the raw sense).
        transpose_axes : None|Tuple[int, ...]
            If presented this should be only `(1, 0)`.
        """

        try:
            _ = self._delete_temp_files
            # something has already defined this, so it's already ready
        except AttributeError:
            self._delete_temp_files = []

        try:
            _ = self._nitf_details
            # something has already defined this, so it's already ready
        except AttributeError:
            if isinstance(nitf_details, str) or is_file_like(nitf_details):
                nitf_details = NITFDetails(nitf_details)
            if not isinstance(nitf_details, NITFDetails):
                raise TypeError('The input argument for NITFReader must be a NITFDetails object.')
            self._nitf_details = nitf_details
        if self._nitf_details.img_headers is None:
            raise SarpyIOError(
                'The input NITF has no image segments,\n\t'
                'so there is no image data to be read.')

        if reverse_axes is not None:
            if isinstance(reverse_axes, int):
                reverse_axes = (reverse_axes, )

            for entry in reverse_axes:
                if not 0 <= entry < 2:
                    raise ValueError('reverse_axes values must be restricted to `{0, 1}`.')
        self._reverse_axes = reverse_axes

        if transpose_axes is not None:
            if transpose_axes != (1, 0):
                raise ValueError('transpose_axes, if not None, must be (1, 0)')
        self._transpose_axes = transpose_axes

        # find image segments which we can not support, for whatever reason
        self._unsupported_segments = self.check_for_compliance()
        if len(self._unsupported_segments) == len(self.nitf_details.img_headers):
            raise SarpyIOError('There are no supported image segments in NITF file {}'.format(self.file_name))

        # our supported images are assembled into collections for joint presentation
        self._image_segment_collections = self.find_image_segment_collections()
        if self._maximum_number_of_images is not None and \
                len(self._image_segment_collections) > self._maximum_number_of_images:
            raise SarpyIOError(
                'Images in this NITF are grouped together in {} collections,\n\t'
                'which exceeds the maximum number of collections permitted ({})\n\t'
                'by class {} implementation'.format(
                    len(self._image_segment_collections), self._maximum_number_of_images, self.__class__))
        self.verify_collection_compliance()

        data_segments = self.get_data_segments()
        BaseReader.__init__(self, data_segments, reader_type=reader_type, close_segments=True)

    @property
    def nitf_details(self) -> NITFDetails:
        """
        NITFDetails: The NITF details object.
        """

        return self._nitf_details

    def get_image_header(self, index: int) -> Union[ImageSegmentHeader, ImageSegmentHeader0]:
        """
        Gets the image subheader at the specified index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ImageSegmentHeader|ImageSegmentHeader0
        """

        return self.nitf_details.img_headers[index]

    @property
    def file_name(self) -> Optional[str]:
        return self._nitf_details.file_name

    @property
    def file_object(self) -> BinaryIO:
        """
        BinaryIO: the binary file like object from which we are reading
        """

        return self._nitf_details.file_object

    @property
    def unsupported_segments(self) -> Tuple[int, ...]:
        """
        Tuple[int, ...]: The image segments deemed not supported.
        """

        return self._unsupported_segments

    @property
    def image_segment_collections(self) -> Tuple[Tuple[int, ...]]:
        """
        The definition for how image segments are grouped together to form the
        output image collection.

        Each entry corresponds to a single output image, and the entry defines
        the image segment indices which are combined to make up the output image.

        Returns
        -------
        Tuple[Tuple[int, ...]]
        """

        return self._image_segment_collections

    def can_use_memmap(self) -> bool:
        """
        Can a memmap be used? This is only supported and/or sensible in the case
        that the file-like object represents a local file.

        Returns
        -------
        bool
        """

        return is_real_file(self.nitf_details.file_object)

    def _read_file_data(self, start_bytes: int, byte_length: int) -> bytes:
        initial_loc = self.file_object.tell()
        self.file_object.seek(start_bytes, os.SEEK_SET)
        the_bytes = self.file_object.read(byte_length)
        self.file_object.seek(initial_loc)
        return the_bytes

    def _check_image_segment_for_compliance(
            self,
            index: int,
            img_header: Union[ImageSegmentHeader, ImageSegmentHeader0]) -> bool:
        """
        Checks whether the image segment can be (or should be) opened.

        Parameters
        ----------
        index : int
            The image segment index (for logging)
        img_header : ImageSegmentHeader|ImageSegmentHeader0
            The image segment header

        Returns
        -------
        bool
        """

        out = True
        if img_header.NBPP not in (8, 16, 32, 64):
            # TODO: is this really true? What about the compression situation?
            # numpy basically only supports traditional typing
            logger.error(
                'Image segment at index {} has bits per pixel per band {},\n\t'
                'only 8, 16, 32, 64 are supported.'.format(index, img_header.NBPP))
            out = False

        if img_header.is_compressed:
            if PIL_Image is None:
                logger.error(
                    'Image segment at index {} has IC value {},\n\t'
                    'and PIL cannot be imported.\n\t'
                    'Currently, compressed image segments require PIL.'.format(
                        index, img_header.IC))
                out = False

        if img_header.IC in self.unsupported_compressions:
            logger.error(
                'Image segment at index {} has IC value `{}`,\n\t'
                'which is not supported.'.format(index, img_header.IMODE))
            out = False
        return out

    def check_for_compliance(self) -> Tuple[int, ...]:
        """
        Gets indices of image segments that cannot (or should not) be opened.

        Returns
        -------
        Tuple[int, ...]
        """

        out = []
        for index, img_header in enumerate(self.nitf_details.img_headers):
            if not self._check_image_segment_for_compliance(index, img_header):
                out.append(index)
        return tuple(out)

    def _construct_block_bounds(self, image_segment_index: int) -> List[Tuple[int, int, int, int]]:
        """
        Construct the bounds for the blocking definition in row/column space for
        the image segment.

        Note that this includes potential pad pixels, since NITF requires that
        each block is the same size.

        Parameters
        ----------
        image_segment_index : int

        Returns
        -------
        List[Tuple[int, int, int, int]]
            This is a list of the form `(row start, row end, column start, column end)`.
        """

        image_header = self.get_image_header(image_segment_index)
        if image_header.NPPBH == 0:
            column_block_size = image_header.NCOLS
        else:
            column_block_size = image_header.NPPBH

        # validate that this makes sense...
        hblocks = column_block_size*image_header.NBPR
        if not (image_header.NCOLS <= hblocks < image_header.NCOLS + column_block_size):
            raise ValueError(
                'Got NCOLS {}, NPPBH {}, and NBPR {}'.format(
                    image_header.NCOLS, image_header.NPPBH, image_header.NBPR))

        if image_header.NPPBV == 0:
            row_block_size = image_header.NROWS
        else:
            row_block_size = image_header.NPPBV

        # validate that this makes sense
        vblocks = row_block_size*image_header.NBPC
        if not (image_header.NROWS <= vblocks < image_header.NROWS + row_block_size):
            raise ValueError(
                'Got NROWS {}, NPPBV {}, and NBPC {}'.format(
                    image_header.NROWS, image_header.NPPBV, image_header.NBPC))

        bounds = []
        block_row_start = 0
        for row_block in range(image_header.NBPC):
            block_row_end = block_row_start + row_block_size
            block_col_start = 0
            for column_block in range(image_header.NBPR):
                block_col_end = block_col_start + column_block_size
                bounds.append((block_row_start, block_row_end, block_col_start, block_col_end))
                block_col_start = block_col_end
            block_row_start = block_row_end
        return bounds

    def _get_mask_details(
            self,
            image_segment_index: int) -> Tuple[Optional[numpy.ndarray], int, int]:
        """
        Gets the mask offset details.

        Parameters
        ----------
        image_segment_index : int

        Returns
        -------
        mask_offsets : Optional[numpy.ndarray]
            The mask byte offset from the end of the mask subheader definition.
            If `IMODE = S`, then this is two dimensional, otherwise it is one
            dimensional
        exclude_value : int
            The offset value for excluded block, should always be `0xFFFFFFFF`.
        additional_offset : int
            The additional offset from the beginning of the image segment data,
            necessary to account for the presence of mask subheader.
        """

        image_header = self.get_image_header(image_segment_index)
        exclude_value = 0xFFFFFFFF
        if image_header.is_masked:
            offset_shift = image_header.mask_subheader.IMDATOFF
            if image_header.mask_subheader.BMR is not None:
                mask_offsets = image_header.mask_subheader.BMR
            elif image_header.mask_subheader.TMR is not None:
                mask_offsets = image_header.mask_subheader.TMR
            else:
                raise ValueError(
                    'Image segment at index {} is marked at masked,\n\t'
                    'but neither BMR nor TMR is defined'.format(image_segment_index))
            if mask_offsets.ndim != 2:
                raise ValueError('Expected two dimensional raw mask offsets array')
            if mask_offsets.shape[0] == 1:
                mask_offsets = numpy.reshape(mask_offsets, (-1, ))
            return mask_offsets, exclude_value, offset_shift
        else:
            return None, exclude_value, 0

    def _get_dtypes(
            self, image_segment_index: int) -> Tuple[numpy.dtype, numpy.dtype, int, Optional[str], Optional[numpy.ndarray]]:
        """
        Gets the information necessary for constructing the format function applicable
        to the given image segment.

        Parameters
        ----------
        image_segment_index : int

        Returns
        -------
        raw_dtype: numpy.ndtype
            The native data type
        formatted_dtype : numpy.dtype
            The formatted data type. Will be `complex64` if `complex_order` is
            populated, the data type of `lut` if it is populated, or same as
            `raw_dtype`.
        formatted_bands : int
            How many bands in the formatted output. Similarly depends on the
            value of `complex_order` and `lut`.
        complex_order : None|str
            If populated, one of `('IQ', 'QI', 'MP', 'PM')` indicating the
            order of complex bands. This will only be populated if consistent.
        lut : None|numpy.ndarray
            If populated, the lookup table presented in the data.
        """
        def get_raw_dtype() -> numpy.dtype:
            if pvtype == 'INT':
                return numpy.dtype('>u{}'.format(bpp))
            elif pvtype == 'SI':
                return numpy.dtype('>i{}'.format(bpp))
            elif pvtype == 'R':
                return numpy.dtype('>f{}'.format(bpp))
            elif pvtype == 'C':
                if bpp not in [4, 8, 16]:
                    raise ValueError(
                        'Got PVTYPE = C and NBPP = {} (not 32, 64 or 128), which is unsupported.'.format(nbpp))
                return numpy.dtype('>c{}'.format(bpp))

        def get_complex_order() -> Optional[str]:
            bands = image_header.Bands
            if (len(bands) % 2) != 0:
                return None
            order = bands[0].ISUBCAT + bands[1].ISUBCAT
            if order not in ['IQ', 'QI', 'MP', 'PM']:
                return None
            for i in range(2, len(bands), 2):
                if order != bands[i].ISUBCAT + bands[i+1].ISUBCAT:
                    return None
            if order in ['IQ', 'QI']:
                if pvtype not in ['SI', 'R']:
                    raise ValueError(
                        'Image segment at index {} appears to be complex of order `{}`, \n\t'
                        'but PVTYPE is `{}`'.format(image_segment_index, order, pvtype))
            if order in ['MP', 'PM']:
                if pvtype not in ['INT', 'R']:
                    raise ValueError(
                        'Image segment at index {} appears to be complex of order `{}`, \n\t'
                        'but PVTYPE is `{}`'.format(image_segment_index, order, pvtype))
            return order

        def get_lut_info() -> Optional[numpy.ndarray]:
            bands = image_header.Bands
            if len(bands) > 1:
                for band in bands:
                    if band.LUT is not None:
                        raise ValueError('There are multiple bands with LUT.')

            # TODO: this isn't really right - handle most significant/least significant nonsense
            lut = bands[0].LUT
            if lut is None:
                return None
            if lut.ndim == 1:
                return lut
            elif lut.ndim == 2:
                return numpy.transpose(lut)
            else:
                raise ValueError('Got lut of shape `{}`'.format(lut.shape))

        image_header = self.get_image_header(image_segment_index)
        nbpp = image_header.NBPP  # previously verified to be one of 8, 16, 32, 64
        bpp = int(nbpp/8)  # bytes per pixel per band
        pvtype = image_header.PVTYPE

        raw_dtype = get_raw_dtype()
        formatted_dtype = raw_dtype
        band_count = len(image_header.Bands)
        formatted_bands = band_count

        # is it one of the assembled complex types?
        complex_order = get_complex_order()
        if complex_order:
            formatted_dtype = numpy.dtype('complex64')
            formatted_bands = int(band_count/2)

        # is there an LUT?
        lut = get_lut_info()
        if lut:
            formatted_dtype = lut.dtype
            formatted_bands = 1 if lut.ndim == 1 else lut.shape[1]

        return raw_dtype, formatted_dtype, formatted_bands, complex_order, lut

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def get_format_function(
            self,
            raw_dtype: numpy.dtype,
            complex_order: Optional[str],
            lut: Optional[numpy.ndarray],
            band_dimension: int,
            image_segment_index: Optional[int] = None,
            **kwargs) -> Optional[FormatFunction]:
        """
        Gets the format function for use in a data segment.

        Parameters
        ----------
        raw_dtype : numpy.dtype
        complex_order : None|str
        lut : None|numpy.ndarray
        band_dimension : int
        image_segment_index : None|int
            For possible use in extension
        kwargs
            Optional keyword argument

        Returns
        -------
        None|FormatFunction
        """

        if complex_order is not None:
            return ComplexFormatFunction(raw_dtype, complex_order, band_dimension=band_dimension)
        elif lut is not None:
            return SingleLUTFormatFunction(lut)
        else:
            return None

    def _verify_image_segment_compatibility(self, index0: int, index1: int) -> bool:
        """
        Verify that the image segments are compatible from the data formatting
        perspective.

        Parameters
        ----------
        index0 : int
        index1 : int

        Returns
        -------
        bool
        """

        img0 = self.get_image_header(index0)
        img1 = self.get_image_header(index1)

        if len(img0.Bands) != len(img1.Bands):
            return False
        if img0.PVTYPE != img1.PVTYPE:
            return False
        if img0.IREP != img1.IREP:
            return False
        if img0.ICAT != img1.ICAT:
            return False
        if img0.NBPP != img1.NBPP:
            return False

        raw_dtype0, _, form_band0, comp_order0, lut0 = self._get_dtypes(index0)
        raw_dtype1, _, form_band1, comp_order1, lut1 = self._get_dtypes(index1)
        if raw_dtype0 != raw_dtype1:
            return False
        if form_band0 != form_band1:
            return False
        if (comp_order0 is None and comp_order1 is not None) or \
                (comp_order0 is not None or comp_order1 is None) or \
                (comp_order0 != comp_order1):
            return False
        if (lut0 is None and lut1 is not None) or \
                (lut0 is not None or lut1 is None) or \
                (numpy.any(lut0 != lut1)):
            return False

    def _correctly_order_image_segment_collection(self, indices: Sequence[int]) -> Tuple[int, ...]:
        """
        Determines the proper order, based on IALVL, for a collection of entries
        which will be assembled into a composite image.

        Parameters
        ----------
        indices: Sequence[int]

        Returns
        -------
        Tuple[int, ...]

        Raises
        ------
        ValueError
            If incompatible IALVL values collection
        """

        if len(indices) < 1:
            raise ValueError('Got empty grouping')
        if len(indices) == 1:
            return (indices[0], )

        img_headers = [self.nitf_details.img_headers[entry] for entry in indices]
        collection = [(entry.IALVL, orig_index) for entry, orig_index in zip(img_headers, indices)]
        collection = sorted(collection, key=lambda x: x[0])   # (stable) order by IALVL

        if all(entry[0] == 0 for entry in collection):
            # all IALVL is 0, and order doesn't matter
            return tuple(indices)
        if all(entry0[0]+1 == entry1[0] for entry0, entry1 in zip(collection[:-1], collection[1:])):
            # ordered, uninterupted sequence of IALVL values
            return tuple(entry[1] for entry in collection)

        raise ValueError(
            'Collection of (IALVL, image segment index) has\n\t'
            'neither all IALVL == 0, or an uninterrupted sequence of IALVL values.\n\t'
            'See {}'.format(collection))

    def find_image_segment_collections(self) -> Tuple[Tuple[int, ...]]:
        """
        Determines the image segments, other than those specifically excluded in
        `unsupported_segments` property value. It is implicitly assumed that the
        elements of a given entry are ordered so that IALVL values are sensible.

        Note that in the default implementation, every image segment is simply
        considered separately.

        Returns
        -------
        Tuple[Tuple[int]]
        """

        out = []
        for index in range(len(self.nitf_details.img_headers)):
            if index not in self.unsupported_segments:
                out.append((index, ))
        return tuple(out)

    def verify_collection_compliance(self) -> None:
        """
        Verify that image segments collections are compatible.

        Raises
        -------
        ValueError
        """

        all_compatible = True
        for collection_index, the_indices in enumerate(self.image_segment_collections):

            if len(the_indices) == 1:
                continue

            compatible = True
            for the_index in the_indices[1:]:
                t_compat = self._verify_image_segment_compatibility(the_indices[0], the_index)
                if not t_compat:
                    logger.error(
                        'Collection index {} has image segments at indices {} and {} incompatible'.format(
                            collection_index, the_indices[0], the_index))
                compatible &= t_compat
            all_compatible &= compatible
        if not all_compatible:
            raise ValueError('Image segment collection incompatibilities')

    def _get_collection_element_coordinate_limits(self, collection_index: int) -> numpy.ndarray:
        """
        For the given image segment collection, as defined in the
        `image_segment_collections` property value, get the relative coordinate
        scheme of the form `[[start_row, end_row, start_column, end_column]]`.

        This relies on inspection of `IALVL` and `ILOC` values for this
        collection of image segments. This assumes that the entries in the
        relevant element of image_segment_collection are in the correct order.

        Parameters
        ----------
        collection_index : int
            The index into the `image_segment_collection` list.

        Returns
        -------
        block_definition: numpy.ndarray
            of the form `[[start_row, end_row, start_column, end_column]]`.
        """

        the_indices = self.image_segment_collections[collection_index]
        the_indices = self._correctly_order_image_segment_collection(the_indices)

        block_definition = numpy.empty((len(the_indices), 4), dtype='int64')
        for i, image_ind in enumerate(the_indices):
            img_header = self.nitf_details.img_headers[image_ind]
            rows = img_header.NROWS
            cols = img_header.NCOLS
            iloc = img_header.ILOC
            if img_header.IALVL == 0 or i == 0:
                previous_indices = numpy.zeros((4, ), dtype='int64')
            else:
                previous_indices = block_definition[i-1, :]
            rel_row_start, rel_col_start = int(iloc[:5]), int(iloc[5:])
            abs_row_start = rel_row_start + previous_indices[1]
            abs_col_start = rel_col_start + previous_indices[3]
            block_definition[i, :] = (abs_row_start, abs_row_start + rows, abs_col_start, abs_col_start + cols)

        # now, renormalize the coordinate system to be sensible
        min_row = numpy.min(block_definition[:, 0])
        min_col = numpy.min(block_definition[:, 2])
        block_definition[:, 0:2:1] -= min_row
        block_definition[:, 2:4:1] -= min_col
        return block_definition

    def _get_transpose(self, formatted_bands: int) -> Optional[Tuple[int, ...]]:
        if self._transpose_axes is None:
            return None
        elif formatted_bands > 1:
            return self._transpose_axes + (2,)
        else:
            return self._transpose_axes

    def _handle_jpeg2k_no_mask(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        # NOTE: it appears that the PIL to numpy array conversion will rearrange
        # bands to be in the final dimension, regardless of storage particulars?

        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'B' or image_header.IC != 'C8':
            raise ValueError(
                'Requires IMODE = `B` and IC = `C8`, got `{}` and `{}` at image segment index {}'.format(
                    image_header.IMODE, image_header.IC, image_segment_index))
        if PIL_Image is None:
            raise ValueError('Image segment {} is compressed, which requires PIL'.format(image_segment_index))

        # get bytes offset to this image segment (relative to start of file)
        offset = self.nitf_details.img_segment_offsets[image_segment_index]
        image_segment_size = self.nitf_details.img_segment_sizes[image_segment_index]
        raw_bands = len(image_header.Bands)
        raw_dtype, formatted_dtype, formatted_bands, complex_order, lut = self._get_dtypes(image_segment_index)
        raw_shape = _get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2)

        # the block details will be handled by the jpeg2000 compression scheme,
        # just read everything and decompress
        the_bytes = self._read_file_data(offset, image_segment_size)
        # create a memmap, and extract all of our jpeg data into it as appropriate
        fi, path_name = mkstemp(suffix='.sarpy_cache', text=False)
        self._delete_temp_files.append(path_name)
        mem_map = numpy.memmap(
            path_name, dtype=raw_dtype, mode='w+', offset=0,
            shape=_get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2))
        # noinspection PyUnresolvedReferences
        img = PIL_Image.open(BytesIO(the_bytes))
        data = numpy.asarray(img)
        mem_map[:] = data[:image_header.NROWS, :image_header.NCOLS]
        mem_map.flush()  # write all the data to the file
        del mem_map  # clean up the memmap
        os.close(fi)

        if apply_format:
            format_function = self.get_format_function(
                raw_dtype, complex_order, lut, 2,
                image_segment_index=image_segment_index)
            reverse_axes = self._reverse_axes
            if self._transpose_axes is None:
                formatted_shape = _get_shape(image_header.NROWS, image_header.NCOLS, formatted_bands, band_dimension=2)
            else:
                formatted_shape = _get_shape(image_header.NCOLS, image_header.NROWS, formatted_bands, band_dimension=2)
            transpose_axes = self._get_transpose(formatted_bands)
        else:
            format_function = None
            reverse_axes = None
            transpose_axes = None
            formatted_dtype = raw_dtype
            formatted_shape = raw_shape

        return NumpyMemmapSegment(
            path_name, 0, raw_dtype, raw_shape, formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode='r', close_file=True)

    def _handle_jpeg2k_with_mask(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        # NOTE: it appears that the PIL to numpy array conversion will rearrange
        # bands to be in the final dimension, regardless of storage particulars?

        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'B' or image_header.IC != 'M8':
            raise ValueError(
                'Requires IMODE = `B` and IC = `M8`, got `{}` and `{}` at image segment index {}'.format(
                    image_header.IMODE, image_header.IC, image_segment_index))
        if PIL_Image is None:
            raise ValueError('Image segment {} is compressed, which requires PIL'.format(image_segment_index))

        # get mask definition details
        mask_offsets, exclude_value, additional_offset = self._get_mask_details(image_segment_index)

        # get bytes offset to this image segment (relative to start of file)
        offset = self.nitf_details.img_segment_offsets[image_segment_index]
        image_segment_size = self.nitf_details.img_segment_sizes[image_segment_index]
        raw_bands = len(image_header.Bands)
        raw_dtype, formatted_dtype, formatted_bands, complex_order, lut = self._get_dtypes(image_segment_index)
        # Establish block pixel bounds
        block_bounds = self._construct_block_bounds(image_segment_index)
        assert isinstance(block_bounds, list)

        if not (isinstance(mask_offsets, numpy.ndarray) and mask_offsets.ndim == 1):
            raise ValueError('Got unexpected mask offsets `{}`'.format(mask_offsets))

        if len(block_bounds) != len(mask_offsets):
            raise ValueError('Got mismatch between block definition and mask offsets definition')

        # jpeg2000 compression, read everything excluding the mask
        the_bytes = self._read_file_data(offset+additional_offset, image_segment_size-additional_offset)

        raw_shape = _get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2)

        # create a memmap, and extract all of our jpeg data into it as appropriate
        fi, path_name = mkstemp(suffix='.sarpy_cache', text=False)
        self._delete_temp_files.append(path_name)
        mem_map = numpy.memmap(
            path_name, dtype=raw_dtype, mode='w+', offset=0,
            shape=_get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2))
        next_jpeg_block = 0
        for mask_index, (mask_offset, block_bound) in enumerate(zip(mask_offsets, block_bounds)):
            if mask_offset == exclude_value:
                continue  # just skip it, because it is masked out

            start_bytes = mask_offset  # TODO: verify that we don't need to account for mask definition length
            end_bytes = len(the_bytes) if mask_index == len(mask_offsets)-1 else mask_offsets[mask_index + 1]
            # noinspection PyUnresolvedReferences
            img = PIL_Image.open(BytesIO(the_bytes[start_bytes:end_bytes]))
            # handle block padding situation
            row_start, row_end = block_bound[0], min(block_bound[1], image_header.NROWS)
            col_start, col_end = block_bound[2], min(block_bound[3], image_header.NCOLS)
            mem_map[row_start: row_end, col_start:col_end] = numpy.asarray(img)[0:row_end - row_start,
                                                             0:col_end - col_start]
            next_jpeg_block += 1

        mem_map.flush()  # write all the data to the file
        del mem_map  # clean up the memmap
        os.close(fi)

        if apply_format:
            format_function = self.get_format_function(
                raw_dtype, complex_order, lut, 2,
                image_segment_index=image_segment_index)
            reverse_axes = self._reverse_axes
            if self._transpose_axes is None:
                formatted_shape = _get_shape(image_header.NROWS, image_header.NCOLS, formatted_bands, band_dimension=2)
            else:
                formatted_shape = _get_shape(image_header.NCOLS, image_header.NROWS, formatted_bands, band_dimension=2)
            transpose_axes = self._get_transpose(formatted_bands)
        else:
            format_function = None
            reverse_axes = None
            transpose_axes = None
            formatted_dtype = raw_dtype
            formatted_shape = raw_shape

        return NumpyMemmapSegment(
            path_name, 0, raw_dtype, raw_shape,
            formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode='r', close_file=True)

    def _handle_jpeg(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        # NOTE: it appears that the PIL to numpy array conversion will rearrange
        # bands to be in the final dimension, regardless of storage particulars?

        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE not in ['B', 'P'] or image_header.IC not in ['I1', 'C3', 'C5', 'M3', 'M5']:
            raise ValueError(
                'Requires IMODE in `(B, P)` and IC in `(I1, C3, C5, M3, M5)`,\n\t'
                'got `{}` and `{}` at image segment index {}'.format(
                    image_header.IMODE, image_header.IC, image_segment_index))
        if PIL_Image is None:
            raise ValueError('Image segment {} is compressed, which requires PIL'.format(image_segment_index))

        # get bytes offset to this image segment (relative to start of file)
        offset = self.nitf_details.img_segment_offsets[image_segment_index]
        image_segment_size = self.nitf_details.img_segment_sizes[image_segment_index]
        raw_bands = len(image_header.Bands)
        raw_dtype, formatted_dtype, formatted_bands, complex_order, lut = self._get_dtypes(image_segment_index)
        # Establish block pixel bounds
        block_bounds = self._construct_block_bounds(image_segment_index)
        assert isinstance(block_bounds, list)

        raw_shape = _get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2)

        # get mask definition details
        mask_offsets, exclude_value, additional_offset = self._get_mask_details(image_segment_index)

        # jpeg compression, read everything (skipping mask) and find the jpeg delimiters
        the_bytes = self._read_file_data(offset+additional_offset, image_segment_size-additional_offset)
        jpeg_delimiters = find_jpeg_delimiters(the_bytes)

        # validate our discovered delimiters and the mask offsets
        if mask_offsets is not None:
            if not (isinstance(mask_offsets, numpy.ndarray) and mask_offsets.ndim == 1):
                raise ValueError('Got unexpected mask offsets `{}`'.format(mask_offsets))

            if len(block_bounds) != len(mask_offsets):
                raise ValueError('Got mismatch between block definition and mask offsets definition')

            # TODO: verify that we don't need to account for mask definition length
            anticipated_jpeg_indices = [index for index, entry in enumerate(mask_offsets)
                                        if entry != exclude_value]
            if len(jpeg_delimiters) != len(anticipated_jpeg_indices):
                raise ValueError(
                    'Found different number of jpeg delimiters ({})\n\t'
                    'than populated blocks ({}) in masked image segment {}'.format(
                        len(jpeg_delimiters), len(anticipated_jpeg_indices), image_segment_index))
            for jpeg_delim, mask_index in zip(jpeg_delimiters, anticipated_jpeg_indices):
                if mask_offsets[mask_index] != jpeg_delim[0]:
                    raise ValueError(
                        'Populated mask offsets ({})\n\t'
                        'do not agree with discovered jpeg offsets ({})\n\t'
                        'with mask subheader length {}'.format(jpeg_delim, mask_offsets, additional_offset))

        else:
            if len(jpeg_delimiters) != len(block_bounds):
                raise ValueError(
                    'Found different number of jpeg delimiters ({}) than blocks ({}) in image segment {}'.format(
                        len(jpeg_delimiters), len(block_bounds), image_segment_index))
            mask_offsets = [entry[0] for entry in jpeg_delimiters]

        # create a memmap, and extract all of our jpeg data into it as appropriate
        fi, path_name = mkstemp(suffix='.sarpy_cache', text=False)
        self._delete_temp_files.append(path_name)
        mem_map = numpy.memmap(
            path_name, dtype=raw_dtype, mode='w+', offset=0,
            shape=_get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2))
        next_jpeg_block = 0
        for mask_offset, block_bound in zip(mask_offsets, block_bounds):
            if mask_offset == exclude_value:
                continue  # just skip it, it's masked out
            jpeg_delim = jpeg_delimiters[next_jpeg_block]
            img = PIL_Image.open(BytesIO(the_bytes[jpeg_delim[0]:jpeg_delim[1]]))
            # handle block padding situation
            row_start, row_end = block_bound[0], min(block_bound[1], image_header.NROWS)
            col_start, col_end = block_bound[2], min(block_bound[3], image_header.NCOLS)
            mem_map[row_start: row_end, col_start:col_end] = numpy.asarray(img)[0:row_end - row_start,
                                                             0:col_end - col_start]
            next_jpeg_block += 1

        mem_map.flush()  # write all the data to the file
        del mem_map  # clean up the memmap
        os.close(fi)

        if apply_format:
            format_function = self.get_format_function(
                raw_dtype, complex_order, lut, 2,
                image_segment_index=image_segment_index)
            reverse_axes = self._reverse_axes
            if self._transpose_axes is None:
                formatted_shape = _get_shape(image_header.NROWS, image_header.NCOLS, formatted_bands, band_dimension=2)
            else:
                formatted_shape = _get_shape(image_header.NCOLS, image_header.NROWS, formatted_bands, band_dimension=2)
            transpose_axes = self._get_transpose(formatted_bands)
        else:
            format_function = None
            reverse_axes = None
            transpose_axes = None
            formatted_dtype = raw_dtype
            formatted_shape = raw_shape

        return NumpyMemmapSegment(
            path_name, 0, raw_dtype, raw_shape,
            formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode='r', close_file=False)

    def _handle_no_compression(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        # NB: Natural order inside the block is (bands, rows, columns)
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE not in ['B', 'R', 'P'] or image_header.IC not in ['NC', 'NM']:
            raise ValueError(
                'Requires IMODE in `(B, R, P)` and IC in `(NC, NM)`,\n\t'
                'got `{}` and `{}` at image segment index {}'.format(
                    image_header.IMODE, image_header.IC, image_segment_index))

        raw_bands = len(image_header.Bands)

        # get bytes offset to this image segment (relative to start of file)
        offset = self.nitf_details.img_segment_offsets[image_segment_index]
        raw_dtype, formatted_dtype, formatted_bands, complex_order, lut = self._get_dtypes(image_segment_index)
        can_use_memmap = self.can_use_memmap()

        block_bounds = self._construct_block_bounds(image_segment_index)
        assert isinstance(block_bounds, list)

        bytes_per_pixel = raw_bands*raw_dtype.itemsize
        block_size = int(image_header.NPPBH*image_header.NPPBV*bytes_per_pixel)

        if image_header.IMODE == 'B':
            # order inside the block is (bands, rows, columns)
            raw_band_dimension = 0
        elif image_header.IMODE == 'R':
            # order inside the block is (rows, bands, columns)
            raw_band_dimension = 1
        elif image_header.IMODE == 'P':
            # order inside the block is (rows, columns, bands)
            raw_band_dimension = 2
        else:
            raise ValueError('Unhandled IMODE `{}`'.format(image_header.IMODE))
        raw_shape = _get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=raw_band_dimension)

        # get mask definition details
        mask_offsets, exclude_value, additional_offset = self._get_mask_details(image_segment_index)

        block_offsets = mask_offsets if mask_offsets is not None else \
            numpy.arange(len(block_bounds), dtype='int64')*block_size

        if not (isinstance(block_offsets, numpy.ndarray) and mask_offsets.ndim == 1):
            raise ValueError('Got unexpected mask offsets `{}`'.format(block_offsets))

        if len(block_bounds) != len(block_offsets):
            raise ValueError('Got mismatch between block definition and block offsets definition')

        # determine output particulars
        if apply_format:
            format_function = self.get_format_function(
                raw_dtype, complex_order, lut, raw_band_dimension,
                image_segment_index=image_segment_index)
            use_transpose = self._transpose_axes
            use_reverse = self._reverse_axes
            if self._transpose_axes is None:
                formatted_shape = _get_shape(image_header.NROWS, image_header.NCOLS, formatted_bands, band_dimension=2)
            else:
                formatted_shape = _get_shape(image_header.NCOLS, image_header.NROWS, formatted_bands, band_dimension=2)
        else:
            format_function = None
            use_transpose = None
            use_reverse = None
            formatted_dtype = raw_dtype
            formatted_shape = _get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2)

        # account for rearrangement of bands to final dimension
        if raw_bands == 1:
            transpose_axes = use_transpose
            reverse_axes = use_reverse
        elif image_header.IMODE == 'B':
            # order inside the block is (bands, rows, columns)
            transpose_axes = (1, 2, 0) if use_transpose is None else (2, 1, 0)
            reverse_axes = None if use_reverse is None else tuple(entry + 1 for entry in use_reverse)
        elif image_header.IMODE == 'R':
            # order inside the block is (rows, bands, columns)
            transpose_axes = (0, 2, 1) if use_transpose is None else (2, 0, 1)
            reverse_mapping = {0: 0, 1: 2}
            reverse_axes = None if use_reverse is None else \
                tuple(reverse_mapping[entry] for entry in use_reverse)
        elif image_header.IMODE == 'P':
            transpose_axes = None if use_transpose is None else use_transpose + (2, )
            reverse_axes = use_reverse
        else:
            raise ValueError('Unhandled IMODE `{}`'.format(image_header.IMODE))

        if len(block_bounds) == 1:
            # there is just a single block, no need to obfuscate behind a
            # block aggregate

            if can_use_memmap:
                return NumpyMemmapSegment(
                    self.file_object, offset, raw_dtype, raw_shape,
                    formatted_dtype, formatted_shape, reverse_axes=reverse_axes,
                    transpose_axes=transpose_axes, format_function=format_function,
                    close_file=False)
            else:
                return FileReadDataSegment(
                    self.file_object, offset, raw_dtype, raw_shape,
                    formatted_dtype, formatted_shape, reverse_axes=reverse_axes,
                    transpose_axes=transpose_axes, format_function=format_function,
                    close_file=False)

        data_segments = []
        child_arrangement = []
        for block_index, (block_definition, block_offset) in enumerate(zip(block_bounds, block_offsets)):
            if block_offset == exclude_value:
                continue  # just skip this, since it's masked out

            b_rows = block_definition[1] - block_definition[0]
            b_cols = block_definition[3] - block_definition[2]
            b_raw_shape = _get_shape(b_rows, b_cols, raw_bands, band_dimension=raw_band_dimension)
            total_offset = offset + additional_offset + block_offset
            if can_use_memmap:
                child_segment = NumpyMemmapSegment(
                    self.file_object, total_offset, raw_dtype, b_raw_shape,
                    raw_dtype, b_raw_shape, close_file=False)
            else:
                child_segment = FileReadDataSegment(
                    self.file_object, total_offset, raw_dtype, b_raw_shape,
                    raw_dtype, b_raw_shape, close_file=False)
            # handle block padding situation
            row_start, row_end = block_definition[0], min(block_definition[1], image_header.NROWS)
            col_start, col_end = block_definition[2], min(block_definition[3], image_header.NCOLS)
            if row_end == block_definition[1] and col_end == block_definition[3]:
                data_segments.append(child_segment)
            else:
                subset_def = _get_subscript_def(
                    0, row_end - row_start, 0, col_end - col_start, raw_bands, raw_band_dimension)
                data_segments.append(
                    SubsetSegment(child_segment, subset_def, 'raw', close_parent=True))

            # determine arrangement of these children
            child_def = _get_subscript_def(
                row_start, row_end, col_start, col_end, raw_bands, raw_band_dimension)
            child_arrangement.append(child_def)

        return BlockAggregateSegment(
            data_segments, child_arrangement, 'raw', 0, raw_shape,
            formatted_dtype, formatted_shape, reverse_axes=reverse_axes,
            transpose_axes=transpose_axes, format_function=format_function,
            close_children=True)

    def _handle_imode_s_jpeg(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        # TODO: handle IMODE S, JPEG data
        # TODO: verify that we don't need to account for mask definition length

        raise NotImplementedError

    def _handle_imode_s_no_compression(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'S' or image_header.IC in ['NC', 'NM']:
            raise ValueError(
                'Requires IMODE = `S` and IC in `(NC, NM)`, got `{}` and `{}` at image segment index {}'.format(
                    image_header.IMODE, image_header.IC, image_segment_index))

        # get bytes offset to this image segment (relative to start of file)
        offset = self.nitf_details.img_segment_offsets[image_segment_index]
        raw_bands = len(image_header.Bands)
        raw_shape = _get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2)
        raw_dtype, formatted_dtype, formatted_bands, complex_order, lut = self._get_dtypes(image_segment_index)
        can_use_memmap = self.can_use_memmap()

        block_bounds = self._construct_block_bounds(image_segment_index)
        assert isinstance(block_bounds, list)

        # get mask definition details
        mask_offsets, exclude_value, additional_offset = self._get_mask_details(image_segment_index)

        bytes_per_pixel = raw_dtype.itemsize  # one band at a time
        block_size = int(image_header.NPPBH*image_header.NPPBV*bytes_per_pixel)
        if mask_offsets is not None:
            block_offsets = mask_offsets
        else:
            block_offsets = numpy.zeros((raw_bands, len(block_bounds)), dtype='int64')
            for i in range(raw_bands):
                block_offsets[i, :] = i*(block_size*len(block_bounds)) + numpy.arange(len(block_bounds), dtype='int64')*block_size

        if not (isinstance(block_offsets, numpy.ndarray) and block_offsets.ndim == 2):
            raise ValueError('Got unexpected block offsets `{}`'.format(block_offsets))

        if len(block_bounds) != block_offsets.shape[1]:
            raise ValueError('Got mismatch between block definition and block offsets definition')

        band_segments = []
        for band_number in range(raw_bands):
            band_raw_shape = _get_shape(image_header.NROWS, image_header.NCOLS, 1, band_dimension=2)
            data_segments = []
            child_arrangement = []
            for block_index, (block_definition, block_offset) in enumerate(block_bounds, block_offsets[band_number, :]):
                if block_offset == exclude_value:
                    continue  # just skip this, since it's masked out

                b_rows = block_definition[1] - block_definition[0]
                b_cols = block_definition[3] - block_definition[2]
                b_raw_shape = _get_shape(b_rows, b_cols, 1, band_dimension=2)
                total_offset = offset + additional_offset + block_offset
                if can_use_memmap:
                    child_segment = NumpyMemmapSegment(
                        self.file_object, total_offset, raw_dtype, b_raw_shape,
                        raw_dtype, b_raw_shape)
                else:
                    child_segment = FileReadDataSegment(
                        self.file_object, total_offset, raw_dtype, b_raw_shape,
                        raw_dtype, b_raw_shape, close_file=False)
                # handle block padding situation
                row_start, row_end = block_definition[0], min(block_definition[1], image_header.NROWS)
                col_start, col_end = block_definition[2], min(block_definition[3], image_header.NCOLS)

                child_def = _get_subscript_def(
                    row_start, row_end, col_start, col_end, 1, 2)
                child_arrangement.append(child_def)
                if row_end == block_definition[1] and col_end == block_definition[3]:
                    data_segments.append(child_segment)
                else:
                    subset_def = _get_subscript_def(
                        0, row_end - row_start, 0, col_end - col_start, 1, 2)
                    data_segments.append(
                        SubsetSegment(child_segment, subset_def, 'raw', close_parent=True))
            band_segments.append(BlockAggregateSegment(
                data_segments, child_arrangement, 'raw', 0, band_raw_shape,
                raw_dtype, band_raw_shape, close_children=True))

        if apply_format:
            format_function = self.get_format_function(
                raw_dtype, complex_order, lut, 2,
                image_segment_index=image_segment_index)
            reverse_axes = self._reverse_axes
            if self._transpose_axes is None:
                formatted_shape = _get_shape(image_header.NROWS, image_header.NCOLS, formatted_bands, band_dimension=2)
            else:
                formatted_shape = _get_shape(image_header.NCOLS, image_header.NROWS, formatted_bands, band_dimension=2)
            transpose_axes = self._get_transpose(formatted_bands)
        else:
            format_function = None
            reverse_axes = None
            transpose_axes = None
            formatted_dtype = raw_dtype
            formatted_shape = raw_shape

        return BandAggregateSegment(
            band_segments, 2, formatted_dtype=formatted_dtype, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, close_children=True)

    def _create_data_segment_from_imode_b(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'B':
            raise ValueError(
                'Requires IMODE = `B`, got `{}` at image segment index {}'.format(
                    image_header.IMODE, image_segment_index))
        # this supports any viable compression scheme
        if image_header.IC in self.unsupported_compressions:
            raise ValueError(
                'Unsupported IC `{}` at image segment index {}'.format(
                    image_header.IC, image_segment_index))

        if image_header.IC in ['NC', 'NM']:
            return self._handle_no_compression(image_segment_index, apply_format)
        elif image_header.IC in ['I1', 'C3', 'C5', 'M3', 'M5']:
            return self._handle_jpeg(image_segment_index, apply_format)
        elif image_header.IC == 'C8':
            return self._handle_jpeg2k_no_mask(image_segment_index, apply_format)
        elif image_header.IC == 'C8':
            return self._handle_jpeg2k_with_mask(image_segment_index, apply_format)
        else:
            raise ValueError('Got unhandled IC `{}`'.format(image_header.IC))

    def _create_data_segment_from_imode_p(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'P':
            raise ValueError(
                'Requires IMODE = `P`, got `{}` at image segment index {}'.format(
                    image_header.IMODE, image_segment_index))

        if image_header.IC not in ['NC', 'NM', 'C3', 'M3', 'C5', 'M5']:
            raise ValueError(
                'IMODE is `P` and the IC is `{}` at image segment index {}'.format(
                    image_header.IC, image_segment_index))

        if image_header.IC in ['NC', 'NM']:
            return self._handle_no_compression(image_segment_index, apply_format)
        elif image_header.IC in ['C3', 'C5', 'M3', 'M5']:
            return self._handle_jpeg(image_segment_index, apply_format)
        else:
            raise ValueError('Got unhandled IC `{}`'.format(image_header.IC))

    def _create_data_segment_from_imode_r(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'R':
            raise ValueError(
                'Requires IMODE = `R`, got `{}` at image segment index {}'.format(
                    image_header.IMODE, image_segment_index))
        if image_header.IMODE not in ['NC', 'NM']:
            raise ValueError(
                'IMODE is `R` and the image is compressed at image segment index {}'.format(
                    image_segment_index))

        if image_header.IC in ['NC', 'NM']:
            return self._handle_no_compression(image_segment_index, apply_format)
        else:
            raise ValueError('Got unhandled IC `{}`'.format(image_header.IC))

    def _create_data_segment_from_imode_s(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'S':
            raise ValueError(
                'Requires IMODE = `S`, got `{}` at image segment index {}'.format(
                    image_header.IMODE, image_segment_index))
        if image_header.IC not in ['NC', 'NM', 'C3', 'M3', 'C5', 'M5']:
            raise ValueError(
                'IMODE is `S` and the IC is `{}` at image segment index {}'.format(
                    image_header.IC, image_segment_index))
        if len(image_header.Bands) < 2:
            raise ValueError('IMODE S is only valid with multiple bands.')
        if image_header.NBPC == 1 and image_header.NBPR == 1:
            raise ValueError('IMODE S is only valid with multiple blocks.')

        if image_header.IC in ['NC', 'NM']:
            return self._handle_imode_s_no_compression(image_segment_index, apply_format)
        elif image_header.IC in ['C3', 'C5', 'M3', 'M5']:
            return self._handle_imode_s_jpeg(image_segment_index, apply_format)
        else:
            raise ValueError('Got unhandled IC `{}`'.format(image_header.IC))

    def create_data_segment_for_image_segment(
            self,
            image_segment_index: int,
            apply_format: bool) -> DataSegment:
        """
        Creates the data segment for the given image segment.

        For consistency of simple usage, any bands will be presented in the
        final formatted/output dimension, regardless of the value of `apply_format`
        or `IMODE`.

        For compressed image segments, the `IMODE` has been
        abstracted away, and the data segment will be consistent with the raw
        shape having bands in the final dimension (analogous to `IMODE=P`).

        Parameters
        ----------
        image_segment_index : int
        apply_format : bool
            Leave data raw (False), or apply format function and global
            `reverse_axes` and `transpose_axes` values?

        Returns
        -------
        DataSegment
        """

        image_header = self.get_image_header(image_segment_index)

        if image_header.IMODE == 'B':
            return self._create_data_segment_from_imode_b(image_segment_index, apply_format)
        elif image_header.IMODE == 'P':
            return self._create_data_segment_from_imode_p(image_segment_index, apply_format)
        elif image_header.IMODE == 'S':
            return self._create_data_segment_from_imode_s(image_segment_index, apply_format)
        elif image_header.IMODE == 'R':
            return self._create_data_segment_from_imode_r(image_segment_index, apply_format)
        else:
            raise ValueError(
                'Got unsupported IMODE `{}` at image segment index `{}`'.format(
                    image_header.IMODE, image_segment_index))

    def _create_data_segment_for_collection_element(self, collection_index: int) -> DataSegment:
        """
        Creates the data segment overarching the given segment collection.

        Parameters
        ----------
        collection_index : int

        Returns
        -------
        DataSegment
        """

        block = self.image_segment_collections[collection_index]

        if len(block) == 1:
            return self.create_data_segment_for_image_segment(block[0], True)

        block_definition = self._get_collection_element_coordinate_limits(collection_index)
        total_rows = int(numpy.max(block_definition[:, 1]))
        total_columns = int(numpy.max(block_definition[:, 3]))

        raw_dtype, formatted_dtype, formatted_bands, complex_order, lut = self._get_dtypes(block[0])
        format_function = self.get_format_function(raw_dtype, complex_order, lut, 2)

        child_segments = []
        child_arrangement = []
        raw_bands = None
        for img_index, block_def in zip(block, block_definition):
            child_segment = self.create_data_segment_for_image_segment(img_index, False)
            # NB: the bands in the formatted data will be in the final dimension
            if raw_bands is None:
                raw_bands = 1 if child_segment.formatted_ndim == 2 else \
                    child_segment.formatted_shape[2]
            child_segments.append(child_segment)
            child_arrangement.append(
                _get_subscript_def(
                    int(block_def[0]), int(block_def[1]), int(block_def[2]), int(block_def[3]), raw_bands, 2))
        transpose = self._get_transpose(formatted_bands)
        raw_shape = (total_rows, total_columns) if raw_bands == 1 else (total_rows, total_columns, raw_bands)

        formatted_shape = raw_shape[:2] if transpose is None else (raw_shape[1], raw_shape[0])
        if formatted_bands > 1:
            formatted_shape = formatted_shape + (formatted_bands, )

        return BlockAggregateSegment(
            child_segments, child_arrangement, 'raw', 0, raw_shape, formatted_dtype, formatted_shape,
            reverse_axes=self._reverse_axes, transpose_axes=transpose, format_function=format_function,
            close_children=True)

    def get_data_segments(self) -> List[DataSegment]:
        """
        Gets a data segment for each of these image segment collection.

        Returns
        -------
        List[DataSegment]
        """

        out = []
        for index in range(len(self.image_segment_collections)):
            out.append(self._create_data_segment_for_collection_element(index))
        return out


#####
# A general nitf writer and associated elements - intended for extension

class ImageDetails(object):
    """
    Helper class for managing the details about a given NITF segment.
    """

    __slots__ = (
        '_bands', '_dtype', '_transform_data', '_parent_index_range',
        '_subheader', '_subheader_offset', '_item_offset',
        '_subheader_written', '_pixels_written')

    def __init__(self, bands, dtype, transform_data, parent_index_range, subheader):
        """

        Parameters
        ----------
        bands : int
            The number of bands.
        dtype : str|numpy.dtype|numpy.number
            The dtype for the associated chipper.
        transform_data : bool|callable
            The transform_data for the associated chipper.
        parent_index_range : Tuple[int]
            Indicates `(start row, end row, start column, end column)` relative to
            the parent image.
        subheader : ImageSegmentHeader
            The image subheader.
        """

        self._subheader_offset = None
        self._item_offset = None
        self._pixels_written = 0
        self._subheader_written = False

        self._bands = int(bands)
        if self._bands <= 0:
            raise ValueError('bands must be positive.')
        self._dtype = dtype
        self._transform_data = transform_data

        if len(parent_index_range) != 4:
            raise ValueError('parent_index_range must have length 4.')
        self._parent_index_range = (
            int(parent_index_range[0]), int(parent_index_range[1]),
            int(parent_index_range[2]), int(parent_index_range[3]))

        if self._parent_index_range[0] < 0 or self._parent_index_range[1] <= self._parent_index_range[0]:
            raise ValueError(
                'Invalid parent row start/end ({}, {})'.format(self._parent_index_range[0],
                                                               self._parent_index_range[1]))
        if self._parent_index_range[2] < 0 or self._parent_index_range[3] <= self._parent_index_range[2]:
            raise ValueError(
                'Invalid parent row start/end ({}, {})'.format(self._parent_index_range[2],
                                                               self._parent_index_range[3]))
        if not isinstance(subheader, ImageSegmentHeader):
            raise TypeError(
                'subheader must be an instance of ImageSegmentHeader, got '
                'type {}'.format(type(subheader)))
        self._subheader = subheader

    @property
    def subheader(self):
        """
        ImageSegmentHeader: The image segment subheader.
        """

        return self._subheader

    @property
    def rows(self):
        """
        int: The number of rows.
        """

        return self._parent_index_range[1] - self._parent_index_range[0]

    @property
    def cols(self):
        """
        int: The number of columns.
        """

        return self._parent_index_range[3] - self._parent_index_range[2]

    @property
    def subheader_offset(self):
        """
        int: The subheader offset.
        """

        return self._subheader_offset

    @subheader_offset.setter
    def subheader_offset(self, value):
        if self._subheader_offset is not None:
            logger.warning("subheader_offset is read only after being initially defined.")
            return
        self._subheader_offset = int(value)
        self._item_offset = self._subheader_offset + self._subheader.get_bytes_length()

    @property
    def item_offset(self):
        """
        int: The image offset.
        """

        return self._item_offset

    @property
    def end_of_item(self):
        """
        int: The position of the end of the image.
        """

        return self.item_offset + self.image_size

    @property
    def total_pixels(self):
        """
        int: The total number of pixels.
        """

        return self.rows*self.cols

    @property
    def image_size(self):
        """
        int: The size of the image in bytes.
        """

        return int(self.total_pixels*self.subheader.NBPP*len(self.subheader.Bands)/8)

    @property
    def pixels_written(self):
        """
        int: The number of pixels written
        """

        return self._pixels_written

    @property
    def subheader_written(self):
        """
        bool: The status of writing the subheader.
        """

        return self._subheader_written

    @subheader_written.setter
    def subheader_written(self, value):
        if self._subheader_written:
            return
        elif value:
            self._subheader_written = True

    @property
    def image_written(self):
        """
        bool: The status of whether the image segment is fully written. This
        naively checks assuming that no pixels have been written redundantly.
        """

        return self._pixels_written >= self.total_pixels

    def count_written(self, index_tuple):
        """
        Count the overlap that we have written in a given step.

        Parameters
        ----------
        index_tuple : Tuple[int]
            Tuple of the form `(row start, row end, column start, column end)`

        Returns
        -------
        None
        """

        new_pixels = (index_tuple[1] - index_tuple[0])*(index_tuple[3] - index_tuple[2])
        self._pixels_written += new_pixels
        if self._pixels_written > self.total_pixels:
            logger.error(
                'A total of {} pixels have been written,\n\t'
                'for an image that should only have {} pixels.'.format(
                self._pixels_written, self.total_pixels))

    def get_overlap(self, index_range):
        """
        Determines overlap for the given image segment.

        Parameters
        ----------
        index_range : Tuple[int]
            Indicates `(start row, end row, start column, end column)` for prospective incoming data.

        Returns
        -------
        Union[Tuple[None], Tuple[Tuple[int]]
            `(None, None)` if there is no overlap. Otherwise, tuple of the form
            `((start row, end row, start column, end column),
                (parent start row, parent end row, parent start column, parent end column))`
            indicating the overlap portion with respect to this image, and the parent image.
        """

        def element_overlap(this_start, this_end, parent_start, parent_end):
            st, ed = None, None
            if this_start <= parent_start <= this_end:
                st = parent_start
                ed = min(int(this_end), parent_end)
            elif parent_start <= this_start <= parent_end:
                st = int(this_start)
                ed = min(int(this_end), parent_end)
            return st, ed

        # do the rows overlap?
        row_s, row_e = element_overlap(index_range[0], index_range[1],
                                       self._parent_index_range[0], self._parent_index_range[1])
        if row_s is None:
            return None, None

        # do the columns overlap?
        col_s, col_e = element_overlap(index_range[2], index_range[3],
                                       self._parent_index_range[2], self._parent_index_range[3])
        if col_s is None:
            return None, None

        return (row_s-self._parent_index_range[0], row_e-self._parent_index_range[0],
                col_s-self._parent_index_range[2], col_e-self._parent_index_range[2]), \
               (row_s, row_e, col_s, col_e)

    def create_writer(self, file_name):
        """
        Creates the BIP writer for this image segment.

        Parameters
        ----------
        file_name : str
            The parent file name.

        Returns
        -------
        BIPWriter
        """

        if self._item_offset is None:
            raise ValueError('The image segment subheader_offset must be defined '
                             'before a writer can be defined.')
        return BIPWriter(
            file_name, (self.rows, self.cols), self._dtype, self._bands,
            self._transform_data, data_offset=self.item_offset)


class DESDetails(object):
    """
    Helper class for managing the details about a given NITF Data Extension Segment.
    """

    __slots__ = (
        '_subheader', '_subheader_offset', '_item_offset', '_des_bytes',
        '_subheader_written', '_des_written')

    def __init__(self, subheader, des_bytes):
        """

        Parameters
        ----------
        subheader : DataExtensionHeader
            The data extension subheader.
        """
        self._subheader_offset = None
        self._item_offset = None
        self._subheader_written = False
        self._des_written = False

        if not isinstance(subheader, DataExtensionHeader):
            raise TypeError(
                'subheader must be an instance of DataExtensionHeader, got '
                'type {}'.format(type(subheader)))
        self._subheader = subheader

        if not isinstance(des_bytes, bytes):
            raise TypeError('des_bytes must be an instance of bytes, got '
                            'type {}'.format(type(des_bytes)))
        self._des_bytes = des_bytes

    @property
    def subheader(self):
        """
        DataExtensionHeader: The data extension subheader.
        """

        return self._subheader

    @property
    def des_bytes(self):
        """
        bytes: The data extension bytes.
        """

        return self._des_bytes

    @property
    def subheader_offset(self):
        """
        int: The subheader offset.
        """

        return self._subheader_offset

    @subheader_offset.setter
    def subheader_offset(self, value):
        if self._subheader_offset is not None:
            logger.warning("subheader_offset is read only after being initially defined.")
            return
        self._subheader_offset = int(value)
        self._item_offset = self._subheader_offset + self._subheader.get_bytes_length()

    @property
    def item_offset(self):
        """
        int: The image offset.
        """

        return self._item_offset

    @property
    def end_of_item(self):
        """
        int: The position of the end of the data extension.
        """

        return self.item_offset + len(self._des_bytes)

    @property
    def subheader_written(self):
        """
        bool: The status of writing the subheader.
        """

        return self._subheader_written

    @subheader_written.setter
    def subheader_written(self, value):
        if self._subheader_written:
            return
        elif value:
            self._subheader_written = True

    @property
    def des_written(self):
        """
        bool: The status of writing the subheader.
        """

        return self._des_written

    @des_written.setter
    def des_written(self, value):
        if self._des_written:
            return
        elif value:
            self._des_written = True


def get_npp_block(value):
    """
    Determine the number of pixels per block value.

    Parameters
    ----------
    value : int

    Returns
    -------
    int
    """

    return 0 if value > 8192 else value


def image_segmentation(rows, cols, pixel_size):
    """
    Determine the appropriate segmentation for the image. This is driven
    by the SICD/SIDD standard, and not the only generally feasible segmentation
    scheme for other NITF file types.

    Parameters
    ----------
    rows : int
    cols : int
    pixel_size : int

    Returns
    -------
    tuple
        Of the form `((row start, row end, column start, column end))`
    """

    im_seg_limit = 10**10 - 2  # as big as can be stored in 10 digits
    im_segments = []
    row_offset = 0
    while row_offset < rows:
        # determine row count, given row_offset and column size
        # how many bytes per row for this column section
        row_memory_size = cols*pixel_size
        # how many rows can we use?
        row_count = min(99999, rows - row_offset, int(im_seg_limit / row_memory_size))
        im_segments.append((row_offset, row_offset + row_count, 0, cols))
        row_offset += row_count  # move the next row offset
    return tuple(im_segments)


def interpolate_corner_points_string(entry, rows, cols, icp):
    """
    Interpolate the corner points for the given subsection from
    the given corner points. This supplies entries for the NITF headers.

    Parameters
    ----------
    entry : numpy.ndarray
        The corner pints of the form `(row_start, row_stop, col_start, col_stop)`
    rows : int
        The number of rows in the parent image.
    cols : int
        The number of cols in the parent image.
    icp : the parent image corner points in geodetic coordinates.

    Returns
    -------
    str
    """

    if icp is None:
        return ''

    if icp.shape[1] == 2:
        icp_new = numpy.zeros((icp.shape[0], 3), dtype=numpy.float64)
        icp_new[:, :2] = icp
        icp = icp_new
    icp_ecf = geodetic_to_ecf(icp)

    const = 1. / (rows * cols)
    pattern = entry[numpy.array([(0, 2), (1, 2), (1, 3), (0, 3)], dtype=numpy.int64)]
    out = []
    for row, col in pattern:
        pt_array = const * numpy.sum(icp_ecf *
                                     (numpy.array([rows - row, row, row, rows - row]) *
                                      numpy.array([cols - col, cols - col, col, col]))[:, numpy.newaxis], axis=0)

        pt = LatLonType.from_array(ecf_to_geodetic(pt_array)[:2])
        dms = pt.dms_format(frac_secs=False)
        out.append('{0:02d}{1:02d}{2:02d}{3:s}'.format(*dms[0]) + '{0:03d}{1:02d}{2:02d}{3:s}'.format(*dms[1]))
    return ''.join(out)


#########
# TODO: update NITf writer...
class NITFWriter(BaseWriter):
    __slots__ = (
        '_file_name', '_security_tags', '_nitf_header', '_nitf_header_written',
        '_img_groups', '_shapes', '_img_details', '_writing_chippers', '_des_details',
        '_closed')

    def __init__(self, file_name, check_existence=True):
        """

        Parameters
        ----------
        file_name : str
        check_existence : bool
            Should we check if the given file already exists, and raises an exception if so?
        """

        if check_existence and os.path.exists(file_name):
            raise SarpyIOError('Given file {} already exists, and a new NITF file cannot be created here.'.format(file_name))

        self._writing_chippers = None
        self._nitf_header_written = False
        self._closed = False
        super(NITFWriter, self).__init__(file_name)
        self._create_security_tags()
        self._create_image_segment_details()
        self._create_data_extension_details()
        self._create_nitf_header()

    @property
    def nitf_header_written(self):  # type: () -> bool
        """
        bool: The status of whether of not we have written the NITF header.
        """

        return self._nitf_header_written

    @property
    def security_tags(self):  # type: () -> NITFSecurityTags
        """
        NITFSecurityTags: The NITF security tags, which will be constructed initially using
        the :func:`default_security_tags` method. This object will be populated **by reference**
        upon construction as the `SecurityTags` property for `nitf_header`, each entry of
        `image_segment_headers`, and `data_extension_header`.

        .. Note:: required edits should be made before adding any data via :func:`write_chip`.
        """

        return self._security_tags

    @property
    def nitf_header(self):  # type: () -> NITFHeader
        """
        NITFHeader: The NITF header object. The `SecurityTags` property is populated
        using `security_tags` **by reference** upon construction.

        .. Note:: required edits should be made before adding any data via :func:`write_chip`.
        """

        return self._nitf_header

    @property
    def image_details(self):  # type: () -> Tuple[ImageDetails]
        """
        Tuple[ImageDetails]: The individual image segment details.
        """

        return self._img_details

    @property
    def des_details(self):  # type: () -> Tuple[DESDetails]
        """
        Tuple[DESDetails]: The individual data extension details.
        """

        return self._des_details

    def _set_offsets(self):
        """
        Sets the offsets for the ImageDetail and DESDetail objects.

        Returns
        -------
        None
        """

        if self.nitf_header is None:
            raise ValueError("The _set_offsets method must be called AFTER the "
                             "_create_nitf_header, _create_image_segment_headers, "
                             "and _create_data_extension_headers methods.")
        if self._img_details is not None and \
                (self.nitf_header.ImageSegments.subhead_sizes.size != len(self._img_details)):
            raise ValueError('The length of _img_details and the defined ImageSegments '
                             'in the NITF header do not match.')
        elif self._img_details is None and \
                self.nitf_header.ImageSegments.subhead_sizes.size != 0:
            raise ValueError('There are no _img_details defined, while there are ImageSegments '
                             'defined in the NITF header.')

        if self._des_details is not None and \
                (self.nitf_header.DataExtensions.subhead_sizes.size != len(self._des_details)):
            raise ValueError('The length of _des_details and the defined DataExtensions '
                             'in the NITF header do not match.')
        elif self._des_details is None and \
                self.nitf_header.DataExtensions.subhead_sizes.size != 0:
            raise ValueError('There are no _des_details defined, while there are DataExtensions '
                             'defined in the NITF header.')

        offset = self.nitf_header.get_bytes_length()

        # set the offsets for the image details
        if self._img_details is not None:
            for details in self._img_details:
                details.subheader_offset = offset
                offset = details.end_of_item

        # set the offsets for the data extensions
        if self._des_details is not None:
            for details in self._des_details:
                details.subheader_offset = offset
                offset = details.end_of_item

        # set the file size in the nitf header
        self.nitf_header.FL = offset
        self.nitf_header.CLEVEL = self._get_clevel(offset)

    def _write_file_header(self):
        """
        Write the file header.

        Returns
        -------
        None
        """

        if self._nitf_header_written:
            return

        logger.info('Writing NITF header.')
        with open(self._file_name, mode='r+b') as fi:
            fi.write(self.nitf_header.to_bytes())
            self._nitf_header_written = True

    def prepare_for_writing(self):
        """
        The NITF file header makes specific reference of the locations/sizes of
        various components, specifically the image segment subheader lengths and
        the data extension subheader and item lengths. These items must be locked
        down BEFORE we can allocate the required file writing specifics from the OS.

        Any desired header modifications (i.e. security tags or any other issues) must be
        finalized, before the final steps to actually begin writing data. Calling
        this method prepares the final versions of the headers, and prepares for actual file
        writing. Any modifications to any header information made AFTER calling this method
        will not be reflected in the produced NITF file.

        .. Note:: This will be implicitly called at first attempted chip writing
            if it has not be explicitly called before.

        Returns
        -------
        None
        """

        if self._nitf_header_written:
            return

        # set the offsets for the images and data extensions,
        #   and the file size in the NITF header
        self._set_offsets()
        self._write_file_header()

        logger.info(
            'Setting up the image segments in virtual memory.\n\t'
            'This may require a large physical memory allocation, and be time consuming.')
        self._writing_chippers = tuple(
            details.create_writer(self._file_name) for details in self.image_details)

    def _write_image_header(self, index):
        """
        Write the image subheader at `index`, if necessary.

        Parameters
        ----------
        index : int

        Returns
        -------
        None
        """

        details = self.image_details[index]

        if details.subheader_written:
            return

        if details.subheader_offset is None:
            raise ValueError('DESDetails.subheader_offset must be defined for index {}.'.format(index))

        logger.info(
            'Writing image segment {} header.\n\t'
            'Depending on OS details, this may require a\n\t'
            'large physical memory allocation, and be time consuming.'.format(index))
        with open(self._file_name, mode='r+b') as fi:
            fi.seek(details.subheader_offset, os.SEEK_SET)
            fi.write(details.subheader.to_bytes())
            details.subheader_written = True

    def _write_des_header(self, index):
        """
        Write the des subheader at `index`, if necessary.

        Parameters
        ----------
        index : int

        Returns
        -------
        None
        """

        details = self.des_details[index]

        if details.subheader_written:
            return

        if details.subheader_offset is None:
            raise ValueError('DESDetails.subheader_offset must be defined for index {}.'.format(index))

        logger.info(
            'Writing data extension {} header.'.format(index))
        with open(self._file_name, mode='r+b') as fi:
            fi.seek(details.subheader_offset, os.SEEK_SET)
            fi.write(details.subheader.to_bytes())
            details.subheader_written = True

    def _write_des_bytes(self, index):
        """
        Write the des bytes at `index`, if necessary.

        Parameters
        ----------
        index : int

        Returns
        -------
        None
        """

        details = self.des_details[index]
        assert isinstance(details, DESDetails)

        if details.des_written:
            return

        if not details.subheader_written:
            self._write_des_header(index)

        logger.info(
            'Writing data extension {}.'.format(index))
        with open(self._file_name, mode='r+b') as fi:
            fi.seek(details.item_offset, os.SEEK_SET)
            fi.write(details.des_bytes)
            details.des_written = True

    def _get_ftitle(self):
        """
        Define the FTITLE for the NITF header.

        Returns
        -------
        str
        """

        raise NotImplementedError

    def _get_fdt(self):
        """
        Gets the NITF header FDT field value.

        Returns
        -------
        str
        """

        return re.sub(r'[^0-9]', '', str(numpy.datetime64('now', 's')))

    def _get_ostaid(self):
        """
        Gets the NITF header OSTAID field value.

        Returns
        -------
        str
        """

        return 'Unknown'

    def _get_clevel(self, file_size):
        """
        Gets the NITF complexity level of the file. This is likely always
        dominated by the memory constraint.

        Parameters
        ----------
        file_size : int
            The file size in bytes

        Returns
        -------
        int
        """

        def memory_level():
            if file_size < 50*(1024**2):
                return 3
            elif file_size < (1024**3):
                return 5
            elif file_size < 2*(int(1024)**3):
                return 6
            elif file_size < 10*(int(1024)**3):
                return 7
            else:
                return 9

        def index_level(ind):
            if ind <= 2048:
                return 3
            elif ind <= 8192:
                return 5
            elif ind <= 65536:
                return 6
            else:
                return 7

        row_max = max(entry[0] for entry in self._shapes)
        col_max = max(entry[1] for entry in self._shapes)

        return max(memory_level(), index_level(row_max), index_level(col_max))

    def write_chip(self, data, start_indices=(0, 0), index=0):
        """
        Write the data to the file(s). This is an alias to :code:`writer(data, start_indices)`.

        Parameters
        ----------
        data : numpy.ndarray
            the complex data
        start_indices : tuple[int, int]
            the starting index for the data.
        index : int
            the chipper index to which to write

        Returns
        -------
        None
        """

        self.__call__(data, start_indices=start_indices, index=index)

    def __call__(self, data, start_indices=(0, 0), index=0):
        """
        Write the data to the file(s).

        Parameters
        ----------
        data : numpy.ndarray
            the complex data
        start_indices : Tuple[int, int]
            the starting index for the data.
        index : int
            the main image index to which to write - parent group of NITF image segments.

        Returns
        -------
        None
        """

        if index >= len(self._img_groups):
            raise IndexError('There are only {} image groups, got index {}'.format(len(self._img_groups), index))

        self.prepare_for_writing()  # no effect if already called

        # validate the index and data arguments
        start_indices = (int(start_indices[0]), int(start_indices[1]))
        shape = self._shapes[index]

        if (start_indices[0] < 0) or (start_indices[1] < 0):
            raise ValueError('start_indices must have positive entries. Got {}'.format(start_indices))
        if (start_indices[0] >= shape[0]) or \
                (start_indices[1] >= shape[1]):
            raise ValueError(
                'start_indices must be bounded from above by {}. Got {}'.format(shape, start_indices))

        index_range = (start_indices[0], start_indices[0] + data.shape[0],
                       start_indices[1], start_indices[1] + data.shape[1])
        if (index_range[1] > shape[0]) or (index_range[3] > shape[1]):
            raise ValueError(
                'Got start_indices = {} and data of shape {}. '
                'This is incompatible with total data shape {}.'.format(start_indices, data.shape, shape))

        # iterate over the image segments for this group, and write as appropriate
        for img_index in self._img_groups[index]:
            details = self._img_details[img_index]

            this_inds, overall_inds = details.get_overlap(index_range)
            if overall_inds is None:
                # there is no overlap here, so skip
                continue

            self._write_image_header(img_index)  # no effect if already called
            # what are the relevant indices into data?
            data_indices = (overall_inds[0] - start_indices[0], overall_inds[1] - start_indices[0],
                            overall_inds[2] - start_indices[1], overall_inds[3] - start_indices[1])
            # write the data
            self._writing_chippers[img_index](data[data_indices[0]:data_indices[1], data_indices[2]: data_indices[3]],
                                              (this_inds[0], this_inds[2]))
            # count the written pixels
            details.count_written(this_inds)

    def close(self):
        """
        Completes any necessary final steps.

        Returns
        -------
        None
        """

        if self._closed:
            return

        # set this status first, in the event of some kind of error
        self._closed = True
        # ensure that all images are fully written
        msg = None
        if self.image_details is not None:
            for i, img_details in enumerate(self.image_details):
                if not img_details.image_written:
                    msg_part = "Image segment {} has only written {} of {} pixels".format(
                        i, img_details.pixels_written, img_details.total_pixels)
                    msg = msg_part if msg is None else msg + '\n\t' + msg_part
                    logger.critical(msg_part)
        # ensure that all data extensions are fully written
        if self.des_details is not None:
            for i, des_detail in enumerate(self.des_details):
                if not des_detail.des_written:
                    self._write_des_bytes(i)
        # close all the chippers
        if self._writing_chippers is not None:
            for entry in self._writing_chippers:
                entry.close()
        if msg is not None:
            raise SarpyIOError(
                'The NITF file {} image data is not fully written, and the file is potentially corrupt.\n{}'.format(self._file_name, msg))

    # require specific implementations
    def _create_security_tags(self):
        """
        Creates the main NITF security tags object with `CLAS` and `CODE`
        attributes set sensibly.

        It is expected that output from this will be modified as appropriate
        and used to set ONLY specific security tags in `data_extension_headers` or
        elements of `image_segment_headers`.

        If simultaneous modification of all security tags attributes for the entire
        NITF is the goal, then directly modify the value(s) using `security_tags`.

        Returns
        -------
        None
        """

        # self._security_tags = <something>
        raise NotImplementedError

    def _create_image_segment_details(self):
        """
        Create the image segment headers.

        Returns
        -------
        None
        """

        if self._security_tags is None:
            raise ValueError(
                "This NITF has no previously defined security tags, so this method "
                "is being called before the _create_secrity_tags method.")

        # _img_groups, _shapes should be defined here or previously.
        # self._img_details = <something>

    def _create_data_extension_details(self):
        """
        Create the data extension headers.

        Returns
        -------
        None
        """

        if self._security_tags is None:
            raise ValueError(
                "This NITF has no previously defined security tags, so this method "
                "is being called before the _create_secrity_tags method.")

        # self._des_details = <something>

    def _get_nitf_image_segments(self):
        """
        Get the ImageSegments component for the NITF header.

        Returns
        -------
        ImageSegmentsType
        """

        if self._img_details is None:
            return ImageSegmentsType(subhead_sizes=None, item_sizes=None)
        else:
            im_sizes = numpy.zeros((len(self._img_details), ), dtype=numpy.int64)
            subhead_sizes = numpy.zeros((len(self._img_details), ), dtype=numpy.int64)
            for i, details in enumerate(self._img_details):
                subhead_sizes[i] = details.subheader.get_bytes_length()
                im_sizes[i] = details.image_size
            return ImageSegmentsType(subhead_sizes=subhead_sizes, item_sizes=im_sizes)

    def _get_nitf_data_extensions(self):
        """
        Get the DataEXtensions component for the NITF header.

        Returns
        -------
        DataExtensionsType
        """

        if self._des_details is None:
            return DataExtensionsType(subhead_sizes=None, item_sizes=None)
        else:
            des_sizes = numpy.zeros((len(self._des_details), ), dtype=numpy.int64)
            subhead_sizes = numpy.zeros((len(self._des_details), ), dtype=numpy.int64)
            for i, details in enumerate(self._des_details):
                subhead_sizes[i] = details.subheader.get_bytes_length()
                des_sizes[i] = len(details.des_bytes)
            return DataExtensionsType(subhead_sizes=subhead_sizes, item_sizes=des_sizes)

    def _create_nitf_header(self):
        """
        Create the main NITF header.

        Returns
        -------
        None
        """

        if self._img_details is None:
            logger.warning(
                "This NITF has no previously defined image segments,\n\t"
                "or the _create_nitf_header method has been called\n\t"
                "BEFORE the _create_image_segment_headers method.")
        if self._des_details is None:
            logger.warning(
                "This NITF has no previously defined data extensions,\n\t"
                "or the _create_nitf_header method has been called\n\t"
                "BEFORE the _create_data_extension_headers method.")

        # NB: CLEVEL and FL will be corrected in prepare_for_writing method
        self._nitf_header = NITFHeader(
            Security=self.security_tags, CLEVEL=3, OSTAID=self._get_ostaid(),
            FDT=self._get_fdt(), FTITLE=self._get_ftitle(), FL=0,
            ImageSegments=self._get_nitf_image_segments(),
            DataExtensions=self._get_nitf_data_extensions())


#######
# Flexible memmap object for opening compressed NITF image segment

class MemMap(object):
    """
    Spoofing necessary memory map functionality to permit READ ONLY opening of a
    compressed NITF image segment using the PIL interface. This is just a thin
    wrapper around the built-in memmap class which accommodates arbitrary offset.

    Note that the bare minimum of functionality is implemented to permit the
    intended use.
    """

    __slots__ = ('_mem_map', '_file_obj', '_offset_shift')

    def __init__(self, file_obj, length, offset):
        """

        Parameters
        ----------
        file_obj : str|BinaryIO
        length : int
        offset : int
        """

        # length and offset validation
        length = int(length)
        offset = int(offset)
        if length < 0 or offset < 0:
            raise ValueError(
                'length ({}) and offset ({}) must be non-negative integers'.format(length, offset))
        # determine offset and length accommodating allocation block size limitation
        self._offset_shift = (offset % mmap.ALLOCATIONGRANULARITY)
        offset = offset - self._offset_shift
        length = length + self._offset_shift
        # establish the mem map
        if isinstance(file_obj, str):
            self._file_obj = open(file_obj, 'rb')
        else:
            self._file_obj = file_obj
        self._mem_map = mmap.mmap(self._file_obj.fileno(), length, access=mmap.ACCESS_READ, offset=offset)

    def read(self, n):
        return self._mem_map.read(n)

    def tell(self):
        return self._mem_map.tell() - self._offset_shift

    def seek(self, pos, whence=0):
        whence = int(whence)
        pos = int(pos)
        if whence == 0:
            self._mem_map.seek(pos+self._offset_shift, 0)
        else:
            self._mem_map.seek(pos, whence)

    @property
    def closed(self):
        return self._file_obj.closed

    def close(self):
        self._file_obj.close()


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name: Union[str, BinaryIO]) -> Union[None, NITFReader]:
    """
    Tests whether a given file_name corresponds to a nitf file. Returns a
    nitf reader instance, if so.

    Parameters
    ----------
    file_name : str|BinaryIO
        the file_name to check

    Returns
    -------
    None|NITFReader
        `NITFReader` instance if nitf file, `None` otherwise
    """

    try:
        nitf_details = NITFDetails(file_name)
        logger.info('File {} is determined to be a nitf file.'.format(file_name))
        return NITFReader(nitf_details)
    except SarpyIOError:
        # we don't want to catch parsing errors, for now
        return None
