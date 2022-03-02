"""
Module laying out basic functionality for reading and writing NITF files.
This is **intended** to represent base functionality to be extended for
SICD and SIDD capability.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
from typing import Union, List, Tuple, BinaryIO
import re
import mmap
from tempfile import mkstemp
from collections import OrderedDict
import struct
import gc

import numpy

from sarpy.io.general.base import BaseReader, AbstractWriter, SubsetChipper, \
    AggregateChipper, BIPChipper, BIPWriter, BSQChipper, BIRChipper, SarpyIOError
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
from sarpy.io.general.utils import is_file_like
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
    import PIL
    # noinspection PyPackageRequirements
    import PIL.Image
except ImportError:
    PIL = None


logger = logging.getLogger(__name__)

_unhandled_version_text = 'Unhandled NITF version `{}`'


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a nitf file. Returns a
    nitf reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    NITFReader|None
        `NITFReader` instance if nitf file, `None` otherwise
    """

    try:
        nitf_details = NITFDetails(file_name)
        logger.info('File {} is determined to be a nitf file.'.format(file_name))
        return NITFReader(nitf_details)
    except SarpyIOError:
        # we don't want to catch parsing errors, for now
        return None


#####
# A general nitf header interpreter - intended for extension

def extract_image_corners(img_header):
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
    corner_strings = [corner_string[start:stop] for start, stop in zip(range(0, 59, 15), range(15, 74, 15))]
    icps = []
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
    This class allows for somewhat general parsing of the header information in a NITF 2.1 file.
    Experimental support for NITF 2.0 is also included.
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

    def __init__(self, file_object):
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

        # Read the first 9 bytes to verify NITF
        self._file_object.seek(0, os.SEEK_SET)
        try:
            version_info = self._file_object.read(9)
        except Exception:
            raise SarpyIOError('Not a NITF 2.1 file.')
        if not isinstance(version_info, bytes):
            raise ValueError('Input file like object not open in bytes mode.')
        try:
            version_info = version_info.decode('utf-8')
        except Exception as e:
            msg = 'Failed checking potential version with error\n\t{}'.format(e)
            logger.info(msg)
            raise SarpyIOError(msg)

        if version_info[:4] != 'NITF':
            raise SarpyIOError('File {} is not a NITF file.'.format(self._file_name))
        self._nitf_version = version_info[4:]
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
    def _element_offsets(cur_loc, item_array_details):
        # type: (int, Union[_ItemArrayHeaders, None]) -> Tuple[int, Union[None, numpy.ndarray], Union[None, numpy.ndarray], Union[None, numpy.ndarray], Union[None, numpy.ndarray]]
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
    def file_name(self):
        """
        str: the file name. This may not be useful if the input was based on a
        file like object
        """

        return self._file_name

    @property
    def file_object(self):
        """
        BinaryIO: The binary file object
        """

        return self._file_object

    @property
    def nitf_header(self):
        # type: () -> Union[NITFHeader, NITFHeader0]
        """
        NITFHeader: the nitf header object
        """

        return self._nitf_header

    @property
    def img_headers(self):
        """
        The image segment headers.

        Returns
        -------
            None|List[sarpy.io.general.nitf_elements.image.ImageSegmentHeader]
        """

        if self._img_headers is not None:
            return self._img_headers

        self._parse_img_headers()
        return self._img_headers

    @property
    def nitf_version(self):
        """
        str: The NITF version number.
        """

        return self._nitf_version

    def _parse_img_headers(self):
        if self.img_segment_offsets is None or self._img_headers is not None:
            return

        self._img_headers = [self.parse_image_subheader(i) for i in range(self.img_subheader_offsets.size)]

    def _fetch_item(self, name, index, offsets, sizes):
        # type: (str, int, numpy.ndarray, numpy.ndarray) -> bytes
        if index >= offsets.size:
            raise IndexError(
                'There are only {0:d} {1:s}, invalid {1:s} position {2:d}'.format(
                    offsets.size, name, index))
        the_offset = offsets[index]
        the_size = sizes[index]
        self._file_object.seek(int(the_offset), os.SEEK_SET)
        the_item = self._file_object.read(int(the_size))
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
        TextSegmentHeader|TextSegmentHeader0
        """

        th = self.get_text_subheader_bytes(index)
        if self._nitf_version == '02.10':
            return TextSegmentHeader.from_bytes(th, 0)
        elif self._nitf_version == '02.00':
            return TextSegmentHeader0.from_bytes(th, 0)
        else:
            raise ValueError(_unhandled_version_text.format(self.nitf_version))

    def get_graphics_subheader_bytes(self, index):
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

    def get_graphics_bytes(self, index):
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

    def parse_graphics_subheader(self, index):
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

    def get_symbol_subheader_bytes(self, index):
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

    def get_symbol_bytes(self, index):
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

    def parse_symbol_subheader(self, index):
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

    def get_label_subheader_bytes(self, index):
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

    def get_label_bytes(self, index):
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

    def parse_label_subheader(self, index):
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
        DataExtensionHeader|DataExtensionHeader0
        """

        dh = self.get_des_subheader_bytes(index)
        if self.nitf_version == '02.10':
            return DataExtensionHeader.from_bytes(dh, 0)
        elif self.nitf_version == '02.00':
            return DataExtensionHeader0.from_bytes(dh, 0)
        else:
            raise ValueError(_unhandled_version_text.format(self.nitf_version))

    def get_res_subheader_bytes(self, index):
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

    def get_res_bytes(self, index):
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

    def parse_res_subheader(self, index):
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

    def get_headers_json(self):
        """
        Get a json representation of the NITF header elements.

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


#####
# A general nitf reader - intended for extension

def _validate_lookup(lookup_table):
    # type: (numpy.ndarray) -> None
    if not isinstance(lookup_table, numpy.ndarray):
        raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
    if lookup_table.dtype.name != 'uint8':
        raise ValueError('requires a numpy.ndarray of uint8 dtype, got {}'.format(lookup_table.dtype))


def single_lut_conversion(lookup_table):
    """
    This constructs the function to convert data using a single lookup table.

    Parameters
    ----------
    lookup_table : numpy.ndarray

    Returns
    -------
    callable
    """

    _validate_lookup(lookup_table)

    def converter(data):
        if not isinstance(data, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(data)))

        if data.dtype.name not in ['uint8', 'uint16']:
            raise ValueError('requires a numpy.ndarray of uint8 or uint16 dtype, '
                             'got {}'.format(data.dtype.name))

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Requires a three-dimensional numpy.ndarray, '
                             'with single band in the last dimension. Got shape {}'.format(data.shape))
        temp = numpy.reshape(data, (-1, ))
        out = lookup_table[temp]
        if lookup_table.ndim == 2:
            return numpy.reshape(out, (data.shape[0], data.shape[1], lookup_table.shape[1]))
        else:
            return numpy.reshape(out, (data.shape[0], data.shape[1], 1))
    return converter


class NITFReader(BaseReader):
    """
    A reader object for NITF 2.10 container files
    """

    __slots__ = ('_nitf_details', '_cached_files', '_symmetry')

    def __init__(self, nitf_details, reader_type="OTHER", symmetry=(False, False, False)):
        """

        Parameters
        ----------
        nitf_details : NITFDetails|BinaryIO|str
            The NITFDetails object or path to a nitf file.
        reader_type : str
            What type of reader is this, options are "SICD", "SIDD", "CPHD", or "OTHER"
        symmetry : tuple
        """

        self._symmetry = symmetry
        self._cached_files = []
        if isinstance(nitf_details, str) or is_file_like(nitf_details):
            nitf_details = NITFDetails(nitf_details)
        if not isinstance(nitf_details, NITFDetails):
            raise TypeError('The input argument for NITFReader must be a NITFDetails object.')
        self._nitf_details = nitf_details

        # determine image segmentation from image headers
        segments = self._find_segments()
        # construct the chippers
        chippers = []
        for i, segment in enumerate(segments):
            this_chip = self._construct_chipper(segment, i)
            if isinstance(this_chip, list):
                chippers.extend(this_chip)
            else:
                chippers.append(this_chip)
        super(NITFReader, self).__init__(tuple(chippers), reader_type=reader_type)

    @property
    def nitf_details(self):
        """
        NITFDetails: The NITF details object.
        """

        return self._nitf_details

    @property
    def file_name(self):
        return self._nitf_details.file_name

    @property
    def file_object(self):
        """
        BinaryIO: the binary file like object from which we are reading
        """

        return self._nitf_details.file_object

    def _compliance_check(self, index):
        """
        Check that the image segment at `index` can be supported.

        Parameters
        ----------
        index : int

        Returns
        -------
        bool
        """

        img_header = self.nitf_details.img_headers[index]

        if img_header.NBPP not in (8, 16, 32, 64):
            # numpy basically only supports traditional typing
            logger.error(
                'Image segment at index {} has bits per pixel per band {},\n\t'
                'only 8, 16, and 32 are supported.'.format(index, img_header.NBPP))
            return False

        if img_header.is_masked and img_header.is_compressed:
            logger.error(
                'Image segment at index {} is both masked and compressed.\n\t'
                'This is not currently supported.'.format(index))
            return False

        if img_header.IC in ['C0', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'I1']:
            if PIL is None:
                logger.error(
                    'Image segment at index {} has IC value {},\n\t'
                    'and PIL cannot be imported.\n\t'
                    'Currently, compressed image segments require PIL.'.format(
                        index, img_header.IC))
                return False
        return True

    def _extract_chipper_params(self, index):
        """
        Gets the basic chipper parameters for the given image segment.

        Parameters
        ----------
        index : int

        Returns
        -------
        tuple
            out the form (native_dtype, output_dtype, native_bands, output_bands, transform_data)
        """

        img_header = self.nitf_details.img_headers[index]
        nbpp = img_header.NBPP  # previously verified to be one of 8, 16, 32, 64
        bpp = int(nbpp/8)  # bytes per pixel per band
        pvtype = img_header.PVTYPE
        if img_header.ICAT.strip() in ['SAR', 'SARIQ'] and ((len(img_header.Bands) % 2) == 0):
            cont = True
            for i in range(0, len(img_header.Bands), 2):
                cont &= (img_header.Bands[i].ISUBCAT == 'I' and
                         img_header.Bands[i+1].ISUBCAT == 'Q')
            if cont:
                native_bands = len(img_header.Bands)
                output_bands = int(native_bands/2)
                if pvtype == 'SI':
                    return numpy.dtype('>i{}'.format(bpp)), numpy.complex64, native_bands, output_bands, 'COMPLEX'
                elif pvtype == 'R':
                    return numpy.dtype('>f{}'.format(bpp)), numpy.complex64, native_bands, output_bands, 'COMPLEX'
            del cont
        if img_header.IREP.strip() == 'MONO' and img_header.Bands[0].LUTD is not None:
            if not (pvtype == 'INT' and bpp not in [1, 2]):
                raise ValueError(
                    'Got IREP = {} with a LUT, but PVTYPE = {} and '
                    'NBPP = {}'.format(img_header.IREP, pvtype, nbpp))
            lut = img_header.Bands[0].LUTD
            if lut.ndim == 1:
                return numpy.dtype('>u{}'.format(bpp)), numpy.dtype('>u1'), 1, 1, single_lut_conversion(lut)
            else:
                return numpy.dtype('>u{}'.format(bpp)), numpy.dtype('>u1'), 1, lut.shape[0], single_lut_conversion(numpy.transpose(lut))
        if img_header.IREP.strip() == 'RGB/LUT':
            lut = img_header.Bands[0].LUTD
            return numpy.dtype('>u1'), numpy.dtype('>u1'), 1, 3, single_lut_conversion(numpy.transpose(lut))

        if pvtype == 'INT':
            return numpy.dtype('>u{}'.format(bpp)), numpy.dtype('>u{}'.format(bpp)), \
                   len(img_header.Bands), len(img_header.Bands), None
        elif pvtype == 'SI':
            return numpy.dtype('>i{}'.format(bpp)), numpy.dtype('>i{}'.format(bpp)), \
                   len(img_header.Bands), len(img_header.Bands), None
        elif pvtype == 'R':
            return numpy.dtype('>f{}'.format(bpp)), numpy.dtype('>f{}'.format(bpp)), \
                   len(img_header.Bands), len(img_header.Bands), None
        elif pvtype == 'C':
            if bpp not in [8, 16]:
                raise ValueError(
                    'Got PVTYPE = C and NBPP = {} (not 64 or 128), which is unsupported.'.format(nbpp))
            bands = len(img_header.Bands)
            return (
                numpy.dtype('>c{}'.format(bpp)),
                numpy.complex64,
                bands,
                bands,
                'COMPLEX')

    @staticmethod
    def _construct_block_bounds(img_header, row_block_size, column_block_size):
        bounds = []
        block_row_start = 0
        for row_block in range(img_header.NBPC):
            block_row_end = block_row_start + row_block_size
            block_col_start = 0
            for column_block in range(img_header.NBPR):
                block_col_end = block_col_start + column_block_size
                bounds.append((block_row_start, block_row_end, block_col_start, block_col_end))
                block_col_start = block_col_end
            block_row_start = block_row_end
        return bounds

    @staticmethod
    def _get_block_sizes(img_header, this_rows, this_cols):
        # column (horizontal) block details
        column_block_size = this_cols
        if img_header.NBPR != 1:
            column_block_size = img_header.NPPBH
        # row (vertical) block details
        row_block_size = this_rows
        if img_header.NBPC != 1:
            row_block_size = img_header.NPPBV
        return row_block_size, column_block_size

    def _create_band_interleaved_by_pixel_chipper(
            self, img_header, data_offset, this_rows, this_cols, raw_dtype, raw_bands,
            output_dtype, output_bands, transform_data, limit_to_raw_bands):

        if img_header.NBPR == 1 and img_header.NBPC == 1:
            # it's one single block
            return BIPChipper(
                self.file_object, raw_dtype, (this_rows, this_cols), raw_bands, output_bands, output_dtype,
                symmetry=self._symmetry, transform_data=transform_data,
                data_offset=data_offset, limit_to_raw_bands=limit_to_raw_bands)

        if img_header.is_masked:
            # update the offset to skip the mask subheader
            data_offset += img_header.mask_subheader.IMDATOFF
            if img_header.mask_subheader.BMR is not None:
                mask_offsets = img_header.mask_subheader.BMR
            elif img_header.mask_subheader.TMR is not None:
                mask_offsets = img_header.mask_subheader.TMR
                logger.warning(
                    'image segment contains a transparency mask - \n\t'
                    'the transparency value is not currently used.')
            else:
                logger.warning(
                    'image segment is masked, but contains neither \n\t'
                    'transparency mask nor block mask? This is unexpected.')
                mask_offsets = None
        else:
            mask_offsets = None
        exclude_value = 0xFFFFFFFF  # only used if there is a mask...

        # column (horizontal) block details
        column_block_size = this_cols
        if img_header.NBPR != 1:
            column_block_size = img_header.NPPBH

        # row (vertical) block details
        row_block_size = this_rows
        if img_header.NBPC != 1:
            row_block_size = img_header.NPPBV

        # iterate over our blocks and determine bounds and create chippers
        bounds = []
        chippers = []
        # NB: NBPP previously verified to be one of 8, 16, 32, 64
        bytes_per_pixel = int(img_header.NBPP*raw_bands/8)

        # outer loop over rows inner loop over columns
        current_offset = 0
        block_row_end = 0
        block_offset = int(img_header.NPPBH*img_header.NPPBV*bytes_per_pixel)
        enumerated_block = 0
        for vblock in range(img_header.NBPC):
            # define the current row block start/end
            block_row_start = block_row_end
            block_row_end += row_block_size
            block_row_end = min(this_rows, block_row_end)
            block_col_end = 0
            for hblock in range(img_header.NBPR):
                # verify whether to include this block, or not
                if mask_offsets is not None:
                    expected_offset = mask_offsets[0][enumerated_block]
                    if expected_offset == exclude_value:
                        # this is a missing block
                        # advance the block enumeration details and skip
                        block_col_end += column_block_size
                        block_col_end = min(this_cols, block_col_end)
                        enumerated_block += 1
                        continue
                    if expected_offset < current_offset:
                        raise ValueError(
                            'expected offset {}, current offset {}, for block {} = ({}, {})'.format(
                                expected_offset, current_offset, enumerated_block, vblock, hblock))
                    elif expected_offset > current_offset:
                        current_offset = expected_offset

                # define the current column block
                block_col_start = block_col_end
                block_col_end += column_block_size
                block_col_end = min(this_cols, block_col_end)
                # store our bounds and offset
                bounds.append((block_row_start, block_row_end, block_col_start, block_col_end))
                # create chipper
                chip_rows = block_row_end - block_row_start
                chip_cols = column_block_size
                if block_col_end == block_col_start + column_block_size:
                    chippers.append(
                        BIPChipper(
                            self.file_object, raw_dtype, (chip_rows, chip_cols),
                            raw_bands, output_bands, output_dtype,
                            symmetry=self._symmetry, transform_data=transform_data,
                            data_offset=int(current_offset+data_offset), limit_to_raw_bands=limit_to_raw_bands))
                else:
                    c_row_start, c_row_end = 0, chip_rows
                    c_col_start, c_col_end = 0, block_col_end - block_col_start
                    if self._symmetry[1]:
                        c_col_start = column_block_size - c_col_end
                        c_col_end = column_block_size - c_col_start
                    if self._symmetry[2]:
                        c_row_start, c_row_end, c_col_start, c_col_end = \
                            c_col_start, c_col_end, c_row_start, c_row_end

                    p_chipper = BIPChipper(
                        self.file_object, raw_dtype, (chip_rows, chip_cols),
                        raw_bands, output_bands, output_dtype,
                        symmetry=self._symmetry, transform_data=transform_data,
                        data_offset=int(current_offset+data_offset), limit_to_raw_bands=limit_to_raw_bands)
                    chippers.append(SubsetChipper(p_chipper, (c_row_start, c_row_end), (c_col_start, c_col_end)))

                # update the block and offset
                enumerated_block += 1
                current_offset += block_offset  # NB: this may contain pad pixels

        bounds = numpy.array(bounds, dtype=numpy.int64)
        total_rows = bounds[-1, 1]
        total_cols = bounds[-1, 3]
        if total_rows != this_rows or total_cols != this_cols:
            raise ValueError('Got unexpected chipper construction')
        if self._symmetry[0]:
            t_bounds = bounds.copy()
            bounds[:, 0] = total_rows - t_bounds[:, 1]
            bounds[:, 1] = total_rows - t_bounds[:, 0]
        if self._symmetry[1]:
            t_bounds = bounds.copy()
            bounds[:, 2] = total_cols - t_bounds[:, 3]
            bounds[:, 3] = total_cols - t_bounds[:, 2]
        if self._symmetry[2]:
            t_bounds = bounds.copy()
            bounds[:, :2] = t_bounds[:, 2:]
            bounds[:, 2:] = t_bounds[:, :2]
        return AggregateChipper(bounds, output_dtype, chippers, output_bands=output_bands)

    def _create_band_sequential_chipper(
            self, img_header, data_offset, this_rows, this_cols, raw_dtype, raw_bands,
            output_dtype, output_bands, transform_data, limit_to_raw_bands):

        if img_header.NBPR == 1 and img_header.NBPC == 1:
            # it's one single block
            chippers = []
            current_offset = 0
            block_offset = int(this_rows*this_cols*img_header.NBPP)

            for i in range(raw_bands):
                chippers.append(BIPChipper(
                    self.file_object, raw_dtype, (this_rows, this_cols), 1, 1, raw_dtype,
                    symmetry=self._symmetry, transform_data=None,
                    data_offset=data_offset+current_offset, limit_to_raw_bands=None))
                current_offset += block_offset
            return BSQChipper(
                chippers, output_dtype, transform_data=transform_data,
                limit_to_raw_bands=limit_to_raw_bands)

        if img_header.is_masked:
            # update the offset to skip the mask subheader
            data_offset += img_header.mask_subheader.IMDATOFF
            if img_header.mask_subheader.BMR is not None:
                mask_offsets = img_header.mask_subheader.BMR
            elif img_header.mask_subheader.TMR is not None:
                mask_offsets = img_header.mask_subheader.TMR
                logger.warning(
                    'image segment contains a transparency mask - \n\t'
                    'the transparency value is not currently used.')
            else:
                logger.warning(
                    'image segment is masked, but contains neither \n\t'
                    'transparency mask nor block mask? This is unexpected.')
                mask_offsets = None
        else:
            mask_offsets = None
        exclude_value = 0xFFFFFFFF  # only used if there is a mask...

        row_block_size, column_block_size = self._get_block_sizes(img_header, this_rows, this_cols)
        block_offset = int(row_block_size*column_block_size*img_header.NBPP/8)  # verified to be integer already
        num_blocks = img_header.NBPC*img_header.NBPR
        bip_chippers = [[] for _ in range(num_blocks)]

        if mask_offsets is None:
            offsets = block_offset*numpy.arange(raw_bands*num_blocks, dtype=numpy.uint64)
            offsets = numpy.reshape(offsets, (raw_bands, num_blocks))
        else:
            offsets = None

        # construct bounds for the chippers
        bounds = self._construct_block_bounds(img_header, row_block_size, column_block_size)

        for band_number in range(raw_bands):
            for block_number in range(num_blocks):
                if mask_offsets is not None:
                    this_offset = mask_offsets[band_number, block_number]
                    if this_offset == exclude_value:
                        continue  # skip this block
                else:
                    this_offset = offsets[band_number, block_number]
                block_bounds = bounds[block_number]
                block_rows = int(block_bounds[1] - block_bounds[0])
                block_cols = int(block_bounds[3] - block_bounds[2])
                bip_chippers[block_number].append(
                    BIPChipper(
                        self.file_object, raw_dtype, (block_rows, block_cols), 1, 1, raw_dtype,
                        symmetry=self._symmetry, transform_data=None, data_offset=int(data_offset+this_offset),
                        limit_to_raw_bands=None))

        bsq_chippers = []
        bsq_bounds = []
        for i, entry in enumerate(bip_chippers):
            if len(entry) == 0:
                continue  # it was masked
            elif len(entry) == raw_bands:
                bsq_chippers.append(
                    BSQChipper(
                        entry, output_dtype, transform_data=transform_data, limit_to_raw_bands=limit_to_raw_bands))
                bsq_bounds.append(bounds[i])
            else:
                raise ValueError(
                    'Failed by partially constructing {} of {} '
                    'chippers for block {}'.format(len(entry), raw_bands, i))
        bsq_bounds = numpy.array(bsq_bounds, dtype='int64')
        return AggregateChipper(bsq_bounds, output_dtype, bsq_chippers, output_bands=output_bands)

    def _create_band_interleaved_by_block_chipper(
            self, img_header, data_offset, this_rows, this_cols, raw_dtype, raw_bands,
            output_dtype, output_bands, transform_data, limit_to_raw_bands):

        if img_header.NBPR == 1 and img_header.NBPC == 1:
            # it's one single block
            chippers = []
            current_offset = 0
            block_offset = int(this_rows*this_cols*img_header.NBPP)

            for i in range(raw_bands):
                chippers.append(BIPChipper(
                    self.file_object, raw_dtype, (this_rows, this_cols), 1, 1, raw_dtype,
                    symmetry=self._symmetry, transform_data=None,
                    data_offset=data_offset+current_offset, limit_to_raw_bands=None))
                current_offset += block_offset
            return BSQChipper(
                chippers, output_dtype, transform_data=transform_data, limit_to_raw_bands=limit_to_raw_bands)

        if img_header.is_masked:
            # update the offset to skip the mask subheader
            data_offset += img_header.mask_subheader.IMDATOFF
            if img_header.mask_subheader.BMR is not None:
                mask_offsets = img_header.mask_subheader.BMR
            elif img_header.mask_subheader.TMR is not None:
                mask_offsets = img_header.mask_subheader.TMR
                logger.warning(
                    'image segment contains a transparency mask - \n\t'
                    'the transparency value is not currently used.')
            else:
                logger.warning(
                    'image segment is masked, but contains neither \n\t'
                    'transparency mask nor block mask? This is unexpected.')
                mask_offsets = None
        else:
            mask_offsets = None
        exclude_value = 0xFFFFFFFF  # only used if there is a mask...

        row_block_size, column_block_size = self._get_block_sizes(img_header, this_rows, this_cols)
        band_offset = int(row_block_size*column_block_size*img_header.NBPP/8)  # verified to be integer already

        num_blocks = img_header.NBPC*img_header.NBPR
        bsq_chippers = [None for _ in range(num_blocks)]  # type: List[Union[None, BSQChipper]]

        # construct bounds for the chippers
        bounds = self._construct_block_bounds(img_header, row_block_size, column_block_size)

        for block_number in range(num_blocks):
            if mask_offsets is not None:
                this_offset = mask_offsets[0, block_number]
                if this_offset == exclude_value:
                    continue  # skip this block

            bip_chippers = []
            for band_number in range(raw_bands):
                block_bounds = bounds[block_number]
                block_rows = int(block_bounds[1] - block_bounds[0])
                block_cols = int(block_bounds[3] - block_bounds[2])
                this_offset = band_number*band_offset
                bip_chippers.append(
                    BIPChipper(
                        self.file_object, raw_dtype, (block_rows, block_cols), 1, 1, raw_dtype,
                        symmetry=self._symmetry, transform_data=None, data_offset=data_offset+this_offset,
                        limit_to_raw_bands=None))
            bsq_chippers[block_number] = BSQChipper(
                bip_chippers, output_dtype, transform_data=transform_data, limit_to_raw_bands=limit_to_raw_bands)

        populated_chippers = []
        populated_bounds = []
        for i, entry in enumerate(bsq_chippers):
            if entry is None:
                continue
            else:
                populated_chippers.append(entry)
                populated_bounds.append(bounds[i])
        populated_bounds = numpy.array(populated_bounds, dtype='int64')
        return AggregateChipper(populated_bounds, output_dtype, populated_chippers, output_bands=output_bands)

    def _create_band_interleaved_by_row_chipper(
            self, img_header, data_offset, this_rows, this_cols, raw_dtype, raw_bands,
            output_dtype, output_bands, transform_data, limit_to_raw_bands):

        if img_header.NBPR == 1 and img_header.NBPC == 1:
            # it's one single block
            return BIRChipper(
                self.file_object, raw_dtype, (this_rows, this_cols), raw_bands, output_bands, output_dtype,
                symmetry=self._symmetry, transform_data=transform_data,
                data_offset=data_offset, limit_to_raw_bands=limit_to_raw_bands)

        if img_header.is_masked:
            # update the offset to skip the mask subheader
            data_offset += img_header.mask_subheader.IMDATOFF
            if img_header.mask_subheader.BMR is not None:
                mask_offsets = img_header.mask_subheader.BMR
            elif img_header.mask_subheader.TMR is not None:
                mask_offsets = img_header.mask_subheader.TMR
                logger.warning(
                    'image segment contains a transparency mask - \n\t'
                    'the transparency value is not currently used.')
            else:
                logger.warning(
                    'image segment is masked, but contains neither \n\t'
                    'transparency mask nor block mask? This is unexpected.')
                mask_offsets = None
        else:
            mask_offsets = None
        exclude_value = 0xFFFFFFFF  # only used if there is a mask...

        # column (horizontal) block details
        column_block_size = this_cols
        if img_header.NBPR != 1:
            column_block_size = img_header.NPPBH

        # row (vertical) block details
        row_block_size = this_rows
        if img_header.NBPC != 1:
            row_block_size = img_header.NPPBV

        # iterate over our blocks and determine bounds and create chippers
        bounds = []
        chippers = []
        # NB: NBPP previously verified to be one of 8, 16, 32, 64
        bytes_per_pixel = int(img_header.NBPP*raw_bands/8)

        # outer loop over rows inner loop over columns
        current_offset = 0
        block_row_end = 0
        block_offset = int(img_header.NPPBH*img_header.NPPBV*bytes_per_pixel)
        enumerated_block = 0
        for vblock in range(img_header.NBPC):
            # define the current row block start/end
            block_row_start = block_row_end
            block_row_end += row_block_size
            block_row_end = min(this_rows, block_row_end)
            block_col_end = 0
            for hblock in range(img_header.NBPR):
                # verify whether to include this block, or not
                if mask_offsets is not None:
                    expected_offset = mask_offsets[0][enumerated_block]
                    if expected_offset == exclude_value:
                        # this is a missing block
                        # advance the block enumeration details and skip
                        block_col_end += column_block_size
                        block_col_end = min(this_cols, block_col_end)
                        enumerated_block += 1
                        continue
                    if expected_offset < current_offset:
                        raise ValueError(
                            'In the masked image, the expected offset is {} and the current offset is {},\n'
                            'for block {} = ({}, {})'.format(
                                expected_offset, current_offset, enumerated_block, vblock, hblock))
                    if expected_offset > current_offset:
                        # don't read masked pixels...
                        current_offset = expected_offset

                # define the current column block
                block_col_start = block_col_end
                block_col_end += column_block_size
                block_col_end = min(this_cols, block_col_end)
                # store our bounds and offset
                bounds.append((block_row_start, block_row_end, block_col_start, block_col_end))
                # create chipper
                chip_rows = block_row_end - block_row_start
                chip_cols = column_block_size
                if block_col_end == block_col_start + column_block_size:
                    chippers.append(
                        BIRChipper(
                            self.file_object, raw_dtype, (chip_rows, chip_cols),
                            raw_bands, output_bands, output_dtype,
                            symmetry=self._symmetry, transform_data=transform_data,
                            data_offset=int(current_offset+data_offset), limit_to_raw_bands=limit_to_raw_bands))
                else:
                    c_row_start, c_row_end = 0, chip_rows
                    c_col_start, c_col_end = 0, block_col_end - block_col_start
                    if self._symmetry[1]:
                        c_col_start = column_block_size - c_col_end
                        c_col_end = column_block_size - c_col_start
                    if self._symmetry[2]:
                        c_row_start, c_row_end, c_col_start, c_col_end = \
                            c_col_start, c_col_end, c_row_start, c_row_end

                    p_chipper = BIRChipper(
                        self.file_object, raw_dtype, (chip_rows, chip_cols),
                        raw_bands, output_bands, output_dtype,
                        symmetry=self._symmetry, transform_data=transform_data,
                        data_offset=int(current_offset+data_offset), limit_to_raw_bands=limit_to_raw_bands)
                    chippers.append(SubsetChipper(p_chipper, (c_row_start, c_row_end), (c_col_start, c_col_end)))

                # update the block and offset
                enumerated_block += 1
                current_offset += block_offset  # NB: this may contain pad pixels

        bounds = numpy.array(bounds, dtype=numpy.int64)
        total_rows = bounds[-1, 1]
        total_cols = bounds[-1, 3]
        if total_rows != this_rows or total_cols != this_cols:
            raise ValueError('Got unexpected chipper construction')
        if self._symmetry[0]:
            t_bounds = bounds.copy()
            bounds[:, 0] = total_rows - t_bounds[:, 1]
            bounds[:, 1] = total_rows - t_bounds[:, 0]
        if self._symmetry[1]:
            t_bounds = bounds.copy()
            bounds[:, 2] = total_cols - t_bounds[:, 3]
            bounds[:, 3] = total_cols - t_bounds[:, 2]
        if self._symmetry[2]:
            t_bounds = bounds.copy()
            bounds[:, :2] = t_bounds[:, 2:]
            bounds[:, 2:] = t_bounds[:, :2]
        return AggregateChipper(bounds, output_dtype, chippers, output_bands=output_bands)

    def _define_chipper(self, index, raw_dtype=None, raw_bands=None, transform_data=None,
                        output_bands=None, output_dtype=None, limit_to_raw_bands=None):
        """
        Gets the chipper for the given image segment.

        Parameters
        ----------
        index : int
            The index of the image segment.
        raw_dtype : None|str|numpy.dtype|numpy.number
            The underlying data type of the image segment.
        raw_bands : None|int
            The number of bands of the image segment.
        transform_data : None|str|callable
            The transform_data parameter for the image segment chipper.
        output_bands : None|int
            The number of output bands from the chipper, after transform_data application.
        output_dtype : None|str|numpy.dtype|numpy.number
            The output data type from the chipper.
        limit_to_raw_bands : None|int|list|tuple|numpy.ndarray
            The parameter for limiting bands for the chipper.

        Returns
        -------
        BIPChipper|AggregateChipper
        """

        def handle_compressed():
            if PIL is None:
                raise ValueError('handling image segments with compression require PIL.')

            # extract the image and dump out to a flat file
            image_offset = self.nitf_details.img_segment_offsets[index]
            image_size = self.nitf_details.nitf_header.ImageSegments.item_sizes[index]
            our_memmap = MemMap(self.file_object, image_size, image_offset)
            img = PIL.Image.open(our_memmap)  # this is a lazy operation
            # dump the extracted image data out to a temp file
            fi, path_name = mkstemp(suffix='.sarpy_cache', text=False)
            logger.warning(
                'Naively trying to use PIL to decompress image segment index {} of {}\n\t'
                'to flat file {}.\n\t'
                'The created cache file should be safely deleted,\n\t'
                'with the possible exception of fatal execution error.\n\t'
                'This is likely to not work unless IMODE ({} here) is `P` and\n\t'
                'the image is a single block.'.format(self.file_name, index, path_name, img_header.IMODE))
            data = numpy.asarray(img)  # create our numpy array from the PIL Image
            if data.shape[:2] != (this_rows, this_cols):
                raise ValueError(
                    'Naively decompressed data of shape {}, but expected ({}, {}, {}). '
                    'This is potentially due to image blocks.'.format(data.shape, this_rows, this_cols, raw_bands))
            mem_map = numpy.memmap(path_name, dtype=data.dtype, mode='w+', offset=0, shape=data.shape)
            mem_map[:] = data
            # clean up this memmap and file overhead
            del mem_map
            os.close(fi)
            self._cached_files.append(path_name)

            return BIPChipper(
                path_name, raw_dtype, (this_rows, this_cols), raw_bands, output_bands, output_dtype,
                symmetry=self._symmetry, transform_data=transform_data, data_offset=0,
                limit_to_raw_bands=limit_to_raw_bands)

        # verify that this image segment is viable
        if not self._compliance_check(index):
            raise ValueError(
                'It is not viable to construct a chipper for the image segment at '
                'index {}'.format(index))

        # define fundamental chipper parameters
        img_header = self.nitf_details.img_headers[index]
        this_raw_dtype, this_output_dtype, this_raw_bands, this_output_bands, \
            this_transform_data = self._extract_chipper_params(index)

        data_offset = self.nitf_details.img_segment_offsets[index]
        # determine basic facts
        this_rows = img_header.NROWS
        this_cols = img_header.NCOLS

        if raw_dtype is None:
            raw_dtype = this_raw_dtype
        if output_dtype is None:
            output_dtype = this_output_dtype
        if raw_bands is None:
            raw_bands = this_raw_bands
        if output_bands is None:
            output_bands = this_output_bands
        if transform_data is None:
            transform_data = this_transform_data
        # define the chipper
        if img_header.IC in ['NC', 'NM']:
            if img_header.IMODE == 'P':
                return self._create_band_interleaved_by_pixel_chipper(
                    img_header, data_offset, this_rows, this_cols, raw_dtype, raw_bands,
                    output_dtype, output_bands, transform_data, limit_to_raw_bands)
            elif img_header.IMODE == 'B':
                return self._create_band_interleaved_by_block_chipper(
                    img_header, data_offset, this_rows, this_cols, raw_dtype, raw_bands,
                    output_dtype, output_bands, transform_data, limit_to_raw_bands)
            elif img_header.IMODE == 'S':
                return self._create_band_sequential_chipper(
                    img_header, data_offset, this_rows, this_cols, raw_dtype, raw_bands,
                    output_dtype, output_bands, transform_data, limit_to_raw_bands)
            elif img_header.IMODE == 'R':
                return self._create_band_interleaved_by_row_chipper(
                    img_header, data_offset, this_rows, this_cols, raw_dtype, raw_bands,
                    output_dtype, output_bands, transform_data, limit_to_raw_bands)
            else:
                raise ValueError('Unsupported IMODE {}'.format(img_header.IMODE))
        elif img_header.IC in ['M1', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']:
            raise ValueError('Compressed and masked image segments not currently supported.')
        elif img_header.IC in ['C0', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'I1']:
            return handle_compressed()
        else:
            raise ValueError('Got unhandled IC code {}'.format(img_header.IC))

    def _get_chipper_partitioning(self, segment, rows, cols):
        """
        Construct the chipper partitioning for the given composite image.

        Parameters
        ----------
        segment : List[int]
            The list of NITF image segments indices which are pieced together
            into a single image.
        rows : int
            The number of rows in composite image.
        cols : int
            The number of cols in the composite image.

        Returns
        -------
        numpy.ndarray
            The chipper partitioning for bounds, of the form
            `[row start, row end, column start, column end]`.
        """

        if len(segment) == 1:
            # nothing to really be done
            return numpy.array([0, rows, 0, cols], dtype=numpy.int64)

        bounds = []
        p_row_start, p_row_end, p_col_start, p_col_end = None, None, None, None
        for i, index in enumerate(segment):
            # get this image subheader
            img_header = self.nitf_details.img_headers[index]

            # get the bytes offset for this nitf image segment
            this_rows, this_cols = img_header.NROWS, img_header.NCOLS
            if this_rows > rows or this_cols > cols:
                raise ValueError(
                    'NITF image segment at index {} has size ({}, {}), and cannot be part of an image of size '
                    '({}, {})'.format(index, this_rows, this_cols, rows, cols))

            # determine where this image segment fits in the overall image
            if i == 0:
                # establish the beginning
                cur_row_start, cur_row_end = 0, this_rows
                cur_col_start, cur_col_end = 0, this_cols
            elif p_col_end < cols:
                if this_rows != (p_row_end - p_row_start):
                    raise ValueError(
                        'Cannot stack a NITF image of size ({}, {}) next to a NITF image of size '
                        '({}, {})'.format(this_rows, this_cols, p_row_end-p_col_end, p_col_end-p_col_start))
                cur_row_start, cur_row_end = p_row_start, p_row_end
                cur_col_start, cur_col_end = p_col_end, p_col_end + this_cols
                if cur_col_end > cols:
                    raise ValueError('Failed at horizontal NITF image segment assembly.')
            elif p_col_end == cols:
                # start a new vertical section
                cur_row_start, cur_row_end = p_row_end, p_row_end + this_rows
                cur_col_start, cur_col_end = 0, this_cols
                if cur_row_end > rows:
                    raise ValueError('Failed at vertical NITF image segment assembly.')
            else:
                raise ValueError('Got unexpected situation in NITF image assembly.')
            bounds.append((cur_row_start, cur_row_end, cur_col_start, cur_col_end))
            p_row_start, p_row_end, p_col_start, p_col_end = cur_row_start, cur_row_end, cur_col_start, cur_col_end
        return numpy.array(bounds, dtype=numpy.int64)

    def _find_segments(self):
        """
        Determine the image segment collections.

        Returns
        -------
        List[List[int]]
        """

        # the default implementation is just to allow every viable segment
        segments = []
        for i, img_segment in enumerate(self.nitf_details.img_headers):
            if self._compliance_check(i):
                segments.append([i, ])
        return segments

    def _construct_chipper(self, segment, index):
        """
        Construct the appropriate chipper given the list of image
        segment indices.

        Parameters
        ----------
        segment : List[int]
            The list of NITF image segments indices which are pieced together
            into a single image.
        index : int
            The index of the composite image, in the composite image collection.
            For a SICD, there will only be one composite image, but there may be
            more for other NITF uses.

        Returns
        -------
        sarpy.io.general.base.BaseChipper|List[sarpy.io.general.base.BaseChipper]
        """

        # this default behavior should be overridden for SICD/SIDD
        return self._define_chipper(segment[0])

    def __del__(self):
        """
        Clean up any cached files.

        Returns
        -------
        None
        """

        if not hasattr(self, '_cached_files'):
            return

        del self._chipper
        if callable(gc.collect):
            gc.collect()

        for fil in self._cached_files:
            if os.path.exists(fil):
                # noinspection PyBroadException
                try:
                    os.remove(fil)
                    logger.info('Deleted cached file {}'.format(fil))
                except Exception:
                    logger.error(
                        'Error in attempt to delete cached file {}.\n\t'
                        'Manually delete this file'.format(fil), exc_info=True)


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
            logger.error('A total of {} pixels have been written,\n\t'
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


class NITFWriter(AbstractWriter):
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
