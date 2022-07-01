"""
Module laying out basic functionality for reading and writing NITF files.

Updated extensively in version 1.3.0.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
from typing import Union, List, Tuple, BinaryIO, Sequence, Optional
from tempfile import mkstemp
from collections import OrderedDict
import struct
from io import BytesIO

import numpy

from sarpy.io.general.base import SarpyIOError, BaseReader, BaseWriter
from sarpy.io.general.format_function import FormatFunction, ComplexFormatFunction, \
    SingleLUTFormatFunction
from sarpy.io.general.data_segment import DataSegment, BandAggregateSegment, \
    BlockAggregateSegment, SubsetSegment, NumpyArraySegment, NumpyMemmapSegment, \
    FileReadDataSegment

# noinspection PyProtectedMember
from sarpy.io.general.nitf_elements.nitf_head import NITFHeader, NITFHeader0, \
    ImageSegmentsType, GraphicsSegmentsType, TextSegmentsType, \
    DataExtensionsType, ReservedExtensionsType, _ItemArrayHeaders
from sarpy.io.general.nitf_elements.text import TextSegmentHeader, TextSegmentHeader0
from sarpy.io.general.nitf_elements.graphics import GraphicsSegmentHeader
from sarpy.io.general.nitf_elements.symbol import SymbolSegmentHeader
from sarpy.io.general.nitf_elements.label import LabelSegmentHeader
from sarpy.io.general.nitf_elements.res import ReservedExtensionHeader, ReservedExtensionHeader0
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, ImageSegmentHeader0, MaskSubheader
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, DataExtensionHeader0
from sarpy.io.general.utils import is_file_like, is_nitf, is_real_file

from sarpy.io.complex.sicd_elements.blocks import LatLonType
from sarpy.geometry.geocoords import ecf_to_geodetic, geodetic_to_ecf
from sarpy.geometry.latlon import num as lat_lon_parser

try:
    # noinspection PyPackageRequirements
    import pyproj
except ImportError:
    pyproj = None

try:
    # noinspection PyPackageRequirements
    from PIL import Image as PIL_Image
    PIL_Image.MAX_IMAGE_PIXELS = None  # get rid of decompression bomb checking
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
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
        next_location = end_block + 2
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


def _construct_block_bounds(
        image_header: Union[ImageSegmentHeader, ImageSegmentHeader0]) -> List[Tuple[int, int, int, int]]:
    """
    Construct the bounds for the blocking definition in row/column space for
    the image segment.

    Note that this includes potential pad pixels, since NITF requires that
    each block is the same size.

    Parameters
    ----------
    image_header : ImageSegmentHeader|ImageSegmentHeader

    Returns
    -------
    List[Tuple[int, int, int, int]]
        This is a list of the form `(row start, row end, column start, column end)`.
    """

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


def _get_dtype(
        image_header: Union[ImageSegmentHeader, ImageSegmentHeader0]
        ) -> Tuple[numpy.dtype, numpy.dtype, int, Optional[str], Optional[numpy.ndarray]]:
    """
    Gets the information necessary for constructing the format function applicable
    to the given image segment.

    Parameters
    ----------
    image_header : ImageSegmentHeader|ImageSegmentHeader

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
                    'Image segment appears to be complex of order `{}`, \n\t'
                    'but PVTYPE is `{}`'.format(order, pvtype))
        if order in ['MP', 'PM']:
            if pvtype not in ['INT', 'R']:
                raise ValueError(
                    'Image segment appears to be complex of order `{}`, \n\t'
                    'but PVTYPE is `{}`'.format(order, pvtype))
        return order

    def get_lut_info() -> Optional[numpy.ndarray]:
        bands = image_header.Bands
        if len(bands) > 1:
            for band in bands:
                if band.LUTD is not None:
                    raise ValueError('There are multiple bands with LUT.')

        # TODO: this isn't really right - handle most significant/least significant nonsense
        lut = bands[0].LUTD
        if lut is None:
            return None
        if lut.ndim == 1:
            return lut
        elif lut.ndim == 2:
            return numpy.transpose(lut)
        else:
            raise ValueError('Got lut of shape `{}`'.format(lut.shape))

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
    if lut is not None:
        formatted_dtype = lut.dtype
        formatted_bands = 1 if lut.ndim == 1 else lut.shape[1]

    return raw_dtype, formatted_dtype, formatted_bands, complex_order, lut


def _get_format_function(
        raw_dtype: numpy.dtype,
        complex_order: Optional[str],
        lut: Optional[numpy.ndarray],
        band_dimension: int) -> Optional[FormatFunction]:
    """
    Gets the format function for use in a data segment.

    Parameters
    ----------
    raw_dtype : numpy.dtype
    complex_order : None|str
    lut : None|numpy.ndarray
    band_dimension : int

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


def _verify_image_segment_compatibility(
        img0: Union[ImageSegmentHeader, ImageSegmentHeader0],
        img1: Union[ImageSegmentHeader, ImageSegmentHeader0]) -> bool:
    """
    Verify that the image segments are compatible from the data formatting
    perspective.

    Parameters
    ----------
    img0 : ImageSegmentHeader
    img1 : ImageSegmentHeader

    Returns
    -------
    bool
    """

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

    raw_dtype0, _, form_band0, comp_order0, lut0 = _get_dtype(img0)
    raw_dtype1, _, form_band1, comp_order1, lut1 = _get_dtype(img1)
    if raw_dtype0 != raw_dtype1:
        return False
    if form_band0 != form_band1:
        return False

    if (comp_order0 is None and comp_order1 is not None) or \
            (comp_order0 is not None and comp_order1 is None):
        return False
    elif comp_order0 is not None and comp_order1 is not None and \
            (comp_order0 != comp_order1):
        return False

    if (lut0 is None and lut1 is not None) or (lut0 is not None and lut1 is None):
        return False
    elif lut0 is not None and lut1 is not None and numpy.any(lut0 != lut1):
        return False

    return True


def _correctly_order_image_segment_collection(
            image_headers: Sequence[Union[ImageSegmentHeader, ImageSegmentHeader0]]) -> Tuple[int, ...]:
    """
    Determines the proper order, based on IALVL, for a collection of entries
    which will be assembled into a composite image.

    Parameters
    ----------
    image_headers : Sequence[ImageSegmentHeader]

    Returns
    -------
    Tuple[int, ...]

    Raises
    ------
    ValueError
        If incompatible IALVL values collection
    """

    collection = [(entry.IALVL, orig_index) for orig_index, entry in enumerate(image_headers)]
    collection = sorted(collection, key=lambda x: x[0])   # (stable) order by IALVL

    if all(entry[0] == 0 for entry in collection):
        # all IALVL is 0, and order doesn't matter
        return tuple(range(len(image_headers)))
    if all(entry0[0]+1 == entry1[0] for entry0, entry1 in zip(collection[:-1], collection[1:])):
        # ordered, uninterupted sequence of IALVL values
        return tuple(entry[1] for entry in collection)

    raise ValueError(
        'Collection of (IALVL, image segment index) has\n\t'
        'neither all IALVL == 0, or an uninterrupted sequence of IALVL values.\n\t'
        'See {}'.format(collection))


def _get_collection_element_coordinate_limits(
        image_headers: Sequence[Union[ImageSegmentHeader, ImageSegmentHeader0]],
        return_clevel: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, int]]:
    """
    For the given collection of image segments, get the relative coordinate
    scheme of the form `[[start_row, end_row, start_column, end_column]]`.

    This relies on inspection of `IALVL` and `ILOC` values for this
    collection of image segments.

    Parameters
    ----------
    image_headers : Sequence[ImageSegmentHeader]
    return_clevel : bool
        Also calculate and return the clevel for this?

    Returns
    -------
    block_definition: numpy.ndarray
        of the form `[[start_row, end_row, start_column, end_column]]`.
    clevel: int
        The CLEVEL for this common coordinate system, only returned if
        `return_clevel=True`
    """

    the_indices = _correctly_order_image_segment_collection(image_headers)

    block_definition = numpy.empty((len(the_indices), 4), dtype='int64')
    for i, image_ind in enumerate(the_indices):
        img_header = image_headers[image_ind]
        rows = img_header.NROWS
        cols = img_header.NCOLS
        iloc = img_header.ILOC
        if img_header.IALVL == 0 or i == 0:
            previous_indices = numpy.zeros((4, ), dtype='int64')
        else:
            previous_indices = block_definition[i-1, :]
        rel_row_start, rel_col_start = int(iloc[:5]), int(iloc[5:])
        abs_row_start = rel_row_start + previous_indices[0]
        abs_col_start = rel_col_start + previous_indices[2]
        block_definition[i, :] = (abs_row_start, abs_row_start + rows, abs_col_start, abs_col_start + cols)

    # now, renormalize the coordinate system to be sensible
    min_row = numpy.min(block_definition[:, 0])
    min_col = numpy.min(block_definition[:, 2])
    block_definition[:, 0:2:1] -= min_row
    block_definition[:, 2:4:1] -= min_col

    if return_clevel:
        dim_size = numpy.max(block_definition)
        if dim_size <= 2048:
            clevel = 3
        elif dim_size <= 8192:
            clevel = 5
        elif dim_size <= 65536:
            clevel = 6
        else:
            clevel = 7
        return block_definition, clevel

    return block_definition


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

        return self._fetch_item(
            'image subheader',
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

    def get_image_bytes(self, index: int) -> bytes:
        """
        Fetches the image bytes at the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        bytes
        """

        return self._fetch_item(
            'image data',
            index,
            self.img_segment_offsets,
            self._nitf_header.ImageSegments.item_sizes)

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

        return self._fetch_item(
            'text subheader',
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

        return self._fetch_item(
            'text segment',
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
            return self._fetch_item(
                'graphics subheader',
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
            return self._fetch_item(
                'graphics segment',
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
            return self._fetch_item(
                'symbol subheader',
                index,
                self.symbol_subheader_offsets,
                self._nitf_header.SymbolSegments.subhead_sizes)
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
            return self._fetch_item(
                'symbol segment',
                index,
                self.symbol_segment_offsets,
                self._nitf_header.SymbolSegments.item_sizes)
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
            return self._fetch_item(
                'label subheader',
                index,
                self.label_subheader_offsets,
                self._nitf_header.LabelSegments.subhead_sizes)
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
            return self._fetch_item(
                'label segment',
                index,
                self.label_segment_offsets,
                self._nitf_header.LabelSegments.item_sizes)
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

        return self._fetch_item(
            'des subheader',
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

        return self._fetch_item(
            'des',
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

        return self._fetch_item(
            'res subheader',
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

        return self._fetch_item(
            'res',
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


class NITFReader(BaseReader):
    """
    A reader implementation based around array-type image data fetching for
    NITF 2.0 or 2.1 files.

    **Significantly revised in version 1.3.0** to accommodate the new data segment
    paradigm. General NITF support is improved from previous version, but there
    remain unsupported edge cases.
    """

    _maximum_number_of_images = None
    unsupported_compressions = ('I1', 'C1', 'C4', 'C6', 'C7', 'M1', 'M4', 'M6', 'M7')

    __slots__ = (
        '_nitf_details', '_unsupported_segments', '_image_segment_collections',
        '_reverse_axes', '_transpose_axes', '_image_segment_data_segments')

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

        self._image_segment_data_segments = {}
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
                'which is not supported.'.format(index, img_header.IC))
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
        image_header = self.get_image_header(image_segment_index)
        # noinspection PyTypeChecker
        return _construct_block_bounds(image_header)

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
            self,
            image_segment_index: int) -> Tuple[numpy.dtype, numpy.dtype, int, Optional[str], Optional[numpy.ndarray]]:
        image_header = self.get_image_header(image_segment_index)
        return _get_dtype(image_header)

    def _get_transpose(self, formatted_bands: int) -> Optional[Tuple[int, ...]]:
        if self._transpose_axes is None:
            return None
        elif formatted_bands > 1:
            return self._transpose_axes + (2,)
        else:
            return self._transpose_axes

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def get_format_function(
            self,
            raw_dtype: numpy.dtype,
            complex_order: Optional[str],
            lut: Optional[numpy.ndarray],
            band_dimension: int,
            image_segment_index: Optional[int] = None,
            **kwargs) -> Optional[FormatFunction]:
        return _get_format_function(raw_dtype, complex_order, lut, band_dimension)

    def _verify_image_segment_compatibility(self, index0: int, index1: int) -> bool:
        img0 = self.get_image_header(index0)
        img1 = self.get_image_header(index1)
        return _verify_image_segment_compatibility(img0, img1)

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
        collection of image segments.

        Parameters
        ----------
        collection_index : int
            The index into the `image_segment_collection` list.

        Returns
        -------
        block_definition: numpy.ndarray
            of the form `[[start_row, end_row, start_column, end_column]]`.
        """

        image_headers = [self.nitf_details.img_headers[image_ind]
                         for image_ind in self.image_segment_collections[collection_index]]
        # noinspection PyTypeChecker
        return _get_collection_element_coordinate_limits(image_headers, return_clevel=False)

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
            mem_map[row_start: row_end, col_start:col_end] = \
                numpy.asarray(img)[0:row_end - row_start, 0:col_end - col_start]
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
        if image_header.IMODE not in ['B', 'P'] or image_header.IC not in ['C3', 'C5', 'M3', 'M5']:
            raise ValueError(
                'Requires IMODE in `(B, P)` and IC in `(C3, C5, M3, M5)`,\n\t'
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
                if anticipated_jpeg_indices[mask_index] != jpeg_delim[0]:
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
        if image_header.is_masked:
            mem_map.fill(0)  # TODO: missing value?

        next_jpeg_block = 0
        for mask_offset, block_bound in zip(mask_offsets, block_bounds):
            if mask_offset == exclude_value:
                continue  # just skip it, it's masked out
            jpeg_delim = jpeg_delimiters[next_jpeg_block]
            # noinspection PyUnresolvedReferences
            the_image_bytes = the_bytes[jpeg_delim[0]:jpeg_delim[1]]
            img = PIL_Image.open(BytesIO(the_image_bytes))
            poo = numpy.array(img)
            # handle block padding situation
            row_start, row_end = block_bound[0], min(block_bound[1], image_header.NROWS)
            col_start, col_end = block_bound[2], min(block_bound[3], image_header.NCOLS)
            mem_map[row_start:row_end, col_start:col_end] = \
                numpy.asarray(img)[0:row_end - row_start, 0:col_end - col_start]
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

        block_size = image_header.get_uncompressed_block_size()

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

        # noinspection PyUnresolvedReferences
        if not (isinstance(block_offsets, numpy.ndarray) and block_offsets.ndim == 1):
            raise ValueError('Got unexpected block offsets `{}`'.format(block_offsets))

        if len(block_bounds) != len(block_offsets):
            raise ValueError('Got mismatch between block definition and block offsets definition')

        final_block_ending = numpy.max(block_offsets[block_offsets != exclude_value]) + block_size + additional_offset
        populated_ending = self.nitf_details.img_segment_sizes[image_segment_index]
        if final_block_ending != populated_ending:
            raise ValueError(
                'Got mismatch between anticipated size {} and populated size {}\n\t'
                'for image segment {}'.format(
                    final_block_ending, populated_ending, image_segment_index))

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
                    mode='r', close_file=False)
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
                    raw_dtype, b_raw_shape, mode='r', close_file=False)
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
                    SubsetSegment(child_segment, subset_def, 'raw', close_parent=True, squeeze=False))

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
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'S' or image_header.IC not in ['C3', 'C5', 'M3', 'M5']:
            raise ValueError(
                'Requires IMODE = `S` and IC in `(C3, C5, M3, M5)`,\n\t'
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

        # get mask definition details
        mask_offsets, exclude_value, additional_offset = self._get_mask_details(image_segment_index)
        # NB: if defined, mask_offsets is a 2-d array here

        # jpeg compression, read everything (skipping mask) and find the jpeg delimiters
        the_bytes = self._read_file_data(offset+additional_offset, image_segment_size-additional_offset)
        jpeg_delimiters = find_jpeg_delimiters(the_bytes)

        # validate our discovered delimiters and the mask offsets
        if mask_offsets is not None:
            if not (isinstance(mask_offsets, numpy.ndarray) and mask_offsets.ndim == 2):
                raise ValueError('Got unexpected mask offsets `{}`'.format(mask_offsets))

            if len(block_bounds) != mask_offsets.shape[1]:
                raise ValueError('Got mismatch between block definition and mask offsets definition')

            # TODO: verify that we don't need to account for mask definition length
            anticipated_jpeg_indices = [index for index, entry in enumerate(mask_offsets.ravel())
                                        if entry != exclude_value]
            if len(jpeg_delimiters) != len(anticipated_jpeg_indices):
                raise ValueError(
                    'Found different number of jpeg delimiters ({})\n\t'
                    'than populated blocks ({}) in masked image segment {}'.format(
                        len(jpeg_delimiters), len(anticipated_jpeg_indices), image_segment_index))
            for jpeg_delim, mask_index in zip(jpeg_delimiters, anticipated_jpeg_indices):
                if anticipated_jpeg_indices[mask_index] != jpeg_delim[0]:
                    raise ValueError(
                        'Populated mask offsets ({})\n\t'
                        'do not agree with discovered jpeg offsets ({})\n\t'
                        'with mask subheader length {}'.format(jpeg_delim, mask_offsets, additional_offset))

        else:
            if len(jpeg_delimiters) != len(block_bounds)*raw_bands:
                raise ValueError(
                    'Found different number of jpeg delimiters ({}) than blocks,\n\t'
                    'bands ({}, {}) in image segment {}'.format(
                        len(jpeg_delimiters), len(block_bounds), raw_bands, image_segment_index))
            mask_offsets = numpy.reshape(
                numpy.array([entry[0] for entry in jpeg_delimiters], dtype='int64'),
                (raw_bands, len(block_bounds)))

        # create a memmap, and extract all of our jpeg data into it as appropriate
        raw_shape = _get_shape(image_header.NROWS, image_header.NCOLS, raw_bands, band_dimension=2)
        fi, path_name = mkstemp(suffix='.sarpy_cache', text=False)
        self._delete_temp_files.append(path_name)
        mem_map = numpy.memmap(
            path_name, dtype=raw_dtype, mode='w+', offset=0,
            shape=raw_shape)
        if image_header.is_masked:
            mem_map.fill(0)  # TODO: missing value?

        next_jpeg_block = 0
        for band_number in range(raw_bands):
            for mask_offset, block_bound in zip(mask_offsets, block_bounds):
                if mask_offset == exclude_value:
                    continue  # just skip it, it's masked out
                jpeg_delim = jpeg_delimiters[next_jpeg_block]
                # noinspection PyUnresolvedReferences
                img = PIL_Image.open(BytesIO(the_bytes[jpeg_delim[0]:jpeg_delim[1]]))
                # handle block padding situation
                row_start, row_end = block_bound[0], min(block_bound[1], image_header.NROWS)
                col_start, col_end = block_bound[2], min(block_bound[3], image_header.NCOLS)
                mem_map[row_start: row_end, col_start:col_end, band_number] = \
                    numpy.asarray(img)[0:row_end - row_start, 0:col_end - col_start]
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

    def _handle_imode_s_no_compression(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'S' or image_header.IC not in ['NC', 'NM']:
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
        block_size = image_header.get_uncompressed_block_size()
        if mask_offsets is not None:
            block_offsets = mask_offsets
        else:
            block_offsets = numpy.zeros((raw_bands, len(block_bounds)), dtype='int64')
            for i in range(raw_bands):
                block_offsets[i, :] = i*(block_size*len(block_bounds)) + \
                                      numpy.arange(len(block_bounds), dtype='int64')*block_size

        if not (isinstance(block_offsets, numpy.ndarray) and block_offsets.ndim == 2):
            raise ValueError('Got unexpected block offsets `{}`'.format(block_offsets))

        if len(block_bounds) != block_offsets.shape[1]:
            raise ValueError('Got mismatch between block definition and block offsets definition')

        block_offsets_flat = block_offsets.ravel()
        final_block_ending = numpy.max(block_offsets_flat[block_offsets_flat != exclude_value]) + \
            block_size + additional_offset
        populated_ending = self.nitf_details.img_segment_sizes[image_segment_index]
        if final_block_ending != populated_ending:
            raise ValueError(
                'Got mismatch between anticipated size {} and populated size {}\n\t'
                'for image segment {}'.format(
                    final_block_ending, populated_ending, image_segment_index))

        band_segments = []
        for band_number in range(raw_bands):
            band_raw_shape = _get_shape(image_header.NROWS, image_header.NCOLS, 1, band_dimension=2)
            data_segments = []
            child_arrangement = []
            for block_index, (block_definition, block_offset) in enumerate(
                    zip(block_bounds, block_offsets[band_number, :])):
                if block_offset == exclude_value:
                    continue  # just skip this, since it's masked out

                b_rows = block_definition[1] - block_definition[0]
                b_cols = block_definition[3] - block_definition[2]
                b_raw_shape = _get_shape(b_rows, b_cols, 1, band_dimension=2)
                total_offset = offset + additional_offset + block_offset
                if can_use_memmap:
                    child_segment = NumpyMemmapSegment(
                        self.file_object, total_offset, raw_dtype, b_raw_shape,
                        raw_dtype, b_raw_shape, mode='r', close_file=False)
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
                        SubsetSegment(child_segment, subset_def, 'raw', close_parent=True, squeeze=False))
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
        elif image_header.IC in ['C3', 'C5', 'M3', 'M5']:
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

        Note that this also stores a reference to the produced data segment in
        the `_image_segment_data_segments` dictionary.

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
            out = self._create_data_segment_from_imode_b(image_segment_index, apply_format)
        elif image_header.IMODE == 'P':
            out = self._create_data_segment_from_imode_p(image_segment_index, apply_format)
        elif image_header.IMODE == 'S':
            out = self._create_data_segment_from_imode_s(image_segment_index, apply_format)
        elif image_header.IMODE == 'R':
            out = self._create_data_segment_from_imode_r(image_segment_index, apply_format)
        else:
            raise ValueError(
                'Got unsupported IMODE `{}` at image segment index `{}`'.format(
                    image_header.IMODE, image_segment_index))
        if image_segment_index in self._image_segment_data_segments:
            logger.warning(
                'Data segment for image segment index {} has already '
                'been created.'.format(image_segment_index))

        self._image_segment_data_segments[image_segment_index] = out
        return out

    def create_data_segment_for_collection_element(self, collection_index: int) -> DataSegment:
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
            out.append(self.create_data_segment_for_collection_element(index))
        return out

    def close(self) -> None:
        self._image_segment_data_segments = None
        BaseReader.close(self)


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name: Union[str, BinaryIO]) -> Optional[NITFReader]:
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


#####
# NITF writing elements

def interpolate_corner_points_string(
        entry: numpy.ndarray,
        rows: int,
        cols: int,
        icp: numpy.ndarray):
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
    icp : numpy.ndarray
        The parent image corner points in geodetic coordinates.

    Returns
    -------
    str
        suitable for IGEOLO entry.
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


def default_image_segmentation(rows: int, cols: int, row_limit: int) -> Tuple[Tuple[int, ...], ...]:
    """
    Determine the appropriate segmentation for the image. This is driven
    by the SICD/SIDD standard, and not the only generally feasible segmentation
    scheme for other NITF file types.

    Parameters
    ----------
    rows : int
    cols : int
    row_limit : int
        It is assumed that this follows the NITF guidelines

    Returns
    -------
    Tuple[Tuple[int, ...], ...]
        Of the form `((row start, row end, column start, column end))`
    """

    im_segments = []
    row_offset = 0
    while row_offset < rows:
        next_rows = min(rows, row_offset + row_limit)
        im_segments.append((row_offset, next_rows, 0, cols))
        row_offset = next_rows
    return tuple(im_segments)


def _flatten_bytes(value: Union[bytes, Sequence]) -> bytes:
    if value is None:
        return b''
    elif isinstance(value, bytes):
        return value
    elif isinstance(value, Sequence):
        return b''.join(_flatten_bytes(entry) for entry in value)
    else:
        raise TypeError('input must be a bytes object, or a sequence with bytes objects as leaves')


class SubheaderManager(object):
    """
    Simple manager object for a NITF subheader, and it's associated information
    in the NITF writing process.

    Introduced in version 1.3.0.
    """

    __slots__ = (
        '_subheader',  '_subheader_offset', '_subheader_size',
        '_item_bytes', '_item_offset', '_item_size',
        '_subheader_written', '_item_written')

    item_bytes_required = True
    """
    Are you required to provide the item bytes?
    """

    subheader_type = None
    """
    What is the type for the subheader?
    """

    def __init__(self, subheader, item_bytes: Optional[bytes] = None):
        if not isinstance(subheader, self.subheader_type):
            raise TypeError(
                'subheader must be of type {} for class {}'.format(
                    self.subheader_type, self.__class__))
        self._subheader = subheader
        self._subheader_size = self._subheader.get_bytes_length()
        self._subheader_offset = None
        self._item_offset = None
        self._subheader_written = False
        self._item_written = False
        self._item_size = None

        self._item_bytes = None
        if item_bytes is None:
            if self.item_bytes_required:
                raise ValueError(
                    'item_bytes is required by class {}.'.format(
                        self.__class__))
        else:
            self.item_bytes = item_bytes

    @property
    def subheader(self):
        """
        The subheader.
        """

        return self._subheader

    @property
    def subheader_offset(self) -> Optional[int]:
        """
        int: The subheader offset.
        """

        return self._subheader_offset

    @subheader_offset.setter
    def subheader_offset(self, value) -> None:
        if self._subheader_offset is not None:
            raise ValueError("subheader_offset is read only after being initially defined.")
        self._subheader_offset = int(value)
        self._item_offset = self._subheader_offset + self.subheader_size

    @property
    def subheader_size(self) -> int:
        """
        int: The subheader size
        """

        return self._subheader_size

    @property
    def item_offset(self) -> Optional[int]:
        """
        int: The item offset.
        """

        return self._item_offset

    @property
    def item_size(self) -> Optional[int]:
        """
        int: The item size
        """

        return self._item_size

    @item_size.setter
    def item_size(self, value) -> None:
        if self._item_size is not None:
            raise ValueError("item_size is read only after being initially defined.")
        self._item_size = int(value)

    @property
    def end_of_item(self) -> Optional[int]:
        """
        int: The position of the end of respective item. This will be the
        offset for the next element.
        """

        if self._item_offset is None:
            return None
        elif self._item_size is None:
            return None

        return self.item_offset + self.item_size

    @property
    def subheader_written(self) -> bool:
        """
        bool: Has this subheader been written?
        """

        return self._subheader_written

    @subheader_written.setter
    def subheader_written(self, value) -> None:
        value = bool(value)
        if self._subheader_written and not value:
            raise ValueError(
                'subheader_written has already been set to True,\n\t'
                'it cannot be reverted to False')
        self._subheader_written = value

    @property
    def item_bytes(self) -> Optional[bytes]:
        """
        None|bytes: The item bytes.
        """

        return self._item_bytes

    @item_bytes.setter
    def item_bytes(self, value: Union[bytes, Sequence]) -> None:
        if self._item_bytes is not None:
            raise ValueError("item_bytes is read only after being initially defined.")
        if value is None:
            self._item_bytes = None
            return

        # TODO: verify the mask information, in the event that value is a sequence?
        value = _flatten_bytes(value)
        if self._item_size is not None and len(value) != self._item_size:
            raise ValueError(
                'item_bytes input has size {},\n\t'
                'but item_size has been defined as {}.'.format(len(value), self._item_size))
        self._item_bytes = value
        self.item_size = len(value)

    @property
    def item_written(self) -> bool:
        """
        bool: Has the item been written?
        """

        return self._item_written

    @item_written.setter
    def item_written(self, value):
        value = bool(value)
        if self._item_written and not value:
            raise ValueError(
                'item_written has already been set to True,\n\t'
                'it cannot be reverted to False')
        self._item_written = value

    def write_subheader(self, file_object: BinaryIO) -> None:
        """
        Write the subheader, at its specified offset, to the file. If writing
        occurs, the file location will be advanced to the end of the subheader
        location.

        Parameters
        ----------
        file_object : BinaryIO
        """

        if self.subheader_written:
            return

        if self.subheader_offset is None:
            return  # nothing to be done

        the_bytes = self.subheader.to_bytes()
        if len(the_bytes) != self._subheader_size:
            raise ValueError(
                'mismatch between the size of the subheader {}\n\t'
                'and the anticipated size of the subheader {}'.format(len(the_bytes), self._subheader_size))

        file_object.seek(self.subheader_offset, os.SEEK_SET)
        file_object.write(the_bytes)
        self.subheader_written = True

    def write_item(self, file_object: BinaryIO) -> None:
        """
        Write the item bytes (if populated), at its specified offset, to the
        file. This requires that the subheader has previously be written. If
        writing occurs, the file location will be advanced to the end of the item
        location.

        Parameters
        ----------
        file_object : BinaryIO

        Returns
        -------
        None
        """

        if self.item_written:
            return

        if self.item_offset is None:
            return  # nothing to be done

        if self.item_bytes is None:
            return  # nothing to be done

        if not self.subheader_written:
            return  # nothing to be done

        file_object.seek(self.item_offset, os.SEEK_SET)
        file_object.write(self.item_bytes)
        self.item_written = True


class ImageSubheaderManager(SubheaderManager):
    item_bytes_required = False
    subheader_type = ImageSegmentHeader

    @property
    def subheader(self) -> ImageSegmentHeader:
        """
        ImageSegmentHeader: The image subheader. Any image mask subheader should
        be populated in the `mask_subheader` property. The size of this will be
        handled independently from the image bytes.
        """

        return self._subheader

    @property
    def item_size(self) -> Optional[int]:
        """
        int: The item size.
        """

        return self._item_size

    @item_size.setter
    def item_size(self, value):
        if self._item_size is not None:
            logger.warning("item_size is read only after being initially defined.")
            return
        if self.subheader.mask_subheader is None:
            self._item_size = int(value)
        else:
            self._item_size = int(value) + self.subheader.mask_subheader.get_bytes_length()

    def write_subheader(self, file_object: BinaryIO) -> None:
        if self.subheader_written:
            return

        SubheaderManager.write_subheader(self, file_object)
        if self.subheader.mask_subheader is not None:
            file_object.write(self.subheader.mask_subheader.to_bytes())

    def write_item(self, file_object: BinaryIO) -> None:
        if self.item_written:
            return

        if self.item_offset is None:
            return

        if self.item_bytes is None:
            return

        if not self.subheader_written:
            return

        if self.subheader.mask_subheader is None:
            file_object.seek(self.item_offset, os.SEEK_SET)
        else:
            file_object.seek(
                self.item_offset+self.subheader.mask_subheader.get_bytes_length(), os.SEEK_SET)
        file_object.write(self.item_bytes)
        self.item_written = True


class GraphicsSubheaderManager(SubheaderManager):
    item_bytes_required = True
    subheader_type = GraphicsSegmentHeader

    @property
    def subheader(self) -> GraphicsSegmentHeader:
        return self._subheader


class TextSubheaderManager(SubheaderManager):
    item_bytes_required = True
    subheader_type = TextSegmentHeader

    @property
    def subheader(self) -> TextSegmentHeader:
        return self._subheader


class DESSubheaderManager(SubheaderManager):
    item_bytes_required = True
    subheader_type = DataExtensionHeader

    @property
    def subheader(self) -> DataExtensionHeader:
        return self._subheader


class RESSubheaderManager(SubheaderManager):
    item_bytes_required = True
    subheader_type = DataExtensionHeader

    @property
    def subheader(self) -> ReservedExtensionHeader:
        return self._subheader


class NITFWritingDetails(object):
    """
    Manager for all the NITF subheader information.

    Note that doing anything which modified the size of the headers after
    initialization (i.e. adding TREs) will not be reflected

    Introduced in version 1.3.0.
    """
    __slots__ = (
        '_header', '_header_size', '_header_written', '_image_managers',
        '_graphics_managers', '_text_managers', '_des_managers', '_res_managers',
        '_image_segment_collections', '_image_segment_coordinates', '_collections_clevel')

    def __init__(
            self,
            header: NITFHeader,
            image_managers: Optional[Tuple[ImageSubheaderManager, ...]] = None,
            image_segment_collections: Optional[Tuple[Tuple[int, ...], ...]] = None,
            image_segment_coordinates: Optional[Tuple[Tuple[Tuple[int, ...], ...], ...]] = None,
            graphics_managers: Optional[Tuple[GraphicsSubheaderManager, ...]] = None,
            text_managers: Optional[Tuple[TextSubheaderManager, ...]] = None,
            des_managers: Optional[Tuple[DESSubheaderManager, ...]] = None,
            res_managers: Optional[Tuple[RESSubheaderManager, ...]] = None):
        """

        Parameters
        ----------
        header : NITFHeader
        image_managers : Optional[Tuple[ImageSubheaderManager, ...]]
            Should be provided, unless the desire is to write NITF without images
        image_segment_collections: Optional[Tuple[Tuple[int, ...], ...]]
            Presence contingent on presence of image_managers
        image_segment_coordinates: Optional[Tuple[Tuple[Tuple[int, ...], ...], ...]]
            Contingent on image_managers. This will be inferred if not provided,
            and validated if provided.
        graphics_managers: Optional[Tuple[GraphicsSubheaderManager, ...]]
        text_managers: Optional[Tuple[TextSubheaderManager, ...]]
        des_managers: Optional[Tuple[DESSubheaderManager, ...]]
        res_managers: Optional[Tuple[RESSubheaderManager, ...]]
        """

        self._collections_clevel = None
        self._header = None
        self._header_written = False
        self._image_managers = None
        self._image_segment_collections = None
        self._image_segment_coordinates = None
        self._graphics_managers = None
        self._text_managers = None
        self._des_managers = None
        self._res_managers = None

        self.header = header
        self.image_managers = image_managers
        self.image_segment_collections = image_segment_collections
        self.image_segment_coordinates = image_segment_coordinates
        self.graphics_managers = graphics_managers
        self.text_managers = text_managers
        self.des_managers = des_managers
        self.res_managers = res_managers

        # set nominal size arrays (for header size purposes), to be corrected later
        self.set_all_sizes(require=False)
        self._header_size = header.get_bytes_length()  # type: int

    @property
    def header(self) -> NITFHeader:
        """
        NITFHeader: The main NITF header. Note that doing anything that changes
        the size of that header (i.e. adding TREs) after initialization will
        result in a broken state.
        """

        return self._header

    @header.setter
    def header(self, value):
        if self._header is not None:
            raise ValueError('header is read-only')
        if not isinstance(value, NITFHeader):
            raise TypeError('header must be of type {}'.format(NITFHeader))
        self._header = value

    @property
    def image_managers(self) -> Optional[Tuple[ImageSubheaderManager, ...]]:
        return self._image_managers

    @image_managers.setter
    def image_managers(self, value):
        if self._image_managers is not None:
            raise ValueError('image_managers is read-only')
        if value is None:
            self._image_managers = None
            return

        if not isinstance(value, tuple):
            raise TypeError('image_managers must be a tuple')
        for entry in value:
            if not isinstance(entry, ImageSubheaderManager):
                raise TypeError('image_managers entries must be of type {}'.format(ImageSubheaderManager))
        self._image_managers = value

    @property
    def image_segment_collections(self) -> Tuple[Tuple[int, ...]]:
        """
        The definition for how image segments are grouped together to form the
        aggregate images.

        Each entry corresponds to a single aggregate image, and the entry defines
        the image segment indices which are combined to make up the aggregate image.

        This must be an ordered partitioning of the set `(0, ..., len(image_managers)-1)`.

        Returns
        -------
        Tuple[Tuple[int, ...]]
        """

        return self._image_segment_collections

    @image_segment_collections.setter
    def image_segment_collections(self, value):
        if self._image_segment_collections is not None:
            raise ValueError('image_segment_collections is read only')

        if self.image_managers is None:
            self._image_segment_collections = None
            return

        if not isinstance(value, tuple):
            raise TypeError('image segment collection must be a tuple')

        last_index = -1
        for entry in value:
            if not isinstance(entry, tuple):
                raise TypeError('image segment collection must be a tuple of tuples')
            if last_index == -1:
                if entry[0] != 0:
                    raise ValueError('The first entry of image segment collection must start at 0.')

            for item in entry:
                if not isinstance(item, int) or item < 0:
                    raise TypeError('image segment collection must be a tuple of tuples of non-negative ints')
                if item != last_index + 1:
                    raise ValueError('image segment collection entries must be arranged in ascending order')
                last_index = item
        if last_index != len(self.image_managers) - 1:
            raise ValueError('Mismatch between the number of image segments and the collection entries')
        self._image_segment_collections = value

    @property
    def image_segment_coordinates(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        """
        The image bounds for the segment collection. This is associated with the
        `image_segment_collection` property.

        Entry `image_segment_coordinates[i]` is associated with the ith aggregate
        image. We have `image_segment_coordinates[i]` is a tuple of tuples of the
        form
        `((row_start, row_end, col_start, col_end)_j,
          (row_start, row_end, col_start, col_end)_{j+1}, ...)`.

        This indicates that the first image segment associated with
        ith aggregate image is at index `j` covering the portion of the aggregate
        image determined by bounds `(row_start, row_end, col_start, col_end)_j`,
        the second image segment is at index `j+1` covering the portion of the
        aggregate determined by bounds `(row_start, row_end, col_start, col_end)_{j+1}`,
        and so on.

        Returns
        -------
        Tuple[Tuple[Tuple[int, ...], ...], ...]:
        """

        return self._image_segment_coordinates

    @image_segment_coordinates.setter
    def image_segment_coordinates(self, value):
        if self._image_segment_coordinates is not None:
            raise ValueError('image_segment_coordinates is read only')

        if self.image_managers is None:
            self._image_segment_coordinates = None
            return

        # create the anticipated version
        anticipated = []
        collections_clevel = []
        for coll in self.image_segment_collections:
            image_headers = [self.image_managers[image_ind].subheader for image_ind in coll]
            coordinate_scheme, clevel = _get_collection_element_coordinate_limits(image_headers, return_clevel=True)
            collections_clevel.append(clevel)
            # noinspection PyTypeChecker
            coordinate_scheme = tuple(tuple(entry) for entry in coordinate_scheme.tolist())
            anticipated.append(coordinate_scheme)
        self._collections_clevel = tuple(collections_clevel)
        anticipated = tuple(anticipated)
        if value is None:
            self._image_segment_coordinates = anticipated
            return

        if not isinstance(value, tuple):
            raise TypeError('image_segment_coordinates must be a tuple')

        if len(value) != len(self.image_segment_collections):
            raise ValueError(
                'Lengths of image_segment_collections and image_segment_coordinates '
                'must match')

        for coords, antic in zip(value, anticipated):
            if not isinstance(coords, tuple):
                raise ValueError('image_segment_coordinates entries must be a tuple')
            if len(coords) != len(antic):
                raise ValueError(
                    'image_segment_collections entries and image_segment_coordinates '
                    'entries must have matching lengths')
            if coords != antic:
                raise ValueError(
                    'image_segment_coordinates does not match the anticipated '
                    'value\n\t{}\n\t{}'.format(value, anticipated))
        self._image_segment_coordinates = value

    @property
    def graphics_managers(self) -> Optional[Tuple[GraphicsSubheaderManager, ...]]:
        return self._graphics_managers

    @graphics_managers.setter
    def graphics_managers(self, value):
        if self._graphics_managers is not None:
            raise ValueError('graphics_managers is read-only')
        if value is None:
            self._graphics_managers = None
            return

        if not isinstance(value, tuple):
            raise TypeError('graphics_managers must be a tuple')
        for entry in value:
            if not isinstance(entry, GraphicsSubheaderManager):
                raise TypeError('graphics_managers entries must be of type {}'.format(GraphicsSubheaderManager))
        self._graphics_managers = value

    @property
    def text_managers(self) -> Optional[Tuple[TextSubheaderManager, ...]]:
        return self._text_managers

    @text_managers.setter
    def text_managers(self, value):
        if self._text_managers is not None:
            raise ValueError('text_managers is read-only')
        if value is None:
            self._text_managers = None
            return

        if not isinstance(value, tuple):
            raise TypeError('text_managers must be a tuple')
        for entry in value:
            if not isinstance(entry, TextSubheaderManager):
                raise TypeError('text_managers entries must be of type {}'.format(TextSubheaderManager))
        self._text_managers = value

    @property
    def des_managers(self) -> Optional[Tuple[DESSubheaderManager, ...]]:
        return self._des_managers

    @des_managers.setter
    def des_managers(self, value):
        if self._des_managers is not None:
            raise ValueError('des_managers is read-only')
        if value is None:
            self._des_managers = None
            return

        if not isinstance(value, tuple):
            raise TypeError('des_managers must be a tuple')
        for entry in value:
            if not isinstance(entry, DESSubheaderManager):
                raise TypeError('des_managers entries must be of type {}'.format(DESSubheaderManager))
        self._des_managers = value

    @property
    def res_managers(self) -> Optional[Tuple[RESSubheaderManager, ...]]:
        return self._res_managers

    @res_managers.setter
    def res_managers(self, value):
        if self._res_managers is not None:
            raise ValueError('res_managers is read-only')
        if value is None:
            self._res_managers = None
            return

        if not isinstance(value, tuple):
            raise TypeError('res_managers must be a tuple')
        for entry in value:
            if not isinstance(entry, RESSubheaderManager):
                raise TypeError('res_managers entries must be of type {}'.format(RESSubheaderManager))
        self._res_managers = value

    def _get_sizes(
            self,
            managers: Optional[Sequence[SubheaderManager]],
            name: str,
            require: bool = False) -> Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        if managers is None:
            return None, None

        subhead_sizes = numpy.zeros((len(managers), ), dtype='int64')
        item_sizes = numpy.zeros((len(managers), ), dtype='int64')
        for i, entry in enumerate(managers):
            subhead_sizes[i] = entry.subheader_size
            item_size = entry.item_size
            if item_size is None:
                if require:
                    raise ValueError('item_size for {} at index {} is unset'.format(name, item_size))
                else:
                    item_size = 0
            item_sizes[i] = item_size
        return subhead_sizes, item_sizes

    def _write_items(self, managers: Optional[Sequence[SubheaderManager]], file_object: BinaryIO) -> None:
        if managers is None:
            return
        for index, entry in enumerate(managers):
            entry.write_subheader(file_object)
            entry.write_item(file_object)

    def _verify_item_written(self, managers: Optional[Sequence[SubheaderManager]], name: str) -> None:
        if managers is None:
            return

        for index, entry in enumerate(managers):
            if not entry.subheader_written:
                logger.error('{} subheader at index {} not written'.format(name, index))
            if not entry.item_written:
                logger.error('{} data at index {} not written'.format(name, index))

    def _get_image_sizes(self, require: bool = False) -> ImageSegmentsType:
        """
        Gets the image sizes details for the NITF header.

        Returns
        -------
        ImageSegmentsType
        """

        subhead_sizes, item_sizes = self._get_sizes(self.image_managers, 'Image', require=require)
        return ImageSegmentsType(subhead_sizes=subhead_sizes, item_sizes=item_sizes)

    def _get_graphics_sizes(self, require: bool = False) -> GraphicsSegmentsType:
        """
        Gets the graphics sizes details for the NITF header.

        Parameters
        ----------
        require : bool
            Require all sizes to be set?

        Returns
        -------
        ImageSegmentsType
        """

        subhead_sizes, item_sizes = self._get_sizes(self.graphics_managers, 'Graphics', require=require)
        return GraphicsSegmentsType(subhead_sizes=subhead_sizes, item_sizes=item_sizes)

    def _get_text_sizes(self, require: bool = False) -> TextSegmentsType:
        """
        Gets the text sizes details for the NITF header.

        Returns
        -------
        TextSegmentsType
        """

        subhead_sizes, item_sizes = self._get_sizes(self.text_managers, 'Text', require=require)
        return TextSegmentsType(subhead_sizes=subhead_sizes, item_sizes=item_sizes)

    def _get_des_sizes(self, require: bool = False) -> DataExtensionsType:
        """
        Gets the image sizes details for the NITF header.

        Returns
        -------
        ImageSegmentsType
        """

        subhead_sizes, item_sizes = self._get_sizes(self.des_managers, 'DES', require=require)
        return DataExtensionsType(subhead_sizes=subhead_sizes, item_sizes=item_sizes)

    def _get_res_sizes(self, require: bool = False) -> ReservedExtensionsType:
        """
        Gets the image sizes details for the NITF header.

        Returns
        -------
        ImageSegmentsType
        """

        subhead_sizes, item_sizes = self._get_sizes(self.res_managers, 'RES', require=require)
        return ReservedExtensionsType(subhead_sizes=subhead_sizes, item_sizes=item_sizes)

    def set_first_image_offset(self) -> None:
        """
        Sets the first image offset from the header length.

        Returns
        -------
        None
        """

        if self.image_managers is None:
            return
        self.image_managers[0].subheader_offset = self._header_size

    def verify_images_have_no_compression(self) -> bool:
        """
        Verify that there is no compression set for every image manager. That is,
        we are going to directly write a NITF file.

        Returns
        -------
        bool
        """

        if self.image_managers is None:
            return True

        out = True
        for entry in self.image_managers:
            out &= (entry.subheader.IC in ['NC', 'NM'])
        return out

    def set_all_sizes(self, require: bool = False) -> None:
        """
        This sets the nominal size information in the nitf header, and optionally
        verifies that all the item_size values are set.

        Parameters
        ----------
        require : bool
            Require all sizes to be set? `0` will be used a as placeholder for
            header information population.

        Returns
        -------
        None
        """

        self.header.ImageSegments = self._get_image_sizes(require=require)
        self.header.GraphicsSegments = self._get_graphics_sizes(require=require)
        self.header.TextSegments = self._get_text_sizes(require=require)
        self.header.DataExtensions = self._get_des_sizes(require=require)
        self.header.ReservedExtensions = self._get_res_sizes(require=require)

    def verify_all_offsets(self, require: bool = False) -> bool:
        """
        This sets and/or verifies all offsets.

        Parameters
        ----------
        require : bool
            Require all offsets to be set?

        Returns
        -------
        bool
        """

        last_offset = self._header_size
        if self.image_managers is not None:
            for index, entry in enumerate(self.image_managers):
                if entry.subheader_offset is None:
                    entry.subheader_offset = last_offset
                elif entry.subheader_offset != last_offset:
                    raise ValueError(
                        'image manager at index {} has subheader offset which does not agree\n\t'
                        'with the end of the previous element'.format(index))
                if entry.item_size is None or entry.item_size == 0:
                    if require:
                        raise ValueError(
                            'image manager at index {} has item_size unpopulated or populated as 0'.format(index))
                    else:
                        return False
                last_offset = entry.end_of_item

        if self.graphics_managers is not None:
            for index, entry in enumerate(self.graphics_managers):
                if entry.subheader_offset is None:
                    entry.subheader_offset = last_offset
                elif entry.subheader_offset != last_offset:
                    raise ValueError(
                        'graphics manager at index {} has subheader offset which does not agree\n\t'
                        'with the end of the previous element'.format(index))
                if entry.item_size is None or entry.item_size == 0:
                    if require:
                        raise ValueError(
                            'graphics manager at index {} has item_size unpopulated or populated as 0'.format(index))
                    else:
                        return False
                last_offset = entry.end_of_item

        if self.text_managers is not None:
            for index, entry in enumerate(self.text_managers):
                if entry.subheader_offset is None:
                    entry.subheader_offset = last_offset
                elif entry.subheader_offset != last_offset:
                    raise ValueError(
                        'text manager at index {} has subheader offset which does not agree\n\t'
                        'with the end of the previous element'.format(index))
                if entry.item_size is None or entry.item_size == 0:
                    if require:
                        raise ValueError(
                            'text manager at index {} has item_size unpopulated or populated as 0'.format(index))
                    else:
                        return False
                last_offset = entry.end_of_item

        if self.des_managers is not None:
            for index, entry in enumerate(self.des_managers):
                if entry.subheader_offset is None:
                    entry.subheader_offset = last_offset
                elif entry.subheader_offset != last_offset:
                    raise ValueError(
                        'des manager at index {} has subheader offset which does not agree\n\t'
                        'with the end of the previous element'.format(index))
                if entry.item_size is None or entry.item_size == 0:
                    if require:
                        raise ValueError(
                            'des manager at index {} has item_size unpopulated or populated as 0'.format(index))
                    else:
                        return False
                last_offset = entry.end_of_item

        if self.res_managers is not None:
            for index, entry in enumerate(self.res_managers):
                if entry.subheader_offset is None:
                    entry.subheader_offset = last_offset
                elif entry.subheader_offset != last_offset:
                    raise ValueError(
                        'res manager at index {} has subheader offset which does not agree\n\t'
                        'with the end of the previous element'.format(index))
                if entry.item_size is None or entry.item_size == 0:
                    if require:
                        raise ValueError(
                            'res manager at index {} has item_size unpopulated or populated as 0'.format(index))
                    else:
                        return False
                last_offset = entry.end_of_item
        self.header.FL = last_offset
        return True

    def set_header_clevel(self) -> None:
        """
        Sets the appropriate CLEVEL. This requires that header.FL (file size) has
        been previously populated correctly (using :meth:`verify_all_offsets`).

        Returns
        -------
        None
        """

        file_size = self.header.FL
        if file_size < 50 * (1024 ** 2):
            mem_clevel = 3
        elif file_size < (1024 ** 3):
            mem_clevel = 5
        elif file_size < 2 * (1024 ** 3):
            mem_clevel = 6
        elif file_size < 10 * (1024 ** 3):
            mem_clevel = 7
        else:
            mem_clevel = 9

        self.header.CLEVEL = mem_clevel if self._collections_clevel is None else \
            max(mem_clevel, max(self._collections_clevel))

    def write_header(self, file_object: BinaryIO, overwrite: bool = False) -> None:
        """
        Write the main NITF header.

        Parameters
        ----------
        file_object : BinaryIO
        overwrite : bool
            Overwrite, if previously written?

        Returns
        -------
        None
        """

        if self._header_written and not overwrite:
            return
        the_bytes = self.header.to_bytes()
        if len(the_bytes) != self._header_size:
            raise ValueError(
                'The anticipated header length {}\n\t'
                'does not match the actual header length {}'.format(self._header_size, len(the_bytes)))
        self.set_header_clevel()
        file_object.seek(0, os.SEEK_SET)
        file_object.write(the_bytes)
        self._header_written = True

    def write_all_populated_items(self, file_object: BinaryIO) -> None:
        """
        Write everything populated. This assumes that the header will start at the
        beginning (position 0) of the file-like object.

        Parameters
        ----------
        file_object : BinaryIO

        Returns
        -------
        None
        """

        self.write_header(file_object, overwrite=False)
        self._write_items(self.image_managers, file_object)
        self._write_items(self.graphics_managers, file_object)
        self._write_items(self.text_managers, file_object)
        self._write_items(self.des_managers, file_object)
        self._write_items(self.res_managers, file_object)

    def verify_all_written(self) -> None:
        if not self._header_written:
            logger.error('NITF header not written')

        self._verify_item_written(self.image_managers, 'image')
        self._verify_item_written(self.graphics_managers, 'graphics')
        self._verify_item_written(self.text_managers, 'text')
        self._verify_item_written(self.des_managers, 'DES')
        self._verify_item_written(self.res_managers, 'RES')


#############
# An array based (for only uncompressed images) nitf 2.1 writer

class NITFWriter(BaseWriter):
    __slots__ = (
        '_file_object', '_file_name', '_in_memory',
        '_nitf_writing_details', '_image_segment_data_segments')

    def __init__(
            self,
            file_object: Union[str, BinaryIO],
            writing_details: NITFWritingDetails,
            check_existence: bool = True):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
        writing_details : NITFWritingDetails
        check_existence : bool
            Should we check if the given file already exists?

        Raises
        ------
        SarpyIOError
            If the given `file_name` already exists
        """

        self._nitf_writing_details = None
        self._image_segment_data_segments = []  # type: List[DataSegment]

        if isinstance(file_object, str):
            if check_existence and os.path.exists(file_object):
                raise SarpyIOError(
                    'Given file {} already exists, and a new NITF file cannot be created here.'.format(file_object))
            file_object = open(file_object, 'wb')

        if not is_file_like(file_object):
            raise ValueError('file_object requires a file path or BinaryIO object')

        self._file_object = file_object
        if is_real_file(file_object):
            self._file_name = file_object.name
            self._in_memory = False
        else:
            self._file_name = None
            self._in_memory = True

        self.nitf_writing_details = writing_details

        if not self.nitf_writing_details.verify_images_have_no_compression():
            raise ValueError(
                'Some image segments indicate compression in the image managers of the nitf_writing_details')

        # set the image offset
        self.nitf_writing_details.set_first_image_offset()

        self._verify_image_segments()
        self.verify_collection_compliance()

        data_segments = self.get_data_segments()

        self.nitf_writing_details.set_all_sizes(require=True)  # NB: while no compression supported...
        if not self._in_memory:
            self.nitf_writing_details.write_all_populated_items(self._file_object)
        BaseWriter.__init__(self, data_segments)

    @property
    def nitf_writing_details(self) -> NITFWritingDetails:
        """
        NITFWritingDetails: The NITF subheader details.
        """

        return self._nitf_writing_details

    @nitf_writing_details.setter
    def nitf_writing_details(self, value):
        if self._nitf_writing_details is not None:
            raise ValueError('nitf_writing_details is read-only')
        if not isinstance(value, NITFWritingDetails):
            raise TypeError('nitf_writing_details must be of type {}'.format(NITFWritingDetails))
        self._nitf_writing_details = value

    @property
    def image_managers(self) -> Tuple[ImageSubheaderManager, ...]:
        return self.nitf_writing_details.image_managers

    def _set_image_size(self, image_segment_index: int, item_size: int) -> None:
        """
        Sets the image size information. This should be without consideration
        for the presence of an image mask, which is handled by with the image
        subheader (if present).

        Parameters
        ----------
        image_segment_index : int
        item_size : int
        """

        self.image_managers[image_segment_index].item_size = item_size

    @property
    def image_segment_collections(self) -> Tuple[Tuple[int, ...]]:
        """
        The definition for how image segments are grouped together to form the
        aggregate image.

        Each entry corresponds to a single output image, and the entry defines
        the image segment indices which are combined to make up the output image.

        Returns
        -------
        Tuple[Tuple[int, ...]]
        """

        return self.nitf_writing_details.image_segment_collections

    def get_image_header(self, index: int) -> ImageSegmentHeader:
        """
        Gets the image subheader at the specified index.

        Parameters
        ----------
        index : int

        Returns
        -------
        ImageSegmentHeader
        """

        return self.image_managers[index].subheader

    # noinspection PyMethodMayBeStatic
    def _check_image_segment_for_compliance(
            self,
            index: int,
            img_header: ImageSegmentHeader) -> None:
        """
        Checks whether the image segment can be (or should be) opened.

        Parameters
        ----------
        index : int
            The image segment index (for logging)
        img_header : ImageSegmentHeader
            The image segment header
        """

        if img_header.NBPP not in (8, 16, 32, 64):
            # numpy basically only supports traditional typing
            raise ValueError(
                'Image segment at index {} has bits per pixel per band {},\n\t'
                'only 8, 16, 32, 64 are supported.'.format(index, img_header.NBPP))

        if img_header.is_compressed:
            if PIL_Image is None:
                raise ValueError(
                    'Image segment at index {} has unsupported IC value {}.'.format(
                        index, img_header.IC))

        if img_header.IMODE not in ['B', 'P', 'R']:
            raise ValueError('Got unsupported IMODE `{}`'.format(img_header.IMODE))

        if img_header.mask_subheader is None:
            if img_header.IC != 'NC':
                raise ValueError('Mask subheader not defined, but IC is not `NC`')
        else:
            if img_header.IC != 'NM':
                raise ValueError('Mask subheader is defined, but IC is not `NM`')

    def _verify_image_segments(self) -> None:
        for index, entry in enumerate(self.image_managers):
            if entry.item_bytes is not None:
                raise ValueError(
                    'The item_bytes is populated for image segment {}.\n\t'
                    'This is incompatible with array-type image writing'.format(index))
            subhead = entry.subheader
            self._check_image_segment_for_compliance(index, subhead)

    def _construct_block_bounds(self, image_segment_index: int) -> List[Tuple[int, int, int, int]]:
        image_header = self.get_image_header(image_segment_index)
        # noinspection PyTypeChecker
        return _construct_block_bounds(image_header)

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
            self,
            image_segment_index: int) -> Tuple[numpy.dtype, numpy.dtype, int, Optional[str], Optional[numpy.ndarray]]:
        image_header = self.get_image_header(image_segment_index)
        return _get_dtype(image_header)

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def get_format_function(
            self,
            raw_dtype: numpy.dtype,
            complex_order: Optional[str],
            lut: Optional[numpy.ndarray],
            band_dimension: int,
            image_segment_index: Optional[int] = None,
            **kwargs) -> Optional[FormatFunction]:
        return _get_format_function(raw_dtype, complex_order, lut, band_dimension)

    def _verify_image_segment_compatibility(self, index0: int, index1: int) -> bool:
        img0 = self.get_image_header(index0)
        img1 = self.get_image_header(index1)
        return _verify_image_segment_compatibility(img0, img1)

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

    def _get_collection_element_coordinate_limits(self, collection_index: int) -> Tuple[Tuple[int, ...], ...]:
        """
        For the given image segment collection, as defined in the
        `image_segment_collections` property value, get the relative coordinate
        scheme of the form `[[start_row, end_row, start_column, end_column]]`.

        This relies on inspection of `IALVL` and `ILOC` values for this
        collection of image segments.

        Parameters
        ----------
        collection_index : int
            The index into the `image_segment_collection` list.

        Returns
        -------
        block_definition: Tuple[Tuple[int, ...], ...]
            of the form `((start_row, end_row, start_column, end_column))`.
        """

        return self.nitf_writing_details.image_segment_coordinates[collection_index]

    def _handle_no_compression(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        # NB: this should definitely set the image size in the manager.

        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE not in ['B', 'R', 'P'] or image_header.IC not in ['NC', 'NM']:
            raise ValueError(
                'Requires IMODE in `(B, R, P)` and IC in `(NC, NM)`,\n\t'
                'got `{}` and `{}` at image segment index {}'.format(
                    image_header.IMODE, image_header.IC, image_segment_index))

        raw_bands = len(image_header.Bands)

        # get bytes offset to this image segment (relative to start of file)
        #   this is only necessary if not in_memory processing
        offset = 0 if self._in_memory else self.image_managers[image_segment_index].item_offset
        raw_dtype, formatted_dtype, formatted_bands, complex_order, lut = self._get_dtypes(image_segment_index)

        block_bounds = self._construct_block_bounds(image_segment_index)
        assert isinstance(block_bounds, list)

        block_size = image_header.get_uncompressed_block_size()

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

        # noinspection PyUnresolvedReferences
        if not (isinstance(block_offsets, numpy.ndarray) and block_offsets.ndim == 1):
            raise ValueError('Got unexpected mask offsets `{}`'.format(block_offsets))

        if len(block_bounds) != len(block_offsets):
            raise ValueError('Got mismatch between block definition and block offsets definition')

        final_block_ending = numpy.max(block_offsets[block_offsets != exclude_value]) + block_size + additional_offset

        # set the details in the image manager...
        self.image_managers[image_segment_index].item_size = final_block_ending - additional_offset
        if not self._in_memory:
            self.image_managers[image_segment_index].item_written = True
            # NB: it's written in principle by the data segment

        # determine output particulars
        if apply_format:
            format_function = self.get_format_function(
                raw_dtype, complex_order, lut, raw_band_dimension,
                image_segment_index=image_segment_index)
            use_transpose = None
            use_reverse = None
            formatted_shape = _get_shape(image_header.NROWS, image_header.NCOLS, formatted_bands, band_dimension=2)
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
            transpose_axes = (1, 2, 0)
            reverse_axes = None
        elif image_header.IMODE == 'R':
            # order inside the block is (rows, bands, columns)
            transpose_axes = (0, 2, 1)
            reverse_axes = None
        elif image_header.IMODE == 'P':
            transpose_axes = None
            reverse_axes = use_reverse
        else:
            raise ValueError('Unhandled IMODE `{}`'.format(image_header.IMODE))

        if len(block_bounds) == 1:
            # there is just a single block, no need to obfuscate behind a
            # block aggregate

            if self._in_memory:
                underlying_array = numpy.full(raw_shape, 0, dtype=raw_dtype)
                return NumpyArraySegment(
                    underlying_array, formatted_dtype, formatted_shape,
                    reverse_axes=reverse_axes, transpose_axes=transpose_axes,
                    format_function=format_function, mode='w')
            else:
                return NumpyMemmapSegment(
                    self._file_name, offset, raw_dtype, raw_shape,
                    formatted_dtype, formatted_shape, reverse_axes=reverse_axes,
                    transpose_axes=transpose_axes, format_function=format_function,
                    mode='w', close_file=False)

        data_segments = []
        child_arrangement = []
        for block_index, (block_definition, block_offset) in enumerate(zip(block_bounds, block_offsets)):
            if block_offset == exclude_value:
                continue  # just skip this, since it's masked out

            b_rows = block_definition[1] - block_definition[0]
            b_cols = block_definition[3] - block_definition[2]
            b_raw_shape = _get_shape(b_rows, b_cols, raw_bands, band_dimension=raw_band_dimension)
            total_offset = offset + additional_offset + block_offset
            if self._in_memory:
                underlying_array = numpy.full(b_raw_shape, 0, dtype=raw_dtype)
                child_segment = NumpyArraySegment(
                    underlying_array, raw_dtype, b_raw_shape, mode='w')
            else:
                child_segment = NumpyMemmapSegment(
                    self._file_name, total_offset, raw_dtype, b_raw_shape,
                    raw_dtype, b_raw_shape, mode='w', close_file=False)
            # handle block padding situation
            row_start, row_end = block_definition[0], min(block_definition[1], image_header.NROWS)
            col_start, col_end = block_definition[2], min(block_definition[3], image_header.NCOLS)
            # NB: we need not establish a subset segment for writing
            if row_end == block_definition[1] and col_end == block_definition[3]:
                data_segments.append(child_segment)
            else:
                subset_def = _get_subscript_def(
                    0, row_end - row_start, 0, col_end - col_start, raw_bands, raw_band_dimension)
                data_segments.append(
                    SubsetSegment(child_segment, subset_def, 'raw', close_parent=True, squeeze=False))

            # determine arrangement of these children
            child_def = _get_subscript_def(
                row_start, row_end, col_start, col_end, raw_bands, raw_band_dimension)
            child_arrangement.append(child_def)

        return BlockAggregateSegment(
            data_segments, child_arrangement, 'raw', 0, raw_shape,
            formatted_dtype, formatted_shape, reverse_axes=reverse_axes,
            transpose_axes=transpose_axes, format_function=format_function,
            close_children=True)

    def _create_data_segment_from_imode_b(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'B':
            raise ValueError(
                'Requires IMODE = `B`, got `{}` at image segment index {}'.format(
                    image_header.IMODE, image_segment_index))
        if image_header.IC in ['NC', 'NM']:
            return self._handle_no_compression(image_segment_index, apply_format)
        else:
            raise ValueError('Got unhandled IC `{}`'.format(image_header.IC))

    def _create_data_segment_from_imode_p(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'P':
            raise ValueError(
                'Requires IMODE = `P`, got `{}` at image segment index {}'.format(
                    image_header.IMODE, image_segment_index))

        if image_header.IC in ['NC', 'NM']:
            return self._handle_no_compression(image_segment_index, apply_format)
        else:
            raise ValueError('Got unhandled IC `{}`'.format(image_header.IC))

    def _create_data_segment_from_imode_r(self, image_segment_index: int, apply_format: bool) -> DataSegment:
        image_header = self.get_image_header(image_segment_index)
        if image_header.IMODE != 'R':
            raise ValueError(
                'Requires IMODE = `R`, got `{}` at image segment index {}'.format(
                    image_header.IMODE, image_segment_index))

        if image_header.IC in ['NC', 'NM']:
            return self._handle_no_compression(image_segment_index, apply_format)
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

        Note that this also stores a reference to the produced data segment in
        the `_image_segment_data_segments` list.

        This will raise an exception if not performed in the order presented in
        the writing manager.

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

        image_manager = self.image_managers[image_segment_index]
        assert isinstance(image_manager, ImageSubheaderManager)
        if not self._in_memory:
            if image_manager.item_offset is None:
                raise ValueError(
                    'Performing file processing and item_offset unpopulated for '
                    'image segment at index {}'.format(image_segment_index))
        if len(self._image_segment_data_segments) != image_segment_index:
            raise ValueError('data segments must be constructed in order.')
        image_header = image_manager.subheader

        if image_header.IMODE == 'B':
            out = self._create_data_segment_from_imode_b(image_segment_index, apply_format)
        elif image_header.IMODE == 'P':
            out = self._create_data_segment_from_imode_p(image_segment_index, apply_format)
        elif image_header.IMODE == 'R':
            out = self._create_data_segment_from_imode_r(image_segment_index, apply_format)
        else:
            raise ValueError(
                'Got unsupported IMODE `{}` at image segment index `{}`'.format(
                    image_header.IMODE, image_segment_index))

        self._image_segment_data_segments.append(out)
        if image_manager.end_of_item is not None:
            if image_segment_index < len(self.image_managers) - 1:
                self.image_managers[image_segment_index + 1].subheader_offset = image_manager.end_of_item
        elif not self._in_memory:
            raise ValueError(
                'file processing, and item_size unpopulated for image segment at index {}'.format(image_segment_index))

        return out

    def create_data_segment_for_collection_element(self, collection_index: int) -> DataSegment:
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

        block_definition = numpy.array(self._get_collection_element_coordinate_limits(collection_index), dtype='int64')

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
        raw_shape = (total_rows, total_columns) if raw_bands == 1 else (total_rows, total_columns, raw_bands)

        formatted_shape = raw_shape[:2]
        if formatted_bands > 1:
            formatted_shape = formatted_shape + (formatted_bands, )

        return BlockAggregateSegment(
            child_segments, child_arrangement, 'raw', 0, raw_shape, formatted_dtype, formatted_shape,
            format_function=format_function, close_children=True)

    def get_data_segments(self) -> List[DataSegment]:
        """
        Gets a data segment for each of these image segment collection.

        Returns
        -------
        List[DataSegment]
        """

        out = []
        for index in range(len(self.image_segment_collections)):
            out.append(self.create_data_segment_for_collection_element(index))
        return out

    def flush(self, force: bool = False) -> None:
        self._validate_closed()

        BaseWriter.flush(self, force=force)

        try:
            if self._in_memory:
                if self._image_segment_data_segments is not None:
                    for index, entry in enumerate(self._image_segment_data_segments):
                        manager = self.nitf_writing_details.image_managers[index]
                        if manager.item_written:
                            continue
                        if manager.item_bytes is not None:
                            continue
                        if force or entry.check_fully_written(warn=force):
                            manager.item_bytes = entry.get_raw_bytes(warn=False)

            check = self.nitf_writing_details.verify_all_offsets(require=False)
            if check:
                self.nitf_writing_details.write_header(self._file_object, overwrite=False)
            self.nitf_writing_details.write_all_populated_items(self._file_object)
        except AttributeError:
            return

    def close(self) -> None:
        BaseWriter.close(self)  # NB: flush called here
        try:
            if self.nitf_writing_details is not None:
                self.nitf_writing_details.verify_all_written()
        except AttributeError:
            pass

        self._nitf_writing_details = None
        self._image_segment_data_segments = None
        self._file_object = None
