# -*- coding: utf-8 -*-

import os
import logging
import sys
from xml.etree import ElementTree
from collections import OrderedDict

import numpy

from .base import BaseChipper, BaseReader, BaseWriter
from .bip import BIPChipper
from ..sicd_elements.SICD import SICDType


def amp_phase_conversion_function(lookup_table):
    """
    The constructs the function to convert from AMP8I_PHS8I format data to complex128 data.

    Parameters
    ----------
    lookup_table

    Returns
    -------
    callable
    """

    def converter(data):
        amp = lookup_table[data[0::2, :, :]]
        theta = data[1::2, :, :]*2*numpy.pi/256
        out = numpy.zeros((data.shape(0) / 2, data.shape(1), data.shape(2)), dtype=numpy.complex128)
        out.real = amp*numpy.cos(theta)
        out.imag = amp*numpy.sin(theta)
        return out
    return converter


class BaseScraper(object):
    """Describes the abstract functionality"""

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def minimum_length(cls):
        """
        The minimum size it takes to write such a header.

        Returns
        -------
        int
        """

        raise NotImplementedError

    @classmethod
    def from_string(cls, value, start, *args):
        """

        Parameters
        ----------
        value: bytes|str
            the header string to scrape

        start : int
            the beginning location in the string
        args : list
            other argumemnts

        Returns
        -------
        """

        raise NotImplementedError

    def to_bytes(self):
        """
        Write the object to a bytes array, suitable for outputing to a file object.

        Returns
        -------
        bytes
        """

        raise NotImplementedError


class HeaderScraper(BaseScraper):
    __slots__ = ()  # the possible attribute collection
    _types = {}  # for elements which will be scraped by another class
    _args = {}  # for elements which will be scraped by another class, these are args for the from_string method
    _lengths = {}  # form attribute: <length> for each entry in __slots__ not in types
    _defaults = {}  # any default values.
    # If a given attribute doesn't have a give default it is assumed that all spaces is the default

    def __init__(self, **kwargs):
        unmatched = [key for key in kwargs if key not in self._defaults]
        if len(unmatched) > 0:
            raise KeyError('Arguments {} are not in the list of allowed '
                           'attributes {}'.format(unmatched, self.__slots__))
        for attribute in self.__slots__:
            if attribute not in kwargs:
                if attribute in self._types:
                    setattr(self, attribute, self._types[attribute]())
                elif attribute in self._defaults:
                    setattr(self, attribute, self._defaults[attribute])
                else:
                    setattr(self, attribute, ' '*self._lengths[attribute])
            else:
                val = kwargs[attribute]
                if attribute not in self._types and not isinstance(val, str):
                    raise ValueError('This requires that argument {} be string '
                                     'instance, and got {}'.format(attribute, type(val)))
                setattr(self, attribute, val)

    def __len__(self):
        length = 0
        for attribute in self.__slots__:
            if attribute in self._types:
                val = getattr(self, attribute)
                length += val.length
            else:
                length += self._lengths[attribute]
        return length

    @classmethod
    def minimum_length(cls):
        min_length = 0
        for attribute in cls.__slots__:
            if attribute in cls._types:
                typ = cls._types[attribute]
                if issubclass(typ, BaseScraper):
                    min_length += typ.minimum_length()
            else:
                min_length += cls._lengths[attribute]
        return min_length

    @classmethod
    def from_string(cls, value, start, *args):
        if value is None:
            return cls()
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if not isinstance(value, str):
            raise TypeError('Requires a bytes or str type input, got {}'.format(type(value)))

        min_length = cls.minimum_length()
        if len(value[start:]) < min_length:
            raise TypeError('Requires a bytes or str type input of length at least {}, '
                            'got {}'.format(min_length, len(value[start:])))

        fields = OrderedDict()
        loc = start
        for attribute in cls.__slots__:
            if attribute in cls._types:
                typ = cls._types[attribute]
                if not issubclass(typ, BaseScraper):
                    raise TypeError('Invalid class definition, any entry of _types must extend BaseScraper')
                args = cls._args.get(attribute, [])
                val = typ.from_string(value, loc, *args)
                lngt = len(val)
            else:
                lngt = cls._lengths[attribute]
                fields[attribute] = value[loc:lngt]
            loc += lngt
        return cls(**fields)

    def to_bytes(self):
        out = ''
        for attribute in self.__slots__:
            flen = self._lengths[attribute]
            fstr = '{0:' + flen + 's}'
            val = getattr(self, attribute, ' ')
            if len(val) <= flen:
                out += fstr.format(val)
            else:
                out += val[:flen]
        return out.encode('utf-8')


class ItemArrayHeaders(BaseScraper):
    """Item array in the NITF header (i.e. Image Segment, Text Segment)"""
    __slots__ = ('subhead_len', 'subhead_sizes', 'item_len', 'item_sizes')

    def __init__(self, subhead_len, subhead_sizes, item_len, item_sizes):
        if subhead_sizes.shape != item_sizes.shape or len(item_sizes.shape) != 1:
            raise ValueError(
                'the subhead_offsets and item_offsets arrays must one-dimensional and the same length')
        self.subhead_len = subhead_len
        self.subhead_sizes = subhead_sizes
        self.item_len = item_len
        self.item_sizes = item_sizes

    def __len__(self):
        return 3 + (self.subhead_len + self.item_len)*self.subhead_sizes.size

    @classmethod
    def minimum_length(cls):
        return 3

    @classmethod
    def from_string(cls, value, start, *args):
        """

        Parameters
        ----------
        value : bytes|str
        start : int
        args : iterable
            this must have two integer elements, which designate the number of
            bytes for the subheader size and the number of bytes for the item size
        Returns
        -------
        ItemArrayHeaders
        """

        subhead_size, item_size = args
        subhead_size, item_size = int(subhead_size), int(item_size)
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if not isinstance(value, str):
            raise TypeError('Requires a bytes or str type input, got {}'.format(type(value)))

        min_length = cls.minimum_length()
        if len(value[start:]) < min_length:
            raise TypeError('Requires a bytes or str type input of length at least {}, '
                            'got {}'.format(min_length, len(value[start:])))
        loc = start
        count = int(value[loc:loc+3])
        loc += 3
        subhead_offsets = numpy.array((count, ), dtype=numpy.int64)
        item_offsets = numpy.array((count, ), dtype=numpy.int64)
        for i in range(count):
            subhead_offsets[i] = int(value[loc: loc+subhead_size])
            loc += subhead_size
            item_offsets[i] = int(value[loc: loc+item_size])
            loc += item_size

    def to_bytes(self):
        out = '{0:3d}'.format(self.subhead_sizes.size)
        subh_frm = '{0:'+self.subhead_len + 'd}'
        item_frm = '{0:' + self.item_len + 'd}'
        for sh_off, it_off in zip(self.subhead_sizes, self.item_sizes):
            out += subh_frm.format(sh_off) + item_frm.format(it_off)
        return out.encode('utf-8')


class OtherHeader(BaseScraper):
    __slots__ = ('overflow', 'header')

    def __init__(self, overflow=None, header=None):
        if overflow is None:
            self.overflow = 0
        else:
            self.overflow = int(overflow)
        if header is None:
            self.header = header
        elif not isinstance(header, str):
            raise ValueError('header must be of string type')
        elif len(header) > 99991:
            logging.warning('Other header will be truncated to 99,9991 characters from {}'.format(header))
            self.header = header[:99991]
            self.overflow += len(header) - 99991
        else:
            self.header = header

    def __len__(self):
        length = 5
        if self.header is not None:
            length += 3 + len(self.header)
        return length

    @classmethod
    def minimum_length(cls):
        return 5

    @classmethod
    def from_string(cls, value, start, *args):
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if not isinstance(value, str):
            raise TypeError('Requires a bytes or str type input, got {}'.format(type(value)))

        if len(value[start:]) < 5:
            raise TypeError('Requires a bytes or str type input of length at least 5, '
                            'got {}'.format(len(value[start:])))

        siz = int(value[start:start+5])
        if siz == 0:
            return cls()
        else:
            overflow = int(value[start+5:start+8])
            header = value[start+8:start+siz]
            return cls(overflow=overflow, header=header)

    def to_bytes(self):
        out = '{0:5d}'.format(self.__len__())
        if self.header is not None:
            if self.overflow > 999:
                out += '999'
            else:
                out += '{0:3d}'.format(self.overflow)
            out += self.header
        return out.encode('utf-8')


class NITFSecurityTags(HeaderScraper):
    """The NITF security tags - described in SICD standard 2014-09-30, Volume II, page 20"""
    __slots__ = (
        'CLAS', 'CLSY', 'CODE', 'CTLH',
        'REL', 'DCTP', 'DCDT', 'DCXM',
        'DG', 'DGDT', 'CLTX', 'CAPT',
        'CAUT', 'CRSN', 'SRDT', 'CTLN')
    _lengths = {
        'CLAS': 1, 'CLSY': 2, 'CODE': 11, 'CTLH': 2,
        'REL': 20, 'DCTP': 2, 'DCDT': 8, 'DCXM': 4,
        'DG': 1, 'DGDT': 8, 'CLTX': 43, 'CAPT': 1,
        'CAUT': 40, 'CRSN': 1, 'SRDT': 8, 'CTLN': 15}
    _defaults = {'CLAS': 'U'}


class ImageSegments(HeaderScraper):
    """The NITF file header - described in SICD standard 2014-09-30, Volume II, page 17"""
    __slots__ = ('NUMI', 'ImageSegments')


class NITFHeader(HeaderScraper):
    """The NITF file header - described in SICD standard 2014-09-30, Volume II, page 17"""
    __slots__ = (
        'FHDR', 'FVER', 'CLEVEL', 'STYPE',
        'OSTAID', 'FDT', 'FTITLE', 'Security',
        'FSCOP', 'FSCPYS', 'ENCRYP', 'FBKGC',
        'ONAME', 'OPHONE', 'FL', 'HL',
        'ImageSegments', 'NUMS', 'NUMX',
        'TextSegments', 'DataExtensions', 'NUMRES',
        'UserHeader', 'ExtendedHeader')
    _types = {
        'Security': NITFSecurityTags,
        'ImageSegments': ItemArrayHeaders,
        'TextSegments': ItemArrayHeaders,
        'DataExtensions': ItemArrayHeaders,
        'UserHeader': OtherHeader,
        'ExtendedHeader': OtherHeader, }
    _args = {
        'ImageSegments': (6, 10),  # these are the sizes for the image subheader and image size info
        'TextSegments': (4, 5),
        'DataExtensions': (4, 9), }
    _lengths = {
        'FHDR': 4, 'FVER': 5, 'CLEVEL': 2, 'STYPE': 4,
        'OSTAID': 10, 'FDT': 14, 'FTITLE': 80,
        'FSCOP': 5, 'FSCPYS': 5, 'ENCRYP': 1, 'FBKGC': 3,
        'ONAME': 24, 'OPHONE': 18, 'FL': 12, 'HL': 6,
        'NUMS': 3, 'NUMX': 3, 'NUMRES': 3, 'UDHDL': 5, }
    _defaults = {
        'FHDR': 'NITF', 'FVER': '02.10', 'STYPE': 'BF01',
        'OSTAID': 'Unknown', 'FSCOP': '00000', 'FSCPYS': '00000',
        'ENCRYP': '0', 'FBKGC': '000', 'NUMS': '000', 'NUMX': '000',
        'NUMRES': '000', 'UDHDL': '00000', }


# TODO: ImageSegmentHeader page 24 of SICD 2014-09-30 Volume II


class NITFOffsets(object):
    """
    SICD (versions 0.3 and above) are stored in NITF 2.1 files, but only a small,
    specific portion of the NITF format is allowed for SICD. This class
    provides a simple method to extract that relevant data subset.
    """

    __slots__ = ('_file_name', 'img_segment_offsets', 'img_segment_rows',
                 'img_segment_columns', 'des_lengths', 'des_offsets', '_is_sicd',
                 '_sicd_meta_data')

    # TODO: rehash this using the NITFHeader parser above.

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF 2.1 file containing a SICD
        extract_sicd : bool
            should we extract the full sicd metadata?
        """

        with open(file_name, mode='rb') as fi:
            # NB: everything in the below assumes that this section of the file
            #   is actually ascii, and will be properly interpreted as ints and
            #   so forth. There are lots of places that may result in some kind
            #   of exception, all of which constitute a failure, and will be
            #   uncaught.

            # Read the first 9 bytes to verify NITF
            head_part = fi.read(9).decode('ascii')
            if head_part != 'NITF02.10':
                raise IOError('Note a NITF 2.1 file, and cannot contain a SICD')
            # TODO: these 354 bytes should probably be parsed
            fi.seek(354)  # offset to first field of interest
            header_length = int(fi.read(6))
            image_segment_count = int(fi.read(3))
            image_segment_subhdr_lengths = numpy.zeros((image_segment_count, ), dtype=numpy.int64)
            image_segment_data_lengths = numpy.zeros((image_segment_count, ), dtype=numpy.int64)
            # the image data in this file is packed as:
            #   header:im1_header:im1_data:im2_header:im2_data:...
            for i in range(image_segment_count):
                image_segment_subhdr_lengths[i] = int(fi.read(6))
                image_segment_data_lengths[i] = int(fi.read(10))
            # Offset to given image segment data from beginning of file in bytes
            self.img_segment_offsets = header_length + numpy.cumsum(image_segment_subhdr_lengths)
            self.img_segment_offsets[1:] += numpy.cumsum(image_segment_data_lengths[:-1])

            if int(fi.read(3)) > 0:
                raise IOError('SICD does not allow for graphics segments.')
            if int(fi.read(3)) > 0:
                raise IOError('SICD does not allow for reserved extension segments.')
            # text segments get packed next, we really only need the size to skip
            text_segment_count = int(fi.read(3))
            text_size = 0
            for i in range(text_segment_count):
                text_size += int(fi.read(4))  # text header length
                text_size += int(fi.read(5))  # text data length
            # data extensions get packed next, we need these
            des_count = int(fi.read(3))  # this really should only be one.
            des_subhdr_lengths = numpy.zeros((des_count, ), dtype=numpy.int64)
            # TODO: we should have an object for an image subheaders and (the sicd) data extension subheader
            #   We have to write them, so it makes sense to operate through useful objects.
            self.des_lengths = numpy.zeros((des_count, ), dtype=numpy.int64)
            for i in range(des_count):
                des_subhdr_lengths[i] = int(fi.read(4))
                self.des_lengths[i] = int(fi.read(9))
            self.des_offsets = \
                self.img_segment_offsets[-1] + image_segment_data_lengths[-1] + text_size + \
                numpy.cumsum(des_subhdr_lengths)
            self.des_offsets[1:] += numpy.cumsum(self.des_lengths[:-1])

            # Number of rows in the given image segment
            self.img_segment_rows = numpy.zeros((image_segment_count,), dtype=numpy.int32)
            # Number of cols in the given image segment
            self.img_segment_columns = numpy.zeros((image_segment_count,), dtype=numpy.int32)
            # Extract these from the respective image headers now
            for i in range(image_segment_count):
                # go to 333 bytes into the ith image header - should parse these headers more effectively
                fi.seek(self.img_segment_offsets[i] - image_segment_subhdr_lengths[i] + 333)
                self.img_segment_rows[i] = int(fi.read(8))
                self.img_segment_columns[i] = int(fi.read(8))

            # SICD Volume 2, File Format Description, section 3.1.1 says that SICD XML
            # metadata must be stored in first Data Extension Segment.
            # Implicitly, this is logically a single collect, so there can't
            # be more than one SICD element.
            self._is_sicd = False
            self._sicd_meta_data = None

            # should we parse the data extension header too? Probably.
            fi.seek(self.des_offsets[0])
            data_extension = fi.read(self.des_lengths[0])
            # do I need to decode this, or will it handle bytes in Python 3?
            try:
                root_node = ElementTree.fromstring(data_extension)
                if root_node.tag.split('}', 1)[-1] == 'SICD':
                    # TODO: I feel like we can do better than the above
                    self._is_sicd = True
            except Exception:
                return

            if self._is_sicd:
                self._sicd_meta_data = SICDType.from_node(root_node.find('SICD'))

    @property
    def is_sicd(self):
        return self._is_sicd

    @property
    def sicd_meta_data(self):
        return self._sicd_meta_data


class MultiSegmentChipper(BaseChipper):
    """
    This is required because NITF files have specified size and row/column limits.
    Any sufficiently large SICD collect must be broken into a series of image segments (along rows).
    We must have a parser to transparently extract data between this collection of image segments.
    """
    __slots__ = ('_file_name', '_data_size', '_dtype', '_complex_out',
                 '_symmetry', '_swap_bytes', '_row_starts', '_row_ends',
                 '_bands_ip', '_child_chippers')

    def __init__(self, file_name, data_sizes, data_offsets, data_type, symmetry=None,
                 complex_type=True, swap_bytes=False, bands_ip=1):
        """

        Parameters
        ----------
        file_name : str
            The name of the file from which to read
        data_sizes : numpy.ndarray
            Two-dimensional array of [row, col] sizes
        data_offsets : numpy.ndarray
            Offset for each image segment from the start of the file
        data_type : numpy.dtype
            The data type of the underlying file
        symmetry : tuple
            See `BaseChipper` for description of 3 element tuple of booleans.
        complex_type : callable|bool
            See `BaseChipper` for description of `complex_type`
        swap_bytes : bool
            are the endian-ness of the os and file different?
        bands_ip : int
            number of bands - this will always be one for sicd.
        """

        if not isinstance(data_sizes, numpy.ndarray):
            raise ValueError('data_sizes must be an numpy.ndarray, not {}'.format(type(data_sizes)))
        if not (len(data_sizes.shape) == 2 and data_sizes.shape[1] == 2):
            raise ValueError('data_sizes must be an Nx2 numpy.ndarray, not shape {}'.format(data_sizes.shape))
        if not numpy.all(data_sizes[0, :] == data_sizes[0, 1]):
            raise ValueError(
                'SICDs are broken up by row only, so the the second column of data_sizes '
                'must be have constant value {}'.format(data_sizes))

        if not isinstance(data_offsets, numpy.ndarray):
            raise ValueError('data_offsets must be an numpy.ndarray, not {}'.format(type(data_offsets)))
        if not (len(data_offsets.shape) == 1):
            raise ValueError(
                'data_sizes must be an one-dimensional numpy.ndarray, '
                'not shape {}'.format(data_offsets.shape))

        if data_sizes.shape[0] != data_offsets.size:
            raise ValueError(
                'data_sizes and data_offsets arguments must have compatible '
                'shape {} - {}'.format(data_sizes.shape, data_sizes.size))

        self._file_name = file_name
        if complex_type is not False:
            self._dtype = numpy.complex64
        else:
            self._dtype = data_type
        # all of the actual reading and reorienting work will be done by these
        # child chippers, which will read from their respective image segments
        self._child_chippers = tuple(
            BIPChipper(file_name, data_type, img_siz, symmetry=symmetry,
                       complex_type=complex_type, data_offset=img_off,
                       swap_bytes=swap_bytes, bands_ip=bands_ip)
            for img_siz, img_off in zip(data_sizes, data_offsets))

        self._row_starts = numpy.zeros((data_sizes.shape[0], ), dtype=numpy.int64)
        self._row_ends = numpy.cumsum(data_sizes[: 0])
        self._row_starts[1:] = self._row_ends[:-1]
        self._bands_ip = int(bands_ip)

        data_size = (self._row_ends[-1], data_sizes[0, 1])
        # all of the actual reading and reorienting done by child chippers,
        # so do not reorient or change type at this level
        super(MultiSegmentChipper, self).__init__(data_size, symmetry=(False, False, False), complex_type=False)

    def _read_raw_fun(self, range1, range2):
        range1, range2 = self._reorder_arguments(range1, range2)
        # this method just assembles the final data from the child chipper pieces
        rows = numpy.arange(*range1, dtype=numpy.int64)  # array
        cols_size = int((range2[1] - range2[0] - 1)/range2[2]) + 1
        if self._bands_ip == 1:
            out = numpy.empty((rows.size, cols_size), dtype=self._dtype)
        else:
            out = numpy.empty((self._bands_ip, rows.size, cols_size), dtype=numpy.complex64)
        for row_start, row_end, child_chipper in zip(self._row_starts, self._row_ends, self._child_chippers):
            row_bool = ((rows >= row_start) & (rows < row_end))
            if numpy.any(row_bool):
                crows = rows[row_bool]
                crange1 = (crows[0]-row_start, crows[-1]+1-row_start, range1[2])
                if self._bands_ip == 1:
                    out[row_bool, :] = child_chipper(crange1, range2)
                else:
                    out[:, row_bool, :] = child_chipper(crange1, range2)
        return out


class SICDReader(BaseReader):
    __slots__ = ('_nitf_offsets', '_sicd_meta', '_chipper')

    def __init__(self, file_name):
        self._nitf_offsets = NITFOffsets(file_name)
        if not self._nitf_offsets.is_sicd:
            raise ValueError(
                'The input file passed in appears to be a NITF 2.1 file that does not contain valid sicd metadata.')

        self._sicd_meta = self._nitf_offsets.sicd_meta_data

        pixel_type = self._sicd_meta.ImageData.PixelType
        complex_type = True
        if pixel_type == 'RE32F_IM32F':
            dtype = numpy.float32
        elif pixel_type == 'RE16I_IM16I':
            dtype = numpy.int16
        elif pixel_type == 'AMP8I_PHS8I':
            dtype = numpy.uint8
            complex_type = amp_phase_conversion_function(self._sicd_meta.ImageData.AmpTable)
            # TODO: is this above right?
            # raise ValueError('Pixel Type `AMP8I_PHS8I` is not currently supported.')
        else:
            raise ValueError('Pixel Type {} not recognized.'.format(pixel_type))

        data_sizes = numpy.column_stack(
            (self._nitf_offsets.img_segment_rows, self._nitf_offsets.img_segment_columns), dtype=numpy.int32)
        # SICDs are required to be stored as big-endian
        swap_bytes = (sys.byteorder != 'big')
        # SICDs require no symmetry reorientation
        symmetry = (False, False, False)
        # construct our chipper
        chipper = MultiSegmentChipper(
            file_name, data_sizes, self._nitf_offsets.img_segment_rows.copy(), dtype,
            symmetry=symmetry, complex_type=complex_type, swap_bytes=swap_bytes, bands_ip=1)

        super(SICDReader, self).__init__(self._nitf_offsets.sicd_meta_data, chipper)


# TODO: complete SICDWriter - make sure it's coherent
