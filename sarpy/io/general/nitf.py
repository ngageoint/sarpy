# -*- coding: utf-8 -*-
"""
Module laying out basic functionality for reading and writing NITF files.
This is **intended** to represent base functionality to be extended for
SICD and SIDD capability.
"""

import logging
import os
from typing import Union, List, Tuple
import re

import numpy

from sarpy.compliance import int_func
from .base import BaseReader, AbstractWriter
from .bip import BIPWriter
# noinspection PyProtectedMember
from .nitf_elements.nitf_head import NITFHeader, ImageSegmentsType, \
    DataExtensionsType, _ItemArrayHeaders
from .nitf_elements.text import TextSegmentHeader
from .nitf_elements.graphics import GraphicsSegmentHeader
from .nitf_elements.res import ReservedExtensionHeader
from .nitf_elements.security import NITFSecurityTags
from .nitf_elements.image import ImageSegmentHeader
from .nitf_elements.des import DataExtensionHeader
from ..complex.sicd_elements.blocks import LatLonType
from sarpy.geometry.geocoords import ecf_to_geodetic, geodetic_to_ecf


#####
# A general nitf header interpreter - intended for extension

class NITFDetails(object):
    """
    This class allows for somewhat general parsing of the header information in a NITF 2.1 file.
    """

    __slots__ = (
        '_file_name', '_nitf_header', '_img_headers',
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
    def nitf_header(self):
        # type: () -> NITFHeader
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


class NITFReader(BaseReader):
    """
    A reader object for **something** in a NITF 2.10 container
    """

    __slots__ = ('_nitf_details', )

    def __init__(self, nitf_details, is_sicd_type=False):
        """

        Parameters
        ----------
        nitf_details : NITFDetails
            The NITFDetails object
        is_sicd_type : bool
            Is this a sicd type reader, or otherwise?
        """

        if not isinstance(nitf_details, NITFDetails):
            raise TypeError('The input argument for NITFReader must be a NITFDetails object.')
        self._nitf_details = nitf_details

        # get sicd structure
        if hasattr(nitf_details, 'sicd_meta'):
            sicd_meta = nitf_details.sicd_meta
        else:
            sicd_meta = None

        # this will redundantly set the _sicd_meta value with the super call,
        # but that is potentially appropriate here for the _find_segments() call.
        self._sicd_meta = sicd_meta
        # determine image segmentation from image headers
        segments = self._find_segments()
        # construct the chippers
        chippers = tuple(self._construct_chipper(segment, i) for i, segment in enumerate(segments))
        # construct regularly
        super(NITFReader, self).__init__(sicd_meta, chippers, is_sicd_type=is_sicd_type)

    @property
    def nitf_details(self):
        """
        NITFDetails: The NITF details object.
        """

        return self._nitf_details

    @property
    def file_name(self):
        return self._nitf_details.file_name

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
        (numpy.ndarray, numpy.ndarray)
            The arrays indicating the chipper partitioning for bounds (of the form
            `[row start, row end, column start, column end]`) and byte offsets.
        """

        bounds = []
        offsets = []

        # verify that (at least) abpp and band count are constant
        abpp, band_count, bytes_per_pixel = None, None, None
        p_row_start, p_row_end, p_col_start, p_col_end = None, None, None, None
        for i, index in enumerate(segment):
            # get this image subheader
            img_header = self.nitf_details.img_headers[index]

            # check for compression
            if img_header.IC != 'NC':
                raise ValueError('Image header at index {} has IC {}. No compression '
                                 'is supported at this time.'.format(index, img_header.IC))

            # check bits per pixel and number of bands
            if abpp is None:
                abpp = img_header.ABPP
                if abpp not in (8, 16, 32):
                    raise ValueError(
                        'Image segment {} has bits per pixel per band {}, only 8, 16, and 32 are supported.'.format(index, abpp))
                band_count = len(img_header.Bands)
                bytes_per_pixel = int_func(abpp*band_count/8)
            elif img_header.ABPP != abpp:
                raise ValueError(
                    'NITF image segment at index {} has ABPP {}, but this is different than the value {}'
                    'previously determined for this composite image'.format(index, img_header.ABPP, abpp))
            elif len(img_header.Bands) != band_count:
                raise ValueError(
                    'NITF image segment at index {} has band count {}, but this is different than the value {}'
                    'previously determined for this composite image'.format(index, len(img_header.Bands), band_count))

            # get the bytes offset for this nitf image segment
            this_rows, this_cols = img_header.NROWS, img_header.NCOLS
            if this_rows > rows or this_cols > cols:
                raise ValueError(
                    'NITF image segment at index {} has size ({}, {}), and cannot be part of an image of size '
                    '({}, {})'.format(index, this_rows, this_cols, rows, cols))

            # horizontal block details
            horizontal_block_size = this_rows
            if img_header.NBPR != 1:
                if (this_cols % img_header.NBPR) != 0:
                    raise ValueError(
                        'The number of blocks per row is listed as {}, but this is '
                        'not equally divisible into the number of columns {}'.format(img_header.NBPR, this_cols))
                horizontal_block_size = int_func(this_cols/img_header.NBPR)

            # vertical block details
            vertical_block_size = this_cols
            if img_header.NBPC != 1:
                if (this_rows % img_header.NBPC) != 0:
                    raise ValueError(
                        'The number of blocks per column is listed as {}, but this is '
                        'not equally divisible into the number of rows {}'.format(img_header.NBPC, this_rows))
                vertical_block_size = int_func(this_rows/img_header.NBPC)

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

            # iterate over our blocks and populate the bounds and offsets
            block_col_end = cur_col_start
            current_offset = self.nitf_details.img_segment_offsets[index]
            for vblock in range(img_header.NBPC):
                block_col_start = block_col_end
                block_col_end += vertical_block_size
                block_row_end = cur_row_start
                for hblock in range(img_header.NBPR):
                    block_row_start = block_row_end
                    block_row_end += horizontal_block_size
                    bounds.append((block_row_start, block_row_end, block_col_start, block_col_end))
                    offsets.append(current_offset)
                    current_offset += horizontal_block_size * vertical_block_size * bytes_per_pixel
            p_row_start, p_row_end, p_col_start, p_col_end = cur_row_start, cur_row_end, cur_col_start, cur_col_end
        return numpy.array(bounds, dtype=numpy.int64), numpy.array(offsets, dtype=numpy.int64)

    def _find_segments(self):
        """
        Determine the image segment collections.

        Returns
        -------
        List[List[int]]
        """

        raise NotImplementedError

    def _construct_chipper(self, segment, index):
        """
        Construct the appropriate multi-segment chipper given the list of image
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
        sarpy.io.general.bip.MultiSegmentChipper
        """

        raise NotImplementedError


class ImageDetails(object):
    """
    Helper class for managing the details about a given NITF segment.
    """

    __slots__ = (
        '_bands', '_dtype', '_complex_type', '_parent_index_range',
        '_subheader', '_subheader_offset', '_item_offset',
        '_subheader_written', '_pixels_written')

    def __init__(self, bands, dtype, complex_type, parent_index_range, subheader):
        """

        Parameters
        ----------
        bands : int
            The number of bands.
        dtype : str|numpy.dtype|numpy.number
            The dtype for the associated chipper.
        complex_type : bool|callable
            The complex_type for the associated chipper.
        parent_index_range : Tuple[int]
            Indicates `(start row, end row, start column, end column)` relative to
            the parent image.
        subheader : ImageSegmentHeader
            The image subheader.
        """

        self._subheader_offset = None
        self._item_offset = None
        self._pixels_written = int_func(0)
        self._subheader_written = False

        self._bands = int_func(bands)
        if self._bands <= 0:
            raise ValueError('bands must be positive.')
        self._dtype = dtype
        self._complex_type = complex_type

        if len(parent_index_range) != 4:
            raise ValueError('parent_index_range must have length 4.')
        self._parent_index_range = (
            int_func(parent_index_range[0]), int_func(parent_index_range[1]),
            int_func(parent_index_range[2]), int_func(parent_index_range[3]))

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
            logging.warning("subheader_offset is read only after being initially defined.")
            return
        self._subheader_offset = int_func(value)
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

        return int_func(self.total_pixels*self.subheader.ABPP*len(self.subheader.Bands)/8)

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
            logging.error('A total of {} pixels have been written for an image that '
                          'should only have {} pixels.'.format(self._pixels_written, self.total_pixels))

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
                ed = min(int_func(this_end), parent_end)
            elif parent_start <= this_start <= parent_end:
                st = int_func(this_start)
                ed = min(int_func(this_end), parent_end)
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
            file_name, (self.rows, self.cols), self._dtype,
            self._complex_type, data_offset=self.item_offset)


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
            logging.warning("subheader_offset is read only after being initially defined.")
            return
        self._subheader_offset = int_func(value)
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
    Determine the appropriate segmentation for the image.

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

    im_seg_limit = 10**10 - 2  # as big as can be stored in 10 digits, given at least 2 bytes per pixel
    dim_limit = 10**5 - 1  # as big as can be stored in 5 digits
    im_segments = []

    row_offset = 0
    col_offset = 0
    col_limit = min(dim_limit, cols)
    while (row_offset < rows) and (col_offset < cols):
        # determine row count, given row_offset, col_offset, and col_limit
        # how many bytes per row for this column section
        row_memory_size = (col_limit - col_offset) * pixel_size
        # how many rows can we use
        row_count = min(dim_limit, rows - row_offset, int_func(im_seg_limit / row_memory_size))
        im_segments.append((row_offset, row_offset + row_count, col_offset, col_limit))
        row_offset += row_count  # move the next row offset
        if row_offset == rows:
            # move over to the next column section
            col_offset = col_limit
            col_limit = min(col_offset + dim_limit, cols)
            row_offset = 0
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

    def __init__(self, file_name):
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

        logging.info('Writing NITF header.')
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

        logging.info(
            'Setting up the image segments in virtual memory. '
            'This may require a large physical memory allocation, '
            'and be time consuming.')
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

        logging.info(
            'Writing image segment {} header. Depending on OS details, this '
            'may require a large physical memory allocation, '
            'and be time consuming.'.format(index))
        with open(self._file_name, mode='r+b') as fi:
            fi.seek(details.subheader_offset)
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

        logging.info(
            'Writing data extension {} header.'.format(index))
        with open(self._file_name, mode='r+b') as fi:
            fi.seek(details.subheader_offset)
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

        logging.info(
            'Writing data extension {}.'.format(index))
        with open(self._file_name, mode='r+b') as fi:
            fi.seek(details.item_offset)
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
            elif file_size < 2*(int_func(1024)**3):
                return 6
            elif file_size < 10*(int_func(1024)**3):
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
        start_indices = (int_func(start_indices[0]), int_func(start_indices[1]))
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

            overall_inds, this_inds = details.get_overlap(index_range)
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
        if self.image_details is not None:
            for i, img_details in enumerate(self.image_details):
                if not img_details.image_written:
                    logging.critical("This NITF file in not completely written and will be corrupt. "
                                     "Image segment {} has only written {} "
                                     "or {}".format(i, img_details.pixels_written, img_details.total_pixels))
        # ensure that all data extensions are fully written
        if self.des_details is not None:
            for i, des_detail in enumerate(self.des_details):
                if not des_detail.des_written:
                    self._write_des_bytes(i)
        # close all the chippers
        if self._writing_chippers is not None:
            for entry in self._writing_chippers:
                entry.close()

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
            logging.warning(
                "This NITF has no previously defined image segments, or the "
                "_create_nitf_header method has been called BEFORE the "
                "_create_image_segment_headers method.")
        if self._des_details is None:
            logging.warning(
                "This NITF has no previously defined data extensions, or the "
                "_create_nitf_header method has been called BEFORE the "
                "_create_data_extension_headers method.")

        # NB: CLEVEL and FL will be corrected in prepare_for_writing method
        self._nitf_header = NITFHeader(
            Security=self.security_tags, CLEVEL=3, OSTAID=self._get_ostaid(),
            FDT=self._get_fdt(), FTITLE=self._get_ftitle(), FL=0,
            ImageSegments=self._get_nitf_image_segments(),
            DataExtensions=self._get_nitf_data_extensions())
