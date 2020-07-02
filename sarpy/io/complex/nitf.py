# -*- coding: utf-8 -*-
"""
Module laying out basic functionality for reading and writing NITF files. This is **intended**
to represent base functionality to be extended for SICD, CPHD, and SIDD capability.
"""

import logging
from typing import List, Tuple
import re

import numpy

from .base import BaseChipper, BaseReader, AbstractWriter, int_func
from .bip import BIPChipper, BIPWriter
from ..nitf.nitf_head import NITFDetails, NITFHeader, ImageSegmentsType, DataExtensionsType
from ..nitf.security import NITFSecurityTags
from ..nitf.image import ImageSegmentHeader
from ..nitf.des import DataExtensionHeader
from .sicd_elements.blocks import LatLonType
from sarpy.geometry.geocoords import ecf_to_geodetic, geodetic_to_ecf


class MultiSegmentChipper(BaseChipper):
    """
    Required chipping object to allow for the fact that a single image in a
    NITF file will often be broken up into a collection of image segments.
    """

    __slots__ = ('_file_name', '_data_size', '_dtype', '_complex_out',
                 '_symmetry', '_bounds', '_bands_ip', '_child_chippers')

    def __init__(self, file_name, bounds, data_offsets, data_type,
                 symmetry=None, complex_type=False, bands_ip=1):
        """

        Parameters
        ----------
        file_name : str
            The name of the file from which to read
        bounds : numpy.ndarray
            Two-dimensional array of [row start, row end, column start, column end]
        data_offsets : numpy.ndarray
            Offset for each image segment from the start of the file
        data_type : numpy.dtype
            The data type of the underlying file
        symmetry : tuple
            See `BaseChipper` for description of 3 element tuple of booleans.
        complex_type : callable|bool
            See `BaseChipper` for description of `complex_type`
        bands_ip : int
            number of bands - this will always be one for sicd.
        """

        if not isinstance(bounds, numpy.ndarray):
            raise ValueError('bounds must be an numpy.ndarray, not {}'.format(type(bounds)))
        if not (bounds.ndim == 2 and bounds.shape[1] == 4):
            raise ValueError('bounds must be an Nx4 numpy.ndarray, not shape {}'.format(bounds.shape))
        data_sizes = numpy.zeros((bounds.shape[0], 2), dtype=numpy.int64)
        p_row_start, p_row_end, p_col_start, p_col_end = None, None, None, None
        for i, entry in enumerate(bounds):
            # Are the order of the entries in bounds sensible?
            if not (0 <= entry[0] < entry[1] and 0 <= entry[2] < entry[3]):
                raise ValueError('entry {} of bounds is {}, and cannot be of the form '
                                 '[row start, row end, column start, column end]'.format(i, entry))

            # Are the elements of bounds sensible in relative terms?
            #   we must traverse by a specific block of columns until we reach the row limit,
            #   and then moving on the next segment of columns - note that this will almost
            #   always be a single block of columns only broken down in row order
            if i > 0:
                if not ((p_row_end == entry[0] and p_col_start == entry[2] and p_col_end == entry[3]) or
                        (p_col_end == entry[2] and entry[0] == 0)):
                    raise ValueError('The relative order for the chipper elements cannot be determined.')
            p_row_start, p_row_end, p_col_start, p_col_end = entry
            # define the data_sizes entry
            data_sizes[i, :] = (entry[1] - entry[0], entry[3] - entry[2])

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
        # all of the actual reading and reorienting work will be done by these
        # child chippers, which will read from their respective image segments
        self._child_chippers = tuple(
            BIPChipper(file_name, data_type, img_siz, symmetry=symmetry,
                       complex_type=complex_type, data_offset=img_off,
                       bands_ip=bands_ip)
            for img_siz, img_off in zip(data_sizes, data_offsets))
        self._bounds = bounds
        self._bands_ip = int_func(bands_ip)

        data_size = (self._bounds[-1, 1], self._bounds[-1, 3])
        # all of the actual reading and reorienting done by child chippers,
        # so do not reorient or change type at this level
        super(MultiSegmentChipper, self).__init__(data_size, symmetry=(False, False, False), complex_type=False)

    def _read_raw_fun(self, range1, range2):
        def subset(rng, start_ind, stop_ind):
            # find our rectangular overlap between the desired indices and chipper bounds
            if rng[2] > 0:
                if rng[1] < start_ind or rng[0] >= stop_ind:
                    return None, None
                # find smallest element rng[0] + mult*rng[2] which is >= start_ind
                mult1 = 0 if start_ind <= rng[0] else int_func(numpy.ceil((start_ind - rng[0])/rng[2]))
                ind1 = rng[0] + mult1*rng[2]
                # find largest element rng[0] + mult*rng[2] which is <= min(stop_ind, rng[1])
                max_ind = min(rng[1], stop_ind)
                mult2 = int_func(numpy.floor((max_ind - rng[0])/rng[2]))
                ind2 = rng[0] + mult2*rng[2]
            else:
                if rng[0] < start_ind or rng[1] >= stop_ind:
                    return None, None
                # find largest element rng[0] + mult*rng[2] which is <= stop_ind-1
                mult1 = 0 if rng[0] < stop_ind else int_func(numpy.floor((stop_ind - 1 - rng[0])/rng[2]))
                ind1 = rng[0] + mult1*rng[2]
                # find smallest element rng[0] + mult*rng[2] which is >= max(start_ind, rng[1]+1)
                mult2 = int_func(numpy.floor((start_ind - rng[0])/rng[2])) if rng[1] < start_ind \
                    else int_func(numpy.floor((rng[1] -1 - rng[0])/rng[2]))
                ind2 = rng[0] + mult2*rng[2]
            return (ind1, ind2, rng[2]), (mult1, mult2)

        range1, range2 = self._reorder_arguments(range1, range2)
        rows_size = int_func((range1[1]-range1[0])/range1[2])
        cols_size = int_func((range2[1]-range2[0])/range2[2])

        if self._bands_ip == 1:
            out = numpy.empty((rows_size, cols_size), dtype=numpy.complex64)
        else:
            out = numpy.empty((rows_size, cols_size, self._bands_ip), dtype=numpy.complex64)
        for entry, child_chipper in zip(self._bounds, self._child_chippers):
            row_start, row_end, col_start, col_end = entry
            # find row overlap for chipper - it's rectangular
            crange1, cinds1 = subset(range1, row_start, row_end)
            if crange1 is None:
                continue  # there is no row overlap for this chipper

            # find column overlap for chipper - it's rectangular
            crange2, cinds2 = subset(range2, col_start, col_end)
            if crange2 is None:
                continue  # there is no column overlap for this chipper

            if self._bands_ip == 1:
                out[cinds1[0]:cinds1[1], cinds2[0]:cinds2[1]] = child_chipper(crange1, crange2)
            else:
                out[cinds1[0]:cinds1[1], cinds2[0]:cinds2[1], :] = child_chipper(crange1, crange2)
        return out


class NITFReader(BaseReader):
    """
    A reader object for **something** in a NITF 2.10 container
    """

    __slots__ = ('_nitf_details', )

    def __init__(self, nitf_details):
        """

        Parameters
        ----------
        nitf_details : NITFDetails
            The NITFDetails object
        """

        if not isinstance(nitf_details, NITFDetails):
            raise TypeError('The input argument for NITFReader must be a NITFDetails object.')
        self._nitf_details = nitf_details

        # get sicd structure
        if hasattr(nitf_details, 'sicd_meta'):
            sicd_meta = nitf_details.sicd_meta
        else:
            sicd_meta = None
        self._sicd_meta = sicd_meta
        # determine image segmentation from image headers
        segments = self._find_segments()
        # construct the chippers
        chippers = tuple(self._construct_chipper(segment, i) for i, segment in enumerate(segments))
        # construct regularly
        super(NITFReader, self).__init__(sicd_meta, chippers)

    @property
    def file_name(self):
        return self._nitf_details.file_name

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
        index : int

        Returns
        -------
        MultiSegmentChipper
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
        dtype : str|numpy.dtype
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
