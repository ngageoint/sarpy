# -*- coding: utf-8 -*-
"""
Module for reading and writing SIDD files - should support SIDD version 1.0 and above.
"""

import logging

import numpy

from .utils import parse_xml_from_string
from .base import int_func, string_types
from .nitf import NITFReader, NITFWriter, ImageDetails, DESDetails
from ..nitf.nitf_head import NITFDetails, NITFHeader, ImageSegmentsType, DataExtensionsType
from ..nitf.des import DataExtensionHeader, SICDDESSubheader
from ..nitf.security import NITFSecurityTags
from ..nitf.image import ImageSegmentHeader, ImageBands, ImageBand
from .sidd_elements.SIDD import SIDDType
from .sicd_elements.SICD import SICDType
from .sicd import MultiSegmentChipper


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a SIDD file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    SIDDReader|None
        `SIDDReader` instance if SIDD file, `None` otherwise
    """

    try:
        nitf_details = SIDDDetails(file_name)
        if nitf_details.is_sidd:
            print('File {} is determined to be a sicd (NITF format) file.'.format(file_name))
            return SIDDReader(nitf_details)
        else:
            return None
    except IOError:
        # we don't want to catch parsing errors, for now
        return None


#########
# Helper object for initially parses NITF header - specifically looking for SICD elements


class SIDDDetails(NITFDetails):
    """
    SIDD are stored in NITF 2.1 files.
    """

    __slots__ = (
        '_img_headers', '_is_sidd', '_sidd_meta', '_sicd_meta',
        'img_segment_rows', 'img_segment_columns')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF 2.1 file containing a SIDD
        """

        self._img_headers = None
        self._is_sidd = False
        self._sidd_meta = None
        self._sicd_meta = None
        super(SIDDDetails, self).__init__(file_name)
        if self._nitf_header.ImageSegments.subhead_sizes.size == 0:
            raise IOError('There are no image segments defined.')
        if self._nitf_header.GraphicsSegments.item_sizes.size > 0:
            raise IOError('A SIDD file does not allow for graphics segments.')
        if self._nitf_header.DataExtensions.subhead_sizes.size == 0:
            raise IOError('A SIDD file requires at least one data extension, containing the '
                          'SIDD xml structure.')
        # define the sidd and sicd metadata
        self._find_sidd()
        # populate the image details
        self.img_segment_rows = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
        self.img_segment_columns = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
        for i, im_header in enumerate(self.img_headers):
            self.img_segment_rows[i] = im_header.NROWS
            self.img_segment_columns[i] = im_header.NCOLS

    @property
    def is_sidd(self):
        """
        bool: whether file name corresponds to a SIDD file, or not.
        """

        return self._is_sidd

    @property
    def sidd_meta(self):
        """
        None|List[sarpy.io.complex.sidd_elements.SIDD.SIDDType]: the sidd meta-data structure(s).
        """

        return self._sidd_meta

    @property
    def sicd_meta(self):
        """
        None|List[sarpy.io.complex.sicd_elements.SICD.SICDType]: the sicd meta-data structure(s).
        """

        return self._sicd_meta

    @property
    def img_headers(self):
        """
        The image segment headers.

        Returns
        -------
            None|List[sarpy.io.nitf.image.ImageSegmentHeader]
        """

        if self._img_headers is not None:
            return self._img_headers

        self._parse_img_headers()
        return self._img_headers

    def _parse_img_headers(self):
        if self.img_segment_offsets is None or self._img_headers is not None:
            return

        self._img_headers = [self.parse_image_subheader(i) for i in range(self.img_subheader_offsets.size)]

    def _find_sidd(self):
        self._is_sidd = False
        if self.des_subheader_offsets is None:
            return

        self._sidd_meta = []
        self._sicd_meta = []

        for i in range(self.des_subheader_offsets.size):
            subhead_bytes = self.get_des_subheader_bytes(i)
            if subhead_bytes.startswith(b'DEXML_DATA_CONTENT'):
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SIDD' in root_node.tag:
                        self._is_sidd = True
                        self._sidd_meta.append(SIDDType.from_node(root_node, xml_ns, ns_key='default'))
                    elif 'SICD' in root_node.tag:
                        self._sicd_meta.append(SICDType.from_node(root_node, xml_ns, ns_key='default'))
                except Exception as e:
                    logging.error('Failed checking des xml header at index {} with error {}'.format(i, e))
                    continue
            elif subhead_bytes.startswith(b'DESIDD_XML'):
                # This is an old format SIDD header
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SIDD' in root_node.tag:
                        self._is_sidd = True
                        self._sidd_meta.append(SIDDType.from_node(root_node, xml_ns, ns_key='default'))
                except Exception as e:
                    logging.error('We found an apparent old-style SIDD DES header at index {}, '
                                  'but failed parsing with error {}'.format(i, e))
                    continue
            elif subhead_bytes.startswith(b'DESICD_XML'):
                # This is an old format SICD header
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SICD' in root_node.tag:
                        self._sicd_meta.append(SICDType.from_node(root_node, xml_ns, ns_key='default'))
                except Exception as e:
                    logging.error('We found an apparent old-style SICD DES header at index {}, '
                                  'but failed parsing with error {}'.format(i, e))
                    continue

        if not self._is_sidd:
            return

        for sicd in self._sicd_meta:
            sicd.derive()


#######
#  The actual reading implementation

def _validate_lookup(lookup_table):
    # type: (numpy.ndarray) -> None
    if not isinstance(lookup_table, numpy.ndarray):
        raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
    if lookup_table.dtype.name != 'uint8':
        raise ValueError('requires a numpy.ndarray of uint8 dtype, got {}'.format(lookup_table.dtype))


def rgb_lut_conversion(lookup_table):
    """
    This constructs the function to convert from RGB/LUT format data to RGB data.

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
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the last dimension), got shape {}'.format(data.shape))
        return lookup_table[data[:, :, 0]]
    return converter


class SIDDReader(NITFReader):
    """
    A reader object for a SIDD file (NITF container with SICD contents)
    """

    __slots__ = ('_sidd_meta', )

    def __init__(self, nitf_details):
        """

        Parameters
        ----------
        nitf_details : str|SIDDDetails
            filename or SIDDDetails object
        """

        if isinstance(nitf_details, string_types):
            nitf_details = SIDDDetails(nitf_details)
        if not isinstance(nitf_details, SIDDDetails):
            raise TypeError('The input argument for SIDDReader must be a filename or '
                            'SIDDDetails object.')

        if not nitf_details.is_sidd:
            raise ValueError(
                'The input file passed in appears to be a NITF 2.1 file that does not contain valid sidd metadata.')
        self._sidd_meta = self._nitf_details.sidd_meta
        super(SIDDReader, self).__init__(nitf_details)

    def _find_segments(self):
        # determine image segmentation from image headers
        segments = [[] for sidd in self._sidd_meta]
        for i, img_header in enumerate(self._nitf_details.img_headers):
            # skip anything but SAR for now (i.e. legend)
            if img_header.ICAT != 'SAR':
                continue

            iid1 = img_header.IID1  # required to be of the form SIDD######
            if not (iid1[:4] == 'SIDD' and iid1[4:].isnumeric()):
                raise ValueError('Got poorly formatted image segment id {} at position {}'.format(iid1, i))
            element = int_func(iid1[4:7])
            if element > len(self._sidd_meta):
                raise ValueError('Got image segment id {}, but there are only {} '
                                 'sidd elements'.format(iid1, len(self._sidd_meta)))
            segments[element - 1].append(i)
        # Ensure that all segments are populated
        for i, entry in enumerate(segments):
            if len(entry) < 1:
                raise ValueError('Did not find any image segments for SIDD {}'.format(i))
        return segments

    def _get_img_details(self, index):
        img_header = self._nitf_details.img_headers[index]
        if img_header.IC != 'NC':
            raise ValueError('Image header at index {} has IC {}. No compression '
                             'is supported at this time.'.format(index, img_header.IC))
        abpp = img_header.ABPP
        if abpp not in [8, 16]:
            raise ValueError('Image header at index {} has ABPP {}. Unsupported.'.format(index, abpp))
        rows = img_header.NROWS
        cols = img_header.NCOLS
        # check for a strangely segmented image
        if img_header.NPPBV not in [rows, 0]:
            raise ValueError('Image header at index {} has NROWS {} and NPPBV {}, '
                             'which is unsupported'.format(index, rows, img_header.NPPBV))
        if img_header.NPPBH not in [cols, 0]:
            raise ValueError('Image header at index {} has NCOLS {} and NPPBH {}, '
                             'which is unsupported'.format(index, cols, img_header.NPPBH))
        # check image type
        irep = img_header.IREP

        if irep not in ['MONO', 'RGB/LUT', 'RGB']:
            raise ValueError('Image header at index {} has IREP {}, which is '
                             'unsupported'.format(index, irep))
        bands = len(img_header.Bands)
        if (irep == 'RGB' and bands != 3) or (irep != 'RGB' and bands != 1):
                raise ValueError('Image header at index {} has IREP {} and {} Bands. '
                                     'This is unsupported.'.format(index, irep, bands))
        # Fetch lookup table
        lut = img_header.Bands[0].LUTD if irep == 'RGB/LUT' else None

        # determine dtype
        if lut is None:
            if abpp == 8:
                dtype = numpy.dtype('>uint8')
            else:
                dtype = numpy.dtype('>uint16')
        else:
            dtype = numpy.dtype('>uint8')

        return rows, cols, dtype, bands, lut

    def _construct_chipper(self, segment, index):
        bounds = numpy.zeros((len(segment), 4), dtype=numpy.uint64)
        offsets = numpy.zeros((len(segment), ), dtype=numpy.int64)
        this_dtype, this_bands, this_lut = None, None, None

        p_row_start, p_row_end, p_col_start, p_col_end = None, None, None, None
        for i, index in enumerate(segment):
            # populate bounds entry
            offsets[i] = self._nitf_details.img_segment_offsets[index]
            rows, cols, dtype, bands, lut = self._get_img_details(index)
            if i == 0:
                cur_row_start, cur_row_end = 0, rows
                cur_col_start, cur_col_end = 0, cols
            elif cols == (p_col_end - p_col_start):
                cur_row_start, cur_row_end = p_row_end, p_row_end + cols
                cur_col_start, cur_col_end = p_col_start, p_col_end
            else:
                cur_row_start, cur_row_end = 0, rows
                cur_col_start, cur_col_end = p_col_end, p_col_end + rows

            bounds[i] = (cur_row_start, cur_row_end, cur_col_start, cur_col_end)
            p_row_start, p_row_end, p_col_start, p_col_end = cur_row_start, cur_row_end, cur_col_start, cur_col_end

            # define the other meta data
            if i == 0:
                this_dtype = dtype
                this_bands = bands
                this_lut = lut
            else:
                if dtype != this_dtype:
                    raise ValueError('Image segments {} form one sidd, but have differing type '
                                     'parameters'.format(segment))
                if bands != this_bands:
                    raise ValueError('Image segments {} form one sidd, but have differing '
                                     'band counts'.format(segment))
                if (this_lut is None and lut is not None) or (this_lut is not None and lut is None):
                    raise ValueError('Image segments {} form one sidd, but have differing '
                                     'look up table information'.format(segment))
                if (this_lut is not None and lut is not None) and \
                        (this_lut.shape != lut.shape or numpy.any(this_lut != lut)):
                    raise ValueError('Image segments {} form one sidd, but have differing '
                                     'look up table information'.format(segment))

        complex_type = False if this_lut is None else rgb_lut_conversion(this_lut)
        return MultiSegmentChipper(
            self._nitf_details.file_name, bounds, offsets, this_dtype,
            symmetry=(False, False, False), complex_type=complex_type, bands_ip=this_bands)
