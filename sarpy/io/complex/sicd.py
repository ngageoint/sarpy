# -*- coding: utf-8 -*-
"""
Module for reading SICD files - should support SICD version 0.3 and above.
"""

import re
import sys
import logging
from typing import Union, Tuple

import numpy

from .base import BaseChipper, BaseReader, BaseWriter, int_func, string_types
from .bip import BIPChipper, BIPWriter
from .utils import parse_xml_from_string
from .sicd_elements.SICD import SICDType
from .sicd_elements.blocks import LatLonType

from ..nitf.nitf_head import NITFDetails, NITFHeader, ImageSegmentsType, DataExtensionsType
# noinspection PyProtectedMember
from ..nitf.des import DataExtensionHeader, SICDDESSubheader, \
    _SICD_SPECIFICATION_NAMESPACE, _SICD_SPECIFICATION_IDENTIFIER
from ..nitf.security import NITFSecurityTags
from ..nitf.image import ImageSegmentHeader, ImageBands, ImageBand


if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences
    from cStringIO import StringIO
    NOT_FOUND_ERROR = IOError
else:
    from io import StringIO
    # noinspection PyUnresolvedReferences
    NOT_FOUND_ERROR = FileNotFoundError

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Wade Schwartzkopf")


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a SICD file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    SICDReader|None
        `SICDReader` instance if SICD file, `None` otherwise
    """

    try:
        nitf_details = SICDDetails(file_name)
        if nitf_details.is_sicd:
            print('File {} is determined to be a sicd (NITF format) file.'.format(file_name))
            return SICDReader(nitf_details)
        else:
            return None
    except IOError:
        # we don't want to catch parsing errors, for now
        return None


#########
# Helper object for initially parses NITF header - specifically looking for SICD elements


class SICDDetails(NITFDetails):
    """
    SICD are stored in NITF 2.1 files.
    """
    __slots__ = (
        '_des_index', '_des_header', '_img_headers',
        '_is_sicd', '_sicd_meta', 'img_segment_rows', 'img_segment_columns')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF 2.1 file containing a SICD
        """

        self._des_index = None
        self._des_header = None
        self._img_headers = None
        self._is_sicd = False
        self._sicd_meta = None
        super(SICDDetails, self).__init__(file_name)
        if self._nitf_header.ImageSegments.subhead_sizes.size == 0:
            raise IOError('There are no image segments defined.')
        if self._nitf_header.GraphicsSegments.item_sizes.size > 0:
            raise IOError('A SICD file does not allow for graphics segments.')
        if self._nitf_header.DataExtensions.subhead_sizes.size == 0:
            raise IOError('A SICD file requires at least one data extension, containing the '
                          'SICD xml structure.')
        # define the sicd metadata
        self._find_sicd()
        # populate the image details
        self.img_segment_rows = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
        self.img_segment_columns = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
        for i, im_header in enumerate(self.img_headers):
            self.img_segment_rows[i] = im_header.NROWS
            self.img_segment_columns[i] = im_header.NCOLS

    @property
    def is_sicd(self):
        """
        bool: whether file name corresponds to a SICD file, or not.
        """

        return self._is_sicd

    @property
    def sicd_meta(self):
        """
        sarpy.io.complex.sicd_elements.SICD.SICDType: the sicd meta-data structure.
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

    @property
    def des_header(self):
        """
        The DES subheader object associated with the SICD.

        Returns
        -------
        None|sarpy.io.nitf.des.DataExtensionHeader
        """

        return self._des_header

    def _parse_img_headers(self):
        if self.img_segment_offsets is None or self._img_headers is not None:
            return

        self._img_headers = [self.parse_image_subheader(i) for i in range(self.img_subheader_offsets.size)]

    def _find_sicd(self):
        self._is_sicd = False
        self._sicd_meta = None
        if self.des_subheader_offsets is None:
            return

        for i in range(self.des_subheader_offsets.size):
            subhead_bytes = self.get_des_subheader_bytes(i)
            if subhead_bytes.startswith(b'DEXML_DATA_CONTENT'):
                des_header = DataExtensionHeader.from_bytes(subhead_bytes, start=0)
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes.decode('utf-8').strip())
                    if 'SIDD' in root_node.tag:  # namespace makes this ugly
                        # NOTE that SIDD files are supposed to have the corresponding
                        # SICD xml as one of the DES AFTER the SIDD xml.
                        # The same basic format is used for both headers.
                        # So, abandon if we find a SIDD xml
                        self._des_index = None
                        self._des_header = None
                        self._is_sicd = False
                        break
                    elif 'SICD' in root_node.tag:  # namespace makes this ugly
                        self._des_index = i
                        self._des_header = des_header
                        self._is_sicd = True
                        self._sicd_meta = SICDType.from_node(root_node, xml_ns, ns_key='default')
                        break
                except Exception:
                    continue
            elif subhead_bytes.startswith(b'DESIDD_XML'):
                # This is an old format SIDD and can't be a SICD
                self._des_index = None
                self._des_header = None
                self._is_sicd = False
                break
            elif subhead_bytes.startswith(b'DESICD_XML'):
                # This is an old format SICD
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SICD' in root_node.tag:  # namespace makes this ugly
                        self._des_index = i
                        self._des_header = None
                        self._is_sicd = True
                        self._sicd_meta = SICDType.from_node(root_node, xml_ns, ns_key='default')
                        break
                except Exception as e:
                    logging.error('We found an apparent old-style SICD DES header, '
                                  'but failed parsing with error {}'.format(e))
                    continue

        if not self._is_sicd:
            return
        self._sicd_meta.derive()
        # TODO: account for the reference frequency offset situation

    def is_des_well_formed(self):
        """
        Returns whether the data extension subheader well-formed. Returns `None`
        if the DataExtensionHeader or the UserHeader section of it was not successfully
        parsed. Currently just checks the `DESSHSI` field for the required value.

        Returns
        -------
        bool|None
        """

        if not self._is_sicd or self.des_header is None or \
                not isinstance(self.des_header, DataExtensionHeader):
            return None

        sicd_des = self._des_header.UserHeader
        if not isinstance(sicd_des, SICDDESSubheader):
            return None
        return sicd_des.DESSHSI.strip() == _SICD_SPECIFICATION_IDENTIFIER

    def repair_des_header(self):
        """
        Determines whether the data extension subheader is well-formed, and tries
        to repair it if not. Currently just sets the `DESSHSI` field to the
        required value.

        Returns `0` if wellformedness could not be evaluated, `1` if no change was
        required, `2` if the subheader was replaced, and `3` if the replacement effort
        failed (details logged at error level).

        Returns
        -------
        int
        """

        stat = self.is_des_well_formed()
        if stat is None:
            return 0
        elif stat:
            return 1

        sicd_des = self._des_header.UserHeader
        # noinspection PyProtectedMember
        sicd_des.DESSHSI = _SICD_SPECIFICATION_IDENTIFIER
        stat = self.rewrite_des_header()
        return 2 if stat else 3

    def rewrite_des_header(self):
        """
        Rewrites the DES subheader associated with the SICD from the current
        value in `des_header` property. This allows minor modifications to the
        security tags or user header information.

        Returns
        -------
        bool
            True is the modification was successful and False otherwise. Note that
            no errors, in particular io errors from write permission issues,
            are caught.
        """

        if not self._is_sicd:
            return False

        des_bytes = self.des_header.to_bytes()
        des_size = self._nitf_header.DataExtensions.subhead_sizes[self._des_index]
        if len(des_bytes) != des_size:
            logging.error(
                "The size of the current des header {} bytes, does not match the "
                "previous {} bytes. They cannot be trivially replaced.".format(des_bytes, des_size))
            return False
        des_loc = self.des_subheader_offsets[self._des_index]
        with open(self._file_name, 'r+b') as fi:
            fi.seek(des_loc)
            fi.write(des_bytes)
        return True


#######
#  The actual reading implementation

def _validate_lookup(lookup_table):  # type: (numpy.ndarray) -> None
    if not isinstance(lookup_table, numpy.ndarray):
        raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
    if lookup_table.dtype.name != 'float64':
        raise ValueError('requires a numpy.ndarray of float64 dtype, got {}'.format(lookup_table.dtype))
    if lookup_table.shape != (256, ):
        raise ValueError('Requires a one-dimensional numpy.ndarray with 256 elements, '
                         'got shape {}'.format(lookup_table.shape))


def amp_phase_to_complex(lookup_table):
    """
    This constructs the function to convert from AMP8I_PHS8I format data to complex64 data.

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

        if data.dtype.name != 'uint8':
            raise ValueError('requires a numpy.ndarray of uint8 dtype, got {}'.format(data.dtype.name))

        if len(data.shape) == 3:
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the first dimension), got shape {}'.format(data.shape))

        out = numpy.zeros((data.shape[0] / 2, data.shape[1], data.shape[2]), dtype=numpy.complex64)
        amp = lookup_table[data[0::2, :, :]]
        theta = data[1::2, :, :]*(2*numpy.pi/256)
        out.real = amp*numpy.cos(theta)
        out.imag = amp*numpy.sin(theta)
        return out
    return converter


class MultiSegmentChipper(BaseChipper):
    """
    required chipping extension to accommodate for the fact that NITF files have specified size
    and row/column limits. Any sufficiently large SICD collect must be broken into a series of
    image segments (along rows). We must have a parser to transparently extract data between
    this collection of image segments.
    """

    __slots__ = ('_file_name', '_data_size', '_dtype', '_complex_out',
                 '_symmetry', '_row_starts', '_row_ends',
                 '_bands_ip', '_child_chippers')

    def __init__(self, file_name, data_sizes, data_offsets, data_type, symmetry=None,
                 complex_type=True, bands_ip=1):
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
        bands_ip : int
            number of bands - this will always be one for sicd.
        """

        if not isinstance(data_sizes, numpy.ndarray):
            raise ValueError('data_sizes must be an numpy.ndarray, not {}'.format(type(data_sizes)))
        if not (len(data_sizes.shape) == 2 and data_sizes.shape[1] == 2):
            raise ValueError('data_sizes must be an Nx2 numpy.ndarray, not shape {}'.format(data_sizes.shape))
        if not numpy.all(data_sizes[:, 1] == data_sizes[0, 1]):
            # TODO: account for things broken up along more than just row order
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
        # all of the actual reading and reorienting work will be done by these
        # child chippers, which will read from their respective image segments
        self._child_chippers = tuple(
            BIPChipper(file_name, data_type, img_siz, symmetry=symmetry,
                       complex_type=complex_type, data_offset=img_off,
                       bands_ip=bands_ip)
            for img_siz, img_off in zip(data_sizes, data_offsets))

        self._row_starts = numpy.zeros((data_sizes.shape[0], ), dtype=numpy.int64)
        self._row_ends = numpy.cumsum(data_sizes[:, 0])
        self._row_starts[1:] = self._row_ends[:-1]
        self._bands_ip = int_func(bands_ip)

        data_size = (self._row_ends[-1], data_sizes[0, 1])
        # all of the actual reading and reorienting done by child chippers,
        # so do not reorient or change type at this level
        super(MultiSegmentChipper, self).__init__(data_size, symmetry=(False, False, False), complex_type=False)

    def _read_raw_fun(self, range1, range2):
        range1, range2 = self._reorder_arguments(range1, range2)
        # this method just assembles the final data from the child chipper pieces
        rows = numpy.arange(*range1, dtype=numpy.int64)  # array
        cols_size = int_func((range2[1] - range2[0] - 1)/range2[2]) + 1
        if self._bands_ip == 1:
            out = numpy.empty((rows.size, cols_size), dtype=numpy.complex64)
        else:
            out = numpy.empty((rows.size, cols_size, self._bands_ip), dtype=numpy.complex64)
        for row_start, row_end, child_chipper in zip(self._row_starts, self._row_ends, self._child_chippers):
            row_bool = ((rows >= row_start) & (rows < row_end))
            if numpy.any(row_bool):
                crows = rows[row_bool]
                crange1 = (crows[0]-row_start, crows[-1]+1-row_start, range1[2])
                if self._bands_ip == 1:
                    out[row_bool, :] = child_chipper(crange1, range2)
                else:
                    out[row_bool, :, :] = child_chipper(crange1, range2)
        return out


class SICDReader(BaseReader):
    """
    A reader object for a SICD file (NITF container with SICD contents)
    """

    __slots__ = ('_nitf_details', '_sicd_meta', '_chipper')

    def __init__(self, nitf_details):
        """

        Parameters
        ----------
        nitf_details : str|sarpy.io.complex.sicd_elements.SICD.SICDDetails
            filename or SICDDetails object
        """

        if isinstance(nitf_details, string_types):
            nitf_details = SICDDetails(nitf_details)
        if not isinstance(nitf_details, SICDDetails):
            raise TypeError('The input argument for SICDReader must be a filename or '
                            'SICDDetails object.')

        self._nitf_details = nitf_details
        if not self._nitf_details.is_sicd:
            raise ValueError(
                'The input file passed in appears to be a NITF 2.1 file that does not contain valid sicd metadata.')

        self._sicd_meta = self._nitf_details.sicd_meta

        pixel_type = self._sicd_meta.ImageData.PixelType
        complex_type = True
        # NB: SICDs are required to be stored as big-endian
        if pixel_type == 'RE32F_IM32F':
            dtype = numpy.dtype('>f4')
        elif pixel_type == 'RE16I_IM16I':
            dtype = numpy.dtype('>i2')
        elif pixel_type == 'AMP8I_PHS8I':
            dtype = numpy.dtype('>u1')
            complex_type = amp_phase_to_complex(self._sicd_meta.ImageData.AmpTable)
        else:
            raise ValueError('Pixel Type {} not recognized.'.format(pixel_type))

        data_sizes = numpy.column_stack(
            (self._nitf_details.img_segment_rows, self._nitf_details.img_segment_columns))
        # TODO: account for things broken up along more than just rows
        # SICDs require no symmetry reorientation
        symmetry = (False, False, False)
        # construct our chipper
        chipper = MultiSegmentChipper(
            nitf_details.file_name, data_sizes, self._nitf_details.img_segment_offsets.copy(), dtype,
            symmetry=symmetry, complex_type=complex_type,
            bands_ip=1)

        super(SICDReader, self).__init__(self._sicd_meta, chipper)

        # should we do a preliminary check that the structure is valid?
        #   note that this results in potentially noisy logging for troubled sicd files
        self._sicd_meta.is_valid(recursive=True)


#######
#  The actual writing implementation

def _validate_input(data):
    # type: (numpy.ndarray) -> tuple
    if not isinstance(data, numpy.ndarray):
        raise ValueError('Requires a numpy.ndarray, got {}'.format(type(data)))
    if data.dtype.name not in ('complex64', 'complex128'):
        raise ValueError('Requires a numpy.ndarray of complex dtype, got {}'.format(data.dtype.name))
    if len(data.shape) != 2:
        raise ValueError('Requires a two-dimensional numpy.ndarray, got {}'.format(data.shape))

    new_shape = (data.shape[0], data.shape[1], 2)
    return new_shape


def complex_to_amp_phase(lookup_table):
    """
    This constructs the function to convert from complex64 or 128 to AMP8I_PHS8I format data.

    Parameters
    ----------
    lookup_table : numpy.ndarray

    Returns
    -------
    callable
    """

    _validate_lookup(lookup_table)

    def converter(data):
        new_shape = _validate_input(data)
        out = numpy.zeros(new_shape, dtype=numpy.uint8)
        # NB: for numpy before 1.10, digitize requires 1-d
        out[:, :, 0] = numpy.digitize(numpy.abs(data).ravel(), lookup_table, right=False).reshape(data.shape)
        out[:, :, 1] = numpy.arctan2(data.real, data.imag)*(256/(2*numpy.pi))
        # truncation takes care of properly rolling negative to positive
        return out

    return converter


def complex_to_int(data):
    """
    This converts from complex64 or 128 data to int16 data.

    Parameters
    ----------
    data : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    # TODO: this is naive. Scaling down to 16-bit limits requires thought.
    new_shape = _validate_input(data)

    if data.dtype.name == 'complex128':
        view_dtype = numpy.float64
    else:
        view_dtype = numpy.float32

    i16_info = numpy.iinfo(numpy.int16)  # for getting max/min type values
    data_view = data.view(dtype=view_dtype).reshape(new_shape)
    out = numpy.zeros(new_shape, dtype=numpy.int16)
    out[:] = numpy.round(numpy.clip(data_view, i16_info.min, i16_info.max))
    # this is nonsense without the clip - gets cast to int64 and then truncated.
    # should we round? Without it, it will be the floor, I believe.
    return out


class SICDWriter(BaseWriter):
    """
    Writer object for SICD file - that is, a NITF file containing SICD data
    following standard 1.2.1
    """

    __slots__ = (
        '_file_name', '_sicd_meta', '_shape', '_pixel_size', '_dtype',
        '_complex_type', '_image_segment_limits',
        '_security_tags', '_image_segment_headers', '_data_extension_header', '_nitf_header',
        '_header_offsets', '_image_offsets',
        '_final_header_info', '_writing_chippers', '_pixels_written', '_des_written')

    def __init__(self, file_name, sicd_meta):
        """

        Parameters
        ----------
        file_name : str
        sicd_meta : sarpy.io.complex.sicd_elements.SICD.SICDType
        """

        super(SICDWriter, self).__init__(file_name, sicd_meta)
        self._shape = (sicd_meta.ImageData.NumRows, sicd_meta.ImageData.NumCols)

        # define _security_tags
        self._security_tags = self.default_security_tags()
        # get image segment details
        self._pixel_size, self._dtype, self._complex_type, pv_type, isubcat, \
            self._image_segment_limits = self._image_segment_details()
        # prepare our pixels written counter
        self._pixels_written = numpy.zeros((self._image_segment_limits.shape[0],), dtype=numpy.int64)
        # define _image_segment_headers
        self._image_segment_headers = self._create_image_segment_headers(pv_type, isubcat)
        # define _data_extension_header
        self._data_extension_header = self._create_data_extension_header()
        # define _nitf_header
        self._nitf_header = self._create_nitf_header()
        self._header_offsets = None
        self._image_offsets = None
        self._final_header_info = None
        self._writing_chippers = None
        self._des_written = False

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
    def image_segment_headers(self):  # type: () -> Tuple[ImageSegmentHeader]
        """
        tuple[ImageSegmentHeader]: The NITF image segment headers. Each entry will have
        the `SecurityTags` property is populated using `security_tags` **by reference**
        upon construction.

        .. Note:: required edits should be made before adding any data via :func:`write_chip`.
        """

        return self._image_segment_headers

    @property
    def data_extension_header(self):  # type: () -> DataExtensionHeader
        """
        DataExtensionHeader: the NITF data extension header. The `SecurityTags`
        property is populated using `security_tags` **by reference** upon construction.

        .. Note:: required edits should be made before adding any data via :func:`write_chip`.
        """

        return self._data_extension_header

    def default_security_tags(self):
        """
        Returns a NITF security tags object with `CLAS` and `CODE`
        attributes set from the SICD.CollectionInfo.Classification value.

        It is expected that output from this will be modified as appropriate
        and used to set ONLY specific security tags in `data_extension_header` or
        elements of `image_segment_headers`.

        If simultaneous modification of all security tags attributes for the entire
        SICD is the goal, then directly modify the value(s) using `security_tags`.

        Returns
        -------
        sarpy.io.nitf.security.NITFSecurityTags
        """

        def get_basic_args():
            out = {}
            if hasattr(self._sicd_meta, '_NITF') and isinstance(self._sicd_meta._NITF, dict):
                sec_tags = self._sicd_meta._NITF.get('Security', {})
                # noinspection PyProtectedMember
                for fld in NITFSecurityTags._ordering:
                    if fld in sec_tags:
                        out[fld] = sec_tags[fld]
            return out

        def get_clas(in_str):
            if 'CLAS' in args:
                return

            if 'UNCLASS' in in_str.upper():
                args['CLAS'] = 'U'
            elif 'CONFIDENTIAL' in in_str.upper():
                args['CLAS'] = 'C'
            elif 'TOP SECRET' in in_str.upper():
                args['CLAS'] = 'T'
            elif 'SECRET' in in_str.upper():
                args['CLAS'] = 'S'
            else:
                logging.critical('Unclear how to extract CLAS for classification string {}. '
                                 'Unpopulated for now, and should be set appropriately.'.format(in_str))

        def get_code(in_str):
            if 'CODE' in args:
                return

            code = re.search('(?<=/)[^/].*', in_str)
            if code is not None:
                args['CODE'] = code.group()

        args = get_basic_args()
        if self._sicd_meta.CollectionInfo is not None:
            get_clas(self._sicd_meta.CollectionInfo.Classification)
            get_code(self._sicd_meta.CollectionInfo.Classification)
        return NITFSecurityTags(**args)

    def _image_segment_details(self):
        # type: () -> (int, numpy.dtype, Union[bool, callable], str, tuple, numpy.ndarray)
        pixel_type = self._sicd_meta.ImageData.PixelType  # required to be defined
        # NB: SICDs are required to be stored as big-endian, so the endian-ness
        #   of the memmap must be explicit
        if pixel_type == 'RE32F_IM32F':
            pv_type, isubcat = 'R', ('I', 'Q')
            pixel_size = 8
            dtype = numpy.dtype('>f4')
            complex_type = True
        elif pixel_type == 'RE16I_IM16I':
            pv_type, isubcat = 'SI', ('I', 'Q')
            pixel_size = 4
            dtype = numpy.dtype('>i2')
            complex_type = complex_to_int
        else:  # pixel_type == 'AMP8I_PHS8I':
            pv_type, isubcat = 'INT', ('M', 'P')
            pixel_size = 2
            dtype = numpy.dtype('>u1')
            complex_type = complex_to_amp_phase(self._sicd_meta.ImageData.AmpTable)

        IM_SEG_LIMIT = 10**10 - 2  # as big as can be stored in 10 digits, given at least 2 bytes per pixel
        DIM_LIMIT = 10**5 - 1  # as big as can be stored in 5 digits
        IM_ROWS = self._sicd_meta.ImageData.NumRows  # required to be defined
        IM_COLS = self._sicd_meta.ImageData.NumCols  # required to be defined
        im_segments = []

        row_offset = 0
        col_offset = 0
        col_limit = min(DIM_LIMIT, IM_COLS)
        while (row_offset < IM_ROWS) and (col_offset < IM_COLS):
            # determine row count, given row_offset, col_offset, and col_limit
            # how many bytes per row for this column section
            row_memory_size = (col_limit-col_offset)*pixel_size
            # how many rows can we use
            row_count = min(DIM_LIMIT, IM_ROWS-row_offset, int_func(IM_SEG_LIMIT/row_memory_size))
            im_segment_size = pixel_size*row_count*(col_limit-col_offset)
            im_segments.append((row_offset, row_offset + row_count, col_offset, col_limit, im_segment_size))
            row_offset += row_count  # move the next row offset
            if row_offset == IM_ROWS:
                # move over to the next column section
                col_offset = col_limit
                col_limit = min(col_offset+DIM_LIMIT, IM_COLS)
                row_offset = 0
        # we now have [(row_start, row_stop, col_start, col_stop, size_in_bytes)]
        #   following the python convention with starts inclusive, stops not inclusive
        return pixel_size, dtype, complex_type, pv_type, isubcat, numpy.array(im_segments, dtype=numpy.int64)

    def _create_image_segment_headers(self, pv_type, isubcat):
        # type: (str, tuple) -> Tuple[ImageSegmentHeader, ...]

        def get_corner_points_string(ent):
            # ent = (row_start, row_stop, col_start, col_stop)
            if icp is None:
                return ''
            const = 1./(rows*cols)
            pattern = ent[numpy.array([(0, 2), (1, 2), (1, 3), (0, 3)], dtype=numpy.int64)]
            out = []
            for row, col in pattern:
                pt_array = const*numpy.sum(icp *
                                           (numpy.array([rows-row, row, row, rows-row]) *
                                            numpy.array([cols-col, cols-col, col, col]))[:, numpy.newaxis], axis=0)
                pt = LatLonType.from_array(pt_array)
                dms = pt.dms_format(frac_secs=False)
                out.append('{0:02d}{1:02d}{2:02d}{3:s}'.format(*dms[0]) + '{0:03d}{1:02d}{2:02d}{3:s}'.format(*dms[1]))
            return ''.join(out)

        if self._sicd_meta.CollectionInfo is not None and self._sicd_meta.CollectionInfo.CoreName is not None:
            ftitle = 'SICD: {}'.format(self._sicd_meta.CollectionInfo.CoreName)
        else:
            ftitle = 'SICD: Unknown'

        idatim = ' '
        if self._sicd_meta.Timeline is not None and self._sicd_meta.Timeline.CollectStart is not None:
            idatim = re.sub(r'[^0-9]', '', str(self._sicd_meta.Timeline.CollectStart.astype('datetime64[s]')))

        isource = 'SICD: Unknown Collector'
        if self._sicd_meta.CollectionInfo is not None and self._sicd_meta.CollectionInfo.CollectorName is not None:
            isource = 'SICD: {}'.format(self._sicd_meta.CollectionInfo.CollectorName)

        icp, rows, cols = None, None, None
        if self._sicd_meta.GeoData is not None and self._sicd_meta.GeoData.ImageCorners is not None:
            # noinspection PyTypeChecker
            icp = self._sicd_meta.GeoData.ImageCorners.get_array(dtype=numpy.float64)
            rows = self._sicd_meta.ImageData.NumRows
            cols = self._sicd_meta.ImageData.NumCols
        abpp = 4*self._pixel_size
        nppbh = 0 if cols > 8192 else cols
        nppbv = 0 if rows > 8192 else rows
        im_seg_heads = []
        bands = [ImageBand(ISUBCAT=entry) for entry in isubcat]
        for i, entry in enumerate(self._image_segment_limits):
            im_seg_heads.append(ImageSegmentHeader(
                IID1='SICD{0:03d}'.format(0 if len(self._image_segment_limits) == 1 else i+1),
                IDATIM=idatim,
                IID2=ftitle,
                ISORCE=isource,
                IREP='NODISPLY',
                ICAT='SAR',
                NROWS=entry[1]-entry[0],
                NCOLS=entry[3]-entry[2],
                PVTYPE=pv_type,
                ABPP=abpp,
                IGEOLO=get_corner_points_string(entry),
                NPPBH=nppbh,
                NPPBV=nppbv,
                NBPP=abpp,
                IDLVL=i+1,
                IALVL=i,
                ILOC='{0:05d}{1:05d}'.format(entry[0], entry[2]),
                Bands=ImageBands(values=bands),
                Security=self._security_tags))
        return tuple(im_seg_heads)

    def _create_data_extension_header(self):
        # type: () -> DataExtensionHeader
        desshdt = str(self._sicd_meta.ImageCreation.DateTime.astype('datetime64[s]'))
        if desshdt[-1] != 'Z':
            desshdt += 'Z'
        desshlpg = ' '
        if self._sicd_meta.GeoData is not None and self._sicd_meta.GeoData.ImageCorners is not None:
            # noinspection PyTypeChecker
            icp = self._sicd_meta.GeoData.ImageCorners.get_array(dtype=numpy.float64)
            temp = []
            for entry in icp:
                temp.append('{0:0=+12.8f}{1:0=+13.8f}'.format(entry[0], entry[1]))
            temp.append(temp[0])
            desshlpg = ''.join(temp)
        return DataExtensionHeader(
            Security=self._security_tags, UserHeader=SICDDESSubheader(DESSHDT=desshdt, DESSHLPG=desshlpg))

    def _create_nitf_header(self):
        # type: () -> NITFHeader
        im_size = self._sicd_meta.ImageData.NumRows*self._sicd_meta.ImageData.NumCols*self._pixel_size
        if im_size < 50*(1024**2):
            clevel = 3
        elif im_size < (1024**3):
            clevel = 5
        elif im_size < 2*(1024**3):
            clevel = 6
        else:
            clevel = 7
        ostaid = 'Unknown '
        fdt = re.sub(r'[^0-9]', '', str(self._sicd_meta.ImageCreation.DateTime.astype('datetime64[s]')))
        if self._sicd_meta.CollectionInfo is not None and self._sicd_meta.CollectionInfo.CoreName is not None:
            ftitle = 'SICD: {}'.format(self._sicd_meta.CollectionInfo.CoreName)
        else:
            ftitle = 'SICD: Unknown'
        # get image segment details - the size of the headers will be redefined when locking down details
        im_sizes = numpy.copy(self._image_segment_limits[:, 4])
        im_segs = ImageSegmentsType(
            subhead_sizes=numpy.zeros(im_sizes.shape, dtype=numpy.int64),
            item_sizes=im_sizes)
        # get data extension details - the size of the headers and items
        #   will be redefined when locking down details
        des = DataExtensionsType(
            subhead_sizes=numpy.array([973, ], dtype=numpy.int64),
            item_sizes=numpy.array([0, ], dtype=numpy.int64))
        return NITFHeader(Security=self.security_tags, CLEVEL=clevel, OSTAID=ostaid, FDT=fdt, FTITLE=ftitle,
                          FL=0, ImageSegments=im_segs, DataExtensions=des)

    def prepare_for_writing(self):
        """
        The NITF file header makes specific reference of the lengths of various components,
        specifically the image segment subheader lengths and the data extension (i.e. SICD xml)
        subheader and item lengths. These items must be locked down BEFORE we can allocate
        the required file writing specifics from the OS.

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

        if self._writing_chippers is not None:
            return

        self._final_header_info = {}
        # get des header, xml and populate nitf_header.DataExtensions information
        des_info = {
            'header': self._data_extension_header.to_bytes(),
            'xml': self._sicd_meta.to_xml_bytes(urn=_SICD_SPECIFICATION_NAMESPACE, tag='SICD')}
        self._nitf_header.DataExtensions.subhead_sizes[0] = len(des_info['header'])
        self._nitf_header.DataExtensions.item_sizes[0] = len(des_info['xml'])  # size should be no issue
        # there would be no satisfactory resolution in the case of an oversized header - we should raise an exception
        if len(des_info['header']) >= 10**4:
            raise ValueError(
                'The data extension subheader is {} characters, and NITF limits the possible '
                'size of a data extension to fewer than 10^4 characters. '
                'This is likely the result of an error.'.format(len(des_info['header'])))
        # there would be no satisfactory resolution in the case of an oversized xml - we should raise an exception
        if len(des_info['xml']) >= 10**9:
            raise ValueError(
                'The xml for our SICD is {} characters, and NITF limits the possible '
                'size of a data extension to fewer than 10^9 characters.'.format(len(des_info['xml'])))
        self._final_header_info['des'] = des_info

        # get image_segment_header strings and populate nitf_header.ImageSegments.subhead_sizes entries
        im_segment_headers = []
        for i, entry in enumerate(self._image_segment_headers):
            head = entry.to_bytes()
            # no satisfactory resolution in the case of an oversized header
            if len(head) >= 10**6:
                raise ValueError(
                    'Image subheader ({} of {}) is {} characters, and NITF limits the possible '
                    'size of an image subheader to fewer than 10^6 characters. '
                    'This is likely the result of an error.'.format(i+1, len(self._image_segment_headers), len(head)))
            im_segment_headers.append(head)
            self._nitf_header.ImageSegments.subhead_sizes[i] = len(head)
        self._final_header_info['image_headers'] = tuple(im_segment_headers)

        # calculate image offsets and file length
        header_length = self._nitf_header.get_bytes_length()
        cumulative_size = header_length
        header_offsets = numpy.zeros((len(self._image_segment_headers), ), dtype=numpy.int64)
        image_offsets = numpy.zeros((len(self._image_segment_headers), ), dtype=numpy.int64)
        for i, (head_size, im_size) in enumerate(
                zip(self._nitf_header.ImageSegments.subhead_sizes, self._nitf_header.ImageSegments.item_sizes)):
            header_offsets[i] = cumulative_size
            cumulative_size += head_size
            image_offsets[i] = cumulative_size
            cumulative_size += im_size
        self._header_offsets = header_offsets
        self._image_offsets = image_offsets
        # NB: this is where text segments would be packed, but we do not enable creation of such currently
        cumulative_size += numpy.sum(
            self._nitf_header.DataExtensions.subhead_sizes + self._nitf_header.DataExtensions.item_sizes)
        if cumulative_size >= 10**12:
            raise ValueError(
                'The calculated file size is {} bytes, and NITF requires it to '
                'be fewer than 10^12 bytes.'.format(cumulative_size))
        self._nitf_header.FL = cumulative_size
        self._final_header_info['nitf'] = self._nitf_header.to_bytes()
        logging.info(
            'Writing NITF header and setting up the chippers, this likely causes '
            'a large physical memory allocation and may be time consuming.')

        # write the nitf header
        with open(self._file_name, mode='r+b') as fi:
            fi.write(self._final_header_info['nitf'])

        # prepare out writing chippers
        self._writing_chippers = tuple(
            BIPWriter(self._file_name, (ent[1]-ent[0], ent[3]-ent[2]),
                      self._dtype, self._complex_type, data_offset=offset)
            for ent, offset in zip(self._image_segment_limits, image_offsets))

    def _write_image_header(self, index):
        # type: (int) -> None
        if self._pixels_written[index] > 0:
            return
        with open(self._file_name, mode='r+b') as fi:
            fi.seek(self._header_offsets[index])
            fi.write(self._final_header_info['image_headers'][index])

    def close(self):
        """
        Checks that data appears to be satisfactorily written, and logs some details
        at error level, if not. Then, it finalizes the SICD file by writing the final
        data extension header and data extension (i.e. SICD xml).

        Returns
        -------
        None
        """

        if not hasattr(self, '_des_written') or self._des_written:
            return

        # let's double check that everything is written
        insufficiently_written = [[], [], []]
        for i, (entry, pix_written) in enumerate(zip(self._image_segment_limits, self._pixels_written)):
            im_siz = entry[4]
            im_written = pix_written*self._pixel_size
            if im_written < im_siz:
                insufficiently_written[0].append(i)
                insufficiently_written[1].append(im_siz)
                insufficiently_written[2].append(im_written)
        if len(insufficiently_written[0]) > 0:
            logging.error(
                'Attempting to create file {}, which will be corrupt. Image segment(s) {} '
                'were expected to be of size {}, but only {} bytes were written.'.format(
                    self._file_name,
                    insufficiently_written[0],
                    insufficiently_written[1],
                    insufficiently_written[2]))

        # let's write the data extension, if we can
        # noinspection PyBroadException
        try:
            with open(self._file_name, mode='r+b') as fi:
                fi.seek(self._image_offsets[-1] + self._image_segment_limits[-1, 4])
                fi.write(self._final_header_info['des']['header'])
                fi.write(self._final_header_info['des']['xml'])
            self._des_written = True
            logging.info('Data file {} fully written.'.format(self._file_name))
        except NOT_FOUND_ERROR as e:
            logging.error(
                'Data file {} not created. Failed with exception {}'.format(self._file_name, e))
        except Exception as e:
            logging.error(
                'Data file {} improperly created, and is likely corrupt. '
                'Failed with exception {}'.format(self._file_name, e))
        # now, we close all the chippers
        if hasattr(self, '_writing_chippers'):
            if self._writing_chippers is not None:
                for entry in self._writing_chippers:
                    entry.close()

    def __call__(self, data, start_indices=(0, 0)):
        def overlap(rrange, crange):
            def element_overlap(this_range, segment_range):
                if segment_range[0] <= this_range[0] < segment_range[1]:
                    # this_range starts in this image segment?
                    if segment_range[0] < this_range[1] <= segment_range[1]:
                        return this_range[0], this_range[1]
                    else:
                        return this_range[0], segment_range[1]
                elif segment_range[0] < this_range[1] <= segment_range[1]:
                    # does this_range stops (but doesn't start) in this image segment
                    return segment_range[0], this_range[1]
                elif (this_range[0] < segment_range[0]) and (segment_range[1] <= this_range[1]):
                    # this_range encompasses this entire segment
                    return segment_range[0], segment_range[1]
                else:
                    return None

            need_segs = numpy.zeros((len(self._image_segment_limits),), dtype=numpy.bool)
            data_ents = numpy.zeros((len(self._image_segment_limits), 4), dtype=numpy.int64)
            for j, ent in enumerate(self._image_segment_limits):
                row_inds = element_overlap(rrange, ent[0:2])
                col_inds = element_overlap(crange, ent[2:4])
                if (row_inds is not None) and (col_inds is not None):
                    need_segs[j] = True
                    data_ents[j, :2] = row_inds
                    data_ents[j, 2:] = col_inds
            return need_segs, data_ents

        if not isinstance(data, numpy.ndarray):
            raise ValueError('data is required to be an instance of numpy.ndarray, got {}'.format(type(data)))

        start_indices = (int_func(start_indices[0]), int_func(start_indices[1]))
        if (start_indices[0] < 0) or (start_indices[1] < 0):
            raise ValueError('start_indices must have positive entries. Got {}'.format(start_indices))
        if (start_indices[0] >= self._shape[0]) or \
                (start_indices[1] >= self._shape[1]):
            raise ValueError(
                'start_indices must be bounded from above by {}. Got {}'.format(self._shape, start_indices))

        row_range = start_indices[0], start_indices[0] + data.shape[0]
        col_range = start_indices[1], start_indices[1] + data.shape[1]
        if (row_range[1] > self._shape[0]) or (col_range[1] > self._shape[1]):
            raise ValueError(
                'Got start_indices = {} and data of shape {}. '
                'This is incompatible with total data shape {}.'.format(start_indices, data.shape, self._shape))

        if self._writing_chippers is None:
            self.prepare_for_writing()

        # which segment(s) will we write in?
        need_segments, data_entries = overlap(row_range, col_range)
        # need_segments - boolean array of which segments that we'll write in
        # data entries - array of [row start, row end, col start, col end] wrt to full image coordinates.
        for i, need_seg in enumerate(need_segments):
            if not need_seg:
                continue
            self._write_image_header(i)  # will just exit if already written
            entry = data_entries[i, :]
            # how many elements will we write?
            write_els = (entry[1] - entry[0])*(entry[3] - entry[2])
            # write the data using this chipper
            drows = (entry[0]-start_indices[0], entry[1]-start_indices[0])
            dcols = (entry[2]-start_indices[1], entry[3]-start_indices[1])
            sinds = (
                start_indices[0] - self._image_segment_limits[i, 0],
                start_indices[1] - self._image_segment_limits[i, 2])
            self._writing_chippers[i](data[drows[0]:drows[1], dcols[0]:dcols[1]], sinds)
            # update how many pixels we have written to this segment
            self._pixels_written[i] += write_els
