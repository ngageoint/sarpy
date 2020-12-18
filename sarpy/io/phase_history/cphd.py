# -*- coding: utf-8 -*-
"""
Module for reading CPHD files - should support CPHD version 0.3 and 1.0.
"""

import struct
import os
from typing import Union, Tuple

import numpy

from sarpy.compliance import int_func, integer_types, string_types
from .cphd1_elements.CPHD import CPHDType, CPHDHeader
from .cphd0_3_elements.CPHD import CPHDType as CPHDType0_3, CPHDHeader as CPHDHeader0_3
from sarpy.io.general.utils import parse_xml_from_string, validate_range
from sarpy.io.general.base import BaseReader
from sarpy.io.general.bip import BIPChipper


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a CPHD file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    CPHDReader1_0|CPHDReader0_3|None
        Appropriate `CPHDReader` instance if CPHD file, `None` otherwise
    """

    try:
        cphd_details = CPHDDetails(file_name)
        print('File {} is determined to be a CPHD version {} file.'.format(file_name, cphd_details.cphd_version))
        return CPHDReader(cphd_details)
    except IOError:
        # we don't want to catch parsing errors, for now?
        return None


def _format_mapping(frm):
    """
    Map the single character struct format element to a numpy dtype and size in bytes.

    Parameters
    ----------
    frm : str

    Returns
    -------
    Tuple[numpy.number, int]
    """

    if frm == 'd':
        return numpy.float64, 8
    elif frm == 'f':
        return numpy.float32, 4
    elif frm == 'Q':
        return numpy.uint64, 8
    elif frm == 'q':
        return numpy.int64, 8
    elif frm in ('L', 'I'):
        return numpy.uint32, 4
    elif frm in ('l', 'i'):
        return numpy.int32, 4
    elif frm == 'H':
        return numpy.uint16, 2
    elif frm == 'h':
        return numpy.int16, 2
    elif frm == 'B':
        return numpy.uint8, 1
    elif frm == 'b':
        return numpy.int8, 1
    else:
        raise ValueError('Got unexpected struct format type {}'.format(frm))


#########
# Helper object for initially parses CPHD elements

class CPHDDetails(object):
    """
    The basic CPHD element parser.
    """

    __slots__ = (
        '_file_name', '_cphd_version', '_cphd_header', '_cphd_meta')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            The path to the CPHD file.
        """

        self._cphd_version = None
        self._cphd_header = None
        self._cphd_meta = None

        if not os.path.exists(file_name) or not os.path.isfile(file_name):
            raise IOError('path {} does not exist or is not a file'.format(file_name))
        self._file_name = file_name

        with open(self.file_name, 'rb') as fi:
            head_bytes = fi.read(10)
            if not head_bytes.startswith(b'CPHD'):
                raise IOError('File {} does not appear to be a CPHD file.'.format(self.file_name))

            self._extract_version(fi)
            self._extract_header(fi)
            self._extract_cphd(fi)

    @property
    def file_name(self):
        # type: () -> str
        """
        str: The CPHD filename.
        """

        return self._file_name

    @property
    def cphd_version(self):
        # type: () -> str
        """
        str: The CPHD version.
        """

        return self._cphd_version

    @property
    def cphd_header(self):
        # type: () -> Union[CPHDHeader, CPHDHeader0_3]
        """
        CPHDHeader|CPHDHeader0_3: The CPHD header object, which is version dependent.
        """

        return self._cphd_header

    @property
    def cphd_meta(self):
        # type: () -> Union[CPHDType, CPHDType0_3]
        """
        CPHDType|CPHDType0_3: The CPHD structure, which is version dependent.
        """

        return self._cphd_meta

    def _extract_version(self, fi):
        """
        Extract the version number from the file. This will advance the file
        object to the end of the initial header line.

        Parameters
        ----------
        fi
            The open file object, required to be opened in binary mode.

        Returns
        -------
        None
        """

        fi.seek(0)
        head_line = fi.readline().strip()
        parts = head_line.split(b'/')
        if len(parts) != 2:
            raise ValueError('Cannot extract CPHD version number from line {}'.format(head_line))
        cphd_version = parts[1].strip().decode('utf-8')
        self._cphd_version = cphd_version

    def _extract_header(self, fi):
        """
        Extract the header from the file. The file object is assumed to be advanced
        to the header location. This will advance to the file object to the end of
        the header section.

        Parameters
        ----------
        fi
            The open file object, required to be opened in binary mode.

        Returns
        -------
        None
        """

        if self.cphd_version.startswith('0.3'):
            self._cphd_header = CPHDHeader0_3.from_file_object(fi)
        elif self.cphd_version.startswith('1.0'):
            self._cphd_header = CPHDHeader.from_file_object(fi)
        else:
            raise ValueError('Got unhandled version number {}'.format(self.cphd_version))

    def _extract_cphd(self, fi):
        """
        Extract and interpret the CPHD structure from the file.

        Parameters
        ----------
        fi
            The open file object, required to be opened in binary mode.

        Returns
        -------
        None
        """

        xml = self._get_cphd_bytes(fi)
        if self.cphd_version.startswith('0.3'):
            the_type = CPHDType0_3
        elif self.cphd_version.startswith('1.0'):
            the_type = CPHDType
        else:
            raise ValueError('Got unhandled version number {}'.format(self.cphd_version))

        root_node, xml_ns = parse_xml_from_string(xml)
        if 'default' in xml_ns:
            self._cphd_meta = the_type.from_node(root_node, xml_ns, ns_key='default')
        else:
            self._cphd_meta = the_type.from_node(root_node, xml_ns)

    def _get_cphd_bytes(self, fi):
        """
        Extract the bytes representation of the CPHD structure.

        Parameters
        ----------
        fi
            The file object opened in binary mode.

        Returns
        -------
        bytes
        """

        header = self.cphd_header
        if header is None:
            raise ValueError('No cphd_header populated.')

        if self.cphd_version.startswith('0.3'):
            assert isinstance(header, CPHDHeader0_3)
            # extract the xml data
            fi.seek(header.XML_BYTE_OFFSET)
            xml = fi.read(header.XML_DATA_SIZE)
        elif self.cphd_version.startswith('1.0'):
            assert isinstance(header, CPHDHeader)
            # extract the xml data
            fi.seek(header.XML_BLOCK_BYTE_OFFSET)
            xml = fi.read(header.XML_BLOCK_SIZE)
        else:
            raise ValueError('Got unhandled version number {}'.format(self.cphd_version))
        return xml

    def get_cphd_bytes(self):
        """
        Extract the bytes representation of the CPHD structure.

        Returns
        -------
        bytes
        """

        with open(self.file_name, 'rb') as fi:
            return self._get_cphd_bytes(fi)


def _validate_cphd_details(cphd_details, version=None):
    """
    Validate the input argument.

    Parameters
    ----------
    cphd_details : str|CPHDDetails
    version : None|str

    Returns
    -------
    CPHDDetails
    """

    if isinstance(cphd_details, string_types):
        cphd_details = CPHDDetails(cphd_details)

    if not isinstance(cphd_details, CPHDDetails):
        raise TypeError('cphd_details is required to be a file path to a CPHD file '
                        'or CPHDDetails, got type {}'.format(cphd_details))

    if version is not None:
        if not cphd_details.cphd_version.startswith(version):
            raise ValueError('This CPHD file is required to be version {}, '
                             'got {}'.format(version, cphd_details.cphd_version))
    return cphd_details


class CPHDReader(BaseReader):
    """
    The Abstract CPHD reader instance, which just selects the proper CPHD reader
    class based on the CPHD version. Note that there is no __init__ method for
    this class, and it would be skipped regardless. Ensure that you make a direct
    call to the BaseReader.__init__() method when extending this class.
    """

    _cphd_details = None

    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            raise ValueError('The first argument of the constructor is required to '
                             'be a file_path or CPHDDetails instance.')
        cphd_details = _validate_cphd_details(args[0])

        if cphd_details.cphd_version.startswith('0.3'):
            return CPHDReader0_3(cphd_details)
        elif cphd_details.cphd_version.startswith('1.0'):
            return CPHDReader1_0(cphd_details)
        else:
            raise ValueError('Got unhandled CPHD version {}'.format(cphd_details.cphd_version))

    @property
    def cphd_details(self):
        # type: () -> CPHDDetails
        """
        CPHDDetails: The cphd details object.
        """

        return self._cphd_details

    @property
    def cphd_version(self):
        # type: () -> str
        """
        str: The CPHD version.
        """

        return self.cphd_details.cphd_version

    @property
    def cphd_meta(self):
        # type: () -> Union[CPHDType, CPHDType0_3]
        """
        CPHDType|CPHDType0_3: The CPHD structure, which is version dependent.
        """

        return self.cphd_details.cphd_meta

    @property
    def cphd_header(self):
        # type: () -> Union[CPHDHeader, CPHDHeader0_3]
        """
        CPHDHeader: The CPHD header object, which is version dependent.
        """

        return self.cphd_details.cphd_header

    @property
    def file_name(self):
        return self.cphd_details.file_name

    def _read_pvp_vector(self, pvp_block_offset, pvp_offset, vector_size, field_offset, frm, fld_siz, row_count, dim_range):
        """
        Read the given Per Vector parameter from the disc.

        Parameters
        ----------
        pvp_block_offset : int
            The Per Vector block offset from the start of the file, in bytes.
        pvp_offset : int
            The offset for the pvp vectors associated with this channel from the start
            of the PVP block, in bytes.
        vector_size : int
            The size of each vector, in bytes.
        field_offset : int
            The offset for field from the start of the vector, in bytes.
        frm : str
            The struct format string for the field
        fld_siz : int
            The size of the field, in bytes - probably 8 or 24.
        row_count : int
            The number of rows present for this CPHD channel.
        dim_range : None|int|Tuple[int, int]|Tuple[int, int, int]
            The indices for the vector parameter.

        Returns
        -------
        numpy.ndarray
        """

        # reformat the range definition
        the_range = validate_range(dim_range, row_count)

        dtype, elem_size = _format_mapping(frm)
        siz = int(fld_siz/elem_size)

        out_count = int_func((the_range[1] - the_range[0])/the_range[2])
        shp = (out_count, ) if siz == 1 else (out_count, siz)
        out = numpy.zeros(shp, dtype=dtype)

        current_offset = pvp_block_offset + pvp_offset + the_range[0]*vector_size + field_offset
        with open(self.cphd_details.file_name, 'rb') as fi:
            for i in range(out_count):
                fi.seek(current_offset)
                element = struct.unpack('>{}{}'.format(siz, frm), fi.read(fld_siz))
                out[i] = element[0] if siz == 1 else element
                current_offset += the_range[2]*vector_size
        return out

    def read_pvp_vector(self, variable, index, the_range=None):
        """
        Read the vector parameter for the given `variable` and CPHD channel.

        Parameters
        ----------
        variable : str
        index : int|str
            The CPHD channel index or identifier.
        the_range : None|int|List[int]|Tuple[int]
            The indices for the vector parameter. `None` returns all, otherwise
            a slice in the (non-traditional) form `([start, [stop, [stride]]])`.

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    def _fetch(self, range1, range2, index):
        """

        Parameters
        ----------
        range1 : tuple
        range2 : tuple
        index : int

        Returns
        -------
        numpy.ndarray
        """


        chipper = self._chipper[index]
        # NB: it is critical that there is no reorientation operation in CPHD.
        # noinspection PyProtectedMember
        range1, range2 = chipper._reorder_arguments(range1, range2)
        data = chipper(range1, range2)

        # fetch the scale data, if there is any
        scale = self.read_pvp_vector('AmpSF', index, the_range=range1)
        if scale is None:
            return data

        scale = numpy.cast['float32'](scale)
        # recast from double, so our data will remain float32
        if scale.size == 1:
            return scale[0]*data
        elif data.ndim == 1:
            return scale*data
        else:
            return scale[:, numpy.newaxis]*data

    def __call__(self, range1, range2, index=0):
        index = self._validate_index(index)
        return self._fetch(range1, range2, index)

    def __getitem__(self, item):
        item, index = self._validate_slice(item)
        chipper = self._chipper[index]
        # noinspection PyProtectedMember
        range1, range2 = chipper._slice_to_args(item)
        return self._fetch(range1, range2, index)

    def read_chip(self, dim1range, dim2range, index=0):
        return self.__call__(dim1range, dim2range, index=index)


class CPHDReader1_0(CPHDReader):
    """
    The CPHD version 1.0 reader.
    """

    def __new__(cls, *args, **kwargs):
        # we must override here, to avoid recursion with
        # the CPHDReader parent
        return object.__new__(cls)

    def __init__(self, cphd_details):
        """

        Parameters
        ----------
        cphd_details : str|CPHDDetails
        """

        self._cphd_details = _validate_cphd_details(cphd_details, version='1.0')
        chipper = self._create_chippers()
        BaseReader.__init__(self, None, chipper, reader_type="CPHD")

    @property
    def cphd_meta(self):
        # type: () -> CPHDType
        """
        CPHDType: The CPHD structure.
        """

        return self.cphd_details.cphd_meta

    @property
    def cphd_header(self):
        # type: () -> CPHDHeader
        """
        CPHDHeader: The CPHD header object.
        """

        return self.cphd_details.cphd_header

    def _create_chippers(self):
        chippers = []

        data = self.cphd_meta.Data
        sample_type = data.SignalArrayFormat
        raw_bands = 2
        output_bands = 1
        output_dtype = 'complex64'
        if sample_type == "CF8":
            raw_dtype = numpy.dtype('>f4')
        elif sample_type == "CI4":
            raw_dtype = numpy.dtype('>i2')
        elif sample_type == "CI2":
            raw_dtype = numpy.dtype('>i1')
        else:
            raise ValueError('Got unhandled signal array format {}'.format(sample_type))
        symmetry = (False, False, False)

        block_offset = self.cphd_header.SIGNAL_BLOCK_BYTE_OFFSET
        for entry in data.Channels:
            img_siz = (entry.NumVectors, entry.NumSamples)
            data_offset = entry.SignalArrayByteOffset
            chippers.append(BIPChipper(
                self.cphd_details.file_name, raw_dtype, img_siz, raw_bands, output_bands, output_dtype,
                symmetry=symmetry, transform_data='COMPLEX', data_offset=block_offset+data_offset))
        return tuple(chippers)

    def _validate_index(self, index):
        """
        Get corresponding integer index for CPHD channel.

        Parameters
        ----------
        index : int|str

        Returns
        -------
        None|int
        """

        cphd_meta = self.cphd_details.cphd_meta

        if isinstance(index, string_types):
            for i, channel in enumerate(cphd_meta.Data.Channels):
                if channel.Identifier == index:
                    return i
            raise KeyError('Cannot find CPHD channel for identifier {}'.format(index))
        else:
            int_index = int_func(index)
            if not (0 <= int_index < cphd_meta.Data.NumCPHDChannels):
                raise ValueError('index must be in the range [0, {})'.format(cphd_meta.Data.NumCPHDChannels))
            return int_index

    def read_pvp_vector(self, variable, index, the_range=None):
        # fetch the header object
        header = self.cphd_header
        # fetch the appropriate details from the cphd structure
        cphd_meta = self.cphd_meta
        int_index = self._validate_index(index)
        channel = cphd_meta.Data.Channels[int_index]
        pvp = cphd_meta.PVP
        row_count = channel.NumVectors
        pvp_block_offset = header.PVP_BLOCK_BYTE_OFFSET
        # get offset for this vector block
        pvp_offset = channel.PVPArrayByteOffset
        vector_size = cphd_meta.Data.NumBytesPVP
        # see if this variable/field is sensible
        res = pvp.get_offset_size_format(variable)
        if res is None:
            return None
        field_offset, fld_siz, frm = res
        return self._read_pvp_vector(
            pvp_block_offset, pvp_offset, vector_size, field_offset, frm, fld_siz, row_count, the_range)

    def read_support_array(self, index, dim1_range, dim2_range):
        """
        Read the support array.

        Parameters
        ----------
        index : int|str
            The support array integer index (of cphd.Data.SupportArrays list) or identifier.
        dim1_range : None|int|Tuple[int, int]|Tuple[int, int, int]
            The row data selection of the form `[start, [stop, [stride]]]`, and
            `None` defaults to all rows (i.e. `(0, NumRows, 1)`)
        dim2_range : None|int|Tuple[int, int]|Tuple[int, int, int]
            The column data selection of the form `[start, [stop, [stride]]]`, and
            `None` defaults to all rows (i.e. `(0, NumCols, 1)`)

        Returns
        -------
        numpy.ndarray
        """

        # find the support array basic details
        the_entry = None
        if isinstance(index, integer_types):
            the_entry = self.cphd_meta.Data.SupportArrays[index]
            identifier = the_entry.Identifier
        elif isinstance(index, string_types):
            identifier = index
            for entry in self.cphd_meta.Data.SupportArrays:
                if entry.Identifier == index:
                    the_entry = entry
                    break
            if the_entry is None:
                raise KeyError('Identifier {} not associated with a support array.'.format(identifier))
        else:
            raise TypeError('Got unexpected type {} for identifier'.format(type(index)))

        # validate the range definition
        range1 = validate_range(dim1_range, the_entry.NumRows)
        range2 = validate_range(dim2_range, the_entry.NumCols)
        # extract the support array metadata details
        details = self.cphd_meta.SupportArray.find_support_array(identifier)
        # determine array byte offset
        offset = self.cphd_header.SUPPORT_BLOCK_BYTE_OFFSET + the_entry.ArrayByteOffset
        # determine numpy dtype and depth of array
        dtype, depth = details.get_numpy_format()
        # set up the numpy memory map
        shape = (the_entry.NumRows, the_entry.NumCols) if depth == 1 else \
            (the_entry.NumRows, the_entry.NumCols, depth)
        mem_map = numpy.memmap(self.cphd_details.file_name,
                               dtype=dtype,
                               mode='r',
                               offset=offset,
                               shape=shape)
        if range1[0] == -1 and range1[2] < 0 and range2[0] == -1 and range2[2] < 0:
            data = mem_map[range1[0]::range1[2], range2[0]::range2[2]]
        elif range1[0] == -1 and range1[2] < 0:
            data = mem_map[range1[0]::range1[2], range2[0]:range2[1]:range2[2]]
        elif range2[0] == -1 and range2[2] < 0:
            data = mem_map[range1[0]:range1[1]:range1[2], range2[0]::range2[2]]
        else:
            data = mem_map[range1[0]:range1[1]:range1[2], range2[0]:range2[1]:range2[2]]
        # clean up the memmap object, probably unnecessary, and there SHOULD be a close method
        del mem_map
        return data


class CPHDReader0_3(CPHDReader):
    """
    The CPHD version 0.3 reader.
    """

    def __new__(cls, *args, **kwargs):
        # we must override here, to avoid recursion with
        # the CPHDReader parent
        return object.__new__(cls)

    def __init__(self, cphd_details):
        """

        Parameters
        ----------
        cphd_details : str|CPHDDetails
        """

        self._cphd_details = _validate_cphd_details(cphd_details, version='0.3')
        chipper = self._create_chippers()
        BaseReader.__init__(self, None, chipper, reader_type="CPHD")

    @property
    def cphd_meta(self):
        # type: () -> CPHDType0_3
        """
        CPHDType0_3: The CPHD structure, which is version dependent.
        """

        return self.cphd_details.cphd_meta

    @property
    def cphd_header(self):
        # type: () -> CPHDHeader0_3
        """
        CPHDHeader0_3: The CPHD header object.
        """

        return self.cphd_details.cphd_header

    def _create_chippers(self):
        chippers = []

        data = self.cphd_meta.Data
        sample_type = data.SampleType
        raw_bands = 2
        output_bands = 1
        output_dtype = 'complex64'
        if sample_type == "RE32F_IM32F":
            raw_dtype = numpy.dtype('>f4')
            bpp = 8
        elif sample_type == "RE16I_IM16I":
            raw_dtype = numpy.dtype('>i2')
            bpp = 4
        elif sample_type == "RE08I_IM08I":
            raw_dtype = numpy.dtype('>i1')
            bpp = 2
        else:
            raise ValueError('Got unhandled sample type {}'.format(sample_type))
        symmetry = (False, False, False)

        data_offset = self.cphd_header.CPHD_BYTE_OFFSET
        for entry in data.ArraySize:
            img_siz = (entry.NumVectors, entry.NumSamples)
            chippers.append(BIPChipper(
                self.cphd_details.file_name, raw_dtype, img_siz, raw_bands, output_bands, output_dtype,
                symmetry=symmetry, transform_data='COMPLEX', data_offset=data_offset))
            data_offset += img_siz[0]*img_siz[1]*bpp
        return tuple(chippers)

    def read_pvp_vector(self, variable, index, the_range=None):
        frm = 'd'  # all fields in CPHD 0.3 are of double type
        # fetch the header object
        header = self.cphd_header
        # fetch the appropriate details from the cphd structure
        cphd_meta = self.cphd_meta
        data = cphd_meta.Data
        if not isinstance(index, integer_types):
            index = int_func(index)
        if not (0 <= index < data.NumCPHDChannels):
            raise ValueError('index must be in the range [0, {})'.format(data.NumCPHDChannels))
        row_count = data.ArraySize[index].NumVectors
        # get overall offset
        pvp_block_offset = header.VB_BYTE_OFFSET
        # get offset for this vector block
        pvp_offset = data.NumBytesVBP*index
        vector_size = data.NumBytesVBP
        # see if this variable/field is sensible
        result = cphd_meta.VectorParameters.get_position_offset_and_size(variable)
        if result is None:
            return None
        field_offset, fld_siz = result
        return self._read_pvp_vector(
            pvp_block_offset, pvp_offset, vector_size, field_offset, frm, fld_siz, row_count, the_range)
