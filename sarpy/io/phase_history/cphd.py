# -*- coding: utf-8 -*-
"""
Module for reading and writing CPHD files - should support reading CPHD version 0.3 and 1.0 and writing version 1.0.
"""

import logging
import os
from typing import Union, Tuple, Dict
from collections import OrderedDict

import numpy

from sarpy.compliance import int_func, integer_types, string_types
from sarpy.io.general.utils import parse_xml_from_string, validate_range
from sarpy.io.general.base import AbstractWriter, BaseReader
from sarpy.io.general.bip import BIPChipper

from sarpy.io.phase_history.cphd1_elements.utils import binary_format_string_to_dtype
# noinspection PyProtectedMember
from sarpy.io.phase_history.cphd1_elements.CPHD import CPHDType, CPHDHeader, _CPHD_SECTION_TERMINATOR
from sarpy.io.phase_history.cphd0_3_elements.CPHD import CPHDType as CPHDType0_3, CPHDHeader as CPHDHeader0_3


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
            raise ValueError(
                'The first argument of the constructor is required to be a file_path '
                'or CPHDDetails instance.')
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
        scale = self.read_pvp_variable('AmpSF', index, the_range=range1)
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
        # type: (Union[None, int, Tuple[int, int], Tuple[int, int, int]], Union[None, int, Tuple[int, int], Tuple[int, int, int]], Union[int, str]) -> numpy.ndarray
        """
        Read the signal block associated with `index`, and given ranges.

        Parameters
        ----------
        dim1range : None|int|Tuple[int, int]|Tuple[int, int, int]
        dim2range : None|int|Tuple[int, int]|Tuple[int, int, int]
        index : int|str

        Returns
        -------
        numpy.ndarray
        """

        return self.__call__(dim1range, dim2range, index=index)

    def read_pvp_variable(self, variable, index, the_range=None):
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

    def read_pvp_array(self, index, the_range=None):
        """
        Read the PVP array from the requested channel.

        Parameters
        ----------
        index : int|str
            The support array integer index (of cphd.Data.Channels list) or identifier.
        the_range : None|int|List[int]|Tuple[int]
            The indices for the vector parameter. `None` returns all, otherwise
            a slice in the (non-traditional) form `([start, [stop, [stride]]])`.

        Returns
        -------
        pvp_array : numpy.ndarray
        """

        raise NotImplementedError

    def read_support_array(self, index, dim1_range, dim2_range):
        # type: (Union[int, str], Union[None, int, Tuple[int, int], Tuple[int, int, int]], Union[None, int, Tuple[int, int], Tuple[int, int, int]]) -> numpy.ndarray
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

        Raises
        ------
        TypeError
            If called on a CPHD version 0.3 reader.
        """

        raise NotImplementedError

    def read_pvp_block(self):
        raise NotImplementedError

    def read_support_block(self):
        raise NotImplementedError

    def read_signal_block(self):
        raise NotImplementedError


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

        self._channel_map = None  # type: Union[None, Dict[str, int]]
        self._support_array_map = None  # type: Union[None, Dict[str, int]]
        self._pvp_memmap = None  # type: Union[None, Dict[str, numpy.ndarray]]
        self._support_array_memmap = None  # type: Union[None, Dict[str, numpy.ndarray]]
        self._cphd_details = _validate_cphd_details(cphd_details, version='1.0')
        chipper = self._create_chippers()
        BaseReader.__init__(self, None, chipper, reader_type="CPHD")
        self._create_pvp_memmaps()
        self._create_support_array_memmaps()

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
        """
        Helper method for creating the various signal reading chipper elements.

        Returns
        -------
        Tuple[BIPChipper]
        """

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

    def _create_pvp_memmaps(self):
        """
        Helper method which creates the pvp mem_maps.

        Returns
        -------
        None
        """

        self._pvp_memmap = None
        if self.cphd_meta.Data.Channels is None:
            logging.error('No Data.Channels defined.')
            return
        if self.cphd_meta.PVP is None:
            logging.error('No PVP object defined.')
            return

        pvp_dtype = self.cphd_meta.PVP.get_vector_dtype()
        self._pvp_memmap = OrderedDict()
        self._channel_map = OrderedDict()
        for i, entry in enumerate(self.cphd_meta.Data.Channels):
            self._channel_map[entry.Identifier] = i
            offset = self.cphd_header.PVP_BLOCK_BYTE_OFFSET + entry.PVPArrayByteOffset
            shape = (entry.NumVectors, )
            self._pvp_memmap[entry.Identifier] = numpy.memmap(
                self.cphd_details.file_name, dtype=pvp_dtype, mode='r', offset=offset, shape=shape)

    def _create_support_array_memmaps(self):
        """
        Helper method which creates the support array mem_maps.

        Returns
        -------
        None
        """

        if self.cphd_meta.Data.SupportArrays is None:
            self._support_array_memmap = None
            return

        self._support_array_memmap = OrderedDict()
        self._support_array_map = OrderedDict()
        for i, entry in enumerate(self.cphd_meta.Data.SupportArrays):
            self._support_array_map[entry.Identifier] = i
            # extract the support array metadata details
            details = self.cphd_meta.SupportArray.find_support_array(entry.Identifier)
            # determine array byte offset
            offset = self.cphd_header.SUPPORT_BLOCK_BYTE_OFFSET + entry.ArrayByteOffset
            # determine numpy dtype and depth of array
            dtype, depth = details.get_numpy_format()
            # set up the numpy memory map
            shape = (entry.NumRows, entry.NumCols) if depth == 1 else (entry.NumRows, entry.NumCols, depth)
            self._support_array_memmap[entry.Identifier] = numpy.memmap(
                self.cphd_details.file_name, dtype=dtype, mode='r', offset=offset, shape=shape)

    def _validate_index(self, index):
        """
        Get corresponding integer index for CPHD channel.

        Parameters
        ----------
        index : int|str

        Returns
        -------
        int
        """

        cphd_meta = self.cphd_details.cphd_meta

        if isinstance(index, string_types):
            if index in self._channel_map:
                return self._channel_map[index]
            else:
                raise KeyError('Cannot find CPHD channel for identifier {}'.format(index))
        else:
            int_index = int_func(index)
            if not (0 <= int_index < cphd_meta.Data.NumCPHDChannels):
                raise ValueError('index must be in the range [0, {})'.format(cphd_meta.Data.NumCPHDChannels))
            return int_index

    def _validate_index_key(self, index):
        """
        Gets the corresponding identifier for the CPHD channel.

        Parameters
        ----------
        index : int|str

        Returns
        -------
        str
        """

        cphd_meta = self.cphd_details.cphd_meta

        if isinstance(index, string_types):
            if index in self._channel_map:
                return index
            else:
                raise KeyError('Cannot find CPHD channel for identifier {}'.format(index))
        else:
            int_index = int_func(index)
            if not (0 <= int_index < cphd_meta.Data.NumCPHDChannels):
                raise ValueError('index must be in the range [0, {})'.format(cphd_meta.Data.NumCPHDChannels))
            return cphd_meta.Data.Channels[int_index].Identifier

    def read_pvp_variable(self, variable, index, the_range=None):
        int_index = self._validate_index(index)
        # fetch the appropriate details from the cphd structure
        cphd_meta = self.cphd_meta
        channel = cphd_meta.Data.Channels[int_index]
        the_range = validate_range(the_range, channel.NumVectors)
        return self._pvp_memmap[channel.Identifier][variable][the_range[0]:the_range[1]:the_range[2]]

    def read_support_array(self, index, dim1_range, dim2_range):
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

    def read_pvp_array(self, index, the_range=None):
        int_index = self._validate_index(index)
        # fetch the appropriate details from the cphd structure
        cphd_meta = self.cphd_meta
        channel = cphd_meta.Data.Channels[int_index]
        the_range = validate_range(the_range, channel.NumVectors)
        return self._pvp_memmap[channel.Identifier][the_range[0]:the_range[1]:the_range[2]]

    def read_pvp_block(self):
        """
        Reads the entirety of the PVP block(s).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the PVP arrays.
        """
        return {chan.Identifier: self.read_pvp_array(chan.Identifier) for chan in self.cphd_meta.Data.Channels}

    def read_support_block(self):
        """
        Reads the entirety of support block(s).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        if self.cphd_meta.Data.SupportArrays:
            return {
                sa.Identifier: self.read_support_array(sa.Identifier, None, None)
                for sa in self.cphd_meta.Data.SupportArrays}
        else:
            return {}

    def read_signal_block(self):
        """
        Reads the entirety of signal block(s).

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        return {chan.Identifier: self.read_chip(None, None, index=chan.Identifier) for chan in self.cphd_meta.Data.Channels}


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
        self._create_pvp_memmaps()

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

    def _validate_index(self, index):
        """
        Validate integer index value for CPHD channel.

        Parameters
        ----------
        index : int

        Returns
        -------
        int
        """

        int_index = int_func(index)
        if not (0 <= int_index < self.cphd_meta.Data.NumCPHDChannels):
            raise ValueError('index must be in the range [0, {})'.format(self.cphd_meta.Data.NumCPHDChannels))
        return int_index

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

    def _create_pvp_memmaps(self):
        """
        Helper method which creates the pvp mem_maps.

        Returns
        -------
        None
        """

        self._pvp_memmap = None
        pvp_dtype = self.cphd_meta.VectorParameters.get_vector_dtype()
        self._pvp_memmap = []
        for i, entry in enumerate(self.cphd_meta.Data.ArraySize):
            offset = self.cphd_header.VB_BYTE_OFFSET + self.cphd_meta.Data.NumBytesVBP*i
            shape = (entry.NumVectors, )
            self._pvp_memmap.append(
                numpy.memmap(
                    self.cphd_details.file_name, dtype=pvp_dtype, mode='r', offset=offset, shape=shape))

    def read_pvp_variable(self, variable, index, the_range=None):
        int_index = self._validate_index(index)
        the_range = validate_range(the_range, self.cphd_meta.Data.ArraySize[int_index].NumVectors)
        return self._pvp_memmap[int_index][variable][the_range[0]:the_range[1]:the_range[2]]

    def read_support_array(self, index, dim1_range, dim2_range):
        raise TypeError('CPHD 0.3 does not support Support Arrays.')

    def read_pvp_array(self, index, the_range=None):
        int_index = self._validate_index(index)
        the_range = validate_range(the_range, self.cphd_meta.Data.ArraySize[int_index].NumVectors)
        return self._pvp_memmap[int_index][the_range[0]:the_range[1]:the_range[2]]

    def read_pvp_block(self):
        """
        Reads the entirety of the PVP block(s).

        Returns
        -------
        List[numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the PVP arrays.
        """

        return [self.read_pvp_array(chan) for chan in range(self.cphd_meta.Data.NumCPHDChannels)]

    def read_support_block(self):
        """
        Reads the entirety of support block(s).

        Returns
        -------
        Dict
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        raise TypeError('CPHD 0.3 does not have support arrays.')

    def read_signal_block(self):
        """
        Reads the full signal block(s).

        Returns
        -------
        List[numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the signal arrays.
        """

        return [self.read_chip(None, None, index=chan) for chan in range(self.cphd_meta.Data.NumCPHDChannels)]


class CPHDWriter1_0(AbstractWriter):
    """
    The CPHD version 1.0 writer.
    """

    __slots__ = ('_file_name', '_cphd_meta')

    def __init__(self, file_name, cphd_meta):
        """

        Parameters
        ----------
        file_name : str
        cphd_meta : sarpy.io.phase_history.cphd1_elements.CPHD.CPHDType
        """

        self._cphd_meta = cphd_meta
        super(CPHDWriter1_0, self).__init__(file_name)
        # TODO:
        #   0.) Make a context manager
        #   1.) We need counters to verify that the pvp, support, and signal arrays have been fully written
        #   2.) Method for fiddling with CPHD structure for defining offsets

    @property
    def cphd_meta(self):
        """
        sarpy.io.phase_history.cphd1_elements.CPHD.CPHDType: The cphd metadata
        """

        return self._cphd_meta

    def write_support_block(self, identifier, data):
        """
        Write support block data to the file.

        Parameters
        ----------
        identifier : str
        data : numpy.ndarray
        """

        # TODO: implement this...should have some kind of validation here
        pass

    def write_pvp_block(self, identifier, data):
        """
        Write the PVP block data to the file.

        Parameters
        ----------
        identifier : str
        data " numpy.ndarray
        """

        # TODO: implement this...should have some kind of validation here
        pass

    def __call__(self, data, start_indices=(0, 0), index=0):
        """
        Write the signal data to the file(s).

        Parameters
        ----------
        data : numpy.ndarray
            The complex data.
        start_indices : Tuple[int, int]
            The starting index for the data.
        index : int|str
            The signal index or identifier to which to write.

        Returns
        -------
        None
        """

        # TODO: implement this...should have some kind of validation here
        pass

    def write_chip(self, data, start_indices=(0, 0), index=0):
        """
        Write the signal data to the file(s). This is an alias to :code:`writer(data, start_indices)`.

        Parameters
        ----------
        data : numpy.ndarray
            The complex data.
        start_indices : tuple[int, int]
            The starting index for the data.
        index : int|str
            The signal index or identifier to which to write.
        """

        self.__call__(data, start_indices=start_indices, index=index)

    def write_file(self, pvp_block, signal_block, support_block=None):
        """
        Write the blocks to the file.

        Parameters
        ----------
        pvp_block: dict
            Dictionary of `numpy.ndarray` containing the PVP arrays.
            Keys must match `signal_block` and be consistent with `self.cphd_meta`
        signal_block: dict
            Dictionary of `numpy.ndarray` containing the signal arrays.
            Keys must match `pvp_block` and be consistent with `self.cphd_meta`
        support_block: dict, optional
            Dictionary of `numpy.ndarray` containing the support arrays.

        """

        # TODO: this should be pulled apart to using the above methods

        header = self.cphd_meta.make_file_header()

        def assert_equal(exp, act, desc='', id=''):
            assert exp == act, '{} {} expected:{}, actual:{}'.format(desc, id, exp, act)

        # Validate support block, if present
        if header.SUPPORT_BLOCK_SIZE or support_block:
            assert header.SUPPORT_BLOCK_SIZE and support_block, 'cphd_meta and support_block are inconsistent'

            expected_support_ids = {s.Identifier for s in self.cphd_meta.Data.SupportArrays}
            assert expected_support_ids == set(support_block.keys()), 'support_block keys do not match those in cphd_meta'

            for sa in self.cphd_meta.Data.SupportArrays:
                input_sa = support_block[sa.Identifier]

                expected_shape = (sa.NumRows, sa.NumCols)
                actual_shape = input_sa.shape[:2] # first two dimensions as dtype may be structured
                assert_equal(expected_shape, actual_shape, 'support array shape', sa.Identifier)

                expected_element_bytes = sa.BytesPerElement
                actual_element_bytes = input_sa.nbytes / numpy.prod(actual_shape)
                assert_equal(expected_element_bytes, actual_element_bytes, 'support array #bytes', sa.Identifier)

        # Verify that channels are consistent
        expected_channels = {c.Identifier for c in self.cphd_meta.Data.Channels}
        assert expected_channels == set(signal_block.keys()), 'signal_block keys do not match those in cphd_meta'
        assert expected_channels == set(pvp_block.keys()), 'pvp_block keys do not match those in cphd_meta'

        # Verify signal_block shapes and sizes
        if self.cphd_meta.Data.SignalCompressionID is None:
            expected_element_bytes = binary_format_string_to_dtype(self.cphd_meta.Data.SignalArrayFormat).itemsize
            for chan in self.cphd_meta.Data.Channels:
                input_sig = signal_block[chan.Identifier]

                expected_shape = (chan.NumVectors, chan.NumSamples)
                actual_shape = input_sig.shape[:2]
                assert_equal(expected_shape, actual_shape, 'signal array shape', chan.Identifier)

                actual_element_bytes = input_sig.nbytes / numpy.prod(actual_shape)
                assert_equal(expected_element_bytes, actual_element_bytes, 'signal array #bytes', chan.Identifier)

        # Validate PVP block
        input_pvp_types = {v.dtype for v in pvp_block.values()}
        assert len(input_pvp_types) == 1, 'All channels in PVP block must be uniformly structured'

        actual_pvp_bytes = input_pvp_types.pop().itemsize
        assert_equal(self.cphd_meta.Data.NumBytesPVP, actual_pvp_bytes, 'Data.NumBytesPVP')
        assert_equal(self.cphd_meta.PVP.get_pvp_dtype().itemsize, actual_pvp_bytes, 'PVP Node #Bytes')

        for chan in self.cphd_meta.Data.Channels:
            input_pvp = pvp_block[chan.Identifier]

            expected_num_vectors = chan.NumVectors
            actual_num_vectors = input_pvp.shape[0]
            assert_equal(expected_num_vectors, actual_num_vectors, 'PVP shape', chan.Identifier)

        def sort_and_check_offsets(meta_items, input_items, offset_name):
            sorted_meta = sorted(meta_items, key=lambda x: getattr(x, offset_name))
            first_item = sorted_meta[0]
            assert getattr(first_item, offset_name) == 0, 'first byte offset in block must be 0'
            actual_offset = input_items[first_item.Identifier].nbytes
            for m in sorted_meta[1:]:
                expected_offset = getattr(m, offset_name)
                assert_equal(expected_offset, actual_offset, offset_name, m.Identifier)

                actual_offset += input_items[m.Identifier].nbytes

            return [x.Identifier for x in sorted_meta]

        if header.SUPPORT_BLOCK_SIZE:
            support_order = sort_and_check_offsets(self.cphd_meta.Data.SupportArrays, support_block, 'ArrayByteOffset')
        signal_order = sort_and_check_offsets(self.cphd_meta.Data.Channels, signal_block, 'SignalArrayByteOffset')
        pvp_order = sort_and_check_offsets(self.cphd_meta.Data.Channels, pvp_block, 'PVPArrayByteOffset')

        # Write file
        def write_array(array):
            array.astype(array.dtype.newbyteorder('big')).tofile(outfile)

        with open(self._file_name, "wb") as outfile:
            # Header
            outfile.write(header.to_string().encode())
            outfile.write(_CPHD_SECTION_TERMINATOR)

            # XML
            outfile.seek(header.XML_BLOCK_BYTE_OFFSET)
            outfile.write(self.cphd_meta.to_xml_bytes())
            outfile.write(_CPHD_SECTION_TERMINATOR)

            # Support Arrays
            if header.SUPPORT_BLOCK_SIZE:
                outfile.seek(header.SUPPORT_BLOCK_BYTE_OFFSET)
                for id in support_order:
                    write_array(support_block[id])

            # PVP
            outfile.seek(header.PVP_BLOCK_BYTE_OFFSET)
            for id in pvp_order:
                write_array(pvp_block[id])

            # Signal
            outfile.seek(header.SIGNAL_BLOCK_BYTE_OFFSET)
            for id in signal_order:
                write_array(signal_block[id])
