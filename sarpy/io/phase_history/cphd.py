"""
Module for reading and writing CPHD files - should support reading CPHD version 0.3 and 1.0 and writing version 1.0.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
from typing import Union, List, Tuple, Dict, BinaryIO, Optional, Sequence
from collections import OrderedDict

import numpy

from sarpy.io.general.utils import is_file_like
from sarpy.io.general.base import AbstractReader, AbstractWriter, SarpyIOError
from sarpy.io.general.data_segment import DataSegment, NumpyMemmapSegment
from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.slice_parsing import verify_subscript, verify_slice

from sarpy.io.phase_history.base import CPHDTypeReader
from sarpy.io.phase_history.cphd1_elements.utils import binary_format_string_to_dtype
# noinspection PyProtectedMember
from sarpy.io.phase_history.cphd1_elements.CPHD import CPHDType as CPHDType1_0, CPHDHeader as CPHDHeader1_0, _CPHD_SECTION_TERMINATOR
from sarpy.io.phase_history.cphd0_3_elements.CPHD import CPHDType as CPHDType0_3, CPHDHeader as CPHDHeader0_3

logger = logging.getLogger(__name__)

_unhandled_version_text = 'Got unhandled CPHD version number `{}`'
_missing_channel_identifier_text = 'Cannot find CPHD channel for identifier `{}`'
_index_range_text = 'index must be in the range `[0, {})`'


#########
# Helper object for initially parses CPHD elements

class CPHDDetails(object):
    """
    The basic CPHD element parser.
    """

    __slots__ = (
        '_file_name', '_file_object', '_closed', '_close_after', '_cphd_version', '_cphd_header', '_cphd_meta')

    def __init__(self, file_object: str):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
            The path to or file like object referencing the CPHD file.
        """

        self._closed = False
        self._close_after = None
        self._cphd_version = None
        self._cphd_header = None
        self._cphd_meta = None
        self._file_object = None  # type: Optional[BinaryIO]

        if isinstance(file_object, str):
            if not os.path.exists(file_object) or not os.path.isfile(file_object):
                raise SarpyIOError('path {} does not exist or is not a file'.format(file_object))
            self._file_name = file_object
            self._file_object = open(file_object, 'rb')
            self._close_after = True
        elif is_file_like(file_object):
            self._file_object = file_object
            if hasattr(file_object, 'name') and isinstance(file_object.name, str):
                self._file_name = file_object.name
            else:
                self._file_name = '<file like object>'
            self._close_after = False
        else:
            raise TypeError('Got unsupported input type {}'.format(type(file_object)))

        self._file_object.seek(0, os.SEEK_SET)
        head_bytes = self._file_object.read(10)
        if not isinstance(head_bytes, bytes):
            raise ValueError('Input file like object not open in bytes mode.')
        if not head_bytes.startswith(b'CPHD'):
            raise SarpyIOError('File {} does not appear to be a CPHD file.'.format(self.file_name))

        self._extract_version()
        self._extract_header()
        self._extract_cphd()

    @property
    def file_name(self) -> str:
        """
        str: The CPHD filename.
        """

        return self._file_name

    @property
    def file_object(self) -> BinaryIO:
        """
        BinaryIO: The binary file object
        """
        # noinspection PyTypeChecker
        return self._file_object

    @property
    def cphd_version(self) -> str:
        """
        str: The CPHD version.
        """

        return self._cphd_version

    @property
    def cphd_meta(self) -> Union[CPHDType1_0, CPHDType0_3]:
        """
        CPHDType1_0|CPHDType0_3: The CPHD metadata object, which is version dependent.
        """

        return self._cphd_meta

    @property
    def cphd_header(self) -> Union[CPHDHeader1_0, CPHDHeader0_3]:
        """
        CPHDHeader1_0|CPHDHeader0_3: The CPHD header object, which is version dependent.
        """

        return self._cphd_header

    def _extract_version(self) -> None:
        """
        Extract the version number from the file. This will advance the file
        object to the end of the initial header line.
        """

        self.file_object.seek(0, os.SEEK_SET)
        head_line = self.file_object.readline().strip()
        parts = head_line.split(b'/')
        if len(parts) != 2:
            raise ValueError('Cannot extract CPHD version number from line {}'.format(head_line))
        cphd_version = parts[1].strip().decode('utf-8')
        self._cphd_version = cphd_version

    def _extract_header(self) -> None:
        """
        Extract the header from the file. The file object is assumed to be advanced
        to the header location. This will advance to the file object to the end of
        the header section.
        """

        if self.cphd_version.startswith('0.3'):
            self._cphd_header = CPHDHeader0_3.from_file_object(self._file_object)
        elif self.cphd_version.startswith('1.0'):
            self._cphd_header = CPHDHeader1_0.from_file_object(self._file_object)
        else:
            raise ValueError(_unhandled_version_text.format(self.cphd_version))

    def _extract_cphd(self) -> None:
        """
        Extract and interpret the CPHD structure from the file.
        """

        xml = self.get_cphd_bytes()
        if self.cphd_version.startswith('0.3'):
            the_type = CPHDType0_3
        elif self.cphd_version.startswith('1.0'):
            the_type = CPHDType1_0
        else:
            raise ValueError(_unhandled_version_text.format(self.cphd_version))

        self._cphd_meta = the_type.from_xml_string(xml)

    def get_cphd_bytes(self) -> bytes:
        """
        Extract the (uninterpreted) bytes representation of the CPHD structure.

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
            self.file_object.seek(header.XML_BYTE_OFFSET, os.SEEK_SET)
            xml = self.file_object.read(header.XML_DATA_SIZE)
        elif self.cphd_version.startswith('1.0'):
            assert isinstance(header, CPHDHeader1_0)
            # extract the xml data
            self.file_object.seek(header.XML_BLOCK_BYTE_OFFSET, os.SEEK_SET)
            xml = self.file_object.read(header.XML_BLOCK_SIZE)
        else:
            raise ValueError(_unhandled_version_text.format(self.cphd_version))
        return xml

    def close(self):
        if self._closed:
            return

        if self._close_after:
            if hasattr(self.file_object, 'close'):
                self.file_object.close()
        self._file_object = None
        self._closed = True

    def __del__(self):
        self.close()


def _validate_cphd_details(cphd_details: Union[str, CPHDDetails],
                           version: Optional[str]=None) -> CPHDDetails:
    """
    Validate the input argument.

    Parameters
    ----------
    cphd_details : str|CPHDDetails
    version : None|str

    Returns
    -------
    CPHDDetails

    Raises
    ------
    TypeError
        The input was neither path to a CPHD file or a CPHDDetails instance
    ValueError
        The CPHD file was the incorrect (specified) version
    """

    if isinstance(cphd_details, str):
        cphd_details = CPHDDetails(cphd_details)

    if not isinstance(cphd_details, CPHDDetails):
        raise TypeError('cphd_details is required to be a file path to a CPHD file '
                        'or CPHDDetails, got type {}'.format(cphd_details))

    if version is not None and not cphd_details.cphd_version.startswith(version):
        raise ValueError('This CPHD file is required to be version {}, '
                         'got {}'.format(version, cphd_details.cphd_version))
    return cphd_details


class CPHDReader(CPHDTypeReader):
    """
    The Abstract CPHD reader instance, which just selects the proper CPHD reader
    class based on the CPHD version. Note that there is no __init__ method for
    this class, and it would be skipped regardless. Ensure that you make a direct
    call to the BaseReader.__init__() method when extending this class.
    """

    __slots__ = ('_cphd_details', )

    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            raise ValueError(
                'The first argument of the constructor is required to be a file_path '
                'or CPHDDetails instance.')
        if is_file_like(args[0]):
            raise ValueError('File like object input not supported for CPHD reading at this time.')
        cphd_details = _validate_cphd_details(args[0])

        if cphd_details.cphd_version.startswith('0.3'):
            return CPHDReader0_3(cphd_details)
        elif cphd_details.cphd_version.startswith('1.0'):
            return CPHDReader1_0(cphd_details)
        else:
            raise ValueError('Got unhandled CPHD version {}'.format(cphd_details.cphd_version))

    @property
    def cphd_details(self) -> CPHDDetails:
        """
        CPHDDetails: The cphd details object.
        """

        return self._cphd_details

    @property
    def cphd_version(self) -> str:
        """
        str: The CPHD version.
        """

        return self.cphd_details.cphd_version

    @property
    def cphd_header(self) -> Union[CPHDHeader1_0, CPHDHeader0_3]:
        """
        CPHDHeader1_0|CPHDHeader0_3: The CPHD header object, which is version dependent.
        """

        return self.cphd_details.cphd_header

    @property
    def file_name(self) -> str:
        return self.cphd_details.file_name

    def read_pvp_variable(
            self,
            variable: str,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> Optional[numpy.ndarray]:
        raise NotImplementedError

    def read_pvp_array(
            self,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> numpy.ndarray:
        raise NotImplementedError

    def read_pvp_block(self) -> Dict[Union[int, str], numpy.ndarray]:
        raise NotImplementedError

    def read_signal_block(self) -> Dict[Union[int, str], numpy.ndarray]:
        raise NotImplementedError

    def read_signal_block_raw(self) -> Dict[Union[int, str], numpy.ndarray]:
        raise NotImplementedError

    def close(self):
        CPHDTypeReader.close(self)
        if hasattr(self, '_cphd_details'):
            if hasattr(self._cphd_details, 'close'):
                self._cphd_details.close()
            del self._cphd_details


class CPHDReader1_0(CPHDReader):
    """
    The CPHD version 1.0 reader.
    """

    def __new__(cls, *args, **kwargs):
        # we must override here, to avoid recursion with
        # the CPHDReader parent
        return object.__new__(cls)

    def __init__(self, cphd_details: Union[str, CPHDDetails]):
        """

        Parameters
        ----------
        cphd_details : str|CPHDDetails
        """

        self._channel_map = None  # type: Union[None, Dict[str, int]]
        self._pvp_memmap = None  # type: Union[None, Dict[str, numpy.ndarray]]
        self._support_array_memmap = None  # type: Union[None, Dict[str, numpy.ndarray]]
        self._cphd_details = _validate_cphd_details(cphd_details, version='1.0')

        CPHDTypeReader.__init__(self, None, self._cphd_details.cphd_meta)
        # set data segments after setting up the pvp information, because
        #   we need the AmpSf to set up the format function for the data segment
        self._create_pvp_memmaps()
        self._create_support_array_memmaps()

        data_segments = self._create_data_segments()
        AbstractReader.__init__(self, data_segments, reader_type='CPHD')

    @property
    def cphd_meta(self) -> CPHDType1_0:
        """
        CPHDType1_0: The CPHD structure.
        """

        return self._cphd_meta

    @property
    def cphd_header(self) -> CPHDHeader1_0:
        """
        CPHDHeader1_0: The CPHD header object.
        """

        return self.cphd_details.cphd_header

    def _create_data_segments(self) -> List[DataSegment]:
        """
        Helper method for creating the various signal data segments.

        Returns
        -------
        List[DataSegment]
        """

        data_segments = []

        data = self.cphd_meta.Data
        sample_type = data.SignalArrayFormat

        if sample_type == "CF8":
            raw_dtype = numpy.dtype('>f4')
        elif sample_type == "CI4":
            raw_dtype = numpy.dtype('>i2')
        elif sample_type == "CI2":
            raw_dtype = numpy.dtype('>i1')
        else:
            raise ValueError('Got unhandled signal array format {}'.format(sample_type))

        block_offset = self.cphd_header.SIGNAL_BLOCK_BYTE_OFFSET
        for entry in data.Channels:
            amp_sf = self.read_pvp_variable('AmpSF', entry.Identifier)
            format_function = ComplexFormatFunction(raw_dtype, order='IQ', amplitude_scaling=amp_sf)
            raw_shape = (entry.NumVectors, entry.NumSamples, 2)
            data_offset = entry.SignalArrayByteOffset
            data_segments.append(
                NumpyMemmapSegment(
                    self.cphd_details.file_object, block_offset+data_offset,
                    raw_dtype, raw_shape, formatted_dtype='complex64', formatted_shape=raw_shape[:2],
                    format_function=format_function, close_file=False))
        return data_segments

    def _create_pvp_memmaps(self) -> None:
        """
        Helper method which creates the pvp mem_maps.

        Returns
        -------
        None
        """

        self._pvp_memmap = None
        if self.cphd_meta.Data.Channels is None:
            logger.error('No Data.Channels defined.')
            return
        if self.cphd_meta.PVP is None:
            logger.error('No PVP object defined.')
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

    def _create_support_array_memmaps(self) -> None:
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
        for i, entry in enumerate(self.cphd_meta.Data.SupportArrays):
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

    def _validate_index(self, index: Union[int, str]) -> int:
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

        if isinstance(index, str):
            if index in self._channel_map:
                return self._channel_map[index]
            else:
                raise KeyError(_missing_channel_identifier_text.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < cphd_meta.Data.NumCPHDChannels):
                raise ValueError(_index_range_text.format(cphd_meta.Data.NumCPHDChannels))
            return int_index

    def _validate_index_key(self, index: Union[int, str]) -> str:
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

        if isinstance(index, str):
            if index in self._channel_map:
                return index
            else:
                raise KeyError(_missing_channel_identifier_text.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < cphd_meta.Data.NumCPHDChannels):
                raise ValueError(_index_range_text.format(cphd_meta.Data.NumCPHDChannels))
            return cphd_meta.Data.Channels[int_index].Identifier

    def read_support_array(self,
                           index: Union[int, str],
                           *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]]) -> numpy.ndarray:
        # find the support array identifier
        if isinstance(index, int):
            the_entry = self.cphd_meta.Data.SupportArrays[index]
            index = the_entry.Identifier
        if not isinstance(index, str):
            raise TypeError('Got unexpected type {} for identifier'.format(type(index)))

        the_memmap = self._support_array_memmap[index]

        if len(ranges) == 0:
            return numpy.copy(the_memmap[:])

        # noinspection PyTypeChecker
        subscript = verify_subscript(ranges, the_memmap.shape)
        return numpy.copy(the_memmap[subscript])

    def read_support_block(self) -> Dict[str, numpy.ndarray]:
        if self.cphd_meta.Data.SupportArrays:
            return {
                sa.Identifier: self.read_support_array(sa.Identifier)
                for sa in self.cphd_meta.Data.SupportArrays}
        else:
            return {}

    def read_pvp_variable(
            self,
            variable: str,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> Optional[numpy.ndarray]:
        index_key = self._validate_index_key(index)
        the_memmap = self._pvp_memmap[index_key]
        the_slice = verify_slice(the_range, the_memmap.shape[0])
        if variable in the_memmap.dtype.fields:
            return numpy.copy(the_memmap[variable][the_slice])
        else:
            return None

    def read_pvp_array(
            self,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> numpy.ndarray:
        index_key = self._validate_index_key(index)
        the_memmap = self._pvp_memmap[index_key]
        the_slice = verify_slice(the_range, the_memmap.shape[0])
        return numpy.copy(the_memmap[the_slice])

    def read_pvp_block(self) -> Dict[str, numpy.ndarray]:
        return {chan.Identifier: self.read_pvp_array(chan.Identifier) for chan in self.cphd_meta.Data.Channels}

    def read_signal_block(self) -> Dict[str, numpy.ndarray]:
        return {chan.Identifier: numpy.copy(self.read(index=chan.Identifier)) for chan in self.cphd_meta.Data.Channels}

    def read_signal_block_raw(self) -> Dict[Union[int, str], numpy.ndarray]:
        return {chan.Identifier: numpy.copy(self.read_raw(index=chan.Identifier)) for chan in self.cphd_meta.Data.Channels}

    def read_chip(self,
             *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]],
             index: Union[int, str]=0,
             squeeze: bool=True) -> numpy.ndarray:
        """
        This is identical to :meth:`read`, and presented for backwards compatibility.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, Tuple[int, ...], slice]]
        index : int|str
        squeeze : bool

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        :meth:`read`.
        """

        return self.__call__(*ranges, index=index, raw=False, squeeze=squeeze)

    def read(self,
             *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]],
             index: Union[int, str]=0,
             squeeze: bool=True) -> numpy.ndarray:
        """
        Read formatted data from the given data segment. Note this is an alias to the
        :meth:`__call__` called as
        :code:`reader(*ranges, index=index, raw=False, squeeze=squeeze)`.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, Tuple[int, ...], slice]]
            The slice definition appropriate for `data_segment[index].read()` usage.
        index : int|str
            The data_segment index or channel identifier.
        squeeze : bool
            Squeeze length 1 dimensions out of the shape of the return array?

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        See :meth:`sarpy.io.general.data_segment.DataSegment.read`.
        """

        return self.__call__(*ranges, index=index, raw=False, squeeze=squeeze)

    def read_raw(self,
                 *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]],
                 index: Union[int, str]=0,
                 squeeze: bool=True) -> numpy.ndarray:
        """
        Read raw data from the given data segment. Note this is an alias to the
        :meth:`__call__` called as
        :code:`reader(*ranges, index=index, raw=True, squeeze=squeeze)`.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, Tuple[int, ...], slice]]
            The slice definition appropriate for `data_segment[index].read()` usage.
        index : int|str
            The data_segment index or cphd channel identifier.
        squeeze : bool
            Squeeze length 1 dimensions out of the shape of the return array?

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        See :meth:`sarpy.io.general.data_segment.DataSegment.read_raw`.
        """

        return self.__call__(*ranges, index=index, raw=True, squeeze=squeeze)

    def __call__(self,
                 *ranges: Sequence[Union[None, int, slice]],
                 index: int=0,
                 raw: bool=False,
                 squeeze: bool=True) -> numpy.ndarray:
        index = self._validate_index(index)
        return AbstractReader.__call__(*ranges, index=index, raw=raw, squeeze=squeeze)


class CPHDReader0_3(CPHDReader):
    """
    The CPHD version 0.3 reader.
    """

    def __new__(cls, *args, **kwargs):
        # we must override here, to avoid recursion with
        # the CPHDReader parent
        return object.__new__(cls)

    def __init__(self, cphd_details: Union[str, CPHDDetails]):
        """

        Parameters
        ----------
        cphd_details : str|CPHDDetails
        """

        self._cphd_details = _validate_cphd_details(cphd_details, version='0.3')
        CPHDTypeReader.__init__(self, None, self._cphd_details.cphd_meta)
        self._create_pvp_memmaps()

        data_segments = self._create_data_segment()
        AbstractReader.__init__(self, data_segments, reader_type="CPHD")

    @property
    def cphd_meta(self) -> CPHDType0_3:
        """
        CPHDType0_3: The CPHD structure, which is version dependent.
        """

        return self._cphd_meta

    @property
    def cphd_header(self) -> CPHDHeader0_3:
        """
        CPHDHeader0_3: The CPHD header object.
        """

        return self.cphd_details.cphd_header

    def _validate_index(self, index: int) -> int:
        """
        Validate integer index value for CPHD channel.

        Parameters
        ----------
        index : int

        Returns
        -------
        int
        """

        int_index = int(index)
        if not (0 <= int_index < self.cphd_meta.Data.NumCPHDChannels):
            raise ValueError(_index_range_text.format(self.cphd_meta.Data.NumCPHDChannels))
        return int_index

    def _create_data_segment(self) -> List[DataSegment]:
        data_segments = []

        data = self.cphd_meta.Data
        sample_type = data.SampleType
        if sample_type == "RE32F_IM32F":
            raw_dtype = numpy.dtype('>f4')
        elif sample_type == "RE16I_IM16I":
            raw_dtype = numpy.dtype('>i2')
        elif sample_type == "RE08I_IM08I":
            raw_dtype = numpy.dtype('>i1')
        else:
            raise ValueError('Got unhandled sample type {}'.format(sample_type))

        data_offset = self.cphd_header.CPHD_BYTE_OFFSET
        for index, entry in enumerate(data.ArraySize):
            amp_sf = self.read_pvp_variable('AmpSF', index)
            format_function = ComplexFormatFunction(raw_dtype, order='IQ', amplitude_scaling=amp_sf)
            raw_shape = (entry.NumVectors, entry.NumSamples, 2)
            data_segments.append(
                NumpyMemmapSegment(
                    self.cphd_details.file_object, data_offset,
                    raw_dtype, raw_shape, formatted_dtype='complex64', formatted_shape=raw_shape[:2],
                    format_function=format_function, close_file=False))
            data_offset += raw_shape[0]*raw_shape[1]*2*raw_dtype.itemsize
        return data_segments

    def _create_pvp_memmaps(self) -> None:
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

    def read_pvp_variable(
            self,
            variable: str,
            index: int,
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> Optional[numpy.ndarray]:
        int_index = self._validate_index(index)
        the_memmap = self._pvp_memmap[int_index]
        the_slice = verify_slice(the_range, the_memmap.shape[0])
        if variable in the_memmap.dtype.fields:
            return numpy.copy(the_memmap[variable][the_slice])
        else:
            return None

    def read_pvp_array(
            self,
            index: int,
            the_range: Union[None, int, Tuple[int, ...], slice]=None) -> numpy.ndarray:
        int_index = self._validate_index(index)
        the_memmap = self._pvp_memmap[int_index]
        the_slice = verify_slice(the_range, the_memmap.shape[0])
        return numpy.copy(the_memmap[the_slice])

    def read_pvp_block(self) -> Dict[int, numpy.ndarray]:
        """
        Reads the entirety of the PVP block(s).

        Returns
        -------
        Dict[int, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the PVP arrays.
        """

        return {chan: self.read_pvp_array(chan) for chan in range(self.cphd_meta.Data.NumCPHDChannels)}

    def read_signal_block(self) -> Dict[int, numpy.ndarray]:
        return {chan: self.read(index=chan) for chan in range(self.cphd_meta.Data.NumCPHDChannels)}

    def read_signal_block_raw(self) -> Dict[int, numpy.ndarray]:
        return {chan: self.read_raw(index=chan) for chan in range(self.cphd_meta.Data.NumCPHDChannels)}

    def __call__(self,
                 *ranges: Sequence[Union[None, int, slice]],
                 index: int=0,
                 raw: bool=False,
                 squeeze: bool=True) -> numpy.ndarray:
        index = self._validate_index(index)
        return AbstractReader.__call__(*ranges, index=index, raw=raw, squeeze=squeeze)


class CPHDWriter1_0(AbstractWriter):
    """
    The CPHD version 1.0 writer.
    """

    __slots__ = (
        '_file_name', '_file_object', '_cphd_meta', '_cphd_header',
        '_pvp_memmaps', '_support_memmaps', '_signal_data_segments',
        '_can_write_regular_data', '_channel_map', '_support_map',
        '_writing_state', '_closed')

    def __init__(self,
                 file_name: str,
                 cphd_meta: CPHDType1_0,
                 check_existence: bool=True):
        """

        Parameters
        ----------
        file_name : str
        cphd_meta : CPHDType1_0
        check_existence : bool
            Should we check if the given file already exists, and raises an exception if so?
        """

        if check_existence and os.path.exists(file_name):
            raise SarpyIOError(
                'File {} already exists, and a new CPHD file can not be created '
                'at this location'.format(file_name))
        self._file_object = open(self._file_name, "wb")

        self._pvp_memmaps = None  # type: Optional[Dict[str, numpy.ndarray]]
        self._support_memmaps = None  # type: Optional[Dict[str, numpy.ndarray]]
        self._signal_data_segments = None  # type: Optional[Dict[str, DataSegment]]
        self._can_write_regular_data = None  # type: Optional[Dict[str, bool]]
        self._channel_map = None  # type: Optional[Dict[str, int]]
        self._support_map = None  # type: Optional[Dict[str, int]]
        self._writing_state = {'header': False, 'pvp': {}, 'support': {}}
        self._closed = False
        self._cphd_meta = cphd_meta
        self._cphd_header = cphd_meta.make_file_header()

        data_segment = self._prepare_for_writing()
        AbstractWriter.__init__(self, data_segment)

    @property
    def cphd_meta(self) -> CPHDType1_0:
        """
        CPHDType1_0: The cphd metadata
        """

        return self._cphd_meta

    @staticmethod
    def _verify_dtype(
            obs_dtype: numpy.dtype,
            exp_dtype: numpy.dtype,
            purpose: str) -> None:
        """
        This is a helper function for comparing two structured array dtypes.

        Parameters
        ----------
        obs_dtype : numpy.dtype
        exp_dtype : numpy.dtype
        purpose : str
        """

        if obs_dtype.fields is None or exp_dtype.fields is None:
            raise ValueError('structure array dtype required.')

        observed_dtype = sorted(
            [(field, dtype_info) for field, dtype_info in obs_dtype.fields.items()],
            key=lambda x: x[1][1])
        expected_dtype = sorted(
            [(field, dtype_info) for field, dtype_info in exp_dtype.fields.items()],
            key=lambda x: x[1][1])
        if len(observed_dtype) != len(expected_dtype):
            raise ValueError('Observed dtype for {} does not match the expected dtype.'.format(purpose))
        for obs_entry, exp_entry in zip(observed_dtype, expected_dtype):
            if obs_entry[1][1] != exp_entry[1][1]:
                raise ValueError(
                    'Observed dtype for {} does not match the expected dtype\nobserved {}\nexpected {}.'.format(
                        purpose, observed_dtype, expected_dtype))
            if obs_entry[0] != exp_entry[0]:
                logger.warning(
                    'Got mismatched field names (observed {}, expected {}) for {}.'.format(
                        obs_entry[0], exp_entry[0], purpose))

    def _validate_channel_index(self, index: Union[int, str]) -> int:
        """
        Get corresponding integer index for CPHD channel.

        Parameters
        ----------
        index : int|str

        Returns
        -------
        int
        """

        if isinstance(index, str):
            if index in self._channel_map:
                return self._channel_map[index]
            else:
                raise KeyError(_missing_channel_identifier_text.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < self.cphd_meta.Data.NumCPHDChannels):
                raise ValueError(_index_range_text.format(self.cphd_meta.Data.NumCPHDChannels))
            return int_index

    def _validate_channel_key(self, index: Union[int, str]) -> str:
        """
        Gets the corresponding identifier for the CPHD channel.

        Parameters
        ----------
        index : int|str

        Returns
        -------
        str
        """

        if isinstance(index, str):
            if index in self._channel_map:
                return index
            else:
                raise KeyError(_missing_channel_identifier_text.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < self.cphd_meta.Data.NumCPHDChannels):
                raise ValueError(_index_range_text.format(self.cphd_meta.Data.NumCPHDChannels))
            return self.cphd_meta.Data.Channels[int_index].Identifier

    def _validate_support_index(self, index: Union[int, str]) -> int:
        """
        Get corresponding integer index for support array.

        Parameters
        ----------
        index : int|str

        Returns
        -------
        int
        """

        if isinstance(index, str):
            if index in self._support_map:
                return self._support_map[index]
            else:
                raise KeyError('Cannot find support array for identifier {}'.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < len(self.cphd_meta.Data.SupportArrays)):
                raise ValueError(_index_range_text.format(len(self.cphd_meta.Data.SupportArrays)))
            return int_index

    def _validate_support_key(self, index: Union[int, str]) -> str:
        """
        Gets the corresponding identifier for the support array.

        Parameters
        ----------
        index : int|str

        Returns
        -------
        str
        """

        if isinstance(index, str):
            if index in self._support_map:
                return index
            else:
                raise KeyError('Cannot find support array for identifier {}'.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < len(self.cphd_meta.Data.SupportArrays)):
                raise ValueError(_index_range_text.format(len(self.cphd_meta.Data.SupportArrays)))
            return self.cphd_meta.Data.SupportArrays[int_index].Identifier

    def _initialize_writing(self) -> List[DataSegment]:
        """
        Initializes the counters/state variables for writing progress checks.
        Expected to be called only by _prepare_for_writing().

        Returns
        -------
        List[DataSegment]
        """

        self._pvp_memmaps = {}

        # setup the PVP memmaps
        pvp_dtype = self.cphd_meta.PVP.get_vector_dtype()
        for i, entry in enumerate(self.cphd_meta.Data.Channels):
            self._channel_map[entry.Identifier] = i
            # create the pvp mem map
            offset = self._cphd_header.PVP_BLOCK_BYTE_OFFSET + entry.PVPArrayByteOffset
            shape = (entry.NumVectors, )
            self._pvp_memmaps[entry.Identifier] = numpy.memmap(
                self._file_name, dtype=pvp_dtype, mode='r+', offset=offset, shape=shape)
            # create the pvp writing state variable
            self._writing_state['pvp'][entry.Identifier] = 0

        self._support_memmaps = {}
        self._support_map = {}
        if self.cphd_meta.Data.SupportArrays is not None:
            for i, entry in enumerate(self.cphd_meta.Data.SupportArrays):
                self._support_map[entry.Identifier] = i
                # extract the support array metadata details
                details = self.cphd_meta.SupportArray.find_support_array(entry.Identifier)
                # determine array byte offset
                offset = self._cphd_header.SUPPORT_BLOCK_BYTE_OFFSET + entry.ArrayByteOffset
                # determine numpy dtype and depth of array
                dtype, depth = details.get_numpy_format()
                # set up the numpy memory map
                shape = (entry.NumRows, entry.NumCols) if depth == 1 else (entry.NumRows, entry.NumCols, depth)
                self._support_memmaps[entry.Identifier] = numpy.memmap(
                    self._file_name, dtype=dtype, mode='r+', offset=offset, shape=shape)
                self._writing_state['support'][entry.Identifier] = 0

        # setup the signal data_segment (this is used for formatting issues)
        no_amp_sf = (self.cphd_meta.PVP.AmpSF is None)
        self._signal_data_segments = {}
        self._channel_map = {}
        self._can_write_regular_data = {}
        signal_data_segments = []
        signal_dtype = binary_format_string_to_dtype(self.cphd_meta.Data.SignalArrayFormat)
        for i, entry in enumerate(self.cphd_meta.Data.Channels):
            self._can_write_regular_data[entry.Identifier] = no_amp_sf
            raw_shape = (entry.NumVectors, entry.NumSamples, 2)
            format_function = ComplexFormatFunction(signal_dtype, order='IQ')

            offset = self._cphd_header.SIGNAL_BLOCK_BYTE_OFFSET + entry.SignalArrayByteOffset
            data_segment = NumpyMemmapSegment(
                self._file_object, offset, signal_dtype, raw_shape,
                formatted_dtype='complex64', formatted_shape=raw_shape[:2],
                format_function=format_function, mode='w', close_file=False)
            signal_data_segments.append(data_segment)

            self._signal_data_segments[entry.Identifier] = data_segment
            # create the signal writing state variable
            self._writing_state['signal'][entry.Identifier] = 0
        return signal_data_segments

    def _prepare_for_writing(self) -> Optional[List[DataSegment]]:
        """
        Prepare all elements for writing.
        """

        if self._writing_state['header']:
            logger.warning('The header for CPHD file {} has already been written. Exiting.'.format(self._file_name))
            return

        # write header
        self._file_object.write(self._cphd_header.to_string().encode())
        self._file_object.write(_CPHD_SECTION_TERMINATOR)
        # write cphd xml
        self._file_object.seek(self._cphd_header.XML_BLOCK_BYTE_OFFSET, os.SEEK_SET)
        self._file_object.write(self.cphd_meta.to_xml_bytes())
        self._file_object.write(_CPHD_SECTION_TERMINATOR)
        self._writing_state['header'] = True

        return self._initialize_writing()

    def write_support_array(self,
                            identifier: Union[int, str],
                            data: numpy.ndarray) -> None:
        """
        Write support array data to the file.

        Parameters
        ----------
        identifier : int|str
        data : numpy.ndarray
        """

        int_index = self._validate_support_index(identifier)
        identifier = self._validate_support_key(identifier)
        entry = self.cphd_meta.Data.SupportArrays[int_index]

        if data.shape != (entry.NumRows, entry.NumCols):
            raise ValueError('Support data shape is not compatible with provided')

        total_pixels = entry.NumRows*entry.NumCols
        # write the data
        self._support_memmaps[identifier][:] = data
        # update the count of written data
        self._writing_state['support'][identifier] += total_pixels

    def write_pvp_array(self,
                        identifier: Union[int, str],
                        data: numpy.ndarray) -> None:
        """
        Write the PVP array data to the file.

        Parameters
        ----------
        identifier : int|str
        data : numpy.ndarray
        """

        def validate_dtype():
            self._verify_dtype(data.dtype, self._pvp_memmaps[identifier].dtype, 'PVP channel {}'.format(identifier))

        int_index = self._validate_channel_index(identifier)
        identifier = self._validate_channel_key(identifier)
        entry = self.cphd_meta.Data.Channels[int_index]
        validate_dtype()

        if data.ndim != 1:
            raise ValueError('Provided data is required to be one dimensional')
        if data.shape[0] != entry.NumVectors:
            raise ValueError('Provided data must have size determined by NumVectors')

        if self.cphd_meta.PVP.AmpSF is not None:
            amp_sf = numpy.copy(data['AmpSF'][:])
            # noinspection PyUnresolvedReferences
            self._signal_data_segments[identifier].format_function.set_amplitude_scaling(amp_sf)
            self._can_write_regular_data[identifier] = True

        self._pvp_memmaps[identifier][:] = data
        self._writing_state['pvp'][identifier] += data.shape[0]

    def write_support_block(self, support_block: Dict[Union[int, str], numpy.ndarray]) -> None:
        """
        Write support block to the file.

        Parameters
        ----------
        support_block: dict
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        expected_support_ids = {s.Identifier for s in self.cphd_meta.Data.SupportArrays}
        assert expected_support_ids == set(support_block), 'support_block keys do not match those in cphd_meta'
        for identifier, array in support_block.items():
            self.write_support_array(identifier, array)

    def write_pvp_block(self, pvp_block: Dict[Union[int, str], numpy.ndarray]) -> None:
        """
        Write PVP block to the file.

        Parameters
        ----------
        pvp_block: dict
            Dictionary of `numpy.ndarray` containing the PVP arrays.
        """

        expected_channels = {c.Identifier for c in self.cphd_meta.Data.Channels}
        assert expected_channels == set(pvp_block), 'pvp_block keys do not match those in cphd_meta'
        for identifier, array in pvp_block.items():
            self.write_pvp_array(identifier, array)

    def write_signal_block(self, signal_block: Dict[Union[int, str], numpy.ndarray]) -> None:
        """
        Write signal block to the file.

        Parameters
        ----------
        signal_block: dict
            Dictionary of `numpy.ndarray` containing the signal arrays in complex64 format.
        """

        expected_channels = {c.Identifier for c in self.cphd_meta.Data.Channels}
        assert expected_channels == set(signal_block), 'signal_block keys do not match those in cphd_meta'
        for identifier, array in signal_block.items():
            self.write(array, index=identifier)

    def write_signal_block_raw(self, signal_block):
        """
        Write signal block to the file.

        Parameters
        ----------
        signal_block: dict
            Dictionary of `numpy.ndarray` containing the the raw formatted
            (i.e. file storage format) signal arrays.
        """

        expected_channels = {c.Identifier for c in self.cphd_meta.Data.Channels}
        assert expected_channels == set(signal_block), 'signal_block keys do not match those in cphd_meta'
        for identifier, array in signal_block.items():
            self.write_raw(array, index=identifier)

    def write_file(self,
                   pvp_block: Dict[Union[int, str], numpy.ndarray],
                   signal_block: Dict[Union[int, str], numpy.ndarray],
                   support_block: Optional[Dict[Union[int, str], numpy.ndarray]]=None):
        """
        Write the blocks to the file.

        Parameters
        ----------
        pvp_block: Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the PVP arrays.
            Keys must be consistent with `self.cphd_meta`
        signal_block: Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the complex64 formatted signal
            arrays.
            Keys must be consistent with `self.cphd_meta`
        support_block: None|Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        self.write_pvp_block(pvp_block)
        if support_block:
            self.write_support_block(support_block)
        self.write_signal_block(signal_block)

    def write_file_raw(self,
                   pvp_block: Dict[Union[int, str], numpy.ndarray],
                   signal_block: Dict[Union[int, str], numpy.ndarray],
                   support_block: Optional[Dict[Union[int, str], numpy.ndarray]]=None):
        """
        Write the blocks to the file.

        Parameters
        ----------
        pvp_block: Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the PVP arrays.
            Keys must be consistent with `self.cphd_meta`
        signal_block: Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the raw formatted
            (i.e. file storage format) signal arrays.
            Keys must be consistent with `self.cphd_meta`
        support_block: None|Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        self.write_pvp_block(pvp_block)
        if support_block:
            self.write_support_block(support_block)
        self.write_signal_block_raw(signal_block)

    def write_chip(self,
              data: numpy.ndarray,
              start_indices: Union[None, int, Tuple[int, ...]] = None,
              subscript: Union[None, Tuple[slice, ...]] = None,
              index: Union[int, str]=0) -> None:
        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def write(self,
              data: numpy.ndarray,
              start_indices: Union[None, int, Tuple[int, ...]] = None,
              subscript: Union[None, Tuple[slice, ...]] = None,
              index: Union[int, str]=0) -> None:
        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def write_raw(self,
              data: numpy.ndarray,
              start_indices: Union[None, int, Tuple[int, ...]]=None,
              subscript: Union[None, Tuple[slice, ...]]=None,
              index: Union[int, str]=0) -> None:
        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def __call__(self,
                 data: numpy.ndarray,
                 start_indices: Union[None, int, Tuple[int, ...]]=None,
                 subscript: Union[None, Tuple[slice, ...]]=None,
                 index: Union[int, str]=0,
                 raw: bool=False) -> None:
        int_index = self._validate_channel_index(index)

        identifier = self._validate_channel_key(index)
        if not raw and not self._can_write_regular_data[identifier]:
            raise ValueError(
                'The channel `{}` has an AmpSF whihc has not been determined,\n\t'
                'but the corresponding PVP block has not yet been written'.format(identifier))

        AbstractWriter.__call__(self, data, start_indices=start_indices, subscript=subscript, index=int_index, raw=raw)

    def _check_fully_written(self) -> bool:
        """
        Verifies that the file is fully written, and logs messages describing any
        unwritten elements at error level. This is expected only to be called by
        the close() method.

        Returns
        -------
        bool
            The status of the file writing completeness.
        """

        if self._closed:
            return True

        if self.cphd_meta is None:
            return True # incomplete initialization or some other inherent problem

        status = True
        pvp_message = ''
        support_message = ''

        for entry in self.cphd_meta.Data.Channels:
            pvp_rows = entry.NumVectors
            if self._writing_state['pvp'][entry.Identifier] < pvp_rows:
                status = False
                pvp_message += 'identifier {}, {} of {} vectors written\n'.format(
                    entry.Identifier, self._writing_state['pvp'][entry.Identifier], pvp_rows)

        if self.cphd_meta.Data.SupportArrays is not None:
            for entry in self.cphd_meta.Data.SupportArrays:
                support_pixels = entry.NumRows*entry.NumCols
                if self._writing_state['support'][entry.Identifier] < support_pixels:
                    status = False
                    support_message += 'identifier {}, {} of {} pixels written\n'.format(
                        entry.Identifier, self._writing_state['support'][entry.Identifier], support_pixels)

        if not status:
            logger.error('CPHD file {} is not completely written, and the result may be corrupt.', self._file_name)
            if pvp_message != '':
                logger.error('PVP block(s) incompletely written\n{}', pvp_message)
            if support_message != '':
                logger.error('Support block(s) incompletely written\n{}',support_message)
        return status

    def close(self):
        if self._closed:
            return

        fully_written = self._check_fully_written()
        AbstractWriter.close(self)

        if hasattr(self, '_file_object') and hasattr(self._file_object, 'close'):
            self._file_object.close()
        self._file_object = None
        if not fully_written:
            raise SarpyIOError('CPHD file {} is not fully written'.format(self._file_name))


def is_a(file_name: str) -> Optional[CPHDReader]:
    """
    Tests whether a given file_name corresponds to a CPHD file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    CPHDReader|None
        Appropriate `CPHDTypeReader` instance if CPHD file, `None` otherwise
    """

    try:
        cphd_details = CPHDDetails(file_name)
        logger.info('File {} is determined to be a CPHD version {} file.'.format(file_name, cphd_details.cphd_version))
        return CPHDReader(cphd_details)
    except SarpyIOError:
        # we don't want to catch parsing errors, for now?
        return None
