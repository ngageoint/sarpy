"""
Module for reading and writing CPHD files. Support reading CPHD version 0.3 and 1
and writing version 1.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import os
from typing import Union, List, Tuple, Dict, BinaryIO, Optional, Sequence
from collections import OrderedDict

import numpy

from sarpy.io.general.utils import is_file_like, is_real_file
from sarpy.io.general.base import BaseReader, BaseWriter, SarpyIOError
from sarpy.io.general.data_segment import DataSegment, NumpyArraySegment, \
    NumpyMemmapSegment
from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.slice_parsing import verify_subscript, verify_slice

from sarpy.io.phase_history.base import CPHDTypeReader
from sarpy.io.phase_history.cphd1_elements.CPHD import CPHDType as CPHDType1, \
    CPHDHeader as CPHDHeader1, CPHD_SECTION_TERMINATOR
from sarpy.io.phase_history.cphd0_3_elements.CPHD import CPHDType as CPHDType0_3, \
    CPHDHeader as CPHDHeader0_3
from sarpy.io.phase_history.cphd_schema import get_namespace, get_default_tuple

logger = logging.getLogger(__name__)

_unhandled_version_text = 'Got unhandled CPHD version number `{}`'
_missing_channel_identifier_text = 'Cannot find CPHD channel for identifier `{}`'
_index_range_text = 'index must be in the range `[0, {})`'


#########
# Helper object for initially parses CPHD elements

class AmpScalingFunction(ComplexFormatFunction):
    __slots__ = (
        '_amplitude_scaling', )
    _allowed_ordering = ('IQ', )

    def __init__(
            self,
            raw_dtype: Union[str, numpy.dtype],
            raw_shape: Optional[Tuple[int, ...]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Tuple[int, ...]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            band_dimension: int = -1,
            amplitude_scaling: Optional[numpy.ndarray] = None):
        """

        Parameters
        ----------
        raw_dtype : str|numpy.dtype
            The raw datatype. Valid options dependent on the value of order.
        raw_shape : None|Tuple[int, ...]
        formatted_shape : None|Tuple[int, ...]
        reverse_axes : None|Tuple[int, ...]
        transpose_axes : None|Tuple[int, ...]
        band_dimension : int
            Which band is the complex dimension, **after** the transpose operation.
        amplitude_scaling : None|numpy.ndarray
            This is here to support the presence of a scaling in CPHD or CRSD usage.
            This requires that `raw_dtype` in `[int8, int16]`, `band_dimension`
            is the final dimension and neither `reverse_axes` nor `transpose_axes`
            is populated.
        """

        ComplexFormatFunction.__init__(
            self, raw_dtype, 'IQ', raw_shape=raw_shape, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes, band_dimension=band_dimension)
        self._amplitude_scaling = None
        self.set_amplitude_scaling(amplitude_scaling)

    @property
    def amplitude_scaling(self) -> Optional[numpy.ndarray]:
        """
        The scaling multiplier array, for CPHD/CRSD usage.

        Returns
        -------
        Optional[numpy.ndarray]
        """

        return self._amplitude_scaling

    def set_amplitude_scaling(
            self,
            array: Optional[numpy.ndarray]) -> None:
        """
        Set the amplitude scaling array.

        Parameters
        ----------
        array : None|numpy.ndarray

        Returns
        -------
        None
        """

        if array is None:
            self._amplitude_scaling = None
            return

        if not isinstance(array, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(array)))
        if array.ndim != 1:
            raise ValueError('requires a one dimensional array')
        if array.dtype.name not in ['float32', 'float64']:
            raise ValueError('requires a numpy.ndarray of float32 or 64 dtype, got {}'.format(array.dtype))
        if array.dtype.name != 'float32':
            array = numpy.cast['float32'](array)

        if self.order not in ['MP', 'PM']:
            logger.warning(
                'A magnitude lookup table has been supplied,\n\t'
                'but the order is not one of `MP` or `PM`')
        # NB: more validation as part of validate_shapes
        if self._raw_dtype.name not in ['int8', 'int16']:
            raise ValueError(
                'A scaling multiplier has been supplied,\n\t'
                'but the raw datatype is not `int8` or `int16`.')
        self._amplitude_scaling = array
        self._validate_amplitude_scaling()

    def _validate_amplitude_scaling(self) -> None:
        if self._amplitude_scaling is None or self._raw_shape is None:
            return

        if self.band_dimension not in [-1, self.raw_ndim-1]:
            raise ValueError('Use of scaling multiplier requires band is the final dimension')
        if self.transpose_axes is not None or self.reverse_axes is not None:
            raise ValueError('Use of scaling multiplier requires null reverse_axes and transpose_axes')
        if self._amplitude_scaling.size != self.raw_shape[0]:
            raise ValueError(
                'Use of scaling multiplier requires the array length\n\t'
                'and the first dimension of raw_shape match.')

    def _forward_functional_step(
            self,
            data: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        out = ComplexFormatFunction._forward_functional_step(self, data, subscript)

        # NB: subscript is in raw coordinates, but we have verified that
        #   the first dimension is unchanged
        if self._amplitude_scaling is not None:
            out = self._amplitude_scaling[subscript[0]]*out

        return out

    def _reverse_functional_step(
            self,
            data: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        # NB: subscript is in formatted coordinates, but we have verified that
        #   transpose_axes is None and band_dimension is the final dimension
        if self._amplitude_scaling is not None:
            data = (1./self._amplitude_scaling[subscript[0]])*data

        return ComplexFormatFunction._reverse_functional_step(self, data, subscript)

    def validate_shapes(self) -> None:
        ComplexFormatFunction.validate_shapes(self)
        self._validate_amplitude_scaling()


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
    def cphd_meta(self) -> Union[CPHDType1, CPHDType0_3]:
        """
        CPHDType1|CPHDType0_3: The CPHD metadata object, which is version dependent.
        """

        return self._cphd_meta

    @property
    def cphd_header(self) -> Union[CPHDHeader1, CPHDHeader0_3]:
        """
        CPHDHeader1|CPHDHeader0_3: The CPHD header object, which is version dependent.
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
        elif self.cphd_version.startswith('1.'):
            self._cphd_header = CPHDHeader1.from_file_object(self._file_object)
        else:
            raise ValueError(_unhandled_version_text.format(self.cphd_version))

    def _extract_cphd(self) -> None:
        """
        Extract and interpret the CPHD structure from the file.
        """

        xml = self.get_cphd_bytes()
        if self.cphd_version.startswith('0.3'):
            the_type = CPHDType0_3
        elif self.cphd_version.startswith('1.'):
            the_type = CPHDType1
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
        elif self.cphd_version.startswith('1.'):
            assert isinstance(header, CPHDHeader1)
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


def _validate_cphd_details(
        cphd_details: Union[str, CPHDDetails],
        version: Union[None, str, Sequence[str]] = None) -> CPHDDetails:
    """
    Validate the input argument.

    Parameters
    ----------
    cphd_details : str|CPHDDetails
    version : None|str|Sequence[str]

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
    if version is not None:
        if isinstance(version, str) and not cphd_details.cphd_version.startswith(version):
            raise ValueError(
                'This CPHD file is required to be version {},\n\t'
                'got {}'.format(version, cphd_details.cphd_version))
        else:
            val = False
            for entry in version:
                if cphd_details.cphd_version.startswith(entry):
                    val = True
                    break
            if not val:
                raise ValueError(
                    'This CPHD file is required to be one of version {},\n\t'
                    'got {}'.format(version, cphd_details.cphd_version))

    return cphd_details


##########
# Reading


class CPHDReader(CPHDTypeReader):
    """
    The Abstract CPHD reader instance, which just selects the proper CPHD reader
    class based on the CPHD version. Note that there is no __init__ method for
    this class, and it would be skipped regardless. Ensure that you make a direct
    call to the BaseReader.__init__() method when extending this class.

    **Updated in version 1.3.0** for reading changes.
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
            return object.__new__(CPHDReader0_3)
        elif cphd_details.cphd_version.startswith('1.'):
            return object.__new__(CPHDReader1)
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
    def cphd_header(self) -> Union[CPHDHeader1, CPHDHeader0_3]:
        """
        CPHDHeader1|CPHDHeader0_3: The CPHD header object, which is version dependent.
        """

        return self.cphd_details.cphd_header

    @property
    def file_name(self) -> str:
        return self.cphd_details.file_name

    def read_pvp_variable(
            self,
            variable: str,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice] = None) -> Optional[numpy.ndarray]:
        raise NotImplementedError

    def read_pvp_array(
            self,
            index: Union[int, str],
            the_range: Union[None, int, Tuple[int, ...], slice] = None) -> numpy.ndarray:
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


class CPHDReader1(CPHDReader):
    """
    The CPHD version 1 reader.

    **Updated in version 1.3.0** for reading changes.
    """
    _allowed_versions = ('1.0', '1.1')

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
        self._cphd_details = _validate_cphd_details(cphd_details, version=self._allowed_versions)

        CPHDTypeReader.__init__(self, None, self._cphd_details.cphd_meta)
        # set data segments after setting up the pvp information, because
        #   we need the AmpSf to set up the format function for the data segment
        self._create_pvp_memmaps()
        self._create_support_array_memmaps()

        data_segments = self._create_data_segments()
        BaseReader.__init__(self, data_segments, reader_type='CPHD')

    @property
    def cphd_meta(self) -> CPHDType1:
        """
        CPHDType1: The CPHD structure.
        """

        return self._cphd_meta

    @property
    def cphd_header(self) -> CPHDHeader1:
        """
        CPHDHeader1: The CPHD header object.
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
            format_function = AmpScalingFunction(raw_dtype, amplitude_scaling=amp_sf)
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

    def read_support_array(
            self,
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
            the_range: Union[None, int, Tuple[int, ...], slice] = None) -> Optional[numpy.ndarray]:
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
            the_range: Union[None, int, Tuple[int, ...], slice] = None) -> numpy.ndarray:
        index_key = self._validate_index_key(index)
        the_memmap = self._pvp_memmap[index_key]
        the_slice = verify_slice(the_range, the_memmap.shape[0])
        return numpy.copy(the_memmap[the_slice])

    def read_pvp_block(self) -> Dict[str, numpy.ndarray]:
        return {chan.Identifier: self.read_pvp_array(chan.Identifier)
                for chan in self.cphd_meta.Data.Channels}

    def read_signal_block(self) -> Dict[str, numpy.ndarray]:
        return {chan.Identifier: numpy.copy(self.read(index=chan.Identifier))
                for chan in self.cphd_meta.Data.Channels}

    def read_signal_block_raw(self) -> Dict[Union[int, str], numpy.ndarray]:
        return {chan.Identifier: numpy.copy(self.read_raw(index=chan.Identifier))
                for chan in self.cphd_meta.Data.Channels}

    def read_chip(
            self,
            *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]],
            index: Union[int, str] = 0,
            squeeze: bool = True) -> numpy.ndarray:
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

    def read(
            self,
            *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]],
            index: Union[int, str] = 0,
            squeeze: bool = True) -> numpy.ndarray:
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

    def read_raw(
            self,
            *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]],
            index: Union[int, str] = 0,
            squeeze: bool = True) -> numpy.ndarray:
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

    def __call__(
            self,
            *ranges: Sequence[Union[None, int, slice]],
            index: int = 0,
            raw: bool = False,
            squeeze: bool = True) -> numpy.ndarray:
        index = self._validate_index(index)
        return BaseReader.__call__(self, *ranges, index=index, raw=raw, squeeze=squeeze)


class CPHDReader0_3(CPHDReader):
    """
    The CPHD version 0.3 reader.

    **Updated in version 1.3.0** for reading changes.
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
        BaseReader.__init__(self, data_segments, reader_type="CPHD")

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
            format_function = AmpScalingFunction(raw_dtype, amplitude_scaling=amp_sf)
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
            the_range: Union[None, int, Tuple[int, ...], slice] = None) -> Optional[numpy.ndarray]:
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
            the_range: Union[None, int, Tuple[int, ...], slice] = None) -> numpy.ndarray:
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

    def __call__(
            self,
            *ranges: Sequence[Union[None, int, slice]],
            index: int = 0,
            raw: bool = False,
            squeeze: bool = True) -> numpy.ndarray:
        index = self._validate_index(index)
        return BaseReader.__call__(self, *ranges, index=index, raw=raw, squeeze=squeeze)


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


###########
# Writing

class ElementDetails(object):
    __slots__ = (
        '_item_offset', '_item_bytes', '_item_written')

    def __init__(self, item_offset: int, item_bytes: Optional[bytes] = None):
        self._item_offset = None
        self._item_bytes = None
        self._item_written = False

        self.item_offset = item_offset
        self.item_bytes = item_bytes

    @property
    def item_offset(self) -> Optional[int]:
        """
        int: The item offset.
        """

        return self._item_offset

    @item_offset.setter
    def item_offset(self, value: int) -> None:
        value = int(value)
        if self._item_offset is not None and self._item_offset != value:
            raise ValueError("item_offset is read only after being initially defined.")
        self._item_offset = value

    @property
    def item_bytes(self) -> Optional[bytes]:
        """
        None|bytes: The item bytes.
        """

        return self._item_bytes

    @item_bytes.setter
    def item_bytes(self, value: bytes) -> None:
        if self._item_bytes is not None:
            raise ValueError("item_bytes is read only after being initially defined.")
        if value is None:
            self._item_bytes = None
            return

        if not isinstance(value, bytes):
            raise TypeError('item_bytes must be of type bytes')
        self._item_bytes = value

    @property
    def item_written(self) -> bool:
        """
        bool: Has the item been written?
        """

        return self._item_written

    @item_written.setter
    def item_written(self, value: bool):
        value = bool(value)
        if self._item_written and not value:
            raise ValueError(
                'item_written has already been set to True,\n\t'
                'it cannot be reverted to False')
        self._item_written = value

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

        file_object.seek(self.item_offset, os.SEEK_SET)
        file_object.write(self.item_bytes)
        self.item_written = True


class CPHDWritingDetails(object):
    __slots__ = (
        '_header', '_header_written', '_meta',
        '_channel_map', '_support_map',
        '_pvp_details', '_support_details', '_signal_details')

    def __init__(self, meta: CPHDType1, check_older_version: bool = False):

        self._header = None
        self._header_written = False
        self._meta = None
        self._channel_map = {}
        self._support_map = {}
        self._pvp_details = None
        self._support_details = None
        self._signal_details = None

        self.meta = meta
        self._set_header(check_older_version)

        # initialize the information for the pvp, support, and signal details
        self._populate_pvp_details()
        self._populate_support_details()
        self._populate_signal_details()

    @property
    def header(self) -> CPHDHeader1:
        return self._header

    def _set_header(self, check_older_version: bool):
        if check_older_version:
            use_version_tuple = self.meta.version_required()
        else:
            use_version_tuple = get_default_tuple()
        use_version_string = '{}.{}.{}'.format(*use_version_tuple)
        self._header = self.meta.make_file_header(use_version=use_version_string)

    @property
    def use_version(self) -> str:
        return self.header.use_version

    @property
    def meta(self) -> CPHDType1:
        """
        CPHDType1: The metadata
        """

        return self._meta

    @meta.setter
    def meta(self, value):
        if self._meta is not None:
            raise ValueError('meta is read only once initialized.')
        if not isinstance(value, CPHDType1):
            raise TypeError('meta must be of type {}'.format(CPHDType1))
        self._meta = value

    def _populate_pvp_details(self) -> None:
        if self._pvp_details is not None:
            raise ValueError('pvp_details can not be initialized again')
        pvp_details = []
        for i, entry in enumerate(self.meta.Data.Channels):
            self._channel_map[entry.Identifier] = i
            offset = self.header.PVP_BLOCK_BYTE_OFFSET + entry.PVPArrayByteOffset
            pvp_details.append(ElementDetails(offset))
        self._pvp_details = tuple(pvp_details)

    def _populate_support_details(self) -> None:
        if self._support_details is not None:
            raise ValueError('support_details can not be initialized again')

        if self.meta.Data.SupportArrays is None:
            self._signal_details = None
            return

        support_details = []
        for i, entry in enumerate(self.meta.Data.SupportArrays):
            self._support_map[entry.Identifier] = i
            offset = self.header.SUPPORT_BLOCK_BYTE_OFFSET + entry.ArrayByteOffset
            support_details.append(ElementDetails(offset))
        self._support_details = tuple(support_details)

    def _populate_signal_details(self) -> None:
        if self._signal_details is not None:
            raise ValueError('signal_details can not be initialized again')

        signal_details = []
        for i, entry in enumerate(self.meta.Data.Channels):
            offset = self.header.SIGNAL_BLOCK_BYTE_OFFSET + entry.SignalArrayByteOffset
            signal_details.append(ElementDetails(offset))
        self._signal_details = tuple(signal_details)

    @property
    def pvp_details(self) -> Optional[Tuple[ElementDetails, ...]]:
        return self._pvp_details

    @property
    def support_details(self) -> Optional[Tuple[ElementDetails, ...]]:
        return self._support_details

    @property
    def signal_details(self) -> Optional[Tuple[ElementDetails, ...]]:
        return self._signal_details

    @property
    def channel_map(self) -> Dict[str, int]:
        return self._channel_map

    @property
    def support_map(self) -> Optional[Dict[str, int]]:
        return self._support_map

    def _write_items(
            self,
            details: Optional[Sequence[ElementDetails]],
            file_object: BinaryIO) -> None:
        if details is None:
            return
        for index, entry in enumerate(details):
            entry.write_item(file_object)

    def _verify_item_written(
            self,
            details: Optional[Sequence[ElementDetails]],
            name: str) -> None:
        if details is None:
            return

        for index, entry in enumerate(details):
            if not entry.item_written:
                logger.error('{} data at index {} not written'.format(name, index))

    def write_header(
            self,
            file_object: BinaryIO,
            overwrite: bool = False) -> None:
        """
        Write the header.The file object will be advanced to the end of the
        block, if writing occurs.

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

        file_object.write(self.header.to_string().encode())
        file_object.write(CPHD_SECTION_TERMINATOR)
        # write xml
        file_object.seek(self.header.XML_BLOCK_BYTE_OFFSET, os.SEEK_SET)
        file_object.write(self.meta.to_xml_bytes(urn=get_namespace(self.use_version)))
        file_object.write(CPHD_SECTION_TERMINATOR)
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
        self._write_items(self.pvp_details, file_object)
        self._write_items(self.support_details, file_object)
        self._write_items(self.signal_details, file_object)

    def verify_all_written(self) -> None:
        if not self._header_written:
            logger.error('header not written')

        self._verify_item_written(self.pvp_details, 'pvp')
        self._verify_item_written(self.support_details, 'support')
        self._verify_item_written(self.signal_details, 'signal')


class CPHDWriter1(BaseWriter):
    """
    The CPHD version 1 writer.

    **Updated in version 1.3.0** for writing changes.
    """
    _writing_details_type = CPHDWritingDetails

    __slots__ = (
        '_file_name', '_file_object', '_in_memory', '_writing_details',
        '_pvp_memmaps', '_support_memmaps', '_signal_data_segments',
        '_can_write_regular_data')

    def __init__(
            self,
            file_object: Union[str, BinaryIO],
            meta: Optional[CPHDType1] = None,
            writing_details: Optional[CPHDWritingDetails] = None,
            check_older_version: bool = False,
            check_existence: bool = True):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
        meta : None|CPHDType1
        writing_details : None|CPHDWritingDetails
        check_older_version : bool
            Try to create an older version CPHD for compliance with other
            NGA applications
        check_existence : bool
            Should we check if the given file already exists, and raises an exception if so?
        """

        self._writing_details = None

        if isinstance(file_object, str):
            if check_existence and os.path.exists(file_object):
                raise SarpyIOError(
                    'Given file {} already exists, and a new CPHD file cannot be created here.'.format(file_object))
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

        if meta is None and writing_details is None:
            raise ValueError('One of meta or writing_details must be provided.')
        if writing_details is None:
            writing_details = self._writing_details_type(meta, check_older_version=check_older_version)
        self.writing_details = writing_details

        self._pvp_memmaps = None  # type: Optional[Dict[str, numpy.ndarray]]
        self._support_memmaps = None  # type: Optional[Dict[str, numpy.ndarray]]
        self._signal_data_segments = None  # type: Optional[Dict[str, DataSegment]]
        self._can_write_regular_data = None  # type: Optional[Dict[str, bool]]
        self._closed = False

        data_segment = self._initialize_data()
        BaseWriter.__init__(self, data_segment)

    @property
    def writing_details(self) -> CPHDWritingDetails:
        return self._writing_details

    @writing_details.setter
    def writing_details(self, value):
        if self._writing_details is not None:
            raise ValueError('writing_details is read-only')
        if not isinstance(value, CPHDWritingDetails):
            raise TypeError('writing_details must be of type {}'.format(CPHDWritingDetails))
        self._writing_details = value

    @property
    def file_name(self) -> Optional[str]:
        return self._file_name

    @property
    def meta(self) -> CPHDType1:
        """
        CPHDType1: The metadata
        """

        return self.writing_details.meta

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
            if index in self.writing_details.channel_map:
                return self.writing_details.channel_map[index]
            else:
                raise KeyError(_missing_channel_identifier_text.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < self.meta.Data.NumCPHDChannels):
                raise ValueError(_index_range_text.format(self.meta.Data.NumCPHDChannels))
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
            if index in self.writing_details.channel_map:
                return index
            else:
                raise KeyError(_missing_channel_identifier_text.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < self.meta.Data.NumCPHDChannels):
                raise ValueError(_index_range_text.format(self.meta.Data.NumCPHDChannels))
            return self.meta.Data.Channels[int_index].Identifier

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
            if index in self.writing_details.support_map:
                return self.writing_details.support_map[index]
            else:
                raise KeyError('Cannot find support array for identifier {}'.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < len(self.meta.Data.SupportArrays)):
                raise ValueError(_index_range_text.format(len(self.meta.Data.SupportArrays)))
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
            if index in self.writing_details.support_map:
                return index
            else:
                raise KeyError('Cannot find support array for identifier {}'.format(index))
        else:
            int_index = int(index)
            if not (0 <= int_index < len(self.meta.Data.SupportArrays)):
                raise ValueError(_index_range_text.format(len(self.meta.Data.SupportArrays)))
            return self.meta.Data.SupportArrays[int_index].Identifier

    def _initialize_data(self) -> List[DataSegment]:
        self._pvp_memmaps = {}
        # setup the PVP memmaps
        pvp_dtype = self.meta.PVP.get_vector_dtype()
        for i, entry in enumerate(self.meta.Data.Channels):
            # create the pvp mem map
            offset = self.writing_details.pvp_details[i].item_offset
            shape = (entry.NumVectors, )
            if self._in_memory:
                self._pvp_memmaps[entry.Identifier] = numpy.empty(shape, dtype=pvp_dtype)
            else:
                self._pvp_memmaps[entry.Identifier] = numpy.memmap(
                    self._file_name, dtype=pvp_dtype, mode='r+', offset=offset, shape=shape)

        self._support_memmaps = {}
        if self.meta.Data.SupportArrays is not None:
            for i, entry in enumerate(self.meta.Data.SupportArrays):
                # extract the support array metadata details
                details = self.meta.SupportArray.find_support_array(entry.Identifier)
                offset = self.writing_details.support_details[i].item_offset
                # determine numpy dtype and depth of array
                dtype, depth = details.get_numpy_format()
                # set up the numpy memory map
                shape = (entry.NumRows, entry.NumCols) if depth == 1 else (entry.NumRows, entry.NumCols, depth)
                if self._in_memory:
                    self._support_memmaps[entry.Identifier] = numpy.empty(shape, dtype=dtype)
                else:
                    self._support_memmaps[entry.Identifier] = numpy.memmap(
                        self._file_name, dtype=dtype, mode='r+', offset=offset, shape=shape)

        # setup the signal data_segment (this is used for formatting issues)
        no_amp_sf = (self.meta.PVP.AmpSF is None)
        self._signal_data_segments = {}
        self._can_write_regular_data = {}
        signal_data_segments = []
        signal_array_format = self.meta.Data.SignalArrayFormat
        if signal_array_format == 'CI2':
            signal_dtype = numpy.dtype('>i1')
        elif signal_array_format == 'CI4':
            signal_dtype = numpy.dtype('>i2')
        elif signal_array_format == 'CF8':
            signal_dtype = numpy.dtype('>f4')
        else:
            raise ValueError('Got unhandled SignalArrayFormat {}'.format(signal_array_format))
        for i, entry in enumerate(self.meta.Data.Channels):
            self._can_write_regular_data[entry.Identifier] = no_amp_sf
            raw_shape = (entry.NumVectors, entry.NumSamples, 2)
            format_function = AmpScalingFunction(signal_dtype)
            offset = self.writing_details.signal_details[i].item_offset
            if self._in_memory:
                underlying_array = numpy.full(raw_shape, 0, dtype=signal_dtype)
                data_segment = NumpyArraySegment(
                    underlying_array, 'complex64', formatted_shape=raw_shape[:2],
                    format_function=format_function, mode='w')
            else:
                data_segment = NumpyMemmapSegment(
                    self._file_object.name, offset, signal_dtype, raw_shape,
                    formatted_dtype='complex64', formatted_shape=raw_shape[:2],
                    format_function=format_function, mode='w', close_file=False)
            signal_data_segments.append(data_segment)

            self._signal_data_segments[entry.Identifier] = data_segment
        return signal_data_segments

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

        self._validate_closed()

        int_index = self._validate_support_index(identifier)
        identifier = self._validate_support_key(identifier)

        out_array = self._support_memmaps[identifier]
        if data.shape != out_array.shape:
            raise ValueError(
                'Support data shape {} is not compatible with\n\t'
                'that provided in metadata {}'.format(data.shape, out_array.shape))

        # write the data
        out_array[:] = data
        # mark it as written
        details = self.writing_details.support_details[int_index]
        if self._in_memory:
            # TODO: we can delete the memmap now?
            details.item_bytes = out_array.tobytes()
        else:
            details.item_written = True

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

        self._validate_closed()

        def validate_dtype():
            self._verify_dtype(data.dtype, self._pvp_memmaps[identifier].dtype, 'PVP channel {}'.format(identifier))

        int_index = self._validate_channel_index(identifier)
        identifier = self._validate_channel_key(identifier)
        entry = self.meta.Data.Channels[int_index]
        validate_dtype()

        if data.ndim != 1:
            raise ValueError('Provided data is required to be one dimensional')
        if data.shape[0] != entry.NumVectors:
            raise ValueError('Provided data must have size determined by NumVectors')

        if self.meta.PVP.AmpSF is not None:
            amp_sf = numpy.copy(data['AmpSF'][:])
            # noinspection PyUnresolvedReferences
            self._signal_data_segments[identifier].format_function.set_amplitude_scaling(amp_sf)
            self._can_write_regular_data[identifier] = True

        # write the data
        self._pvp_memmaps[identifier][:] = data
        # mark it as written
        details = self.writing_details.pvp_details[int_index]
        if self._in_memory:
            # TODO: we can likely delete the memmap now?
            details.item_bytes = self._pvp_memmaps[identifier].tobytes()
        else:
            details.item_written = True

    def write_support_block(self, support_block: Dict[Union[int, str], numpy.ndarray]) -> None:
        """
        Write support block to the file.

        Parameters
        ----------
        support_block: dict
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        expected_support_ids = {s.Identifier for s in self.meta.Data.SupportArrays}
        assert expected_support_ids == set(support_block), 'support_block keys do not match those in meta'
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

        expected_channels = {c.Identifier for c in self.meta.Data.Channels}
        assert expected_channels == set(pvp_block), 'pvp_block keys do not match those in meta'
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

        expected_channels = {c.Identifier for c in self.meta.Data.Channels}
        assert expected_channels == set(signal_block), 'signal_block keys do not match those in meta'
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

        expected_channels = {c.Identifier for c in self.meta.Data.Channels}
        assert expected_channels == set(signal_block), 'signal_block keys do not match those in meta'
        for identifier, array in signal_block.items():
            self.write_raw(array, index=identifier)

    def write_file(
            self,
            pvp_block: Dict[Union[int, str], numpy.ndarray],
            signal_block: Dict[Union[int, str], numpy.ndarray],
            support_block: Optional[Dict[Union[int, str], numpy.ndarray]] = None):
        """
        Write the blocks to the file.

        Parameters
        ----------
        pvp_block: Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the PVP arrays.
            Keys must be consistent with `self.meta`
        signal_block: Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the complex64 formatted signal
            arrays.
            Keys must be consistent with `self.meta`
        support_block: None|Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        self.write_pvp_block(pvp_block)
        if support_block:
            self.write_support_block(support_block)
        self.write_signal_block(signal_block)

    def write_file_raw(
            self,
            pvp_block: Dict[Union[int, str], numpy.ndarray],
            signal_block: Dict[Union[int, str], numpy.ndarray],
            support_block: Optional[Dict[Union[int, str], numpy.ndarray]] = None):
        """
        Write the blocks to the file.

        Parameters
        ----------
        pvp_block: Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the PVP arrays.
            Keys must be consistent with `self.meta`
        signal_block: Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the raw formatted
            (i.e. file storage format) signal arrays.
            Keys must be consistent with `self.meta`
        support_block: None|Dict[str, numpy.ndarray]
            Dictionary of `numpy.ndarray` containing the support arrays.
        """

        self.write_pvp_block(pvp_block)
        if support_block:
            self.write_support_block(support_block)
        self.write_signal_block_raw(signal_block)

    def write_chip(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Tuple[slice, ...]] = None,
            index: Union[int, str] = 0) -> None:
        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def write(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Tuple[slice, ...]] = None,
            index: Union[int, str] = 0) -> None:
        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Tuple[slice, ...]] = None,
            index: Union[int, str] = 0) -> None:
        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=True)

    def __call__(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Tuple[slice, ...]] = None,
            index: Union[int, str] = 0,
            raw: bool = False) -> None:
        int_index = self._validate_channel_index(index)

        identifier = self._validate_channel_key(index)
        if not raw and not self._can_write_regular_data[identifier]:
            raise ValueError(
                'The channel `{}` has an AmpSF which has not been determined,\n\t'
                'but the corresponding PVP block has not yet been written'.format(identifier))

        BaseWriter.__call__(self, data, start_indices=start_indices, subscript=subscript, index=int_index, raw=raw)

        # check if it's fully written
        # NB: this could be refactored out, but leaving it makes the most logical
        #   sense given the pvp/support approach
        fully_written = self.data_segment[int_index].check_fully_written(warn=False)
        if fully_written:
            self.writing_details.signal_details[int_index].item_written = True

    def flush(self, force: bool = False) -> None:
        self._validate_closed()

        BaseWriter.flush(self, force=force)

        try:
            if self._in_memory:
                if self.data_segment is not None:
                    for index, entry in enumerate(self.data_segment):
                        details = self.writing_details.signal_details[index]
                        if details.item_written:
                            continue
                        if details.item_bytes is not None:
                            continue
                        if force or entry.check_fully_written(warn=force):
                            details.item_bytes = entry.get_raw_bytes(warn=False)

            self.writing_details.write_all_populated_items(self._file_object)
        except AttributeError:
            return

    def close(self):
        if hasattr(self, '_closed') and self._closed:
            return

        BaseWriter.close(self)  # NB: flush called here
        try:
            if self.writing_details is not None:
                self.writing_details.verify_all_written()
        except AttributeError:
            pass
        self._writing_details = None
        self._file_object = None
