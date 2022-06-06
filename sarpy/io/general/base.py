"""
The basic definitions for file-like reading and writing. This is generally
centered on image-like file efforts, and array-like interaction with image data.

This module completely revamped in version 1.3.0 for data segment usage.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import os
import logging
from typing import Union, List, Tuple, Sequence, Optional, Callable
from importlib import import_module
import pkgutil

import numpy

from sarpy.compliance import SarpyError
from sarpy.io.general.format_function import FormatFunction
from sarpy.io.general.data_segment import DataSegment, extract_string_from_subscript, \
    NumpyArraySegment

logger = logging.getLogger(__name__)

READER_TYPES = ('SICD', 'SIDD', 'CPHD', 'CRSD', 'OTHER')
"""
The reader_type enum 
"""


class SarpyIOError(SarpyError):
    """A custom exception class for discovered input/output errors."""


############
# module walking to register openers

def check_for_openers(start_package: str, register_method: Callable) -> None:
    """
    Walks the package, and registers the discovered openers. That is, the modules
    with an :meth:`is_a` method.

    Parameters
    ----------
    start_package : str
    register_method : Callable
    """

    module = import_module(start_package)
    for details in pkgutil.walk_packages(module.__path__, start_package+'.'):
        _, module_name, is_pkg = details
        if is_pkg:
            # don't bother checking for packages
            continue
        sub_module = import_module(module_name)
        if hasattr(sub_module, 'is_a'):
            register_method(sub_module.is_a)


#############
# reader implementation for array like data

class BaseReader(object):
    """
    The basic reader definition, using array-like data fetching.

    **Changed in version 1.3.0**
    """

    __slots__ = (
        '_data_segment', '_reader_type', '_closed', '_close_segments',
        '_delete_temp_files')

    def __init__(
            self,
            data_segment: Union[None, DataSegment, Sequence[DataSegment]],
            reader_type: str = 'OTHER',
            close_segments: bool = True,
            delete_files: Optional[Union[str, Sequence[str]]] = None):
        """

        Parameters
        ----------
        data_segment : None|DataSegment|Sequence[DataSegment]
            None is a feasible value for extension tricks, ultimately the data_segments
            must be defined on initialization (by some extension).
        reader_type : str
        close_segments : bool
            Call segment.close() for each data segment on reader.close()?
        delete_files : None|str|Sequence[str]
            Any temp files which should be cleaned up on reader.close()?
            This will occur after closing segments.
        """

        # NB: it's entirely possible under multiple inheritance (class extends
        # two classes each of which extends BaseReader), that this initializer
        # has already been called. Don't override appropriate values, in that case.

        # override regardless here
        reader_type = reader_type.upper()
        if reader_type not in READER_TYPES:
            logger.info(
                'reader_type has value {}, while it is generally expected to be '
                'one of {}'.format(reader_type, READER_TYPES))
        self._reader_type = reader_type

        try:
            _ = self._closed
            # we didn't get an attribute error, so something has already defined
        except AttributeError:
            self._closed = False

        try:
            _ = self._close_segments
        except AttributeError:
            self._close_segments = close_segments

        try:
            _ = self._delete_temp_files
        except AttributeError:
            self._delete_temp_files = []  # type: List[str]

        if delete_files is None:
            pass
        elif isinstance(delete_files, str):
            if delete_files not in self._delete_temp_files:
                self._delete_temp_files.append(delete_files)
        else:
            for entry in delete_files:
                if entry not in self._delete_temp_files:
                    self._delete_temp_files.append(entry)

        try:
            _ = self._data_segment
            # we didn't get an attribute error, so something has already defined it
            self._set_data_segment(data_segment)
            # NB: this will raise a ValueError upon repeated definition attempts.
        except AttributeError:
            self._data_segment = None
            self._set_data_segment(data_segment)

    @property
    def file_name(self) -> Optional[str]:
        """
        None|str: Defined as a convenience property.
        """

        return None

    @property
    def reader_type(self) -> str:
        """
        str: A descriptive string for the type of reader
        """
        return self._reader_type

    @property
    def data_segment(self) -> Union[DataSegment, Tuple[DataSegment, ...]]:
        """
        DataSegment|Tuple[DataSegment, ...]: The data segment collection.
        """

        return self._data_segment

    def _set_data_segment(
            self,
            data_segment: Union[DataSegment, Sequence[DataSegment]]) -> None:
        """
        Sets the data segment collection. This can only be performed once.

        Parameters
        ----------
        data_segment : DataSegment|Sequence[DataSegment]

        Returns
        -------
        None
        """

        if data_segment is None:
            return  # do nothing

        if self._data_segment is not None:
            raise ValueError('data_segment is read only, once set.')

        if isinstance(data_segment, DataSegment):
            data_segment = [data_segment, ]
        if not isinstance(data_segment, Sequence):
            raise TypeError('data_segment must be an instance of DataSegment or a sequence of such instances')

        for entry in data_segment:
            if not isinstance(entry, DataSegment):
                raise TypeError(
                    'Requires all data segment entries to be an instance of DataSegment.\n\t'
                    'Got type {}'.format(type(entry)))
            if not entry.mode == 'r':
                raise ValueError('Each data segment must have mode="r"')

        if len(data_segment) == 1:
            self._data_segment = data_segment[0]
        else:
            self._data_segment = tuple(data_segment)

    @property
    def image_count(self) -> int:
        """
        int: The number of images/data segments from which to read.
        """

        if isinstance(self.data_segment, DataSegment):
            return 1
        else:
            return len(self.data_segment)

    def get_data_segment_as_tuple(self) -> Tuple[DataSegment, ...]:
        """
        Get the data segment collection as a tuple, to avoid the need for redundant
        checking issues.

        Returns
        -------
        Tuple[DataSegment, ...]
        """

        return (self.data_segment, ) if self.image_count == 1 else self._data_segment

    @property
    def data_size(self) -> Union[Tuple[int, ...], Tuple[Tuple[int, ...]]]:
        """
        Tuple[int, ...]|Tuple[Tuple[int, ...], ...]: the output/formatted data
        size(s) of the data segment(s). If there is a single data segment, then
        this will be `Tuple[int, ...]`, otherwise it will be
        `Tuple[Tuple, int, ...], ...]`.
        """

        return self.data_segment.formatted_shape if self.image_count == 1 else \
            tuple(entry.formatted_shape for entry in self.data_segment)

    def get_data_size_as_tuple(self) -> Tuple[Tuple[int, ...], ...]:
        """
        Get the data size collection as a tuple of tuples, to avoid the need
        for redundant checking issues.

        Returns
        -------
        Tuple[Tuple[int, ...], ...]
        """

        return (self.data_size, ) if self.image_count == 1 else self.data_size

    @property
    def raw_data_size(self) -> Union[Tuple[int, ...], Tuple[Tuple[int, ...]]]:
        """
        Tuple[int, ...]|Tuple[Tuple[int, ...], ...]: the raw data size(s) of the
        data segment(s). If there is a single data segment, then this will be
        `Tuple[int, ...]`, otherwise it will be `Tuple[Tuple, int, ...], ...]`.
        """

        return self.data_segment.raw_shape if self.image_count == 1 else \
            tuple(entry.raw_shape for entry in self.data_segment)

    def get_raw_data_size_as_tuple(self) -> Tuple[Tuple[int, ...], ...]:
        """
        Get the raw data size collection as a tuple of tuples, to avoid the need
        for redundant checking issues.

        Returns
        -------
        Tuple[Tuple[int, ...], ...]
        """

        return (self.data_size, ) if self.image_count == 1 else self.data_size

    @property
    def files_to_delete_on_close(self) -> List[str]:
        """
        List[str]: A collection of files to delete on the close operation.
        """

        return self._delete_temp_files

    @property
    def closed(self) -> bool:
        """
        bool: Is the reader closed? Reading will result in a ValueError
        """

        return self._closed

    def _validate_closed(self):
        if not hasattr(self, '_closed') or self._closed:
            raise ValueError('I/O operation of closed writer')

    def read_chip(
            self,
            *ranges: Sequence[Union[None, int, Tuple[int, ...], slice]],
            index: int = 0,
            squeeze: bool = True) -> numpy.ndarray:
        """
        This is identical to :meth:`read`, and presented for backwards compatibility.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, Tuple[int, ...], slice]]
        index : int
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
            *ranges: Union[None, int, Tuple[int, ...], slice],
            index: int = 0,
            squeeze: bool = True) -> numpy.ndarray:
        """
        Read formatted data from the given data segment. Note this is an alias to the
        :meth:`__call__` called as
        :code:`reader(*ranges, index=index, raw=False, squeeze=squeeze)`.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, Tuple[int, ...], slice]]
            The slice definition appropriate for `data_segment[index].read()` usage.
        index : int
            The data_segment index. This is ignored if `image_count== 1`.
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
            *ranges: Union[None, int, Tuple[int, ...], slice],
            index: int = 0,
            squeeze: bool = True) -> numpy.ndarray:
        """
        Read raw data from the given data segment. Note this is an alias to the
        :meth:`__call__` called as
        :code:`reader(*ranges, index=index, raw=True, squeeze=squeeze)`.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, Tuple[int, ...], slice]]
            The slice definition appropriate for `data_segment[index].read()` usage.
        index : int
            The data_segment index. This is ignored if `image_count== 1`.
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
            *ranges: Union[None, int, Tuple[int, ...], slice],
            index: int = 0,
            raw: bool = False,
            squeeze: bool = True) -> numpy.ndarray:

        self._validate_closed()

        if len(ranges) == 0:
            subscript = None
        else:
            subscript = []
            for rng in ranges:
                if rng is None:
                    subscript.append(slice(None, None, 1))
                elif isinstance(rng, int):
                    subscript.append(slice(rng))
                elif isinstance(rng, tuple):
                    subscript.append(slice(*rng))
                elif isinstance(rng, slice) or rng is Ellipsis:
                    subscript.append(rng)
                else:
                    raise TypeError('Got unexpected type `{}` value for range/slice'.format(type(rng)))
        if isinstance(self._data_segment, tuple):
            ds = self.data_segment[index]
        else:
            ds = self.data_segment

        if raw:
            return ds.read_raw(subscript, squeeze=squeeze)
        else:
            return ds.read(subscript, squeeze=squeeze)

    def __getitem__(self, subscript) -> numpy.ndarray:
        # TODO: document the str usage and index determination

        subscript, string_entries = extract_string_from_subscript(subscript)
        if not isinstance(subscript, (tuple, list)):
            subscript = (subscript, )

        raw = ('raw' in string_entries)
        squeeze = ('nosqueeze' not in string_entries)

        if isinstance(subscript[-1], int):
            the_index = subscript[-1]
            if -self.image_count < the_index < self.image_count:
                return self.__call__(*subscript[:-1], index=subscript[-1], raw=raw, squeeze=squeeze)
        return self.__call__(*subscript, index=0, raw=raw, squeeze=squeeze)

    def close(self) -> None:
        """
        This should perform any necessary clean-up operations, like closing
        open file handles, deleting any temp files, etc.
        """

        if not hasattr(self, '_closed') or self._closed:
            return
        # close all the segments
        if self._close_segments and self._data_segment is not None:
            if isinstance(self._data_segment, DataSegment):
                self._data_segment.close()
            else:
                for entry in self._data_segment:
                    entry.close()
        self._data_segment = None

        # delete temp files
        while len(self._delete_temp_files) > 0:
            filename = self._delete_temp_files.pop()
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(
                    'Reader attempted to delete temp file {},\n\t'
                    'but got error {}'.format(filename, e))
        self._closed = True

    def __del__(self):
        # NB: this is called when the object is marked for garbage collection
        # (i.e. reference count goes to 0), and the order in which this happens
        # may be unreliable
        self.close()


class FlatReader(BaseReader):
    """
    Class for passing a numpy array straight through as a reader.

    Changed in version 1.3.0
    """

    def __init__(
            self,
            underlying_array: numpy.ndarray,
            reader_type: str = 'OTHER',
            formatted_dtype: Optional[Union[str, numpy.dtype]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Union[int, Sequence[int]]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            close_segments: bool = True):
        """

        Parameters
        ----------
        underlying_array : numpy.ndarray
        reader_type : str
        formatted_dtype : None|str|numpy.dtype
        formatted_shape : None|Tuple[int, ...]
        reverse_axes : None|Sequence[int]
        transpose_axes : None|Tuple[int, ...]
        format_function : None|FormatFunction
        close_segments : bool
        """

        data_segment = NumpyArraySegment(
            underlying_array, formatted_dtype=formatted_dtype, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode='r')
        BaseReader.__init__(
            self, data_segment, reader_type=reader_type, close_segments=close_segments)


class AggregateReader(BaseReader):
    """
    Aggregate multiple readers into a single reader instance. This default
    aggregate implementation will not preserve any other metadata structures.
    """

    __slots__ = ('_readers', '_index_mapping', '_close_readers')

    def __init__(
            self,
            readers: Sequence[BaseReader],
            reader_type: str = "OTHER",
            close_readers: bool = False):
        """

        Parameters
        ----------
        readers : Sequence[BaseReader]
            The readers.
        reader_type : str
            The reader type string.
        close_readers: bool
            Close all readers on this reader close?
        """

        self._close_readers = close_readers
        self._index_mapping = None
        self._readers = self._validate_readers(readers)
        data_segments = self._define_index_mapping()
        # NB: close_segments is and should be handled by the constituent readers
        BaseReader.__init__(
            self, data_segment=data_segments, reader_type=reader_type, close_segments=False)

    @staticmethod
    def _validate_readers(readers: Sequence[BaseReader]) -> Tuple[BaseReader]:
        """
        Validate the input reader/file collection.

        Parameters
        ----------
        readers : Sequence[BaseReader]

        Returns
        -------
        Tuple[BaseReader]
        """

        if not isinstance(readers, Sequence):
            raise TypeError('input argument must be a sequence of readers. Got type {}'.format(type(readers)))

        # validate each entry
        the_readers = []
        for i, entry in enumerate(readers):
            if not isinstance(entry, BaseReader):
                raise TypeError(
                    'All elements of the input argument must be reader instances. '
                    'Entry {} is of type {}'.format(i, type(entry)))
            the_readers.append(entry)
        return tuple(the_readers)

    def _define_index_mapping(self) -> List[DataSegment]:
        """
        Define the index mapping.

        Returns
        -------
        List[DataSegment]
        """

        # prepare the index mapping workspace
        index_mapping = []

        segments = []
        for i, reader in enumerate(self._readers):
            for j, segment in enumerate(reader.get_data_segment_as_tuple()):
                segments.append(segment)
                index_mapping.append((i, j))
        self._index_mapping = tuple(index_mapping)
        return segments

    @property
    def index_mapping(self) -> Tuple[Tuple[int, int]]:
        """
        Tuple[Tuple[int, int]]: The index mapping of the form (reader index, segment index in reader).
        """

        return self._index_mapping

    def close(self) -> None:
        """
        This should perform any necessary clean-up operations, like closing
        open file handles, deleting any temp files, etc.
        """

        if not hasattr(self, '_closed') or self._closed:
            return

        BaseReader.close(self)
        if self._close_readers and self._readers is not None:
            for entry in self._readers:
                entry.close()
        self._readers = None


#############
# writer implementation for array like data

class BaseWriter(object):
    """
    Writer definition, using array-like data writing.

    Introduced in version 1.3.0
    """

    __slots__ = ('_data_segment', '_closed')

    def __init__(
            self,
            data_segment: Union[DataSegment, Sequence[DataSegment]]):

        self._closed = False
        if isinstance(data_segment, DataSegment):
            data_segment = [data_segment, ]
        if not isinstance(data_segment, Sequence):
            raise TypeError('data_segment must be an instance of DataSegment or a sequence of such instances')

        for entry in data_segment:
            if not isinstance(entry, DataSegment):
                raise TypeError(
                    'Requires all data segment entries to be an instance of DataSegment.\n\t'
                    'Got type {}'.format(type(entry)))
            if not entry.mode == 'w':
                raise ValueError('Each data segment must have mode="w" for writing')

        self._data_segment = tuple(data_segment)

    @property
    def file_name(self) -> Optional[str]:
        """
        None|str: Defined as a convenience property.
        """

        return None

    @property
    def data_segment(self) -> Tuple[DataSegment, ...]:
        """
        Tuple[DataSegment, ...]: The data segment collection.
        """

        return self._data_segment

    @property
    def image_count(self) -> int:
        """
        int: The number of overall images/data segments.
        """

        return len(self.data_segment)

    @property
    def data_size(self) -> Tuple[Tuple[int, ...]]:
        """
        Tuple[Tuple[int, ...], ...]: the formatted data sizes of the data
        segments.
        """

        return tuple(entry.formatted_shape for entry in self.data_segment)

    @property
    def raw_data_size(self) -> Union[Tuple[int, ...], Tuple[Tuple[int, ...]]]:
        """
        Tuple[Tuple[int, ...], ...]: the raw data sizes of the data segments.
        """

        return tuple(entry.raw_shape for entry in self.data_segment)

    @property
    def closed(self) -> bool:
        """
        bool: Is the reader closed? Reading will result in a ValueError
        """

        return self._closed

    def _validate_closed(self):
        if not hasattr(self, '_closed') or self._closed:
            raise ValueError('I/O operation of closed writer')

    def write_chip(
            self,
            data: numpy.ndarray,
            start_indices: Optional[Union[int, Tuple[int, ...]]] = None,
            subscript: Optional[Tuple[slice, ...]] = None,
            index: int = 0) -> None:
        """
        This is identical to :meth:`write`, and presented for backwards compatibility.

        Parameters
        ----------
        data : numpy.ndarray
        start_indices : None|int|Tuple[int, ...]
        subscript : None|Tuple[slice, ...]
        index : int

        See Also
        --------
        See :meth:`sarpy.io.general.data_segment.DataSegment.write`.
        """

        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def write(
            self,
            data: numpy.ndarray,
            start_indices: Optional[Union[int, Tuple[int, ...]]] = None,
            subscript: Optional[Tuple[slice, ...]] = None,
            index: int = 0) -> None:
        """
        Write the data to the appropriate data segment. This is an alias to
        :code:`writer(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)`.

        **Only one of `start_indices` and `subscript` should be specified.**

        Parameters
        ----------
        data : numpy.ndarray
            The data to write.
        start_indices : None|int|Tuple[int, ...]
            Assuming a contiguous chunk of data, this provides the starting
            indices of the chunk. Any missing (tail) coordinates will be filled
            in with 0's.
        subscript : None|Tuple[slice, ...]
            In contrast to providing `start_indices`, the slicing definition in
            formatted coordinates pertinent to the specified data segment.
        index : int
            The index of the

        See Also
        --------
        See :meth:`sarpy.io.general.data_segment.DataSegment.write`.
        """

        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Optional[Union[int, Tuple[int, ...]]] = None,
            subscript: Optional[Tuple[slice, ...]] = None,
            index: int = 0) -> None:
        """
        Write the raw data to the file(s). This is an alias to
        :code:`writer(data, start_indices=start_indices, subscript=subscript, index=index, raw=True)`.

        **Only one of `start_indices` and `subscript` should be specified.**

        Parameters
        ----------
        data : numpy.ndarray
            The data to write.
        start_indices : None|int|Tuple[int, ...]
            Assuming a contiguous chunk of data, this provides the starting
            indices of the chunk. Any missing (tail) coordinates will be filled
            in with 0's.
        subscript : None|Tuple[slice, ...]
            In contrast to providing `start_indices`, the slicing definition in
            raw coordinates pertinent to the specified data segment.
        index : int

        See Also
        --------
        See :meth:`sarpy.io.general.data_segment.DataSegment.write_raw`.
        """

        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=True)

    def __call__(
            self,
            data: numpy.ndarray,
            start_indices: Optional[Union[int, Tuple[int, ...]]] = None,
            subscript: Optional[Tuple[slice, ...]] = None,
            index: int = 0,
            raw: bool = False) -> None:
        """
        Write the data to the given data segment.

        Parameters
        ----------
        data : numpy.ndarray
            The data to write.
        start_indices : None|int|Tuple[int, ...]
            Assuming a contiguous chunk of data, this provides the starting
            indices of the chunk. Any missing (tail) coordinates will be filled
            in with 0's.
        subscript : None|Tuple[slice, ...]
            In contrast to providing `start_indices`, the slicing definition in
            coordinates pertinent to the specified data segment and `raw` value.
        index : int
        raw : bool
        """

        self._validate_closed()
        ds = self.data_segment[index]

        if raw:
            return ds.write_raw(data, start_indices=start_indices, subscript=subscript)
        else:
            if not ds.can_write_regular:
                raise ValueError(
                    'The data segment at index {} can not convert from formatted data to raw data.\n\t'
                    'It is only permitted to use the write_raw() function on this data set,\n\t'
                    'and to write the data in raw (i.e. unformatted) form.')
            return ds.write(data, start_indices=start_indices, subscript=subscript)

    def flush(self, force: bool=False) -> None:
        """
        Try to perform any necessary steps to flush written data to the disk/buffer.

        Parameters
        ----------
        force : bool
            Try force flushing, even for incompletely written data.

        Returns
        -------
        None
        """

        self._validate_closed()
        if self._data_segment is not None:
            for data_segment in self.data_segment:
                data_segment.flush()

    def close(self) -> None:
        """
        Completes any necessary final steps.
        """

        if not hasattr(self, '_closed') or self._closed:
            return

        try:
            # flush the data
            self.flush(force=True)
            # close all the segments
            if self._data_segment is not None:
                for entry in self._data_segment:
                    entry.close()
            self._data_segment = None
            self._closed = True
        except AttributeError:
            self._closed = True
            return

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

        if exception_type is not None:
            logger.error(
                'The {} file writer generated an exception during processing'.format(
                    self.__class__.__name__))
            # The exception will be reraised.
            # It's unclear how any exception could really be caught.
