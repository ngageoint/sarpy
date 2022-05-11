"""
The basic definitions for file-like reading and writing. This is generally
centered on image-like file efforts, and array-like interaction with image data.

This module completely updated in version 1.3.0.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

# TODO: handle some other kind of writing here?
#   We should be able to write a file in memory...
#   I'll work through the NITF situation, and then decide


import os
import logging
from typing import Union, List, Tuple, Sequence, Optional
from importlib import import_module
import pkgutil

import numpy

from sarpy.compliance import SarpyError
from sarpy.io.general.format_function import FormatFunction
from sarpy.io.general.data_segment import DataSegment, NumpyArraySegment

logger = logging.getLogger(__name__)

READER_TYPES = ('SICD', 'SIDD', 'CPHD', 'CRSD', 'OTHER')
"""
The reader_type enum 
"""


class SarpyIOError(SarpyError):
    """A custom exception class for discovered input/output errors."""


############
# module walking to register openers

def check_for_openers(start_package, register_method):
    """
    Walks the package, and registers the discovered openers.

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

class AbstractReader(object):
    """
    The basic reader definition, using array-like data fetching.

    **Changed in version 1.3.0**
    """

    __slots__ = (
        '_data_segment', '_reader_type', '_closed', '_close_segments',
        '_delete_temp_files')

    def __init__(self,
                 data_segment: Union[DataSegment, Sequence[DataSegment]],
                 reader_type: str='OTHER',
                 close_segments: bool=True,
                 delete_files: Union[None, str, Sequence[str]]=None):
        """

        Parameters
        ----------
        data_segment : DataSegment|Sequence[DataSegment]
        reader_type : str
        close_segments : bool
            Call segment.close() for each data segment on reader.close()?
        delete_files : None|Sequence[str]
            Any temp files which should be cleaned up on reader.close()?
            This will occur after closing segments.
        """

        self._closed = False
        self._close_segments = close_segments

        self._delete_temp_files = []  # type: List[str]
        if delete_files is None:
            pass
        elif isinstance(delete_files, str):
            self._delete_temp_files.append(delete_files)
        else:
            for entry in delete_files:
                self._delete_temp_files.append(entry)

        reader_type = reader_type.upper()
        if reader_type not in READER_TYPES:
            logger.error(
                'reader_type has value {}, while it is expected to be '
                'one of {}'.format(reader_type, READER_TYPES))
        self._reader_type = reader_type

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

    @property
    def image_count(self) -> int:
        """
        int: The number of images/data segments from which to read.
        """

        if isinstance(self.data_segment, DataSegment):
            return 1
        else:
            return len(self.data_segment)

    def get_data_segment_as_tuple(self):
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

        return (self.data_size,) if self.image_count == 1 else self.data_size

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

        return (self.data_size,) if self.image_count == 1 else self.data_size

    @property
    def files_to_delete_on_close(self) -> List[str]:
        """
        List[str]: A collection of files to delete on the close operation.
        """

        return self._delete_temp_files

    def read_chip(self,
             *ranges: Sequence[Union[None, int, slice, Ellipsis]],
             index: int=0,
             squeeze: bool=True) -> numpy.ndarray:
        """
        This is identical to :meth:`read`, and presented for backwards compatibility.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, slice, Ellipsis]]
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

    def read(self,
             *ranges: Sequence[Union[None, int, slice, Ellipsis]],
             index: int=0,
             squeeze: bool=True) -> numpy.ndarray:
        """
        Read formatted data from the given data segment. Note this is an alias to the
        :meth:`__call__` called as
        :code:`reader(*ranges, index=index, raw=False, squeeze=squeeze)`.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, slice, Ellipsis]]
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

    def read_raw(self,
                 *ranges: Sequence[Union[None, int, slice, Ellipsis]],
                 index: int=0,
                 squeeze: bool=True) -> numpy.ndarray:
        """
        Read raw data from the given data segment. Note this is an alias to the
        :meth:`__call__` called as
        :code:`reader(*ranges, index=index, raw=True, squeeze=squeeze)`.

        Parameters
        ----------
        ranges : Sequence[Union[None, int, slice, Ellipsis]]
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

    def __call__(self,
                 *ranges: Sequence[Union[None, int, slice, Ellipsis]],
                 index: int=0,
                 raw: bool=False,
                 squeeze: bool=True) -> numpy.ndarray:
        if len(ranges) == 0:
            subscript = None
        else:
            subscript = []
            for rng in ranges:
                if rng is None:
                    subscript.append(slice(None, None, None))
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
        if self.image_count == 1:
            return self.__call__(subscript, index=0, raw=False, squeeze=True)

        if isinstance(subscript, tuple):
            if isinstance(subscript[-1], int):
                return self.__call__(subscript[:-1], index=subscript[-1], raw=False, squeeze=True)
        return self.__call__(subscript, index=0, raw=False, squeeze=True)

    def close(self):
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


class FlatReader(AbstractReader):
    """
    Class for passing a numpy array straight through as a reader.

    **Changed in version 1.3.0**
    """

    def __init__(self,
                 underlying_array: numpy.ndarray,
                 reader_type: str='OTHER',
                 formatted_dtype: Union[None, str, numpy.dtype] = None,
                 formatted_shape: Union[None, Tuple[int, ...]] = None,
                 reverse_axes: Union[None, int, Sequence[int]] = None,
                 transpose_axes: Union[None, Tuple[int, ...]] = None,
                 format_function: Union[None, FormatFunction] = None):
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
        """

        data_segment = NumpyArraySegment(
            underlying_array, formatted_dtype=formatted_dtype, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode='r')
        AbstractReader.__init__(self, data_segment, reader_type=reader_type, close_segments=False)


class AggregateReader(AbstractReader):
    """
    Aggregate multiple readers into a single reader instance. This default
    aggregate implementation will not preserve any other metadata structures.
    """

    __slots__ = ('_readers', '_index_mapping', '_close_readers')

    def __init__(self,
                 readers: Sequence[AbstractReader],
                 reader_type: str="OTHER",
                 close_readers: bool=False):
        """

        Parameters
        ----------
        readers : Sequence[AbstractReader]
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
        super(AggregateReader, self).__init__(data_segment=data_segments, reader_type=reader_type, close_segments=False)

    @staticmethod
    def _validate_readers(readers: Sequence[AbstractReader]):
        """
        Validate the input reader/file collection.

        Parameters
        ----------
        readers : Sequence[AbstractReader]

        Returns
        -------
        Tuple[AbstractReader]
        """

        if not isinstance(readers, Sequence):
            raise TypeError('input argument must be a sequence of readers. Got type {}'.format(type(readers)))

        # validate each entry
        the_readers = []
        for i, entry in enumerate(readers):
            if not isinstance(entry, AbstractReader):
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
        # assemble the chipper arguments
        segments = []
        for i, reader in enumerate(self._readers):
            for j, segment in enumerate(reader.get_data_segment_as_tuple()):
                segment.append(segment)
                index_mapping.append((i, j))
        self._index_mapping = tuple(index_mapping)
        return segments

    @property
    def index_mapping(self):
        # type: () -> Tuple[Tuple[int, int]]
        """
        Tuple[Tuple[int, int]]: The index mapping of the form (reader index, segment index in reader).
        """

        return self._index_mapping

    def close(self):
        """
        This should perform any necessary clean-up operations, like closing
        open file handles, deleting any temp files, etc.
        """

        if not hasattr(self, '_closed') or self._closed:
            return

        AbstractReader.close(self)
        if self._close_readers and self._readers is not None:
            for entry in self._readers:
                entry.close()
        self._readers = None


#############
# writer implementation for array like data

class AbstractWriter(object):
    """
    Abstract writer definition, using array-like data writing.

    **Changed in version 1.3.0**
    """

    __slots__ = ('_data_segment', '_closed')

    def __init__(self, data_segment: Union[DataSegment, Sequence[DataSegment]]):

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
                raise ValueError('Each data segment must have mode="r"')

        self._data_segment = tuple(data_segment)
        self._closed = False

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

    def write_chip(self,
              data: numpy.ndarray,
              start_indices: Union[None, int, Tuple[int, ...]] = None,
              subscript: Union[None, Tuple[slice, ...]] = None,
              index=0) -> None:
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

    def write(self,
              data: numpy.ndarray,
              start_indices: Union[None, int, Tuple[int, ...]]=None,
              subscript: Union[None, Tuple[slice, ...]]=None,
              index=0) -> None:
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
            The subscript definition in formatted coordinates pertinent to the
            specified data segment.
        index : int
            The index of the

        See Also
        --------
        See :meth:`sarpy.io.general.data_segment.DataSegment.write`.
        """

        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def write_raw(self,
              data: numpy.ndarray,
              start_indices: Union[None, int, Tuple[int, ...]]=None,
              subscript: Union[None, Tuple[slice, ...]]=None,
              index=0) -> None:
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
        index : int

        See Also
        --------
        See :meth:`sarpy.io.general.data_segment.DataSegment.write_raw`.
        """

        self.__call__(data, start_indices=start_indices, subscript=subscript, index=index, raw=False)

    def __call__(self,
                 data: numpy.ndarray,
                 start_indices: Union[None, int, Tuple[int, ...]]=None,
                 subscript: Union[None, Tuple[slice, ...]]=None,
                 index: int=0,
                 raw: bool=False) -> None:
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
        index : int
        raw : bool
        """

        ds = self.data_segment[index]

        if raw:
            return ds.write_raw(data, start_indices=start_indices, subscript=subscript)
        else:
            if ds.can_write_regular:
                raise ValueError(
                    'The data segment at index {} can not convert from formatted data to raw data.\n\t'
                    'It is only permitted to use the write_raw() function on this data set,\n\t'
                    'and to write the data in raw (i.e. unformatted) form.')
            return ds.write(data, start_indices=start_indices, subscript=subscript)

    def close(self):
        """
        Completes any necessary final steps.
        """

        if not hasattr(self, '_closed') or self._closed:
            return

        # close all the segments
        if self._data_segment is not None:
            for entry in self._data_segment:
                entry.close()
        self._data_segment = None
        self._closed = True

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
