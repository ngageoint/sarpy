"""
The object definitions for reading and writing data in single conceptual units
using an interface based on slicing definitions and numpy arrays with formatting
operations.

This module introduced in version 1.3.0.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
import os
from typing import Union, Tuple, Sequence, BinaryIO, Optional

import numpy

from sarpy.io.general.format_function import FormatFunction, IdentityFunction
from sarpy.io.general.slice_parsing import verify_subscript, get_slice_result_size, \
    get_subscript_result_size
from sarpy.io.general.utils import h5py, is_file_like

if h5py is not None:
    from h5py import File as h5pyFile, Dataset as h5pyDataset
else:
    h5pyFile = None
    h5pyDataset = None

logger = logging.getLogger(__name__)


####
# helper functions

def _reverse_slice(slice_in: slice) -> slice:
    """
    Given a slice with negative step, this returns a slice which will define the
    same elements traversed in the opposite direction. Note that this is not
    the same as the mirror operation.

    Parameters
    ----------
    slice_in : slice

    Returns
    -------
    slice

    Raises
    ------
    ValueError
    """

    if slice_in.step > 0:
        raise ValueError('This is only applicable to slices with negative step value')
    stop = 0 if slice_in.stop is None else slice_in.stop + 1
    mult = int(numpy.floor((stop - slice_in.start)/slice_in.step))
    final_entry = slice_in.start + mult*slice_in.step
    return slice(final_entry, slice_in.start-slice_in.step, -slice_in.step)


def _find_slice_overlap(
        slice_in: slice,
        ref_slice: slice) -> Tuple[Optional[slice], Optional[slice]]:
    """
    Finds the overlap of the slice with a contiguous interval slice.

    Parameters
    ----------
    slice_in : slice
    ref_slice : slice

    Returns
    -------
    child_slice : None|slice
        The overlap expressed as a slice relative to the sliced coordinates.
        None if there is no overlap.
    parent_slice : None|slice
        The overlap expressed as a slice relative to the overall indices.
        None if there is no overlap.
    """

    if ref_slice.step not in [1, -1]:
        raise ValueError('Reference slice must have step +/-1')

    if ref_slice.step > 0:
        start_ind = ref_slice.start
        stop_ind = ref_slice.stop
    else:
        start_ind = 0 if ref_slice.stop is None else ref_slice.stop + 1
        stop_ind = ref_slice.start + 1

    if slice_in.step > 0:
        if slice_in.stop <= start_ind or slice_in.start >= stop_ind:
            # there is no overlap

            return None, None
        # find minimum multiplier so that slice_in.start + mult*slice_in.step >= start_ind
        #    mult >= (start_ind - slice_in.start)/slice_in.step
        child_start = 0 if start_ind <= slice_in.start else \
            int(numpy.ceil((start_ind - slice_in.start)/slice_in.step))
        parent_start = slice_in.start - start_ind + child_start*slice_in.step
        # find end - maximum multiplier so that slice_in.start + mult*slice_in.step <= stop_ind
        #    mult <= (stop_ind - slice_in.start)/slice_in.step
        max_ind = min(slice_in.stop, stop_ind)
        child_stop = int(numpy.floor((max_ind - slice_in.start)/slice_in.step))
        parent_stop = slice_in.start - start_ind + child_stop*slice_in.step
    else:
        if slice_in.start < start_ind or (slice_in.stop is not None and slice_in.stop >= stop_ind):
            # there is no overlap

            # noinspection PyTypeChecker
            return None, None

        # find minimum multiplier so that slice_in.start + mult*slice_in.step <= stop_ind-1
        #    mult >= (stop_ind - 1 - slice_in.start)/slice_in.step
        child_start = 0 if slice_in.start < stop_ind else int(numpy.ceil((stop_ind - 1 - slice_in.start)/slice_in.step))
        parent_start = slice_in.start - start_ind + child_start*slice_in.step
        # find end - first multiplier so that slice_in.start + mult*slice_in.step < start_ind - 1
        #    mult > (start_ind - slice_in.start)/slice_in.step
        if slice_in.stop is None:
            min_ind = max(start_ind-1, -1)
        else:
            min_ind = max(start_ind-1, slice_in.stop)
        child_stop = int(numpy.ceil((min_ind - slice_in.start)/slice_in.step))
        parent_stop = slice_in.start - start_ind + child_stop*slice_in.step
        if parent_stop < 0:
            parent_stop = None

    if ref_slice.step < 0:
        # noinspection PyTypeChecker
        return _reverse_slice(slice(parent_start, parent_stop, slice_in.step)), \
               _reverse_slice(slice(child_start, child_stop, 1))
    else:
        # noinspection PyTypeChecker
        return slice(parent_start, parent_stop, slice_in.step), \
               slice(child_start, child_stop, 1)


def _infer_subscript_for_write(
        data: numpy.ndarray,
        start_indices: Union[None, int, Tuple[int, ...]],
        subscript: Union[None, Sequence[slice]],
        full_shape: Tuple[int, ...]) -> Tuple[slice, ...]:
    """
    Helper function, for writing operation, which infers the subscript definition
    between the given start_indices or (possibly partially defined) subscript.

    Parameters
    ----------
    data : numpy.ndarray
    start_indices : None|int|Tuple[int, ...]
    subscript : None|Sequence[slice]
    full_shape : Tuple[int, ...]

    Returns
    -------
    Tuple[slice, ...]
    """

    if start_indices is None and subscript is None:
        if data.shape == full_shape:
            return verify_subscript(None, full_shape)
        else:
            raise ValueError('One of start_indices or subscript must be provided.')

    if start_indices is not None:
        if isinstance(start_indices, int):
            start_indices = (start_indices, )

        if len(start_indices) < len(full_shape):
            start_indices = start_indices + tuple(0 for _ in range(len(start_indices), len(full_shape)))
        subscript = tuple([slice(entry1, entry1+entry2, 1) for entry1, entry2 in zip(start_indices, data.shape)])
    subscript, result_shape = get_subscript_result_size(subscript, full_shape)
    if result_shape != data.shape:
        raise ValueError(
            'Inferred subscript `{}` with shape `{}`\n\t'
            'does not match data.shape `{}`'.format(subscript, result_shape, data.shape))
    return subscript


def extract_string_from_subscript(
        subscript: Union[None, int, slice, Tuple]) -> Tuple[Union[None, int, slice, Sequence], Tuple[str, ...]]:
    """
    Extracts any string elements (stripped and made all lowercase) from subscript entries.

    Parameters
    ----------
    subscript : None|str|int|slice|Sequence

    Returns
    -------
    subscript: None|int|slice|Sequence
        With string entries removed
    strings : Tuple[str, ...]
        The string entries, stripped and made all lower case.
    """

    string_entries = []
    if isinstance(subscript, str):
        string_entries.append(subscript.strip().lower())
        subscript = None
    elif isinstance(subscript, Sequence):
        new_subscript = []
        for entry in subscript:
            if isinstance(entry, str):
                string_entries.append(entry.strip().lower())
            else:
                new_subscript.append(entry)
        if len(string_entries) > 0:
            subscript = tuple(new_subscript)
    return subscript, tuple(string_entries)


#####
# Abstract data segment definition and derived element implementations

class DataSegment(object):
    """
    Partially abstract base class representing one conceptual fragment of data
    read or written as an array. This is generally designed for images, but is
    general enough to support other usage.

    Introduced in version 1.3.0.

    .. warning::
        The format function instance will be modified in place. Do not use the
        same format function instance across multiple data segments.
    """

    _allowed_modes = ('r', 'w')

    __slots__ = (
        '_closed', '_mode',
        '_raw_dtype', '_raw_shape', '_formatted_dtype', '_formatted_shape',
        '_reverse_axes', '_transpose_axes', '_reverse_transpose_axes',
        '_format_function')

    def __init__(
            self,
            raw_dtype: Union[str, numpy.dtype],
            raw_shape: Tuple[int, ...],
            formatted_dtype: Union[str, numpy.dtype],
            formatted_shape: Tuple[int, ...],
            reverse_axes: Union[None, int, Sequence[int]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            mode: str = 'r'):
        """

        Parameters
        ----------
        raw_dtype : str|numpy.dtype
        raw_shape : Tuple[int, ...]
        formatted_dtype : str|numpy.dtype
        formatted_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
            Note that the format function instance
        mode : str
        """

        self._closed = False
        self._mode = None
        self._set_mode(mode)

        self._raw_shape = None
        self._set_raw_shape(raw_shape)

        self._raw_dtype = None
        self._set_raw_dtype(raw_dtype)

        self._formatted_shape = None
        self._set_formatted_shape(formatted_shape)

        self._formatted_dtype = None
        self._set_formatted_dtype(formatted_dtype)

        self._reverse_axes = None
        self._set_reverse_axes(reverse_axes)

        self._transpose_axes = None
        self._reverse_transpose_axes = None
        self._set_transpose_axes(transpose_axes)

        self._format_function = None
        self._set_format_function(format_function)
        self._validate_shapes()

    @property
    def raw_shape(self) -> Tuple[int, ...]:
        """
        Tuple[int, ...]: The raw shape.
        """

        return self._raw_shape

    def _set_raw_shape(self, value: Tuple[int, ...]) -> None:
        if not isinstance(value, tuple):
            raise TypeError(
                'raw_shape must be specified by a tuple of ints, got type `{}`'.format(type(value)))
        for entry in value:
            if not isinstance(entry, int):
                raise TypeError(
                    'raw_shape must be specified by a tuple of ints, got `{}`'.format(value))
            if entry <= 0:
                raise ValueError(
                    'raw_shape must be specified by a tuple of positive ints, got `{}`'.format(value))
        self._raw_shape = value

    @property
    def raw_ndim(self) -> int:
        """
        int: The number of raw dimensions.
        """

        return len(self._raw_shape)

    @property
    def mode(self) -> str:
        """
        str: The mode.
        """

        return self._mode

    def _set_mode(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError('Mode must be a string value')
        value = value.strip().lower()

        if value not in self._allowed_modes:
            raise ValueError('mode must be one of {}'.format(self._allowed_modes))

        self._mode = value

    @property
    def raw_dtype(self) -> numpy.dtype:
        """
        numpy.dtype: The data type of the data returned by the :func:`read_raw` function.
        """

        return self._raw_dtype

    def _set_raw_dtype(self, value) -> None:
        if not isinstance(value, numpy.dtype):
            try:
                value = numpy.dtype(value)
            except Exception as e:
                raise ValueError(
                    'Tried interpreting raw_dtype value as a numpy.dtype, '
                    'and failed with error\n\t{}'.format(e))
        self._raw_dtype = value

    @property
    def formatted_shape(self) -> Tuple[int, ...]:
        """
        Tuple[int, ...]: The formatted data shape.
        """

        return self._formatted_shape

    def _set_formatted_shape(self, value: Tuple[int, ...]) -> None:
        if not isinstance(value, tuple):
            raise TypeError(
                'formatted_shape must be specified by a tuple of ints, got type `{}`'.format(type(value)))
        for entry in value:
            if not isinstance(entry, int):
                raise TypeError(
                    'formatted_shape must be specified by a tuple of ints, got `{}`'.format(value))
            if entry <= 0:
                raise ValueError(
                    'formatted_shape must be specified by a tuple of positive ints, got `{}`'.format(value))
        self._formatted_shape = value

    @property
    def formatted_dtype(self) -> numpy.dtype:
        """
        numpy.dtype: The data type of the formatted data, which will be returned
        by the :func:`read` function.
        """

        return self._formatted_dtype

    def _set_formatted_dtype(self, value) -> None:
        if not isinstance(value, numpy.dtype):
            try:
                value = numpy.dtype(value)
            except Exception as e:
                raise ValueError(
                    'Tried interpreting formatted_dtype value as a numpy.dtype, '
                    'and failed with error\n\t{}'.format(e))
        self._formatted_dtype = value

    @property
    def formatted_ndim(self) -> int:
        """
        int: The number of formatted dimensions.
        """

        return len(self._formatted_shape)

    @property
    def reverse_axes(self) -> Optional[Tuple[int, ...]]:
        """
        None|Tuple[int, ...]: The collection of axes (with respect to raw order)
        along which we will reverse as part of transformation to formatted data order.
        If not `None`, then this will be a tuple in strictly increasing order.
        """

        return self._reverse_axes

    def _set_reverse_axes(self, value: Union[None, int, Tuple[int, ...]]) -> None:
        if value is None:
            self._reverse_axes = None
            return

        if isinstance(value, int):
            value = (value, )
        else:
            value = tuple(sorted(list(set(int(entry) for entry in value))))

        for entry in value:
            if not (0 <= entry < self.raw_ndim):
                raise ValueError('reverse_axes entries must be less than raw_ndim')

        self._reverse_axes = value

    @property
    def transpose_axes(self) -> Tuple[int, ...]:
        """
        None|Tuple[int, ...]: The transpose order for switching from raw order to
        formatted order, prior to applying any format function.

        If populated, this must be a permutation of `(0, 1, ..., raw_ndim-1)`.
        """

        return self._transpose_axes

    def _set_transpose_axes(self, value: Union[None, Tuple[int, ...]]) -> None:
        if value is None:
            self._transpose_axes = None
            return
        value = tuple([int(entry) for entry in value])
        if set(value) != set(range(self.raw_ndim)):
            raise ValueError('transpose_axes must be a permutation of range(raw_ndim), got\n\t{}'.format(value))
        self._transpose_axes = value

    @property
    def format_function(self) -> FormatFunction:
        """
        The format function which will be applied to the raw data.

        Returns
        -------
        FormatFunction
        """

        return self._format_function

    def _set_format_function(self, value: Optional[FormatFunction]) -> None:
        if value is None:
            value = IdentityFunction()
        if not isinstance(value, FormatFunction):
            raise ValueError('Got unexpected format_function value of type `{}`'.format(type(value)))

        # set our important property values
        value.set_raw_shape(self.raw_shape)
        value.set_formatted_shape(self.formatted_shape)
        value.set_reverse_axes(self.reverse_axes)
        value.set_transpose_axes(self.transpose_axes)
        self._format_function = value

    @property
    def can_write_regular(self) -> bool:
        """
        bool: Can this data segment write regular data, which requires a function
        inverse?
        """

        return self.mode == 'w' and self.format_function.has_inverse

    @property
    def closed(self) -> bool:
        """
        bool: Is the data segment closed? Reading or writing will result in a ValueError
        """

        return self._closed

    def _validate_closed(self):
        if not hasattr(self, '_closed') or self._closed:
            raise ValueError('I/O operation of closed data segment')

    def _validate_shapes(self) -> None:
        """
        Validate the raw_shape and formatted_shape values.
        """

        self.format_function.validate_shapes()

    # read related methods
    def verify_raw_subscript(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]]) -> Tuple[slice, ...]:
        """
        Verifies that the structure of the subscript is in keeping with the raw
        shape, and fills in any missing dimensions.

        Parameters
        ----------
        subscript : None|int|slice|Sequence[int|slice|Tuple[int, ...]]

        Returns
        -------
        Tuple[slice, ...]
            Guaranteed to be a tuple of slices of length `raw_ndim`.
        """

        return verify_subscript(subscript, self._raw_shape)

    def verify_formatted_subscript(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]]) -> Tuple[slice, ...]:
        """
        Verifies that the structure of the subscript is in keeping with the formatted
        shape, and fills in any missing dimensions.

        Parameters
        ----------
        subscript : None|int|slice|Sequence[int|slice|Tuple[int, ...]]

        Returns
        -------
        Tuple[slice, ...]
            Guaranteed to be a tuple of slices of length `formatted_ndim`.
        """

        return verify_subscript(subscript, self._formatted_shape)

    def _interpret_subscript(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            raw: bool = False) -> Tuple[slice, ...]:
        """
        Restructures the subscript to be a tuple of slices guaranteed to be the same
        length as the dimension of the return.

        Parameters
        ----------
        subscript : None|int|slice|Tuple[slice, ...]
        raw : bool
            If `True` then this should apply to raw coordinates, if `False` it
            should apply to original coordinates.

        Returns
        -------
        Tuple[slice, ...]
        """

        if raw:
            return verify_subscript(subscript, self._raw_shape)
        else:
            return verify_subscript(subscript, self._formatted_shape)

    def __getitem__(
            self,
            subscript: Union[None, int, slice, str, Sequence[Union[None, int, slice, str]]]) -> numpy.ndarray:
        """
        Fetch the data via slice definition.

        Parameters
        ----------
        subscript : None|int|slice|str|tuple

        Returns
        -------
        numpy.ndarray
        """

        # TODO: document the string entries situation
        self._validate_closed()

        subscript, string_entries = extract_string_from_subscript(subscript)
        use_raw = ('raw' in string_entries)
        squeeze = ('nosqueeze' not in string_entries)

        if use_raw:
            return self.read_raw(subscript, squeeze=squeeze)
        else:
            return self.read(subscript, squeeze=squeeze)

    def read(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:
        """
        In keeping with data segment mode, read the data slice specified relative
        to the formatted data coordinates. This requires that `mode` is `'r'`.

        Parameters
        ----------
        subscript : None|int|slice|Sequence[int|slice|Tuple[int, ...]]
        squeeze : bool
            Apply the numpy.squeeze operation, which eliminates dimension of size 1?

        Returns
        -------
        numpy.ndarray
        """

        self._validate_closed()

        if self.mode != 'r':
            raise ValueError('Requires mode = "r"')
        norm_subscript = self.verify_formatted_subscript(subscript)
        raw_subscript = self.format_function.transform_formatted_slice(norm_subscript)
        raw_data = self.read_raw(raw_subscript, squeeze=False)
        return self.format_function(raw_data, raw_subscript, squeeze=squeeze)

    # noinspection PyTypeChecker
    def read_raw(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:
        """
        In keeping with data segment mode, read raw data from the source, without
        reformatting and or applying symmetry operations. This requires that `mode`
        is `'r'`.

        Parameters
        ----------
        subscript : None|int|slice|Sequence[int|slice|Tuple[int, ...]]
            These arguments are relative to raw data shape and order, no symmetry
            operations have been applied.
        squeeze : bool
            Apply numpy.squeeze, which eliminates any dimensions of size 1?

        Returns
        -------
        numpy.ndarray
            This will be of data type given by `raw_dtype`.
        """

        if self.mode != 'r':
            raise ValueError('Requires mode == "r"')

        raise NotImplementedError

    def _verify_write_raw_details(self, data: numpy.ndarray) -> None:
        if self.mode != 'w':
            raise ValueError('I/O Error, functionality requires mode == "w"')

        if data.dtype.itemsize != self.raw_dtype.itemsize:
            raise ValueError(
                'Expected data dtype itemsize `{}`, got `{}`'.format(self.raw_dtype.itemsize, data.dtype.itemsize))
        if data.dtype != self.raw_dtype:
            logger.warning('Expected data dtype `{}`, got `{}`.'.format(self.raw_dtype, data.dtype))

    def write(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Sequence[slice]] = None,
            **kwargs) -> None:
        """
        In keeping with data segment mode, write the data provided in formatted
        form, assuming the slice specified relative to the formatted data coordinates.

        This requires that `mode` is `'w'`, and `format_function.has_inverse == True`,
        because we have to apply the format function inverse to the provided data.

        **Only one of `start_indices` and `subscript` should be specified.**

        Parameters
        ----------
        data : numpy.ndarray
            The data in formatted form, to be transferred to raw form and written.
        start_indices : None|int|Tuple[int, ...]
            Assuming a contiguous chunk of data, this provides the starting
            indices of the chunk. Any missing (tail) coordinates will be filled
            in with 0's.
        subscript : None|Sequence[slice]
            The subscript definition in formatted coordinates.
        kwargs

        Returns
        -------
        None
        """

        self._validate_closed()

        if not self.can_write_regular:
            raise ValueError(
                'I/O error, functionality requires mode = "w"\n\t'
                'and the ability to invert the format function')

        if data.dtype.itemsize != self.formatted_dtype.itemsize:
            raise ValueError(
                'Expected data dtype itemsize `{}`, got `{}`'.format(
                    self.formatted_dtype.itemsize, data.dtype.itemsize))
        if data.dtype != self.formatted_dtype:
            logger.warning('Expected data dtype `{}`, got `{}`.'.format(
                self.formatted_dtype, data.dtype))

        subscript = _infer_subscript_for_write(
            data, start_indices, subscript, self.formatted_shape)

        raw_data = self.format_function.inverse(data, subscript)
        raw_subscript = self.format_function.transform_formatted_slice(subscript)
        self.write_raw(raw_data, subscript=raw_subscript, **kwargs)

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Sequence[slice]] = None,
            **kwargs) -> None:
        """
        In keeping with data segment mode, write the data provided in raw form,
        assuming the slice specified relative to raw data coordinates. This
        requires that `mode` is `'w'`.

        **Only one of `start_indices` and `subscript` should be specified.**

        Parameters
        ----------
        data : numpy.ndarray
            The data in raw form.
        start_indices : None|int|Tuple[int, ...]
            Assuming a contiguous chunk of data, this provides the starting
            indices of the chunk. Any missing (tail) coordinates will be filled
            in with 0's.
        subscript : None|Tuple[slice, ...]
            The subscript definition in raw coordinates.
        kwargs

        Returns
        -------
        None
        """

        raise NotImplementedError

    def check_fully_written(self, warn: bool = False) -> bool:
        """
        Checks that all expected pixel data is fully written.

        Parameters
        ----------
        warn : bool
            Log warning with some details if not fully written.

        Returns
        -------
        bool
        """

        raise NotImplementedError

    def get_raw_bytes(self, warn: bool = True) -> Union[bytes, Tuple]:
        """
        This returns the bytes for the underlying raw data.

        .. warning::
            A data segment is *conceptually* represented in raw data as a single
            numpy array of appropriate shape and data type. When the data segment
            is formed from component pieces, then the return of this function may
            deviate significantly from the raw byte representation of such an
            array after consideration of data order and pad pixels.

        Parameters
        ----------
        warn : bool
            If `True`, then a check will be performed to ensure that the data
            has been fully written and warnings printed if the answer is no.

        Returns
        -------
        bytes|Tuple
            The result will be a `bytes` object, unless the data segment is
            made up of a collection of child data segments, in which case the
            result will be a Tuple consisting of their `get_raw_bytes` returns.
        """

        raise NotImplementedError

    def flush(self) -> None:
        """
        Should perform, if possible, any necessary steps to flush any unwritten
        data to the file.

        Returns
        -------
        None
        """

        return

    def close(self):
        """
        This should perform any necessary clean-up operations, like closing
        open file handles, deleting any temp files, etc
        """

        if not hasattr(self, '_closed') or self._closed:
            return
        self._closed = True

    def __del__(self):
        # NB: this is called when the object is marked for garbage collection
        # (i.e. reference count goes to 0)
        # This order in which this happens may be unreliable
        self.close()


class ReorientationSegment(DataSegment):
    """
    Define a basic ordering of a given DataSegment. The raw data will be
    presented as the parent data segment's formatted data.

    Introduced in version 1.3.0.
    """

    __slots__ = ('_parent', '_close_parent')

    def __init__(
            self,
            parent: DataSegment,
            formatted_dtype: Optional[Union[str, numpy.dtype]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Union[int, Sequence[int]]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            close_parent: bool = True):
        """
        Parameters
        ----------
        parent : DataSegment
        formatted_dtype : str|numpy.dtype
        formatted_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform, after applying any axis reversal,
            and before applying any format function
        close_parent : bool
            Call parent.close() when close is called?
        """

        self._parent = None
        self._close_parent = None
        self.close_parent = close_parent
        intermediate_shape = self._set_parent(parent, transpose_axes)
        if format_function is None:
            formatted_dtype = parent.formatted_dtype
            formatted_shape = intermediate_shape
        else:
            if formatted_dtype is None or formatted_shape is None:
                raise ValueError(
                    'If format_function is provided,\n\t'
                    'then formatted_dtype and formatted_shape must be provided.')
        mode = parent.mode
        if mode == 'w' and not parent.can_write_regular:
            raise ValueError('Requires that the parent can write regular data')

        DataSegment.__init__(
            self, parent.formatted_dtype, parent.formatted_shape, formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode=mode)

    @property
    def parent(self) -> DataSegment:
        return self._parent

    def _set_parent(self,
                    parent: DataSegment,
                    transpose_axes: Union[None, Tuple[int, ...]]) -> Tuple[int, ...]:
        if transpose_axes is None:
            trans_axes = tuple(range(parent.formatted_ndim))
        else:
            if len(transpose_axes) != parent.formatted_ndim:
                raise ValueError('transpose_axes must have length {}'.format(parent.formatted_ndim))
            trans_axes = transpose_axes
        self._parent = parent
        return tuple([parent.formatted_shape[index] for index in trans_axes])

    @property
    def close_parent(self) -> bool:
        """
        bool: Call parent.close() when close is called?
        """

        return self._close_parent

    @close_parent.setter
    def close_parent(self, value):
        self._close_parent = bool(value)

    def read_raw(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:

        self._validate_closed()
        if self.mode != 'r':
            raise ValueError('Requires mode == "r"')

        return self.parent.read(subscript, squeeze=squeeze)

    def check_fully_written(self, warn: bool = False) -> bool:
        return self.parent.check_fully_written(warn=warn)

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Sequence[slice]] = None,
            **kwargs):
        """
        In keeping with data segment mode, write the data provided in raw form,
        assuming the slice specified relative to raw data coordinates. This
        requires that `mode` is `'w'`.

        Note that raw form/order for the data segment is simply a reordered version
        of the formatted data for parent. This **is not** related to raw data
        with respect to the parent. To write raw data with respect to the parent,
        use :func:`parent.write_raw` instead.

        **Only one of `start_indices` and `subscript` should be specified.**

        Parameters
        ----------
        data : numpy.ndarray
            The data in raw form.
        start_indices : None|int|Tuple[int, ...]
            Assuming a contiguous chunk of data, this provides the starting
            indices of the chunk. Any missing (tail) coordinates will be filled
            in with 0's.
        subscript : None|Sequence[slice]
            The subscript definition in raw coordinates.
        kwargs

        Returns
        -------
        None
        """

        self._validate_closed()
        self._verify_write_raw_details(data)

        subscript = _infer_subscript_for_write(data, start_indices, subscript, self.raw_shape)
        parent_form_subscript = self.format_function.transform_formatted_slice(subscript)
        self.parent.write(data, subscript=parent_form_subscript, **kwargs)

    def get_raw_bytes(self, warn: bool = True) -> Union[bytes, Tuple]:
        self._validate_closed()
        return self.parent.get_raw_bytes(warn=warn)

    def flush(self) -> None:
        self._validate_closed()
        try:
            self.parent.flush()
        except AttributeError:
            return

    def close(self):
        try:
            if self._closed:
                return

            self.flush()
            if self.close_parent:
                self.parent.close()
            DataSegment.close(self)
            self._parent = None
        except AttributeError:
            return


class SubsetSegment(DataSegment):
    """
    Define a subset of a given DataSegment, with formatting handled by the
    parent data segment.

    Introduced in version 1.3.0.
    """
    _allowed_modes = ('r', 'w')

    __slots__ = (
        '_parent', '_formatted_subset_definition', '_raw_subset_definition',
        '_original_formatted_indices', '_original_raw_indices', '_squeeze',
        '_close_parent', '_pixels_written', '_expected_pixels_written')

    def __init__(
            self,
            parent: DataSegment,
            subset_definition: Tuple[slice, ...],
            coordinate_basis: str,
            squeeze: bool = True,
            close_parent: bool = True):
        """
        Parameters
        ----------
        parent : DataSegment
        subset_definition : Tuple[slice, ...]
        coordinate_basis : str
            The coordinate basis for the subset definition, it should be one of
            `('raw', 'formatted')`.
        squeeze: bool
            Eliminate the dimensions that are size 1 in the subset?
        close_parent : bool
            Call parent.close() when close is called?
        """

        self._close_parent = None
        self.close_parent = close_parent
        self._original_formatted_indices = None  # the original indices matched to the new, with entry -1 when flat
        self._original_raw_indices = None
        self._formatted_subset_definition = None
        self._raw_subset_definition = None
        self._squeeze = squeeze
        self._parent = parent

        raw_shape, formatted_shape = self._validate_subset_definition(
            subset_definition, coordinate_basis)
        DataSegment.__init__(
            self, parent.raw_dtype, raw_shape, parent.formatted_dtype, formatted_shape,
            mode=parent.mode)
        self._pixels_written = 0
        if self.mode == 'w':
            self._expected_pixels_written = int(numpy.prod(raw_shape))
        else:
            self._expected_pixels_written = 0

    def _validate_shapes(self) -> None:
        # handled else where
        pass

    @property
    def parent(self) -> DataSegment:
        return self._parent

    @property
    def formatted_subset_definition(self) -> Tuple[slice, ...]:
        """
        Tuple[slice]: The subset definition, in formatted coordinates.
        """

        return self._formatted_subset_definition

    @property
    def raw_subset_definition(self) -> Tuple[slice, ...]:
        """
        Tuple[slice]: The subset definition, in raw coordinates.
        """

        return self._raw_subset_definition

    @property
    def close_parent(self) -> bool:
        """
        bool: Call parent.close() when close is called?
        """

        return self._close_parent

    @close_parent.setter
    def close_parent(self, value):
        self._close_parent = bool(value)

    def _validate_subset_definition(
            self,
            subset_definition: Tuple[slice, ...],
            coordinate_basis: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Validates the subset definition.

        Parameters
        ----------
        subset_definition : Tuple[slice, ...]
        coordinate_basis : str

        Returns
        -------
        raw_shape : Tuple[int, ...]
        formatted_shape : Tuple[int, ...]
        """

        raw_shape = []
        formatted_shape = []
        original_indices = []
        raw_indices = []

        coordinate_basis = coordinate_basis.strip().lower()
        if coordinate_basis == 'raw':
            raw_def = self.parent.verify_raw_subscript(subset_definition)
            form_def = self.parent.format_function.transform_raw_slice(raw_def)
        elif coordinate_basis == 'formatted':
            form_def = self.parent.verify_formatted_subscript(subset_definition)
            raw_def = self.parent.format_function.transform_formatted_slice(form_def)
        else:
            raise ValueError('Got unexpected coordinate basis `{}`'.format(coordinate_basis))

        for index, entry in enumerate(form_def):
            this_size = get_slice_result_size(entry)
            if self._squeeze and this_size == 1:
                logger.info('Entry at index {} of subset definition yields a single entry'.format(index))
                original_indices.append(-1)
            else:
                formatted_shape.append(this_size)
                original_indices.append(index)

        self._formatted_subset_definition = form_def
        self._raw_subset_definition = raw_def
        self._original_formatted_indices = tuple(original_indices)

        for index, entry in enumerate(raw_def):
            this_size = get_slice_result_size(entry)
            if self._squeeze and this_size == 1:
                logger.info('Raw slice at index {} of subset definition yields a single entry'.format(index))
                raw_indices.append(-1)
            else:
                raw_shape.append(this_size)
                raw_indices.append(index)
        self._original_raw_indices = tuple(raw_indices)
        return tuple(raw_shape), tuple(formatted_shape)

    def _get_parent_subscript(
            self,
            norm_subscript: Tuple[slice, ...],
            this_shape: Tuple[int, ...],
            full_shape: Tuple[int, ...],
            use_indices: Tuple[int, ...],
            subset_definition: Tuple[slice, ...]) -> Tuple[slice, ...]:
        """
        Helper function for defining a parent subscript from the subset subscript definition.

        Parameters
        ----------
        norm_subscript : Tuple[slice, ...]
            The normalized subset subscript.
        this_shape : Tuple[int, ...]
            The shape in the subset domain.
        full_shape : Tuple[int, ...]
            The full parent shape in the given domain.
        use_indices : Tuple[int, ...]
            Structure helping to identify the dimensions from the parent which
            have been preserved, and which have collapsed.
        subset_definition : Tuple[slice, ...]
            The subset definition with respect to parent coordinates.

        Returns
        -------
        Tuple[slice, ...]
        """

        out = []
        for full_size, out_index, slice_def in zip(full_shape, use_indices, subset_definition):
            if out_index == -1:
                out.append(slice_def)
            else:
                part_def = norm_subscript[out_index]
                step = part_def.step*slice_def.step
                # now, extract start and stop
                # the logic of this is terrible, so let's just do it the easy way
                the_array = numpy.arange(full_size)[slice_def][part_def]
                if len(the_array) < 1:
                    raise KeyError('Got invalid slice definition {} for shape {}'.format(norm_subscript, this_shape))
                start = the_array[0]
                stop = the_array[-1] + step
                if stop < 0:
                    stop = None
                elif stop > full_size:
                    stop = full_size
                out.append(slice(start, stop, step))
        return tuple(out)

    def get_parent_raw_subscript(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice]]]) -> Tuple[slice, ...]:
        """
        Gets the raw parent subscript from the raw subset subscript definition.

        Parameters
        ----------
        subscript : None|int|slice|Sequence[int|slice]

        Returns
        -------
        Tuple[slice, ...]
        """

        return self._get_parent_subscript(
            self.verify_raw_subscript(subscript), self.raw_shape, self.parent.raw_shape,
            self._original_raw_indices, self._raw_subset_definition)

    def get_parent_formatted_subscript(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice]]]) -> Tuple[slice, ...]:
        """
        Gets the formatted parent subscript from the formatted subset subscript definition.

        Parameters
        ----------
        subscript : None|int|slice|Sequence[int|slice]

        Returns
        -------
        Tuple[slice, ...]
        """

        out = self._get_parent_subscript(
            self.verify_formatted_subscript(subscript), self.formatted_shape, self.parent.formatted_shape,
            self._original_formatted_indices, self._formatted_subset_definition)
        return out

    def read_raw(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:
        self._validate_closed()
        if self.mode != 'r':
            raise ValueError('Requires mode == "r"')

        norm_subscript = self.get_parent_raw_subscript(subscript)
        if squeeze:
            return self.parent.read_raw(norm_subscript, squeeze=True)
        else:
            data = self.parent.read_raw(norm_subscript, squeeze=False)
            use_shape = []
            for check, size in zip(self._original_raw_indices, data.shape):
                if not self._squeeze or check != -1:
                    use_shape.append(size)
            if squeeze:
                return numpy.squeeze(data)
            else:
                return numpy.reshape(data, tuple(use_shape))

    def read(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice]]],
            squeeze=True) -> numpy.ndarray:
        self._validate_closed()
        if self.mode != 'r':
            raise ValueError('Requires mode == "r"')

        norm_subscript = self.get_parent_formatted_subscript(subscript)
        if squeeze:
            return self.parent.read(norm_subscript, squeeze=True)
        else:
            data = self.parent.read(norm_subscript, squeeze=False)
            use_shape = []
            for check, size in zip(self._original_formatted_indices, data.shape):
                if not self._squeeze or check != -1:
                    use_shape.append(size)
            if squeeze:
                return numpy.squeeze(data)
            else:
                return numpy.reshape(data, tuple(use_shape))

    def check_fully_written(self, warn: bool = False) -> bool:
        if self.mode == 'r':
            return True

        if self._pixels_written < self._expected_pixels_written:
            if warn:
                logger.error(
                    'Segment expected {} pixels written, but only {} pixels were written'.format(
                        self._expected_pixels_written, self._pixels_written))
            return False
        elif self._pixels_written == self._expected_pixels_written:
            return True
        else:
            if warn:
                logger.error(
                    'Segment expected {} pixels written,\n\t'
                    'but {} pixels were written.\n\t'
                    'This redundancy may be an error'.format(
                        self._expected_pixels_written, self._pixels_written))
            return False

    def _update_pixels_written(self, written: int) -> None:
        new_pixels_written = self._pixels_written + written
        if self._pixels_written <= self._expected_pixels_written < new_pixels_written:
            logger.error(
                'Segment expected {} pixels written,\n\t'
                'but now has {} pixels written.\n\t'
                'This redundancy may be an error'.format(
                    self._expected_pixels_written, new_pixels_written))
        self._pixels_written = new_pixels_written

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Sequence[slice]] = None,
            **kwargs):
        self._validate_closed()
        self._verify_write_raw_details(data)
        subscript = _infer_subscript_for_write(data, start_indices, subscript, self.raw_shape)
        parent_subscript = self.get_parent_raw_subscript(subscript)
        self.parent.write_raw(data, subscript=parent_subscript, **kwargs)
        self._update_pixels_written(data.size)

    def get_raw_bytes(self, warn: bool = True) -> Union[bytes, Tuple]:
        """
        This returns the bytes for the underlying raw data **of the parent segment.**

        The only writing use case considered at present is for the blocks including
        padding inside a NITF file.

        Parameters
        ----------
        warn : bool
            If `True`, then a check will be performed to ensure that the data
            has been fully written.

        Returns
        -------
        bytes|Tuple
            The result will be a `bytes` object, unless the data segment is
            made up of a collection of child data segments, in which case the
            result will be a Tuple consisting of their `get_raw_bytes` returns.
        """

        self._validate_closed()
        if warn and not self.check_fully_written(warn=True):
            logger.error(
                'There has been a call to `get_raw_bytes` from {},\n\t'
                'but all pixels are not fully written'.format(self.__class__))
        return self.parent.get_raw_bytes(warn=False)

    def flush(self) -> None:
        self._validate_closed()
        try:
            self.parent.flush()
        except AttributeError:
            return

    def close(self):
        try:
            if self._closed:
                return

            self.flush()
            if self.close_parent:
                self.parent.close()
            DataSegment.close(self)
            self._parent = None
        except (ValueError, AttributeError):
            return


class BandAggregateSegment(DataSegment):
    """
    This stacks a collection of data segments, which must have compatible details,
    together along a new (final) band dimension.

    Note that :func:`read` and :func:`read_raw` return identical results here.
    To access raw data from the children, use access on the `children` property.

    Introduced in version 1.3.0.
    """

    __slots__ = ('_band_dimension', '_children', '_close_children')

    def __init__(
            self,
            children: Sequence[DataSegment],
            band_dimension: int,
            formatted_dtype: Optional[Union[str, numpy.dtype]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Union[int, Sequence[int]]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            close_children: bool = True):
        """

        Parameters
        ----------
        children : Sequence[DataSegment]
        band_dimension : int
            The band dimension. This must remain unchanged in transpose_axes,
            and is not permitted to be reversed by reverse_axes.
        formatted_dtype : str|numpy.dtype
        formatted_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform, after applying any axis reversal,
            and before applying any format function
        format_function : None|FormatFunction
        close_children : bool
        """

        self._band_dimension = None
        self._close_children = None
        self.close_children = close_children
        self._children = None
        self._set_band_dimension(band_dimension, reverse_axes, transpose_axes)
        raw_dtype, raw_shape, form_shape, the_mode = self._set_children(children, transpose_axes)

        if format_function is None:
            formatted_dtype = raw_dtype
            formatted_shape = form_shape
        else:
            if formatted_dtype is None or formatted_shape is None:
                raise ValueError(
                    'If format_function is provided,\n\t'
                    'then formatted_dtype and formatted_shape must be provided.')
        DataSegment.__init__(
            self, raw_dtype, raw_shape, formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode=the_mode)

    @property
    def band_dimension(self) -> int:
        """
        int: The band dimension, in raw data after the transpose operation.
        """

        return self._band_dimension

    def _set_band_dimension(
            self,
            value: int,
            reverse_axes: Union[None, int, Sequence[int]],
            transpose_axes: Union[None, Tuple[int, ...]]) -> None:
        if not isinstance(value, int):
            raise TypeError('band_dimension must be an integer')
        if value < 0:
            raise TypeError('band_dimension must be non-negative')

        if transpose_axes is not None:
            if value != transpose_axes[value]:
                raise ValueError('band_dimension is not permitted to be changed by transpose_axes.')

        if reverse_axes is None:
            pass
        elif isinstance(reverse_axes, int):
            if value == reverse_axes:
                raise ValueError('Reversal along the band dimension is not permitted')
        else:
            if value in reverse_axes:
                raise ValueError('Reversal along the band dimension is not permitted')

        if self._band_dimension is not None:
            if value != self._band_dimension:
                raise ValueError('band_dimension is read only once set')
            return  # nothing to be done
        self._band_dimension = value

    @property
    def close_children(self) -> bool:
        """
        bool: Call child.close() when close is called?
        """

        return self._close_children

    @close_children.setter
    def close_children(self, value):
        self._close_children = bool(value)

    @property
    def children(self) -> Tuple[DataSegment, ...]:
        """
        The collection of children that we are stacking.

        Returns
        -------
        Tuple[DataSegment, ...]
        """

        return self._children

    def _set_children(
            self,
            children: Sequence[DataSegment],
            transpose_axes: Optional[Tuple[int, ...]]) -> Tuple[numpy.dtype, Tuple[int, ...], Tuple[int, ...], str]:
        if len(children) < 2:
            raise ValueError('Cannot define a BandAggregateSegment based on fewer than 2 segments.')

        child_shape = children[0].formatted_shape
        the_dtype = children[0].formatted_dtype
        the_mode = children[0].mode

        if transpose_axes is None:
            transpose_axes = tuple(range(0, len(child_shape) + 1))

        raw_shape = [entry for entry in child_shape]
        raw_shape.insert(self.band_dimension, len(children))
        raw_shape = tuple(raw_shape)

        form_shape = tuple(raw_shape[entry] for entry in transpose_axes)

        use_children = []
        for child in children:
            if child.formatted_shape != child_shape:
                raise ValueError('All children must have the same formatted shape')
            if child.formatted_dtype != the_dtype:
                raise ValueError('All children must have the same formatted dtype')
            if child.mode != the_mode:
                raise ValueError('All children must have the same mode')
            if child.mode == 'w' and not child.can_write_regular:
                raise ValueError('write mode requires that all children can write regular data')
            use_children.append(child)
        self._children = tuple(use_children)
        return the_dtype, raw_shape, form_shape, the_mode

    @property
    def bands(self) -> int:
        """
        int: The number of bands (child data segments)
        """

        return len(self.children)

    def read_raw(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:
        self._validate_closed()
        if self.mode != 'r':
            raise ValueError('Requires mode == "r"')

        norm_subscript, the_shape = get_subscript_result_size(subscript, self.raw_shape)
        out = numpy.empty(the_shape, dtype=self.raw_dtype)
        full_band_subscript = tuple(slice(0, entry, 1) for entry in the_shape)

        for out_index, index in enumerate(numpy.arange(self.bands)[norm_subscript[self.band_dimension]]):
            child_subscript = norm_subscript[:self.band_dimension] + norm_subscript[self.band_dimension+1:]
            band_subscript = full_band_subscript[:self.band_dimension] + \
                (out_index, ) + \
                full_band_subscript[self.band_dimension+1:]
            out[band_subscript] = self.children[index].read(child_subscript, squeeze=False)

        if squeeze:
            return numpy.squeeze(out)
        else:
            return out

    def check_fully_written(self, warn: bool = False) -> bool:
        if self.mode == 'r':
            return True

        out = True
        for i, child in enumerate(self.children):
            done = child.check_fully_written(warn=warn)
            if warn and not done:
                logger.error('Band {} of BandAggregateSegment indicates incomplete writing'.format(i))
            out &= done
        return out

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Sequence[slice]] = None,
            **kwargs):
        """
        In keeping with data segment mode, write the data provided in raw form,
        assuming the slice specified relative to raw data coordinates. This
        requires that `mode` is `'w'`.

        Note that raw form/order for the data segment is simply a reordered version
        of the formatted data for parent. This **is not** related to raw data
        with respect to the parent. To write raw data with respect to the parent,
        use :func:`parent.write_raw` instead.

        **Only one of `start_indices` and `subscript` should be specified.**

        Parameters
        ----------
        data : numpy.ndarray
            The data in raw form.
        start_indices : None|int|Tuple[int, ...]
            Assuming a contiguous chunk of data, this provides the starting
            indices of the chunk. Any missing (tail) coordinates will be filled
            in with 0's.
        subscript : None|Sequence[slice]
            The subscript definition in raw coordinates.
        kwargs

        Returns
        -------
        None
        """

        self._validate_closed()
        self._verify_write_raw_details(data)

        norm_subscript = _infer_subscript_for_write(
            data, start_indices, subscript, self.raw_shape)
        # iterate over each band, and write the raw data there...
        for out_index, index in enumerate(
                numpy.arange(self.bands)[norm_subscript[self.band_dimension]]):
            child_subscript = norm_subscript[:self.band_dimension] + \
                norm_subscript[self.band_dimension+1:]
            band_subscript = norm_subscript[:self.band_dimension] + \
                (out_index, ) + \
                norm_subscript[self.band_dimension+1:]
            self.children[index].write(
                data[band_subscript], subscript=child_subscript, **kwargs)

    def get_raw_bytes(self, warn: bool = True) -> Union[bytes, Tuple]:
        self._validate_closed()
        return tuple(entry.get_raw_bytes(warn=warn) for entry in self.children)

    def flush(self) -> None:
        self._validate_closed()
        try:
            if self.children is not None:
                for child in self.children:
                    child.flush()
        except AttributeError:
            return

    def close(self):
        try:
            if self._closed:
                return

            self.flush()
            if self._children is not None:
                for entry in self._children:
                    entry.close()
            DataSegment.close(self)
            self._children = None
        except AttributeError:
            return


class BlockAggregateSegment(DataSegment):
    """
    Combines a collection of child data segments, according to a given
    block definition. All children must have the same formatted_dtype. This
    implementation is motivated by a two-dimensional block arrangement, but is
    entirely general.

    No effort is made to ensure that the block definition spans the whole space,
    nor is any effort made to ensure that blocks do not overlap.

    If there are holes present in block definition, then data read across any
    hole will be populated with `missing_data_value`. Data attempted to write
    across any hole will simply be ignored.

    Introduced in version 1.3.0.
    """

    __slots__ = (
        '_children', '_formatted_child_arrangement', '_raw_child_arrangement',
        '_missing_data_value', '_close_children')

    def __init__(
            self,
            children: Sequence[DataSegment],
            child_arrangement: Sequence[Tuple[slice, ...]],
            coordinate_basis: str,
            missing_data_value,
            raw_shape: Tuple[int, ...],
            formatted_dtype: Union[str, numpy.dtype],
            formatted_shape: Tuple[int, ...],
            reverse_axes: Optional[Union[int, Sequence[int]]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            close_children: bool = True):
        """

        Parameters
        ----------
        children : Sequence[DataSegment]
            The collection of children
        child_arrangement : Sequence[Tuple[slice, ...]]
            The collection of definitions for each block. Overlap in definition
            is permitted.
        missing_data_value
            Missing data value, which must be compatible with
            raw_dtype=child.formatted_dtype.
        close_children : bool
        """

        self.close_children = close_children
        self._close_children = None

        self._children = None
        self._formatted_child_arrangement = None
        self._raw_child_arrangement = None
        self._missing_data_value = missing_data_value
        raw_dtype = children[0].formatted_dtype
        the_mode = children[0].mode
        DataSegment.__init__(
            self, raw_dtype, raw_shape, formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode=the_mode)
        self._set_children(children, child_arrangement, coordinate_basis)

    @property
    def close_children(self) -> bool:
        """
        bool: Call child.close() when close is called?
        """

        return self._close_children

    @close_children.setter
    def close_children(self, value):
        self._close_children = bool(value)

    @property
    def children(self) -> Tuple[DataSegment, ...]:
        """
        The collection of children that we are stacking together.

        Returns
        -------
        Tuple[DataSegment, ...]
        """

        return self._children

    def _set_children(
            self,
            children: Sequence[DataSegment],
            child_arrangement: Sequence[Tuple[slice, ...]],
            coordinate_basis: str) -> None:

        if len(children) != len(child_arrangement):
            raise ValueError('We must have the same number of children as child_arrangement entries')

        coordinate_basis = coordinate_basis.strip().lower()
        if coordinate_basis == 'raw':
            raw_arrangement = [self.verify_raw_subscript(entry) for entry in child_arrangement]
            formatted_arrangement = [self.format_function.transform_raw_slice(entry) for entry in raw_arrangement]
        elif coordinate_basis == 'formatted':
            formatted_arrangement = [self.verify_formatted_subscript(entry) for entry in child_arrangement]
            raw_arrangement = [self.format_function.transform_formatted_slice(entry) for entry in formatted_arrangement]
        else:
            raise ValueError('Got unexpected coordinate basis `{}`'.format(coordinate_basis))

        for i, (child, raw_def, form_def) in enumerate(zip(children, raw_arrangement, formatted_arrangement)):
            if child.formatted_dtype != self.raw_dtype:
                raise ValueError(
                    'Each child.formatted_dtype must be identical to\n\t'
                    'self.raw_dtype = {}'.format(self.raw_dtype))

            for entry in raw_def:
                if entry.step not in [1, -1]:
                    raise ValueError('Each entry of child_arrangement must have step +/-1.')
            for entry in form_def:
                if entry.step not in [1, -1]:
                    raise ValueError('Each entry of child_arrangement must have step +/-1.')

            # verify the shape is sensible
            _, result_shape = get_subscript_result_size(raw_def, self.raw_shape)
            if result_shape != child.formatted_shape:
                raise ValueError(
                    'child_arrangement definition expects child {} to have formatted_shape {},\n\t'
                    'but it has formatted_shape {}'.format(i, result_shape, child.formatted_shape))

        self._children = tuple(children)
        self._raw_child_arrangement = tuple(raw_arrangement)
        self._formatted_child_arrangement = tuple(formatted_arrangement)

    def read_raw(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:
        self._validate_closed()
        if self.mode != 'r':
            raise ValueError('Requires mode == "r"')

        subscript, formatted_shape = get_subscript_result_size(subscript, self.raw_shape)
        out = numpy.full(formatted_shape, fill_value=self._missing_data_value, dtype=self.raw_dtype)

        for entry, child in zip(self._raw_child_arrangement, self._children):
            use_block = True
            parent_subscript = []
            child_subscript = []
            for data_slice, block_slice in zip(subscript, entry):
                child_entry, par_entry = _find_slice_overlap(data_slice, block_slice)
                if par_entry is None:
                    use_block = False
                    # there is no overlap
                    break
                else:
                    parent_subscript.append(par_entry)
                    child_subscript.append(child_entry)
            if use_block:
                out[tuple(parent_subscript)] = child.read_raw(tuple(child_subscript), squeeze=False)

        if squeeze:
            return numpy.squeeze(out)
        else:
            return out

    def check_fully_written(self, warn: bool = False) -> bool:
        if self.mode == 'r':
            return True

        out = True
        for i, child in enumerate(self.children):
            done = child.check_fully_written(warn=warn)
            if warn and not done:
                logger.error('Block {} of BlockAggregateSegment indicates incomplete writing'.format(i))
            out &= done
        return out

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Sequence[slice]] = None,
            **kwargs):
        """
        In keeping with data segment mode, write the data provided in raw form,
        assuming the slice specified relative to raw data coordinates. This
        requires that `mode` is `'w'`.

        Note that raw form/order for the data segment is broken into blocks of
        reordered version of the formatted data for each child. This **is not**
        related to raw data with respect to the child. To write raw data with
        respect to the parent, use :func:`child.write_raw` instead.

        **Only one of `start_indices` and `subscript` should be specified.**

        Parameters
        ----------
        data : numpy.ndarray
            The data in raw form.
        start_indices : None|int|Tuple[int, ...]
            Assuming a contiguous chunk of data, this provides the starting
            indices of the chunk. Any missing (tail) coordinates will be filled
            in with 0's.
        subscript : None|Sequence[slice]
            The subscript definition in raw coordinates.
        kwargs

        Returns
        -------
        None
        """

        self._validate_closed()
        self._verify_write_raw_details(data)

        norm_subscript = _infer_subscript_for_write(data, start_indices, subscript, self.raw_shape)
        for entry, child in zip(self._raw_child_arrangement, self._children):
            # determine if there is overlap of norm_subscript with this block,
            # and write the appropriate data, if so.
            use_block = True
            data_subscript = []
            child_subscript = []
            for lim, data_slice, block_slice in zip(self.raw_shape, norm_subscript, entry):
                child_entry, par_entry = _find_slice_overlap(data_slice, block_slice)
                # this expresses the overlap between our data slice in overall coordinates
                #   relative to the entire raw_shape, and coordinates relative to
                #   just the block in question
                if par_entry is None:
                    use_block = False
                    # there is no overlap
                    break

                # we need to convert from overall coordinates relative to the entire
                # raw image, to coordinates relative to just the data shape
                _, data_entry = _find_slice_overlap(slice(0, lim, 1), par_entry)
                data_subscript.append(data_entry)
                child_subscript.append(child_entry)

            if use_block:
                child.write(data[tuple(data_subscript)], subscript=tuple(child_subscript), **kwargs)

    def get_raw_bytes(self, warn: bool = True) -> Union[bytes, Tuple]:
        self._validate_closed()
        return tuple(entry.get_raw_bytes(warn=warn) for entry in self.children)

    def flush(self) -> None:
        self._validate_closed()
        try:
            if self.children is not None:
                for child in self.children:
                    child.flush()
        except AttributeError:
            return

    def close(self):
        try:
            if self._closed:
                return

            self.flush()
            if self._children is not None:
                for entry in self._children:
                    entry.close()
            DataSegment.close(self)
            self._children = None
        except AttributeError:
            return


####
# Concrete implementations

class NumpyArraySegment(DataSegment):
    """
    DataSegment based on reading from a numpy.ndarray.

    Introduced in version 1.3.0.
    """

    __slots__ = ('_underlying_array', '_pixels_written', '_expected_pixels_written')

    def __init__(
            self,
            underlying_array: numpy.ndarray,
            formatted_dtype: Optional[Union[str, numpy.dtype]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Union[int, Sequence[int]]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            mode: str = 'r'):
        """

        Parameters
        ----------
        underlying_array : numpy.ndarray
        formatted_dtype : str|numpy.dtype
        formatted_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
        mode : str
        """

        if not isinstance(underlying_array, numpy.ndarray):
            raise TypeError(
                'underlying array must be a numpy.ndarray, got type `{}`'.format(
                    type(underlying_array)))
        self._underlying_array = underlying_array
        self._pixels_written = 0
        if formatted_dtype is None:
            if format_function is None:
                formatted_dtype = underlying_array.dtype
            else:
                raise ValueError(
                    'Format function is provided, so formatted_dtype must be provided.')
        if formatted_shape is None:
            if format_function is None:
                if transpose_axes is None:
                    formatted_shape = underlying_array.shape
                else:
                    formatted_shape = [underlying_array.shape[index] for index in transpose_axes]
            else:
                raise ValueError(
                    'Format function is provided, so formatted_shape must be provided.')

        DataSegment.__init__(
            self, underlying_array.dtype, underlying_array.shape, formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes, format_function=format_function,
            mode=mode)
        if self.mode == 'w':
            self._expected_pixels_written = self._underlying_array.size
        else:
            self._expected_pixels_written = 0

    @property
    def underlying_array(self) -> numpy.ndarray:
        """
        The underlying data array.

        Returns
        -------
        numpy.ndarray
        """

        return self._underlying_array

    def read_raw(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:
        self._validate_closed()
        if self.mode != 'r':
            raise ValueError('Requires mode == "r"')

        subscript, out_shape = get_subscript_result_size(subscript, self.raw_shape)
        out = self._underlying_array[subscript]  # squeezed by default

        if squeeze:
            return numpy.squeeze(out)
        else:
            return numpy.reshape(out, out_shape)

    def check_fully_written(self, warn: bool = False) -> bool:
        if self.mode == 'r':
            return True

        if self._pixels_written < self._expected_pixels_written:
            if warn:
                logger.error(
                    'Segment expected {} pixels written, but only {} pixels were written'.format(
                        self._expected_pixels_written, self._pixels_written))
            return False
        elif self._pixels_written == self._expected_pixels_written:
            return True
        else:
            if warn:
                logger.error(
                    'Segment expected {} pixels written,\n\t'
                    'but {} pixels were written.\n\t'
                    'This redundancy may be an error'.format(
                        self._expected_pixels_written, self._pixels_written))
            return False

    def _update_pixels_written(self, written: int) -> None:
        new_pixels_written = self._pixels_written + written
        if self._pixels_written <= self._expected_pixels_written < new_pixels_written:
            logger.error(
                'Segment expected {} pixels written,\n\t'
                'but now has {} pixels written.\n\t'
                'This redundancy may be an error'.format(
                    self._expected_pixels_written, new_pixels_written))
        self._pixels_written = new_pixels_written

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Optional[Union[int, Tuple[int, ...]]] = None,
            subscript: Optional[Sequence[slice]] = None,
            **kwargs):
        self._validate_closed()
        self._verify_write_raw_details(data)
        subscript = _infer_subscript_for_write(data, start_indices, subscript, self.raw_shape)
        self._underlying_array[subscript] = data
        self._update_pixels_written(data.size)

    def get_raw_bytes(self, warn: bool = False) -> Union[bytes, Tuple]:
        self._validate_closed()
        if warn and not self.check_fully_written(warn=True):
            logger.error(
                'There has been a call to `get_raw_bytes` from {},\n\t'
                'but all pixels are not fully written'.format(self.__class__))
        return self.underlying_array.view('B').reshape(-1)

    def flush(self) -> None:
        self._validate_closed()
        try:
            if self.mode == 'w' and hasattr(self._underlying_array, 'flush'):
                # noinspection PyUnresolvedReferences
                self._underlying_array.flush()
        except AttributeError:
            return

    def close(self) -> None:
        try:
            if self._closed:
                return

            self.flush()
            self._underlying_array = None
            DataSegment.close(self)
        except AttributeError:
            return


class NumpyMemmapSegment(NumpyArraySegment):
    """
    DataSegment based on establishing a numpy memmap, and using that as the
    underlying array.

    Introduced in version 1.3.0.
    """

    __slots__ = (
        '_file_object', '_memory_map', '_close_file')

    def __init__(
            self,
            file_object: Union[str, BinaryIO],
            data_offset: int,
            raw_dtype: Union[str, numpy.dtype],
            raw_shape: Tuple[int, ...],
            formatted_dtype: Optional[Union[str, numpy.dtype]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Union[int, Sequence[int]]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            mode: str = 'r',
            close_file: bool = False):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
        data_offset : int
        raw_dtype : str|numpy.dtype
        raw_shape : Tuple[int, ...]
        formatted_dtype : str|numpy.dtype
        formatted_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
        mode : str
        close_file : bool
        """

        self._close_file = None
        if isinstance(file_object, str):
            close_file = True
        self.close_file = close_file
        self._file_object = file_object

        self._pixels_written = 0
        self._expected_pixels_written = 0

        mmap_mode = 'r' if mode == 'r' else 'r+'
        self._memory_map = numpy.memmap(
            file_object,
            offset=data_offset,
            dtype=raw_dtype,
            shape=raw_shape,
            mode=mmap_mode)

        NumpyArraySegment.__init__(
            self, self._memory_map, formatted_dtype=formatted_dtype, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode=mode)

    @property
    def close_file(self) -> bool:
        """
        bool: Close the file object when complete?
        """

        return self._close_file

    @close_file.setter
    def close_file(self, value):
        self._close_file = bool(value)

    def flush(self) -> None:
        try:
            if self.mode == 'w':
                self._memory_map.flush()
        except AttributeError:
            pass

    def close(self):
        try:
            if self._closed:
                return

            NumpyArraySegment.close(self)  # NB: this calls flush
            self._memory_map = None
            if self._close_file and hasattr(self._file_object, 'close'):
                self._file_object.close()
            self._file_object = None
        except AttributeError:
            return


class HDF5DatasetSegment(DataSegment):
    """
    DataSegment based on reading from an hdf5 file, using the h5py library.

    Introduced in version 1.3.0.
    """
    _allowed_modes = ('r', )

    __slots__ = (
        '_file_object', '_data_set', '_close_file')

    def __init__(
            self,
            file_object: Union[str, h5pyFile],
            data_set: Union[str, h5pyDataset],
            formatted_dtype: Optional[Union[str, numpy.dtype]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Union[int, Sequence[int]]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            close_file: bool = False):
        """

        Parameters
        ----------
        file_object : str|h5py.File
        data_set : str|h5py.Dataset
        formatted_dtype : str|numpy.dtype
        formatted_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
        close_file : bool
        """

        self._close_file = None
        self._file_object = None
        self._data_set = None

        if h5py is None:
            raise ValueError(
                'h5py was not successfully imported, and no hdf5 file can be read')

        if isinstance(file_object, str):
            close_file = True

        self._set_file_object(file_object)
        self._set_data_set(data_set)

        if format_function is not None:
            if formatted_dtype is None or formatted_shape is None:
                raise ValueError(
                    'format_function is supplied, so formatted_dtype and formatted_shape must also be supplied')
        else:
            formatted_dtype = self.data_set.dtype
            raw_shape = self.data_set.shape
            if transpose_axes is None:
                formatted_shape = raw_shape
            else:
                formatted_shape = tuple(raw_shape[entry] for entry in transpose_axes)
        self.close_file = close_file

        DataSegment.__init__(
            self, self.data_set.dtype, self.data_set.shape, formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode='r')

    @property
    def close_file(self) -> bool:
        """
        bool: Close the file object when complete?
        """

        return self._close_file

    @close_file.setter
    def close_file(self, value):
        self._close_file = bool(value)

    @property
    def file_object(self) -> h5pyFile:
        return self._file_object

    def _set_file_object(self, value) -> None:
        if isinstance(value, str):
            value = h5py.File(value, mode='r')
        if not isinstance(value, h5py.File):
            raise ValueError('Requires a path to a hdf5 file or h5py.File object')
        self._file_object = value

    @property
    def data_set(self) -> h5pyDataset:
        return self._data_set

    def _set_data_set(self, value) -> None:
        if isinstance(value, str):
            value = self.file_object[value]
        if not isinstance(value, h5py.Dataset):
            raise ValueError('Requires a dataset path or h5py.Dataset object')
        self._data_set = value

    def read_raw(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:
        self._validate_closed()
        subscript, out_shape = get_subscript_result_size(subscript, self.raw_shape)

        # NB: h5py does not support slicing with a negative step (right now)
        #   we need to read the identical elements in positive order,
        #   then reverse. Note that this is not the same as using the mirror
        #   image slice, which is used in the reverse_axes operations.

        reverse = []
        use_subscript = []
        for index, (the_size, entry) in enumerate(zip(self.raw_shape, subscript)):
            if entry.step < 0:
                use_subscript.append(_reverse_slice(entry))
                reverse.append(index)
            else:
                use_subscript.append(entry)
        use_subscript = tuple(use_subscript)

        out = numpy.reshape(self.data_set[use_subscript], out_shape)
        for index in reverse:
            out = numpy.flip(out, axis=index)

        if squeeze:
            return numpy.squeeze(out)
        else:
            return out

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Sequence[slice]] = None,
            **kwargs):
        raise NotImplementedError

    def get_raw_bytes(self, warn: bool = True) -> Union[bytes, Tuple]:
        raise NotImplementedError

    def check_fully_written(self, warn: bool = False) -> bool:
        return True

    def close(self) -> None:
        try:
            if self._closed:
                return

            self._data_set = None
            if self._close_file and hasattr(self.file_object, 'close'):
                self.file_object.close()
            self._file_object = None
            DataSegment.close(self)
        except AttributeError:
            pass


class FileReadDataSegment(DataSegment):
    """
    Read a data array manually from a file - this is primarily intended for cloud
    usage.

    Introduced in version 1.3.0.
    """
    _allowed_modes = ('r', )

    __slots__ = (
        '_file_object', '_data_offset', '_close_file')

    def __init__(
            self,
            file_object: BinaryIO,
            data_offset: int,
            raw_dtype: Union[str, numpy.dtype],
            raw_shape: Tuple[int, ...],
            formatted_dtype: Union[str, numpy.dtype],
            formatted_shape: Tuple[int, ...],
            reverse_axes: Optional[Union[int, Sequence[int]]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            format_function: Optional[FormatFunction] = None,
            close_file: bool = False):
        """

        Parameters
        ----------
        file_object : BinaryIO
        data_offset : int
        raw_dtype : str|numpy.dtype
        raw_shape : Tuple[int, ...]
        formatted_dtype : str|numpy.dtype
        formatted_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
        close_file : bool
        """

        self._file_object = None
        self._data_offset = None
        self._close_file = None
        self.close_file = close_file
        self._set_data_offset(data_offset)
        self._set_file_object(file_object)
        DataSegment.__init__(
            self, raw_dtype, raw_shape, formatted_dtype, formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode='r')

    @property
    def close_file(self) -> bool:
        """
        bool: Close the file object when complete?
        """

        return self._close_file

    @close_file.setter
    def close_file(self, value):
        self._close_file = bool(value)

    @property
    def file_object(self) -> BinaryIO:
        return self._file_object

    def _set_file_object(self, value) -> None:
        if not is_file_like(value):
            raise ValueError('Requires a file-like object')
        self._file_object = value

    @property
    def data_offset(self) -> int:
        """
        int: The offset of the data in bytes from the start of the file-like
        object.
        """

        return self._data_offset

    def _set_data_offset(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ValueError('data_offset must be non-negative.')
        self._data_offset = value

    def read_raw(
            self,
            subscript: Union[None, int, slice, Sequence[Union[int, slice, Tuple[int, ...]]]],
            squeeze=True) -> numpy.ndarray:
        self._validate_closed()
        subscript, out_shape = get_subscript_result_size(subscript, self.raw_shape)

        init_slice = subscript[0]
        init_reverse = (init_slice.step < 0)
        if init_reverse:
            init_slice = _reverse_slice(init_slice)

        pixel_per_row = 1 if self.formatted_ndim == 1 else int(numpy.prod(self.raw_shape[1:]))
        row_stride = self.raw_dtype.itemsize*pixel_per_row

        start_row = init_slice.start
        rows = init_slice.stop - init_slice.start

        # read the whole contiguous chunk from start_row up to the final row
        # seek to the proper start location
        start_loc = self._data_offset + start_row*row_stride
        self.file_object.seek(start_loc, os.SEEK_SET)
        total_size = rows*row_stride
        # read our data
        data = self.file_object.read(total_size)
        if len(data) != total_size:
            raise ValueError(
                'Tried to read {} bytes of data, but received {}.\n'
                'The most likely reason for this is a malformed chipper, \n'
                'which attempts to read more data than the file contains'.format(total_size, len(data)))
        # define temp array from this data
        data = numpy.frombuffer(data, self._raw_dtype, rows*pixel_per_row)
        data = numpy.reshape(data, (rows, ) + self.raw_shape[1:])
        # extract our data
        out = data[(slice(None, None, init_slice.step), ) + subscript[1:]]
        out = numpy.reshape(out, out_shape)
        if init_reverse:
            out = numpy.flip(out, axis=0)

        if squeeze:
            out = numpy.copy(numpy.squeeze(out))
        else:
            out = numpy.copy(out)
        del data
        return out

    def write_raw(
            self,
            data: numpy.ndarray,
            start_indices: Union[None, int, Tuple[int, ...]] = None,
            subscript: Union[None, Sequence[slice]] = None,
            **kwargs):

        if self.mode != 'w':
            raise ValueError('I/O Error, functionality requires mode == "w"')
        raise NotImplementedError

    def get_raw_bytes(self, warn: bool = True) -> Union[bytes, Tuple]:
        raise NotImplementedError

    def check_fully_written(self, warn: bool = False) -> bool:
        return True

    def close(self) -> None:
        try:
            if self._closed:
                return

            if self._close_file:
                if hasattr(self.file_object, 'close'):
                    self.file_object.close()
            self._file_object = None
            DataSegment.close(self)
        except AttributeError:
            return
