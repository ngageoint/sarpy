"""
The fundamental general objects and methods for reading and presenting data
from files.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
import os
from typing import Union, Tuple, Sequence, Callable, BinaryIO

import numpy

from sarpy.io.general.format_function import FormatFunction, IdentityFunction
from sarpy.io.general.utils import h5py, is_file_like, verify_subscript, \
    result_size

logger = logging.getLogger(__name__)


####
# helper functions

def _find_overlap(slice_in: slice, start_ind: int, stop_ind: int):
    """
    Finds the overlap of the slice with the interval.

    Parameters
    ----------
    slice_in : slice
    start_ind : int
    stop_ind : int

    Returns
    -------
    parent_slice : None|slice
        The overlap expressed as a slice relative to the overall indices
    child_slice : None|slice
        The overlap expressed as a slice relative to the sliced coordinates
    """

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
        return slice(parent_start, parent_stop, slice_in.step), slice(child_start, child_stop, 1)
    else:
        if slice_in.start < start_ind or (slice_in.stop is not None and slice_in.stop >= stop_ind):
            # there is no overlap
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
        return slice(parent_start, parent_stop, slice_in.step), slice(child_start, child_stop, 1)


def _reverse_slice(slice_in: slice) -> slice:
    """
    Given a slice with negative step, this returns a slice which will define the
    same elements traversed in the opposite direction.

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


#####
# Basic data segment definitions and assemblies

class DataSegmentBase(object):
    """
    Partially abstract base class representing one conceptual fragment of data,
    potentially to be used as a component piece to be assembled into a larger
    array of data.

    This is geared somewhat towards images, but is general enough to support
    other usage.
    """

    __slots__ = (
        '_closed', '_raw_dtype', '_raw_shape', '_output_dtype', '_output_shape',
        '_reverse_axes', '_transpose_axes', '_reverse_transpose_axes',
        '_format_function')

    def __init__(self,
                 raw_dtype: Union[str, numpy.dtype],
                 raw_shape: Tuple[int, ...],
                 output_dtype: Union[str, numpy.dtype],
                 output_shape: Tuple[int, ...],
                 reverse_axes: Union[None, int, Sequence[int]]=None,
                 transpose_axes: Union[None, Tuple[int, ...]]=None,
                 format_function: Union[None, str, Callable]=None):
        """

        Parameters
        ----------
        raw_dtype : str|numpy.dtype
        raw_shape : Tuple[int, ...]
        output_dtype : str|numpy.dtype
        output_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int, ...]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
        """

        self._closed = False
        self._raw_shape = None
        self._set_raw_shape(raw_shape)

        self._raw_dtype = None
        self._set_raw_dtype(raw_dtype)

        self._output_shape = None
        self._set_output_shape(output_shape)

        self._output_dtype = None
        self._set_output_dtype(output_dtype)

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
    def output_shape(self) -> Tuple[int, ...]:
        """
        Tuple[int, ...]: The output shape. For multi-band images, it is the tacit
        assumption that the data has been reorganized so that the band is
        in the final dimension.
        """

        return self._output_shape

    def _set_output_shape(self, value: Tuple[int, ...]) -> None:
        if not isinstance(value, tuple):
            raise TypeError(
                'output_shape must be specified by a tuple of ints, got type `{}`'.format(type(value)))
        for entry in value:
            if not isinstance(entry, int):
                raise TypeError(
                    'output_shape must be specified by a tuple of ints, got `{}`'.format(value))
            if entry <= 0:
                raise ValueError(
                    'output_shape must be specified by a tuple of positive ints, got `{}`'.format(value))
        self._output_shape = value

    @property
    def output_dtype(self) -> numpy.dtype:
        """
        numpy.dtype: The data type of the data returned by the :func:`read` function.
        """

        return self._output_dtype

    def _set_output_dtype(self, value) -> None:
        if not isinstance(value, numpy.dtype):
            try:
                value = numpy.dtype(value)
            except Exception as e:
                raise ValueError(
                    'Tried interpreting output_dtype value as a numpy.dtype, '
                    'and failed with error\n\t{}'.format(e))
        self._output_dtype = value

    @property
    def ndim(self) -> int:
        """
        int: The number of output dimensions.
        """

        return len(self._output_shape)

    @property
    def reverse_axes(self) -> Union[None, Tuple[int, ...]]:
        """
        None|Tuple[int, ...]: The collection of axes (with respect to raw order)
        along which we will reverse as part of transformation to output data order.
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
                raise ValueError('reverse_axes entries must be less than ndim')

        self._reverse_axes = value

    @property
    def transpose_axes(self) -> Tuple[int, ...]:
        """
        None|Tuple[int, ...]: The transpose order for switching from raw order to
        output order, prior to applying any format function.

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

    def _set_format_function(self, value: Union[None, FormatFunction]) -> None:
        if value is None:
            value = IdentityFunction()
        if not isinstance(value, FormatFunction):
            raise ValueError('Got unexpected input for format_function of type `{}`'.format(type(value)))

        # set our important property values
        value.set_input_shape(self.raw_shape)
        value.set_output_shape(self.output_shape)
        value.set_reverse_axes(self.reverse_axes)
        value.set_transpose_axes(self.transpose_axes)
        self._format_function = value

    def _validate_shapes(self) -> None:
        """
        Do our best at validating the raw_shape and output_shape.
        """

        self.format_function.validate_shapes()

    def _interpret_subscript(self, subscript: Union[None, int, slice, Tuple[slice, ...]], raw=False) -> Tuple[slice, ...]:
        """
        Restructures the input to be a tuple of slices guaranteed to be the same
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
            return verify_subscript(subscript, self._output_shape)

    def read(self, subscript: Union[int, slice, Tuple[slice, ...]], squeeze=True) -> numpy.ndarray:
        """
        Read the data slice specified relative to the output data coordinates.

        .. warning::
            Attempting to slice on bands that are modified by format_function will
            likely fail, and probably not gracefully.

        Parameters
        ----------
        subscript : int|slice|tuple
        squeeze : bool
            Apply the numpy.squeeze operation, which eliminates dimension of size 1?

        Returns
        -------
        numpy.ndarray
        """

        norm_subscript = self._interpret_subscript(subscript, raw=False)
        raw_subscript = self.format_function.transform_slice(norm_subscript)
        raw_data = self.read_raw(raw_subscript, squeeze=False)
        return self.format_function(raw_data, squeeze=squeeze)

    def __getitem__(self, subscript):
        """
        Fetch the data via slice definition.

        Parameters
        ----------
        subscript : int|slice|tuple

        Returns
        -------
        numpy.ndarray
        """

        if isinstance(subscript, tuple) and isinstance(subscript[-1], dict):
            kwargs = subscript[-1]
            subscript = subscript[:-1]
        else:
            kwargs = {}
        if kwargs.get('raw', False):
            return self.read(subscript)
        else:
            return self.read_raw(subscript, squeeze=kwargs.get('squeeze', True))

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

    def read_raw(self, subscript: Union[slice, Tuple[slice, ...]], squeeze=True) -> numpy.ndarray:
        """
        Read raw data from the source, without reformatting and or applying
        symmetry operations.

        Parameters
        ----------
        subscript : slice|Tuple[slice, ...]
            These arguments are relative to raw data shape and order, no symmetry
            operations have been applied. These slice arguments are expected to
            be fully defined.
        squeeze : bool
            Apply numpy.squeeze, which eliminates any dimensions of size 1?

        Returns
        -------
        numpy.ndarray
            This will be of data type given by `raw_dtype`.
        """

        raise NotImplementedError


class ReorientationSegment(DataSegmentBase):
    """
    Define a basic ordering of a given DataSegmentBase. The raw data will be
    presented as the parent data segments output data.
    """

    __slots__ = ('_parent', '_close_parent')

    def __init__(self,
                 parent: DataSegmentBase,
                 output_dtype: Union[None, str, numpy.dtype]=None,
                 output_shape: Union[None, Tuple[int, ...]]=None,
                 reverse_axes: Union[None, int, Sequence[int]]=None,
                 transpose_axes: Union[None, Tuple[int, ...]]=None,
                 format_function: Union[None, str, Callable]=None,
                 close_parent: bool=True):
        """
        Parameters
        ----------
        parent : DataSegmentBase
        output_dtype : str|numpy.dtype
        output_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int, ...]
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
            output_dtype = parent.output_dtype
            output_shape = intermediate_shape
        else:
            if output_dtype is None or output_shape is None:
                raise ValueError(
                    'If format_function is provided,\n\t'
                    'then output_dtype and output_shape must be provided.')

        DataSegmentBase.__init__(
            self, parent.output_dtype, parent.output_shape, output_dtype, output_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes, format_function=format_function)

    @property
    def parent(self) -> DataSegmentBase:
        return self._parent

    def _set_parent(self,
                    parent: DataSegmentBase,
                    transpose_axes: Union[None, Tuple[int, ...]]) -> Tuple[int, ...]:
        if transpose_axes is None:
            trans_axes = tuple(range(parent.ndim))
        else:
            if len(transpose_axes) != parent.ndim:
                raise ValueError('transpose_axes must have length {}'.format(parent.ndim))
            trans_axes = transpose_axes
        self._parent = parent
        return tuple([parent.output_shape[index] for index in trans_axes])

    @property
    def close_parent(self) -> bool:
        """
        bool: Call parent.close() when close is called?
        """

        return self._close_parent

    @close_parent.setter
    def close_parent(self, value):
        self._close_parent = bool(value)

    def read_raw(self, subscript: Union[slice, Tuple[slice, ...]], squeeze=True) -> numpy.ndarray:
        return self.parent.read(subscript, squeeze=squeeze)

    def close(self):
        if not (getattr(self, 'close_parent', True) is False):
            if hasattr(self.parent, 'close'):
                self.parent.close()
        DataSegmentBase.close(self)


class SubsetSegment(DataSegmentBase):
    """
    Define a subset of a given DataSegmentBase.
    """

    __slots__ = (
        '_parent', '_subset_definition', '_raw_subset_definition',
        '_original_output_indices', '_original_raw_indices',
        '_close_parent')

    def __init__(self,
                 parent: DataSegmentBase,
                 subset_definition: Tuple[slice, ...],
                 close_parent: bool=True):
        """
        Parameters
        ----------
        parent : DataSegmentBase
        subset_definition : tuple
        close_parent : bool
            Call parent.close() when close is called?
        """

        self._close_parent = None
        self.close_parent = close_parent
        self._original_output_indices = None  # the original indices matched to the new, with entry -1 when flat
        self._original_raw_indices = None
        self._subset_definition = None
        self._raw_subset_definition = None
        self._parent = parent
        raw_shape, output_shape = self._validate_subset_definition(subset_definition)
        DataSegmentBase.__init__(
            self, parent.raw_dtype, raw_shape, parent.output_dtype, output_shape)

    @property
    def parent(self) -> DataSegmentBase:
        return self._parent

    @property
    def subset_definition(self) -> Tuple[slice]:
        """
        Tuple[slice]: Gets the subset definition, in output coordinates.
        """

        return self._subset_definition

    @property
    def close_parent(self) -> bool:
        """
        bool: Call parent.close() when close is called?
        """

        return self._close_parent

    @close_parent.setter
    def close_parent(self, value):
        self._close_parent = bool(value)

    def _validate_subset_definition(self, subset_definition):
        """
        Validates the subset definition.

        Parameters
        ----------
        subset_definition : tuple

        Returns
        -------
        raw_shape : tuple
        output_shape : tuple
        """

        sub_def = []
        raw_shape = []
        output_shape = []
        original_indices = []
        raw_indices = []

        for index, entry in enumerate(self.parent._interpret_subscript(subset_definition)):
            if entry.step is None or entry.step < 0:
                raise ValueError(
                    'Entry {} at index {} of subset definition does not have slice with\n\t'
                    'positive step defined'.format(entry, index))

            siz = self.parent.output_shape[index]
            start = 0 if entry.start is None else entry.start
            stop = siz if entry.stop is None else entry.stop
            step = entry.step
            this_slice = slice(start, stop, step)
            test_array = numpy.arange(siz)[this_slice]
            if test_array.size == 0:
                raise ValueError('Entry at index {} of subset definition yields empty result'.format(index))
            elif test_array.size == 1:
                logger.info('Entry at index {} of subset definition yields a single entry'.format(index))
                original_indices.append(-1)
            else:
                output_shape.append(test_array.size)
                original_indices.append(index)
            sub_def.append(this_slice)

        self._subset_definition = tuple(sub_def)
        self._original_output_indices = tuple(original_indices)

        self._raw_subset_definition = self.parent.format_function.transform_slice(self._subset_definition)
        for index, siz in enumerate(self.parent.raw_shape):
            test_array = numpy.arange(siz)[self._raw_subset_definition[index]]
            if test_array.size == 0:
                raise ValueError('Raw slice at index {} yield empty result'.format(index))
            elif test_array.size == 1:
                logger.info('Raw slice at index {} of subset definition yields a single entry'.format(index))
                raw_indices.append(-1)
            else:
                raw_shape.append(test_array.size)
                raw_indices.append(index)
        self._original_raw_indices = tuple(raw_indices)
        return tuple(raw_shape), tuple(output_shape)

    def _interpret_subscript(self, subscript: Union[None, int, tuple, slice], raw=False) -> tuple:
        norm_subscript = DataSegmentBase._interpret_subscript(self, subscript, raw=raw)
        out = []
        if raw:
            use_inds = self._original_raw_indices
            use_def = self._raw_subset_definition
            full_shapes = self.parent.raw_shape
        else:
            use_inds = self._original_output_indices
            use_def = self._subset_definition
            full_shapes = self.parent.output_shape

        for full_size, out_index, slice_def in zip(full_shapes, use_inds, use_def):
            if out_index == -1:
                out.append(slice_def)
            else:
                part_def = norm_subscript[out_index]
                step = part_def.step*slice_def.step
                # now, extract start and stop
                # the logic of this is terrible, so let's just do it the easy way
                the_array = numpy.arange(full_size)[slice_def][part_def]
                start = the_array[0]
                stop = the_array[-1] + step
                if stop < 0:
                    stop = None
                out.append(slice(start, stop, step))
        return tuple(out)

    def read_raw(self, subscript: Union[slice, tuple], squeeze=True) -> numpy.ndarray:
        norm_subscript = self._interpret_subscript(subscript, raw=True)
        if squeeze:
            return self.parent.read_raw(norm_subscript, squeeze=True)
        else:
            data = self.parent.read_raw(norm_subscript, squeeze=False)
            use_shape = []
            for check, size in zip(self._original_raw_indices, data.shape):
                if check != -1:
                    use_shape.append(size)
            return numpy.reshape(data, tuple(use_shape))

    def read(self, subscript: Union[int, tuple, slice, numpy.ndarray], squeeze=True) -> numpy.ndarray:
        norm_subscript = self._interpret_subscript(subscript, raw=False)
        if squeeze:
            return self.parent.read(norm_subscript, squeeze=True)
        else:
            data = self.parent.read(norm_subscript, squeeze=False)
            use_shape = []
            for check, size in zip(self._original_output_indices, data.shape):
                if check != -1:
                    use_shape.append(size)
            return numpy.reshape(data, tuple(use_shape))

    def close(self):
        if not (getattr(self, 'close_parent', True) is False):
            if hasattr(self.parent, 'close'):
                self.parent.close()
        DataSegmentBase.close(self)


class BandAggregateSegment(DataSegmentBase):
    """
    This stacks a collection of data segments, which must have compatible details,
    together along a new (final) band dimension.

    Note that :func:`read` and :func:`read_raw` return identical results here.
    To access raw data from the children, use access on the `children` property.
    """

    __slots__ = ('_children', '_close_children')

    def __init__(self,
                 children: Sequence[DataSegmentBase],
                 output_dtype: Union[None, str, numpy.dtype]=None,
                 output_shape: Union[None, Tuple[int, ...]]=None,
                 format_function: Union[None, str, Callable]=None,
                 close_children: bool=True):
        """

        Parameters
        ----------
        children : Sequence[DataSegmentBase]
        output_dtype : str|numpy.dtype
        output_shape : Tuple[int, ...]
        format_function : None|FormatFunction
        close_children : bool
        """

        self._close_children = None
        self.close_children = close_children
        self._children = None
        raw_dtype, raw_shape = self._set_children(children)

        if format_function is None:
            output_dtype = raw_dtype
            output_shape = raw_shape
        else:
            if output_dtype is None or output_shape is None:
                raise ValueError(
                    'If format_function is provided,\n\t'
                    'then output_dtype and output_shape must be provided.')
        DataSegmentBase.__init__(self, raw_dtype, raw_shape, output_dtype,
                                 output_shape, format_function=format_function)

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
    def children(self) -> Tuple[DataSegmentBase, ...]:
        """
        The collection of children that we are stacking.

        Returns
        -------
        Tuple[DataSegmentBase, ...]
        """

        return self._children

    def _set_children(self, children: Sequence[DataSegmentBase]) -> (numpy.dtype, Tuple[int, ...]):
        if len(children) < 2:
            raise ValueError('Cannot define a BandAggregateSegment based on fewer than 2 segments.')

        child_shape = children[0].output_shape
        the_shape = child_shape + (len(children), )
        the_dtype = children[0].output_dtype

        use_children = []
        for child in children:
            if child.output_shape != child_shape:
                raise ValueError('All children must have the same output shape')
            if child.output_dtype != the_dtype:
                raise ValueError('All children must have the same output dtype')
            use_children.append(child)
        self._children = tuple(use_children)
        return the_dtype, the_shape

    @property
    def bands(self) -> int:
        """
        int: The number of bands (child data segments)
        """

        return len(self.children)

    def read_raw(self, subscript: Union[slice, tuple], squeeze=True) -> numpy.ndarray:
        norm_subscript, the_shape = result_size(subscript, self.raw_shape)
        out = numpy.empty(the_shape, dtype=self.raw_dtype)
        for out_index, index in enumerate(numpy.arange(self.bands)[norm_subscript[-1]]):
            out[..., out_index] = self.children[index].read(subscript, squeeze=False)
        if squeeze:
            return numpy.squeeze(out)
        else:
            return out

    def close(self):
        if not (getattr(self, 'close_children', True) is False):
            if self._children is not None:
                for entry in self._children:
                    if hasattr(entry, 'close'):
                        entry.close()
        DataSegmentBase.close(self)


class BlockAggregateSegment(DataSegmentBase):
    """
    Combines a collection of compatible data segments, according to a
    two-dimensional block definition applying to the first two-dimensions.

    All children must be identical for any dimensions beyond the first two,
    and have the same output_dtype.
    """

    __slots__ = (
        '_children', '_child_arrangement', '_missing_data_value',
        '_close_children')

    def __init__(self,
                 children: Sequence[DataSegmentBase],
                 child_arrangement: numpy.ndarray,
                 missing_data_value,
                 close_children: bool = True):
        """

        Parameters
        ----------
        children : Sequence[DataSegmentBase]
        child_arrangement : numpy.ndarray
            Two-dimensional array of `[[row start, row end, column start, column end]]`,
            where `row` indicates the first dimension and `column` indicates the
            second dimension. Any holes in the definition will have data reading
            augmented by `missing_data_value` in the holes. Overlap in definition
            is permitted.
        missing_data_value
            Missing data value, which must be compatible with child.output_dtype.
        close_children : bool
        """

        self.close_children = close_children
        self._close_children = None

        self._children = None
        self._child_arrangement = None
        self._missing_data_value = missing_data_value

        the_dtype, the_shape = self._set_children(children, child_arrangement)
        DataSegmentBase.__init__(self, the_dtype, the_shape, the_dtype, the_shape)

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
    def children(self) -> Tuple[DataSegmentBase, ...]:
        """
        The collection of children that we are stacking.

        Returns
        -------
        Tuple[DataSegmentBase, ...]
        """

        return self._children

    def _set_children(self,
                      children: Sequence[DataSegmentBase],
                      child_arrangement: numpy.ndarray) -> (numpy.dtype, Tuple[int, ...]):
        if child_arrangement.ndim != 2 or \
                child_arrangement.shape[1] != 4 or \
                child_arrangement.shape[0] != len(children):
            raise ValueError('Mismatch in child collection and arrangement array')

        init_child = children[0]
        first_child_shape = init_child.output_shape
        raw_dtype = init_child.output_dtype
        raw_shape_init = [0, 0]
        my_children = []
        my_arrangement = []

        if init_child.ndim < 2:
            raise ValueError('Each child must have ndim >= 2')
        else:
            terminal_shape = first_child_shape[2:]

        for index, child in enumerate(children):
            if raw_dtype != child.output_dtype:
                raise ValueError(
                    'Require all children to have identical output_dtype,\n\t'
                    'got {} and {}'.format(raw_dtype, child.output_dtype))
            if child.ndim < 2:
                raise ValueError('Each child must have ndim >= 2')

            child_shape = init_child.output_shape
            if child_shape[2:] != terminal_shape:
                raise ValueError('Incompatible children shapes `{}` and `{}`'.format(child_shape, first_child_shape))
            arrangement = child_arrangement[index, :]
            row_start = int(arrangement[0])
            row_end = int(arrangement[1])
            col_start = int(arrangement[2])
            col_end = int(arrangement[3])
            if not 0 <= row_start < row_end or 0 <= col_start < col_end:
                raise ValueError('arrangement definition has invalid entry {}'.format(arrangement))
            if not ((row_end - row_start) == child_shape[0] and (col_end - col_start) == child_shape[1]):
                raise ValueError(
                    'arrangement definition entry `{}`\n\t'
                    'incompatible with child shape `{}`'.format(arrangement, child_shape))

            raw_shape_init[0] = max(raw_shape_init[0], row_end)
            raw_shape_init[1] = max(raw_shape_init[1], col_end)
            my_children.append(child)
            my_arrangement.append([row_start, row_end, col_start, col_end])

        self._children = tuple(my_children)
        self._child_arrangement = numpy.array(my_arrangement, dtype='int32')
        return raw_dtype, tuple(raw_shape_init) + terminal_shape

    def read_raw(self, subscript: Union[slice, tuple], squeeze=True) -> numpy.ndarray:
        subscript, output_shape = result_size(subscript, self.raw_shape)
        out = numpy.full(output_shape, fill_value=self._missing_data_value, dtype=self.raw_dtype)

        for entry, child in zip(self._child_arrangement, self._children):
            row_start, row_end, col_start, col_end = entry
            row_slice, child_row_slice = _find_overlap(subscript[0], row_start, row_end)
            col_slice, child_col_slice = _find_overlap(subscript[1], col_start, col_end)

            parent_subscript = (row_slice, col_slice) + subscript[2:]
            child_subscript = (child_row_slice, child_col_slice) + subscript[2:]
            out[parent_subscript] = child.read_raw(child_subscript, squeeze=False)
        if squeeze:
            return numpy.squeeze(out)
        else:
            return out

    def close(self):
        if not (getattr(self, 'close_children', True) is False):
            if self._children is not None:
                for entry in self._children:
                    if hasattr(entry, 'close'):
                        entry.close()
        DataSegmentBase.close(self)


####
# Concrete reading implementations

class NumpyArraySegment(DataSegmentBase):
    """
    DataSegment based on reading from a numpy.ndarray
    """

    __slots__ = ('_underlying_array', )

    def __init__(self,
                 underlying_array : numpy.ndarray,
                 output_dtype: Union[str, numpy.dtype],
                 output_shape: Tuple[int, ...],
                 reverse_axes: Union[None, int, Sequence[int]]=None,
                 transpose_axes: Union[None, Tuple[int, ...]]=None,
                 format_function: Union[None, str, Callable]=None):
        """

        Parameters
        ----------
        underlying_array : numpy.ndarray
        output_dtype : str|numpy.dtype
        output_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int, ...]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
        """

        if not isinstance(underlying_array, numpy.ndarray):
            raise TypeError(
                'underlying array must be a numpy.ndarray, got type `{}`'.format(
                    type(underlying_array)))
        self._underlying_array = underlying_array
        DataSegmentBase.__init__(self, underlying_array.dtype, underlying_array.shape,
                                 output_dtype, output_shape,
                                 reverse_axes=reverse_axes, transpose_axes=transpose_axes, format_function=format_function)

    @property
    def underlying_array(self) -> numpy.ndarray:
        """
        The underlying data array.

        Returns
        -------
        numpy.ndarray
        """

        return self._underlying_array

    def read_raw(self, subscript: Union[slice, Tuple[slice, ...]], squeeze=True) -> numpy.ndarray:
        subscript, out_shape = result_size(subscript, self.raw_shape)
        out = self._underlying_array[subscript]  # squeezed by default

        if squeeze:
            return out
        else:
            return numpy.reshape(out, out_shape)

    def close(self):
        self._underlying_array = None
        DataSegmentBase.close(self)


class NumpyMemmapSegment(NumpyArraySegment):
    """
    DataSegment based on establishing a numpy memmap, and using that as the
    underlying array.
    """

    __slots__ = (
        '_file_object', '_close_file')

    def __init__(self,
                 file_object: Union[str, BinaryIO],
                 data_offset: int,
                 raw_dtype: Union[str, numpy.dtype],
                 raw_shape: Tuple[int, ...],
                 output_dtype: Union[str, numpy.dtype],
                 output_shape: Tuple[int, ...],
                 reverse_axes: Union[None, int, Sequence[int]]=None,
                 transpose_axes: Union[None, Tuple[int, ...]]=None,
                 format_function: Union[None, str, Callable]=None,
                 close_file: bool=False):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
        data_offset : int
        raw_dtype : str|numpy.dtype
        raw_shape : Tuple[int, ...]
        output_dtype : str|numpy.dtype
        output_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int, ...]
            The collection of axes (in raw order) to reverse, prior to applying
            transpose operation
        transpose_axes : None|Tuple[int, ...]
            The transpose operation to perform to the raw data, after applying
            any axis reversal, and before applying any format function
        format_function : None|FormatFunction
        close_file : bool
        """

        self._close_file = None
        self.close_file = close_file

        memory_map = numpy.memmap(file_object,
                                  dtype=raw_dtype,
                                  mode='r',
                                  offset=data_offset,
                                  shape=raw_shape)

        NumpyArraySegment.__init__(self,
                                   memory_map, output_dtype, output_shape,
                                   reverse_axes=reverse_axes, transpose_axes=transpose_axes,
                                   format_function=format_function)

    @property
    def close_file(self) -> bool:
        """
        bool: Close the file object when complete?
        """

        return self._close_file

    @close_file.setter
    def close_file(self, value):
        self._close_file = bool(value)

    def close(self):
        NumpyArraySegment.close(self)
        if self._close_file:
            if self._file_object is not None and \
                    hasattr(self._file_object, 'closed') and \
                    not self._file_object.closed:
                self._file_object.close()


class HDF5Segment(DataSegmentBase):
    """
    DataSegment based on reading from an hdf5 file, using the h5py library
    """

    __slots__ = (
        '_file_object', '_data_set', '_close_file')

    def __init__(self,
                 file_object: Union[str, h5py.File],
                 data_set: Union[str, h5py.Dataset],
                 output_dtype: Union[str, numpy.dtype],
                 output_shape: Tuple[int, ...],
                 reverse_axes: Union[None, int, Sequence[int]]=None,
                 transpose_axes: Union[None, Tuple[int, ...]]=None,
                 format_function: Union[None, str, Callable]=None,
                 close_file: bool=False):
        """

        Parameters
        ----------
        file_object : str|h5py.File
        data_set : str|h5py.Dataset
        output_dtype : str|numpy.dtype
        output_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int, ...]
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

        self._set_file_object(file_object)
        self._set_data_set(data_set)

        self.close_file = close_file

        DataSegmentBase.__init__(self, self.data_set.dtype, self.data_set.shape,
                                 output_dtype, output_shape,
                                 reverse_axes=reverse_axes, transpose_axes=transpose_axes,
                                 format_function=format_function)

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
    def file_object(self) -> h5py.File:
        return self._file_object

    def _set_file_object(self, value):
        if isinstance(value, str):
            value = h5py.File(value, mode='r')
        if not isinstance(value, h5py.File):
            raise ValueError('Requires a path to a hdf5 file or h5py.File object')
        self._file_object = value

    @property
    def data_set(self) -> h5py.Dataset:
        return self._data_set

    def _set_data_set(self, value):
        if isinstance(value, str):
            value = self.file_object[value]
        if not isinstance(value, h5py.Dataset):
            raise ValueError('Requires a dataset path or h5py.Dataset object')
        self._data_set = value

    def close(self):
        self._data_set = None
        if self._close_file:
            if hasattr(self.file_object, 'close'):
                self.file_object.close()
        self._file_object = None
        DataSegmentBase.close(self)

    def read_raw(self, subscript: Union[slice, Tuple[slice, ...]], squeeze=True) -> numpy.ndarray:
        subscript, out_shape = result_size(subscript, self.raw_shape)

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


class FileReadDataSegment(DataSegmentBase):
    """
    Read a data array manually from a file - this is really only for cloud usage.
    """

    __slots__ = (
        '_file_object', '_data_offset', '_close_file')

    def __init__(self,
                 file_object: BinaryIO,
                 data_offset : int,
                 raw_dtype: Union[str, numpy.dtype],
                 raw_shape: Tuple[int, ...],
                 output_dtype: Union[str, numpy.dtype],
                 output_shape: Tuple[int, ...],
                 reverse_axes: Union[None, int, Sequence[int]]=None,
                 transpose_axes: Union[None, Tuple[int, ...]]=None,
                 format_function: Union[None, str, Callable]=None,
                 close_file: bool=False):
        """

        Parameters
        ----------
        file_object : BinaryIO
        data_offset : int
        raw_dtype : str|numpy.dtype
        raw_shape : Tuple[int, ...]
        output_dtype : str|numpy.dtype
        output_shape : Tuple[int, ...]
        reverse_axes : None|int|Sequence[int, ...]
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
        DataSegmentBase.__init__(self, raw_dtype, raw_shape,
                                 output_dtype, output_shape,
                                 reverse_axes=reverse_axes, transpose_axes=transpose_axes,
                                 format_function=format_function)

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

    def _set_file_object(self, value):
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

    def close(self):
        if self._close_file:
            if hasattr(self.file_object, 'close'):
                self.file_object.close()
        self._file_object = None
        DataSegmentBase.close(self)

    def read_raw(self, subscript: Union[slice, Tuple[slice, ...]], squeeze=True) -> numpy.ndarray:
        subscript, out_shape = result_size(subscript, self.raw_shape)

        init_slice = subscript[0]
        init_reverse = (init_slice.step < 0)
        if init_reverse:
            init_slice = _reverse_slice(init_slice)

        pixel_per_row = 1 if self.ndim == 1 else int(numpy.prod(self.raw_shape[1:]))
        row_stride = self.raw_dtype.itemsize*pixel_per_row

        start_row = init_slice.start
        rows = out_shape[0]

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
        out = data[(init_slice, ) + subscript[1:]]
        out = numpy.reshape(out, out_shape)
        if init_reverse:
            out = numpy.flip(out, axis=0)

        if squeeze:
            out = numpy.copy(numpy.squeeze(out))
        else:
            out = numpy.copy(out)
        del data
        return out
