"""
The general base elements for reading and writing image files.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import io
import os
import logging
from typing import Union, Tuple, BinaryIO, Sequence, Optional
from importlib import import_module
import pkgutil

import numpy

from sarpy.compliance import SarpyError
from sarpy.io.general.utils import validate_range, reverse_range, is_file_like

logger = logging.getLogger(__name__)

##################
# module variables
_SUPPORTED_TRANSFORM_VALUES = ('COMPLEX', )
READER_TYPES = ('SICD', 'SIDD', 'CPHD', 'CRSD', 'OTHER')


class SarpyIOError(SarpyError):
    """A custom exception class for discovered input/output errors."""


#################
# Chipper definitions - this is base functionality for the most basic reading

def validate_transform_data(transform_data):
    """
    Validate the transform_data value.

    Parameters
    ----------
    transform_data : None|str|Callable

    Returns
    -------
    None|str|Callable
    """

    if transform_data is None or callable(transform_data):
        return transform_data
    elif isinstance(transform_data, str):
        transform_data = transform_data.upper()
        if transform_data not in _SUPPORTED_TRANSFORM_VALUES:
            raise ValueError('transform_data is string {}, which is not supported'.format(transform_data))
        return transform_data
    else:
        raise ValueError('transform_data must be None, a string, or callable')


class BaseChipper(object):
    """
    Base class defining basic functionality for the literal extraction of data
    from a file. The intent of this class is to be a callable in the following form:
    :code:`data = BaseChipper(entry1, entry2)`, where each entry is a tuple or
    int of the form `[[[start], stop,] step]`.

    Similarly, we are able to use more traditional Python slicing syntax

    .. code-block:: python

        data = BaseChipper[slice1[, slice2]]

    **Extension Requirement:** This provides the basic implementation for the work
    flow, but it is **required** that any extension provide a concrete implementation
    for actually reading from the raw file in `read_raw_fun`.

    **Extension Consideration:** It is possible that the basic functionality for
    conversion of raw data to complex data requires something more nuanced than
    the default provided in the `_transform_data_method` method.
    """

    __slots__ = ('_data_size', '_transform_data', '_symmetry')

    def __init__(self, data_size, symmetry=(False, False, False), transform_data=False):
        """

        Parameters
        ----------
        data_size : tuple
            The shape of the raw data (i.e. in the file).
        symmetry : tuple
            Describes any required data transformation. See the `symmetry` property.
        transform_data : Callable|str|None
            For data transformation after reading.
            If `None`, then no transformation will be applied. If `callable`, then
            this is expected to be the transformation method for the raw data. If
            string valued and `'complex'`, then the assumption is that real/imaginary
            components are stored in adjacent bands, which will be combined into a
            single band upon extraction. Other situations will yield and value error.
        """

        self._transform_data = validate_transform_data(transform_data)

        if not isinstance(symmetry, tuple):
            symmetry = tuple(symmetry)
        if len(symmetry) != 3:
            raise ValueError(
                'The symmetry parameter must have length 3, and got {}.'.format(symmetry))
        self._symmetry = tuple([bool(entry) for entry in symmetry])

        if not isinstance(data_size, tuple):
            data_size = tuple(data_size)
        if len(data_size) != 2:
            raise ValueError(
                'The data_size parameter must have length 2, and got {}.'.format(data_size))
        data_size = (int(data_size[0]), int(data_size[1]))
        if data_size[0] < 0 or data_size[1] < 0:
            raise ValueError('All entries of data_size {} must be non-negative.'.format(data_size))
        if self._symmetry[2]:
            self._data_size = (data_size[1], data_size[0])
        else:
            self._data_size = data_size

    @property
    def symmetry(self):
        """
        Tuple[bool, bool, bool]: Entries of the form (`flip1`, `flip2`, `swap_axes`).
        This describes necessary symmetry transformation to be performed to convert
        from raw (file storage) order into the order expected (analysis order).

        * `flip1=True` - we reverse order in the first axis, with respect to the raw order.

        * `flip2=True` - we reverse order in the second axis, with respect to the raw order.

        * `swap_axes=True` - we switch the two axes, after any required flipping.
        """

        return self._symmetry

    @property
    def data_size(self):
        """
        Tuple[int, int]: Two element tuple of the form `(rows, columns)`, which provides the
        size of the data, after any necessary symmetry transformations have been applied.
        Note that this excludes the number of bands in the image.
        """

        return self._data_size

    def __call__(self, range1, range2):
        """
        Reads and fetches data. Note that :code:`chipper(range1, range2)` is an alias
        for :code:`chipper.read_chip(range1, range2)`.

        Parameters
        ----------
        range1 : None|int|tuple
        range2 : none|int|tuple

        Returns
        -------
        numpy.ndarray
        """

        data = self._read_raw_fun(range1, range2)
        data = self._transform_data_method(data)

        # make a one band image flat
        if data.ndim == 3 and data.shape[2] == 1:
            data = numpy.reshape(data, data.shape[:-1])

        data = self._reorder_data(data)
        return data

    def __getitem__(self, item):
        """
        Reads and returns data using more traditional to python slice functionality.
        After slice interpretation, this is analogous to :func:`__call__` or :func:`read_chip`.

        Parameters
        ----------
        item : None|int|slice|tuple

        Returns
        -------
        numpy.ndarray
        """

        range1, range2 = self._slice_to_args(item)
        return self.__call__(range1, range2)

    @staticmethod
    def _slice_to_args(item):
        # type: (Union[None, int, slice, tuple]) -> tuple
        def parse(entry):
            if isinstance(entry, int):
                return entry, entry+1, 1
            if isinstance(entry, slice):
                return entry.start, entry.stop, entry.step

        # this input is assumed to come from slice parsing
        if isinstance(item, tuple) and len(item) > 2:
            raise ValueError(
                'Chipper received slice argument {}. We cannot slice on more than two dimensions.'.format(item))
        if isinstance(item, tuple):
            return parse(item[0]), parse(item[1])
        else:
            return parse(item), None

    def _reorder_arguments(self, range1, range2):
        """
        Reinterpret the range arguments into actual "physical" arguments of memory,
        in light of the symmetry attribute.

        Parameters
        ----------
        range1 : None|int|tuple
            * if `None`, then the range is not limited in first axis
            * if `int` = start
            * if (`int`, `int`) = `start`, `stop`
            * if (`int`, `int`, `int`) = `start`, `stop`, `step size`
        range2 : None|int|tuple
            same as `range1`, except for the second axis.

        Returns
        -------
        None|int|tuple
            actual range 1 - in light of `range1`, `range2` and symmetry
        None|int|tuple
            actual range 2 - in light of `range1`, `range2` and symmetry
        """

        if isinstance(range1, (numpy.ndarray, list)):
            range1 = tuple(int(el) for el in range1)
        if isinstance(range2, (numpy.ndarray, list)):
            range2 = tuple(int(el) for el in range2)

        if not (range1 is None or isinstance(range1, int) or isinstance(range1, tuple)):
            raise TypeError('range1 is of type {}, but must be an instance of None, '
                            'int or tuple.'.format(range1))
        if isinstance(range1, tuple) and len(range1) > 3:
            raise TypeError('range1 must have no more than 3 entries, received {}.'.format(range1))

        if not (range2 is None or isinstance(range2, int) or isinstance(range2, tuple)):
            raise TypeError('range2 is of type {}, but must be an instance of None, '
                            'int or tuple.'.format(range2))
        if isinstance(range2, tuple) and len(range2) > 3:
            raise TypeError('range2 must have no more than 3 entries, received {}.'.format(range2))

        # switch the axes symmetry dictates
        if self._symmetry[2]:
            range1, range2 = range2, range1
            lim1, lim2 = self._data_size[1], self._data_size[0]
        else:
            lim1, lim2 = self._data_size[0], self._data_size[1]

        # validate the first range
        if self._symmetry[0]:
            real_arg1 = reverse_range(range1, lim1)
        else:
            real_arg1 = validate_range(range1, lim1)
        # validate the second range
        if self._symmetry[1]:
            real_arg2 = reverse_range(range2, lim2)
        else:
            real_arg2 = validate_range(range2, lim2)

        return real_arg1, real_arg2

    def _transform_data_method(self, data):
        # type: (numpy.ndarray) -> numpy.ndarray
        if self._transform_data is None:
            # nothing to be done
            return data
        elif callable(self._transform_data):
            return self._transform_data(data)
        elif isinstance(self._transform_data, str):
            if self._transform_data == 'COMPLEX':
                if numpy.iscomplexobj(data):
                    return data
                out = numpy.zeros((data.shape[0], data.shape[1], int(data.shape[2]/2)), dtype=numpy.complex64)
                out.real = data[:, :, 0::2]
                out.imag = data[:, :, 1::2]
                return out
            else:
                raise ValueError('Unsupported transform_data value `{}`'.format(self._transform_data))
        raise ValueError('Unsupported transform_data value {}'.format(self._transform_data))

    def _reorder_data(self, data):
        # type: (numpy.ndarray) -> numpy.ndarray
        if self._symmetry[2]:
            return numpy.copy(numpy.swapaxes(data, 1, 0))
        return data

    def _read_raw_fun(self, range1, range2):
        """
        Reads data as stored in a file, before any complex data and symmetry
        transformations are applied. The one potential exception to the "raw"
        file orientation of the data is that bands will always be returned in the
        first dimension (data[n,:,:] is the nth band -- "band sequential" or BSQ,
        as stored in Python's memory), regardless of how the data is stored in
        the file.

        Parameters
        ----------
        range1 : None|int|tuple
            * if `None`, then the range is not limited in first axis
            * if `int` = start
            * if (`int`, `int`) = `start`, `stop`
            * if (`int`, `int`, `int`) = `start`, `stop`, `step size`
        range2 : None|int|tuple
            same as `range1`, except for the second axis.

        Returns
        -------
        numpy.ndarray
            the (mostly) raw data read from the file
        """

        # Should generally begin as:
        # arange1, arange2 = self._reorder_arguments(range1, range2)

        raise NotImplementedError


class SubsetChipper(BaseChipper):
    """
    Permits transparent extraction from a particular subset of the possible data range
    """

    __slots__ = ('_data_size', '_transform_data', '_symmetry', 'shift1', 'shift2', 'parent_chipper')

    def __init__(self, parent_chipper, dim1bounds, dim2bounds):
        """

        Parameters
        ----------
        parent_chipper : BaseChipper
        dim1bounds : numpy.ndarray|list|tuple
        dim2bounds: numpy.ndarray|list|tuple
        """

        if not isinstance(parent_chipper, BaseChipper):
            raise TypeError('parent_chipper is required to be an instance of BaseChipper, '
                            'got type {}'.format(type(parent_chipper)))

        data_size = (dim1bounds[1] - dim1bounds[0], dim2bounds[1] - dim2bounds[0])
        self.shift1 = dim1bounds[0]
        self.shift2 = dim2bounds[0]
        self.parent_chipper = parent_chipper
        super(SubsetChipper, self).__init__(data_size, symmetry=(False, False, False), transform_data=None)

    def _reformat_bounds(self, range1, range2):
        def _get_start(entry, shift, bound, step):
            if entry is None:
                return shift if step > 0 else bound + shift - 1
            entry = int(entry)
            if -bound < entry < 0:
                return shift + bound + entry
            elif 0 <= entry < bound:
                return shift + entry
            raise ValueError(
                'Got slice start entry {}, which must be in the range '
                '({}, {})'.format(entry, -bound, bound))

        def _get_end(entry, shift, bound, step):
            if entry is None:
                return bound + shift if step > 0 else shift - 1
            entry = int(entry)
            if -bound <= entry < 0:
                return shift + bound + entry
            elif 0 <= entry <= bound:
                return shift + entry
            raise ValueError(
                'Got slice end entry {}, which must be in the range '
                '[{}, {}]'.format(entry, -bound, bound))

        def _get_range(rng, shift, bound):
            step = 1 if rng[2] is None else rng[2]
            return (
                _get_start(rng[0], shift, bound, step),
                _get_end(rng[1], shift, bound, step),
                step)
        arange1 = _get_range(range1, self.shift1, self._data_size[0])
        arange2 = _get_range(range2, self.shift2, self._data_size[1])
        return arange1, arange2

    def _read_raw_fun(self, range1, range2):
        arange1, arange2 = self._reformat_bounds(range1, range2)
        return self.parent_chipper.__call__(arange1, arange2)


class AggregateChipper(BaseChipper):
    """
    Class which allows assembly of a collection of chippers into a single
    chipper object.
    """

    __slots__ = ('_child_chippers', '_bounds', '_dtype', '_output_bands')

    def __init__(self, bounds, output_dtype, child_chippers, output_bands=1):
        """

        Parameters
        ----------
        bounds : numpy.ndarray
            Two-dimensional array of `[[row start, row end, column start, column end]]`.
        output_dtype : str|numpy.dtype|numpy.number
            The data type of the output data
        child_chippers : tuple|list
            The list or tuple of child chipper objects.
        output_bands : int
            The number of bands (after intermediate chipper adjustments).
        """

        # validate bounds
        data_sizes = self._validate_bounds(bounds)
        # validate child chippers
        child_chippers = tuple(child_chippers)
        if bounds.shape[0] != len(child_chippers):
            raise ValueError('bounds and child_chippers must have compatible lengths')
        for i, entry in enumerate(child_chippers):
            if not isinstance(entry, BaseChipper):
                raise TypeError(
                    'Each entry of child_chippers must be an instance of BaseChipper, '
                    'got type {} at entry {}'.format(type(entry), i))
            expected_shape = (int(data_sizes[i, 0]), int(data_sizes[i, 1]))
            if entry.data_size != expected_shape:
                raise ValueError(
                    'chipper at index {} has expected shape {}, but '
                    'actual shape {}'.format(i, expected_shape, entry.data_size))
        self._child_chippers = child_chippers
        self._bounds = bounds
        self._dtype = output_dtype
        self._output_bands = int(output_bands)
        data_size = (
            int(numpy.max(self._bounds[:, 1])),
            int(numpy.max(self._bounds[:, 3])))
        # all of the actual reading and reorienting done by child chippers,
        # so do not reorient or change type at this level
        super(AggregateChipper, self).__init__(data_size, symmetry=(False, False, False), transform_data=None)

    @staticmethod
    def _validate_bounds(bounds):
        """
        Validate the bounds array.

        Parameters
        ----------
        bounds : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        if not isinstance(bounds, numpy.ndarray):
            raise ValueError('bounds must be an numpy.ndarray, not {}'.format(type(bounds)))
        if not issubclass(bounds.dtype.type, numpy.integer):
            raise ValueError('bounds must be an integer dtype numpy.ndarray, got dtype {}'.format(bounds.dtype))
        if not (bounds.ndim == 2 and bounds.shape[1] == 4):
            raise ValueError('bounds must be an Nx4 numpy.ndarray, not shape {}'.format(bounds.shape))

        # determine data sizes and sensibility
        data_sizes = numpy.zeros((bounds.shape[0], 2), dtype=numpy.int64)

        for i, entry in enumerate(bounds):
            # Are the order of the entries in bounds sensible?
            if not (0 <= entry[0] < entry[1] and 0 <= entry[2] < entry[3]):
                raise ValueError('entry {} of bounds is {}, and cannot be of the form '
                                 '[row start, row end, column start, column end]'.format(i, entry))
            # define the data_sizes entry
            data_sizes[i, :] = (entry[1] - entry[0], entry[3] - entry[2])
        return data_sizes

    def _subset(self, rng, start_ind, stop_ind):
        """
        Finds the rectangular overlap between the desired indices and given chipper bounds.

        Parameters
        ----------
        rng
        start_ind
        stop_ind

        Returns
        -------
        tuple, tuple
        """

        if rng[2] > 0:
            if rng[1] <= start_ind or rng[0] >= stop_ind:
                # there is no overlap
                return None, None
            # find smallest element rng[0] + mult*rng[2] which is >= start_ind
            mult1 = 0 if start_ind <= rng[0] else int(numpy.ceil((start_ind - rng[0]) / rng[2]))
            ind1 = rng[0] + mult1 * rng[2]
            # find largest element rng[0] + mult*rng[2] which is <= min(stop_ind, rng[1])
            max_ind = min(rng[1], stop_ind)
            mult2 = int(numpy.floor((max_ind - rng[0]) / rng[2]))
            ind2 = rng[0] + mult2 * rng[2]
        else:
            if rng[0] <= start_ind or rng[1] >= stop_ind:
                return None, None
            # find largest element rng[0] + mult*rng[2] which is <= stop_ind-1
            mult1 = 0 if rng[0] < stop_ind else int(numpy.floor((stop_ind - 1 - rng[0])/rng[2]))
            ind1 = rng[0] + mult1*rng[2]
            # find smallest element rng[0] + mult*rng[2] which is >= max(start_ind, rng[1]+1)
            mult2 = int(numpy.floor((start_ind - rng[0])/rng[2])) if rng[1] < start_ind \
                else int(numpy.floor((rng[1] -1 - rng[0])/rng[2]))
            ind2 = rng[0] + mult2*rng[2]
        return (ind1-start_ind, ind2-start_ind, rng[2]), (mult1, mult2)

    def _read_raw_fun(self, range1, range2):
        range1, range2 = self._reorder_arguments(range1, range2)
        rows_size = int((range1[1]-range1[0])/range1[2])
        cols_size = int((range2[1]-range2[0])/range2[2])

        if self._output_bands == 1:
            out = numpy.zeros((rows_size, cols_size), dtype=self._dtype)
        else:
            out = numpy.zeros((rows_size, cols_size, self._output_bands), dtype=self._dtype)
            # TODO: missing/unmapped data will appear as zeros.
            #   Does this make sense in all cases? should it be nan for floating point?

        for entry, child_chipper in zip(self._bounds, self._child_chippers):
            row_start, row_end, col_start, col_end = entry
            # find row overlap for chipper - it's rectangular
            crange1, cinds1 = self._subset(range1, row_start, row_end)
            if crange1 is None:
                continue  # there is no row overlap for this chipper

            # find column overlap for chipper - it's rectangular
            crange2, cinds2 = self._subset(range2, col_start, col_end)
            if crange2 is None:
                continue  # there is no column overlap for this chipper

            if self._output_bands == 1:
                out[cinds1[0]:cinds1[1], cinds2[0]:cinds2[1]] = \
                    child_chipper[crange1[0]:crange1[1]:crange1[2], crange2[0]:crange2[1]:crange2[2]]
            else:
                out[cinds1[0]:cinds1[1], cinds2[0]:cinds2[1], :] = \
                    child_chipper[crange1[0]:crange1[1]:crange1[2], crange2[0]:crange2[1]:crange2[2]]
        return out


#################
# Base Reader definition

class AbstractReader(object):
    """
    The abstract reader basic definition - essentially just an interface definition.
    """

    __slots__ = ('_chipper', '_data_size', '_reader_type')

    @property
    def reader_type(self):
        """
        str: A descriptive string for the type of reader
        """
        return self._reader_type

    @property
    def data_size(self):
        # type: () -> Union[Tuple[int, int], Tuple[Tuple[int, int]]]
        """
        Tuple[int, int]|Tuple[Tuple[int, int], ...]: the data size(s) of the form (rows, cols).
        """

        return self._data_size

    @property
    def image_count(self):
        """
        int: The number of images from which to read.
        """

        if isinstance(self._chipper, tuple):
            return len(self._chipper)
        else:
            return 1

    @property
    def file_name(self):
        # type: () -> Optional[str]
        """
        None|str: Defined a convenience property.
        """

        return None

    def get_data_size_as_tuple(self):
        """
        Get the data size wrapped in a tuple - for simplicity and ease of use.

        Returns
        -------
        Tuple[Tuple[int, int]]
        """

        if isinstance(self._chipper, tuple):
            return self._data_size
        else:
            return (self._data_size, )

    def _get_chippers_as_tuple(self):
        """
        Get the chipper collection as a tuple.

        Returns
        -------
        Tuple[BaseChipper]
        """

        if isinstance(self._chipper, tuple):
            return self._chipper
        else:
            return (self._chipper, )

    def _validate_index(self, index):
        if isinstance(self._chipper, BaseChipper) or index is None:
            return 0

        if not isinstance(index, int):
            raise ValueError('Cannot slice in multiple indices on the third dimension.')
        index = int(index)
        siz = len(self._chipper)
        if not (-siz < index < siz):
            raise ValueError('index must be in the range ({}, {})'.format(-siz, siz))
        return index

    def _validate_slice(self, item):
        if isinstance(item, tuple):
            if len(item) > 3:
                raise ValueError(
                    'Reader received slice argument {}. We cannot slice on more than '
                    'three dimensions.'.format(item))
            if len(item) == 3:
                index = self._validate_index(item[2])
                return item[:2], index
        return item, 0

    def __call__(self, range1, range2, index=0):
        """
        Reads and fetches data.

        Parameters
        ----------
        range1 : None|int|tuple
            The row data selection of the form `[start, [stop, [stride]]]`, and
            `None` defaults to all rows (i.e. `(0, NumRows, 1)`)
        range2 : None|int|tuple
            The column data selection of the form `[start, [stop, [stride]]]`, and
            `None` defaults to all rows (i.e. `(0, NumCols, 1)`)
        index : None|int

        Returns
        -------
        numpy.ndarray
            If complex, the data type will be `complex64`, so be sure to upcast to
            `complex128` if so desired.

        Examples
        --------

        Basic fetching
        :code:`data = reader((start1, stop1, stride1), (start2, stop2, stride2), index=0)`

        Also, basic fetching can be accomplished via Python style basic slice syntax
        :code:`data = reader[start1:stop1:stride1, start:stop:stride]` or
        :code:`data = reader[start:stop:stride, start:stop:stride, index]`

        Here the slice on index (dimension 3) is limited to a single integer, and
        no slice on the index :code:`reader[:, :]` will default to `index=0`,
        :code:`reader[:, :, 0]` (where appropriate).

        :code:`reader((start1, stop1, stride1), (start2, stop2, stride2))`
        yields the same as :code:`reader[start1:stop1:stride1, start2:stop2:stride2]`.
        """

        if isinstance(self._chipper, tuple):
            index = self._validate_index(index)
            return self._chipper[index](range1, range2)
        else:
            return self._chipper(range1, range2)

    def __getitem__(self, item):
        """
        Reads and returns data using more traditional to python slice functionality.
        After slice interpretation, this is analogous to :func:`__call__` or
        :func:`read_chip`.

        Parameters
        ----------
        item : None|int|slice|tuple

        Returns
        -------
        numpy.ndarray
            If complex, the data type will be `complex64`, so be sure to upcast to
            `complex128` if so desired.

        Examples
        --------
        This is the familiar Python style basic slice syntax
        :code:`data = reader[start1:stop1:stride1, start:stop:stride]` or
        :code:`data = reader[start:stop:stride, start:stop:stride, index]`

        Here the slice on index (dimension 3) is limited to a single integer, and
        no slice on index :code:`reader[:, :]` will default to `index=0`,
        :code:`reader[:, :, 0]` (where appropriate).
        """

        item, index = self._validate_slice(item)
        if isinstance(self._chipper, tuple):
            return self._chipper[index].__getitem__(item)
        else:
            return self._chipper.__getitem__(item)

    def read_chip(self, dim1range, dim2range, index=None):
        """
        Read the given section of data as an array. Note that
        :code:`reader.read_chip(range1, range2, index)` is an alias for
        :code:`reader(range1, range2, index)`.

        Parameters
        ----------
        dim1range : None|int|Tuple[int, int]|Tuple[int, int, int]
            The row data selection of the form `[start, [stop, [stride]]]`, and
            `None` defaults to all rows (i.e. `(0, NumRows, 1)`)
        dim2range : None|int|Tuple[int, int]|Tuple[int, int, int]
            The column data selection of the form `[start, [stop, [stride]]]`, and
            `None` defaults to all rows (i.e. `(0, NumCols, 1)`)
        index : int|None
            Relative to which sicd/chipper, and only used in the event of multiple
            sicd/chippers. Defaults to `0`, if not provided.

        Returns
        -------
        numpy.ndarray
            If complex, the data type will be `complex64`, so be sure to upcast to
            `complex128` if so desired.

        Examples
        --------
        The **preferred syntax** is to use Python slice syntax or call syntax,
        and the following yield equivalent results

        .. code-block:: python

            data = reader[start:stop:stride, start:stop:stride, index]
            data = reader((start1, stop1, stride1), (start2, stop2, stride2), index=index)`
            data = reader.read_chip((start1, stop1, stride1), (start2, stop2, stride2) index=index)

        Here the slice on index (dimension 3) is limited to a single integer. No
        slice on index will default to `index=0`, that is :code:`reader[:, :]` and
        :code:`reader[:, :, 0]` yield equivalent results.
        """

        return self.__call__(dim1range, dim2range, index=index)


class BaseReader(AbstractReader):
    """
    Abstract file reader class
    """

    __slots__ = ('_chipper', '_data_size', '_reader_type')

    def __init__(self, chipper, reader_type="OTHER"):
        """

        Parameters
        ----------
        chipper : BaseChipper|Tuple[BaseChipper]
            a chipper object, or tuple of chipper objects
        reader_type : str
            What kind of reader is this? Allowable options are "SICD", "SIDD",
            "CPHD", or "OTHER".
        """
        # set the reader_type state
        if not isinstance(reader_type, str):
            raise ValueError('reader_type must be a string, got {}'.format(type(reader_type)))
        if reader_type not in READER_TYPES:
            logger.error(
                'reader_type has value {}, while it is expected to be '
                'one of {}'.format(reader_type, READER_TYPES))
        self._reader_type = reader_type
        # adjust chipper inputs
        if isinstance(chipper, list):
            chipper = tuple(chipper)

        # validate chipper input
        if isinstance(chipper, tuple):
            for el in chipper:
                if not isinstance(el, BaseChipper):
                    raise TypeError(
                        'Got a collection for chipper, and all elements are required '
                        'to be instances of BaseChipper.')
        elif not isinstance(chipper, BaseChipper):
            raise TypeError(
                'chipper argument is required to be a BaseChipper instance, or collection of BaseChipper objects')
        self._chipper = chipper

        # determine data_size
        if isinstance(chipper, BaseChipper):
            data_size = chipper.data_size
        else:
            data_size = tuple(el.data_size for el in chipper)
        self._data_size = data_size

    @property
    def file_name(self):
        # type: () -> str
        """
        str: The file/path name for the reader object.
        """

        raise NotImplementedError


class SubsetReader(BaseReader):
    """
    Permits extraction from a particular subset of the possible data range.
    """

    __slots__ = ('_parent_reader', )

    def __init__(self, parent_reader, dim1bounds, dim2bounds):
        """

        Parameters
        ----------
        parent_reader : AbstractReader
        dim1bounds : tuple
        dim2bounds : tuple
        """

        self._parent_reader = parent_reader
        # noinspection PyProtectedMember
        chipper = SubsetChipper(parent_reader._chipper, dim1bounds, dim2bounds)
        super(SubsetReader, self).__init__(chipper, reader_type=parent_reader.reader_type)

    @property
    def file_name(self):
        return self._parent_reader.file_name


class AggregateReader(BaseReader):
    """
    Aggregate multiple files and/or readers into a single reader instance. This default
    aggregate implementation will not preserve any SICD or SIDD structures.
    """

    __slots__ = ('_readers', '_index_mapping')

    def __init__(self, readers, reader_type="OTHER"):
        """

        Parameters
        ----------
        readers : Sequence[AbstractReader]
        reader_type : str
            The reader type string.
        """

        self._index_mapping = None
        self._readers = self._validate_readers(readers)
        the_chippers = self._define_index_mapping()
        super(AggregateReader, self).__init__(chipper=the_chippers, reader_type=reader_type)

    @staticmethod
    def _validate_readers(readers):
        """
        Validate the input reader/file collection.

        Parameters
        ----------
        readers : Sequence[AbstractReader]

        Returns
        -------
        Tuple[AbstractReader]
        """

        if not isinstance(readers, (list, tuple)):
            raise TypeError('input argument must be a list or tuple of readers/files. Got type {}'.format(type(readers)))

        # validate each entry
        the_readers = []
        for i, entry in enumerate(readers):
            if not isinstance(entry, AbstractReader):
                raise TypeError(
                    'All elements of the input argument must be file names or reader instances. '
                    'Entry {} is of type {}'.format(i, type(entry)))
            the_readers.append(entry)
        return tuple(the_readers)

    def _define_index_mapping(self):
        """
        Define the index mapping.

        Returns
        -------
        Tuple[BaseChipper]
        """

        # prepare the index mapping workspace
        index_mapping = []
        # assemble the chipper arguments
        the_chippers = []
        for i, reader in enumerate(self._readers):
            for j, chipper in enumerate(reader._get_chippers_as_tuple()):
                the_chippers.append(chipper)
                index_mapping.append((i, j))

        self._index_mapping = tuple(index_mapping)
        return tuple(the_chippers)

    @property
    def index_mapping(self):
        # type: () -> Tuple[Tuple[int, int]]
        """
        Tuple[Tuple[int, int]]: The index mapping of the form (reader index, sicd index).
        """

        return self._index_mapping

    @property
    def file_name(self):
        # type: () -> Tuple[Optional[str]]
        """
        Tuple[Optional[str]]: The filename collection.
        """

        return tuple(entry.file_name for entry in self._readers)


class FlatReader(BaseReader):
    """
    Class for passing a numpy array or memmap straight through as a reader.
    """

    def __init__(self, array, reader_type='OTHER', output_bands=None, output_dtype=None,
                 symmetry=(False, False, False), transform_data=None, limit_to_raw_bands=None):
        """

        Parameters
        ----------
        array : numpy.ndarray
        reader_type : str
            What kind of reader is this? Should be "SICD", or "OTHER" here. If SICD,
            the (output) data is expected to be complex.
        output_bands : None|int
        output_dtype : None|str|numpy.dtype|numpy.number
        symmetry : tuple
            Describes any required data transformation. See the `symmetry` property.
        transform_data : None|str|Callable
            For data transformation after reading.
            If `None`, then no transformation will be applied. If `callable`, then
            this is expected to be the transformation method for the raw data. If
            string valued and `'complex'`, then the assumption is that real/imaginary
            components are stored in adjacent bands, which will be combined into a
            single band upon extraction. Other situations will yield and value error.
        limit_to_raw_bands : None|int|numpy.ndarray|list|tuple
            The collection of raw bands to which to read. `None` is all bands.
        """

        if array.ndim not in [2, 3]:
            raise ValueError('Requires two or three-dimensional array')

        raw_dtype = array.dtype
        raw_bands = 1 if array.ndim == 2 else array.shape[3]
        data_size = array.shape[:2]

        if output_bands is None:
            if transform_data is not None or limit_to_raw_bands is not None:
                raise ValueError(
                    'output_bands is not populated, but transform_data or limit_to_raw_bands is populated.')
            output_bands = raw_bands

        if output_dtype is None:
            if transform_data is not None:
                raise ValueError(
                    'output_dtype is not populated, but transform_data is populated.')
            output_dtype = raw_dtype

        chipper = BIPChipper(
            array, raw_dtype, data_size, raw_bands, output_bands, output_dtype,
            symmetry=symmetry, transform_data=transform_data, limit_to_raw_bands=limit_to_raw_bands)
        super(FlatReader, self).__init__(chipper, reader_type=reader_type)

    @property
    def file_name(self):
        return None


#################
# Base Writer definition

class AbstractWriter(object):
    """
    Abstract file writer class for SICD data.
    """

    __slots__ = ('_file_name', )

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        self._file_name = file_name
        if not os.path.exists(self._file_name):
            with open(self._file_name, 'wb') as fi:
                fi.write(b'')

    def close(self):
        """
        Completes any necessary final steps.

        Returns
        -------
        None
        """

        pass

    def write_chip(self, data, start_indices=(0, 0)):
        """
        Write the data to the file(s). This is an alias to :code:`writer(data, start_indices)`.

        Parameters
        ----------
        data : numpy.ndarray
            the complex data
        start_indices : tuple[int, int]
            the starting index for the data.

        Returns
        -------
        None
        """

        self.__call__(data, start_indices=start_indices)

    def __call__(self, data, start_indices=(0, 0)):
        """
        Write the data to the file(s).

        Parameters
        ----------
        data : numpy.ndarray
            the complex data
        start_indices : Tuple[int, int]
            the starting index for the data.

        Returns
        -------
        None
        """

        raise NotImplementedError

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is None:
            self.close()
        else:
            logger.error(
                'The {} file writer generated an exception during processing.\n\t'
                'The file {} may be only partially generated and corrupt.'.format(
                    self.__class__.__name__, self._file_name))
            # The exception will be reraised.
            # It's unclear how any exception could be caught.


#############
# concrete chipper implementations for NITF file patterns

def _validate_limit_to_raw_bands(limit_to_raw_bands, raw_bands):
    if limit_to_raw_bands is None:
        return None

    if isinstance(limit_to_raw_bands, int):
        limit_to_raw_bands = numpy.array([limit_to_raw_bands, ], dtype='int32')
    if isinstance(limit_to_raw_bands, (list, tuple)):
        limit_to_raw_bands = numpy.array(limit_to_raw_bands, dtype='int32')
    if not isinstance(limit_to_raw_bands, numpy.ndarray):
        raise TypeError('limit_to_raw_bands got unsupported input of type {}'.format(type(limit_to_raw_bands)))
    # ensure that limit_to_raw_bands make sense...
    if numpy.any((limit_to_raw_bands < 0) | (limit_to_raw_bands >= raw_bands)):
        raise ValueError(
            'all entries of limit_to_raw_bands ({}) must be in the range 0 <= value < {}'.format(limit_to_raw_bands, raw_bands))
    return limit_to_raw_bands


class BIPChipper(BaseChipper):
    """
    Band interleaved format file chipper.
    """

    __slots__ = (
        '_file_name', '_file_object', '_data_offset', '_shape',
        '_raw_bands', '_raw_dtype', '_output_bands', '_output_dtype',
        '_limit_to_raw_bands', '_memory_map', '_close_after')

    def __init__(self, data_input, raw_dtype, data_size, raw_bands, output_bands, output_dtype,
                 symmetry=(False, False, False), transform_data=None,
                 data_offset=0, limit_to_raw_bands=None):
        """

        Parameters
        ----------
        data_input : str|BinaryIO|numpy.ndarray
            The name of a file or binary file like object from which to read,
            or a numpy array or memmap to use directly.
        raw_dtype : str|numpy.dtype|numpy.number
            The data type of the underlying file. **Note: specify endianness where necessary.**
        data_size : tuple
            The `(rows, columns)` of the raw data. See `data_size` property.
        raw_bands : int
            The number of bands in the file.
        output_bands : int
            The number of bands in the output data.
        output_dtype : str|numpy.dtype|numpy.number
            The data type of the return data. This should be in keeping with `transform_data`.
        symmetry : tuple
            Describes any required data transformation. See the `symmetry` property.
        transform_data : None|str|Callable
            For data transformation after reading.
            If `None`, then no transformation will be applied. If `callable`, then
            this is expected to be the transformation method for the raw data. If
            string valued and `'complex'`, then the assumption is that real/imaginary
            components are stored in adjacent bands, which will be combined into a
            single band upon extraction. Other situations will yield and value error.
        data_offset : int
            byte offset from the start of the file at which the data actually starts
        limit_to_raw_bands : None|int|numpy.ndarray|list|tuple
            The collection of raw bands to which to read. `None` is all bands.
        """

        self._limit_to_raw_bands = None
        self._file_name = None #type: Union[None, str]
        self._file_object = None  # type: Union[None, BinaryIO]
        self._memory_map = None  # type: Union[None, numpy.ndarray]
        self._close_after = False  # type: bool
        super(BIPChipper, self).__init__(data_size, symmetry=symmetry, transform_data=transform_data)

        raw_bands = int(raw_bands)
        if raw_bands < 1:
            raise ValueError('raw_bands must be a positive integer')
        self._raw_bands = raw_bands

        output_bands = int(output_bands)
        if output_bands < 1:
            raise ValueError('output_bands must be a positive integer.')
        self._output_bands = output_bands

        self._raw_dtype = raw_dtype
        self._output_dtype = output_dtype

        data_offset = int(data_offset)
        if data_offset < 0:
            raise ValueError('data_offset must be a non-negative integer. Got {}'.format(data_offset))
        self._data_offset = int(data_offset)

        self._shape = (int(data_size[0]), int(data_size[1]), self._raw_bands)

        if isinstance(data_input, numpy.ndarray):
            self._memory_map = data_input
            self._file_name = None
            self._file_object = None
            # NB: we assume the the rest of the data is set up properly
        elif is_file_like(data_input):
            self._file_object = data_input
            self._close_after = False
            self._file_name = data_input.name if hasattr(data_input, 'name') else None

            try_memmap = False
            if hasattr(data_input, 'fileno'):
                try:
                    # check that fileno actually works, not just exists.
                    data_input.fileno()
                    try_memmap = True
                except io.UnsupportedOperation:
                    pass

            if try_memmap:
                # noinspection PyBroadException
                try:
                    self._memory_map = numpy.memmap(data_input,
                                                    dtype=raw_dtype,
                                                    mode='r',
                                                    offset=data_offset,
                                                    shape=self._shape)
                except Exception as e:
                    # fall back to direct reading
                    logger.error(
                        'Error setting up a BIP chipper - {}\n\t'
                        'This may actually be fatal.'.format(e))
                    self._memory_map = None
            else:
                self._memory_map = None
        elif isinstance(data_input, str):
            if not os.path.isfile(data_input):
                raise SarpyIOError('Path {} either does not exists, or is not a file.'.format(data_input))
            if not os.access(data_input, os.R_OK):
                raise SarpyIOError('User does not appear to have read access for file {}.'.format(data_input))
            self._file_name = data_input
            try:
                self._memory_map = numpy.memmap(self._file_name,
                                                dtype=raw_dtype,
                                                mode='r',
                                                offset=data_offset,
                                                shape=self._shape)
                self._file_object = None
                self._close_after = False
            except (OverflowError, OSError):
                # fallback after failing to construct the mem map
                #   the most likely cause to be here is that 32-bit python fails
                #   constructing a memmap for any file larger than 2GB
                self._file_object = open(self._file_name, mode='rb')
                self._close_after = True
                logger.warning(
                    'Falling back to reading file {} manually, instead of using '
                    'numpy memmap.'.format(self._file_name))
        self._validate_limit_to_raw_bands(limit_to_raw_bands)

    def _validate_limit_to_raw_bands(self, limit_to_raw_bands):
        limit_to_raw_bands = _validate_limit_to_raw_bands(limit_to_raw_bands, self._raw_bands)
        if self._memory_map is None and limit_to_raw_bands is not None:
            raise ValueError(
                'BIP chipper cannot utilize limit_to_raw_bands except when using a local file.')
        self._limit_to_raw_bands = limit_to_raw_bands

    def __del__(self):
        if not self._close_after:
            return
        if self._file_object is not None and \
                hasattr(self._file_object, 'closed') and \
                not self._file_object.closed:
            self._file_object.close()

    def _read_raw_fun(self, range1, range2):
        t_range1, t_range2 = self._reorder_arguments(range1, range2)
        if self._memory_map is not None:
            return self._read_memory_map(t_range1, t_range2)
        else:
            return self._read_file(t_range1, t_range2)

    def _read_memory_map(self, range1, range2):
        if (range1[1] == -1 and range1[2] < 0) and (range2[1] == -1 and range2[2] < 0):
            if self._limit_to_raw_bands is None:
                out = numpy.array(
                    self._memory_map[range1[0]::range1[2], range2[0]::range2[2]],
                    dtype=self._raw_dtype)
            else:
                out = numpy.array(
                    self._memory_map[range1[0]::range1[2], range2[0]::range2[2], self._limit_to_raw_bands],
                    dtype=self._raw_dtype)
        elif range1[1] == -1 and range1[2] < 0:
            if self._limit_to_raw_bands is None:
                out = numpy.array(
                    self._memory_map[range1[0]::range1[2], range2[0]:range2[1]:range2[2]],
                    dtype=self._raw_dtype)
            else:
                out = numpy.array(
                    self._memory_map[range1[0]::range1[2], range2[0]:range2[1]:range2[2], self._limit_to_raw_bands],
                    dtype=self._raw_dtype)
        elif range2[1] == -1 and range2[2] < 0:
            if self._limit_to_raw_bands is None:
                out = numpy.array(
                    self._memory_map[range1[0]:range1[1]:range1[2], range2[0]::range2[2]],
                    dtype=self._raw_dtype)
            else:
                out = numpy.array(
                    self._memory_map[range1[0]:range1[1]:range1[2], range2[0]::range2[2], self._limit_to_raw_bands],
                    dtype=self._raw_dtype)
        else:
            if self._limit_to_raw_bands is None:
                out = numpy.array(
                    self._memory_map[range1[0]:range1[1]:range1[2], range2[0]:range2[1]:range2[2]],
                    dtype=self._raw_dtype)
            else:
                out = numpy.array(
                    self._memory_map[range1[0]:range1[1]:range1[2], range2[0]:range2[1]:range2[2], self._limit_to_raw_bands],
                    dtype=self._raw_dtype)
        return out

    def _read_file(self, range1, range2):
        if self._limit_to_raw_bands is None:
            band_collection = numpy.arange(self._raw_bands, dtype='int32')
        else:
            band_collection = self._limit_to_raw_bands

        init_location = self._file_object.tell()
        # we have to manually map out the stride and all that for the array ourselves
        element_size = int(numpy.dtype(self._raw_dtype).itemsize*self._raw_bands)
        stride = element_size*int(self._shape[1])  # how much to skip a whole (real) row?
        # let's determine the specific row/column arrays that we are going to read
        dim1array = numpy.arange(*range1)
        dim2array = numpy.arange(*range2)
        # determine the contiguous chunk to read
        start_row = min(dim1array[0], dim1array[-1])
        rows = abs(range1[1] - range1[0])
        # allocate our output array
        out = numpy.empty((len(dim1array), len(dim2array), len(band_collection)), dtype=self._raw_dtype)

        # seek to the proper start location
        start_loc = self._data_offset + start_row*stride
        self._file_object.seek(start_loc, os.SEEK_SET)
        # read our data
        total_entries = int(rows)*self._shape[1]*self._raw_bands
        total_size = int(rows)*stride
        data = self._file_object.read(total_size)
        if len(data) != total_size:
            raise ValueError(
                'Tried to read {} bytes of data, but received {}.\n'
                'The most likely reason for this is a malformed chipper, \n'
                'which attempts to read more data than the file contains'.format(total_size, len(data)))
        # cast and shape
        data = numpy.frombuffer(data, self._raw_dtype, total_entries)
        data = numpy.reshape(data, (rows, self._shape[1], self._raw_bands))
        # reduce, as necessary
        out[:, :] = data[range1[0]-start_row:range1[1]-start_row:range1[2], range2[0]:range2[1]:range2[2], band_collection]
        self._file_object.seek(init_location, os.SEEK_SET)
        return out


class BSQChipper(BaseChipper):
    """
    Chipper enabling Band-sequential and Band Interleaved by Block formats,
    assembled from single band BIP constituents
    """

    __slots__ = ('_child_chippers', '_dtype', '_limit_to_raw_bands')

    def __init__(self, child_chippers, output_dtype, transform_data=None, limit_to_raw_bands=None):
        """

        Parameters
        ----------
        child_chippers : tuple|list
            The list or tuple of child chipper objects.
        output_dtype : str|numpy.dtype|numpy.number
            The data type of the output data
        transform_data : None|str|Callable
            For data transformation after reading.
            If `None`, then no transformation will be applied. If `callable`, then
            this is expected to be the transformation method for the raw data. If
            string valued and `'complex'`, then the assumption is that real/imaginary
            components are stored in adjacent bands, which will be combined into a
            single band upon extraction. Other situations will yield and value error.
        limit_to_raw_bands : None|int|numpy.ndarray|list|tuple
            The collection of raw bands to which to read. `None` is all bands.
        """

        self._dtype = output_dtype
        self._limit_to_raw_bands = None
        # validate that the data_sizes are all the same
        data_size = None
        for i, entry in enumerate(child_chippers):
            if not isinstance(entry, BaseChipper):
                raise TypeError(
                    'Each entry of child_chippers must be an instance of BaseChipper, '
                    'got type {} at entry {}'.format(type(entry), i))
            if data_size is None:
                data_size = entry.data_size
            elif entry.data_size != data_size:
                raise ValueError(
                    'chipper at index {} has expected shape {}, but '
                    'actual shape {}'.format(i, data_size, entry.data_size))
        self._child_chippers = child_chippers
        # NB: it is left to the constructor to know that these all fit otherwise
        super(BSQChipper, self).__init__(data_size, symmetry=(False, False, False), transform_data=transform_data)
        self._validate_limit_to_raw_bands(limit_to_raw_bands)

    @property
    def output_bands(self):
        """
        int: The number of bands.
        """

        return len(self._child_chippers)

    def _validate_limit_to_raw_bands(self, limit_to_raw_bands):
        self._limit_to_raw_bands = _validate_limit_to_raw_bands(limit_to_raw_bands, self.output_bands)

    def _read_raw_fun(self, range1, range2):
        range1, range2 = self._reorder_arguments(range1, range2)
        rows_size = int(numpy.ceil((range1[1]-range1[0])/range1[2]))
        cols_size = int(numpy.ceil((range2[1]-range2[0])/range2[2]))

        if self._limit_to_raw_bands is None:
            out = numpy.zeros((rows_size, cols_size, self.output_bands), dtype=self._dtype)
            for band_number, child_chipper in enumerate(self._child_chippers):
                out[:rows_size, :cols_size, band_number] = \
                    child_chipper[range1[0]:range1[1]:range1[2], range2[0]:range2[1]:range2[2]]
        else:
            out = numpy.zeros((rows_size, cols_size, self._limit_to_raw_bands.size), dtype=self._dtype)
            for i, band_number in enumerate(self._limit_to_raw_bands):
                child_chipper = self._child_chippers[int(band_number)]
                out[:rows_size, :cols_size, i] = \
                    child_chipper[range1[0]:range1[1]:range1[2], range2[0]:range2[1]:range2[2]]
        return out


class BIRChipper(BaseChipper):
    """
    Band Interleaved by Row chipper.
    """

    __slots__ = (
        '_file_name', '_file_object', '_data_offset', '_shape',
        '_raw_bands', '_raw_dtype', '_output_bands', '_output_dtype',
        '_limit_to_raw_bands', '_memory_map', '_close_after')

    def __init__(self, data_input, raw_dtype, data_size, raw_bands, output_bands, output_dtype,
                 symmetry=(False, False, False), transform_data=None,
                 data_offset=0, limit_to_raw_bands=None):
        """

        Parameters
        ----------
        data_input : str|BinaryIO|numpy.ndarray
            The name of a file or binary file like object from which to read,
            or a numpy array or memmap to use directly.
        raw_dtype : str|numpy.dtype|numpy.number
            The data type of the underlying file. **Note: specify endianness where necessary.**
        data_size : tuple
            The `(rows, columns)` of the raw data. See `data_size` property.
        raw_bands : int
            The number of bands in the file.
        output_bands : int
            The number of bands in the output data.
        output_dtype : str|numpy.dtype|numpy.number
            The data type of the return data. This should be in keeping with `transform_data`.
        symmetry : tuple
            Describes any required data transformation. See the `symmetry` property.
        transform_data : None|str|Callable
            For data transformation after reading.
            If `None`, then no transformation will be applied. If `callable`, then
            this is expected to be the transformation method for the raw data. If
            string valued and `'complex'`, then the assumption is that real/imaginary
            components are stored in adjacent bands, which will be combined into a
            single band upon extraction. Other situations will yield and value error.
        data_offset : int
            byte offset from the start of the file at which the data actually starts
        limit_to_raw_bands : None|int|numpy.ndarray|list|tuple
            The collection of raw bands to which to read. `None` is all bands.
        """

        self._limit_to_raw_bands = None
        self._file_name = None #type: Union[None, str]
        self._file_object = None  # type: Union[None, BinaryIO]
        self._memory_map = None  # type: Union[None, numpy.ndarray]
        self._close_after = False  # type: bool
        super(BIRChipper, self).__init__(data_size, symmetry=symmetry, transform_data=transform_data)

        raw_bands = int(raw_bands)
        if raw_bands < 1:
            raise ValueError('raw_bands must be a positive integer')
        self._raw_bands = raw_bands

        output_bands = int(output_bands)
        if output_bands < 1:
            raise ValueError('output_bands must be a positive integer.')
        self._output_bands = output_bands

        self._raw_dtype = raw_dtype
        self._output_dtype = output_dtype

        data_offset = int(data_offset)
        if data_offset < 0:
            raise ValueError('data_offset must be a non-negative integer. Got {}'.format(data_offset))
        self._data_offset = int(data_offset)

        self._shape = (int(data_size[0]), self._raw_bands, int(data_size[1]))

        if isinstance(data_input, numpy.ndarray):
            self._memory_map = data_input
            self._file_name = None
            self._file_object = None
            # NB: we assume the the rest of the data is set up properly
        elif is_file_like(data_input):
            self._file_object = data_input
            self._close_after = False
            self._file_name = data_input.name if hasattr(data_input, 'name') else None

            if hasattr(data_input, 'fileno'):
                # noinspection PyBroadException
                try:
                    self._memory_map = numpy.memmap(data_input,
                                                    dtype=raw_dtype,
                                                    mode='r',
                                                    offset=data_offset,
                                                    shape=self._shape)
                except Exception:
                    # fall back to direct reading
                    self._memory_map = None
            else:
                self._memory_map = None
        elif isinstance(data_input, str):
            if not os.path.isfile(data_input):
                raise SarpyIOError('Path {} either does not exists, or is not a file.'.format(data_input))
            if not os.access(data_input, os.R_OK):
                raise SarpyIOError('User does not appear to have read access for file {}.'.format(data_input))
            self._file_name = data_input
            try:
                self._memory_map = numpy.memmap(self._file_name,
                                                dtype=raw_dtype,
                                                mode='r',
                                                offset=data_offset,
                                                shape=self._shape)
                self._file_object = None
                self._close_after = False
            except (OverflowError, OSError):
                # fallback after failing to construct the mem map
                #   the most likely cause to be here is that 32-bit python fails
                #   constructing a memmap for any file larger than 2GB
                self._file_object = open(self._file_name, mode='rb')
                self._close_after = True
                logger.warning(
                    'Falling back to reading file {} manually,\n\t'
                    'instead of using numpy memmap.'.format(self._file_name))
        self._validate_limit_to_raw_bands(limit_to_raw_bands)

    def _validate_limit_to_raw_bands(self, limit_to_raw_bands):
        limit_to_raw_bands = _validate_limit_to_raw_bands(limit_to_raw_bands, self._raw_bands)
        self._limit_to_raw_bands = limit_to_raw_bands

    def __del__(self):
        if not self._close_after:
            return
        if self._file_object is not None and \
                hasattr(self._file_object, 'closed') and \
                not self._file_object.closed:
            self._file_object.close()

    def _read_raw_fun(self, range1, range2):
        t_range1, t_range2 = self._reorder_arguments(range1, range2)
        if self._memory_map is not None:
            out = self._read_memory_map(t_range1, t_range2)
        else:
            out = self._read_file(t_range1, t_range2)
        return numpy.copy(numpy.transpose(out, (0, 2, 1)))

    def _read_memory_map(self, range1, range2):
        if (range1[1] == -1 and range1[2] < 0) and (range2[1] == -1 and range2[2] < 0):
            if self._limit_to_raw_bands is None:
                out = numpy.array(
                    self._memory_map[range1[0]::range1[2], :, range2[0]::range2[2]],
                    dtype=self._raw_dtype)
            else:
                out = numpy.array(
                    self._memory_map[range1[0]::range1[2], self._limit_to_raw_bands, range2[0]::range2[2]],
                    dtype=self._raw_dtype)
        elif range1[1] == -1 and range1[2] < 0:
            if self._limit_to_raw_bands is None:
                out = numpy.array(
                    self._memory_map[range1[0]::range1[2], :, range2[0]:range2[1]:range2[2]],
                    dtype=self._raw_dtype)
            else:
                out = numpy.array(
                    self._memory_map[range1[0]::range1[2], self._limit_to_raw_bands, range2[0]:range2[1]:range2[2]],
                    dtype=self._raw_dtype)
        elif range2[1] == -1 and range2[2] < 0:
            if self._limit_to_raw_bands is None:
                out = numpy.array(
                    self._memory_map[range1[0]:range1[1]:range1[2], :, range2[0]::range2[2]],
                    dtype=self._raw_dtype)
            else:
                out = numpy.array(
                    self._memory_map[range1[0]:range1[1]:range1[2], self._limit_to_raw_bands, range2[0]::range2[2]],
                    dtype=self._raw_dtype)
        else:
            if self._limit_to_raw_bands is None:
                out = numpy.array(
                    self._memory_map[range1[0]:range1[1]:range1[2], :, range2[0]:range2[1]:range2[2]],
                    dtype=self._raw_dtype)
            else:
                out = numpy.array(
                    self._memory_map[range1[0]:range1[1]:range1[2], self._limit_to_raw_bands, range2[0]:range2[1]:range2[2]],
                    dtype=self._raw_dtype)
        return out

    def _read_file(self, range1, range2):
        if self._limit_to_raw_bands is None:
            band_collection = numpy.arange(self._raw_bands, dtype='int32')
        else:
            band_collection = self._limit_to_raw_bands

        init_location = self._file_object.tell()
        # we have to manually map out the stride and all that for the array ourselves
        element_size = int(numpy.dtype(self._raw_dtype).itemsize)
        stride = element_size*int(self._shape[2])*self._raw_bands  # how much to skip a whole (real) row?
        # let's determine the specific row/column arrays that we are going to read
        dim1array = numpy.arange(*range1)
        dim2array = numpy.arange(*range2)
        # determine the contiguous chunk to read
        start_row = min(dim1array[0], dim1array[-1])
        rows = abs(range1[1] - range1[0])
        # allocate our output array
        out = numpy.empty((len(dim1array), len(band_collection), len(dim2array)), dtype=self._raw_dtype)

        # seek to the proper start location
        start_loc = self._data_offset + start_row*stride
        self._file_object.seek(start_loc, os.SEEK_SET)
        # read our data, then cast and reduce
        total_entries = int(rows)*self._shape[2]*self._raw_bands
        total_size = self._shape[2]*stride
        data = self._file_object.read(total_size)
        data = numpy.frombuffer(data, self._raw_dtype, total_entries)
        data = numpy.reshape(data, (rows, self._raw_bands, self._shape[2]))
        out[:, :, :] = data[range1[0]-start_row:range1[1]-start_row:range1[2], band_collection, range2[0]:range2[1]:range2[2]]
        self._file_object.seek(init_location, os.SEEK_SET)
        return out


############
# concrete writing chipper

class BIPWriter(AbstractWriter):
    """
    For writing the SICD data into the NITF container. This is abstracted generally
    because an array of these writers is used for multi-image segment NITF files.
    That is, SICD with enough rows/columns.
    """

    __slots__ = (
        '_raw_dtype', '_transform_data', '_data_offset',
        '_shape', '_memory_map', '_fid')

    def __init__(self, file_name, data_size, raw_dtype, output_bands, transform_data, data_offset=0):
        """
        For writing the SICD data into the NITF container. This is abstracted generally
        because an array of these writers is used for multi-image segment NITF files.
        That is, SICD with enough rows/columns.

        Parameters
        ----------
        file_name : str
            the file_name
        data_size : tuple
            the shape of the form (rows, cols)
        raw_dtype : str|numpy.dtype|numpy.number
            the underlying data type of the output data. Specify endianess here if necessary.
        output_bands : int
            The number of output bands written to the file.
        transform_data : callable|str
            For complex type handling.

            * If callable, then this is expected to transform the complex data
              to the raw data. A ValueError will be raised if the data type of
              the output doesn't match `raw_dtype`. By the sicd standard,
              `raw_dtype` should be int16 or uint8.

            * If `COMPLEX`, then the data is dtype complex64 or complex128, and will
              be written out to raw after appropriate manipulation. This requires
              that `raw_dtype` is float32 - for the sicd standard.

            * Otherwise, the then data will be written directly to raw. A ValueError
              will be raised if the data type of the data to be written doesn't
              match `raw_dtype`.
        data_offset : int
            byte offset from the start of the file at which the data actually starts
        """

        super(BIPWriter, self).__init__(file_name)
        if not isinstance(data_size, tuple):
            data_size = tuple(data_size)

        self._transform_data = validate_transform_data(transform_data)

        if len(data_size) != 2:
            raise ValueError(
                'The data_size parameter must have length 2, and got {}.'.format(data_size))
        data_size = (int(data_size[0]), int(data_size[1]))
        for i, entry in enumerate(data_size):
            if entry <= 0:
                raise ValueError('Entries {} of data_size is {}, but must be strictly positive.'.format(i, entry))
        output_bands = int(output_bands)
        if output_bands < 1:
            raise ValueError('output_bands must be a positive integer.')
        self._shape = (data_size[0], data_size[1], output_bands)

        self._raw_dtype = numpy.dtype(raw_dtype)

        if self._transform_data == 'COMPLEX' and self._raw_dtype.name != 'float32':
            raise ValueError(
                'transform_data = `COMPLEX`, which requires that data for writing has '
                'dtype complex64/128, and output is written as float32 (raw_dtype). '
                'raw_dtype is given as {}.'.format(raw_dtype))
        if callable(self._transform_data) and self._raw_dtype.name not in ('uint8', 'int16'):
            raise ValueError(
                'transform_data is callable, which requires that dtype complex64/128, '
                'and output is written as uint8 or uint16. '
                'raw_dtype is given as {}.'.format(self._raw_dtype.name))

        self._data_offset = int(data_offset)

        self._memory_map = None
        self._fid = None
        try:
            self._memory_map = numpy.memmap(self._file_name,
                                            dtype=self._raw_dtype,
                                            mode='r+',
                                            offset=self._data_offset,
                                            shape=self._shape)
        except (OverflowError, OSError):
            # if 32-bit python, then we'll fail for any file larger than 2GB
            # we fall-back to a slower version of reading manually
            self._fid = open(self._file_name, mode='r+b')
            logger.warning(
                'Falling back to writing file {} manually (instead of using mem-map).\n\t'
                'This has almost certainly occurred because you are 32-bit python\n\t'
                'to try to read (portions of) a file which is larger than 2GB.'.format(
                    self._file_name))

    def write_chip(self, data, start_indices=(0, 0)):
        self.__call__(data, start_indices=start_indices)

    def __call__(self, data, start_indices=(0, 0)):
        """
        Write the specified data.

        Parameters
        ----------
        data : numpy.ndarray
        start_indices : tuple

        Returns
        -------
        None
        """

        # NB: it is expected that start-indices has been validate before getting here
        if not isinstance(data, numpy.ndarray):
            raise TypeError('Requires data is a numpy.ndarray, got {}'.format(type(data)))

        start1, stop1 = start_indices[0], start_indices[0] + data.shape[0]
        start2, stop2 = start_indices[1], start_indices[1] + data.shape[1]

        # make sure we are using the proper data ordering
        if not data.flags.c_contiguous:
            data = numpy.ascontiguousarray(data)

        if self._transform_data is None:
            if data.dtype.name != self._raw_dtype.name:
                raise ValueError(
                    'Writer expects data type {}, and got data of type {}.'.format(self._raw_dtype, data.dtype))
            self._call(start1, stop1, start2, stop2, data)
        elif callable(self._transform_data):
            new_data = self._transform_data(data)
            if new_data.dtype.name != self._raw_dtype.name:
                raise ValueError(
                    'Writer expects data type {}, and got data of type {} from the '
                    'callable method transform_data.'.format(self._raw_dtype, new_data.dtype))
            self._call(start1, stop1, start2, stop2, new_data)
        else:  # transform_data is True
            if data.dtype.name not in ('complex64', 'complex128'):
                raise ValueError(
                    'Writer expects data type {}, and got data of type {} from the '
                    'callable method transform_data.'.format(self._raw_dtype, data.dtype))
            if data.dtype.name != 'complex64':
                data = data.astype(numpy.complex64)

            data_view = data.view(numpy.float32).reshape((data.shape[0], data.shape[1], 2))
            self._call(start1, stop1, start2, stop2, data_view)

    def _call(self, start1, stop1, start2, stop2, data):
        if self._memory_map is not None:
            if data.ndim == 2:
                self._memory_map[start1:stop1, start2:stop2] = data[:, :, numpy.newaxis]
            else:
                self._memory_map[start1:stop1, start2:stop2] = data
            return

        # we have to fall-back to manually write
        element_size = int(self._raw_dtype.itemsize)
        if len(self._shape) == 3:
            element_size *= int(self._shape[2])
        stride = element_size*int(self._shape[0])
        # go to the appropriate spot in the file for first entry
        self._fid.seek(self._data_offset + stride*start1 + element_size*start2, os.SEEK_SET)
        if start1 == 0 and stop1 == self._shape[0]:
            # we can write the block all at once
            data.astype(self._raw_dtype).tofile(self._fid)
        else:
            # have to write one row at a time
            bytes_to_skip_per_row = element_size*(self._shape[0]-(stop1-start1))
            for i, row in enumerate(data):
                # we the row, and then skip to where the next row starts
                row.astype(self._raw_dtype).tofile(self._fid)
                if i < len(data) - 1:
                    # don't seek on last entry (avoid segfault, or whatever)
                    self._fid.seek(bytes_to_skip_per_row, os.SEEK_CUR)

    def close(self):
        """
        **Should be called on exit.** Cleanly close the file. This is actually only
        required if memory map failed, and we fell back to manually writing the file.

        Returns
        -------
        None
        """

        if hasattr(self, '_fid') and self._fid is not None and \
                hasattr(self._fid, 'closed') and not self._fid.closed:
            self._fid.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is None:
            self.close()
        else:
            logger.error(
                'The {} file writer generated an exception during processing.\n\t'
                'The file {} may be only partially generated and corrupt.'.format(
                    self.__class__.__name__, self._file_name))
            # The exception will be reraised.
            # It's unclear how any exception could be caught.


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
