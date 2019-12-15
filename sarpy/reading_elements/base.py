# -*- coding: utf-8 -*-
"""
The base elements for reading and writing files as appropriate.

** It is expected that this is not the final location for these files. **
"""

import numpy

__classification__ = "UNCLASSIFIED"


class BaseChipper(object):
    """
    Abstract base class defining basic functionality for the literal extraction of data
    from a file. The intent of this class is to be a callable in the following form:
    .. code-block:
        data = BaseChipper(entry1, entry2)
    where each entry is a tuple or int of the form `[[start], stop,] step`.
    Similarly, we are able to use more traditional Python slicing syntax
    .. code-block:
        data = BaseChipper[slice1[, slice2]]

    **Extension Requirement:** This provides the basic implementation for the work
    flow, but it is **required** that any extension provide a concrete implementation
    for actually reading from the raw file in `read_raw_fun`.

    **Extension Consideration:** It is possible that the basic functionality for
    conversion of raw data to complex data requires something more nuanced than
    the default provided in the `_data_to_complex` method.
    """

    _data_size = None
    _complex_type = None
    _symmetry = (False, False, False)

    def __init__(self, data_size, symmetry=None, complex_type=None):
        """

        Parameters
        ----------
        data_size : tuple
            The full size of the data *after* any required transformation. See
            `data_size` property.
        symmetry : tuple
            Describes any required data transformation. See the `symmetry` property.
        complex_type : callable|bool
            For complex type handling.
            If callable, then this is expected to transform the raw data to the complex data.
            If this evaluates to `True`, then the assumption is that real/imaginary
            components are stored in adjacent bands, which will be combined into a
            single band upon extraction.
        """

        self._complex_type = complex_type

        if not isinstance(data_size, tuple):
            data_size = tuple(data_size)
        if len(data_size) != 2:
            raise ValueError(
                'The data_size parameter must have length 2, and got {}.'.format(data_size))
        data_size = (int(data_size[0]), int(data_size[1]))
        if data_size[0] < 0 or data_size[1] < 0:
            raise ValueError('All entries of data_size {} must be non-negative.'.format(data_size))
        self._data_size = data_size

        if not isinstance(symmetry, tuple):
            symmetry = tuple(symmetry)
        if len(symmetry) != 3:
            raise ValueError(
                'The symmetry parameter must have length 3, and got {}.'.format(symmetry))
        junk = tuple([bool(entry) for entry in symmetry])
        self._symmetry = junk

    @property
    def symmetry(self):
        """
        tuple: with boolean entries of the form (`flip1`, `flip2`, `swapaxes`).
        This describes necessary symmetry transformation to be performed to convert
        from raw (file storage) order into the order expected (analysis order).

        * `flip1=True` - we reverse order in the first axis, wrt raw order.

        * `flip2=True` - we reverse order in the second axis, wrt raw order).

        * `swapaxes=True` - we switch the two axes, after any required flipping.
        """

        return self._symmetry

    @property
    def data_size(self):
        """
        tuple: Two element tuple of the form `(rows, columns)`, which provides the
        size of the data, after any necessary symmetry transformations have been applied.
        Note that this excludes the number of bands in the image.
        """

        return self._data_size

    @property
    def shape(self):
        if self._symmetry[2]:
            return self._data_size[::-1]
        else:
            return self._data_size

    def __call__(self, range1, range2):
        data = self._read_raw_fun(range1, range2)
        data = self._data_to_complex(data)

        # make a one band image flat
        if data.ndim == 3 and data.shape[0] == 1:
            data = numpy.reshape(data, data.shape[1:])

        data = self._reorder_data(data)
        return data

    def __getitem__(self, item):
        range1, range2 = self._slice_to_args(item)
        return self.__call__(range1, range2)

    @staticmethod
    def _slice_to_args(item):
        def parse(entry):
            if isinstance(entry, int):
                return item, item+1, None
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
            * if `int` = step size
            * if (`int`, `int`) = `end`, `step size`
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

        def extract(arg, siz):
            start, stop, step = None, None, None
            if isinstance(arg, int):
                step = arg
            elif len(arg) == 1:
                step = arg[0]
            elif len(arg) == 2:
                stop, step = arg
            elif len(arg) == 3:
                start, stop, step = arg
            start = 0 if start is None else int(start)
            stop = siz if stop is None else int(stop)
            step = 1 if step is None else int(step)
            # basic validity check
            if not (-siz < start < siz):
                raise ValueError(
                    'Range argument {} has extracted start {}, which is required '
                    'to be in the range [0, {})'.format(arg, start, siz))
            if not (-siz < stop <= siz):
                raise ValueError(
                    'Range argument {} has extracted stop {}, which is required '
                    'to be in the range [0, {}]'.format(arg, start, siz))
            if not ((0 < step < siz) or (-siz < step < 0)):
                raise ValueError(
                    'Range argument {} has extracted step {}, for an axis of length '
                    '{}'.format(arg, start, siz))
            if ((step < 0) and (stop > start)) or ((step > 0) and (start > stop)):
                raise ValueError(
                    'Range argument {} has extracted start {}, stop {}, step {}, '
                    'which is not valid.'.format(arg, start, stop, step))

            # reform negative values for start/stop appropriately
            if start < 0:
                start += siz
            if stop < 0:
                stop += siz
            return start, stop, step

        def reverse_arg(arg, siz):
            start, stop, step = extract(arg, siz)
            # read backwards
            return (siz - 1) - start, (siz - 1) - stop, -step

        if isinstance(range1, (numpy.ndarray, list)):
            range1 = tuple(range1)
        if isinstance(range2, (numpy.ndarray, list)):
            range2 = tuple(range2)

        if not (range1 is None or isinstance(range1, (int, tuple))):
            raise TypeError('range1 is of type {}, but must be an instance of None, '
                            'int or tuple.'.format(range1))
        if not (range2 is None or isinstance(range2, (int, tuple))):
            raise TypeError('range2 is of type {}, but must be an instance of None, '
                            'int or tuple.'.format(range2))
        if isinstance(range1, tuple) and len(range1) > 3:
            raise TypeError('range1 must have no more than 3 entries, received {}.'.format(range1))
        if isinstance(range2, tuple) and len(range2) > 3:
            raise TypeError('range2 must have no more than 3 entries, received {}.'.format(range2))

        # switch the axes if necessary
        real_arg1, real_arg2 = (range2, range1) if self._symmetry[2] else (range1, range2)
        # validate the first range
        if self._symmetry[0]:
            real_arg1 = reverse_arg(real_arg1, self._data_size[0])
        else:
            real_arg1 = extract(real_arg1, self._data_size[0])
        # validate the second range
        if self._symmetry[1]:
            real_arg2 = reverse_arg(real_arg2, self._data_size[1])
        else:
            real_arg2 = extract(real_arg2, self._data_size[1])
        return real_arg1, real_arg2

    def _data_to_complex(self, data):
        if callable(self._complex_type):
            return self._complex_type(data)  # is this actually necessary?
        if self._complex_type:
            # TODO: complex128 or complex64?
            out = numpy.zeros((data.shape(0)/2, data.shape(1), data.shape(2)), dtype=numpy.complex128)
            out.real = data[0::2, :, :]
            out.imag = data[1::2, :, :]
            return out
        return data

    def _reorder_data(self, data):
        if self._symmetry[2]:
            data = numpy.swapaxes(data, data.ndim-1, data.ndim-2)
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
            * if `int` = step size
            * if (`int`, `int`) = `end`, `step size`
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
    """Permits transparent extraction from a particular subset of the possible data range"""

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
        super(SubsetChipper, self).__init__(data_size, symmetry=parent_chipper.symmetry)

    def _data_to_complex(self, data):
        return self.parent_chipper._data_to_complex(data)

    def _read_raw_fun(self, range1, range2):
        arange1 = (range1[0] + self.shift1, range1[1] + self.shift1, range1[2])
        arange2 = (range2[0] + self.shift2, range2[1] + self.shift2, range2[2])
        return self.parent_chipper._read_raw_fun(arange1, arange2)


class BaseReader(object):
    """Abstract file reader class"""
    _sicd_meta = None

    # TODO: establish more generic capability?

    def read_chip(self, dim1range, dim2range):
        # TODO: document
        raise NotImplementedError


class BaseWriter(object):
    """Abstract file writer class"""
    _sicd_meta = None

    # TODO: establish more generic capability?

    def write_chip(self, data, start_indices):
        # TODO: document
        raise NotImplementedError
