"""
The base elements for reading and writing files as appropriate.

** It is expected that this is not the final location for these files. **
"""

import numpy


class BaseChipper(object):
    # TODO: put good and informative documentation
    _complex_type = False
    _data_size = (0, 0)
    _symmetry = (False, False, False)

    def __init__(self, data_size, complex_type=None, symmetry=None):
        # TODO: is that all this should be doing?
        self._data_size = data_size
        if callable(complex_type) or isinstance(complex_type, bool):
            self._complex_type = complex_type
        if isinstance(symmetry, tuple) and len(symmetry) >= 3:
            self._symmetry = symmetry

    def __call__(self, range1, range2):
        data = self.read_raw_fun(self._reorder_arguments(range1, range2))
        data = self._data_to_complex(data)

        # make a one band image flat
        if data.ndim == 3 and data.shape[0] == 1:
            data = numpy.reshape(data, data.shape[1:])

        data = self._reorder_data(data)
        return data

    def __getitem__(self, item):
        dim1range, dim2range = self._slice_to_args(item)
        return self.__call__(dim1range, dim2range)

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
            # TODO: why not just implement it here? I think this is just total object-oriented confusion.
            return self._complex_type(data)
        elif self._complex_type:
            # TODO: 128 or 64?
            out = numpy.zeros((data.shape(0)/2, data.shape(1), data.shape(2)), dtype=numpy.complex128)
            out.real = data[0::2, :, :]
            out.imag = data[1::2, :, :]
            return out
        else:
            return data

    def _reorder_data(self, data):
        if self._symmetry[2]:
            data = numpy.swapaxes(data, data.ndim-1, data.ndim-2)
        return data

    def read_raw_fun(self, range1, range2):
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

        # TODO: why would a user want access to this? Should be private.
        raise NotImplementedError

# TODO: subset chipper - from line utils/chipper.py line 217. There's still some confusion.


class BaseReader(object):
    """Abstract file reader class"""
    _sicd_meta = None

    # TODO: establish more generic capability?

    def read_chip(self, dim1range, dim2range):
        # TODO: document
        raise NotImplementedError


class Writer(object):
    """Abstract file writer class"""
    _sicd_meta = None

    # TODO: establish more generic capability?

    def write_chip(self, data, start_indices):
        # TODO: document
        raise NotImplementedError
