# -*- coding: utf-8 -*-
"""
This provides implementation of Reading and writing capabilities for files with
data stored in *Band Interleaved By Pixel (BIP)* format.
"""

import logging
import os

import numpy

from sarpy.compliance import int_func, integer_types
from sarpy.io.general.base import BaseChipper, AbstractWriter, validate_transform_data


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BIPChipper(BaseChipper):
    """
    Band interleaved format file chipper.
    """

    __slots__ = (
        '_file_name', '_raw_dtype', '_data_offset', '_shape', '_raw_bands',
        '_output_bands', '_output_dtype', '_limit_to_raw_bands', '_memory_map', '_fid')

    def __init__(self, file_name, raw_dtype, data_size, raw_bands, output_bands, output_dtype,
                 symmetry=(False, False, False), transform_data=None,
                 data_offset=0, limit_to_raw_bands=None):
        """

        Parameters
        ----------
        file_name : str
            The name of the file from which to read
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
        super(BIPChipper, self).__init__(data_size, symmetry=symmetry, transform_data=transform_data)

        raw_bands = int_func(raw_bands)
        if raw_bands < 1:
            raise ValueError('raw_bands must be a positive integer')
        self._raw_bands = raw_bands
        self._validate_limit_to_raw_bands(limit_to_raw_bands)

        output_bands = int_func(output_bands)
        if output_bands < 1:
            raise ValueError('output_bands must be a positive integer.')
        self._output_bands = output_bands

        self._raw_dtype = raw_dtype
        self._output_dtype = output_dtype

        data_offset = int_func(data_offset)
        if data_offset < 0:
            raise ValueError('data_offset must be a non-negative integer. Got {}'.format(data_offset))
        self._data_offset = int_func(data_offset)

        self._shape = (int_func(data_size[0]), int_func(data_size[1]), self._raw_bands)

        if not os.path.isfile(file_name):
            raise IOError('Path {} either does not exists, or is not a file.'.format(file_name))
        if not os.access(file_name, os.R_OK):
            raise IOError('User does not appear to have read access for file {}.'.format(file_name))
        self._file_name = file_name

        self._memory_map = None
        self._fid = None
        try:
            self._memory_map = numpy.memmap(self._file_name,
                                            dtype=raw_dtype,
                                            mode='r',
                                            offset=data_offset,
                                            shape=self._shape)  # type: numpy.memmap
        except (OverflowError, OSError):
            if self._limit_to_raw_bands is not None:
                raise ValueError(
                    'Unsupported effort with limit_to_raw_bands is not None, and falling '
                    'back to a manual file reading. This is presumably because this is '
                    '32-bit python and you are reading a large (> 2GB) file.')
            # if 32-bit python, then we'll fail for any file larger than 2GB
            # we fall-back to a slower version of reading manually
            self._fid = open(self._file_name, mode='rb')
            logging.warning(
                'Falling back to reading file {} manually (instead of using mem-map). This has almost '
                'certainly occurred because you are 32-bit python to try to read (portions of) a file '
                'which is larger than 2GB.'.format(self._file_name))

    def _validate_limit_to_raw_bands(self, limit_to_raw_bands):
        if limit_to_raw_bands is None:
            self._limit_to_raw_bands = None
            return

        if isinstance(limit_to_raw_bands, integer_types):
            limit_to_raw_bands = numpy.array([limit_to_raw_bands, ], dtype='int32')
        if isinstance(limit_to_raw_bands, (list, tuple)):
            limit_to_raw_bands = numpy.array(limit_to_raw_bands, dtype='int32')
        if not isinstance(limit_to_raw_bands, numpy.ndarray):
            raise TypeError('limit_to_raw_bands got unsupported input of type {}'.format(type(limit_to_raw_bands)))
        # ensure that limit_to_raw_bands make sense...
        if numpy.any((limit_to_raw_bands < 0) | (limit_to_raw_bands >= self._raw_bands)):
            raise ValueError(
                'all entries of limit_to_raw_bands ({}) must be in the range 0 <= value < {}'.format(limit_to_raw_bands, self._raw_bands))
        self._limit_to_raw_bands = limit_to_raw_bands

    def __del__(self):
        if hasattr(self, '_fid') and self._fid is not None and \
                hasattr(self._fid, 'closed') and not self._fid.closed:
            self._fid.close()

    def _read_raw_fun(self, range1, range2):
        t_range1, t_range2 = self._reorder_arguments(range1, range2)
        if self._memory_map is not None:
            return self._read_memory_map(t_range1, t_range2)
        elif self._fid is not None:
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
        def get_row_location(rr, cc):
            return self._data_offset + \
                   rr*stride + \
                   cc*element_size

        # we have to manually map out the stride and all that for the array ourselves
        element_size = int_func(numpy.dtype(self._raw_dtype).itemsize*self._raw_bands)
        stride = element_size*int_func(self._shape[0])  # how much to skip a whole (real) row?
        entries_per_row = abs(range1[1] - range1[0])  # not including the stride, if not +/-1
        # let's determine the specific row/column arrays that we are going to read
        dim1array = numpy.arange(range1)
        dim2array = numpy.arange(range2)
        # allocate our output array
        out = numpy.empty((len(dim1array), len(dim2array), self._raw_bands), dtype=self._raw_dtype)
        # determine the first column reading location (may be reading cols backwards)
        col_begin = dim2array[0] if range2[2] > 0 else dim2array[-1]

        for i, row in enumerate(dim1array):
            # go to the appropriate point in the file for (row/col)
            self._fid.seek(get_row_location(row, col_begin))
            # interpret this of line as numpy.ndarray - inherently flat array
            line = numpy.fromfile(self._fid, self._raw_dtype, entries_per_row*self._raw_bands)
            # note that we purposely read without considering skipping elements, which
            #   is factored in (along with any potential order reversal) below
            out[i, :, :] = line[::range2[2]]
        return out


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
        data_size = (int_func(data_size[0]), int_func(data_size[1]))
        for i, entry in enumerate(data_size):
            if entry <= 0:
                raise ValueError('Entries {} of data_size is {}, but must be strictly positive.'.format(i, entry))
        output_bands = int_func(output_bands)
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

        self._data_offset = int_func(data_offset)

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
            logging.warning(
                'Falling back to writing file {} manually (instead of using mem-map). This has almost '
                'certainly occurred because you are 32-bit python to try to read (portions of) a file '
                'which is larger than 2GB.'.format(self._file_name))

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
        element_size = int_func(self._raw_dtype.itemsize)
        if len(self._shape) == 3:
            element_size *= int_func(self._shape[2])
        stride = element_size*int_func(self._shape[0])
        # go to the appropriate spot in the file for first entry
        self._fid.seek(self._data_offset + stride*start1 + element_size*start2)
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
            logging.error(
                'The {} file writer generated an exception during processing. The file {} may be '
                'only partially generated and corrupt.'.format(self.__class__.__name__, self._file_name))
            # The exception will be reraised.
            # It's unclear how any exception could be caught.
