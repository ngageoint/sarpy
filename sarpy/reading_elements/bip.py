# -*- coding: utf-8 -*-
"""
This provides implementation of Reading and writing capabilities for files with
data stored in *Band Interleaved By Pixel (BIP)* format.
"""

import os

import numpy

from .base import BaseChipper


__classification__ = "UNCLASSIFIED"


class BIPChipper(BaseChipper):
    __slots__ = ('_file_name', '_data_size', '_complex_type', '_symmetry', '_memory_map')

    def __init__(self, file_name, data_type, data_size,
                 symmetry=(False, False, False), complex_type=False,
                 data_offset=0, bands_ip=1):
        """

        Parameters
        ----------
        file_name : str
            The name of the file from which to read
        data_type : numpy.dtype
            The data type of the underlying file
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
        data_offset : int
            byte offset from the start of the file at which the data actually starts
        bands_ip : int
            number of bands - really intended for complex data
        """

        super(BIPChipper, self).__init__(data_size, symmetry=symmetry, complex_type=complex_type)

        bands = int(bands_ip)
        if self._complex_type is not False:
            bands *= 2

        true_shape = self._data_size + (bands, )
        data_offset = int(data_offset)

        if not os.path.isfile(file_name):
            raise IOError('Path {} either does not exists, or is not a file.'.format(file_name))
        if not os.access(file_name, os.R_OK):
            raise IOError('User does not appear to have read access for file {}.'.format(file_name))
        self._file_name = file_name

        # NOTE: does holding a memory map open come with any penalty?
        #   should we just open on read_raw_fun?
        self._memory_map = numpy.memmap(self._file_name,
                                        dtype=data_type,
                                        mode='r',
                                        offset=data_offset,
                                        shape=true_shape)

    def _read_raw_fun(self, range1, range2):
        range1, range2 = self._reorder_arguments(range1, range2)
        # just read the data from the memory mapped region
        data = numpy.array(self._memory_map[range1[0]:range1[1]:range1[2], range2[0]:range2[1]:range2[2]])
        return data.transpose((2, 0, 1))  # switches from band interleaved to band sequential


class BIPWriter(object):
    """
    For writing the SICD data into the NITF container. This is abstracted generally
    because an array of these writers is used for multi-image segment NITF files.
    That is, SICD with enough rows/columns.
    """

    def __init__(self, file_name, data_size, data_type, complex_type, data_offset=0):
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
        data_type : numpy.dtype
            the underlying data type of the output data.
        complex_type : callable|bool
            For complex type handling.

            * If callable, then this is expected to transform the complex data
              to the raw data. A ValueError will be raised if the data type of
              the output doesn't match `data_type`. By the sicd standard,
              `data_type` should be int16 or uint8.

            * If `True`, then the data is dtype complex64 or complex128, and will
              be written out to raw after appropriate manipulation. This requires
              that `data_type` is float32 - for the sicd standard.

            * If `False`, the then data will be written directly to raw. A ValueError
              will be raised if the data type of the data to be written doesn't
              match `data_type`.
        data_offset : int
            byte offset from the start of the file at which the data actually starts
        """

        if not isinstance(data_size, tuple):
            data_size = tuple(data_size)
        if len(data_size) != 2:
            raise ValueError(
                'The data_size parameter must have length 2, and got {}.'.format(data_size))
        data_size = (int(data_size[0]), int(data_size[1]))
        if data_size[0] < 0 or data_size[1] < 0:
            raise ValueError('All entries of data_size {} must be non-negative.'.format(data_size))
        self._data_size = data_size

        self._data_type = numpy.dtype(data_type)
        if not (isinstance(complex_type, bool) or callable(complex_type)):
            raise ValueError('complex-type must be a boolean or a callable')
        self._complex_type = complex_type

        if self._complex_type is True:
            raise ValueError(
                'complex_type = `True`, which requires that data for writing has '
                'dtype complex64/128, and output is written as float32 (data_type). '
                'data_type is given as {}.'.format(data_type))
        if callable(self._complex_type) and self._data_type not in (numpy.uint8, numpy.int16):
            raise ValueError(
                'complex_type is callable, which requires that dtype complex64/128, '
                'and output is written as uint8 or uint16. '
                'data_type is given as {}.'.format(data_type))

        self._file_name = file_name
        self._data_offset = int(data_offset)
        if self._complex_type is False:
            self._shape = self._data_size
        else:
            self._shape = (self._data_size[0], self._data_size[1], 2)

        self._memory_map = numpy.memmap(self._file_name,
                                        dtype=self._data_type,
                                        mode='r+',
                                        offset=self._data_offset,
                                        shape=self._shape)

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

        if not isinstance(data, numpy.ndarray):
            raise TypeError('Requires data is a numpy.ndarray, got {}'.format(type(data)))

        start1, stop1 = start_indices[0], start_indices[0] + data.shape[0]
        start2, stop2 = start_indices[1], start_indices[1] + data.shape[1]

        # make sure we are using the proper data ordering for memory map
        if not data.flags.c_contiguous:
            data = numpy.ascontiguousarray(data)

        if self._complex_type is False:
            if data.dtype != self._data_type:
                raise ValueError(
                    'Writer expects data type {}, and got data of type {}.'.format(self._data_type, data.dtype))
            self._memory_map[start1:stop1, start2:stop2] = data
        elif callable(self._complex_type):
            new_data = self._complex_type(data)
            if new_data.dtype != self._data_type:
                raise ValueError(
                    'Writer expects data type {}, and got data of type {} from the '
                    'callable method complex_type.'.format(self._data_type, new_data.dtype))
            self._memory_map[start1:stop1, start2:stop2, :] = new_data
        else:  # complex_type is True
            if data.dtype not in (numpy.complex64, numpy.complex128):
                raise ValueError(
                    'Writer expects data type {}, and got data of type {} from the '
                    'callable method complex_type.'.format(self._data_type, data.dtype))
            if data.dtype != numpy.complex64:
                data = data.astype(numpy.complex64)

            data_view = data.view(numpy.float32).reshape((data.shape[0], data.shape[1], 2))
            self._memory_map[start1:stop1, start2:stop2, :] = data_view
