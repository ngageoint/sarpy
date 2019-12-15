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
    def __init__(self, file_name, data_type, data_size, symmetry=(False, False, False),
                 complex_type=False, data_offset=0, swap_bytes=False, bands_ip=1):
        """

        Parameters
        ----------
        filename : str
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
        swap_bytes : bool
            are the endian-ness of the os and file different?
        bands_ip : int
            number of bands - really intended for complex data
        """

        super(BIPChipper, self).__init__(data_size, symmetry=symmetry, complex_type=complex_type)

        bands = int(bands_ip)
        if self._complex_type is not False:
            bands *= 2

        true_shape = self._data_size + (bands, )
        data_offset = int(data_offset)

        # try to set up a memory map. If this fails, default to pure file reading.
        self._swap_bytes = bool(swap_bytes)

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
        if self._swap_bytes:
            data.byteswap(True)  # this swaps bytes in place
        return data.transpose((2, 0, 1))  # switches from band interleaved to band sequential


class BIPWriter(object):
    """Work in progress carried over from legacy."""
    # TODO: what should be improved here?
    #   This is much more limited than the reader.

    def __init__(self, file_name, data_size, data_type, complex_type, data_offset=0):
        # no swap_bytes argument?

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
        self._complex_type = bool(complex_type)
        self._data_offset = int(data_offset)
        self._file_name = file_name
        if self._complex_type:
            self._shape = (self._data_size[0], self._data_size[1]*2)
        else:
            self._shape = self._data_size

        self._memory_map = numpy.memmap(self._file_name,
                                        dtype=self._data_type,
                                        mode='r+',
                                        offset=self._data_offset,
                                        shape=self._shape)

    def __call__(self, data, start_indices=(0, 0)):
        """

        Parameters
        ----------
        data : numpy.ndarray
        start_indices : tuple

        Returns
        -------
        None
        """
        # TODO: no strides?

        if self._complex_type:
            start_indices = (start_indices[0], 2*start_indices[1])
            if data.dtype.name != 'complex64':
                data = data.astype(numpy.complex64)
            if not data.flags.c_contiguous:
                data = numpy.ascontiguousarray(data)  # can't memory map otherwise?
            data_view = data.view(numpy.float32)  # now shape = (rows, cols, 2)
        else:
            data_view = data.view()
        start1, stop1 = start_indices[0], start_indices[0]+data_view.shape[0]
        start2, stop2 = start_indices[1], start_indices[1]+data_view.shape[1]
        self._memory_map[start1:stop1, start2:stop2] = data_view
