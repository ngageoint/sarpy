"""This module has utilities to read and write files stored band interleaved by pixel (BIP)."""

import numpy as np
import os
from . import chipper

__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"
__email__ = "wschwartzkopf@integrity-apps.com"


class Chipper(chipper.Base):
    """Chipper function object for files band interleaved by pixel (BIP)."""
    def __init__(self, filename, datasize, datatype, complextype,  # Required params
                 data_offset=0,  # Start of data in bytes from start of file
                 swapbytes=False,   # Is reading endian same as file endian
                 symmetry=(False, False, False),  # Assume no reorientation
                 bands_ip=1):  # This means bands of complex data (if data is complex)
        # Easier way to copy arguments to class attributes in Python?
        self.complextype = complextype
        self.datasize = datasize
        self.symmetry = symmetry
        if symmetry[2]:
            self.shape = tuple(datasize[::-1])
        else:
            self.shape = tuple(datasize)
        # Double bands for complex.   complextype could be boolean or function
        bands = bands_ip * ((complextype is not False) + 1)
        # Can we use memory mapping?
        # If we are reading and writing from/to files in order, not much
        # advantage to memory mapping.  If however, we do require random access
        # to file read/write, memory mapping can be MUCH faster than other IO.
        try:
            # NumPy's memmap is a bit finicky with its input arguments.  It does
            # not allow uint64 for offset and shape, and shape must be a tuple,
            # not a NumPy array.
            memmap = np.memmap(filename, dtype=datatype, mode='r',
                               offset=np.int64(data_offset),
                               shape=(tuple(np.int64(datasize)) +
                                      (np.int64(bands),)))
            self.read_raw_fun = lambda dim1range, dim2range: \
                read_bip_mm(memmap, swapbytes, dim1range, dim2range)
            # No need to close memory map explicitly like fid, since Python
            # garbage collection will close when self.memmap is deleted.
        except (OverflowError, OSError):  # Python's memmap is dumb.  Can't handle large files.
            # Fall back to regular file reads
            self.fid = open(filename, mode='rb')
            self.read_raw_fun = lambda dim1range, dim2range: \
                read_bip(self.fid, datasize, data_offset, datatype,
                         bands, swapbytes, dim1range, dim2range)

    def __del__(self):
        if (hasattr(self, 'fid') and hasattr(self.fid, 'closed') and
           not self.fid.closed):
            self.fid.close()


class Writer:
    """This class provides for the localized writing of bip data.  It does
    not (yet) allow for various symmetry options or bands beyond complex
    values like the Chipper class for reading bip does."""
    def __init__(self, filename, datasize, datatype, complextype,  # Required params
                 data_offset=0):  # Start of data in bytes from start of file
        self.complextype = complextype
        try:
            if complextype:
                # Allow for real/imaginary interleaving.
                mm_datasize = (datasize[0], datasize[1]*2)
            else:
                mm_datasize = datasize
            self.memmap = np.memmap(filename, dtype=datatype, mode='r+',
                                    offset=np.int64(data_offset),
                                    shape=mm_datasize)
        except:  # Python's memmap is dumb.  Can't handle large files.
            # Fall back to regular file reads
            self.fid = open(filename, mode='r+b')
            self.datasize = datasize
            self.datatype = datatype
            self.data_offset = data_offset

    def __call__(self, data, start_indices=(0, 0)):
        if self.complextype:
            if (data.dtype.name != 'complex64'):
                data = data.astype(np.dtype('complex64'))
            if not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)
            data2 = data.view(np.dtype('float32'))  # Interleaves without copying
            start_indices = (start_indices[0], 2*start_indices[1])
        else:
            data2 = data.view()
        if hasattr(self, 'memmap'):
            # Trivial.  Too bad this doesn't always work in Python.
            self.memmap[start_indices[0]:(start_indices[0]+data2.shape[0]),
                        start_indices[1]:(start_indices[1]+data2.shape[1])] = data2
        elif (hasattr(self, 'fid') and hasattr(self.fid, 'closed') and
              not self.fid.closed):
            element_size = self.datatype.itemsize * (bool(self.complextype) + 1)
            offset = (self.data_offset +  # Start of data in file
                      (element_size * int(start_indices[0]) *
                       int(self.datasize[1])) +  # First row
                      (element_size * int(start_indices[1])))  # First column
            self.fid.seek(offset)
            bytes_per_row_to_skip = element_size * (self.datasize[1] - data.shape[1])
            if bytes_per_row_to_skip == 0:  # Easiest case
                data2.astype(self.datatype).tofile(self.fid)
            else:
                # NOTE: MATLAB allows a "skip" parameter in its fwrite function.
                # This allows one to do very fast reads when subsample equals 1
                # using only a single line of code-- no loops!  Not sure of an
                # equivalent way to do this in Python, so we have to use "for"
                # loops-- yuck!
                for i in range(data.shape[0]-1):
                    data2[i, :].astype(self.datatype).tofile(self.fid)
                    self.fid.seek(bytes_per_row_to_skip, os.SEEK_CUR)
                # No seek with last write
                data2[-1, :].astype(self.datatype).tofile(self.fid)

    def __del__(self):
        if (hasattr(self, 'fid') and hasattr(self.fid, 'closed') and
           not self.fid.closed):
            self.fid.close()


def read_bip(fid, datasize, offset=0, datatype='float32', bands=1,
             swapbytes=False, dim1range=None, dim2range=None):
    """Generic function for reading data band interleaved by pixel.

    Data is read directly from disk with no transformation.  The most quickly
    incresing dimension on disk will be the most quickly increasing dimension in
    the array in memory.  No assumptions are made as to what the bands
    represent (complex i/q, etc.)

    INPUTS:
       fid: File identifier from open().  Must refer to a file that is open for
          reading as binary.
       datasize: 1x2 tuple/list (number of elements in first dimension, number
          of elements in the second dimension).  In keeping with the Python
          standard, the second dimension is the more quickly increasing as
          written in the file.
       offset: Index (in bytes) from the beginning of the file to the beginning
          of the data.  Default is 0 (beginning of file).
       datatype: Data type specifying binary data precision.  Default is
          dtype('float32').
       bands: Number of bands in data.  Default is 1.
       swapbytes: Whether the "endianness" of the data matches the "endianess"
          of our file reads.  Default is False.
       dim1range: ([start, stop,] step).  Similar syntax as Python range() or
          NumPy arange() functions.  This is the range of data to read in the
          less quickly increasing dimension (as written in the file).  Default
          is entire range.
       dim2range: ([start, stop,] step).  Similar syntax as Python range() or
          NumPy arange() functions.  This is the range of data to read in the
          more quickly increasing dimension (as written in the file).  Default
          is entire range.

    OUTPUT: Array of complex data values read from file.

    """

    # Check input arguments
    datasize, dim1range, dim2range = chipper.check_args(
        datasize, dim1range, dim2range)
    offset = np.array(offset, dtype='uint64')
    if offset.size == 1:   # Second term of offset allows for line prefix/suffix
        offset = np.append(offset, np.array(0, dtype='uint64'))
    # Determine element size
    datatype = np.dtype(datatype)  # Allows caller to pass dtype or string
    elementsize = np.uint64(datatype.itemsize * bands)

    # Read data (region of interest only)
    fid.seek(offset[0] +  # Beginning of data
             (dim1range[0] * (datasize[1] * elementsize + offset[1])) +  # Skip to first row
             (dim2range[0] * elementsize))  # Skip to first column
    dim2size = dim2range[1] - dim2range[0]
    lendim1range = len(range(*dim1range))
    dataout = np.zeros((bands, lendim1range, len(range(*dim2range))), datatype)
    # NOTE: MATLAB allows a "skip" parameter in its fread function.  This allows
    # one to do very fast reads when subsample equals 1 using only a single line
    # of code-- no loops!  Not sure of an equivalent way to do this in Python,
    # so we have to use "for" loops-- yuck!
    for i in range(lendim1range):
        single_line = np.fromfile(fid, datatype, np.uint64(bands) * dim2size)
        for j in range(bands):  # Pixel intervleaved
            dataout[j, i, :] = single_line[j::dim2range[2]*np.uint64(bands)]
        fid.seek(((datasize[1] * elementsize) + offset[1]) * (dim1range[2] - np.uint64(1)) +  # Skip unread rows
                 ((datasize[1] - dim2size) * elementsize) + offset[1], 1)  # Skip to beginning of dim2range
    if swapbytes:
        dataout.byteswap(True)
    return dataout


def read_bip_mm(memmap, swapbytes=False, dim1range=None, dim2range=None):
    """Generic function for reading data band interleaved by pixel using memory-mapped IO.

    INPUTS:
        memmap: File identifier from numpy.memmap.
        swapbytes, dim1range, and dim2range: Same as read_bip
    OUTPUT: Same as read_bip

    Note this disclaimer from the NumPy documentation: "Memory-mapped arrays use
    the Python memory-map object which (prior to Python 2.5) does not allow
    files to be larger than a certain size depending on the platform. This size
    is always < 2GB even on 64-bit systems."

    """

    # Check input parameters
    datasize = memmap.shape[:2]
    datasize, dim1range, dim2range = chipper.check_args(
        datasize, dim1range, dim2range)
    # Read data (region of interest only)
    dataout = np.array(memmap[dim1range[0]:dim1range[1]:dim1range[2],
                              dim2range[0]:dim2range[1]:dim2range[2], :])
    if swapbytes:
        dataout.byteswap(True)
    # Convert from band interleaved by pixel (BIP) to band sequential (BSQ).
    # Puts bands in 1st dimension, consistent with read_bip()
    dataout = dataout.transpose((2, 0, 1))
    return dataout
