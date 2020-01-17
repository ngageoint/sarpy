"""This module describes a class of callable "chipper" objects.

    A chipper function object is a simplified function from a file reader.  A
    chipper is callable and take only two arguments, the range of elements to
    take in each dimension (using a similar notation as Python's range()
    function) and return data over an AOI.  This allows us to setup a chipper
    once (possibly without a users knowledge), and then allow for multiple AOI
    reads from that file with this simple notation.

    data = chipper(([start, stop,] step), ([start, stop,] step))

    Internally a chipper contains all information such as file pointers and file
    layout so that AOI data can be easily extracted with this brief notation
    without having to pass all of the file description info every time.

"""

# TODO: Should check_args go in __call__ rather than each derived class?

import numpy as np

__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"
__email__ = "wschwartzkopf@integrity-apps.com"


class Base():
    """Abstract class from which read chipper function object can be derived.

    FlatFile is a basic implementation of a chipper function object for complex
    data.  It uses raw readers (like bip.read_bip, which return the data
    exactly as stored in a file) but adds ways to convert raw multiband data
    into complex data, as well as letting the caller define the orientation of
    the data with respect to how it is stored in a file.

    If one wanted to implement a chipper, one only has to extend this class by
    defining the read_raw_fun method and setting the complextype, datasize, and
    symmetry attributes to describe how to convert that data to complex and how
    the data is oriented in the file.

    METHODS:
        read_raw_fun: Function with the same input arguments as the chipper
            function.  However, this function returns data as stored in a file
            potentially before any complex data and symmetry transformations are
            applied.  There is one exception to the "raw" file orientation of
            the data:  In data returned from read_raw_fun, bands are always in
            the first dimension (data[n,:,:] is the nth band-- "band sequential"
            or BSQ, as stored in Python's memory), regardless of how the data is
            stored in the file.

    ATTRIBUTES:
        complextype: Whether bands returned from the read_raw_fun method parts
            of complex values (for example, I/Q).  Can be either a boolean or a
            function defining how to convert multiband data into complex values.
            True signifies bands are I/Q values.  False means no conversion
            necessary.  If another conversion from bands to complex values is
            necessary (e.g. amplitude/phase), then that can be passed in as a
            function, which takes as an argument a single array, in which the
            first dimension is the bands, and outputs a single array of those
            bands combined into complex values.
        datasize: 1x2 tuple/list (number of elements in first dimension, number
            of elements in the second dimension).  Size of the data (after
            symmetry transformations have been applied), as it would be returned
            in an array if the entire dataset was read in the chipper.  The
            number of bands (that would make a third dimension) is not included
            in this datasize attribute.
        symmetry: This parameter is used to perform symmetry operations on the
            imported image data.  A combination of mirroring in the first
            dimension (m1), mirroring in the second dimension (m2) and
            transpose (tr) is used to accomplish the symmetry operations.  The
            symmetry attribute is defined as [ m1 m2 tr ], where each of
            the elements can be either False or True (0 or 1) to indicate
            whether to apply a transformation or not.

            [ 0 0 0 ] - default setting; successive pixels on disk are
                interpreted to fill the last dimension in a NumPy array
            [ 0 1 1 ] -  90 degrees CCW from [ 0 0 0 ]
            [ 1 0 1 ] -  90 degrees  CW from [ 0 0 0 ]
            [ 1 1 0 ] - 180 degrees     from [ 0 0 0 ]
            [ 0 0 1 ] - transpose of         [ 0 0 0 ]
            [ 1 0 0 ] -  90 degrees CCW from [ 0 0 1 ]
            [ 0 1 0 ] -  90 degrees  CW from [ 0 0 1 ]
            [ 1 1 1 ] - 180 degrees     from [ 0 0 1 ]

    """

    # TODO: Add decimation types
    def __call__(self, dim1range=None, dim2range=None):
        dim1range, dim2range = reorient_chipper_args(
            self.symmetry, self.datasize, dim1range, dim2range)
        data_out = self.read_raw_fun(dim1range, dim2range)
        data_out = data2complex(data_out, self.complextype)
        if data_out.ndim > 2 and data_out.shape[0] == 1:  # Remove only multi-band dimension
            data_out = data_out.reshape(data_out.shape[-2:])
        data_out = reorient_chipper_data(self.symmetry, data_out)
        return data_out

    def __getitem__(self, key):
        dim1range, dim2range = self.slice_to_dim_args(key)
        return self.__call__(dim1range, dim2range)

    def slice_to_dim_args(self, key):
        dimrange = [None, None]
        for i in [0, 1]:
            if(key[i].start is None):
                kstart = 0
            else:
                kstart = key[i].start
            if(key[i].stop is None):
                if self.symmetry[2]:
                    kstop = self.datasize[i-1]
                else:
                    kstop = self.datasize[i]
            else:
                kstop = key[i].stop
            if(key[i].step is None):
                kstep = 1
            else:
                kstep = key[i].step
            dimrange[i] = (kstart, kstop, kstep)
        return dimrange[0], dimrange[1]


def data2complex(data_in, complextype):
    """Takes data in multiple bands and converts them to complex."""
    if complextype is False:  # Not complex
        return data_in
    elif complextype is True:  # Standard I/Q
        return data_in[0::2, :, :] + (data_in[1::2, :, :]*1j)
    elif callable(complextype):  # Custom complex types (i.e. amplitude/phase)
        return complextype(data_in)


def reorient_chipper_args(symmetry, datasize, dim1range=None, dim2range=None):
    """Applies a symmetry transform to the chipper indexing

    See module docstring for more information on the definition of symmetry.

    """

    if symmetry[2]:
        dim1range, dim2range = dim2range, dim1range
    if (symmetry[0] and (dim1range is not None) and
       not np.isscalar(dim1range) and len(dim1range) > 1):
        dim1range[0:2] = datasize[0] - dim1range[1::-1]
    if (symmetry[1] and (dim2range is not None) and
       not np.isscalar(dim2range) and len(dim2range) > 1):
        dim2range[0:2] = datasize[1] - dim2range[1::-1]
    return dim1range, dim2range


def reorient_chipper_data(symmetry, data_in):
    """Applies a symmetry transform to the chipper data

    See module docstring for more information on the definition of symmetry.

    """

    # Data may have 3 dimensions if multi-band.  If so, first dimension is band,
    # and we only want to change the last two.
    reoriented_data = data_in
    if symmetry[0]:
        reoriented_data = reoriented_data[..., ::-1, :]
    if symmetry[1]:
        reoriented_data = reoriented_data[..., :, ::-1]
    if symmetry[2]:
        last_axis = reoriented_data.ndim - 1
        reoriented_data = reoriented_data.swapaxes(last_axis, last_axis - 1)
    return reoriented_data


def check_args(datasize, dim1range, dim2range):
    """Standardize input arguments for chipper functions.

    This function does 4 things:

    1) Checks arguments for validity.  If error, returns meaningful error
    message.
    2) Fills in missing values with defaults, so that everything is explicitly
    listed.
    3) Casts everything to uint64.  Calculating offsets for seeks into large
    files (>4 Gig) using 32-bit integer values can result in overflows.
    4) We also wrap everything in a numPy array for consistent handling.  This
    way the caller can pass a tuple, list, scalar, or NumPy array, and it
    should all work the same.  Why Python requires this is an ugliness of the
    language.  MATLAB superior here.

    """

    # Check datasize first
    datasize = np.array(datasize, dtype='uint64')
    if datasize.shape != (2,) or (datasize[0] <= 0) or (datasize[1] <= 0):
        raise(ValueError('Invalid datasize.'))
    # Then check ranges.  They follow the syntax of Python's range() function.
    if dim1range is None:
        dim1range = (0, datasize[0], 1)
    if dim2range is None:
        dim2range = (0, datasize[1], 1)
    dim1range = np.array(dim1range, dtype='uint64')
    dim2range = np.array(dim2range, dtype='uint64')
    # If only one value given, use as the step size.  This varies slightly from
    # the range() usage, but is much more useful for our purposes.
    if dim1range.size == 1:
        dim1range = np.array((0, datasize[0], np.asscalar(dim1range)), dtype=np.uint64)
    if dim2range.size == 1:
        dim2range = np.array((0, datasize[1], np.asscalar(dim2range)), dtype=np.uint64)
    if ((dim1range[1] > datasize[0]) or (dim1range[0] < 0) or (dim1range[0] >= dim1range[1]) or
       (dim2range[1] > datasize[1]) or (dim2range[0] < 0) or (dim2range[0] >= dim2range[1])):
        raise(ValueError('Invalid subimage index range.'))
    # If only two values given, they are start and stop.  Third value is step
    # size, which defaults to 1.
    if dim1range.shape[0] == 2:
        dim1range = np.append(dim1range, np.uint64(1))
    if dim2range.shape[0] == 2:
        dim2range = np.append(dim2range, np.uint64(1))

    return datasize, dim1range, dim2range


def subset(full_image_chipper, dim1bounds, dim2bounds):
    """Take a chipper object and return another chipper object that only has visibility into a
    subset of the original."""
    class SubsetChipper(Base):
        """Object wrapper for calling base chipper."""
        def __init__(self, full_image_chipper, dim1bounds, dim2bounds):
            self.datasize = [np.diff(dim1bounds)[0], np.diff(dim2bounds)[0]]
            self.symmetry = [0, 0, 0]
            self.complextype = False
            self.full_image_chipper = full_image_chipper

        def read_raw_fun(self, dim1range, dim2range):
            datasize, dim1range, dim2range = check_args(self.datasize, dim1range, dim2range)
            return self.full_image_chipper(
                [dim1range[0] + dim1bounds[0], dim1range[1] + dim1bounds[0], dim1range[2]],
                [dim2range[0] + dim2bounds[0], dim2range[1] + dim2bounds[0], dim2range[2]])

    return SubsetChipper(full_image_chipper, dim1bounds, dim2bounds)
