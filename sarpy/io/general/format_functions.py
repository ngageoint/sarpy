"""
Standard format functions for use in DataSegmentReader definition. 
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import numpy

from typing import Union

logger = logging.getLogger(__name__)


class FormatFunction(object):
    """
    Stateful reformat functions for data reading/formatting operations. 
    This allows transformation between actual and expected input slice 
    arguments.
    """

    def __call__(self, array: numpy.ndarray) -> numpy.ndarray:
        """
        Performs the reformatting operation. The output data is 
        expected to have the dimensions of size 1 squeezed AFTER
        this operation, not before.

        Parameters
        ----------
        array : numpy.ndarray   
            The input raw array.

        Returns
        -------
        numpy.ndarray   
            The formatted array.
        """

        raise NotImplementedError
    
    def reverse_slice_transform(self, subscript : Tuple[slice, ...]) -> Tuple[slice, ...]:
        """
        Transform the desired output subscript definition to the corresponding 
        input data slice definition. 

        Parameters
        ----------
        subscript : Tuple[slice, ...]

        Returns
        -------
        Tuple[slice, ...]

        Raises
        ------
        ValueError
            This should raise a value error if the desired input requirement cannot 
            be expressed as a slice along the given dimension
        """

        raise NotImplementedError


class ComplexFormatFunction(FormatFunction):
    """
    Reformats input data from real/imaginary dimension pairs 
    to complex64 output, assuming that the input data has 
    fixed dimensionality and the real/imaginary pairs fall 
    along a given dimension. 
    """

    def __init__(self, total_dimension: int, band_dimension: int, order='IQ'):
        order = order.upper()
        if order not in ['IQ', 'QI']:
            raise ValueError('Order is required to be one of `IQ` or `QI`,\n\tgot `{}`'.format(order))
        self.order = order  # type: str

        self.total_dimension = total_dimension  # type: int
        self.band_dimension = band_dimension  # type: int

    def __call__(self, array: numpy.ndarray) -> numpy.ndarray:
        if array.ndim != self.total_dimension or (array.shape[self.band_dimension] % 2) != 0:
            raise ValueError(
                'Requires {} dimensional input with even size along dimension {}'.format(
                    self.total_dimension, self.band_dimension))
        
        band_dim_size = array.shape[self.band_dimension]
        out_shape = numpy.array(array.shape, dtype='int64')
        out_shape[self.band_dimension] = int(band_dim_size/2)

        out = numpy.zeros(tuple(out_shape.tolist()), dtype='complex64')
        if self.order == 'IQ':
            out.real = array.take(indices=range(0, band_dim_size, 2), axis=self.band_dimension)
            out.imag = array.take(indices=range(1, band_dim_size, 2), axis=self.band_dimension)
        else:
            out.imag = array.take(indices=range(0, band_dim_size, 2), axis=self.band_dimension)
            out.real = array.take(indices=range(1, band_dim_size, 2), axis=self.band_dimension)
        return out
    
    def reverse_slice_transform(self, subscript : Tuple[slice, ...]) -> Tuple[slice, ...]:
        # NB: this could accompany a missing dimension at the band_dimension location
        if len(subscript) == self.total_dimension - 1:
            out = []
            for index, entry in enumerate(subscript):
                if index == self.band_dimension:
                    out.append(slice(0, 2, 1))
                else:
                    out.append(entry)
            if self.band_dimension == self.total_dimension - 1:
                out.append(slice(0, 2, 1))                
            return tuple(out)
                    
        if len(subscript) != self.total_dimension:
            raise ValueError('Requires input of length {}, got length {}'.format(self.total_dimension, len(subscript)))
        
        out = []
        for index, entry in enumerate(subscript):
            if index == self.band_dimension:
                if entry.step != 1:
                    raise ValueError(
                        'Slicing along the complex dimension and applying this format function\n\t''
                        'is only only permitted using step == 1.')
                start = None if entry.start is None else 2*entry.start
                stop = None if entry.stop is None else 2*entry.stop
                out.append(slice(start, stop, 1))
            else:
                out.append(entry)
        return tuple(out)


class MagnitudePhaseFormatFunction(FormatFunction):
    """
    Reformats input data from magnitude/phase dimension pairs 
    to complex64 output, assuming that the input data has 
    fixed dimensionality and the pairs fall 
    along a given dimension. 
    """

    def __init__(self, total_dimension: int, band_dimension: int, order='MP'):
        order = order.upper()
        if order not in ['MP', 'PM']:
            raise ValueError('Order is required to be one of `MP` or `PM`,\n\tgot `{}`'.format(order))
        self.order = order  # type: str

        self.total_dimension = total_dimension  # type: int
        self.band_dimension = band_dimension  # type: int

    def __call__(self, array: numpy.ndarray) -> numpy.ndarray:
        if array.ndim != self.total_dimension or (array.shape[self.band_dimension] % 2) != 0:
            raise ValueError(
                'Requires {} dimensional input with even size along dimension {}'.format(
                    self.total_dimension, self.band_dimension))

        if array.dtype.name not in ['uint8', 'uint16', 'uint32', 'uint64']:
            raise ValueError(
                'Requires a numpy.ndarray of unsigned integer type.')
        bit_depth = array.dtype.itemsize*8
        
        band_dim_size = array.shape[self.band_dimension]
        out_shape = numpy.array(array.shape, dtype='int64')
        out_shape[self.band_dimension] = int(band_dim_size/2)

        out = numpy.zeros(tuple(out_shape.tolist()), dtype='complex64')
        if self.order == 'MP':
            mag = array.take(indices=range(0, band_dim_size, 2), axis=self.band_dimension)
            theta = array.take(indices=range(1, band_dim_size, 2), axis=self.band_dimension)*(2*numpy.pi/(1 << bit_depth))
        else:
            mag = array.take(indices=range(1, band_dim_size, 2), axis=self.band_dimension)
            theta = array.take(indices=range(2, band_dim_size, 2), axis=self.band_dimension)*(2*numpy.pi/(1 << bit_depth))
        
        out.real = mag*numpy.cos(theta)
        out.imag = mag*numpy.sin(theta)
        return out
    
    def reverse_slice_transform(self, subscript : Tuple[slice, ...]) -> Tuple[slice, ...]:
        # NB: this could accompany a missing dimension at the band_dimension location
        if len(subscript) == self.total_dimension - 1:
            out = []
            for index, entry in enumerate(subscript):
                if index == self.band_dimension:
                    out.append(slice(0, 2, 1))
                else:
                    out.append(entry)
            if self.band_dimension == self.total_dimension - 1:
                out.append(slice(0, 2, 1))                
            return tuple(out)
                    
        if len(subscript) != self.total_dimension:
            raise ValueError('Requires input of length {}, got length {}'.format(self.total_dimension, len(subscript)))
        
        out = []
        for index, entry in enumerate(subscript):
            if index == self.band_dimension:
                if entry.step != 1:
                    raise ValueError(
                        'Slicing along the complex dimension and applying this format function\n\t''
                        'is only only permitted using step == 1.')
                start = None if entry.start is None else 2*entry.start
                stop = None if entry.stop is None else 2*entry.stop
                out.append(slice(start, stop, 1))
            else:
                out.append(entry)
        return tuple(out)


class SingleLUTFormatFunction(object):

    """
    Reformats input data according to the use of a single 8-bit lookup table. 
    """

    def __init__(self, lookup_table: numpy.ndarray):
        if not isinstance(lookup_table, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
        if lookup_table.dtype.name != 'uint8':
            raise ValueError('requires a numpy.ndarray of uint8 dtype, got {}'.format(lookup_table.dtype))
        self.lookup_table = lookup_table

    def __call__(self, array: numpy.ndarray) -> numpy.ndarray:
        if not isinstance(array, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(array)))

        if array.dtype.name not in ['uint8', 'uint16']:
            raise ValueError('requires a numpy.ndarray of uint8 or uint16 dtype, '
                             'got {}'.format(array.dtype.name))

        if len(array.shape) == 2:
            raise ValueError('Requires a two-dimensional numpy.ndarray, got shape {}'.format(array.shape))
        temp = numpy.reshape(array, (-1, ))
        out = self.lookup_table[temp]
        if self.lookup_table.ndim == 2:
            return numpy.reshape(out, (array.shape[0], array.shape[1], lookup_table.shape[1]))
        else:
            return numpy.reshape(out, (array.shape[0], array.shape[1], 1))
    
    def reverse_slice_transform(self, subscript : Tuple[slice, ...]) -> Tuple[slice, ...]:
        if len(subscript) in [2, 3]:
            return (subscript[0], subscript[1])
        else:
            raise ValueError('Only two dimensional input permitted')


