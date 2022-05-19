"""
Stateful functions for use in format operations for data segments.

This module introduced in version 1.3.0.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from typing import Union, Tuple, Optional

import numpy

from sarpy.io.general.slice_parsing import get_subscript_result_size

logger = logging.getLogger(__name__)


#######
# slice helper functions

def reformat_slice(
        sl_in: slice,
        limit_in: int,
        mirror: bool) -> slice:
    """
    Reformat the slice, with optional reverse operation.

    Note that the mirror operation doesn't run the slice backwards across the
    same elements, but rather creates a mirror image of the slice. This is
    to properly accommodate the data segment reverse symmetry transform.

    Parameters
    ----------
    sl_in : slice
        From prior processing, it is expected that `sl_in.step` is populated,
        and `sl_in.start` is non-negative, and `sl_in.stop` is non-negative or
        `None` (only in th event that `sl_in.step < 0`.
    limit_in : int
        The upper limit for the axis to which this slice pertains.
    mirror : bool
        Create the mirror image slice?

    Returns
    -------
    slice
    """

    if sl_in.step is None:
        raise ValueError('input slice has unpopulated step value')
    if sl_in.start is not None and sl_in.start < 0:
        raise ValueError('input slice has negative start value')
    if sl_in.stop is not None and sl_in.stop < 0:
        raise ValueError('input slice has negative stop value')

    if mirror:
        # reverse the indexing/ordering of the slice, and ensure that the
        # step gets handled properly
        if sl_in.step > 0:
            start_in = 0 if sl_in.start is None else sl_in.start
            stop_in = limit_in if sl_in.stop is None else sl_in.stop
            if sl_in.step > (stop_in - start_in):
                step_in = stop_in - start_in
            else:
                step_in = sl_in.step

            # are we fetching the whole dimension?
            if step_in == 1 and start_in == 0 and stop_in == limit_in:
                return slice(limit_in - 1, None, -1)

            count = int((stop_in - start_in) / float(step_in))

            if count == 1:
                return slice(limit_in - (start_in + 1), limit_in - start_in, 1)

            last_entry = start_in + (count - 1) * sl_in.step
            if start_in == 0:
                return slice(limit_in - 1 - last_entry, None, -sl_in.step)
            else:
                return slice(limit_in - 1 - last_entry, limit_in - start_in, -sl_in.step)
        else:
            start_in = limit_in - 1 if sl_in.start is None else sl_in.start
            stop_in = -1 if sl_in.stop is None else sl_in.stop

            # are we fetching the whole dimension?
            if sl_in.step == -1 and start_in == limit_in - 1 and stop_in == -1:
                return slice(0, None, 1)

            count = int((stop_in - start_in) / float(sl_in.step))

            if count == 0:
                return slice(limit_in - (start_in + 1), limit_in - start_in, 1)

            last_entry = start_in + (count - 1) * sl_in.step
            return slice(limit_in - 1 - last_entry, limit_in - start_in, -sl_in.step)
    else:
        return sl_in


#########
# format function implementations

class FormatFunction(object):
    """
    Stateful function for data orientation and formatting operations associated
    with reading data. *This is specifically intended for use in conjunction
    with `DataSegment`.*

    This allows mapping from raw data to formatted data, for reading data from
    a file and converting it to the form of intended use.

    If the reverse process is implemented, it enables converting from formatted
    data to raw data, for taking common use data and converting it to the raw
    form, for writing data to a file.

    Introduced in version 1.3.0.
    """

    has_inverse = False
    """
    Indicates whether this format function has the inverse call implemented.
    """

    __slots__ = ('_raw_shape', '_formatted_shape', '_reverse_axes', '_transpose_axes', '_reverse_transpose_axes')

    def __init__(
            self,
            raw_shape: Optional[Tuple[int, ...]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Tuple[int, ...]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None):
        """

        Parameters
        ----------
        raw_shape : None|Tuple[int, ...]
        formatted_shape : None|Tuple[int, ...]
        reverse_axes : None|Tuple[int, ...]
        transpose_axes : None|Tuple[int, ...]
        """

        self._raw_shape = None
        self._formatted_shape = None
        self._reverse_axes = None
        self._transpose_axes = None
        self._reverse_transpose_axes = None
        self.set_raw_shape(raw_shape)
        self.set_formatted_shape(formatted_shape)
        self.set_reverse_axes(reverse_axes)
        self.set_transpose_axes(transpose_axes)

    @property
    def raw_shape(self) -> Optional[Tuple[int, ...]]:
        """
        None|Tuple[int, ...]: The expected full possible raw shape.
        """

        return self._raw_shape

    def set_raw_shape(self, value: Optional[Tuple[int, ...]]) -> None:
        if self._raw_shape is not None:
            if value is None or value != self._raw_shape:
                raise ValueError('raw_shape is read only once set')
            return  # nothing to be done
        self._raw_shape = value

    @property
    def raw_ndim(self) -> int:
        if self.raw_shape is None:
            raise ValueError('raw_shape must be set')
        return len(self._raw_shape)

    @property
    def formatted_shape(self) -> Optional[Tuple[int, ...]]:
        """
        None|Tuple[int, ...]: The expected output shape basis.
        """

        return self._formatted_shape

    def set_formatted_shape(self, value: Optional[Tuple[int, ...]]) -> None:
        if self._formatted_shape is not None:
            if value is None or value != self._formatted_shape:
                raise ValueError('formatted_shape is read only once set')
            return  # nothing to be done
        self._formatted_shape = value

    @property
    def formatted_ndim(self) -> int:
        if self.formatted_shape is None:
            raise ValueError('formatted_shape must be set')
        return len(self._formatted_shape)

    @property
    def reverse_axes(self) -> Optional[Tuple[int, ...]]:
        """
        None|Tuple[int, ...]: The collection of axes (with respect to raw order)
        along which we will reverse as part of transformation to output data order.
        If not `None`, then this will be a tuple in strictly increasing order.
        """

        return self._reverse_axes

    def set_reverse_axes(self, value: Optional[Tuple[int, ...]]) -> None:
        if self._reverse_axes is not None:
            if value is None or value != self._reverse_axes:
                raise ValueError('reverse_axes is read only once set')
            return  # nothing to be done
        self._reverse_axes = value

    @property
    def transpose_axes(self) -> Tuple[int, ...]:
        """
        None|Tuple[int, ...]: The transpose order for switching from raw order to
        output order, prior to applying any format function.
        """

        return self._transpose_axes

    def set_transpose_axes(self, value: Optional[Tuple[int, ...]]) -> None:
        if self._transpose_axes is not None:
            if value is None or value != self._transpose_axes:
                raise ValueError('transpose_axes is read only once set')
            return  # nothing to be done
        if value is None:
            return  # nothing to be done

        self._transpose_axes = value
        # inverts the transpose axes mapping
        self._reverse_transpose_axes = tuple([value.index(i) for i in range(len(value))])

    def _get_populated_transpose_axes(self) -> Tuple[int, ...]:
        trans_axes = tuple(range(len(self.raw_shape))) if self.transpose_axes is None else \
            self.transpose_axes
        return trans_axes

    def _verify_shapes_set(self) -> None:
        if self.raw_shape is None or self.formatted_shape is None:
            raise ValueError('raw_shape and formatted_shape must both be set.')

    def _reverse_and_transpose(
            self,
            array: numpy.ndarray,
            inverse=False) -> numpy.ndarray:
        """
        Performs the reverse and transpose operations. This applies to data in raw
        format.

        Parameters
        ----------
        array : numpy.ndarray
        inverse : bool
            If `True`, then this should be the opposite operation.

        Returns
        -------
        numpy.ndarray
        """

        if array.ndim != self.raw_ndim:
            raise ValueError('Got unexpected raw data shape')

        if inverse:
            if self.transpose_axes is not None:
                # NB: this requires a copy, if not trivial
                array = numpy.transpose(array, axes=self._reverse_transpose_axes)
            if self.reverse_axes is not None:
                # NB: these are simply view operations
                for index in self.reverse_axes:
                    array = numpy.flip(array, axis=index)
        else:
            if self.reverse_axes is not None:
                # NB: these are simply view operations
                for index in self.reverse_axes:
                    array = numpy.flip(array, axis=index)
            if self.transpose_axes is not None:
                # NB: this requires a copy, if not trivial
                array = numpy.transpose(array, axes=self.transpose_axes)
        return array

    def __call__(
            self,
            array: numpy.ndarray,
            subscript: Tuple[slice, ...],
            squeeze=True) -> numpy.ndarray:
        """
        Performs the reformatting operation. The output data will have
        dimensions of size 1 squeezed by this operation, it should not generally
        be done before.

        Parameters
        ----------
        array : numpy.ndarray
            The input raw array.
        subscript : Tuple[slice, ...]
            The slice definition which yielded the input raw array.
        squeeze : bool
            Apply numpy.squeeze operation, which eliminates dimensions of size 1?

        Returns
        -------
        numpy.ndarray
            The output formatted array.
        """

        array = self._reverse_and_transpose(array, inverse=False)
        array = self._forward_functional_step(array, subscript)
        if squeeze:
            return numpy.squeeze(array)
        else:
            return array

    def inverse(
            self,
            array: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        """
        Inverse operation which takes in formatted data, and returns
        corresponding raw data.

        Parameters
        ----------
        array : numpy.ndarray
            The input formatted data.
        subscript : Tuple[slice, ...]
            The slice definition which yielded the formatted data.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            A value error should be raised if `inverse=True` and
            `has_inverse=False`.
        """

        if not self.has_inverse:
            raise ValueError('has_inverse is False')

        array = self._reverse_functional_step(array, subscript)
        array = self._reverse_and_transpose(array, inverse=True)
        return array

    def validate_shapes(self) -> None:
        """
        Validates that the provided `raw_shape` and `formatted_shape` are sensible.

        This should be called only after setting the appropriate values for the
        `raw_shape`, `formatted_shape`, `reverse_axes` and `transpose_axes` properties.

        Raises
        ------
        ValueError
            Raises a ValueError if the shapes are not compatible according to this
            function and the transpose axes argument.
        """

        raise NotImplementedError

    def transform_formatted_slice(
            self,
            subscript: Tuple[slice, ...]) -> Tuple[slice, ...]:
        """
        Transform from the subscript definition in formatted coordinates to
        subscript definition with respect to raw coordinates.

        Parameters
        ----------
        subscript : Tuple[slice, ...]

        Returns
        -------
        Tuple[slice, ...]

        Raises
        ------
        ValueError
            Raised if the desired requirement cannot be met.
        """

        raise NotImplementedError

    def transform_raw_slice(
            self,
            subscript: Tuple[slice, ...]) -> Tuple[slice, ...]:
        """
        Transform from the subscript definition in raw coordinates to
        subscript definition with respect to formatted coordinates.

        Parameters
        ----------
        subscript : Tuple[slice, ...]

        Returns
        -------
        Tuple[slice, ...]

        Raises
        ------
        ValueError
            Raised if the desired requirement cannot be met.
        """

        raise NotImplementedError

    def _forward_functional_step(
            self,
            array: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        """
        Performs the functional operation. This should perform on raw data following
        the reorientation operations provided by :func:`_reverse_and_transpose`.

        Parameters
        ----------
        array : numpy.ndarray
            The raw data to be transformed.
        subscript : Tuple[int, ...]
            The subscript in raw coordinates which would yield the raw data.

        Returns
        -------
        numpy.ndarray
        """

        raise NotImplementedError

    # noinspection PyTypeChecker
    def _reverse_functional_step(
            self,
            array: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        """
        Performs the reverse functional operation. This should perform on formatted data,
        followed by the reorientation operations provided by :func:`_reverse_and_transpose`.

        Parameters
        ----------
        array : numpy.ndarray
            The formatted data to be inverted.
        subscript : Tuple[slice, ...]
            The subscript in formatted coordinates which would yield the formatted data.

        Returns
        -------
        numpy.ndarray
        """

        if not self.has_inverse:
            raise ValueError('has_inverse is False')

        raise NotImplementedError


class IdentityFunction(FormatFunction):
    """
    A format function allowing only reversing and transposing operations, the
    actual functional step is simply the identity function.

    Introduced in version 1.3.0.
    """
    has_inverse = True

    def validate_shapes(self) -> None:
        self._verify_shapes_set()
        if self.raw_ndim != self.formatted_ndim:
            raise ValueError('raw_shape and formatted_shape must have the same length ')

        trans_axes = self._get_populated_transpose_axes()
        if self.raw_ndim != len(trans_axes):
            raise ValueError('raw_shape and transpose_axes must have the same length ')

        # we should have formatted_shape[i] == raw_shape[trans_axes[i]]
        expected_formatted_shape = tuple([self.raw_shape[index] for index in trans_axes])
        if expected_formatted_shape != self.formatted_shape:
            raise ValueError(
                'Input_shape `{}` and transpose_axes `{}` yields expected output shape `{}`\n\t'
                'got formatted_shape `{}`'.format(
                    self.raw_shape, self.transpose_axes, expected_formatted_shape, self.formatted_shape))

    def transform_formatted_slice(
            self,
            subscript: Tuple[slice, ...]) -> Tuple[slice, ...]:
        if len(subscript) != self.formatted_ndim:
            raise ValueError('The length of subscript and formatted_shape must match')

        reverse_axes = () if self.reverse_axes is None else self.reverse_axes
        rev_transpose_axes = tuple(range(len(self.raw_shape))) if self.transpose_axes is None else \
            self._reverse_transpose_axes

        # we will reorder from formatted order into raw order, using the opposite
        # of the transpose axes definition, reversing any axes required according
        # to reverse_axes definition (in raw order)
        out = []
        for i, index in enumerate(rev_transpose_axes):
            # index in raw order, i in formatted order
            rev = (index in reverse_axes)
            shape_limit = self.raw_shape[index]  # also self.formatted_shape[i]
            out.append(reformat_slice(subscript[index], shape_limit, rev))
        return tuple(out)

    def transform_raw_slice(
            self,
            subscript: Tuple[slice, ...]) -> Tuple[slice, ...]:
        if len(subscript) != self.raw_ndim:
            raise ValueError('The length of subscript and raw_shape must match')

        reverse_axes = () if self.reverse_axes is None else self.reverse_axes
        transpose_axes = tuple(range(len(self.formatted_shape))) if self.transpose_axes is None else \
            self.transpose_axes

        # we will reorder from raw order into formatted order, using the transpose
        # axes definition, reversing any axes required according to reverse_axes
        # definition (in raw order)
        out = []
        for i, index in enumerate(transpose_axes):
            # index in formatted order, i in raw order
            rev = (i in reverse_axes)
            shape_limit = self.formatted_shape[index]  # also self.raw_shape[i]
            out.append(reformat_slice(subscript[index], shape_limit, rev))
        return tuple(out)

    def _forward_functional_step(
            self,
            array: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        # the only operations are reordering/reversing, performed by _reverse_and_transpose
        return array

    def _reverse_functional_step(
            self,
            array: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        return array


class ComplexFormatFunction(FormatFunction):
    """
    Reformats data from real/imaginary dimension pairs
    to complex64 output, assuming that the raw data has
    fixed dimensionality and the real/imaginary pairs fall 
    along a given dimension.

    Introduced in version 1.3.0.
    """
    has_inverse = True
    _allowed_ordering = ('IQ', 'QI', 'MP', 'PM')

    __slots__ = (
        '_band_dimension', '_order', '_raw_dtype', '_magnitude_lookup_table',
        '_amplitude_scaling')

    def __init__(
            self,
            raw_dtype: Union[str, numpy.dtype],
            order: str,
            raw_shape: Optional[Tuple[int, ...]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Tuple[int, ...]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            band_dimension: int = -1,
            magnitude_lookup_table: Optional[numpy.ndarray] = None,
            amplitude_scaling: Optional[numpy.ndarray] = None):
        """

        Parameters
        ----------
        raw_dtype : str|numpy.dtype
            The raw datatype. Valid options dependent on the value of order.
        order : str
            One of `('IQ', 'QI', 'MP', 'PM')`. The options `('IQ', 'QI')` allow
            raw_dtype `('int8', 'int16', 'int32', 'float32', 'float64')`. The
            options `('MP', 'PM')` allow raw_dtype
            `('uint8', 'uint16', 'uint32', 'float32', 'float64')`.
        raw_shape : None|Tuple[int, ...]
        formatted_shape : None|Tuple[int, ...]
        reverse_axes : None|Tuple[int, ...]
        transpose_axes : None|Tuple[int, ...]
        band_dimension : int
            Which band is the complex dimension, **after** the transpose operation.
        magnitude_lookup_table : None|numpy.ndarray
            This is here to support the 'AMP8I_PH8I' pixel type for SICD usage,
            and will be ignored unless `order` in `['MP', 'PM']` and `raw_dtype`
            is `uint8`.
        amplitude_scaling : None|numpy.ndarray
            This is here to support the presence of a scaling in CPHD or CRSD usage.
            This requires that `raw_dtype` in `[int8, int16]`, `band_dimension`
            is the final dimension and neither `reverse_axes` nor `transpose_axes`
            is populated.
        """

        self._raw_dtype = numpy.dtype(raw_dtype)  # type: numpy.dtype
        self._band_dimension = None
        self._order = None
        self._set_order(order)
        self._magnitude_lookup_table = None
        self.set_magnitude_lookup(magnitude_lookup_table)
        self._amplitude_scaling = None
        self.set_amplitude_scaling(amplitude_scaling)

        FormatFunction.__init__(
            self, raw_shape=raw_shape, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes)
        self._set_band_dimension(band_dimension)

    @property
    def band_dimension(self) -> int:
        """
        int: The band dimension, in raw data after the transpose operation.
        """

        return self._band_dimension

    def _set_band_dimension(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError('band_dimension must be an integer')

        if not (-self.raw_ndim <= value < self.raw_ndim):
            raise ValueError('band_dimension out of bounds.')

        if value < 0:
            value = value + self.raw_ndim

        if self._band_dimension is not None:
            if value != self._band_dimension:
                raise ValueError('band_dimension is read only once set')
            return  # nothing to be done
        self._band_dimension = value

    @property
    def order(self) -> str:
        """
        str: The order string, once of `('IQ', 'QI', 'MP', 'PM')`.
        """

        return self._order

    def _set_order(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError('order must be an string')

        value = value.strip().upper()
        if value not in self._allowed_ordering:
            raise ValueError(
                'Order is required to be one of {},\n\t'
                'got `{}`'.format(self._allowed_ordering, value))
        if self._order is not None:
            if value != self._order:
                raise ValueError('order is read only once set')
        self._order = value
        if self._order in ['IQ', 'QI']:
            if self._raw_dtype.name not in ['int8', 'int16', 'int32', 'float32', 'float64']:
                raise ValueError(
                    'order is {}, and raw_dtype must be one of '
                    'int8, int16, int32, float32, or float64'.format(self._order))
        elif self._order in ['MP', 'PM']:
            if self._raw_dtype.name not in ['uint8', 'uint16', 'uint32', 'float32', 'float64']:
                raise ValueError(
                    'order is {}, and raw_dtype must be one of '
                    'uint8, uint16, uint32, float32, or float64'.format(self._order))
        else:
            raise ValueError('Got unhandled ordering value `{}`'.format(self._order))

    @property
    def magnitude_lookup_table(self) -> Optional[numpy.ndarray]:
        """
        The magnitude lookup table, for SICD usage with `AMP8I_PH8I` pixel type.

        Returns
        -------
        Optional[numpy.ndarray]
        """

        return self._magnitude_lookup_table

    def set_magnitude_lookup(
            self,
            lookup_table: Optional[numpy.ndarray]) -> None:
        if lookup_table is None:
            self._magnitude_lookup_table = None
            return

        if not isinstance(lookup_table, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
        if lookup_table.dtype.name not in ['float32', 'float64']:
            raise ValueError('requires a numpy.ndarray of float32 or 64 dtype, got {}'.format(lookup_table.dtype))
        if lookup_table.dtype.name != 'float32':
            lookup_table = numpy.cast['float32'](lookup_table)
        if lookup_table.shape != (256,):
            raise ValueError('Requires a one-dimensional numpy.ndarray with 256 elements, '
                             'got shape {}'.format(lookup_table.shape))

        if self.order not in ['MP', 'PM']:
            logger.warning(
                'A magnitude lookup table has been supplied,\n\t'
                'but the order is not one of `MP` or `PM`')
        if self._raw_dtype.name != 'uint8':
            raise ValueError(
                'A magnitude lookup table has been supplied,\n\t'
                'but the raw datatype is not `uint8`.')
        self._magnitude_lookup_table = lookup_table

    @property
    def amplitude_scaling(self) -> Optional[numpy.ndarray]:
        """
        The scaling multiplier array, for CPHD/CRSD usage.

        Returns
        -------
        Optional[numpy.ndarray]
        """

        return self._amplitude_scaling

    def set_amplitude_scaling(
            self,
            array: Optional[numpy.ndarray]) -> None:
        if array is None:
            self._amplitude_scaling = None
            return

        if not isinstance(array, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(array)))
        if array.ndim != 1:
            raise ValueError('requires a one dimensional array')
        if array.dtype.name not in ['float32', 'float64']:
            raise ValueError('requires a numpy.ndarray of float32 or 64 dtype, got {}'.format(array.dtype))
        if array.dtype.name != 'float32':
            array = numpy.cast['float32'](array)

        if self.order not in ['MP', 'PM']:
            logger.warning(
                'A magnitude lookup table has been supplied,\n\t'
                'but the order is not one of `MP` or `PM`')
        # NB: more validation as part of validate_shapes
        if self._raw_dtype.name not in ['int8', 'int16']:
            raise ValueError(
                'A scaling multiplier has been supplied,\n\t'
                'but the raw datatype is not `int8` or `int16`.')
        self._amplitude_scaling = array
        self._validate_amplitude_scaling()

    def _validate_amplitude_scaling(self) -> None:
        if self._amplitude_scaling is None or self._raw_shape is None:
            return

        if self.band_dimension not in [-1, self.raw_ndim-1]:
            raise ValueError('Use of scaling multiplier requires band is the final dimension')
        if self.transpose_axes is not None or self.reverse_axes is not None:
            raise ValueError('Use of scaling multiplier requires null reverse_axes and transpose_axes')
        if self._amplitude_scaling.size != self.raw_shape[0]:
            raise ValueError(
                'Use of scaling multiplier requires the array length\n\t'
                'and the first dimension of raw_shape match.')

    def validate_shapes(self) -> None:
        self._verify_shapes_set()
        self._set_band_dimension(self._band_dimension)
        trans_axes = self._get_populated_transpose_axes()
        if self.raw_ndim != len(trans_axes):
            raise ValueError('raw_shape and transpose_axes must have the same length ')

        arranged_shape = tuple([self.raw_shape[index] for index in trans_axes])
        if (arranged_shape[self.band_dimension] % 2) != 0:
            raise ValueError(
                'Input_shape `{}`, transpose_axes `{}` yields rearranged shape `{}`\n\t'
                'entry in band_dimension `{}` should be even'.format(
                    self.raw_shape, self.transpose_axes, arranged_shape, self.band_dimension))
        after_mapping_shape = [entry for entry in arranged_shape]
        after_mapping_shape[self.band_dimension] = int(after_mapping_shape[self.band_dimension]/2)
        after_mapping_shape = tuple(after_mapping_shape)

        if self.raw_ndim == self.formatted_ndim:
            if after_mapping_shape != self.formatted_shape:
                raise ValueError(
                    'Input_shape `{}`, transpose_axes `{}`, band dimension `{}` yields expected output shape `{}`\n\t'
                    'got formatted_shape `{}`'.format(
                        self.raw_shape, self.transpose_axes, self.band_dimension, arranged_shape, self.formatted_shape))
        elif self.raw_ndim == self.formatted_ndim + 1:
            reduced_shape = [entry for entry in arranged_shape]
            reduced_shape.pop(self.band_dimension)
            reduced_shape = tuple(reduced_shape)
            if reduced_shape != self.formatted_shape:
                raise ValueError(
                    'Input_shape `{}`, transpose_axes `{}`, band dimension `{}` yields expected output shape `{}`\n\t'
                    'got formatted_shape `{}`'.format(
                        self.raw_shape, self.transpose_axes, self.band_dimension, reduced_shape, self.formatted_shape))
        else:
            raise ValueError(
                'Input_shape `{}`, transpose_axes `{}`, band dimension `{}` yields expected output shape `{}`\n\t'
                'got formatted_shape `{}`'.format(
                    self.raw_shape, self.transpose_axes, self.band_dimension, arranged_shape, self.formatted_shape))
        self._validate_amplitude_scaling()

    def transform_formatted_slice(
            self,
            subscript: Tuple[slice, ...]) -> Tuple[slice, ...]:
        if len(subscript) != len(self.formatted_shape):
            raise ValueError('The length of subscript and formatted_shape must match')

        reverse_axes = () if self.reverse_axes is None else self.reverse_axes
        rev_transpose_axes = tuple(range(len(self.raw_shape))) if self.transpose_axes is None else \
            self._reverse_transpose_axes

        if self.raw_ndim == self.formatted_ndim:
            # there has been no collapse in dimension
            use_subscript = subscript
        else:
            # pad at band dimension (in the order after transpose operation)
            use_subscript = [entry for entry in subscript]
            use_subscript.insert(self.band_dimension, slice(0, 2, 1))

        # we will reorder from formatted order into raw order, using the opposite
        # of the transpose axes definition, reversing any axes required according
        # to reverse_axes definition (in raw order)
        out = []
        for i, index in enumerate(rev_transpose_axes):
            # index in raw order, i in formatted order (possibly padded for missing band dimension)
            rev = (index in reverse_axes)
            shape_limit = self.raw_shape[index]  # also self.formatted_shape[i]
            if self.raw_ndim == self.formatted_ndim:
                # the band dimension is not flattened, we have to transform
                temp_sl = reformat_slice(use_subscript[index], shape_limit, rev)
                if temp_sl.step not in [-1, 1]:
                    raise ValueError(
                        'Slicing along the complex dimension and applying this format function\n\t'
                        'is only only permitted using step +/-1')
                if temp_sl.step > 0:
                    start = 2*temp_sl.start
                    # noinspection PyTypeChecker
                    stop = 2*temp_sl.stop
                    out.append(slice(start, stop, 1))
                elif temp_sl.step < 0:
                    start = 2*temp_sl.start
                    # noinspection PyTypeChecker
                    stop = None if temp_sl.stop is None else 2*temp_sl.stop
                    out.append(slice(start, stop, -1))
            else:
                out.append(reformat_slice(use_subscript[index], shape_limit, rev))
        return tuple(out)

    def transform_raw_slice(
            self,
            subscript: Tuple[slice, ...]) -> Tuple[slice, ...]:
        if len(subscript) != self.raw_ndim:
            raise ValueError('The length of subscript and raw_shape must match')

        reverse_axes = () if self.reverse_axes is None else self.reverse_axes
        transpose_axes = tuple(range(len(self.formatted_shape))) if self.transpose_axes is None else \
            self.transpose_axes

        # we will reorder from raw order into formatted order, using the transpose
        # axes definition, reversing any axes required according to reverse_axes
        # definition (in raw order)
        out = []
        for i, index in enumerate(transpose_axes):
            # index in formatted order, i in raw order
            rev = (i in reverse_axes)
            shape_limit = self.formatted_shape[index]  # also self.raw_shape[i]
            if index == self.band_dimension and self.formatted_ndim < self.raw_ndim:
                # the band dimension has collapsed, so omit anything here
                continue
            else:
                out.append(reformat_slice(subscript[index], shape_limit, rev))
        return tuple(out)

    def _forward_functional_step(
            self,
            data: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        if data.ndim != self.raw_ndim:
            raise ValueError('Expected raw data of dimension {}'.format(self.raw_ndim))
        if (data.shape[self.band_dimension] % 2) != 0:
            raise ValueError(
                'Requires {} dimensional raw data with even size along dimension {}'.format(
                    self.raw_ndim, self.band_dimension))

        band_dim_size = data.shape[self.band_dimension]
        if self.formatted_ndim < self.raw_ndim:
            out_shape = data.shape[:self.band_dimension] + data.shape[self.band_dimension + 1:]
        else:
            out_shape = data.shape[:self.band_dimension] + \
                        (int(band_dim_size/2), ) + \
                        data.shape[self.band_dimension + 1:]

        out = numpy.empty(out_shape, dtype='complex64')
        if self.order == 'IQ':
            out.real = data.take(indices=range(0, band_dim_size, 2), axis=self.band_dimension)
            out.imag = data.take(indices=range(1, band_dim_size, 2), axis=self.band_dimension)
        elif self.order == 'QI':
            out.imag = data.take(indices=range(0, band_dim_size, 2), axis=self.band_dimension)
            out.real = data.take(indices=range(1, band_dim_size, 2), axis=self.band_dimension)
        elif self.order in ['MP', 'PM']:
            if self.order == 'MP':
                mag = data.take(indices=range(0, band_dim_size, 2), axis=self.band_dimension)
                theta = data.take(indices=range(1, band_dim_size, 2), axis=self.band_dimension)
            else:
                mag = data.take(indices=range(0, band_dim_size, 2), axis=self.band_dimension)
                theta = data.take(indices=range(1, band_dim_size, 2), axis=self.band_dimension)

            if self.magnitude_lookup_table is not None:
                mag = self.magnitude_lookup_table[mag]

            if data.dtype.name in ['uint8', 'uint16', 'uint32']:
                bit_depth = data.dtype.itemsize*8
                theta = theta*2*numpy.pi/(1 << bit_depth)
            out.real = mag*numpy.cos(theta)
            out.imag = mag*numpy.sin(theta)
        else:
            raise ValueError('Unhandled order value {}'.format(self.order))

        # NB: subscript is in raw coordinates, but we have verified that
        #   the first dimension is unchanged
        if self._amplitude_scaling is not None:
            out = self._amplitude_scaling[subscript[0]]*out

        return out

    def _reverse_functional_step(
            self,
            data: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
        if data.ndim != self.formatted_ndim:
            raise ValueError('Expected formatted data of dimension {}'.format(self.raw_ndim))

        if self.formatted_ndim < self.raw_ndim:
            out_shape = data.shape[:self.band_dimension] + (2,) + data.shape[self.band_dimension:]
        else:
            band_dim_size = data.shape[self.band_dimension]
            out_shape = data.shape[:self.band_dimension] + \
                (2*band_dim_size, ) + \
                data.shape[self.band_dimension + 1:]

        # NB: subscript is in formatted coordinates, but we have verified that
        #   transpose_axes is None and band_dimension is the final dimension
        if self._amplitude_scaling is not None:
            data = (1./self._amplitude_scaling[subscript[0]])*data

        slice0 = []
        slice1 = []
        for index, siz in out_shape:
            if index == self.band_dimension:
                slice0.append(slice(0, siz, 2))
                slice1.append(slice(1, siz, 2))
            else:
                slice0.append(slice(0, siz, 1))
                slice1.append(slice(0, siz, 1))
        slice0 = tuple(slice0)
        slice1 = tuple(slice1)

        out = numpy.empty(out_shape, dtype=self._raw_dtype)
        if self.order == 'IQ':
            out[slice0] = data.real
            out[slice1] = data.imag
        elif self.order == 'QI':
            out[slice1] = data.real
            out[slice0] = data.imag
        elif self.order in ['MP', 'PM']:
            magnitude = numpy.abs(data)
            if self.magnitude_lookup_table is not None:
                magnitude = numpy.digitize(
                    magnitude.ravel(), self.magnitude_lookup_table, right=False).reshape(data.shape)

            theta = numpy.arctan2(data.imag, data.real)
            theta[theta < 0] += 2*numpy.pi

            if self._raw_dtype.name in ['uint8', 'uint16', 'uint32']:
                bit_depth = self._raw_dtype.itemsize*8
                theta *= (1 << bit_depth)/(2*numpy.pi)

            if self.order == 'MP':
                out[slice0] = magnitude
                out[slice1] = theta
            else:
                out[slice1] = magnitude
                out[slice0] = theta
        else:
            raise ValueError('Unhandled order value {}'.format(self.order))
        return out


class SingleLUTFormatFunction(FormatFunction):
    """
    Reformat the raw data according to the use of a single 8-bit lookup table.
    In the case of a 2-d LUT, and effort to slice on the final dimension
    (from the LUT) is not supported.

    Introduced in version 1.3.0.
    """

    has_inverse = False

    __slots__ = ('_lookup_table', )

    def __init__(
            self,
            lookup_table: numpy.ndarray,
            raw_shape: Optional[Tuple[int, ...]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Tuple[int, ...]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None):
        """

        Parameters
        ----------
        lookup_table : numpy.ndarray
            The 8-bit lookup table.
        raw_shape : None|Tuple[int, ...]
        formatted_shape : None|Tuple[int, ...]
        reverse_axes : None|Tuple[int, ...]
        transpose_axes : None|Tuple[int, ...]
        """

        self._lookup_table = None
        if not isinstance(lookup_table, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
        if lookup_table.dtype.name != 'uint8':
            raise ValueError('requires a numpy.ndarray of uint8 dtype, got {}'.format(lookup_table.dtype))
        self._lookup_table = lookup_table

        FormatFunction.__init__(self, raw_shape=raw_shape, formatted_shape=formatted_shape,
                                reverse_axes=reverse_axes, transpose_axes=transpose_axes)

    @property
    def lookup_table(self) -> numpy.ndarray:
        return self._lookup_table

    def validate_shapes(self) -> None:
        self._verify_shapes_set()
        trans_axes = self._get_populated_transpose_axes()
        if self.raw_ndim != len(trans_axes):
            raise ValueError('raw_shape and transpose_axes must have the same length')

        arranged_shape = [self.raw_shape[index] for index in trans_axes]
        if self.lookup_table.ndim == 2:
            arranged_shape.append(self.lookup_table.shape[1])
        arranged_shape = tuple(arranged_shape)

        if arranged_shape != self.formatted_shape:
            raise ValueError(
                'Input_shape `{}`, transpose_axes `{}` and lookup table\n\t'
                'yields expected output shape `{}` got formatted_shape `{}`'.format(
                    self.raw_shape, self.transpose_axes, arranged_shape, self.formatted_shape))

    def transform_formatted_slice(
            self,
            subscript: Tuple[slice, ...]) -> Tuple[slice, ...]:
        if len(subscript) != self.formatted_ndim:
            raise ValueError('The length of subscript and formatted_shape must match')

        reverse_axes = () if self.reverse_axes is None else self.reverse_axes
        rev_transpose_axes = tuple(range(len(self.raw_shape))) if self.transpose_axes is None else \
            self._reverse_transpose_axes

        # we will reorder from formatted order into raw order, using the opposite
        # of the transpose axes definition, reversing any axes required according
        # to reverse_axes definition (in raw order)
        out = []
        # NB: for 2-d LUT, the final slice will be ignored here (as it should)
        for i, index in enumerate(rev_transpose_axes):
            # index in raw order, i in formatted order
            rev = (index in reverse_axes)
            shape_limit = self.raw_shape[index]  # also self.formatted_shape[i]
            out.append(reformat_slice(subscript[index], shape_limit, rev))
        return tuple(out)

    def transform_raw_slice(
            self,
            subscript: Tuple[slice, ...]) -> Tuple[slice, ...]:
        if len(subscript) != self.raw_ndim:
            raise ValueError('The length of subscript and raw_shape must match')

        reverse_axes = () if self.reverse_axes is None else self.reverse_axes
        transpose_axes = tuple(range(len(self.formatted_shape))) if self.transpose_axes is None else \
            self.transpose_axes

        # we will reorder from raw order into formatted order, using the transpose
        # axes definition, reversing any axes required according to reverse_axes
        # definition (in raw order)
        out = []
        for i, index in enumerate(transpose_axes):
            # index in formatted order, i in raw order
            rev = (i in reverse_axes)
            shape_limit = self.formatted_shape[index]  # also self.raw_shape[i]
            out.append(reformat_slice(subscript[index], shape_limit, rev))
        if self.raw_ndim < self.formatted_ndim:
            # 2-d lookup table
            lim = self.formatted_shape[-1]
            out.append(slice(0, lim, 1))
        return tuple(out)

    def _forward_functional_step(
            self,
            array: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> numpy.ndarray:
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
            out_shape = array.shape + (self.lookup_table.shape[1], )
        else:
            out_shape = array.shape
        return numpy.reshape(out, out_shape)

    def __call__(
            self,
            array: numpy.ndarray,
            subscript: Tuple[slice, ...],
            squeeze=True) -> numpy.ndarray:
        array = self._reverse_and_transpose(array, inverse=False)
        array = self._forward_functional_step(array, subscript)
        if self.raw_ndim < self.formatted_ndim:
            # apply slice in the band (final dimension)
            array = array.take(
                indices=numpy.arange(self.formatted_shape[-1])[subscript[-1]], axis=-1)
            # ensure shape is as expected - any squeeze handled consistently
            out_shape = get_subscript_result_size(subscript, self.formatted_shape)
            array = numpy.reshape(array, out_shape)
        if squeeze:
            return numpy.squeeze(array)
        else:
            return array


if __name__ == '__main__':

    # TODO: formulate this simple testing into unit tests...

    # check the reverse_axes and transpose_axes functionality
    t_data = numpy.reshape(numpy.arange(6, dtype='int32'), (3, 2))

    # check just reverse axes
    # for axis in [0, 1]:
    #     func_rev = IdentityFunction(raw_shape=(3, 2), formatted_shape=(3, 2), reverse_axes=(axis, ), transpose_axes=None)
    #     out_data = func_rev(t_data, None)
    #     test_data = numpy.flip(t_data, axis=axis)
    #     print(f'Reverse axis {axis} Success? {numpy.all(test_data == out_data)}')

    # check just transpose axes
    # func_transpose = IdentityFunction(raw_shape=(3, 2), formatted_shape=(2, 3), transpose_axes=(1, 0))
    # out_data = func_transpose(t_data, None)
    # test_data = numpy.transpose(t_data, (1, 0))
    # print(f'transpose axes success? {numpy.all(test_data == out_data)}')

    # check both together
    for axis in ((0, ), (1, ), (0, 1)):
        func = IdentityFunction(raw_shape=(3, 2), formatted_shape=(2, 3), reverse_axes=axis, transpose_axes=(1, 0))
        out_data = func(t_data, None)
        test_data = numpy.transpose(numpy.flip(t_data, axis=axis), (1, 0))
        print(f'combined axis {axis} success? {numpy.all(test_data == out_data)}')