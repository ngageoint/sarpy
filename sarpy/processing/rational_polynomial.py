"""
General purpose rational polynomial tools
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from typing import List, Sequence

import numpy
from numpy.polynomial import polynomial
from scipy.linalg import lstsq, LinAlgError

from sarpy.compliance import SarpyError

logger = logging.getLogger(__name__)


class SarpyRatPolyError(SarpyError):
    """A custom exception class for rational polynomial fitting errors."""


#################
# helper functions

def _get_num_variables(coeff_list):
    """
    Determine the number of variables by inspection of the coefficient list

    Parameters
    ----------
    coeff_list : Sequence

    Returns
    -------
    int
    """

    variables = None
    for entry in coeff_list:
        if isinstance(entry, int):
            if variables is None:
                variables = 1
            elif variables != 1:
                raise ValueError('Entry order mismatch')
        else:
            if variables is None:
                variables = len(entry)
            elif variables != len(entry):
                raise ValueError('Entry order mismatch')
    if variables is None:
        raise ValueError('Unable to determine the number of variables')
    return variables


def _map_list_to_poly_matrix(coeffs, coeff_list):
    """
    Maps the coefficients and coefficient listing to corresponding
    numpy polynomial coefficient matrix.

    Parameters
    ----------
    coeffs : Sequence
    coeff_list : Sequence

    Returns
    -------
    coefficient_array : numpy.ndarray
    """

    variables = _get_num_variables(coeff_list)

    matrix_shape = []
    for i in range(variables):
        matrix_shape.append(max(entry[i] for entry in coeff_list)+1)

    coefficient_array = numpy.zeros(tuple(matrix_shape), dtype='float64')
    for i, entry in enumerate(coeff_list):
        coefficient_array[entry] = coeffs[i]
    return coefficient_array


def get_default_coefficient_ordering(variables, order):
    """
    Gets a sensible coefficient ordering of a polynomial of given number of
    variables and order.

    Parameters
    ----------
    variables : int
    order : int

    Returns
    -------
    coefficient_list : List[tuple]
        List of the form `[(exponent 0, exponent 1, ...)]`, determining the ordering
        of monomial terms in the associated multivariable polynomial.
    """

    variables = int(variables)
    order = int(order)
    if variables < 1:
        raise ValueError('variables must be at least 1')
    if order < 1:
        raise ValueError('order must be at least 1')

    shape_details = tuple([order + 1 for _ in range(variables)])
    coefficient_list = []
    for index in numpy.ndindex(shape_details):
        total_exponent = sum(index)
        if total_exponent <= order:
            coefficient_list.append(index)
    return coefficient_list


###################
# base rational polynomial fitting functions

def rational_poly_fit_1d(x, data, coeff_list, cond=None):
    """
    Fits a one variable rational polynomial according to the input coefficient
    listing order.

    Parameters
    ----------
    x : numpy.ndarray
    data : numpy.ndarray
    coeff_list : List
    cond : None|float
        Passed through to :func:`scipy.linalg.lstsq`.
    """

    if coeff_list[0] not in [0, (0, )]:
        raise ValueError(
            'The first entry of coeff_list is required to be the constant term `0`')

    if not (x.size == data.size):
        raise ValueError('Size mismatch among data entries')
    x = x.flatten()
    data = data.flatten()

    # Enforcing that the denominator has constant term 1,
    #   P(x)/(1 + Q(x)) = d ->
    #   P(x) - d*Q(x) = d
    # This can be formulated as a strictly linear problem A*t = d

    A = numpy.empty((x.size, 2*len(coeff_list) - 1), dtype=numpy.float64)
    for i, entry in enumerate(coeff_list):
        if not (isinstance(entry, int) or (isinstance(entry, tuple) and len(entry) == 1 and isinstance(entry[0], int))):
            raise TypeError('coeff_list must be a list of integers or length 1 tuples of ints')
        if isinstance(entry, tuple):
            entry = entry[0]

        u = 1
        if entry > 0:
            u *= numpy.power(x, entry)
        A[:, i] = u
        if i > 0:
            A[:, i+len(coeff_list) - 1] = -u*data

    # perform least squares fit
    try:
        sol, residuals, rank, sing_values = lstsq(A, data, cond=cond)
    except LinAlgError as e:
        raise SarpyRatPolyError(str(e))

    #if len(residuals) != 0:
    residuals /= float(x.size)
    logger.info(
        'Performed rational polynomial fit, got\n\t'
        'residuals {}\n\t'
        'rank {}\n\t'
        'singular values {}'.format(residuals, rank, sing_values))

    numerator = numpy.zeros((len(coeff_list), ), dtype='float64')
    denominator = numpy.zeros((len(coeff_list), ), dtype='float64')
    denominator[0] = 1.0
    numerator[:] = sol[:len(coeff_list)]
    denominator[1:] = sol[len(coeff_list):]
    return numerator, denominator


def rational_poly_fit_2d(x, y, data, coeff_list, cond=None):
    """
    Fits a two variable rational polynomial according to the input coefficient
    listing order.

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    data : numpy.ndarray
    coeff_list : List
    cond : None|float
        Passed through to :func:`scipy.linalg.lstsq`.
    """

    if coeff_list[0] != (0, 0):
        raise ValueError(
            'The first entry of coeff_list is required to be the constant term `(0, 0)`')

    if not (x.size == y.size and x.size == data.size):
        raise ValueError('Size mismatch among data entries')
    x = x.flatten()
    y = y.flatten()
    data = data.flatten()

    # Enforcing that the denominator has constant term 1,
    #   P(x, y)/(1 + Q(x, y)) = d ->
    #   P(x, y) - d*Q(x, y) = d
    # This can be formulated as a strictly linear problem A*t = d

    A = numpy.empty((x.size, 2 * len(coeff_list) - 1), dtype=numpy.float64)
    for i, entry in enumerate(coeff_list):
        if len(entry) != 2:
            raise TypeError('coeff_list must be a list of tuples of length 2')

        u = 1
        if entry[0] > 0:
            u *= numpy.power(x, entry[0])
        if entry[1] > 0:
            u *= numpy.power(y, entry[1])
        A[:, i] = u
        if i > 0:
            A[:, i + len(coeff_list) - 1] = -u*data

    # perform least squares fit
    try:
        sol, residuals, rank, sing_values = lstsq(A, data, cond=cond)
    except LinAlgError as e:
        raise SarpyRatPolyError(str(e))

    # if len(residuals) != 0:
    residuals /= float(x.size)
    logger.info(
        'Performed rational polynomial fit, got\n\t'
        'residuals {}\n\t'
        'rank {}\n\t'
        'singular values {}'.format(residuals, rank, sing_values))

    numerator = numpy.zeros((len(coeff_list),), dtype='float64')
    denominator = numpy.zeros((len(coeff_list),), dtype='float64')
    denominator[0] = 1.0
    numerator[:] = sol[:len(coeff_list)]
    denominator[1:] = sol[len(coeff_list):]
    return numerator, denominator


def rational_poly_fit_3d(x, y, z, data, coeff_list, cond=None):
    """
    Fits a three variable rational polynomial according to the input coefficient
    listing order.

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    z : numpy.ndarray
    data : numpy.ndarray
    coeff_list : List
    cond : None|float
        Passed through to :func:`scipy.linalg.lstsq`.
    """

    if coeff_list[0] != (0, 0, 0):
        raise ValueError(
            'The first entry of coeff_list is required to be the constant term `(0, 0, 0)`')

    if not (x.size == y.size and x.size == z.size and x.size == data.size):
        raise ValueError('Size mismatch among data entries')
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    data = data.flatten()

    # Enforcing that the denominator has constant term 1,
    #   P(x, y, z)/(1 + Q(x, y, z)) = d ->
    #   P(x, y, z) - d*Q(x, y, z) = d
    # This can be formulated as a strictly linear problem A*t = d

    A = numpy.empty((x.size, 2*len(coeff_list) - 1), dtype=numpy.float64)
    for i, entry in enumerate(coeff_list):
        if len(entry) != 3:
            raise TypeError('coeff_list must be a list of tuples of length 3')

        u = 1
        if entry[0] > 0:
            u *= numpy.power(x, entry[0])
        if entry[1] > 0:
            u *= numpy.power(y, entry[1])
        if entry[2] > 0:
            u *= numpy.power(z, entry[2])
        A[:, i] = u
        if i > 0:
            A[:, i + len(coeff_list) - 1] = -u*data

    # perform least squares fit
    try:
        sol, residuals, rank, sing_values = lstsq(A, data, cond=cond)
    except LinAlgError as e:
        raise SarpyRatPolyError(str(e))

    #if len(residuals) != 0:
    residuals /= float(x.size)
    logger.info(
        'Performed rational polynomial fit, got\n\t'
        'residuals {}\n\t'
        'rank {}\n\t'
        'singular values {}'.format(residuals, rank, sing_values))

    numerator = numpy.zeros((len(coeff_list),), dtype='float64')
    denominator = numpy.zeros((len(coeff_list),), dtype='float64')
    denominator[0] = 1.0
    numerator[:] = sol[:len(coeff_list)]
    denominator[1:] = sol[len(coeff_list):]
    return numerator, denominator


####################
# rational polynomial definition

class RationalPolynomial(object):
    r"""
    A basic rational polynomial implementation. This assumes the data model
    `input_data -> output_data` via the relation

    .. math::
        X = (x, y, ...) & = (input\_data - input\_offset)/input\_scale \\
        (output\_data - output\_offset)/output\_scale & = Data = numerator(X)/denominator(X) \\
        output\_data & = (numerator(X)/denominator(X))*output_scale + output\_offset

    This object is callable, and acts as the evaluation function after construction.
    That is, suppose we have

    .. code::
        rational_poly = RationalPolynomial(...)  # suppose constructed as 2 variables
        output_0 = rational_poly([x, y])  # pass in the two variables as a single array
        output_1 = rational_poly(x, y)  # pass in the two variables individually
        # output_0 and output_1 should be identical

        output_fail = rational_poly(x, y, z)
        # this raises an exception for mismatch with the number of variables
    """

    __slots__ = (
        '_numerator', '_denominator', '_coeff_list', '_variables', '_input_offsets', '_input_scales',
        '_output_offset', '_output_scale', '_numerator_array', '_denominator_array')

    def __init__(self, numerator, denominator, coeff_list, input_offsets, input_scales, output_offset, output_scale):
        """

        Parameters
        ----------
        numerator : Sequence|numpy.ndarray
        denominator : Sequence|numpy.ndarray
        coeff_list : Sequence
        input_offsets : Sequence
        input_scales : Sequence
        output_offset : float
        output_scale : float
        """

        self._coeff_list = coeff_list
        self._variables = _get_num_variables(coeff_list)
        if self._variables not in [1, 2, 3]:
            raise ValueError('Functionality allows only 1, 2, or 3 variables.')

        if len(numerator) != len(self._coeff_list):
            raise ValueError('numerator must be the same length as coeff_list')
        self._numerator = numerator
        if len(denominator) != len(self._coeff_list):
            raise ValueError('denominator must be the same length as coeff_list')
        self._denominator = denominator

        if len(input_offsets) != self._variables:
            raise ValueError('The input_offsets must be the same length as the number of variables')
        self._input_offsets = input_offsets

        if len(input_scales) != self._variables:
            raise ValueError('The input_scale must be the same length as the number of variables')
        self._input_scales = input_scales

        self._output_offset = float(output_offset)
        self._output_scale = float(output_scale)

        self._numerator_array = _map_list_to_poly_matrix(numerator, coeff_list)
        self._denominator_array = _map_list_to_poly_matrix(denominator, coeff_list)

    @property
    def variables(self):
        """
        The number of independent variables.

        Returns
        -------
        int
        """

        return self._variables

    @property
    def coefficient_list(self):
        """
        The coefficient list.

        Returns
        -------
        Sequence
        """

        return self._coeff_list

    @property
    def numerator(self):
        """
        The numerator coefficients.

        Returns
        -------
        Sequence
        """

        return self._numerator

    @property
    def denominator(self):
        """
        The denominator coefficients.

        Returns
        -------
        Sequence
        """

        return self._denominator

    def __call__(self, *input_variables):
        def ensure_the_type(data):
            if isinstance(data, (numpy.number, int, float, numpy.ndarray)):
                return data
            else:
                return numpy.array(data)

        if len(input_variables) not in [1, self.variables]:
            raise ValueError('Got an unexpected number of input arguments')
        if len(input_variables) == 1:
            separate = False
            inp_vars = ensure_the_type(input_variables[0])
        else:
            separate = True
            inp_vars = [ensure_the_type(entry) for entry in input_variables]

        # todo: should we evaluate the viability of the input?
        if self.variables == 1:
            x = (inp_vars - self._input_offsets[0])/self._input_scales[0]
            value = polynomial.polyval(x, self._numerator_array) / \
                polynomial.polyval(x, self._denominator_array)
        elif self.variables == 2:
            if separate:
                x = (inp_vars[0] - self._input_offsets[0])/self._input_scales[0]
                y = (inp_vars[1] - self._input_offsets[1])/self._input_scales[1]
            else:
                if inp_vars.shape[-1] != 2:
                    raise ValueError(
                        'Final dimension of input data ({}) must match the number '
                        'of variables ({}).'.format(inp_vars.shape, self.variables))
                x = (inp_vars[..., 0] - self._input_offsets[0])/self._input_scales[0]
                y = (inp_vars[..., 1] - self._input_offsets[1])/self._input_scales[1]
            value = polynomial.polyval2d(x, y, self._numerator_array) / \
                polynomial.polyval2d(x, y, self._denominator_array)
        elif self.variables == 3:
            if separate:
                x = (inp_vars[0] - self._input_offsets[0])/self._input_scales[0]
                y = (inp_vars[1] - self._input_offsets[1])/self._input_scales[1]
                z = (inp_vars[2] - self._input_offsets[2])/self._input_scales[2]
            else:
                if inp_vars.shape[-1] != 3:
                    raise ValueError(
                        'Final dimension of input data ({}) must match the number '
                        'of variables ({}).'.format(inp_vars.shape, self.variables))
                x = (inp_vars[..., 0] - self._input_offsets[0])/self._input_scales[0]
                y = (inp_vars[..., 1] - self._input_offsets[1])/self._input_scales[1]
                z = (inp_vars[..., 2] - self._input_offsets[2]) / self._input_scales[2]
            value = polynomial.polyval3d(x, y, z, self._numerator_array) / \
                polynomial.polyval3d(x, y, z, self._denominator_array)
        else:
            raise ValueError('More than 3 variables is unsupported')
        return value*self._output_scale + self._output_offset


def _get_scale_and_offset(array):
    # type: (numpy.ndarray) -> (float, float)
    min_value = numpy.min(array)
    max_value = numpy.max(array)
    scale_value = 0.5*(max_value - min_value)
    offset_value = 0.5*(max_value + min_value)
    return offset_value, scale_value


def get_rational_poly_1d(x, data, coeff_list=None, order=None, cond=None):
    """
    Gets the RationalPolynomial instance that comes from fitting the provided data.

    Parameters
    ----------
    x : numpy.ndarray
    data : numpy.ndarray
    coeff_list : None|Sequence
    order : None|int
    cond : None|float
        Passed through to :func:`scipy.linalg.lstsq`.
    """

    if (coeff_list is None and order is None) or \
            (coeff_list is not None and order is not None):
        raise ValueError('Exact one of  coeff_list and order must be provided.')

    if order is not None:
        coeff_list = get_default_coefficient_ordering(1, int(order))

    if _get_num_variables(coeff_list) != 1:
        raise ValueError('The number of variables defined by the coefficient list must be 1.')

    scale_x, offset_x = _get_scale_and_offset(x)
    scale_data, offset_data = _get_scale_and_offset(data)

    numerator, denominator = rational_poly_fit_1d(
        (x-offset_x)/scale_x,
        (data-offset_data)/scale_data, coeff_list, cond=cond)

    return RationalPolynomial(
        numerator, denominator, coeff_list,
        (offset_x, ), (scale_x, ),
        offset_data, scale_data)


def get_rational_poly_2d(x, y, data, coeff_list=None, order=None, cond=None):
    """
    Gets the RationalPolynomial instance that comes from fitting the provided data.

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    data : numpy.ndarray
    coeff_list : None|Sequence
    order : None|int
    cond : None|float
        Passed through to :func:`scipy.linalg.lstsq`.
    """

    if (coeff_list is None and order is None) or \
            (coeff_list is not None and order is not None):
        raise ValueError('Exact one of coeff_list and order must be provided.')

    if order is not None:
        coeff_list = get_default_coefficient_ordering(2, int(order))

    if _get_num_variables(coeff_list) != 2:
        raise ValueError('The number of variables defined by the coefficient list must be 2.')

    scale_x, offset_x = _get_scale_and_offset(x)
    scale_y, offset_y = _get_scale_and_offset(y)
    scale_data, offset_data = _get_scale_and_offset(data)

    numerator, denominator = rational_poly_fit_2d(
        (x-offset_x)/scale_x, (y-offset_y)/scale_y,
        (data-offset_data)/scale_data, coeff_list, cond=cond)

    return RationalPolynomial(
        numerator, denominator, coeff_list,
        (offset_x, offset_y), (scale_x, scale_y),
        offset_data, scale_data)


def get_rational_poly_3d(x, y, z, data, coeff_list=None, order=None, cond=None):
    """
    Gets the RationalPolynomial instance that comes from fitting the provided data.

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    z : numpy.ndarray
    data : numpy.ndarray
    coeff_list : None|Sequence
    order : None|int
    cond : None|float
        Passed through to :func:`scipy.linalg.lstsq`.
    """

    if (coeff_list is None and order is None) or \
            (coeff_list is not None and order is not None):
        raise ValueError('Exact one of coeff_list and order must be provided.')

    if order is not None:
        coeff_list = get_default_coefficient_ordering(3, int(order))

    if _get_num_variables(coeff_list) != 3:
        raise ValueError('The number of variables defined by the coefficient list must be 3.')

    scale_x, offset_x = _get_scale_and_offset(x)
    scale_y, offset_y = _get_scale_and_offset(y)
    scale_z, offset_z = _get_scale_and_offset(z)
    scale_data, offset_data = _get_scale_and_offset(data)

    numerator, denominator = rational_poly_fit_3d(
        (x-offset_x)/scale_x, (y-offset_y)/scale_y, (z-offset_z)/scale_z,
        (data-offset_data)/scale_data, coeff_list, cond=cond)

    return RationalPolynomial(
        numerator, denominator, coeff_list,
        (offset_x, offset_y, offset_z), (scale_x, scale_y, scale_z),
        offset_data, scale_data)


####################
# collective rational polynomial function

class CombinedRationalPolynomial(object):
    """
    Assemble a collection of RationalPolynomial objects with the same number of variables
    into a single multi-variable output object.
    """

    __slots__ = ('_collection', )

    def __init__(self, *collection):
        if len(collection) == 1 and isinstance(collection[0], (list, tuple)):
            collection = collection[0]
        if len(collection) < 2:
            raise ValueError('This requires more than a single input')

        coll = []
        variables = None
        for entry in collection:
            if not isinstance(entry, RationalPolynomial):
                raise TypeError(
                    'Every input must be an instance of RationalPolynomial,\n\t'
                    'got type `{}`'.format(type(entry)))
            if variables is None:
                variables = entry.variables
            elif variables != entry.variables:
                raise TypeError(
                    'Every input must be an instance of RationalPolynomial with\n\t'
                    'the same number of variables, got type `{}` and `{}`'.format(variables, entry.variables))
            coll.append(entry)
        self._collection = tuple(coll)

    def __call__(self, *args, combine=True):
        out = tuple([entry(*args) for entry in self._collection])
        if combine:
            return numpy.stack(out, axis=-1)
        else:
            return out
