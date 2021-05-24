"""
Common utils for CPHD 1.0 functionality.
"""

import numpy

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Daniel Pressler, Valkyrie")


#########
# Module variables
_DTYPE_LOOKUP = {
    "U1": numpy.dtype('>u1'),
    "U2": numpy.dtype('>u2'),
    "U4": numpy.dtype('>u4'),
    "U8": numpy.dtype('>u8'),
    "I1": numpy.dtype('>i1'),
    "I2": numpy.dtype('>i2'),
    "I4": numpy.dtype('>i4'),
    "I8": numpy.dtype('>i8'),
    "F4": numpy.dtype('>f4'),
    "F8": numpy.dtype('>f8'),
    "CI2": numpy.dtype([('real', '>i1'), ('imag', '>i1')]),
    "CI4": numpy.dtype([('real', '>i2'), ('imag', '>i2')]),
    "CI8": numpy.dtype([('real', '>i4'), ('imag', '>i4')]),
    "CI16": numpy.dtype([('real', '>i8'), ('imag', '>i8')]),
    "CF8": numpy.dtype('>c8'),
    "CF16": numpy.dtype('>c16')}


def _single_binary_format_string_to_dtype(form):
    """
    Convert a CPHD datatype into a dtype.

    Parameters
    ----------
    form

    Returns
    -------
    numpy.dtype
    """

    if form.startswith('S'):
        return numpy.dtype(form)
    else:
        return _DTYPE_LOOKUP[form]


def binary_format_string_to_dtype(format_string):
    """
    Return the numpy.dtype for CPHD Binary Format string (table 10-2).

    Parameters
    ----
    format_string: str
        PVP type designator (e.g., :code:`'I1', 'I4', 'CF8'`, etc.).

    Returns
    -------
    numpy.dtype
        The equivalent `numpy.dtype` of the PVP format string
        (e.g., :code:`numpy.int8, numpy.int32, numpy.complex64`, etc.).
    """

    components = format_string.split(';')
    if '=' in components[0]:
        assert format_string.endswith(';'), 'Format strings describing multiple parameters must end with a semi-colon'
        comptypes = []
        for comp in components[:-1]:
            kvp = comp.split('=')
            comptypes.append((kvp[0], _single_binary_format_string_to_dtype(kvp[1])))

        # special handling of XYZ types
        keys, types = list(zip(*comptypes))
        if keys == ('X', 'Y', 'Z') and len(set(types)) == 1:
            dtype = numpy.dtype((comptypes[0][1], 3))
        else:
            dtype = numpy.dtype(comptypes)
    else:
        dtype = _single_binary_format_string_to_dtype(components[0])
    return dtype


def homogeneous_dtype(format_string, return_length=False):
    """
    Determine a numpy.dtype (including endianness) from a CPHD format string, requiring
    that any multiple parts are all identical.

    Parameters
    ----------
    format_string : str
    return_length : bool
        Return the number of elements?

    Returns
    -------
    numpy.dtype|(numpy.dtype, int)
        Tuple of (`numpy.dtype`, # of elements) if `return_length`, `numpy.dtype` otherwise

    """

    raw_dtype = binary_format_string_to_dtype(format_string)
    if raw_dtype.names is None:
        # Simple or subarray
        dtype = raw_dtype.base
        num_elements = max(1, sum(raw_dtype.shape))
    else:
        # Structured
        dtype_set = {v[0] for v in raw_dtype.fields.values()}
        if len(dtype_set) == 1:
            dtype = dtype_set.pop()
            num_elements = len(raw_dtype.names)
        else:
            raise ValueError("Format string {} was heterogeneous (dtype_set={})".format(format_string, dtype_set))

    if return_length:
        return dtype, num_elements
    else:
        return dtype
