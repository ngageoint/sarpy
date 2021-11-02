"""
Module for converting between various latitude/longitude representations.
"""

import sys
import re

import numpy


__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"


def string(value, latlon, num_units=3, precision=None, delimiter='',
           include_symbols=True, signed=False, padded=True):
    """
    Convert latitude/longitude numeric values to customizable string format.

    Supports ISO 6709:2008 formatted geographic coordinates:

    * Annex D (human interface)
        delimiter = ''; include_symbols = true; padded = true; signed = false
    * Annex H (string representation)
        delimiter = ''; include_symbols = false; padded = true; signed = true

    Parameters
    ----------
    value : float|numpy.ndarray|list|tuple
        Value of latitude or longitude in decimal degrees or dms vector.
    latlon : str
        One of {'lat', 'lon'}, required for formatting string.
    num_units : int
        1 - decimal degrees; 2 - degrees/minutes; 3 - degrees/minutes/seconds.
        Default is 3.
    delimiter : str|list|tuple
        Separators between degrees/minutes/seconds/hemisphere.  Default is '' (empty).
    include_symbols : bool
        Whether to include degree, minute, second symbols.  Default is true.
    signed : bool
        Whether to use +/- or N/S/E/W to represent hemisphere.
        Default is false (N/S/E/W).
    precision : int
        Number of decimal points shown in finest unit.  Default is 5 if
        num_units==1, otherwise 0.
    padded : bool
        Whether to use zeros to pad out to consistent string length (3 digits for
        longitude degrees, 2 digits for all other elements).  Default is true.
    """

    if isinstance(value, (numpy.ndarray, list, tuple)):
        value = num(value)
    elif not isinstance(value, float):
        value = float(value)
    # value should now be in in decimal degrees

    # Precision.  Default is dependent on other input arguments.
    if precision is None:
        if num_units == 1:
            precision = 5
        else:
            precision = 0
    # Symbols
    if include_symbols:
        latlon_symbols = ('\xB0', "'", '"')
    else:
        latlon_symbols = ('', '', '')
    # Delimiters
    try:
        if len(delimiter) != 3:
            delimiter = [delimiter]*num_units
    except:  # Must be a scalar if len() didn't work
        delimiter = [delimiter]*num_units
    if signed:
        delimiter[num_units-1] = ''  # No separator needed for hemisphere
    # Differences between latitude and longitude
    if latlon == 'lat':
        if value > 0:
            hemisphere = 'N'
            latlon_sign = '+'
        else:
            hemisphere = 'S'
            latlon_sign = '-'
        degrees_digits = 2
    elif latlon == 'lon':
        if value > 180:
            value = value - 360
        if value > 0:
            hemisphere = 'E'
            latlon_sign = '+'
        else:
            hemisphere = 'W'
            latlon_sign = '-'
        degrees_digits = 3
    # Compute degrees/minutes/seconds
    new_value = abs(value)
    value = [None]*num_units
    for i in range(num_units):
        fraction = new_value % 1.
        value[i] = int(new_value)
        new_value = fraction*60
    value[-1] = value[-1] + fraction
    if num_units > 1 and round(value[-1],precision) == 60:  # Seconds of 60 is invalid
        value[-1] = 0 
        value[-2] = value[-2] + 1
        if num_units == 3 and value[-2] == 60:  # If adding 1 to mintues makes minutes 60 which is also invalid
            value[-2] = 0
            value[-3] = value[-3] + 1
    # Build string
    latlon_string = ''
    for i in range(num_units):
        if padded:
            if i == 0:
                int_digits = degrees_digits
            else:
                int_digits = 2  # True for all but longitude degrees
        else:
            int_digits = 1
        if (i + 1) == num_units:
            precision_digits = precision
        else:
            precision_digits = 0
        if precision_digits > 0:
            int_digits = int_digits + 1  # Account for the decimal point
        latlon_string = '%s%0*.*f%s%s' % \
                        (latlon_string, int_digits + precision_digits, precision_digits,
                         abs(value[i]), latlon_symbols[i], delimiter[i])
    if signed:
        latlon_string = latlon_sign + latlon_string
    else:
        latlon_string = latlon_string + hemisphere
    return latlon_string


def dms(degrees):
    """
    Calculate degrees, minutes, seconds representation from decimal degrees.

    Parameters
    ----------
    degrees : float

    Returns
    -------
    (int, int, float)
    """

    degrees_int = int(abs(degrees))	 # integer degrees
    degrees_frac = abs(degrees) - degrees_int  # fractional degrees, used to compute minutes
    minutes_int = float(int(degrees_frac * 60))  # integer minutes
    minutes_frac = degrees_frac - minutes_int / 60  # fractional minutes, used to compute seconds
    seconds = minutes_frac * 3600  # decimal seconds

    # Handle sign.  Degrees portion will contain the sign of the coordinate.
    # Minutes and seconds will always be positive.
    # sign function returns -1, 0, +1 for x < 0, x == 0, x > 0, respectively
    if degrees < 0:
        degrees_int *= -1

    return degrees_int, minutes_int, seconds


def num(latlon_input):
    """
    Convert a variety of lat/long formats into decimal degree value.

    This should handle any string compliant with the ISO 6709:2008 standard
    or any of a number of variants for describing lat/long coordinates.
    Also handles degree/minutes/seconds passed in as a tuple/list/array.

    Parameters
    ----------
    latlon_input : numpy.ndarray|list|tuple|str

    Returns
    -------
    float
    """

    # Vector format degrees/minutes/seconds
    if isinstance(latlon_input, (numpy.ndarray, list, tuple)):
        if len(latlon_input) == 3:
            return latlon_input[0] + latlon_input[1]/60. + latlon_input[2]/3600.

    if not isinstance(latlon_input, str):
        raise ValueError('Expected a (degree, minutes, seconds) tuple of string. '
                         'Got type {}'.format(type(latlon_input)))
    # String input
    # Handles decimal degrees and degree/minutes/second with delimiters
    # Any non-numeric characters in string are considered delimiters
    tokens_str = list(filter(lambda x: len(x.strip()) > 0, re.split('[^.\d]', latlon_input)))
    tokens = [float(x) for x in tokens_str]
    decimal_degrees = numpy.polynomial.polynomial.polyval(1/60., numpy.abs(tokens))
    if ('W' in latlon_input or 'S' in latlon_input) != ('-' in latlon_input):
        decimal_degrees = -decimal_degrees
    # Handles degree/minutes/second with no delimiters DDD,DDDMM,DDDMMSS
    if len(tokens) == 1:
        for i in range(min(3, int(len(tokens_str[0].split('.')[0])/2)-1)):
            decimal_degrees = (numpy.fix(decimal_degrees/100) +
                               numpy.fmod(decimal_degrees, 100)/60.)
    # Error checking should occur here
    if (len(tokens) < 1 or len(tokens) > 3 or
       decimal_degrees < -180 or decimal_degrees > 360 or
       sum(c.isalpha() for c in latlon_input) > 1):
        decimal_degrees = float('nan')  # Unparseable inputs are returned as NaN
    return float(decimal_degrees)
