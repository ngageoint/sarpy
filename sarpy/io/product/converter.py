"""
This module provide utilities for opening any files analogous to Sensor
Independent Derived Data, namely detected images in the ground plane.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import os

from sarpy.io.general.base import SarpyIOError, check_for_openers
from sarpy.io.product.base import SIDDTypeReader

###########
# Module variables
_openers = []
_parsed_openers = False


def register_opener(open_func):
    """
    Provide a new opener.

    Parameters
    ----------
    open_func
        This is required to be a function which takes a single argument (file name).
        This function should return a sarpy.io.product.base.SIDDTypeReader instance
        if the referenced file is viable for the underlying type, and None otherwise.

    Returns
    -------
    None
    """

    if not callable(open_func):
        raise TypeError('open_func must be a callable')
    if open_func not in _openers:
        _openers.append(open_func)


def parse_openers():
    """
    Automatically find the viable openers (i.e. :func:`is_a`) in the various modules.

    Returns
    -------

    """

    global _parsed_openers
    if _parsed_openers:
        return
    _parsed_openers = True

    check_for_openers('sarpy.io.product', register_opener)


def open_product(file_name):
    """
    Given a file, try to find and return the appropriate reader object.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    SIDDTypeReader

    Raises
    ------
    SarpyIOError
    """

    if not os.path.exists(file_name):
        raise SarpyIOError('File {} does not exist.'.format(file_name))
    # parse openers, if not already done
    parse_openers()
    # see if we can find a reader though trial and error
    for opener in _openers:
        reader = opener(file_name)
        if reader is not None:
            return reader

    # If for loop completes, no matching file format was found.
    raise SarpyIOError('Unable to determine product image format.')
