"""
This module provides utilities for attempting to open other image files not
opened by the sicd, sidd, cphd, or other_image reader collections.
"""

import os
from sarpy.io.general.base import SarpyIOError, check_for_openers

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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
        This function should return a sarpy.io.general.base.BaseReader instance
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

    check_for_openers('sarpy.io.general', register_opener)


def open_general(file_name):
    """
    Given a file, try to find and return the appropriate reader object.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    sarpy.io.general.base.BaseReader

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
    raise SarpyIOError('Unable to determine image format.')
