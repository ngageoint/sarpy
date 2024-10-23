"""
This module provide utilities for reading essentially Compensated Phase History Data.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import os
from typing import BinaryIO, Callable, Union

from sarpy.io.general.base import SarpyIOError, check_for_openers
from sarpy.io.phase_history.base import CPHDTypeReader
from sarpy.io.general.utils import is_file_like


###########
# Module variables
_openers = []
_parsed_openers = False


def register_opener(open_func: Callable) -> None:
    """
    Provide a new opener.

    Parameters
    ----------
    open_func : Callable
        This is required to be a function which takes a single argument (file name).
        This function should return a sarpy.io.phase_history.base.CPHDTypeReader instance
        if the referenced file is viable for the underlying type, and None otherwise.

    Returns
    -------
    None
    """

    if not callable(open_func):
        raise TypeError('open_func must be a callable')
    if open_func not in _openers:
        _openers.append(open_func)


def parse_openers() -> None:
    """
    Automatically find the viable openers (i.e. :func:`is_a`) in the various modules.
    """

    global _parsed_openers
    if _parsed_openers:
        return
    _parsed_openers = True

    check_for_openers('sarpy.io.phase_history', register_opener)


def open_phase_history(filename_or_obj:Union[str, BinaryIO]) -> CPHDTypeReader:
    """
    Given a file, try to find and return the appropriate reader object.

    Parameters
    ----------
    file_name : str|BinaryIO

    Returns
    -------
    CPHDTypeReader

    Raises
    ------
    SarpyIOError
    """

    if (not is_file_like(filename_or_obj)) and (not os.path.exists(filename_or_obj)):
        raise SarpyIOError('File {} does not exist.'.format(filename_or_obj))
    # parse openers, if not already done
    parse_openers()
    # see if we can find a reader though trial and error
    for opener in _openers:
        reader = opener(filename_or_obj)
        if reader is not None:
            return reader

    # If for loop completes, no matching file format was found.
    raise SarpyIOError('Unable to determine phase history image format.')
