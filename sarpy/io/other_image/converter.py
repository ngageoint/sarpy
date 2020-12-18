# -*- coding: utf-8 -*-
"""
This module provide utilities for attempting to open other image files not opened by
the sicd, sidd, or cphd reader collections.
"""

import os
import sys
import pkgutil
from importlib import import_module

from sarpy.io.general.base import BaseReader


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

    def check_module(mod_name):
        # import the module
        import_module(mod_name)
        # fetch the module from the modules dict
        module = sys.modules[mod_name]
        # see if it has an is_a function, if so, register it
        if hasattr(module, 'is_a'):
            register_opener(module.is_a)

        # walk down any subpackages
        path, fil = os.path.split(module.__file__)
        if not fil.startswith('__init__.py'):
            # there are no subpackages
            return
        for sub_module in pkgutil.walk_packages([path, ]):
            _, sub_module_name, _ = sub_module
            sub_name = "{}.{}".format(mod_name, sub_module_name)
            check_module(sub_name)

    check_module('sarpy.io.other_image')


def open_other(file_name):
    """
    Given a file, try to find and return the appropriate reader object.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    BaseReader

    Raises
    ------
    IOError
    """

    if not os.path.exists(file_name):
        raise IOError('File {} does not exist.'.format(file_name))
    # parse openers, if not already done
    parse_openers()
    # see if we can find a reader though trial and error
    for opener in _openers:
        reader = opener(file_name)
        if reader is not None:
            return reader

    # If for loop completes, no matching file format was found.
    raise IOError('Unable to determine image format.')
