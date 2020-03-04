"""
Module for maintaining the TRE registry
"""

import logging

from ..headers import TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas Mccullough"

_TRE_Registry = {}
_parsed_package = False


def register_tre(tre_type, tre_id=None, replace=False):
    """
    Register a type in the TRE registry.

    Parameters
    ----------
    tre_type : type
        A subclass of TRE
    tre_id : None|str
        The id for the type. The class name will be used if not supplied.
    replace : bool
        Should we replace if a TRE with given id if already registered?

    Returns
    -------
    None
    """

    if not issubclass(tre_type, TRE):
        raise TypeError('tre_type must be a subclass of sarpy.io.nitf.header.TRE')

    if tre_id is None:
        tre_id = tre_type.__name__
    if not isinstance(tre_id, str):
        raise TypeError('tre_id must be a string, got type {}'.format(type(tre_id)))

    if tre_id in _TRE_Registry:
        if replace:
            logging.warning(
                'TRE with id {} is already registered. We are replacing the definition.'.format(tre_type))
        else:
            logging.warning(
                'TRE with id {} is already registered. We are NOT replacing the definition.'.format(tre_type))
            return
    _TRE_Registry[tre_id] = tre_type


def find_tre(tre_id):
    return _TRE_Registry.get(tre_id, None)


def parse_package():
    """
    Walk the path below sarpy.io.nitf.tres, find all subclasses of TRE, and register them

    Returns
    -------
    None
    """

    global _parsed_package
    if _parsed_package:
        return
    # walk the path below sarpy.io.nitf.tres, find all subclasses of TRE, dump them into our dictionary

    def check_module(module):
        pass

    _parsed_package = True
