"""
Module for maintaining the TRE registry
"""

import logging
import pkgutil
import inspect
import os

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas Mccullough"

_TRE_Registry = {}
_parsed_package = False
_default_tre_packages = 'sarpy.io.nitf.tres'


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

    from sarpy.io.nitf.headers import TRE, UnknownTRE

    if not issubclass(tre_type, TRE):
        raise TypeError('tre_type must be a subclass of sarpy.io.nitf.header.TRE')

    if tre_type in [TRE, UnknownTRE]:
        return

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
    """
    Try to find a TRE with given id in our registry. Return `None` if not found.

    Parameters
    ----------
    tre_id : str|bytes

    Returns
    -------
    sarpy.io.nitf.headers.TRE|None
    """

    if not _parsed_package:
        parse_package()

    if isinstance(tre_id, bytes):
        tre_id = tre_id.decode('utf-8')
    if not isinstance(tre_id, str):
        raise TypeError('tre_id must be of type string. Got {}'.format(tre_id))
    return _TRE_Registry.get(tre_id.strip(), None)


def parse_package(packages=None):
    """
    Walk the packages contained in `packages`, find all subclasses of TRE, and register them

    Returns
    -------
    None
    """

    from sarpy.io.nitf.headers import TRE

    if packages is None:
        global _parsed_package
        if _parsed_package:
            return  # already parsed the default packages
        else:
            _parsed_package = True
            packages = _default_tre_packages

    if isinstance(packages, str):
        packages = [packages, ]

    logging.info('Finding and registering TREs contained in packages {}'.format(packages))
    # walk the packages, find all subclasses of TRE, dump them into our dictionary

    def check_module(module_name):
        # get the module loader
        loader = pkgutil.get_loader(module_name)
        # load the module
        module = loader.load_module(module_name)
        # check all classes of the module itself
        for element_name, element_type in inspect.getmembers(module, inspect.isclass):
            if issubclass(element_type, TRE) and element_type != TRE:
                register_tre(element_type, tre_id=element_name, replace=False)
        # walk down any subpackages
        path, fil = os.path.split(module.__file__)
        if fil != '__init__.py':
            # there are no subpackages
            return
        for sub_module in pkgutil.walk_packages([path, ]):
            _, sub_module_name, _ = sub_module
            sub_name = "{}.{}".format(module_name, sub_module_name)
            check_module(sub_name)
    for pack in packages:
        check_module(pack)
    logging.info('We now have {} registered TREs'.format(len(_TRE_Registry)))
