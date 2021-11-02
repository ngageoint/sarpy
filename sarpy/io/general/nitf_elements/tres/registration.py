"""
Module for maintaining the TRE registry
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas Mccullough"


import logging
import pkgutil
from importlib import import_module
import inspect

from sarpy.compliance import bytes_to_string

logger = logging.getLogger(__name__)

###############
# module variables
_TRE_Registry = {}
_parsed_package = False
_default_tre_packages = 'sarpy.io.general.nitf_elements.tres'


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

    from sarpy.io.general.nitf_elements.tres.tre_elements import TREExtension

    if not issubclass(tre_type, TREExtension):
        raise TypeError('tre_type must be a subclass of sarpy.io.general.nitf_elements.header.TRE')

    if tre_type in [TREExtension, ]:
        return

    if tre_id is None:
        tre_id = tre_type.__name__
    if not isinstance(tre_id, str):
        raise TypeError('tre_id must be a string, got type {}'.format(type(tre_id)))

    if tre_id in _TRE_Registry:
        if replace:
            logger.warning(
                'TRE with id {} is already registered.\n\t'
                'We are replacing the definition.'.format(tre_type))
        else:
            logger.warning(
                'TRE with id {} is already registered.\n\t'
                'We are NOT replacing the definition.'.format(tre_type))
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
    sarpy.io.general.nitf_elements.base.TRE|None
    """

    if not _parsed_package:
        parse_package()

    if isinstance(tre_id, bytes):
        tre_id = bytes_to_string(tre_id)
    if not isinstance(tre_id, str):
        raise TypeError('tre_id must be of type string. Got {}'.format(tre_id))
    return _TRE_Registry.get(tre_id.strip(), None)


def parse_package(packages=None):
    """
    Walk the packages contained in `packages`, find all subclasses of TRE, and register them.

    Returns
    -------
    None
    """

    def evaluate(the_module):
        for element_name, element_type in inspect.getmembers(the_module, inspect.isclass):
            if issubclass(element_type, TREExtension) and element_type != TREExtension:
                register_tre(element_type, tre_id=element_name, replace=False)

    from sarpy.io.general.nitf_elements.tres.tre_elements import TREExtension

    if packages is None:
        global _parsed_package
        if _parsed_package:
            return  # already parsed the default packages
        else:
            _parsed_package = True
            packages = _default_tre_packages

    if isinstance(packages, str):
        packages = [packages, ]

    logger.info('Finding and registering TREs contained in packages {}'.format(packages))
    # walk the packages, find all subclasses of TRE, dump them into our dictionary

    for start_package in packages:
        module = import_module(start_package)
        evaluate(module)
        for details in pkgutil.walk_packages(module.__path__, start_package + '.'):
            _, module_name, is_pkg = details
            sub_module = import_module(module_name)
            evaluate(sub_module)

    logger.info('We now have {} registered TREs'.format(len(_TRE_Registry)))
