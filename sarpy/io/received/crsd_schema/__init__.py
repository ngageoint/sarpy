"""
This package contains the CRSD schema
"""

__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

import os
import re
from typing import List, Dict, Tuple, Union


_CRSD_DEFAULT_TUPLE = (1, 0, 0)

_the_directory = os.path.split(__file__)[0]

urn_mapping = {
    'urn:CRSD:1.0.0': {
        'version': '1.0',
        'release': '1.0.0',
        'date': '2021-06-12T00:00:00Z',
        'schema': os.path.join(_the_directory, 'CRSD_schema_V1.0.0_2021_06_12.xsd')},
}
WRITABLE_VERSIONS = ('1.0.0', )

# validate the defined paths
for key, entry in urn_mapping.items():
    schema_path = entry.get('schema', None)
    if schema_path is not None and not os.path.exists(schema_path):
        raise ValueError('`{}` has nonexistent schema path {}'.format(key, schema_path))


def get_default_tuple() -> Tuple[int, int, int]:
    """
    Get the default CRSD version tuple.

    Returns
    -------
    Tuple[int, int, int]
    """

    return _CRSD_DEFAULT_TUPLE


def get_default_version_string() -> str:
    """
    Get the default CRSD version string.

    Returns
    -------
    str
    """

    return '{}.{}.{}'.format(*_CRSD_DEFAULT_TUPLE)


def get_namespace(version: Union[str, Tuple[int, int, int]]) -> str:
    if isinstance(version, (list, tuple)):
        version = '{}.{}.{}'.format(version[0], version[1], version[2])
    return 'http://api.nsgreg.nga.mil/schema/crsd/{}'.format(version)


def check_urn(urn_string: str) -> str:
    """
    Checks that the urn string follows the correct pattern.

    Parameters
    ----------
    urn_string : str

    Raises
    ------
    ValueError
        This raises an exception for a poorly formed or unmapped CRSD urn.
    """

    if not isinstance(urn_string, str):
        raise TypeError(
            'Expected a urn input of string type, got type {}'.format(type(urn_string)))

    the_match = re.match(r'^\d.\d.\d$', urn_string)
    if the_match is not None:
        urn_string = 'urn:CRSD:{}'.format(urn_string)

    the_match = re.match(r'^urn:CRSD:\d.\d.\d$', urn_string)
    if the_match is None:
        raise ValueError(
            'Input provided as `{}`,\nbut should be of the form '
            '`urn:CRSD:<major>.<minor>.<release>'.format(urn_string))
    return urn_string


def get_urn_details(urn_string: str) -> Dict[str, str]:
    """
    Gets the associated details for the given CRSD urn, or raise an exception for
    poorly formatted or unrecognized urn.

    Parameters
    ----------
    urn_string : str

    Returns
    -------
    Dict[str, str]
    """

    urn_string = check_urn(urn_string)
    out = urn_mapping.get(urn_string, None)

    if out is None:
        raise KeyError(
            'Got correctly formatted, but unmapped CRSD urn {}.'.format(urn_string))
    return out


def get_schema_path(the_urn: str) -> str:
    """
    Gets the path to the proper schema file for the given urn.

    Parameters
    ----------
    the_urn : str

    Returns
    -------
    str
    """

    result = get_urn_details(the_urn)
    return result.get('schema', None)


def get_versions() -> List[str]:
    """
    Gets a list of recognized CRSD urns.

    Returns
    -------
    List[str]
    """

    return list(sorted(urn_mapping.keys()))
