"""
This package contains the CPHD schema
"""

__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

import os
import re
from typing import List, Dict, Tuple, Union


_CPHD_DEFAULT_TUPLE = (1, 1, 0)

_the_directory = os.path.split(__file__)[0]

urn_mapping = {
    'urn:CPHD:0.3.0': {
        'tuple': (0, 3, 0),
        'version': '0.3',
        'release': '0.3.0',
        'date': ''},
    'urn:CPHD:1.0.1': {
        'tuple': (1, 0, 1),
        'version': '1.0',
        'release': '1.0.1',
        'date': '2018-05-21T00:00:00Z',
        'schema': os.path.join(_the_directory, 'CPHD_schema_V1.0.1_2018_05_21.xsd')},
    'urn:CPHD:1.1.0': {
        'tuple': (1, 1, 0),
        'version': '1.1',
        'release': '1.1.0',
        'date': '2022-06-09T00:00:00Z',
        'schema': os.path.join(_the_directory, 'CPHD_schema_V1.1.0_2022_06_23.xsd')},
}
WRITABLE_VERSIONS = ('1.0.1', '1.1.0')

# validate the defined paths
for key, entry in urn_mapping.items():
    schema_path = entry.get('schema', None)
    if schema_path is not None and not os.path.exists(schema_path):
        raise ValueError('`{}` has nonexistent schema path {}'.format(key, schema_path))


def get_default_tuple() -> Tuple[int, int, int]:
    """
    Get the default CPHD version tuple.

    Returns
    -------
    Tuple[int, int, int]
    """

    return _CPHD_DEFAULT_TUPLE


def get_default_version_string() -> str:
    """
    Get the default CPHD version string.

    Returns
    -------
    str
    """

    return '{}.{}.{}'.format(*_CPHD_DEFAULT_TUPLE)


def get_namespace(version: Union[str, Tuple[int, int, int]]) -> str:
    if isinstance(version, (list, tuple)):
        version = '{}.{}.{}'.format(version[0], version[1], version[2])
    return 'http://api.nsgreg.nga.mil/schema/cphd/{}'.format(version)


def check_urn(urn_string: str) -> str:
    """
    Checks that the urn string follows the correct pattern.

    Parameters
    ----------
    urn_string : str

    Raises
    ------
    ValueError
        This raises an exception for a poorly formed or unmapped CPHD urn.
    """

    if not isinstance(urn_string, str):
        raise TypeError(
            'Expected a urn input of string type, got type {}'.format(type(urn_string)))

    the_match = re.match(r'^\d.\d.\d$', urn_string)
    if the_match is not None:
        urn_string = 'urn:CPHD:{}'.format(urn_string)

    the_match = re.match(r'^urn:CPHD:\d.\d.\d$', urn_string)
    if the_match is None:
        raise ValueError(
            'Input provided as `{}`,\nbut should be of the form '
            '`urn:CPHD:<major>.<minor>.<release>'.format(urn_string))
    return urn_string


def get_urn_details(urn_string: str) -> Dict[str, str]:
    """
    Gets the associated details for the given CPHD urn, or raise an exception for
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
            'Got correctly formatted, but unmapped CPHD urn {}.'.format(urn_string))
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
    Gets a list of recognized CPHD urns.

    Returns
    -------
    List[str]
    """

    return list(sorted(urn_mapping.keys(), key=lambda x: urn_mapping[x]['tuple']))
