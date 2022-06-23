"""
Tools for inspecting a SICD urn url and providing basic details.
"""

__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

import os
import re
from typing import List, Tuple, Dict, Optional


_SICD_DEFAULT_TUPLE = (1, 3, 0)
_SICD_SPECIFICATION_IDENTIFIER = 'SICD Volume 1 Design & Implementation Description Document'

_the_directory = os.path.split(__file__)[0]
urn_mapping = {
    'urn:SICD:0.3.1': {
        'tuple': (0, 3, 1),
        'version': '0.3',
        'release': '0.3.1',
        'date': '2009-03-17T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V0.3.1_2009_03_17.xsd')},
    'urn:SICD:0.4.0': {
        'tuple': (0, 4, 0),
        'version': '0.4',
        'release': '0.4.0',
        'date': '2010-02-12T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V0.4.0_2010_02_12.xsd')},
    'urn:SICD:0.4.1': {
        'tuple': (0, 4, 1),
        'version': '0.4',
        'release': '0.4.1',
        'date': '2010-07-15T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V0.4.1_2010_07_15.xsd')},
    'urn:SICD:0.5.0': {
        'tuple': (0, 5, 0),
        'version': '0.5',
        'release': '0.5.0',
        'date': '2011-01-12T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V0.5.0_2011_01_12.xsd')},
    'urn:SICD:1.0.0': {
        'tuple': (1, 0, 0),
        'version': '1.0',
        'release': '1.0.0',
        'date': '2011-08-31T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1.0.0_2011_08_31.xsd')},
    'urn:SICD:1.0.1': {
        'tuple': (1, 0, 1),
        'version': '1.0',
        'release': '1.0.1',
        'date': '2013-02-25T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1.0.1_2013_02_25.xsd')},
    'urn:SICD:1.1.0': {
        'tuple': (1, 1, 0),
        'version': '1.1',
        'release': '1.1.0',
        'date': '2014-09-30T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1.1.0_2014_09_30.xsd')},
    'urn:SICD:1.2.0': {
        'tuple': (1, 2, 0),
        'version': '1.2',
        'release': '1.2.0',
        'date': '2016-06-30T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1.2.0_2016_06_30.xsd')},
    'urn:SICD:1.2.1': {
        'tuple': (1, 2, 1),
        'version': '1.2',
        'release': '1.2.1',
        'date': '2018-12-13T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1.2.1_2018_12_13.xsd')},
    'urn:SICD:1.3.0': {
        'tuple': (1, 3, 0),
        'version': '1.3',
        'release': '1.3.0',
        'date': '2022-06-09T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1.3.0_2021_11_30.xsd')}
}
WRITABLE_VERSIONS = tuple(entry['release'] for key, entry in urn_mapping.items() if entry['tuple'] >= (1, 0, 0))

# validate the defined paths
for key, entry in urn_mapping.items():
    schema_path = entry.get('schema', None)
    if schema_path is not None and not os.path.exists(schema_path):
        raise ValueError('`{}` has nonexistent schema path {}'.format(key, schema_path))


def get_default_tuple() -> Tuple[int, int, int]:
    """
    Get the default SICD version tuple.

    Returns
    -------
    Tuple[int, int, int]
    """

    return _SICD_DEFAULT_TUPLE


def get_default_version_string() -> str:
    """
    Get the default SICD version string.

    Returns
    -------
    str
    """

    return '{}.{}.{}'.format(*_SICD_DEFAULT_TUPLE)


def get_specification_identifier() -> str:
    """
    Get the SICD specification identifier string.

    Returns
    -------
    str
    """

    return _SICD_SPECIFICATION_IDENTIFIER


def check_urn(urn_string: str) -> str:
    """
    Checks that the urn string follows the correct pattern.

    Parameters
    ----------
    urn_string : str

    Returns
    -------
    str

    Raises
    ------
    ValueError
        This raises an exception for a poorly formed or unmapped SICD urn.
    """

    if not isinstance(urn_string, str):
        raise TypeError(
            'Expected a urn input of string type, got type {}'.format(type(urn_string)))

    the_match = re.match(r'^\d.\d.\d$', urn_string)
    if the_match is not None:
        urn_string = 'urn:SICD:{}'.format(urn_string)

    the_match = re.match(r'^urn:SICD:\d.\d.\d$', urn_string)
    if the_match is None:
        raise ValueError(
            'Input provided as `{}`,\nbut should be of the form '
            '`urn:SICD:<major>.<minor>.<release>'.format(urn_string))
    return urn_string


def get_urn_details(urn_string: str) -> Dict[str, str]:
    """
    Gets the associated details for the given SICD urn, or raise an exception for
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
            'Got correctly formatted, but unmapped SICD urn {}.'.format(urn_string))
    return out


def get_schema_path(the_urn: str) -> Optional[str]:
    """
    Gets the path to the proper schema file for the given urn.

    Parameters
    ----------
    the_urn : str

    Returns
    -------
    None|str
    """

    result = get_urn_details(the_urn)
    return result.get('schema', None)


def get_versions() -> List[str]:
    """
    Gets a list of recognized SICD urn.

    Returns
    -------
    List[str]
    """

    return list(sorted(urn_mapping.keys(), key=lambda x: urn_mapping[x]['tuple']))
