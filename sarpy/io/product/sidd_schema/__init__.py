__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

import os
import re
from typing import List

from sarpy.compliance import string_types


_the_directory = os.path.split(__file__)[0]

urn_mapping = {
    'urn:SIDD:1.0.0': {
        'version': '1.0',
        'release': '1.0.0',
        'date': '2011-08-32T00:00:00Z',
        'schema': os.path.join(_the_directory, 'version1', 'SIDD_schema_V1.0.0_2011_08_31.xsd')},
    'urn:SIDD:2.0.0': {
        'version': '2.0',
        'release': '2.0.0',
        'date': '2019-05-31T00:00:00Z',
        'schema': os.path.join(_the_directory, 'version2', 'SIDD_schema_V2.0.0_2019_05_31.xsd')},
}
_SIDD_SPECIFICATION_IDENTIFIER = 'SIDD Volume 1 Design & Implementation Description Document'


def get_specification_identifier():
    """
    Get the SIDD specification identifier string.

    Returns
    -------
    str
    """

    return _SIDD_SPECIFICATION_IDENTIFIER


def check_urn(urn_string):
    """
    Checks that the urn string follows the correct pattern. This raises an
    exception for a poorly formed or unmapped SIDD urn.

    Parameters
    ----------
    urn_string : str
    """

    if not isinstance(urn_string, string_types):
        raise TypeError(
            'Expected a urn input of string type, got type {}'.format(type(urn_string)))

    the_match = re.match(r'^urn:SIDD:\d.\d.\d$', urn_string)
    if the_match is None:
        raise ValueError(
            'Input provided as `{}`,\nbut should be of the form '
            '`urn:SIDD:<major>.<minor>.<release>'.format(urn_string))


def get_urn_details(urn_string):
    """
    Gets the associated details for the given SIDD urn, or raise an exception for
    poorly formatted or unrecognized urn.

    Parameters
    ----------
    urn_string

    Returns
    -------
    dict
    """

    check_urn(urn_string)
    out = urn_mapping.get(urn_string, None)

    if out is None:
        raise KeyError(
            'Got correctly formatted, but unmapped SIDD urn {}.'.format(urn_string))
    return out


def get_schema_path(the_urn):
    """
    Gets the path to the proper schema file for the given SIDD urn.

    Parameters
    ----------
    the_urn : str

    Returns
    -------
    str
    """

    result = get_urn_details(the_urn)
    return result['schema']


def get_versions():
    """
    Gets a list of recognized SIDD urn.

    Returns
    -------
    List[str]
    """

    return list(sorted(urn_mapping.keys()))
