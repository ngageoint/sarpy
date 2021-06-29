"""
Tools for inspecting a SICD urn url and providing basic details.
"""

__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

import os
import re
from typing import List

from sarpy.compliance import string_types


_the_directory = os.path.split(__file__)[0]

urn_mapping = {
    'urn:SICD:0.3.1': {
        'version': '0.3',
        'release': '0.3.1',
        'date': '2009-03-17T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V0_3_1_2009_03_17.xsd')},
    'urn:SICD:0.4.0': {
        'version': '0.4',
        'release': '0.4.0',
        'date': '2010-02-12T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V0_4_0_2010_02_12.xsd')},
    'urn:SICD:0.4.1': {
        'version': '0.4',
        'release': '0.4.1',
        'date': '2010-07-15T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V0_4_1_2010_07_15.xsd')},
    'urn:SICD:0.5.0': {
        'version': '0.5',
        'release': '0.5.0',
        'date': '2011-01-12T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V0_5_0_2011_01_12.xsd')},
    'urn:SICD:1.0.0': {
        'version': '1.0',
        'release': '1.0.0',
        'date': '2011-08-31T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1_0_0_2011_08_31.xsd')},
    'urn:SICD:1.0.1': {
        'version': '1.0',
        'release': '1.0.1',
        'date': '2013-02-25T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1_0_1_2013_02_25.xsd')},
    'urn:SICD:1.1.0': {
        'version': '1.1',
        'release': '1.1.0',
        'date': '2014-07-08T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1_1_0_2014_07_08.xsd')},
    'urn:SICD:1.2.0': {
        'version': '1.2',
        'release': '1.2.0',
        'date': '2016-06-30T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1_2_0_2016_06_30.xsd')},
    'urn:SICD:1.2.1': {
        'version': '1.2',
        'release': '1.2.1',
        'date': '2018-12-13T00:00:00Z',
        'schema': os.path.join(_the_directory, 'SICD_schema_V1_2_1_2018_12_13.xsd')}
}
_SICD_SPECIFICATION_IDENTIFIER = 'SICD Volume 1 Design & Implementation Description Document'


def get_specification_identifier():
    """
    Get the SICD specification identifier string.

    Returns
    -------
    str
    """

    return _SICD_SPECIFICATION_IDENTIFIER


def check_urn(urn_string):
    """
    Checks that the urn string follows the correct pattern. This raises an
    exception for a poorly formed or unmapped SICD urn.

    Parameters
    ----------
    urn_string : str
    """

    if not isinstance(urn_string, string_types):
        raise TypeError(
            'Expected a urn input of string type, got type {}'.format(type(urn_string)))

    the_match = re.match(r'^urn:SICD:\d.\d.\d$', urn_string)
    if the_match is None:
        raise ValueError(
            'Input provided as `{}`,\nbut should be of the form '
            '`urn:SICD:<major>.<minor>.<release>'.format(urn_string))


def get_urn_details(urn_string):
    """
    Gets the associated details for the given SICD urn, or raise an exception for
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
            'Got correctly formatted, but unmapped SICD urn {}.'.format(urn_string))
    return out


def get_schema_path(the_urn):
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
    return result['schema']


def get_versions():
    """
    Gets a list of recognized SICD urn.

    Returns
    -------
    List[str]
    """

    return list(sorted(urn_mapping.keys()))
