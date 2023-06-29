__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

import os
import re
import logging
from typing import List, Dict, Optional


logger = logging.getLogger('validation')

_the_directory = os.path.split(__file__)[0]
urn_mapping = {
    'urn:SIDD:1.0.0': {
        'ism_urn': 'urn:us:gov:ic:ism',
        'sfa_urn': 'urn:SFA:1.2.0',
        'sicommon_urn': 'urn:SICommon:0.1',
        'version': '1.0.0',
        'release': '1.0.0',
        'date': '2011-08-31T00:00:00Z',
        'schema': os.path.join(_the_directory, 'version1', 'SIDD_schema_V1.0.0_2011_08_31.xsd')},
    'urn:SIDD:2.0.0': {
        'ism_urn': 'urn:us:gov:ic:ism:13',
        'sfa_urn': 'urn:SFA:1.2.0',
        'sicommon_urn': 'urn:SICommon:1.0',
        'version': '2.0.0',
        'release': '2.0.0',
        'date': '2019-05-31T00:00:00Z',
        'schema': os.path.join(_the_directory, 'version2', 'SIDD_schema_V2.0.0_2019_05_31.xsd')},
    'urn:SIDD:3.0.0': {
        'ism_urn': 'urn:us:gov:ic:ism:13',
        'sfa_urn': 'urn:SFA:1.2.0',
        'sicommon_urn': 'urn:SICommon:1.0',
        'version': '3.0.0',
        'release': '3.0.0',
        'date': '2020-06-02T00:00:00Z',
        'schema': os.path.join(_the_directory, 'version3', 'SIDD_schema_V3.0.0.xsd')},
}
_SIDD_SPECIFICATION_IDENTIFIER = 'SIDD Volume 1 Design & Implementation Description Document'


def get_specification_identifier() -> str:
    """
    Get the SIDD specification identifier string.

    Returns
    -------
    str
    """

    return _SIDD_SPECIFICATION_IDENTIFIER


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
        This raises an exception for a poorly formed or unmapped SIDD urn.
    """

    if not isinstance(urn_string, str):
        raise TypeError(
            'Expected a urn input of string type, got type {}'.format(type(urn_string)))

    the_match = re.match(r'^\d.\d.\d$', urn_string)
    if the_match is not None:
        urn_string = 'urn:SIDD:{}'.format(urn_string)

    the_match = re.match(r'^urn:SIDD:\d.\d.\d$', urn_string)
    if the_match is None:
        raise ValueError(
            'Input provided as `{}`,\nbut should be of the form '
            '`urn:SIDD:<major>.<minor>.<release>'.format(urn_string))
    return urn_string


def get_urn_details(urn_string: str) -> Dict[str, str]:
    """
    Gets the associated details for the given SIDD urn, or raise an exception for
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
            'Got correctly formatted, but unmapped SIDD urn {}.'.format(urn_string))
    return out


def get_schema_path(the_urn: str) -> Optional[str]:
    """
    Gets the path to the proper schema file for the given SIDD urn.

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
    Gets a list of recognized SIDD urn.

    Returns
    -------
    List[str]
    """

    return list(sorted(urn_mapping.keys()))


def validate_xml_ns(xml_ns: Dict[str, str], ns_key: str = 'default') -> bool:
    """
    Validate the parsed SIDD xml namespace dictionary. This is expected to
    accompany the use of :func:`sarpy.io.general.utils.parse_xml_from_string`.

    Parameters
    ----------
    xml_ns : dict
        The xml namespace dictionary.
    ns_key : str
        The main SIDD element or default namespace.

    Returns
    -------
    bool
    """

    if not isinstance(xml_ns, dict):
        raise ValueError('xml_ns must be a dictionary for SIDD interpretation.')

    if ns_key not in xml_ns:
        raise ValueError('ns_key must be a key in xml_ns.')

    sidd_urn = xml_ns[ns_key]

    try:
        details = get_urn_details(sidd_urn)
    except KeyError:
        logger.error('Got unmapped sidd urn `{}`'.format(sidd_urn))
        return False

    # key => (expected prefix, required)
    expected_ns = {
        'ism': ('urn:us:gov:ic:ism', True),
        'sfa': ('urn:sfa:', False),
        'sicommon': ('urn:sicommon:', True),
    }

    ns_to_add = dict()
    for expected_key, (expected_prefix, _) in expected_ns.items():
        if expected_key not in xml_ns:
            for actual_ns in xml_ns.values():
                if isinstance(actual_ns, str) and actual_ns.lower().startswith(expected_prefix):
                    ns_to_add[expected_key] = actual_ns
                    break
    xml_ns.update(ns_to_add)

    valid = True
    for key, (_, required) in expected_ns.items():
        if key in xml_ns and xml_ns[key] != details[f'{key}_urn']:
            valid = False
            logger.error(
                'SIDD: SIDD {} `{}` namespace urn is expected to be "{}", but we got "{}".\n\t'
                'Differences in standard may lead to deserialization and/or '
                'validation errors.'.format(sidd_urn, key, details[f'{key}_urn'], xml_ns[key]))
        if required and key not in xml_ns:
            valid = False
            logger.error(f'SIDD: No `{key}` namespace defined.')
    return valid
