__classification__ = 'UNCLASSIFIED'
__author__ = "Thomas McCullough"

import os
import re
import logging
from typing import List


logger = logging.getLogger('validation')

_the_directory = os.path.split(__file__)[0]
urn_mapping = {
    'urn:SIDD:1.0.0': {
        'ism_urn': 'urn:us:gov:ic:ism',
        'sfa_urn': 'urn:SFA:1.2.0',
        'sicommon_urn': 'urn:SICommon:0.1',
        'version': '1.0',
        'release': '1.0.0',
        'date': '2011-08-31T00:00:00Z',
        'schema': os.path.join(_the_directory, 'version1', 'SIDD_schema_V1.0.0_2011_08_31.xsd')},
    'urn:SIDD:2.0.0': {
        'ism_urn': 'urn:us:gov:ic:ism:13',
        'sfa_urn': 'urn:SFA:1.2.0',
        'sicommon_urn': 'urn:SICommon:1.0',
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

    if not isinstance(urn_string, str):
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


def validate_xml_ns(xml_ns, ns_key='default'):
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

    def validate_ism_urn():
        if 'ism' not in xml_ns:
            the_val = None
            for key in xml_ns:
                val = xml_ns[key]
                if val.lower().startswith('urn:us:gov:ic:ism'):
                    the_val = val
            xml_ns['ism'] = the_val

        valid = True
        if 'ism' not in xml_ns:
            logger.error('SIDD: No `ism` namespace defined.')
            valid = False
        elif xml_ns['ism'] != details['ism_urn']:
            logger.error(
                'SIDD: SIDD {} `ISM` namespace urn is expected to be "{}", but we got "{}".\n\t'
                'Differences in standard may lead to deserialization and/or '
                'validation errors.'.format(sidd_urn, details['ism_urn'], xml_ns['ism']))
            valid = False
        return valid

    def validate_sfa_urn():
        if 'sfa' not in xml_ns:
            the_val = None
            for key in xml_ns:
                val = xml_ns[key]
                if val.lower().startswith('urn:sfa:'):
                    the_val = val
            xml_ns['sfa'] = the_val

        valid = True
        if 'ism' not in xml_ns:
            logger.error('SIDD: No `sfa` namespace defined.')
            valid = False
        elif xml_ns['sfa'] != details['sfa_urn']:
            logger.error(
                'SIDD: SIDD {} `SFA` namespace urn is expected to be "{}", but we got "{}".\n\t'
                'Differences in standard may lead to deserialization and/or '
                'validation errors.'.format(sidd_urn, details['sfa_urn'], xml_ns['sfa']))
            valid = False
        return valid

    def validate_sicommon_urn():
        if 'sicommon' not in xml_ns:
            the_val = None
            for key in xml_ns:
                val = xml_ns[key]
                if val.lower().startswith('urn:sicommon:'):
                    the_val = val
            xml_ns['sicommon'] = the_val

        valid = True
        if 'sicommon' not in xml_ns:
            logger.error('SIDD: No `sicommon` namespace defined.')
            valid = False
        elif xml_ns['sicommon'] != details['sicommon_urn']:
            logger.error(
                'SIDD: SIDD {} `SICommon` namespace urn is expected to be "{}", but we got "{}".\n\t'
                'Differences in standard may lead to deserialization and/or '
                'validation errors.'.format(sidd_urn, details['sicommon_urn'], xml_ns['sicommon']))
            valid = False
        return valid

    if not isinstance(xml_ns, dict):
        return ValueError('xml_ns must be a dictionary for SIDD interpretation.')

    if ns_key not in xml_ns:
        raise ValueError('ns_key must be a key in xml_ns.')

    sidd_urn = xml_ns[ns_key]

    try:
        details = get_urn_details(sidd_urn)
    except KeyError:
        logger.error('Got unmapped sidd urn `{}`'.format(sidd_urn))
        return False

    valid_ns = validate_ism_urn()
    valid_ns &= validate_sfa_urn()
    valid_ns &= validate_sicommon_urn()
    return valid_ns
