# -*- coding: utf-8 -*-
"""
A module for performing a selection of validation checks on a SICD (nitf) file,
or the xml file containing the sicd structure.

Use the `check_file` function directly, or perform using the command line
>>> python sarpy.consistency.sicd_consistency <file_name>
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
import sys
import argparse
from lxml import etree

from sarpy.io.general.utils import parse_xml_from_string
from sarpy.io.complex.sicd_schema import get_schema_path
from sarpy.io.complex.sicd import SICDDetails
from sarpy.io.complex.sicd_elements.SICD import SICDType


def _evaluate_xml_versus_schema(xml_string, urn_string):
    """
    Check validity of the xml string versus the appropriate schema.

    Parameters
    ----------
    xml_string : str
    urn_string : str

    Returns
    -------
    bool
    """

    # get schema path
    try:
        the_schema = get_schema_path(urn_string)
    except ValueError:
        logging.error('Failed finding the schema for urn {}'.format(urn_string))
        return False
    xml_doc = etree.fromstring(xml_string)
    xml_schema = etree.XMLSchema(file=the_schema)
    validity = xml_schema.validate(xml_doc)
    if not validity:
        for entry in xml_schema.error_log:
            logging.error('validation error on line {}: {}'.format(entry.line, entry.message.encode('utf-8')))
    return validity


def _get_sicd_xml_from_nitf(sicd_details):
    """
    Fetch the xml string for the SICD.

    Parameters
    ----------
    sicd_details : SICDDetails
    """

    for i in range(sicd_details.des_subheader_offsets.size):
        subhead_bytes = sicd_details.get_des_subheader_bytes(i)
        if subhead_bytes.startswith(b'DEXML_DATA_CONTENT'):
            des_bytes = sicd_details.get_des_bytes(i).decode('utf-8').strip()
            # noinspection PyBroadException
            try:
                root_node, xml_ns = parse_xml_from_string(des_bytes)
                if 'SICD' in root_node.tag:  # namespace makes this ugly
                    return des_bytes, root_node, xml_ns
            except Exception:
                continue
        elif subhead_bytes.startswith(b'DESICD_XML'):
            # This is an old format SICD
            des_bytes = sicd_details.get_des_bytes(i).decode('utf-8').strip()
            try:
                root_node, xml_ns = parse_xml_from_string(des_bytes)
                if 'SICD' in root_node.tag:  # namespace makes this ugly
                    return des_bytes, root_node, xml_ns
            except Exception as e:
                logging.error('We found an apparent old-style SICD DES header, '
                              'but failed parsing with error {}'.format(e))
                continue
    return None, None, None


def check_file(file_name):
    """
    Check the SICD validity for the given file SICD (i.e. appropriately styled NITF)
    or xml file containing the SICD structure alone.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    bool
    """

    sicd_xml, root_node, xml_ns, urn_string = None, None, None, None
    # check if this is just an xml file
    with open(file_name, 'rb') as fi:
        initial_bits = fi.read(100)
        if initial_bits.startswith(b'<?xml') or initial_bits.startswith(b'<SICD'):
            sicd_xml = fi.read().decode('utf-8')
            root_node, xml_ns = parse_xml_from_string(sicd_xml)

    if sicd_xml is None:
        # try to first test whether this is SICD file
        try:
            sicd_details = SICDDetails(file_name)
            if not sicd_details.is_sicd:
                logging.error('File {} is a NITF file, but is apparently not a SICD file.')
                return False
            sicd_xml, root_node, xml_ns = _get_sicd_xml_from_nitf(sicd_details)
        except IOError:
            pass
    if sicd_xml is None:
        logging.error('Could not interpret file {}'.format(file_name))
        return False

    urn_string = xml_ns['default']
    valid_xml = _evaluate_xml_versus_schema(sicd_xml, urn_string)

    the_sicd = SICDType.from_node(root_node, xml_ns=xml_ns)
    valid_sicd_contents = the_sicd.is_valid(recursive=True, stack=False)
    return valid_xml & valid_sicd_contents


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SICD Consistency')
    parser.add_argument('file_name')
    parser.add_argument('-l', '--level', default='WARNING',
                        help="Logging level. Should be one of INFO, WARNING, or ERROR.")
    config = parser.parse_args()

    logging.basicConfig(level=config.level)
    logger = logging.getLogger('validation')
    logger.setLevel(config.level)
    validity = check_file(config.file_name)
    if validity:
        logging.info('\nSICD {} has been validated with no errors'.format(config.file_name))
    else:
        print('\nSICD {} has apparent errors'.format(config.file_name))
    sys.exit(int(validity))
