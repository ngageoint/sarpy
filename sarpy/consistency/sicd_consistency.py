"""
A module for performing a selection of validation checks on a SICD (nitf) file,
or the xml file containing the sicd structure.

Use the `check_file` function directly, or perform using the command line

>>> python -m sarpy.consistency.sicd_consistency <file_name>

For more information, about command line usage, see

>>> python -m sarpy.consistency.sicd_consistency --help

"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
import sys
import argparse
try:
    from lxml import etree
except ImportError:
    etree = None

from sarpy.io.general.base import SarpyIOError
from sarpy.io.general.utils import parse_xml_from_string
from sarpy.io.complex.sicd_schema import get_schema_path
from sarpy.io.complex.sicd import SICDDetails, SICDType

logger = logging.getLogger('validation')


def evaluate_xml_versus_schema(xml_string, urn_string):
    """
    Check validity of the xml string versus the appropriate schema.

    Parameters
    ----------
    xml_string : str
    urn_string : str

    Returns
    -------
    None|bool
    """

    if etree is None:
        return None

    # get schema path
    try:
        the_schema = get_schema_path(urn_string)
    except ValueError:
        logger.error('Failed finding the schema for urn {}'.format(urn_string))
        return False
    xml_doc = etree.fromstring(xml_string)
    xml_schema = etree.XMLSchema(file=the_schema)
    validity = xml_schema.validate(xml_doc)
    if not validity:
        for entry in xml_schema.error_log:
            logger.error('SICD schema validation error on line {}'
                         '\n\t{}'.format(entry.line, entry.message.encode('utf-8')))
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
                logger.error('We found an apparent old-style SICD DES header, '
                              'but failed parsing with error {}'.format(e))
                continue
    return None, None, None


def check_file(file_name):
    """
    Check the SICD validity for the given file SICD (i.e. appropriately styled NITF)
    or xml file containing the SICD structure alone.

    Parameters
    ----------
    file_name : str|SICDDetails

    Returns
    -------
    bool
    """

    sicd_xml, root_node, xml_ns, urn_string = None, None, None, None
    if isinstance(file_name, SICDDetails):
        sicd_xml, root_node, xml_ns = _get_sicd_xml_from_nitf(file_name)
    else:
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
                    logger.error('File {} is a NITF file, but is apparently not a SICD file.')
                    return False
                sicd_xml, root_node, xml_ns = _get_sicd_xml_from_nitf(sicd_details)
            except SarpyIOError:
                pass

    if sicd_xml is None:
        logger.error('Could not interpret input {}'.format(file_name))
        return False

    urn_string = xml_ns['default']
    valid_xml = evaluate_xml_versus_schema(sicd_xml, urn_string)
    if valid_xml is None:
        valid_xml = True

    the_sicd = SICDType.from_node(root_node, xml_ns=xml_ns)
    valid_sicd_contents = the_sicd.is_valid(recursive=True, stack=False)
    return valid_xml & valid_sicd_contents


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SICD Consistency')
    parser.add_argument('file_name')
    parser.add_argument(
        '-l', '--level', default='WARNING',
        choices=['INFO', 'WARNING', 'ERROR'], help="Logging level")
    config = parser.parse_args()

    logging.basicConfig(level=config.level)
    logger.setLevel(config.level)
    validity = check_file(config.file_name)
    if validity:
        logger.info('\nSICD {} has been validated with no errors'.format(config.file_name))
    else:
        logger.error('\nSICD {} has apparent errors'.format(config.file_name))
    sys.exit(int(validity))
