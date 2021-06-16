"""
A module for performing a selection of validation checks on a SIDD (nitf) file,
or an xml file containing the sidd structure.

Use the `check_file` function directly, or perform using the command line

>>> python -m sarpy.consistency.sidd_consistency <file_name>

For more information, about command line usage, see

>>> python -m sarpy.consistency.sidd_consistency --help

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

from sarpy.io.product.sidd import SIDDDetails
from sarpy.io.product.sidd_schema import get_schema_path
from sarpy.io.product.sidd2_elements.SIDD import SIDDType

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
    bool
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
            logger.error('SIDD schema validation error on line {}'
                         '\n\t{}'.format(entry.line, entry.message.encode('utf-8')))
    return validity


def _get_sidd_xml_from_nitf(sidd_details):
    """
    Fetch the collection of xml strings for the SIDD.

    Parameters
    ----------
    sidd_details : SIDDDetails
    """

    the_bytes = []
    the_root = []
    the_xml_ns = []

    for i in range(sidd_details.des_subheader_offsets.size):
        subhead_bytes = sidd_details.get_des_subheader_bytes(i)
        if subhead_bytes.startswith(b'DEXML_DATA_CONTENT'):
            des_bytes = sidd_details.get_des_bytes(i).decode('utf-8').strip()
            # noinspection PyBroadException
            try:
                root_node, xml_ns = parse_xml_from_string(des_bytes)
                if 'SIDD' in root_node.tag:  # namespace makes this ugly
                    the_bytes.append(des_bytes)
                    the_root.append(root_node)
                    the_xml_ns.append(xml_ns)
            except Exception:
                continue
        elif subhead_bytes.startswith(b'DESIDD_XML'):
            # This is an old format SIDD
            des_bytes = sidd_details.get_des_bytes(i).decode('utf-8').strip()
            try:
                root_node, xml_ns = parse_xml_from_string(des_bytes)
                if 'SIDD' in root_node.tag:  # namespace makes this ugly
                    the_bytes.append(des_bytes)
                    the_root.append(root_node)
                    the_xml_ns.append(xml_ns)
            except Exception as e:
                logger.error('We found an apparent old-style SICD DES header, '
                              'but failed parsing with error {}'.format(e))
                continue
    if len(the_bytes) < 1:
        return None, None, None
    return the_bytes, the_root, the_xml_ns


def check_file(file_name):
    """
    Check the SIDD validity for the given file SIDD (i.e. appropriately styled NITF)
    or xml file containing the SIDD structure alone.

    Parameters
    ----------
    file_name : str|SIDDDetails

    Returns
    -------
    bool
    """

    sidd_xml, root_node, xml_ns = None, None, None
    if isinstance(file_name, SIDDDetails):
        sidd_xml, root_node, xml_ns = _get_sidd_xml_from_nitf(file_name)
    else:
        # check if this is just an xml file
        with open(file_name, 'rb') as fi:
            initial_bits = fi.read(100)
            if initial_bits.startswith(b'<?xml') or initial_bits.startswith(b'<SIDD'):
                sidd_xml = fi.read().decode('utf-8')
                root_node, xml_ns = parse_xml_from_string(sidd_xml)
                sidd_xml = [sidd_xml, ]
                root_node = [root_node, ]
                xml_ns = [xml_ns, ]

        if sidd_xml is None:
            # try to first test whether this is SIDD file
            try:
                sicd_details = SIDDDetails(file_name)
                if not sicd_details.is_sidd:
                    logger.error('File {} is a NITF file, but is apparently not a SIDD file.')
                    return False
                sidd_xml, root_node, xml_ns = _get_sidd_xml_from_nitf(sicd_details)
            except SarpyIOError:
                pass

    if sidd_xml is None:
        logger.error('Could not interpret input {}'.format(file_name))
        return False

    out = True
    for i, (xml_bytes, root, ns) in enumerate(zip(sidd_xml, root_node, xml_ns)):
        urn_string = ns['default']
        valid_xml = evaluate_xml_versus_schema(xml_bytes, urn_string)
        if valid_xml is None:
            valid_xml = True

        the_sidd = SIDDType.from_node(root, xml_ns=ns, ns_key='default')
        valid_sidd_contents = the_sidd.is_valid(recursive=True, stack=False)
        out &= valid_xml & valid_sidd_contents
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SIDD Consistency')
    parser.add_argument('file_name')
    parser.add_argument(
        '-l', '--level', default='WARNING',
        choices=['INFO', 'WARNING', 'ERROR'], help="Logging level")
    config = parser.parse_args()

    logging.basicConfig(level=config.level)
    logger.setLevel(config.level)
    validity = check_file(config.file_name)
    if validity:
        logger.info('\nSIDD {} has been validated with no errors'.format(config.file_name))
    else:
        logger.error('\nSIDD {} has apparent errors'.format(config.file_name))
    sys.exit(int(validity))
