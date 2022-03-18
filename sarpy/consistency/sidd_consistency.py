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
import os
import re

from sarpy.consistency.sicd_consistency import check_sicd_data_extension
from sarpy.io.general.nitf import NITFDetails
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, \
    DataExtensionHeader0
from sarpy.io.xml.base import parse_xml_from_string, validate_xml_from_string
from sarpy.io.product.sidd_schema import get_schema_path, get_urn_details, \
    get_specification_identifier

from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1


logger = logging.getLogger('validation')


def evaluate_xml_versus_schema(xml_string, urn_string):
    """
    Check validity of the xml string versus the appropriate schema.

    Parameters
    ----------
    xml_string : str|bytes
    urn_string : str

    Returns
    -------
    bool
    """

    # get schema path
    try:
        the_schema = get_schema_path(urn_string)
    except KeyError:
        logger.exception('SIDD: Failed finding the schema for urn {}'.format(urn_string))
        return False

    try:
        return validate_xml_from_string(xml_string, the_schema, output_logger=logger)
    except ImportError:
        return None


def _evaluate_xml_string_validity(xml_string=None):
    """
    Check the validity of the SIDD xml, as defined by the given string.

    Parameters
    ----------
    xml_string : str|bytes

    Returns
    -------
    is_valid : bool
    sidd_urn : str
    sidd : SIDDType1|SIDDType2
    """

    root_node, xml_ns = parse_xml_from_string(xml_string)
    if 'default' not in xml_ns:
        raise ValueError(
            'Could not properly interpret the namespace collection from xml\n{}'.format(xml_ns))

    # todo: validate the xml namespace elements here

    sidd_urn = xml_ns['default']
    # check that our urn is mapped
    try:
        _ = get_urn_details(sidd_urn)
        check_schema = True
    except Exception as e:
        logger.exception('SIDD: The SIDD namespace has unrecognized value')
        check_schema = False

    valid_xml = None
    if check_schema:
        valid_xml = evaluate_xml_versus_schema(xml_string, sidd_urn)
    if valid_xml is None:
        valid_xml = True

    # perform the various sidd structure checks
    if sidd_urn == 'urn:SIDD:1.0.0':
        the_sidd = SIDDType1.from_node(root_node, xml_ns=xml_ns)
        valid_sidd_contents = the_sidd.is_valid(recursive=True, stack=False)
    elif sidd_urn == 'urn:SIDD:2.0.0':
        the_sidd = SIDDType2.from_node(root_node, xml_ns=xml_ns)
        valid_sidd_contents = the_sidd.is_valid(recursive=True, stack=False)
    else:
        raise ValueError('Got unhandled urn {}'.format(sidd_urn))
    return valid_xml & valid_sidd_contents, sidd_urn, the_sidd


def check_sidd_data_extension(nitf_details, des_header, xml_string):
    """
    Evaluate a SIDD data extension for validity.

    Parameters
    ----------
    nitf_details : NITFDetails
    des_header : DataExtensionHeader|DataExtensionHeader0
    xml_string : str|bytes

    Returns
    -------
    is_valid : bool
    sidd : SIDDType1|SIDDType2
    """

    def check_des_header_fields():
        # type: () -> bool

        des_id = des_header.DESID.strip() if nitf_details.nitf_version == '02.10' else des_header.DESTAG.strip()

        if des_id != 'XML_DATA_CONTENT':
            logger.warning('SIDD: Found old style SIDD DES Header. This is deprecated.')
            return True

        # make sure that the NITF urn is evaluated for sensibility
        nitf_urn = des_header.UserHeader.DESSHTN.strip()
        try:
            nitf_urn_details = get_urn_details(nitf_urn)
        except Exception:
            logger.exception('SIDD: The SIDD DES.DESSHTN must be a recognized urn')
            return False

        # make sure that the NITF urn and SICD urn actually agree
        header_good = True
        if nitf_urn != xml_urn:
            logger.error('SIDD: The SIDD DES.DESSHTN ({}) and urn ({}) must agree'.format(nitf_urn, xml_urn))
            header_good = False

        # make sure that the NITF DES fields are populated appropriately for NITF urn
        if des_header.UserHeader.DESSHSI.strip() != get_specification_identifier():
            logger.error(
                'SIDD: DES.DESSHSI has value `{}`,\n\tbut should have value `{}`'.format(
                    des_header.UserHeader.DESSHSI.strip(), get_specification_identifier()))
            header_good = False

        nitf_version = nitf_urn_details['version']
        if des_header.UserHeader.DESSHSV.strip() != nitf_version:
            logger.error(
                'SIDD: DES.DESSHSV has value `{}`,\n\tbut should have value `{}` based on DES.DESSHTN `{}`'.format(
                    des_header.UserHeader.DESSHSV.strip(), nitf_version, nitf_urn))
            header_good = False

        nitf_date = nitf_urn_details['date']
        if des_header.UserHeader.DESSHSD.strip() != nitf_date:
            logger.warning(
                'SIDD: DES.DESSHSD has value `{}`,\n\tbut should have value `{}` based on DES.DESSHTN `{}`'.format(
                    des_header.UserHeader.DESSHSV.strip(), nitf_date, nitf_urn))
        return header_good

    def compare_sidd_class():
        # type: () -> bool

        if the_sidd.ProductCreation is None or the_sidd.ProductCreation.Classification is None or \
                the_sidd.ProductCreation.Classification.classification is None:
            logger.error(
                'SIDD: SIDD.ProductCreation.Classification.classification is not populated,\n\t'
                'so can not be compared with SIDD DES.DESCLAS `{}`'.format(des_header.Security.CLAS.strip()))
            return False

        extracted_class = the_sidd.ProductCreation.Classification.classification
        if extracted_class != des_header.Security.CLAS.strip():
            logger.warning(
                'SIDD: DES.DESCLAS is `{}`,\n\tand SIDD.ProductCreation.Classification.classification '
                'is {}'.format(des_header.Security.CLAS.strip(), extracted_class))

        if des_header.Security.CLAS.strip() != nitf_details.nitf_header.Security.CLAS.strip():
            logger.warning(
                'SIDD: DES.DESCLAS is `{}`,\n\tand NITF.CLAS is `{}`'.format(
                    des_header.Security.CLAS.strip(), nitf_details.nitf_header.Security.CLAS.strip()))
        return True

    # check sicd xml structure for validity
    valid_sicd, xml_urn, the_sidd = _evaluate_xml_string_validity(xml_string)
    # check that the sicd information and header information appropriately match
    valid_header = check_des_header_fields()
    # check that the classification seems to make sense
    valid_class = compare_sidd_class()
    return valid_sicd & valid_header & valid_class, the_sidd


def check_sidd_file(nitf_details):
    """
    Check the validity of the given NITF file as a SICD file.

    Parameters
    ----------
    nitf_details : str|NITFDetails

    Returns
    -------
    bool
    """

    def find_des():
        for i in range(nitf_details.des_subheader_offsets.size):
            subhead_bytes = nitf_details.get_des_subheader_bytes(i)
            if nitf_details.nitf_version == '02.00':
                des_header = DataExtensionHeader0.from_bytes(subhead_bytes, start=0)
            elif nitf_details.nitf_version == '02.10':
                des_header = DataExtensionHeader.from_bytes(subhead_bytes, start=0)
            else:
                raise ValueError('Got unhandled NITF version {}'.format(nitf_details.nitf_version))

            if subhead_bytes.startswith(b'DEXML_DATA_CONTENT'):
                des_bytes = nitf_details.get_des_bytes(i).decode('utf-8').strip().encode()
                # noinspection PyBroadException
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SIDD' in root_node.tag:  # namespace makes this ugly
                        sidd_des.append((i, des_bytes, root_node, xml_ns, des_header))
                    elif 'SICD' in root_node.tag:
                        sicd_des.append((i, des_bytes, root_node, xml_ns, des_header))
                except Exception:
                    continue
            elif subhead_bytes.startswith(b'DESIDD_XML'):
                # This is an old format SIDD
                des_bytes = nitf_details.get_des_bytes(i).decode('utf-8').strip().encode()
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SIDD' in root_node.tag:  # namespace makes this ugly
                        sidd_des.append((i, des_bytes, root_node, xml_ns, des_header))
                except Exception as e:
                    logger.exception('SIDD: Old-style SIDD DES header at index {}, but failed parsing'.format(i))
                    continue
            elif subhead_bytes.startswith(b'DESICD_XML'):
                # This is an old format SICD
                des_bytes = nitf_details.get_des_bytes(i).decode('utf-8').strip().encode()
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SICD' in root_node.tag:  # namespace makes this ugly
                        sicd_des.append((i, des_bytes, root_node, xml_ns, des_header))
                except Exception as e:
                    logger.exception('SIDD: Old-style SICD DES header at index {}, but failed parsing'.format(i))
                    continue

    def check_image_data():
        valid_images = True
        # verify that all images have the correct pixel type
        for i, img_header in enumerate(nitf_details.img_headers):
            if img_header.ICAT.strip() != 'SAR':
                continue

            iid1 = img_header.IID1.strip()
            if re.match(r'^SIDD\d\d\d\d\d\d', iid1) is None:
                valid_images = False
                logger.error(
                    'SIDD: image segment at index {} of {} has IID1 = `{}`,\n\t'
                    'expected to be of the form `SIDDXXXYYY`'.format(i, len(nitf_details.img_headers), iid1))
                continue

            sidd_index = int(iid1[4:7])
            if not (0 < sidd_index <= len(sidd_des)):
                valid_images = False
                logger.error(
                    'SIDD: image segment at index {} of {} has IID1 = `{}`,\n\t'
                    'it is unclear with which of the {} SIDDs '
                    'this is associated'.format(i, len(nitf_details.img_headers), iid1, len(sidd_des)))
                continue

            has_image[sidd_index - 1] = True

            type_information = sidd_nitf_details[sidd_index - 1]
            pixel_type = type_information['pixel_type']
            if pixel_type is None:
                continue  # we already noted the failure here

            exp_nbpp, exp_pvtype  = type_information['nbpp'], type_information['pvtype']
            if img_header.PVTYPE.strip() != exp_pvtype:
                valid_images = False
                logger.error(
                    'SIDD: image segment at index {} of {} has PVTYPE = `{}`,\n\t'
                    'expected to be `{}` based on pixel type {}'.format(
                        i, len(nitf_details.img_headers), img_header.PVTYPE.strip(), exp_pvtype, pixel_type))
            if img_header.NBPP != exp_nbpp:
                valid_images = False
                logger.error(
                    'SIDD: image segment at index {} of {} has NBPP = `{}`,\n\t'
                    'expected to be `{}` based on pixel type {}'.format(
                        i, len(nitf_details.img_headers), img_header.NBPP, exp_nbpp, pixel_type))

        for sidd_index, entry in enumerate(has_image):
            if not entry:
                logger.error(
                    'SIDD: No image segments appear to be associated with the sidd at index {}'.format(sidd_index))
                valid_images = False

        return valid_images

    if isinstance(nitf_details, str):
        if not os.path.isfile(nitf_details):
            raise ValueError('Got string input, but it is not a valid path')
        nitf_details = NITFDetails(nitf_details)

    if not isinstance(nitf_details, NITFDetails):
        raise TypeError(
            'Input is expected to be a path to a NITF file, or a NITFDetails object instance')

    sidd_des = []
    sicd_des = []
    sidd_nitf_details = []
    has_image = []

    find_des()
    if len(sidd_des) < 1:
        logger.error('SIDD: No SIDD DES found, this is not a valid SIDD file.')
        return False

    valid_sidd_des = True
    for entry in sidd_des:
        this_sidd_valid, the_sidd = check_sidd_data_extension(nitf_details, entry[4], entry[1])
        valid_sidd_des &= this_sidd_valid
        has_image.append(False)
        if the_sidd.Display is None or the_sidd.Display.PixelType is None:
            valid_sidd_des = False
            logger.error('SIDD: SIDD.Display.PixelType is not populated, and can not be compared to NITF image details')
            sidd_nitf_details.append({'pixel_type': None})
        elif the_sidd.Display.PixelType in ['MONO8I', 'MONO8LU', 'RGB8LU']:
            sidd_nitf_details.append({'nbpp': 8, 'pvtype': 'INT', 'pixel_type': the_sidd.Display.PixelType})
        elif the_sidd.Display.PixelType == 'MONO16I':
            sidd_nitf_details.append({'nbpp': 16, 'pvtype': 'INT', 'pixel_type': the_sidd.Display.PixelType})
        elif the_sidd.Display.PixelType == 'RGB24I':
            sidd_nitf_details.append({'nbpp': 24, 'pvtype': 'INT', 'pixel_type': the_sidd.Display.PixelType})
        else:
            raise ValueError('Got unhandled pixel type {}'.format(the_sidd.Display.PixelType))

    valid_sicd_des = True
    for entry in sicd_des:
        this_sicd_valid, _ = check_sicd_data_extension(nitf_details, entry[4], entry[1])
        valid_sicd_des &= this_sicd_valid
    valid_image = check_image_data()

    return valid_sidd_des & valid_sicd_des & valid_image


def check_file(file_name):
    """
    Check the validity for the given file SIDD (i.e. appropriately styled NITF)
    or xml file containing the SIDD structure alone.

    Parameters
    ----------
    file_name : str|NITFDetails

    Returns
    -------
    bool
    """

    if isinstance(file_name, str):
        if not os.path.isfile(file_name):
            raise ValueError('Got string input, but it is not a valid path')

        # check if this is just an xml file
        with open(file_name, 'rb') as fi:
            initial_bits = fi.read(30)
            if initial_bits.startswith(b'<?xml') or initial_bits.startswith(b'<SIDD'):
                sicd_xml = fi.read().decode('utf-8')
                return _evaluate_xml_string_validity(sicd_xml)[0]

    return check_sidd_file(file_name)


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
        logger.info('\nSIDD: {} has been validated with no errors'.format(config.file_name))
    else:
        logger.error('\nSIDD: {} has apparent errors'.format(config.file_name))
    sys.exit(int(validity))
