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
import os
from typing import Union

from sarpy.compliance import string_types
from sarpy.io.xml.base import parse_xml_from_string
from sarpy.io.general.nitf import NITFDetails
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, \
    DataExtensionHeader0
from sarpy.io.complex.sicd import SICDReader, SICDDetails, extract_clas
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_schema import get_urn_details, get_schema_path, \
    get_specification_identifier

try:
    from lxml import etree
except ImportError:
    etree = None

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
    except Exception as e:
        logger.exception('SICD: Failed getting the schema for urn {}'.format(urn_string))
        return False

    xml_doc = etree.fromstring(xml_string)
    xml_schema = etree.XMLSchema(file=the_schema)
    validity = xml_schema.validate(xml_doc)
    if not validity:
        for entry in xml_schema.error_log:
            logger.error('SICD: SICD schema validation error on line {}'
                         '\n\t{}'.format(entry.line, entry.message.encode('utf-8')))
    return validity


def _evaluate_xml_string_validity(xml_string):
    """
    Check the validity of the SICD xml, as defined by the given string.

    Parameters
    ----------
    xml_string : str

    Returns
    -------
    (bool, str, SICDType)
    """

    root_node, xml_ns = parse_xml_from_string(xml_string)
    if xml_ns is None:
        raise ValueError(
            'SICD XML invalid, because no apparent namespace defined in the xml,\n\t'
            'which starts `{}...`'.format(xml_string[:15]))

    if 'default' not in xml_ns:
        raise ValueError(
            'Could not properly interpret the namespace collection from xml\n{}'.format(xml_ns))

    sicd_urn = xml_ns['default']
    # check that our urn is mapped
    try:
        _ = get_urn_details(sicd_urn)
        check_schema = True
    except Exception as e:
        logger.exception('SICD: The SICD namespace has unrecognized value')
        check_schema = False

    valid_xml = None
    if check_schema:
        valid_xml = evaluate_xml_versus_schema(xml_string, sicd_urn)
    if valid_xml is None:
        valid_xml = True

    # perform the various sicd structure checks
    the_sicd = SICDType.from_node(root_node, xml_ns=xml_ns)
    valid_sicd_contents = the_sicd.is_valid(recursive=True, stack=False)
    return valid_xml & valid_sicd_contents, sicd_urn, the_sicd


def check_sicd_data_extension(nitf_details, des_header, xml_string):
    """
    Evaluate a SICD data extension for validity.

    Parameters
    ----------
    nitf_details : NITFDetails
    des_header : DataExtensionHeader|DataExtensionHeader0
    xml_string : str

    Returns
    -------
    (bool, SICDType)
    """

    def check_des_header_fields():
        # type: () -> bool

        des_id = des_header.DESID.strip() if nitf_details.nitf_version == '02.10' else des_header.DESTAG.strip()

        if des_id != 'XML_DATA_CONTENT':
            logger.warning('SICD: Found old style SICD DES Header. This is deprecated.')
            return True

        # make sure that the NITF urn is evaluated for sensibility
        nitf_urn = des_header.UserHeader.DESSHTN.strip()
        try:
            nitf_urn_details = get_urn_details(nitf_urn)
        except Exception:
            logger.exception('SICD: The SICD DES.DESSHTN must be a recognized urn')
            return False

        # make sure that the NITF urn and SICD urn actually agree
        header_good = True
        if nitf_urn != xml_urn:
            logger.error('SICD: The SICD DES.DESSHTN ({}) and urn ({}) must agree'.format(nitf_urn, xml_urn))
            header_good = False

        # make sure that the NITF DES fields are populated appropriately for NITF urn
        if des_header.UserHeader.DESSHSI.strip() != get_specification_identifier():
            logger.error(
                'SICD: DES.DESSHSI has value `{}`,\n\tbut should have value `{}`'.format(
                    des_header.UserHeader.DESSHSI.strip(), get_specification_identifier()))
            header_good = False

        nitf_version = nitf_urn_details['version']
        if des_header.UserHeader.DESSHSV.strip() != nitf_version:
            logger.error(
                'SICD: DES.DESSHSV has value `{}`,\n\tbut should have value `{}` based on DES.DESSHTN `{}`'.format(
                    des_header.UserHeader.DESSHSV.strip(), nitf_version, nitf_urn))
            header_good = False

        nitf_date = nitf_urn_details['date']
        if des_header.UserHeader.DESSHSD.strip() != nitf_date:
            logger.warning(
                'SICD: DES.DESSHSD has value `{}`,\n\tbut should have value `{}` based on DES.DESSHTN `{}`'.format(
                    des_header.UserHeader.DESSHSD.strip(), nitf_date, nitf_urn))
        return header_good

    def compare_sicd_class():
        # type: () -> bool

        if the_sicd.CollectionInfo is None or the_sicd.CollectionInfo.Classification is None:
            logger.error(
                'SICD: SICD.CollectionInfo.Classification is not populated,\n\t'
                'so can not be compared with SICD DES.DESCLAS `{}`'.format(des_header.Security.CLAS.strip()))
            return False

        sicd_class = the_sicd.CollectionInfo.Classification
        extracted_class = extract_clas(the_sicd)
        if extracted_class != des_header.Security.CLAS.strip():
            logger.warning(
                'SICD: DES.DESCLAS is `{}`,\n\tand SICD.CollectionInfo.Classification '
                'is {}'.format(des_header.Security.CLAS.strip(), sicd_class))

        if des_header.Security.CLAS.strip() != nitf_details.nitf_header.Security.CLAS.strip():
            logger.warning(
                'SICD: DES.DESCLAS is `{}`,\n\tand NITF.CLAS is `{}`'.format(
                    des_header.Security.CLAS.strip(), nitf_details.nitf_header.Security.CLAS.strip()))
        return True

    # check sicd xml structure for validity
    valid_sicd, xml_urn, the_sicd = _evaluate_xml_string_validity(xml_string)
    # check that the sicd information and header information appropriately match
    valid_header = check_des_header_fields()
    # check that the classification seems to make sense
    valid_class = compare_sicd_class()
    return valid_sicd & valid_header & valid_class, the_sicd


def check_sicd_file(nitf_details):
    """
    Check the validity of the given NITF file as a SICD file.

    Parameters
    ----------
    nitf_details : str|NITFDetails
        The path to the NITF file, or a `NITFDetails` object.

    Returns
    -------
    bool
    """

    def check_data_extension_headers():
        # type: () -> (str, Union[DataExtensionHeader, DataExtensionHeader0])
        sicd_des = []

        for i in range(nitf_details.des_subheader_offsets.size):
            subhead_bytes = nitf_details.get_des_subheader_bytes(i)
            des_bytes = None
            if subhead_bytes.startswith(b'DEXML_DATA_CONTENT'):
                des_bytes = nitf_details.get_des_bytes(i)

            elif subhead_bytes.startswith(b'DESIDD_XML'):
                raise ValueError(
                    'This file contains an old format SIDD DES, and should be a SIDD file')
            elif subhead_bytes.startswith(b'DESICD_XML'):
                des_bytes = nitf_details.get_des_bytes(i)

            if des_bytes is None:
                continue

            # compare the SICD structure and the des header structure
            if nitf_details.nitf_version == '02.00':
                des_header = DataExtensionHeader0.from_bytes(subhead_bytes, start=0)
            elif nitf_details.nitf_version == '02.10':
                des_header = DataExtensionHeader.from_bytes(subhead_bytes, start=0)
            else:
                raise ValueError('Got unhandled NITF version {}'.format(nitf_details.nitf_version))

            try:
                des_string = des_bytes.decode('utf-8').strip()
                root_node, xml_ns = parse_xml_from_string(des_string)
                # namespace makes this ugly
                if 'SIDD' in root_node.tag:
                    raise ValueError(
                        'This file contains a SIDD DES, and should be a SIDD file')
                elif 'SICD' in root_node.tag:
                    sicd_des.append((i, des_string, des_header))
            except Exception as e:
                logger.error('Failed parsing the xml DES entry {} as xml'.format(i))
                raise e

        if len(sicd_des) == 0:
            raise ValueError('No SICD DES values found, so this is not a viable SICD file')
        elif len(sicd_des) > 1:
            raise ValueError(
                'Multiple SICD DES values found at indices {},\n'
                'so this is not a viable SICD file'.format([entry[0] for entry in sicd_des]))

        return sicd_des[0][1], sicd_des[0][2]

    def check_image_data():
        # type: () -> bool

        # get pixel type
        pixel_type = the_sicd.ImageData.PixelType
        if pixel_type == 'RE32F_IM32F':
            exp_nbpp = 32
            exp_pvtype = 'R'
        elif pixel_type == 'RE16I_IM16I':
            exp_nbpp = 16
            exp_pvtype = 'SI'
        elif pixel_type == 'AMP8I_PHS8I':
            exp_nbpp = 8
            exp_pvtype = 'INT'
        else:
            raise ValueError('Got unexpected pixel type {}'.format(pixel_type))

        valid_images = True
        # verify that all images have the correct pixel type
        for i, img_header in enumerate(nitf_details.img_headers):
            if img_header.ICAT.strip() != 'SAR':
                valid_images = False
                logger.error(
                    'SICD: image segment at index {} of {} has ICAT = `{}`,\n\texpected to be `SAR`'.format(
                        i, len(nitf_details.img_headers), img_header.ICAT.strip()))

            if img_header.PVTYPE.strip() != exp_pvtype:
                valid_images = False
                logger.error(
                    'SICD: image segment at index {} of {} has PVTYPE = `{}`,\n\t'
                    'expected to be `{}` based on pixel type {}'.format(
                        i, len(nitf_details.img_headers), img_header.PVTYPE.strip(), exp_pvtype, pixel_type))
            if img_header.NBPP != exp_nbpp:
                valid_images = False
                logger.error(
                    'SICD: image segment at index {} of {} has NBPP = `{}`,\n\t'
                    'expected to be `{}` based on pixel type {}'.format(
                        i, len(nitf_details.img_headers), img_header.NBPP, exp_nbpp, pixel_type))

            if len(img_header.Bands) != 2:
                valid_images = False
                logger.error('SICD: image segment at index {} of {} does not have two (I/Q or M/P) bands'.format(
                    i, len(nitf_details.img_headers)))
                continue

            if pixel_type == 'AMP8I_PHS8I':
                if img_header.Bands[0].ISUBCAT.strip() != 'M' and img_header.Bands[1].ISUBCAT.strip() != 'P':
                    valid_images = False
                    logger.error(
                        'SICD: pixel_type is {}, image segment at index {} of {}\n\t'
                        'has bands with ISUBCAT {}, expected ("M", "P")'.format(
                            pixel_type, i, len(nitf_details.img_headers),
                            (img_header.Bands[0].ISUBCAT.strip(), img_header.Bands[1].ISUBCAT.strip())))
            else:
                if img_header.Bands[0].ISUBCAT.strip() != 'I' and img_header.Bands[1].ISUBCAT.strip() != 'Q':
                    valid_images = False
                    logger.error(
                        'SICD: pixel_type is {}, image segment at index {} of {}\n\t'
                        'has bands with ISUBCAT {}, expected ("I", "Q")'.format(
                            pixel_type, i, len(nitf_details.img_headers),
                            (img_header.Bands[0].ISUBCAT.strip(), img_header.Bands[1].ISUBCAT.strip())))

        return valid_images

    if isinstance(nitf_details, string_types):
        if not os.path.isfile(nitf_details):
            raise ValueError('Got string input, but it is not a valid path')
        nitf_details = NITFDetails(nitf_details)

    if not isinstance(nitf_details, NITFDetails):
        raise TypeError(
            'Input is expected to be a path to a NITF file, or a NITFDetails object instance')

    # find the sicd header
    sicd_xml_string, des_header = check_data_extension_headers()
    # check that the sicd and header are valid
    valid_sicd_des, the_sicd = check_sicd_data_extension(nitf_details, des_header, sicd_xml_string)
    # check that the image segments all make sense compared to the sicd structure
    valid_img = check_image_data()
    all_valid = valid_sicd_des & valid_img

    if valid_img:
        try:
            reader = SICDReader(nitf_details.file_name)
        except Exception as e:
            logger.exception(
                'SICD: All image segments appear viable for the SICD,\n\t'
                'but SICDReader construction failed')
    return all_valid


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

    if isinstance(file_name, string_types):
        if not os.path.isfile(file_name):
            raise ValueError('Got string input, but it is not a valid path')

        # check if this is just an xml file
        with open(file_name, 'rb') as fi:
            initial_bits = fi.read(30)
            if initial_bits.startswith(b'<?xml') or initial_bits.startswith(b'<SICD'):
                sicd_xml = fi.read().decode('utf-8')
                return _evaluate_xml_string_validity(sicd_xml)[0]

    return check_sicd_file(file_name)


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
        logger.info('\nSICD: {} has been validated with no errors'.format(config.file_name))
    else:
        logger.error('\nSICD: {} has apparent errors'.format(config.file_name))
    sys.exit(int(validity))
