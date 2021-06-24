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
from sarpy.io.general.base import SarpyIOError
from sarpy.io.general.nitf import NITFDetails
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, DataExtensionHeader0
from sarpy.io.general.utils import parse_xml_from_string
from sarpy.io.complex.sicd import SICDReader, SICDDetails, extract_clas
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.SICD import SICDType, _SICD_SPECIFICATION_IDENTIFIER
from sarpy.io.complex.sicd_schema import get_schema_path


try:
    from lxml import etree
except ImportError:
    etree = None

logger = logging.getLogger('validation')

version_mapping = {
    '0.3.1': {'date': '2009-03-17T00:00:00Z'},
    '0.4.0': {'date': '2010-02-12T00:00:00Z'},
    '0.4.1': {'date': '2010-07-15T00:00:00Z'},
    '0.5.0': {'date': '2011-01-12T00:00:00Z'},
    '1.0.0': {'date': '2011-08-31T00:00:00Z'},
    '1.0.1': {'date': '2013-02-25T00:00:00Z'},
    '1.1.0': {'date': '2014-07-08T00:00:00Z'},
    '1.2.0': {'date': '2016-06-30T00:00:00Z'},
    '1.2.1': {'date': '2018-12-13T00:00:00Z'}}
versions = list(sorted(version_mapping.keys()))


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


def _evaluate_xml_string_validity(xml_string):
    """
    Check the validity of the SICD, as defined by the given string.

    Parameters
    ----------
    xml_string : str

    Returns
    -------
    (bool, str, SICDType)
    """

    root_node, xml_ns = parse_xml_from_string(xml_string)
    if 'default' not in xml_ns:
        raise ValueError(
            'Could not properly interpret the namespace collection from xml\n{}'.format(xml_ns))

    # validate versus the given urn...
    valid_xml = evaluate_xml_versus_schema(xml_string, xml_ns['default'])
    if valid_xml is None:
        valid_xml = True

    # perform the various sicd structure checks
    the_sicd = SICDType.from_node(root_node, xml_ns=xml_ns)
    valid_sicd_contents = the_sicd.is_valid(recursive=True, stack=False)
    return valid_xml & valid_sicd_contents, xml_ns['default'], the_sicd


def check_sicd_file(nitf_details):
    """
    Check the validity of the given NITF file as a SICD file.

    Parameters
    ----------
    nitf_details : str|NITFDetails

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
                raise ValueError('This file contains an old format SIDD DES, and should be a SIDD file')
            elif subhead_bytes.startswith(b'DESICD_XML'):
                logger.warning('DES extry {} is an old format SICD DES, which is outmoded'.format(i))
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
                    raise ValueError('This file contains a SIDD DES, and should be a SIDD file')
                elif 'SICD' in root_node.tag:
                    sicd_des.append((i, des_string, des_header))
            except Exception as e:
                logging.error('Failed parsing the DES entry {} as xml'.format(i))
                raise e

        if len(sicd_des) == 0:
            raise ValueError('No SICD DES values found, so this is not a viable SICD file')
        elif len(sicd_des) > 1:
            raise ValueError(
                'Multiple SICD DES values found at indices {},\n'
                'so this is not a viable SICD file'.format([entry[0] for entry in sicd_des]))

        return sicd_des[0][1], sicd_des[0][2]

    def check_des_header_fields():
        # type: () -> bool

        if des_header.DESTAG.strip() != 'XML_DATA_CONTENT':
            return True

        header_good = True
        if des_header.UserHeader.DESSHLI.strip() != _SICD_SPECIFICATION_IDENTIFIER:
            logger.error(
                'DES.DESSHLI has value `{}`,\nbut should have value `{}`'.format(
                    des_header.UserHeader.DESSHLI.strip(), _SICD_SPECIFICATION_IDENTIFIER))
            return False

        if urn_string[:9] != 'urn:SICD:':
            logger.error(
                'detected urn string {}, which should have the form "urn:SICD:<version>"'.format(urn_string))
            return False

        urn_version = urn_string[9:]
        split_version = urn_version.split('.')
        for entry in split_version:
            try:
                val = int(entry)
                if val < 0:
                    logger.error(
                        'Got version {} for urn {}, which is expected to be of the form '
                        '<major>.<minor>.<release>'.format(urn_version, urn_string))
                    return False
            except Exception:
                logger.error(
                    'Got version {} for urn {}, which is expected to be of the form '
                    '<major>.<minor>.<release>'.format(urn_version, urn_string))
                return False

        if len(split_version) != 3:
            logger.error(
                'Got version {} for urn {}, which is expected to be of the form '
                '<major>.<minor>.<release>'.format(urn_version, urn_string))
            return False

        short_version = split_version[0] + '.' + split_version[1]
        if des_header.UserHeader.DESSHSV.strip() != short_version:
            logger.error(
                'DES.DESSHSV has value `{}`,\nbut should have value `{}` based on sicd urn {}'.format(
                    des_header.UserHeader.DESSHSV.strip(), short_version, urn_string))
            header_good = False

        if urn_version not in versions:
            logger.error(
                'detected urn string {} with version {},\n'
                'but version is expected to be one of {}'.format(urn_string, urn_version, versions))
            return False

        values = version_mapping[urn_version]
        if des_header.UserHeader.DESSHSD.strip() != values['date']:
            logger.warning(
                'DES.DESSHSD has value `{}`,\nbut should have value `{}` based on sicd urn {}'.format(
                    des_header.UserHeader.DESSHSV.strip(), values['date'], urn_string))
        if des_header.UserHeader.DESSHTN.strip() != urn_string:
            logger.error(
                'DES.DESSHTN has value `{}`,\nbut should have value `{}` based on sicd urn {}'.format(
                    des_header.UserHeader.DESSHTN.strip(), urn_string, urn_string))
            header_good = False
        return header_good

    def compare_sicd_class():
        # type: () -> bool

        if the_sicd.CollectionInfo is None or the_sicd.CollectionInfo.Classification is None:
            logger.error(
                'SICD.CollectionInfo.Classification is not populated, '
                'so can not be compared with DES.DESCLAS {}'.format(des_header.Security.CLAS.strip()))
            return False

        sicd_class = the_sicd.CollectionInfo.Classification
        extracted_class = extract_clas(the_sicd)
        if extracted_class != des_header.Security.CLAS.strip():
            logger.error(
                'DES.DESCLAS is {}, and inferred '
                'SICD.CollectionInfo.Classification is {}'.format(des_header.Security.CLAS.strip(), sicd_class))
            return False

        if des_header.Security.CLAS.strip() != nitf_details.nitf_header.Security.CLAS.strip():
            logger.warning(
                'DES.DESCLAS is {}, and NITF.CLAS is {}'.format(
                    des_header.Security.CLAS.strip(), nitf_details.nitf_header.Security.CLAS.strip()))
        return True

    def check_image_data():
        # type: () -> bool

        # get pixel type
        pixel_type = the_sicd.ImageData.PixelType
        if pixel_type == 'RE32F_IM32F':
            exp_nbpp = 8
            exp_pvtype = 'R'
        elif pixel_type == 'RE16I_IM16I':
            exp_nbpp = 4
            exp_pvtype = 'SI'
        elif pixel_type == 'AMP8I_PHS8I':
            exp_nbpp = 2
            exp_pvtype = 'INT'
        else:
            raise ValueError('Got unexpected pixel type {}'.format(pixel_type))

        valid_images = True
        # verify that all images have the correct pixel type
        for i, img_header in enumerate(nitf_details.img_headers):
            if img_header.ICAT.strip() != 'SAR':
                valid_images = False
                logger.error(
                    'image segment at index {} of {} has ICAT = `{}`,\nexpected to be `SAR`'.format(
                        i, len(nitf_details.img_headers), img_header.ICAT.strip()))

            if img_header.PVTYPE.strip() != exp_pvtype:
                valid_images = False
                logger.error(
                    'image segment at index {} of {} has PVTYPE = `{}`,\n'
                    'expected to be `{}` based on pixel type {}'.format(
                        i, len(nitf_details.img_headers), img_header.PVTYPE.strip(), exp_pvtype, pixel_type))
            if img_header.NBPP != exp_nbpp:
                valid_images = False
                logger.error(
                    'image segment at index {} of {} has NBPP = `{}`,\n'
                    'expected to be `{}` based on pixel type {}'.format(
                        i, len(nitf_details.img_headers), img_header.NBPP, exp_nbpp, pixel_type))

            if len(img_header.Bands) != 2:
                valid_images = False
                logger.error('image segment at index {} of {} does not have two bands'.format(
                    i, len(nitf_details.img_headers)))
                continue

            if pixel_type == 'AMP8I_PHS8I':
                if img_header.Bands[0].ISUBCAT.strip() != 'M' and img_header.Bands[1].ISUBCAT.strip() != 'P':
                    valid_images = False
                    logger.error(
                        'pixel_type is {}, image segment at index {} of {}\n'
                        'has bands with ISUBCAT {}, expected ("M", "P")'.format(
                            pixel_type, i, len(nitf_details.img_headers),
                            (img_header.Bands[0].ISUBCAT.strip(), img_header.Bands[1].ISUBCAT.strip())))
            else:
                if img_header.Bands[0].ISUBCAT.strip() != 'I' and img_header.Bands[1].ISUBCAT.strip() != 'Q':
                    valid_images = False
                    logger.error(
                        'pixel_type is {}, image segment at index {} of {}\n'
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

    sicd_xml_string, des_header = check_data_extension_headers()

    valid_sicd, urn_string, the_sicd = _evaluate_xml_string_validity(sicd_xml_string)
    valid_header = check_des_header_fields()
    valid_header &= compare_sicd_class()
    valid_img = check_image_data()

    all_valid = valid_sicd & valid_header & valid_img

    if valid_img:
        try:
            reader = SICDReader(nitf_details.file_name)
        except Exception as e:
            logger.error(
                'All image segments appear viable for the SICD,\n'
                'but SICDReader construction failed with error\n{}'.format(e))
            
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
        logger.info('\nSICD {} has been validated with no errors'.format(config.file_name))
    else:
        logger.error('\nSICD {} has apparent errors'.format(config.file_name))
    sys.exit(int(validity))
