"""
Module for reading and writing SIDD files - should support SIDD version 1.0 and above.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
import sys
from functools import reduce
import re
from typing import List, Union, BinaryIO
from datetime import datetime

import numpy

from sarpy.io.xml.base import parse_xml_from_string
from sarpy.io.general.utils import is_file_like
from sarpy.io.general.base import AggregateChipper, SarpyIOError
from sarpy.io.general.nitf import NITFDetails, NITFReader, NITFWriter, ImageDetails, DESDetails, \
    image_segmentation, get_npp_block, interpolate_corner_points_string
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, XMLDESSubheader
from sarpy.io.general.nitf_elements.security import NITFSecurityTags
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, ImageBands, ImageBand

from sarpy.io.product.base import SIDDTypeReader
from sarpy.io.product.sidd2_elements.SIDD import SIDDType
from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd import extract_clas as extract_clas_sicd


logger = logging.getLogger(__name__)

########
# module variables
_class_priority = {'U': 0, 'R': 1, 'C': 2, 'S': 3, 'T': 4}


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a SIDD file.
    Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    SIDDReader|None
        `SIDDReader` instance if SIDD file, `None` otherwise
    """

    try:
        nitf_details = SIDDDetails(file_name)
        if nitf_details.is_sidd:
            logger.info('File {} is determined to be a SIDD (NITF format) file.'.format(file_name))
            return SIDDReader(nitf_details)
        else:
            return None
    except SarpyIOError:
        # we don't want to catch parsing errors, for now
        return None


#########
# Helper object for initially parses NITF header - specifically looking for SICD elements


class SIDDDetails(NITFDetails):
    """
    SIDD are stored in NITF 2.1 files.
    """

    __slots__ = (
        '_is_sidd', '_sidd_meta', '_sicd_meta',
        'img_segment_rows', 'img_segment_columns')

    def __init__(self, file_object):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
            file name or file like object for a NITF 2.1 or 2.0 containing a SIDD
        """

        self._img_headers = None
        self._is_sidd = False
        self._sidd_meta = None
        self._sicd_meta = None
        super(SIDDDetails, self).__init__(file_object)
        if self._nitf_header.ImageSegments.subhead_sizes.size == 0:
            raise SarpyIOError('There are no image segments defined.')
        if self._nitf_header.GraphicsSegments.item_sizes.size > 0:
            raise SarpyIOError('A SIDD file does not allow for graphics segments.')
        if self._nitf_header.DataExtensions.subhead_sizes.size == 0:
            raise SarpyIOError(
                'A SIDD file requires at least one data extension, containing the '
                'SIDD xml structure.')
        # define the sidd and sicd metadata
        self._find_sidd()
        if not self.is_sidd:
            raise SarpyIOError('Could not find SIDD xml data extensions.')
        # populate the image details
        self.img_segment_rows = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
        self.img_segment_columns = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
        for i, im_header in enumerate(self.img_headers):
            self.img_segment_rows[i] = im_header.NROWS
            self.img_segment_columns[i] = im_header.NCOLS

    @property
    def is_sidd(self):
        """
        bool: whether file name corresponds to a SIDD file, or not.
        """

        return self._is_sidd

    @property
    def sidd_meta(self):
        """
        None|List[sarpy.io.product.sidd2_elements.SIDD.SIDDType]: the sidd meta-data structure(s).
        """

        return self._sidd_meta

    @property
    def sicd_meta(self):
        """
        None|List[sarpy.io.complex.sicd_elements.SICD.SICDType]: the sicd meta-data structure(s).
        """

        return self._sicd_meta

    def _find_sidd(self):
        self._is_sidd = False
        if self.des_subheader_offsets is None:
            return

        self._sidd_meta = []
        self._sicd_meta = []

        for i in range(self.des_subheader_offsets.size):
            subhead_bytes = self.get_des_subheader_bytes(i)
            if subhead_bytes.startswith(b'DEXML_DATA_CONTENT'):
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SIDD' in root_node.tag:
                        self._is_sidd = True
                        self._sidd_meta.append(SIDDType.from_node(root_node, xml_ns, ns_key='default'))
                    elif 'SICD' in root_node.tag:
                        self._sicd_meta.append(SICDType.from_node(root_node, xml_ns, ns_key='default'))
                except Exception as e:
                    logger.error('Failed checking des xml header at index {} with error {}'.format(i, e))
                    continue
            elif subhead_bytes.startswith(b'DESIDD_XML'):
                # This is an old format SIDD header
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SIDD' in root_node.tag:
                        self._is_sidd = True
                        self._sidd_meta.append(SIDDType.from_node(root_node, xml_ns, ns_key='default'))
                except Exception as e:
                    logger.error(
                        'We found an apparent old-style SIDD DES header at index {},\n\t'
                        'but failed parsing with error {}'.format(i, e))
                    continue
            elif subhead_bytes.startswith(b'DESICD_XML'):
                # This is an old format SICD header
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
                    if 'SICD' in root_node.tag:
                        self._sicd_meta.append(SICDType.from_node(root_node, xml_ns, ns_key='default'))
                except Exception as e:
                    logger.error(
                        'We found an apparent old-style SICD DES header at index {},\n\t'
                        'but failed parsing with error {}'.format(i, e))
                    continue

        if not self._is_sidd:
            return

        for sicd in self._sicd_meta:
            sicd.derive()


#######
#  The actual reading implementation

def _check_iid_format(iid1, i):
    if not (iid1[:4] == 'SIDD' and iid1[4:].isnumeric()):
        raise ValueError('Got poorly formatted image segment id {} at position {}'.format(iid1, i))


class SIDDReader(NITFReader, SIDDTypeReader):
    """
    A reader object for a SIDD file (NITF container with SICD contents)
    """

    def __init__(self, nitf_details):
        """

        Parameters
        ----------
        nitf_details : str|BinaryIO|SIDDDetails
            filename, file-like object, or SIDDDetails object
        """

        if isinstance(nitf_details, str) or is_file_like(nitf_details):
            nitf_details = SIDDDetails(nitf_details)
        if not isinstance(nitf_details, SIDDDetails):
            raise TypeError('The input argument for SIDDReader must be a filename or '
                            'SIDDDetails object.')

        if not nitf_details.is_sidd:
            raise ValueError(
                'The input file passed in appears to be a NITF 2.1 file that does not contain '
                'valid sidd metadata.')

        self._nitf_details = nitf_details
        SIDDTypeReader.__init__(self, self.nitf_details.sidd_meta, self.nitf_details.sicd_meta)
        NITFReader.__init__(self, nitf_details, reader_type="SIDD")

    @property
    def nitf_details(self):
        # type: () -> SIDDDetails
        """
        SIDDDetails: The SIDD NITF details object.
        """

        return self._nitf_details

    def _find_segments(self):
        # determine image segmentation from image headers
        segments = [[] for _ in self._sidd_meta]
        for i, img_header in enumerate(self.nitf_details.img_headers):
            # skip anything but SAR for now (i.e. legend)
            if img_header.ICAT != 'SAR':
                continue

            iid1 = img_header.IID1  # required to be of the form SIDD######
            _check_iid_format(iid1, i)
            element = int(iid1[4:7])
            if element > len(self._sidd_meta):
                raise ValueError('Got image segment id {}, but there are only {} '
                                 'sidd elements'.format(iid1, len(self._sidd_meta)))
            segments[element - 1].append(i)
        # Ensure that all segments are populated
        for i, entry in enumerate(segments):
            if len(entry) < 1:
                raise ValueError('Did not find any image segments for SIDD {}'.format(i))
        return segments

    def _check_img_details(self, segment):
        raw_dtype, output_dtype, raw_bands, output_bands, transform_data = self._extract_chipper_params(segment[0])
        for this_index in segment[1:]:
            this_raw_dtype, this_output_dtype, this_raw_bands, this_output_bands, \
                _ = self._extract_chipper_params(this_index)
            if this_raw_dtype.name != raw_dtype.name:
                raise ValueError(
                    'Segments at index {} and {} have incompatible data types '
                    '{} and {}'.format(segment[0], this_index, raw_dtype, this_raw_dtype))
            if this_output_dtype.name != output_dtype.name:
                raise ValueError(
                    'Segments at index {} and {} have incompatible output data types '
                    '{} and {}'.format(segment[0], this_index, output_dtype, this_output_dtype))
            if this_raw_bands != raw_bands:
                raise ValueError(
                    'Segments at index {} and {} have incompatible input bands '
                    '{} and {}'.format(segment[0], this_index, raw_bands, this_raw_bands))
            if this_output_bands != output_bands:
                raise ValueError(
                    'Segments at index {} and {} have incompatible output bands '
                    '{} and {}'.format(segment[0], this_index, output_bands, this_output_bands))
        return raw_dtype, output_dtype, raw_bands, output_bands, transform_data

    def _construct_chipper(self, segment, index):
        # get the image size
        sidd = self.sidd_meta[index]
        rows = sidd.Measurement.PixelFootprint.Row
        cols = sidd.Measurement.PixelFootprint.Col

        # extract the basic elements for the chippers
        raw_dtype, output_dtype, raw_bands, output_bands, transform_data = self._check_img_details(segment)
        if len(segment) == 1:
            return self._define_chipper(
                segment[0], raw_dtype=raw_dtype, raw_bands=raw_bands, transform_data=transform_data,
                output_dtype=output_dtype, output_bands=output_bands)
        else:
            # get the bounds definition
            bounds = self._get_chipper_partitioning(segment, rows, cols)
            # define the chippers
            chippers = [
                self._define_chipper(img_index, raw_dtype=raw_dtype, raw_bands=raw_bands, transform_data=transform_data,
                                     output_dtype=output_dtype, output_bands=output_bands) for img_index in segment]
            # define the aggregate chipper
            return AggregateChipper(bounds, output_dtype, chippers, output_bands=output_bands)


#########
# The writer implementation

def validate_sidd_for_writing(sidd_meta):
    """
    Helper method which ensures the provided SIDD structure is appropriate.

    Parameters
    ----------
    sidd_meta : SIDDType|List[SIDDType]|SIDDType1|List[SIDDType1]

    Returns
    -------
    Tuple[Union[SIDDType, SIDDType1]]
    """

    def inspect_sidd(the_sidd):
        # we must have the image size
        if the_sidd.Measurement is None:
            raise ValueError('The sidd_meta has un-populated Measurement, '
                             'and nothing useful can be inferred.')
        if the_sidd.Measurement.PixelFootprint is None:
            raise ValueError('The sidd_meta has un-populated Measurement.PixelFootprint, '
                             'and this is not valid for writing.')

        # we must have the pixel type
        if the_sidd.Display is None:
            raise ValueError('The sidd_meta has un-populated Display, '
                             'and nothing useful can be inferred.')
        if the_sidd.Display.PixelType is None:
            raise ValueError('The sidd_meta has un-populated Display.PixelType, '
                             'and nothing useful can be inferred.')
        # No support for LUT until necessary
        if the_sidd.Display.PixelType in ('MONO8LU', 'RGB8LU'):
            raise ValueError('PixelType requiring lookup table currently unsupported.')

        # we must have collection time
        if the_sidd.ExploitationFeatures is None:
            raise ValueError('The sidd_meta has un-populated ExploitationFeatures.')
        if the_sidd.ExploitationFeatures.Collections is None or len(the_sidd.ExploitationFeatures.Collections) == 0:
            raise ValueError('The sidd_meta has un-populated ExploitationFeatures.Collections.')
        if the_sidd.ExploitationFeatures.Collections[0].Information is None:
            raise ValueError('The sidd_meta has un-populated ExploitationFeatures.Collections.Information.')
        if the_sidd.ExploitationFeatures.Collections[0].Information.CollectionDateTime is None:
            raise ValueError('The sidd_meta has un-populated ExploitationFeatures.Collections.'
                             'Information.CollectionDateTime')

        # try validating versus the appropriate schema
        xml_str = the_sidd.to_xml_string(tag='SIDD')
        from sarpy.consistency.sidd_consistency import evaluate_xml_versus_schema
        if isinstance(the_sidd, SIDDType1):
            urn = 'urn:SIDD:1.0.0'
        elif isinstance(the_sidd, SIDDType):
            urn = 'urn:SIDD:2.0.0'
        else:
            raise ValueError('Unhandled type {}'.format(type(the_sidd)))
        result = evaluate_xml_versus_schema(xml_str, urn)
        if result is False:
            logger.warning(
                'The provided SIDD does not properly validate\n\t'
                'against the schema for {}'.format(urn))

    if isinstance(sidd_meta, (SIDDType, SIDDType1)):
        inspect_sidd(sidd_meta)
        return (sidd_meta, )
    elif isinstance(sidd_meta, (tuple, list)):
        out = []
        for entry in sidd_meta:
            if not isinstance(entry, (SIDDType, SIDDType1)):
                raise TypeError('All entries are required to be an instance of SIDDType, '
                                'got type {}'.format(type(entry)))
            inspect_sidd(entry)
            out.append(entry)
        return tuple(out)
    else:
        raise TypeError('sidd_meta is required to be an instance of SIDDType or a list/tuple '
                        'of such instances, got {}'.format(type(sidd_meta)))


def validate_sicd_for_writing(sicd_meta):
    """
    Helper method which ensures the provided SICD structure is appropriate.

    Parameters
    ----------
    sicd_meta : SICDType|List[SICDType]

    Returns
    -------
    None|Tuple[SICDType]
    """

    if sicd_meta is None:
        return None
    if isinstance(sicd_meta, SICDType):
        return (sicd_meta, )
    elif isinstance(sicd_meta, (tuple, list)):
        out = []
        for entry in sicd_meta:
            if not isinstance(entry, SICDType):
                raise TypeError('All entries are required to be an instance of SICDType, '
                                'got type {}'.format(type(entry)))
            out.append(entry)
        return tuple(out)
    else:
        raise TypeError('sicd_meta is required to be an instance of SICDType or a list/tuple '
                        'of such instances, got {}'.format(type(sicd_meta)))


def extract_clas(the_sidd):
    """
    Extract the classification string from a SIDD as appropriate for NITF Security
    tags CLAS attribute.

    Parameters
    ----------
    the_sidd : SIDDType|SIDDType1

    Returns
    -------
    str
    """

    class_str = the_sidd.ProductCreation.Classification.classification

    if class_str is None or class_str == '':
        return 'U'
    else:
        return class_str[:1]


def extract_clsy(the_sidd):
    """
    Extract the ownerProducer string from a SIDD as appropriate for NITF Security
    tags CLSY attribute.

    Parameters
    ----------
    the_sidd : SIDDType|SIDDType1

    Returns
    -------
    str
    """

    owner = the_sidd.ProductCreation.Classification.ownerProducer.upper()
    if owner is None:
        return ''
    elif owner in ('USA', 'CAN', 'AUS', 'NZL'):
        return owner[:2]
    elif owner == 'GBR':
        return 'UK'
    elif owner == 'NATO':
        return 'XN'
    else:
        logger.warning(
            'Got owner {}, and the CLSY will be truncated\n\t'
            'to two characters.'.format(owner))
        return owner[:2]


class SIDDWriter(NITFWriter):
    """
    Writer class for a SIDD file - a NITF file containing data derived from complex radar data and
    SIDD and SICD data extension(s).
    """

    __slots__ = ('_sidd_meta', '_sicd_meta', )

    def __init__(self, file_name, sidd_meta, sicd_meta, check_existence=True):
        """

        Parameters
        ----------
        file_name : str
        sidd_meta : SIDDType|List[SIDDType]|SIDDType1|List[SIDDType1]
        sicd_meta : SICDType|List[SICDType]
        check_existence : bool
            Should we check if the given file already exists, and raises an exception if so?
        """

        self._shapes = None
        self._sidd_meta = validate_sidd_for_writing(sidd_meta)
        self._set_shapes()
        self._sicd_meta = validate_sicd_for_writing(sicd_meta)
        self._security_tags = None
        self._nitf_header = None
        self._img_groups = None
        self._img_details = None
        self._des_details = None
        super(SIDDWriter, self).__init__(file_name, check_existence=check_existence)

    @property
    def sidd_meta(self):
        """
        Tuple[SIDDType]: The sidd metadata.
        """

        return self._sidd_meta

    @property
    def sicd_meta(self):
        """
        None|Tuple[SICDType]: The sicd metadata.
        """

        return self._sicd_meta

    def _set_shapes(self):
        if isinstance(self._sidd_meta, tuple):
            self._shapes = tuple(
                (entry.Measurement.PixelFootprint.Row, entry.Measurement.PixelFootprint.Col)
                for entry in self._sidd_meta)
        else:
            self._shapes = (
                (self._sidd_meta.Measurement.PixelFootprint.Row,
                 self._sidd_meta.Measurement.PixelFootprint.Col), )

    def _get_security_tags(self, index):
        """
        Gets the security tags for SIDD at `index`.

        Parameters
        ----------
        index : int

        Returns
        -------
        NITFSecurityTags
        """

        args = {}
        sidd = self.sidd_meta[index]
        if sidd.ProductCreation is not None and sidd.ProductCreation.Classification is not None:
            args['CLAS'] = extract_clas(sidd)
            args['CLSY'] = extract_clsy(sidd)

        security = sidd.NITF.get('Security', {})
        # noinspection PyProtectedMember
        for key in NITFSecurityTags._ordering:
            if key in security:
                args[key] = security[key]
        return NITFSecurityTags(**args)

    def _create_security_tags(self):
        def class_priority(cls1, cls2):
            p1 = _class_priority[cls1]
            p2 = _class_priority[cls2]
            if p1 >= p2:
                return cls1
            return cls2

        # determine the highest priority clas string from all the sidds & sicds
        clas_collection = [extract_clas(sidd) for sidd in self.sidd_meta]
        if self.sicd_meta is not None:
            clas_collection.extend([extract_clas_sicd(sicd) for sicd in self.sicd_meta])
        clas = reduce(class_priority, clas_collection)
        # try to determine clsy from all sidds
        clsy_collection = list(set(extract_clsy(sidd) for sidd in self.sidd_meta))
        clsy = clsy_collection[0] if len(clsy_collection) else None
        # populate the attribute
        self._security_tags = NITFSecurityTags(CLAS=clas, CLSY=clsy)

    def _get_iid2(self, index):
        """
        Get the IID2 for the sidd at `index`.

        Parameters
        ----------
        index : int

        Returns
        -------
        str
        """

        sidd = self.sidd_meta[index]

        iid2 = sidd.NITF.get('IID2', None)
        if iid2 is None:
            iid2 = sidd.NITF.get('FTITLE', None)
        if iid2 is None:
            iid2 = sidd.NITF.get('SUGGESTED_NAME', None)
        if iid2 is None:
            iid2 = 'SIDD: Unknown'
        return iid2

    def _get_ftitle(self, index=0):
        sidd = self.sidd_meta[index]
        ftitle = sidd.NITF.get('FTITLE', None)
        if ftitle is None:
            ftitle = sidd.NITF.get('SUGGESTED_NAME', None)
        if ftitle is None:
            ftitle = 'SIDD: Unknown'
        return ftitle

    def _get_fdt(self):
        sidd = self.sidd_meta[0]
        if sidd.ExploitationFeatures.Collections[0].Information.CollectionDateTime is not None:
            the_time = sidd.ExploitationFeatures.Collections[0].Information.CollectionDateTime.astype('datetime64[s]')
            return re.sub(r'[^0-9]', '', str(the_time))
        else:
            return super(SIDDWriter, self)._get_fdt()

    def _get_ostaid(self, index=0):
        sidd = self.sidd_meta[index]
        ostaid = sidd.NITF.get('OSTAID', 'Unknown')
        return ostaid

    def _get_isorce(self, index=0):
        sidd = self.sidd_meta[index]
        isorce = sidd.NITF.get('ISORCE', sidd.ExploitationFeatures.Collections[0].Information.SensorName)
        if isorce is None:
            isorce = 'Unknown'
        return isorce

    def _image_parameters(self, index):
        """
        Get the image parameters for the sidd at `index`.

        Parameters
        ----------
        index

        Returns
        -------
        (int, int, str, numpy.dtype, Union[bool, str, callable], str, int, tuple, tuple)
            pixel_size - the size of each pixel in bytes.
            abpp - the actual bits per pixel.
            irep - the image representation
            raw_dtype - the data type.
            transform_data -
            pv_type - the pixel type string.
            band_count - the number of bands
            irepband - the image representation.
            im_segments - Segmentation of the form `((row start, row end, column start, column end))`
        """

        sidd = self.sidd_meta[index]
        assert isinstance(sidd, (SIDDType, SIDDType1))
        if sidd.Display.PixelType == 'MONO8I':
            pixel_size = 1
            abpp = 8
            irep = 'MONO'
            raw_dtype = numpy.dtype('>u1')
            transform_data = None
            pv_type = 'INT'
            band_count = 1
            irepband = ('M', )
        elif sidd.Display.PixelType == 'MONO16I':
            pixel_size = 2
            abpp = 16
            irep = 'MONO'
            raw_dtype = numpy.dtype('>u2')
            transform_data = None
            pv_type = 'INT'
            band_count = 1
            irepband = ('M', )
        elif sidd.Display.PixelType == 'RGB24I':
            pixel_size = 3
            abpp = 8
            irep = 'RGB'
            raw_dtype = numpy.dtype('>u1')
            transform_data = None
            pv_type = 'INT'
            band_count = 3
            irepband = ('R', 'G', 'B')
        else:
            raise ValueError('Unsupported PixelType {}'.format(sidd.Display.PixelType))

        image_segment_limits = image_segmentation(
            sidd.Measurement.PixelFootprint.Row, sidd.Measurement.PixelFootprint.Col, pixel_size)
        return pixel_size, abpp, irep, raw_dtype, transform_data, pv_type, band_count, irepband, image_segment_limits

    @staticmethod
    def _get_icp(sidd):
        """
        Get the Image corner point array, if possible.

        Parameters
        ----------
        sidd : SIDDType|SIDDType1

        Returns
        -------
        numpy.ndarray
        """

        if isinstance(sidd, SIDDType) and sidd.GeoData is not None and sidd.GeoData.ImageCorners is not None:
            return sidd.GeoData.ImageCorners.get_array(dtype=numpy.dtype('float64'))
        elif isinstance(sidd, SIDDType1) and sidd.GeographicAndTarget is not None and \
                sidd.GeographicAndTarget.GeographicCoverage is not None and \
                sidd.GeographicAndTarget.GeographicCoverage.Footprint is not None:
            return sidd.GeographicAndTarget.GeographicCoverage.Footprint.get_array(dtype=numpy.dtype('float64'))
        return None

    def _create_image_segment(self, index, img_groups, img_details):
        cur_count = len(img_details)
        pixel_size, abpp, irep, raw_dtype, transform_data, pv_type, band_count, irepband, \
            image_segment_limits = self._image_parameters(index)
        new_count = len(image_segment_limits)
        img_groups.append(tuple(range(cur_count, cur_count+new_count)))
        security = self._get_security_tags(index)

        iid2 = self._get_iid2(index)
        sidd = self.sidd_meta[index]

        idatim = self._get_fdt()

        isorce = self._get_isorce(index)

        rows = sidd.Measurement.PixelFootprint.Row
        cols = sidd.Measurement.PixelFootprint.Col

        icp = self._get_icp(sidd)
        bands = [ImageBand(ISUBCAT='', IREPBAND=entry) for entry in irepband]

        for i, entry in enumerate(image_segment_limits):
            this_rows = entry[1]-entry[0]
            this_cols = entry[3]-entry[2]
            subhead = ImageSegmentHeader(
                IID1='SIDD{0:03d}{1:03d}'.format(len(img_groups), i+1),
                IDATIM=idatim,
                IID2=iid2,
                ISORCE=isorce,
                IREP=irep,
                ICAT='SAR',
                NROWS=this_rows,
                NCOLS=this_cols,
                PVTYPE=pv_type,
                ABPP=abpp,
                IGEOLO=interpolate_corner_points_string(numpy.array(entry, dtype=numpy.int64), rows, cols, icp),
                IMODE='P',
                NPPBH=get_npp_block(this_cols),
                NPPBV=get_npp_block(this_rows),
                NBPP=abpp,
                IDLVL=i+1+len(img_details),
                IALVL=i+len(img_details),
                ILOC='{0:05d}{1:05d}'.format(entry[0], entry[2]),
                Bands=ImageBands(values=bands),
                Security=security)
            img_details.append(ImageDetails(band_count, raw_dtype, transform_data, entry, subhead))

    def _create_image_segment_details(self):
        super(SIDDWriter, self)._create_image_segment_details()
        img_groups = []
        img_details = []
        for index in range(len(self.sidd_meta)):
            self._create_image_segment(index, img_groups, img_details)
        self._img_groups = tuple(img_groups)
        self._img_details = tuple(img_details)

    def _create_sidd_des_segment(self, index):
        """
        Create the details for the sidd data extension at `index`.

        Parameters
        ----------
        index : int

        Returns
        -------
        DESDetails
        """

        imgs = self._img_groups[index]
        security = self.image_details[imgs[0]].subheader.Security
        sidd = self.sidd_meta[index]
        uh_args = sidd.get_des_details()

        try:
            desshdt = str(sidd.ProductCreation.ProcessorInformation.ProcessingDateTime.astype('datetime64[s]'))
        except AttributeError:
            desshdt = str(numpy.datetime64('now'))
        if desshdt[-1] != 'Z':
            desshdt += 'Z'
        uh_args['DESSHDT'] = desshdt

        desshlpg = ''
        icp = self._get_icp(sidd)
        if icp is not None:
            temp = []
            for entry in icp:
                temp.append('{0:0=+12.8f}{1:0=+13.8f}'.format(entry[0], entry[1]))
            temp.append(temp[0])
            desshlpg = ''.join(temp)
        uh_args['DESSHLPG'] = desshlpg
        subhead = DataExtensionHeader(
            Security=security,
            UserHeader=XMLDESSubheader(**uh_args))
        return DESDetails(subhead, sidd.to_xml_bytes(tag='SIDD'))

    def _create_sicd_des_segment(self, index):
        """
        Create the details for the sicd data extension at `index`.

        Parameters
        ----------
        index : int

        Returns
        -------
        DESDetails
        """

        security_tags = self.security_tags
        sicd = self.sicd_meta[index]
        uh_args = sicd.get_des_details(check_version1_compliance=True)
        if sicd.ImageCreation.DateTime is None:
            desshdt = datetime.utcnow().isoformat('T', timespec='seconds')
        else:
            desshdt = str(sicd.ImageCreation.DateTime.astype('datetime64[s]'))
        if desshdt[-1] != 'Z':
            desshdt += 'Z'
        uh_args['DESSHDT'] = desshdt

        desshlpg = ''
        if sicd.GeoData is not None and sicd.GeoData.ImageCorners is not None:
            # noinspection PyTypeChecker
            icp = sicd.GeoData.ImageCorners.get_array(dtype=numpy.float64)
            temp = []
            for entry in icp:
                temp.append('{0:0=+12.8f}{1:0=+13.8f}'.format(entry[0], entry[1]))
            temp.append(temp[0])
            desshlpg = ''.join(temp)
        uh_args['DESSHLPG'] = desshlpg

        subhead = DataExtensionHeader(
            Security=security_tags,
            UserHeader=XMLDESSubheader(**uh_args))

        return DESDetails(subhead, sicd.to_xml_bytes(tag='SICD', urn=uh_args['DESSHTN']))

    def _create_data_extension_details(self):
        super(SIDDWriter, self)._create_data_extension_details()
        des_details = []
        for index in range(len(self.sidd_meta)):
            des_details.append(self._create_sidd_des_segment(index))
        if self.sicd_meta is not None:
            for index in range(len(self.sicd_meta)):
                des_details.append(self._create_sicd_des_segment(index))
        self._des_details = tuple(des_details)
