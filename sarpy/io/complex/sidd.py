# -*- coding: utf-8 -*-
"""
Module for reading and writing SIDD files - should support SIDD version 1.0 and above.
"""

import logging
from functools import reduce
from datetime import datetime
import re

import numpy

from .utils import parse_xml_from_string
from .base import int_func, string_types
from .nitf import NITFReader, NITFWriter, ImageDetails, DESDetails, image_segmentation, get_npp_block, \
    interpolate_corner_points_string
from ..nitf.nitf_head import NITFDetails
from ..nitf.des import DataExtensionHeader, SICDDESSubheader
from ..nitf.security import NITFSecurityTags
from ..nitf.image import ImageSegmentHeader, ImageBands, ImageBand
from .sidd_elements.SIDD import SIDDType
from .sidd_elements.sidd1_elements.SIDD import SIDDType as SIDDType1
from .sicd_elements.SICD import SICDType
from .sicd import MultiSegmentChipper, extract_clas as extract_clas_sicd


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


########
# module variables
_class_priority = {'U': 0, 'R' : 1, 'C': 2, 'S': 3, 'T': 4}


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
            print('File {} is determined to be a SIDD (NITF format) file.'.format(file_name))
            return SIDDReader(nitf_details)
        else:
            return None
    except IOError:
        # we don't want to catch parsing errors, for now
        return None


#########
# Helper object for initially parses NITF header - specifically looking for SICD elements


class SIDDDetails(NITFDetails):
    """
    SIDD are stored in NITF 2.1 files.
    """

    __slots__ = (
        '_img_headers', '_is_sidd', '_sidd_meta', '_sicd_meta',
        'img_segment_rows', 'img_segment_columns')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF 2.1 file containing a SIDD
        """

        self._img_headers = None
        self._is_sidd = False
        self._sidd_meta = None
        self._sicd_meta = None
        super(SIDDDetails, self).__init__(file_name)
        if self._nitf_header.ImageSegments.subhead_sizes.size == 0:
            raise IOError('There are no image segments defined.')
        if self._nitf_header.GraphicsSegments.item_sizes.size > 0:
            raise IOError('A SIDD file does not allow for graphics segments.')
        if self._nitf_header.DataExtensions.subhead_sizes.size == 0:
            raise IOError('A SIDD file requires at least one data extension, containing the '
                          'SIDD xml structure.')
        # define the sidd and sicd metadata
        self._find_sidd()
        if not self.is_sidd:
            raise IOError('Could not find SIDD xml data extensions.')
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
        None|List[sarpy.io.complex.sidd_elements.SIDD.SIDDType]: the sidd meta-data structure(s).
        """

        return self._sidd_meta

    @property
    def sicd_meta(self):
        """
        None|List[sarpy.io.complex.sicd_elements.SICD.SICDType]: the sicd meta-data structure(s).
        """

        return self._sicd_meta

    @property
    def img_headers(self):
        """
        The image segment headers.

        Returns
        -------
            None|List[sarpy.io.nitf.image.ImageSegmentHeader]
        """

        if self._img_headers is not None:
            return self._img_headers

        self._parse_img_headers()
        return self._img_headers

    def _parse_img_headers(self):
        if self.img_segment_offsets is None or self._img_headers is not None:
            return

        self._img_headers = [self.parse_image_subheader(i) for i in range(self.img_subheader_offsets.size)]

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
                    logging.error('Failed checking des xml header at index {} with error {}'.format(i, e))
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
                    logging.error('We found an apparent old-style SIDD DES header at index {}, '
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
                    logging.error('We found an apparent old-style SICD DES header at index {}, '
                                  'but failed parsing with error {}'.format(i, e))
                    continue

        if not self._is_sidd:
            return

        for sicd in self._sicd_meta:
            sicd.derive()


#######
#  The actual reading implementation

def _validate_lookup(lookup_table):
    # type: (numpy.ndarray) -> None
    if not isinstance(lookup_table, numpy.ndarray):
        raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
    if lookup_table.dtype.name != 'uint8':
        raise ValueError('requires a numpy.ndarray of uint8 dtype, got {}'.format(lookup_table.dtype))


def rgb_lut_conversion(lookup_table):
    """
    This constructs the function to convert from RGB/LUT format data to RGB data.

    Parameters
    ----------
    lookup_table : numpy.ndarray

    Returns
    -------
    callable
    """

    _validate_lookup(lookup_table)

    def converter(data):
        if not isinstance(data, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(data)))

        if data.dtype.name not in ['uint8', 'uint16']:
            raise ValueError('requires a numpy.ndarray of uint8 or uint16 dtype, '
                             'got {}'.format(data.dtype.name))

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the last dimension), got shape {}'.format(data.shape))
        return lookup_table[data[:, :, 0]]
    return converter


class SIDDReader(NITFReader):
    """
    A reader object for a SIDD file (NITF container with SICD contents)
    """

    __slots__ = ('_sidd_meta', )

    def __init__(self, nitf_details):
        """

        Parameters
        ----------
        nitf_details : str|SIDDDetails
            filename or SIDDDetails object
        """

        if isinstance(nitf_details, string_types):
            nitf_details = SIDDDetails(nitf_details)
        if not isinstance(nitf_details, SIDDDetails):
            raise TypeError('The input argument for SIDDReader must be a filename or '
                            'SIDDDetails object.')

        if not nitf_details.is_sidd:
            raise ValueError(
                'The input file passed in appears to be a NITF 2.1 file that does not contain valid sidd metadata.')
        super(SIDDReader, self).__init__(nitf_details, is_sicd_type=False)
        self._sidd_meta = self.nitf_details.sidd_meta

    @property
    def nitf_details(self):
        # type: () -> SIDDDetails
        """
        SIDDDetails: The SIDD NITF details object.
        """

        return self._nitf_details

    @property
    def sidd_meta(self):
        """
        None|List[sarpy.io.complex.sidd_elements.SIDD.SIDDType]: the sidd meta-data structure(s).
        """

        return self.nitf_details.sidd_meta

    @property
    def nitf_details(self):
        # type: () -> SIDDDetails
        # noinspection PyTypeChecker
        return self._nitf_details

    def _find_segments(self):
        # determine image segmentation from image headers
        segments = [[] for sidd in self._sidd_meta]
        for i, img_header in enumerate(self.nitf_details.img_headers):
            # skip anything but SAR for now (i.e. legend)
            if img_header.ICAT != 'SAR':
                continue

            iid1 = img_header.IID1  # required to be of the form SIDD######
            if not (iid1[:4] == 'SIDD' and iid1[4:].isnumeric()):
                raise ValueError('Got poorly formatted image segment id {} at position {}'.format(iid1, i))
            element = int_func(iid1[4:7])
            if element > len(self._sidd_meta):
                raise ValueError('Got image segment id {}, but there are only {} '
                                 'sidd elements'.format(iid1, len(self._sidd_meta)))
            segments[element - 1].append(i)
        # Ensure that all segments are populated
        for i, entry in enumerate(segments):
            if len(entry) < 1:
                raise ValueError('Did not find any image segments for SIDD {}'.format(i))
        return segments

    def _get_img_details(self, index):
        img_header = self.nitf_details.img_headers[index]
        if img_header.IC != 'NC':
            raise ValueError('Image header at index {} has IC {}. No compression '
                             'is supported at this time.'.format(index, img_header.IC))
        abpp = img_header.ABPP
        if abpp not in [8, 16]:
            raise ValueError('Image header at index {} has ABPP {}. Unsupported.'.format(index, abpp))
        rows = img_header.NROWS
        cols = img_header.NCOLS
        # check for a strangely segmented image
        if img_header.NPPBV not in [rows, 0]:
            raise ValueError('Image header at index {} has NROWS {} and NPPBV {}, '
                             'which is unsupported'.format(index, rows, img_header.NPPBV))
        if img_header.NPPBH not in [cols, 0]:
            raise ValueError('Image header at index {} has NCOLS {} and NPPBH {}, '
                             'which is unsupported'.format(index, cols, img_header.NPPBH))
        # check image type
        irep = img_header.IREP

        if irep not in ['MONO', 'RGB/LUT', 'RGB']:
            raise ValueError('Image header at index {} has IREP {}, which is '
                             'unsupported'.format(index, irep))
        bands = len(img_header.Bands)
        if (irep == 'RGB' and bands != 3) or (irep != 'RGB' and bands != 1):
                raise ValueError('Image header at index {} has IREP {} and {} Bands. '
                                     'This is unsupported.'.format(index, irep, bands))
        # Fetch lookup table
        lut = img_header.Bands[0].LUTD if irep == 'RGB/LUT' else None

        # determine dtype
        if lut is None:
            if abpp == 8:
                dtype = numpy.dtype('>uint8')
            else:
                dtype = numpy.dtype('>uint16')
        else:
            dtype = numpy.dtype('>uint8')

        return rows, cols, dtype, bands, lut

    def _construct_chipper(self, segment, index):
        bounds = numpy.zeros((len(segment), 4), dtype=numpy.uint64)
        offsets = numpy.zeros((len(segment), ), dtype=numpy.int64)
        this_dtype, this_bands, this_lut = None, None, None

        p_row_start, p_row_end, p_col_start, p_col_end = None, None, None, None
        for i, index in enumerate(segment):
            # populate bounds entry
            offsets[i] = self.nitf_details.img_segment_offsets[index]
            rows, cols, dtype, bands, lut = self._get_img_details(index)
            if i == 0:
                cur_row_start, cur_row_end = 0, rows
                cur_col_start, cur_col_end = 0, cols
            elif cols == (p_col_end - p_col_start):
                cur_row_start, cur_row_end = p_row_end, p_row_end + cols
                cur_col_start, cur_col_end = p_col_start, p_col_end
            else:
                cur_row_start, cur_row_end = 0, rows
                cur_col_start, cur_col_end = p_col_end, p_col_end + rows

            bounds[i] = (cur_row_start, cur_row_end, cur_col_start, cur_col_end)
            p_row_start, p_row_end, p_col_start, p_col_end = cur_row_start, cur_row_end, cur_col_start, cur_col_end

            # define the other meta data
            if i == 0:
                this_dtype = dtype
                this_bands = bands
                this_lut = lut
            else:
                if dtype != this_dtype:
                    raise ValueError('Image segments {} form one sidd, but have differing type '
                                     'parameters'.format(segment))
                if bands != this_bands:
                    raise ValueError('Image segments {} form one sidd, but have differing '
                                     'band counts'.format(segment))
                if (this_lut is None and lut is not None) or (this_lut is not None and lut is None):
                    raise ValueError('Image segments {} form one sidd, but have differing '
                                     'look up table information'.format(segment))
                if (this_lut is not None and lut is not None) and \
                        (this_lut.shape != lut.shape or numpy.any(this_lut != lut)):
                    raise ValueError('Image segments {} form one sidd, but have differing '
                                     'look up table information'.format(segment))

        complex_type = False if this_lut is None else rgb_lut_conversion(this_lut)
        return MultiSegmentChipper(
            self.file_name, bounds, offsets, this_dtype,
            symmetry=(False, False, False), complex_type=complex_type, bands_ip=this_bands)


def validate_sidd_for_writing(sidd_meta):
    """
    Helper method which ensures the provided SIDD structure is appropriate.

    Parameters
    ----------
    sidd_meta : SIDDType|List[SIDDType]|SIDDType1|List[SIDDType1]

    Returns
    -------
    Tuple[SIDDType, SIDDType1]
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
        if the_sidd.ExploitationFeatures.Collections is None or the_sidd.ExploitationFeatures.Collections.size == 0:
            raise ValueError('The sidd_meta has un-populated ExploitationFeatures.Collections.')
        if the_sidd.ExploitationFeatures.Collections[0].Information is None:
            raise ValueError('The sidd_meta has un-populated ExploitationFeatures.Collections.Information.')
        if the_sidd.ExploitationFeatures.Collections[0].Information.CollectionDateTime is None:
            raise ValueError('The sidd_meta has un-populated ExploitationFeatures.Collections.'
                             'Information.CollectionDateTime')

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
        logging.warning('Got owner {}, and the CLSY will be truncated '
                        'to two characters.'.format(owner))
        return owner[:2]


class SIDDWriter(NITFWriter):
    """
    Writer class for a SIDD file - a NITF file containing data derived from complex radar data and
    SIDD and SICD data extension(s).
    """

    __slots__ = ('_sidd_meta', '_sicd_meta', )

    def __init__(self, file_name, sidd_meta, sicd_meta):
        """

        Parameters
        ----------
        file_name : str
        sidd_meta : SIDDType|List[SIDDType]|SIDDType1|List[SIDDType1]
        sicd_meta : SICDType|List[SICDType]
        """

        self._sidd_meta = validate_sidd_for_writing(sidd_meta)
        # self._shapes = ((self.sicd_meta.ImageData.NumRows, self.sicd_meta.ImageData.NumCols), )
        self._sicd_meta = validate_sicd_for_writing(sicd_meta)
        self._security_tags = None
        self._nitf_header = None
        self._img_groups = None
        self._img_details = None
        self._des_details = None
        super(SIDDWriter, self).__init__(file_name)

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
            # TODO: this should be more robust
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

        iid2 = None
        if hasattr(sidd, '_NITF') and isinstance(sidd._NITF, dict):
            iid2 = sidd._NITF.get('SUGGESTED_NAME', None)
        if iid2 is None:
            iid2 = 'SIDD: Unknown'
        return iid2

    def _get_ftitle(self):
        return self._get_iid2(0)

    def _get_fdt(self):
        sidd = self.sidd_meta[0]
        if sidd.ExploitationFeatures.Collections[0].Information.CollectionDateTime is not None:
            the_time = sidd.ExploitationFeatures.Collections[0].Information.CollectionDateTime.astype('datetime64[s]')
            return re.sub(r'[^0-9]', '', str(the_time))
        else:
            return super(SIDDWriter, self)._get_fdt()

    def _get_ostaid(self):
        ostaid = 'Unknown'
        sidd = self.sidd_meta[0]
        if hasattr(sidd, '_NITF') and isinstance(sidd._NITF, dict):
            ostaid = sidd._NITF.get('OSTAID', 'Unknown')
        return ostaid

    def _image_parameters(self, index):
        """
        Get the image parameters for the sidd at `index`.

        Parameters
        ----------
        index

        Returns
        -------
        (int, int, str, numpy.dtype, Union[bool, callable], str, tuple, tuple)
            pixel_size - the size of each pixel in bytes.
            abpp - the actual bits per pixel.
            irep - the image representation
            dtype - the data type.
            complex_type -
            pv_type - the pixel type string.
            irepband - the image representation.
            im_segments - Segmentation of the form `((row start, row end, column start, column end))`
        """

        sidd = self.sidd_meta[index]
        assert isinstance(sidd, (SIDDType, SIDDType1))
        if sidd.Display.PixelType == 'MONO8I':
            pixel_size = 1
            abpp = 8
            irep = 'MONO'
            dtype = numpy.dtype('>u1')
            complex_type = False
            pv_type = 'INT'
            irepband = ('M', )
        elif sidd.Display.PixelType == 'MONO16I':
            pixel_size = 2
            abpp = 16
            irep = 'MONO'
            dtype = numpy.dtype('>u1')
            complex_type = False
            pv_type = 'INT'
            irepband = ('M', )
        elif sidd.Display.PixelType == 'RGB24I':
            pixel_size = 3
            abpp = 8
            irep = 'RGB'
            dtype = numpy.dtype('>u1')
            complex_type = False
            pv_type = 'INT'
            irepband = ('R', 'G', 'B')
        else:
            raise ValueError('Unsupported PixelType {}'.format(sidd.Display.PixelType))

        image_segment_limits = image_segmentation(
            sidd.Measurement.PixelFootprint.Row, sidd.Measurement.PixelFootprint.Col, pixel_size)
        return pixel_size, abpp, irep, dtype, complex_type, pv_type, irepband, image_segment_limits

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
        pixel_size, abpp, irep, dtype, complex_type, pv_type, irepband, \
            image_segment_limits = self._image_parameters(index)
        new_count = len(image_segment_limits)
        img_groups.append(tuple(range(cur_count, cur_count+new_count)))
        security = self._get_security_tags(index)

        iid2 = self._get_iid2(index)
        sidd = self.sidd_meta[index]

        idatim = self._get_fdt()

        isorce = ''
        if sidd.ExploitationFeatures.Collections[0].Information.SensorName is not None:
            isorce = sidd.ExploitationFeatures.Collections[0].Information.SensorName

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
                IMODE='P' if irep == 'RGB' else 'B',
                NPPBH=get_npp_block(this_cols),
                NPPBV=get_npp_block(this_rows),
                NBPP=abpp,
                IDLVL=i+1+len(img_details),
                IALVL=i+len(img_details),
                ILOC='{0:05d}{1:05d}'.format(entry[0], entry[2]),
                Bands=ImageBands(values=bands),
                Security=security)
            img_details.append(ImageDetails(1, dtype, complex_type, entry, subhead))

    def _create_image_segment_details(self):
        super(SIDDWriter, self)._create_image_segment_details()
        img_groups = []
        img_details = []
        for index in range(len(self.sidd_meta)):
            self._create_image_segment(index, img_groups, img_details)
        self._img_groups = tuple(img_groups)
        self._img_details = tuple(img_details)

    def _create_des_segment(self, index):
        """
        Create the details for the data extension at `index`.

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
            desshdt = str(sidd.ProductCreation.ProcessorInformation.ProcessingDateTime)
        except AttributeError:
            desshdt = str(numpy.datetime64(datetime.now()))
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
            UserHeader=SICDDESSubheader(**uh_args))
        return DESDetails(subhead, sidd.to_xml_bytes(tag='SIDD'))

    def _create_data_extension_details(self):
        super(SIDDWriter, self)._create_data_extension_details()
        self._des_details = tuple(self._create_des_segment(index) for index in range(len(self.sidd_meta)))
