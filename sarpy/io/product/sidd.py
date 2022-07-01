"""
Module for reading and writing SIDD files
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from functools import reduce
import re
from typing import List, Tuple, Sequence, Union, BinaryIO, Optional

import numpy

from sarpy.io.xml.base import parse_xml_from_string

from sarpy.io.general.utils import is_file_like
from sarpy.io.general.base import SarpyIOError
from sarpy.io.general.nitf import NITFDetails, NITFReader, NITFWriter, \
    interpolate_corner_points_string, ImageSubheaderManager, \
    GraphicsSubheaderManager, TextSubheaderManager, DESSubheaderManager, \
    RESSubheaderManager, NITFWritingDetails, default_image_segmentation
from sarpy.io.general.nitf_elements.nitf_head import NITFHeader
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, XMLDESSubheader
from sarpy.io.general.nitf_elements.security import NITFSecurityTags
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, \
    ImageSegmentHeader0, ImageBands, ImageBand

from sarpy.io.product.base import SIDDTypeReader
from sarpy.io.product.sidd2_elements.SIDD import SIDDType as SIDDType2
from sarpy.io.product.sidd1_elements.SIDD import SIDDType as SIDDType1
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd import extract_clas as extract_clas_sicd, \
    create_security_tags_from_sicd


logger = logging.getLogger(__name__)

########
# module variables
_class_priority = {'U': 0, 'R': 1, 'C': 2, 'S': 3, 'T': 4}


#########
# Helper object for initially parses NITF header - specifically looking for SICD elements


class SIDDDetails(NITFDetails):
    """
    SIDD are stored in NITF 2.1 files.
    """

    __slots__ = (
        '_is_sidd', '_sidd_meta', '_sicd_meta')

    def __init__(self, file_object: Union[str, BinaryIO]):
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
        NITFDetails.__init__(self, file_object)

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

    @property
    def is_sidd(self) -> bool:
        """
        bool: whether file name corresponds to a SIDD file, or not.
        """

        return self._is_sidd

    @property
    def sidd_meta(self) -> Union[SIDDType2, SIDDType1, List[SIDDType2], List[SIDDType1]]:
        """
        None|SIDDType2|SIDDType1|List[SIDDType2]|List[SIDDType1]: the sidd meta-data structure(s).
        """

        return self._sidd_meta

    @property
    def sicd_meta(self) -> Optional[List[SICDType]]:
        """
        None|List[SICDType]: the sicd meta-data structure(s).
        """

        return self._sicd_meta

    def _find_sidd(self) -> None:
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
                        self._sidd_meta.append(SIDDType2.from_node(root_node, xml_ns, ns_key='default'))
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
                        self._sidd_meta.append(SIDDType2.from_node(root_node, xml_ns, ns_key='default'))
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

def _check_iid_format(iid1: str) -> bool:
    if not (iid1[:4] == 'SIDD' and iid1[4:].isnumeric()):
        return False
    return True


class SIDDReader(NITFReader, SIDDTypeReader):
    """
    A reader object for a SIDD file (NITF container with SIDD contents)
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
        SIDDTypeReader.__init__(self, None, self.nitf_details.sidd_meta, self.nitf_details.sicd_meta)
        NITFReader.__init__(self, nitf_details, reader_type="SIDD")
        self._check_sizes()

    @property
    def nitf_details(self) -> SIDDDetails:
        """
        SIDDDetails: The SIDD NITF details object.
        """

        return self._nitf_details

    def _check_image_segment_for_compliance(
            self,
            index: int,
            img_header: Union[ImageSegmentHeader, ImageSegmentHeader0]) -> bool:

        out = NITFReader._check_image_segment_for_compliance(self, index, img_header)
        if not out:
            return out

        out = True
        # skip anything but SAR for now (i.e. legend)
        if img_header.ICAT != 'SAR':
            logger.info('Image segment at index {} has ICAT != SAR'.format(index))
            out = False
        if not _check_iid_format(img_header.IID1):
            logger.info('Image segment at index {} has IID1 {}'.format(index, img_header.IID1))
            out = False
        return out

    def find_image_segment_collections(self) -> Tuple[Tuple[int, ...]]:
        segments = [[] for _ in self._sidd_meta]  # the number of segments that we should have

        for i, img_header in enumerate(self.nitf_details.img_headers):
            if i in self.unsupported_segments:
                continue  # skip these...

            iid1 = img_header.IID1  # required to be of the form SIDD######
            if not _check_iid_format(iid1):
                raise ValueError('Got unhandled image segment at index {}'.format(i))
            element = int(iid1[4:7])
            if element > len(self._sidd_meta):
                raise ValueError(
                    'Got image segment iid1 {}, but there are only {} sidd elements'.format(
                        iid1, len(self._sidd_meta)))
            segments[element - 1].append(i)
        # Ensure that all segments are populated with something...
        for i, entry in enumerate(segments):
            if len(entry) < 1:
                raise ValueError('Did not find any image segments for SIDD {}'.format(i))
        # noinspection PyTypeChecker
        return tuple(tuple(entry) for entry in segments)


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name: Union[str, BinaryIO]) -> Optional[SIDDReader]:
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
# The writer implementation

def validate_sidd_for_writing(
        sidd_meta: Union[SIDDType2, SIDDType1, List[SIDDType2], List[SIDDType1]]) -> Union[Tuple[SIDDType2, ...], Tuple[SIDDType1, ...]]:
    """
    Helper method which ensures the provided SIDD structure is appropriate.

    Parameters
    ----------
    sidd_meta : SIDDType2|List[SIDDType2]|SIDDType1|List[SIDDType1]

    Returns
    -------
    Tuple[SIDDType2, ...]|Tuple[SIDDType1, ...]
    """

    def inspect_sidd(the_sidd: Union[SIDDType2, SIDDType1]) -> None:
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
        elif isinstance(the_sidd, SIDDType2):
            urn = 'urn:SIDD:2.0.0'
        else:
            raise ValueError('Unhandled type {}'.format(type(the_sidd)))
        result = evaluate_xml_versus_schema(xml_str, urn)
        if result is False:
            logger.warning(
                'The provided SIDD does not properly validate\n\t'
                'against the schema for {}'.format(urn))

    if isinstance(sidd_meta, (SIDDType2, SIDDType1)):
        inspect_sidd(sidd_meta)
        # noinspection PyRedundantParentheses
        return (sidd_meta, )
    elif isinstance(sidd_meta, (tuple, list)):
        out = []
        for entry in sidd_meta:
            if not isinstance(entry, (SIDDType2, SIDDType1)):
                raise TypeError('All entries are required to be an instance of SIDDType, '
                                'got type {}'.format(type(entry)))
            inspect_sidd(entry)
            out.append(entry)
        return tuple(out)
    else:
        raise TypeError('sidd_meta is required to be an instance of SIDDType or a list/tuple '
                        'of such instances, got {}'.format(type(sidd_meta)))


def validate_sicd_for_writing(sicd_meta: Union[SICDType, Sequence[SICDType]]) -> Optional[Tuple[SICDType, ...]]:
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
        # noinspection PyRedundantParentheses
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


def extract_clas(the_sidd: Union[SIDDType2, SIDDType1]) -> str:
    """
    Extract the classification string from a SIDD as appropriate for NITF Security
    tags CLAS attribute.

    Parameters
    ----------
    the_sidd : SIDDType2|SIDDType1

    Returns
    -------
    str
    """

    class_str = the_sidd.ProductCreation.Classification.classification

    if class_str is None or class_str == '':
        return 'U'
    else:
        return class_str[:1]


def extract_clsy(the_sidd: Union[SIDDType2, SIDDType1]) -> str:
    """
    Extract the ownerProducer string from a SIDD as appropriate for NITF Security
    tags CLSY attribute.

    Parameters
    ----------
    the_sidd : SIDDType2|SIDDType1

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


def create_security_tags_from_sidd(sidd_meta: Union[SIDDType2, SIDDType1]) -> NITFSecurityTags:
    def get_basic_args():
        out = {}
        sec_tags = sidd_meta.NITF.get('Security', {})
        # noinspection PyProtectedMember
        for fld in NITFSecurityTags._ordering:
            if fld in sec_tags:
                out[fld] = sec_tags[fld]
        return out

    def get_clas():
        if 'CLAS' in args:
            return
        args['CLAS'] = extract_clas(sidd_meta)

    def get_code():
        # TODO: what to do here?
        if 'CODE' in args:
            return

    def get_clsy():
        if 'CLSY' in args:
            return
        args['CLSY'] = extract_clsy(sidd_meta)

    args = get_basic_args()
    if sidd_meta.ProductCreation is not None and \
            sidd_meta.ProductCreation.Classification is not None:
        get_clas()
        get_code()
        get_clsy()
    return NITFSecurityTags(**args)


class SIDDWritingDetails(NITFWritingDetails):
    __slots__ = (
        '_sidd_meta', '_sicd_meta', '_security_tags',
        '_sidd_security_tags', '_sicd_security_tags',
        '_row_limit')

    def __init__(
            self,
            sidd_meta: Union[SIDDType2, SIDDType1, Sequence[SIDDType2], Sequence[SIDDType1]],
            sicd_meta: Optional[Union[SICDType, Sequence[SICDType]]],
            row_limit: Optional[int] = None,
            additional_des: Optional[Sequence[DESSubheaderManager]] = None,
            graphics_managers: Optional[Tuple[GraphicsSubheaderManager, ...]] = None,
            text_managers: Optional[Tuple[TextSubheaderManager, ...]] = None,
            res_managers: Optional[Tuple[RESSubheaderManager, ...]] = None):
        """

        Parameters
        ----------
        sidd_meta : SIDDType2|List[SIDDType2]|SIDDType1|List[SIDDType1]
        sicd_meta : SICDType
        row_limit : None|int
            Desired row limit for the sicd image segments. Non-positive values
            or values > 99999 will be ignored.
        additional_des : None|Sequence[DESSubheaderManager]
        graphics_managers: Optional[Tuple[GraphicsSubheaderManager, ...]]
        text_managers: Optional[Tuple[TextSubheaderManager, ...]]
        res_managers: Optional[Tuple[RESSubheaderManager, ...]]
        """

        self._sidd_meta = None
        self._sidd_security_tags = None
        self._set_sidd_meta(sidd_meta)

        self._sicd_meta = None
        self._sicd_security_tags = None
        self._set_sicd_meta(sicd_meta)

        self._security_tags = None
        self._create_security_tags()

        self._row_limit = None
        self._set_row_limit(row_limit)

        header = self._create_header()
        image_managers, image_segment_collection, image_segment_coordinates = self._create_image_segments()
        des_managers = self._create_des_segments(additional_des)
        NITFWritingDetails.__init__(
            self,
            header,
            image_managers=image_managers,
            image_segment_collections=image_segment_collection,
            image_segment_coordinates=image_segment_coordinates,
            graphics_managers=graphics_managers,
            text_managers=text_managers,
            des_managers=des_managers,
            res_managers=res_managers)

    @property
    def sidd_meta(self) -> Union[Tuple[SIDDType2, ...], Tuple[SIDDType1, ...]]:
        """
        Tuple[SIDDType2, ...]: The sidd metadata.
        """

        return self._sidd_meta

    def _set_sidd_meta(self, value) -> None:
        if self._sidd_meta is not None:
            raise ValueError('sidd_meta is read only')
        if value is None:
            raise ValueError('sidd_meta is required.')

        self._sidd_meta = validate_sidd_for_writing(value)
        # noinspection PyTypeChecker
        self._sidd_security_tags = tuple(create_security_tags_from_sidd(entry) for entry in self._sidd_meta)

    @property
    def sicd_meta(self) -> Tuple[SICDType, ...]:
        """
        Tuple[SICDType, ...]: The sicd metadata
        """

        return self._sicd_meta

    def _set_sicd_meta(self, value) -> None:
        if self._sicd_meta is not None:
            raise ValueError('sicd_meta is read only')
        if value is None:
            self._sicd_meta = None
            self._sicd_security_tags = None
            return

        self._sicd_meta = validate_sicd_for_writing(value)
        # noinspection PyTypeChecker
        self._sicd_security_tags = tuple(create_security_tags_from_sicd(entry) for entry in self._sicd_meta)

    @property
    def row_limit(self) -> Tuple[int, ...]:
        return self._row_limit

    def _set_row_limit(self, value) -> None:
        if value is not None:
            if not isinstance(value, int):
                raise TypeError('row_bounds must be an integer')
            if value < 1:
                value = None

        if value is None or value > 99999:
            value = 99999

        im_seg_limit = 10**10 - 2  # allowable image segment size
        row_limit = []
        for sidd in self.sidd_meta:
            row_memory_size = sidd.Measurement.PixelFootprint.Col*sidd.Display.get_pixel_size()
            memory_limit = int(numpy.floor(im_seg_limit/row_memory_size))
            row_limit.append(min(value, memory_limit))
        self._row_limit = tuple(row_limit)

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

    def _get_iid2(self, index: int) -> str:
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

    def _get_ftitle(self, index: int = 0) -> str:
        sidd = self.sidd_meta[index]
        ftitle = sidd.NITF.get('FTITLE', None)
        if ftitle is None:
            ftitle = sidd.NITF.get('SUGGESTED_NAME', None)
        if ftitle is None:
            ftitle = 'SIDD: Unknown'
        return ftitle

    def _get_fdt(self, index: int) -> Optional[str]:
        sidd = self.sidd_meta[index]
        if sidd.ExploitationFeatures.Collections[0].Information.CollectionDateTime is not None:
            the_time = sidd.ExploitationFeatures.Collections[0].Information.CollectionDateTime.astype('datetime64[s]')
            return re.sub(r'[^0-9]', '', str(the_time))
        else:
            return None

    def _get_ostaid(self, index: int=0) -> str:
        sidd = self.sidd_meta[index]
        ostaid = sidd.NITF.get('OSTAID', 'Unknown')
        return ostaid

    def _get_isorce(self, index: int=0) -> str:
        sidd = self.sidd_meta[index]
        isorce = sidd.NITF.get('ISORCE', sidd.ExploitationFeatures.Collections[0].Information.SensorName)
        if isorce is None:
            isorce = 'Unknown'
        return isorce

    def _get_icp(self, sidd_index: int) -> Optional[numpy.ndarray]:
        """
        Get the Image corner point array, if possible.

        Parameters
        ----------
        sidd_index : int

        Returns
        -------
        None|numpy.ndarray
        """

        sidd = self.sidd_meta[sidd_index]  # type: Union[SIDDType2, SIDDType1]
        if isinstance(sidd, SIDDType2) and sidd.GeoData is not None and sidd.GeoData.ImageCorners is not None:
            return sidd.GeoData.ImageCorners.get_array(dtype=numpy.dtype('float64'))
        elif isinstance(sidd, SIDDType1) and sidd.GeographicAndTarget is not None and \
                sidd.GeographicAndTarget.GeographicCoverage is not None and \
                sidd.GeographicAndTarget.GeographicCoverage.Footprint is not None:
            return sidd.GeographicAndTarget.GeographicCoverage.Footprint.get_array(dtype=numpy.dtype('float64'))
        return None

    def _create_header(self) -> NITFHeader:
        """
        Create the main NITF header.

        Returns
        -------
        NITFHeader
        """

        # NB: CLEVEL and FL will be corrected...
        return NITFHeader(
            Security=self._security_tags, CLEVEL=3, OSTAID=self._get_ostaid(),
            FDT=self._get_fdt(0), FTITLE=self._get_ftitle(), FL=0)

    def _create_image_segment_for_sidd(
            self,
            sidd_index: int,
            starting_index: int) -> Tuple[List[ImageSubheaderManager], Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:

        image_managers = []
        sidd = self.sidd_meta[sidd_index]

        if isinstance(sidd, SIDDType2) and sidd.Compression is not None:
            raise ValueError('Compression not currently supported.')

        basic_args = {
            'ICAT': 'SAR',
            'IC': 'NC',
            'IID2' : self._get_iid2(sidd_index),
            'ISORCE': self._get_isorce(sidd_index),
            'IDATIM': self._get_fdt(sidd_index)
        }

        if sidd.Display.PixelType == 'MONO8I':
            basic_args['PVTYPE'] = 'INT'
            basic_args['NBPP'] = 8
            basic_args['ABPP'] = 8
            basic_args['IREP'] = 'MONO'
            basic_args['IMODE'] = 'B'
            irepband = ('M', )
        elif sidd.Display.PixelType == 'MONO16I':
            basic_args['PVTYPE'] = 'INT'
            basic_args['NBPP'] = 16
            basic_args['ABPP'] = 16
            basic_args['IREP'] = 'MONO'
            basic_args['IMODE'] = 'B'
            irepband = ('M', )
        elif sidd.Display.PixelType == 'RGB24I':
            basic_args['PVTYPE'] = 'INT'
            basic_args['NBPP'] = 8
            basic_args['ABPP'] = 8
            basic_args['IREP'] = 'RGB'
            basic_args['IMODE'] = 'P'
            irepband = ('R', 'G', 'B')
        else:
            raise ValueError('Unsupported PixelType {}'.format(sidd.Display.PixelType))

        rows = sidd.Measurement.PixelFootprint.Row
        cols = sidd.Measurement.PixelFootprint.Col
        icp = self._get_icp(sidd_index)
        image_segment_limits = default_image_segmentation(rows, cols, self._row_limit[sidd_index])

        image_segment_indices = []
        total_image_count = starting_index
        for i, entry in enumerate(image_segment_limits):
            if i == 0:
                iloc = '0000000000'
            else:
                prev_lims = image_segment_limits[i-1]
                prev_rows = prev_lims[1] - prev_lims[0]
                iloc = '{0:05d}00000'.format(prev_rows)

            this_rows = entry[1]-entry[0]
            this_cols = entry[3]-entry[2]
            subhead = ImageSegmentHeader(
                IID1='SIDD{0:03d}{1:03d}'.format(sidd_index+1, i+1),
                NROWS=this_rows,
                NCOLS=this_cols,
                IGEOLO=interpolate_corner_points_string(numpy.array(entry, dtype=numpy.int64), rows, cols, icp),
                NPPBH=0 if this_cols > 8192 else this_cols,
                NPPBV=0 if this_rows > 8192 else this_rows,
                IDLVL=sidd_index + i + 2,
                IALVL=sidd_index + i+ 1,
                ILOC=iloc,
                Bands=ImageBands(values=[ImageBand(ISUBCAT='', IREPBAND=entry) for entry in irepband]),
                Security=self._sidd_security_tags[sidd_index],
                **basic_args)
            image_managers.append(ImageSubheaderManager(subhead))
            image_segment_indices.append(total_image_count)
            total_image_count += 1
        return image_managers, tuple(image_segment_indices), image_segment_limits

    def _create_image_segments(self) -> Tuple[Tuple[ImageSubheaderManager, ...], Tuple[Tuple[int, ...], ...], Tuple[Tuple[Tuple[int, ...], ...]]]:
        image_managers = []
        image_segment_collection = []
        image_segment_coordinates = []
        starting_index = 0
        for i in range(len(self.sidd_meta)):
            t_managers, t_indices, t_limit = self._create_image_segment_for_sidd(i, starting_index)
            image_managers.extend(t_managers)
            image_segment_collection.append(t_indices)
            image_segment_coordinates.append(t_limit)
            starting_index = t_indices[-1] + 1
        return tuple(image_managers), tuple(image_segment_collection), tuple(image_segment_coordinates)

    def _create_des_segment_for_sidd(self, sidd_index: int) -> DESSubheaderManager:
        sidd = self.sidd_meta[sidd_index]
        uh_args = sidd.get_des_details()

        try:
            desshdt = str(sidd.ProductCreation.ProcessorInformation.ProcessingDateTime.astype('datetime64[s]'))
        except AttributeError:
            desshdt = str(numpy.datetime64('now'))
        if desshdt[-1] != 'Z':
            desshdt += 'Z'
        uh_args['DESSHDT'] = desshdt

        desshlpg = ''
        icp = self._get_icp(sidd_index)
        if icp is not None:
            temp = []
            for entry in icp:
                temp.append('{0:0=+12.8f}{1:0=+13.8f}'.format(entry[0], entry[1]))
            temp.append(temp[0])
            desshlpg = ''.join(temp)
        uh_args['DESSHLPG'] = desshlpg
        subhead = DataExtensionHeader(
            Security=self._sidd_security_tags[sidd_index],
            UserHeader=XMLDESSubheader(**uh_args))
        return DESSubheaderManager(
            subhead, sidd.to_xml_bytes(tag='SIDD'))

    def _create_sidd_des_segments(self) -> List[DESSubheaderManager]:
        return [self._create_des_segment_for_sidd(index) for index in range(len(self.sidd_meta))]

    def _create_des_segment_for_sicd(self, sicd_index: int) -> DESSubheaderManager:
        sicd = self.sicd_meta[sicd_index]
        uh_args = sicd.get_des_details()

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
            Security=self._sicd_security_tags[sicd_index],
            UserHeader=XMLDESSubheader(**uh_args))
        return DESSubheaderManager(
            subhead, sicd.to_xml_bytes(tag='SICD', urn=uh_args['DESSHTN']))

    def _create_sicd_des_segments(self) -> List[DESSubheaderManager]:
        if self.sicd_meta is None:
            return []
        return [self._create_des_segment_for_sicd(index) for index in range(len(self.sicd_meta))]

    def _create_des_segments(
            self,
            additional_des: Optional[Sequence[DESSubheaderManager]]) -> Tuple[DESSubheaderManager, ...]:

        if additional_des is not None:
            des_managers = list(additional_des)
        else:
            des_managers = []
        des_managers.extend(self._create_sidd_des_segments())
        des_managers.extend(self._create_sicd_des_segments())
        return tuple(des_managers)


class SIDDWriter(NITFWriter):
    """
    Writer class for a SIDD file - a NITF file following certain rules.

    **Changed in version 1.3.0** to reflect NITFWriter changes.
    """

    def __init__(
            self,
            file_object: Union[str, BinaryIO],
            sidd_meta: Optional[Union[SIDDType2, SIDDType1, Sequence[SIDDType2], Sequence[SIDDType1]]] = None,
            sicd_meta: Optional[Union[SICDType, Sequence[SICDType]]] = None,
            sidd_writing_details: Optional[SIDDWritingDetails] = None,
            check_existence: bool = True):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
        sidd_meta : None|SIDDType2|SIDDType1|Sequence[SIDDType2]|Sequence[SIDDType1]
        sicd_meta : None|SICDType|Sequence[SICDType]
        sidd_writing_details : None|SIDDWritingDetails
        check_existence : bool
            Should we check if the given file already exists?
        """

        if sidd_meta is None and sidd_writing_details is None:
            raise ValueError('One of sidd_meta or sidd_writing_details must be provided.')
        if sidd_writing_details is None:
            sidd_writing_details = SIDDWritingDetails(sidd_meta, sicd_meta=sicd_meta)
        NITFWriter.__init__(
            self, file_object, sidd_writing_details, check_existence=check_existence)

    @property
    def nitf_writing_details(self) -> SIDDWritingDetails:
        """
        SIDDWritingDetails: The SIDD/NITF subheader details.
        """

        # noinspection PyTypeChecker
        return self._nitf_writing_details

    @nitf_writing_details.setter
    def nitf_writing_details(self, value):
        if self._nitf_writing_details is not None:
            raise ValueError('nitf_writing_details is read-only')
        if not isinstance(value, SIDDWritingDetails):
            raise TypeError('nitf_writing_details must be of type {}'.format(SIDDWritingDetails))
        self._nitf_writing_details = value
