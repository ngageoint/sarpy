"""
Module for reading and writing SICD files
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import re
import logging
from datetime import datetime
from typing import BinaryIO, Union, Optional, Dict, Tuple, Sequence

import numpy

from sarpy.__about__ import __title__, __version__
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType

from sarpy.io.general.base import SarpyIOError
from sarpy.io.general.format_function import FormatFunction, ComplexFormatFunction
from sarpy.io.general.nitf import NITFDetails, NITFReader, NITFWriter, \
    interpolate_corner_points_string, default_image_segmentation, \
    ImageSubheaderManager, TextSubheaderManager, DESSubheaderManager, \
    RESSubheaderManager, NITFWritingDetails
from sarpy.io.general.nitf_elements.nitf_head import NITFHeader
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, XMLDESSubheader
from sarpy.io.general.nitf_elements.security import NITFSecurityTags
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, \
    ImageSegmentHeader0, ImageBands, ImageBand
from sarpy.io.general.utils import is_file_like

from sarpy.io.xml.base import parse_xml_from_string

logger = logging.getLogger(__name__)


#########
# Helper object for initially parses NITF header


class AmpLookupFunction(ComplexFormatFunction):
    __slots__ = ('_magnitude_lookup_table', )
    _allowed_ordering = ('MP', )

    def __init__(
            self,
            raw_dtype: Union[str, numpy.dtype],
            magnitude_lookup_table: numpy.ndarray,
            raw_shape: Optional[Tuple[int, ...]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Tuple[int, ...]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            band_dimension: int = -1):
        """

        Parameters
        ----------
        raw_dtype : str|numpy.dtype
            The raw datatype. Must be `uint8` up to endianness.
        magnitude_lookup_table : numpy.ndarray
        raw_shape : None|Tuple[int, ...]
        formatted_shape : None|Tuple[int, ...]
        reverse_axes : None|Tuple[int, ...]
        transpose_axes : None|Tuple[int, ...]
        band_dimension : int
            Which band is the complex dimension, **after** the transpose operation.
        """

        ComplexFormatFunction.__init__(
            self, raw_dtype, 'MP', raw_shape=raw_shape, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes, band_dimension=band_dimension)
        self._magnitude_lookup_table = None
        self.set_magnitude_lookup(magnitude_lookup_table)

    @property
    def magnitude_lookup_table(self) -> numpy.ndarray:
        """
        The magnitude lookup table, for SICD usage with `AMP8I_PH8I` pixel type.

        Returns
        -------
        numpy.ndarray
        """

        return self._magnitude_lookup_table

    def set_magnitude_lookup(self, lookup_table: numpy.ndarray) -> None:
        if not isinstance(lookup_table, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
        if lookup_table.dtype.name not in ['float32', 'float64']:
            raise ValueError('requires a numpy.ndarray of float32 or 64 dtype, got {}'.format(lookup_table.dtype))
        if lookup_table.dtype.name != 'float32':
            lookup_table = numpy.cast['float32'](lookup_table)
        if lookup_table.shape != (256,):
            raise ValueError('Requires a one-dimensional numpy.ndarray with 256 elements, '
                             'got shape {}'.format(lookup_table.shape))

        if self._raw_dtype.name != 'uint8':
            raise ValueError(
                'A magnitude lookup table has been supplied,\n\t'
                'but the raw datatype is not `uint8`.')
        self._magnitude_lookup_table = lookup_table

    def _forward_magnitude_theta(
            self,
            data: numpy.ndarray,
            out: numpy.ndarray,
            magnitude: numpy.ndarray,
            theta: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> None:
        magnitude = self.magnitude_lookup_table[magnitude]
        ComplexFormatFunction._forward_magnitude_theta(
            self, data, out, magnitude, theta, subscript)

    def _reverse_magnitude_theta(
            self,
            data: numpy.ndarray,
            out: numpy.ndarray,
            magnitude: numpy.ndarray,
            theta: numpy.ndarray,
            slice0: Tuple[slice, ...],
            slice1: Tuple[slice, ...]) -> None:
        magnitude = numpy.digitize(
            numpy.round(magnitude.ravel()), self.magnitude_lookup_table, right=False).reshape(data.shape)

        ComplexFormatFunction._reverse_magnitude_theta(self, data, out, magnitude, theta, slice0, slice1)


class SICDDetails(NITFDetails):
    """
    SICD are stored in NITF 2.1 files.
    """
    __slots__ = (
        '_des_index', '_des_header', '_is_sicd', '_sicd_meta')

    def __init__(self, file_object: Union[str, BinaryIO]):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
            file name or file like object for a NITF 2.1 or 2.0 containing a SICD.
        """

        self._des_index = None
        self._des_header = None
        self._img_headers = None
        self._is_sicd = False
        self._sicd_meta = None
        NITFDetails.__init__(self, file_object)

        if self._nitf_header.ImageSegments.subhead_sizes.size == 0:
            raise SarpyIOError('There are no image segments defined.')
        if self._nitf_header.GraphicsSegments.item_sizes.size > 0:
            raise SarpyIOError('A SICD file does not allow for graphics segments.')
        if self._nitf_header.DataExtensions.subhead_sizes.size == 0:
            raise SarpyIOError(
                'A SICD file requires at least one data extension, containing the '
                'SICD xml structure.')

        # define the sicd metadata
        self._find_sicd()
        if not self.is_sicd:
            raise SarpyIOError('Could not find the SICD XML des.')

    @property
    def is_sicd(self) -> bool:
        """
        bool: whether file name corresponds to a SICD file, or not.
        """

        return self._is_sicd

    @property
    def sicd_meta(self) -> SICDType:
        """
        SICDType: the sicd meta-data structure.
        """

        return self._sicd_meta

    @property
    def des_header(self) -> Optional[DataExtensionHeader]:
        """
        The DES subheader object associated with the SICD.

        Returns
        -------
        DataExtensionHeader
        """

        return self._des_header

    def _find_sicd(self) -> None:
        self._is_sicd = False
        self._sicd_meta = None
        if self.des_subheader_offsets is None:
            return

        for i in range(self.des_subheader_offsets.size):
            subhead_bytes = self.get_des_subheader_bytes(i)
            if subhead_bytes.startswith(b'DEXML_DATA_CONTENT'):
                des_header = DataExtensionHeader.from_bytes(subhead_bytes, start=0)
                des_bytes = self.get_des_bytes(i)
                # noinspection PyBroadException
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes.decode('utf-8').strip().encode())
                    if 'SIDD' in root_node.tag:  # namespace makes this ugly
                        # NOTE that SIDD files are supposed to have the corresponding
                        # SICD xml as one of the DES AFTER the SIDD xml.
                        # The same basic format is used for both headers.
                        # So, abandon if we find a SIDD xml
                        self._des_index = None
                        self._des_header = None
                        self._is_sicd = False
                        break
                    elif 'SICD' in root_node.tag:  # namespace makes this ugly
                        self._des_index = i
                        self._des_header = des_header
                        self._is_sicd = True
                        if xml_ns is None:
                            self._sicd_meta = SICDType.from_node(root_node, xml_ns, ns_key=None)
                        else:
                            self._sicd_meta = SICDType.from_node(root_node, xml_ns, ns_key='default')
                        break
                except Exception:
                    continue
            elif subhead_bytes.startswith(b'DESIDD_XML'):
                # This is an old format SIDD and can't be a SICD
                self._des_index = None
                self._des_header = None
                self._is_sicd = False
                break
            elif subhead_bytes.startswith(b'DESICD_XML'):
                # This is an old format SICD
                des_bytes = self.get_des_bytes(i)
                try:
                    root_node, xml_ns = parse_xml_from_string(des_bytes.decode('utf-8').strip().encode())
                    if 'SICD' in root_node.tag:  # namespace makes this ugly
                        self._des_index = i
                        self._des_header = None
                        self._is_sicd = True
                        if xml_ns is None:
                            self._sicd_meta = SICDType.from_node(root_node, xml_ns, ns_key=None)
                        else:
                            self._sicd_meta = SICDType.from_node(root_node, xml_ns, ns_key='default')
                        break
                except Exception as e:
                    logger.error(
                        'We found an apparent old-style SICD DES header,\n\t'
                        'but failed parsing with error {}'.format(e))
                    continue

        if not self._is_sicd:
            return

        # noinspection PyBroadException
        try:
            self._sicd_meta.derive()
        except Exception:
            pass
        # TODO: account for the reference frequency offset situation


#######
#  The actual reading implementation

class SICDReader(NITFReader, SICDTypeReader):
    """
    A SICD reader implementation - file is NITF container following specified rules.

    **Changed in version 1.3.0** for reading changes.
    """
    _maximum_number_of_images = 1

    def __init__(self, nitf_details):
        """

        Parameters
        ----------
        nitf_details :  : str|BinaryIO|SICDDetails
            filename, file-like object, or SICDDetails object
        """

        if isinstance(nitf_details, str) or is_file_like(nitf_details):
            nitf_details = SICDDetails(nitf_details)
        if not isinstance(nitf_details, SICDDetails):
            raise TypeError(
                'The input argument for SICDReader must be a filename, file-like object, '
                'or SICDDetails object.')

        SICDTypeReader.__init__(self, None, nitf_details.sicd_meta)
        NITFReader.__init__(self, nitf_details, reader_type='SICD')
        self._check_sizes()

    @property
    def nitf_details(self) -> SICDDetails:
        """
        SICDDetails: The SICD NITF details object.
        """

        # noinspection PyTypeChecker
        return self._nitf_details

    def get_nitf_dict(self) -> Dict:
        """
        Populate a dictionary with the pertinent NITF header information. This
        is for use in more faithful preservation of NITF header information
        in copying or rewriting sicd files.

        Returns
        -------
        dict
        """

        out = {}
        security = {}
        security_obj = self.nitf_details.nitf_header.Security
        # noinspection PyProtectedMember
        for field in NITFSecurityTags._ordering:
            value = getattr(security_obj, field).strip()
            if value != '':
                security[field] = value
        if len(security) > 0:
            out['Security'] = security

        out['OSTAID'] = self.nitf_details.nitf_header.OSTAID
        out['FTITLE'] = self.nitf_details.nitf_header.FTITLE
        out['ISORCE'] = self.nitf_details.img_headers[0].ISORCE
        out['IID2'] = self.nitf_details.img_headers[0].IID2
        return out

    def populate_nitf_information_into_sicd(self):
        """
        Populate some pertinent NITF header information into the SICD structure.
        This provides more faithful copying or rewriting options.
        """

        self._sicd_meta.NITF = self.get_nitf_dict()

    def depopulate_nitf_information(self):
        """
        Eliminates the NITF information dict from the SICD structure.
        """

        self._sicd_meta.NITF = {}

    def get_format_function(
            self,
            raw_dtype: numpy.dtype,
            complex_order: Optional[str],
            lut: Optional[numpy.ndarray],
            band_dimension: int,
            image_segment_index: Optional[int] = None,
            **kwargs) -> Optional[FormatFunction]:
        if complex_order is not None and complex_order != 'IQ':
            if complex_order != 'MP' or raw_dtype.name != 'uint8' or band_dimension != 2:
                raise ValueError('Got unsupported SICD band type definition')
            if self.sicd_meta.ImageData.PixelType != 'AMP8I_PH8I' or self.sicd_meta.ImageData.AmpTable is None:
                raise ValueError('Expected AMP8I_PH8I')
            return AmpLookupFunction(raw_dtype, self.sicd_meta.ImageData.AmpTable)
        return NITFReader.get_format_function(
            self, raw_dtype, complex_order, lut, band_dimension, image_segment_index, **kwargs)

    def _check_image_segment_for_compliance(
            self,
            index: int,
            img_header: Union[ImageSegmentHeader, ImageSegmentHeader0]) -> bool:
        out = NITFReader._check_image_segment_for_compliance(self, index, img_header)
        if not out:
            return out

        raw_dtype, formatted_dtype, formatted_bands, complex_order, lut = self._get_dtypes(index)
        if complex_order is None or complex_order not in ['IQ', 'MP']:
            logger.error(
                'Image segment at index {} is not of appropriate type for a SICD Image Segment'.format(index))
            return False
        if formatted_bands != 1:
            logger.error(
                'Image segment at index {} has multiple complex bands'.format(index))
            return False

        raw_name = raw_dtype.name
        pixel_type = self.sicd_meta.ImageData.PixelType
        if pixel_type == 'RE32F_IM32F':
            if complex_order != 'IQ' or raw_name != 'float32':
                logger.error(
                    'Image segment at index {} required to be compatible\n\t'
                    'with PIXEL_TYPE {}'.format(index, pixel_type))
                return False
        elif pixel_type == 'RE16I_IM16I':
            if complex_order != 'IQ' or raw_name != 'int16':
                logger.error(
                    'Image segment at index {} required to be compatible\n\t'
                    'with PIXEL_TYPE {}'.format(index, pixel_type))
                return False
        elif pixel_type == 'AMP8I_PHS8I':
            if complex_order != 'MP' or raw_name != 'uint8':
                logger.error(
                    'Image segment at index {} required to be compatible\n\t'
                    'with PIXEL_TYPE {}'.format(index, pixel_type))
                return False
        else:
            raise ValueError('Unhandled PIXEL_TYPE {}'.format(pixel_type))
        return True

    def find_image_segment_collections(self) -> Tuple[Tuple[int, ...]]:
        return (
            tuple(index for index in range(len(self.nitf_details.img_headers))
                  if index not in self.unsupported_segments), )


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name: Union[str, BinaryIO]) -> Optional[SICDReader]:
    """
    Tests whether a given file_name corresponds to a SICD file, and returns
    a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    SICDReader|None
    """

    try:
        nitf_details = SICDDetails(file_name)
        if nitf_details.is_sicd:
            logger.info('File {} is determined to be a SICD (NITF format) file.'.format(file_name))
            return SICDReader(nitf_details)
        else:
            return None
    except SarpyIOError:
        return None


#######
#  The actual writing implementation

def validate_sicd_for_writing(sicd_meta: SICDType) -> SICDType:
    """
    Helper method which ensures the provided SICD structure provides enough
    information to support file writing, as well as ensures a few basic items
    are populated as appropriate.

    Parameters
    ----------
    sicd_meta : SICDType

    Returns
    -------
    SICDType
        This returns a deep copy of the provided SICD structure, with any
        necessary modifications.
    """

    if not isinstance(sicd_meta, SICDType):
        raise ValueError('sicd_meta is required to be an instance of SICDType, got {}'.format(type(sicd_meta)))
    if sicd_meta.ImageData is None:
        raise ValueError('The sicd_meta has un-populated ImageData, and nothing useful can be inferred.')
    if sicd_meta.ImageData.NumCols is None or sicd_meta.ImageData.NumRows is None:
        raise ValueError('The sicd_meta has ImageData with unpopulated NumRows or NumCols, '
                         'and nothing useful can be inferred.')
    if sicd_meta.ImageData.PixelType is None:
        logger.warning('The PixelType for sicd_meta is unset, so defaulting to RE32F_IM32F.')
        sicd_meta.ImageData.PixelType = 'RE32F_IM32F'

    sicd_meta = sicd_meta.copy()

    profile = '{} {}'.format(__title__, __version__)
    if sicd_meta.ImageCreation is None:
        sicd_meta.ImageCreation = ImageCreationType(
            Application=profile,
            DateTime=numpy.datetime64(datetime.now()),
            Profile=profile)
    else:
        sicd_meta.ImageCreation.Profile = profile
        if sicd_meta.ImageCreation.DateTime is None:
            sicd_meta.ImageCreation.DateTime = numpy.datetime64(datetime.now())
    return sicd_meta


def extract_clas(sicd: SICDType) -> str:
    """
    Extract the classification string from a SICD as appropriate for NITF Security
    tags CLAS attribute.

    Parameters
    ----------
    sicd : SICDType

    Returns
    -------
    str
    """
    if sicd.CollectionInfo is None or sicd.CollectionInfo.Classification is None:
        return 'U'

    c_str = sicd.CollectionInfo.Classification.upper().strip()

    if 'UNCLASS' in c_str or c_str == 'U':
        return 'U'
    elif 'CONFIDENTIAL' in c_str or c_str == 'C' or c_str.startswith('C/'):
        return 'C'
    elif 'TOP SECRET' in c_str or c_str == 'TS' or c_str.startswith('TS/'):
        return 'T'
    elif 'SECRET' in c_str or c_str == 'S' or c_str.startswith('S/'):
        return 'S'
    elif 'FOUO' in c_str.upper() or 'RESTRICTED' in c_str.upper():
        return 'R'
    else:
        logger.error(
            'Unclear how to extract CLAS for classification string {}.\n\t'
            'Should be set appropriately.'.format(c_str))
        return 'U'


def create_security_tags_from_sicd(sicd_meta: SICDType) -> NITFSecurityTags:
    def get_basic_args():
        out = {}
        sec_tags = sicd_meta.NITF.get('Security', {})
        # noinspection PyProtectedMember
        for fld in NITFSecurityTags._ordering:
            if fld in sec_tags:
                out[fld] = sec_tags[fld]
        return out

    def get_clas():
        if 'CLAS' in args:
            return
        args['CLAS'] = extract_clas(sicd_meta)

    def get_code(in_str):
        if 'CODE' in args:
            return

        # TODO: this is pretty terrible...
        code = re.search('(?<=/)[^/].*', in_str)
        if code is not None:
            args['CODE'] = code.group()

    def get_clsy():
        if args.get('CLSY', '').strip() == '':
            args['CLSY'] = 'US'

    args = get_basic_args()
    if sicd_meta.CollectionInfo is not None:
        get_clas()
        get_code(sicd_meta.CollectionInfo.Classification)
        get_clsy()

    return NITFSecurityTags(**args)


class SICDWritingDetails(NITFWritingDetails):
    """
    Manager for all the NITF subheader information associated with the SICD.

    Introduced in version 1.3.0.
    """

    __slots__ = (
        '_sicd_meta', '_security_tags', '_row_limit', '_check_older_version',
        '_required_version')

    def __init__(
            self,
            sicd_meta: SICDType,
            row_limit: Optional[int] = None,
            additional_des: Optional[Sequence[DESSubheaderManager]] = None,
            text_managers: Optional[Tuple[TextSubheaderManager, ...]] = None,
            res_managers: Optional[Tuple[RESSubheaderManager, ...]] = None,
            check_older_version: bool = False):
        """

        Parameters
        ----------
        sicd_meta : SICDType
        row_limit : None|int
            Desired row limit for the sicd image segments. Non-positive values
            or values > 99999 will be ignored.
        additional_des : None|Sequence[DESSubheaderManager]
        text_managers: Optional[Tuple[TextSubheaderManager, ...]]
        res_managers: Optional[Tuple[RESSubheaderManager, ...]]
        check_older_version : bool
            Try to create an older version sicd, for compliance
        """

        self._check_older_version = bool(check_older_version)
        self._security_tags = None
        self._sicd_meta = None
        self._set_sicd_meta(sicd_meta)
        self._required_version = self.sicd_meta.version_required()
        self._create_security_tags()
        self._row_limit = None
        self._set_row_limit(row_limit)

        header = self._create_header()
        image_managers, image_segment_collections, image_segment_coordinates = self._create_image_segments()
        des_managers = self._create_des_segments(additional_des)

        # NB: graphics not permitted in sicd
        NITFWritingDetails.__init__(
            self,
            header,
            image_managers=image_managers,
            image_segment_collections=image_segment_collections,
            image_segment_coordinates=image_segment_coordinates,
            text_managers=text_managers,
            des_managers=des_managers,
            res_managers=res_managers)

    @property
    def sicd_meta(self) -> SICDType:
        """
        SICDType: The sicd metadata
        """

        return self._sicd_meta

    def _set_sicd_meta(self, value):
        if self._sicd_meta is not None:
            raise ValueError('sicd_meta is read only')
        self._sicd_meta = validate_sicd_for_writing(value)

    @property
    def requires_version(self) -> Tuple[int, int, int]:
        """
        Tuple[int, int, int]: What is the required (at minimum) sicd version?
        """

        return self._required_version

    @property
    def row_limit(self) -> int:
        return self._row_limit

    def _set_row_limit(self, value):
        if value is not None:
            if not isinstance(value, int):
                raise TypeError('row_bounds must be an integer')
            if value < 1:
                value = None

        if value is None or value > 99999:
            value = 99999

        im_seg_limit = 10**10 - 2  # allowable image segment size
        row_memory_size = self.sicd_meta.ImageData.NumCols*self.sicd_meta.ImageData.get_pixel_size()

        memory_limit = int(numpy.floor(im_seg_limit/row_memory_size))
        self._row_limit = min(value, memory_limit)

    @property
    def security_tags(self) -> NITFSecurityTags:
        """
        NITFSecurityTags: The default NITF security tags for use.
        """

        return self._security_tags

    def _create_security_tags(self) -> None:
        """
        Creates a NITF security tags object with `CLAS` and `CODE` attributes in
        the sicd_meta.NITF property and/or extracted from the
        SICD.CollectionInfo.Classification value.

        Returns
        -------
        None
        """

        self._security_tags = create_security_tags_from_sicd(self.sicd_meta)

    def _get_ftitle(self) -> str:
        ftitle = self.sicd_meta.NITF.get('FTITLE', None)
        if ftitle is None:
            ftitle = self.sicd_meta.NITF.get('SUGGESTED_NAME', None)
        if ftitle is None:
            ftitle = self.sicd_meta.get_suggested_name(1)
        if ftitle is None and self.sicd_meta.CollectionInfo is not None and \
                self.sicd_meta.CollectionInfo.CoreName is not None:
            ftitle = 'SICD: {}'.format(self.sicd_meta.CollectionInfo.CoreName)
        if ftitle is None:
            ftitle = 'SICD: Unknown'
        if self._check_older_version and self._required_version < (1, 2, 0) and \
                not ftitle.startswith('SICD:'):
            ftitle = 'SICD:' + ftitle
        return ftitle

    def _get_fdt(self):
        return re.sub(r'[^0-9]', '', str(self.sicd_meta.ImageCreation.DateTime.astype('datetime64[s]')))

    def _get_idatim(self) -> str:
        idatim = ' '
        if self.sicd_meta.Timeline is not None and self.sicd_meta.Timeline.CollectStart is not None:
            idatim = re.sub(r'[^0-9]', '', str(self.sicd_meta.Timeline.CollectStart.astype('datetime64[s]')))
        return idatim

    def _get_ostaid(self) -> str:
        ostaid = self.sicd_meta.NITF.get('OSTAID', 'Unknown')
        return ostaid

    def _get_isorce(self) -> str:
        isorce = self.sicd_meta.NITF.get('ISORCE', None)
        if isorce is None and \
                self.sicd_meta.CollectionInfo is not None and \
                self.sicd_meta.CollectionInfo.CollectorName is not None:
            isorce = 'SICD: {}'.format(self.sicd_meta.CollectionInfo.CollectorName)
        if isorce is None:
            isorce = 'SICD: Unknown Collector'
        return isorce

    def _get_iid2(self) -> str:
        iid2 = self.sicd_meta.NITF.get('IID2', self._get_ftitle())
        if self._check_older_version and self._required_version < (1, 2, 0) and \
                not iid2.startswith('SICD:'):
            iid2 = 'SICD:' + iid2
        return iid2

    def _create_header(self) -> NITFHeader:
        """
        Create the main NITF header.

        Returns
        -------
        NITFHeader
        """

        # NB: CLEVEL and FL will be corrected...
        return NITFHeader(
            Security=self.security_tags, CLEVEL=3, OSTAID=self._get_ostaid(),
            FDT=self._get_fdt(), FTITLE=self._get_ftitle(), FL=0)

    def _create_image_segments(self) -> Tuple[
            Tuple[ImageSubheaderManager, ...],
            Tuple[Tuple[int, ...], ...],
            Tuple[Tuple[Tuple[int, ...], ...]]]:
        image_managers = []
        basic_args = {
            'IREP': 'NODISPLY',
            'IC': 'NC',
            'ICAT': 'SAR',
            'IID2': self._get_iid2(),
            'IDATIM': self._get_idatim(),
            'ISORCE': self._get_isorce()
        }
        pixel_type = self.sicd_meta.ImageData.PixelType  # required to be defined
        # NB: SICDs are required to be stored as big-endian, so the endian-ness
        #   of the memmap must be explicit
        if pixel_type == 'RE32F_IM32F':
            basic_args['PVTYPE'] = 'R'
            basic_args['NBPP'] = 32
            basic_args['ABPP'] = 32
            isubcat = ('I', 'Q')
        elif pixel_type == 'RE16I_IM16I':
            basic_args['PVTYPE'] = 'SI'
            basic_args['NBPP'] = 16
            basic_args['ABPP'] = 16
            isubcat = ('I', 'Q')
        elif pixel_type == 'AMP8I_PHS8I':
            basic_args['PVTYPE'] = 'INT'
            basic_args['NBPP'] = 8
            basic_args['ABPP'] = 8
            isubcat = ('M', 'P')
        else:
            raise ValueError('Got unhandled pixel_type {}'.format(pixel_type))

        rows = self.sicd_meta.ImageData.NumRows
        cols = self.sicd_meta.ImageData.NumCols
        icp = None
        if self.sicd_meta.GeoData is not None and self.sicd_meta.GeoData.ImageCorners is not None:
            # noinspection PyTypeChecker
            icp = self.sicd_meta.GeoData.ImageCorners.get_array(dtype=numpy.float64)

        image_segment_limits = default_image_segmentation(rows, cols, self.row_limit)

        image_segment_collection = (tuple(range(len(image_segment_limits))), )
        image_segment_coordinates = (image_segment_limits, )

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
                IID1='SICD{0:03d}'.format(0 if len(image_segment_limits) == 1 else i+1),
                NROWS=this_rows,
                NCOLS=this_cols,
                IGEOLO=interpolate_corner_points_string(numpy.array(entry, dtype=numpy.int64), rows, cols, icp),
                NPPBH=0 if this_cols > 8192 else this_cols,
                NPPBV=0 if this_rows > 8192 else this_rows,
                NBPC=1,
                NBPR=1,
                IDLVL=i+1,
                IALVL=i,
                ILOC=iloc,
                Bands=ImageBands(values=[ImageBand(ISUBCAT=entry) for entry in isubcat]),
                Security=self._security_tags,
                **basic_args)
            image_managers.append(ImageSubheaderManager(subhead))
        return tuple(image_managers), image_segment_collection, image_segment_coordinates

    def _create_sicd_des(self) -> DESSubheaderManager:
        uh_args = self.sicd_meta.get_des_details(self._check_older_version)
        desshdt = str(self.sicd_meta.ImageCreation.DateTime.astype('datetime64[s]'))
        if desshdt[-1] != 'Z':
            desshdt += 'Z'
        uh_args['DESSHDT'] = desshdt

        desshlpg = ''
        if self.sicd_meta.GeoData is not None and self.sicd_meta.GeoData.ImageCorners is not None:
            # noinspection PyTypeChecker
            icp = self.sicd_meta.GeoData.ImageCorners.get_array(dtype=numpy.float64)
            temp = []
            for entry in icp:
                temp.append('{0:0=+12.8f}{1:0=+13.8f}'.format(entry[0], entry[1]))
            temp.append(temp[0])
            desshlpg = ''.join(temp)
        uh_args['DESSHLPG'] = desshlpg

        subhead = DataExtensionHeader(
            Security=self.security_tags,
            UserHeader=XMLDESSubheader(**uh_args))
        return DESSubheaderManager(
            subhead, self.sicd_meta.to_xml_bytes(tag='SICD', urn=uh_args['DESSHTN']))

    def _create_des_segments(
            self,
            additional_des: Optional[Sequence[DESSubheaderManager]]) -> Tuple[DESSubheaderManager, ...]:

        if additional_des is not None:
            des_managers = list(additional_des)
        else:
            des_managers = []
        des_managers.append(self._create_sicd_des())
        return tuple(des_managers)


class SICDWriter(NITFWriter):
    """
    Writer class for a SICD file - a NITF file containing complex radar data and
    SICD data extension.

    **Changed in version 1.3.0** to reflect NITFWriter changes.
    """

    def __init__(
            self,
            file_object: Union[str, BinaryIO],
            sicd_meta: Optional[SICDType] = None,
            sicd_writing_details: Optional[SICDWritingDetails] = None,
            check_older_version: bool = False,
            check_existence: bool = True):
        """

        Parameters
        ----------
        file_object : str|BinaryIO
        sicd_meta : None|SICDType
        sicd_writing_details : None|SICDWritingDetails
        check_older_version : bool
            Try to create an older version sicd, for compliance with standard
            NGA applications like SOCET or RemoteView
        check_existence : bool
            Should we check if the given file already exists?
        """

        if sicd_meta is None and sicd_writing_details is None:
            raise ValueError('One of sicd_meta or sicd_writing_details must be provided.')
        if sicd_writing_details is None:
            sicd_writing_details = SICDWritingDetails(sicd_meta, check_older_version=check_older_version)
        NITFWriter.__init__(
            self, file_object, sicd_writing_details, check_existence=check_existence)

    @property
    def nitf_writing_details(self) -> SICDWritingDetails:
        """
        SICDWritingDetails: The SICD/NITF subheader details.
        """

        # noinspection PyTypeChecker
        return self._nitf_writing_details

    @nitf_writing_details.setter
    def nitf_writing_details(self, value):
        if self._nitf_writing_details is not None:
            raise ValueError('nitf_writing_details is read-only')
        if not isinstance(value, SICDWritingDetails):
            raise TypeError('nitf_writing_details must be of type {}'.format(SICDWritingDetails))
        self._nitf_writing_details = value

    @property
    def sicd_meta(self) -> SICDType:
        return self.nitf_writing_details.sicd_meta

    def get_format_function(
            self,
            raw_dtype: numpy.dtype,
            complex_order: Optional[str],
            lut: Optional[numpy.ndarray],
            band_dimension: int,
            image_segment_index: Optional[int] = None,
            **kwargs) -> Optional[FormatFunction]:
        if complex_order is not None and complex_order != 'IQ':
            if complex_order != 'MP' or raw_dtype.name != 'uint8' or band_dimension != 2:
                raise ValueError('Got unsupported SICD band type definition')
            if self.sicd_meta.ImageData.PixelType != 'AMP8I_PH8I' or self.sicd_meta.ImageData.AmpTable is None:
                raise ValueError('Expected AMP8I_PH8I')
            return AmpLookupFunction(raw_dtype, self.sicd_meta.ImageData.AmpTable)
        return NITFWriter.get_format_function(
            self, raw_dtype, complex_order, lut, band_dimension, image_segment_index, **kwargs)
