"""
Module for reading SICD files - should support SICD version 0.3 and above.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Wade Schwartzkopf")


import re
import logging
import os
from datetime import datetime
from typing import BinaryIO

import numpy

from sarpy.__about__ import __title__, __version__
from sarpy.io.general.base import AggregateChipper, SarpyIOError
from sarpy.io.general.nitf import NITFReader, NITFWriter, ImageDetails, DESDetails, \
    image_segmentation, get_npp_block, interpolate_corner_points_string
from sarpy.io.xml.base import parse_xml_from_string
from sarpy.io.general.utils import is_file_like
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType, get_specification_identifier
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType

from sarpy.io.general.nitf import NITFDetails
from sarpy.io.general.nitf_elements.des import DataExtensionHeader, XMLDESSubheader
from sarpy.io.general.nitf_elements.security import NITFSecurityTags
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, ImageBands, ImageBand


logger = logging.getLogger(__name__)


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
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


#########
# Helper object for initially parses NITF header - specifically looking for SICD elements


class SICDDetails(NITFDetails):
    """
    SICD are stored in NITF 2.1 files.
    """
    __slots__ = (
        '_des_index', '_des_header', '_is_sicd', '_sicd_meta',
        'img_segment_rows', 'img_segment_columns')

    def __init__(self, file_object):
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
        super(SICDDetails, self).__init__(file_object)
        if self._nitf_header.ImageSegments.subhead_sizes.size == 0:
            raise SarpyIOError('There are no image segments defined.')
        if self._nitf_header.GraphicsSegments.item_sizes.size > 0:
            raise SarpyIOError('A SICD file does not allow for graphics segments.')
        if self._nitf_header.DataExtensions.subhead_sizes.size == 0:
            raise SarpyIOError('A SICD file requires at least one data extension, containing the '
                          'SICD xml structure.')
        # define the sicd metadata
        self._find_sicd()
        if not self.is_sicd:
            raise SarpyIOError('Could not find the SICD XML des.')
        # populate the image details
        self.img_segment_rows = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
        self.img_segment_columns = numpy.zeros(self.img_segment_offsets.shape, dtype=numpy.int64)
        for i, im_header in enumerate(self.img_headers):
            self.img_segment_rows[i] = im_header.NROWS
            self.img_segment_columns[i] = im_header.NCOLS

    @property
    def is_sicd(self):
        """
        bool: whether file name corresponds to a SICD file, or not.
        """

        return self._is_sicd

    @property
    def sicd_meta(self):
        """
        sarpy.io.complex.sicd_elements.SICD.SICDType: the sicd meta-data structure.
        """

        return self._sicd_meta

    @property
    def des_header(self):
        """
        The DES subheader object associated with the SICD.

        Returns
        -------
        None|sarpy.io.general.nitf_elements.des.DataExtensionHeader
        """

        return self._des_header

    def _find_sicd(self):
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
                    root_node, xml_ns = parse_xml_from_string(des_bytes.decode('utf-8').strip())
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
                    root_node, xml_ns = parse_xml_from_string(des_bytes)
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
        except Exception as e:
            pass
        # TODO: account for the reference frequency offset situation

    def is_des_well_formed(self):
        """
        Returns whether the data extension subheader well-formed. Returns `None`
        if the DataExtensionHeader or the UserHeader section of it was not successfully
        parsed. Currently just checks the `DESSHSI` field for the required value.

        Returns
        -------
        bool|None
        """

        if not self._is_sicd or self.des_header is None or \
                not isinstance(self.des_header, DataExtensionHeader):
            return None

        sicd_des = self._des_header.UserHeader
        if not isinstance(sicd_des, XMLDESSubheader):
            return None
        return sicd_des.DESSHSI.strip() == get_specification_identifier()

    def repair_des_header(self):
        """
        Determines whether the data extension subheader is well-formed, and tries
        to repair it if not. Currently just sets the `DESSHSI` field to the
        required value.

        Returns `0` if wellformedness could not be evaluated, `1` if no change was
        required, `2` if the subheader was replaced, and `3` if the replacement effort
        failed (details logged at error level).

        Returns
        -------
        int
        """

        stat = self.is_des_well_formed()
        if stat is None:
            return 0
        elif stat:
            return 1

        sicd_des = self._des_header.UserHeader
        sicd_des.DESSHSI = get_specification_identifier()
        stat = self.rewrite_des_header()
        return 2 if stat else 3

    def rewrite_des_header(self):
        """
        Rewrites the DES subheader associated with the SICD from the current
        value in `des_header` property. This allows minor modifications to the
        security tags or user header information.

        Returns
        -------
        bool
            True is the modification was successful and False otherwise. Note that
            no errors, in particular io errors from write permission issues,
            are caught.
        """

        if not self._is_sicd:
            return False

        des_bytes = self.des_header.to_bytes()
        des_size = self._nitf_header.DataExtensions.subhead_sizes[self._des_index]
        if len(des_bytes) != des_size:
            logger.error(
                "The size of the current des header {} bytes,\n\t"
                "does not match the previous {} bytes.\n\t"
                "They cannot be trivially replaced.".format(des_bytes, des_size))
            return False
        des_loc = self.des_subheader_offsets[self._des_index]
        if not os.path.exists(self._file_name):
            raise ValueError('Operation not allowed.')
        with open(self._file_name, 'r+b') as fi:
            fi.seek(des_loc, os.SEEK_SET)
            fi.write(des_bytes)
        return True


#######
#  The actual reading implementation

def _validate_lookup(lookup_table):
    # type: (numpy.ndarray) -> None
    if not isinstance(lookup_table, numpy.ndarray):
        raise ValueError('requires a numpy.ndarray, got {}'.format(type(lookup_table)))
    if lookup_table.dtype.name != 'float64':
        raise ValueError('requires a numpy.ndarray of float64 dtype, got {}'.format(lookup_table.dtype))
    if lookup_table.shape != (256, ):
        raise ValueError('Requires a one-dimensional numpy.ndarray with 256 elements, '
                         'got shape {}'.format(lookup_table.shape))


def amp_phase_to_complex(lookup_table):
    """
    This constructs the function to convert from AMP8I_PHS8I format data to complex64 data.

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

        if data.dtype.name != 'uint8':
            raise ValueError('requires a numpy.ndarray of uint8 dtype, got {}'.format(data.dtype.name))

        if len(data.shape) != 3:
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the last dimension), got shape {}'.format(data.shape))

        out = numpy.zeros((data.shape[0], data.shape[1], int(data.shape[2]/2)), dtype=numpy.complex64)
        amp = lookup_table[data[:, :, 0::2]]
        theta = data[:, :, 1::2]*(2*numpy.pi/256)
        out.real = amp*numpy.cos(theta)
        out.imag = amp*numpy.sin(theta)
        return out
    return converter


class SICDReader(NITFReader, SICDTypeReader):
    """
    A reader object for a SICD file (NITF container with SICD contents)
    """

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

        SICDTypeReader.__init__(self, nitf_details.sicd_meta)
        NITFReader.__init__(self, nitf_details, reader_type='SICD')
        self._check_sizes()

    @property
    def nitf_details(self):
        # type: () -> SICDDetails
        """
        SICDDetails: The SICD NITF details object.
        """

        # noinspection PyTypeChecker
        return self._nitf_details

    def get_nitf_dict(self):
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

    def _find_segments(self):
        return [list(range(self.nitf_details.img_segment_offsets.size)), ]

    def _construct_chipper(self, segment, index):
        meta = self.sicd_meta
        pixel_type = meta.ImageData.PixelType
        # NB: SICDs are required to be stored as big-endian
        if pixel_type == 'RE32F_IM32F':
            raw_dtype = numpy.dtype('>f4')
            transform_data = 'COMPLEX'
        elif pixel_type == 'RE16I_IM16I':
            raw_dtype = numpy.dtype('>i2')
            transform_data = 'COMPLEX'
        elif pixel_type == 'AMP8I_PHS8I':
            raw_dtype = numpy.dtype('>u1')
            transform_data = amp_phase_to_complex(meta.ImageData.AmpTable)
        else:
            raise ValueError('Pixel Type {} not recognized.'.format(pixel_type))

        # verify that the collective output of _extract_chipper_params makes sense
        for img_index in segment:
            inp_dtype, _, _, _, _ = self._extract_chipper_params(img_index)
            if inp_dtype.name != raw_dtype.name:
                raise ValueError(
                    'Image segment at index {} apparently has dtype {}, expected {} '
                    'from the SICD definition'.format(img_index, inp_dtype, raw_dtype))

        if len(segment) == 1:
            return self._define_chipper(
                segment[0], raw_dtype=raw_dtype, raw_bands=2, transform_data=transform_data,
                output_dtype='complex64', output_bands=1)
        else:
            # get the bounds definition
            bounds = self._get_chipper_partitioning(segment, meta.ImageData.NumRows, meta.ImageData.NumCols)
            # define the chippers collection
            chippers = [
                self._define_chipper(img_index, raw_dtype=raw_dtype, raw_bands=2, transform_data=transform_data,
                                     output_dtype='complex64', output_bands=1) for img_index in segment]
            # define the aggregate chipper
            return AggregateChipper(bounds, 'complex64', chippers, output_bands=1)


#######
#  The actual writing implementation

def validate_sicd_for_writing(sicd_meta):
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


def _validate_input(data):
    # type: (numpy.ndarray) -> tuple
    if not isinstance(data, numpy.ndarray):
        raise ValueError('Requires a numpy.ndarray, got {}'.format(type(data)))
    if data.dtype.name not in ('complex64', 'complex128'):
        raise ValueError('Requires a numpy.ndarray of complex dtype, got {}'.format(data.dtype.name))
    if len(data.shape) != 2:
        raise ValueError('Requires a two-dimensional numpy.ndarray, got {}'.format(data.shape))

    new_shape = (data.shape[0], data.shape[1], 2)
    return new_shape


def complex_to_amp_phase(lookup_table):
    """
    This constructs the function to convert from complex64 or 128 to AMP8I_PHS8I format data.

    Parameters
    ----------
    lookup_table : numpy.ndarray

    Returns
    -------
    callable
    """

    _validate_lookup(lookup_table)

    def converter(data):
        new_shape = _validate_input(data)
        out = numpy.zeros(new_shape, dtype=numpy.uint8)
        # NB: for numpy before 1.10, digitize requires 1-d
        out[:, :, 0] = numpy.digitize(numpy.abs(data).ravel(), lookup_table, right=False).reshape(data.shape)
        out[:, :, 1] = numpy.arctan2(data.real, data.imag)*(256/(2*numpy.pi))
        # truncation takes care of properly rolling negative to positive
        return out

    return converter


def complex_to_int(data):
    """
    This converts from complex64 or 128 data to int16 data.

    Parameters
    ----------
    data : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    new_shape = _validate_input(data)

    if data.dtype.name == 'complex128':
        view_dtype = numpy.float64
    else:
        view_dtype = numpy.float32

    i16_info = numpy.iinfo(numpy.int16)  # for getting max/min type values
    data_view = data.view(dtype=view_dtype).reshape(new_shape)
    out = numpy.zeros(new_shape, dtype=numpy.int16)
    out[:] = numpy.round(numpy.clip(data_view, i16_info.min, i16_info.max))
    # this is nonsense without the clip - gets cast to int64 and then truncated.
    # should we round? Without it, it will be the floor, I believe.
    return out


def extract_clas(sicd):
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
        logger.critical(
            'Unclear how to extract CLAS for classification string {}.\n\t'
            'Should be set appropriately.'.format(c_str))
        return 'U'


class SICDWriter(NITFWriter):
    """
    Writer class for a SICD file - a NITF file containing complex radar data and 
    SICD data extension. 
    """

    __slots__ = ('_sicd_meta', '_check_older_version')

    def __init__(self, file_name, sicd_meta, check_older_version=False, check_existence=True):
        """

        Parameters
        ----------
        file_name : str
        sicd_meta : sarpy.io.complex.sicd_elements.SICD.SICDType
        check_older_version : bool
            Try to create a version 1.1 sicd, if possible?
        check_existence : bool
            Should we check if the given file already exists, and raises an exception if so?
        """

        self._check_older_version = check_older_version
        self._sicd_meta = validate_sicd_for_writing(sicd_meta)
        self._security_tags = None
        self._nitf_header = None
        self._img_groups = None
        self._img_details = None
        self._des_details = None
        self._shapes = ((self.sicd_meta.ImageData.NumRows, self.sicd_meta.ImageData.NumCols), )
        super(SICDWriter, self).__init__(file_name, check_existence=check_existence)

    @property
    def sicd_meta(self):
        """
        sarpy.io.complex.sicd_elements.SICD.SICDType: The sicd metadata
        """

        return self._sicd_meta

    def default_security_tags(self):
        """
        Returns a NITF security tags object with `CLAS` and `CODE`
        attributes set from the SICD.CollectionInfo.Classification value.

        It is expected that output from this will be modified as appropriate
        and used to set ONLY specific security tags in `data_extension_header` or
        elements of `image_segment_headers`.

        If simultaneous modification of all security tags attributes for the entire
        SICD is the goal, then directly modify the value(s) using `security_tags`.

        Returns
        -------
        sarpy.io.general.nitf_elements.security.NITFSecurityTags
        """

        def get_basic_args():
            out = {}
            sec_tags = self._sicd_meta.NITF.get('Security', {})
            # noinspection PyProtectedMember
            for fld in NITFSecurityTags._ordering:
                if fld in sec_tags:
                    out[fld] = sec_tags[fld]
            return out

        def get_clas():
            if 'CLAS' in args:
                return
            args['CLAS'] = extract_clas(self.sicd_meta)

        def get_code(in_str):
            if 'CODE' in args:
                return

            code = re.search('(?<=/)[^/].*', in_str)
            if code is not None:
                args['CODE'] = code.group()

        def get_clsy():
            if args.get('CLSY', '').strip() == '':
                args['CLSY'] = 'US'

        args = get_basic_args()
        if self._sicd_meta.CollectionInfo is not None:
            get_clas()
            get_code(self._sicd_meta.CollectionInfo.Classification)
            get_clsy()
        return NITFSecurityTags(**args)

    def _create_security_tags(self):
        self._security_tags = self.default_security_tags()

    def _get_ftitle(self):  # type: () -> str
        ftitle = self._sicd_meta.NITF.get('FTITLE', None)
        if ftitle is None:
            ftitle = self._sicd_meta.NITF.get('SUGGESTED_NAME', None)
        if ftitle is None:
            ftitle = self._sicd_meta.get_suggested_name(1)
        if ftitle is None and self._sicd_meta.CollectionInfo is not None and \
                self._sicd_meta.CollectionInfo.CoreName is not None:
            ftitle = 'SICD: {}'.format(self._sicd_meta.CollectionInfo.CoreName)
        if ftitle is None:
            ftitle = 'SICD: Unknown'
        if self._check_older_version and not ftitle.startswith('SICD:'):
            ftitle = 'SICD:' + ftitle
        return ftitle

    def _get_fdt(self):
        return re.sub(r'[^0-9]', '', str(self.sicd_meta.ImageCreation.DateTime.astype('datetime64[s]')))

    def _get_ostaid(self):
        ostaid = self._sicd_meta.NITF.get('OSTAID', 'Unknown')
        return ostaid

    def _get_isorce(self):
        isorce = self._sicd_meta.NITF.get('ISORCE', None)
        if isorce is None and \
                self.sicd_meta.CollectionInfo is not None and \
                self.sicd_meta.CollectionInfo.CollectorName is not None:
            isorce = 'SICD: {}'.format(self.sicd_meta.CollectionInfo.CollectorName)
        if isorce is None:
            isorce = 'SICD: Unknown Collector'
        return isorce

    def _get_iid2(self):
        iid2 = self._sicd_meta.NITF.get('IID2', self._get_ftitle())
        if self._check_older_version and not iid2.startswith('SICD:'):
            iid2 = 'SICD:' + iid2
        return iid2

    def _image_parameters(self):
        """
        Get the image parameters.

        Returns
        -------
        (int, numpy.dtype, Union[bool, callable], str, tuple, tuple)
            pixel_size - the size of each pixel in bytes.
            dtype - the data type.
            transform_data - the transform_data parameters
            pv_type - the pixel type string.
            isubcat - the image subcategory.
            im_segments - Segmentation of the form `((row start, row end, column start, column end))`
        """

        pixel_type = self.sicd_meta.ImageData.PixelType  # required to be defined
        # NB: SICDs are required to be stored as big-endian, so the endian-ness
        #   of the memmap must be explicit
        if pixel_type == 'RE32F_IM32F':
            pv_type, isubcat = 'R', ('I', 'Q')
            pixel_size = 8
            dtype = numpy.dtype('>f4')
            transform_data = 'COMPLEX'
        elif pixel_type == 'RE16I_IM16I':
            pv_type, isubcat = 'SI', ('I', 'Q')
            pixel_size = 4
            dtype = numpy.dtype('>i2')
            transform_data = complex_to_int
        elif pixel_type == 'AMP8I_PHS8I':
            pv_type, isubcat = 'INT', ('M', 'P')
            pixel_size = 2
            dtype = numpy.dtype('>u1')
            transform_data = complex_to_amp_phase(self.sicd_meta.ImageData.AmpTable)
        else:
            raise ValueError('Got unhandled pixel_type {}'.format(pixel_type))
        image_segment_limits = image_segmentation(
            self.sicd_meta.ImageData.NumRows, self.sicd_meta.ImageData.NumCols, pixel_size)
        return pixel_size, dtype, transform_data, pv_type, isubcat, image_segment_limits

    def _create_image_segment_details(self):
        super(SICDWriter, self)._create_image_segment_details()

        pixel_size, dtype, transform_data, pv_type, isubcat, image_segment_limits = self._image_parameters()
        img_groups = tuple(range(len(image_segment_limits)))
        self._img_groups = (img_groups, )

        iid2 = self._get_iid2()
        idatim = ' '
        if self.sicd_meta.Timeline is not None and self.sicd_meta.Timeline.CollectStart is not None:
            idatim = re.sub(r'[^0-9]', '', str(self.sicd_meta.Timeline.CollectStart.astype('datetime64[s]')))

        isource = self._get_isorce()

        icp, rows, cols = None, None, None
        if self.sicd_meta.GeoData is not None and self.sicd_meta.GeoData.ImageCorners is not None:
            # noinspection PyTypeChecker
            icp = self.sicd_meta.GeoData.ImageCorners.get_array(dtype=numpy.float64)
            rows = self.sicd_meta.ImageData.NumRows
            cols = self.sicd_meta.ImageData.NumCols
        abpp = 4*pixel_size
        bands = [ImageBand(ISUBCAT=entry) for entry in isubcat]

        img_details = []

        for i, entry in enumerate(image_segment_limits):
            this_rows = entry[1]-entry[0]
            this_cols = entry[3]-entry[2]
            subhead = ImageSegmentHeader(
                IID1='SICD{0:03d}'.format(0 if len(image_segment_limits) == 1 else i+1),
                IDATIM=idatim,
                IID2=iid2,
                ISORCE=isource,
                IREP='NODISPLY',
                ICAT='SAR',
                NROWS=this_rows,
                NCOLS=this_cols,
                PVTYPE=pv_type,
                ABPP=abpp,
                IGEOLO=interpolate_corner_points_string(numpy.array(entry, dtype=numpy.int64), rows, cols, icp),
                NBPC=1,
                NPPBH=get_npp_block(this_cols),
                NBPR=1,
                NPPBV=get_npp_block(this_rows),
                NBPP=abpp,
                IDLVL=i+1,
                IALVL=i,
                ILOC='{0:05d}{1:05d}'.format(entry[0], entry[2]),
                Bands=ImageBands(values=bands),
                Security=self._security_tags)
            img_details.append(ImageDetails(2, dtype, transform_data, entry, subhead))

        self._img_details = tuple(img_details)

    def _create_data_extension_details(self):
        super(SICDWriter, self)._create_data_extension_details()
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
            Security=self._security_tags,
            UserHeader=XMLDESSubheader(**uh_args))

        self._des_details = (
            DESDetails(subhead, self.sicd_meta.to_xml_bytes(tag='SICD', urn=uh_args['DESSHTN'])), )
