"""
Module providing api consistent with other file types for reading tiff files.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Daniel Haverporth")

# It was the original intent to use gdal for the bulk of tiff reading
# Unfortunately, the necessary sarpy functionality can only be obtained by
# gdal_dataset.GetVirtualMemArray(). As of July 2020, this is supported only
# on Linux platforms - unclear what more constraints. So, using gdal to provide
# the reading capability is not feasible at present.

import logging
import os

import numpy
import re
from typing import Union, Tuple, Dict, BinaryIO, Sequence

from sarpy.io.general.base import BaseReader, SarpyIOError
from sarpy.io.general.format_function import ComplexFormatFunction
from sarpy.io.general.data_segment import NumpyMemmapSegment

logger = logging.getLogger(__name__)


_BASELINE_TAGS = {
    254: 'NewSubfileType',
    255: 'SubfileType',
    256: 'ImageWidth',
    257: 'ImageLength',
    258: 'BitsPerSample',
    259: 'Compression',
    262: 'PhotometricInterpretation',
    263: 'Thresholding',
    264: 'CellWidth',
    265: 'CellLength',
    266: 'FillOrder',
    270: 'ImageDescription',
    271: 'Make',
    272: 'Model',
    273: 'StripOffsets',
    274: 'Orientation',
    277: 'SamplesPerPixel',
    278: 'RowsPerStrip',
    279: 'StripByteCounts',
    280: 'MinSampleValue',
    281: 'MaxSampleValue',
    282: 'XResolution',
    283: 'YResolution',
    284: 'PlanarConfiguration',
    288: 'FreeOffsets',
    289: 'FreeByteCounts',
    290: 'GrayResponseUnit',
    291: 'GrayResponseCurve',
    296: 'ResolutionUnit',
    305: 'Software',
    306: 'DateTime',
    315: 'Artist',
    316: 'HostComputer',
    320: 'ColorMap',
    338: 'ExtraSamples',
    33432: 'Copyright',
}
_EXTENSION_TAGS = {
    269: 'DocumentName',
    285: 'PageName',
    286: 'XPosition',
    287: 'YPosition',
    292: 'T4Options',
    293: 'T6Options',
    297: 'PageNumber',
    301: 'TransferDunction',
    317: 'Predictor',
    318: 'WhitePoint',
    319: 'PrimaryChromaticities',
    321: 'HalftoneHints',
    322: 'TileWidth',
    323: 'TileLength',
    324: 'TileOffsets',
    325: 'TileByteCounts',
    326: 'BadFaxLines',
    327: 'CleanFaxData',
    328: 'ConsecutiveBadFaxLines',
    330: 'SubIFDs',
    332: 'InkSet',
    333: 'InkNames',
    334: 'NumberOfInks',
    336: 'DotRange',
    337: 'TargetPrinter',
    339: 'SampleFormat',
    340: 'SMinSampleValue',
    341: 'SMaxSampleValue',
    342: 'TransferRange',
    343: 'ClipPath',
    344: 'XClipPathUnits',
    345: 'YClipPathUnits',
    346: 'Indexed',
    347: 'JPEGTables',
    351: 'OPIProxy',
    400: 'GlobalParametersIFD',
    401: 'ProfileType',
    402: 'FaxProfile',
    403: 'CodingMethods',
    404: 'VersionYear',
    405: 'ModeNumber',
    433: 'Decode',
    434: 'DefaultImageColor',
    512: 'JPEGProc',
    513: 'JPEGInterchangeFormat',
    514: 'JPEGInterchangeFormatLength',
    515: 'JPEGRestartInterval',
    517: 'JPEGLosslessPredictors',
    518: 'JPEGPointTransforms',
    519: 'JPEGQTables',
    520: 'JPEGDCTables',
    521: 'JPEGACTables',
    529: 'YCbCrCoefficients',
    530: 'YCbCrSubSampling',
    531: 'YCbCrPositioning',
    532: 'ReferenceBlackWhite',
    559: 'StripRowCounts',
    700: 'XMP',
    32781: 'ImageID',
    34732: 'ImageLayer',
}
_GEOTIFF_TAGS = {
    33550: 'ModelPixelScaleTag',
    33922: 'ModelTiePointTag',
    34264: 'ModelTransformationTag',
    34735: 'GeoKeyDirectoryTag',
    34736: 'GeoDoubleParamsTag',
    34737: 'GeoAsciiParamsTag',
}


##########

class TiffDetails(object):
    """
    For checking tiff metadata, and parsing in the event we are not using GDAL
    """

    __slots__ = ('_file_name', '_endian', '_magic_number', '_tags')
    _DTYPES = {i+1: entry for i, entry in enumerate(
        ['u1', 'a', 'u2', 'u4', 'u4',
         'i1', 'u1', 'i2', 'i4', 'i4',
         'f4', 'f8', 'u4', None, None,
         'u8', 'i8', 'u8'])}
    _SIZES = numpy.array(
        [1, 1, 2, 4, 8,
         1, 1, 2, 4, 8,
         4, 8, 4, 0, 0,
         8, 8, 8], dtype=numpy.int64)
    # no definition for entries for 14 & 15

    def __init__(self, file_name: str):
        """

        Parameters
        ----------
        file_name : str
        """

        if not (isinstance(file_name, str) and os.path.isfile(file_name)):
            raise SarpyIOError('Not a TIFF file.')

        with open(file_name, 'rb') as fi:
            # Try to read the basic tiff header
            try:
                fi_endian = fi.read(2).decode('utf-8')
            except Exception as e:
                raise SarpyIOError('Failed decoding the 2 character tiff header with error\n\t{}'.format(e))

            if fi_endian == 'II':
                self._endian = '<'
            elif fi_endian == 'MM':
                self._endian = '>'
            else:
                raise SarpyIOError('Invalid tiff endian string {}'.format(fi_endian))
            # check the magic number
            self._magic_number = numpy.fromfile(fi, dtype='{}i2'.format(self._endian), count=1)[0]
            if self._magic_number not in [42, 43]:
                raise SarpyIOError('Not a valid tiff file, got magic number {}'.format(self._magic_number))

            if self._magic_number == 43:
                rhead = numpy.fromfile(fi, dtype='{}i2'.format(self._endian), count=2)
                if rhead[0] != 8:
                    raise SarpyIOError('Not a valid bigtiff. The offset size is given as {}'.format(rhead[0]))
                if rhead[1] != 0:
                    raise SarpyIOError('Not a valid bigtiff. The reserved entry of '
                                  'the header is given as {} != 0'.format(rhead[1]))
        self._file_name = file_name
        self._tags = None

    @property
    def file_name(self) -> str:
        """
        str: READ ONLY. The file name.
        """

        return self._file_name

    @property
    def endian(self) -> str:
        """
        str: READ ONLY. The numpy dtype style ``('>' = big, '<' = little)`` endian string for the tiff file.
        """

        return self._endian

    @property
    def tags(self) -> Dict[str, Union[str, numpy.ndarray]]:
        """
        Dict: READ ONLY. The tiff tags dictionary,
        provided that :meth:`parse_tags` has been called. This dictionary is
        of the form `{<tag name> : str|numpy.ndarray}`, even for those tags
        containing only a single entry (i.e. `count=1`).
        """

        if self._tags is None:
            self.parse_tags()
        return self._tags

    def parse_tags(self) -> None:
        """
        Parse the tags from the file, if desired. This sets the `tags` attribute.

        Returns
        -------
        None
        """

        if self._magic_number == 42:
            type_dtype = numpy.dtype('{}u2'.format(self._endian))
            count_dtype = numpy.dtype('{}u2'.format(self._endian))
            offset_dtype = numpy.dtype('{}u4'.format(self._endian))
            offset_size = 4
        elif self._magic_number == 43:
            type_dtype = numpy.dtype('{}u2'.format(self._endian))
            count_dtype = numpy.dtype('{}i8'.format(self._endian))
            offset_dtype = numpy.dtype('{}i8'.format(self._endian))
            offset_size = 8
        else:
            raise ValueError('Unrecognized magic number {}'.format(self._magic_number))

        with open(self._file_name, 'rb') as fi:
            # skip the basic header
            fi.seek(offset_size, os.SEEK_SET)
            # extract the tags information
            tags = {}
            self._parse_ifd(fi, tags, type_dtype, count_dtype, offset_dtype, offset_size)
        self._tags = tags

    def _read_tag(self,
                  fi: BinaryIO,
                  tiff_type: int,
                  num_tag: int,
                  count: int) -> Dict:
        """
        Parse the specific tag information.

        Parameters
        ----------
        fi
            The file type object
        tiff_type : int
            The tag data type identifier value
        num_tag : int
            The numeric tag identifier
        count : int
            The number of such tags

        Returns
        -------
        dict
        """

        # find which tags we belong to
        if num_tag in _BASELINE_TAGS:
            ext = 'BaselineTag'
            name = _BASELINE_TAGS[num_tag]
        elif num_tag in _EXTENSION_TAGS:
            ext = 'ExtensionTag'
            name = _EXTENSION_TAGS[num_tag]
        elif num_tag in _GEOTIFF_TAGS:
            ext = 'GeoTiffTag'
            name = _GEOTIFF_TAGS[num_tag]
        else:
            ext, name = None, None
        # Now extract from file based on type number
        dtype = self._DTYPES.get(int(tiff_type), None)
        if dtype is None:
            logger.warning(
                'Failed to extract tiff data type {},\n\t'
                'for {} - {}'.format(tiff_type, ext, name))
            return {'Value': None, 'Name': name, 'Extension': ext}
        if tiff_type == 2:  # ascii field - read directly and decode?
            val = fi.read(count)  # this will be a string for python 2, and we decode for python 3
            if not isinstance(val, str):
                val = val.decode('utf-8')
            # eliminate the null characters
            val = re.sub('\x00', '', val)
        elif tiff_type in [5, 10]:  # unsigned or signed rational
            val = numpy.fromfile(fi, dtype='{}{}'.format(self._endian, dtype), count=numpy.int64(2*count)).reshape((-1, 2))
        else:
            val = numpy.fromfile(fi, dtype='{}{}'.format(self._endian, dtype), count=count)
            if count == 1:
                val = val[0]
        return {'Value': val, 'Name': name, 'Extension': ext}

    def _parse_ifd(self,
                   fi: BinaryIO,
                   tags: dict,
                   type_dtype: Union[str, numpy.dtype],
                   count_dtype: Union[str, numpy.dtype],
                   offset_dtype: Union[str, numpy.dtype],
                   offset_size: int) -> None:
        """
        Recursively parses the tag data and populates a provided dictionary
        Parameters
        ----------
        fi
            The file type object
        tags : dict
            The tag data dictionary being populated
        type_dtype : str|numpy.dtype
            The data type for the data element - note that endian-ness is included
        count_dtype : str|numpy.dtype
            The data type for the number of directories - note that endian-ness is included
        offset_dtype : str|numpy.dtype
            The data type for the offset - note that endian-ness is included
        offset_size : int
            The size of the offset
        Returns
        -------
        None
        """

        nifd = numpy.fromfile(fi, dtype=offset_dtype, count=1)[0]
        if nifd == 0:
            return  # termination criterion

        fi.seek(nifd)
        num_entries = numpy.fromfile(fi, dtype=count_dtype, count=1)[0]
        for entry in range(int(num_entries)):
            num_tag, tiff_type = numpy.fromfile(fi, dtype=type_dtype, count=2)
            count = numpy.fromfile(fi, dtype=offset_dtype, count=1)[0]
            total_size = self._SIZES[tiff_type-1]*count
            if total_size <= offset_size:
                save_ptr = fi.tell() + offset_size  # we should advance past the entire block
                value = self._read_tag(fi, tiff_type, num_tag, count)
                fi.seek(save_ptr)
            else:
                offset = numpy.fromfile(fi, dtype=offset_dtype, count=1)[0]
                save_ptr = fi.tell()  # save our current spot
                fi.seek(offset)  # get to the offset location
                value = self._read_tag(fi, tiff_type, num_tag, count)  # read the tag value
                fi.seek(save_ptr)  # return to our location
            tags[value['Name']] = value['Value']
        self._parse_ifd(fi, tags, type_dtype, count_dtype, offset_dtype, offset_size)  # recurse

    def check_compression(self):
        """
        Check the Compression tag, and verify uncompressed.

        Returns
        -------
        None
        """

        if self.tags['Compression'] != 1:
            raise ValueError(
                'The file {} indicates some kind of tiff compression, and the sarpy API requirements '
                'do not presently support reading of compressed tiff files. Consider using gdal to '
                'translate this tiff to an uncompressed file via the commmand\n\t'
                '"gdal_translate -co TILED=no <input_file> <output_file>"')

    def check_tiled(self):
        """
        Check if the tiff file is tiled.

        Returns
        -------
        None
        """

        if 'TileLength' in self.tags or 'TileWidth' in self.tags:
            raise ValueError(
                'The file {} indicates that this is a tiled file, and the sarpy API requirements '
                'do not presently support reading of tiled tiff files. Consider using gdal to '
                'translate this tiff to a flat file via the commmand\n\t'
                '"gdal_translate -co TILED=no <input_file> <output_file>"')


class NativeTiffDataSegment(NumpyMemmapSegment):
    """
    Direct reading of data from tiff file, failing if compression is present.

    This is a very complex SAR specific implementation, and not general.
    """

    __slots__ = ('_tiff_details', )
    _SAMPLE_FORMATS = {
        1: 'u', 2: 'i', 3: 'f', 5: 'i', 6: 'f'}  # 5 and 6 are complex int/float

    def __init__(self,
                 tiff_details: Union[str, TiffDetails],
                 reverse_axes: Union[None, int, Sequence[int]] = None,
                 transpose_axes: Union[None, Tuple[int, ...]] = None):
        """
        If format function and format_dtype are not provided, then SAR specific
        (not necessarily general) choices will be made.

        Parameters
        ----------
        tiff_details : TiffDetails
        reverse_axes : None|Tuple[int, ...]
        transpose_axes : None|Tuple[int, ...]
        """

        if isinstance(tiff_details, str):
            tiff_details = TiffDetails(tiff_details)
        if not isinstance(tiff_details, TiffDetails):
            raise TypeError('NativeTiffChipper input argument must be a filename '
                            'or TiffDetails object.')

        tiff_details.check_compression()
        tiff_details.check_tiled()

        self._tiff_details = tiff_details
        if isinstance(tiff_details.tags['SampleFormat'], numpy.ndarray):
            samp_form = tiff_details.tags['SampleFormat'][0]
        else:
            samp_form = tiff_details.tags['SampleFormat']
        if samp_form not in self._SAMPLE_FORMATS:
            raise ValueError('Invalid sample format {}'.format(samp_form))
        if isinstance(tiff_details.tags['BitsPerSample'], numpy.ndarray):
            bits_per_sample = tiff_details.tags['BitsPerSample'][0]
        else:
            bits_per_sample = tiff_details.tags['BitsPerSample']

        raw_bands = int(tiff_details.tags['SamplesPerPixel'])

        if samp_form in [5, 6]:
            transform_data = 'COMPLEX'
            output_bands = int(raw_bands)
            raw_bands *= 2
            bits_per_sample /= 2
            output_dtype = 'complex64'
        elif raw_bands == 2:
            # NB: this is heavily skewed towards SAR and obviously not general
            transform_data = 'COMPLEX'
            output_dtype = 'complex64'
            output_bands = 1
        else:
            transform_data = None
            output_bands = raw_bands
            output_dtype = None

        raw_shape = (int(tiff_details.tags['ImageLength']), int(tiff_details.tags['ImageWidth']), raw_bands)
        raw_dtype = numpy.dtype('{0:s}{1:s}{2:d}'.format(
            self._tiff_details.endian, self._SAMPLE_FORMATS[samp_form], int(bits_per_sample/8)))
        if output_dtype is None:
            output_dtype = raw_dtype
        data_offset = int(tiff_details.tags['StripOffsets'][0])

        format_function = None
        if transform_data == 'COMPLEX':
            format_function = ComplexFormatFunction(raw_dtype, order='IQ')

        if reverse_axes is not None:
            if isinstance(reverse_axes, int):
                reverse_axes = (reverse_axes, )
            for entry in reverse_axes:
                if not entry < 2:
                    raise ValueError('reversing of axes on permitted along the first two axes.')

        if transpose_axes is not None:
            if len(transpose_axes) < 2 or len(transpose_axes) > 3:
                raise ValueError('transpose axes must have length 2 or 3')
            elif len(transpose_axes) == 2:
                transpose_axes = transpose_axes + (2, )

            if transpose_axes[2] != 2:
                raise ValueError(
                    'The transpose operation must preserve the location of the band data,\n\t'
                    'in the final dimension')

        if transpose_axes is None or transpose_axes == (0, 1, 2):
            output_shape = raw_shape[:2]
        else:
            output_shape = (raw_shape[1], raw_shape[0])

        if output_bands > 1:
            output_shape = output_shape + (output_bands, )

        NumpyMemmapSegment.__init__(
            self, tiff_details.file_name, data_offset, raw_dtype, raw_shape,
            formatted_dtype=output_dtype, formatted_shape=output_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes,
            format_function=format_function, mode='r', close_file=True)

    @property
    def tiff_details(self) -> TiffDetails:
        return self._tiff_details


class TiffReader(BaseReader):
    def __init__(self,
                 tiff_details: Union[str, TiffDetails],
                 reverse_axes: Union[None, int, Sequence[int]] = None,
                 transpose_axes: Union[None, Tuple[int, ...]] = None):
        """

        Parameters
        ----------
        tiff_details : TiffDetails
        reverse_axes : None|int|Sequence[int]
        transpose_axes : None|Tuple[int, ...]
        """

        data_segment = NativeTiffDataSegment(tiff_details, reverse_axes=reverse_axes, transpose_axes=transpose_axes)
        BaseReader.__init__(self, data_segment, reader_type='OTHER', close_segments=True)

    @property
    def data_segment(self) -> NativeTiffDataSegment:
        """
        NativeTiffDataSegment: The tiff data segment.
        """

        return self._data_segment

    @property
    def tiff_details(self) -> TiffDetails:
        """
        TiffDetails: The tiff details object.
        """

        return self.data_segment.tiff_details

    @property
    def file_name(self):
        return self.tiff_details.file_name

########
# base expected functionality for a module with an implemented Reader


def is_a(file_name: str) -> Union[None, TiffReader]:
    """
    Tests whether a given file_name corresponds to a tiff file. Returns a
    tiff reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    None|TiffReader
        `TiffReader` instance if tiff file, `None` otherwise
    """

    try:
        tiff_details = TiffDetails(file_name)
        logger.info('File {} is determined to be a tiff file.'.format(file_name))
        return TiffReader(tiff_details)
    except SarpyIOError:
        # we don't want to catch parsing errors, for now
        return None
