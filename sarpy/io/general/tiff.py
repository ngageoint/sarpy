# -*- coding: utf-8 -*-
"""
Module providing api consistent with other file types for reading tiff files.
"""

import logging
import numpy
import warnings

from .base import BaseChipper, BaseReader, int_func
from .bip import BIPChipper

try:
    from osgeo import gdal
    _HAS_GDAL = True
except ImportError:
    warnings.warn("gdal is not successfully imported, which precludes reading tiffs via gdal")
    gdal = None
    _HAS_GDAL = False


__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Daniel Haverporth")


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

########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a tiff file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    TiffReader|None
        `TiffReader` instance if tiff file, `None` otherwise
    """

    try:
        tiff_details = TiffDetails(file_name)
        print('File {} is determined to be a tiff file.'.format(file_name))
        return TiffReader(tiff_details)
    except IOError:
        # we don't want to catch parsing errors, for now
        return None


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

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        with open(file_name, 'rb') as fi:
            # Try to read the basic tiff header
            try:
                fi_endian = fi.read(2).decode('utf-8')
            except:
                raise IOError('Not a TIFF file.')

            if fi_endian == 'II':
                self._endian = '<'
            elif fi_endian == 'MM':
                self._endian = '>'
            else:
                raise IOError('Invalid tiff endian string {}'.format(fi_endian))
            # check the magic number
            self._magic_number = numpy.fromfile(fi, dtype='{}i2'.format(self._endian), count=1)[0]
            if self._magic_number not in [42, 43]:
                raise IOError('Not a valid tiff file, got magic number {}'.format(self._magic_number))

            if self._magic_number == 43:
                rhead = numpy.fromfile(fi, dtype='{}i2'.format(self._endian), count=2)
                if rhead[0] != 8:
                    raise IOError('Not a valid bigtiff. The offset size is given as {}'.format(rhead[0]))
                if rhead[1] != 0:
                    raise IOError('Not a valid bigtiff. The reserved entry of '
                                  'the header is given as {} != 0'.format(rhead[1]))
        self._file_name = file_name
        self._tags = None

    @property
    def file_name(self):
        """
        str: READ ONLY. The file name.
        """

        return self._file_name

    @property
    def endian(self):
        """
        str: READ ONLY. The numpy dtype style ``('>' = big, '<' = little)`` endian string for the tiff file.
        """

        return self._endian

    @property
    def tags(self):
        """
        None|dict: READ ONLY. the tiff tags dictionary, provided that func:`parse_tags` has been called.
        This dictionary is of the form ``{<tag name> : numpy.ndarray value}``, even for
        those tags containing only a single entry (i.e. `count=1`).
        """
        if self._tags is None:
            self.parse_tags()
        return self._tags

    def parse_tags(self):
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
            fi.seek(offset_size)
            # extract the tags information
            tags = {}
            self._parse_ifd(fi, tags, type_dtype, count_dtype, offset_dtype, offset_size)
        self._tags = tags

    def _read_tag(self, fi, tiff_type, num_tag, count):
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
            logging.warning('Failed to extract tiff data type {}, for {} - {}'.format(tiff_type, ext, name))
            return {'Value': None, 'Name': name, 'Extension': ext}
        if tiff_type == 2:  # ascii field
            val = str(numpy.fromfile(fi, '{}{}{}'.format(self._endian, dtype, count), count=1))
        elif tiff_type in [5, 10]:  # unsigned or signed rational
            val = numpy.fromfile(fi, dtype='{}{}'.format(self._endian, dtype), count=numpy.int64(2*count)).reshape((-1, 2))
        else:
            val = numpy.fromfile(fi, dtype='{}{}'.format(self._endian, dtype), count=count)
        return {'Value': val, 'Name': name, 'Extension': ext}

    def _parse_ifd(self, fi, tags, type_dtype, count_dtype, offset_dtype, offset_size):
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


class NativeTiffChipper(BIPChipper):
    """
    Direct reading of data from tiff file, failing if compression is present
    """

    __slots__ = ('_tiff_details', )
    _SAMPLE_FORMATS = {
        1: 'u', 2: 'i', 3: 'f', 5: 'i', 6: 'f'}  # 5 and 6 are complex int/float

    def __init__(self, tiff_details, symmetry=(False, False, True)):
        """

        Parameters
        ----------
        tiff_details : TiffDetails
        symmetry : Tuple[bool]
        """

        if isinstance(tiff_details, str):
            tiff_details = TiffDetails(tiff_details)
        if not isinstance(tiff_details, TiffDetails):
            raise TypeError('NativeTiffChipper input argument must be a filename '
                            'or TiffDetails object.')

        compression_tag = int(tiff_details.tags['Compression'][0])
        if compression_tag != 1:
            raise ValueError('Tiff has compression tag {}, but only 1 (no compression) '
                             'is supported.'.format(compression_tag))

        self._tiff_details = tiff_details
        samp_form = tiff_details.tags['SampleFormat'][0]
        if samp_form not in self._SAMPLE_FORMATS:
            raise ValueError('Invalid sample format {}'.format(samp_form))
        bits_per_sample = tiff_details.tags['BitsPerSample'][0]
        complex_type = (int(tiff_details.tags['SamplesPerPixel'][0]) == 2)  # NB: this is obviously not general
        if samp_form in [5, 6]:
            bits_per_sample /= 2
            complex_type = True
        data_size = (int_func(tiff_details.tags['ImageLength'][0]), int_func(tiff_details.tags['ImageWidth'][0]))
        data_type = numpy.dtype('{0:s}{1:s}{2:d}'.format(self._tiff_details.endian,
                                                         self._SAMPLE_FORMATS[samp_form],
                                                         int(bits_per_sample/8)))
        data_offset = int_func(tiff_details.tags['StripOffsets'][0])

        super(NativeTiffChipper, self).__init__(
            tiff_details.file_name, data_type, data_size, symmetry=symmetry, complex_type=complex_type,
            data_offset=data_offset, bands_ip=1)


class GdalTiffChipper(BaseChipper):
    """
    Utilizing gdal for reading of data from tiff file, should be much more robust
    """
    # TODO: this is a work in progress and not quite functional
    __slots__ = ('_tiff_details', '_data_set', '_bands', '_virt_array')

    def __init__(self, tiff_details, symmetry=(False, False, True)):
        """

        Parameters
        ----------
        tiff_details : TiffDetails
        symmetry : Tuple[bool]
        """
        if isinstance(tiff_details, str):
            tiff_details = TiffDetails(tiff_details)
        if not isinstance(tiff_details, TiffDetails):
            raise TypeError('GdalTiffChipper input argument must be a filename '
                            'or TiffDetails object.')

        self._tiff_details = tiff_details
        # initialize our dataset - NB: this should close gracefully on garbage collection
        self._data_set = gdal.Open(tiff_details.file_name, gdal.GA_ReadOnly)
        if self._data_set is None:
            raise ValueError(
                'GDAL failed with unspecified error in opening file {}'.format(tiff_details.file_name))
        # get data_size information
        data_size = (self._data_set.RasterYSize, self._data_set.RasterXSize)
        self._bands = self._data_set.RasterCount
        # TODO: get data_type information - specifically how does complex really work?
        complex_type = ''
        super(GdalTiffChipper, self).__init__(data_size, symmetry=symmetry, complex_type=complex_type)
        # 5.) set up our virtual array using GetVirtualMemArray
        try:
            self._virt_array = self._data_set.GetVirtualMemArray()
        except Exception:
            logging.error(
                msg="There has been some error using the gdal method GetVirtualMemArray(). "
                    "Consider falling back to the base sarpy tiff reader implementation (use_gdal=False)")
            raise
        # TODO: this does not generally work should we clunkily fall back to dataset.band.ReadAsArray()?
        #   This doesn't support slicing...

    def _read_raw_fun(self, range1, range2):
        arange1, arange2 = self._reorder_arguments(range1, range2)
        if self._bands == 1:
            out = self._virt_array[arange1[0]:arange1[1]:arange1[2], arange2[0]:arange2[1]:arange2[2]]
        elif self._data_set.band_sequential:
            # push the bands to the end
            out = (self._virt_array[:, arange1[0]:arange1[1]:arange1[2], arange2[0]:arange2[1]:arange2[2]]).transpose((2, 0, 1))
        else:
            # push the bands to the end
            out = self._virt_array[arange1[0]:arange1[1]:arange1[2], arange2[0]:arange2[1]:arange2[2], :]
        return out


class TiffReader(BaseReader):
    __slots__ = ('_tiff_details', '_sicd_meta', '_chipper')
    _DEFAULT_SYMMETRY = (False, False, False)

    def __init__(self, tiff_details, sicd_meta=None, symmetry=None, use_gdal=False):
        """

        Parameters
        ----------
        tiff_details : TiffDetails
        sicd_meta : None|sarpy.io.complex.sicd_elements.SICD.SICDType
        symmetry : Tuple[bool]
        use_gdal : bool
            Should we use gdal to read the tiff (required if compressed)
        """

        if isinstance(tiff_details, str):
            tiff_details = TiffDetails(tiff_details)
        if not isinstance(tiff_details, TiffDetails):
            raise TypeError('TiffReader input argument must be a filename '
                            'or TiffDetails object.')

        self._tiff_details = tiff_details
        if symmetry is None:
            symmetry = self._DEFAULT_SYMMETRY

        if use_gdal:
            # TODO: finish this capability
            logging.warning(
                msg="The option use_gdal=True, but this functionality is a work in progress. "
                    "Falling back to the sarpy base version.")
            use_gdal = False

        if use_gdal and not _HAS_GDAL:
            logging.warning(
                msg="The option use_gdal=True, but there does not appear to be "
                    "a functional gdal installed. Falling back to the sarpy base version.")
            use_gdal = False
        if use_gdal:
            chipper = GdalTiffChipper(tiff_details, symmetry=symmetry)
        else:
            chipper = NativeTiffChipper(tiff_details, symmetry=symmetry)
        super(TiffReader, self).__init__(sicd_meta, chipper)

    @property
    def tiff_details(self):
        # type: () -> TiffDetails
        """
        TiffDetails: The tiff details object.
        """

        return self._tiff_details

    @property
    def file_name(self):
        return self.tiff_details.file_name
