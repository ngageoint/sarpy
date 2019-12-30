# -*- coding: utf-8 -*-
"""
Module providing api consistent with other file types for reading tiff files.
"""

import sys
import numpy


try:
    from osgeo import gdal
    _USE_GDAL = True
except ImportError:
    gdal = None
    _USE_GDAL = False

from .base import BaseReader, BaseChipper
from .bip import BIPChipper


__classification__ = "UNCLASSIFIED"


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

# TODO:
#   1.) port tiff version in io.complex
#   2.) make similar bigtiff port
#   3.) make version using gdal


class TiffMetadataParser(object):
    __slots__ = ('_file_name', '_endian', '_magic_number', 'tags', '_TYPE_SIZES', '_TYPE_DTYPES')

    def __init__(self, file_name):
        with open(file_name, 'rb') as fi:
            # Try to read the basic tiff header
            fi_endian = fi.read(2).decode('utf-8')
            if fi_endian == 'II':
                self._endian = '<'
            elif fi_endian == 'MM':
                self._endian = '>'
            else:
                raise ValueError('Invalid tiff endian string {}'.format(fi_endian))
            # check the magic number
            self._magic_number = numpy.fromfile(fi, dtype='{}i2'.format(self._endian), count=1)[0]
            if self._magic_number not in [42, 43]:
                raise ValueError('Not a valid tiff file, got magic number {}'.format(self._magic_number))

            if self._magic_number == 43:
                rhead = numpy.fromfile(fi, dtype='{}i2'.format(self._endian), count=2)
                if rhead[0] != 8:
                    raise ValueError('Not a valid bigtiff. The offset size is given as {}'.format(rhead[0]))
                if rhead[1] != 0:
                    raise ValueError('Not a valid bigtiff. The reserved entry of '
                                     'the header is given as {} != 0'.format(rhead[1]))
        self._file_name = file_name
        self.tags = None

        self._TYPE_SIZES = numpy.array([1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 0, 0, 0, 8, 8, 8], dtype=numpy.uint64)
        self._TYPE_DTYPES = ['{}{}'.format(self._endian, entry) if entry is not None else None
                             for entry in ['u1', 'a', 'u2', 'u4', None,
                                           'i1', 'u1', 'i2', 'i4', None,
                                           'f4', 'f8', None, None, None,
                                           'u8', 'i8', 'u8']]
        # TODO: no definition for entries for 13-15?
        #   these last 3 entries (16-18) are for bigtiff only. no rational extension?

    def parse_tags(self):
        if self._magic_number == 42:
            type_dtype = '{}u2'.format(self._endian)
            offset_dtype = '{}u4'.format(self._endian)
            offset_size = 4
        elif self._magic_number == 43:
            type_dtype = '{}u2'.format(self._endian)
            offset_dtype = '{}u8'.format(self._endian)
            offset_size = 8
        else:
            raise ValueError('Unrecognized magic number {}'.format(self._magic_number))

        with open(self._file_name, 'rb') as fi:
            # skip the basic header
            fi.seek(offset_size)
            # extract the tags information
            tags = {}
            self._parse_ifd(fi, tags, type_dtype, offset_dtype, offset_size)

    def _read_tag(self, fi, tiff_type, num_tag, count):
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
        if tiff_type == 5:  # unsigned rational
            val = numpy.fromfile(fi, dtype='{}u4'.format(self._endian), count=2*count).reshape((-1, 2))
        elif tiff_type == 10:  # signed rational
            val = numpy.fromfile(fi, dtype='{}i4'.format(self._endian), count=2*count).reshape((-1, 2))
        elif tiff_type == 2:
            val = str(numpy.fromfile(fi, '{}a{}'.format(self._endian, count), count=1))
        else:
            dtype = self._TYPE_DTYPES[tiff_type]
            val = numpy.fromfile(fi, dtype=dtype, count=count)
        return {'Value': val, 'Name': name, 'Extension': ext}

    def _parse_ifd(self, fi, tags, type_dtype, offset_dtype, offset_size):
        nifd = numpy.fromfile(fi, dtype=offset_dtype, count=1)[0]
        if nifd == 0:
            return
        fi.seek(nifd)
        num_entries = numpy.fromfile(dtype=type_dtype, count=1)[0]
        for entry in range(int(num_entries)):
            num_tag, tiff_type = numpy.fromfile(fi, dtype=type_dtype, count=2)
            count = numpy.fromfile(fi, dtype=offset_dtype, count=1)[0]
            total_size = self._TYPE_SIZES[tiff_type-1]*count
            if total_size <= offset_size:
                value = self._read_tag(fi, tiff_type, num_tag, count)
            else:
                offset = numpy.fromfile(fi, dtype=offset_dtype, count=1)[0]
                save_ptr = fi.tell()  # save our current spot
                fi.seek(offset)  # got to the offset location
                value = self._read_tag(fi, tiff_type, num_tag, count)  # read the tag value
                fi.seek(save_ptr)  # return to our location
            tags[value['Name']] = value['Value']
        self._parse_ifd(fi, tags, type_dtype, offset_dtype, offset_size)  # recurse


class TiffChipper(BIPChipper):
    __slots__ = ('_tiff_meta', )

    def __init__(self, file_name, tiff_meta, symmetry=(False, False, True)):
        self._tiff_meta = tiff_meta
        # TODO: extract the below from the tiff meta-data, line 63 from complex/tiff.py
        data_type = ''
        data_size = ''
        complex_type = False
        data_offset = 0
        bands_ip = 1

        super(TiffChipper, self).__init__(
            file_name, data_type, data_size, symmetry=symmetry, complex_type=complex_type,
            data_offset=data_offset, bands_ip=bands_ip)


class GdalChipper(BaseChipper):
    pass
