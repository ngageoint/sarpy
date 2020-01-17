# SarPy imports
from .sicd import MetaNode
from .utils import bip
from . import Reader as ReaderSuper  # Reader superclass
# Python standard library imports
import sys
# External dependencies
import numpy as np

__classification__ = "UNCLASSIFIED"
__author__ = "Daniel Haverporth"
__email__ = "Daniel.L.Haverporth@nga.mil"

# TODO: Add way to find associated metadata files for sensor-specific formats that use TIFF


def isa(filename):
    """Test to see if file is a TIFF.  If so, return reader object."""
    with open(filename, mode='rb') as fid:
        # identify endian
        try:
            endianflag = fid.read(2).decode('ascii')
            if endianflag == 'II':
                endian = '<'
                magicNumber = np.fromfile(fid, dtype=endian + 'i2', count=1)[0]
            elif endianflag == 'MM':
                endian = '>'
                magicNumber = np.fromfile(fid, dtype=endian + 'i2', count=1)[0]
        except UnicodeDecodeError:  # Data might not have been valid ASCII
            pass
        if 'magicNumber' in locals() and magicNumber == 42:
            return Reader


class Reader(ReaderSuper):
    """Creates a file reader object for a TIFF file."""

    def __init__(self, filename):
        tiffmeta = read_meta(filename)
        DEFAULT_SYMMETRY = (False, False, True)  # True for all Sentinel-1 and some (not all) RS2
        self.read_chip = chipper(filename, DEFAULT_SYMMETRY, tiffmeta=tiffmeta)

        # Populate a minimal SICD metadata since we know nothing about radar parameters (if this is
        # even a radar dataset)
        self.sicdmeta = MetaNode()
        self.sicdmeta.ImageData = MetaNode()
        if DEFAULT_SYMMETRY[2]:
            self.sicdmeta.ImageData.NumCols = tiffmeta['ImageLength'][0]
            self.sicdmeta.ImageData.NumRows = tiffmeta['ImageWidth'][0]
        else:
            self.sicdmeta.ImageData.NumCols = tiffmeta['ImageWidth'][0]
            self.sicdmeta.ImageData.NumRows = tiffmeta['ImageLength'][0]
        self.sicdmeta.native = MetaNode()
        self.sicdmeta.native.tiff = tiffmeta


def chipper(filename, symmetry=(False, False, True), tiffmeta=None):
    """Separates the creation of the chipper function, so that other formats that use TIFF can
    reuse this code."""

    if tiffmeta is None:  # Passing this allows one to avoid reparsing TIFF header
        tiffmeta = read_meta(filename)
    SAMPLE_TYPES = (None, 'uint', 'int', 'float', None, 'int', 'float')
    bits_per_sample = tiffmeta['BitsPerSample'][0]
    if (tiffmeta['SampleFormat'][0] > len(SAMPLE_TYPES) or
       SAMPLE_TYPES[tiffmeta['SampleFormat'][0]] is None):
        raise(ValueError('Invalid pixel type.'))
    if tiffmeta['SampleFormat'][0] in [5, 6]:
        bits_per_sample /= 2
    datatype = np.dtype(SAMPLE_TYPES[tiffmeta['SampleFormat'][0]] +
                        str(int(bits_per_sample)))
    complextype = (tiffmeta['SampleFormat'][0] in [5, 6] or
                   int(tiffmeta['SamplesPerPixel'][0]) == 2)  # int makes bool instead of bool_
    datasize = np.array([tiffmeta['ImageLength'][0], tiffmeta['ImageWidth'][0]])
    swapbytes = ((sys.byteorder == 'big') != (tiffmeta['endian'] == '>'))
    data_offset = tiffmeta['StripOffsets'][0]
    return bip.Chipper(filename, datasize, datatype, complextype,
                       data_offset, swapbytes, symmetry, bands_ip=1)


def read_meta(filename):
    """Read metadata from TIFF file."""

    # We have to open as binary, since there is some binary data in the file.
    # Python doesn't seem to let us read just part of the file as utf-8.
    with open(filename, mode='rb') as fid:
        # Read TIFF file header
        endian = fid.read(2).decode('ascii')
        if endian == 'II':
            endian = '<'
        elif endian == 'MM':
            endian = '>'
        else:
            raise(ValueError('Invalid endian type.'))
        fid.seek(2)  # Offset to first field of interest
        magicNumber = np.fromfile(fid, dtype=endian + 'i2', count=1)[0]  # should be 42
        if magicNumber != 42:
            raise(ValueError('Not a valid TIFF file. Magic number does not match.'))
        # Pointer to Next Image Segment
        nextIFD = np.fromfile(fid, dtype=endian + 'u4', count=1)[0]
        while nextIFD > 0:
            fid.seek(nextIFD)
            tags = read_ifd(fid, endian)
            nextIFD = tags['next_ifd']
        tags['endian'] = endian
        return tags


def read_ifd(fid, endian):
    # Define parameters of TIFF File
    TYPE_SIZE = np.array([1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8])
    num_ifd_entries = np.fromfile(fid, dtype=endian + 'u2', count=1)[0]
    Tags = {}
    for entry_i in range(num_ifd_entries):
        tag_numeric = np.fromfile(fid, dtype=endian + 'u2', count=1)[0]
        tiff_type = np.fromfile(fid, dtype=endian + 'u2', count=1)[0]
        count = np.fromfile(fid, dtype=endian + 'u4', count=1)[0]
        if tiff_type > 0 and tiff_type <= TYPE_SIZE.size:
            total_size = TYPE_SIZE[tiff_type-1]*count
            if total_size <= 4:
                value = readTiffTag(fid, tiff_type, tag_numeric, count, endian)
                fid.seek(4-total_size, 1)
            else:
                offset = np.fromfile(fid, dtype=endian + 'u4', count=1)[0]
                save_ptr = fid.tell()
                fid.seek(offset)
                value = readTiffTag(fid, tiff_type, tag_numeric, count, endian)
                fid.seek(save_ptr)
            Tags[value['Name']] = value['Value']
    next_ifd = np.fromfile(fid, dtype=endian + 'u4', count=1)[0]
    Tags['next_ifd'] = next_ifd
    return Tags


def readTiffTag(fid, tiff_type, type_numeric, count, endian):
    switcher = {  # Baseline Tag Values
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
    switcher2 = {  # Extension Tags
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
        34732: 'ImageLayer'
        }
    switcher3 = {  # GeoTiff Tags
        33550: 'ModelPixelScaleTag',
        33922: 'ModelTiePointTag',
        34264: 'ModelTransformationTag',
        34735: 'GeoKeyDirectoryTag',
        34736: 'GeoDoubleParamsTag',
        34737: 'GeoAsciiParamsTag',
        }

    # Determine Tag Name
    BaselineTag = switcher.get(type_numeric, 0)
    ExtensionTag = switcher2.get(type_numeric, 0)
    GeoTiffTag = switcher3.get(type_numeric, 0)
    if BaselineTag != 0:
        Extension = 'BaselineTag'
        Name = BaselineTag
    elif ExtensionTag != 0:
        Extension = 'ExtensionTag'
        Name = ExtensionTag
    elif GeoTiffTag != 0:
        Extension = 'GeoTiffTag'
        Name = GeoTiffTag
    else:
        Extension = 'Unidentified'
        Name = 'Unknown'

    # Now extract from file based on type number
    if tiff_type == 1:
        Value = np.fromfile(fid, dtype=endian + 'u1', count=count)
    elif tiff_type == 2:
        Value = np.fromfile(fid, dtype=endian + 'a' + str(count), count=1)
    elif tiff_type == 3:
        Value = np.fromfile(fid, dtype=endian + 'u2', count=count)
    elif tiff_type == 4:
        Value = np.fromfile(fid, dtype=endian + 'u4', count=count)
    elif tiff_type == 6:
        Value = np.fromfile(fid, dtype=endian + 'i1', count=count)
    elif tiff_type == 7:
        Value = np.fromfile(fid, dtype=endian + 'u1', count=count)
    elif tiff_type == 8:
        Value = np.fromfile(fid, dtype=endian + 'i2', count=count)
    elif tiff_type == 9:
        Value = np.fromfile(fid, dtype=endian + 'i4', count=count)
    elif tiff_type == 11:
        Value = np.fromfile(fid, dtype=endian + 'f4', count=count)
    elif tiff_type == 12:
        Value = np.fromfile(fid, dtype=endian + 'f8', count=count)
    else:
        # Most notably, this code won't handle the XResolution and
        # YResolution tags, which are of type RATIONAL, but are
        # required is many TIFFs.
        fid.seek(4)
        Value = None

    return {'Value': Value, 'Name': Name, 'Extension': Extension}
