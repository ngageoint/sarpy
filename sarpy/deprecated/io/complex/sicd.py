"""Module for reading SICD files (version 0.3 and above)."""

# SarPy imports
from . import Reader as ReaderSuper  # Reader superclass
from . import Writer as WriterSuper  # Writer superclass
from .utils import bip
from .utils import chipper
from sarpy.geometry import geocoords as gc, latlon as ll, point_projection as point
# Python standard library imports
import copy
from datetime import datetime
import os
import re
import sys
import xml.etree.ElementTree as ET
# External dependencies
import numpy as np
from numpy.polynomial import polynomial as poly

__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"
__email__ = "Wade.C.Schwartzkopf.ctr@nga.mil"


def isa(filename):
    """Test to see if file is a SICD.  If so, return reader object."""
    try:
        # A non-NITF file will probably result in an exception in read_nitf_offsets.
        # A non-SICD NITF will probably result in an exception in ParseString.
        nitf = read_nitf_offsets(filename)
        with open(filename, mode='rb') as fid:
            fid.seek(nitf['des_offsets'][0])
            root_node = ET.fromstring(fid.read(nitf['des_lengths'][0]))
        if root_node.tag.split('}', 1)[-1] == 'SICD':
            return Reader
    except:
        pass  # Not a SICD, but that's OK


class Reader(ReaderSuper):
    """Creates a file reader object for a SICD file."""

    schema_info = None  # Class variable.  Should only have to be populated once for all instances

    def __init__(self, filename):
        schema_filename = os.path.join(os.path.dirname(__file__),
                                       'SICD_schema_V1.1.0_2014_09_30.xsd')  # Most current schema
        # Schema the same for all SICDs.  Only parse once for first instance
        # and then keep around for all future instances.
        if (os.path.exists(schema_filename) and
           os.path.isfile(schema_filename) and
           (Reader.schema_info is None)):
            Reader.schema_info = parse_schema(schema_filename)

        self.sicdmeta, nitfmeta = read_meta(filename, Reader.schema_info)
        data_offset = nitfmeta['img_segment_offsets']
        datasize = np.column_stack((nitfmeta['img_segment_rows'],
                                    nitfmeta['img_segment_columns']))
        if self.sicdmeta.ImageData.PixelType == 'RE32F_IM32F':
            datatype = np.dtype('float32')
        elif self.sicdmeta.ImageData.PixelType == 'RE16I_IM16I':
            datatype = np.dtype('int16')
        elif self.sicdmeta.ImageData.PixelType == 'AMP8I_PHS8I':
            raise(ValueError('AMP8I_PHS8I is currently an unsupported pixel type.'))
        else:
            raise(ValueError('Invalid pixel type.'))
        complextype = True
        swapbytes = (sys.byteorder != 'big')  # All SICDs are big-endian
        symmetry = (False, False, False)
        self.read_chip = self.multisegment(filename, datasize, datatype,
                                           complextype, data_offset, swapbytes,
                                           symmetry, bands_ip=1)

    class multisegment(chipper.Base):
        """Chipper function for SICDs with multiple image segments."""
        def __init__(self, filename, datasize, datatype, complextype,  # Required params
                     data_offset=0,  # Start of data in bytes from start of file
                     swapbytes=False,  # Is reading endian same as file endian
                     symmetry=(False, False, False),  # Assume no reorientation
                     bands_ip=1):  # This means bands of complex data (if data is complex)
            if datasize.shape[0] != data_offset.size:
                raise(ValueError('DATASIZE and DATA_OFFSET must have matching sizes.'))
            # Complex type set to False here, since conversion to complex will
            # be done in individual chippers for each image segment.
            self.complextype = False
            self.symmetry = symmetry
            # Build individual chippers here
            self.chippers = []
            for i in range(data_offset.size):
                self.chippers.append(bip.Chipper(filename, datasize[i],
                                                 datatype, complextype,
                                                 data_offset[i], swapbytes,
                                                 symmetry, bands_ip))
            self.rowends = datasize[:, 0].cumsum()
            # Doesn't work on older version of NumPy due to an unsafe cast
            # self.rowstarts = np.insert(self.rowends[:-1], 0, 0)
            # This should work in all versions of numpy:
            self.rowstarts = np.hstack((np.uint32(0), self.rowends[:-1]))
            self.datasize = [self.rowends[-1], datasize[0, 1]]
            self.read_raw_fun = lambda dim1range, dim2range: \
                self.combined_chipper(dim1range, dim2range)

        def combined_chipper(self, dim1range=None, dim2range=None):
            """A unified chipper that calls chippers for each image segment and
            returns outputs as if it were a single contiguous dataset"""
            datasize, dim1range, dim2range = chipper.check_args(
                self.datasize, dim1range, dim2range)
            dim1ind = np.array(range(*dim1range))
            output = np.zeros((len(dim1ind), len(range(*dim2range))),
                              dtype=np.complex64)
            # Decide which image segments have request data
            for i in range(self.rowstarts.size):
                dim1blockvalid = ((dim1ind < self.rowends[i]) &
                                  (dim1ind >= self.rowstarts[i]))
                # Extract data from relevent image segments
                if any(dim1blockvalid):
                    blockdim1range = [min(dim1ind[dim1blockvalid]),
                                      max(dim1ind[dim1blockvalid]) + 1,
                                      dim1range[2]]
                    blockdim1range[:2] -= self.rowstarts[i]
                    output[np.where(dim1blockvalid), :] = \
                        self.chippers[i](blockdim1range, dim2range)
            return output


class Writer(WriterSuper):
    """Creates a file writer object for a SICD file."""

    # Class variable.  Should only have to be populated once for all instances
    schema_info = None

    # Class constants
    ISSIZEMAX = 9999999998  # Image segment size maximum
    ILOCMAX = 99999  # Largest value we can put in image location field
    IS_SUBHEADER_LENGTH = 512  # Fixed for two bands image segments
    # DES_HEADER_LENGTH = 200  # Harded-coded from SICD spec (0.5 and before)
    DES_HEADER_LENGTH = 973  # Harded-coded from SICD spec (1.0)

    def __init__(self, filename, sicdmeta):
        schema_filename = os.path.join(os.path.dirname(__file__),
                                       'SICD_schema_V1.1.0_2014_09_30.xsd')  # Most current schema
        # Schema the same for all SICDs.  Only parse once for first instance
        # and then keep around for all future instances.
        if (os.path.exists(schema_filename) and
           os.path.isfile(schema_filename) and
           (Writer.schema_info is None)):
            Writer.schema_info = parse_schema(schema_filename)

        # Compute image segment parameters
        self.filename = filename
        self.sicdmeta = sicdmeta
        if (hasattr(sicdmeta, 'ImageData') and
           hasattr(sicdmeta.ImageData, 'PixelType')):
            if sicdmeta.ImageData.PixelType == 'RE32F_IM32F':
                bytes_per_pixel = 8
                datatype = np.dtype('>f4')
            elif sicdmeta.ImageData.PixelType == 'RE16I_IM16I':
                bytes_per_pixel = 4
                datatype = np.dtype('>i2')
            elif sicdmeta.ImageData.PixelType == 'AMP8I_PHS8I':
                bytes_per_pixel = 2
                datatype = np.dtype('>u1')
                raise(ValueError('AMP8I_PHS8I is currently an unsupported pixel type.'))
            else:
                raise(ValueError('PixelType must be RE32F_IM32F, RE16I_IM16I, or AMP8I_PHS8I.'))
        else:
            sicdmeta.ImageData.PixelType = 'RE32F_IM32F'
            bytes_per_pixel = 8
            datatype = np.dtype('>f4')
        self.bytes_per_row = int(sicdmeta.ImageData.NumCols) * bytes_per_pixel
        num_rows_limit = min(int(np.floor(self.ISSIZEMAX / float(self.bytes_per_row))),
                             self.ILOCMAX)
        # Number of image segments
        self.num_is = int(np.ceil(float(sicdmeta.ImageData.NumRows)/num_rows_limit))
        # Row index of the first row in each segment
        self.first_row_is = np.arange(self.num_is) * num_rows_limit
        self.num_rows_is = np.empty_like(self.first_row_is)
        self.num_rows_is[:-1] = num_rows_limit  # Number of rows in each segment
        self.num_rows_is[-1] = (sicdmeta.ImageData.NumRows -
                                ((self.num_is - 1) * num_rows_limit))
        # Compute DES parameters
        self.nitf_header_length = 401 + (16 * self.num_is)
        self.des_data = struct2xml(sicdmeta, self.schema_info, inc_newline=True)
        # Open the file and write the NITF file header data
        with open(filename, mode='wb') as self.fid:
            self._write_fileheader()
        # Setup image segment writers that will be used by write_chip for writing pixels
        self.writer_is = []
        for is_count in range(self.num_is):
            is_size = (self.num_rows_is[is_count], sicdmeta.ImageData.NumCols)
            self.writer_is.append(bip.Writer(filename, is_size, datatype, True,
                                             self.nitf_header_length +
                                             (self.IS_SUBHEADER_LENGTH * (is_count + 1)) +
                                             (int(sum(self.num_rows_is[0:is_count])) *
                                              self.bytes_per_row)))

    def __del__(self):
        # Write image subheaders upon closing.  We don't do this during
        # __init__, since if the image is not written yet, jumping to any
        # image subheader beyond the first will result in gigabytes of
        # file being created, which could cause unecessary delay.
        # Another (perhaps better) option would be to write each header
        # the first time any pixel data is written to a segment.
        with open(self.filename, mode='r+b') as self.fid:
            pos = self.nitf_header_length
            for i in range(self.num_is):
                self.fid.seek(pos)
                self._write_imsubhdr(i)
                pos = (pos + self.IS_SUBHEADER_LENGTH +
                       (int(self.num_rows_is[i]) * self.bytes_per_row))
            # Write DES
            self.fid.seek(pos)  # Seek to end of image data
            self._write_dessubhdr()
            self.fid.write(self.des_data)

    # All of these subfunctions for writing NITF component subheaders are a lot of lines
    # of code just to generate a bunch of fields in the file, most of which will likely
    # never be read or used in any way, since SICD stores all of its metadata in the XML.
    def _write_security_tags(self):
        """Writes the NITF security tags at the current file pointer."""
        if (hasattr(self.sicdmeta, 'CollectionInfo') and
           hasattr(self.sicdmeta.CollectionInfo, 'Classification')):
            classification = self.sicdmeta.CollectionInfo.Classification
        else:
            classification = ' '
        code = re.search('(?<=/)[^/].*', classification)
        if code is not None:
            code = code.group()
        else:
            code = ''
        self.fid.write(classification[0].encode())  # CLAS
        self.fid.write(b'US')  # CLSY
        self.fid.write(('%-11.11s' % code).encode())  # 11 spaces reserverd for classification code
        self.fid.write(b'  ')  # 2 spaces for CLTH
        self.fid.write(b' '*20)  # 20 spaces reserverd for REL
        self.fid.write(b'  ')  # 2 spaces reserved for DCTP
        self.fid.write(b' '*8)  # 8 spaces reserved for DCDT
        self.fid.write(b' '*4)  # 4 spaces reserver for DCXM
        self.fid.write(b' ')  # 1 space reserved for DG
        self.fid.write(b' '*8)  # 8 spaces reserverd for DGDT
        self.fid.write(b' '*43)  # 43 spaces reserved for CLTX
        self.fid.write(b' ')  # 1 space reserved for CATP
        self.fid.write(b' '*40)  # 40 spaces reserverd for CAUT
        self.fid.write(b' ')  # 1 for CRSN
        self.fid.write(b' '*8)  # 8 for SRDT
        self.fid.write(b' '*15)  # 15 for CLTN

    def _write_fileheader(self):
        self.fid.write(b'NITF02.10')
        image_data_size = int(sum(self.num_rows_is)) * self.bytes_per_row
        if image_data_size < 50*(1024*1024):  # less than 50 MB
            complexity = b'03'
        elif image_data_size < 1024**3:  # less than 1 GB
            complexity = b'05'
        elif image_data_size < 2*1024**3:  # less than 2 GB
            complexity = b'06'
        else:
            complexity = b'07'
        self.fid.write(complexity)
        self.fid.write(b'BF01')  # SType
        self.fid.write(b'Unknown   ')  # OSTAID (not supposed to be blank)
        try:  # May not have ImageCreation.DateTime field or it may be misformed
            # Creation time of original image
            fdt = datetime.strftime(self.sicdmeta.ImageCreation.DateTime, '%Y%m%d%H%M%S')
        except:
            # Creation time of this NITF
            fdt = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        self.fid.write(fdt.encode())  # creation time
        # FTITLE.  The 'SICD:' prefix used to be required, but is not any longer.
        # We keep it just in case older tools expect it.
        if (hasattr(self.sicdmeta, 'CollectionInfo') and
           hasattr(self.sicdmeta.CollectionInfo, 'CoreName')):
            self.fid.write(('SICD: %-74.74s' % self.sicdmeta.CollectionInfo.CoreName).encode())
        else:
            self.fid.write('SICD: '.ljust(80).encode())
        self._write_security_tags()
        self.fid.write(b'00000')  # FSCOPY
        self.fid.write(b'00000')  # FSCPYS
        self.fid.write(b'0')  # Encryption 0 = no encrpytion
        temp = np.array(0, dtype='uint8')
        temp.tofile(self.fid)  # FBKGC
        temp.tofile(self.fid)  # Backgournd Color
        temp.tofile(self.fid)  # red, green, blue
        self.fid.write(b' '*24)  # 24 spaces reserved for originator name
        self.fid.write(b' '*18)  # 18 spaces reservers for orginator phone
        fileLength = int(self.nitf_header_length + (self.IS_SUBHEADER_LENGTH * self.num_is) +
                         image_data_size + self.DES_HEADER_LENGTH + len(self.des_data))
        self.fid.write(('%012d' % fileLength).encode())
        self.fid.write(('%06d' % self.nitf_header_length).encode())
        # Image Segment Description
        self.fid.write(('%03d' % self.num_is).encode())
        for i in range(self.num_is):
            self.fid.write(('%06d' % self.IS_SUBHEADER_LENGTH).encode())
            self.fid.write(('%010d' % (int(self.num_rows_is[i])*self.bytes_per_row)).encode())
        # Graphic Segments not allowed in SICD
        self.fid.write(b'000')
        # Reserved Extensiion Segments Not allowed in SICD
        self.fid.write(b'000')
        #  Text Segments Not Generally Used in SICD
        self.fid.write(b'000')
        #  DES Segment
        self.fid.write(b'001')
        self.fid.write(('%04d' % self.DES_HEADER_LENGTH).encode())
        self.fid.write(('%09d' % len(self.des_data)).encode())
        # Reserved Extension Segment Not Generally used in SICD
        self.fid.write(b'000')
        # User Defined Header not generally used in SICD
        self.fid.write(b'00000')
        # Extended Headers not generally used in SICD
        self.fid.write(b'00000')

    def _write_imsubhdr(self, im_seg_number=0):
        self.fid.write(b'IM')
        self.fid.write(('SICD%03d   ' % im_seg_number).encode())
        try:  # May not have ImageCreation.DateTime field or it may be misformed
            # Creation time of original image
            fdt = datetime.strftime(self.sicdmeta.ImageCreation.DateTime, '%Y%m%d%H%M%S')
        except:
            # Creation time of this NITF
            fdt = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        self.fid.write(fdt.encode())  # creation time
        self.fid.write(b' '*17)  # TGTID
        # IID2.  The 'SICD:' prefix used to be required, but is not any longer.
        # We keep it just in case older tools expect it.
        if (hasattr(self.sicdmeta, 'CollectionInfo') and
           hasattr(self.sicdmeta.CollectionInfo, 'CoreName')):
            self.fid.write(('SICD: %-74.74s' % self.sicdmeta.CollectionInfo.CoreName).encode())
        else:
            self.fid.write('SICD: '.ljust(80).encode())
        self._write_security_tags()
        self.fid.write(b'0')
        if (hasattr(self.sicdmeta, 'CollectionInfo') and
           hasattr(self.sicdmeta.CollectionInfo, 'CollectorName')):
            self.fid.write(('SICD: %-36.36s' %
                            self.sicdmeta.CollectionInfo.CollectorName).encode())
        else:
            self.fid.write('SICD: '.ljust(42).encode())
        self.fid.write(('%08d' % self.num_rows_is[im_seg_number]).encode())
        self.fid.write(('%08d' % self.sicdmeta.ImageData.NumCols).encode())
        if self.sicdmeta.ImageData.PixelType == 'RE32F_IM32F':
            pvtype = 'R'
            abpp = 32
            isubcat1 = 'I'
            isubcat2 = 'Q'
        elif self.sicdmeta.ImageData.PixelType == 'RE16I_IM16I':
            pvtype = 'SI'
            abpp = 16
            isubcat1 = 'I'
            isubcat2 = 'Q'
        elif self.sicdmeta.ImageData.PixelType == 'AMP8I_PHS8I':
            pvtype = 'SI'
            abpp = 8
            isubcat1 = 'M'
            isubcat2 = 'P'
        self.fid.write(pvtype.ljust(3).encode())
        self.fid.write(b'NODISPLY')
        self.fid.write('SAR'.ljust(8).encode())
        self.fid.write(('%02d' % abpp).encode())
        self.fid.write(b'R')
        self.fid.write(b'G')
        # TODO: The corner lat/lons used here aren't really right for the case of
        #   multiple image segments, since GeoData.ImageCorners describes the
        #   entire image, not each segment.  However, these fields in the image
        #   subheader aren't really used by any tool we know anyway, since all SICD
        #   metadata should be extracted from the DES XML.
        try:  # Use TRY here since Lat/lon strings may be invalid
            frfc_lat = ll.string(self.sicdmeta.GeoData.ImageCorners.FRFC.Lat, 'lat',
                                 num_units=3, include_symbols=False)
        except:
            frfc_lat = ''
        try:
            frfc_lon = ll.string(self.sicdmeta.GeoData.ImageCorners.FRFC.Lon, 'lon',
                                 num_units=3, include_symbols=False)
        except:
            frfc_lon = ''
        try:
            frlc_lat = ll.string(self.sicdmeta.GeoData.ImageCorners.FRLC.Lat, 'lat',
                                 num_units=3, include_symbols=False)
        except:
            frlc_lat = ''
        try:
            frlc_lon = ll.string(self.sicdmeta.GeoData.ImageCorners.FRLC.Lon, 'lon',
                                 num_units=3, include_symbols=False)
        except:
            frlc_lon = ''
        try:
            lrlc_lat = ll.string(self.sicdmeta.GeoData.ImageCorners.LRLC.Lat, 'lat',
                                 num_units=3, include_symbols=False)
        except:
            lrlc_lat = ''
        try:
            lrlc_lon = ll.string(self.sicdmeta.GeoData.ImageCorners.LRLC.Lon, 'lon',
                                 num_units=3, include_symbols=False)
        except:
            lrlc_lon = ''
        try:
            lrfc_lat = ll.string(self.sicdmeta.GeoData.ImageCorners.LRFC.Lat, 'lat',
                                 num_units=3, include_symbols=False)
        except:
            lrfc_lat = ''
        try:
            lrfc_lon = ll.string(self.sicdmeta.GeoData.ImageCorners.LRFC.Lon, 'lon',
                                 num_units=3, include_symbols=False)
        except:
            lrfc_lon = ''
        self.fid.write(frfc_lat.ljust(7).encode())
        self.fid.write(frfc_lon.ljust(8).encode())
        self.fid.write(frlc_lat.ljust(7).encode())
        self.fid.write(frlc_lon.ljust(8).encode())
        self.fid.write(lrlc_lat.ljust(7).encode())
        self.fid.write(lrlc_lon.ljust(8).encode())
        self.fid.write(lrfc_lat.ljust(7).encode())
        self.fid.write(lrfc_lon.ljust(8).encode())
        self.fid.write(b'0')
        self.fid.write(b'NC')
        self.fid.write(b'2')
        self.fid.write(b'  ')
        self.fid.write(isubcat1.ljust(6).encode())
        self.fid.write(b'N')
        self.fid.write(b'   ')
        self.fid.write(b'0')
        self.fid.write(b'  ')
        self.fid.write(isubcat2.ljust(6).encode())
        self.fid.write(b'N')
        self.fid.write(b'   ')
        self.fid.write(b'0')
        self.fid.write(b'0')
        self.fid.write(b'P')
        self.fid.write(b'0001')
        self.fid.write(b'0001')
        if self.sicdmeta.ImageData.NumCols > 8192:
            nppbh = 0  # (zero means "use NCOLS")
        else:
            nppbh = self.sicdmeta.ImageData.NumCols
        self.fid.write(('%04d' % nppbh).encode())
        if int(self.num_rows_is[im_seg_number]) > 8192:
            nppbv = 0  # (zero means "use NROWS")
        else:
            nppbv = self.num_rows_is[im_seg_number]
        self.fid.write(('%04d' % nppbv).encode())
        self.fid.write(('%02d' % abpp).encode())
        self.fid.write(('%03d' % (im_seg_number+1)).encode())
        self.fid.write(('%03d' % im_seg_number).encode())
        if im_seg_number == 0:
            self.fid.write(b'00000')
        else:
            self.fid.write(('%05d' % self.num_rows_is[im_seg_number]).encode())
        self.fid.write(b'00000')
        self.fid.write(b'1.0 ')
        self.fid.write(b'00000')
        self.fid.write(b'00000')

    def _write_dessubhdr(self):
        self.fid.write(b'DE')  # DE
        self.fid.write('XML_DATA_CONTENT'.ljust(25).encode())  # DESID
        self.fid.write(b'01')  # DESVER
        self._write_security_tags()
        self.fid.write(b'0773')  # DESSHL
        self.fid.write(b'99999')  # DESCRC - CRC not computed
        self.fid.write(b'XML     ')  # DESSHFT
        try:  # May not have ImageCreation.DateTime field or it may be misformed
            # Creation time of original image
            fdt = datetime.strftime(self.sicdmeta.ImageCreation.DateTime, '%Y-%m-%dT%H:%M:%SZ')
        except:
            # Creation time of this NITF
            fdt = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%SZ')
        self.fid.write(fdt.encode())  # DESSHDT
        self.fid.write(b' '*40)  # DESSHRP
        self.fid.write(b'SICD Volume 1 Design & Implementation Description Document  ')  # DESSHSI
        self.fid.write(b'1.1       ')
        self.fid.write(b'2014-09-30T00:00:00Z')  # DESSHDS
        self.fid.write(('urn:SICD:1.1.0' + ' '*106).encode())  # DESSHTN
        if (hasattr(self.sicdmeta, 'GeoData') and
           hasattr(self.sicdmeta.GeoData, 'ImageCorners') and
           hasattr(self.sicdmeta.GeoData.ImageCorners, 'ICP')):
            self.fid.write(('%+012.8f%+013.8f%+012.8f%+013.8f%+012.8f%+013.8f' +
                           '%+012.8f%+013.8f%+012.8f%+013.8f' %
                            (self.sicdmeta.GeoData.ImageCorners.ICP.FRFC.Lat,
                             self.sicdmeta.GeoData.ImageCorners.ICP.FRFC.Lon,
                             self.sicdmeta.GeoData.ImageCorners.ICP.FRLC.Lat,
                             self.sicdmeta.GeoData.ImageCorners.ICP.FRLC.Lon,
                             self.sicdmeta.GeoData.ImageCorners.ICP.LRLC.Lat,
                             self.sicdmeta.GeoData.ImageCorners.ICP.LRLC.Lon,
                             self.sicdmeta.GeoData.ImageCorners.ICP.LRFC.Lat,
                             self.sicdmeta.GeoData.ImageCorners.ICP.LRFC.Lon,
                             self.sicdmeta.GeoData.ImageCorners.ICP.FRFC.Lat,
                             self.sicdmeta.GeoData.ImageCorners.ICP.FRFC.Lon)).encode())
        else:
            self.fid.write(b' '*125)
        self.fid.write(b' '*25)
        self.fid.write(b' '*20)
        self.fid.write(b' '*120)
        self.fid.write(b' '*200)

    def write_chip(self, data, start_indices=(0, 0)):
        """Writes the given data to a selected place in the already-opened file."""
        # All of the work done here is distributing the file writing across
        # the multiple NITF image segments in the SICD.  The actual writing
        # to file is done by calling a set of per-segment writer objects
        # that were setup in the constructor.
        lastrows = self.first_row_is + self.num_rows_is
        # Write data to file one segment at a time
        for i in range(self.num_is):
            # Is there anything to write in this segment?
            if ((start_indices[0] < lastrows[i]) and
               ((start_indices[0] + data.shape[0]) >= self.first_row_is[i])):
                # Indices of rows in entire image that we will be writing
                rowrange = np.array((max(start_indices[0], self.first_row_is[i]),
                                     min(start_indices[0] + data.shape[0], lastrows[i])))
                # Indices of rows in data input parameter that we will be writing from
                datarange = rowrange - start_indices[0]
                # Indices of NITF image segment that we will be writing to
                segmentrange = rowrange - self.first_row_is[i]
                self.writer_is[i](
                        data[datarange[0]:datarange[1], :],
                        [segmentrange[0], start_indices[1]])


def read_meta(filename, schema_struct=None):
    """Read metadata from Sensor Independent Complex Data (SICD) file, versions 0.3+"""

    nitf = read_nitf_offsets(filename)
    # SICD Volume 2, File Format Description, section 3.1.1 says that SICD XML
    # metadata must be stored in first DES.  We could also check content to
    # select which DES has the SICD XML in a multiple DES SICD.
    sicd_des_offset = nitf['des_offsets'][0]
    sicd_des_length = nitf['des_lengths'][0]

    # Read SICD XML metadata from the data extension segment
    with open(filename, mode='rb') as fid:
        fid.seek(sicd_des_offset)
        sicd_xml_string = fid.read(sicd_des_length)
    sicd_meta_struct = xml2struct(ET.fromstring(sicd_xml_string), schema_struct)

    # Adjust frequencies in metadata to be true, not offset values, if
    # reference frequency is available
    if (hasattr(sicd_meta_struct, 'RadarCollection') and
       hasattr(sicd_meta_struct.RadarCollection, 'RefFreqIndex') and
       sicd_meta_struct.RadarCollection.RefFreqIndex):
        try:
            import sicd_ref_freq
            apply_ref_freq(sicd_meta_struct, sicd_ref_freq.sicd_ref_freq)
        except ImportError:
            pass  # module doesn't exist, deal with it.
    return sicd_meta_struct, nitf


def read_nitf_offsets(filename):
    """Read NITF fields relevant to parsing SICD

    SICD (versions 0.3 and above) is stored in a NITF container.  NITF is a
    complicated format that involves lots of fields and configurations
    possibilities. Fortunately, SICD only really uses a small, specific
    portion of the NITF format.  This function extracts only the few parts of
    the NITF metadata necessary for reading a SICD NITF file.

    """

    # We have to open as binary, since there is some binary data in the file.
    # Python doesn't seem to let us read just part of the file as utf-8.
    with open(filename, mode='rb') as fid:
        # Read NITF file header
        if fid.read(9).decode('ascii') != "NITF02.10":  # Check format
            raise(IOError('SICD files must be NITF version 2.1'))
        fid.seek(354)  # Offset to first field of interest
        hl = np.uint32(fid.read(6))  # File header length
        numi = np.uint32(fid.read(3))  # Number of image segments
        img_segment_subhdr_lengths = np.zeros(numi, 'uint64')
        img_segment_data_lengths = np.zeros(numi, 'uint64')
        nitf = {}
        # Offset to image segment data from beginning of file (in bytes)
        nitf['img_segment_offsets'] = np.zeros(numi, 'uint64')
        # Number of rows in each image segment (in case data is spread across
        # multiple image segments)
        nitf['img_segment_rows'] = np.zeros(numi, 'uint32')
        # Number of columns in each image segment (in case data is spread
        # across multiple image segments)
        nitf['img_segment_columns'] = np.zeros(numi, 'uint32')
        for i in range(numi):
            img_segment_subhdr_lengths[i] = np.uint64(fid.read(6))
            nitf['img_segment_offsets'][i] = (
                hl +
                np.sum(img_segment_subhdr_lengths) +
                np.sum(img_segment_data_lengths))
            img_segment_data_lengths[i] = np.uint64(fid.read(10))
        segment_length = np.uint64(fid.read(3))
        if segment_length > 0:
            raise(IOError('SICD does not allow for graphics segments.'))
        segment_length = np.uint64(fid.read(3))
        if segment_length > 0:
            raise(IOError('SICD does not allow for reserved extension segments.'))
        numt = np.uint64(fid.read(3))
        text_segment_subhdr_lengths = np.zeros(numt, 'uint64')
        text_segment_data_lengths = np.zeros(numt, 'uint64')
        for i in range(numt):
            text_segment_subhdr_lengths[i] = np.uint64(fid.read(4))
            text_segment_data_lengths[i] = np.uint64(fid.read(5))
        numdes = np.uint32(fid.read(3))  # Number of data extension segments
        des_subhdr_lengths = np.zeros(numdes, 'uint64')
        des_data_lengths = np.zeros(numdes, 'uint64')
        for i in range(numdes):
            # Length of data extension segment subheader
            des_subhdr_lengths[i] = np.uint32(fid.read(4))
            # Length of data extension segment data
            des_data_lengths[i] = np.uint32(fid.read(9))
        nitf['des_lengths'] = des_data_lengths
        nitf['des_offsets'] = (
            hl + np.sum(img_segment_subhdr_lengths) +
            np.sum(img_segment_data_lengths) +
            np.sum(text_segment_subhdr_lengths) +
            np.sum(text_segment_data_lengths) +
            np.cumsum(des_subhdr_lengths) +
            # Doesn't work on older version of NumPy due to an unsafe cast
            # np.cumsum(np.insert(des_data_lengths[:-1], 0, 0))
            # This should work in all versions of numpy:
            np.cumsum(np.hstack((np.uint64(0), des_data_lengths[:-1]))))
        # Get number of rows for each image segment from image segment headers
        next_img_subhdr_offset = hl
        for i in range(numi):
            fid.seek(next_img_subhdr_offset)  # Jump to ith image segment
            fid.seek(333, 1)  # Jump to number of rows field
            nitf['img_segment_rows'][i] = np.uint32(fid.read(8))
            nitf['img_segment_columns'][i] = np.uint32(fid.read(8))
            next_img_subhdr_offset = (
                next_img_subhdr_offset +
                img_segment_subhdr_lengths[i] + img_segment_data_lengths[i])
    return nitf


def xml2struct(root_node, schema_struct=None):
    """Convert SICD XML into a structure

    Converts SICD XML into a Python object that is easy to browse and
    reference (similar in style and syntax to a MATLAB structure).  Makes sure
    all data types are read as the correct Python type, arrays are stored as
    arrays, etc.
    """
    def _recursfun(current_node, schema_struct, schema_types):
        """Recursive portion of the XML traversal."""
        current_struct = MetaNode()
        for child_node in current_node:
            # This is a stupid solution to remove namespace
            current_name = child_node.tag.split('}', 1)[-1]
            # Does schema contain information on this field?
            if schema_struct and hasattr(schema_struct, current_name):
                child_schema_struct = getattr(schema_struct, current_name)
                # Find base structure or primitive string
                while (hasattr(child_schema_struct, 'SCHEMA_type') and
                       hasattr(schema_types, child_schema_struct.SCHEMA_type)):
                    child_schema_struct = getattr(schema_types, child_schema_struct.SCHEMA_type)
            else:
                # We try to be flexible and read all fields, regardless
                # of whether the field is described in the schema or
                # not.  This allows extra custom fields to be included
                # that may not fit the spec.  Also, if we are reading a
                # SICD of a different version than the schema we are
                # using, this should allow that to work (at least
                # partially) as well.
                child_schema_struct = MetaNode()

            # Parse this child node's content
            if len(child_node) > 0:  # Substructure
                value = _recursfun(child_node, child_schema_struct, schema_types)
            else:  # Leaf (text/data node)
                # Three ways to get the class of a node
                if (child_schema_struct is not None and
                   hasattr(child_schema_struct, 'SCHEMA_type')):
                    # Current way, from schema
                    class_string = child_schema_struct.SCHEMA_type
                elif 'class' in child_node.attrib:
                    # Old SICDs (<0.5) used to have class info included in nodes
                    class_string = child_node.attrib['class']
                else:  # We will have to guess at the class
                    class_string = None
                if class_string:  # We know class
                    in_string = child_node.text
                    if class_string == 'xs:string':
                        value = in_string  # nothing to do
                    elif class_string == 'xs:double':
                        try:
                            value = float(in_string)
                        except ValueError:
                            value = float('nan')
                    elif class_string == 'xs:int':
                        try:
                            value = int(in_string)
                        except ValueError:
                            value = int('nan')
                    elif class_string == 'xs:dateTime':
                        value = re.search('\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.)?\d{,6}',
                                          in_string).group(0)
                        try:
                            value = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')
                        except ValueError:
                            value = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')
                    elif class_string == 'xs:boolean':
                        value = in_string == 'true'
                    else:  # unrecognized class
                        value = None
                else:  # Guess at class
                    value = child_node.text
                    try:
                        # If value is numeric, store as such
                        value = float(value)
                    except ValueError:
                        try:
                            datestr = re.search('\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.)?\d{,6}',
                                                                        value)
                            if datestr:  # dateTime
                                datestr = datestr.group(0)
                                try:
                                    value = datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S.%f')
                                except ValueError:
                                    value = datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S')
                            elif value.lower() in ['true', 'false']:  # boolean
                                value = value.lower() == 'true'
                        except ValueError:
                            pass


            # 'name' attribute requires special handling.  For the most part, in
            # SICD, the XML attributes don't need to be explicitly given to the
            # user since they don't do much more than order elements.  'name' is an
            # exception to this rule and does contain valuable content that a user
            # would need to see.
            if 'name' in child_node.attrib:
                if isinstance(value, MetaNode):  # Add attribute as subfield
                    value.name = child_node.attrib['name']
                else:  # Single text node.  Save as name/value pair
                    name_node = MetaNode()
                    name_node.name = child_node.attrib['name']
                    name_node.value = value
                    value = name_node

            # Handle special array cases
            if current_name == 'ICP':  # Index ICP by name, rather than number
                # Index values are '1:FRFC', '2:FRLC', '3:LRLC', '4:LRFC'
                # Use index as node name, rather than 'ICP'
                current_name = child_node.attrib['index'][2:]
                setattr(current_struct, current_name, value)
            elif ('index' in child_node.attrib and
                  len(current_node.findall('./' + child_node.tag)) > 1):  # Ordered elements
                if not hasattr(current_struct, current_name):  # Initialize list
                    setattr(current_struct, current_name, [None] *
                            len(current_node.findall('./' + child_node.tag)))
                getattr(current_struct, current_name)[int(child_node.attrib['index'])-1] = \
                    value
            elif 'exponent1' in child_node.attrib:  # Another type of ordered elements
                if not hasattr(current_struct, current_name):
                    # Initialize array.  Exponents must be of type float.
                    if 'order2' in current_node.attrib:
                        setattr(current_struct, current_name,
                                np.zeros((int(current_node.attrib['order1']) + 1,
                                          int(current_node.attrib['order2']) + 1), float))
                    else:
                        setattr(current_struct, current_name,
                                np.zeros(int(current_node.attrib['order1']) + 1, float))
                index1 = int(child_node.attrib['exponent1'])
                if 'exponent2' in child_node.attrib:
                    index2 = int(child_node.attrib['exponent2'])
                    getattr(current_struct, current_name)[index1, index2] = value
                else:
                    getattr(current_struct, current_name)[index1] = value
            elif hasattr(current_struct, current_name):  # Multiple occurences of a field name
                if isinstance(getattr(current_struct, current_name), list):
                    getattr(current_struct, current_name).append(value)
                else:
                    setattr(current_struct, current_name,
                            [getattr(current_struct, current_name), value])
            else:  # Normal non-array case
                setattr(current_struct, current_name, value)

        # Arrays where each element is numeric and in a separate XML node ('Coef'
        # and 'Wgt') are collapsed into a single array node.
        if ((('order1' in current_node.attrib) and  # 1- or 2-D polynomial
           isinstance(getattr(current_struct, current_name), np.ndarray)) or
           (('size' in current_node.attrib) and  # List of numbers
           isinstance(getattr(current_struct, current_name), list) and
           isinstance(getattr(current_struct, current_name)[0], (float, int)))):
            current_struct = np.array(getattr(current_struct, current_name))

        return current_struct

    if (root_node.tag.find('SICD') < 0) and (root_node.tag.find('CPHD') < 0):
        raise(IOError('Not a SICD or CPHD XML object.'))
    if schema_struct is None:
        schema_struct = MetaNode()
        schema_struct.master = MetaNode()
        schema_struct.types = MetaNode()
    output_struct = _recursfun(root_node, schema_struct.master, schema_struct.types)
    m = re.search(r'{urn:SICD:(?P<version_str>\d*\.\d*\.\d*)}SICD', root_node.tag)
    if m is not None:  # Extract and save SICD version
        # Starts with 'urn:SICD:' or 'urn:CPHD:'
        output_struct.SICDVersion = m.group('version_str')
        update_meta(output_struct, output_struct.SICDVersion)
    return output_struct


def struct2xml(sicdmeta, schema_struct=None, inc_newline=True):
    """Converts SICD metadata structure as created by the SarPy IO/complex framework
    into an XML string that can be written to a SICD file.  This function
    essentially inverts what xml2struct does.
    """
    def _recursfun(current_node, sicdmeta, schema_struct, schema_types):
        # Order fieldnames according to schema
        if schema_struct is not None:
            child_names = [i for (ord, i) in
                           sorted([(val.SCHEMA_order, key)
                                  for (key, val) in schema_struct.__dict__.items()
                                  if key in sicdmeta.__dict__.keys()])]
            # Add fields in structure that were not found in schema
            child_names.extend(sorted(set(sicdmeta.__dict__.keys()) -
                                      set(schema_struct.__dict__.keys())))
        else:
            child_names = sorted(sicdmeta.__dict__.keys())
        # Traverse sicdmeta structure in order
        for current_child_name in child_names:
            # Get schema info for this child node if available
            if hasattr(schema_struct, current_child_name):
                child_schema_struct = copy.deepcopy(getattr(schema_struct,
                                                            current_child_name))
                # Find base structure or primitive string
                while (hasattr(child_schema_struct, 'SCHEMA_type') and
                        hasattr(schema_types, child_schema_struct.SCHEMA_type)):
                    if hasattr(child_schema_struct, 'SCHEMA_attributes'):  # Schema "extension"
                        attributes_to_pass = child_schema_struct.SCHEMA_attributes
                    else:
                        attributes_to_pass = None
                    child_schema_struct = copy.deepcopy(getattr(
                            schema_types, child_schema_struct.SCHEMA_type))
                    if attributes_to_pass:  # Pass extension attributes to base type
                        if hasattr(child_schema_struct, 'SCHEMA_attributes'):
                            # Some of these may have already been passed through
                            # in previous uses of this structure, so we use a
                            # set to make sure they don't get added multiple times.
                            child_schema_struct.SCHEMA_attributes = \
                                list(set(attributes_to_pass).union(
                                    child_schema_struct.SCHEMA_attributes))
                        else:
                            child_schema_struct.SCHEMA_attributes = attributes_to_pass
            else:
                # We try to be flexible and read all fields, regardless
                # of whether the field is described in the schema or
                # not.  This allows extra custom fields to be included that
                # may not fit the spec.  Also, if our metadata comes from
                # a SICD of a different version than the schema we are
                # using, this should allow that to work (at least
                # partially) as well.
                child_schema_struct = None
            # Process structure content into XML
            current_child = getattr(sicdmeta, current_child_name)
            multiple_instances = isinstance(current_child, list)
            if not multiple_instances:
                # Multiple or single instances of the same field name should look the same
                current_child = [current_child]
            for i in range(len(current_child)):
                # Process single node
                if current_child_name in ('native', 'SICDVersion'):
                    # Non-spec fields often added by SarPy for internal use
                    pass
                elif current_child_name == 'ImageCorners':
                    # Special case: ICP Indexed by name rather than number
                    icp_node = ET.SubElement(current_node, current_child_name)
                    ICP_FIELDS = ('FRFC', 'FRLC', 'LRLC', 'LRFC')
                    for j in range(len(ICP_FIELDS)):
                        if hasattr(current_child[i], ICP_FIELDS[j]):
                            child_node = ET.SubElement(icp_node, 'ICP')
                            child_node.set('index', '%d:%s' % (j + 1, ICP_FIELDS[j]))
                            _recursfun(child_node,
                                       getattr(current_child[i], ICP_FIELDS[j]),
                                       child_schema_struct, schema_types)
                else:
                    child_node = ET.SubElement(current_node, current_child_name)
                    if hasattr(current_child[i], 'name'):
                        # Special attribute: Used in 'Paramater', 'Desc', 'GeoInfo', 'RefPt'
                        child_node.set('name', current_child[i].name)
                    if hasattr(current_child[i], 'value'):
                        # Special structure field: Used in 'Parameter', 'Desc'
                        child_node.text = current_child[i].value
                    elif isinstance(current_child[i], MetaNode):
                        if hasattr(current_child[i], 'name'):
                            child_copy = copy.deepcopy(current_child[i])
                            del child_copy.name
                        else:  # No need to make copy
                            child_copy = current_child[i]
                        _recursfun(child_node, child_copy, child_schema_struct, schema_types)
                        if ((child_schema_struct is None and
                             multiple_instances and current_child_name != 'GeoInfo') or
                            (hasattr(child_schema_struct, 'SCHEMA_attributes') and
                             ('index' in child_schema_struct.SCHEMA_attributes))):
                            child_node.set('index', '%d' % (i + 1))
                    elif (isinstance(current_child[i], np.ndarray) or
                          (hasattr(child_schema_struct, 'SCHEMA_attributes') and
                           (('order1' in child_schema_struct.SCHEMA_attributes) or
                           ('size' in child_schema_struct.SCHEMA_attributes)))):
                        sicdmeta2d = np.atleast_1d(current_child[i])  # Allow for scalars
                        is_more_than_1d = (sicdmeta2d.ndim > 1)
                        if not is_more_than_1d:
                            sicdmeta2d = sicdmeta2d[:, np.newaxis]
                        for j in range(sicdmeta2d.shape[0]):
                            for k in range(sicdmeta2d.shape[1]):
                                if current_child_name == 'WgtFunct':
                                    coef_node = ET.SubElement(child_node, 'Wgt')
                                    attribute_name = 'index'
                                    val = '%d' % (j + 1)
                                else:
                                    coef_node = ET.SubElement(child_node, 'Coef')
                                    attribute_name = 'exponent1'
                                    val = '%d' % j
                                coef_node.set(attribute_name, val)
                                if (is_more_than_1d or
                                        (hasattr(child_schema_struct, 'SCHEMA_attributes') and
                                         ('order2' in child_schema_struct.SCHEMA_attributes))):
                                    coef_node.set('exponent2', '%d' % k)
                                coef_node.text = '%.15E' % sicdmeta2d[j, k]
                    else:  # Scalar
                        # First check schema, then check MATLAB class of
                        # value in metadata structure. If variable in
                        # memory and schema disagree, we must convert type.
                        if hasattr(child_schema_struct, 'SCHEMA_type'):
                            class_str = child_schema_struct.SCHEMA_type
                            if class_str == 'xs:string':
                                if isinstance(current_child[i], str):
                                    str_value = current_child[i]
                                else:  # May have been incorrectly populated
                                    str_value = str(current_child[i])
                            elif class_str == 'xs:double':
                                str_value = '%.15E' % current_child[i]
                            elif class_str == 'xs:int':
                                str_value = '%d' % current_child[i]
                            elif class_str == 'xs:dateTime':
                                # Does %f work in Python 2.5?
                                str_value = datetime.strftime(
                                    current_child[i], '%Y-%m-%dT%H:%M:%S.%fZ')
                            elif class_str == 'xs:boolean':
                                if current_child[i]:
                                    str_value = 'true'
                                else:
                                    str_value = 'false'
                            else:
                                raise(ValueError('Unrecognized class type in SICD schema.'))
                        else:  # Field not found in schema.  Guess class based on value
                            # Special case: DateTime needs to be formatted/converted from double
                            # to string
                            if (current_child_name in ('DateTime', 'CollectStart') and
                               isinstance(current_child[i], datetime)):
                                # Does %f work in Python 2.5?
                                str_value = datetime.strftime(
                                    current_child[i], '%Y-%m-%dT%H:%M:%S.%fZ')
                                class_str = 'xs:dateTime'
                            elif isinstance(current_child[i], bool):
                                # bool is a subclass of int, so we have to check this first
                                if current_child[i]:
                                    str_value = 'true'
                                else:
                                    str_value = 'false'
                                class_str = 'xs:boolean'
                            elif isinstance(current_child[i], (int, np.long)):
                                str_value = str(current_child[i])
                                class_str = 'xs:int'
                            elif isinstance(current_child[i], float):
                                str_value = '%.15E' % current_child[i]
                                class_str = 'xs:double'
                            else:
                                str_value = current_child[i]
                                class_str = 'xs:string'
                        child_node.text = str_value
                        # if inc_class_attributes:  # No longer used in SICD
                        #     child_node.set('class', class_str)
                # Add size attributes, if necessary
                if hasattr(child_schema_struct, 'SCHEMA_attributes'):
                    if 'size' in child_schema_struct.SCHEMA_attributes:
                        child_node.set('size', str(len(child_node)))
                    elif 'order1' in child_schema_struct.SCHEMA_attributes:
                        child_node.set('order1', str(current_child[i].shape[0] - 1))
                        if 'order2' in child_schema_struct.SCHEMA_attributes:
                            child_node.set('order2', str(current_child[i].shape[1] - 1))

    root_node = ET.Element('SICD')
    if schema_struct is not None:
        root_node.set('xmlns', 'urn:SICD:1.1.0')
    if hasattr(schema_struct, 'master') and hasattr(schema_struct, 'types'):
        _recursfun(root_node, sicdmeta, schema_struct.master, schema_struct.types)
    else:
        _recursfun(root_node, sicdmeta, None, None)
    # It would be nice to run a validation against the schema here, as a
    # simple quality check against which a warning could be thrown, but
    # there doesn't appear to be a good, easy way to do this without
    # pulling in non-standard or compiled libraries.
    return ET.tostring(root_node)


def parse_schema(filename):
    """Parse SICD/CPHD schema XSD into a structure

    It is MUCH faster to traverse through a structure in memory than XML,
    so we want to convert the schema info into a structure before traversing
    through the XML.

    """
    def _recursfun_schema(current_node):
        """Recursive portion of the schema traversal."""
        # Order of the fields (but not attributes) in a schema matters, at
        # least for sequences.  We make effort to record not just the structure,
        # but also the order of the nodes here.
        output_struct = MetaNode()
        for child in current_node:
            tag = child.tag.split('}', 1)[-1]  # This is a stupid solution to remove namespace
            if tag == 'element':
                if 'type' in child.attrib:
                    # Uglier syntax than MATLAB structures...
                    setattr(output_struct, child.attrib['name'], MetaNode())
                    setattr(getattr(output_struct, child.attrib['name']),
                            'SCHEMA_order', len(output_struct.__dict__.keys()) - 1)
                    setattr(getattr(output_struct, child.attrib['name']),
                            'SCHEMA_type', child.attrib['type'])
                else:  # Element with empty type.  Should have a structure defined within it.
                    setattr(output_struct, child.attrib['name'],
                            _recursfun_schema(child))
                    setattr(getattr(output_struct, child.attrib['name']),
                            'SCHEMA_order', len(output_struct.__dict__.keys()) - 1)
            elif tag in ['restriction', 'extension']:
                output_struct = _recursfun_schema(child)  # Adds any attributes
                output_struct.SCHEMA_type = child.attrib['base']
            elif tag in ['simpleType', 'simpleContent', 'complexType', 'complexContent']:
                output_struct = _recursfun_schema(child)
            elif tag in ['sequence', 'choice', 'all']:
                new_struct = _recursfun_schema(child)
                # Shallow merge of new_struct with output_struct.
                # Take care to maintain ordering.
                init_length = len(output_struct.__dict__.keys())
                for key, value in new_struct.__dict__.items():
                    if hasattr(value, 'SCHEMA_order'):
                        setattr(value, 'SCHEMA_order', init_length +
                                getattr(value, 'SCHEMA_order'))
                    setattr(output_struct, key, value)
            elif tag == 'attribute':
                if hasattr(output_struct, 'SCHEMA_attributes'):
                    output_struct.SCHEMA_attributes.append(child.attrib['name'])
                else:
                    output_struct.SCHEMA_attributes = [child.attrib['name']]
            elif tag in ['minInclusive', 'maxInclusive', 'enumeration']:
                pass  # These fields are expected, but we don't use them for anything.
            else:
                raise(IOError('Unrecognized node type in XSD.'))

        return output_struct

    schema_struct = MetaNode()
    schema_struct.types = MetaNode()
    for child in ET.parse(filename).getroot():
        tag = child.tag.split('}', 1)[-1]  # This is a stupid solution to remove namespace
        if tag in ['simpleType', 'complexType']:  # Type definitions
            setattr(schema_struct.types, child.attrib['name'],
                    _recursfun_schema(child))
        elif tag == 'element':  # Master node (should be only one)
            schema_struct.master = _recursfun_schema(child)
        else:
            raise(IOError('This type of node not expected in SICD schema.'))

    return schema_struct


def update_meta(sicd_meta, version_string):
    """Master function for updating SICD metadata structure from old versions
    to current one.  Nested functions break this version upgrade up into
    sections specific for each SICD version."""
    def sicd_update_meta_0_4(sicd_meta):
        """Update a SICD metadata structure from version 0.4 to current version
        (whatever that may be)"""
        # Update WgtType format
        for i in ['Row', 'Col']:
            if hasattr(sicd_meta, 'Grid') and hasattr(sicd_meta.Grid, i):
                grid_struct = getattr(sicd_meta.Grid, i)
                if hasattr(grid_struct, 'WgtType') and isinstance(grid_struct.WgtType, str):
                    wgt_name = grid_struct.WgtType.split()
                    parameters = wgt_name[1:]
                    wgt_name = wgt_name[0]
                    grid_struct.WgtType = MetaNode()  # Change from string to structure
                    grid_struct.WgtType.WindowName = wgt_name
                    if parameters:
                        grid_struct.WgtType.Parameter = []
                        for cur_par_str in parameters:
                            parameter_parts = cur_par_str.split('=')
                            if len(parameter_parts) > 1:
                                cur_struct = MetaNode()
                                cur_struct.name = parameter_parts[0]
                                cur_struct.value = parameter_parts[1]
                                grid_struct.WgtType.Parameter.append(cur_struct)
            setattr(sicd_meta.Grid, i, grid_struct)
        # We are now updated to version 0.5.  Now do rest of updates.
        sicd_update_meta_0_5(sicd_meta)

    def sicd_update_meta_0_5(sicd_meta):
        """Update a SICD metadata structure from version 0.5 to current version
        (whatever that may be)"""
        # Add RadarCollection.TxPolarization, now required, but optional prior to version 1.0
        if (hasattr(sicd_meta, 'RadarCollection') and
                not hasattr(sicd_meta.RadarCollection, 'TxPolarization') and
                hasattr(sicd_meta.RadarCollection, 'RcvChannels') and
                hasattr(sicd_meta.RadarCollection.RcvChannels, 'ChanParameters')):
            ChanPars = sicd_meta.RadarCollection.RcvChannels.ChanParameters  # Shorten notation
            if isinstance(ChanPars, list):
                sicd_meta.RadarCollection.TxPolarization = 'SEQUENCE'
                # Set comprehension to avoid repeats.  Conversion to list lets us index into it.
                tx_pols = list(set(i.TxRcvPolarization[0] for i in ChanPars))
                if not hasattr(sicd_meta.RadarCollection, 'TxSequence'):
                    sicd_meta.RadarCollection.TxSequence = MetaNode()
                if not hasattr(sicd_meta.RadarCollection.TxSequence, 'TxStep'):
                    # Should always be a list for SEQUENCE
                    sicd_meta.RadarCollection.TxSequence.TxStep = []
                for i in range(len(tx_pols)):
                    if (i + 1) > len(sicd_meta.RadarCollection.TxSequence.TxStep):
                        sicd_meta.RadarCollection.TxSequence.TxStep.append(MetaNode())
                    sicd_meta.RadarCollection.TxSequence.TxStep[i].TxPolarization = tx_pols[i]
                # Note: If there are multiple waveforms and multiple polarizations,
                # there is no deconfliction done here.
            else:
                sicd_meta.RadarCollection.TxPolarization = ChanPars.TxRcvPolarization[0]

        # RadarCollection.Area.Corner was optional in version 0.5, but required in
        # version 1.0.  Fortunately, Corner is easily derived from Plane.
        if (hasattr(sicd_meta, 'RadarCollection') and
           hasattr(sicd_meta.RadarCollection, 'Area') and
           not hasattr(sicd_meta.RadarCollection.Area, 'Corner') and
           hasattr(sicd_meta.RadarCollection.Area, 'Plane')):
            try:  # If Plane substructure is misformed, this may fail
                plane = sicd_meta.RadarCollection.Area.Plane  # For concise notation
                ref_pt = np.array([plane.RefPt.ECF.X, plane.RefPt.ECF.Y, plane.RefPt.ECF.Z])
                x_uvect = np.array([plane.XDir.UVectECF.X, plane.XDir.UVectECF.Y,
                                    plane.XDir.UVectECF.Z])
                y_uvect = np.array([plane.YDir.UVectECF.X, plane.YDir.UVectECF.Y,
                                    plane.YDir.UVectECF.Z])
                x_offsets = np.array([plane.XDir.FirstLine, plane.XDir.FirstLine,
                                      plane.XDir.NumLines, plane.XDir.NumLines])
                y_offsets = np.array([plane.YDir.FirstSample, plane.YDir.NumSamples,
                                      plane.YDir.NumSamples, plane.YDir.FirstSample])
                sicd_meta.RadarCollection.Area.Corner = MetaNode()
                sicd_meta.RadarCollection.Area.Corner.ACP = [MetaNode() for _ in range(4)]
                for i in range(4):
                    acp_ecf = ref_pt + \
                        x_uvect * plane.XDir.LineSpacing * (x_offsets[i] - plane.RefPt.Line) + \
                        y_uvect * plane.YDir.SampleSpacing * (y_offsets[i] - plane.RefPt.Sample)
                    acp_llh = gc.ecf_to_geodetic(acp_ecf).squeeze()
                    sicd_meta.RadarCollection.Area.Corner.ACP[i].Lat = acp_llh[0]
                    sicd_meta.RadarCollection.Area.Corner.ACP[i].Lon = acp_llh[1]
                    sicd_meta.RadarCollection.Area.Corner.ACP[i].HAE = acp_llh[2]
            except AttributeError:  # OK.  Just means fields missing
                pass
            except ImportError:  # ecf_to_geodetic module not in expected place
                pass  # Just continue without computing corners

        # PolarizationHVAnglePoly no longer a valid field in version 1.0.
        if (hasattr(sicd_meta, 'RadarCollection') and
           hasattr(sicd_meta.RadarCollection, 'PolarizationHVAnglePoly')):
            del sicd_meta.RadarCollection.PolarizationHVAnglePoly

        # Antenna.Tx/Rcv/TwoWay.HPBW no longer a valid field in version 1.0.
        if hasattr(sicd_meta, 'Antenna'):
            if (hasattr(sicd_meta.Antenna, 'Tx') and
                    hasattr(sicd_meta.Antenna.Tx, 'HPBW')):
                del sicd_meta.Antenna.Tx.HPBW
            if (hasattr(sicd_meta.Antenna, 'Rcv') and
                    hasattr(sicd_meta.Antenna.Rcv, 'HPBW')):
                del sicd_meta.Antenna.Rcv.HPBW
            if (hasattr(sicd_meta.Antenna, 'TwoWay') and
                    hasattr(sicd_meta.Antenna.TwoWay, 'HPBW')):
                del sicd_meta.Antenna.TwoWay.HPBW

        # NoiseLevel got its own substructure between SICD 0.5 and SICD 1.0
        if (hasattr(sicd_meta, 'Radiometric') and
                hasattr(sicd_meta.Radiometric, 'NoisePoly')):
            sicd_meta.Radiometric.NoiseLevel = MetaNode()
            sicd_meta.Radiometric.NoiseLevel.NoisePoly = \
                sicd_meta.Radiometric.NoisePoly
            del sicd_meta.Radiometric.NoisePoly
            if hasattr(sicd_meta.Radiometric, 'NoiseLevelType'):
                sicd_meta.Radiometric.NoiseLevel.NoiseLevelType = \
                    sicd_meta.Radiometric.NoiseLevelType
                del sicd_meta.Radiometric.NoiseLevelType
            else:
                # Even if NoiseLevelType wasn't given, we know that relative noise
                # levels should be 1 at SCP.
                if abs(sicd_meta.Radiometric.NoiseLevel.NoisePoly.flatten()[0]-1) < np.spacing(1):
                    sicd_meta.Radiometric.NoiseLevel.NoiseLevelType = 'RELATIVE'
                else:
                    sicd_meta.Radiometric.NoiseLevel.NoiseLevelType = 'ABSOLUTE'

        # MatchInfo
        if hasattr(sicd_meta, 'MatchInfo'):
            newMatchInfo = MetaNode()  # Clear this out so we can reconstruct it
            # MatchType was optional field in 0.5
            if hasattr(sicd_meta.MatchInfo, 'Collect'):
                if not isinstance(sicd_meta.MatchInfo.Collect, list):
                    sicd_meta.MatchInfo.Collect = [sicd_meta.MatchInfo.Collect]
                # Making a set to remove duplicates
                types = set(i.MatchType for i in sicd_meta.MatchInfo.Collect)
            else:
                types = set([''])  # TypeID (equivalent of MatchType) required in 1.0
            newMatchInfo.NumMatchTypes = len(types)
            newMatchInfo.MatchType = []
            for current_type in types:
                collects = [j for j in sicd_meta.MatchInfo.Collect
                            if hasattr(j, 'MatchType') and j.MatchType == current_type]
                matchtype = MetaNode()
                matchtype.TypeID = current_type.strip()
                # SICD version 0.5 included current instance as one of the
                # collections whereas version 1.0 did not
                matchtype.NumMatchCollections = len(collects) - 1
                matchtype.MatchCollection = []
                for current_collect in collects:
                    if hasattr(current_collect, 'Parameter'):
                        if isinstance(current_collect.Parameter, list):
                            # Multiple parameters
                            current_index = next((int(k.value) for k in current_collect.Parameter
                                                 if k.name.strip() == 'CURRENT_INSTANCE'), None)
                        elif current_collect.Parameter.name.strip() == 'CURRENT_INSTANCE':
                            current_index = int(current_collect.Parameter.value)
                        else:
                            current_index = None
                    else:
                        current_index = None
                    if current_index is not None:
                        matchtype.CurrentIndex = current_index
                    else:
                        matchcollection = MetaNode()
                        if hasattr(current_collect, 'CoreName'):  # Required field
                            matchcollection.CoreName = current_collect.CoreName
                        if hasattr(current_collect, 'Parameter'):
                            matchcollection.Parameter = current_collect.Parameter
                        matchtype.MatchCollection.append(matchcollection)
                if len(matchtype.MatchCollection) == 0:
                    del matchtype.MatchCollection
                newMatchInfo.MatchType.append(matchtype)
            sicd_meta.MatchInfo = newMatchInfo

        # Add AzimAng and LayoverAng to SCPCOA
        sicd_meta = derived_fields(sicd_meta)

    try:
        version_parts = [int(i) for i in version_string.split('.')]
    except ValueError:  # Version string misformed
        pass
    # Update metadata structure to current version if necessary
    if version_parts >= [1]:  # Nothing to change between 1.0 and 1.1
        pass
    elif version_parts >= [0, 5]:  # Version 0.5
        sicd_update_meta_0_5(sicd_meta)
    elif version_parts >= [0, 4]:  # Version 0.4
        sicd_update_meta_0_4(sicd_meta)
    else:  # Either older, unrecognized version or mislablled version
        sicd_update_meta_0_4(sicd_meta)  # Attempt to update what we can


def derived_fields(meta, set_default_values=True):
    """This function attempts to populate missing fields from a SICD metadata
    structure.  Using this function should allow one to more simply (with
    less replicated code) create SICDs from a number of different sources by
    defining only the fundamental fields and then calling this function to
    populate all of the derived fields.

    There are two types of fields which are populated in this function:
    1) DERIVED values: These fields can be computed exactly from other
    fields. SICD includes many redundant parameters for ease of access.  This
    function tries to see which core, fundamental fields are available and
    calculate as many derived fields from those as possible.  Example:
    SCPCOA.SCPTime must equal Grid.TimeCOAPoly(1).
    2) DEFAULT values: These are fields which may not be given exactly, but
    for which we can make a reasonable guess or approximation based on the
    most common types of SAR processing.  In fact, some of these fields are
    so common that they are just assumed and not even explicitly given in
    other file formats. Population of these fields can be turned off through
    the SET_DEFAULT_VALUES variable since they are not absolutely known.
    Example: The PFA image plane normal is often the instantaneous slant
    plane at center of aperture.

    Within the code and comments, we attempt to label whether the value being
    populated is a DERIVED or DEFAULT value.  Note that if a field is already
    populated in the input metadata structure, this function will not
    overwrite it for either the DERIVED or DEFAULT cases."""

    def _hamming_ipr(x, a):
        return a*(np.sin(np.pi*x)/(np.pi*x)) + \
            ((1-a)*(np.sin(np.pi*(x-1))/(np.pi*(x-1)))/2) + \
            ((1-a)*(np.sin(np.pi*(x+1))/(np.pi*(x+1)))/2) - \
            a/np.sqrt(2)

    # Fields DERIVED from Grid parameters
    if hasattr(meta.ImageData, 'ValidData'):  # Test vertices
        valid_vertices = [(v.Row, v.Col) for v in meta.ImageData.ValidData.Vertex]
    else:  # Use edges of full image if ValidData not available
        valid_vertices = [(0, 0),
                          (0, meta.ImageData.NumCols-1),
                          (meta.ImageData.NumRows-1, meta.ImageData.NumCols-1),
                          (meta.ImageData.NumRows-1, 0)]
    for current_fieldname in ('Row', 'Col'):
        if hasattr(meta, 'Grid') and hasattr(meta.Grid, current_fieldname):
            row_column = getattr(meta.Grid, current_fieldname)
            # WgtFunct DERIVED from WgtType
            if (hasattr(row_column, 'WgtType') and
               hasattr(row_column.WgtType, 'WindowName') and
               not hasattr(row_column, 'WgtFunct') and
               row_column.WgtType.WindowName not in ['UNIFORM', 'UNKNOWN']):
                try:  # Will error if WgtFunct cannot be created from WgtType
                    DEFAULT_WGT_SIZE = 512
                    row_column.WgtFunct = weight2fun(row_column)(DEFAULT_WGT_SIZE)
                except Exception:
                    pass
            broadening_factor = None
            if (hasattr(row_column, 'WgtType') and
               hasattr(row_column.WgtType, 'WindowName')):
                try:  # If scipy not available, don't crash
                    from scipy.optimize import fsolve  # Only import if needed
                    if row_column.WgtType.WindowName.upper() == 'UNIFORM':  # 0.886
                        broadening_factor = 2 * fsolve(lambda x: _hamming_ipr(x, 1), .1)[0]
                    elif row_column.WgtType.WindowName.upper() == 'HAMMING':
                        if (not hasattr(row_column.WgtType, 'Parameter') or
                           not hasattr(row_column.WgtType.Parameter, 'value')):
                            # A Hamming window is defined in many places as a
                            # raised cosine of weight .54, so this is the default.
                            # However some data use a generalized raised cosine and
                            # call it HAMMING, so we allow for both uses.
                            coef = 0.54
                        else:
                            coef = float(row_column.WgtType.Parameter.value)
                        broadening_factor = 2 * fsolve(lambda x: _hamming_ipr(x, coef), .1)[0]
                    elif row_column.WgtType.WindowName.upper() == 'HANNING':
                        broadening_factor = 2 * fsolve(lambda x: _hamming_ipr(x, 0.5), .1)[0]
                except ImportError:
                    pass
            if broadening_factor is None and hasattr(row_column, 'WgtFunct'):
                OVERSAMPLE = 1024
                imp_resp = abs(np.fft.fft(row_column.WgtFunct,  # Oversampled response function
                                          int(row_column.WgtFunct.size * OVERSAMPLE)))
                imp_resp = imp_resp/sum(row_column.WgtFunct)  # Normalize to unit peak
                # Samples surrounding half-power point
                ind = np.flatnonzero(imp_resp < (1/np.sqrt(2)))[0] + np.array([-1, 0])
                # Linear interpolation to solve for half-power point
                ind = ind[0] + ((1/np.sqrt(2)) - imp_resp[ind[0]]) / np.diff(imp_resp[ind])[0]
                broadening_factor = 2*ind/OVERSAMPLE
            # Resolution can be DERIVED from bandwidth and weighting type
            if broadening_factor is not None:
                if hasattr(row_column, 'ImpRespBW') and not hasattr(row_column, 'ImpRespWid'):
                    row_column.ImpRespWid = broadening_factor/row_column.ImpRespBW
                elif hasattr(row_column, 'ImpRespWid') and not hasattr(row_column, 'ImpRespBW'):
                    row_column.ImpRespBW = broadening_factor/row_column.ImpRespWid
            # DeltaK1/2 can be APPROXIMATED from DeltaKCOAPoly
            if (hasattr(row_column, 'ImpRespBW') and
               hasattr(row_column, 'SS') and
               (not hasattr(row_column, 'DeltaK1')) and
               (not hasattr(row_column, 'DeltaK2'))):
                if hasattr(row_column, 'DeltaKCOAPoly'):
                    min_dk = np.Inf
                    max_dk = -np.Inf
                    # Here, we assume the min and max of DeltaKCOAPoly must be on
                    # the vertices of the image, since it is smooth and monotonic
                    # in most cases-- although in actuality this is not always the
                    # case.  To be totally generic, we would have to search for an
                    # interior min and max as well.
                    for vertex in valid_vertices:
                        currentDeltaK = poly.polyval2d(
                            vertex[0], vertex[1], row_column.DeltaKCOAPoly)
                        min_dk = min(min_dk, currentDeltaK)
                        max_dk = max(max_dk, currentDeltaK)
                else:
                    min_dk = 0
                    max_dk = 0
                min_dk = min_dk - (row_column.ImpRespBW/2)
                max_dk = max_dk + (row_column.ImpRespBW/2)
                # Wrapped spectrum
                if (min_dk < -(1/row_column.SS)/2) or (max_dk > (1/row_column.SS)/2):
                    min_dk = -(1/row_column.SS)/2
                    max_dk = -min_dk
                row_column.DeltaK1 = min_dk
                row_column.DeltaK2 = max_dk
    # SCPTime can always be DERIVED from Grid.TimeCOAPoly
    if (((not hasattr(meta, 'SCPCOA')) or (not hasattr(meta.SCPCOA, 'SCPTime'))) and
       hasattr(meta, 'Grid') and hasattr(meta.Grid, 'TimeCOAPoly')):
        if not hasattr(meta, 'SCPCOA'):
            meta.SCPCOA = MetaNode()
        meta.SCPCOA.SCPTime = meta.Grid.TimeCOAPoly[0, 0]
    # and sometimes Grid.TimeCOAPoly can be DERIVED from SCPTime
    elif ((not hasattr(meta, 'Grid') or not hasattr(meta.Grid, 'TimeCOAPoly')) and
          hasattr(meta, 'SCPCOA') and hasattr(meta.SCPCOA, 'SCPTime') and
          hasattr(meta, 'CollectionInfo') and
          hasattr(meta.CollectionInfo, 'RadarMode') and
          hasattr(meta.CollectionInfo.RadarMode, 'ModeType') and
          meta.CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT'):
        if not hasattr(meta, 'Grid'):
            meta.Grid = MetaNode()
        meta.Grid.TimeCOAPoly = np.atleast_2d(meta.SCPCOA.SCPTime)
    # ARP Pos/Vel/ACC fields can be DERIVED from ARPPoly and SCPTime
    if (hasattr(meta, 'Position') and hasattr(meta.Position, 'ARPPoly') and
       hasattr(meta, 'SCPCOA') and hasattr(meta.SCPCOA, 'SCPTime')):
        if not hasattr(meta.SCPCOA, 'ARPPos'):
            meta.SCPCOA.ARPPos = MetaNode()
            meta.SCPCOA.ARPPos.X = poly.polyval(meta.SCPCOA.SCPTime, meta.Position.ARPPoly.X)
            meta.SCPCOA.ARPPos.Y = poly.polyval(meta.SCPCOA.SCPTime, meta.Position.ARPPoly.Y)
            meta.SCPCOA.ARPPos.Z = poly.polyval(meta.SCPCOA.SCPTime, meta.Position.ARPPoly.Z)
        # Velocity is derivative of position
        if not hasattr(meta.SCPCOA, 'ARPVel'):
            meta.SCPCOA.ARPVel = MetaNode()
        meta.SCPCOA.ARPVel.X = poly.polyval(meta.SCPCOA.SCPTime,
                                            poly.polyder(meta.Position.ARPPoly.X))
        meta.SCPCOA.ARPVel.Y = poly.polyval(meta.SCPCOA.SCPTime,
                                            poly.polyder(meta.Position.ARPPoly.Y))
        meta.SCPCOA.ARPVel.Z = poly.polyval(meta.SCPCOA.SCPTime,
                                            poly.polyder(meta.Position.ARPPoly.Z))
        # Acceleration is second derivative of position
        if not hasattr(meta.SCPCOA, 'ARPAcc'):
            meta.SCPCOA.ARPAcc = MetaNode()
        meta.SCPCOA.ARPAcc.X = poly.polyval(meta.SCPCOA.SCPTime,
                                            poly.polyder(meta.Position.ARPPoly.X, 2))
        meta.SCPCOA.ARPAcc.Y = poly.polyval(meta.SCPCOA.SCPTime,
                                            poly.polyder(meta.Position.ARPPoly.Y, 2))
        meta.SCPCOA.ARPAcc.Z = poly.polyval(meta.SCPCOA.SCPTime,
                                            poly.polyder(meta.Position.ARPPoly.Z, 2))
    # A simple ARPPoly can be DERIVED from SCPCOA Pos/Vel/Acc if that was all that was defined.
    if (hasattr(meta, 'SCPCOA') and hasattr(meta.SCPCOA, 'ARPPos') and
       hasattr(meta.SCPCOA, 'ARPVel') and hasattr(meta.SCPCOA, 'SCPTime') and
       (not hasattr(meta, 'Position') or not hasattr(meta.Position, 'ARPPoly'))):
        if not hasattr(meta.SCPCOA, 'ARPAcc'):
            meta.SCPCOA.ARPAcc = MetaNode()
            meta.SCPCOA.ARPAcc.X = 0
            meta.SCPCOA.ARPAcc.Y = 0
            meta.SCPCOA.ARPAcc.Z = 0
        if not hasattr(meta, 'Position'):
            meta.Position = MetaNode()
        if not hasattr(meta.Position, 'ARPPoly'):
            meta.Position.ARPPoly = MetaNode()
        for i in ('X', 'Y', 'Z'):
            setattr(meta.Position.ARPPoly, i, [
                # Constant
                getattr(meta.SCPCOA.ARPPos, i) -
                (getattr(meta.SCPCOA.ARPVel, i) * meta.SCPCOA.SCPTime) +
                ((getattr(meta.SCPCOA.ARPAcc, i)/2) * (meta.SCPCOA.SCPTime**2)),
                # Linear
                getattr(meta.SCPCOA.ARPVel, i) -
                (getattr(meta.SCPCOA.ARPAcc, i) * meta.SCPCOA.SCPTime),
                # Quadratic
                getattr(meta.SCPCOA.ARPAcc, i)/2])
    # Transmit bandwidth
    if (hasattr(meta, 'RadarCollection') and
       hasattr(meta.RadarCollection, 'Waveform') and
       hasattr(meta.RadarCollection.Waveform, 'WFParameters')):
        # DERIVED: Redundant WFParameters fields
        if isinstance(meta.RadarCollection.Waveform.WFParameters, list):
            wfparameters = meta.RadarCollection.Waveform.WFParameters
        else:
            wfparameters = [meta.RadarCollection.Waveform.WFParameters]
        for wfp in wfparameters:
            if (hasattr(wfp, 'RcvDemodType') and wfp.RcvDemodType == 'CHIRP' and
               not hasattr(wfp, 'RcvFMRate')):
                wfp.RcvFMRate = 0
            if (hasattr(wfp, 'RcvFMRate') and (wfp.RcvFMRate == 0) and
               not hasattr(wfp, 'RcvDemodType')):
                wfp.RcvDemodType = 'CHIRP'
            if (not hasattr(wfp, 'TxRFBandwidth') and hasattr(wfp, 'TxPulseLength') and
               hasattr(wfp, 'TxFMRate')):
                wfp.TxRFBandwidth = wfp.TxPulseLength * wfp.TxFMRate
            if (hasattr(wfp, 'TxRFBandwidth') and not hasattr(wfp, 'TxPulseLength') and
               hasattr(wfp, 'TxFMRate')):
                wfp.TxPulseLength = wfp.TxRFBandwidth / wfp.TxFMRate
            if (hasattr(wfp, 'TxRFBandwidth') and hasattr(wfp, 'TxPulseLength') and
               not hasattr(wfp, 'TxFMRate')):
                wfp.TxFMRate = wfp.TxRFBandwidth / wfp.TxPulseLength
        # DERIVED: These values should be equal.
        if (not hasattr(meta.RadarCollection, 'TxFrequency') or
                not hasattr(meta.RadarCollection.TxFrequency, 'Min')):
            meta.RadarCollection.TxFrequency.Min = \
                min([wfp.TxFreqStart for wfp in wfparameters])
        if (not hasattr(meta.RadarCollection, 'TxFrequency') or
                not hasattr(meta.RadarCollection.TxFrequency, 'Max')):
            meta.RadarCollection.TxFrequency.Max = \
                max([(wfp.TxFreqStart+wfp.TxRFBandwidth) for wfp in wfparameters])
    if (hasattr(meta, 'RadarCollection') and
       hasattr(meta.RadarCollection, 'TxFrequency') and
       hasattr(meta.RadarCollection.TxFrequency, 'Min') and
       hasattr(meta.RadarCollection.TxFrequency, 'Max')):
        # DEFAULT: We often assume that all transmitted bandwidth was
        # processed, if given no other information.
        if set_default_values:
            if (not hasattr(meta, 'ImageFormation') or
               not hasattr(meta.ImageFormation, 'TxFrequencyProc') or
               not hasattr(meta.ImageFormation.TxFrequencyProc, 'MinProc')):
                meta.ImageFormation.TxFrequencyProc.MinProc = \
                    meta.RadarCollection.TxFrequency.Min
            if (not hasattr(meta, 'ImageFormation') or
               not hasattr(meta.ImageFormation, 'TxFrequencyProc') or
               not hasattr(meta.ImageFormation.TxFrequencyProc, 'MaxProc')):
                meta.ImageFormation.TxFrequencyProc.MaxProc = \
                    meta.RadarCollection.TxFrequency.Max
        # DERIVED: These values should be equal.
        if (hasattr(meta.RadarCollection, 'Waveform') and
                hasattr(meta.RadarCollection.Waveform, 'WFParameters') and
                isinstance(meta.RadarCollection.Waveform.WFParameters, MetaNode)):
            if not hasattr(meta.RadarCollection.Waveform.WFParameters, 'TxFreqStart'):
                meta.RadarCollection.Waveform.WFParameters.TxFreqStart = \
                    meta.RadarCollection.TxFrequency.Min
            if not hasattr(meta.RadarCollection.Waveform.WFParameters, 'TxRFBandwidth'):
                meta.RadarCollection.Waveform.WFParameters.TxRFBandwidth = \
                    (meta.RadarCollection.TxFrequency.Max -
                     meta.RadarCollection.TxFrequency.Min)
    # We might use center processed frequency later
    if (hasattr(meta, 'ImageFormation') and
       hasattr(meta.ImageFormation, 'TxFrequencyProc') and
       hasattr(meta.ImageFormation.TxFrequencyProc, 'MinProc') and
       hasattr(meta.ImageFormation.TxFrequencyProc, 'MaxProc') and
       (not hasattr(meta.RadarCollection, 'RefFreqIndex') or
       (meta.RadarCollection.RefFreqIndex == 0))):
        fc = (meta.ImageFormation.TxFrequencyProc.MinProc +
              meta.ImageFormation.TxFrequencyProc.MaxProc)/2
    # DERIVED: GeoData.SCP
    if (hasattr(meta, 'GeoData') and hasattr(meta.GeoData, 'SCP') and
            hasattr(meta.GeoData.SCP, 'ECF') and not hasattr(meta.GeoData.SCP, 'LLH')):
        llh = gc.ecf_to_geodetic([meta.GeoData.SCP.ECF.X,
                                  meta.GeoData.SCP.ECF.Y,
                                  meta.GeoData.SCP.ECF.Z])[0]
        meta.GeoData.SCP.LLH = MetaNode()
        meta.GeoData.SCP.LLH.Lat = llh[0]
        meta.GeoData.SCP.LLH.Lon = llh[1]
        meta.GeoData.SCP.LLH.HAE = llh[2]
    if (hasattr(meta, 'GeoData') and hasattr(meta.GeoData, 'SCP') and
            hasattr(meta.GeoData.SCP, 'LLH') and not hasattr(meta.GeoData.SCP, 'ECF')):
        ecf = gc.geodetic_to_ecf([meta.GeoData.SCP.LLH.Lat,
                                  meta.GeoData.SCP.LLH.Lon,
                                  meta.GeoData.SCP.LLH.HAE])[0]
        meta.GeoData.SCP.ECF = MetaNode()
        meta.GeoData.SCP.ECF.X = ecf[0]
        meta.GeoData.SCP.ECF.Y = ecf[1]
        meta.GeoData.SCP.ECF.Z = ecf[2]
    # Many fields (particularly in SCPCOA) can be DERIVED from ARPPos, ARPVel and SCP
    if (hasattr(meta, 'SCPCOA') and hasattr(meta.SCPCOA, 'ARPPos') and
            hasattr(meta.SCPCOA, 'ARPVel') and hasattr(meta, 'GeoData') and
            hasattr(meta.GeoData, 'SCP') and hasattr(meta.GeoData.SCP, 'ECF')):
        SCP = np.array([meta.GeoData.SCP.ECF.X, meta.GeoData.SCP.ECF.Y, meta.GeoData.SCP.ECF.Z])
        ARP = np.array([meta.SCPCOA.ARPPos.X, meta.SCPCOA.ARPPos.Y, meta.SCPCOA.ARPPos.Z])
        ARP_v = np.array([meta.SCPCOA.ARPVel.X, meta.SCPCOA.ARPVel.Y, meta.SCPCOA.ARPVel.Z])
        uLOS = (SCP - ARP)/np.linalg.norm(SCP - ARP)
        left = np.cross(ARP/np.linalg.norm(ARP), ARP_v/np.linalg.norm(ARP))
        look = np.sign(np.dot(left, uLOS))
        if not hasattr(meta.SCPCOA, 'SideOfTrack'):
            if look < 0:
                meta.SCPCOA.SideOfTrack = 'R'
            else:
                meta.SCPCOA.SideOfTrack = 'L'
        if not hasattr(meta.SCPCOA, 'SlantRange'):
            meta.SCPCOA.SlantRange = np.linalg.norm(SCP - ARP)
        if not hasattr(meta.SCPCOA, 'GroundRange'):
            meta.SCPCOA.GroundRange = (np.linalg.norm(SCP) *
                                       np.arccos(np.dot(ARP, SCP) /
                                                 (np.linalg.norm(SCP) * np.linalg.norm(ARP))))
        if not hasattr(meta.SCPCOA, 'DopplerConeAng'):
            # Doppler Cone Angle is angle of slant range vector from velocity vector
            meta.SCPCOA.DopplerConeAng = np.rad2deg(np.arccos(np.dot((
                ARP_v / np.linalg.norm(ARP_v)), uLOS)))
        # Earth Tangent Plane (ETP) at the SCP is the plane tangent to the
        # surface of constant height above the WGS 84 ellipsoid (HAE) that
        # contains the SCP. The ETP is an approximation to the ground plane at
        # the SCP.
        ETP = gc.wgs_84_norm(SCP)
        if not hasattr(meta.SCPCOA, 'GrazeAng'):
            # Angle between ground plane and line-of-site vector
            meta.SCPCOA.GrazeAng = np.rad2deg(np.arcsin(np.dot(ETP, -uLOS)))
        if not hasattr(meta.SCPCOA, 'IncidenceAng'):
            # Angle between ground plane normal and line-of-site vector
            # meta.SCPCOA.IncidenceAng = np.rad2deg(np.arcsin(np.dot(ETP, -uLOS)))
            meta.SCPCOA.IncidenceAng = 90 - meta.SCPCOA.GrazeAng
        # Instantaneous slant plane unit normal at COA (also called uSPZ in SICD spec)
        spn = look * np.cross(ARP_v, uLOS)
        spn = spn/np.linalg.norm(spn)
        # Project range vector (from SCP toward ARP) onto ground plane
        uGPX = -uLOS - np.dot(ETP, -uLOS) * ETP
        uGPX = uGPX/np.linalg.norm(uGPX)
        if not hasattr(meta.SCPCOA, 'TwistAng'):
            # 1) Equations from SICD spec:
            uGPY = np.cross(ETP, uGPX)
            # Angle from the +GPY axis and to the +SPY axis in the plane of incidence
            meta.SCPCOA.TwistAng = -np.rad2deg(np.arcsin(np.dot(uGPY, spn)))
            # 2) Another implementation (seems to turn out exactly the same):
            # meta.SCPCOA.TwistAng = asind(cross(ETP, spn) * (-uLOS) / norm(cross((-uLOS), ETP)));
        if not hasattr(meta.SCPCOA, 'SlopeAng'):
            # Angle between slant and ground planes
            meta.SCPCOA.SlopeAng = np.rad2deg(np.arccos(np.dot(ETP, spn)))
        north_ground = [0, 0, 1] - np.dot(ETP, [0, 0, 1]) * ETP  # Project north onto ground plane
        # Unit vector in ground plane in north direction
        uNORTH = north_ground/np.linalg.norm(north_ground)
        uEAST = np.cross(uNORTH, ETP)  # Unit vector in ground plane in east direction
        if not hasattr(meta.SCPCOA, 'AzimAng'):
            # Component of ground-projected range vector in north direction
            az_north = np.dot(uGPX, uNORTH)
            # Component of ground-projected range vector in east direction
            az_east = np.dot(uGPX, uEAST)
            meta.SCPCOA.AzimAng = np.arctan2(az_east, az_north)
            # Assure in [0,360], not [-pi,pi]
            meta.SCPCOA.AzimAng = np.remainder(meta.SCPCOA.AzimAng*180/np.pi, 360)
        if not hasattr(meta.SCPCOA, 'LayoverAng'):
            # Layover direction in ground plane
            layover_ground = ETP - (1 / np.dot(ETP, spn)) * spn
            lo_north = np.dot(layover_ground, uNORTH)  # Component of layover in north direction
            lo_east = np.dot(layover_ground, uEAST)  # Component of layover in east direction
            meta.SCPCOA.LayoverAng = np.arctan2(lo_east, lo_north)
            # Assure in [0,360], not [-pi,pi]
            meta.SCPCOA.LayoverAng = np.remainder(meta.SCPCOA.LayoverAng*180/np.pi, 360)
        # Compute IFP specific parameters (including Row/Col.UVectECF) here
        SPEED_OF_LIGHT = 299792458.
        if (hasattr(meta, 'ImageFormation') and
           hasattr(meta.ImageFormation, 'ImageFormAlgo') and
           hasattr(meta.ImageFormation.ImageFormAlgo, 'upper')):
            # We will need these structures for all IFP types
            if not hasattr(meta, 'Grid'):
                meta.Grid = MetaNode()
            if not hasattr(meta.Grid, 'Row'):
                meta.Grid.Row = MetaNode()
            if not hasattr(meta.Grid, 'Col'):
                meta.Grid.Col = MetaNode()
            if meta.ImageFormation.ImageFormAlgo.upper() == 'RGAZCOMP':
                # In theory, we could even derive Grid.TimeCOAPoly for the RGAZCOMP
                # case if IPPPoly was include, since it must just  be the time
                # computed for the vector index: v_coa = (1/2) * (v_ps + v_pe)
                # DERIVED: RGAZCOMP image formation must result in a SLANT, RGAZIM grid
                if not hasattr(meta.Grid, 'ImagePlane'):
                    meta.Grid.ImagePlane = 'SLANT'
                if not hasattr(meta.Grid, 'Type'):
                    meta.Grid.Type = 'RGAZIM'
                # DERIVED: RgAzComp.AzSF
                if not hasattr(meta, 'RgAzComp'):
                    meta.RgAzComp = MetaNode()
                if not hasattr(meta.RgAzComp, 'AzSF'):
                    meta.RgAzComp.AzSF = (-look * np.sin(np.deg2rad(meta.SCPCOA.DopplerConeAng)) /
                                          meta.SCPCOA.SlantRange)
                # DERIVED: RgAzComp.KazPoly
                if (hasattr(meta, 'Timeline') and
                        hasattr(meta.Timeline, 'IPP') and
                        hasattr(meta.Timeline.IPP, 'Set') and
                        isinstance(meta.Timeline.IPP.Set, MetaNode) and
                        hasattr(meta.Timeline.IPP.Set, 'IPPPoly') and
                        hasattr(meta.Grid.Row, 'KCtr') and
                        not hasattr(meta.RgAzComp, 'KazPoly')):
                    krg_coa = meta.Grid.Row.KCtr
                    if hasattr(meta.Grid.Row, 'DeltaKCOAPoly'):
                        krg_coa = krg_coa + meta.Grid.Row.DeltaKCOAPoly
                    st_rate_coa = poly.polyval(meta.SCPCOA.SCPTime,
                                               poly.polyder(meta.Timeline.IPP.Set.IPPPoly))
                    # Scale factor described in SICD spec
                    delta_kaz_per_delta_v = (look * krg_coa *
                                             (np.linalg.norm(ARP_v) *
                                              np.sin(np.deg2rad(meta.SCPCOA.DopplerConeAng)) /
                                              meta.SCPCOA.SlantRange) / st_rate_coa)
                    meta.RgAzComp.KazPoly = (delta_kaz_per_delta_v *
                                             meta.Timeline.IPP.Set.IPPPoly)
                # DERIVED: UVectECF
                if (not hasattr(meta.Grid.Row, 'UVectECF') and
                   not hasattr(meta.Grid.Col, 'UVectECF')):
                    meta.Grid.Row.UVectECF = MetaNode()
                    meta.Grid.Row.UVectECF.X = uLOS[0]
                    meta.Grid.Row.UVectECF.Y = uLOS[1]
                    meta.Grid.Row.UVectECF.Z = uLOS[2]
                    uAZ = np.cross(spn, uLOS)
                    meta.Grid.Col.UVectECF = MetaNode()
                    meta.Grid.Col.UVectECF.X = uAZ[0]
                    meta.Grid.Col.UVectECF.Y = uAZ[1]
                    meta.Grid.Col.UVectECF.Z = uAZ[2]
                # DERIVED: KCtr/DeltaKCOAPoly
                # In SICD, if the optional DeltaKCOAPoly field is omitted,
                # it is assumed to be zero. If the creator of the partial
                # SICD metadata just forgot it, or didn't know it, rather
                # than leaving the field off as an explicit declaration of
                # a zero value, the KCtr computation will be wrong if the
                # DFT was not "centered" (s_0 = s_coa and v_0 = v_coa in
                # the terminology of the SICD spec).
                if 'fc' in locals():
                    if (not hasattr(meta.Grid.Row, 'KCtr')):
                        if hasattr(meta.Grid.Row, 'DeltaKCOAPoly'):
                            # DeltaKCOAPoly populated, but not KCtr (would be odd)
                            meta.Grid.Row.KCtr = (fc * (2/SPEED_OF_LIGHT)) - \
                                meta.Grid.Row.DeltaKCOAPoly.flat[0]
                        else:  # Neither KCtr or DeltaKCOAPoly populated
                            meta.Grid.Row.KCtr = fc * (2/SPEED_OF_LIGHT)
                            # DeltaKCOAPoly assumed to be zero
                    elif not hasattr(meta.Grid.Row, 'DeltaKCOAPoly'):
                        # KCtr populated, but not DeltaKCOAPoly
                        meta.Grid.Row.DeltaKCOAPoly = (fc * (2/SPEED_OF_LIGHT)) - \
                            meta.Grid.Row.KCtr
                if (not hasattr(meta.Grid.Col, 'KCtr')):
                    meta.Grid.Col.KCtr = 0
                    if hasattr(meta.Grid.Col, 'DeltaKCOAPoly'):
                        # DeltaKCOAPoly populated, but not KCtr (would be odd)
                        meta.Grid.Col.KCtr = -meta.Grid.Col.DeltaKCOAPoly.flat[0]
                    else:  # Neither KCtr or DeltaKCOAPoly populated
                        # DeltaKCOAPoly assumed to be zero
                        pass
                elif not hasattr(meta.Grid.Col, 'DeltaKCOAPoly'):
                    # KCtr populated, but not DeltaKCOAPoly
                    meta.Grid.Col.DeltaKCOAPoly = -meta.Grid.Col.KCtr
            elif meta.ImageFormation.ImageFormAlgo.upper() == 'PFA':
                if not hasattr(meta, 'PFA'):
                    meta.PFA = MetaNode()
                # DEFAULT: RGAZIM grid is the natural result of PFA
                if set_default_values and not hasattr(meta.Grid, 'Type'):
                    meta.Grid.Type = 'RGAZIM'
                # Reasonable DEFAULT guesses for PFA parameters IPN, FPN,
                # and PolarAngRefTime
                if set_default_values and not hasattr(meta.PFA, 'IPN'):
                    meta.PFA.IPN = MetaNode()
                    if hasattr(meta.Grid, 'ImagePlane'):
                        if meta.Grid.ImagePlane == 'SLANT':
                            # Instantaneous slant plane at center of aperture
                            meta.PFA.IPN = spn[0]
                            meta.PFA.IPN = spn[1]
                            meta.PFA.IPN = spn[2]
                        elif meta.Grid.ImagePlane == 'GROUND':
                            meta.PFA.IPN = ETP[0]
                            meta.PFA.IPN = ETP[1]
                            meta.PFA.IPN = ETP[2]
                    else:  # Guess slant plane (the most common IPN) if not specified
                        meta.PFA.IPN = spn[0]
                        meta.PFA.IPN = spn[1]
                        meta.PFA.IPN = spn[2]
                if set_default_values and not hasattr(meta.PFA, 'FPN'):
                    meta.PFA.FPN = MetaNode()
                    meta.PFA.FPN = ETP[0]
                    meta.PFA.FPN = ETP[1]
                    meta.PFA.FPN = ETP[2]
                if (hasattr(meta, 'Position') and hasattr(meta.Position, 'ARPPoly') and
                   hasattr(meta.PFA, 'PolarAngRefTime')):  # Compute exactly if possible
                    pol_ref_pos = [
                        poly.polyval(meta.PFA.PolarAngRefTime, meta.Position.ARPPoly.X),
                        poly.polyval(meta.PFA.PolarAngRefTime, meta.Position.ARPPoly.Y),
                        poly.polyval(meta.PFA.PolarAngRefTime, meta.Position.ARPPoly.Z)]
                elif set_default_values:  # DEFAULT: Otherwise guess PolarAngRefTime = SCPTime
                    pol_ref_pos = ARP
                    if hasattr(meta, 'SCPCOA') and hasattr(meta.SCPCOA, 'SCPTime'):
                        meta.PFA.PolarAngRefTime = meta.SCPCOA.SCPTime
                # TODO: PolarAngPoly, SpatialFreqSFPoly
                if (hasattr(meta.PFA, 'IPN') and hasattr(meta.PFA, 'FPN') and
                   not hasattr(meta.Grid.Row, 'UVectECF') and
                   not hasattr(meta.Grid.Col, 'UVectECF')):
                    ipn = np.array([meta.PFA.IPN.X, meta.PFA.IPN.Y, meta.PFA.IPN.Z])
                    fpn = np.array([meta.PFA.FPN.X, meta.PFA.FPN.Y, meta.PFA.FPN.Z])
                    # Row.UVect should be the range vector at zero polar
                    # angle time projected into IPN
                    # Projection of a point along a given direction to a
                    # plane is just the intersection of the line defined by
                    # that point (l0) and direction (l) and the plane
                    # defined by a point in the plane (p0) and the normal
                    # (p):
                    # l0 + ((l0 - p0).p/(l.p))*l
                    # where . represents the dot product.
                    # Distance from point to plane in line_direction:
                    d = np.dot((SCP - pol_ref_pos), ipn) / np.dot(fpn, ipn)
                    ref_pos_ipn = pol_ref_pos + (d * fpn)
                    uRG = SCP - ref_pos_ipn
                    uRG = uRG/np.linalg.norm(uRG)
                    uAZ = np.cross(ipn, uRG)
                    meta.Grid.Row.UVectECF = MetaNode()
                    meta.Grid.Row.UVectECF.X = uRG[0]
                    meta.Grid.Row.UVectECF.Y = uRG[1]
                    meta.Grid.Row.UVectECF.Z = uRG[2]
                    meta.Grid.Col.UVectECF = MetaNode()
                    meta.Grid.Col.UVectECF.X = uAZ[0]
                    meta.Grid.Col.UVectECF.Y = uAZ[1]
                    meta.Grid.Col.UVectECF.Z = uAZ[2]
                # DEFAULT value. Almost always zero for PFA
                if set_default_values and not hasattr(meta.Grid.Col, 'KCtr'):
                    meta.Grid.Col.KCtr = 0
                    # Sometimes set to a nonzero (PFA.Kaz1 + PFA.Kaz2)/2
                if set_default_values and not hasattr(meta.Grid.Row, 'KCtr'):
                    if hasattr(meta.PFA, 'Krg1') and hasattr(meta.PFA, 'Krg2'):
                        # DEFAULT: The most reasonable way to compute this
                        meta.Grid.Row.KCtr = (meta.PFA.Krg1 + meta.PFA.Krg2)/2
                    elif 'fc' in locals():
                        # APPROXIMATION: This may not be quite right, due
                        # to rectangular inscription loss in PFA, but it
                        # should be close.
                        meta.Grid.Row.KCtr = (fc * (2/SPEED_OF_LIGHT) *
                                              meta.PFA.SpatialFreqSFPoly[0])
            elif meta.ImageFormation.ImageFormAlgo.upper() == 'RMA':
                if hasattr(meta, 'RMA') and hasattr(meta.RMA, 'ImageType'):
                    rmatype = meta.RMA.ImageType.upper()
                    # RMAT/RMCR cases
                    if rmatype in ('RMAT', 'RMCR'):
                        if set_default_values:
                            if not hasattr(meta.Grid, 'ImagePlane'):
                                meta.Grid.ImagePlane = 'SLANT'
                            if not hasattr(meta.Grid, 'Type'):
                                # DEFAULT: XCTYAT grid is the natural result of RMA/RMAT
                                if rmatype == 'RMAT':
                                    meta.Grid.Type = 'XCTYAT'
                                # DEFAULT: XRGYCR grid is the natural result of RMA/RMCR
                                elif rmatype == 'RMCR':
                                    meta.Grid.Type = 'XRGYCR'
                            if not hasattr(meta.RMA, rmatype):
                                setattr(meta.RMA, rmatype, MetaNode())
                            # DEFAULT: Set PosRef/VelRef to SCPCOA Pos/Vel
                            rmafield = getattr(meta.RMA, rmatype)
                            if not hasattr(rmafield, 'PosRef'):
                                setattr(rmafield, 'PosRef', copy.deepcopy(meta.SCPCOA.ARPPos))
                            if not hasattr(rmafield, 'VelRef'):
                                setattr(rmafield, 'VelRef', copy.deepcopy(meta.SCPCOA.ARPVel))
                        if hasattr(meta.RMA, rmatype):
                            if (hasattr(getattr(meta.RMA, rmatype), 'PosRef') and
                               hasattr(getattr(meta.RMA, rmatype), 'VelRef')):
                                rmafield = getattr(meta.RMA, rmatype)
                                PosRef = np.array([rmafield.PosRef.X,
                                                   rmafield.PosRef.Y,
                                                   rmafield.PosRef.Z])
                                VelRef = np.array([rmafield.VelRef.X,
                                                   rmafield.VelRef.Y,
                                                   rmafield.VelRef.Z])
                                # Range unit vector
                                uLOS = (SCP - PosRef)/np.linalg.norm(SCP - PosRef)
                                left = np.cross(PosRef/np.linalg.norm(PosRef),
                                                VelRef/np.linalg.norm(VelRef))
                                look = np.sign(np.dot(left, uLOS))
                                # DCA is a DERIVED field
                                if not hasattr(rmafield, 'DopConeAngRef'):
                                    rmafield.DopConeAngRef = np.rad2deg(np.arccos(
                                        np.dot(VelRef/np.linalg.norm(VelRef), uLOS)))
                                # Row/Col.UVectECF are DERIVED fields
                                if (not hasattr(meta.Grid.Row, 'UVectECF') and
                                   not hasattr(meta.Grid.Col, 'UVectECF')):
                                    if rmatype == 'RMAT':
                                        # Along track unit vector
                                        uYAT = -look * (VelRef / np.linalg.norm(VelRef))
                                        spn = np.cross(uLOS, uYAT)
                                        # Reference slant plane normal
                                        spn = spn / np.linalg.norm(spn)
                                        uXCT = np.cross(uYAT, spn)  # Cross track unit vector
                                        meta.Grid.Row.UVectECF = MetaNode()
                                        meta.Grid.Row.UVectECF.X = uXCT[0]
                                        meta.Grid.Row.UVectECF.Y = uXCT[1]
                                        meta.Grid.Row.UVectECF.Z = uXCT[2]
                                        meta.Grid.Col.UVectECF = MetaNode()
                                        meta.Grid.Col.UVectECF.X = uYAT[0]
                                        meta.Grid.Col.UVectECF.Y = uYAT[1]
                                        meta.Grid.Col.UVectECF.Z = uYAT[2]
                                    elif rmatype == 'RMCR':
                                        uXRG = uLOS  # Range unit vector
                                        spn = look * np.cross(VelRef / np.linalg.norm(VelRef),
                                                              uXRG)
                                        # Reference slant plane normal
                                        spn = spn / np.linalg.norm(spn)
                                        uYCR = np.cross(spn, uXRG)  # Cross range unit vector
                                        meta.Grid.Row.UVectECF = MetaNode()
                                        meta.Grid.Row.UVectECF.X = uXRG[0]
                                        meta.Grid.Row.UVectECF.Y = uXRG[1]
                                        meta.Grid.Row.UVectECF.Z = uXRG[2]
                                        meta.Grid.Col.UVectECF = MetaNode()
                                        meta.Grid.Col.UVectECF.X = uYCR[0]
                                        meta.Grid.Col.UVectECF.Y = uYCR[1]
                                        meta.Grid.Col.UVectECF.Z = uYCR[2]
                        # DEFAULT: RMAT/RMCR Row/Col.KCtr
                        if set_default_values and 'fc' in locals():
                            k_f_c = fc * (2/SPEED_OF_LIGHT)
                            if rmatype == 'RMAT' and hasattr(meta.RMA.RMAT, 'DopConeAngRef'):
                                if not hasattr(meta.Grid.Row, 'KCtr'):
                                    meta.Grid.Row.KCtr = k_f_c * \
                                        np.sin(np.deg2rad(meta.RMA.RMAT.DopConeAngRef))
                                if not hasattr(meta.Grid.Col, 'KCtr'):
                                    meta.Grid.Col.KCtr = k_f_c * \
                                        np.cos(np.deg2rad(meta.RMA.RMAT.DopConeAngRef))
                            elif rmatype == 'RMCR':
                                if not hasattr(meta.Grid.Row, 'KCtr'):
                                    meta.Grid.Row.KCtr = k_f_c
                                if not hasattr(meta.Grid.Col, 'KCtr'):
                                    meta.Grid.Col.KCtr = 0
                    # INCA
                    elif rmatype == 'INCA' and hasattr(meta.RMA, 'INCA'):
                        # DEFAULT: RGZERO grid is the natural result of RMA/INCA
                        if not hasattr(meta.Grid, 'Type'):
                            meta.Grid.Type = 'RGZERO'
                        if (hasattr(meta.RMA.INCA, 'TimeCAPoly') and
                           hasattr(meta, 'Position') and
                           hasattr(meta.Position, 'ARPPoly')):
                            # INCA UVects are DERIVED from closest approach
                            # position/velocity, not center of aperture
                            ca_pos = [poly.polyval(meta.RMA.INCA.TimeCAPoly[0],
                                                   meta.Position.ARPPoly.X),
                                      poly.polyval(meta.RMA.INCA.TimeCAPoly[0],
                                                   meta.Position.ARPPoly.Y),
                                      poly.polyval(meta.RMA.INCA.TimeCAPoly[0],
                                                   meta.Position.ARPPoly.Z)]
                            ca_vel = [poly.polyval(meta.RMA.INCA.TimeCAPoly[0],
                                                   poly.polyder(meta.Position.ARPPoly.X)),
                                      poly.polyval(meta.RMA.INCA.TimeCAPoly[0],
                                                   poly.polyder(meta.Position.ARPPoly.Y)),
                                      poly.polyval(meta.RMA.INCA.TimeCAPoly[0],
                                                   poly.polyder(meta.Position.ARPPoly.Z))]
                            if not hasattr(meta.RMA.INCA, 'R_CA_SCP'):
                                meta.RMA.INCA.R_CA_SCP = np.linalg.norm(ca_pos-SCP)
                            if (((not hasattr(meta.Grid, 'Row') or
                               not hasattr(meta.Grid.Row, 'UVectECF')) and
                               (not hasattr(meta.Grid, 'Col') or
                               not hasattr(meta.Grid.Col, 'UVectECF')))):
                                # Range unit vector
                                uRG = (SCP - ca_pos)/np.linalg.norm(SCP - ca_pos)
                                left = np.cross(ca_pos/np.linalg.norm(ca_pos),
                                                ca_vel/np.linalg.norm(ca_pos))
                                look = np.sign(np.dot(left, uRG))
                                spn = -look * np.cross(uRG, ca_vel)
                                spn = spn/np.linalg.norm(spn)  # Slant plane unit normal
                                uAZ = np.cross(spn, uRG)
                                meta.Grid.Row.UVectECF = MetaNode()
                                meta.Grid.Row.UVectECF.X = uRG[0]
                                meta.Grid.Row.UVectECF.Y = uRG[1]
                                meta.Grid.Row.UVectECF.Z = uRG[2]
                                meta.Grid.Col.UVectECF = MetaNode()
                                meta.Grid.Col.UVectECF.X = uAZ[0]
                                meta.Grid.Col.UVectECF.Y = uAZ[1]
                                meta.Grid.Col.UVectECF.Z = uAZ[2]
                        # DERIVED: Always the case for INCA
                        if not hasattr(meta.Grid.Col, 'KCtr'):
                            meta.Grid.Col.KCtr = 0
                        # DEFAULT: The frequency used for computing Doppler
                        # Centroid values is often the center transmitted
                        # frequency.
                        if (set_default_values and hasattr(meta, 'RadarCollection') and
                           hasattr(meta.RadarCollection, 'TxFrequency') and
                           hasattr(meta.RadarCollection.TxFrequency, 'Min') and
                           hasattr(meta.RadarCollection.TxFrequency, 'Max') and
                           not hasattr(meta.RMA.INCA, 'FreqZero')):
                            meta.RMA.INCA.FreqZero = (meta.RadarCollection.TxFrequency.Min +
                                                      meta.RadarCollection.TxFrequency.Max)/2
                        # Row.KCtr/FreqZero DERIVED relationship is exact
                        # (although FreqZero may be set to default above.)
                        if hasattr(meta.RMA.INCA, 'FreqZero'):
                            if not hasattr(meta.Grid.Row, 'KCtr'):
                                meta.Grid.Row.KCtr = meta.RMA.INCA.FreqZero * 2 / SPEED_OF_LIGHT
    # DERIVED: Add corners coords if they don't already exist
    if (not hasattr(meta, 'GeoData')) or (not hasattr(meta.GeoData, 'ImageCorners')):
        try:
            update_corners(meta)
        except:
            pass
    # DERIVED: Add ValidData geocoords
    if (hasattr(meta, 'ImageData') and hasattr(meta.ImageData, 'ValidData') and
       ((not hasattr(meta, 'GeoData')) or (not hasattr(meta.GeoData, 'ValidData')))):
        if not hasattr(meta, 'GeoData'):
            meta.GeoData = MetaNode()
        if not hasattr(meta.GeoData, 'ValidData'):
            meta.GeoData.ValidData = MetaNode()
        if not hasattr(meta.GeoData.ValidData, 'Vertex'):
            meta.GeoData.ValidData.Vertex = [None]*len(meta.ImageData.ValidData.Vertex)
        try:
            for i in range(len(meta.ImageData.ValidData.Vertex)):
                meta.GeoData.ValidData.Vertex[i] = MetaNode()
                valid_latlon = point.image_to_ground_geo(
                    [meta.ImageData.ValidData.Vertex[i].Row,
                     meta.ImageData.ValidData.Vertex[i].Col], meta)[0]
                meta.GeoData.ValidData.Vertex[i].Lat = valid_latlon[0]
                meta.GeoData.ValidData.Vertex[i].Lon = valid_latlon[1]
        except:
            pass
    # Its difficult to imagine a scenario where GeoData.ValidData would be
    # populated, but ImageData.ValidData would not, so we don't handle deriving
    # the other direction.  Also, since HAE is not provided with each Vertex,
    # and since the height used could have been a constant height across image
    # area or from an external DEM, its not clear that there is a precise way
    # to do this.
    # DERIVED: Radiometric parameters RCS, sigma_0, gamma, beta, can be derived from each other
    if (hasattr(meta, 'Radiometric') and hasattr(meta, 'Grid')):
        # Calculate slant plane area
        if hasattr(meta.Grid.Row, 'WgtFunct'):
            rng_wght_f = np.mean(meta.Grid.Row.WgtFunct**2) \
                       / (np.mean(meta.Grid.Row.WgtFunct)**2)
        else:  # If no weight in metadata SICD assumes 1.0
            rng_wght_f = 1.0

        if hasattr(meta.Grid.Col, 'WgtFunct'):
            az_wght_f = np.mean(meta.Grid.Col.WgtFunct**2) \
                       / (np.mean(meta.Grid.Col.WgtFunct)**2)
        else:  # If no weight in metadata SICD assumes 1.0
            az_wght_f = 1.0
        area_sp = (rng_wght_f*az_wght_f)/(meta.Grid.Row.ImpRespBW*meta.Grid.Col.ImpRespBW)
        # To make the implementation shorter, first use whatever is present to
        # derive the Beta poly.
        if ((not hasattr(meta.Radiometric, 'BetaZeroSFPoly')) and
           (hasattr(meta.Radiometric, 'RCSSFPoly'))):
                meta.Radiometric.BetaZeroSFPoly = (
                                 meta.Radiometric.RCSSFPoly / area_sp)
        elif ((not hasattr(meta.Radiometric, 'BetaZeroSFPoly')) and
              (hasattr(meta.Radiometric, 'SigmaZeroSFPoly'))):
                meta.Radiometric.BetaZeroSFPoly = (
                                 meta.Radiometric.SigmaZeroSFPoly /
                                 np.cos(meta.SCPCOA.SlopeAng*np.pi/180))
        elif ((not hasattr(meta.Radiometric, 'BetaZeroSFPoly')) and
              (hasattr(meta.Radiometric, 'GammaZeroSFPoly'))):
                meta.Radiometric.BetaZeroSFPoly = (
                                 meta.Radiometric.GammaZeroSFPoly *
                                 np.sin(meta.SCPCOA.GrazeAng*np.pi/180) /
                                 np.cos(meta.SCPCOA.SlopeAng*np.pi/180))
        # Now use the Beta poly to derive the other (if empty) fields.
        if hasattr(meta.Radiometric, 'BetaZeroSFPoly'):
            if not hasattr(meta.Radiometric, 'RCSSFPoly'):
                meta.Radiometric.RCSSFPoly = (
                                 meta.Radiometric.BetaZeroSFPoly * area_sp)
            if not hasattr(meta.Radiometric, 'SigmaZeroSFPoly'):
                meta.Radiometric.SigmaZeroSFPoly = (
                                 meta.Radiometric.BetaZeroSFPoly *
                                 np.cos(meta.SCPCOA.SlopeAng*np.pi/180))
            if not hasattr(meta.Radiometric, 'GammaZeroSFPoly'):
                meta.Radiometric.GammaZeroSFPoly = (
                                 meta.Radiometric.BetaZeroSFPoly /
                                 np.sin(meta.SCPCOA.GrazeAng*np.pi/180) *
                                 np.cos(meta.SCPCOA.SlopeAng*np.pi/180))


def update_corners(meta):
    """Add corner coords to SICD metadata if they can be computed from other metadata."""
    if not hasattr(meta, 'GeoData'):
        meta.GeoData = MetaNode()
    if not hasattr(meta.GeoData, 'ImageCorners'):
        meta.GeoData.ImageCorners = MetaNode()
    corner_latlons = point.image_to_ground_geo(
        [[0, 0],
         [0, meta.ImageData.NumCols-1],
         [meta.ImageData.NumRows-1, meta.ImageData.NumCols-1],
         [meta.ImageData.NumRows-1, 0]], meta)
    if not hasattr(meta.GeoData.ImageCorners, 'FRFC'):
        meta.GeoData.ImageCorners.FRFC = MetaNode()
    meta.GeoData.ImageCorners.FRFC.Lat = corner_latlons[0, 0]
    meta.GeoData.ImageCorners.FRFC.Lon = corner_latlons[0, 1]
    if not hasattr(meta.GeoData.ImageCorners, 'FRLC'):
        meta.GeoData.ImageCorners.FRLC = MetaNode()
    meta.GeoData.ImageCorners.FRLC.Lat = corner_latlons[1, 0]
    meta.GeoData.ImageCorners.FRLC.Lon = corner_latlons[1, 1]
    if not hasattr(meta.GeoData.ImageCorners, 'LRLC'):
        meta.GeoData.ImageCorners.LRLC = MetaNode()
    meta.GeoData.ImageCorners.LRLC.Lat = corner_latlons[2, 0]
    meta.GeoData.ImageCorners.LRLC.Lon = corner_latlons[2, 1]
    if not hasattr(meta.GeoData.ImageCorners, 'LRFC'):
        meta.GeoData.ImageCorners.LRFC = MetaNode()
    meta.GeoData.ImageCorners.LRFC.Lat = corner_latlons[3, 0]
    meta.GeoData.ImageCorners.LRFC.Lon = corner_latlons[3, 1]


def apply_ref_freq(sicd_meta, ref_freq):
    """Adjust all of the fields possibly affected by RadarCollection.RefFreqIndex"""
    if hasattr(sicd_meta, 'RadarCollection'):
        if hasattr(sicd_meta.RadarCollection, 'TxFrequency'):
            if hasattr(sicd_meta.RadarCollection.TxFrequency, 'Min'):
                sicd_meta.RadarCollection.TxFrequency.Min = \
                    sicd_meta.RadarCollection.TxFrequency.Min + ref_freq
            if hasattr(sicd_meta.RadarCollection.TxFrequency, 'Max'):
                sicd_meta.RadarCollection.TxFrequency.Max = \
                    sicd_meta.RadarCollection.TxFrequency.Max + ref_freq
        if (hasattr(sicd_meta.RadarCollection, 'Waveform') and
                hasattr(sicd_meta.RadarCollection.Waveform, 'WFParameters')):
            if isinstance(sicd_meta.RadarCollection.Waveform.WFParameters, list):  # Ugly
                for i in range(len(sicd_meta.RadarCollection.Waveform.WFParameters)):
                    if hasattr(sicd_meta.RadarCollection.Waveform.WFParameters[i], 'TxFreqStart'):
                        sicd_meta.RadarCollection.Waveform.WFParameters[i].TxFreqStart = \
                            sicd_meta.RadarCollection.Waveform.WFParameters[i].TxFreqStart + \
                            ref_freq
                    if hasattr(sicd_meta.RadarCollection.Waveform.WFParameters[i], 'RcvFreqStart'):
                        sicd_meta.RadarCollection.Waveform.WFParameters[i].RcvFreqStart = \
                            sicd_meta.RadarCollection.Waveform.WFParameters[i].RcvFreqStart + \
                            ref_freq
            else:
                if hasattr(sicd_meta.RadarCollection.Waveform.WFParameters, 'TxFreqStart'):
                    sicd_meta.RadarCollection.Waveform.WFParameters.TxFreqStart = \
                        sicd_meta.RadarCollection.Waveform.WFParameters.TxFreqStart + ref_freq
                if hasattr(sicd_meta.RadarCollection.Waveform.WFParameters, 'RcvFreqStart'):
                    sicd_meta.RadarCollection.Waveform.WFParameters.RcvFreqStart = \
                        sicd_meta.RadarCollection.Waveform.WFParameters.RcvFreqStart + ref_freq
    if (hasattr(sicd_meta, 'ImageFormation') and
       hasattr(sicd_meta.ImageFormation, 'TxFrequencyProc')):
        if hasattr(sicd_meta.ImageFormation.TxFrequencyProc, 'MinProc'):
            sicd_meta.ImageFormation.TxFrequencyProc.MinProc = \
                sicd_meta.ImageFormation.TxFrequencyProc.MinProc + ref_freq
        if hasattr(sicd_meta.ImageFormation.TxFrequencyProc, 'MaxProc'):
            sicd_meta.ImageFormation.TxFrequencyProc.MaxProc = \
                sicd_meta.ImageFormation.TxFrequencyProc.MaxProc + ref_freq
    if hasattr(sicd_meta, 'Antenna'):
        if (hasattr(sicd_meta.Antenna, 'Tx') and
           hasattr(sicd_meta.Antenna.Tx, 'FreqZero')):
            sicd_meta.Antenna.Tx.FreqZero = \
                sicd_meta.Antenna.Tx.FreqZero + ref_freq
        if (hasattr(sicd_meta.Antenna, 'Rcv') and
           hasattr(sicd_meta.Antenna.Rcv, 'FreqZero')):
            sicd_meta.Antenna.Rcv.FreqZero = \
                sicd_meta.Antenna.Rcv.FreqZero + ref_freq
        if (hasattr(sicd_meta.Antenna, 'TwoWay') and
           hasattr(sicd_meta.Antenna.TwoWay, 'FreqZero')):
            sicd_meta.Antenna.TwoWay.FreqZero = \
                sicd_meta.Antenna.TwoWay.FreqZero + ref_freq
    if (hasattr(sicd_meta, 'RMA') and hasattr(sicd_meta.RMA, 'INCA') and
            hasattr(sicd_meta.RMA.INCA, 'FreqZero')):
        sicd_meta.RMA.INCA.FreqZero = sicd_meta.RMA.INCA.FreqZero + ref_freq
    sicd_meta.RadarCollection.RefFreqIndex = 0


def weight2fun(grid_rowcol):
    """Make a function from a SICD data structure description of a complex image weighting

    Input:
        grid_rowcol       Either the Grid.Row or Grid.Col SICD field depending on
                          which direction is being processed.  Should have either
                          the WgtType or WgtFunct subfields.
    Output:
        output_fun        Function that generates weighting.  Takes
                          a single input parameter, which is the number
                          of elements in the resulting weighting vector."""
    def _raised_cos(n, coef):
        N = np.arange(np.ceil(n/2.))
        w = coef - (1-coef)*np.cos(2*np.pi*N / (n-1))
        if (n % 2) == 0:
            w = np.append(w, w[::-1])
        else:
            w = np.append(w, w[-1::-1])
        return(w)

    # Taylor weighting not a function easily available in standard Python libraries,
    # so we make a quick one here.
    def _taylor_win(n, nbar=4, sll=-30):
        a = np.arccosh(10**(-sll/20))/np.pi
        # Taylor pulse widening (dilation) factor.
        sp2 = (nbar**2)/((a**2) + ((nbar-.5)**2))
        xi = (np.arange(n)-(0.5*n)+0.5)/n
        summation = 0
        n_nbar = np.arange(1, nbar)
        for m in n_nbar:
            # Calculate the cosine weights.
            num = np.prod((1 - (m**2/sp2)/(a**2+np.power(n_nbar-0.5, 2))))
            den = np.prod((1 - m**2/np.power(np.delete(n_nbar, m-1), 2)))
            f = (((-1)**(m+1))*num)/(2*den)
            summation = f*np.cos(2*np.pi*m*xi)+summation
        return 1 + 2*summation

    useWgtFunct = False
    # First try to compute function analytically
    if hasattr(grid_rowcol, 'WgtType'):
        try:  # SICD metadata is sometimes misformed
            # SICD versions <0.5 will not have the same WgtType structure.  We hope
            # that update_meta() will have fixed this structure upon ingest though.
            if grid_rowcol.WgtType.WindowName.upper() == 'UNIFORM':
                # We could do this:
                # output_fun = lambda x: np.ones(x)
                # Instead we just pass out None as a simple way to let calling
                # calling functions know that no weighting function was applied.
                output_fun = None
            elif grid_rowcol.WgtType.WindowName.upper() == 'HAMMING':
                if (not hasattr(grid_rowcol.WgtType, 'Parameter') or
                   not hasattr(grid_rowcol.WgtType.Parameter, 'value')):
                    # A Hamming window is defined in many places as a
                    # raised cosine of weight .54, so this is the default.
                    # However some data use a generalized raised cosine and
                    # call it HAMMING, so we allow for both uses.
                    coef = 0.54
                else:
                    coef = float(grid_rowcol.WgtType.Parameter.value)
                output_fun = lambda n: _raised_cos(n, coef)
            elif grid_rowcol.WgtType.WindowName.upper() == 'HANNING':
                output_fun = lambda n: _raised_cos(n, 0.5)
            elif grid_rowcol.WgtType.WindowName.upper() == 'KAISER':
                output_fun = lambda n: np.kaiser(n, float(grid_rowcol.WgtType.Parameter.value))
            elif grid_rowcol.WgtType.WindowName.upper() == 'TAYLOR':
                nbar = float([param.value for param in grid_rowcol.WgtType.Parameter
                             if (param.name).upper() == 'NBAR'][0])
                sll = float([param.value for param in grid_rowcol.WgtType.Parameter
                            if (param.name).upper() == 'SLL'][0])
                # A couple conventions for how SLL may be populated,
                # but only one sign makes sense for taylor function
                sll = -abs(sll)
                output_fun = lambda n:  _taylor_win(n, nbar, sll)/max(_taylor_win(n, nbar, sll))
            else:
                useWgtFunct = True
            if output_fun is not None:
                # Run once just to make sure the function we created doesn't throw error
                output_fun(2)
        except Exception:
            useWgtFunct = True
    else:
        useWgtFunct = True
    # If analytic approach didn't work, use sampled data
    if useWgtFunct:
        if not hasattr(grid_rowcol, 'WgtFunct'):
            raise ValueError('Insufficient metadata to determine weighting function.')
            # Alternative for calling functions, if they catch this error, is
            # to determine weighting from the complex data itself.
        else:
            # We would really like to not be dependent on SCIPY here.  Perhaps at
            # some point make our own implementation of MATLAB's interpft.
            import scipy.signal as sig
            output_fun = lambda n: sig.resample(grid_rowcol.WgtFunct, n)
    return output_fun


class MetaNode(object):
    """Empty object just used as structure.  We define nothing here except
    methods for display.

    We prefer using the object syntax to represent structures over
    dictionaries, since many interfaces will auto-complete the object
    attributes when typing, but not dictionary keywords.  A dictionary can
    always be easily be derived from Python objects using __dict__ anyway.
    We would like the MATLAB syntax of struct.('fieldname') for cleaner clode,
    which was considered by Python, but rejected, as described in PEP 363.
    """
    def __str__(self):  # For human readability
        return MetaNode._pprint_sicd_node(self)

    def __repr__(self):  # Complete string description of data structure
        # Python 2 works with or without the decode()
        return struct2xml(self).decode()  # Python 3 needs the decode()
        # Another equally valid, but less-SICD way to do this:
        # return repr(self.__dict__)

    def merge(self, newnode):  # Adds fields in new structure to current one.
        # Fields already in self will not be changed, but all unconflicting
        # fields in newnode will be added to self.
        for key, value in newnode.__dict__.items():
            if hasattr(self, key):
                if (isinstance(getattr(self, key), MetaNode) and
                   isinstance(value, MetaNode)):
                    getattr(self, key).merge(getattr(newnode, key))
            else:
                setattr(self, key, value)

    @staticmethod
    def _pprint_sicd_node(sicd_meta_node, indent_level=0):
        """Pretty print for SicdMetaNode class."""
        INDENT_SIZE = 3
        new_string = ''
        for key, value in sorted(sicd_meta_node.__dict__.items()):  # Sorts by keys
            key_str = ' ' * INDENT_SIZE * indent_level + str(key)
            if isinstance(value, list) and isinstance(value[0], MetaNode):
                for i in range(len(value)):
                    new_string += key_str
                    new_string += '\n' + MetaNode._pprint_sicd_node(value[i], indent_level+1)
            else:
                new_string += key_str
                if isinstance(value, MetaNode):
                    new_string += '\n' + MetaNode._pprint_sicd_node(value, indent_level+1)
                else:
                    str_val = str(value)
                    if len(str_val) > 200:  # Truncate line if very long
                        str_val = str_val[0:200] + '...'
                    new_string += ': ' + str_val + '\n'
        return new_string
