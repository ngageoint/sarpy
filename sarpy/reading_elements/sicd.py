
import os
import sys
from xml.etree import ElementTree

import numpy

from .base import BaseChipper, BaseReader, BaseWriter
from ..sicd_elements.SICD import SICDType


def amp_phase_conversion_function(lookup_table):
    """
    The constructs the function to convert from AMP8I_PHS8I format data to complex128 data.

    Parameters
    ----------
    lookup_table

    Returns
    -------
    callable
    """

    def converter(data):
        amp = lookup_table[data[0::2, :, :]]
        theta = data[1::2, :, :]*2*numpy.pi/256
        out = numpy.zeros((data.shape(0) / 2, data.shape(1), data.shape(2)), dtype=numpy.complex128)
        out.real = amp*numpy.cos(theta)
        out.imag = amp*numpy.sin(theta)
        return out
    return converter


class NITFOffsets(object):
    """
    SICD (versions 0.3 and above) are stored in NITF 2.1 files, but only a small,
    specific portion of the NITF format is used at all for SICD. This class
    provides a simple method to extract those limited data.
    """

    __slots__ = ('img_segment_offsets', 'img_segment_rows', 'img_segment_columns',
                 'data_ex_lengths', 'data_ex_offsets', 'is_sicd', 'sicd_meta_data')

    def __init__(self, file_name, extract_sicd=False):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF 2.1 file containing a SICD
        extract_sicd : bool
            should we extract the full sicd metadata?
        """

        with open(file_name, mode='rb') as fi:
            # NB: everything in the below assumes that this section of the file
            #   is actually ascii, and will be properly interpreted as ints and
            #   so forth. There are lots of places that may result in some kind
            #   of exception, all of which constitute a failure, and will be
            #   uncaught.

            # Read the first 9 bytes to verify NITF
            head_part = fi.read(9).decode('ascii')
            if head_part != 'NITF02.10':
                raise IOError('Note a NITF 2.1 file, and cannot contain a SICD')
            # offset to first field of interest
            fi.seek(354)
            header_length = int(fi.read(6))
            image_segment_count = int(fi.read(3))
            image_segment_subhdr_lengths = numpy.zeros((image_segment_count, ), dtype=numpy.int64)
            image_segment_data_lengths = numpy.zeros((image_segment_count, ), dtype=numpy.int64)
            # the image data in this file is packed as:
            #   header:im1_header:im1_data:im2_header:im2_data:...
            for i in range(image_segment_count):
                image_segment_subhdr_lengths[i] = int(fi.read(6))
                image_segment_data_lengths[i] = int(fi.read(10))
            # Offset to given image segment data from beginning of file in bytes
            self.img_segment_offsets = header_length + numpy.cumsum(image_segment_subhdr_lengths)
            self.img_segment_offsets[1:] += numpy.cumsum(image_segment_data_lengths[:-1])

            if int(fi.read(3)) > 0:
                raise IOError('SICD does not allow for graphics segments.')
            if int(fi.read(3)) > 0:
                raise IOError('SICD does not allow for reserved extension segments.')
            # text segments get packed next, we really only need the size to skip
            text_segment_count = int(fi.read(3))
            text_size = 0
            for i in range(text_segment_count):
                text_size += int(fi.read(4))  # text header length
                text_size += int(fi.read(5))  # text data length
            # data extensions get packed next, we need these
            data_ex_count = int(fi.read(3))
            data_ex_subhdr_lengths = numpy.zeros((data_ex_count, ), dtype=numpy.int64)
            self.data_ex_lengths = numpy.zeros((data_ex_count, ), dtype=numpy.int64)
            for i in range(data_ex_count):
                data_ex_subhdr_lengths[i] = int(fi.read(4))
                self.data_ex_lengths[i] = int(fi.read(9))
            self.data_ex_offsets = \
                self.img_segment_offsets[-1] + image_segment_data_lengths[-1] + text_size + \
                numpy.cumsum(data_ex_subhdr_lengths)
            self.data_ex_offsets[1:] += numpy.cumsum(self.data_ex_lengths[:-1])

            # Number of rows in the given image segment
            self.img_segment_rows = numpy.zeros((image_segment_count,), dtype=numpy.int32)
            # Number of cols in the given image segment
            self.img_segment_columns = numpy.zeros((image_segment_count,), dtype=numpy.int32)
            # Extract these from the respective image headers now
            for i in range(image_segment_count):
                # go to 333 bytes into the ith image header
                fi.seek(self.img_segment_offsets[i] - image_segment_subhdr_lengths[i] + 333)
                self.img_segment_rows[i] = int(fi.read(8))
                self.img_segment_columns[i] = int(fi.read(8))

            # SICD Volume 2, File Format Description, section 3.1.1 says that SICD XML
            # metadata must be stored in first Data Extension Segment.
            # TODO: How do we know that's the one that we want? What if it isn't?
            self.is_sicd = False
            self.sicd_meta_data = None

            fi.seek(self.data_ex_offsets[0])
            data_extension = fi.read(self.data_ex_lengths[0])
            try:
                root_node = ElementTree.fromstring(data_extension)
                if root_node.tag.split('}', 1)[-1] == 'SICD':
                    self.is_sicd = True
            except Exception:
                return

            if extract_sicd:
                self.sicd_meta_data = SICDType.from_node(root_node.find('SICD'))


class SICDReader(BaseReader):
    def __init__(self, file_name):
        self._nitf_offsets = NITFOffsets(file_name, extract_sicd=True)
        self._sicd_meta = self._nitf_offsets.sicd_meta_data
        self._sicd_meta.derive()  # try to fix up the structure and derive any missing attributes
        # TODO: should we check that it's valid?

        pixel_type = self._sicd_meta.ImageData.PixelType
        complex_type = True
        if pixel_type == 'RE32F_IM32F':
            dtype = numpy.float32
        elif pixel_type == 'RE16I_IM16I':
            dtype = numpy.int16
        elif pixel_type == 'AMP8I_PHS8I':
            dtype = numpy.uint8
            complex_type = amp_phase_conversion_function(self._sicd_meta.ImageData.AmpTable)
            # TODO: is this above right?
            # raise ValueError('Pixel Type `AMP8I_PHS8I` is not currently supported.')
        else:
            raise ValueError('Pixel Type {} not recognized.'.format(pixel_type))

        data_sizes = numpy.column_stack(
            (self._nitf_offsets.img_segment_rows, self._nitf_offsets.img_segment_columns), dtype=numpy.int32)
        # SICDs are required to be stored as big-endian
        swap_bytes = (sys.byteorder != 'big')
        symmetry = (False, False, False)
        # TODO: craft the chipper


class MultiSegmentChipper(BaseChipper):
    def __init__(self, data_size, symmetry=None, complex_type=None):

        # this data_size should NOT be passed straight through.
        super(MultiSegmentChipper, self).__init__(data_size, symmetry=symmetry, complex_type=False)

        pass

