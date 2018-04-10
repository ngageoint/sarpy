"""Module for reading SIO files."""

# SarPy imports
from . import Reader as ReaderSuper  # Reader superclass
from . import Writer as WriterSuper  # Writer superclass
from .utils import bip
from . import sicd
# Python standard library imports
import os.path
import warnings
import re
# External dependencies
import numpy as np

__classification__ = "UNCLASSIFIED"
__author__ = "Wade Schwartzkopf"
__email__ = "wschwartzkopf@integrity-apps.com"

MAGIC_NUMBERS = (int('FF017FFE', 16),
                 int('FE7F01FF', 16),
                 int('FF027FFD', 16),
                 int('FD7F02FF', 16))


def isa(filename):
    """Test to see if file is a SIO.  If so, return reader object."""
    with open(filename, mode='rb') as fid:
        if np.fromfile(fid, dtype='uint32', count=1) in MAGIC_NUMBERS:
            return Reader


class Reader(ReaderSuper):
    """Creates a file reader object for an SIO file."""
    def __init__(self, filename):
        # Read SIO itself
        ihdr, swapbytes, data_offset, user_data = read_meta(filename)
        self.sicdmeta = meta2sicd(ihdr, user_data)
        # ihdr is uint32 in file.  Without double cast, file size computations
        # over 4 Gig can overflow
        ihdr = np.array(ihdr, dtype='uint64')
        # Check that header matches filesize
        if os.path.getsize(filename) != (np.uint64(data_offset) +
                                         (ihdr[1] * ihdr[2] * ihdr[4])):
            warnings.warn(UserWarning('File appears to be SIO, ' +
                                      'but header does not match file size.'))

        # Check for CASPR metadata file and read
        caspr_filename = locate_caspr(filename)
        symmetry = (False, False, False)  # Energy from top
        if caspr_filename:
            native_metadata = read_caspr(caspr_filename)
            if ('Image Parameters' in native_metadata and
               'image illumination direction [top, left, bottom, right]' in native_metadata['Image Parameters']):
                if native_metadata['Image Parameters']['image illumination direction [top, left, bottom, right]'] == 'left':
                    symmetry = (True, False, True)
                    # Reorient size info
                    self.sicdmeta = meta2sicd(ihdr[[0, 2, 1, 3, 4]], user_data)
                elif not native_metadata['Image Parameters']['image illumination direction [top, left, bottom, right]'] == 'top':
                    ValueError('Unhandled illumination direction.')
            # TODO: Convert CASPR metadata to SICD format and merge with other SICD metadata
            # self.sicdmeta.merge(meta2sicd_caspr(native_metadata))
            if not hasattr(self.sicdmeta, 'native'):
                self.sicdmeta.native = sicd.MetaNode()
            self.sicdmeta.native.caspr = native_metadata

        # Build object
        datasize = ihdr[1:3]
        datatype, complexbool = sio2numpytype(ihdr[3], ihdr[4])
        self.read_chip = bip.Chipper(filename, datasize, datatype, complexbool,
                                     data_offset, swapbytes, symmetry,
                                     bands_ip=1)


class Writer(WriterSuper):
    def __init__(self, filename, sicdmeta):
        # SIO format allows for either endianness, but we just pick it arbitrarily.
        ENDIAN = '>'

        self.filename = filename
        self.sicdmeta = sicdmeta
        if (hasattr(sicdmeta, 'ImageData') and
           hasattr(sicdmeta.ImageData, 'PixelType')):
            if sicdmeta.ImageData.PixelType == 'RE32F_IM32F':
                datatype = np.dtype(ENDIAN + 'f4')
                element_type = 13
                element_length = 8
            elif sicdmeta.ImageData.PixelType == 'RE16I_IM16I':
                datatype = np.dtype(ENDIAN + 'i2')
                element_type = 12
                element_length = 4
            elif sicdmeta.ImageData.PixelType == 'AMP8I_PHS8I':
                datatype = np.dtype(ENDIAN + 'u1')
                element_type = 11
                element_length = 2
                raise(ValueError('AMP8I_PHS8I is currently an unsupported pixel type.'))
            else:
                raise(ValueError('PixelType must be RE32F_IM32F, RE16I_IM16I, or AMP8I_PHS8I.'))
        else:
            sicdmeta.ImageData.PixelType = 'RE32F_IM32F'
            datatype = np.dtype(ENDIAN + 'f4')
        # Write header
        with open(filename, mode='w') as fid:
            header = np.array([MAGIC_NUMBERS[0],
                               sicdmeta.ImageData.NumRows,
                               sicdmeta.ImageData.NumCols,
                               element_type,
                               element_length], dtype=np.dtype(ENDIAN + 'u4'))
            header.tofile(fid)
        # Setup pixel writing
        image_size = (sicdmeta.ImageData.NumRows, sicdmeta.ImageData.NumCols)
        self.write_chip = bip.Writer(filename, image_size, datatype, True, 20)


def read_meta(filename):
    """Parse SIO header."""
    with open(filename, mode='rb') as fid:
        ihdr = np.fromfile(fid, dtype='uint32', count=5)
        if ihdr[0] == MAGIC_NUMBERS[0]:  # File is the same endian as our file IO
            data_offset = 20
            swapbytes = False
            user_data = {}
        elif ihdr[0] == MAGIC_NUMBERS[1]:  # File is different endian
            data_offset = 20
            swapbytes = True
            ihdr = ihdr.byteswap()
            user_data = {}
        elif ihdr[0] == MAGIC_NUMBERS[2]:  # Same endian, with user data
            swapbytes = False
            user_data, user_data_length = _read_userdata(fid, swapbytes)
            data_offset = 20 + user_data_length
        elif ihdr[0] == MAGIC_NUMBERS[3]:  # Different endian, with user data
            swapbytes = True
            ihdr = ihdr.byteswap()
            user_data, user_data_length = _read_userdata(fid, swapbytes)
            data_offset = 20 + user_data_length
        else:  # Not an SIO file
            ihdr = swapbytes = data_offset = user_data = None

        return ihdr, swapbytes, data_offset, user_data


def _read_userdata(fid, swapbytes):
    """Extracts user data from SIO files

    Assumes that you already know the endianness and that the file has user data

    """
    num_data_pairs = np.fromfile(fid, dtype='uint32', count=1)
    if swapbytes:
        num_data_pairs.byteswap(True)
    userdata_length_in_bytes = 4  # Size in bytes of num_data_pairs
    userdata_dict = {}  # Initialize dictionary
    for i in range(num_data_pairs):
        namebytes = np.fromfile(fid, dtype='uint32', count=1)  # 4 bytes
        if swapbytes:
            namebytes.byteswap(True)
        # Switching back and forth repeatedly between NumPy np.fromfile and
        # Python fid.read apparently results in bad things happening--
        # fid.tell() gets confused, among other things.  Thus we will try to
        # force all our reads to be NumPy fromfile for consistency.
        # name = fid.read(namebytes).decode('ascii') # This causes problems
        name = np.fromfile(fid, dtype='int8', count=namebytes)  # This works
        name = ''.join(map(chr, name))  # Decode bytes to ASCII string

        valuebytes = np.fromfile(fid, dtype='uint32', count=1)  # 4 bytes
        if swapbytes:
            valuebytes.byteswap(True)
        # value = fid.read(valuebytes) # This cause problems
        value = np.fromfile(fid, dtype='int8', count=valuebytes)  # This works
        try:
            value = ''.join(map(chr, value))  # Decode bytes to ASCII string
        except UnicodeDecodeError:  # Not all userdata values are strings.
            pass  # Leave as bytes so user can cast to int, float, etc.

        userdata_dict[name] = value
        userdata_length_in_bytes = userdata_length_in_bytes + namebytes + valuebytes + 8

    return userdata_dict, userdata_length_in_bytes


def sio2numpytype(element_type, element_length):
    """Convert the SIO description of data type in a NumPy dtype."""
    iscomplex = int(element_type/10)  # Tens place is zero => real, tens place is one => complex
    if iscomplex > 1:  # Tens place is two => vector
        raise(ValueError('Vector types for SIO files are not supported by this reader.'))
    iscomplex = iscomplex > 0  # Convert to boolean
    datatypenum = element_type % 10  # 1: unsigned int, 2: signed int, 3: float
    if datatypenum == 1:
        datatype = 'uint'
    elif datatypenum == 2:
        datatype = 'int'
    elif datatypenum == 3:
        datatype = 'float'
    else:
        raise(ValueError('Reader only recognizes unsigned and signed integers and floats.'))
    datalength = element_length * 8
    if(iscomplex):
        datalength = int(datalength / 2)
    return np.dtype(datatype + str(datalength)), iscomplex


def meta2sicd(sio_hdr, user_data={}):
    """Converts SIO header into SICD-style structure

    Really not much to do here since SIO header is so minimal.  Just fill in
    image size and datatype.

    """

    # Check to see if we have a SICD-in-SIO type file.  If there's a SICDMETA
    # field in the user_data structure we'll assume that has all the valid
    # metadata for this file and we'll use that.  Otherwise we'll just return a
    # small 'stub' SICD metadata structure (not much information in the SIO
    # header).
    if 'SICDMETA' in user_data:  # user_data should be dictionary
        # We assume this condition is rarely used, so we put include inside here
        from xml.dom.minidom import parseString
        return sicd.sicdxml2struct(parseString(user_data['SICDMETA']))
    else:
        sicdstruct = sicd.MetaNode()
        sicdstruct.ImageData = sicd.MetaNode()
        sicdstruct.ImageData.NumRows = sio_hdr[1]
        sicdstruct.ImageData.NumCols = sio_hdr[2]
        # Assume full image, but no way to know for sure
        sicdstruct.ImageData.FullImage = sicd.MetaNode()
        sicdstruct.ImageData.FullImage.NumRows = sio_hdr[1]
        sicdstruct.ImageData.FullImage.NumCols = sio_hdr[2]
        sicdstruct.ImageData.FirstRow = int(0)
        sicdstruct.ImageData.FirstCol = int(0)
        if (sio_hdr[3] == 13) and (sio_hdr[4] == 8):
            sicdstruct.ImageData.PixelType = 'RE32F_IM32F'
        elif (sio_hdr[3] == 12) and (sio_hdr[4] == 4):
            sicdstruct.ImageData.PixelType = 'RE16I_IM16I'
        # Not usually given explicitly in accompanying metadata, so just guess center
        sicdstruct.ImageData.SCPPixel = sicd.MetaNode()
        sicdstruct.ImageData.SCPPixel.Row = int(sicdstruct.ImageData.NumRows/2)
        sicdstruct.ImageData.SCPPixel.Col = int(sicdstruct.ImageData.NumCols/2)
        sicdstruct.native = sicd.MetaNode()
        sicdstruct.native.sio = user_data
        return sicdstruct


# Everything below handles CASPR files, the metadata files that often accompany SIOs
def locate_caspr(filename):
    """Locate CASPR metadata file that might be associated with an SIO file."""
    path, extension = os.path.splitext(filename)
    path, name = os.path.split(path)
    possible_filenames = (os.path.join(path, name + '.hydra'),
                          os.path.join(path, name + '.hdr'),
                          os.path.join(path, '..', 'RPHDHeader.out'),
                          os.path.join(path, name + '_RPHDHeader.out'))
    for i in possible_filenames:
        if os.path.exists(i) and os.path.isfile(i):
            return i


def read_caspr(filename):
    """Read metadata from CASPR header."""
    with open(filename, mode='r') as fid:
        current_subfield = ''
        reading_subfield = False
        metadata = {}
        for linetoparse in fid:
            linetoparse = linetoparse.rstrip('\n')
            if linetoparse != '' and not linetoparse.startswith(' '):  # Not empty
                if linetoparse.startswith(';;;'):
                    reading_subfield = not reading_subfield
                    if reading_subfield:
                        current_subfield = ''
                    else:
                        metadata[current_subfield] = {}
                else:
                    quoted_token = re.match('"(?P<quoted>[^"]+)"', linetoparse)
                    if quoted_token:  # Some values with spaces are surrounded by quotes
                        import pdb
                        pdb.set_trace()
                        tokens = [quoted_token.group('quoted'),
                                  linetoparse[quoted_token.end('quoted')+1:].strip()]
                    else:  # No quoted values were found
                        # If not using quotes, split with whitespace
                        tokens = linetoparse.split(None, 1)
                    if (len(tokens) > 1) and tokens[1] != '':
                        if reading_subfield:  # Subsection heading
                            current_subfield = current_subfield + tokens[1]
                        elif not current_subfield == '':  # Actual field value
                            try:
                                floatval = float(tokens[0])
                                metadata[current_subfield][tokens[1]] = floatval
                            except ValueError:  # Value is string not numeric
                                metadata[current_subfield][tokens[1]] = tokens[0]
        return metadata
