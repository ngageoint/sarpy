__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"
# Written on: 2025-10
#

import numpy
import os
from pathlib import Path
import sys

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.general.base import SarpyIOError

class SIOWriter(object):
    """
    A class for writing SICD data in the Stream-oriented Input Output (SIO) 
    format.

    SIO files have a 20 byte header, followed by the SICD meta data in the user 
    data section, followed by the image data.

    The SICD metadata is written in a user data field called "SICDMETA" in the 
    standard SIO file format. The SICD metadata in xml format is prefixed with
    a Uniform Resource Name (urn) derived from the _SICD_SPEC_DETAILS and 
    _SICD_VERSION_DEFAULT located in sarpy.io.complex.sicd_elements.SICD.

    The caller can also turn off the user data, since not all SIO readers handle 
    this.

    Example:
    # Given an image with SICD meta data, write both to an SIO formatted file.
    # Within an interactive Python shell

    import numpy as np
    import struct

    from sarpy.io.complex.sicd_elements.blocks import RowColType
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.complex.sicd_elements.ImageData import ImageDataType, FullImageType
    from sarpy.io.complex.sio_processor.sio_writer import SIOWriter as SIOWriter

    example_image_data = np.arange(13*17*2, dtype=np.float32).reshape(13, 17,2)
    example_sicd_meta_data = SICDType(
        ImageData=ImageDataType(
            NumRows=example_image_data.shape[0],
                NumCols=example_image_data.shape[1],
                PixelType="RE32F_IM32F",
                FirstRow=0,
                FirstCol=0,
                FullImage=FullImageType(
                    NumRows=example_image_data.shape[0],
                    NumCols=example_image_data.shape[1]
                ),
                SCPPixel=RowColType(Row=example_image_data.shape[0] // 2, Col=example_image_data.shape[1] // 2)
        ),
    )
    output_file_name = "SIOWriterExampleOutput.sio"

    my_sio_writer = SIOWriter(str(output_file_name), example_image_data, 
        example_sicd_meta_data, param_start_indices=[0, 0])
    my_sio_writer.write()
    my_sio_writer.close()
    """
    
    def __init__(
            self,
            param_filename:              str|Path, 
            param_image_data:            numpy.array, 
            param_sicdmeta:              SICDType|None = None,
            param_start_indices:         list          = [0, 0],
            param_include_sicd_metadata: bool          = True
    ):
        """
        Parameters:

        param_filename:
            The path and filename to write the data to.

        param_image_data:
            The actual data that makes up the image stored as a numpy array.

        param_sicdmeta:
            SICD meta data in a SICDType object.

        param_start_indices:
            The indicices of the image data where you want to start writing from.
            This is used to skip over unwanted parts of the image.

        param_include_sicd_metadata: 
            A boolean value used to indicate if the SICD metadata is to be 
            included in the SIO file or not. This is used to prevent writing the
            SICD meta data when you know you will ingest it later with another
            reader that cannot process SICD data from an SIO file.
            True: Write the SICD meta data to the SIO file
            False: Do not write the SICD meta data to the SIO file.

        Returns:
        An SIOWriter object.
        """
        # Parse inputs
        self._filename               = param_filename
        if isinstance(self._filename, str):
            self._filename = Path(self._filename)
            if os.path.dirname(self._filename) == '':
                self._filename = Path.cwd() / self._filename
        if not isinstance(self._filename, Path):
            raise TypeError('Filename must be a pathlib.Path or string')
        self._image_data             = param_image_data
        if param_sicdmeta is not None:
            self._sicdmeta_xml_bytes = param_sicdmeta.to_xml_bytes()
        else:
            self._sicdmeta_xml_bytes = None
        self._start_indices          = param_start_indices
        self._include_sicd_metadata  = param_include_sicd_metadata
        
        # The default SIO header is 20 bytes
        # It is comprised of 5 uint32 words, which are 4 bytes
        # Determine the endianness of the system
        self._endianness = sys.byteorder
        # Check dtype of _image_data numpy array incase the endianness was set  
        # differently than the system default.
        if self._image_data.dtype.byteorder == '<':
            self._endianness = 'little'
        if self._image_data.dtype.byteorder == '>':
            self._endianness = 'big'
        if self._sicdmeta_xml_bytes is not None and self._include_sicd_metadata:
            if self._endianness == "big":
                self._magic_key   = 0xFF027FFD # Indicates big endian, with user-data
            else:
                self._magic_key   = 0xFD7F02FF # Indicates little endian, with user-data
        else:
            if self._endianness == "big":
                self._magic_key  = 0xFF017FFE # Indicates big endian, with no user-data
            else :
                self._magic_key  = 0xFE7F01FF # Indicates little endian, with no user-data
        self._image_shape    = self._image_data.shape
        
        # Data type and size
        self._numpy_data_type_to_sio()
        
        if not os.path.exists(os.path.dirname(self._filename)):
            raise SarpyIOError('Path {} is not a file'.format(self._filename))
        self._fid = open(self._filename, 'wb')
        
    
    def _numpy_data_type_to_sio(self):
        """
        Private function: Given the numpy data type from the _image_data, set the data type code and size.
        """
        match self._image_data.dtype.name:
            case 'uint8':
                self._data_type_code = 1
                self._data_size      = 1
            case 'int16':
                self._data_type_code = 2
                self._data_size      = 2
            case 'float32':
                self._data_type_code = 3
                self._data_size      = 4
            case 'complex64':
                self._data_type_code = 13
                self._data_size      = 8
            case _ : #Default if other cases don't match
                raise TypeError('Writer only recognizes floats, complex and signed or unsigned integers')
                
    def write(self):
        """
        Write _image_data and meta data to the file indicated by _filename
        
        Fixed definition SIO header with (possibly) one user-data segment.
        Header will the look like this:
            Core SIO header:
               Magic key   (4 byte uint, either 0xFF027FFD or 0xFF017FFE)
               Rows        (4 byte uint)
               Columns     (4 byte uint)
               Data type   (4 byte uint)
               Data size   (4 byte uint, # bytes per element)
            Optional "user data":
               Num pairs   (4 byte uint, # pairs of user data, fixed at 1)
               Name bytes  (4 byte uint, # bytes in name of user element,
                            fixed at 8 for name "SICDMETA")
               Name        (8 bytes containing "SICDMETA")
               Value bytes (4 byte uint, value is length of XML string)
               Value       (XML string holding SICD metadata)

        Parameters:
            None

        Returns:
            The number of bytes written to file.
        """

        # Write the header in order as defined above.
        # Write the _magic_key, which is alreay in proper byte order so don't 
        # set byteorder
        self._fid.write(self._magic_key.to_bytes(4))
        # For remaining numerical values, include byteorder indicator to 
        # properly format the endianness to be consistent with the endianness
        # of the _image_data numpy array.
        # Rows and columns
        self._fid.write(self._image_shape[0].to_bytes(4, byteorder=self._endianness))
        self._fid.write(self._image_shape[1].to_bytes(4, byteorder=self._endianness))
        # Data type information
        self._fid.write(self._data_type_code.to_bytes(4, byteorder=self._endianness))
        self._fid.write(self._data_size.to_bytes(4, byteorder=self._endianness))
        # User data
        if self._sicdmeta_xml_bytes is not None and self._include_sicd_metadata:
            # Num pairs of user data, always 1 because we only write 1 SICD at a time.
            self._fid.write((1).to_bytes(4, byteorder=self._endianness))
            self._fid.write((8).to_bytes(4, byteorder=self._endianness)) # Name length
            self._fid.write('SICDMETA'.encode('utf-8'))      # Always SICDMEATA
            self._fid.write(len(self._sicdmeta_xml_bytes).to_bytes(4, byteorder=self._endianness)) 
            self._fid.write(self._sicdmeta_xml_bytes)  

        num_bytes_written = self._fid.write(self._image_data.tobytes())
        return num_bytes_written
    
    def close(self):
        """
        Close the file we opened to write to.
        """
        self._fid.close()