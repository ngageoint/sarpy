__classification__ = "UNCLASSIFIED"
__author__ = "Tex Peterson"
# Written on: 2025-10
#

import numpy
import os

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.general.base import SarpyIOError

class SIOReader(object):
    """
    A class for reading SICD data from the Stream-oriented Input Output (SIO) 
    format.

    SIO files have a 20 byte header, possibly followed by the SICD meta data in 
    the user data section, followed by the image data.

    Example:
    # Given an SIO file.
    # Within an interactive Python shell

    from sarpy.io.complex.sio_processor.sio_reader import SIOReader as SIOReader

    input_sio_file = "SIOReaderExampleOutput.sio"

    my_sio_reader = SIOReader(str(input_sio_file))

    """

    def __init__(
        self,
        param_filename
    ):
        """
        Parameters:

        param_filename:
            The path and filename to read the data from.

        Returns:
        An SIOReader object.

        """
        self._filename           =  param_filename
        if not os.path.exists(os.path.dirname(self._filename)):
            raise SarpyIOError('Path {} is not a file'.format(self._filename))
        self._fid                = open(self._filename, 'rb') 
        self._magic_key          = int.from_bytes(self._fid.read(4))
        byte_order = 'big'
        # We have to check the _magic_key multiple times for different scenarios
        # Check the _magic_key for little endian, and set the byte_order accordingly
        if self._magic_key in [0xFE7F01FF, 0xFD7F02FF]:
            byte_order = 'little'
        self._rows               = int.from_bytes(self._fid.read(4), byteorder=byte_order)
        self._columns            = int.from_bytes(self._fid.read(4), byteorder=byte_order)
        self._data_type_code     = int.from_bytes(self._fid.read(4), byteorder=byte_order)
        self._data_size          = int.from_bytes(self._fid.read(4), byteorder=byte_order)
        self._user_data          = None
        self._sicdmeta           = None
        # Set the data type and size from the code in the header.
        self._sio_code_to_numpy_data_type()
        if byte_order == 'little':
            self._data_type_str = '<' + self._data_type_str
        else:
            self._data_type_str = '>' + self._data_type_str
        # Magic Key indicates there is SICD meta data in the user data
        if self._magic_key in [0xFF027FFD, 0xFD7F02FF]:
            # SICD meta data header format:
            # Num pairs   (4 byte uint, # pairs of user data, fixed at 1)
            # Name bytes  (4 byte uint, # bytes in name of user element,
            #             fixed at 8 for name "SICDMETA")
            # Name        (8 bytes containing "SICDMETA")
            # Value bytes (4 byte uint, value is length of XML string)
            # Value       (XML string holding SICD metadata)
            self._num_pairs      = int.from_bytes(self._fid.read(4), byteorder=byte_order)
            user_data_name_bytes = int.from_bytes(self._fid.read(4), byteorder=byte_order)
            self._user_data_name = self._fid.read(
                user_data_name_bytes).decode("utf-8")
            self._user_data_size = int.from_bytes(self._fid.read(4), byteorder=byte_order)
            self._user_data      = self._fid.read(
                self._user_data_size).decode("utf-8")
            self._sicdmeta       = SICDType.from_xml_string(self._user_data)
        # Check to ensure the _magic_key is valid
        if self._magic_key in [0xFF017FFE, 0xFF027FFD, 0xFE7F01FF, 0xFD7F02FF]:
            if 'c4' in self._data_type_str:
                local_datatype_str = self._data_type_str[0] + 'i2'
                local_raw_data = numpy.frombuffer(self._fid.read(),
                                                     dtype=local_datatype_str
                                                     )
                real_start = 0
                imag_start = 1
                if byte_order == 'little':
                    real_start = 1
                    imag_start = 0
                local_real_part = local_raw_data[real_start::2]
                local_imag_part = local_raw_data[imag_start::2]
                local_image_data = numpy.empty(local_real_part.shape, dtype=numpy.complex64)
                local_image_data.real = local_real_part
                local_image_data.imag = local_imag_part
                self._image_data = local_image_data.reshape(self._rows,
                                                            self._columns)
            else:
                self._image_data  = numpy.frombuffer(self._fid.read(),
                                                     dtype=self._data_type_str
                                                     ).reshape(self._rows,
                                                               self._columns)
        else:
            raise SarpyIOError('Magic Key {} is not valid'.format(
                self._magic_key))

    def _sio_code_to_numpy_data_type(self):
        """
        Private function: Given the data type code from the header, set the 
        numpy data type and size.
        """
        match self._data_type_code:
            case 1:
                self._data_type_str = 'u1'
            case 2:
                self._data_type_str = 'i2'
                if self._data_size == 4:
                    self._data_type_str = 'c4'
            case 3:
                self._data_type_str = 'f4'
            case 12:
                self._data_type_str = 'c4'
            case 13:
                self._data_type_str = 'c8'
            case _ : #Default if other cases don't match
                raise TypeError('Reader only recognizes floats, complex and ' + \
                                'signed or unsigned integers')        
