# -*- coding: utf-8 -*-
"""
Functionality for reading SIO data into a SICD model.
"""

import os
import struct
import logging
import re
from typing import Union, Dict, Tuple

import numpy

from sarpy.io.complex.sicd_elements.blocks import RowColType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType, FullImageType
from sarpy.io.general.base import BaseReader
from sarpy.io.general.bip import BIPChipper, BIPWriter
from sarpy.io.complex.sicd import complex_to_amp_phase, complex_to_int, amp_phase_to_complex
from sarpy.io.general.utils import parse_xml_from_string

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Wade Schwartzkopf")


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a SIO file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    SIOReader|None
        `SIOReader` instance if SIO file, `None` otherwise
    """

    try:
        sio_details = SIODetails(file_name)
        print('File {} is determined to be a SIO file.'.format(file_name))
        return SIOReader(sio_details)
    except IOError:
        return None


###########
# parser and interpreter for hdf5 attributes

class SIODetails(object):
    __slots__ = (
        '_file_name', '_magic_number', '_head', '_user_data', '_data_offset',
        '_caspr_data', '_symmetry', '_sicd')

    # NB: there are really just two types of SIO file (with user_data and without),
    #   with endian-ness layered on top
    ENDIAN = {
        0xFF017FFE: '>', 0xFE7F01FF: '<',  # no user data
        0xFF027FFD: '>', 0xFD7F02FF: '<'}  # with user data

    def __init__(self, file_name):
        self._file_name = file_name
        self._user_data = None
        self._data_offset = 20
        self._caspr_data = None
        self._symmetry = (False, False, False)
        self._sicd = None

        if not os.path.isfile(file_name):
            raise IOError('Path {} is not a file'.format(file_name))

        with open(file_name, 'rb') as fi:
            self._magic_number = struct.unpack('I', fi.read(4))[0]
            endian = self.ENDIAN.get(self._magic_number, None)
            if endian is None:
                raise IOError('File {} is not an SIO file. Got magic number {}'.format(file_name, self._magic_number))

            # reader basic header - (rows, columns, data_type, pixel_size)?
            init_head = numpy.array(struct.unpack('{}4I'.format(endian), fi.read(16)), dtype=numpy.uint64)
            if not (numpy.all(init_head[2:] == numpy.array([13, 8]))
                    or numpy.all(init_head[2:] == numpy.array([12, 4]))
                    or numpy.all(init_head[2:] == numpy.array([11, 2]))):
                raise IOError('Got unsupported sio data type/pixel size = {}'.format(init_head[2:]))
            self._head = init_head

    @property
    def file_name(self):  # type: () -> str
        return self._file_name

    @property
    def symmetry(self):  # type: () -> Tuple[bool, bool, bool]
        return self._symmetry

    @property
    def data_offset(self):  # type: () -> int
        return self._data_offset

    @property
    def data_size(self):  # type: () -> Union[None, Tuple[int, int]]
        if self._head is None:
            return None
        rows, cols = self._head[:2]
        if self._symmetry[0]:
            return cols, rows
        else:
            return rows, cols

    @property
    def data_type(self):  # type: () -> Union[None, str]
        # head[2] = (2X = vector, 1X = complex/scalar, 0X = real/scalar), where
        #   X = (1 = unsigned int, 2 = signed int, 3 = float, (4=double? I would guess)
        # head[3] = pixel size in bytes (2*bit depth for complex, or band*bit depth for vector)
        pixel_size = self._head[3]
        # we require (for sicd) that either head[2:] is [13, 8], [12, 4], [11, 2]
        if pixel_size == 8:
            return 'float32'
        elif pixel_size == 4:
            return 'int16'
        elif pixel_size == 2:
            return 'uint8'
        else:
            raise ValueError('Got unsupported sio data type/pixel size = {}'.format(self._head[2:]))

    @property
    def pixel_type(self):  # type: () -> Union[None, str]
        if self._head[2] == 13 and self._head[3] == 8:
            return 'RE32F_IM32F'
        elif self._head[2] == 12 and self._head[3] == 4:
            return 'RE16I_IM16I'
        elif self._head[2] == 11 and self._head[3] == 2:
            return 'AMP8I_PHS8I'
        else:
            raise ValueError('Got unsupported sio data type/pixel size = {}'.format(self._head[2:]))

    def _read_user_data(self):
        if self._user_data is not None:
            return
        if self._magic_number in (0xFF017FFE, 0xFE7F01FF):  # no user data
            self._user_data = {}
        else:
            def read_user_data():
                out = {}
                user_dat_len = 0
                if self._magic_number in (0xFF017FFE, 0xFE7F01FF):  # no user data
                    return out, user_dat_len

                num_data_pairs = struct.unpack('{}I'.format(endian), fi.read(4))[0]

                for i in range(num_data_pairs):
                    name_length = struct.unpack('{}I'.format(endian), fi.read(4))[0]
                    name = struct.unpack('{}{}s'.format(endian, name_length), fi.read(name_length))[0].decode('utf-8')
                    value_length = struct.unpack('{}I'.format(endian), fi.read(4))[0]
                    value = struct.unpack('{}{}s'.format(endian, value_length), fi.read(value_length))[0]
                    try:
                        value = value.decode('utf-8')
                    except UnicodeDecodeError:
                        # leave value as bytes - it may just be some other type
                        pass
                    out[name] = value
                    user_dat_len += 4 + name_length + 4 + value_length
                return out, user_dat_len

            endian = self.ENDIAN[self._magic_number]
            with open(self._file_name, 'rb') as fi:
                fi.seek(20)  # skip the basic header
                # read the user data (some type of header), if necessary
                user_data, user_data_length = read_user_data()
            self._user_data = user_data
            self._data_offset = 20 + user_data_length
        # validate file size
        exp_file_size = self._data_offset + self._head[0]*self._head[1]*self._head[3]
        act_file_size = os.path.getsize(self._file_name)
        if exp_file_size != act_file_size:
            logging.warning(
                'File {} appears to be an SIO file, but size calculated from the header {} '
                'does not match the actual file size {}'.format(self._file_name, exp_file_size, act_file_size))

    def _find_caspr_data(self):
        def find_caspr():
            dir_name, fil_name = os.path.split(self._file_name)
            file_stem = os.path.splitext(fil_name)
            for fil in [os.path.join(dir_name, '{}.hydra'.format(file_stem)),
                        os.path.join(dir_name, '{}.hdr'.format(file_stem)),
                        os.path.join(dir_name, '..', 'RPHDHeader.out'),
                        os.path.join(dir_name, '..', '_RPHDHeader.out')]:
                if os.path.exists(fil) and os.path.isfile(fil):
                    # generally redundant, except for broken link?
                    return fil
            return None

        casp_fil = find_caspr()
        if casp_fil is None:
            return

        out = {}
        with open(casp_fil, 'r') as fi:
            lines = fi.read().splitlines(keepends=False)
        # this is generally just copied from the previous version - maybe refactor eventually
        current_subfield = ''
        reading_subfield = False
        for line in lines:
            if len(line.strip()) == 0:
                continue  # skip blank lines
            if line.startswith(';;;'):  # I guess this is the subfield delimiter?
                reading_subfield = ~reading_subfield  # change state
                if reading_subfield:
                    current_subfield = ''
                else:
                    out[current_subfield] = {}  # prepare the workspace
            else:
                quoted_token = re.match('"(?P<quoted>[^"]+)"', line)
                if quoted_token:  # Some values with spaces are surrounded by quotes
                    # import pdb
                    # pdb.set_trace()
                    tokens = [quoted_token.group('quoted'),
                              line[quoted_token.end('quoted') + 1:].strip()]
                else:  # No quoted values were found
                    # If not using quotes, split with whitespace
                    tokens = line.split(None, 1)
                if (len(tokens) > 1) and tokens[1] != '':
                    if reading_subfield:  # Subsection heading
                        current_subfield = current_subfield + tokens[1]
                    elif not current_subfield == '':  # Actual field value
                        try:
                            out[current_subfield][tokens[1]] = float(tokens[0])
                        except ValueError:  # Value is string not numeric
                            out[current_subfield][tokens[1]] = tokens[0]
        self._caspr_data = out
        # set symmetry
        im_params = out.get('Image Parameters', None)
        if im_params is None:
            return
        illum_dir = im_params.get('image illumination direction [top, left, bottom, right]', None)
        if illum_dir is None:
            return
        elif illum_dir == 'left':
            self._symmetry = (True, False, True)
        elif illum_dir != 'top':
            raise ValueError('unhandled illumination direction {}'.format(illum_dir))

    def get_sicd(self):
        """
        Extract the SICD details.

        Returns
        -------
        SICDType
        """

        if self._sicd is not None:
            return self._sicd
        if self._user_data is None:
            self._read_user_data()
        if self._caspr_data is None:
            self._find_caspr_data()
        # Check if the user data contains a sicd structure.
        sicd_string = None
        for nam in ['SICDMETA', 'SICD_META', 'SICD']:
            if sicd_string is None:
                sicd_string = self._user_data.get(nam, None)
        # If so, assume that this SICD is valid and simply present it
        if sicd_string is not None:
            root_node, xml_ns = parse_xml_from_string(sicd_string)
            self._sicd = SICDType.from_node(root_node, xml_ns)
            self._sicd.derive()
        else:
            # otherwise, we populate a really minimal sicd structure
            num_rows, num_cols = self.data_size
            self._sicd = SICDType(ImageData=ImageDataType(NumRows=num_rows,
                                                          NumCols=num_cols,
                                                          FirstRow=0,
                                                          FirstCol=0,
                                                          PixelType=self.pixel_type,
                                                          FullImage=FullImageType(NumRows=num_rows,
                                                                                  NumCols=num_cols),
                                                          SCPPixel=RowColType(Row=num_rows/2,
                                                                              Col=num_cols/2)))
        return self._sicd


#######
#  The actual reading implementation

class SIOReader(BaseReader):
    __slots__ = ('_sio_details', )

    def __init__(self, sio_details):
        """

        Parameters
        ----------
        sio_details : str|SIODetails
            filename or SIODetails object
        """

        if isinstance(sio_details, str):
            sio_details = SIODetails(sio_details)
        if not isinstance(sio_details, SIODetails):
            raise TypeError('The input argument for SIOReader must be a filename or '
                            'SIODetails object.')
        self._sio_details = sio_details
        sicd_meta = sio_details.get_sicd()
        if sicd_meta.ImageData.PixelType == 'AMP8I_PHS8I':
            transform_data = amp_phase_to_complex(sicd_meta.ImageData.AmpTable)
        else:
            transform_data = 'COMPLEX'
        raw_bands = 2
        output_bands = 1
        output_dtype = 'complex64'
        chipper = BIPChipper(sio_details.file_name, sio_details.data_type, sio_details.data_size,
                             raw_bands, output_bands, output_dtype,
                             symmetry=sio_details.symmetry, transform_data=transform_data,
                             data_offset=sio_details.data_offset)
        super(SIOReader, self).__init__(sicd_meta, chipper, reader_type="SICD")

    @property
    def sio_details(self):
        # type: () -> SIODetails
        """
        SIODetails: The sio details object.
        """

        return self._sio_details

    @property
    def file_name(self):
        return self.sio_details.file_name


#######
#  The actual writing implementation

class SIOWriter(BIPWriter):
    def __init__(self, file_name, sicd_meta, user_data=None, check_older_version=False):
        """

        Parameters
        ----------
        file_name : str
        sicd_meta : SICDType
        user_data : None|Dict[str, str]
        check_older_version : bool
            Try to use an older version (1.1) of the SICD standard, for possible
            application compliance issues?
        """

        # choose magic number (with user data) and corresponding endian-ness
        magic_number = 0xFD7F02FF
        endian = SIODetails.ENDIAN[magic_number]

        # define basic image details
        image_size = (sicd_meta.ImageData.NumRows, sicd_meta.ImageData.NumCols)
        pixel_type = sicd_meta.ImageData.PixelType
        if pixel_type == 'RE32F_IM32F':
            data_type = numpy.dtype('{}f4'.format(endian))
            element_type = 13
            element_size = 8
            transform_data = 'COMPLEX'
        elif pixel_type == 'RE16I_IM16I':
            data_type = numpy.dtype('{}i2'.format(endian))
            element_type = 12
            element_size = 4
            transform_data = complex_to_int
        else:
            data_type = numpy.dtype('{}u1'.format(endian))
            element_type = 11
            element_size = 2
            transform_data = complex_to_amp_phase(sicd_meta.ImageData.AmpTable)
        # construct the sio header
        header = numpy.array(
            [magic_number, image_size[0], image_size[1], element_type, element_size],
            dtype='>u4')
        # construct the user data - must be {str : str}
        if user_data is None:
            user_data = {}
        uh_args = sicd_meta.get_des_details(check_older_version)
        user_data['SICDMETA'] = sicd_meta.to_xml_string(tag='SICD', urn=uh_args['DESSHTN'])
        data_offset = 20
        with open(file_name, 'wb') as fi:
            fi.write(struct.pack('{}5I'.format(endian), *header))
            # write the user data - name size, name, value size, value
            for name in user_data:
                name_bytes = name.encode('utf-8')
                fi.write(struct.pack('{}I'.format(endian), len(name_bytes)))
                fi.write(struct.pack('{}{}s'.format(endian, len(name_bytes), name_bytes)))
                val_bytes = user_data[name].encode('utf-8')
                fi.write(struct.pack('{}I'.format(endian), len(val_bytes)))
                fi.write(struct.pack('{}{}s'.format(endian, len(val_bytes), val_bytes)))
                data_offset += 4 + len(name_bytes) + 4 + len(val_bytes)
        # initialize the bip writer - we're ready to go
        output_bands = 2
        super(SIOWriter, self).__init__(file_name, image_size, data_type, output_bands,
                                        transform_data=transform_data, data_offset=data_offset)
