"""
Functionality for reading a GFF file into a SICD model
"""

# some sample datasets here: https://www.sandia.gov/radar/complex-data/

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import os
import struct
from typing import BinaryIO

import numpy

from sarpy.io.general.base import SarpyIOError

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, \
    RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType


####################
# utility functions

def _get_string(bytes_in):
    bytes_in = bytes_in.replace(b'\x00', b'')
    return bytes_in.decode('utf-8')


####################
# version specific header parsing

class _GFFHeader_1_6(object):
    """
    Interpreter for the GFF version 1.6 header
    """

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self.version = '1.6'
        fi.seek(12, os.SEEK_SET)
        # starting at line 3 of def
        self.header_length = struct.unpack(estr+'I', fi.read(4))[0]
        if self.header_length < 952:
            raise ValueError(
                'The provided header is apparently too short to be a version 1.6 GFF header')
        # TODO: should this be 1024 all the time?

        fi.read(2)  # redundant
        self.creator = _get_string(fi.read(24))
        self.date_time = struct.unpack(estr+'6H', fi.read(6*2))  # year,month, day, hour, minute, second
        fi.read(2)  # endian, already parsed
        self.bytes_per_pixel, self.frame_count, self.image_type, \
            self.row_major, self.range_count, self.azimuth_count = \
            struct.unpack(estr+'6I', fi.read(6*4))
        self.scale_exponent, self.scale_mantissa, self.offset_exponent, self.offset_mantissa = \
            struct.unpack(estr+'4i', fi.read(4*4))
        # at line 17 of def

        fi.read(2) # redundant
        self.comment = _get_string(fi.read(166))
        self.image_plane, self.range_pixel_size, self.azimuth_pixel_size = \
            struct.unpack(estr+'3I', fi.read(3*4))
        self.azimuth_overlap, self.srp_lat, self.srp_lon, self.srp_alt, self.rfoa, \
            self.x_to_srp = struct.unpack(estr+'6i', fi.read(6*4))
        fi.read(2)
        self.phase_name = _get_string(fi.read(128))
        fi.read(2)
        self.image_name = _get_string(fi.read(128))
        # at line 32 of def

        self.look_count, self.param_ref_ap, self.param_ref_pos, self.graze_angle, \
            self.squint, self.gta, self.range_beam_ctr, self.flight_time = \
            struct.unpack(estr+'4I2i2I', fi.read(8*4))

        self.range_chirp_rate, self.x_to_start, self.mo_comp_mode, self.v_x = \
            struct.unpack(estr+'fi2I', fi.read(4*4))
        # at line 44 of def

        self.apc_lat, self.apc_lon, self.apc_alt = struct.unpack(estr+'3i', fi.read(3*4))
        self.cal_parm, self.logical_block_address = struct.unpack(estr+'2I', fi.read(2*4))
        self.az_resolution, self.range_resolution = struct.unpack(estr+'2I', fi.read(2*4))
        self.des_sigma_n, self.des_graze, self.des_squint, self.des_range, \
            self.scene_track_angle = struct.unpack(estr+'iIiIi', fi.read(5*4))
        # at line 56 of def

        self.user_param = fi.read(48)  # leave uninterpreted
        self.coarse_snr, self.coarse_azimuth_sub, self.coarse_range_sub, \
            self.max_azimuth_shift, self.max_range_shift, \
            self.coarse_delta_azimuth, self.coarse_delta_range = \
            struct.unpack(estr+'7i', fi.read(7*4))
        self.tot_procs, self.tpt_box_cmode, self.snr_thresh, self.range_size, \
            self.map_box_size, self.box_size, self.box_spc, self.tot_tpts, \
            self.good_tpts, self.range_seed, self.range_shift, self.azimuth_shift = \
            struct.unpack(estr+'12i', fi.read(12*4))
        # at line 76 of def

        self.sum_x_ramp, self.sum_y_ramp = struct.unpack(estr+'2i', fi.read(2*4))
        self.cy9k_tape_block, self.nominal_center_frequency = struct.unpack(estr+'If', fi.read(2*4))
        self.image_flags, self.line_number, self.patch_number = struct.unpack(estr+'3I', fi.read(3*4))
        self.lambda0, self.srange_pix_space = struct.unpack(estr+'2f', fi.read(2*4))
        self.dopp_pix_space, self.dopp_offset, self.dopp_range_scale, self.mux_time_delay = \
            struct.unpack(estr+'4f', fi.read(4*4))
        # at line 89 of def

        self.apc_ecef = struct.unpack(estr+'3d', fi.read(3*8))
        self.vel_ecef = struct.unpack(estr+'3f', fi.read(3*4))
        self.phase_cal = struct.unpack(estr+'f', fi.read(4))[0]
        self.srp_ecef = struct.unpack(estr+'3d', fi.read(3*8))
        self.res5 = fi.read(64)  # leave uninterpreted


class _radar_1_8(object):
    """
    The radar details, for version 1.8
    """

    def __init__(self, the_bytes, estr):
        """

        Parameters
        ----------
        the_bytes : bytes
            This will be required to have length 76
        estr : str
            The endianness format string
        """

        if not (isinstance(the_bytes, bytes) and len(the_bytes) == 76):
            raise ValueError('Incorrect length input')

        self.platform = _get_string(the_bytes[:24])
        self.proc_id = _get_string(the_bytes[24:36])
        self.radar_model = _get_string(the_bytes[36:48])
        self.radar_id = struct.unpack(estr+'I', the_bytes[48:52])[0]
        self.swid = _get_string(the_bytes[52:76])


class _GFFHeader_1_8(object):
    """
    Interpreter for the GFF version 1.8 header
    """

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self.version = '1.8'
        fi.seek(12, os.SEEK_SET)
        # starting at line 3 of def
        self.header_length = struct.unpack(estr+'I', fi.read(4))[0]
        if self.header_length < 2040:  # TODO: correct this
            raise ValueError(
                'The provided header is apparently too short to be a version 1.8 GFF header')
        # TODO: should this be 2048 all the time?

        fi.read(2)  # redundant
        self.creator = _get_string(fi.read(24))
        self.date_time = struct.unpack(estr+'6H', fi.read(6*2))  # year,month, day, hour, minute, second
        fi.read(2)  # endian, already parsed
        self.bytes_per_pixel, self.frame_count, self.image_type, \
            self.row_major, self.range_count, self.azimuth_count = \
            struct.unpack(estr+'f5I', fi.read(6*4))
        self.scale_exponent, self.scale_mantissa, self.offset_exponent, self.offset_mantissa = \
            struct.unpack(estr+'4i', fi.read(4*4))
        # at line 17 of def

        self.res2 = fi.read(32)  # leave uninterpreted

        fi.read(2) # redundant
        self.comment = _get_string(fi.read(166))
        self.image_plane, self.range_pixel_size, self.azimuth_pixel_size = \
            struct.unpack(estr+'3I', fi.read(3*4))
        self.azimuth_overlap, self.srp_lat, self.srp_lon, self.srp_alt, self.rfoa, \
            self.x_to_srp = struct.unpack(estr+'6i', fi.read(6*4))

        self.res2 = fi.read(32)  # leave uninterpreted

        fi.read(2)
        self.phase_name = _get_string(fi.read(128))
        fi.read(2)
        self.image_name = _get_string(fi.read(128))
        # at line 34 of def

        self.look_count, self.param_ref_ap, self.param_ref_pos, self.graze_angle, \
            self.squint, self.gta, self.range_beam_ctr, self.flight_time = \
            struct.unpack(estr+'4I2i2I', fi.read(8*4))

        self.range_chirp_rate, self.x_to_start, self.mo_comp_mode, self.v_x = \
            struct.unpack(estr+'fi2I', fi.read(4*4))
        # at line 46 of def

        self.apc_lat, self.apc_lon, self.apc_alt = struct.unpack(estr+'3i', fi.read(3*4))
        self.cal_parm, self.logical_block_address = struct.unpack(estr+'2I', fi.read(2*4))
        self.az_resolution, self.range_resolution = struct.unpack(estr+'2I', fi.read(2*4))
        self.des_sigma_n, self.des_graze, self.des_squint, self.des_range, \
            self.scene_track_angle = struct.unpack(estr+'iIiIi', fi.read(5*4))
        # at line 58 of def

        self.user_param = fi.read(48)  # leave uninterpreted
        self.coarse_snr, self.coarse_azimuth_sub, self.coarse_range_sub, \
            self.max_azimuth_shift, self.max_range_shift, \
            self.coarse_delta_azimuth, self.coarse_delta_range = \
            struct.unpack(estr+'7i', fi.read(7*4))
        self.tot_procs, self.tpt_box_cmode, self.snr_thresh, self.range_size, \
            self.map_box_size, self.box_size, self.box_spc, self.tot_tpts, \
            self.good_tpts, self.range_seed, self.range_shift, self.azimuth_shift = \
            struct.unpack(estr+'12i', fi.read(12*4))
        # at line 78 of def

        self.sum_x_ramp, self.sum_y_ramp = struct.unpack(estr+'2i', fi.read(2*4))
        self.cy9k_tape_block, self.nominal_center_frequency = struct.unpack(estr+'If', fi.read(2*4))
        self.image_flags, self.line_number, self.patch_number = struct.unpack(estr+'3I', fi.read(3*4))
        self.lambda0, self.srange_pix_space = struct.unpack(estr+'2f', fi.read(2*4))
        self.dopp_pix_space, self.dopp_offset, self.dopp_range_scale, self.mux_time_delay = \
            struct.unpack(estr+'4f', fi.read(4*4))
        # at line 91 of def

        self.apc_ecef = struct.unpack(estr+'3d', fi.read(3*8))
        self.vel_ecef = struct.unpack(estr+'3f', fi.read(3*4))
        self.phase_cal = struct.unpack(estr+'f', fi.read(4))[0]
        self.srp_ecef = struct.unpack(estr+'3d', fi.read(3*8))

        self.res5 = fi.read(64)  # leave uninterpreted
        # at line 102

        self.header_length1 = struct.unpack(estr+'I', fi.read(4))[0]
        self.image_date = struct.unpack(estr+'6H', fi.read(6*2))  # year,month, day, hour, minute, second
        self.comp_file_name = _get_string(fi.read(128))
        self.ref_file_name = _get_string(fi.read(128))

        self.IE = _radar_1_8(fi.read(76), estr)  # TODO: what do IE/IF/PH represent?
        self.IF = _radar_1_8(fi.read(76), estr)
        self.if_algo = _get_string(fi.read(8))
        self.PH = _radar_1_8(fi.read(76), estr)
        # at line 122 of def

        self.ph_data_rcd, self.proc_product = struct.unpack(estr+'2i', fi.read(2*4))
        self.mission_text = _get_string(fi.read(8))
        self.ph_source, self.gps_week = struct.unpack(estr+'iI', fi.read(2*4))
        self.data_collect_reqh = _get_string(fi.read(14))
        self.res6 = fi.read(2)  # leave uninterpreted
        # at line 129

        self.grid_name = _get_string(fi.read(24))
        self.pix_val_linearity, self.complex_or_real, self.bits_per_magnitude, \
            self.bits_per_phase = struct.unpack(estr+'2i2H', fi.read(2*4+2*2))
        self.complex_order_type, self.pix_data_type, self.image_length, \
            self.image_cmp_scheme = struct.unpack(estr+'4i', fi.read(4*4))
        # at line 138

        self.apbo, self.asa_pitch, self.asa_squint, self.dsa_pitch, self.ira = \
            struct.unpack(estr+'5f', fi.read(5*4))
        self.rx_polarization = struct.unpack(estr+'2f', fi.read(2*4))
        self.tx_polarization = struct.unpack(estr+'2f', fi.read(2*4))
        self.v_avg = struct.unpack(estr+'3f', fi.read(3*4))
        self.apc_avg = struct.unpack(estr+'3f', fi.read(3*4))
        self.averaging_time, self.dgta = struct.unpack(estr+'2f', fi.read(2*4))
        # at line 153

        self.velocity_y, self.velocity_z = struct.unpack(estr+'2I', fi.read(2*4))
        self.ba, self.be = struct.unpack(estr+'2f', fi.read(2*4))
        self.az_geom_corr, self.range_geom_corr, self.az_win_fac_bw, \
            self.range_win_fac_bw = struct.unpack(estr+'2i2f', fi.read(4*4))
        self.az_win_id = _get_string(fi.read(48))
        self.range_win_id = _get_string(fi.read(48))
        # at line 163

        self.keep_out_viol_prcnt = struct.unpack(estr+'f', fi.read(4))[0]
        self.az_coeff = struct.unpack(estr+'6f', fi.read(6*4))
        self.pos_uncert = struct.unpack(estr+'3f', fi.read(3*4))
        self.nav_aiding_type = struct.unpack(estr+'i', fi.read(4))[0]
        self.two_dnl_phase_coeffs = struct.unpack(estr+'10f', fi.read(10*4))
        self.clutter_snr_thresh = struct.unpack(estr+'f', fi.read(4))[0]
        # at line 171

        self.elevation_coeff = struct.unpack(estr+'9f', fi.read(9*4))
        self.monopulse_coeff = struct.unpack(estr+'12f', fi.read(12*4))
        self.twist_pt_err_prcnt, self.tilt_pt_err_prcnt, self.az_pt_err_prcnt = \
            struct.unpack(estr+'3f', fi.read(3*4))
        self.sigma_n, self.take_num = struct.unpack(estr+'Ii', fi.read(2*4))
        self.if_sar_flags = struct.unpack(estr+'5i', fi.read(5*4))
        self.mu_threshold, self.gff_app_type = struct.unpack(estr+'fi', fi.read(2*4))
        self.res7 = fi.read(8)  # leave uninterpreted


class _GFFHeader_2(object):
    """
    Interpreter for the GFF version 2.* header
    """

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        pass


####################
# object for creation of sicd structure from GFF header object

class _GFFInterpretter1(object):
    """
    Extractor of SICD structure and parameters from gff_header_1*
    object
    """

    def __init__(self, header):
        """

        Parameters
        ----------
        header : _GFFHeader_1_6|_GFFHeader_1_8
        """

        self.header = header
        if self.header.image_type == 0:
            raise ValueError(
                'ImageType indicates a magnitude only image, which is incompatible with SICD')


    def get_sicd(self):
        """
        Gets the SICD structure.

        Returns
        -------
        SICDType
        """

        def get_collection_info():
            # type: () -> CollectionInfoType
            core_name = self.header.image_name.replace(':', '_')
            return CollectionInfoType(
                CoreName=core_name,
                CollectType='MONOSTATIC',
                RadarMode=RadarModeType(
                    ModeType='SPOTLIGHT',
                    ModeID=None),
                Classification='UNCLASSIFIED')

        def get_image_creation():
            # type: () -> ImageCreationType
            from sarpy.__about__ import __version__
            from datetime import datetime
            return ImageCreationType(
                Application=self.header.creator,
                DateTime=numpy.datetime64(datetime(*details.header.date_time)),
                Site='Unknown',
                Profile='sarpy {}'.format(__version__))

        def get_image_data():
            # type: () -> ImageDataType
            return ImageDataType()

        collection_info = get_collection_info()
        image_creation = get_image_creation()
        image_data = get_image_data()

        return SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            ImageData=image_data)



####################
# formatting functions properly reading the data


class GFFDetails(object):
    __slots__ = (
        '_file_name', '_file_object', '_close_after',
        '_endianness', '_major_version', '_minor_version',
        '_header')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        self._endianness = None
        self._major_version = None
        self._minor_version = None
        self._header = None
        self._close_after = True

        if not os.path.isfile(file_name):
            raise SarpyIOError('Path {} is not a file'.format(file_name))
        self._file_name = file_name
        self._file_object = open(self._file_name, 'rb')

        check = self._file_object.read(7)
        if check != b'GSATIMG':
            self._file_object.close()
            self._close_after = False
            raise SarpyIOError('file {} is not a GFF file'.format(self._file_name))

        # initialize things
        self._initialize()

    @property
    def file_name(self):
        """
        str: the file name
        """

        return self._file_name

    @property
    def endianness(self):
        """
        str: The endian format of the GFF storage. Returns '<' if little-endian or '>' if big endian.
        """

        return self._endianness

    @property
    def major_version(self):
        """
        int: The major GFF version number
        """

        return self._major_version

    @property
    def minor_version(self):
        """
        int: The minor GFF version number
        """

        return self._minor_version

    @property
    def version(self):
        """
        str: The GFF version number
        """

        return '{}.{}'.format(self._major_version, self._minor_version)

    @property
    def header(self):
        """
        The GFF header object.

        Returns
        -------
        _GFFHeader_1_6|_GFFHeader_1_8|_GFFHeader_2
        """

        return self._header

    def _initialize(self):
        """
        Initialize the various elements
        """

        self._file_object.seek(7, os.SEEK_SET)
        check = self._file_object.read(1)
        if check == b'\x20':
            # this should be version 1.*, but we will verify
            self._file_object.seek(54, os.SEEK_SET)
            endianness = struct.unpack('H', self._file_object.read(2))[0] # 0 if little endian
            estr = '<' if endianness == 0 else '>'

            self._file_object.seek(8, os.SEEK_SET)
            self._minor_version, self._major_version = struct.unpack('{}HH'.format(estr), self._file_object.read(4))
        elif check == b'\x00':
            # this should be a version 2.*, but we will verify
            estr = '<'
            self._file_object.seek(16, os.SEEK_SET)
            self._major_version, self._minor_version = struct.unpack('{}HH'.format(estr), self._file_object.read(4))
        else:
            raise ValueError('Got unexpected check byte')

        self._file_object.seek(0, os.SEEK_SET)
        self._endianness = estr
        print(self.version)
        if self.version == '1.6':
            self._header = _GFFHeader_1_6(self._file_object, self.endianness)
        elif self.version == '1.8':
            self._header = _GFFHeader_1_8(self._file_object, self.endianness)
        else:
            raise ValueError('Not yet')

    def __del__(self):
        if self._close_after:
            self._close_after = False
            # noinspection PyBroadException
            try:
                self._file_object.close()
            except Exception:
                pass


if __name__ == '__main__':
    import os
    from datetime import datetime
    the_file = os.path.expanduser('~/Downloads/GFF/example_files/MiniSAR20050519p0009image003/MiniSAR20050519p0001image008.gff')
    details = GFFDetails(the_file)
    print(f'{details.header.date_time}, {numpy.datetime64(datetime(*details.header.date_time), "s")}')
