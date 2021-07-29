"""
Functionality for reading a GFF file into a SICD model.

Note: This has been tested on files of version 1.8 and 2.5, but hopefully works for others.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
import os
import struct
from typing import Union, BinaryIO

import numpy

from sarpy.compliance import int_func, string_types
from sarpy.io.general.base import BaseReader, BIPChipper, is_file_like, \
    SarpyIOError
from sarpy.geometry.geocoords import geodetic_to_ecf

from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, \
    RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, \
    WgtTypeType
from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a Cosmo Skymed file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str|BinaryIO
        the file_name to check

    Returns
    -------
    CSKReader|None
        `CSKReader` instance if Cosmo Skymed file, `None` otherwise
    """

    if is_file_like(file_name):
        return None

    try:
        gff_details = GFFDetails(file_name)
        logging.info('File {} is determined to be a GFF version {} file.'.format(
            file_name, gff_details.version))
        return GFFReader(gff_details)
    except SarpyIOError:
        return None


####################
# utility functions

def _get_string(bytes_in):
    bytes_in = bytes_in.replace(b'\x00', b'')
    return bytes_in.decode('utf-8')


def _rescale_float(int_in, scale):
    return float(int_in)/scale


####################
# version 1 specific header parsing

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

        self.file_object = fi

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

        fi.read(2)  # redundant
        self.comment = _get_string(fi.read(166))
        self.image_plane = struct.unpack(estr+'I', fi.read(4))[0]
        range_pixel_size, azimuth_pixel_size, azimuth_overlap = struct.unpack(estr+'3I', fi.read(3*4))
        self.range_pixel_size = _rescale_float(range_pixel_size, 1 << 16)
        self.azimuth_pixel_size = _rescale_float(azimuth_pixel_size, 1 << 16)
        self.azimuth_overlap = _rescale_float(azimuth_overlap, 1 << 16)

        srp_lat, srp_lon, srp_alt, rfoa, x_to_srp = struct.unpack(estr+'5i', fi.read(5*4))
        self.srp_lat = _rescale_float(srp_lat, 1 << 23)
        self.srp_lon = _rescale_float(srp_lon, 1 << 23)
        self.srp_alt = _rescale_float(srp_alt, 1 << 16)
        self.rfoa = _rescale_float(rfoa, 1 << 23)
        self.x_to_srp = _rescale_float(x_to_srp, 1 << 16)

        fi.read(2)
        self.phase_name = _get_string(fi.read(128))
        fi.read(2)
        self.image_name = _get_string(fi.read(128))
        # at line 32 of def

        self.look_count, self.param_ref_ap, self.param_ref_pos = \
            struct.unpack(estr+'3I', fi.read(3*4))

        graze_angle, squint, gta, range_beam_ctr, flight_time = \
            struct.unpack(estr + 'I2i2I', fi.read(5*4))
        self.graze_angle = _rescale_float(graze_angle, 1 << 23)
        self.squint = _rescale_float(squint, 1 << 23)
        self.gta = _rescale_float(gta, 1 << 23)
        self.range_beam_ctr = _rescale_float(range_beam_ctr, 1 << 8)
        self.flight_time = _rescale_float(flight_time, 1000)

        self.range_chirp_rate, x_to_start, self.mo_comp_mode, v_x = \
            struct.unpack(estr+'fi2I', fi.read(4*4))
        self.x_to_start = _rescale_float(x_to_start, 1 << 16)
        self.v_x = _rescale_float(v_x, 1 << 16)
        # at line 44 of def

        apc_lat, apc_lon, apc_alt = struct.unpack(estr+'3i', fi.read(3*4))
        self.apc_lat = _rescale_float(apc_lat, 1 << 23)
        self.apc_lon = _rescale_float(apc_lon, 1 << 23)
        self.apc_alt = _rescale_float(apc_alt, 1 << 16)

        cal_parm, self.logical_block_address = struct.unpack(estr+'2I', fi.read(2*4))
        self.cal_parm = _rescale_float(cal_parm, 1 << 24)
        az_resolution, range_resolution = struct.unpack(estr+'2I', fi.read(2*4))
        self.az_resolution = _rescale_float(az_resolution, 1 << 16)
        self.range_resolution = _rescale_float(range_resolution, 1 << 16)

        des_sigma_n, des_graze, des_squint, des_range, scene_track_angle = \
            struct.unpack(estr+'iIiIi', fi.read(5*4))
        self.des_sigma_n = _rescale_float(des_sigma_n, 1 << 23)
        self.des_graze = _rescale_float(des_graze, 1 << 23)
        self.des_squint = _rescale_float(des_squint, 1 << 23)
        self.des_range = _rescale_float(des_range, 1 << 8)
        self.scene_track_angle = _rescale_float(scene_track_angle, 1 << 23)
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


class _Radar_1_8(object):
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

        self.file_object = fi

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
        self.date_time = struct.unpack(estr+'6H', fi.read(6*2))  # year, month, day, hour, minute, second
        fi.read(2)  # endian, already parsed
        self.bytes_per_pixel = int_func(struct.unpack(estr+'f', fi.read(4))[0])
        self.frame_count, self.image_type, self.row_major, self.range_count, \
            self.azimuth_count = struct.unpack(estr+'5I', fi.read(5*4))
        self.scale_exponent, self.scale_mantissa, self.offset_exponent, self.offset_mantissa = \
            struct.unpack(estr+'4i', fi.read(4*4))
        # at line 17 of def

        self.res1 = fi.read(32)  # leave uninterpreted

        fi.read(2) # redundant
        self.comment = _get_string(fi.read(166))
        self.image_plane = struct.unpack(estr+'I', fi.read(4))[0]
        range_pixel_size, azimuth_pixel_size, azimuth_overlap = struct.unpack(estr+'3I', fi.read(3*4))
        self.range_pixel_size = _rescale_float(range_pixel_size, 1 << 16)
        self.azimuth_pixel_size = _rescale_float(azimuth_pixel_size, 1 << 16)
        self.azimuth_overlap = _rescale_float(azimuth_overlap, 1 << 16)

        srp_lat, srp_lon, srp_alt, rfoa, x_to_srp = struct.unpack(estr+'5i', fi.read(5*4))
        self.srp_lat = _rescale_float(srp_lat, 1 << 23)
        self.srp_lon = _rescale_float(srp_lon, 1 << 23)
        self.srp_alt = _rescale_float(srp_alt, 1 << 16)
        self.rfoa = _rescale_float(rfoa, 1 << 23)
        self.x_to_srp = _rescale_float(x_to_srp, 1 << 16)

        self.res2 = fi.read(32)  # leave uninterpreted

        fi.read(2)
        self.phase_name = _get_string(fi.read(128))
        fi.read(2)
        self.image_name = _get_string(fi.read(128))
        # at line 34 of def

        self.look_count, self.param_ref_ap, self.param_ref_pos = \
            struct.unpack(estr + '3I', fi.read(3 * 4))

        graze_angle, squint, gta, range_beam_ctr, flight_time = \
            struct.unpack(estr + 'I2i2I', fi.read(5 * 4))
        self.graze_angle = _rescale_float(graze_angle, 1 << 23)
        self.squint = _rescale_float(squint, 1 << 23)
        self.gta = _rescale_float(gta, 1 << 23)
        self.range_beam_ctr = _rescale_float(range_beam_ctr, 1 << 8)
        self.flight_time = _rescale_float(flight_time, 1000)

        self.range_chirp_rate, x_to_start, self.mo_comp_mode, v_x = \
            struct.unpack(estr + 'fi2I', fi.read(4 * 4))
        self.x_to_start = _rescale_float(x_to_start, 1 << 16)
        self.v_x = _rescale_float(v_x, 1 << 16)
        # at line 46 of def

        apc_lat, apc_lon, apc_alt = struct.unpack(estr + '3i', fi.read(3 * 4))
        self.apc_lat = _rescale_float(apc_lat, 1 << 23)
        self.apc_lon = _rescale_float(apc_lon, 1 << 23)
        self.apc_alt = _rescale_float(apc_alt, 1 << 16)

        cal_parm, self.logical_block_address = struct.unpack(estr + '2I', fi.read(2 * 4))
        self.cal_parm = _rescale_float(cal_parm, 1 << 24)
        az_resolution, range_resolution = struct.unpack(estr + '2I', fi.read(2 * 4))
        self.az_resolution = _rescale_float(az_resolution, 1 << 16)
        self.range_resolution = _rescale_float(range_resolution, 1 << 16)

        des_sigma_n, des_graze, des_squint, des_range, scene_track_angle = \
            struct.unpack(estr + 'iIiIi', fi.read(5 * 4))
        self.des_sigma_n = _rescale_float(des_sigma_n, 1 << 23)
        self.des_graze = _rescale_float(des_graze, 1 << 23)
        self.des_squint = _rescale_float(des_squint, 1 << 23)
        self.des_range = _rescale_float(des_range, 1 << 8)
        self.scene_track_angle = _rescale_float(scene_track_angle, 1 << 23)
        # at line 58 of def

        self.user_param = fi.read(48)  # leave uninterpreted

        self.coarse_snr, self.coarse_azimuth_sub, self.coarse_range_sub, \
        self.max_azimuth_shift, self.max_range_shift, \
        self.coarse_delta_azimuth, self.coarse_delta_range = \
            struct.unpack(estr + '7i', fi.read(7 * 4))

        self.tot_procs, self.tpt_box_cmode, self.snr_thresh, self.range_size, \
        self.map_box_size, self.box_size, self.box_spc, self.tot_tpts, \
        self.good_tpts, self.range_seed, self.range_shift, self.azimuth_shift = \
            struct.unpack(estr + '12i', fi.read(12 * 4))
        # at line 78 of def

        self.sum_x_ramp, self.sum_y_ramp = struct.unpack(estr + '2i', fi.read(2 * 4))
        self.cy9k_tape_block, self.nominal_center_frequency = struct.unpack(estr + 'If', fi.read(2 * 4))
        self.image_flags, self.line_number, self.patch_number = struct.unpack(estr + '3I', fi.read(3 * 4))
        self.lambda0, self.srange_pix_space = struct.unpack(estr + '2f', fi.read(2 * 4))
        self.dopp_pix_space, self.dopp_offset, self.dopp_range_scale, self.mux_time_delay = \
            struct.unpack(estr + '4f', fi.read(4 * 4))
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

        self.IE = _Radar_1_8(fi.read(76), estr)  # TODO: what do IE/IF/PH represent?
        self.IF = _Radar_1_8(fi.read(76), estr)
        self.if_algo = _get_string(fi.read(8))
        self.PH = _Radar_1_8(fi.read(76), estr)
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

        velocity_y, velocity_z = struct.unpack(estr+'2I', fi.read(2*4))
        self.velocity_y = _rescale_float(velocity_y, 1 << 16)
        self.velocity_z = _rescale_float(velocity_z, 1 << 16)

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
        sigma_n, self.take_num = struct.unpack(estr+'Ii', fi.read(2*4))
        self.sigma_n = _rescale_float(sigma_n, 1 << 23)

        self.if_sar_flags = struct.unpack(estr+'5i', fi.read(5*4))
        self.mu_threshold, self.gff_app_type = struct.unpack(estr+'fi', fi.read(2*4))
        self.res7 = fi.read(8)  # leave uninterpreted


#####################
# version 2 specific header parsing


class _BlockHeader_2(object):
    """
    Read and interpret a block "sub"-header. This generically precedes every version
    2 data block, including the main file header
    """

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self.name = _get_string(fi.read(16))
        self.major_version, self.minor_version = struct.unpack(estr+'HH', fi.read(2*2))
        self.size = struct.unpack(estr+'I', fi.read(4))[0]
        fi.read(4)  # not sure what this is from looking the matlab. Probably a check sum or something?


class _GSATIMG_2_5(object):
    """
    Interpreter for the GSATIMG object
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

class _GFFInterpreter(object):
    """
    Extractor for the sicd details
    """

    def get_sicd(self):
        """
        Gets the SICD structure.

        Returns
        -------
        SICDType
        """

        raise NotImplementedError

    def get_chipper(self):
        """
        Gets the chipper for reading the data.

        Returns
        -------
        BIPChipper
        """

        raise NotImplementedError


class _GFFInterpreter1(_GFFInterpreter):
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
        def get_collection_info():
            # type: () -> CollectionInfoType
            core_name = self.header.image_name.replace(':', '_')
            return CollectionInfoType(
                CoreName=core_name,
                CollectType='MONOSTATIC',
                RadarMode=RadarModeType(
                    ModeType='SPOTLIGHT'),
                Classification='UNCLASSIFIED')

        def get_image_creation():
            # type: () -> ImageCreationType
            from sarpy.__about__ import __version__
            from datetime import datetime
            return ImageCreationType(
                Application=self.header.creator,
                DateTime=numpy.datetime64(datetime(*self.header.date_time)),
                Site='Unknown',
                Profile='sarpy {}'.format(__version__))

        def get_image_data():
            # type: () -> ImageDataType
            return ImageDataType(
                PixelType='RE32F_IM32F',
                NumRows=num_rows,
                NumCols=num_cols,
                FullImage=(num_rows, num_cols),
                FirstRow=0,
                FirstCol=0,
                SCPPixel=(scp_row, scp_col))

        def get_geo_data():
            # type: () -> GeoDataType

            return GeoDataType(
                SCP=SCPType(
                    LLH=[self.header.srp_lat, self.header.srp_lon, self.header.srp_alt]))

        def get_grid():
            # type: () -> GridType
            image_plane = 'GROUND' if self.header.image_plane == 0 else 'SLANT'
            # we presume that image_plane in [0, 1]

            def get_wgt(str_in):
                # type: (str) -> Union[None, WgtTypeType]
                if str_in == '':
                    return None

                elements = str_in.split()
                win_name = elements[0].upper()
                parameters = None
                if win_name == 'TAYLOR':
                    if len(elements) < 2:
                        raise ValueError('Got unparseable window definition `{}`'.format(str_in))
                    params = elements[1].split(',')
                    if len(params) != 2:
                        raise ValueError('Got unparseable window definition `{}`'.format(str_in))
                    parameters = {'SLL': params[0].strip(), 'NBAR': params[1].strip()}
                return WgtTypeType(
                    WindowName=win_name,
                    Parameters=parameters)

            row_ss = self.header.range_pixel_size
            col_ss = self.header.azimuth_pixel_size
            row_bw = 1./row_ss
            col_bw = 1./col_ss
            if self.header.version == '1.8':
                if self.header.range_win_fac_bw > 0:
                    row_bw = self.header.range_win_fac_bw/row_ss
                if self.header.az_win_fac_bw > 0:
                    col_bw = self.header.az_win_fac_bw/col_ss

            row = DirParamType(
                Sgn=-1,
                SS=row_ss,
                ImpRespWid=self.header.range_resolution,
                ImpRespBW=row_bw,
                DeltaK1=0.5*row_bw,
                DeltaK2=-0.5*row_bw,
                WgtType=get_wgt(self.header.range_win_id if self.header.version == '1.8' else ''),
                DeltaKCOAPoly=[[0, ], ]  # TODO: revisit this?
            )

            col = DirParamType(
                Sgn=-1,
                SS=col_ss,
                ImpRespWid=self.header.az_resolution,
                ImpRespBW=col_bw,
                DeltaK1=0.5*col_bw,
                DeltaK2=-0.5*col_bw,
                WgtType=get_wgt(self.header.az_win_id if self.header.version == '1.8' else ''),
                DeltaKCOAPoly=[[0, ], ]  # TODO: revisit this?
            )

            return GridType(
                ImagePlane=image_plane,
                Type='RGAZIM',
                Row=row,
                Col=col)

        def get_scpcoa():
            # type: () -> Union[None, SCPCOAType]
            side_of_track = 'L' if self.header.squint < 0 else 'R'

            apc_llh = numpy.array(
                [self.header.apc_lat, self.header.apc_lon, self.header.apc_alt],
                dtype='float64')
            if numpy.all(apc_llh == 0):
                arp_pos = None
            else:
                arp_pos = geodetic_to_ecf(apc_llh, ordering='latlon')

            return SCPCOAType(
                ARPPos=arp_pos,
                GrazeAng=self.header.graze_angle,
                SideOfTrack=side_of_track)

        num_rows = self.header.range_count
        num_cols = self.header.azimuth_count
        scp_row = int_func(0.5*num_rows)
        scp_col = int_func(0.5*num_cols)

        collection_info = get_collection_info()
        image_creation = get_image_creation()
        image_data = get_image_data()
        geo_data = get_geo_data()
        grid = get_grid()
        scpcoa = get_scpcoa()

        return SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            ImageData=image_data,
            GeoData=geo_data,
            Grid=grid,
            SCPCOA=scpcoa)

    def get_chipper(self):
        if self.header.bits_per_phase not in [8, 16, 32]:
            raise ValueError('Got unexpected bits per phase {}'.format(self.header.bits_per_phase))
        if self.header.bits_per_phase != self.header.bits_per_magnitude:
            raise ValueError('Got a different value for bits per phase and bits per magnitude.')

        raw_bands = 2
        output_bands = 1
        output_dtype = 'complex64'
        raw_dtype = 'uint{}'.format(self.header.bits_per_phase)
        data_size = (self.header.range_count, self.header.azimuth_count)
        symmetry = (False, False, False)
        data_offset = self.header.header_length
        if self.header.image_type == 1:
            # phase/magnitude which is integer
            transform_data = phase_amp_to_complex(self.header.bits_per_phase)
            return BIPChipper(
                self.header.file_object, raw_dtype, data_size, raw_bands, output_bands, output_dtype,
                symmetry=symmetry, transform_data=transform_data,
                data_offset=data_offset, limit_to_raw_bands=None)
        else:
            raise ValueError('Got unsupported image type `{}`'.format(self.header.image_type))


####################
# formatting functions properly reading the data

def phase_amp_to_complex(bit_depth):
    """
    This constructs the function to convert from phase/amplitude format data of
    a given bit-depth to complex64 data.

    Parameters
    ----------
    bit_depth

    Returns
    -------
    callable
    """

    def converter(data):
        dtype_name = 'uint{}'.format(bit_depth)
        if not isinstance(data, numpy.ndarray):
            raise ValueError('requires a numpy.ndarray, got {}'.format(type(data)))

        if len(data.shape) != 3 and data.shape[2] != 2:
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the last dimension), got shape {}'.format(data.shape))

        if data.dtype.name != dtype_name:
            raise ValueError('requires a numpy.ndarray of dtype `{}`, got `{}`'.format(
                dtype_name, data.dtype.name))

        out = numpy.zeros((data.shape[0], data.shape[1], 1), dtype=numpy.complex64)
        amp = data[:, :, 1]
        theta = data[:, :, 0]*(2*numpy.pi/(1 << bit_depth))
        out[:, :, 0].real = amp*numpy.cos(theta)  # handle shape nonsense
        out[:, :, 0].imag = amp*numpy.sin(theta)
        return out

    return converter


####################
# the actual reader implementation

class GFFDetails(object):
    __slots__ = (
        '_file_name', '_file_object', '_close_after',
        '_endianness', '_major_version', '_minor_version',
        '_header', '_interpreter')

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

    @property
    def interpreter(self):
        """
        The GFF interpreter object.

        Returns
        -------
        _GFFInterpreter
        """

        return self._interpreter

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
        if self.version == '1.6':
            self._header = _GFFHeader_1_6(self._file_object, self.endianness)
            self._interpreter = _GFFInterpreter1(self._header)
        elif self.version == '1.8':
            self._header = _GFFHeader_1_8(self._file_object, self.endianness)
            self._interpreter = _GFFInterpreter1(self._header)
        else:
            # raise ValueError('Not yet')
            pass

    def get_sicd(self):
        """
        Gets the sicd structure.

        Returns
        -------
        SICDType
        """

        return self._interpreter.get_sicd()

    def get_chipper(self):
        """
        Gets the chipper for reading data.

        Returns
        -------
        BIPChipper
        """

        return self._interpreter.get_chipper()

    def __del__(self):
        if self._close_after:
            self._close_after = False
            # noinspection PyBroadException
            try:
                self._file_object.close()
            except Exception:
                pass


class GFFReader(BaseReader, SICDTypeReader):
    """
    Gets a reader type object for GFF files
    """

    __slots__ = ('_gff_details', )

    def __init__(self, gff_details):
        """

        Parameters
        ----------
        gff_details : str|GFFDetails
            file name or GFFDetails object
        """

        if isinstance(gff_details, string_types):
            gff_details = GFFDetails(gff_details)
        if not isinstance(gff_details, GFFDetails):
            raise TypeError('The input argument for a GFFReader must be a '
                            'filename or GFFDetails object')
        self._gff_details = gff_details
        sicd = gff_details.get_sicd()
        chipper = gff_details.get_chipper()

        SICDTypeReader.__init__(self, sicd)
        BaseReader.__init__(self, chipper, reader_type="SICD")

    @property
    def gff_details(self):
        # type: () -> GFFDetails
        """
        GFFDetails: The details object.
        """

        return self._gff_details

    @property
    def file_name(self):
        return self.gff_details.file_name


if __name__ == '__main__':
    import os
    from datetime import datetime

    # gff_root = r'R:\sar\Data_SomeDomestic\Sandia\FARAD_Phoenix\OSAPF\NoAF\PS0004'
    # the_file = os.path.join(gff_root, '20150408_0408P07_PS0004_PT000001_N03_M1_CH3_OSAPF.gff')

    gff_root = r'R:\sar\Data_SomeDomestic\Sandia\dionysius\gff_example'
    the_file = os.path.join(gff_root, 'Patch005.gff')

    details = GFFDetails(the_file)
    print(f'{details.version}')
    # print(f'{details.header.tx_polarization, details.header.rx_polarization}')

    # interp = _GFFInterpreter1(details.header)
    # sicd = interp.get_sicd()
    # print(sicd)
