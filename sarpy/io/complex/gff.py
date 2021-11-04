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
from datetime import datetime
from tempfile import mkstemp
import zlib
import gc

import numpy
from scipy.constants import speed_of_light

from sarpy.io.general.base import BaseReader, BIPChipper, BSQChipper, \
    is_file_like, SarpyIOError
from sarpy.io.general.nitf import MemMap
from sarpy.geometry.geocoords import geodetic_to_ecf, wgs_84_norm, ned_to_ecf

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
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    WaveformParametersType, ChanParametersType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, \
    RcvChanProcType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, \
    NoiseLevelType_


try:
    import PIL
except ImportError:
    PIL = None

logger = logging.getLogger(__name__)


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
        logger.info('File {} is determined to be a GFF version {} file.'.format(
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
        self.estr = estr

        self.version = '1.6'
        fi.seek(12, os.SEEK_SET)
        # starting at line 3 of def
        self.header_length = struct.unpack(estr+'I', fi.read(4))[0]
        if self.header_length < 952:
            raise ValueError(
                'The provided header is apparently too short to be a version 1.6 GFF header')

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
        self.estr = estr

        self.version = '1.8'
        fi.seek(12, os.SEEK_SET)
        # starting at line 3 of def
        self.header_length = struct.unpack(estr+'I', fi.read(4))[0]
        if self.header_length < 2040:
            raise ValueError(
                'The provided header is apparently too short to be a version 1.8 GFF header')

        fi.read(2)  # redundant
        self.creator = _get_string(fi.read(24))
        self.date_time = struct.unpack(estr+'6H', fi.read(6*2))  # year, month, day, hour, minute, second
        fi.read(2)  # endian, already parsed
        self.bytes_per_pixel = int(struct.unpack(estr+'f', fi.read(4))[0])
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
            struct.unpack(estr + '3I', fi.read(3*4))

        graze_angle, squint, gta, range_beam_ctr, flight_time = \
            struct.unpack(estr + 'I2i2I', fi.read(5*4))
        self.graze_angle = _rescale_float(graze_angle, 1 << 23)
        self.squint = _rescale_float(squint, 1 << 23)
        self.gta = _rescale_float(gta, 1 << 23)
        self.range_beam_ctr = _rescale_float(range_beam_ctr, 1 << 8)
        self.flight_time = _rescale_float(flight_time, 1000)

        self.range_chirp_rate, x_to_start, self.mo_comp_mode, v_x = \
            struct.unpack(estr + 'fi2I', fi.read(4*4))
        self.x_to_start = _rescale_float(x_to_start, 1 << 16)
        self.v_x = _rescale_float(v_x, 1 << 16)
        # at line 46 of def

        apc_lat, apc_lon, apc_alt = struct.unpack(estr + '3i', fi.read(3*4))
        self.apc_lat = _rescale_float(apc_lat, 1 << 23)
        self.apc_lon = _rescale_float(apc_lon, 1 << 23)
        self.apc_alt = _rescale_float(apc_alt, 1 << 16)

        cal_parm, self.logical_block_address = struct.unpack(estr + '2I', fi.read(2*4))
        self.cal_parm = _rescale_float(cal_parm, 1 << 24)
        az_resolution, range_resolution = struct.unpack(estr + '2I', fi.read(2*4))
        self.az_resolution = _rescale_float(az_resolution, 1 << 16)
        self.range_resolution = _rescale_float(range_resolution, 1 << 16)

        des_sigma_n, des_graze, des_squint, des_range, scene_track_angle = \
            struct.unpack(estr + 'iIiIi', fi.read(5*4))
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
            struct.unpack(estr + '7i', fi.read(7*4))

        self.tot_procs, self.tpt_box_cmode, self.snr_thresh, self.range_size, \
        self.map_box_size, self.box_size, self.box_spc, self.tot_tpts, \
        self.good_tpts, self.range_seed, self.range_shift, self.azimuth_shift = \
            struct.unpack(estr + '12i', fi.read(12*4))
        # at line 78 of def

        self.sum_x_ramp, self.sum_y_ramp = struct.unpack(estr + '2i', fi.read(2*4))
        self.cy9k_tape_block, self.nominal_center_frequency = struct.unpack(estr + 'If', fi.read(2*4))
        self.image_flags, self.line_number, self.patch_number = struct.unpack(estr + '3I', fi.read(3*4))
        self.lambda0, self.srange_pix_space = struct.unpack(estr + '2f', fi.read(2*4))
        self.dopp_pix_space, self.dopp_offset, self.dopp_range_scale, self.mux_time_delay = \
            struct.unpack(estr + '4f', fi.read(4*4))
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

        self.IE = _Radar_1_8(fi.read(76), estr)
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

# NB: I am only parsing the GSATIMG, APINFO, IFINFO, and GEOINFO blocks
#   because those are the only blocks referenced in the matlab that I
#   am mirroring


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
        what0 = fi.read(4)  # not sure what this is from looking at the matlab.
        self.size = struct.unpack(estr+'I', fi.read(4))[0]
        what1 = fi.read(4)  # not sure what this is from looking at the matlab.
        if (self.version == '2.0' and self.size == 64) or (self.version == '1.0' and self.size == 52):
            self.name = 'RADARINFO'  # fix known issue for some early version 2 GFF files

    @property
    def version(self):
        """
        str: The version
        """

        return '{}.{}'.format(self.major_version, self.minor_version)


# APINFO definitions
class _APInfo_1_0(object):
    """
    The APINFO block
    """
    serialized_length = 314

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self.missionText = _get_string(fi.read(8))
        self.swVerNum = _get_string(fi.read(8))
        self.radarSerNum, self.phSource = struct.unpack(estr+'2I', fi.read(2*4))
        fi.read(2)
        self.phName = _get_string(fi.read(128))
        self.ctrFreq, self.wavelength = struct.unpack(estr+'2f', fi.read(2*4))
        self.rxPolarization, self.txPolarization = struct.unpack(estr+'2I', fi.read(2*4))
        self.azBeamWidth, self.elBeamWidth = struct.unpack(estr+'2f', fi.read(2*4))
        self.grazingAngle, self.squintAngle, self.gta, self.rngToBeamCtr = \
            struct.unpack(estr+'4f', fi.read(4*4))
        # line 16

        self.desSquint, self.desRng, self.desGTA, self.antPhaseCtrBear = \
            struct.unpack(estr+'4f', fi.read(4*4))
        self.ApTimeUTC = struct.unpack(estr+'6H', fi.read(6*2))
        self.flightTime, self.flightWeek = struct.unpack(estr+'2I', fi.read(2*4))
        self.chirpRate, self.xDistToStart = struct.unpack(estr+'2f', fi.read(2*4))
        self.momeasMode, self.radarMode = struct.unpack(estr+'2I', fi.read(2*4))
        # line 32

        self.rfoa = struct.unpack(estr+'f', fi.read(4))[0]
        self.apcVel = struct.unpack(estr+'3d', fi.read(3*8))
        self.apcLLH = struct.unpack(estr+'3d', fi.read(3*8))
        self.keepOutViol, self.gimStopTwist, self.gimStopTilt, self.gimStopAz = \
            struct.unpack(estr+'4f', fi.read(4*4))


class _APInfo_2_0(_APInfo_1_0):
    """
    The APINFO block
    """
    serialized_length = 318

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        _APInfo_1_0.__init__(self, fi, estr)
        self.apfdFactor = struct.unpack(estr+'i', fi.read(4))[0]


class _APInfo_3_0(_APInfo_2_0):
    """
    The APINFO block
    """
    serialized_length = 334

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        _APInfo_2_0.__init__(self, fi, estr)
        self.fastTimeSamples, self.adSampleFreq, self.apertureTime, \
            self.numPhaseHistories = struct.unpack(estr+'I2fI', fi.read(4*4))


class _APInfo_4_0(object):
    """
    The APINFO block
    """
    serialized_length = 418

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        # essentially the same as version 3, except the first two fields are longer
        self.missionText = _get_string(fi.read(50))
        self.swVerNum = _get_string(fi.read(50))
        self.radarSerNum, self.phSource = struct.unpack(estr+'2I', fi.read(2*4))
        fi.read(2)
        self.phName = _get_string(fi.read(128))
        self.ctrFreq, self.wavelength = struct.unpack(estr+'2f', fi.read(2*4))
        self.rxPolarization, self.txPolarization = struct.unpack(estr+'2I', fi.read(2*4))
        self.azBeamWidth, self.elBeamWidth = struct.unpack(estr+'2f', fi.read(2*4))
        self.grazingAngle, self.squintAngle, self.gta, self.rngToBeamCtr = \
            struct.unpack(estr+'4f', fi.read(4*4))
        # line 16

        self.desSquint, self.desRng, self.desGTA, self.antPhaseCtrBear = \
            struct.unpack(estr+'4f', fi.read(4*4))
        self.ApTimeUTC = struct.unpack(estr+'6H', fi.read(6*2))
        self.flightTime, self.flightWeek = struct.unpack(estr+'2I', fi.read(2*4))
        self.chirpRate, self.xDistToStart = struct.unpack(estr+'2f', fi.read(2*4))
        self.momeasMode, self.radarMode = struct.unpack(estr+'2I', fi.read(2*4))
        # line 32

        self.rfoa = struct.unpack(estr+'f', fi.read(4))[0]
        self.apcVel = struct.unpack(estr+'3d', fi.read(3*8))
        self.apcLLH = struct.unpack(estr+'3d', fi.read(3*8))
        self.keepOutViol, self.gimStopTwist, self.gimStopTilt, self.gimStopAz = \
            struct.unpack(estr+'4f', fi.read(4*4))

        self.apfdFactor = struct.unpack(estr+'i', fi.read(4))[0]
        self.fastTimeSamples, self.adSampleFreq, self.apertureTime, \
            self.numPhaseHistories = struct.unpack(estr+'I2fI', fi.read(4*4))


class _APInfo_5_0(_APInfo_4_0):
    """
    The APINFO block
    """
    serialized_length = 426

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        _APInfo_4_0.__init__(self, fi, estr)
        self.lightSpeed = struct.unpack(estr+'d', fi.read(8))[0]  # really?


class _APInfo_5_1(_APInfo_5_0):
    """
    The APINFO block
    """
    serialized_length = 430

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        _APInfo_5_0.__init__(self, fi, estr)
        self.delTanApAngle = struct.unpack(estr+'f', fi.read(4))[0]


class _APInfo_5_2(_APInfo_5_1):
    """
    The APINFO block
    """
    serialized_length = 434

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        _APInfo_5_1.__init__(self, fi, estr)
        self.metersInSampledDoppler = struct.unpack(estr+'f', fi.read(4))[0]


# IFINFO definitions
class _IFInfo_1_0(object):
    """
    Interpreter for IFInfo object
    """
    serialized_length = 514

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self.procProduct = struct.unpack(estr+'I', fi.read(4))[0]
        fi.read(2)
        self.imgFileName = _get_string(fi.read(128))
        self.azResolution, self.rngResolution = struct.unpack(estr+'2f', fi.read(2*4))
        self.imgCalParam, self.sigmaN = struct.unpack(estr+'2f', fi.read(2*4))
        self.sampLocDCRow, self.sampLocDCCol = struct.unpack(estr+'2i', fi.read(2*4))
        self.ifAlgo = _get_string(fi.read(8))
        self.imgFlag = struct.unpack(estr+'i', fi.read(4))[0]
        self.azCoeff = struct.unpack(estr+'6f', fi.read(6*4))
        self.elCoeff = struct.unpack(estr+'9f', fi.read(9*4))
        self.azGeoCorrect, self.rngGeoCorrect = struct.unpack(estr+'2i', fi.read(2*4))
        self.wndBwFactAz, self.wndBwFactRng = struct.unpack(estr+'2f', fi.read(2*4))
        self.wndFncIdAz = _get_string(fi.read(48))
        self.wndFncIdRng = _get_string(fi.read(48))
        fi.read(2)
        self.cmtText = _get_string(fi.read(166))
        self.autoFocusInfo = struct.unpack(estr+'i', fi.read(4))[0]


class _IFInfo_2_0(_IFInfo_1_0):
    """
    Interpreter for IFInfo object - identical with version 2.1 and 2.2
    """
    serialized_length = 582

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        _IFInfo_1_0.__init__(self, fi, estr)
        self.rngFFTSize = struct.unpack(estr+'i', fi.read(4))[0]
        self.RangePaneFilterCoeff = struct.unpack(estr+'11f', fi.read(11*4))
        self.AzPreFilterCoeff = struct.unpack(estr+'5f', fi.read(5*4))


class _IFInfo_3_0(_IFInfo_2_0):
    """
    Interpreter for IFInfo object
    """
    serialized_length = 586

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        _IFInfo_2_0.__init__(self, fi, estr)
        self.afPeakQuadComp = struct.unpack(estr+'f', fi.read(4))[0]


# GEOINFO definitions
class _GeoInfo_1(object):
    """
    Interpreter for GeoInfo object - note that versions 1.0 and 1.1 are identical
    """
    serialized_length = 52

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self.imagePlane = struct.unpack(estr+'i', fi.read(4))[0]
        self.rangePixSpacing, self.desiredGrazAng, self.azPixSpacing = \
            struct.unpack(estr+'3f', fi.read(3*4))
        self.patchCtrLLH = struct.unpack(estr+'3d', fi.read(3*8))
        self.pixLocImCtrRow, self.pixLocImCtrCol = struct.unpack(estr+'2I', fi.read(2*4))
        self.imgRotAngle = struct.unpack(estr+'f', fi.read(4))[0]


# GSATIMG definition

def _get_complex_domain_code(code_int):
    # type: (int) -> str
    if code_int in [0, 3]:
        return 'IQ'
    elif code_int in [1, 4]:
        return 'QI'
    elif code_int in [2, 5]:
        return 'MP'
    elif code_int == 6:
        return 'PM'
    elif code_int == 7:
        return 'M'
    elif code_int == 8:
        return 'P'
    else:
        raise ValueError('Got unexpected code `{}`'.format(code_int))


def _get_band_order(code_int):
    # type: (int) -> str
    if code_int in [0, 1, 2, 7, 8]:
        return 'interleaved'
    elif code_int in [3, 4, 5, 6]:
        return 'sequential'
    else:
        raise ValueError('Got unexpected code `{}`'.format(code_int))


class _PixelFormat(object):
    """
    Interpreter for pixel format object
    """

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self.comp0_bitSize, self.comp0_dataType = struct.unpack(estr+'HI', fi.read(2+4))
        self.comp1_bitSize, self.comp1_dataType = struct.unpack(estr+'HI', fi.read(2+4))
        self.cmplxDomain, self.numComponents = struct.unpack(estr+'Ii', fi.read(2*4))


class _GSATIMG_2(object):
    """
    Interpreter for the GSATIMG object
    """
    serialized_length = 82

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self.endian = struct.unpack(estr+'I', fi.read(4))[0]
        fi.read(2)
        self.imageCreator = _get_string(fi.read(24))
        self.rangePixels, self.azPixels = struct.unpack(estr+'2I', fi.read(2*4))
        self.pixOrder, self.imageLengthBytes, self.imageCompressionScheme, \
            self.pixDataType = struct.unpack(estr+'4I', fi.read(4*4))
        self.pixelFormat = _PixelFormat(fi, estr)
        self.pixValLin, self.autoScaleFac = struct.unpack(estr+'if', fi.read(2*4))

        complex_domain = _get_complex_domain_code(self.pixelFormat.cmplxDomain)
        if complex_domain not in ['IQ', 'QI', 'MP', 'PM']:
            raise ValueError('We got unsupported complex domain `{}`'.format(complex_domain))


# combined GFF version 2 header collection
def _check_serialization(block_header, expected_length):
    # type: (_BlockHeader_2, int) -> None
    if block_header.size == expected_length:
        return

    raise ValueError(
        'Got `{}` block of version `{}` and serialized length {},\n\t'
        'but expected serialized length {}'.format(
            block_header.name, block_header.version, block_header.size, expected_length))


class _GFFHeader_2(object):
    """
    Interpreter for the GFF version 2.* header
    """

    __slots__ = (
        'file_object', 'estr', '_gsat_img', '_ap_info', '_if_info', '_geo_info',
        '_image_header', '_image_offset')

    def __init__(self, fi, estr):
        """

        Parameters
        ----------
        fi : BinaryIO
        estr : str
            The endianness string for format interpretation, one of `['<', '>']`
        """

        self._gsat_img = None
        self._ap_info = None
        self._if_info = None
        self._geo_info = None
        self._image_header = None
        self._image_offset = None
        self.file_object = fi
        self.estr = estr

        # extract the initial file location
        init_location = fi.tell()

        # go to the begining of the file
        fi.seek(0, os.SEEK_SET)
        gsat_header = _BlockHeader_2(fi, estr)
        self._gsat_img = _GSATIMG_2(fi, estr)

        while True:
            block_header = _BlockHeader_2(fi, estr)
            if block_header.name == 'IMAGEDATA':
                self._image_header = block_header
                self._image_offset = fi.tell()
                break
            elif block_header.name == 'APINFO':
                self._parse_apinfo(fi, estr, block_header)
            elif block_header.name == 'IFINFO':
                self._parse_ifinfo(fi, estr, block_header)
            elif block_header.name == 'GEOINFO':
                self._parse_geoinfo(fi, estr, block_header)
            else:
                # we are not parsing this block, so just skip it
                fi.seek(block_header.size, os.SEEK_CUR)

        # return to the initial file location
        fi.seek(init_location, os.SEEK_SET)
        self._check_valid(gsat_header)

    @property
    def gsat_img(self):
        # type: () -> _GSATIMG_2
        return self._gsat_img

    @property
    def ap_info(self):
        # type: () -> Union[_APInfo_1_0, _APInfo_2_0, _APInfo_3_0, _APInfo_4_0, _APInfo_5_0, _APInfo_5_1, _APInfo_5_2]
        return self._ap_info

    @property
    def if_info(self):
        # type: () -> Union[_IFInfo_1_0, _IFInfo_2_0, _IFInfo_3_0]
        return self._if_info

    @property
    def geo_info(self):
        # type: () -> _GeoInfo_1
        return self._geo_info

    @property
    def image_header(self):
        # type: () -> _BlockHeader_2
        return self._image_header

    @property
    def image_offset(self):
        # type: () -> int
        return self._image_offset

    def _parse_apinfo(self, fi, estr, block_header):
        if block_header.name != 'APINFO':
            return

        if block_header.major_version == 1:
            _check_serialization(block_header, _APInfo_1_0.serialized_length)
            self._ap_info = _APInfo_1_0(fi, estr)
        elif block_header.major_version == 2:
            _check_serialization(block_header, _APInfo_2_0.serialized_length)
            self._ap_info = _APInfo_2_0(fi, estr)
        elif block_header.major_version == 3:
            _check_serialization(block_header, _APInfo_3_0.serialized_length)
            self._ap_info = _APInfo_3_0(fi, estr)
        elif block_header.major_version == 4:
            _check_serialization(block_header, _APInfo_4_0.serialized_length)
            self._ap_info = _APInfo_4_0(fi, estr)
        elif block_header.major_version == 5:
            if block_header.minor_version == 0:
                _check_serialization(block_header, _APInfo_5_0.serialized_length)
                self._ap_info = _APInfo_5_0(fi, estr)
            elif block_header.minor_version == 1:
                _check_serialization(block_header, _APInfo_5_1.serialized_length)
                self._ap_info = _APInfo_5_1(fi, estr)
            elif block_header.minor_version == 2:
                _check_serialization(block_header, _APInfo_5_2.serialized_length)
                self._ap_info = _APInfo_5_2(fi, estr)
        else:
            raise ValueError(
                'Could not parse required `{}` block version `{}`'.format(
                    block_header.name, block_header.version))

    def _parse_ifinfo(self, fi, estr, block_header):
        if block_header.name != 'IFINFO':
            return

        if block_header.major_version == 1:
            _check_serialization(block_header, _IFInfo_1_0.serialized_length)
            self._if_info = _IFInfo_1_0(fi, estr)
        elif block_header.major_version == 2:
            _check_serialization(block_header, _IFInfo_2_0.serialized_length)
            self._if_info = _IFInfo_2_0(fi, estr)
        elif block_header.major_version == 3:
            _check_serialization(block_header, _IFInfo_3_0.serialized_length)
            self._if_info = _IFInfo_3_0(fi, estr)
        else:
            raise ValueError(
                'Could not parse required `{}` block version `{}`'.format(
                    block_header.name, block_header.version))

    def _parse_geoinfo(self, fi, estr, block_header):
        if block_header.name != 'GEOINFO':
            return

        _check_serialization(block_header, _GeoInfo_1.serialized_length)
        self._geo_info = _GeoInfo_1(fi, estr)

    def _check_valid(self, gsat_header):
        # ensure that the required elements are all set
        valid = True
        if self._ap_info is None:
            valid = False
            logger.error(
                'GFF version {} file did not present APINFO block'.format(
                    gsat_header.version))
        if self._if_info is None:
            valid = False
            logger.error(
                'GFF version {} file did not present IFINFO block'.format(
                    gsat_header.version))
        if self._geo_info is None:
            valid = False
            logger.error(
                'GFF version {} file did not present GEOINFO block'.format(
                    gsat_header.version))
        if not valid:
            raise ValueError('GFF file determined to be invalid')

    def get_arp_vel(self):
        """
        Gets the aperture velocity in ECF coordinates

        Returns
        -------
        numpy.ndarray
        """

        # TODO: this is not correct

        # get the aperture velocity in its native frame of reference (rotated ENU)
        arp_vel_orig = numpy.array(self.ap_info.apcVel, dtype='float64')
        # TODO: arp_vel_orig is in what coordinate system? Rick said "rotated ENU", wrt gta?
        # gets the angle wrt to True North for the radar frame of reference
        angle = numpy.deg2rad(self.ap_info.rfoa)
        cosine, sine = numpy.cos(angle), numpy.sin(angle)
        # construct the NED velocity vector
        transform = numpy.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, -1]], dtype='float64')
        ned_velocity = transform.dot(arp_vel_orig)
        # convert to ECF
        orp = geodetic_to_ecf(self.ap_info.apcLLH, ordering='latlon')
        out = ned_to_ecf(ned_velocity, orp, absolute_coords=False)
        return out


####################
# object for creation of sicd structure from GFF header object

def _get_wgt(str_in):
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


def _get_polarization_string(int_value):
    # type: (int) -> Union[None, str]
    if int_value == 0:
        return 'H'
    elif int_value == 1:
        return 'V'
    elif int_value == 2:
        return 'LHC'
    elif int_value == 3:
        return 'RHC'
    elif int_value in [4, 5]:
        # TODO: according to their enum, we have 4 -> "T" and 5 -> "P"
        #   what does that mean?
        return 'OTHER'
    else:
        return 'UNKNOWN'


def _get_tx_rcv_polarization(tx_pol_int, rcv_pol_int):
    # type: (int, int) -> (str, str)
    tx_pol = _get_polarization_string(tx_pol_int)
    rcv_pol = _get_polarization_string(rcv_pol_int)
    if tx_pol in ['OTHER', 'UNKNOWN'] or rcv_pol in ['OTHER', 'UNKNOWN']:
        tx_rcv_pol = 'OTHER'
    else:
        tx_rcv_pol = '{}:{}'.format(tx_pol, rcv_pol)
    return tx_pol, tx_rcv_pol


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
                WgtType=_get_wgt(
                    self.header.range_win_id if self.header.version == '1.8' else ''),
                DeltaKCOAPoly=[[0, ], ]  # TODO: revisit this?
            )

            col = DirParamType(
                Sgn=-1,
                SS=col_ss,
                ImpRespWid=self.header.az_resolution,
                ImpRespBW=col_bw,
                DeltaK1=0.5*col_bw,
                DeltaK2=-0.5*col_bw,
                WgtType=_get_wgt(
                    self.header.az_win_id if self.header.version == '1.8' else ''),
                DeltaKCOAPoly=[[0, ], ]  # TODO: revisit this?
            )

            return GridType(
                ImagePlane=image_plane,
                Type='PLANE',
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
        scp_row = int(0.5*num_rows)
        scp_col = int(0.5*num_cols)

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
        if self.header.bits_per_magnitude not in [8, 16, 32]:
            raise ValueError('Got unexpected bits per phase {}'.format(self.header.bits_per_magnitude))

        # creating a custom phase/magnitude data type
        phase_dtype = numpy.dtype('{}u{}'.format(self.header.estr, int(self.header.bits_per_phase/8)))
        magnitude_dtype = numpy.dtype('{}u{}'.format(self.header.estr, int(self.header.bits_per_magnitude/8)))
        raw_dtype = numpy.dtype([('phase', phase_dtype), ('magnitude', magnitude_dtype)])

        raw_bands = 1
        output_bands = 1
        output_dtype = 'complex64'
        data_size = (self.header.range_count, self.header.azimuth_count)
        if self.header.row_major:
            symmetry = (True, True, True)
        else:
            symmetry = (True, True, False)
        data_offset = self.header.header_length
        if self.header.image_type == 1:
            # phase/magnitude which is integer
            transform_data = phase_amp_int_to_complex()
            return BIPChipper(
                self.header.file_object, raw_dtype, data_size, raw_bands, output_bands, output_dtype,
                symmetry=symmetry, transform_data=transform_data,
                data_offset=data_offset, limit_to_raw_bands=None)
        else:
            raise ValueError('Got unsupported image type `{}`'.format(self.header.image_type))


def _get_numpy_dtype(data_type_int):
    # type: (int) -> str
    if data_type_int == 0:
        return 'u1'
    elif data_type_int == 1:
        return 'u2'
    elif data_type_int == 2:
        return 'u4'
    elif data_type_int == 3:
        return 'u8'
    elif data_type_int == 4:
        return 'i1'
    elif data_type_int == 5:
        return 'i2'
    elif data_type_int == 6:
        return 'i4'
    elif data_type_int == 7:
        return 'i8'
    elif data_type_int == 8:
        return 'f4'
    elif data_type_int == 9:
        return 'f8'
    else:
        raise ValueError('Got unsupported data type code `{}`'.format(data_type_int))


class _GFFInterpreter2(_GFFInterpreter):
    """
    Extractor of SICD structure and parameters from GFFHeader_2 object
    """

    def __init__(self, header):
        """

        Parameters
        ----------
        header : _GFFHeader_2
        """

        self.header = header
        self._cached_files = []
        if self.header.gsat_img.pixelFormat.numComponents != 2:
            raise ValueError(
                'The pixel format indicates that the number of components is `{}`, '
                'which is not supported for a complex image'.format(
                    self.header.gsat_img.pixelFormat.numComponents))

    def get_sicd(self):
        def get_collection_info():
            # type: () -> CollectionInfoType
            core_name = self.header.ap_info.phName  # TODO: double check this...
            return CollectionInfoType(
                CollectorName=self.header.ap_info.missionText,
                CoreName=core_name,
                CollectType='MONOSTATIC',
                RadarMode=RadarModeType(
                    ModeType='SPOTLIGHT'),
                Classification='UNCLASSIFIED')

        def get_image_creation():
            # type: () -> ImageCreationType
            from sarpy.__about__ import __version__
            application = '{} {}'.format(self.header.gsat_img.imageCreator, self.header.ap_info.swVerNum)
            date_time = None  # todo: really?
            return ImageCreationType(
                Application=application,
                DateTime=date_time,
                Site='Unknown',
                Profile='sarpy {}'.format(__version__))

        def get_image_data():
            # type: () -> ImageDataType

            pix_data_type = self.header.gsat_img.pixDataType
            amp_table = None
            if pix_data_type == 12:
                pixel_type = 'AMP8I_PHS8I'
                amp_table = numpy.arange(256, dtype='float64')
            elif pix_data_type in [1, 3, 4, 6, 8, 9, 10, 11]:
                pixel_type = 'RE32F_IM32F'
            elif pix_data_type in [2, 7]:
                pixel_type = 'RE16I_IM16I'
            else:
                raise ValueError('Unhandled pixTypeData value `{}`'.format(pix_data_type))

            return ImageDataType(
                PixelType=pixel_type,
                AmpTable=amp_table,
                NumRows=num_rows,
                NumCols=num_cols,
                FullImage=(num_rows, num_cols),
                FirstRow=0,
                FirstCol=0,
                SCPPixel=(scp_row, scp_col))

        def get_geo_data():
            # type: () -> GeoDataType
            return GeoDataType(SCP=SCPType(ECF=scp))

        def get_grid():
            # type: () -> GridType
            image_plane = 'GROUND' if self.header.geo_info.imagePlane == 0 else 'SLANT'
            # we presume that image_plane in [0, 1]

            # derive row/col uvect
            ground_uvec = wgs_84_norm(scp)

            urng = scp - arp_pos  # unit vector for row in the slant plane
            urng /= numpy.linalg.norm(urng)
            if image_plane == 'GROUND':
                row_uvec = urng - numpy.dot(urng, ground_uvec)*ground_uvec
                row_uvec /= numpy.linalg.norm(row_uvec)
            else:
                row_uvec = urng

            col_uvec = arp_vel/numpy.linalg.norm(arp_vel)
            if self.header.ap_info.squintAngle < 0:
                col_uvec *= -1

            # verify that my orientation makes some sense
            dumb_check = ground_uvec.dot(numpy.cross(row_uvec, col_uvec))
            if dumb_check <= 0:
                raise ValueError(
                    'The range vector, velocity vector, and squint angle have '
                    'incompatible orientations')

            parallel_component = numpy.dot(row_uvec, col_uvec)
            if numpy.abs(parallel_component) > 1e-7:
                col_uvec = col_uvec - parallel_component*row_uvec
                col_uvec /= numpy.linalg.norm(col_uvec)

            row_ss = self.header.geo_info.rangePixSpacing
            row_bw = self.header.if_info.wndBwFactRng/self.header.if_info.rngResolution
            row_delta_kcoa_constant = 0.5*(1 - (self.header.if_info.sampLocDCRow/int(0.5*num_rows)))/row_ss
            row = DirParamType(
                Sgn=-1,
                SS=row_ss,
                UVectECF=row_uvec,
                ImpRespWid=self.header.if_info.rngResolution,
                ImpRespBW=row_bw,
                KCtr=2*center_frequency/speed_of_light,
                DeltaK1=0.5*row_bw,
                DeltaK2=-0.5*row_bw,
                WgtType=_get_wgt(self.header.if_info.wndFncIdRng),
                DeltaKCOAPoly=[[row_delta_kcoa_constant, ], ])

            col_ss = self.header.geo_info.azPixSpacing
            col_bw = self.header.if_info.wndBwFactAz/self.header.if_info.azResolution
            col_delta_kcoa_constant = 0.5*(1 - (self.header.if_info.sampLocDCCol/int(0.5*num_cols)))/col_ss
            col = DirParamType(
                Sgn=-1,
                SS=col_ss,
                UVectECF=col_uvec,
                ImpRespWid=self.header.if_info.azResolution,
                ImpRespBW=col_bw,
                KCtr=0,
                DeltaK1=0.5*col_bw,
                DeltaK2=-0.5*col_bw,
                WgtType=_get_wgt(self.header.if_info.wndFncIdAz),
                DeltaKCOAPoly=[[col_delta_kcoa_constant, ], ])

            return GridType(
                ImagePlane=image_plane,
                Type=grid_type,
                Row=row,
                Col=col)

        def get_scpcoa():
            # type: () -> SCPCOAType
            return SCPCOAType(
                ARPPos=arp_pos,
                ARPVel=arp_vel,
                SCPTime=0.5*collect_duration)

        def get_timeline():
            # type: () -> TimelineType
            try:
                # only exists for APINFO version 3 and above
                ipp_end = self.header.ap_info.numPhaseHistories
                ipp = [IPPSetType(
                    TStart=0,
                    TEnd=collect_duration,
                    IPPStart=0,
                    IPPEnd=ipp_end,
                    IPPPoly=[0, ipp_end/collect_duration]), ]
            except AttributeError:
                ipp = None
            return TimelineType(
                CollectStart=start_time,
                CollectDuration=collect_duration,
                IPP=ipp)

        def get_radar_collection():
            # type: () -> RadarCollectionType

            try:
                sample_rate = self.header.ap_info.adSampleFreq
                pulse_length = float(self.header.ap_info.fastTimeSamples)/sample_rate
                waveform = [
                    WaveformParametersType(ADCSampleRate=sample_rate, TxPulseLength=pulse_length), ]
            except AttributeError:
                waveform = None

            rcv_channels = [ChanParametersType(TxRcvPolarization=tx_rcv_pol, index=1), ]

            return RadarCollectionType(
                TxFrequency=(center_frequency-0.5*band_width, center_frequency+0.5*band_width),
                Waveform=waveform,
                TxPolarization=tx_pol,
                RcvChannels=rcv_channels)

        def get_image_formation():
            # type: () -> ImageFormationType
            return ImageFormationType(
                RcvChanProc=RcvChanProcType(ChanIndices=[1, ]),
                TxRcvPolarizationProc=tx_rcv_pol,
                TxFrequencyProc=(
                    center_frequency-0.5*band_width,
                    center_frequency+0.5*band_width),
                TStartProc=0,
                TEndProc=collect_duration,
                ImageFormAlgo=image_form_algo,
                STBeamComp='NO',
                ImageBeamComp='NO',
                AzAutofocus='NO',
                RgAutofocus='NO')

        def repair_scpcoa():
            # call after deriving the sicd fields
            if out_sicd.SCPCOA.GrazeAng is None:
                out_sicd.SCPCOA.GrazeAng = self.header.ap_info.grazingAngle
            if out_sicd.SCPCOA.IncidenceAng is None:
                out_sicd.SCPCOA.IncidenceAng = 90 - out_sicd.SCPCOA.GrazeAng
            if out_sicd.SCPCOA.SideOfTrack is None:
                out_sicd.SCPCOA.SideOfTrack = 'L' if self.header.ap_info.squintAngle < 0 else 'R'

        def populate_radiometric():
            # call after deriving the sicd fields
            rcs_constant = self.header.if_info.imgCalParam**2
            radiometric = RadiometricType(RCSSFPoly=[[rcs_constant, ]])
            # noinspection PyProtectedMember
            radiometric._derive_parameters(out_sicd.Grid, out_sicd.SCPCOA)
            if radiometric.SigmaZeroSFPoly is not None:
                noise_constant = self.header.if_info.sigmaN - 10*numpy.log10(radiometric.SigmaZeroSFPoly[0, 0])
                radiometric.NoiseLevel = NoiseLevelType_(
                    NoiseLevelType='ABSOLUTE',
                    NoisePoly=[[noise_constant, ]])
            out_sicd.Radiometric = radiometric

        num_rows = self.header.gsat_img.rangePixels
        num_cols = self.header.gsat_img.azPixels
        scp_row = self.header.geo_info.pixLocImCtrRow
        scp_col = self.header.geo_info.pixLocImCtrCol

        collect_duration = self.header.ap_info.apertureTime
        scp_time_utc_us = numpy.datetime64(datetime(*self.header.ap_info.ApTimeUTC), 'us').astype('int64')
        start_time = (scp_time_utc_us - int(0.5*collect_duration*1e6)).astype('datetime64[us]')
        tx_pol, tx_rcv_pol = _get_tx_rcv_polarization(
            self.header.ap_info.txPolarization, self.header.ap_info.rxPolarization)
        center_frequency = self.header.ap_info.ctrFreq
        band_width = 0.0  # TODO: is this defined anywhere?

        scp = geodetic_to_ecf(self.header.geo_info.patchCtrLLH)
        arp_llh = self.header.ap_info.apcLLH
        arp_pos = geodetic_to_ecf(arp_llh, ordering='latlon')
        arp_vel = self.header.get_arp_vel()

        if self.header.if_info.ifAlgo in ['PFA', 'OSAPF']:
            # if self.header.if_info.ifAlgo == 'PFA':
            image_form_algo = 'PFA'
            grid_type = 'RGAZIM'
        else:
            image_form_algo = 'OTHER'
            grid_type = 'PLANE'

        collection_info = get_collection_info()
        image_creation = get_image_creation()
        image_data = get_image_data()
        geo_data = get_geo_data()
        scp = geo_data.SCP.ECF.get_array()

        grid = get_grid()
        scpcoa = get_scpcoa()
        timeline = get_timeline()
        radar_collection = get_radar_collection()
        image_formation = get_image_formation()

        out_sicd = SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            ImageData=image_data,
            GeoData=geo_data,
            Grid=grid,
            SCPCOA=scpcoa,
            Timeline=timeline,
            RadarCollection=radar_collection,
            ImageFormation=image_formation)

        out_sicd.derive()
        repair_scpcoa()
        populate_radiometric()
        out_sicd.populate_rniirs(override=False)
        return out_sicd

    def _get_size_and_symmetry(self):
        # type: () -> ((int, int), (bool, bool, bool))
        if self.header.gsat_img.pixOrder == 0:
            # in range consecutive order, opposite from a SICD
            data_size = (self.header.gsat_img.azPixels, self.header.gsat_img.rangePixels)
            symmetry = (True, True, True)
        elif self.header.gsat_img.pixOrder == 1:
            # in azimuth consecutive order, like a SICD
            data_size = (self.header.gsat_img.rangePixels, self.header.gsat_img.azPixels)
            symmetry = (True, True, False)
        else:
            raise ValueError('Got unexpected pixel order `{}`'.format(self.header.gsat_img.pixOrder))
        return data_size, symmetry

    def _check_image_validity(self, band_order):
        # type: (str) -> None

        if self.header.gsat_img.pixelFormat.numComponents != 2:
            raise ValueError(
                'Got unexpected number of components `{}`'.format(
                    self.header.gsat_img.pixelFormat.numComponents))

        image_compression_scheme = self.header.gsat_img.imageCompressionScheme
        if image_compression_scheme in [1, 3]:
            if band_order == 'sequential':
                raise ValueError(
                    'GFF with sequential bands and jpeg or jpeg 2000 compression currently unsupported.')
            if PIL is None:
                raise ValueError(
                    'The GFF image is compressed using jpeg or jpeg 2000 compression, '
                    'and decompression requires the PIL library')

    def _extract_zlib_image(self):
        # type: () -> str
        if self.header.gsat_img.imageCompressionScheme != 2:
            raise ValueError('The image is not zlib compressed')
        self.header.file_object.seek(self.header.image_offset, os.SEEK_SET)
        data_bytes = zlib.decompress(self.header.file_object.read(self.header.image_header.size))
        fi, path_name = mkstemp(suffix='.sarpy_cache', text=False)
        os.close(fi)
        self._cached_files.append(path_name)
        logger.info('Created cached file {} for decompressed data'.format(path_name))
        with open(path_name, 'wb') as the_file:
            the_file.write(data_bytes)
        logger.info('Filled cached file {}'.format(path_name))
        return path_name

    def _extract_pil_image(self, band_order, data_size):
        # type: (str, (int, int)) -> str
        if band_order == 'sequential':
            raise ValueError(
                'GFF with sequential bands and jpeg or jpeg 2000 compression currently unsupported.')
        our_memmap = MemMap(self.header.file_object.name, self.header.image_header.size, self.header.image_offset)
        img = PIL.Image.open(our_memmap)  # this is a lazy operation
        # dump the extracted image data out to a temp file
        fi, path_name = mkstemp(suffix='.sarpy_cache', text=False)
        os.close(fi)
        self._cached_files.append(path_name)
        logger.info('Created cached file {} for decompressed data'.format(path_name))
        data = numpy.asarray(img)  # create our numpy array from the PIL Image
        if data.shape[:2] != data_size:
            raise ValueError(
                'Naively decompressed data of shape {}, but expected ({}, {}, {}).'.format(
                    data.shape, data_size[0], data_size[1], 2))
        mem_map = numpy.memmap(path_name, dtype=data.dtype, mode='w+', offset=0, shape=data.shape)
        mem_map[:] = data
        # clean up this memmap and file overhead
        del mem_map
        logger.info('Filled cached file {}'.format(path_name))
        return path_name

    def _get_interleaved_chipper(self):
        complex_domain = _get_complex_domain_code(self.header.gsat_img.pixelFormat.cmplxDomain)

        dtype0 = _get_numpy_dtype(self.header.gsat_img.pixelFormat.comp0_dataType)
        dtype1 = _get_numpy_dtype(self.header.gsat_img.pixelFormat.comp1_dataType)

        raw_bands = 1
        output_bands = 1
        output_dtype = 'complex64'
        data_size, symmetry = self._get_size_and_symmetry()
        if complex_domain == 'IQ':
            raw_dtype = numpy.dtype([('real', dtype0), ('imag', dtype1)])
            transform_data = I_Q_to_complex()
        elif complex_domain == 'QI':
            raw_dtype = numpy.dtype([('imag', dtype0), ('real', dtype1)])
            transform_data = I_Q_to_complex()
        elif complex_domain == 'MP':
            raw_dtype = numpy.dtype([('magnitude', dtype0), ('phase', dtype1)])
            transform_data = phase_amp_int_to_complex()
        elif complex_domain == 'PM':
            raw_dtype = numpy.dtype([('phase', dtype0), ('magnitude', dtype1)])
            transform_data = phase_amp_int_to_complex()
        else:
            raise ValueError('Got unexpected complex domain `{}`'.format(complex_domain))

        image_compression_scheme = self.header.gsat_img.imageCompressionScheme
        if image_compression_scheme == 0:
            # no compression
            the_file = self.header.file_object
            data_offset = self.header.image_offset
        elif image_compression_scheme in [1, 3]:
            # jpeg or jpeg 2000 compression
            the_file = self._extract_pil_image('interleaved', data_size)
            data_offset = 0
        elif image_compression_scheme == 2:
            # zlib compression
            the_file = self._extract_zlib_image()
            data_offset = 0
        else:
            raise ValueError('Got unhandled image compression scheme code `{}`'.format(image_compression_scheme))
        return BIPChipper(
            the_file, raw_dtype, data_size, raw_bands, output_bands, output_dtype,
            symmetry=symmetry, transform_data=transform_data,
            data_offset=data_offset, limit_to_raw_bands=None)

    def _get_sequential_chipper(self):
        image_compression_scheme = self.header.gsat_img.imageCompressionScheme
        complex_domain = _get_complex_domain_code(self.header.gsat_img.pixelFormat.cmplxDomain)

        if self.header.gsat_img.pixelFormat.comp0_dataType != self.header.gsat_img.pixelFormat.comp1_dataType:
            raise ValueError(
                'GFF with sequential bands has the two components with different data types.\n\t'
                'This is currently unsupported.')

        raw_dtype = numpy.dtype(_get_numpy_dtype(self.header.gsat_img.pixelFormat.comp0_dataType))
        raw_bands = 1
        data_size, symmetry = self._get_size_and_symmetry()

        band_size = data_size[0]*data_size[1]*raw_dtype.itemsize
        if complex_domain in ['IQ', 'QI']:
            transform_data = 'complex'
        elif complex_domain in ['MP', 'PM']:
            transform_data = phase_amp_seq_to_complex()
        else:
            raise ValueError('Got unexpected complex domain `{}`'.format(complex_domain))

        if image_compression_scheme == 0:
            # no compression
            the_file = self.header.file_object
            main_offset = self.header.image_offset
        elif image_compression_scheme == 2:
            the_file = self._extract_zlib_image()
            main_offset = 0
        else:
            raise ValueError('Unhandled image compression scheme `{}`'.format(image_compression_scheme))

        if complex_domain in ['IQ', 'MP']:
            chippers = (
                BIPChipper(
                    the_file, raw_dtype, data_size, raw_bands, raw_bands, raw_dtype,
                    symmetry=symmetry, transform_data=None,
                    data_offset=main_offset, limit_to_raw_bands=None),
                BIPChipper(
                    the_file, raw_dtype, data_size, raw_bands, raw_bands, raw_dtype,
                    symmetry=symmetry, transform_data=None,
                    data_offset=main_offset+band_size, limit_to_raw_bands=None))
        else:
            # construct as IQ/MP order
            chippers = (
                BIPChipper(
                    the_file, raw_dtype, data_size, raw_bands, raw_bands, raw_dtype,
                    symmetry=symmetry, transform_data=None,
                    data_offset=main_offset+band_size, limit_to_raw_bands=None),
                BIPChipper(
                    the_file, raw_dtype, data_size, raw_bands, raw_bands, raw_dtype,
                    symmetry=symmetry, transform_data=None,
                    data_offset=main_offset, limit_to_raw_bands=None))

        return BSQChipper(chippers, raw_dtype, transform_data=transform_data)

    def get_chipper(self):
        band_order = _get_band_order(self.header.gsat_img.pixelFormat.cmplxDomain)
        self._check_image_validity(band_order)

        if band_order == 'interleaved':
            return self._get_interleaved_chipper()
        elif band_order == 'sequential':
            return self._get_sequential_chipper()
        else:
            raise ValueError('Unhandled band order `{}`'.format(band_order))

    def __del__(self):
        """
        Clean up any cached files.

        Returns
        -------
        None
        """

        for fil in self._cached_files:
            if os.path.exists(fil):
                # noinspection PyBroadException
                try:
                    os.remove(fil)
                    logger.info('Deleted cached file {}'.format(fil))
                except Exception:
                    logger.error(
                        'Error in attempt to delete cached file {}.\n\t'
                        'Manually delete this file'.format(fil), exc_info=True)


####################
# formatting functions properly reading the data

def phase_amp_seq_to_complex():
    """
    This constructs the function to convert from phase/magnitude format data,
    assuming that data type is simple with two bands, to complex64 data.

    Returns
    -------
    callable
    """

    def converter(data):
        if not isinstance(data, numpy.ndarray):
            raise TypeError(
                'Requires a numpy.ndarray, got {}'.format(type(data)))

        if len(data.shape) != 3 and data.shape[2] != 2:
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the last dimension), got shape {}'.format(data.shape))

        if data.dtype.name not in ['uint8', 'uint16', 'uint32', 'uint64']:
            raise ValueError(
                'Requires a numpy.ndarray of unsigned integer type.')

        bit_depth = data.dtype.itemsize*8

        out = numpy.zeros(data.shape[:2] + (1, ), dtype=numpy.complex64)
        mag = data[:, :, 0]
        theta = data[:, :, 1]*(2*numpy.pi/(1 << bit_depth))
        out[:, :, 0].real = mag*numpy.cos(theta)
        out[:, :, 0].imag = mag*numpy.sin(theta)
        return out
    return converter


def phase_amp_int_to_complex():
    """
    This constructs the function to convert from phase/magnitude or magnitude/phase
    format data, assuming that the data type is custom with a single band, to complex64 data.

    Returns
    -------
    callable
    """

    def converter(data):
        if not isinstance(data, numpy.ndarray):
            raise TypeError(
                'Requires a numpy.ndarray, got {}'.format(type(data)))

        if len(data.shape) != 3 and data.shape[2] != 1:
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the last dimension), got shape {}'.format(data.shape))

        if data.dtype['phase'].name not in ['uint8', 'uint16', 'uint32', 'uint64'] or \
                data.dtype['magnitude'].name not in ['uint8', 'uint16', 'uint32', 'uint64']:
            raise ValueError(
                'Requires a numpy.ndarray of composite dtype with phase and magnitude '
                'of unsigned integer type.')

        bit_depth = data.dtype['phase'].itemsize*8
        out = numpy.zeros(data.shape, dtype=numpy.complex64)
        mag = data['magnitude']
        theta = data['phase']*(2*numpy.pi/(1 << bit_depth))
        out[:].real = mag*numpy.cos(theta)
        out[:].imag = mag*numpy.sin(theta)
        return out
    return converter


def I_Q_to_complex():
    """
    For simple consistency, this constructs the function to simply convert from
    I/Q or Q/I format data of a given bit-depth to complex64 data.

    Returns
    -------
    callable
    """

    def converter(data):
        if not isinstance(data, numpy.ndarray):
            raise TypeError(
                'Requires a numpy.ndarray, got {}'.format(type(data)))

        if len(data.shape) != 3 and data.shape[2] != 1:
            raise ValueError('Requires a three-dimensional numpy.ndarray (with band '
                             'in the last dimension), got shape {}'.format(data.shape))

        out = numpy.zeros(data.shape, dtype='complex64')
        out.real = data['real']
        out.imag = data['imag']
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
        version = self.version
        if version == '1.6':
            self._header = _GFFHeader_1_6(self._file_object, self.endianness)
            self._interpreter = _GFFInterpreter1(self._header)
        elif version == '1.8':
            self._header = _GFFHeader_1_8(self._file_object, self.endianness)
            self._interpreter = _GFFInterpreter1(self._header)
        elif self.major_version == 2:
            self._header = _GFFHeader_2(self._file_object, self.endianness)
            self._interpreter = _GFFInterpreter2(self._header)
        else:
            raise ValueError('Got unhandled GFF version `{}`'.format(version))

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

        if isinstance(gff_details, str):
            gff_details = GFFDetails(gff_details)
        if not isinstance(gff_details, GFFDetails):
            raise TypeError('The input argument for a GFFReader must be a '
                            'filename or GFFDetails object')
        self._gff_details = gff_details
        sicd = gff_details.get_sicd()
        chipper = gff_details.get_chipper()
        SICDTypeReader.__init__(self, sicd)
        BaseReader.__init__(self, chipper, reader_type="SICD")
        self._check_sizes()

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

    def __del__(self):
        # noinspection PyBroadException
        try:
            del self._chipper  # you have to explicitly delete and garbage collect the chipper to delete any temp file
            gc.collect()
            del self._gff_details
        except Exception:
            pass
