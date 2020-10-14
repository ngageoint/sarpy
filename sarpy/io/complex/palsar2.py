# -*- coding: utf-8 -*-
"""
Functionality for reading PALSAR ALOS 2 data into a SICD model.
"""

import logging
import os
import struct
from typing import Union, Tuple, List

import numpy

from sarpy.io.general.base import BaseChipper, SubsetChipper
from sarpy.io.general.bip import BIPChipper

from sarpy.compliance import int_func

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


##########
# basic helper functions

def _determine_file_type(file_name):
    """
    This checks the initial bit of the header to determine file type.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    None|str
    """

    with open(file_name, 'rb') as fi:
        # read the first 12 bytes of the file
        header = fi.read(12)
    parts = struct.unpack('BBBB', header[4:8])
    if parts == (192, 192, 18, 18):
        return 'VOL'
    elif parts == (11, 192, 18, 18):
        return 'LED'
    elif parts == (50, 192, 18, 18):
        return 'IMG'
    elif parts == (63, 192, 18, 18):
        return 'TRL'
    else:
        return None


def _make_float(bytes_in):
    """
    Try to parse as a float.

    Parameters
    ----------
    bytes_in : bytes

    Returns
    -------
    float
    """

    if len(bytes_in.strip()) == 0:
        return numpy.nan
    else:
        return float(bytes_in)


##########
# helper class that contains common header elements

class _BaseElements(object):
    """
    Common header element
    """
    __slots__ = (
        'rec_num', 'rec_subtype1', 'rec_type', 'rec_subtype2', 'rec_subtype3', 'rec_length')

    def __init__(self, fi):
        """

        Parameters
        ----------
        fi
            The file object.
        """

        the_bytes = fi.read(12)
        self.rec_num, self.rec_subtype1, self.rec_type, self.rec_subtype2, self.rec_subtype3, \
        self.rec_length = struct.unpack('>IBBBBI', the_bytes)  # type: int, int, int, int, int, int


class _BaseElements2(_BaseElements):
    __slots__ = ('ascii_ebcdic', )

    def __init__(self, fi):
        super(_BaseElements2, self).__init__(fi)
        self.ascii_ebcdic = fi.read(2).decode('utf-8')  # type: str
        fi.seek(2, 1) # skip reserved field


############
# Header elements common to IMG, LED, TRL, and VOL

class _CommonElements(_BaseElements2):
    """
    Parser and interpreter for the elements common to IMG, LED, TRL, and VOL files
    """

    __slots__ = (
        'doc_id', 'doc_rev', 'rec_rev', 'soft_rel_rev')

    def __init__(self, fi):
        super(_CommonElements, self).__init__(fi)
        self.doc_id = fi.read(12).decode('utf-8')  # type: str
        self.doc_rev = fi.read(2).decode('utf-8')  # type: str
        self.rec_rev = fi.read(2).decode('utf-8')  # type: str
        self.soft_rel_rev = fi.read(12).decode('utf-8')  # type: str


###############
# Elements common to IMG, LED, and TRL

class _CommonElements2(_CommonElements):
    """
    Parser and interpreter for the elements common to IMG, LED, TRL files
    """

    __slots__ = (
        'file_num', 'file_id', 'rec_seq_loc_type_flag', 'seq_num_loc', 'fld_len_seq',
        'rec_code_loc_type_flag', 'loc_rec_code', 'fld_len_code', 'rec_len_loc_type_flag',
        'loc_rec_len', 'len_rec_len', 'num_data_rec', 'data_len')

    def __init__(self, fi):
        """

        Parameters
        ----------
        fi
            The open file object.
        """

        super(_CommonElements2, self).__init__(fi)
        self.file_num = fi.read(4).decode('utf-8')  # type: str
        self.file_id = fi.read(16).decode('utf-8')  # type: str
        self.rec_seq_loc_type_flag = fi.read(4).decode('utf-8')  # type: str
        self.seq_num_loc = fi.read(8).decode('utf-8')  # type: str
        self.fld_len_seq = fi.read(4).decode('utf-8')  # type: str
        self.rec_code_loc_type_flag = fi.read(4).decode('utf-8')  # type: str
        self.loc_rec_code = fi.read(8).decode('utf-8')  # type: str
        self.fld_len_code = fi.read(4).decode('utf-8')  # type: str
        self.rec_len_loc_type_flag = fi.read(4).decode('utf-8')  # type: str
        self.loc_rec_len = fi.read(8).decode('utf-8')  # type: str
        self.len_rec_len = fi.read(4).decode('utf-8')  # type: str
        fi.seek(68, 1)  # skip reserved field
        self.num_data_rec = int_func(struct.unpack('6s', fi.read(6))[0])  # type: int
        self.data_len = int_func(struct.unpack('6s', fi.read(6))[0])  # type: int


################
# Further common to LED and TRL

class _CommonElements3(_CommonElements2):
    """
    Parser and interpreter for the elements common to LED and TRL files
    """

    __slots__ = (
        'num_map_rec', 'map_len', 'num_pos_rec', 'pos_len', 'num_att_rec', 'att_len',
        'num_rad_rec', 'rad_len', 'num_rad_comp_rec', 'rad_comp_len',
        'num_data_qual_rec', 'data_qual_len', 'num_hist_rec', 'hist_len',
        'num_rng_spect_rec', 'rng_spect_len', 'num_dem_rec', 'dem_len',
        'num_radar_rec', 'radar_len', 'num_annot_rec', 'annot_len',
        'num_proc_rec', 'proc_len', 'num_cal_rec', 'cal_len',
        'num_gcp_rec', 'gcp_len', 'num_fac_data_rec', 'fac_data_len')

    def __init__(self, fi):
        super(_CommonElements3, self).__init__(fi)
        self.num_map_rec = int_func(fi.read(6))  # type: int
        self.map_len = int_func(fi.read(6))  # type: int
        self.num_pos_rec = int_func(fi.read(6))  # type: int
        self.pos_len = int_func(fi.read(6))  # type: int
        self.num_att_rec = int_func(fi.read(6))  # type: int
        self.att_len = int_func(fi.read(6))  # type: int
        self.num_rad_rec = int_func(fi.read(6))  # type: int
        self.rad_len = int_func(fi.read(6))  # type: int
        self.num_rad_comp_rec = int_func(fi.read(6))  # type: int
        self.rad_comp_len = int_func(fi.read(6))  # type: int
        self.num_data_qual_rec = int_func(fi.read(6))  # type: int
        self.data_qual_len = int_func(fi.read(6))  # type: int
        self.num_hist_rec = int_func(fi.read(6))  # type: int
        self.hist_len = int_func(fi.read(6))  # type: int
        self.num_rng_spect_rec = int_func(fi.read(6))  # type: int
        self.rng_spect_len = int_func(fi.read(6))  # type: int
        self.num_dem_rec = int_func(fi.read(6))  # type: int
        self.dem_len = int_func(fi.read(6))  # type: int
        self.num_radar_rec = int_func(fi.read(6))  # type: int
        self.radar_len = int_func(fi.read(6))  # type: int
        self.num_annot_rec = int_func(fi.read(6))  # type: int
        self.annot_len = int_func(fi.read(6))  # type: int
        self.num_proc_rec = int_func(fi.read(6))  # type: int
        self.proc_len = int_func(fi.read(6))  # type: int
        self.num_cal_rec = int_func(fi.read(6))  # type: int
        self.cal_len = int_func(fi.read(6))  # type: int
        self.num_gcp_rec = int_func(fi.read(6))  # type: int
        self.gcp_len = int_func(fi.read(6))  # type: int
        fi.seek(60, 1)  # skip reserved fields
        # the five data facility records
        num_fac_data_rec = []
        fac_data_len = []
        for i in range(5):
            num_fac_data_rec.append(int_func(fi.read(6)))
            fac_data_len.append(int_func(fi.read(8)))
        self.num_fac_data_rec = tuple(num_fac_data_rec)  # type: Tuple[int]
        self.fac_data_len = tuple(fac_data_len)  # type: Tuple[int]


##########
# IMG file interpretation

class _IMG_SignalElements(_BaseElements):
    """
    Parser and interpreter for the signal header part of an IMG file.
    """

    __slots__ = (
        # general info
        'line_num', 'sar_rec_ind', 'left_fill', 'num_pixels', 'right_fill',
        # sensor parameters
        'update_flg', 'year', 'day', 'msec', 'chan_id', 'chan_code',
        'tx_pol', 'rcv_pol', 'prf', 'scan_id', 'rng_comp_flg', 'chirp_type',
        'chirp_length', 'chirp_const', 'chirp_lin', 'chirp_quad', 'usec',
        'gain', 'invalid_flg', 'elec_ele', 'mech_ele', 'elec_squint',
        'mech_squint', 'slant_rng', 'wind_pos',
        # platform reference
        'pos_update_flg', 'plat_lat', 'plat_lon', 'plat_alt', 'grnd_spd',
        'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'track',
        'true_track', 'pitch', 'roll', 'yaw',
        # sensor/facility specific auxiliary data
        'lat_first', 'lat_center', 'lat_last', 'lon_first', 'lon_center',
        'lon_last',
        # scansar parameters
        'burst_num', 'line_num',
        # general
        'frame_num')

    def __init__(self, fi):
        """

        Parameters
        ----------
        fi
            The file object, which has been advanced to the start
            of the record.
        """
        # define the initial common header
        super(_IMG_SignalElements, self).__init__(fi)
        # prefix data - general information
        self.line_num, self.sar_rec_ind, self.left_fill, self.num_pixels, self.right_fill = \
            struct.unpack('>IIIII', fi.read(4*5))  # type: int, int, int, int, int
        # prefix data - sensor parameters
        self.update_flg, self.year, self.day, self.msec = \
            struct.unpack('>IIII', fi.read(4*4))  # type: int, int, int, int
        self.chan_id, self.chan_code, self.tx_pol, self.rcv_pol = \
            struct.unpack('>HHHH', fi.read(4*2))  # type: int, int, int, int
        self.prf, self.scan_id = struct.unpack('>II', fi.read(2*4))  # type: int, int
        self.rng_comp_flg, self.chirp_type = struct.unpack('>HH', fi.read(2*2))  # type: int, int
        self.chirp_length, self.chirp_const, self.chirp_lin, self.chirp_quad = \
            struct.unpack('>IIII', fi.read(4*4))  # type: int, int, int, int
        self.usec = struct.unpack('>Q', fi.read(8))[0]  # type: int
        self.gain, self.invalid_flg, self.elec_ele, self.mech_ele, \
        self.elec_squint, self.mech_squint, self.slant_rng, self.wind_pos = \
            struct.unpack('>IIIIIIII', fi.read(8*4)) # type: int, int, int, int, int, int, int, int
        fi.seek(4, 1)  # skip reserved fields
        # prefix data - platform reference information
        self.pos_update_flg, self.plat_lat, self.plat_lon, self.plat_alt, self.grnd_spd = \
            struct.unpack('>IIIII', fi.read(5*4))  # type: int, int, int, int, int
        self.vel_x, self.vel_y, self.vel_z, self.acc_x, self.acc_y, self.acc_z = \
            struct.unpack('>IIIIII', fi.read(6*4))  # type: int, int, int, int, int, int
        self.track, self.true_track, self.pitch, self.roll, self.yaw = \
            struct.unpack('>IIIII', fi.read(5*4))  # type: int, int, int, int, int
        # prefix data - sensor/facility auxiliary data
        self.lat_first, self.lat_center, self.lat_last = \
            struct.unpack('>III', fi.read(3*4))  # type: int, int, int
        self.lon_first, self.lon_center, self.lon_last = \
            struct.unpack('>III', fi.read(3*4))  # type: int, int, int
        # scan sar
        self.burst_num, self.line_num = struct.unpack('>II', fi.read(2*4))  # type: int, int
        fi.seek(60, 1)  # reserved field
        self.frame_num = struct.unpack('>I', fi.read(4))[0]  # type: int
        # NB: there are remaining unparsed fields of no interest before data


class _IMG_Elements(_CommonElements2):
    """
    IMG file header parsing and interpretation
    """

    __slots__ = (
        # sample group data
        'sample_len', 'num_samples', 'num_bytes', 'just_order',
        # SAR related data
        'num_chan', 'num_lines', 'num_left', 'num_pixels', 'num_right', 'num_top',
        'num_bottom', 'interleave',
        # record data
        'phys_rec_line', 'phys_rec_multi_chan', 'prefix_bytes', 'sar_data_bytes',
        'suffix_bytes', 'pre_suf_rpt_flg',
        # prefix/suffix data locations
        'loc_sar_data', 'loc_sar_chan_num', 'loc_time', 'loc_leftfill', 'loc_rightfill',
        'pad_pixels', 'loc_data_qual', 'loc_cal_info', 'loc_gain', 'loc_bias',
        'sar_datatype', 'sar_datatype_code', 'num_leftfill', 'num_rightfill',
        'max_data_range', 'scansar_num_bursts', 'scansar_num_lines',
        'scansar_num_overlap',
        # some reserved fields for class metadata
        '_file_name', '_signal_start_location', '_signal_elements'
    )

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        if _determine_file_type(file_name) != 'IMG':
            raise IOError('file {} does not appear to be an IMG file'.format(file_name))
        self._file_name = file_name  # type: str
        self._signal_elements = None  # type: Union[None, Tuple[_IMG_SignalElements]]

        with open(self._file_name, 'rb') as fi:
            super(_IMG_Elements, self).__init__(fi)
            self._parse_fields(fi)
            self._basic_signal(fi)

    def _parse_fields(self, fi):
        """
        Parses all the field data.

        Parameters
        ----------
        fi
            The file object.

        Returns
        -------
        None
        """

        # this has advanced past the data_len field
        fi.seek(24, 1)  # skip reserved field
        # sample group data
        self.sample_len = int_func(fi.read(4))  # type: int
        self.num_samples = int_func(fi.read(4))  # type: int
        self.num_bytes = int_func(fi.read(4))  # type: int
        self.just_order = fi.read(4).decode('utf-8')  # type: str
        # SAR related data
        self.num_chan = int_func(fi.read(4))  # type: int
        self.num_lines = int_func(fi.read(8))  # type: int
        self.num_left = int_func(fi.read(4))  # type: int
        self.num_pixels = int_func(fi.read(8))  # type: int
        self.num_right = int_func(fi.read(4))  # type: int
        self.num_top = int_func(fi.read(4))  # type: int
        self.num_bottom = int_func(fi.read(4))  # type: int
        self.interleave = fi.read(4).decode('utf-8')  # type: str
        # record data
        self.phys_rec_line = int_func(fi.read(2))  # type: int
        self.phys_rec_multi_chan = int_func(fi.read(2))  # type: int
        self.prefix_bytes = int_func(fi.read(4))  # type: int
        self.sar_data_bytes = int_func(fi.read(8))  # type: int
        self.suffix_bytes = int_func(fi.read(4))  # type: int
        self.pre_suf_rpt_flg = fi.read(4).decode('utf-8')  # type: str
        # prefix/suffix data locations
        self.loc_sar_data = fi.read(8).decode('utf-8')  # type: str
        self.loc_sar_chan_num = fi.read(8).decode('utf-8')  # type: str
        self.loc_time = fi.read(8).decode('utf-8')  # type: str
        self.loc_leftfill = fi.read(8).decode('utf-8')  # type: str
        self.loc_rightfill = fi.read(8).decode('utf-8')  # type: str
        self.pad_pixels = fi.read(4).decode('utf-8')  # type: str
        fi.seek(28, 1)  # skip resevered fields
        self.loc_data_qual = fi.read(8).decode('utf-8')  # type: str
        self.loc_cal_info = fi.read(8).decode('utf-8')  # type: str
        self.loc_gain = fi.read(8).decode('utf-8')  # type: str
        self.loc_bias = fi.read(8).decode('utf-8')  # type: str
        self.sar_datatype = fi.read(28).decode('utf-8')  # type: str
        self.sar_datatype_code = fi.read(4).decode('utf-8')  # type: str
        self.num_leftfill = fi.read(4).decode('utf-8')  # type: str
        self.num_rightfill = fi.read(4).decode('utf-8')  # type: str
        self.max_data_range = fi.read(8).decode('utf-8')  # type: str
        self.scansar_num_bursts = fi.read(4).decode('utf-8')  # type: str
        self.scansar_num_lines = fi.read(4).decode('utf-8')  # type: str
        self.scansar_num_overlap = fi.read(4).decode('utf-8')  # type: str
        fi.seek(260, 1)  # skip reserved field
        self._signal_start_location = fi.tell()

    def _parse_signal(self, fi, index):
        """
        Parse the signal element at the given index.

        Parameters
        ----------
        fi
            The open file object.
        index : int

        Returns
        -------
        _IMG_SignalElements
        """

        index = int_func(index)
        if not (0 <= index < self.num_data_rec):
            raise KeyError('index {} must be in range [0, {})'.format(index, self.num_data_rec))
        # find offset for the given record, and traverse to it
        record_offset = self._signal_start_location + \
                        (self.prefix_bytes + self.num_pixels*self.num_bytes)*index
        # go to the start of the given record
        fi.seek(record_offset, 0)
        return _IMG_SignalElements(fi)

    def _basic_signal(self, fi):
        """
        Parse the signal portion of the IMG file.

        Parameters
        ----------
        fi
            The file object.

        Returns
        -------
        None
        """

        file_id = self.file_id[7]
        if file_id == 'B':
            # signal data records
            #   we only need the first and potentially last record (for now?)
            self._signal_elements = (
                self._parse_signal(fi, 0),
                self._parse_signal(fi, self.num_data_rec-1))

        elif file_id in ['C', 'D']:
            logging.warning('Processed data IMG file {}'.format(self._file_name))
        else:
            raise ValueError('Got unhandled file_id {} in IMG file {}'.format(
                self.file_id, self._file_name))

    def construct_chipper(self, side_of_track):
        """
        Construct the chipper associated with the IMG file.

        Parameters
        ----------
        side_of_track : str

        Returns
        -------
        BaseChipper
        """

        if self.sar_datatype_code.strip() == 'C*8':
            raw_dtype = 'float32'
            raw_bands = 2
            pixel_size = 8
        else:
            raw_dtype = 'int16'
            raw_bands = 2
            pixel_size = 4

        if side_of_track == 'L':
            symmetry = (False, True, True)
        else:
            symmetry = (False, False, True)

        prefix_bytes = self.prefix_bytes
        suffix_bytes = self.suffix_bytes
        if (prefix_bytes % pixel_size) != 0:
            raise ValueError('prefix size is not compatible with pixel size')
        if (suffix_bytes % pixel_size) != 0:
            raise ValueError('suffix size is not compatible with pixel size')
        pref_cols = int(prefix_bytes/pixel_size)
        suf_cols = int(suffix_bytes/pixel_size)
        data_size = (self.num_lines, pref_cols+self.num_pixels+suf_cols)
        sub_start = suf_cols if symmetry[1] else pref_cols
        sub_end = self.num_pixels+pref_cols if symmetry[1] else self.num_pixels+suf_cols

        output_bands = 1
        output_dtype = 'complex64'
        transform_data = 'COMPLEX'
        p_chipper = BIPChipper(
            self._file_name, raw_dtype, data_size, raw_bands, output_bands, output_dtype,
            symmetry, transform_data, data_offset=self.rec_length)
        return SubsetChipper(p_chipper, dim1bounds=(sub_start, sub_end), dim2bounds=(0, self.num_lines))


###########
# LED file interpretation

class _LED_Data(_BaseElements):
    """
    The data set summary in the LED file.
    """

    __slots__ = (
        'rec_seq', 'sar_id', 'scene_id', 'num_scene_ref', 'scene_ctr_time',
        'geo_lat', 'geo_long', 'heading', 'ellips', 'semimajor', 'semiminor',
        'earth_mass', 'grav_const', 'J2', 'J3', 'J4', 'avg_terr', 'ctr_line',
        'ctr_pixel', 'proc_length', 'proc_width', 'sar_chan', 'platform',
        'sensor_id_mode', 'orbit', 'sensor_lat', 'sensor_lon', 'sensor_heading',
        'clock_angle', 'incidence', 'wavelength', 'mocomp', 'range_pulse_code',
        'range_pulse_amp_coef', 'range_pulse_phs_coef', 'chirp_index',
        'sampling_rate', 'range_gate', 'pulse_width', 'baseband_flg',
        'range_compressed_flg', 'rec_gain_like_pol', 'rec_gain_cross_pol',
        'quant_bit', 'quant_desc', 'dc_bias_i', 'dc_bias_q', 'gain_imbalance',
        'elec_bores', 'mech_bores', 'echo_tracker', 'prf', 'ant_beam_2way_el',
        'ant_beam_2way_az', 'sat_time', 'sat_clock', 'sat_clock_inc',
        'proc_fac', 'proc_sys', 'proc_ver', 'proc_fac_code', 'proc_lvl_code',
        'prod_type', 'proc_alg', 'num_look_az', 'num_look_rng', 'bw_per_look_az',
        'bw_per_look_rng', 'bw_az', 'bw_rng', 'wgt_az', 'wgt_rng',
        'data_input_src', 'res_grnd_rng', 'res_az', 'rad_bias', 'rad_gain',
        'at_dop', 'xt_dop', 'time_dir_pixel', 'time_dir_line',
        'at_dop_rate', 'xt_dop_rate', 'line_constant', 'clutter_lock_flg',
        'autofocus_flg', 'line_spacing', 'pixel_spacing', 'rng_comp_des', 'dop_freq',
        'cal_mode_loc_flag', 'start_line_cal_start', 'end_line_cal_start',
        'start_line_cal_end', 'end_line_cal_end',
        'prf_switch', 'prf_switch_line', 'beam_ctr_dir', 'yaw_steer_flag',
        'param_table_num', 'off_nadir', 'ant_beam_num', 'incidence_ang',
        'num_annot')

    def __init__(self, fi):
        super(_LED_Data, self).__init__(fi)
        self.rec_seq = fi.read(4).decode('utf-8')  # type: str
        self.sar_id = fi.read(4).decode('utf-8')  # type: str
        self.scene_id = fi.read(32).decode('utf-8')  # type: str
        self.num_scene_ref = fi.read(16).decode('utf-8')  # type: str
        self.scene_ctr_time = fi.read(32).decode('utf-8')  # type: str
        fi.seek(16, 1)  # skip reserved fields
        self.geo_lat = fi.read(16).decode('utf-8')  # type: str
        self.geo_long = fi.read(16).decode('utf-8')  # type: str
        self.heading = fi.read(16).decode('utf-8')  # type: str
        self.ellips = fi.read(16).decode('utf-8')  # type: str
        self.semimajor = fi.read(16).decode('utf-8')  # type: str
        self.semiminor = fi.read(16).decode('utf-8')  # type: str
        self.earth_mass = fi.read(16).decode('utf-8')  # type: str
        self.grav_const = fi.read(16).decode('utf-8')  # type: str
        self.J2 = fi.read(16).decode('utf-8')  # type: str
        self.J3 = fi.read(16).decode('utf-8')  # type: str
        self.J4 = fi.read(16).decode('utf-8')  # type: str
        fi.seek(16, 1)  # skip reserved fields
        self.avg_terr = fi.read(16).decode('utf-8')  # type: str
        self.ctr_line = int_func(fi.read(8))  # type: int
        self.ctr_pixel = int_func(fi.read(8))  # type: int
        self.proc_length = fi.read(16).decode('utf-8')  # type: str
        self.proc_width = fi.read(16).decode('utf-8')  # type: str
        fi.seek(16, 1)  # skip reserved fields
        self.sar_chan = fi.read(4).decode('utf-8')  # type: str
        fi.seek(4, 1)  # skip reserved fields
        self.platform = fi.read(16).decode('utf-8')  # type: str
        self.sensor_id_mode = fi.read(32).decode('utf-8')  # type: str
        self.orbit = fi.read(8).decode('utf-8')  # type: str
        self.sensor_lat = fi.read(8).decode('utf-8')  # type: str
        self.sensor_lon = fi.read(8).decode('utf-8')  # type: str
        self.sensor_heading = fi.read(8).decode('utf-8')  # type: str
        self.clock_angle = float(fi.read(8))  # type: float
        self.incidence = float(fi.read(8))  # type: float
        fi.seek(8, 1)  # skip reserved fields
        self.wavelength = float(fi.read(16))  # type: float
        self.mocomp = fi.read(2).decode('utf-8')  # type: str
        self.range_pulse_code = fi.read(16).decode('utf-8')  # type: str
        self.range_pulse_amp_coef = [fi.read(16).decode('utf-8') for _ in range(5)]  # type: List[str]
        self.range_pulse_phs_coef = [fi.read(16).decode('utf-8') for _ in range(5)]  # type: List[str]
        self.chirp_index = fi.read(8).decode('utf-8')  # type: str
        fi.seek(8, 1)  # skip reserved fields
        self.sampling_rate = float(fi.read(16))  # type: float
        self.range_gate = float(fi.read(16))  # type: float
        self.pulse_width = float(fi.read(16))  # type: float
        self.baseband_flg = fi.read(4).decode('utf-8')  # type: str
        self.range_compressed_flg = fi.read(4).decode('utf-8')  # type: str
        self.rec_gain_like_pol = float(fi.read(16))  # type: float
        self.rec_gain_cross_pol = float(fi.read(16))  # type: float
        self.quant_bit = fi.read(8).decode('utf-8')  # type: str
        self.quant_desc = fi.read(12).decode('utf-8')  # type: str
        self.dc_bias_i = float(fi.read(16))  # type: float
        self.dc_bias_q = float(fi.read(16))  # type: float
        self.gain_imbalance = float(fi.read(16))  # type: float
        fi.seek(32, 1)  # skip reserved fields
        self.elec_bores = float(fi.read(16))  # type: float
        self.mech_bores = float(fi.read(16))  # type: float
        self.echo_tracker = fi.read(4).decode('utf-8')  # type: str
        self.prf = float(fi.read(16))  # type: float
        self.ant_beam_2way_el = float(fi.read(16))  # type: float
        self.ant_beam_2way_az = float(fi.read(16))  # type: float
        self.sat_time = fi.read(16).decode('utf-8')  # type: str
        self.sat_clock = fi.read(32).decode('utf-8')  # type: str
        self.sat_clock_inc = fi.read(16).decode('utf-8')  # type: str
        self.proc_fac = fi.read(16).decode('utf-8')  # type: str
        self.proc_sys = fi.read(8).decode('utf-8')  # type: str
        self.proc_ver = fi.read(8).decode('utf-8')  # type: str
        self.proc_fac_code = fi.read(16).decode('utf-8')  # type: str
        self.proc_lvl_code = fi.read(16).decode('utf-8')  # type: str
        self.prod_type = fi.read(32).decode('utf-8')  # type: str
        self.proc_alg = fi.read(32).decode('utf-8')  # type: str
        self.num_look_az = float(fi.read(16))  # type: float
        self.num_look_rng = float(fi.read(16))  # type: float
        self.bw_per_look_az = float(fi.read(16))  # type: float
        self.bw_per_look_rng = float(fi.read(16))  # type: float
        self.bw_az = float(fi.read(16))  # type: float
        self.bw_rng = float(fi.read(16))  # type: float
        self.wgt_az = fi.read(32).decode('utf-8')  # type: str
        self.wgt_rng = fi.read(32).decode('utf-8')  # type: str
        self.data_input_src = fi.read(16).decode('utf-8')  # type: str
        self.res_grnd_rng = fi.read(16).decode('utf-8')  # type: str
        self.rad_bias = fi.read(16).decode('utf-8')  # type: str
        self.res_az = fi.read(16).decode('utf-8')  # type: str
        self.rad_gain = fi.read(16).decode('utf-8')  # type: str
        self.at_dop = [float(fi.read(16)) for _ in range(3)]  # type: List[float]
        fi.seek(16, 1)  # skip reserved fields
        self.xt_dop = [float(fi.read(16)) for _ in range(3)]  # type: List[float]
        self.time_dir_pixel = fi.read(8).decode('utf-8')  # type: str
        self.time_dir_line = fi.read(8).decode('utf-8')  # type: str
        self.at_dop_rate = [float(fi.read(16)) for _ in range(3)]  # type: List[float]
        fi.seek(16, 1)  # skip reserved fields
        self.xt_dop_rate = [float(fi.read(16)) for _ in range(3)]  # type: List[float]
        fi.seek(16, 1)  # skip reserved fields
        self.line_constant = fi.read(8).decode('utf-8')  # type: str
        self.clutter_lock_flg = fi.read(4).decode('utf-8')  # type: str
        self.autofocus_flg = fi.read(4).decode('utf-8')  # type: str
        self.line_spacing = float(fi.read(16))  # type: float
        self.pixel_spacing = float(fi.read(16))  # type: float
        self.rng_comp_des = fi.read(16).decode('utf-8')  # type: str
        self.dop_freq = [float(fi.read(16)) for _ in range(2)]  # type: List[float]
        self.cal_mode_loc_flag = fi.read(4).decode('utf-8')  # type: str
        self.start_line_cal_start = fi.read(8).decode('utf-8')  # type: str
        self.end_line_cal_start = fi.read(8).decode('utf-8')  # type: str
        self.start_line_cal_end = fi.read(8).decode('utf-8')  # type: str
        self.end_line_cal_end = fi.read(8).decode('utf-8')  # type: str
        self.prf_switch = fi.read(4).decode('utf-8')  # type: str
        self.prf_switch_line = fi.read(8).decode('utf-8')  # type: str
        self.beam_ctr_dir = float(fi.read(16))  # type: float
        self.yaw_steer_flag = fi.read(4).decode('utf-8')  # type: str
        self.param_table_num = fi.read(4).decode('utf-8')  # type: str
        self.off_nadir = float(fi.read(16))  # type: float
        self.ant_beam_num = fi.read(4).decode('utf-8')  # type: str
        fi.seek(28, 1)  # skip reserved fields
        self.incidence_ang = [float(fi.read(20)) for _ in range(6)]  # type: List[float]
        self.num_annot = float(fi.read(8))  # type: float
        fi.seek(8 + 64*32 + 26, 1)  # skip reserved fields, for map projection?


class _LED_Position(_BaseElements):
    """
    The position summary in the LED file.
    """

    __slots__ = (
        'orb_elem', 'pos', 'vel', 'num_pts', 'year', 'month', 'day', 'day_in_year',
        'sec', 'int', 'ref_coord_sys', 'greenwich_mean_hr_ang',
        'at_pos_err', 'ct_pos_err', 'rad_pos_err',
        'at_vel_err', 'ct_vel_err', 'rad_vel_err',
        'pts_pos', 'pts_vel', 'leap_sec')

    def __init__(self, fi):
        super(_LED_Position, self).__init__(fi)
        self.orb_elem = fi.read(32).decode('utf-8')  # type: str
        self.pos = numpy.array([float(fi.read(16)) for _ in range(3)], dtype='float64')  # type: numpy.ndarray
        self.vel = numpy.array([float(fi.read(16)) for _ in range(3)], dtype='float64')  # type: numpy.ndarray
        self.num_pts = int_func(fi.read(4))  # type: int
        self.year = int_func(fi.read(4))  # type: int
        self.month = int_func(fi.read(4))  # type: int
        self.day = int_func(fi.read(4))  # type: int
        self.day_in_year = int_func(fi.read(4))  # type: int
        self.sec = float(fi.read(22))  # type: float
        self.int = float(fi.read(22))  # type: float
        self.ref_coord_sys = fi.read(64).decode('utf-8')  # type: str
        self.greenwich_mean_hr_ang = fi.read(22).decode('utf-8')  # type: str
        self.at_pos_err = float(fi.read(16))  # type: float
        self.ct_pos_err = float(fi.read(16))  # type: float
        self.rad_pos_err = float(fi.read(16))  # type: float
        self.at_vel_err = float(fi.read(16))  # type: float
        self.ct_vel_err = float(fi.read(16))  # type: float
        self.rad_vel_err = float(fi.read(16))  # type: float
        self.pts_pos = numpy.zeros((self.num_pts, 3), dtype='float64')  # type: numpy.ndarray
        self.pts_vel = numpy.zeros((self.num_pts, 3), dtype='float64')  # type: numpy.ndarray
        for i in range(self.num_pts):
            self.pts_pos[i, :] = [float(fi.read(22)) for _ in range(3)]
            self.pts_vel[i, :] = [float(fi.read(22)) for _ in range(3)]
        fi.seek(18, 1)  # skip reserved fields
        self.leap_sec = fi.read(1).decode('utf-8')  # type: str
        fi.seek(579, 1)  # skip reserved fields


class _LED_AttitudePoint(object):
    """
    An attitude point.
    """
    __slots__ = (
        'day_year', 'msec_day',
        'pitch_flag', 'roll_flag', 'yaw_flag',
        'pitch', 'roll', 'yaw',
        'pitch_rate_flag', 'roll_rate_flag', 'yaw_rate_flag',
        'pitch_rate', 'roll_rate', 'yaw_rate')

    def __init__(self, fi):
        self.day_year = int_func(fi.read(4))  # type: int
        self.msec_day = int_func(fi.read(8))  # type: int
        self.pitch_flag = fi.read(4).decode('utf-8')  # type: str
        self.roll_flag = fi.read(4).decode('utf-8')  # type: str
        self.yaw_flag = fi.read(4).decode('utf-8')  # type: str
        self.pitch = float(fi.read(14))  # type: float
        self.roll = float(fi.read(14))  # type: float
        self.yaw = float(fi.read(14))  # type: float
        self.pitch_rate_flag = fi.read(4).decode('utf-8')  # type: str
        self.roll_rate_flag = fi.read(4).decode('utf-8')  # type: str
        self.yaw_rate_flag = fi.read(4).decode('utf-8')  # type: str
        self.pitch_rate = float(fi.read(14))  # type: float
        self.roll_rate = float(fi.read(14))  # type: float
        self.yaw_rate = float(fi.read(14))  # type: float


class _LED_Attitude(_BaseElements):
    """
    The attitude summary in the LED file.
    """

    __slots__ = (
        'num_pts', 'pts')

    def __init__(self, fi):
        super(_LED_Attitude, self).__init__(fi)
        self.num_pts = int_func(fi.read(4))  # type: int
        self.pts = [_LED_AttitudePoint(fi) for _ in range(self.num_pts)]
        extra = self.rec_length - (16 + 120*self.num_pts)
        fi.seek(extra, 1)  # skip remaining reserved


class _LED_Radiometric(_BaseElements):
    """
    The attitude summary in the LED file.
    """

    __slots__ = (
        'seq_num', 'num_pts', 'cal_factor', 'tx_distortion', 'rcv_distortion')

    def __init__(self, fi):
        super(_LED_Radiometric, self).__init__(fi)
        self.seq_num = int_func(fi.read(4))  # type: int
        self.num_pts = int_func(fi.read(4))  # type: int
        self.cal_factor = float(fi.read(16))  # type: float
        self.tx_distortion = numpy.zeros((2, 2), dtype='complex128')  # type: numpy.ndarray
        self.rcv_distortion = numpy.zeros((2, 2), dtype='complex128')  # type: numpy.ndarray
        for i in range(2):
            for j in range(2):
                self.tx_distortion[i, j] = complex(real=float(fi.read(16)), imag=float(fi.read(16)))
        for i in range(2):
            for j in range(2):
                self.rcv_distortion[i, j] = complex(real=float(fi.read(16)), imag=float(fi.read(16)))
        extra = self.rec_length - (12 + 24 + 16*2*4*2)
        fi.seek(extra, 1)  # skip remaining reserved


class _LED_DataQuality(_BaseElements):
    """
    The data quality summary in the LED file.
    """

    __slots__ = (
        'dq_rec_num', 'chan_id', 'date', 'num_chans', 'islr', 'pslr',
        'aar', 'rar', 'snr', 'ber', 'sr_res', 'az_res', 'rad_res', 'dyn_rng',
        'abs_cal_mag', 'abs_cal_phs', 'rel_cal_mag', 'rel_cal_phs',
        'abs_err_at', 'abs_err_ct', 'distort_line', 'distort_pixel',
        'distort_skew', 'orient_err', 'at_misreg_err', 'ct_misreg_err')

    def __init__(self, fi):
        super(_LED_DataQuality, self).__init__(fi)
        self.dq_rec_num = fi.read(4).decode('utf-8')  # type: str
        self.chan_id = fi.read(4).decode('utf-8')  # type: str
        self.date = fi.read(6).decode('utf-8')  # type: str
        self.num_chans = int_func(fi.read(4))  # type: int
        self.islr = float(fi.read(16))  # type: float
        self.pslr = float(fi.read(16))  # type: float
        self.aar = float(fi.read(16))  # type: float
        self.rar = float(fi.read(16))  # type: float
        self.snr = float(fi.read(16))  # type: float
        self.ber = fi.read(16).decode('utf-8')  # type: str
        self.sr_res = float(fi.read(16))  # type: float
        self.az_res = float(fi.read(16))  # type: float
        self.rad_res = fi.read(16).decode('utf-8')  # type: str
        self.dyn_rng = fi.read(16).decode('utf-8')  # type: str
        self.abs_cal_mag = fi.read(16).decode('utf-8')  # type: str
        self.abs_cal_phs = fi.read(16).decode('utf-8')  # type: str
        self.rel_cal_mag = numpy.zeros((self.num_chans, ), dtype='float64')  # type: numpy.ndarray
        self.rel_cal_phs = numpy.zeros((self.num_chans, ), dtype='float64')  # type: numpy.ndarray
        for i in range(self.num_chans):
            self.rel_cal_mag[i] = _make_float(fi.read(16))
            self.rel_cal_phs[i] = _make_float(fi.read(16))
        fi.seek(480 - self.num_chans*32, 1)  # skip reserved
        self.abs_err_at = fi.read(16).decode('utf-8')  # type: str
        self.abs_err_ct = fi.read(16).decode('utf-8')  # type: str
        self.distort_line = fi.read(16).decode('utf-8')  # type: str
        self.distort_skew = fi.read(16).decode('utf-8')  # type: str
        self.orient_err = fi.read(16).decode('utf-8')  # type: str
        self.at_misreg_err = numpy.zeros((self.num_chans, ), dtype='float64')  # type: numpy.ndarray
        self.ct_misreg_err = numpy.zeros((self.num_chans, ), dtype='float64')  # type: numpy.ndarray
        for i in range(self.num_chans):
            self.at_misreg_err[i] = _make_float(fi.read(16))
            self.ct_misreg_err[i] = _make_float(fi.read(16))
        this_size = 782 + 32*self.num_chans
        fi.seek(self.rec_length-this_size, 1)  # skip reserved


class _LED_Facility(_BaseElements):
    """
    The data quality summary in the LED file.
    """

    __slots__ = (
        'fac_seq_num', 'mapproj2pix', 'mapproj2line', 'cal_mode_data_loc_flg',
        'start_line_upper', 'end_line_upper',
        'start_line_bottom', 'end_line_bottom',
        'prf_switch_flag', 'prf_switch_line',
        'num_loss_lines_10', 'num_loss_lines_11',
        'pixelline2lat', 'pixelline2lon', 'origin_pixel', 'origin_line',
        'latlon2pixel', 'latlon2line', 'origin_lat', 'origin_lon')

    def __init__(self, fi, parse_all=False):
        super(_LED_Facility, self).__init__(fi)
        self.fac_seq_num = struct.unpack('>I', fi.read(4))[0]  # type: int
        if not parse_all:
            fi.seek(self.rec_length-16, 1)
            return

        self.mapproj2pix = numpy.zeros((10, ), dtype='float64')  # type: numpy.ndarray
        for i in range(10):
            self.mapproj2pix[i] = _make_float(fi.read(20))
        self.mapproj2line = numpy.zeros((10, ), dtype='float64')  # type: numpy.ndarray
        for i in range(10):
            self.mapproj2line[i] = _make_float(fi.read(20))

        self.cal_mode_data_loc_flg = fi.read(4).decode('utf-8')  # type: str
        self.start_line_upper = fi.read(8).decode('utf-8')  # type: str
        self.end_line_upper = fi.read(8).decode('utf-8')  # type: str
        self.start_line_bottom = fi.read(8).decode('utf-8')  # type: str
        self.end_line_bottom = fi.read(8).decode('utf-8')  # type: str
        self.prf_switch_flag = fi.read(4).decode('utf-8')  # type: str
        self.prf_switch_line = fi.read(8).decode('utf-8')  # type: str
        fi.seek(8, 1)  # skip reserved field
        self.num_loss_lines_10 = fi.read(8).decode('utf-8')  # type: str
        self.num_loss_lines_11 = fi.read(8).decode('utf-8')  # type: str
        fi.seek(312, 1)  # skip empty fields
        fi.seek(224, 1)  # skip reserved fields

        self.pixelline2lat = numpy.zeros((25, ), dtype='float64')  # type: numpy.ndarray
        self.pixelline2lon = numpy.zeros((25, ), dtype='float64')  # type: numpy.ndarray
        for i in range(25):
            self.pixelline2lat[i] = _make_float(fi.read(20))
        for i in range(25):
            self.pixelline2lon[i] = _make_float(fi.read(20))

        self.origin_pixel = float(fi.read(20))  # type: float
        self.origin_line = float(fi.read(20))  # type: float

        self.latlon2pixel = numpy.zeros((25, ), dtype='float64')  # type: numpy.ndarray
        self.latlon2line = numpy.zeros((25, ), dtype='float64')  # type: numpy.ndarray
        for i in range(25):
            self.latlon2pixel[i] = _make_float(fi.read(20))
        for i in range(25):
            self.latlon2line[i] = _make_float(fi.read(20))

        self.origin_lat = float(fi.read(20))  # type: float
        self.origin_lon = float(fi.read(20))  # type: float
        fi.seek(1896, 1)  # skip empty fields


class _LED_Elements(_CommonElements3):
    """
    LED file header parsing and interpretation
    """

    __slots__ = (
        '_file_name', 'data', 'position', 'attitude', 'radiometric',
        'data_quality', 'facility')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        if _determine_file_type(file_name) != 'LED':
            raise IOError('file {} does not appear to be an LED file'.format(file_name))
        self._file_name = file_name  # type: str
        with open(self._file_name, 'rb') as fi:
            super(_LED_Elements, self).__init__(fi)
            fi.seek(230, 1)  # skip reserved fields
            self.data = _LED_Data(fi)  # type: _LED_Data
            if self.num_map_rec > 0:
                # skip any map projection
                # should not be present for level 1.1
                fi.seek(self.map_len, 1)
            self.position = _LED_Position(fi)  # type: _LED_Position
            self.attitude = _LED_Attitude(fi)  # type: _LED_Attitude
            self.radiometric = _LED_Radiometric(fi)  # type: _LED_Radiometric
            self.data_quality = _LED_DataQuality(fi)  # type: _LED_DataQuality
            self.facility = [_LED_Facility(fi, parse_all=False) for _ in range(4)]  # type: List[_LED_Facility]
            self.facility.append(_LED_Facility(fi, parse_all=True))


############
# TRL file interpretation

class _TRL_LowResRecord(object):
    """
    Low resolution record in TRL file.
    """

    __slots__ = ('length', 'pixels', 'lines', 'bytes')

    def __init__(self, fi):
        self.length = int_func(fi.read(8))
        self.pixels = int_func(fi.read(6))
        self.lines = int_func(fi.read(6))
        self.bytes = int_func(fi.read(6))


class _TRL_Elements(_CommonElements3):
    """
    TRL file header parsing and interpretation
    """

    __slots__ = (
        '_file_name', 'num_low_res_rec', 'low_res')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        if _determine_file_type(file_name) != 'TRL':
            raise IOError('file {} does not appear to be an LED file'.format(file_name))
        self._file_name = file_name  # type: str
        with open(self._file_name, 'rb') as fi:
            super(_TRL_Elements, self).__init__(fi)
            self.num_low_res_rec = int_func(fi.read(6))  # type: int
            self.low_res = tuple([_TRL_LowResRecord(fi) for _ in range(self.num_low_res_rec)])
            fi.seek(720, 1)  # skip reserved data
            # comment carried over from matlab -
            #   There seems to be an array the size of the low resolution image on
            #   the end of this file, but it doesn't seem to contain any data


############
# VOL file interpretation

class _VOL_File(_BaseElements2):
    """
    The VOL file object.
    """

    __slots__ = (
        'num', 'name', 'clas', 'clas_code', 'typ', 'typ_code', 'num_recs',
        'len_first_rec', 'max_rec_len', 'rec_len_type', 'rec_len_type_code',
        'phys_vol_first', 'phys_vol_last', 'rec_num_first', 'rec_num_last')

    def __init__(self, fi):
        super(_VOL_File, self).__init__(fi)
        self.num = fi.read(4).decode('utf-8')  # type: str
        self.name = fi.read(16).decode('utf-8')  # type: str
        self.clas = fi.read(28).decode('utf-8')  # type: str
        self.clas_code = fi.read(4).decode('utf-8')  # type: str
        self.typ = fi.read(28).decode('utf-8')  # type: str
        self.typ_code = fi.read(4).decode('utf-8')  # type: str
        self.num_recs = int_func(fi.read(8))  # type: int
        self.len_first_rec = int_func(fi.read(8))  # type: int
        self.max_rec_len = int_func(fi.read(8))  # type: int
        self.rec_len_type = fi.read(12).decode('utf-8')  # type: str
        self.rec_len_type_code = fi.read(4).decode('utf-8')  # type: str
        self.phys_vol_first = int_func(fi.read(2))  # type: int
        self.phys_vol_last = int_func(fi.read(2))  # type: int
        self.rec_num_first = int_func(fi.read(8))  # type: int
        self.rec_num_last = int_func(fi.read(8))  # type: int
        fi.seek(200, 1)  # skipping reserved fields


class _VOL_Text(_BaseElements2):
    """
    The VOL text object.
    """

    __slots__ = (
        'prod_id', 'location', 'phys_id', 'scene_id', 'scene_loc_id')

    def __init__(self, fi):
        super(_VOL_Text, self).__init__(fi)
        self.prod_id = fi.read(40).decode('utf-8')  # type: str
        self.location = fi.read(60).decode('utf-8')  # type: str
        self.phys_id = fi.read(40).decode('utf-8')  # type: str
        self.scene_id = fi.read(40).decode('utf-8')  # type: str
        self.scene_loc_id = fi.read(40).decode('utf-8')  # type: str
        fi.seek(124, 1)  # skip reserved fields


class _VOL_Elements(_CommonElements):
    """
    VOL file header parsing and interpretation
    """

    __slots__ = (
        '_file_name', 'phys_vol_id', 'log_vol_id', 'vol_set_id', 'num_phys_vol',
        'phys_seq_first', 'phys_seq_last', 'phys_seq_cur', 'file_num', 'log_vol',
        'log_vol_num', 'log_vol_create_date', 'log_vol_create_time', 'log_vol_co',
        'log_vol_agency', 'log_vol_facility', 'num_file_ptr', 'num_text_rec',
        'files', 'texts')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        if _determine_file_type(file_name) != 'VOL':
            raise IOError('file {} does not appear to be an LED file'.format(file_name))
        self._file_name = file_name  # type: str
        with open(self._file_name, 'rb') as fi:
            super(_VOL_Elements, self).__init__(fi)
            self.phys_vol_id = fi.read(16).decode('utf-8')  # type: str
            self.log_vol_id = fi.read(16).decode('utf-8')  # type: str
            self.vol_set_id = fi.read(16).decode('utf-8')  # type: str
            self.num_phys_vol = fi.read(2).decode('utf-8')  # type: str
            self.phys_seq_first = fi.read(2).decode('utf-8')  # type: str
            self.phys_seq_last = fi.read(2).decode('utf-8')  # type: str
            self.phys_seq_cur = fi.read(2).decode('utf-8')  # type: str
            self.file_num = fi.read(4).decode('utf-8')  # type: str
            self.log_vol = fi.read(4).decode('utf-8')  # type: str
            self.log_vol_num = fi.read(4).decode('utf-8')  # type: str
            self.log_vol_create_date = fi.read(8).decode('utf-8')  # type: str
            self.log_vol_create_time = fi.read(8).decode('utf-8')  # type: str
            self.log_vol_co = fi.read(12).decode('utf-8')  # type: str
            self.log_vol_agency = fi.read(8).decode('utf-8')  # type: str
            self.log_vol_facility = fi.read(12).decode('utf-8')  # type: str
            self.num_file_ptr = int_func(fi.read(4))  # type: int
            self.num_text_rec = int_func(fi.read(4))  # type: int
            fi.seek(192, 1)
            self.files = tuple([_VOL_File(fi) for _ in range(self.num_file_ptr)])  # type: Tuple[_VOL_File]
            self.texts = tuple([_VOL_Text(fi) for _ in range(self.num_text_rec)])  # type: Tuple[_VOL_Text]


if __name__ == '__main__':
    root_dir = os.path.expanduser(
        '~/Desktop/sarpy_testing/palsar'
        '/0000000000_001001_ALOS2229210750-180821-L1.1CEOS')

    # # test img
    # img_file = os.path.join(root_dir, 'IMG-HH-ALOS2229210750-180821-HBQR1.1__A')
    # img_elements = _IMG_Elements(img_file)
    # print(img_elements.doc_id)

    # # test led
    # led_file = os.path.join(root_dir, 'LED-ALOS2229210750-180821-HBQR1.1__A')
    # led_elements = _LED_Elements(led_file)
    # print(led_elements.doc_id)

    # # test trl
    # trl_file = os.path.join(root_dir, 'TRL-ALOS2229210750-180821-HBQR1.1__A')
    # trl_elements = _TRL_Elements(trl_file)
    # print(trl_elements.doc_id)

    # # test vol
    # vol_file = os.path.join(root_dir, 'VOL-ALOS2229210750-180821-HBQR1.1__A')
    # vol_elements = _VOL_Elements(vol_file)
    # print(vol_elements.doc_id)
