# -*- coding: utf-8 -*-
"""
Functionality for reading PALSAR ALOS 2 data into a SICD model.
"""

import logging
import os
import struct
from typing import Union, Tuple

import numpy

from sarpy.io.general.base import BaseChipper, SubsetChipper
from sarpy.io.general.bip import BIPChipper

from sarpy.compliance import int_func

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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

##########
# helper class that contains common header elements

class _BaseHeader(object):
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

        self.rec_num = struct.unpack('I', fi.read(4))[0]  # type: int
        self.rec_subtype1, self.rec_type, self.rec_subtype2, self.rec_subtype3 = \
            struct.unpack('BBBB', fi.read(4)) # type: int, int, int, int
        self.rec_length = struct.unpack('I', fi.read(4))[0]  # type: int


############
# Header elements common to IMG, LED, TRL, and VOL

class _CommonHeader(_BaseHeader):
    """
    Parser and interpreter for the elements common to IMG, LED, TRL, and VOL files
    """

    __slots__ = (
        'ascii_ebcdic', 'doc_id', 'doc_rev', 'rec_rev', 'soft_rel_rev')

    def __init__(self, fi):
        """

        Parameters
        ----------
        fi
            The open file reader.
        """

        super(_CommonHeader, self).__init__(fi)
        self.ascii_ebcdic = fi.read(2).decode('utf-8')  # type: str
        fi.seek(2, 1) # skip reserved field
        self.doc_id = fi.read(12).decode('utf-8')  # type: str
        self.doc_rev = fi.read(2).decode('utf-8')  # type: str
        self.rec_rev = fi.read(2).decode('utf-8')  # type: str
        self.soft_rel_rev = fi.read(12).decode('utf-8')  # type: str

###############
# Header elements common to IMG, LED, and TRL

class _CommonHeader2(_CommonHeader):
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

        super(_CommonHeader2, self).__init__(fi)
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


##########
# IMG file interpretation

class _SignalHeader(_BaseHeader):
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
        super(_SignalHeader, self).__init__(fi)
        # prefix data - general information
        self.line_num, self.sar_rec_ind, self.left_fill, self.num_pixels, self.right_fill = \
            struct.unpack('5I', fi.read(4*5))  # type: int, int, int, int, int
        # prefix data - sensor parameters
        self.update_flg, self.year, self.day, self.msec = \
            struct.unpack('4I', fi.read(4*4))  # type: int, int, int, int
        self.chan_id, self.chan_code, self.tx_pol, self.rcv_pol = \
            struct.unpack('4H', fi.read(4*2))  # type: int, int, int, int
        self.prf, self.scan_id = struct.unpack('2I', fi.read(2*4))  # type: int, int
        self.rng_comp_flg, self.chirp_type = struct.unpack('2H', fi.read(2*2))  # type: int, int
        self.chirp_length, self.chirp_const, self.chirp_lin, self.chirp_quad = \
            struct.unpack('4I', fi.read(4*4))  # type: int, int, int, int
        self.usec = struct.unpack('Q', fi.read(8))[0]  # type: int
        self.gain, self.invalid_flg, self.elec_ele, self.mech_ele, \
        self.elec_squint, self.mech_squint, self.slant_rng, self.wind_pos = \
            struct.unpack('8I', fi.read(8*4)) # type: int, int, int, int, int, int, int, int
        fi.seek(4, 1)  # skip reserved fields
        # prefix data - platform reference information
        self.pos_update_flg, self.plat_lat, self.plat_lon, self.plat_alt, self.grnd_spd = \
            struct.unpack('5I', fi.read(5*4))  # type: int, int, int, int, int
        self.vel_x, self.vel_y, self.vel_z, self.acc_x, self.acc_y, self.acc_z = \
            struct.unpack('6I', fi.read(6*4))  # type: int, int, int, int, int, int
        self.track, self.true_track, self.pitch, self.roll, self.yaw = \
            struct.unpack('5I', fi.read(5*4))  # type: int, int, int, int, int
        # prefix data - sensor/facility auxiliary data
        self.lat_first, self.lat_center, self.lat_last = \
            struct.unpack('3I', fi.read(3*4))  # type: int, int, int
        self.lon_first, self.lon_center, self.lon_last = \
            struct.unpack('3I', fi.read(3*4))  # type: int, int, int
        # scan sar
        self.burst_num, self.line_num = struct.unpack('2I', fi.read(2*4))  # type: int, int
        fi.seek(60, 1)  # reserved field
        self.frame_num = struct.unpack('I', fi.read(4))[0]  # type: int
        # NB: there are remaining unparsed fields of no interest before data


class _IMGHeader(_CommonHeader2):
    """
    IMG file header parsing and interpretation
    """

    __slots__ = (
        # standard initial header
        'rec_num', 'rec_subtype1', 'rec_type', 'rec_subtype2', 'rec_subtype3',
        'rec_length', 'ascii_ebcdic',
        'doc_id', 'doc_rev', 'rec_rev', 'soft_rel_rev',
        # common to IMG, LED, and TRL
        'file_num', 'file_id',
        'rec_seq_loc_type_flag', 'seq_num_loc', 'fld_len_seq', 'rec_code_loc_type_flag',
        'loc_rec_code', 'fld_len_code', 'rec_len_loc_type_flag', 'loc_rec_len',
        'len_rec_len', 'num_data_rec', 'data_len',

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
        self._signal_elements = None  # type: Union[None, Tuple[_SignalHeader]]

        with open(self._file_name, 'rb') as fi:
            super(_IMGHeader, self).__init__(fi)
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
        _SignalHeader
        """

        index = int_func(index)
        if not (0 <= index < self.num_data_rec):
            raise KeyError('index {} must be in range [0, {})'.format(index, self.num_data_rec))
        # find offset for the given record, and traverse to it
        record_offset = self._signal_start_location + self.prefix_bytes + \
                         self.num_pixels*self.num_bytes*index
        # go to the start of the given record
        fi.seek(record_offset, 0)
        return _SignalHeader(fi)

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

# data class
class _LED_Data(_BaseHeader):
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
        'range_compressed', 'rec_gain_like_pol', 'rec_gain_cross_pol',
        'quant_bit', 'quant_desc', 'dc_bias_i', 'dc_bias_q', 'gain_imbalance',
        'elec_bores', 'mech_bores', 'echo_tracker', 'prf', 'ant_beam_2way_el',
        'ant_beam_2way_az', 'sat_time', 'sta_clock_inc',
        'proc_fac', 'proc_sys', 'proc_ver', 'proc_lvl_code', 'proc_type',
        'proc_alg', 'num_look_az', 'num_look_rng', 'bw_per_look_az',
        'bw_per_look_rng', 'bw_az', 'bw_rng', 'wgt_az', 'wgt_rng',
        'data_input_src', 'res_grnd_rng', 'res_az', 'rad_bias', 'rad_gain',
        'at_dop', 'xt_dop', 'time_dir_pixel', 'time_dir_line',
        'at_dop_rate', 'xt_dop_rate', 'line_constant', 'clutter_lock_flg',
        'autofocus_flg', 'line_spacing', 'rng_comp_des', 'dop_freq',
        'cal_mode_loc_flag', 'start_line_cal_start', 'end_line_cal_start',
        'prf_switch', 'prf_switch_line', 'beam_ctr_dir', 'yaw_steer_flag',
        'param_table_num', 'off_nadir', 'ant_beam_num', 'incidence_ang',
        'num_annot')

    def __init__(self, fi):
        super(_LED_Data, self).__init__(fi)
        # TODO: line 89 - 226

# position class
# attribute class
# radiometric class
# data_quality class
# facility class



class _LEDHeader(_CommonHeader2):
    """
    LED file header parsing and interpretation
    """

    __slots__ = (
        'num_map_rec', 'map_len', 'num_pos_rec', 'pos_len', 'num_att_rec', 'att_len',
        'num_rad_rec', 'rad_len', 'num_rad_comp_rec', 'rad_comp_len',
        'num_data_qual_rec', 'data_qual_len', 'num_hist_rec', 'hist_len',
        'num_rng_spect_rec', 'rng_spect_len', 'num_dem_rec', 'dem_len',
        'num_radar_rec', 'radar_len', 'num_annot_rec', 'annot_len',
        'num_proc_rec', 'proc_len', 'num_cal_rec', 'cal_len',
        'num_gcp_rec', 'gcp_len', 'num_fac_data_rec', 'fac_data_len',
        # some reserved fields for class metadata
        '_file_name', )

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
            super(_LEDHeader, self).__init__(fi)
            self._parse_fields(fi)

    def _parse_fields(self, fi):
        """
        Parse the fields.

        Returns
        -------
        None
        """

        # this has parsed everything up to data_len
        self.num_map_rec = int_func(fi.read(6))
        self.map_len = int_func(fi.read(6))
        self.num_pos_rec = int_func(fi.read(6))
        self.pos_len = int_func(fi.read(6))
        self.num_att_rec = int_func(fi.read(6))
        self.att_len = int_func(fi.read(6))
        self.num_rad_rec = int_func(fi.read(6))
        self.rad_len = int_func(fi.read(6))
        self.num_rad_comp_rec = int_func(fi.read(6))
        self.rad_comp_len = int_func(fi.read(6))
        self.num_data_qual_rec = int_func(fi.read(6))
        self.data_qual_len = int_func(fi.read(6))
        self.num_hist_rec = int_func(fi.read(6))
        self.hist_len = int_func(fi.read(6))
        self.num_rng_spect_rec = int_func(fi.read(6))
        self.rng_spect_len = int_func(fi.read(6))
        self.num_dem_rec = int_func(fi.read(6))
        self.dem_len = int_func(fi.read(6))
        self.num_radar_rec = int_func(fi.read(6))
        self.radar_len = int_func(fi.read(6))
        self.num_annot_rec = int_func(fi.read(6))
        self.annot_len = int_func(fi.read(6))
        self.num_proc_rec = int_func(fi.read(6))
        self.proc_len = int_func(fi.read(6))
        self.num_cal_rec = int_func(fi.read(6))
        self.cal_len = int_func(fi.read(6))
        self.num_gcp_rec = int_func(fi.read(6))
        self.gcp_len = int_func(fi.read(6))
        fi.seek(60, 1)  # skip reserved fields
        # the five data facility records
        num_fac_data_rec = []
        fac_data_len = []
        for i in range(5):
            num_fac_data_rec.append(int_func(fi.read(6)))
            fac_data_len.append(int_func(fi.read(6)))
        self.num_fac_data_rec = tuple(num_fac_data_rec)
        self.fac_data_len = tuple(fac_data_len)
        fi.seek(230, 1)  # skip reserved fields



if __name__ == '__main__':
    root_dir = os.path.expanduser(
        '~/Desktop/sarpy_testing/palsar'
        '/0000000000_001001_ALOS2229210750-180821-L1.1CEOS')

    img_file = os.path.join(root_dir, 'IMG-HH-ALOS2229210750-180821-HBQR1.1__A')
    img_header = _IMGHeader(img_file)
    pref = img_header.prefix_bytes
    suff = img_header.suffix_bytes
    datatype_code = img_header.sar_datatype_code
    print('pref = *{}*, suff = *{}*, datatype_code = *{}*'.format(pref, suff, datatype_code))

    # test _determine_file_type
    # for fil in [
    #     'BRS-HH-ALOS2229210750-180821-HBQR1.1__A.jpg',
    #     'IMG-HH-ALOS2229210750-180821-HBQR1.1__A',
    #     'LED-ALOS2229210750-180821-HBQR1.1__A',
    #     'TRL-ALOS2229210750-180821-HBQR1.1__A',
    #     'VOL-ALOS2229210750-180821-HBQR1.1__A']:
    #     print(_determine_file_type(os.path.join(root_dir, fil)), fil)
