# -*- coding: utf-8 -*-
"""
Selected unclassified NITF file header TRE objects.
"""

from ..headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ACCHZB(TRE):
    class ACHZs(NITFLoop):
        class ACHZ(NITFElement):
            class PTSs(NITFLoop):
                class PTS(NITFElement):
                    __slots__ = ('LON', 'LAT')
                    _formats = {'LON': '15d', 'LAT': '15d'}

                _child_class = PTS
                _count_size = 3

            __slots__ = ('UNIAAH', 'AAH', 'UNIAPH', 'APH', '_PTSs')
            _formats = {'UNIAAH': '3s', 'AAH': '5d', 'UNIAPH': '3s', 'APH': '5d'}
            _types = {'_PTSs': PTSs}
            _defaults = {'_PTSs': {}}
            _if_skips = {
                'UNIAAH': {'condition': ' == ""', 'vars': ['AAH', ]},
                'UNIAPH': {'condition': ' == ""', 'vars': ['APH', ]}}

        _child_class = ACHZ
        _count_size = 2

    __slots__ = ('TAG', '_ACHZs')
    _formats = {'TAG': '6s'}
    _types = {'_ACHZs': ACHZs}
    _defaults = {'_ACHZs': {}}


class ACCPOB(TRE):
    class ACPOs(NITFLoop):
        class ACPO(NITFElement):
            class PTSs(NITFLoop):
                class PTS(NITFElement):
                    __slots__ = ('LON', 'LAT')
                    _formats = {'LON': '15d', 'LAT': '15d'}

                _child_class = PTS
                _count_size = 3

            __slots__ = (
                'UNIAAH', 'AAH', 'UNIAAV', 'AAV', 'UNIAPH', 'APH', 'UNIAPV', 'APV', '_PTSs')
            _formats = {
                'UNIAAH': '3s', 'AAH': '5d', 'UNIAAV': '3s', 'AAV': '5d', 'UNIAPH': '3s',
                'APH': '5d', 'UNIAPV': '3s', 'APV': '5d'}
            _types = {'_PTSs': PTSs}
            _defaults = {'_PTSs': {}}

        _child_class = ACPO
        _count_size = 2

    __slots__ = ('TAG', '_ACPOs')
    _formats = {'TAG': '6s'}
    _types = {'_ACPOs': ACPOs}
    _defaults = {'_ACPOs': {}}


class ACCVTB(TRE):
    class ACVTs(NITFLoop):
        class ACVT(NITFElement):
            class PTSs(NITFLoop):
                class PTS(NITFElement):
                    __slots__ = ('LON', 'LAT')
                    _formats = {'LON': '15d', 'LAT': '15d'}

                _child_class = PTS
                _count_size = 3

            __slots__ = ('UNIAAV', 'AAV', 'UNIAPV', 'APV', '_PTSs')
            _formats = {'UNIAAV': '3s', 'AAV': '5d', 'UNIAPV': '3s', 'APV': '5d'}
            _types = {'_PTSs': PTSs}
            _defaults = {'_PTSs': {}}
            _if_skips = {
                'UNIAAV': {'condition': ' == ""', 'vars': ['AAV', ]},
                'UNIAPV': {'condition': ' == ""', 'vars': ['APV', ]}}

        _child_class = ACVT
        _count_size = 2

    __slots__ = ('TAG', '_ACVTs')
    _formats = {'TAG': '6s'}
    _types = {'_ACVTs': ACVTs}
    _defaults = {'_ACVTs': {}}


class ACFTB(TRE):
    __slots__ = (
        'TAG', 'AC_MSN_ID', 'AC_TAIL_NO', 'AC_TO', 'SENSOR_ID_TYPE', 'SENSOR_ID',
        'SCENE_SOURCE', 'SCNUM', 'PDATE', 'IMHOSTNO', 'IMREQID', 'MPLAN', 'ENTLOC',
        'LOC_ACCY', 'ENTELV', 'ELV_UNIT', 'EXITLOC', 'EXITELV', 'TMAP', 'ROW_SPACING',
        'ROW_SPACING_UNITS', 'COL_SPACING', 'COL_SPACING_UNITS', 'FOCAL_LENGTH',
        'SENSERIAL', 'ABSWVER', 'CAL_DATE', 'PATCH_TOT', 'MTI_TOT')
    _formats = {
        'TAG': '6s', 'AC_MSN_ID': '20s', 'AC_TAIL_NO': '10s', 'AC_TO': '12s',
        'SENSOR_ID_TYPE': '4s', 'SENSOR_ID': '6s', 'SCENE_SOURCE': '1d',
        'SCNUM': '6d', 'PDATE': '8d', 'IMHOSTNO': '6d', 'IMREQID': '5d',
        'MPLAN': '3d', 'ENTLOC': '25s', 'LOC_ACCY': '6s', 'ENTELV': '6s',
        'ELV_UNIT': '1s', 'EXITLOC': '25s', 'EXITELV': '6s', 'TMAP': '7s',
        'ROW_SPACING': '7s', 'ROW_SPACING_UNITS': '1s', 'COL_SPACING': '7s',
        'COL_SPACING_UNITS': '1s', 'FOCAL_LENGTH': '6s', 'SENSERIAL': '6s',
        'ABSWVER': '7s', 'CAL_DATE': '8s', 'PATCH_TOT': '4d', 'MTI_TOT': '3d'}


class AIMIDB(TRE):
    __slots__ = (
        'TAG', 'ACQUISITION_DATE', 'MISSION_NO', 'MISSION_IDENTIFICATION',
        'FLIGHT_NO', 'OP_NUM', 'CURRENT_SEGMENT', 'REPRO_NUM', 'REPLAY',
        'RESERVED_001', 'START_TILE_COLUMN', 'START_TILE_ROW', 'END_SEGMENT',
        'END_TILE_COLUMN', 'END_TILE_ROW', 'COUNTRY', 'RESERVED002', 'LOCATION',
        'RESERVED003')
    _formats = {
        'TAG': '6s', 'ACQUISITION_DATE': '14s', 'MISSION_NO': '4s',
        'MISSION_IDENTIFICATION': '10s', 'FLIGHT_NO': '2s', 'OP_NUM': '3d',
        'CURRENT_SEGMENT': '2s', 'REPRO_NUM': '2d', 'REPLAY': '3s',
        'RESERVED_001': '1s', 'START_TILE_COLUMN': '3d', 'START_TILE_ROW': '5d',
        'END_SEGMENT': '2s', 'END_TILE_COLUMN': '3d', 'END_TILE_ROW': '5d',
        'COUNTRY': '2s', 'RESERVED002': '4s', 'LOCATION': '11s', 'RESERVED003': '13s'}


class AIPBCA(TRE):
    __slots__ = (
        'TAG', 'Patch_Width', 'u_hat_x', 'u_hat_y', 'u_hat_z', 'v_hat_x', 'v_hat_y', 'v_hat_z', 'n_hat_x', 'n_hat_y',
        'n_hat_z', 'Dep_Angle', 'CT_Track_Range', 'eta_0', 'eta_1', 'x_Img_u', 'x_Img_v', 'x_Img_n', 'y_Img_u',
        'y_Img_v',
        'y_Img_n', 'z_Img_u', 'z_Img_v', 'z_Img_n', 'ct_hat_x', 'ct_hat_y', 'ct_hat_z', 'scl_pt_u', 'scl_pt_v',
        'scl_pt_n',
        'sigma_sm', 'sigma_sn', 's_off', 'Rn_offset', 'CRP_Range', 'Ref_Dep_Ang', 'Ref_Asp_Ang', 'n_Skip_Az',
        'n_Skip_Range')
    _formats = {
        'TAG': '6s', 'Patch_Width': '5d', 'u_hat_x': '16s', 'u_hat_y': '16s', 'u_hat_z': '16s',
        'v_hat_x': '16s', 'v_hat_y': '16s', 'v_hat_z': '16s', 'n_hat_x': '16s', 'n_hat_y': '16s',
        'n_hat_z': '16s', 'Dep_Angle': '7d', 'CT_Track_Range': '10d', 'eta_0': '16s', 'eta_1': '16s',
        'x_Img_u': '9s', 'x_Img_v': '9s', 'x_Img_n': '9s', 'y_Img_u': '9s', 'y_Img_v': '9s', 'y_Img_n': '9s',
        'z_Img_u': '9s', 'z_Img_v': '9s', 'z_Img_n': '9s', 'ct_hat_x': '9s', 'ct_hat_y': '9s', 'ct_hat_z': '9s',
        'scl_pt_u': '13s', 'scl_pt_v': '13s', 'scl_pt_n': '13s', 'sigma_sm': '13s', 'sigma_sn': '13s',
        's_off': '10s', 'Rn_offset': '12s', 'CRP_Range': '11d', 'Ref_Dep_Ang': '7d', 'Ref_Asp_Ang': '9d',
        'n_Skip_Az': '1d', 'n_Skip_Range': '1d'}


class ASTORA(TRE):
    __slots__ = (
        'TAG', 'IMG_TOTAL_ROWS', 'IMG_TOTAL_COLS', 'IMG_INDEX_ROW', 'IMG_INDEX_COL', 'GEOID_OFFSET', 'ALPHA_0', 'K_L',
        'C_M', 'AC_ROLL', 'AC_PITCH', 'AC_YAW', 'AC_TRACK_HEADING', 'AP_ORIGIN_X', 'AP_ORIGIN_Y', 'AP_ORIGIN_Z',
        'AP_DIR_X', 'AP_DIR_Y', 'AP_DIR_Z', 'X_AP_START', 'X_AP_END', 'SS_ROW_SHIFT', 'SS_COL_SHIFT', 'U_hat_x',
        'U_hat_y', 'U_hat_z', 'V_hat_x', 'V_hat_y', 'V_hat_z', 'N_hat_x', 'N_hat_y', 'N_hat_z', 'Eta_0', 'Sigma_sm',
        'Sigma_sn', 'S_off', 'Rn_offset', 'R_scl', 'R_nav', 'R_sc_exact', 'C_sc_x', 'C_sc_y', 'C_sc_z', 'K_hat_x',
        'K_hat_y', 'K_hat_z', 'L_hat_x', 'L_hat_y', 'L_hat_z', 'P_Z', 'Theta_c', 'Alpha_sl', 'Sigma_tc')
    _formats = {
        'TAG': '6s', 'IMG_TOTAL_ROWS': '6d', 'IMG_TOTAL_COLS': '6d', 'IMG_INDEX_ROW': '6d', 'IMG_INDEX_COL': '6d',
        'GEOID_OFFSET': '7s', 'ALPHA_0': '16s', 'K_L': '2d', 'C_M': '15s', 'AC_ROLL': '16s', 'AC_PITCH': '16s',
        'AC_YAW': '16s', 'AC_TRACK_HEADING': '16s', 'AP_ORIGIN_X': '13s', 'AP_ORIGIN_Y': '13s', 'AP_ORIGIN_Z': '13s',
        'AP_DIR_X': '16s', 'AP_DIR_Y': '16s', 'AP_DIR_Z': '16s', 'X_AP_START': '12s', 'X_AP_END': '12s',
        'SS_ROW_SHIFT': '4d', 'SS_COL_SHIFT': '4d', 'U_hat_x': '16s', 'U_hat_y': '16s', 'U_hat_z': '16s',
        'V_hat_x': '16s', 'V_hat_y': '16s', 'V_hat_z': '16s', 'N_hat_x': '16s', 'N_hat_y': '16s', 'N_hat_z': '16s',
        'Eta_0': '16s', 'Sigma_sm': '13s', 'Sigma_sn': '13s', 'S_off': '10s', 'Rn_offset': '12d', 'R_scl': '16d',
        'R_nav': '16d', 'R_sc_exact': '16d', 'C_sc_x': '16s', 'C_sc_y': '16s', 'C_sc_z': '16s', 'K_hat_x': '16s',
        'K_hat_y': '16s', 'K_hat_z': '16s', 'L_hat_x': '16s', 'L_hat_y': '16s', 'L_hat_z': '16s', 'P_Z': '16s',
        'Theta_c': '16s', 'Alpha_sl': '16d', 'Sigma_tc': '16d'}


class BANDSA(TRE):
    class BANDs(NITFLoop):
        class BAND(NITFElement):
            __slots__ = (
                'BANDPEAK', 'BANDLBOUND', 'BANDUBOUND', 'BANDWIDTH', 'BANDCALDRK', 'BANDCALINC', 'BANDRESP', 'BANDASD',
                'BANDGSD')
            _formats = {
                'BANDPEAK': '5s', 'BANDLBOUND': '5s', 'BANDUBOUND': '5s', 'BANDWIDTH': '5s', 'BANDCALDRK': '6s',
                'BANDCALINC': '5s', 'BANDRESP': '5s', 'BANDASD': '5s', 'BANDGSD': '5s'}

        _child_class = BAND
        _count_size = 4

    __slots__ = (
        'TAG', 'ROW_SPACING', 'ROW_SPACING_UNITS', 'COL_SPACING', 'COL_SPACING_UNITS', 'FOCAL_LENGTH', '_BANDs')
    _formats = {
        'TAG': '6s', 'ROW_SPACING': '7s', 'ROW_SPACING_UNITS': '1s', 'COL_SPACING': '7s', 'COL_SPACING_UNITS': '1s',
        'FOCAL_LENGTH': '6s'}
    _types = {'_BANDs': BANDs}
    _defaults = {'_BANDs': {}}


class BCKGDA(TRE):
    __slots__ = (
        'TAG', 'BGWIDTH', 'BGHEIGHT', 'BGRED', 'BGGREEN', 'BGBLUE', 'PIXSIZE', 'PIXUNITS')
    _formats = {
        'TAG': '6s', 'BGWIDTH': '8d', 'BGHEIGHT': '8d', 'BGRED': '8d', 'BGGREEN': '8d', 'BGBLUE': '8d', 'PIXSIZE': '8d',
        'PIXUNITS': '8d'}


class BLOCKA(TRE):
    __slots__ = (
        'TAG', 'BLOCK_INSTANCE', 'N_GRAY', 'L_LINES', 'LAYOVER_ANGLE', 'SHADOW_ANGLE', 'RESERVED-001', 'FRLC_LOC',
        'LRLC_LOC', 'LRFC_LOC', 'FRFC_LOC', 'RESERVED-002')
    _formats = {
        'TAG': '6s', 'BLOCK_INSTANCE': '2d', 'N_GRAY': '5s', 'L_LINES': '5d', 'LAYOVER_ANGLE': '3s',
        'SHADOW_ANGLE': '3s', 'RESERVED-001': '16s', 'FRLC_LOC': '21s', 'LRLC_LOC': '21s', 'LRFC_LOC': '21s',
        'FRFC_LOC': '21s', 'RESERVED-002': '5s'}


class BNDPLB(TRE):
    class PTSs(NITFLoop):
        class PTS(NITFElement):
            __slots__ = ('LON', 'LAT')
            _formats = {'LON': '15d', 'LAT': '15d'}

        _child_class = PTS
        _count_size = 4

    __slots__ = ('TAG', '_PTSs')
    _formats = {'TAG': '6s'}
    _types = {'_PTSs': PTSs}
    _defaults = {'_PTSs': {}}


class CLCTNA(TRE):
    __slots__ = (
        'TAG', 'VERNUM', 'CLCTN_NAME', 'CLCTN_DESCRIPT', 'CLCTN_STDATE', 'CLCTN_SPDATE', 'CLCTN_LOC', 'COUNTRY',
        'SPONSOR', 'PERSONNEL', 'SCLCTN_NAME', 'SDESCRIPTION', 'SCLCTN_Z_OFF', 'SCLCTN_STDATE', 'SCLCTN_SPDATE',
        'SECURITY', 'SCG', 'SITE', 'SITE_NUM', 'SCN_NUM', 'FLIGHT_NUM', 'PASS_NUM', 'SCN_CNTR', 'ALTITUDE',
        'SCN_CONTENT', 'BGRND_TYPE', 'WX_STATION', 'WX_OVERVIEW', 'WX_FILE')
    _formats = {
        'TAG': '6s', 'VERNUM': '4d', 'CLCTN_NAME': '25s', 'CLCTN_DESCRIPT': '255s', 'CLCTN_STDATE': '8s',
        'CLCTN_SPDATE': '8s', 'CLCTN_LOC': '11s', 'COUNTRY': '2s', 'SPONSOR': '20s', 'PERSONNEL': '100s',
        'SCLCTN_NAME': '20s', 'SDESCRIPTION': '255s', 'SCLCTN_Z_OFF': '3s', 'SCLCTN_STDATE': '8s',
        'SCLCTN_SPDATE': '8s', 'SECURITY': '7s', 'SCG': '15s', 'SITE': '15s', 'SITE_NUM': '3s', 'SCN_NUM': '3s',
        'FLIGHT_NUM': '2s', 'PASS_NUM': '2s', 'SCN_CNTR': '11s', 'ALTITUDE': '5s', 'SCN_CONTENT': '50s',
        'BGRND_TYPE': '50s', 'WX_STATION': '20s', 'WX_OVERVIEW': '15s', 'WX_FILE': '30s'}


class CLCTNB(TRE):
    class SITEs(NITFLoop):
        class SITE(NITFElement):
            __slots__ = (
                'SCLCTN_NAME', 'SDESCRIPTION', 'SITE_NUM', 'SCN_NUM', 'SCLCTN_STDATE', 'SCLCTN_SPDATE', 'SCN_CNTR',
                'ALTITUDE', 'SCN_CONTENT', 'BGRND_TYPE', 'SITE_COV')
            _formats = {
                'SCLCTN_NAME': '20s', 'SDESCRIPTION': '255s', 'SITE_NUM': '3s', 'SCN_NUM': '3s', 'SCLCTN_STDATE': '8s',
                'SCLCTN_SPDATE': '8s', 'SCN_CNTR': '11s', 'ALTITUDE': '5s', 'SCN_CONTENT': '50s', 'BGRND_TYPE': '50s',
                'SITE_COV': '1s'}

        _child_class = SITE
        _count_size = 1

    __slots__ = (
        'TAG', 'VERNUM', 'CLCTN_NAME', 'CLCTN_DESCRIPT', 'CLCTN_STDATE', 'CLCTN_SPDATE', 'CLCTN_LOC', 'SITE', 'COUNTRY',
        'SPONSOR', 'PERSONNEL', '_SITEs', 'SCLCTN_Z_OFF', 'SECURITY', 'SCG', 'FLIGHT_NUM', 'PASS_NUM', 'WX_STATION',
        'WX_OVERVIEW', 'WX_FILE')
    _formats = {
        'TAG': '6s', 'VERNUM': '4d', 'CLCTN_NAME': '25s', 'CLCTN_DESCRIPT': '255s', 'CLCTN_STDATE': '8s',
        'CLCTN_SPDATE': '8s', 'CLCTN_LOC': '11s', 'SITE': '15s', 'COUNTRY': '2s', 'SPONSOR': '20s', 'PERSONNEL': '100s',
        'SCLCTN_Z_OFF': '3s', 'SECURITY': '7s', 'SCG': '15s', 'FLIGHT_NUM': '2s', 'PASS_NUM': '2s', 'WX_STATION': '20s',
        'WX_OVERVIEW': '15s', 'WX_FILE': '30s'}
    _types = {'_SITEs': SITEs}
    _defaults = {'_SITEs': {}}


class CSCCGA(TRE):
    __slots__ = (
        'TAG', 'CCG_SOURCE', 'REG_SENSOR', 'ORIGIN_LINE', 'ORIGIN_SAMPLE', 'AS_CELL_SIZE', 'CS_CELL_SIZE',
        'CCG_MAX_LINE', 'CCG_MAX_SAMPLE')
    _formats = {
        'TAG': '6s', 'CCG_SOURCE': '18s', 'REG_SENSOR': '6s', 'ORIGIN_LINE': '7d', 'ORIGIN_SAMPLE': '5d',
        'AS_CELL_SIZE': '7d', 'CS_CELL_SIZE': '5d', 'CCG_MAX_LINE': '7d', 'CCG_MAX_SAMPLE': '5d'}


class CSCRNA(TRE):
    __slots__ = (
        'TAG', 'PREDICT_CORNERS', 'ULCNR_LAT', 'ULCNR_LONG', 'ULCNR_HT', 'URCNR_LAT', 'URCNR_LONG', 'URCNR_HT',
        'LRCNR_LAT', 'LRCNR_LONG', 'LRCNR_HT', 'LLCNR_LAT', 'LLCNR_LONG', 'LLCNR_HT')
    _formats = {
        'TAG': '6s', 'PREDICT_CORNERS': '1s', 'ULCNR_LAT': '9d', 'ULCNR_LONG': '10d', 'ULCNR_HT': '8d',
        'URCNR_LAT': '9d', 'URCNR_LONG': '10d', 'URCNR_HT': '8d', 'LRCNR_LAT': '9d', 'LRCNR_LONG': '10d',
        'LRCNR_HT': '8d', 'LLCNR_LAT': '9d', 'LLCNR_LONG': '10d', 'LLCNR_HT': '8d'}


class CSDIDA(TRE):
    __slots__ = (
        'TAG', 'DAY', 'MONTH', 'YEAR', 'PLATFORM_CODE', 'VEHICLE_ID', 'PASS', 'OPERATION', 'SENSOR_ID', 'PRODUCT_ID',
        'RESERVED_1', 'TIME', 'PROCESS_TIME', 'RESERVED_2', 'RESERVED_3', 'RESERVED_4', 'RESERVED_5',
        'SOFTWARE_VERSION_NUMBER')
    _formats = {
        'TAG': '6s', 'DAY': '2d', 'MONTH': '3s', 'YEAR': '4d', 'PLATFORM_CODE': '2s', 'VEHICLE_ID': '2d', 'PASS': '2d',
        'OPERATION': '3d', 'SENSOR_ID': '2s', 'PRODUCT_ID': '2s', 'RESERVED_1': '4s', 'TIME': '14d',
        'PROCESS_TIME': '14d', 'RESERVED_2': '2d', 'RESERVED_3': '2d', 'RESERVED_4': '1s', 'RESERVED_5': '1s',
        'SOFTWARE_VERSION_NUMBER': '10s'}


class CSEPHA(TRE):
    class EPHEMs(NITFLoop):
        class EPHEM(NITFElement):
            __slots__ = ('EPHEM_X', 'EPHEM_Y', 'EPHEM_Z')
            _formats = {'EPHEM_X': '12d', 'EPHEM_Y': '12d', 'EPHEM_Z': '12d'}

        _child_class = EPHEM
        _count_size = 3

    __slots__ = ('TAG', 'EPHEM_FLAG', 'DT_EPHEM', 'DATE_EPHEM', 'T0_EPHEM', '_EPHEMs')
    _formats = {'TAG': '6s', 'EPHEM_FLAG': '12s', 'DT_EPHEM': '5d', 'DATE_EPHEM': '8d', 'T0_EPHEM': '13d'}
    _types = {'_EPHEMs': EPHEMs}
    _defaults = {'_EPHEMs': {}}


class CSEXRA(TRE):
    __slots__ = (
        'TAG', 'SENSOR', 'TIME_FIRST_LINE_IMAGE', 'TIME_IMAGE_DURATION', 'MAX_GSD', 'ALONG_SCAN_GSD', 'CROSS_SCAN_GSD',
        'GEO_MEAN_GSD', 'A_S_VERT_GSD', 'C_S_VERT_GSD', 'GEO_MEAN_VERT_GSD', 'GSD_BETA_ANGLE', 'DYNAMIC_RANGE',
        'NUM_LINES', 'NUM_SAMPLES', 'ANGLE_TO_NORTH', 'OBLIQUITY_ANGLE', 'AZ_OF_OBLIQUITY', 'GRD_COVER',
        'SNOW_DEPTH_CAT', 'SUN_AZIMUTH', 'SUN_ELEVATION', 'PREDICTED_NIIRS', 'CIRCL_ERR', 'LINEAR_ERR')
    _formats = {
        'TAG': '6s', 'SENSOR': '6s', 'TIME_FIRST_LINE_IMAGE': '12d', 'TIME_IMAGE_DURATION': '12d', 'MAX_GSD': '5d',
        'ALONG_SCAN_GSD': '5s', 'CROSS_SCAN_GSD': '5s', 'GEO_MEAN_GSD': '5s', 'A_S_VERT_GSD': '5s',
        'C_S_VERT_GSD': '5s', 'GEO_MEAN_VERT_GSD': '5s', 'GSD_BETA_ANGLE': '5s', 'DYNAMIC_RANGE': '5d',
        'NUM_LINES': '7d', 'NUM_SAMPLES': '5d', 'ANGLE_TO_NORTH': '7d', 'OBLIQUITY_ANGLE': '6d',
        'AZ_OF_OBLIQUITY': '7d', 'GRD_COVER': '1d', 'SNOW_DEPTH_CAT': '1d', 'SUN_AZIMUTH': '7d', 'SUN_ELEVATION': '7d',
        'PREDICTED_NIIRS': '3s', 'CIRCL_ERR': '3d', 'LINEAR_ERR': '3d'}


class CSPROA(TRE):
    __slots__ = (
        'TAG', 'RESERVED_0', 'RESERVED_1', 'RESERVED_2', 'RESERVED_3', 'RESERVED_4', 'RESERVED_5', 'RESERVED_6',
        'RESERVED_7', 'RESERVED_8', 'BWC')
    _formats = {
        'TAG': '6s', 'RESERVED_0': '12s', 'RESERVED_1': '12s', 'RESERVED_2': '12s', 'RESERVED_3': '12s',
        'RESERVED_4': '12s', 'RESERVED_5': '12s', 'RESERVED_6': '12s', 'RESERVED_7': '12s', 'RESERVED_8': '12s',
        'BWC': '12s'}


class CSSFAA(TRE):
    class BANDs(NITFLoop):
        class BAND(NITFElement):
            __slots__ = (
                'BAND_TYPE', 'BAND_ID', 'FOC_LENGTH', 'NUM_DAP', 'NUM_FIR', 'DELTA', 'OPPOFF_X', 'OPPOFF_Y', 'OPPOFF_Z',
                'START_X', 'START_Y', 'FINISH_X', 'FINISH_Y')
            _formats = {
                'BAND_TYPE': '1s', 'BAND_ID': '6s', 'FOC_LENGTH': '11d', 'NUM_DAP': '8d', 'NUM_FIR': '8d',
                'DELTA': '7d', 'OPPOFF_X': '7d', 'OPPOFF_Y': '7d', 'OPPOFF_Z': '7d', 'START_X': '11d', 'START_Y': '11d',
                'FINISH_X': '11d', 'FINISH_Y': '11d'}

        _child_class = BAND
        _count_size = 1

    __slots__ = ('TAG', '_BANDs')
    _formats = {'TAG': '6s'}
    _types = {'_BANDs': BANDs}
    _defaults = {'_BANDs': {}}


class CSSHPA(TRE):
    __slots__ = (
        'TAG', 'SHAPE_USE', 'SHAPE_CLASS', 'CC_SOURCE', 'SHAPE1_NAME', 'SHAPE1_START', 'SHAPE2_NAME', 'SHAPE2_START',
        'SHAPE3_NAME', 'SHAPE3_START')
    _formats = {
        'TAG': '6s', 'SHAPE_USE': '25s', 'SHAPE_CLASS': '10s', 'CC_SOURCE': '18s', 'SHAPE1_NAME': '3s',
        'SHAPE1_START': '6d', 'SHAPE2_NAME': '3s', 'SHAPE2_START': '6d', 'SHAPE3_NAME': '3s', 'SHAPE3_START': '6d'}
    _if_skips = {'SHAPE_USE': {'condition': '!= "CLOUD_SHAPES"', 'vars': ['CC_SOURCE', ]}}


class EXOPTA(TRE):
    __slots__ = (
        'TAG', 'ANGLETONORTH', 'MEANGSD', 'RESERV01', 'DYNAMICRANGE', 'RESERV02', 'OBLANG', 'ROLLANG', 'PRIMEID',
        'PRIMEBE', 'RESERV03', 'NSEC', 'RESERV04', 'RESERV05', 'NSEG', 'MAXLPSEG', 'RESERV06', 'SUNEL', 'SUNAZ')
    _formats = {
        'TAG': '6s', 'ANGLETONORTH': '3s', 'MEANGSD': '5s', 'RESERV01': '1s', 'DYNAMICRANGE': '5s', 'RESERV02': '7s',
        'OBLANG': '5s', 'ROLLANG': '6s', 'PRIMEID': '12s', 'PRIMEBE': '15s', 'RESERV03': '5s', 'NSEC': '3s',
        'RESERV04': '2s', 'RESERV05': '7s', 'NSEG': '3s', 'MAXLPSEG': '6s', 'RESERV06': '12s', 'SUNEL': '5s',
        'SUNAZ': '5s'}


class EXPLTB(TRE):
    __slots__ = (
        'TAG', 'ANGLE_TO_NORTH', 'ANGLE_TO_NORTH_ACCY', 'SQUINT_ANGLE', 'SQUINT_ANGLE_ACCY', 'MODE', 'RESVD001',
        'GRAZE_ANG', 'GRAZE_ANG_ACCY', 'SLOPE_ANG', 'POLAR', 'NSAMP', 'RESVD002', 'SEQ_NUM', 'PRIME_ID', 'PRIME_BE',
        'RESVD003', 'N_SEC', 'IPR')
    _formats = {
        'TAG': '6s', 'ANGLE_TO_NORTH': '7s', 'ANGLE_TO_NORTH_ACCY': '6s', 'SQUINT_ANGLE': '7s',
        'SQUINT_ANGLE_ACCY': '6s', 'MODE': '3s', 'RESVD001': '16s', 'GRAZE_ANG': '5s', 'GRAZE_ANG_ACCY': '5s',
        'SLOPE_ANG': '5s', 'POLAR': '2s', 'NSAMP': '5s', 'RESVD002': '1s', 'SEQ_NUM': '1s', 'PRIME_ID': '12s',
        'PRIME_BE': '15s', 'RESVD003': '1s', 'N_SEC': '2s', 'IPR': '2s'}


class GEOLOB(TRE):
    __slots__ = ('TAG', 'ARV', 'BRV', 'LSO', 'PSO')
    _formats = {'TAG': '6s', 'ARV': '9d', 'BRV': '9d', 'LSO': '15d', 'PSO': '15d'}


class GEOPSB(TRE):
    __slots__ = (
        'TAG', 'TYP', 'UNI', 'DAG', 'DCD', 'ELL', 'ELC', 'DVR', 'VDCDVR', 'SDA', 'VDCSDA', 'ZOR', 'GRD', 'GRN', 'ZNA')
    _formats = {
        'TAG': '6s', 'TYP': '3s', 'UNI': '3s', 'DAG': '80s', 'DCD': '4s', 'ELL': '80s', 'ELC': '3s', 'DVR': '80s',
        'VDCDVR': '4s', 'SDA': '80s', 'VDCSDA': '4s', 'ZOR': '15d', 'GRD': '3s', 'GRN': '80s', 'ZNA': '4d'}


class GRDPSB(TRE):
    class GRDs(NITFLoop):
        class GRD(NITFElement):
            __slots__ = ('ZVL', 'BAD', 'LOD', 'LAD', 'LSO', 'PSO')
            _formats = {'ZVL': '10d', 'BAD': '10s', 'LOD': '12d', 'LAD': '12d', 'LSO': '11d', 'PSO': '11d'}

        _child_class = GRD
        _count_size = 2

    __slots__ = ('TAG', '_GRDs')
    _formats = {'TAG': '6s'}
    _types = {'_GRDs': GRDs}
    _defaults = {'_GRDs': {}}


class HISTOA(TRE):
    class EVENTs(NITFLoop):
        class EVENT(NITFElement):
            class IPCOMs(NITFLoop):
                class IPCOM_(NITFElement):
                    __slots__ = ('IPCOM', )
                    _formats = {'IPCOM': '80s'}

                _child_class = IPCOM_
                _count_size = 1

            __slots__ = (
                'PDATE', 'PSITE', 'PAS', '_IPCOMs', 'IBPP', 'IPVTYPE', 'INBWC', 'DISP_FLAG', 'ROT_FLAG', 'ROT_ANGLE',
                'ASYM_FLAG', 'ZOOMROW', 'ZOOMCOL', 'PROJ_FLAG', 'SHARP_FLAG', 'SHARPFAM', 'SHARPMEM', 'MAG_FLAG',
                'MAG_LEVEL', 'DRA_FLAG', 'DRA_MULT', 'DRA_SUB', 'TTC_FLAG', 'TTCFAM', 'TTCMEM', 'DEVLUT_FLAG', 'OBPP',
                'OPVTYPE', 'OUTBWC')
            _formats = {
                'PDATE': '14s', 'PSITE': '10s', 'PAS': '10s', 'IBPP': '2d', 'IPVTYPE': '3s', 'INBWC': '10s',
                'DISP_FLAG': '1s', 'ROT_FLAG': '1d', 'ROT_ANGLE': '8s', 'ASYM_FLAG': '1s', 'ZOOMROW': '7s',
                'ZOOMCOL': '7s', 'PROJ_FLAG': '1s', 'SHARP_FLAG': '1d', 'SHARPFAM': '2d', 'SHARPMEM': '2d',
                'MAG_FLAG': '1d', 'MAG_LEVEL': '7s', 'DRA_FLAG': '1d', 'DRA_MULT': '7s', 'DRA_SUB': '5d',
                'TTC_FLAG': '1d', 'TTCFAM': '2d', 'TTCMEM': '2d', 'DEVLUT_FLAG': '1d', 'OBPP': '2d', 'OPVTYPE': '3s',
                'OUTBWC': '10s'}
            _types = {'_IPCOMs': IPCOMs}
            _defaults = {'_IPCOMs': {}}
            _if_skips = {
                'ROT_FLAG': {'condition': '!= 1', 'vars': ['ROT_ANGLE', ]},
                'ASYM_FLAG': {'condition': '!= 1', 'vars': ['ZOOMROW', 'ZOOMCOL']},
                'SHARP_FLAG': {'condition': '!= 1', 'vars': ['SHARPFAM', 'SHARPMEM']},
                'MAG_FLAG': {'condition': '!= 1', 'vars': ['MAG_LEVEL', ]},
                'DRA_FLAG': {'condition': '!= 1', 'vars': ['DRA_MULT', 'DRA_SUB']},
                'TTC_FLAG': {'condition': '!= 1', 'vars': ['TTCFAM', 'TTCMEM']},
            }

        _child_class = EVENT
        _count_size = 2

    __slots__ = ('TAG', 'SYSTYPE', 'PC', 'PE', 'REMAP_FLAG', 'LUTID', '_EVENTs')
    _formats = {'TAG': '6s', 'SYSTYPE': '20s', 'PC': '12s', 'PE': '4s', 'REMAP_FLAG': '1s', 'LUTID': '2d'}
    _types = {'_EVENTs': EVENTs}
    _defaults = {'_EVENTs': {}}


class ICHIPB(TRE):
    __slots__ = (
        'TAG', 'XFRM_FLAG', 'SCALE_FACTOR', 'ANAMRPH_CORR', 'SCANBLK_NUM', 'OP_ROW_11', 'OP_COL_11', 'OP_ROW_12',
        'OP_COL_12', 'OP_ROW_21', 'OP_COL_21', 'OP_ROW_22', 'OP_COL_22', 'FI_ROW_11', 'FI_COL_11', 'FI_ROW_12',
        'FI_COL_12', 'FI_ROW_21', 'FI_COL_21', 'FI_ROW_22', 'FI_COL_22', 'FI_ROW', 'FI_COL')
    _formats = {
        'TAG': '6s', 'XFRM_FLAG': '2s', 'SCALE_FACTOR': '10s', 'ANAMRPH_CORR': '2s', 'SCANBLK_NUM': '2s',
        'OP_ROW_11': '12s', 'OP_COL_11': '12s', 'OP_ROW_12': '12s', 'OP_COL_12': '12s', 'OP_ROW_21': '12s',
        'OP_COL_21': '12s', 'OP_ROW_22': '12s', 'OP_COL_22': '12s', 'FI_ROW_11': '12s', 'FI_COL_11': '12s',
        'FI_ROW_12': '12s', 'FI_COL_12': '12s', 'FI_ROW_21': '12s', 'FI_COL_21': '12s', 'FI_ROW_22': '12s',
        'FI_COL_22': '12s', 'FI_ROW': '8s', 'FI_COL': '8s'}


class J2KLRA(TRE):
    class LAYERs(NITFLoop):
        class LAYER(NITFElement):
            __slots__ = ('LAYER_ID', 'BITRATE')
            _formats = {'LAYER_ID': '3d', 'BITRATE': '9s'}

        _child_class = LAYER
        _count_size = 3

    __slots__ = (
        'TAG', 'ORIG', 'NLEVELS_O', 'NBANDS_O', '_LAYERs', 'NLEVELS_I', 'NBANDS_I', 'NLAYERS_I')
    _formats = {
        'TAG': '6s', 'ORIG': '1d', 'NLEVELS_O': '2d', 'NBANDS_O': '5d', 'NLEVELS_I': '2d', 'NBANDS_I': '5d',
        'NLAYERS_I': '3d'}
    _types = {'_LAYERs': LAYERs}
    _defaults = {'_LAYERSs': {}}
    _if_skips = {'ORIG': {'condition': 'not in [1, 3, 9]', 'vars': ['NLEVELS_I', 'NBANDS_I', 'NLAYERS_I']}}


class MAPLOB(TRE):
    __slots__ = ('TAG', 'UNILOA', 'LOD', 'LAD', 'LSO', 'PSO')
    _formats = {'TAG': '6s', 'UNILOA': '3s', 'LOD': '5d', 'LAD': '5d', 'LSO': '15d', 'PSO': '15d'}


class MENSRB(TRE):
    __slots__ = (
        'TAG', 'ACFT_LOC', 'ACFT_LOC_ACCY', 'ACFT_ALT', 'RP_LOC', 'RP_LOC_ACCY', 'RP_ELV', 'OF_PC_R', 'OF_PC_A',
        'COSGRZ', 'RGCRP', 'RLMAP', 'RP_ROW', 'RP_COL', 'C_R_NC', 'C_R_EC', 'C_R_DC', 'C_AZ_NC', 'C_AZ_EC', 'C_AZ_DC',
        'C_AL_NC', 'C_AL_EC', 'C_AL_DC', 'TOTAL_TILES_COLS', 'TOTAL_TILES_ROWS')
    _formats = {
        'TAG': '6s', 'ACFT_LOC': '25s', 'ACFT_LOC_ACCY': '6s', 'ACFT_ALT': '6s', 'RP_LOC': '25s', 'RP_LOC_ACCY': '6s',
        'RP_ELV': '6s', 'OF_PC_R': '7s', 'OF_PC_A': '7s', 'COSGRZ': '7s', 'RGCRP': '7s', 'RLMAP': '1s', 'RP_ROW': '5s',
        'RP_COL': '5s', 'C_R_NC': '10s', 'C_R_EC': '10s', 'C_R_DC': '10s', 'C_AZ_NC': '9s', 'C_AZ_EC': '9s',
        'C_AZ_DC': '9s', 'C_AL_NC': '9s', 'C_AL_EC': '9s', 'C_AL_DC': '9s', 'TOTAL_TILES_COLS': '3s',
        'TOTAL_TILES_ROWS': '5s'}


class MPDSRA(TRE):
    __slots__ = (
        'TAG', 'BLKNO', 'CDIPR', 'NBLKW', 'NRBLK', 'NCBLK', 'ORPX', 'ORPY', 'ORPZ', 'ORPRO', 'ORPCO', 'FPNVX', 'FPNVY',
        'FPNVZ', 'ARPTM', 'RESV1', 'ARPPN', 'ARPPE', 'ARPPD', 'ARPVN', 'ARPVE', 'ARPVD', 'ARPAN', 'ARPAE', 'ARPAD',
        'RESV2')
    _formats = {
        'TAG': '6s', 'BLKNO': '2s', 'CDIPR': '2s', 'NBLKW': '2d', 'NRBLK': '5d', 'NCBLK': '5d', 'ORPX': '9s',
        'ORPY': '9s', 'ORPZ': '9s', 'ORPRO': '5d', 'ORPCO': '5d', 'FPNVX': '7s', 'FPNVY': '7s', 'FPNVZ': '7s',
        'ARPTM': '9s', 'RESV1': '14s', 'ARPPN': '9s', 'ARPPE': '9s', 'ARPPD': '9s', 'ARPVN': '9s', 'ARPVE': '9s',
        'ARPVD': '9s', 'ARPAN': '8s', 'ARPAE': '8s', 'ARPAD': '8s', 'RESV2': '13s'}


class MSTGTA(TRE):
    __slots__ = (
        'TAG', 'TGTNUM', 'TGTID', 'TGTBE', 'TGTPRI', 'TGTREQ', 'TGTLTIOV', 'TGTTYPE', 'TGTCOLL', 'TGTCAT', 'TGTUTC',
        'TGTELEV', 'TGTELEVUNIT', 'TGTLOC')
    _formats = {
        'TAG': '6s', 'TGTNUM': '5s', 'TGTID': '12s', 'TGTBE': '15s', 'TGTPRI': '3s', 'TGTREQ': '12s', 'TGTLTIOV': '12s',
        'TGTTYPE': '1s', 'TGTCOLL': '1s', 'TGTCAT': '5s', 'TGTUTC': '7s', 'TGTELEV': '6s', 'TGTELEVUNIT': '1s',
        'TGTLOC': '21s'}


class MTIRPA(TRE):
    class VTGTs(NITFLoop):
        class VTGT(NITFElement):
            __slots__ = ('TGLOC', 'TGRDV', 'TGGSP', 'TGHEA', 'TGSIG', 'TGCAT')
            _formats = {'TGLOC': '21s', 'TGRDV': '4s', 'TGGSP': '3s', 'TGHEA': '3s', 'TGSIG': '2s', 'TGCAT': '1s'}

        _child_class = VTGT
        _count_size = 3

    __slots__ = (
        'TAG', 'DESTP', 'MTPID', 'PCHNO', 'WAMFN', 'WAMBN', 'UTC', 'SQNTA', 'COSGZ', '_VTGTs')
    _formats = {
        'TAG': '6s', 'DESTP': '2s', 'MTPID': '3s', 'PCHNO': '4s', 'WAMFN': '5s', 'WAMBN': '1s', 'UTC': '8s',
        'SQNTA': '5s', 'COSGZ': '7s'}
    _types = {'_VTGTs': VTGTs}
    _defaults = {'_VTGTs': {}}


class MTIRPB(TRE):
    class VTGTs(NITFLoop):
        class VTGT(NITFElement):
            __slots__ = (
                'TGLOC', 'TGLCA', 'TGRDV', 'TGGSP', 'TGHEA', 'TGSIG', 'TGCAT')
            _formats = {
                'TGLOC': '23s', 'TGLCA': '6s', 'TGRDV': '4s', 'TGGSP': '3s', 'TGHEA': '3s', 'TGSIG': '2s',
                'TGCAT': '1s'}

        _child_class = VTGT
        _count_size = 3

    __slots__ = (
        'TAG', 'DESTP', 'MTPID', 'PCHNO', 'WAMFN', 'WAMBN', 'UTC', 'ACLOC', 'ACALT', 'ACALU', 'ACHED', 'MTILR', 'SQNTA',
        'COSGZ', '_VTGTs')
    _formats = {
        'TAG': '6s', 'DESTP': '2s', 'MTPID': '3s', 'PCHNO': '4s', 'WAMFN': '5s', 'WAMBN': '1s', 'UTC': '14s',
        'ACLOC': '21s', 'ACALT': '6s', 'ACALU': '1s', 'ACHED': '3s', 'MTILR': '1s', 'SQNTA': '5s', 'COSGZ': '7s'}
    _types = {'_VTGTs': VTGTs}
    _defaults = {'_VTGTs': {}}


class NBLOCA(TRE):
    class FRAMEs(NITFLoop):
        class FRAME(NITFElement):
            __slots__ = ('FRAME_OFFSET',)
            _formats = {'FRAME_OFFSET': '4b'}

        _child_class = FRAME
        _count_size = 4

    __slots__ = ('TAG', 'FRAME_1_OFFSET', '_FRAMEs')
    _formats = {'TAG': '6s', 'FRAME_1_OFFSET': '4b'}
    _types = {'_FRAMEs': FRAMEs}
    _defaults = {'_FRAMEs': {}}


class OFFSET(TRE):
    __slots__ = ('TAG', 'LINE', 'SAMPLE')
    _formats = {'TAG': '6s', 'LINE': '8d', 'SAMPLE': '8d'}


class PATCHB(TRE):
    __slots__ = (
        'TAG', 'PAT_NO', 'LAST_PAT_FLAG', 'LNSTRT', 'LNSTOP', 'AZL', 'NVL', 'FVL', 'NPIXEL', 'FVPIX', 'FRAME', 'UTC',
        'SHEAD', 'GRAVITY', 'INS_V_NC', 'INS_V_EC', 'INS_V_DC', 'OFFLAT', 'OFFLONG', 'TRACK', 'GSWEEP', 'SHEAR',
        'BATCH_NO')
    _formats = {
        'TAG': '6s', 'PAT_NO': '4s', 'LAST_PAT_FLAG': '1s', 'LNSTRT': '7s', 'LNSTOP': '7s', 'AZL': '5s', 'NVL': '5s',
        'FVL': '3s', 'NPIXEL': '5s', 'FVPIX': '5s', 'FRAME': '3s', 'UTC': '8s', 'SHEAD': '7s', 'GRAVITY': '7s',
        'INS_V_NC': '5s', 'INS_V_EC': '5s', 'INS_V_DC': '5s', 'OFFLAT': '8s', 'OFFLONG': '8s', 'TRACK': '3s',
        'GSWEEP': '6s', 'SHEAR': '8s', 'BATCH_NO': '6s'}


class PIAEQA(TRE):
    __slots__ = (
        'TAG', 'EQPCODE', 'EQPNOMEN', 'EQPMAN', 'OBTYPE', 'ORDBAT', 'CTRYPROD', 'CTRYDSN', 'OBJVIEW')
    _formats = {
        'TAG': '6s', 'EQPCODE': '7s', 'EQPNOMEN': '45s', 'EQPMAN': '64s', 'OBTYPE': '1s', 'ORDBAT': '3s',
        'CTRYPROD': '2s', 'CTRYDSN': '2s', 'OBJVIEW': '6s'}


class PIAEVA(TRE):
    __slots__ = ('TAG', 'EVENTNAME', 'EVENTTYPE')
    _formats = {'TAG': '6s', 'EVENTNAME': '38s', 'EVENTTYPE': '8s'}


class PIAIMB(TRE):
    __slots__ = (
        'TAG', 'CLOUD', 'STDRD', 'SMODE', 'SNAME', 'SRCE', 'CMGEN', 'SQUAL', 'MISNM', 'CSPEC', 'PJTID', 'GENER',
        'EXPLS', 'OTHRC')
    _formats = {
        'TAG': '6s', 'CLOUD': '3s', 'STDRD': '1s', 'SMODE': '12s', 'SNAME': '18s', 'SRCE': '255s', 'CMGEN': '2s',
        'SQUAL': '1s', 'MISNM': '7s', 'CSPEC': '32s', 'PJTID': '2s', 'GENER': '1s', 'EXPLS': '1s', 'OTHRC': '2s'}


class PIAIMC(TRE):
    __slots__ = (
        'TAG', 'CLOUDCVR', 'SRP', 'SENSMODE', 'SENSNAME', 'SOURCE', 'COMGEN', 'SUBQUAL', 'PIAMSNNUM', 'CAMSPECS',
        'PROJID', 'GENERATION', 'ESD', 'OTHERCOND', 'MEANGSD', 'IDATUM', 'IELLIP', 'PREPROC', 'IPROJ', 'SATTRACK')
    _formats = {
        'TAG': '6s', 'CLOUDCVR': '3d', 'SRP': '1s', 'SENSMODE': '12s', 'SENSNAME': '18s', 'SOURCE': '255s',
        'COMGEN': '2d', 'SUBQUAL': '1s', 'PIAMSNNUM': '7s', 'CAMSPECS': '32s', 'PROJID': '2s', 'GENERATION': '1d',
        'ESD': '1s', 'OTHERCOND': '2s', 'MEANGSD': '7d', 'IDATUM': '3s', 'IELLIP': '3s', 'PREPROC': '2s', 'IPROJ': '2s',
        'SATTRACK': '8d'}


class PIAPEA(TRE):
    __slots__ = ('TAG', 'LASTNME', 'FIRSTNME', 'MIDNME', 'DOB', 'ASSOCTRY')
    _formats = {'TAG': '6s', 'LASTNME': '28s', 'FIRSTNME': '28s', 'MIDNME': '28s', 'DOB': '6s', 'ASSOCTRY': '2s'}


class PIAPEB(TRE):
    __slots__ = ('TAG', 'LASTNME', 'FIRSTNME', 'MIDNME', 'DOB', 'ASSOCTRY')
    _formats = {'TAG': '6s', 'LASTNME': '28s', 'FIRSTNME': '28s', 'MIDNME': '28s', 'DOB': '8s', 'ASSOCTRY': '2s'}


class PIAPRC(TRE):
    class STs(NITFLoop):
        class ST(NITFElement):
            __slots__ = ('SECTITLE',)
            _formats = {'SECTITLE': '48s'}

        _child_class = ST
        _count_size = 2

    class ROs(NITFLoop):
        class RO(NITFElement):
            __slots__ = ('REQORG',)
            _formats = {'REQORG': '64s'}

        _child_class = RO
        _count_size = 2

    class KWs(NITFLoop):
        class KW(NITFElement):
            __slots__ = ('KEYWORD',)
            _formats = {'KEYWORD': '255s'}

        _child_class = KW
        _count_size = 2

    class ARs(NITFLoop):
        class AR(NITFElement):
            __slots__ = ('ASSRPT',)
            _formats = {'ASSRPT': '20s'}

        _child_class = AR
        _count_size = 2

    class ATs(NITFLoop):
        class AT(NITFElement):
            __slots__ = ('ATEXT',)
            _formats = {'ATEXT': '255s'}

        _child_class = AT
        _count_size = 2

    __slots__ = (
        'TAG', 'ACCID', 'FMCTL', 'SDET', 'PCODE', 'PSUBE', 'PIDNM', 'PNAME', 'MAKER', 'CTIME', 'MAPID', '_STs', '_ROs',
        '_KWs', '_ARs', '_ATs')
    _formats = {
        'TAG': '6s', 'ACCID': '64s', 'FMCTL': '32s', 'SDET': '1s', 'PCODE': '2s', 'PSUBE': '6s', 'PIDNM': '20s',
        'PNAME': '10s', 'MAKER': '2s', 'CTIME': '14s', 'MAPID': '40s'}
    _types = {'_STs': STs, '_ROs': ROs, '_KWs': KWs, '_ARs': ARs, '_ATs': ATs}
    _defaults = {'_STs': {}, '_ROs': {}, '_KWs': {}, '_ARs': {}, '_ATs': {}}


class PIAPRD(TRE):
    class SECTs(NITFLoop):
        class SECT(NITFElement):
            __slots__ = ('SECTITLE', 'PPNUM', 'TPP')
            _formats = {'SECTITLE': '40s', 'PPNUM': '5s', 'TPP': '3d'}

        _child_class = SECT
        _count_size = 2

    class RQORGs(NITFLoop):
        class RQORG(NITFElement):
            __slots__ = ('REQORG',)
            _formats = {'REQORG': '64s'}

        _child_class = RQORG
        _count_size = 2

    class KEYWDs(NITFLoop):
        class KEYWD(NITFElement):
            __slots__ = ('KEYWORD',)
            _formats = {'KEYWORD': '255s'}

        _child_class = KEYWD
        _count_size = 2

    class ASRPTs(NITFLoop):
        class ASRPT(NITFElement):
            __slots__ = ('ASSRPT',)
            _formats = {'ASSRPT': '20s'}

        _child_class = ASRPT
        _count_size = 2

    class ATEXTs(NITFLoop):
        class ATEXT(NITFElement):
            __slots__ = ('ATEXT',)
            _formats = {'ATEXT': '255s'}

        _child_class = ATEXT
        _count_size = 2

    __slots__ = (
        'TAG', 'ACCESSID', 'FMCNTROL', 'SUBDET', 'PRODCODE', 'PRODCRSE', 'PRODIDNO', 'PRODSNME', 'PRODCRCD', 'PRODCRTM',
        'MAPID', '_SECTs', '_RQORGs', '_KEYWDs', '_ASRPTs', '_ATEXTs')
    _formats = {
        'TAG': '6s', 'ACCESSID': '64s', 'FMCNTROL': '32s', 'SUBDET': '1s', 'PRODCODE': '2s', 'PRODCRSE': '6s',
        'PRODIDNO': '20s', 'PRODSNME': '10s', 'PRODCRCD': '2s', 'PRODCRTM': '14s', 'MAPID': '40s'}
    _types = {'_SECTs': SECTs, '_RQORGs': RQORGs, '_KEYWDs': KEYWDs, '_ASRPTs': ASRPTs, '_ATEXTs': ATEXTs}
    _defaults = {'_SECTs': {}, '_RQORGs': {}, '_KEYWDs': {}, '_ASRPTs': {}, '_ATEXTs': {}}


class PIATGA(TRE):
    __slots__ = (
        'TAG', 'TGTUTM', 'PIATGAID', 'PIACTRY', 'PIACAT', 'TGTGEO', 'DATUM', 'TGTNAME', 'PERCOVER')
    _formats = {
        'TAG': '6s', 'TGTUTM': '15s', 'PIATGAID': '15s', 'PIACTRY': '2s', 'PIACAT': '5s', 'TGTGEO': '15s',
        'DATUM': '3s', 'TGTNAME': '38s', 'PERCOVER': '3d'}


class PIATGB(TRE):
    __slots__ = (
        'TAG', 'TGTUTM', 'PIATGAID', 'PIACTRY', 'PIACAT', 'TGTGEO', 'DATUM', 'TGTNAME', 'PERCOVER', 'TGTLAT', 'TGTLON')
    _formats = {
        'TAG': '6s', 'TGTUTM': '15s', 'PIATGAID': '15s', 'PIACTRY': '2s', 'PIACAT': '5s', 'TGTGEO': '15s',
        'DATUM': '3s', 'TGTNAME': '38s', 'PERCOVER': '3d', 'TGTLAT': '10s', 'TGTLON': '11s'}


class PRJPSB(TRE):
    class PRJs(NITFLoop):
        class PRJ(NITFElement):
            __slots__ = ('PRJ',)
            _formats = {'PRJ': '15d'}

        _child_class = PRJ
        _count_size = 1

    __slots__ = ('TAG', 'PRN', 'PCO', '_PRJs', 'XOR', 'YOR')
    _formats = {'TAG': '6s', 'PRN': '80s', 'PCO': '2s', 'XOR': '15d', 'YOR': '15d'}
    _types = {'_PRJs': PRJs}
    _defaults = {'_PRJs': {}}


class REGPTB(TRE):
    class PTSs(NITFLoop):
        class PTS(NITFElement):
            __slots__ = ('PID', 'LON', 'LAT', 'ZVL', 'DIX', 'DIY')
            _formats = {
                'PID': '10s', 'LON': '15d', 'LAT': '15d', 'ZVL': '15d', 'DIX': '11d', 'DIY': '11d'}

        _child_class = PTS
        _count_size = 4

    __slots__ = ('TAG', '_PTSs')
    _formats = {'TAG': '6s'}
    _types = {'_PTSs': PTSs}
    _defaults = {'_PTSs': {}}


class RPFHDR(TRE):
    __slots__ = (
        'TAG', 'ENDIAN', 'HDSECL', 'FILENM', 'NEWFLG', 'STDNUM', 'STDDAT', 'CLASS', 'COUNTR', 'RELEAS', 'LOCSEC')
    _formats = {
        'TAG': '6s', 'ENDIAN': '1b', 'HDSECL': '2b', 'FILENM': '12s', 'NEWFLG': '1b', 'STDNUM': '15s', 'STDDAT': '8s',
        'CLASS': '1s', 'COUNTR': '2s', 'RELEAS': '2s', 'LOCSEC': '4b'}


class RSMGIA(TRE):
    __slots__ = (
        'TAG', 'IID', 'EDITION', 'GR0', 'GRX', 'GRY', 'GRZ', 'GRXX', 'GRXY', 'GRXZ', 'GRYY', 'GRYZ', 'GRZZ', 'GC0',
        'GCX', 'GCY', 'GCZ', 'GCXX', 'GCXY', 'GCXZ', 'GCYY', 'GCYZ', 'GCZZ', 'GRNIS', 'GCNIS', 'GTNIS', 'GRSSIZ',
        'GCSSIZ')
    _formats = {
        'TAG': '6s', 'IID': '80s', 'EDITION': '40s', 'GR0': '21s', 'GRX': '21s', 'GRY': '21s', 'GRZ': '21s',
        'GRXX': '21s', 'GRXY': '21s', 'GRXZ': '21s', 'GRYY': '21s', 'GRYZ': '21s', 'GRZZ': '21s', 'GC0': '21s',
        'GCX': '21s', 'GCY': '21s', 'GCZ': '21s', 'GCXX': '21s', 'GCXY': '21s', 'GCXZ': '21s', 'GCYY': '21s',
        'GCYZ': '21s', 'GCZZ': '21s', 'GRNIS': '3d', 'GCNIS': '3d', 'GTNIS': '3d', 'GRSSIZ': '21s', 'GCSSIZ': '21s'}


class RSMIDA(TRE):
    __slots__ = (
        'TAG', 'IID', 'EDITION', 'ISID', 'SID', 'STID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'NRG',
        'NCG', 'TRG', 'TCG', 'GRNDD', 'XUOR', 'YUOR', 'ZUOR', 'XUXR', 'XUYR', 'XUZR', 'YUXR', 'YUYR', 'YUZR', 'ZUXR',
        'ZUYR', 'ZUZR', 'V1X', 'V1Y', 'V1Z', 'V2X', 'V2Y', 'V2Z', 'V3X', 'V3Y', 'V3Z', 'V4X', 'V4Y', 'V4Z', 'V5X',
        'V5Y', 'V5Z', 'V6X', 'V6Y', 'V6Z', 'V7X', 'V7Y', 'V7Z', 'V8X', 'V8Y', 'V8Z', 'GRPX', 'GRPY', 'GRPZ', 'FULLR',
        'FULLC', 'MINR', 'MAXR', 'MINC', 'MAXC', 'IE0', 'IER', 'IEC', 'IERR', 'IERC', 'IECC', 'IA0', 'IAR', 'IAC',
        'IARR', 'IARC', 'IACC', 'SPX', 'SVX', 'SAX', 'SPY', 'SVY', 'SAY', 'SPZ', 'SVZ', 'SAZ')
    _formats = {
        'TAG': '6s', 'IID': '80s', 'EDITION': '40s', 'ISID': '40s', 'SID': '40s', 'STID': '40s', 'YEAR': '4d',
        'MONTH': '2d', 'DAY': '2d', 'HOUR': '2d', 'MINUTE': '2d', 'SECOND': '9d', 'NRG': '8s', 'NCG': '8s',
        'TRG': '21s', 'TCG': '21s', 'GRNDD': '1s', 'XUOR': '21s', 'YUOR': '21s', 'ZUOR': '21s', 'XUXR': '21s',
        'XUYR': '21s', 'XUZR': '21s', 'YUXR': '21s', 'YUYR': '21s', 'YUZR': '21s', 'ZUXR': '21s', 'ZUYR': '21s',
        'ZUZR': '21s', 'V1X': '21s', 'V1Y': '21s', 'V1Z': '21s', 'V2X': '21s', 'V2Y': '21s', 'V2Z': '21s', 'V3X': '21s',
        'V3Y': '21s', 'V3Z': '21s', 'V4X': '21s', 'V4Y': '21s', 'V4Z': '21s', 'V5X': '21s', 'V5Y': '21s', 'V5Z': '21s',
        'V6X': '21s', 'V6Y': '21s', 'V6Z': '21s', 'V7X': '21s', 'V7Y': '21s', 'V7Z': '21s', 'V8X': '21s', 'V8Y': '21s',
        'V8Z': '21s', 'GRPX': '21s', 'GRPY': '21s', 'GRPZ': '21s', 'FULLR': '8s', 'FULLC': '8s', 'MINR': '8s',
        'MAXR': '8s', 'MINC': '8s', 'MAXC': '8s', 'IE0': '21s', 'IER': '21s', 'IEC': '21s', 'IERR': '21s',
        'IERC': '21s', 'IECC': '21s', 'IA0': '21s', 'IAR': '21s', 'IAC': '21s', 'IARR': '21s', 'IARC': '21s',
        'IACC': '21s', 'SPX': '21s', 'SVX': '21s', 'SAX': '21s', 'SPY': '21s', 'SVY': '21s', 'SAY': '21s', 'SPZ': '21s',
        'SVZ': '21s', 'SAZ': '21s'}


class RSMPCA(TRE):
    class RNTRMs(NITFLoop):
        class RNTRM(NITFElement):
            __slots__ = ('RNPCF',)
            _formats = {'RNPCF': '21s'}

        _child_class = RNTRM
        _count_size = 3

    class RDTRMs(NITFLoop):
        class RDTRM(NITFElement):
            __slots__ = ('RDPCF',)
            _formats = {'RDPCF': '21s'}

        _child_class = RDTRM
        _count_size = 3

    class CNTRMs(NITFLoop):
        class CNTRM(NITFElement):
            __slots__ = ('CNPCF',)
            _formats = {'CNPCF': '21s'}

        _child_class = CNTRM
        _count_size = 3

    class CDTRMs(NITFLoop):
        class CDTRM(NITFElement):
            __slots__ = ('CDPCF',)
            _formats = {'CDPCF': '21s'}

        _child_class = CDTRM
        _count_size = 3

    __slots__ = (
        'TAG', 'IID', 'EDITION', 'RSN', 'CSN', 'RFEP', 'CFEP', 'RNRMO', 'CNRMO', 'XNRMO', 'YNRMO', 'ZNRMO', 'RNRMSF',
        'CNRMSF', 'XNRMSF', 'YNRMSF', 'ZNRMSF', 'RNPWRX', 'RNPWRY', 'RNPWRZ', '_RNTRMs', 'RDPWRX', 'RDPWRY', 'RDPWRZ',
        '_RDTRMs', 'CNPWRX', 'CNPWRY', 'CNPWRZ', '_CNTRMs', 'CDPWRX', 'CDPWRY', 'CDPWRZ', '_CDTRMs')
    _formats = {
        'TAG': '6s', 'IID': '80s', 'EDITION': '40s', 'RSN': '3d', 'CSN': '3d', 'RFEP': '21s', 'CFEP': '21s',
        'RNRMO': '21s', 'CNRMO': '21s', 'XNRMO': '21s', 'YNRMO': '21s', 'ZNRMO': '21s', 'RNRMSF': '21s',
        'CNRMSF': '21s', 'XNRMSF': '21s', 'YNRMSF': '21s', 'ZNRMSF': '21s', 'RNPWRX': '1d', 'RNPWRY': '1d',
        'RNPWRZ': '1d', 'RDPWRX': '1d', 'RDPWRY': '1d', 'RDPWRZ': '1d', 'CNPWRX': '1d', 'CNPWRY': '1d', 'CNPWRZ': '1d',
        'CDPWRX': '1d', 'CDPWRY': '1d', 'CDPWRZ': '1d'}
    _types = {'_RNTRMs': RNTRMs, '_RDTRMs': RDTRMs, '_CNTRMs': CNTRMs, '_CDTRMs': CDTRMs}
    _defaults = {'_RNTRMs': {}, '_RDTRMs': {}, '_CNTRMs': {}, '_CDTRMs': {}}


class RSMPIA(TRE):
    __slots__ = (
        'TAG', 'IID', 'EDITION', 'R0', 'RX', 'RY', 'RZ', 'RXX', 'RXY', 'RXZ', 'RYY', 'RYZ', 'RZZ', 'C0', 'CX', 'CY',
        'CZ', 'CXX', 'CXY', 'CXZ', 'CYY', 'CYZ', 'CZZ', 'RNIS', 'CNIS', 'TNIS', 'RSSIZ', 'CSSIZ')
    _formats = {
        'TAG': '6s', 'IID': '80s', 'EDITION': '40s', 'R0': '21s', 'RX': '21s', 'RY': '21s', 'RZ': '21s', 'RXX': '21s',
        'RXY': '21s', 'RXZ': '21s', 'RYY': '21s', 'RYZ': '21s', 'RZZ': '21s', 'C0': '21s', 'CX': '21s', 'CY': '21s',
        'CZ': '21s', 'CXX': '21s', 'CXY': '21s', 'CXZ': '21s', 'CYY': '21s', 'CYZ': '21s', 'CZZ': '21s', 'RNIS': '3d',
        'CNIS': '3d', 'TNIS': '3d', 'RSSIZ': '21s', 'CSSIZ': '21s'}


class SECTGA(TRE):
    __slots__ = ('TAG', 'SEC_ID', 'SEC_BE', 'RESVD001')
    _formats = {'TAG': '6s', 'SEC_ID': '12s', 'SEC_BE': '15s', 'RESVD001': '1s'}


class SENSRA(TRE):
    __slots__ = (
        'TAG', 'REFROW', 'REFCOL', 'SNSMODEL', 'SNSMOUNT', 'SENSLOC', 'SNALTSRC', 'SENSALT', 'SNALUNIT', 'SENSAGL',
        'SNSPITCH', 'SENSROLL', 'SENSYAW', 'PLTPITCH', 'PLATROLL', 'PLATHDG', 'GRSPDSRC', 'GRDSPEED', 'GRSPUNIT',
        'GRDTRACK', 'VERTVEL', 'VERTVELU', 'SWATHFRM', 'NSWATHS', 'SPOTNUM')
    _formats = {
        'TAG': '6s', 'REFROW': '8d', 'REFCOL': '8d', 'SNSMODEL': '6s', 'SNSMOUNT': '3s', 'SENSLOC': '21s',
        'SNALTSRC': '1s', 'SENSALT': '6s', 'SNALUNIT': '1s', 'SENSAGL': '5s', 'SNSPITCH': '7s', 'SENSROLL': '8s',
        'SENSYAW': '8s', 'PLTPITCH': '7s', 'PLATROLL': '8s', 'PLATHDG': '5s', 'GRSPDSRC': '1s', 'GRDSPEED': '6s',
        'GRSPUNIT': '1s', 'GRDTRACK': '5s', 'VERTVEL': '5s', 'VERTVELU': '1s', 'SWATHFRM': '4s', 'NSWATHS': '4d',
        'SPOTNUM': '3d'}


class SNSPSB(TRE):
    class SNSs(NITFLoop):
        class SNS(NITFElement):
            class BPs(NITFLoop):
                class BP(NITFElement):
                    class PTSs(NITFLoop):
                        class PTS(NITFElement):
                            __slots__ = ('LON', 'LAT')
                            _formats = {'LON': '15d', 'LAT': '15d'}

                        _child_class = PTS
                        _count_size = 2

                    __slots__ = ('_PTSs')
                    _types = {'_PTSs': PTSs}
                    _defaults = {'_PTSs': {}}

                _child_class = BP
                _count_size = 2

            class BNDs(NITFLoop):
                class BND(NITFElement):
                    __slots__ = ('BID', 'WS1', 'WS2')
                    _formats = {'BID': '5s', 'WS1': '5d', 'WS2': '5d'}

                _child_class = BND
                _count_size = 2

            class AUXs(NITFLoop):
                class AUX(NITFElement):
                    __slots__ = ('API', 'APF', 'UNIAPX', 'APN', 'APR', 'APA')
                    _formats = {
                        'API': '20s', 'APF': '1s', 'UNIAPX': '7s', 'APN': '10d', 'APR': '20d', 'APA': '20s'}

                _child_class = AUX
                _count_size = 3

            __slots__ = (
                '_BPs', '_BNDs', 'UNIRES', 'REX', 'REY', 'GSX', 'GSY', 'GSL', 'PLTFM', 'INS', 'MOD', 'PRL', 'ACT',
                'UNINOA', 'NOA', 'UNIANG', 'ANG', 'UNIALT', 'ALT', 'LONSCC', 'LATSCC', 'UNISAE', 'SAZ', 'SEL', 'UNIRPY',
                'ROL', 'PIT', 'YAW', 'UNIPXT', 'PIXT', 'UNISPE', 'ROS', 'PIS', 'YAS', '_AUXs')
            _formats = {
                'UNIRES': '3s', 'REX': '6d', 'REY': '6d', 'GSX': '6d', 'GSY': '6d', 'GSL': '12s', 'PLTFM': '8s',
                'INS': '8s', 'MOD': '4s', 'PRL': '5s', 'ACT': '18s', 'UNINOA': '3s', 'NOA': '7d', 'UNIANG': '3s',
                'ANG': '7d', 'UNIALT': '3s', 'ALT': '9d', 'LONSCC': '10d', 'LATSCC': '10d', 'UNISAE': '3s', 'SAZ': '7d',
                'SEL': '7d', 'UNIRPY': '3s', 'ROL': '7d', 'PIT': '7d', 'YAW': '7d', 'UNIPXT': '3s', 'PIXT': '14d',
                'UNISPE': '7s', 'ROS': '22d', 'PIS': '22d', 'YAS': '22d'}
            _types = {'_BPs': BPs, '_BNDs': BNDs, '_AUXs': AUXs}
            _defaults = {'_BPs': {}, '_BNDs': {}, '_AUXs': {}}

        _child_class = SNS
        _count_size = 2

    __slots__ = ('TAG', '_SNSs')
    _formats = {'TAG': '6s'}
    _types = {'_SNSs': SNSs}
    _defaults = {'_SNSs': {}}


class SOURCB(TRE):
    class SOURs(NITFLoop):
        class SOUR(NITFElement):
            class BPs(NITFLoop):
                class BP(NITFElement):
                    class PTSs(NITFLoop):
                        class PTS(NITFElement):
                            __slots__ = ('LON', 'LAT')
                            _formats = {'LON': '15d', 'LAT': '15d'}

                        _child_class = PTS
                        _count_size = 3

                    __slots__ = ('_PTSs',)
                    _types = {'_PTSs': PTSs}
                    _defaults = {'_PTSs': {}}

                _child_class = BP
                _count_size = 2

            class MIs(NITFLoop):
                class MI(NITFElement):
                    __slots__ = (
                        'CDV30', 'UNIRAT', 'RAT', 'UNIGMA', 'GMA', 'LONGMA', 'LATGMA', 'UNIGCA', 'GCA')
                    _formats = {
                        'CDV30': '8s', 'UNIRAT': '3s', 'RAT': '8d', 'UNIGMA': '3s', 'GMA': '8d', 'LONGMA': '15d',
                        'LATGMA': '15d', 'UNIGCA': '3s', 'GCA': '8d'}

                _child_class = MI
                _count_size = 2

            class LIs(NITFLoop):
                class LI(NITFElement):
                    __slots__ = ('BAD',)
                    _formats = {'BAD': '10s'}

                _child_class = LI
                _count_size = 2

            class PRJs(NITFLoop):
                class PRJ(NITFElement):
                    __slots__ = ('PRJ',)
                    _formats = {'PRJ': '15d'}

                _child_class = PRJ
                _count_size = 1

            class INs(NITFLoop):
                class IN(NITFElement):
                    __slots__ = (
                        'INT', 'INS_SCA', 'NTL', 'TTL', 'NVL', 'TVL', 'NTR', 'TTR', 'NVR', 'TVR', 'NRL', 'TRL', 'NSL',
                        'TSL', 'NRR', 'TRR', 'NSR', 'TSR')
                    _formats = {
                        'INT': '10s', 'INS_SCA': '9d', 'NTL': '15d', 'TTL': '15d', 'NVL': '15d', 'TVL': '15d',
                        'NTR': '15d', 'TTR': '15d', 'NVR': '15d', 'TVR': '15d', 'NRL': '15d', 'TRL': '15d',
                        'NSL': '15d', 'TSL': '15d', 'NRR': '15d', 'TRR': '15d', 'NSR': '15d', 'TSR': '15d'}

                _child_class = IN
                _count_size = 2

            __slots__ = (
                '_BPs', 'PRT', 'URF', 'EDN', 'NAM', 'CDP', 'CDV', 'CDV27', 'SRN', 'SCA', 'UNISQU', 'SQU', 'UNIPCI',
                'PCI', 'WPC', 'NST', 'UNIHKE', 'HKE', 'LONHKE', 'LATHKE', 'QSS', 'QOD', 'CDV10', 'QLE', 'CPY', '_MIs',
                '_LIs', 'DAG', 'DCD', 'ELL', 'ELC', 'DVR', 'VDCDVR', 'SDA', 'VDCSDA', 'PRN', 'PCO', '_PRJs', 'XOR',
                'YOR', 'GRD', 'GRN', 'ZNA', '_INs')
            _formats = {
                'PRT': '10s', 'URF': '20s', 'EDN': '7s', 'NAM': '20s', 'CDP': '3d', 'CDV': '8s', 'CDV27': '8s',
                'SRN': '80s', 'SCA': '9s', 'UNISQU': '3s', 'SQU': '10d', 'UNIPCI': '3s', 'PCI': '4d', 'WPC': '3d',
                'NST': '3d', 'UNIHKE': '3s', 'HKE': '6d', 'LONHKE': '15d', 'LATHKE': '15d', 'QSS': '1s', 'QOD': '1s',
                'CDV10': '8s', 'QLE': '80s', 'CPY': '80s', 'DAG': '80s', 'DCD': '4s', 'ELL': '80s', 'ELC': '3s',
                'DVR': '80s', 'VDCDVR': '4s', 'SDA': '80s', 'VDCSDA': '4s', 'PRN': '80s', 'PCO': '2s', 'XOR': '15d',
                'YOR': '15d', 'GRD': '3s', 'GRN': '80s', 'ZNA': '4d'}
            _types = {'_BPs': BPs, '_MIs': MIs, '_LIs': LIs, '_PRJs': PRJs, '_INs': INs}
            _defaults = {'_BPs': {}, '_MIs': {}, '_LIs': {}, '_PRJs': {}, '_INs': {}}

        _child_class = SOUR
        _count_size = 2

    __slots__ = ('TAG', 'IS_SCA', 'CPATCH', '_SOURs')
    _formats = {'TAG': '6s', 'IS_SCA': '9d', 'CPATCH': '10s'}
    _types = {'_SOURs': SOURs}
    _defaults = {'_SOURs': {}}


class STDIDC(TRE):
    __slots__ = (
        'TAG', 'ACQUISITION_DATE', 'MISSION', 'PASS', 'OP_NUM', 'START_SEGMENT', 'REPRO_NUM', 'REPLAY_REGEN',
        'BLANK_FILL', 'START_COLUMN', 'START_ROW', 'END_SEGMENT', 'END_COLUMN', 'END_ROW', 'COUNTRY', 'WAC', 'LOCATION',
        'RESERV01', 'RESERV02')
    _formats = {
        'TAG': '6s', 'ACQUISITION_DATE': '14s', 'MISSION': '14s', 'PASS': '2s', 'OP_NUM': '3s', 'START_SEGMENT': '2s',
        'REPRO_NUM': '2s', 'REPLAY_REGEN': '3s', 'BLANK_FILL': '1s', 'START_COLUMN': '3s', 'START_ROW': '5s',
        'END_SEGMENT': '2s', 'END_COLUMN': '3s', 'END_ROW': '5s', 'COUNTRY': '2s', 'WAC': '4s', 'LOCATION': '11s',
        'RESERV01': '5s', 'RESERV02': '8s'}


class TRGTA(TRE):
    class SCENE_TGTSs(NITFLoop):
        class SCENE_TGTS(NITFElement):
            class TGT_QCs(NITFLoop):
                class TGT_QC(NITFElement):
                    __slots__ = ('TGT_QCOMMENT',)
                    _formats = {'TGT_QCOMMENT': '40s'}

                _child_class = TGT_QC
                _count_size = 1

            class TGT_CCs(NITFLoop):
                class TGT_CC(NITFElement):
                    __slots__ = ('TGT_CCOMMENT',)
                    _formats = {'TGT_CCOMMENT': '40s'}

                _child_class = TGT_CC
                _count_size = 1

            class REF_PTs(NITFLoop):
                class REF_PT(NITFElement):
                    __slots__ = (
                        'TGT_REF', 'TGT_LL', 'TGT_ELEV', 'TGT_BAND', 'TGT_ROW', 'TGT_COL', 'TGT_PROW', 'TGT_PCOL')
                    _formats = {
                        'TGT_REF': '10s', 'TGT_LL': '21s', 'TGT_ELEV': '8s', 'TGT_BAND': '3s', 'TGT_ROW': '8d',
                        'TGT_COL': '8d', 'TGT_PROW': '8d', 'TGT_PCOL': '8d'}

                _child_class = REF_PT
                _count_size = 1

            __slots__ = (
                'TGT_NAME', 'TGT_TYPE', 'TGT_VER', 'TGT_CAT', 'TGT_BE', 'TGT_SN', 'TGT_POSNUM', 'TGT_ATTITUDE_PITCH',
                'TGT_ATTITUDE_ROLL', 'TGT_ATTITUDE_YAW', 'TGT_DIM_LENGTH', 'TGT_DIM_WIDTH', 'TGT_DIM_HEIGHT',
                'TGT_AZIMUTH', 'TGT_CLTR_RATIO', 'TGT_STATE', 'TGT_COND', 'TGT_OBSCR', 'TGT_OBSCR%', 'TGT_CAMO',
                'TGT_CAMO%', 'TGT_UNDER', 'TGT_OVER', 'TGT_TTEXTURE', 'TGT_PAINT', 'TGT_SPEED', 'TGT_HEADING',
                '_TGT_QCs', '_TGT_CCs', '_REF_PTs')
            _formats = {
                'TGT_NAME': '25s', 'TGT_TYPE': '15s', 'TGT_VER': '6s', 'TGT_CAT': '5s', 'TGT_BE': '17s',
                'TGT_SN': '10s', 'TGT_POSNUM': '2s', 'TGT_ATTITUDE_PITCH': '6s', 'TGT_ATTITUDE_ROLL': '6s',
                'TGT_ATTITUDE_YAW': '6s', 'TGT_DIM_LENGTH': '5s', 'TGT_DIM_WIDTH': '5s', 'TGT_DIM_HEIGHT': '5s',
                'TGT_AZIMUTH': '6s', 'TGT_CLTR_RATIO': '8s', 'TGT_STATE': '10s', 'TGT_COND': '30s', 'TGT_OBSCR': '20s',
                'TGT_OBSCR%': '3s', 'TGT_CAMO': '20s', 'TGT_CAMO%': '3s', 'TGT_UNDER': '12s', 'TGT_OVER': '30s',
                'TGT_TTEXTURE': '45s', 'TGT_PAINT': '40s', 'TGT_SPEED': '3s', 'TGT_HEADING': '3s'}
            _types = {'_TGT_QCs': TGT_QCs, '_TGT_CCs': TGT_CCs, '_REF_PTs': REF_PTs}
            _defaults = {'_TGT_QCs': {}, '_TGT_CCs': {}, '_REF_PTs': {}}

        _child_class = SCENE_TGTS
        _count_size = 3

    class ATTRIBUTESs(NITFLoop):
        class ATTRIBUTES(NITFElement):
            __slots__ = ('ATTR_TGT_NUM', 'ATTR_NAME', 'ATTR_CONDTN', 'ATTR_VALUE')
            _formats = {'ATTR_TGT_NUM': '3d', 'ATTR_NAME': '30s', 'ATTR_CONDTN': '35s', 'ATTR_VALUE': '10s'}

        _child_class = ATTRIBUTES
        _count_size = 3

    __slots__ = ('TAG', 'VERNUM', 'NO_VALID_TGTS', '_SCENE_TGTSs', '_ATTRIBUTESs')
    _formats = {'TAG': '6s', 'VERNUM': '4d', 'NO_VALID_TGTS': '3d'}
    _types = {'_SCENE_TGTSs': SCENE_TGTSs, '_ATTRIBUTESs': ATTRIBUTESs}
    _defaults = {'_SCENE_TGTSs': {}, '_ATTRIBUTESs': {}}


class USE00A(TRE):
    __slots__ = (
        'TAG', 'ANGLE_TO_NORTH', 'MEAN_GSD', 'RSRVD01', 'DYNAMIC_RANGE', 'RSRVD02', 'RSRVD03', 'RSRVD04', 'OBL_ANG',
        'ROLL_ANG', 'RSRVD05', 'RSRVD06', 'RSRVD07', 'RSRVD08', 'RSRVD09', 'RSRVD10', 'RSRVD11', 'N_REF', 'REV_NUM',
        'N_SEG', 'MAX_LP_SEG', 'RSRVD12', 'RSRVD13', 'SUN_EL', 'SUN_AZ')
    _formats = {
        'TAG': '6s', 'ANGLE_TO_NORTH': '3d', 'MEAN_GSD': '5s', 'RSRVD01': '1s', 'DYNAMIC_RANGE': '5d', 'RSRVD02': '3s',
        'RSRVD03': '1s', 'RSRVD04': '3s', 'OBL_ANG': '5s', 'ROLL_ANG': '6s', 'RSRVD05': '12s', 'RSRVD06': '15s',
        'RSRVD07': '4s', 'RSRVD08': '1s', 'RSRVD09': '3s', 'RSRVD10': '1s', 'RSRVD11': '1s', 'N_REF': '2d',
        'REV_NUM': '5d', 'N_SEG': '3d', 'MAX_LP_SEG': '6d', 'RSRVD12': '6s', 'RSRVD13': '6s', 'SUN_EL': '5s',
        'SUN_AZ': '5s'}
