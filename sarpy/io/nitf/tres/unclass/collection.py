# -*- coding: utf-8 -*-
"""
Selected simple unclassified NITF file header TRE objects.
"""

from ...headers import TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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
    _defaults = {'TAG': 'ACFTB'}
    _enums = {'TAG': {'ACFTB', }}


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
    _defaults = {'TAG': 'AIMIDB'}
    _enums = {'TAG': {'AIMIDB', }}


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
    _defaults = {'TAG': 'AIPBCA'}
    _enums = {'TAG': {'AIPBCA', }}


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
    _defaults = {'TAG': 'ASTORA'}
    _enums = {'TAG': {'ASTORA', }}


class BCKGDA(TRE):
    __slots__ = (
        'TAG', 'BGWIDTH', 'BGHEIGHT', 'BGRED', 'BGGREEN', 'BGBLUE', 'PIXSIZE', 'PIXUNITS')
    _formats = {
        'TAG': '6s', 'BGWIDTH': '8d', 'BGHEIGHT': '8d', 'BGRED': '8d', 'BGGREEN': '8d', 'BGBLUE': '8d', 'PIXSIZE': '8d',
        'PIXUNITS': '8d'}
    _defaults = {'TAG': 'BCKGDA'}
    _enums = {'TAG': {'BCKGDA', }}


class BLOCKA(TRE):
    __slots__ = (
        'TAG', 'BLOCK_INSTANCE', 'N_GRAY', 'L_LINES', 'LAYOVER_ANGLE', 'SHADOW_ANGLE', 'RESERVED-001', 'FRLC_LOC',
        'LRLC_LOC', 'LRFC_LOC', 'FRFC_LOC', 'RESERVED-002')
    _formats = {
        'TAG': '6s', 'BLOCK_INSTANCE': '2d', 'N_GRAY': '5s', 'L_LINES': '5d', 'LAYOVER_ANGLE': '3s',
        'SHADOW_ANGLE': '3s', 'RESERVED-001': '16s', 'FRLC_LOC': '21s', 'LRLC_LOC': '21s', 'LRFC_LOC': '21s',
        'FRFC_LOC': '21s', 'RESERVED-002': '5s'}
    _defaults = {'TAG': 'BLOCKA'}
    _enums = {'TAG': {'BLOCKA', }}


class CSCCGA(TRE):
    __slots__ = (
        'TAG', 'CCG_SOURCE', 'REG_SENSOR', 'ORIGIN_LINE', 'ORIGIN_SAMPLE', 'AS_CELL_SIZE', 'CS_CELL_SIZE',
        'CCG_MAX_LINE', 'CCG_MAX_SAMPLE')
    _formats = {
        'TAG': '6s', 'CCG_SOURCE': '18s', 'REG_SENSOR': '6s', 'ORIGIN_LINE': '7d', 'ORIGIN_SAMPLE': '5d',
        'AS_CELL_SIZE': '7d', 'CS_CELL_SIZE': '5d', 'CCG_MAX_LINE': '7d', 'CCG_MAX_SAMPLE': '5d'}
    _defaults = {'TAG': 'CSCCGA'}
    _enums = {'TAG': {'CSCCGA', }}


class CSCRNA(TRE):
    __slots__ = (
        'TAG', 'PREDICT_CORNERS', 'ULCNR_LAT', 'ULCNR_LONG', 'ULCNR_HT', 'URCNR_LAT', 'URCNR_LONG', 'URCNR_HT',
        'LRCNR_LAT', 'LRCNR_LONG', 'LRCNR_HT', 'LLCNR_LAT', 'LLCNR_LONG', 'LLCNR_HT')
    _formats = {
        'TAG': '6s', 'PREDICT_CORNERS': '1s', 'ULCNR_LAT': '9d', 'ULCNR_LONG': '10d', 'ULCNR_HT': '8d',
        'URCNR_LAT': '9d', 'URCNR_LONG': '10d', 'URCNR_HT': '8d', 'LRCNR_LAT': '9d', 'LRCNR_LONG': '10d',
        'LRCNR_HT': '8d', 'LLCNR_LAT': '9d', 'LLCNR_LONG': '10d', 'LLCNR_HT': '8d'}
    _defaults = {'TAG': 'CSCRNA'}
    _enums = {'TAG': {'CSCRNA', }}


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
    _defaults = {'TAG': 'CSDIDA'}
    _enums = {'TAG': {'CSDIDA', }}


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
    _defaults = {'TAG': 'CSEXRA'}
    _enums = {'TAG': {'CSEXRA', }}


class CSPROA(TRE):
    __slots__ = (
        'TAG', 'RESERVED_0', 'RESERVED_1', 'RESERVED_2', 'RESERVED_3', 'RESERVED_4', 'RESERVED_5', 'RESERVED_6',
        'RESERVED_7', 'RESERVED_8', 'BWC')
    _formats = {
        'TAG': '6s', 'RESERVED_0': '12s', 'RESERVED_1': '12s', 'RESERVED_2': '12s', 'RESERVED_3': '12s',
        'RESERVED_4': '12s', 'RESERVED_5': '12s', 'RESERVED_6': '12s', 'RESERVED_7': '12s', 'RESERVED_8': '12s',
        'BWC': '12s'}
    _defaults = {'TAG': 'CSPROA'}
    _enums = {'TAG': {'CSPROA', }}


class CSSHPA(TRE):
    __slots__ = (
        'TAG', 'SHAPE_USE', 'SHAPE_CLASS', 'CC_SOURCE', 'SHAPE1_NAME', 'SHAPE1_START', 'SHAPE2_NAME', 'SHAPE2_START',
        'SHAPE3_NAME', 'SHAPE3_START')
    _formats = {
        'TAG': '6s', 'SHAPE_USE': '25s', 'SHAPE_CLASS': '10s', 'CC_SOURCE': '18s', 'SHAPE1_NAME': '3s',
        'SHAPE1_START': '6d', 'SHAPE2_NAME': '3s', 'SHAPE2_START': '6d', 'SHAPE3_NAME': '3s', 'SHAPE3_START': '6d'}
    _if_skips = {'SHAPE_USE': {'condition': '!= "CLOUD_SHAPES"', 'vars': ['CC_SOURCE', ]}}
    _defaults = {'TAG': 'CSSHPA'}
    _enums = {'TAG': {'CSSHPA', }}


class EXOPTA(TRE):
    __slots__ = (
        'TAG', 'ANGLETONORTH', 'MEANGSD', 'RESERV01', 'DYNAMICRANGE', 'RESERV02', 'OBLANG', 'ROLLANG', 'PRIMEID',
        'PRIMEBE', 'RESERV03', 'NSEC', 'RESERV04', 'RESERV05', 'NSEG', 'MAXLPSEG', 'RESERV06', 'SUNEL', 'SUNAZ')
    _formats = {
        'TAG': '6s', 'ANGLETONORTH': '3s', 'MEANGSD': '5s', 'RESERV01': '1s', 'DYNAMICRANGE': '5s', 'RESERV02': '7s',
        'OBLANG': '5s', 'ROLLANG': '6s', 'PRIMEID': '12s', 'PRIMEBE': '15s', 'RESERV03': '5s', 'NSEC': '3s',
        'RESERV04': '2s', 'RESERV05': '7s', 'NSEG': '3s', 'MAXLPSEG': '6s', 'RESERV06': '12s', 'SUNEL': '5s',
        'SUNAZ': '5s'}
    _defaults = {'TAG': 'EXOPTA'}
    _enums = {'TAG': {'EXOPTA', }}


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
    _defaults = {'TAG': 'EXPLTB'}
    _enums = {'TAG': {'EXPLTB', }}


class GEOLOB(TRE):
    __slots__ = ('TAG', 'ARV', 'BRV', 'LSO', 'PSO')
    _formats = {'TAG': '6s', 'ARV': '9d', 'BRV': '9d', 'LSO': '15d', 'PSO': '15d'}
    _defaults = {'TAG': 'GEOLOB'}
    _enums = {'TAG': {'GEOLOB', }}


class GEOPSB(TRE):
    __slots__ = (
        'TAG', 'TYP', 'UNI', 'DAG', 'DCD', 'ELL', 'ELC', 'DVR', 'VDCDVR', 'SDA', 'VDCSDA', 'ZOR', 'GRD', 'GRN', 'ZNA')
    _formats = {
        'TAG': '6s', 'TYP': '3s', 'UNI': '3s', 'DAG': '80s', 'DCD': '4s', 'ELL': '80s', 'ELC': '3s', 'DVR': '80s',
        'VDCDVR': '4s', 'SDA': '80s', 'VDCSDA': '4s', 'ZOR': '15d', 'GRD': '3s', 'GRN': '80s', 'ZNA': '4d'}
    _defaults = {'TAG': 'GEOPSB'}
    _enums = {'TAG': {'GEOPSB', }}


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
    _defaults = {'TAG': 'ICHIPB'}
    _enums = {'TAG': {'ICHIPB', }}


class MAPLOB(TRE):
    __slots__ = ('TAG', 'UNILOA', 'LOD', 'LAD', 'LSO', 'PSO')
    _formats = {'TAG': '6s', 'UNILOA': '3s', 'LOD': '5d', 'LAD': '5d', 'LSO': '15d', 'PSO': '15d'}
    _defaults = {'TAG': 'MAPLOB'}
    _enums = {'TAG': {'MAPLOB', }}


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
    _defaults = {'TAG': 'MENSRB'}
    _enums = {'TAG': {'MENSRB', }}


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
    _defaults = {'TAG': 'MPDSRA'}
    _enums = {'TAG': {'MPDSRA', }}


class MSTGTA(TRE):
    __slots__ = (
        'TAG', 'TGTNUM', 'TGTID', 'TGTBE', 'TGTPRI', 'TGTREQ', 'TGTLTIOV', 'TGTTYPE', 'TGTCOLL', 'TGTCAT', 'TGTUTC',
        'TGTELEV', 'TGTELEVUNIT', 'TGTLOC')
    _formats = {
        'TAG': '6s', 'TGTNUM': '5s', 'TGTID': '12s', 'TGTBE': '15s', 'TGTPRI': '3s', 'TGTREQ': '12s', 'TGTLTIOV': '12s',
        'TGTTYPE': '1s', 'TGTCOLL': '1s', 'TGTCAT': '5s', 'TGTUTC': '7s', 'TGTELEV': '6s', 'TGTELEVUNIT': '1s',
        'TGTLOC': '21s'}
    _defaults = {'TAG': 'MSTGTA'}
    _enums = {'TAG': {'MSTGTA', }}


class OFFSET(TRE):
    __slots__ = ('TAG', 'LINE', 'SAMPLE')
    _formats = {'TAG': '6s', 'LINE': '8d', 'SAMPLE': '8d'}
    _defaults = {'TAG': 'OFFSET'}
    _enums = {'TAG': {'OFFSET', }}


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
    _defaults = {'TAG': 'PATCHB'}
    _enums = {'TAG': {'PATCHB', }}


class RPFHDR(TRE):
    __slots__ = (
        'TAG', 'ENDIAN', 'HDSECL', 'FILENM', 'NEWFLG', 'STDNUM', 'STDDAT', 'CLASS', 'COUNTR', 'RELEAS', 'LOCSEC')
    _formats = {
        'TAG': '6s', 'ENDIAN': '1b', 'HDSECL': '2b', 'FILENM': '12s', 'NEWFLG': '1b', 'STDNUM': '15s', 'STDDAT': '8s',
        'CLASS': '1s', 'COUNTR': '2s', 'RELEAS': '2s', 'LOCSEC': '4b'}
    _defaults = {'TAG': 'RPFHDR'}
    _enums = {'TAG': {'RPFHDR', }}


class SECTGA(TRE):
    __slots__ = ('TAG', 'SEC_ID', 'SEC_BE', 'RESVD001')
    _formats = {'TAG': '6s', 'SEC_ID': '12s', 'SEC_BE': '15s', 'RESVD001': '1s'}
    _defaults = {'TAG': 'SECTGA'}
    _enums = {'TAG': {'SECTGA', }}


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
    _defaults = {'TAG': 'SENSRA'}
    _enums = {'TAG': {'SENSRA', }}


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
    _defaults = {'TAG': 'STDIDC'}
    _enums = {'TAG': {'STDIDC', }}


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
    _defaults = {'TAG': 'USE00A'}
    _enums = {'TAG': {'USE00A', }}
