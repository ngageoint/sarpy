# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ASTORAType(TREElement):
    def __init__(self, value):
        super(ASTORAType, self).__init__()
        self.add_field('IMG_TOTAL_ROWS', 's', 6, value)
        self.add_field('IMG_TOTAL_COLS', 's', 6, value)
        self.add_field('IMG_INDEX_ROW', 's', 6, value)
        self.add_field('IMG_INDEX_COL', 's', 6, value)
        self.add_field('GEOID_OFFSET', 's', 7, value)
        self.add_field('ALPHA_0', 's', 16, value)
        self.add_field('K_L', 's', 2, value)
        self.add_field('C_M', 's', 15, value)
        self.add_field('AC_ROLL', 's', 16, value)
        self.add_field('AC_PITCH', 's', 16, value)
        self.add_field('AC_YAW', 's', 16, value)
        self.add_field('AC_TRACK_HEADING', 's', 16, value)
        self.add_field('AP_ORIGIN_X', 's', 13, value)
        self.add_field('AP_ORIGIN_Y', 's', 13, value)
        self.add_field('AP_ORIGIN_Z', 's', 13, value)
        self.add_field('AP_DIR_X', 's', 16, value)
        self.add_field('AP_DIR_Y', 's', 16, value)
        self.add_field('AP_DIR_Z', 's', 16, value)
        self.add_field('X_AP_START', 's', 12, value)
        self.add_field('X_AP_END', 's', 12, value)
        self.add_field('SS_ROW_SHIFT', 's', 4, value)
        self.add_field('SS_COL_SHIFT', 's', 4, value)
        self.add_field('U_HAT_X', 's', 16, value)
        self.add_field('U_HAT_Y', 's', 16, value)
        self.add_field('U_HAT_Z', 's', 16, value)
        self.add_field('V_HAT_X', 's', 16, value)
        self.add_field('V_HAT_Y', 's', 16, value)
        self.add_field('V_HAT_Z', 's', 16, value)
        self.add_field('N_HAT_X', 's', 16, value)
        self.add_field('N_HAT_Y', 's', 16, value)
        self.add_field('N_HAT_Z', 's', 16, value)
        self.add_field('ETA_0', 's', 16, value)
        self.add_field('SIGMA_SM', 's', 13, value)
        self.add_field('SIGMA_SN', 's', 13, value)
        self.add_field('S_OFF', 's', 10, value)
        self.add_field('RN_OFFSET', 's', 12, value)
        self.add_field('R_SCL', 's', 16, value)
        self.add_field('R_NAV', 's', 16, value)
        self.add_field('R_SC_EXACT', 's', 16, value)
        self.add_field('C_SC_X', 's', 16, value)
        self.add_field('C_SC_Y', 's', 16, value)
        self.add_field('C_SC_Z', 's', 16, value)
        self.add_field('K_HAT_X', 's', 16, value)
        self.add_field('K_HAT_Y', 's', 16, value)
        self.add_field('K_HAT_Z', 's', 16, value)
        self.add_field('L_HAT_X', 's', 16, value)
        self.add_field('L_HAT_Y', 's', 16, value)
        self.add_field('L_HAT_Z', 's', 16, value)
        self.add_field('P_Z', 's', 16, value)
        self.add_field('THETA_C', 's', 16, value)
        self.add_field('ALPHA_SL', 's', 16, value)
        self.add_field('SIGMA_TC', 's', 16, value)


class ASTORA(TREExtension):
    _tag_value = 'ASTORA'
    _data_type = ASTORAType
