# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PLTFMAType(TREElement):
    def __init__(self, value):
        super(PLTFMAType, self).__init__()
        self.add_field('VERNUM', 's', 4, value)
        self.add_field('P_NAME', 's', 12, value)
        self.add_field('P_DESCR', 's', 40, value)
        self.add_field('P_DATE', 's', 8, value)
        self.add_field('P_TIME', 's', 9, value)
        self.add_field('P_TYPE', 's', 1, value)
        if self.P_TYPE in ['G', 'T', 'M']:
            self.add_field('SNSR_HT', 's', 3, value)
            self.add_field('SNSRLOC', 's', 21, value)
            self.add_field('SNSRHDNG', 's', 3, value)
        elif self.P_TYPE == 'A':
            self.add_field('AC_TYPE', 's', 15, value)
            self.add_field('AC_SERIAL', 's', 12, value)
            self.add_field('AC_T_NUM', 's', 10, value)
            self.add_field('AC_PITCH', 's', 5, value)
            self.add_field('AC_ROLL', 's', 5, value)
            self.add_field('AC_HDNG', 's', 3, value)
            self.add_field('AC_REF_PT', 's', 1, value)
            self.add_field('AC_POS_X', 's', 9, value)
            self.add_field('AC_POS_Y', 's', 9, value)
            self.add_field('AC_POS_Z', 's', 9, value)
            self.add_field('AC_VEL_X', 's', 9, value)
            self.add_field('AC_VEL_Y', 's', 9, value)
            self.add_field('AC_POS_Z', 's', 9, value)
            self.add_field('AC_ACC_X', 's', 8, value)
            self.add_field('AC_ACC_Y', 's', 8, value)
            self.add_field('AC_POS_Z', 's', 8, value)
            self.add_field('AC_SPEED', 's', 5, value)
            self.add_field('ENTLOC', 's', 21, value)
            self.add_field('ENTALT', 's', 6, value)
            self.add_field('EXITLOC', 's', 21, value)
            self.add_field('EXITALTH', 's', 6, value)
            self.add_field('INS_V_NC', 's', 5, value)
            self.add_field('INS_V_EC', 's', 5, value)
            self.add_field('INS_V_DC', 's', 5, value)


class PLTFMA(TREExtension):
    _tag_value = 'PLTFMA'
    _data_type = PLTFMAType
