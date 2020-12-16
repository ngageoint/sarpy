# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class VER(TREElement):
    def __init__(self, value):
        super(VER, self).__init__()
        self.add_field('VER_NAME', 's', 15, value)
        self.add_field('VERNUM', 's', 10, value)


class IMGDTAType(TREElement):
    def __init__(self, value):
        super(IMGDTAType, self).__init__()
        self.add_field('VERNUM', 's', 4, value)
        self.add_field('FILENAME', 's', 32, value)
        self.add_field('PARENT_FNAME', 's', 32, value)
        self.add_field('CHECKSUM', 's', 32, value)
        self.add_field('ISIZE', 's', 10, value)
        self.add_field('STATUS', 's', 1, value)
        self.add_field('CDATE', 's', 8, value)
        self.add_field('CTIME', 's', 10, value)
        self.add_field('PDATE', 's', 8, value)
        self.add_field('SENTYPE', 's', 1, value)
        self.add_field('DATA_PLANE', 's', 1, value)
        self.add_field('DATA_TYPE', 's', 4, value)
        self.add_field('NUM_ROWS', 's', 6, value)
        self.add_field('NUM_COLS', 's', 6, value)
        self.add_field('SEN_POS', 's', 1, value)
        self.add_field('SEN_CAL_FAC', 's', 15, value)
        self.add_field('IMGQUAL', 's', 50, value)
        self.add_field('NUM_VER', 'd', 2, value)
        self.add_loop('VERs', self.NUM_VER, VER, value)
        if self.SENTYPE == 'R':
            self.add_field('SEN_LOOK', 's', 1, value)
            self.add_field('CR_RES', 's', 7, value)
            self.add_field('RANGE_RES', 's', 7, value)
            self.add_field('CR_PIXELSP', 's', 7, value)
            self.add_field('RANGE_PIXELSP', 's', 7, value)
            self.add_field('CR_WEIGHT', 's', 40, value)
            self.add_field('RANGE_WEIGHT', 's', 40, value)
            self.add_field('R_OVR_SAMP', 's', 6, value)
            self.add_field('CR_OVR_SAMP', 's', 6, value)
            self.add_field('D_DEPRES', 's', 6, value)
            self.add_field('D_GP_SQ', 's', 7, value)
            self.add_field('D_SP_SQ', 's', 7, value)
            self.add_field('D_RANGE', 's', 7, value)
            self.add_field('D_AP_LL', 's', 21, value)
            self.add_field('D_AP_ELV', 's', 7, value)
            self.add_field('M_DEPRES', 's', 6, value)
            self.add_field('M_GP_SQ', 's', 7, value)
            self.add_field('M_SP_SQ', 's', 7, value)
            self.add_field('M_RANGE', 's', 7, value)
            self.add_field('M_AP_LL', 's', 21, value)
            self.add_field('M_AP_ELV', 's', 7, value)
        elif self.SENTYPE in ['E', 'I']:
            self.add_field('GRNDSAMPDIS', 's', 6, value)
            self.add_field('SWATHSIZE', 's', 6, value)
            self.add_field('D_RANGE', 's', 7, value)
            self.add_field('D_AZ_LOOK', 's', 6, value)
            self.add_field('D_EL_LOOK', 's', 5, value)
            self.add_field('M_RANGE', 's', 7, value)
            self.add_field('M_AZ_LOOK', 's', 6, value)
            self.add_field('M_EL_LOOK', 's', 5, value)


class IMGDTA(TREExtension):
    _tag_value = 'IMGDTA'
    _data_type = IMGDTAType
