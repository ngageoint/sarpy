# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class MENSRBType(TREElement):
    def __init__(self, value):
        super(MENSRBType, self).__init__()
        self.add_field('ACFT_LOC', 's', 25, value)
        self.add_field('ACFT_LOC_ACCY', 's', 6, value)
        self.add_field('ACFT_ALT', 's', 6, value)
        self.add_field('RP_LOC', 's', 25, value)
        self.add_field('RP_LOC_ACCY', 's', 6, value)
        self.add_field('RP_ELV', 's', 6, value)
        self.add_field('OF_PC_R', 's', 7, value)
        self.add_field('OF_PC_A', 's', 7, value)
        self.add_field('COSGRZ', 's', 7, value)
        self.add_field('RGCRP', 's', 7, value)
        self.add_field('RLMAP', 's', 1, value)
        self.add_field('RP_ROW', 's', 5, value)
        self.add_field('RP_COL', 's', 5, value)
        self.add_field('C_R_NC', 's', 10, value)
        self.add_field('C_R_EC', 's', 10, value)
        self.add_field('C_R_DC', 's', 10, value)
        self.add_field('C_AZ_NC', 's', 9, value)
        self.add_field('C_AZ_EC', 's', 9, value)
        self.add_field('C_AZ_DC', 's', 9, value)
        self.add_field('C_AL_NC', 's', 9, value)
        self.add_field('C_AL_EC', 's', 9, value)
        self.add_field('C_AL_DC', 's', 9, value)
        self.add_field('TOTAL_TILES_COLS', 's', 3, value)
        self.add_field('TOTAL_TILES_ROWS', 's', 5, value)


class MENSRB(TREExtension):
    _tag_value = 'MENSRB'
    _data_type = MENSRBType
