# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BAND(TREElement):
    def __init__(self, value):
        super(BAND, self).__init__()
        self.add_field('BAND_TYPE', 's', 1, value)
        self.add_field('BAND_ID', 's', 6, value)
        self.add_field('FOC_LENGTH', 's', 11, value)
        self.add_field('NUM_DAP', 's', 8, value)
        self.add_field('NUM_FIR', 's', 8, value)
        self.add_field('DELTA', 's', 7, value)
        self.add_field('OPPOFF_X', 's', 7, value)
        self.add_field('OPPOFF_Y', 's', 7, value)
        self.add_field('OPPOFF_Z', 's', 7, value)
        self.add_field('START_X', 's', 11, value)
        self.add_field('START_Y', 's', 11, value)
        self.add_field('FINISH_X', 's', 11, value)
        self.add_field('FINISH_Y', 's', 11, value)


class CSSFAAType(TREElement):
    def __init__(self, value):
        super(CSSFAAType, self).__init__()
        self.add_field('NUM_BANDS', 'd', 1, value)
        self.add_loop('BANDs', self.NUM_BANDS, BAND, value)


class CSSFAA(TREExtension):
    _tag_value = 'CSSFAA'
    _data_type = CSSFAAType
