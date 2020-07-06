# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BAND(TREElement):
    def __init__(self, value):
        super(BAND, self).__init__()
        self.add_field('BAND_TYPE', 's', 1, value)
        self.add_field('BAND_ID', 's', 6, value)
        self.add_field('FOC_LENGTH', 'd', 11, value)
        self.add_field('NUM_DAP', 'd', 8, value)
        self.add_field('NUM_FIR', 'd', 8, value)
        self.add_field('DELTA', 'd', 7, value)
        self.add_field('OPPOFF_X', 'd', 7, value)
        self.add_field('OPPOFF_Y', 'd', 7, value)
        self.add_field('OPPOFF_Z', 'd', 7, value)
        self.add_field('START_X', 'd', 11, value)
        self.add_field('START_Y', 'd', 11, value)
        self.add_field('FINISH_X', 'd', 11, value)
        self.add_field('FINISH_Y', 'd', 11, value)


class CSSFAAType(TREElement):
    def __init__(self, value):
        super(CSSFAAType, self).__init__()
        self.add_field('NUM_BANDS', 'd', 1, value)
        self.add_loop('BANDs', self.NUM_BANDS, BAND, value)


class CSSFAA(TREExtension):
    _tag_value = 'CSSFAA'
    _data_type = CSSFAAType
