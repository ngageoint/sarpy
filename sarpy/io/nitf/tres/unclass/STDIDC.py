# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class STDIDCType(TREElement):
    def __init__(self, value):
        super(STDIDCType, self).__init__()
        self.add_field('ACQUISITION_DATE', 's', 14, value)
        self.add_field('MISSION', 's', 14, value)
        self.add_field('PASS', 's', 2, value)
        self.add_field('OP_NUM', 's', 3, value)
        self.add_field('START_SEGMENT', 's', 2, value)
        self.add_field('REPRO_NUM', 's', 2, value)
        self.add_field('REPLAY_REGEN', 's', 3, value)
        self.add_field('BLANK_FILL', 's', 1, value)
        self.add_field('START_COLUMN', 's', 3, value)
        self.add_field('START_ROW', 's', 5, value)
        self.add_field('END_SEGMENT', 's', 2, value)
        self.add_field('END_COLUMN', 's', 3, value)
        self.add_field('END_ROW', 's', 5, value)
        self.add_field('COUNTRY', 's', 2, value)
        self.add_field('WAC', 's', 4, value)
        self.add_field('LOCATION', 's', 11, value)
        self.add_field('RESERV01', 's', 5, value)
        self.add_field('RESERV02', 's', 8, value)


class STDIDC(TREExtension):
    _tag_value = 'STDIDC'
    _data_type = STDIDCType
