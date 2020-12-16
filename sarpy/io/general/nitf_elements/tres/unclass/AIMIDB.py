# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class AIMIDBType(TREElement):
    def __init__(self, value):
        super(AIMIDBType, self).__init__()
        self.add_field('ACQUISITION_DATE', 's', 14, value)
        self.add_field('MISSION_NO', 's', 4, value)
        self.add_field('MISSION_IDENTIFICATION', 's', 10, value)
        self.add_field('FLIGHT_NO', 's', 2, value)
        self.add_field('OP_NUM', 's', 3, value)
        self.add_field('CURRENT_SEGMENT', 's', 2, value)
        self.add_field('REPRO_NUM', 's', 2, value)
        self.add_field('REPLAY', 's', 3, value)
        self.add_field('RESERVED_001', 's', 1, value)
        self.add_field('START_TILE_COLUMN', 's', 3, value)
        self.add_field('START_TILE_ROW', 's', 5, value)
        self.add_field('END_SEGMENT', 's', 2, value)
        self.add_field('END_TILE_COLUMN', 's', 3, value)
        self.add_field('END_TILE_ROW', 's', 5, value)
        self.add_field('COUNTRY', 's', 2, value)
        self.add_field('RESERVED002', 's', 4, value)
        self.add_field('LOCATION', 's', 11, value)
        self.add_field('RESERVED003', 's', 13, value)


class AIMIDB(TREExtension):
    _tag_value = 'AIMIDB'
    _data_type = AIMIDBType
