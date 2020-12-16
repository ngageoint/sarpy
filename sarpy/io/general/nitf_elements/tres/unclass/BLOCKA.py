# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BLOCKAType(TREElement):
    def __init__(self, value):
        super(BLOCKAType, self).__init__()
        self.add_field('BLOCK_INSTANCE', 's', 2, value)
        self.add_field('N_GRAY', 's', 5, value)
        self.add_field('L_LINES', 's', 5, value)
        self.add_field('LAYOVER_ANGLE', 's', 3, value)
        self.add_field('SHADOW_ANGLE', 's', 3, value)
        self.add_field('RESERVED_001', 's', 16, value)
        self.add_field('FRLC_LOC', 's', 21, value)
        self.add_field('LRLC_LOC', 's', 21, value)
        self.add_field('LRFC_LOC', 's', 21, value)
        self.add_field('FRFC_LOC', 's', 21, value)
        self.add_field('RESERVED_002', 's', 5, value)


class BLOCKA(TREExtension):
    _tag_value = 'BLOCKA'
    _data_type = BLOCKAType
