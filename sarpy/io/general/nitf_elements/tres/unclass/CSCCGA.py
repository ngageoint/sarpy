# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CSCCGAType(TREElement):
    def __init__(self, value):
        super(CSCCGAType, self).__init__()
        self.add_field('CCG_SOURCE', 's', 18, value)
        self.add_field('REG_SENSOR', 's', 6, value)
        self.add_field('ORIGIN_LINE', 's', 7, value)
        self.add_field('ORIGIN_SAMPLE', 's', 5, value)
        self.add_field('AS_CELL_SIZE', 's', 7, value)
        self.add_field('CS_CELL_SIZE', 's', 5, value)
        self.add_field('CCG_MAX_LINE', 's', 7, value)
        self.add_field('CCG_MAX_SAMPLE', 's', 5, value)


class CSCCGA(TREExtension):
    _tag_value = 'CSCCGA'
    _data_type = CSCCGAType
