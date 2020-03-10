# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class STEROBType(TREElement):
    def __init__(self, value):
        super(STEROBType, self).__init__()
        self.add_field('ST_ID', 's', 60, value)
        self.add_field('N_MATES', 's', 1, value)
        self.add_field('MATE_INSTANCE', 's', 1, value)
        self.add_field('B_CONV', 's', 5, value)
        self.add_field('E_CONV', 's', 5, value)
        self.add_field('B_ASYM', 's', 5, value)
        self.add_field('E_ASYM', 's', 5, value)
        self.add_field('B_BIE', 's', 6, value)
        self.add_field('E_EIE', 's', 6, value)


class STEROB(TREExtension):
    _tag_value = 'STEROB'
    _data_type = STEROBType
