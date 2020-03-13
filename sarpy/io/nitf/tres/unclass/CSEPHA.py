# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class EPHEM(TREElement):
    def __init__(self, value):
        super(EPHEM, self).__init__()
        self.add_field('EPHEM_X', 'd', 12, value)
        self.add_field('EPHEM_Y', 'd', 12, value)
        self.add_field('EPHEM_Z', 'd', 12, value)


class CSEPHAType(TREElement):
    def __init__(self, value):
        super(CSEPHAType, self).__init__()
        self.add_field('EPHEM_FLAG', 's', 12, value)
        self.add_field('DT_EPHEM', 'd', 5, value)
        self.add_field('DATE_EPHEM', 'd', 8, value)
        self.add_field('T0_EPHEM', 'd', 13, value)
        self.add_field('NUM_EPHEM', 'd', 3, value)
        self.add_loop('EPHEMs', self.NUM_EPHEM, EPHEM, value)


class CSEPHA(TREExtension):
    _tag_value = 'CSEPHA'
    _data_type = CSEPHAType
