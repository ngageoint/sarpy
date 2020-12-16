# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PT(TREElement):
    def __init__(self, value):
        super(PT, self).__init__()
        self.add_field('LON', 's', 15, value)
        self.add_field('LAT', 's', 15, value)


class ACVT(TREElement):
    def __init__(self, value):
        super(ACVT, self).__init__()
        self.add_field('UNIAAV', 's', 3, value)
        if self.UNIAAV != '':
            self.add_field('AAV', 's', 5, value)
        self.add_field('UNIAPV', 's', 3, value)
        if self.UNIAPV != '':
            self.add_field('APV', 's', 5, value)
        self.add_field('NUMPTS', 'd', 3, value)
        self.add_loop('PTs', self.NUMPTS, PT, value)


class ACCVTBType(TREElement):
    def __init__(self, value):
        super(ACCVTBType, self).__init__()
        self.add_field('NUMACVT', 'd', 2, value)
        self.add_loop('ACVTs', self.NUMACVT, ACVT, value)


class ACCVTB(TREExtension):
    _tag_value = 'ACCVTB'
    _data_type = ACCVTBType
