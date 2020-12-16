# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PT(TREElement):
    def __init__(self, value):
        super(PT, self).__init__()
        self.add_field('LON', 's', 15, value)
        self.add_field('LAT', 's', 15, value)


class BNDPLBType(TREElement):
    def __init__(self, value):
        super(BNDPLBType, self).__init__()
        self.add_field('NUMPTS', 'd', 4, value)
        self.add_loop('PTs', self.NUMPTS, PT, value)


class BNDPLB(TREExtension):
    _tag_value = 'BNDPLB'
    _data_type = BNDPLBType
