# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PRJ(TREElement):
    def __init__(self, value):
        super(PRJ, self).__init__()
        self.add_field('PRJ', 'd', 15, value)


class PRJPSBType(TREElement):
    def __init__(self, value):
        super(PRJPSBType, self).__init__()
        self.add_field('PRN', 's', 80, value)
        self.add_field('PCO', 's', 2, value)
        self.add_field('NUM_PRJ', 'd', 1, value)
        self.add_loop('PRJs', self.NUM_PRJ, PRJ, value)
        self.add_field('XOR', 'd', 15, value)
        self.add_field('YOR', 'd', 15, value)


class PRJPSB(TREExtension):
    _tag_value = 'PRJPSB'
    _data_type = PRJPSBType
