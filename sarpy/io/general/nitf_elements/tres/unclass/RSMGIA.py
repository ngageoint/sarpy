# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class RSMGIAType(TREElement):
    def __init__(self, value):
        super(RSMGIAType, self).__init__()
        self.add_field('IID', 's', 80, value)
        self.add_field('EDITION', 's', 40, value)
        self.add_field('GR0', 's', 21, value)
        self.add_field('GRX', 's', 21, value)
        self.add_field('GRY', 's', 21, value)
        self.add_field('GRZ', 's', 21, value)
        self.add_field('GRXX', 's', 21, value)
        self.add_field('GRXY', 's', 21, value)
        self.add_field('GRXZ', 's', 21, value)
        self.add_field('GRYY', 's', 21, value)
        self.add_field('GRYZ', 's', 21, value)
        self.add_field('GRZZ', 's', 21, value)
        self.add_field('GC0', 's', 21, value)
        self.add_field('GCX', 's', 21, value)
        self.add_field('GCY', 's', 21, value)
        self.add_field('GCZ', 's', 21, value)
        self.add_field('GCXX', 's', 21, value)
        self.add_field('GCXY', 's', 21, value)
        self.add_field('GCXZ', 's', 21, value)
        self.add_field('GCYY', 's', 21, value)
        self.add_field('GCYZ', 's', 21, value)
        self.add_field('GCZZ', 's', 21, value)
        self.add_field('GRNIS', 's', 3, value)
        self.add_field('GCNIS', 's', 3, value)
        self.add_field('GTNIS', 's', 3, value)
        self.add_field('GRSSIZ', 's', 21, value)
        self.add_field('GCSSIZ', 's', 21, value)


class RSMGIA(TREExtension):
    _tag_value = 'RSMGIA'
    _data_type = RSMGIAType
