# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class GRD(TREElement):
    def __init__(self, value):
        super(GRD, self).__init__()
        self.add_field('ZVL', 'd', 10, value)
        self.add_field('BAD', 's', 10, value)
        self.add_field('LOD', 'd', 12, value)
        self.add_field('LAD', 'd', 12, value)
        self.add_field('LSO', 'd', 11, value)
        self.add_field('PSO', 'd', 11, value)


class GRDPSBType(TREElement):
    def __init__(self, value):
        super(GRDPSBType, self).__init__()
        self.add_field('NUM_GRDS', 'd', 2, value)
        self.add_loop('GRDs', self.NUM_GRDS, GRD, value)


class GRDPSB(TREExtension):
    _tag_value = 'GRDPSB'
    _data_type = GRDPSBType
