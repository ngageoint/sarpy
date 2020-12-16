# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class MAPLOBType(TREElement):
    def __init__(self, value):
        super(MAPLOBType, self).__init__()
        self.add_field('UNILOA', 's', 3, value)
        self.add_field('LOD', 's', 5, value)
        self.add_field('LAD', 's', 5, value)
        self.add_field('LSO', 's', 15, value)
        self.add_field('PSO', 's', 15, value)


class MAPLOB(TREExtension):
    _tag_value = 'MAPLOB'
    _data_type = MAPLOBType
