# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class GEOLOBType(TREElement):
    def __init__(self, value):
        super(GEOLOBType, self).__init__()
        self.add_field('ARV', 'd', 9, value)
        self.add_field('BRV', 'd', 9, value)
        self.add_field('LSO', 'd', 15, value)
        self.add_field('PSO', 'd', 15, value)


class GEOLOB(TREExtension):
    _tag_value = 'GEOLOB'
    _data_type = GEOLOBType
