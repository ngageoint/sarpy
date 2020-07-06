# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BCKGDAType(TREElement):
    def __init__(self, value):
        super(BCKGDAType, self).__init__()
        self.add_field('BGWIDTH', 'd', 8, value)
        self.add_field('BGHEIGHT', 'd', 8, value)
        self.add_field('BGRED', 'd', 8, value)
        self.add_field('BGGREEN', 'd', 8, value)
        self.add_field('BGBLUE', 'd', 8, value)
        self.add_field('PIXSIZE', 'd', 8, value)
        self.add_field('PIXUNITS', 'd', 8, value)


class BCKGDA(TREExtension):
    _tag_value = 'BCKGDA'
    _data_type = BCKGDAType
