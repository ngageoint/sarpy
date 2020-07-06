# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class EXOPTAType(TREElement):
    def __init__(self, value):
        super(EXOPTAType, self).__init__()
        self.add_field('ANGLETONORTH', 's', 3, value)
        self.add_field('MEANGSD', 's', 5, value)
        self.add_field('RESERV01', 's', 1, value)
        self.add_field('DYNAMICRANGE', 's', 5, value)
        self.add_field('RESERV02', 's', 7, value)
        self.add_field('OBLANG', 's', 5, value)
        self.add_field('ROLLANG', 's', 6, value)
        self.add_field('PRIMEID', 's', 12, value)
        self.add_field('PRIMEBE', 's', 15, value)
        self.add_field('RESERV03', 's', 5, value)
        self.add_field('NSEC', 's', 3, value)
        self.add_field('RESERV04', 's', 2, value)
        self.add_field('RESERV05', 's', 7, value)
        self.add_field('NSEG', 's', 3, value)
        self.add_field('MAXLPSEG', 's', 6, value)
        self.add_field('RESERV06', 's', 12, value)
        self.add_field('SUNEL', 's', 5, value)
        self.add_field('SUNAZ', 's', 5, value)


class EXOPTA(TREExtension):
    _tag_value = 'EXOPTA'
    _data_type = EXOPTAType
