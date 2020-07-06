# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PIAIMBType(TREElement):
    def __init__(self, value):
        super(PIAIMBType, self).__init__()
        self.add_field('CLOUD', 's', 3, value)
        self.add_field('STDRD', 's', 1, value)
        self.add_field('SMODE', 's', 12, value)
        self.add_field('SNAME', 's', 18, value)
        self.add_field('SRCE', 's', 255, value)
        self.add_field('CMGEN', 's', 2, value)
        self.add_field('SQUAL', 's', 1, value)
        self.add_field('MISNM', 's', 7, value)
        self.add_field('CSPEC', 's', 32, value)
        self.add_field('PJTID', 's', 2, value)
        self.add_field('GENER', 's', 1, value)
        self.add_field('EXPLS', 's', 1, value)
        self.add_field('OTHRC', 's', 2, value)


class PIAIMB(TREExtension):
    _tag_value = 'PIAIMB'
    _data_type = PIAIMBType
