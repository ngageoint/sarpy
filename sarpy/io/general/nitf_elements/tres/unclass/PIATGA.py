# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PIATGAType(TREElement):
    def __init__(self, value):
        super(PIATGAType, self).__init__()
        self.add_field('TGTUTM', 's', 15, value)
        self.add_field('PIATGAID', 's', 15, value)
        self.add_field('PIACTRY', 's', 2, value)
        self.add_field('PIACAT', 's', 5, value)
        self.add_field('TGTGEO', 's', 15, value)
        self.add_field('DATUM', 's', 3, value)
        self.add_field('TGTNAME', 's', 38, value)
        self.add_field('PERCOVER', 's', 3, value)


class PIATGA(TREExtension):
    _tag_value = 'PIATGA'
    _data_type = PIATGAType
