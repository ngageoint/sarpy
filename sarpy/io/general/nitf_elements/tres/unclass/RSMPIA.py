# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class RSMPIAType(TREElement):
    def __init__(self, value):
        super(RSMPIAType, self).__init__()
        self.add_field('IID', 's', 80, value)
        self.add_field('EDITION', 's', 40, value)
        self.add_field('R0', 's', 21, value)
        self.add_field('RX', 's', 21, value)
        self.add_field('RY', 's', 21, value)
        self.add_field('RZ', 's', 21, value)
        self.add_field('RXX', 's', 21, value)
        self.add_field('RXY', 's', 21, value)
        self.add_field('RXZ', 's', 21, value)
        self.add_field('RYY', 's', 21, value)
        self.add_field('RYZ', 's', 21, value)
        self.add_field('RZZ', 's', 21, value)
        self.add_field('C0', 's', 21, value)
        self.add_field('CX', 's', 21, value)
        self.add_field('CY', 's', 21, value)
        self.add_field('CZ', 's', 21, value)
        self.add_field('CXX', 's', 21, value)
        self.add_field('CXY', 's', 21, value)
        self.add_field('CXZ', 's', 21, value)
        self.add_field('CYY', 's', 21, value)
        self.add_field('CYZ', 's', 21, value)
        self.add_field('CZZ', 's', 21, value)
        self.add_field('RNIS', 's', 3, value)
        self.add_field('CNIS', 's', 3, value)
        self.add_field('TNIS', 's', 3, value)
        self.add_field('RSSIZ', 's', 21, value)
        self.add_field('CSSIZ', 's', 21, value)


class RSMPIA(TREExtension):
    _tag_value = 'RSMPIA'
    _data_type = RSMPIAType
