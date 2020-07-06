# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CSCRNAType(TREElement):
    def __init__(self, value):
        super(CSCRNAType, self).__init__()
        self.add_field('PREDICT_CORNERS', 's', 1, value)
        self.add_field('ULCNR_LAT', 'd', 9, value)
        self.add_field('ULCNR_LONG', 'd', 10, value)
        self.add_field('ULCNR_HT', 'd', 8, value)
        self.add_field('URCNR_LAT', 'd', 9, value)
        self.add_field('URCNR_LONG', 'd', 10, value)
        self.add_field('URCNR_HT', 'd', 8, value)
        self.add_field('LRCNR_LAT', 'd', 9, value)
        self.add_field('LRCNR_LONG', 'd', 10, value)
        self.add_field('LRCNR_HT', 'd', 8, value)
        self.add_field('LLCNR_LAT', 'd', 9, value)
        self.add_field('LLCNR_LONG', 'd', 10, value)
        self.add_field('LLCNR_HT', 'd', 8, value)


class CSCRNA(TREExtension):
    _tag_value = 'CSCRNA'
    _data_type = CSCRNAType
