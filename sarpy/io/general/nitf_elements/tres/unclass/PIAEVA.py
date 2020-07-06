# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PIAEVAType(TREElement):
    def __init__(self, value):
        super(PIAEVAType, self).__init__()
        self.add_field('EVENTNAME', 's', 38, value)
        self.add_field('EVENTTYPE', 's', 8, value)


class PIAEVA(TREExtension):
    _tag_value = 'PIAEVA'
    _data_type = PIAEVAType
