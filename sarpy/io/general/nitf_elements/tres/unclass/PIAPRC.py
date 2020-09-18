# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ST(TREElement):
    def __init__(self, value):
        super(ST, self).__init__()
        self.add_field('SECTITLE', 's', 40, value)
        self.add_field('PPNUM', 's', 5, value)
        self.add_field('TPP', 's', 3, value)


class RO(TREElement):
    def __init__(self, value):
        super(RO, self).__init__()
        self.add_field('REQORG', 's', 64, value)


class KW(TREElement):
    def __init__(self, value):
        super(KW, self).__init__()
        self.add_field('KEYWORD', 's', 255, value)


class AR(TREElement):
    def __init__(self, value):
        super(AR, self).__init__()
        self.add_field('ASSRPT', 's', 20, value)


class AT(TREElement):
    def __init__(self, value):
        super(AT, self).__init__()
        self.add_field('ATEXT', 's', 255, value)


class PIAPRCType(TREElement):
    def __init__(self, value):
        super(PIAPRCType, self).__init__()
        self.add_field('ACCID', 's', 64, value)
        self.add_field('FMCTL', 's', 32, value)
        self.add_field('SDET', 's', 1, value)
        self.add_field('PCODE', 's', 2, value)
        self.add_field('PSUBE', 's', 6, value)
        self.add_field('PIDNM', 's', 20, value)
        self.add_field('PNAME', 's', 10, value)
        self.add_field('MAKER', 's', 2, value)
        self.add_field('CTIME', 's', 14, value)
        self.add_field('MAPID', 's', 40, value)
        self.add_field('STREP', 'd', 2, value)
        self.add_loop('STs', self.STREP, ST, value)
        self.add_field('ROREP', 'd', 2, value)
        self.add_loop('ROs', self.ROREP, RO, value)
        self.add_field('KWREP', 'd', 2, value)
        self.add_loop('KWs', self.KWREP, KW, value)
        self.add_field('ARREP', 'd', 2, value)
        self.add_loop('ARs', self.ARREP, AR, value)
        self.add_field('ATREP', 'd', 2, value)
        self.add_loop('ATs', self.ATREP, AT, value)


class PIAPRC(TREExtension):
    _tag_value = 'PIAPRC'
    _data_type = PIAPRCType
