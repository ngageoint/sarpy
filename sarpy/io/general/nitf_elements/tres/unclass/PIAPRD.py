# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SECTT(TREElement):
    def __init__(self, value):
        super(SECTT, self).__init__()
        self.add_field('SECTITLE', 's', 40, value)
        self.add_field('PPNUM', 's', 5, value)
        self.add_field('TPP', 's', 3, value)


class RQORG(TREElement):
    def __init__(self, value):
        super(RQORG, self).__init__()
        self.add_field('REQORG', 's', 64, value)


class KEYWD(TREElement):
    def __init__(self, value):
        super(KEYWD, self).__init__()
        self.add_field('KEYWORD', 's', 255, value)


class ASRPT(TREElement):
    def __init__(self, value):
        super(ASRPT, self).__init__()
        self.add_field('ASSRPT', 's', 20, value)


class ATEXT(TREElement):
    def __init__(self, value):
        super(ATEXT, self).__init__()
        self.add_field('ATEXT', 's', 255, value)


class PIAPRDType(TREElement):
    def __init__(self, value):
        super(PIAPRDType, self).__init__()
        self.add_field('ACCESSID', 's', 64, value)
        self.add_field('FMCNTROL', 's', 32, value)
        self.add_field('SUBDET', 's', 1, value)
        self.add_field('PRODCODE', 's', 2, value)
        self.add_field('PRODCRSE', 's', 6, value)
        self.add_field('PRODIDNO', 's', 20, value)
        self.add_field('PRODSNME', 's', 10, value)
        self.add_field('PRODCRCD', 's', 2, value)
        self.add_field('PRODCRTM', 's', 14, value)
        self.add_field('MAPID', 's', 40, value)
        self.add_field('SECTTREP', 'd', 2, value)
        self.add_loop('SECTTs', self.SECTTREP, SECTT, value)
        self.add_field('RQORGREP', 'd', 2, value)
        self.add_loop('RQORGs', self.RQORGREP, RQORG, value)
        self.add_field('KEYWDREP', 'd', 2, value)
        self.add_loop('KEYWDs', self.KEYWDREP, KEYWD, value)
        self.add_field('ASRPTREP', 'd', 2, value)
        self.add_loop('ASRPTs', self.ASRPTREP, ASRPT, value)
        self.add_field('ATEXTREP', 'd', 2, value)
        self.add_loop('ATEXTs', self.ATEXTREP, ATEXT, value)


class PIAPRD(TREExtension):
    _tag_value = 'PIAPRD'
    _data_type = PIAPRDType
