# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class MAP(TREElement):
    def __init__(self, value):
        super(MAP, self).__init__()
        self.add_field('MAP', 's', 21, value)


class CORSEG(TREElement):
    def __init__(self, value):
        super(CORSEG, self).__init__()
        self.add_field('CORSEG', 's', 21, value)
        self.add_field('TAUSEG', 's', 21, value)


class ERRCVG(TREElement):
    def __init__(self, value):
        super(ERRCVG, self).__init__()
        self.add_field('ERRCVG', 's', 21, value)


class ERRCOVDATA(TREElement):
    def __init__(self, value):
        super(ERRCOVDATA, self).__init__()
        self.add_field('NUMOPG', 'd', 2, value)
        self.add_loop(
            'ERRCVGs', int((self.NUMOPG+1)*self.NUMOPG/2), ERRCVG, value)
        self.add_field('TCDF', 'd', 1, value)
        self.add_field('NCSEG', 'd', 1, value)
        self.add_loop('CORSEGs', self.NCSEG, CORSEG, value)


class UCORSEGR(TREElement):
    def __init__(self, value):
        super(UCORSEGR, self).__init__()
        self.add_field('UCORSR', 's', 21, value)
        self.add_field('UTAUSR', 's', 21, value)


class UCORSEGC(TREElement):
    def __init__(self, value):
        super(UCORSEGC, self).__init__()
        self.add_field('UCORSC', 's', 21, value)
        self.add_field('UTAUSC', 's', 21, value)


class RSMECAType(TREElement):
    def __init__(self, value):
        super(RSMECAType, self).__init__()
        self.add_field('IID', 's', 80, value)
        self.add_field('EDITION', 's', 40, value)
        self.add_field('TID', 's', 40, value)
        self.add_field('INCLIC', 's', 1, value)
        self.add_field('INCLUC', 's', 1, value)

        if self.INCLIC == 'Y':
            self.add_field('NPAR', 'd', 2, value)
            self.add_field('NPARO', 'd', 2, value)
            self.add_field('IGN', 'd', 2, value)
            self.add_field('CVDATE', 's', 8, value)
            self.add_field('XUOL', 's', 21, value)
            self.add_field('YUOL', 's', 21, value)
            self.add_field('ZUOL', 's', 21, value)
            self.add_field('XUXL', 's', 21, value)
            self.add_field('XUYL', 's', 21, value)
            self.add_field('XUZL', 's', 21, value)
            self.add_field('YUXL', 's', 21, value)
            self.add_field('YUYL', 's', 21, value)
            self.add_field('YUZL', 's', 21, value)
            self.add_field('ZUXL', 's', 21, value)
            self.add_field('ZUYL', 's', 21, value)
            self.add_field('ZUZL', 's', 21, value)
            self.add_field('IR0', 's', 2, value)
            self.add_field('IRX', 's', 2, value)
            self.add_field('IRY', 's', 2, value)
            self.add_field('IRZ', 's', 2, value)
            self.add_field('IRXX', 's', 2, value)
            self.add_field('IRXY', 's', 2, value)
            self.add_field('IRXZ', 's', 2, value)
            self.add_field('IRYY', 's', 2, value)
            self.add_field('IRYZ', 's', 2, value)
            self.add_field('IRZZ', 's', 2, value)
            self.add_field('IC0', 's', 2, value)
            self.add_field('ICX', 's', 2, value)
            self.add_field('ICY', 's', 2, value)
            self.add_field('ICZ', 's', 2, value)
            self.add_field('ICXX', 's', 2, value)
            self.add_field('ICXY', 's', 2, value)
            self.add_field('ICXZ', 's', 2, value)
            self.add_field('ICYY', 's', 2, value)
            self.add_field('ICYZ', 's', 2, value)
            self.add_field('ICZZ', 's', 2, value)
            self.add_field('GX0', 's', 2, value)
            self.add_field('GY0', 's', 2, value)
            self.add_field('GZ0', 's', 2, value)
            self.add_field('GXR', 's', 2, value)
            self.add_field('GYR', 's', 2, value)
            self.add_field('GZR', 's', 2, value)
            self.add_field('GS', 's', 2, value)
            self.add_field('GXX', 's', 2, value)
            self.add_field('GXY', 's', 2, value)
            self.add_field('GXZ', 's', 2, value)
            self.add_field('GYX', 's', 2, value)
            self.add_field('GYY', 's', 2, value)
            self.add_field('GYZ', 's', 2, value)
            self.add_field('GZX', 's', 2, value)
            self.add_field('GZY', 's', 2, value)
            self.add_field('GZZ', 's', 2, value)
            self.add_loop('ERRCOVDATAs', self.IGN, ERRCOVDATA, value)
            self.add_loop('MAPMAT', int(self.NPAR * self.NPARO), MAP, value)

        if self.INCLUC == 'Y':
            self.add_field('URR', 's', 21, value)
            self.add_field('URC', 's', 21, value)
            self.add_field('UCC', 's', 21, value)
            self.add_field('UNCSR', 'd', 1, value)
            self.add_loop('UCORSEGRs', self.UNCSR, UCORSEGR, value)
            self.add_field('UNCSC', 'd', 1, value)
            self.add_loop('UCORSEGCs', self.UNCSC, UCORSEGC, value)


class RSMECA(TREExtension):
    _tag_value = 'RSMECA'
    _data_type = RSMECAType
