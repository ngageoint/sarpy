# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class RNT(TREElement):
    def __init__(self, value):
        super(RNT, self).__init__()
        self.add_field('RNPCF', 's', 21, value)


class RDT(TREElement):
    def __init__(self, value):
        super(RDT, self).__init__()
        self.add_field('RDPCF', 's', 21, value)


class CNT(TREElement):
    def __init__(self, value):
        super(CNT, self).__init__()
        self.add_field('CNPCF', 's', 21, value)


class CDT(TREElement):
    def __init__(self, value):
        super(CDT, self).__init__()
        self.add_field('CDPCF', 's', 21, value)


class RSMPCAType(TREElement):
    def __init__(self, value):
        super(RSMPCAType, self).__init__()
        self.add_field('IID', 's', 80, value)
        self.add_field('EDITION', 's', 40, value)
        self.add_field('RSN', 'd', 3, value)
        self.add_field('CSN', 'd', 3, value)
        self.add_field('RFEP', 's', 21, value)
        self.add_field('CFEP', 's', 21, value)
        self.add_field('RNRMO', 's', 21, value)
        self.add_field('CNRMO', 's', 21, value)
        self.add_field('XNRMO', 's', 21, value)
        self.add_field('YNRMO', 's', 21, value)
        self.add_field('ZNRMO', 's', 21, value)
        self.add_field('RNRMSF', 's', 21, value)
        self.add_field('CNRMSF', 's', 21, value)
        self.add_field('XNRMSF', 's', 21, value)
        self.add_field('YNRMSF', 's', 21, value)
        self.add_field('ZNRMSF', 's', 21, value)
        self.add_field('RNPWRX', 'd', 1, value)
        self.add_field('RNPWRY', 'd', 1, value)
        self.add_field('RNPWRZ', 'd', 1, value)
        self.add_field('RNTRMS', 'd', 3, value)
        self.add_loop('RNTs', self.RNTRMS, RNT, value)
        self.add_field('RDPWRX', 'd', 1, value)
        self.add_field('RDPWRY', 'd', 1, value)
        self.add_field('RDPWRZ', 'd', 1, value)
        self.add_field('RDTRMS', 'd', 3, value)
        self.add_loop('RDTs', self.RDTRMS, RDT, value)
        self.add_field('CNPWRX', 'd', 1, value)
        self.add_field('CNPWRY', 'd', 1, value)
        self.add_field('CNPWRZ', 'd', 1, value)
        self.add_field('CNTRMS', 'd', 3, value)
        self.add_loop('CNTs', self.CNTRMS, CNT, value)
        self.add_field('CDPWRX', 'd', 1, value)
        self.add_field('CDPWRY', 'd', 1, value)
        self.add_field('CDPWRZ', 'd', 1, value)
        self.add_field('CDTRMS', 'd', 3, value)
        self.add_loop('CDTs', self.CDTRMS, CDT, value)


class RSMPCA(TREExtension):
    _tag_value = 'RSMPCA'
    _data_type = RSMPCAType
