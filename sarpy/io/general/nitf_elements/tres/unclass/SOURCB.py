# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PT(TREElement):
    def __init__(self, value):
        super(PT, self).__init__()
        self.add_field('LON', 's', 15, value)
        self.add_field('LAT', 's', 15, value)


class BP(TREElement):
    def __init__(self, value):
        super(BP, self).__init__()
        self.add_field('NUM_PTS', 'd', 3, value)
        self.add_loop('PTs', self.NUM_PTS, PT, value)


class MI(TREElement):
    def __init__(self, value):
        super(MI, self).__init__()
        self.add_field('CDV30', 's', 8, value)
        self.add_field('UNIRAT', 's', 3, value)
        self.add_field('RAT', 's', 8, value)
        self.add_field('UNIGMA', 's', 3, value)
        self.add_field('GMA', 's', 8, value)
        self.add_field('LONGMA', 's', 15, value)
        self.add_field('LATGMA', 's', 15, value)
        self.add_field('UNIGCA', 's', 3, value)
        self.add_field('GCA', 's', 8, value)


class LI(TREElement):
    def __init__(self, value):
        super(LI, self).__init__()
        self.add_field('BAD', 's', 10, value)


class PRJ(TREElement):
    def __init__(self, value):
        super(PRJ, self).__init__()
        self.add_field('PRJ', 's', 15, value)


class IN(TREElement):
    def __init__(self, value):
        super(IN, self).__init__()
        self.add_field('INT', 's', 10, value)
        self.add_field('INS_SCA', 's', 9, value)
        self.add_field('NTL', 's', 15, value)
        self.add_field('TTL', 's', 15, value)
        self.add_field('NVL', 's', 15, value)
        self.add_field('TVL', 's', 15, value)
        self.add_field('NTR', 's', 15, value)
        self.add_field('TTR', 's', 15, value)
        self.add_field('NVR', 's', 15, value)
        self.add_field('TVR', 's', 15, value)
        self.add_field('NRL', 's', 15, value)
        self.add_field('TRL', 's', 15, value)
        self.add_field('NSL', 's', 15, value)
        self.add_field('TSL', 's', 15, value)
        self.add_field('NRR', 's', 15, value)
        self.add_field('TRR', 's', 15, value)
        self.add_field('NSR', 's', 15, value)
        self.add_field('TSR', 's', 15, value)


class SOUR(TREElement):
    def __init__(self, value):
        super(SOUR, self).__init__()
        self.add_field('NUM_BP', 'd', 2, value)
        self.add_loop('BPs', self.NUM_BP, BP, value)
        self.add_field('PRT', 's', 10, value)
        self.add_field('URF', 's', 20, value)
        self.add_field('EDN', 's', 7, value)
        self.add_field('NAM', 's', 20, value)
        self.add_field('CDP', 's', 3, value)
        self.add_field('CDV', 's', 8, value)
        self.add_field('CDV27', 's', 8, value)
        self.add_field('SRN', 's', 80, value)
        self.add_field('SCA', 's', 9, value)
        self.add_field('UNISQU', 's', 3, value)
        self.add_field('SQU', 's', 10, value)
        self.add_field('UNIPCI', 's', 3, value)
        self.add_field('PCI', 's', 4, value)
        self.add_field('WPC', 's', 3, value)
        self.add_field('NST', 's', 3, value)
        self.add_field('UNIHKE', 's', 3, value)
        self.add_field('HKE', 's', 6, value)
        self.add_field('LONHKE', 's', 15, value)
        self.add_field('LATHKE', 's', 15, value)
        self.add_field('QSS', 's', 1, value)
        self.add_field('QOD', 's', 1, value)
        self.add_field('CDV10', 's', 8, value)
        self.add_field('QLE', 's', 80, value)
        self.add_field('CPY', 's', 80, value)
        self.add_field('NMI', 'd', 2, value)
        self.add_loop('MIs', self.NMI, MI, value)
        self.add_field('NLI', 'd', 2, value)
        self.add_loop('LIs', self.NLI, LI, value)
        self.add_field('DAG', 's', 80, value)
        self.add_field('DCD', 's', 4, value)
        self.add_field('ELL', 's', 80, value)
        self.add_field('ELC', 's', 3, value)
        self.add_field('DVR', 's', 80, value)
        self.add_field('VDCDVR', 's', 4, value)
        self.add_field('SDA', 's', 80, value)
        self.add_field('VDCSDA', 's', 4, value)
        self.add_field('PRN', 's', 80, value)
        self.add_field('PCO', 's', 2, value)
        self.add_field('NUM_PRJ', 'd', 1, value)
        self.add_loop('PRJs', self.NUM_PRJ, PRJ, value)
        self.add_field('XOR', 's', 15, value)
        self.add_field('YOR', 's', 15, value)
        self.add_field('GRD', 's', 3, value)
        self.add_field('GRN', 's', 80, value)
        self.add_field('ZNA', 's', 4, value)
        self.add_field('NIN', 'd', 2, value)
        self.add_loop('INs', self.NIN, IN, value)


class SOURCBType(TREElement):
    def __init__(self, value):
        super(SOURCBType, self).__init__()
        self.add_field('IS_SCA', 's', 9, value)
        self.add_field('CPATCH', 's', 10, value)
        self.add_field('NUM_SOUR', 'd', 2, value)
        self.add_loop('SOURs', self.NUM_SOUR, SOUR, value)


class SOURCB(TREExtension):
    _tag_value = 'SOURCB'
    _data_type = SOURCBType
