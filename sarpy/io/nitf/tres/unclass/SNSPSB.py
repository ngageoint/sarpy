# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PT(TREElement):
    def __init__(self, value):
        super(PT, self).__init__()
        self.add_field('LON', 'd', 15, value)
        self.add_field('LAT', 'd', 15, value)


class BP(TREElement):
    def __init__(self, value):
        super(BP, self).__init__()
        self.add_field('NUM_PTS', 'd', 2, value)
        self.add_loop('PTs', self.NUM_PTS, PT, value)


class BND(TREElement):
    def __init__(self, value):
        super(BND, self).__init__()
        self.add_field('BID', 's', 5, value)
        self.add_field('WS1', 'd', 5, value)
        self.add_field('WS2', 'd', 5, value)


class AUX(TREElement):
    def __init__(self, value):
        super(AUX, self).__init__()
        self.add_field('API', 's', 20, value)
        self.add_field('APF', 's', 1, value)
        self.add_field('UNIAPX', 's', 7, value)
        self.add_field('APN', 'd', 10, value)
        self.add_field('APR', 'd', 20, value)
        self.add_field('APA', 's', 20, value)


class SNS(TREElement):
    def __init__(self, value):
        super(SNS, self).__init__()
        self.add_field('NUM_BP', 'd', 2, value)
        self.add_loop('BPs', self.NUM_BP, BP, value)
        self.add_field('NUM_BND', 'd', 2, value)
        self.add_loop('BNDs', self.NUM_BND, BND, value)
        self.add_field('UNIRES', 's', 3, value)
        self.add_field('REX', 'd', 6, value)
        self.add_field('REY', 'd', 6, value)
        self.add_field('GSX', 'd', 6, value)
        self.add_field('GSY', 'd', 6, value)
        self.add_field('GSL', 's', 12, value)
        self.add_field('PLTFM', 's', 8, value)
        self.add_field('INS', 's', 8, value)
        self.add_field('MOD', 's', 4, value)
        self.add_field('PRL', 's', 5, value)
        self.add_field('ACT', 's', 18, value)
        self.add_field('UNINOA', 's', 3, value)
        self.add_field('NOA', 'd', 7, value)
        self.add_field('UNIANG', 's', 3, value)
        self.add_field('ANG', 'd', 7, value)
        self.add_field('UNIALT', 's', 3, value)
        self.add_field('ALT', 'd', 9, value)
        self.add_field('LONSCC', 'd', 10, value)
        self.add_field('LATSCC', 'd', 10, value)
        self.add_field('UNISAE', 's', 3, value)
        self.add_field('SAZ', 'd', 7, value)
        self.add_field('SEL', 'd', 7, value)
        self.add_field('UNIRPY', 's', 3, value)
        self.add_field('ROL', 'd', 7, value)
        self.add_field('PIT', 'd', 7, value)
        self.add_field('YAW', 'd', 7, value)
        self.add_field('UNIPXT', 's', 3, value)
        self.add_field('PIXT', 'd', 14, value)
        self.add_field('UNISPE', 's', 7, value)
        self.add_field('ROS', 'd', 22, value)
        self.add_field('PIS', 'd', 22, value)
        self.add_field('YAS', 'd', 22, value)
        self.add_field('NUM_AUX', 'd', 3, value)
        self.add_loop('AUXs', self.NUM_AUX, AUX, value)


class SNSPSBType(TREElement):
    def __init__(self, value):
        super(SNSPSBType, self).__init__()
        self.add_field('NUMSNS', 'd', 2, value)
        self.add_loop('SNSs', self.NUMSNS, SNS, value)


class SNSPSB(TREExtension):
    _tag_value = 'SNSPSB'
    _data_type = SNSPSBType
