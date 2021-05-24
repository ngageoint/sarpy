
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
        self.add_field('NUM_PTS', 'd', 2, value)
        self.add_loop('PTs', self.NUM_PTS, PT, value)


class BND(TREElement):
    def __init__(self, value):
        super(BND, self).__init__()
        self.add_field('BID', 's', 5, value)
        self.add_field('WS1', 's', 5, value)
        self.add_field('WS2', 's', 5, value)


class AUX(TREElement):
    def __init__(self, value):
        super(AUX, self).__init__()
        self.add_field('API', 's', 20, value)
        self.add_field('APF', 's', 1, value)
        self.add_field('UNIAPX', 's', 7, value)
        self.add_field('APN', 's', 10, value)
        self.add_field('APR', 's', 20, value)
        self.add_field('APA', 's', 20, value)


class SNS(TREElement):
    def __init__(self, value):
        super(SNS, self).__init__()
        self.add_field('NUM_BP', 'd', 2, value)
        self.add_loop('BPs', self.NUM_BP, BP, value)
        self.add_field('NUM_BND', 'd', 2, value)
        self.add_loop('BNDs', self.NUM_BND, BND, value)
        self.add_field('UNIRES', 's', 3, value)
        self.add_field('REX', 's', 6, value)
        self.add_field('REY', 's', 6, value)
        self.add_field('GSX', 's', 6, value)
        self.add_field('GSY', 's', 6, value)
        self.add_field('GSL', 's', 12, value)
        self.add_field('PLTFM', 's', 8, value)
        self.add_field('INS', 's', 8, value)
        self.add_field('MOD', 's', 4, value)
        self.add_field('PRL', 's', 5, value)
        self.add_field('ACT', 's', 18, value)
        self.add_field('UNINOA', 's', 3, value)
        self.add_field('NOA', 's', 7, value)
        self.add_field('UNIANG', 's', 3, value)
        self.add_field('ANG', 's', 7, value)
        self.add_field('UNIALT', 's', 3, value)
        self.add_field('ALT', 's', 9, value)
        self.add_field('LONSCC', 's', 10, value)
        self.add_field('LATSCC', 's', 10, value)
        self.add_field('UNISAE', 's', 3, value)
        self.add_field('SAZ', 's', 7, value)
        self.add_field('SEL', 's', 7, value)
        self.add_field('UNIRPY', 's', 3, value)
        self.add_field('ROL', 's', 7, value)
        self.add_field('PIT', 's', 7, value)
        self.add_field('YAW', 's', 7, value)
        self.add_field('UNIPXT', 's', 3, value)
        self.add_field('PIXT', 's', 14, value)
        self.add_field('UNISPE', 's', 7, value)
        self.add_field('ROS', 's', 22, value)
        self.add_field('PIS', 's', 22, value)
        self.add_field('YAS', 's', 22, value)
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
