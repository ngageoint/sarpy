# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class VTGT(TREElement):
    def __init__(self, value):
        super(VTGT, self).__init__()
        self.add_field('TGLOC', 's', 23, value)
        self.add_field('TGLCA', 's', 6, value)
        self.add_field('TGRDV', 's', 4, value)
        self.add_field('TGGSP', 's', 3, value)
        self.add_field('TGHEA', 's', 3, value)
        self.add_field('TGSIG', 's', 2, value)
        self.add_field('TGCAT', 's', 1, value)


class MTIRPBType(TREElement):
    def __init__(self, value):
        super(MTIRPBType, self).__init__()
        self.add_field('DESTP', 's', 2, value)
        self.add_field('MTPID', 's', 3, value)
        self.add_field('PCHNO', 's', 4, value)
        self.add_field('WAMFN', 's', 5, value)
        self.add_field('WAMBN', 's', 1, value)
        self.add_field('UTC', 's', 14, value)
        self.add_field('ACLOC', 's', 21, value)
        self.add_field('ACALT', 's', 6, value)
        self.add_field('ACALU', 's', 1, value)
        self.add_field('ACHED', 's', 3, value)
        self.add_field('MTILR', 's', 1, value)
        self.add_field('SQNTA', 's', 5, value)
        self.add_field('COSGZ', 's', 7, value)
        self.add_field('NVTGT', 'd', 3, value)
        self.add_loop('VTGTs', self.NVTGT, VTGT, value)


class MTIRPB(TREExtension):
    _tag_value = 'MTIRPB'
    _data_type = MTIRPBType
