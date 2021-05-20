
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class VTGT(TREElement):
    def __init__(self, value):
        super(VTGT, self).__init__()
        self.add_field('TGLOC', 's', 21, value)
        self.add_field('TGRDV', 's', 4, value)
        self.add_field('TGGSP', 's', 3, value)
        self.add_field('TGHEA', 's', 3, value)
        self.add_field('TGSIG', 's', 2, value)
        self.add_field('TGCAT', 's', 1, value)


class MTIRPAType(TREElement):
    def __init__(self, value):
        super(MTIRPAType, self).__init__()
        self.add_field('DESTP', 's', 2, value)
        self.add_field('MTPID', 's', 3, value)
        self.add_field('PCHNO', 's', 4, value)
        self.add_field('WAMFN', 's', 5, value)
        self.add_field('WAMBN', 's', 1, value)
        self.add_field('UTC', 's', 8, value)
        self.add_field('SQNTA', 's', 5, value)
        self.add_field('COSGZ', 's', 7, value)
        self.add_field('NVTGT', 'd', 3, value)
        self.add_loop('VTGTs', self.NVTGT, VTGT, value)


class MTIRPA(TREExtension):
    _tag_value = 'MTIRPA'
    _data_type = MTIRPAType
