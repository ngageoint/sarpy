
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BAND(TREElement):
    def __init__(self, value):
        super(BAND, self).__init__()
        self.add_field('BANDPEAK', 's', 5, value)
        self.add_field('BANDLBOUND', 's', 5, value)
        self.add_field('BANDUBOUND', 's', 5, value)
        self.add_field('BANDWIDTH', 's', 5, value)
        self.add_field('BANDCALDRK', 's', 6, value)
        self.add_field('BANDCALINC', 's', 5, value)
        self.add_field('BANDRESP', 's', 5, value)
        self.add_field('BANDASD', 's', 5, value)
        self.add_field('BANDGSD', 's', 5, value)


class BANDSAType(TREElement):
    def __init__(self, value):
        super(BANDSAType, self).__init__()
        self.add_field('ROW_SPACING', 's', 7, value)
        self.add_field('ROW_SPACING_UNITS', 's', 1, value)
        self.add_field('COL_SPACING', 's', 7, value)
        self.add_field('COL_SPACING_UNITS', 's', 1, value)
        self.add_field('FOCAL_LENGTH', 's', 6, value)
        self.add_field('BANDCOUNT', 'd', 4, value)
        self.add_loop('BANDs', self.BANDCOUNT, BAND, value)


class BANDSA(TREExtension):
    _tag_value = 'BANDSA'
    _data_type = BANDSAType
