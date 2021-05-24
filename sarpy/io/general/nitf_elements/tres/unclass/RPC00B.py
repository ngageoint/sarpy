
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class LINE_NUM_COEFF(TREElement):
    def __init__(self, value):
        super(LINE_NUM_COEFF, self).__init__()
        self.add_field('LINE_NUM_COEFF', 's', 12, value)


class LINE_DEN_COEFF(TREElement):
    def __init__(self, value):
        super(LINE_DEN_COEFF, self).__init__()
        self.add_field('LINE_DEN_COEFF', 's', 12, value)


class SAMP_NUM_COEFF(TREElement):
    def __init__(self, value):
        super(SAMP_NUM_COEFF, self).__init__()
        self.add_field('SAMP_NUM_COEFF', 's', 12, value)


class SAMP_DEN_COEFF(TREElement):
    def __init__(self, value):
        super(SAMP_DEN_COEFF, self).__init__()
        self.add_field('SAMP_DEN_COEFF', 's', 12, value)


class RPC00BType(TREElement):
    def __init__(self, value):
        super(RPC00BType, self).__init__()
        self.add_field('SUCCESS', 's', 1, value)
        self.add_field('ERR_BIAS', 's', 7, value)
        self.add_field('ERR_RAND', 's', 7, value)
        self.add_field('LINE_OFF', 's', 6, value)
        self.add_field('SAMP_OFF', 's', 5, value)
        self.add_field('LAT_OFF', 's', 8, value)
        self.add_field('LONG_OFF', 's', 9, value)
        self.add_field('HEIGHT_OFF', 's', 5, value)
        self.add_field('LINE_SCALE', 's', 6, value)
        self.add_field('SAMP_SCALE', 's', 5, value)
        self.add_field('LAT_SCALE', 's', 8, value)
        self.add_field('LONG_SCALE', 's', 9, value)
        self.add_field('HEIGHT_SCALE', 's', 5, value)
        self.add_loop('LINE_NUM_COEFFs', 20, LINE_NUM_COEFF, value)
        self.add_loop('LINE_DEN_COEFFs', 20, LINE_DEN_COEFF, value)
        self.add_loop('SAMP_NUM_COEFFs', 20, SAMP_NUM_COEFF, value)
        self.add_loop('SAMP_DEN_COEFFs', 20, SAMP_DEN_COEFF, value)


class RPC00B(TREExtension):
    _tag_value = 'RPC00B'
    _data_type = RPC00BType
