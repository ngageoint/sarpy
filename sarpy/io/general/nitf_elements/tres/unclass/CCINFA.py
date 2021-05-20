
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CODE(TREElement):
    def __init__(self, value):
        super(CODE, self).__init__()
        self.add_field('CODE_LEN', 'd', 1, value)
        self.add_field('CODE', 's', self.CODE_LEN, value)
        self.add_field('EQTYPE', 's', 1, value)
        self.add_field('ESURN_LEN', 'd', 2, value)
        self.add_field('ESURN', 's', self.ESURN_LEN, value)
        self.add_field('DETAIL_LEN', 'd', 5, value)
        if self.DETAIL_LEN > 0:
            self.add_field('DETAIL_CMPR', 's', 1, value)
            self.add_field('DETAIL', 's', self.DETAIL_LEN, value)


class CCINFAType(TREElement):
    def __init__(self, value):
        super(CCINFAType, self).__init__()
        self.add_field('NUMCODE', 'd', 3, value)
        self.add_loop('CODEs', self.NUMCODE, CODE, value)


class CCINFA(TREExtension):
    _tag_value = 'CCINFA'
    _data_type = CCINFAType
