
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SECTGAType(TREElement):
    def __init__(self, value):
        super(SECTGAType, self).__init__()
        self.add_field('SEC_ID', 's', 12, value)
        self.add_field('SEC_BE', 's', 15, value)
        self.add_field('RESVD001', 's', 1, value)


class SECTGA(TREExtension):
    _tag_value = 'SECTGA'
    _data_type = SECTGAType
