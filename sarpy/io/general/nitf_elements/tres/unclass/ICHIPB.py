
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ICHIPBType(TREElement):
    def __init__(self, value):
        super(ICHIPBType, self).__init__()
        self.add_field('XFRM_FLAG', 's', 2, value)
        self.add_field('SCALE_FACTOR', 's', 10, value)
        self.add_field('ANAMRPH_CORR', 's', 2, value)
        self.add_field('SCANBLK_NUM', 's', 2, value)
        self.add_field('OP_ROW_11', 's', 12, value)
        self.add_field('OP_COL_11', 's', 12, value)
        self.add_field('OP_ROW_12', 's', 12, value)
        self.add_field('OP_COL_12', 's', 12, value)
        self.add_field('OP_ROW_21', 's', 12, value)
        self.add_field('OP_COL_21', 's', 12, value)
        self.add_field('OP_ROW_22', 's', 12, value)
        self.add_field('OP_COL_22', 's', 12, value)
        self.add_field('FI_ROW_11', 's', 12, value)
        self.add_field('FI_COL_11', 's', 12, value)
        self.add_field('FI_ROW_12', 's', 12, value)
        self.add_field('FI_COL_12', 's', 12, value)
        self.add_field('FI_ROW_21', 's', 12, value)
        self.add_field('FI_COL_21', 's', 12, value)
        self.add_field('FI_ROW_22', 's', 12, value)
        self.add_field('FI_COL_22', 's', 12, value)
        self.add_field('FI_ROW', 's', 8, value)
        self.add_field('FI_COL', 's', 8, value)


class ICHIPB(TREExtension):
    _tag_value = 'ICHIPB'
    _data_type = ICHIPBType
