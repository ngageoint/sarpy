
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class EXPLTBType(TREElement):
    def __init__(self, value):
        super(EXPLTBType, self).__init__()
        self.add_field('ANGLE_TO_NORTH', 's', 7, value)
        self.add_field('ANGLE_TO_NORTH_ACCY', 's', 6, value)
        self.add_field('SQUINT_ANGLE', 's', 7, value)
        self.add_field('SQUINT_ANGLE_ACCY', 's', 6, value)
        self.add_field('MODE', 's', 3, value)
        self.add_field('RESVD001', 's', 16, value)
        self.add_field('GRAZE_ANG', 's', 5, value)
        self.add_field('GRAZE_ANG_ACCY', 's', 5, value)
        self.add_field('SLOPE_ANG', 's', 5, value)
        self.add_field('POLAR', 's', 2, value)
        self.add_field('NSAMP', 's', 5, value)
        self.add_field('RESVD002', 's', 1, value)
        self.add_field('SEQ_NUM', 's', 1, value)
        self.add_field('PRIME_ID', 's', 12, value)
        self.add_field('PRIME_BE', 's', 15, value)
        self.add_field('RESVD003', 's', 1, value)
        self.add_field('N_SEC', 's', 2, value)
        self.add_field('IPR', 's', 2, value)


class EXPLTB(TREExtension):
    _tag_value = 'EXPLTB'
    _data_type = EXPLTBType
