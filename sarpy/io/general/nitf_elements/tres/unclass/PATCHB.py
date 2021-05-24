
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PATCHBType(TREElement):
    def __init__(self, value):
        super(PATCHBType, self).__init__()
        self.add_field('PAT_NO', 's', 4, value)
        self.add_field('LAST_PAT_FLAG', 's', 1, value)
        self.add_field('LNSTRT', 's', 7, value)
        self.add_field('LNSTOP', 's', 7, value)
        self.add_field('AZL', 's', 5, value)
        self.add_field('NVL', 's', 5, value)
        self.add_field('FVL', 's', 3, value)
        self.add_field('NPIXEL', 's', 5, value)
        self.add_field('FVPIX', 's', 5, value)
        self.add_field('FRAME', 's', 3, value)
        self.add_field('UTC', 's', 8, value)
        self.add_field('SHEAD', 's', 7, value)
        self.add_field('GRAVITY', 's', 7, value)
        self.add_field('INS_V_NC', 's', 5, value)
        self.add_field('INS_V_EC', 's', 5, value)
        self.add_field('INS_V_DC', 's', 5, value)
        self.add_field('OFFLAT', 's', 8, value)
        self.add_field('OFFLONG', 's', 8, value)
        self.add_field('TRACK', 's', 3, value)
        self.add_field('GSWEEP', 's', 6, value)
        self.add_field('SHEAR', 's', 8, value)
        self.add_field('BATCH_NO', 's', 6, value)


class PATCHB(TREExtension):
    _tag_value = 'PATCHB'
    _data_type = PATCHBType
