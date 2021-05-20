
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class TGT_QC(TREElement):
    def __init__(self, value):
        super(TGT_QC, self).__init__()
        self.add_field('TGT_QCOMMENT', 's', 40, value)


class TGT_CC(TREElement):
    def __init__(self, value):
        super(TGT_CC, self).__init__()
        self.add_field('TGT_CCOMMENT', 's', 40, value)


class REF_PT(TREElement):
    def __init__(self, value):
        super(REF_PT, self).__init__()
        self.add_field('TGT_REF', 's', 10, value)
        self.add_field('TGT_LL', 's', 21, value)
        self.add_field('TGT_ELEV', 's', 8, value)
        self.add_field('TGT_BAND', 's', 3, value)
        self.add_field('TGT_ROW', 's', 8, value)
        self.add_field('TGT_COL', 's', 8, value)
        self.add_field('TGT_PROW', 's', 8, value)
        self.add_field('TGT_PCOL', 's', 8, value)


class VALID_TGT(TREElement):
    def __init__(self, value):
        super(VALID_TGT, self).__init__()
        self.add_field('TGT_NAME', 's', 25, value)
        self.add_field('TGT_TYPE', 's', 15, value)
        self.add_field('TGT_VER', 's', 6, value)
        self.add_field('TGT_CAT', 's', 5, value)
        self.add_field('TGT_BE', 's', 17, value)
        self.add_field('TGT_SN', 's', 10, value)
        self.add_field('TGT_POSNUM', 's', 2, value)
        self.add_field('TGT_ATTITUDE_PITCH', 's', 6, value)
        self.add_field('TGT_ATTITUDE_ROLL', 's', 6, value)
        self.add_field('TGT_ATTITUDE_YAW', 's', 6, value)
        self.add_field('TGT_DIM_LENGTH', 's', 5, value)
        self.add_field('TGT_DIM_WIDTH', 's', 5, value)
        self.add_field('TGT_DIM_HEIGHT', 's', 5, value)
        self.add_field('TGT_AZIMUTH', 's', 6, value)
        self.add_field('TGT_CLTR_RATIO', 's', 8, value)
        self.add_field('TGT_STATE', 's', 10, value)
        self.add_field('TGT_COND', 's', 30, value)
        self.add_field('TGT_OBSCR', 's', 20, value)
        self.add_field('TGT_OBSCR%', 's', 3, value)
        self.add_field('TGT_CAMO', 's', 20, value)
        self.add_field('TGT_CAMO%', 's', 3, value)
        self.add_field('TGT_UNDER', 's', 12, value)
        self.add_field('TGT_OVER', 's', 30, value)
        self.add_field('TGT_TTEXTURE', 's', 45, value)
        self.add_field('TGT_PAINT', 's', 40, value)
        self.add_field('TGT_SPEED', 's', 3, value)
        self.add_field('TGT_HEADING', 's', 3, value)
        self.add_field('TGT_QC_NUM', 'd', 1, value)
        self.add_loop('TGT_QCs', self.TGT_QC_NUM, TGT_QC, value)
        self.add_field('TGT_CC_NUM', 'd', 1, value)
        self.add_loop('TGT_CCs', self.TGT_CC_NUM, TGT_CC, value)
        self.add_field('NO_REF_PT', 'd', 1, value)
        self.add_loop('REF_PTs', self.NO_REF_PT, REF_PT, value)


class ATTRIBUTE(TREElement):
    def __init__(self, value):
        super(ATTRIBUTE, self).__init__()
        self.add_field('ATTR_TGT_NUM', 's', 3, value)
        self.add_field('ATTR_NAME', 's', 30, value)
        self.add_field('ATTR_CONDTN', 's', 35, value)
        self.add_field('ATTR_VALUE', 's', 10, value)


class TRGTAType(TREElement):
    def __init__(self, value):
        super(TRGTAType, self).__init__()
        self.add_field('VERNUM', 's', 4, value)
        self.add_field('NO_VALID_TGTS', 'd', 3, value)
        self.add_field('NO_SCENE_TGTS', 'd', 3, value)
        self.add_loop('VALID_TGTs', self.NO_VALID_TGTS, VALID_TGT, value)
        self.add_field('NO_ATTRIBUTES', 'd', 3, value)
        self.add_loop('ATTRIBUTEs', self.NO_ATTRIBUTES, ATTRIBUTE, value)


class TRGTA(TREExtension):
    _tag_value = 'TRGTA'
    _data_type = TRGTAType
