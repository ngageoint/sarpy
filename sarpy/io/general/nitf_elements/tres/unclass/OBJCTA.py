# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class OBJ(TREElement):
    def __init__(self, value):
        super(OBJ, self).__init__()
        self.add_field('OBJ_TY', 's', 20, value)
        self.add_field('OBJ_NM', 's', 15, value)
        self.add_field('OBJ_POS', 's', 2, value)
        self.add_field('OBJ_SN', 's', 10, value)
        self.add_field('OBJ_LL', 's', 21, value)
        self.add_field('OBJ_ELEV', 's', 8, value)
        self.add_field('OBJ_ROW', 's', 8, value)
        self.add_field('OBJ_COL', 's', 8, value)
        self.add_field('OBJ_PROW', 's', 8, value)
        self.add_field('OBJ_PCOL', 's', 8, value)
        self.add_field('OBJ_ATTR', 's', 20, value)
        self.add_field('OBJ_SEN', 's', 2, value)
        if self.OBJ_SEN == 'R':
            self.add_field('OBJ_AZ_3DB_WIDTH', 's', 7, value)
            self.add_field('OBJ_RNG_3DB_WIDTH', 's', 7, value)
            self.add_field('OBJ_AZ_18DB_WIDTH', 's', 7, value)
            self.add_field('OBJ_RNG_18DB_WIDTH', 's', 7, value)
            self.add_field('OBJ_AZ_3_18DB_RATIO', 's', 8, value)
            self.add_field('OBJ_RNG_3_18DB_RATIO', 's', 8, value)
            self.add_field('OBJ_AZ_PK_SL_RATIO', 's', 8, value)
            self.add_field('OBJ_RNG_PK_SL_RATIO', 's', 8, value)
            self.add_field('OBJ_AZ_INT_SL_RATIO', 's', 8, value)
            self.add_field('OBJ_RNGINT_SL_RATIO', 's', 8, value)
        elif self.OBJ_SEN in ['EO', 'IR']:
            self.add_field('OBJ_CAL_TEMP', 's', 6, value)


class OBJCTAType(TREElement):
    def __init__(self, value):
        super(OBJCTAType, self).__init__()
        self.add_field('VERNUM', 's', 4, value)
        self.add_field('NUM_OBJ', 's', 3, value)
        self.add_field('OBJ_REF', 's', 10, value)
        self.add_field('NUM_SCENE_OBJ', 'd', 3, value)
        self.add_loop('OBJs', self.NUM_OBJ, OBJ, value)


class OBJCTA(TREExtension):
    _tag_value = 'OBJCTA'
    _data_type = OBJCTAType
