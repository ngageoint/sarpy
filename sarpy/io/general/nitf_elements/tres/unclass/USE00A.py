# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class USE00AType(TREElement):
    def __init__(self, value):
        super(USE00AType, self).__init__()
        self.add_field('ANGLE_TO_NORTH', 's', 3, value)
        self.add_field('MEAN_GSD', 's', 5, value)
        self.add_field('RSRVD01', 's', 1, value)
        self.add_field('DYNAMIC_RANGE', 's', 5, value)
        self.add_field('RSRVD02', 's', 3, value)
        self.add_field('RSRVD03', 's', 1, value)
        self.add_field('RSRVD04', 's', 3, value)
        self.add_field('OBL_ANG', 's', 5, value)
        self.add_field('ROLL_ANG', 's', 6, value)
        self.add_field('RSRVD05', 's', 12, value)
        self.add_field('RSRVD06', 's', 15, value)
        self.add_field('RSRVD07', 's', 4, value)
        self.add_field('RSRVD08', 's', 1, value)
        self.add_field('RSRVD09', 's', 3, value)
        self.add_field('RSRVD10', 's', 1, value)
        self.add_field('RSRVD11', 's', 1, value)
        self.add_field('N_REF', 's', 2, value)
        self.add_field('REV_NUM', 's', 5, value)
        self.add_field('N_SEG', 's', 3, value)
        self.add_field('MAX_LP_SEG', 's', 6, value)
        self.add_field('RSRVD12', 's', 6, value)
        self.add_field('RSRVD13', 's', 6, value)
        self.add_field('SUN_EL', 's', 5, value)
        self.add_field('SUN_AZ', 's', 5, value)


class USE00A(TREExtension):
    _tag_value = 'USE00A'
    _data_type = USE00AType
