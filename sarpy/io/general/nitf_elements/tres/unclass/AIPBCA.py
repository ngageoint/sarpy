# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class AIPBCAType(TREElement):
    def __init__(self, value):
        super(AIPBCAType, self).__init__()
        self.add_field('PATCH_WIDTH', 's', 5, value)
        self.add_field('U_HAT_X', 's', 16, value)
        self.add_field('U_HAT_Y', 's', 16, value)
        self.add_field('U_HAT_Z', 's', 16, value)
        self.add_field('V_HAT_X', 's', 16, value)
        self.add_field('V_HAT_Y', 's', 16, value)
        self.add_field('V_HAT_Z', 's', 16, value)
        self.add_field('N_HAT_X', 's', 16, value)
        self.add_field('N_HAT_Y', 's', 16, value)
        self.add_field('N_HAT_Z', 's', 16, value)
        self.add_field('DEP_ANGLE', 's', 7, value)
        self.add_field('CT_TRACK_RANGE', 's', 10, value)
        self.add_field('ETA_0', 's', 16, value)
        self.add_field('ETA_1', 's', 16, value)
        self.add_field('X_IMG_U', 's', 9, value)
        self.add_field('X_IMG_V', 's', 9, value)
        self.add_field('X_IMG_N', 's', 9, value)
        self.add_field('Y_IMG_U', 's', 9, value)
        self.add_field('Y_IMG_V', 's', 9, value)
        self.add_field('Y_IMG_N', 's', 9, value)
        self.add_field('Z_IMG_U', 's', 9, value)
        self.add_field('Z_IMG_V', 's', 9, value)
        self.add_field('Z_IMG_N', 's', 9, value)
        self.add_field('CT_HAT_X', 's', 9, value)
        self.add_field('CT_HAT_Y', 's', 9, value)
        self.add_field('CT_HAT_Z', 's', 9, value)
        self.add_field('SCL_PT_U', 's', 13, value)
        self.add_field('SCL_PT_V', 's', 13, value)
        self.add_field('SCL_PT_N', 's', 13, value)
        self.add_field('SIGMA_SM', 's', 13, value)
        self.add_field('SIGMA_SN', 's', 13, value)
        self.add_field('S_OFF', 's', 10, value)
        self.add_field('RN_OFFSET', 's', 12, value)
        self.add_field('CRP_RANGE', 's', 11, value)
        self.add_field('REF_DEP_ANG', 's', 7, value)
        self.add_field('REF_ASP_ANG', 's', 9, value)
        self.add_field('N_SKIP_AZ', 's', 1, value)
        self.add_field('N_SKIP_RANGE', 's', 1, value)


class AIPBCA(TREExtension):
    _tag_value = 'AIPBCA'
    _data_type = AIPBCAType
