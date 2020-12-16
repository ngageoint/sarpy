# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class LAYER(TREElement):
    def __init__(self, value):
        super(LAYER, self).__init__()
        self.add_field('LAYER_ID', 's', 3, value)
        self.add_field('BITRATE', 's', 9, value)


class J2KLRAType(TREElement):
    def __init__(self, value):
        super(J2KLRAType, self).__init__()
        self.add_field('ORIG', 's', 1, value)
        self.add_field('NLEVELS_O', 's', 2, value)
        self.add_field('NBANDS_O', 's', 5, value)
        self.add_field('NLAYERS_O', 'd', 3, value)
        self.add_loop('LAYERs', self.NLAYERS_O, LAYER, value)
        if self.ORIG in [1, 3, 9]:
            self.add_field('NLEVELS_I', 's', 2, value)
            self.add_field('NBANDS_I', 's', 5, value)
            self.add_field('NLAYERS_I', 's', 3, value)


class J2KLRA(TREExtension):
    _tag_value = 'J2KLRA'
    _data_type = J2KLRAType
