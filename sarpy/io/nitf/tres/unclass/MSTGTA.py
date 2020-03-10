# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class MSTGTAType(TREElement):
    def __init__(self, value):
        super(MSTGTAType, self).__init__()
        self.add_field('TGTNUM', 's', 5, value)
        self.add_field('TGTID', 's', 12, value)
        self.add_field('TGTBE', 's', 15, value)
        self.add_field('TGTPRI', 's', 3, value)
        self.add_field('TGTREQ', 's', 12, value)
        self.add_field('TGTLTIOV', 's', 12, value)
        self.add_field('TGTTYPE', 's', 1, value)
        self.add_field('TGTCOLL', 's', 1, value)
        self.add_field('TGTCAT', 's', 5, value)
        self.add_field('TGTUTC', 's', 7, value)
        self.add_field('TGTELEV', 's', 6, value)
        self.add_field('TGTELEVUNIT', 's', 1, value)
        self.add_field('TGTLOC', 's', 21, value)


class MSTGTA(TREExtension):
    _tag_value = 'MSTGTA'
    _data_type = MSTGTAType
