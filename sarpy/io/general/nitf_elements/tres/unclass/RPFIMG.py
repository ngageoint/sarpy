# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SECREC(TREElement):
    def __init__(self, value):
        super(SECREC, self).__init__()
        self.add_field('LOCID', 'b', 1, value)
        self.add_field('SECLEN', 'b', 1, value)
        self.add_field('PHYSIDX', 'b', 1, value)


class RPFIMGType(TREElement):
    def __init__(self, value):
        super(RPFIMGType, self).__init__()
        self.add_field('LOCLEN', 'b', 1, value)
        self.add_field('CLTOFF', 'b', 1, value)
        self.add_field('SECRECS', 'b', 1, value)
        self.add_field('RECLEN', 'b', 1, value)
        self.add_field('AGGLEN', 'b', 1, value)
        self.add_loop('SECRECList', int(self.SECRECS), SECREC, value)
        self.add_field('UNKNOWN', 'b', len(value) - self._bytes_length, value)


class RPFIMG(TREExtension):
    _tag_value = 'RPFIMG'
    _data_type = RPFIMGType
