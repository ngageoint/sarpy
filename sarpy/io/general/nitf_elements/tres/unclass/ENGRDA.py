
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"



class RECORD(TREElement):
    def __init__(self, value):
        super(RECORD, self).__init__()
        self.add_field('ENGLN', 'd', 2, value)
        self.add_field('ENGLBL', 's', self.ENGLN, value)
        self.add_field('ENGMTXC', 's', 4, value)
        self.add_field('ENGMTXR', 's', 4, value)
        self.add_field('ENGTYP', 's', 1, value)
        self.add_field('ENGDTS', 'd', 1, value)
        self.add_field('ENGDTU', 's', 2, value)
        self.add_field('ENGDATC', 'd', 8, value)
        self.add_field('ENGDATA', 'b', self.ENGDATC*self.ENGDTS, value)


class ENGRDAType(TREElement):
    def __init__(self, value):
        super(ENGRDAType, self).__init__()
        self.add_field('RESRC', 's', 20, value)
        self.add_field('RECNT', 'd', 3, value)
        self.add_loop('RECORDSs', self.RECNT, RECORD, value)


class ENGRDA(TREExtension):
    _tag_value = 'ENGRDA'
    _data_type = ENGRDAType
