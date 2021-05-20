
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SECREC(TREElement):
    def __init__(self, value):
        super(SECREC, self).__init__()
        self.add_field('LOCID', 'b', 1, value)
        self.add_field('SECLEN', 'b', 1, value)
        self.add_field('PHYSIDX', 'b', 1, value)


class RPFDESType(TREElement):
    def __init__(self, value):
        super(RPFDESType, self).__init__()
        self.add_field('LOCLEN', 'b', 1, value)
        # TODO: It's unclear whether this should be interpreted as an 8-bit int, or string?
        if int(self.LOCLEN) > 0:
            self.add_field('CLTOFF', 'b', 1, value)
            self.add_field('SECRECS', 'b', 1, value)
            self.add_field('RECLEN', 'b', 1, value)
            self.add_field('AGGLEN', 'b', 1, value)
            self.add_loop('SECRECList', int(self.SECRECS), SECREC, value)
        self.add_field('UNKNOWN', 'b', len(value) - self._bytes_length, value)


class RPFDES(TREExtension):
    _tag_value = 'RPFDES'
    _data_type = RPFDESType
