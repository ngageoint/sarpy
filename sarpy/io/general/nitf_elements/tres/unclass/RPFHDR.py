
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class RPFHDRType(TREElement):
    def __init__(self, value):
        super(RPFHDRType, self).__init__()
        self.add_field('ENDIAN', 'b', 1, value)
        self.add_field('HDSECL', 'b', 2, value)
        self.add_field('FILENM', 's', 12, value)
        self.add_field('NEWFLG', 'b', 1, value)
        self.add_field('STDNUM', 's', 15, value)
        self.add_field('STDDAT', 's', 8, value)
        self.add_field('CLASS', 's', 1, value)
        self.add_field('COUNTR', 's', 2, value)
        self.add_field('RELEAS', 's', 2, value)
        self.add_field('LOCSEC', 'b', 4, value)


class RPFHDR(TREExtension):
    _tag_value = 'RPFHDR'
    _data_type = RPFHDRType
