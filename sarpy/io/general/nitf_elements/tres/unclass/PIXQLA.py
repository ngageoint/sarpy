
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class AIS(TREElement):
    def __init__(self, value):
        super(AIS, self).__init__()
        self.add_field('AISDLVL', 's', 3, value)


class PIXQUAL(TREElement):
    def __init__(self, value):
        super(PIXQUAL, self).__init__()
        self.add_field('PQ_CONDITION', 's', 40, value)


class PIXQLAType(TREElement):
    def __init__(self, value):
        super(PIXQLAType, self).__init__()
        self.add_field('NUMAIS', 's', 3, value)
        if self.NUMAIS != 'ALL':
            self.add_loop('AISs', int(self.NUMAIS), AIS, value)
        self.add_field('NPIXQUAL', 'd', 4, value)
        self.add_field('PQ_BIT_VALUE', 's', 1, value)
        self.add_loop('PIXQUALs', self.NPIXQUAL, PIXQUAL, value)


class PIXQLA(TREExtension):
    _tag_value = 'PIXQLA'
    _data_type = PIXQLAType
