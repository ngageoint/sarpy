
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PT(TREElement):
    def __init__(self, value):
        super(PT, self).__init__()
        self.add_field('LON', 's', 15, value)
        self.add_field('LAT', 's', 15, value)


class ACHZ(TREElement):
    def __init__(self, value):
        super(ACHZ, self).__init__()
        self.add_field('UNIAAH', 's', 3, value)
        if self.UNIAAH != '':
            self.add_field('AAH', 's', 5, value)
        self.add_field('UNIAPH', 's', 3, value)
        if self.UNIAPH != '':
            self.add_field('APH', 's', 5, value)
        self.add_field('NUMPTS', 'd', 3, value)
        self.add_loop('PTs', self.NUMPTS, PT, value)


class ACCHZBType(TREElement):
    def __init__(self, value):
        super(ACCHZBType, self).__init__()
        self.add_field('NUMACHZ', 'd', 2, value)
        self.add_loop('ACHZs', self.NUMACHZ, ACHZ, value)


class ACCHZB(TREExtension):
    _tag_value = 'ACCHZB'
    _data_type = ACCHZBType
