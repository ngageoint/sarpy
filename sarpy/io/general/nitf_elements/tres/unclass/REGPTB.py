
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PT(TREElement):
    def __init__(self, value):
        super(PT, self).__init__()
        self.add_field('PID', 's', 10, value)
        self.add_field('LON', 's', 15, value)
        self.add_field('LAT', 's', 15, value)
        self.add_field('ZVL', 's', 15, value)
        self.add_field('DIX', 's', 11, value)
        self.add_field('DIY', 's', 11, value)


class REGPTBType(TREElement):
    def __init__(self, value):
        super(REGPTBType, self).__init__()
        self.add_field('NUMPTS', 'd', 4, value)
        self.add_loop('PTs', self.NUMPTS, PT, value)


class REGPTB(TREExtension):
    _tag_value = 'REGPTB'
    _data_type = REGPTBType
