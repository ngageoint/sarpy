# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PT(TREElement):
    def __init__(self, value):
        super(PT, self).__init__()
        self.add_field('LON', 'd', 15, value)
        self.add_field('LAT', 'd', 15, value)


class ACPO(TREElement):
    def __init__(self, value):
        super(ACPO, self).__init__()
        self.add_field('UNIAAH', 's', 3, value)
        self.add_field('AAH', 'd', 5, value)
        self.add_field('UNIAAV', 's', 3, value)
        self.add_field('AAV', 'd', 5, value)
        self.add_field('UNIAPH', 's', 3, value)
        self.add_field('APH', 'd', 5, value)
        self.add_field('UNIAPV', 's', 3, value)
        self.add_field('APV', 'd', 5, value)
        self.add_field('NUMPTS', 'd', 3, value)
        self.add_loop('PTs', self.NUMPTS, PT, value)


class ACCPOBType(TREElement):
    def __init__(self, value):
        super(ACCPOBType, self).__init__()
        self.add_field('NUMACPO', 'd', 2, value)
        self.add_loop('ACPOs', self.NUMACPO, ACPO, value)


class ACCPOB(TREExtension):
    _tag_value = 'ACCPOB'
    _data_type = ACCPOBType
