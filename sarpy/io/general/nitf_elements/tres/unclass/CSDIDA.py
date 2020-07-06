# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CSDIDAType(TREElement):
    def __init__(self, value):
        super(CSDIDAType, self).__init__()
        self.add_field('DAY', 'd', 2, value)
        self.add_field('MONTH', 's', 3, value)
        self.add_field('YEAR', 'd', 4, value)
        self.add_field('PLATFORM_CODE', 's', 2, value)
        self.add_field('VEHICLE_ID', 'd', 2, value)
        self.add_field('PASS', 'd', 2, value)
        self.add_field('OPERATION', 'd', 3, value)
        self.add_field('SENSOR_ID', 's', 2, value)
        self.add_field('PRODUCT_ID', 's', 2, value)
        self.add_field('RESERVED_1', 's', 4, value)
        self.add_field('TIME', 'd', 14, value)
        self.add_field('PROCESS_TIME', 'd', 14, value)
        self.add_field('RESERVED_2', 'd', 2, value)
        self.add_field('RESERVED_3', 'd', 2, value)
        self.add_field('RESERVED_4', 's', 1, value)
        self.add_field('RESERVED_5', 's', 1, value)
        self.add_field('SOFTWARE_VERSION_NUMBER', 's', 10, value)


class CSDIDA(TREExtension):
    _tag_value = 'CSDIDA'
    _data_type = CSDIDAType
