
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CSDIDAType(TREElement):
    def __init__(self, value):
        super(CSDIDAType, self).__init__()
        self.add_field('DAY', 's', 2, value)
        self.add_field('MONTH', 's', 3, value)
        self.add_field('YEAR', 's', 4, value)
        self.add_field('PLATFORM_CODE', 's', 2, value)
        self.add_field('VEHICLE_ID', 's', 2, value)
        self.add_field('PASS', 's', 2, value)
        self.add_field('OPERATION', 's', 3, value)
        self.add_field('SENSOR_ID', 's', 2, value)
        self.add_field('PRODUCT_ID', 's', 2, value)
        self.add_field('RESERVED_1', 's', 4, value)
        self.add_field('TIME', 's', 14, value)
        self.add_field('PROCESS_TIME', 's', 14, value)
        self.add_field('RESERVED_2', 's', 2, value)
        self.add_field('RESERVED_3', 's', 2, value)
        self.add_field('RESERVED_4', 's', 1, value)
        self.add_field('RESERVED_5', 's', 1, value)
        self.add_field('SOFTWARE_VERSION_NUMBER', 's', 10, value)


class CSDIDA(TREExtension):
    _tag_value = 'CSDIDA'
    _data_type = CSDIDAType
