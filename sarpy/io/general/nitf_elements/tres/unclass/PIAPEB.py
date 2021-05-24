
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PIAPEBType(TREElement):
    def __init__(self, value):
        super(PIAPEBType, self).__init__()
        self.add_field('LASTNME', 's', 28, value)
        self.add_field('FIRSTNME', 's', 28, value)
        self.add_field('MIDNME', 's', 28, value)
        self.add_field('DOB', 's', 8, value)
        self.add_field('ASSOCTRY', 's', 2, value)


class PIAPEB(TREExtension):
    _tag_value = 'PIAPEB'
    _data_type = PIAPEBType
