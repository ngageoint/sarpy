
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PIAEQAType(TREElement):
    def __init__(self, value):
        super(PIAEQAType, self).__init__()
        self.add_field('EQPCODE', 's', 7, value)
        self.add_field('EQPNOMEN', 's', 45, value)
        self.add_field('EQPMAN', 's', 64, value)
        self.add_field('OBTYPE', 's', 1, value)
        self.add_field('ORDBAT', 's', 3, value)
        self.add_field('CTRYPROD', 's', 2, value)
        self.add_field('CTRYDSN', 's', 2, value)
        self.add_field('OBJVIEW', 's', 6, value)


class PIAEQA(TREExtension):
    _tag_value = 'PIAEQA'
    _data_type = PIAEQAType
