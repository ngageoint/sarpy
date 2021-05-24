
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BCKGDAType(TREElement):
    def __init__(self, value):
        super(BCKGDAType, self).__init__()
        self.add_field('BGWIDTH', 's', 8, value)
        self.add_field('BGHEIGHT', 's', 8, value)
        self.add_field('BGRED', 's', 8, value)
        self.add_field('BGGREEN', 's', 8, value)
        self.add_field('BGBLUE', 's', 8, value)
        self.add_field('PIXSIZE', 's', 8, value)
        self.add_field('PIXUNITS', 's', 8, value)


class BCKGDA(TREExtension):
    _tag_value = 'BCKGDA'
    _data_type = BCKGDAType
