
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SENSRAType(TREElement):
    def __init__(self, value):
        super(SENSRAType, self).__init__()
        self.add_field('REFROW', 's', 8, value)
        self.add_field('REFCOL', 's', 8, value)
        self.add_field('SNSMODEL', 's', 6, value)
        self.add_field('SNSMOUNT', 's', 3, value)
        self.add_field('SENSLOC', 's', 21, value)
        self.add_field('SNALTSRC', 's', 1, value)
        self.add_field('SENSALT', 's', 6, value)
        self.add_field('SNALUNIT', 's', 1, value)
        self.add_field('SENSAGL', 's', 5, value)
        self.add_field('SNSPITCH', 's', 7, value)
        self.add_field('SENSROLL', 's', 8, value)
        self.add_field('SENSYAW', 's', 8, value)
        self.add_field('PLTPITCH', 's', 7, value)
        self.add_field('PLATROLL', 's', 8, value)
        self.add_field('PLATHDG', 's', 5, value)
        self.add_field('GRSPDSRC', 's', 1, value)
        self.add_field('GRDSPEED', 's', 6, value)
        self.add_field('GRSPUNIT', 's', 1, value)
        self.add_field('GRDTRACK', 's', 5, value)
        self.add_field('VERTVEL', 's', 5, value)
        self.add_field('VERTVELU', 's', 1, value)
        self.add_field('SWATHFRM', 's', 4, value)
        self.add_field('NSWATHS', 's', 4, value)
        self.add_field('SPOTNUM', 's', 3, value)


class SENSRA(TREExtension):
    _tag_value = 'SENSRA'
    _data_type = SENSRAType
