
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CSSHPAType(TREElement):
    def __init__(self, value):
        super(CSSHPAType, self).__init__()
        self.add_field('SHAPE_USE', 's', 25, value)
        self.add_field('SHAPE_CLASS', 's', 10, value)
        if self.SHAPE_USE.strip() == 'CLOUD_SHAPES':
            self.add_field('CC_SOURCE', 's', 18, value)
        self.add_field('SHAPE1_NAME', 's', 3, value)
        self.add_field('SHAPE1_START', 's', 6, value)
        self.add_field('SHAPE2_NAME', 's', 3, value)
        self.add_field('SHAPE2_START', 's', 6, value)
        self.add_field('SHAPE3_NAME', 's', 3, value)
        self.add_field('SHAPE3_START', 's', 6, value)


class CSSHPA(TREExtension):
    _tag_value = 'CSSHPA'
    _data_type = CSSHPAType
