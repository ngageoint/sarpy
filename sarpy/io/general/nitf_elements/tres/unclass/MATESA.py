
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Brad Hards"

# From STDI-0002 Volume 1 Appendix AK: MATESA 1.1

class MATE(TREElement):
    def __init__(self, value):
        super(MATE, self).__init__()
        self.add_field('SOURCE', 's', 42, value)
        self.add_field('MATE_TYPE', 's', 16, value)
        self.add_field('MATE_ID_LEN', 'd', 4, value)
        self.add_field('MATE_ID', 's', self.MATE_ID_LEN, value)

class GROUP(TREElement):
    def __init__(self, value):
        super(GROUP, self).__init__()
        self.add_field('RELATIONSHIP', 's', 24, value)
        self.add_field('NUM_MATES', 'd', 4, value)
        self.add_loop('MATEs', self.NUM_MATES, MATE, value)

class MATESAType(TREElement):
    def __init__(self, value):
        super(MATESAType, self).__init__()
        self.add_field('CUR_SOURCE', 's', 42, value)
        self.add_field('CUR_MATE_TYPE', 's', 16, value)
        self.add_field('CUR_FILE_ID_LEN', 'd', 4, value)
        self.add_field('CUR_FILE_ID', 's', self.CUR_FILE_ID_LEN, value)
        self.add_field('NUM_GROUPS', 'd', 4, value)
        self.add_loop('GROUPs', self.NUM_GROUPS, GROUP, value)

class MATESA(TREExtension):
    _tag_value = 'MATESA'
    _data_type = MATESAType
