
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ACFTBType(TREElement):
    def __init__(self, value):
        super(ACFTBType, self).__init__()
        self.add_field('AC_MSN_ID', 's', 20, value)
        self.add_field('AC_TAIL_NO', 's', 10, value)
        self.add_field('AC_TO', 's', 12, value)
        self.add_field('SENSOR_ID_TYPE', 's', 4, value)
        self.add_field('SENSOR_ID', 's', 6, value)
        self.add_field('SCENE_SOURCE', 's', 1, value)
        self.add_field('SCNUM', 's', 6, value)
        self.add_field('PDATE', 's', 8, value)
        self.add_field('IMHOSTNO', 's', 6, value)
        self.add_field('IMREQID', 's', 5, value)
        self.add_field('MPLAN', 's', 3, value)
        self.add_field('ENTLOC', 's', 25, value)
        self.add_field('LOC_ACCY', 's', 6, value)
        self.add_field('ENTELV', 's', 6, value)
        self.add_field('ELV_UNIT', 's', 1, value)
        self.add_field('EXITLOC', 's', 25, value)
        self.add_field('EXITELV', 's', 6, value)
        self.add_field('TMAP', 's', 7, value)
        self.add_field('ROW_SPACING', 's', 7, value)
        self.add_field('ROW_SPACING_UNITS', 's', 1, value)
        self.add_field('COL_SPACING', 's', 7, value)
        self.add_field('COL_SPACING_UNITS', 's', 1, value)
        self.add_field('FOCAL_LENGTH', 's', 6, value)
        self.add_field('SENSERIAL', 's', 6, value)
        self.add_field('ABSWVER', 's', 7, value)
        self.add_field('CAL_DATE', 's', 8, value)
        self.add_field('PATCH_TOT', 's', 4, value)
        self.add_field('MTI_TOT', 's', 3, value)


class ACFTB(TREExtension):
    _tag_value = 'ACFTB'
    _data_type = ACFTBType
