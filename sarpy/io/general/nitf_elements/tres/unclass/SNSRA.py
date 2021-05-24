
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SENBAND1(TREElement):
    def __init__(self, value):
        super(SENBAND1, self).__init__()
        self.add_field('SENBAND', 's', 10, value)
        self.add_field('SEN_BANDWL', 's', 3, value)
        self.add_field('SEN_CEN_F', 's', 3, value)
        self.add_field('POLARIZATION', 's', 2, value)
        self.add_field('AZ_BWIDTH', 's', 6, value)
        self.add_field('EL_BWIDTH', 's', 6, value)
        self.add_field('DYN_RNGE', 's', 4, value)
        self.add_field('SENCALFAC', 's', 15, value)


class SENBAND2(TREElement):
    def __init__(self, value):
        super(SENBAND2, self).__init__()
        self.add_field('SENBAND', 's', 10, value)
        self.add_field('SEN_FOV_T', 's', 3, value)
        self.add_field('SEN_FOV_T_U', 's', 1, value)
        self.add_field('SEN_IFOV_T', 's', 3, value)
        self.add_field('SEN_IFOV_T_U', 's', 1, value)
        self.add_field('SEN_FOV_CT', 's', 5, value)
        self.add_field('SEN_IFOV_CT', 's', 3, value)
        self.add_field('SEN_IFOV_CT_U', 's', 1, value)
        self.add_field('SEN_FOR_T', 's', 3, value)
        self.add_field('SEN_FOR_CT', 's', 3, value)
        self.add_field('SEN_L_WAVE', 's', 4, value)
        self.add_field('SEN_U_WAVE', 's', 4, value)
        self.add_field('SUBBANDS', 's', 3, value)
        self.add_field('SENFLENGTH', 's', 4, value)
        self.add_field('SENFNUM', 's', 4, value)
        self.add_field('LINESAMPLES', 's', 4, value)
        self.add_field('DETECTTYPE', 's', 12, value)
        self.add_field('POLARIZATION', 's', 2, value)
        self.add_field('DYN_RNGE', 's', 4, value)
        self.add_field('SENCALFAC', 's', 15, value)


class SNSRAType(TREElement):
    def __init__(self, value):
        super(SNSRAType, self).__init__()
        self.add_field('VERNUM', 's', 4, value)
        self.add_field('SENNAME', 's', 20, value)
        self.add_field('SENTYPE', 's', 1, value)
        self.add_field('SENMODE', 's', 10, value)
        self.add_field('SENSCAN', 's', 12, value)
        self.add_field('SENSOR_ID', 's', 10, value)
        self.add_field('MPLAN', 's', 3, value)
        self.add_field('SENSERIAL', 's', 4, value)
        self.add_field('SENOPORG', 's', 10, value)
        self.add_field('SENMFG', 's', 12, value)
        self.add_field('ABSWVER', 's', 7, value)
        self.add_field('AVG_ALT', 's', 5, value)
        if self.SENTYPE == 'R':
            self.add_field('FOC_X', 's', 7, value)
            self.add_field('FOC_Y', 's', 7, value)
            self.add_field('FOC_Z', 's', 7, value)
            self.add_field('NUM_SENBAND', 'd', 1, value)
            self.add_loop('SENBANDs', self.NUM_SENBAND, SENBAND1, value)
        if self.SENTYPE in ['I', 'E']:
            self.add_field('NUM_SENBAND', 'd', 1, value)
            self.add_loop('SENBANDs', self.NUM_SENBAND, SENBAND2, value)


class SNSRA(TREExtension):
    _tag_value = 'SNSRA'
    _data_type = SNSRAType
