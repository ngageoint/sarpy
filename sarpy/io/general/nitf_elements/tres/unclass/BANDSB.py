
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

# TODO: I'm not entirely confident that these bit operation work as expected


class PARAMETER(TREElement):
    def __init__(self, value, EXISTENCE_MASK):
        super(PARAMETER, self).__init__()
        if EXISTENCE_MASK & 0x10000000:
            self.add_field('BANDID', 's', 50, value)
        if EXISTENCE_MASK & 0x08000000:
            self.add_field('BAD_BAND', 'd', 1, value)
        if EXISTENCE_MASK & 0x04000000:
            self.add_field('NIIRS', 's', 3, value)
        if EXISTENCE_MASK & 0x02000000:
            self.add_field('FOCAL_LEN', 'd', 5, value)
        if EXISTENCE_MASK & 0x01000000:
            self.add_field('CWAVE', 's', 7, value)
        if EXISTENCE_MASK & 0x00800000:
            self.add_field('FWHM', 's', 7, value)
        if EXISTENCE_MASK & 0x00400000:
            self.add_field('FWHM_UNC', 's', 7, value)
        if EXISTENCE_MASK & 0x00200000:
            self.add_field('NOM_WAVE', 's', 7, value)
        if EXISTENCE_MASK & 0x00100000:
            self.add_field('NOM_WAVE_UNC', 's', 7, value)
        if EXISTENCE_MASK & 0x00080000:
            self.add_field('LBOUND', 's', 7, value)
            self.add_field('UBOUND', 's', 7, value)
        if EXISTENCE_MASK & 0x00040000:
            self.add_field('SCALE_FACTOR', 'ieee754_binary32', 4, value)
            self.add_field('ADDITIVE_FACTOR', 'ieee754_binary32', 4, value)
        if EXISTENCE_MASK & 0x00020000:
            self.add_field('START_TIME', 's', 16, value)
        if EXISTENCE_MASK & 0x00010000:
            self.add_field('INT_TIME', 's', 6, value)
        if EXISTENCE_MASK & 0x00008000:
            self.add_field('CALDRK', 's', 6, value)
            self.add_field('CALIBRATION_SENSITIVITY', 's', 5, value)
        if EXISTENCE_MASK & 0x00004000:
            self.add_field('ROW_GSD', 's', 7, value)
            if EXISTENCE_MASK & 0x00002000:
                self.add_field('ROW_GSD_UNC', 's', 7, value)
            self.add_field('ROW_GSD_UNIT', 's', 1, value)
            self.add_field('COL_GSD', 's', 7, value)
            if EXISTENCE_MASK & 0x00002000:
                self.add_field('COL_GSD_UNC', 's', 7, value)
            self.add_field('COL_GSD_UNIT', 's', 1, value)
        if EXISTENCE_MASK & 0x00001000:
            self.add_field('BKNOISE', 's', 5, value)
            self.add_field('SCNNOISE', 's', 5, value)
        if EXISTENCE_MASK & 0x00000800:
            self.add_field('SPT_RESP_FUNCTION_ROW', 's', 7, value)
            if EXISTENCE_MASK & 0x00000400:
                self.add_field('SPT_RESP_UNC_ROW', 's', 7, value)
            self.add_field('SPT_RESP_UNIT_ROW', 's', 1, value)
            self.add_field('SPT_RESP_FUNCTION_COL', 's', 7, value)
            if EXISTENCE_MASK & 0x00000400:
                self.add_field('SPT_RESP_UNC_COL', 's', 7, value)
            self.add_field('SPT_RESP_UNIT_COL', 's', 1, value)
        if EXISTENCE_MASK & 0x00000200:
            self.add_field('DATA_FLD_3', 'b', 16, value)
        if EXISTENCE_MASK & 0x00000100:
            self.add_field('DATA_FLD_4', 'b', 24, value)
        if EXISTENCE_MASK & 0x00000080:
            self.add_field('DATA_FLD_5', 'b', 32, value)
        if EXISTENCE_MASK & 0x00000040:
            self.add_field('DATA_FLD_6', 'b', 48, value)


class BAND(TREElement):
    def __init__(self, value):
        super(BAND, self).__init__()
        if self.BAPF == 'I':
            self.add_field('APN', 'd', 10, value)
        if self.BAPF == 'R':
            self.add_field('APR', 'ieee754_binary32', 4, value)
        if self.BAPF == 'A':
            self.add_field('APA', 's', 20, value)


class AUX_B(TREElement):
    def __init__(self, value):
        super(AUX_B, self).__init__()
        self.add_field('BAPF', 's', 1, value)
        self.add_field('UBAP', 's', 7, value)
        self.add_loop('BANDs', self.COUNT, BAND, value)


class AUX_C(TREElement):
    def __init__(self, value):
        super(AUX_C, self).__init__()
        self.add_field('CAPF', 's', 1, value)
        self.add_field('UCAP', 's', 7, value)
        if self.CAPF == 'I':
            self.add_field('APN', 'd', 10, value)
        if self.CAPF == 'R':
            self.add_field('APR', 'ieee754_binary32', 4, value)
        if self.CAPF == 'A':
            self.add_field('APA', 's', 20, value)


class BANDSBType(TREElement):
    def __init__(self, value):
        super(BANDSBType, self).__init__()
        self.add_field('COUNT', 'd', 5, value)
        self.add_field('RADIOMETRIC_QUANTITY', 's', 24, value)
        self.add_field('RADIOMETRIC_QUANTITY_UNIT', 's', 1, value)
        self.add_field('SCALE_FACTOR', 'ieee754_binary32', 4, value)
        self.add_field('ADDITIVE_FACTOR', 'ieee754_binary32', 4, value)
        self.add_field('ROW_GSD', 's', 7, value)
        self.add_field('ROW_GSD_UNIT', 's', 1, value)
        self.add_field('COL_GSD', 's', 7, value)
        self.add_field('COL_GSD_UNIT', 's', 1, value)
        self.add_field('SPT_RESP_ROW', 's', 7, value)
        self.add_field('SPT_RESP_UNIT_ROW', 's', 1, value)
        self.add_field('SPT_RESP_COL', 's', 7, value)
        self.add_field('SPT_RESP_UNIT_COL', 's', 1, value)
        self.add_field('DATA_FLD_1', 'b', 48, value)
        self.add_field('EXISTENCE_MASK', 'b', 4, value)
        existence_mask = int.from_bytes(self.EXISTENCE_MASK, byteorder='big')
        if existence_mask & 0x80000000:
            self.add_field('RADIOMETRIC_ADJUSTMENT_SURFACE', 's', 24, value)
            self.add_field('ATMOSPHERIC_ADJUSTMENT_ALTITUDE', 'ieee754_binary32', 4, value)
        if existence_mask & 0x40000000:
            self.add_field('DIAMETER', 's', 7, value)
        if existence_mask & 0x20000000:
            self.add_field('DATA_FLD_2', 'b', 32, value)
        if existence_mask & 0x01F80000:
            self.add_field('WAVE_LENGTH_UNIT', 's', 1, value)
        self.add_loop('PARAMETERs', self.COUNT, PARAMETER, value, existence_mask)
        if existence_mask & 0x00000001:
            self.add_field('NUM_AUX_B', 'd', 2, value)
            self.add_field('NUM_AUX_C', 'd', 2, value)
            self.add_loop('AUX_Bs', self.NUM_AUX_B, AUX_B, value)
            self.add_loop('AUX_Cs', self.NUM_AUX_C, AUX_C, value)


class BANDSB(TREExtension):
    _tag_value = 'BANDSB'
    _data_type = BANDSBType
