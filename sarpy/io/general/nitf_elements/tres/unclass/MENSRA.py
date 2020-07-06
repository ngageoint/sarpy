# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class MENSRA_155Type(TREElement):
    def __init__(self, value):
        super(MENSRA_155Type, self).__init__()
        self.add_field('CCRP_LOC', 's', 21, value)
        self.add_field('CCRP_ALT', 's', 6, value)
        self.add_field('OF_PC_R', 's', 7, value)
        self.add_field('OF_PC_A', 's', 7, value)
        self.add_field('COSGRZ', 's', 7, value)
        self.add_field('RGCCRP', 's', 7, value)
        self.add_field('RLMAP', 's', 1, value)
        self.add_field('CCRP_ROW', 's', 5, value)
        self.add_field('CCRP_COL', 's', 5, value)
        self.add_field('ACFT_LOC', 's', 21, value)
        self.add_field('ACFT_ALT', 's', 5, value)
        self.add_field('C_R_NC', 's', 7, value)
        self.add_field('C_R_EC', 's', 7, value)
        self.add_field('C_R_DC', 's', 7, value)
        self.add_field('C_AZ_NC', 's', 7, value)
        self.add_field('C_AZ_EC', 's', 7, value)
        self.add_field('C_AZ_DC', 's', 7, value)
        self.add_field('C_AL_NC', 's', 7, value)
        self.add_field('C_AL_EC', 's', 7, value)
        self.add_field('C_AL_DC', 's', 7, value)


class MENSRA_155(TREExtension):
    _tag_value = 'MENSRA'
    _data_type = MENSRA_155Type


class MENSRA_174Type(TREElement):
    def __init__(self, value):
        super(MENSRA_174Type, self).__init__()
        self.add_field('ACFT_LOC', 's', 21, value)
        self.add_field('ACFT_ALT', 's', 6, value)
        self.add_field('CCRP_LOC', 's', 21, value)
        self.add_field('CCRP_ALT', 's', 6, value)
        self.add_field('OF_PC_R', 's', 7, value)
        self.add_field('OF_PC_A', 's', 7, value)
        self.add_field('COSGRZ', 's', 7, value)
        self.add_field('RGCCRP', 's', 7, value)
        self.add_field('RLMAP', 's', 1, value)
        self.add_field('CCRP_ROW', 's', 5, value)
        self.add_field('CCRP_COL', 's', 5, value)
        self.add_field('C_R_NC', 's', 9, value)
        self.add_field('C_R_EC', 's', 9, value)
        self.add_field('C_R_DC', 's', 9, value)
        self.add_field('C_AZ_NC', 's', 9, value)
        self.add_field('C_AZ_EC', 's', 9, value)
        self.add_field('C_AZ_DC', 's', 9, value)
        self.add_field('C_AL_NC', 's', 9, value)
        self.add_field('C_AL_EC', 's', 9, value)
        self.add_field('C_AL_DC', 's', 9, value)


class MENSRA_174(TREExtension):
    _tag_value = 'MENSRA'
    _data_type = MENSRA_174Type


class MENSRA_185Type(TREElement):
    def __init__(self, value):
        super(MENSRA_185Type, self).__init__()
        self.add_field('ACFT_LOC', 's', 25, value)
        self.add_field('ACFT_ALT', 's', 6, value)
        self.add_field('CCRP_LOC', 's', 25, value)
        self.add_field('CCRP_ALT', 's', 6, value)
        self.add_field('OF_PC_R', 's', 7, value)
        self.add_field('OF_PC_A', 's', 7, value)
        self.add_field('COSGZ', 's', 7, value)
        self.add_field('RGCCRP', 's', 7, value)
        self.add_field('RLMAP', 's', 1, value)
        self.add_field('CCRP_ROW', 's', 5, value)
        self.add_field('CCRP_COL', 's', 5, value)
        self.add_field('C_R_NC', 's', 10, value)
        self.add_field('C_R_EC', 's', 10, value)
        self.add_field('C_R_DC', 's', 10, value)
        self.add_field('C_AZ_NC', 's', 9, value)
        self.add_field('C_AZ_EC', 's', 9, value)
        self.add_field('C_AZ_DC', 's', 9, value)
        self.add_field('C_AL_NC', 's', 9, value)
        self.add_field('C_AL_EC', 's', 9, value)
        self.add_field('C_AL_DC', 's', 9, value)
        self.add_field('TOTAL_TILES_COLS', 's', 3, value)
        self.add_field('TOTAL_TILES_ROWS', 's', 5, value)


class MENSRA_185(TREExtension):
    _tag_value = 'MENSRA'
    _data_type = MENSRA_185Type


class MENSRA(TREExtension):
    _tag_value = 'MENSRA'

    def __init__(self):
        raise ValueError(
            'Not to be implemented directly. '
            'Use of one MENSRA_155, MENSRA_174, or MENSRA_185')

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value : bytes
        start : int

        Returns
        -------
        MENSRA_155|MENSRA_174|MENSRA_185
        """

        tag_value = value[start:start+6].decode('utf-8').strip()
        if tag_value != cls._tag_value:
            raise ValueError('tag value must be {}. Got {}'.format(cls._tag_value, tag_value))

        lng = int(value[start+6:start+11])
        if lng == 155:
            return MENSRA_155.from_bytes(value, start)
        elif lng == 174:
            return MENSRA_174.from_bytes(value, start)
        elif lng == 185:
            return MENSRA_185.from_bytes(value, start)
        else:
            raise ValueError('the data must be length 155, 174, or 185. Got {}'.format(lng))
