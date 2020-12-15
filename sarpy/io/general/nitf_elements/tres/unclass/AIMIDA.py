# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class AIMIDA_69Type(TREElement):
    def __init__(self, value):
        super(AIMIDA_69Type, self).__init__()
        self.add_field('MISSION_DATE', 's', 7, value)
        self.add_field('MISSION_NO', 's', 4, value)
        self.add_field('FLIGHT_NO', 's', 2, value)
        self.add_field('OP_NUM', 's', 3, value)
        self.add_field('START_SEGMENT', 's', 2, value)
        self.add_field('REPRO_NUM', 's', 2, value)
        self.add_field('REPLAY', 's', 3, value)
        self.add_field('RESVD001', 's', 1, value)
        self.add_field('START_COLUMN', 's', 2, value)
        self.add_field('START_ROW', 's', 5, value)
        self.add_field('END_SEGMENT', 's', 2, value)
        self.add_field('END_COLUMN', 's', 2, value)
        self.add_field('END_ROW', 's', 5, value)
        self.add_field('COUNTRY', 's', 2, value)
        self.add_field('RESVD002', 's', 4, value)
        self.add_field('LOCATION', 's', 11, value)
        self.add_field('TIME', 's', 5, value)
        self.add_field('CREATION_DATE', 's', 7, value)


class AIMIDA_69(TREExtension):
    _tag_value = 'AIMIDA'
    _data_type = AIMIDA_69Type


class AIMIDA_73Type(TREElement):
    def __init__(self, value):
        super(AIMIDA_73Type, self).__init__()
        self.add_field('MISSION_DATE', 's', 8, value)
        self.add_field('MISSION_NO', 's', 4, value)
        self.add_field('FLIGHT_NO', 's', 2, value)
        self.add_field('OP_NUM', 's', 3, value)
        self.add_field('RESVD001', 's', 2, value)
        self.add_field('REPRO_NUM', 's', 2, value)
        self.add_field('REPLAY', 's', 3, value)
        self.add_field('RESVD002', 's', 1, value)
        self.add_field('START_COLUMN', 's', 3, value)
        self.add_field('START_ROW', 's', 5, value)
        self.add_field('RESVD003', 's', 2, value)
        self.add_field('END_COLUMN', 's', 3, value)
        self.add_field('END_ROW', 's', 5, value)
        self.add_field('COUNTRY', 's', 2, value)
        self.add_field('RESVD004', 's', 4, value)
        self.add_field('LOCATION', 's', 11, value)
        self.add_field('TIME', 's', 5, value)
        self.add_field('CREATION_DATE', 's', 8, value)


class AIMIDA_73(TREExtension):
    _tag_value = 'AIMIDA'
    _data_type = AIMIDA_73Type


class AIMIDA_89Type(TREElement):
    def __init__(self, value):
        super(AIMIDA_89Type, self).__init__()
        self.add_field('MISSION_DATE_TIME', 's', 14, value)
        self.add_field('MISSION_NO', 's', 4, value)
        self.add_field('MISSION_ID', 's', 10, value)
        self.add_field('FLIGHT_NO', 's', 2, value)
        self.add_field('OP_NUM', 's', 3, value)
        self.add_field('CURRENT_SEGMENT', 's', 2, value)
        self.add_field('REPRO_NUM', 's', 2, value)
        self.add_field('REPLAY', 's', 3, value)
        self.add_field('RESVD001', 's', 1, value)
        self.add_field('START_COLUMN', 's', 3, value)
        self.add_field('START_ROW', 's', 5, value)
        self.add_field('END_SEGMENT', 's', 2, value)
        self.add_field('END_COLUMN', 's', 3, value)
        self.add_field('END_ROW', 's', 5, value)
        self.add_field('COUNTRY', 's', 2, value)
        self.add_field('RESVD002', 's', 4, value)
        self.add_field('LOCATION', 's', 11, value)
        self.add_field('RESVD003', 's', 13, value)


class AIMIDA_89(TREExtension):
    _tag_value = 'AIMIDA'
    _data_type = AIMIDA_89Type


class AIMIDA(TREExtension):
    _tag_value = 'AIMIDA'

    def __init__(self):
        raise ValueError(
            'Not to be implemented directly. '
            'Use of one AIMIDA_69, AIMIDA_73, or AIMIDA_89')

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value : bytes
        start : int

        Returns
        -------
        AIMIDA_69|AIMIDA_73|AIMIDA_89
        """

        tag_value = value[start:start+6].decode('utf-8').strip()
        if tag_value != cls._tag_value:
            raise ValueError('tag value must be {}. Got {}'.format(cls._tag_value, tag_value))

        lng = int(value[start+6:start+11])
        if lng == 69:
            return AIMIDA_69.from_bytes(value, start)
        elif lng == 73:
            return AIMIDA_73.from_bytes(value, start)
        elif lng == 89:
            return AIMIDA_89.from_bytes(value, start)
        else:
            raise ValueError('the data must be length 69, 73, or 89. Got {}'.format(lng))
