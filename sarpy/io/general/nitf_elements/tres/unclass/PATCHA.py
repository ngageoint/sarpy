
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PATCHA_74Type(TREElement):
    def __init__(self, value):
        super(PATCHA_74Type, self).__init__()
        self.add_field('PAT_NO', 's', 4, value)
        self.add_field('LAST_PAT_FLAG', 's', 1, value)
        self.add_field('LNSTRT', 's', 7, value)
        self.add_field('LNSTOP', 's', 7, value)
        self.add_field('AZL', 's', 5, value)
        self.add_field('NVL', 's', 5, value)
        self.add_field('FVL', 's', 3, value)
        self.add_field('NPIXEL', 's', 5, value)
        self.add_field('FVPIX', 's', 5, value)
        self.add_field('FRAME', 's', 3, value)
        self.add_field('GMT', 's', 8, value)
        self.add_field('SHEAD', 's', 7, value)
        self.add_field('GSWEEP', 's', 6, value)
        self.add_field('SHEAR', 's', 8, value)


class PATCHA_74(TREExtension):
    _tag_value = 'PATCHA'
    _data_type = PATCHA_74Type


class PATCHA_115Type(TREElement):
    def __init__(self, value):
        super(PATCHA_115Type, self).__init__()
        self.add_field('PAT_NO', 's', 4, value)
        self.add_field('LAST_PAT_FLAG', 's', 1, value)
        self.add_field('LNSTRT', 's', 7, value)
        self.add_field('LNSTOP', 's', 7, value)
        self.add_field('AZL', 's', 5, value)
        self.add_field('NVL', 's', 5, value)
        self.add_field('FVL', 's', 3, value)
        self.add_field('NPIXEL', 's', 5, value)
        self.add_field('FVPIX', 's', 5, value)
        self.add_field('FRAME', 's', 3, value)
        self.add_field('UTC', 's', 8, value)
        self.add_field('SHEAD', 's', 7, value)
        self.add_field('GRAVITY', 's', 7, value)
        self.add_field('INS_V_NC', 's', 5, value)
        self.add_field('INS_V_EC', 's', 5, value)
        self.add_field('INS_V_DC', 's', 5, value)
        self.add_field('OFFLAT', 's', 8, value)
        self.add_field('OFFLONG', 's', 8, value)
        self.add_field('TRACK', 's', 3, value)
        self.add_field('GSWEEP', 's', 6, value)
        self.add_field('SHEAR', 's', 8, value)


class PATCHA_115(TREExtension):
    _tag_value = 'PATCHA'
    _data_type = PATCHA_115Type


class PATCHA(TREExtension):
    _tag_value = 'PATCHA'

    def __init__(self):
        raise ValueError(
            'Not to be implemented directly. '
            'Use of one PATCHA_74 or PATCHA_115')

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value : bytes
        start : int

        Returns
        -------
        PATCHA_74|PATCHA_115
        """

        tag_value = value[start:start+6].decode('utf-8').strip()
        if tag_value != cls._tag_value:
            raise ValueError('tag value must be {}. Got {}'.format(cls._tag_value, tag_value))

        lng = int(value[start+6:start+11])
        if lng == 74:
            return PATCHA_74.from_bytes(value, start)
        elif lng == 115:
            return PATCHA_115.from_bytes(value, start)
        else:
            raise ValueError('the data must be length 74 or 115. Got {}'.format(lng))
