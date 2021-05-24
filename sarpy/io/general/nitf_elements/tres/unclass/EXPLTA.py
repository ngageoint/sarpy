
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class EXPLTA_87Type(TREElement):
    def __init__(self, value):
        super(EXPLTA_87Type, self).__init__()
        self.add_field('ANGLE_TO_NORTH', 's', 3, value)
        self.add_field('SQUINT_ANGLE', 's', 3, value)
        self.add_field('MODE', 's', 3, value)
        self.add_field('RESVD001', 's', 16, value)
        self.add_field('GRAZE_ANG', 's', 2, value)
        self.add_field('SLOPE_ANG', 's', 2, value)
        self.add_field('POLAR', 's', 2, value)
        self.add_field('NSAMP', 's', 5, value)
        self.add_field('RESVD002', 's', 1, value)
        self.add_field('SEQ_NUM', 's', 1, value)
        self.add_field('PRIME_ID', 's', 12, value)
        self.add_field('PRIME_BE', 's', 15, value)
        self.add_field('RESVD003', 's', 1, value)
        self.add_field('N_SEC', 's', 2, value)
        self.add_field('IPR', 's', 2, value)
        self.add_field('RESVD004', 's', 2, value)
        self.add_field('RESVD005', 's', 2, value)
        self.add_field('RESVD006', 's', 5, value)
        self.add_field('RESVD007', 's', 8, value)


class EXPLTA_87(TREExtension):
    _tag_value = 'EXPLTA'
    _data_type = EXPLTA_87Type


class EXPLTA_101Type(TREElement):
    def __init__(self, value):
        super(EXPLTA_101Type, self).__init__()
        self.add_field('ANGLE_TO_NORTH', 's', 7, value)
        self.add_field('SQUINT_ANGLE', 's', 7, value)
        self.add_field('MODE', 's', 3, value)
        self.add_field('RESVD001', 's', 16, value)
        self.add_field('GRAZE_ANG', 's', 5, value)
        self.add_field('SLOPE_ANG', 's', 5, value)
        self.add_field('POLAR', 's', 2, value)
        self.add_field('NSAMP', 's', 5, value)
        self.add_field('RESVD002', 's', 1, value)
        self.add_field('SEQ_NUM', 's', 1, value)
        self.add_field('PRIME_ID', 's', 12, value)
        self.add_field('PRIME_BE', 's', 15, value)
        self.add_field('RESVD003', 's', 1, value)
        self.add_field('N_SEC', 's', 2, value)
        self.add_field('IPR', 's', 2, value)


class EXPLTA_101(TREExtension):
    _tag_value = 'EXPLTA'
    _data_type = EXPLTA_101Type


class EXPLTA(TREExtension):
    _tag_value = 'EXPLTA'

    def __init__(self):
        raise ValueError(
            'Not to be implemented directly. '
            'Use of one EXPLTA_87 or  EXPLTA_101')

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value : bytes
        start : int

        Returns
        -------
        EXPLTA_87|EXPLTA_101
        """

        tag_value = value[start:start+6].decode('utf-8').strip()
        if tag_value != cls._tag_value:
            raise ValueError('tag value must be {}. Got {}'.format(cls._tag_value, tag_value))

        lng = int(value[start+6:start+11])
        if lng == 87:
            return EXPLTA_87.from_bytes(value, start)
        elif lng == 101:
            return EXPLTA_101.from_bytes(value, start)
        else:
            raise ValueError('the data must be length 87 or 101. Got {}'.format(lng))
