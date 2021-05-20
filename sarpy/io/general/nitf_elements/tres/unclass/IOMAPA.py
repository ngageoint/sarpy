
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class IOMAPA_6Type(TREElement):
    def __init__(self, value):
        super(IOMAPA_6Type, self).__init__()
        self.add_field('BAND_NUMBER', 's', 3, value)
        self.add_field('MAP_SELECT', 's', 1, value)
        self.add_field('S2', 's', 2, value)


class IOMAPA_6(TREExtension):
    _tag_value = 'IOMAPA'
    _data_type = IOMAPA_6Type


class IOMAPA_16Type(TREElement):
    def __init__(self, value):
        super(IOMAPA_16Type, self).__init__()
        self.add_field('BAND_NUMBER', 's', 3, value)
        self.add_field('MAP_SELECT', 's', 1, value)
        self.add_field('TABLE_ID', 's', 2, value)
        self.add_field('S1', 's', 2, value)
        self.add_field('S2', 's', 2, value)
        self.add_field('R_WHOLE', 's', 3, value)
        self.add_field('R_FRACTION', 's', 3, value)


class IOMAPA_16(TREExtension):
    _tag_value = 'IOMAPA'
    _data_type = IOMAPA_16Type


class SEGMENT(TREElement):
    def __init__(self, value):
        super(SEGMENT, self).__init__()
        self.add_field('OUT_B0', 'b', 4, value)
        self.add_field('OUT_B1', 'b', 4, value)
        self.add_field('OUT_B2', 'b', 4, value)
        self.add_field('OUT_B3', 'b', 4, value)
        self.add_field('OUT_B4', 'b', 4, value)
        self.add_field('OUT_B5', 'b', 4, value)


class IOMAPA_91Type(TREElement):
    def __init__(self, value):
        super(IOMAPA_91Type, self).__init__()
        self.add_field('BAND_NUMBER', 's', 3, value)
        self.add_field('MAP_SELECT', 's', 1, value)
        self.add_field('TABLE_ID', 's', 1, value)
        self.add_field('S1', 's', 2, value)
        self.add_field('S2', 's', 2, value)
        self.add_field('NO_OF_SEGMENTS', 'd', 1, value)
        self.add_field('XOB_1', 's', 4, value)
        self.add_field('XOB_2', 's', 4, value)
        self.add_loop('SEGMENTs', self.NO_OF_SEGMENTS, SEGMENT, value)


class IOMAPA_91(TREExtension):
    _tag_value = 'IOMAPA'
    _data_type = IOMAPA_91Type


class MAP(TREElement):
    def __init__(self, value):
        super(MAP, self).__init__()
        self.add_field('OUTPUT_MAP_VALUE', 's', 2, value)


class IOMAPA_8202Type(TREElement):
    def __init__(self, value):
        super(IOMAPA_8202Type, self).__init__()
        self.add_field('BAND_NUMBER', 's', 3, value)
        self.add_field('MAP_SELECT', 's', 1, value)
        self.add_field('TABLE_ID', 's', 2, value)
        self.add_field('S1', 's', 2, value)
        self.add_field('S2', 's', 2, value)
        self.add_loop('MAPs', 4096, MAP, value)


class IOMAPA_8202(TREExtension):
    _tag_value = 'IOMAPA'
    _data_type = IOMAPA_8202Type


class IOMAPA(TREExtension):
    _tag_value = 'IOMAPA'

    def __init__(self):
        raise ValueError(
            'Not to be implemented directly. '
            'Use of one IOMAPA_6, IOMAPA_16, IOMAPA_91, or IOMAPA_8202')

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value : bytes
        start : int

        Returns
        -------
        IOMAPA_6|IOMAPA_16|IOMAPA_91|IOMAPA_8202
        """

        tag_value = value[start:start+6].decode('utf-8').strip()
        if tag_value != cls._tag_value:
            raise ValueError('tag value must be {}. Got {}'.format(cls._tag_value, tag_value))

        lng = int(value[start+6:start+11])
        if lng == 6:
            return IOMAPA_6.from_bytes(value, start)
        elif lng == 16:
            return IOMAPA_16.from_bytes(value, start)
        elif lng == 91:
            return IOMAPA_91.from_bytes(value, start)
        elif lng == 8202:
            return IOMAPA_8202.from_bytes(value, start)
        else:
            raise ValueError('the data must be length 6, 16, 91, or 8202. Got {}'.format(lng))
