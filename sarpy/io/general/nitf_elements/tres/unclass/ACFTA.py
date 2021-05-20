
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ACFTA_132Type(TREElement):
    def __init__(self, value):
        super(ACFTA_132Type, self).__init__()
        self.add_field('AC_MSN_ID', 's', 10, value)
        self.add_field('SCTYPE', 's', 1, value)
        self.add_field('SCNUM', 's', 4, value)
        self.add_field('SENSOR_ID', 's', 3, value)
        self.add_field('PATCH_TOT', 's', 4, value)
        self.add_field('MTI_TOT', 's', 3, value)
        self.add_field('PDATE', 's', 7, value)
        self.add_field('IMHOSTNO', 's', 3, value)
        self.add_field('IMREQID', 's', 5, value)
        self.add_field('SCENE_SOURCE', 's', 1, value)
        self.add_field('MPLAN', 's', 2, value)
        self.add_field('ENTLOC', 's', 21, value)
        self.add_field('ENTALT', 's', 6, value)
        self.add_field('EXITLOC', 's', 21, value)
        self.add_field('EXITALT', 's', 6, value)
        self.add_field('TMAP', 's', 7, value)
        self.add_field('RCS', 's', 3, value)
        self.add_field('ROW_SPACING', 's', 7, value)
        self.add_field('COL_SPACING', 's', 7, value)
        self.add_field('SENSERIAL', 's', 4, value)
        self.add_field('ABSWVER', 's', 7, value)


class ACFTA_132(TREExtension):
    _tag_value = 'ACFTA'
    _data_type = ACFTA_132Type


class ACFTA_154Type(TREElement):
    def __init__(self, value):
        super(ACFTA_154Type, self).__init__()
        self.add_field('AC_MSN_ID', 's', 10, value)
        self.add_field('AC_TAIL_NO', 's', 10, value)
        self.add_field('SENSOR_ID', 's', 10, value)
        self.add_field('SCENE_SOURCE', 's', 1, value)
        self.add_field('SCNUM', 's', 6, value)
        self.add_field('PDATE', 's', 8, value)
        self.add_field('IMHOSTNO', 's', 6, value)
        self.add_field('IMREQID', 's', 5, value)
        self.add_field('MPLAN', 's', 3, value)
        self.add_field('ENTLOC', 's', 21, value)
        self.add_field('ENTALT', 's', 6, value)
        self.add_field('EXITLOC', 's', 21, value)
        self.add_field('EXITALT', 's', 6, value)
        self.add_field('TMAP', 's', 7, value)
        self.add_field('ROW_SPACING', 's', 7, value)
        self.add_field('COL_SPACING', 's', 7, value)
        self.add_field('SENSERIAL', 's', 6, value)
        self.add_field('ABSWVER', 's', 7, value)
        self.add_field('PATCH_TOT', 's', 4, value)
        self.add_field('MTI_TOT', 's', 3, value)


class ACFTA_154(TREExtension):
    _tag_value = 'ACFTA'
    _data_type = ACFTA_154Type


class ACFTA_199Type(TREElement):
    def __init__(self, value):
        super(ACFTA_199Type, self).__init__()
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
        self.add_field('ENTELV', 's', 6, value)
        self.add_field('ELVUNIT', 's', 1, value)
        self.add_field('EXITLOC', 's', 25, value)
        self.add_field('EXITELV', 's', 6, value)
        self.add_field('TMAP', 's', 7, value)
        self.add_field('RESERVD1', 's', 1, value)
        self.add_field('ROW_SPACING', 's', 7, value)
        self.add_field('COL_SPACING', 's', 7, value)
        self.add_field('FOCAL_LENGTH', 's', 6, value)
        self.add_field('SENSERIAL', 's', 6, value)
        self.add_field('ABSWVER', 's', 7, value)
        self.add_field('CAL_DATE', 's', 8, value)
        self.add_field('PATCH_TOT', 's', 4, value)
        self.add_field('MTI_TOT', 's', 3, value)


class ACFTA_199(TREExtension):
    _tag_value = 'ACFTA'
    _data_type = ACFTA_199Type


class ACFTA(TREExtension):
    _tag_value = 'ACFTA'

    def __init__(self):
        raise ValueError(
            'Not to be implemented directly. '
            'Use of one ACFTA_132, ACFTA_154, or ACFTA_199')

    @classmethod
    def from_bytes(cls, value, start):
        """

        Parameters
        ----------
        value : bytes
        start : int

        Returns
        -------
        ACFTA_132|ACFTA_154|ACFTA_199
        """

        tag_value = value[start:start+6].decode('utf-8').strip()
        if tag_value != cls._tag_value:
            raise ValueError('tag value must be {}. Got {}'.format(cls._tag_value, tag_value))

        lng = int(value[start+6:start+11])
        if lng == 132:
            return ACFTA_132.from_bytes(value, start)
        elif lng == 154:
            return ACFTA_154.from_bytes(value, start)
        elif lng == 199:
            return ACFTA_199.from_bytes(value, start)
        else:
            raise ValueError('the data must be length 132, 154, or 199. Got {}'.format(lng))
