
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class IPCOM(TREElement):
    def __init__(self, value):
        super(IPCOM, self).__init__()
        self.add_field('IPCOM', 's', 80, value)


class EVENT(TREElement):
    def __init__(self, value):
        super(EVENT, self).__init__()
        self.add_field('PDATE', 's', 14, value)
        self.add_field('PSITE', 's', 10, value)
        self.add_field('PAS', 's', 10, value)
        self.add_field('NIPCOM', 'd', 1, value)
        self.add_loop('IPCOMs', self.NIPCOM, IPCOM, value)
        self.add_field('IBPP', 's', 2, value)
        self.add_field('IPVTYPE', 's', 3, value)
        self.add_field('INBWC', 's', 10, value)
        self.add_field('DISP_FLAG', 's', 1, value)
        self.add_field('ROT_FLAG', 'd', 1, value)
        if self.ROT_FLAG == 1:
            self.add_field('ROT_ANGLE', 's', 8, value)
        self.add_field('ASYM_FLAG', 's', 1, value)
        if self.ASYM_FLAG == 1:
            self.add_field('ZOOMROW', 's', 7, value)
            self.add_field('ZOOMCOL', 's', 7, value)
        self.add_field('PROJ_FLAG', 's', 1, value)
        self.add_field('SHARP_FLAG', 'd', 1, value)
        if self.SHARP_FLAG == 1:
            self.add_field('SHARPFAM', 's', 2, value)
            self.add_field('SHARPMEM', 's', 2, value)
        self.add_field('MAG_FLAG', 'd', 1, value)
        if self.MAG_FLAG == 1:
            self.add_field('MAG_LEVEL', 's', 7, value)
        self.add_field('DRA_FLAG', 'd', 1, value)
        if self.DRA_FLAG == 1:
            self.add_field('DRA_MULT', 's', 7, value)
            self.add_field('DRA_SUB', 's', 5, value)
        self.add_field('TTC_FLAG', 'd', 1, value)
        if self.TTC_FLAG == 1:
            self.add_field('TTCFAM', 's', 2, value)
            self.add_field('TTCMEM', 's', 2, value)
        self.add_field('DEVLUT_FLAG', 'd', 1, value)
        self.add_field('OBPP', 's', 2, value)
        self.add_field('OPVTYPE', 's', 3, value)
        self.add_field('OUTBWC', 's', 10, value)


class HISTOAType(TREElement):
    def __init__(self, value):
        super(HISTOAType, self).__init__()
        self.add_field('SYSTYPE', 's', 20, value)
        self.add_field('PC', 's', 12, value)
        self.add_field('PE', 's', 4, value)
        self.add_field('REMAP_FLAG', 's', 1, value)
        self.add_field('LUTID', 's', 2, value)
        self.add_field('NEVENTS', 'd', 2, value)
        self.add_loop('EVENTs', self.NEVENTS, EVENT, value)


class HISTOA(TREExtension):
    _tag_value = 'HISTOA'
    _data_type = HISTOAType
