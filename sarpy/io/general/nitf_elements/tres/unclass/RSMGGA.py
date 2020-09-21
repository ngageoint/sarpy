# -*- coding: utf-8 -*-

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PTOFF(TREElement):
    def __init__(self, value):
        super(PTOFF, self).__init__()
        self.add_field('IXO', 's', 4, value)
        self.add_field('IYO', 's', 4, value)


class GDPT(TREElement):
    def __init__(self, value, TNUMRD, TNUMCD):
        super(GDPT, self).__init__()
        self.add_field('RCOORD', 's', TNUMRD, value)
        self.add_field('CCOORD', 's', TNUMCD, value)


class GRID(TREElement):
    def __init__(self, value, TNUMRD, TNUMCD):
        super(GRID, self).__init__()
        self.add_field('NXPTS', 'd', 3, value)
        self.add_field('NYPTS', 'd', 3, value)
        if len(value) < self.NXPTS*self.NYPTS*(TNUMRD+TNUMCD):
            raise ValueError(
                'The input string is length {}, but there are supposed '
                'to be {}={}x{} elements each of length {}'.format(
                    len(value), self.NXPTS*self.NYPTS, self.NXPTS, self.NYPTS, TNUMRD+TNUMCD))
        self.add_loop('GDPTs', self.NXPTS*self.NYPTS, GDPT, value, TNUMRD, TNUMCD)


class RSMGGAType(TREElement):
    def __init__(self, value):
        super(RSMGGAType, self).__init__()
        self.add_field('IID', 's', 80, value)
        self.add_field('EDITION', 's', 40, value)
        self.add_field('GGRSN', 'd', 3, value)
        self.add_field('GGCSN', 'd', 3, value)
        self.add_field('GGRFEP', 's', 21, value)
        self.add_field('GGCFEP', 's', 21, value)
        self.add_field('INTORD', 's', 1, value)
        self.add_field('NPLN', 'd', 3, value)
        self.add_field('DELTAZ', 's', 21, value)
        self.add_field('DELTAX', 's', 21, value)
        self.add_field('DELTAY', 's', 21, value)
        self.add_field('ZPLN1', 's', 21, value)
        self.add_field('XIPLN1', 's', 21, value)
        self.add_field('YIPLN1', 's', 21, value)
        self.add_field('REFROW', 's', 9, value)
        self.add_field('REFCOL', 's', 9, value)
        self.add_field('TNUMRD', 'd', 2, value)
        self.add_field('TNUMCD', 'd', 2, value)
        self.add_field('FNUMRD', 'd', 1, value)
        self.add_field('FNUMCD', 'd', 1, value)
        self.add_loop('PTOFFs', self.NPLN-1, PTOFF, value)
        self.add_loop('GRIDs', self.NPLN, GRID, value, self.TNUMRD, self.TNUMCD)


class RSMGGA(TREExtension):
    _tag_value = 'RSMGGA'
    _data_type = RSMGGAType
