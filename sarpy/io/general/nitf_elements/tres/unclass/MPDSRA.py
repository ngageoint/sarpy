
from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class MPDSRAType(TREElement):
    def __init__(self, value):
        super(MPDSRAType, self).__init__()
        self.add_field('BLKNO', 's', 2, value)
        self.add_field('CDIPR', 's', 2, value)
        self.add_field('NBLKW', 's', 2, value)
        self.add_field('NRBLK', 's', 5, value)
        self.add_field('NCBLK', 's', 5, value)
        self.add_field('ORPX', 's', 9, value)
        self.add_field('ORPY', 's', 9, value)
        self.add_field('ORPZ', 's', 9, value)
        self.add_field('ORPRO', 's', 5, value)
        self.add_field('ORPCO', 's', 5, value)
        self.add_field('FPNVX', 's', 7, value)
        self.add_field('FPNVY', 's', 7, value)
        self.add_field('FPNVZ', 's', 7, value)
        self.add_field('ARPTM', 's', 9, value)
        self.add_field('RESV1', 's', 14, value)
        self.add_field('ARPPN', 's', 9, value)
        self.add_field('ARPPE', 's', 9, value)
        self.add_field('ARPPD', 's', 9, value)
        self.add_field('ARPVN', 's', 9, value)
        self.add_field('ARPVE', 's', 9, value)
        self.add_field('ARPVD', 's', 9, value)
        self.add_field('ARPAN', 's', 8, value)
        self.add_field('ARPAE', 's', 8, value)
        self.add_field('ARPAD', 's', 8, value)
        self.add_field('RESV2', 's', 13, value)


class MPDSRA(TREExtension):
    _tag_value = 'MPDSRA'
    _data_type = MPDSRAType
