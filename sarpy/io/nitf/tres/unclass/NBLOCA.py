# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class FRAME(NITFElement):
    __slots__ = ('FRAME_OFFSET',)
    _formats = {'FRAME_OFFSET': '4b'}


class FRAMEs(NITFLoop):
    _child_class = FRAME
    _count_size = 4


class NBLOCA(TRE):
    __slots__ = ('TAG', 'FRAME_1_OFFSET', '_FRAMEs')
    _formats = {'TAG': '6s', 'FRAME_1_OFFSET': '4b'}
    _types = {'_FRAMEs': FRAMEs}
    _defaults = {'_FRAMEs': {}, 'TAG': 'NBLOCA'}
    _enums = {'TAG': {'NBLOCA', }}

    @property
    def FRAMEs(self):  # type: () -> FRAMEs
        return self._FRAMEs

    @FRAMEs.setter
    def FRAMEs(self, value):
        # noinspection PyAttributeOutsideInit
        self._FRAMEs = value
