# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BAND(NITFElement):
    __slots__ = (
        'BAND_TYPE', 'BAND_ID', 'FOC_LENGTH', 'NUM_DAP', 'NUM_FIR', 'DELTA', 'OPPOFF_X', 'OPPOFF_Y', 'OPPOFF_Z',
        'START_X', 'START_Y', 'FINISH_X', 'FINISH_Y')
    _formats = {
        'BAND_TYPE': '1s', 'BAND_ID': '6s', 'FOC_LENGTH': '11d', 'NUM_DAP': '8d', 'NUM_FIR': '8d',
        'DELTA': '7d', 'OPPOFF_X': '7d', 'OPPOFF_Y': '7d', 'OPPOFF_Z': '7d', 'START_X': '11d', 'START_Y': '11d',
        'FINISH_X': '11d', 'FINISH_Y': '11d'}


class BANDs(NITFLoop):
    _child_class = BAND
    _count_size = 1


class CSSFAA(TRE):
    __slots__ = ('TAG', '_BANDs')
    _formats = {'TAG': '6s'}
    _types = {'_BANDs': BANDs}
    _defaults = {'_BANDs': {}, 'TAG': 'CSSFAA'}
    _enums = {'TAG': {'CSSFAA', }}

    @property
    def BANDs(self):  # type: () -> BANDs
        return self._BANDs

    @BANDs.setter
    def BANDs(self, value):
        # noinspection PyAttributeOutsideInit
        self._BANDs = value
