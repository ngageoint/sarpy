# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class BAND(NITFElement):
    __slots__ = (
        'BANDPEAK', 'BANDLBOUND', 'BANDUBOUND', 'BANDWIDTH', 'BANDCALDRK', 'BANDCALINC', 'BANDRESP', 'BANDASD',
        'BANDGSD')
    _formats = {
        'BANDPEAK': '5s', 'BANDLBOUND': '5s', 'BANDUBOUND': '5s', 'BANDWIDTH': '5s', 'BANDCALDRK': '6s',
        'BANDCALINC': '5s', 'BANDRESP': '5s', 'BANDASD': '5s', 'BANDGSD': '5s'}


class BANDs(NITFLoop):
    _child_class = BAND
    _count_size = 4


class BANDSA(TRE):
    __slots__ = (
        'TAG', 'ROW_SPACING', 'ROW_SPACING_UNITS', 'COL_SPACING', 'COL_SPACING_UNITS', 'FOCAL_LENGTH', '_BANDs')
    _formats = {
        'TAG': '6s', 'ROW_SPACING': '7s', 'ROW_SPACING_UNITS': '1s', 'COL_SPACING': '7s', 'COL_SPACING_UNITS': '1s',
        'FOCAL_LENGTH': '6s'}
    _types = {'_BANDs': BANDs}
    _defaults = {'_BANDs': {}, 'TAG': 'BANDSA'}
    _enums = {'TAG': {'BANDSA', }}

    @property
    def BANDs(self):  # type: () -> BANDs
        return self._BANDs

    @BANDs.setter
    def BANDs(self, value):
        # noinspection PyAttributeOutsideInit
        self._BANDs = value
