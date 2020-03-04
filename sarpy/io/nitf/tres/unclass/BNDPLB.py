# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PTS(NITFElement):
    __slots__ = ('LON', 'LAT')
    _formats = {'LON': '15d', 'LAT': '15d'}


class PTSs(NITFLoop):
    _child_class = PTS
    _count_size = 4


class BNDPLB(TRE):
    __slots__ = ('TAG', '_PTSs')
    _formats = {'TAG': '6s'}
    _types = {'_PTSs': PTSs}
    _defaults = {'_PTSs': {}, 'TAG': 'BNDPLB'}
    _enums = {'TAG': {'BNDPLB', }}

    @property
    def PTSs(self):  # type: () -> PTSs
        return self._PTSs

    @PTSs.setter
    def PTSs(self, value):
        # noinspection PyAttributeOutsideInit
        self._PTSs = value
