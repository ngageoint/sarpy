# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PTS(NITFElement):
    __slots__ = ('PID', 'LON', 'LAT', 'ZVL', 'DIX', 'DIY')
    _formats = {
        'PID': '10s', 'LON': '15d', 'LAT': '15d', 'ZVL': '15d', 'DIX': '11d', 'DIY': '11d'}


class PTSs(NITFLoop):
    _child_class = PTS
    _count_size = 4


class REGPTB(TRE):
    __slots__ = ('TAG', '_PTSs')
    _formats = {'TAG': '6s'}
    _types = {'_PTSs': PTSs}
    _defaults = {'_PTSs': {}, 'TAG': 'REGPTB'}
    _enums = {'TAG': {'REGPTB', }}

    @property
    def PTSs(self):  # type: () -> PTSs
        return self._PTSs

    @PTSs.setter
    def PTSs(self, value):
        # noinspection PyAttributeOutsideInit
        self._PTSs = value
