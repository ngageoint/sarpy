# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PTS(NITFElement):
    __slots__ = ('LON', 'LAT')
    _formats = {'LON': '15d', 'LAT': '15d'}


class PTSs(NITFLoop):
    _child_class = PTS
    _count_size = 3


###########
# ACCHZB

class ACHZ(NITFElement):
    __slots__ = ('UNIAAH', 'AAH', 'UNIAPH', 'APH', '_PTSs')
    _formats = {'UNIAAH': '3s', 'AAH': '5d', 'UNIAPH': '3s', 'APH': '5d'}
    _types = {'_PTSs': PTSs}
    _defaults = {'_PTSs': {}}
    _if_skips = {
        'UNIAAH': {'condition': ' == ""', 'vars': ['AAH', ]},
        'UNIAPH': {'condition': ' == ""', 'vars': ['APH', ]}}

    @property
    def PTSs(self):  # type: () -> PTSs
        return self._PTSs

    @PTSs.setter
    def PTSs(self, value):
        # noinspection PyAttributeOutsideInit
        self._PTSs = value


class ACHZs(NITFLoop):
    _child_class = ACHZ
    _count_size = 2


class ACCHZB(TRE):
    __slots__ = ('TAG', '_ACHZs')
    _formats = {'TAG': '6s'}
    _types = {'_ACHZs': ACHZs}
    _defaults = {'_ACHZs': {}, 'TAG': 'ACCHZB'}
    _enums = {'TAG': {'ACCHZB', }}

    @property
    def ACHZs(self):  # type: () -> ACHZs
        return self._ACHZs

    @ACHZs.setter
    def ACHZs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ACHZs = value


############
# ACCPOB

class ACPO(NITFElement):
    __slots__ = (
        'UNIAAH', 'AAH', 'UNIAAV', 'AAV', 'UNIAPH', 'APH', 'UNIAPV', 'APV', '_PTSs')
    _formats = {
        'UNIAAH': '3s', 'AAH': '5d', 'UNIAAV': '3s', 'AAV': '5d', 'UNIAPH': '3s',
        'APH': '5d', 'UNIAPV': '3s', 'APV': '5d'}
    _types = {'_PTSs': PTSs}
    _defaults = {'_PTSs': {}}

    @property
    def PTSs(self):  # type: () -> PTSs
        return self._PTSs

    @PTSs.setter
    def PTSs(self, value):
        # noinspection PyAttributeOutsideInit
        self._PTSs = value


class ACPOs(NITFLoop):
    _child_class = ACPO
    _count_size = 2


class ACCPOB(TRE):
    __slots__ = ('TAG', '_ACPOs')
    _formats = {'TAG': '6s'}
    _types = {'_ACPOs': ACPOs}
    _defaults = {'_ACPOs': {}, 'TAG': 'ACCPOB'}
    _enums = {'TAG': {'ACCPOB', }}

    @property
    def ACPOs(self):  # type: () -> ACPOs
        return self._ACPOs

    @ACPOs.setter
    def ACPOs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ACPOs = value


#############
# ACCVTB

class ACVT(NITFElement):
    __slots__ = ('UNIAAV', 'AAV', 'UNIAPV', 'APV', '_PTSs')
    _formats = {'UNIAAV': '3s', 'AAV': '5d', 'UNIAPV': '3s', 'APV': '5d'}
    _types = {'_PTSs': PTSs}
    _defaults = {'_PTSs': {}}
    _if_skips = {
        'UNIAAV': {'condition': ' == ""', 'vars': ['AAV', ]},
        'UNIAPV': {'condition': ' == ""', 'vars': ['APV', ]}}

    @property
    def PTSs(self):  # type: () -> PTSs
        return self._PTSs

    @PTSs.setter
    def PTSs(self, value):
        # noinspection PyAttributeOutsideInit
        self._PTSs = value


class ACVTs(NITFLoop):
    _child_class = ACVT
    _count_size = 2


class ACCVTB(TRE):
    __slots__ = ('TAG', '_ACVTs')
    _formats = {'TAG': '6s'}
    _types = {'_ACVTs': ACVTs}
    _defaults = {'_ACVTs': {}, 'TAG': 'ACCVTB'}
    _enums = {'TAG': {'ACCVTB', }}

    @property
    def ACVTs(self):  # type: () -> ACVTs
        return self._ACVTs

    @ACVTs.setter
    def ACVTs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ACVTs = value
