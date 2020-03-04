# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PTS(NITFElement):
    __slots__ = ('LON', 'LAT')
    _formats = {'LON': '15d', 'LAT': '15d'}


class PTSs(NITFLoop):
    _child_class = PTS
    _count_size = 2


class BP(NITFElement):
    __slots__ = ('_PTSs', )
    _types = {'_PTSs': PTSs}
    _defaults = {'_PTSs': {}}

    @property
    def PTSs(self):  # type: () -> PTSs
        return self._PTSs

    @PTSs.setter
    def PTSs(self, value):
        # noinspection PyAttributeOutsideInit
        self._PTSs = value


class BPs(NITFLoop):
    _child_class = BP
    _count_size = 2


class BND(NITFElement):
    __slots__ = ('BID', 'WS1', 'WS2')
    _formats = {'BID': '5s', 'WS1': '5d', 'WS2': '5d'}


class BNDs(NITFLoop):
    _child_class = BND
    _count_size = 2


class AUX(NITFElement):
    __slots__ = ('API', 'APF', 'UNIAPX', 'APN', 'APR', 'APA')
    _formats = {
        'API': '20s', 'APF': '1s', 'UNIAPX': '7s', 'APN': '10d', 'APR': '20d', 'APA': '20s'}


class AUXs(NITFLoop):
    _child_class = AUX
    _count_size = 3


class SNS(NITFElement):
    __slots__ = (
        '_BPs', '_BNDs', 'UNIRES', 'REX', 'REY', 'GSX', 'GSY', 'GSL', 'PLTFM', 'INS', 'MOD', 'PRL', 'ACT',
        'UNINOA', 'NOA', 'UNIANG', 'ANG', 'UNIALT', 'ALT', 'LONSCC', 'LATSCC', 'UNISAE', 'SAZ', 'SEL', 'UNIRPY',
        'ROL', 'PIT', 'YAW', 'UNIPXT', 'PIXT', 'UNISPE', 'ROS', 'PIS', 'YAS', '_AUXs')
    _formats = {
        'UNIRES': '3s', 'REX': '6d', 'REY': '6d', 'GSX': '6d', 'GSY': '6d', 'GSL': '12s', 'PLTFM': '8s',
        'INS': '8s', 'MOD': '4s', 'PRL': '5s', 'ACT': '18s', 'UNINOA': '3s', 'NOA': '7d', 'UNIANG': '3s',
        'ANG': '7d', 'UNIALT': '3s', 'ALT': '9d', 'LONSCC': '10d', 'LATSCC': '10d', 'UNISAE': '3s', 'SAZ': '7d',
        'SEL': '7d', 'UNIRPY': '3s', 'ROL': '7d', 'PIT': '7d', 'YAW': '7d', 'UNIPXT': '3s', 'PIXT': '14d',
        'UNISPE': '7s', 'ROS': '22d', 'PIS': '22d', 'YAS': '22d'}
    _types = {'_BPs': BPs, '_BNDs': BNDs, '_AUXs': AUXs}
    _defaults = {'_BPs': {}, '_BNDs': {}, '_AUXs': {}}

    @property
    def BPs(self):  # type: () -> BPs
        return self._BPs

    @BPs.setter
    def BPs(self, value):
        # noinspection PyAttributeOutsideInit
        self._BPs = value

    @property
    def BNDs(self):  # type: () -> BNDs
        return self._BNDs

    @BNDs.setter
    def BNDs(self, value):
        # noinspection PyAttributeOutsideInit
        self._BNDs = value

    @property
    def AUXs(self):  # type: () -> AUXs
        return self._AUXs

    @AUXs.setter
    def AUXs(self, value):
        # noinspection PyAttributeOutsideInit
        self._AUXs = value


class SNSs(NITFLoop):
    _child_class = SNS
    _count_size = 2


class SNSPSB(TRE):
    __slots__ = ('TAG', '_SNSs')
    _formats = {'TAG': '6s'}
    _types = {'_SNSs': SNSs}
    _defaults = {'_SNSs': {}, 'TAG': 'SNSPSB'}
    _enums = {'TAG': {'SNSPSB', }}

    @property
    def SNSs(self):  # type: () -> SNSs
        return self._SNSs

    @SNSs.setter
    def SNSs(self, value):
        # noinspection PyAttributeOutsideInit
        self._SNSs = value
