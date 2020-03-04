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


class BP(NITFElement):
    __slots__ = ('_PTSs',)
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


class MI(NITFElement):
    __slots__ = (
        'CDV30', 'UNIRAT', 'RAT', 'UNIGMA', 'GMA', 'LONGMA', 'LATGMA', 'UNIGCA', 'GCA')
    _formats = {
        'CDV30': '8s', 'UNIRAT': '3s', 'RAT': '8d', 'UNIGMA': '3s', 'GMA': '8d', 'LONGMA': '15d',
        'LATGMA': '15d', 'UNIGCA': '3s', 'GCA': '8d'}


class MIs(NITFLoop):
    _child_class = MI
    _count_size = 2


class LI(NITFElement):
    __slots__ = ('BAD',)
    _formats = {'BAD': '10s'}


class LIs(NITFLoop):
    _child_class = LI
    _count_size = 2


class PRJ(NITFElement):
    __slots__ = ('PRJ',)
    _formats = {'PRJ': '15d'}


class PRJs(NITFLoop):
    _child_class = PRJ
    _count_size = 1


class IN(NITFElement):
    __slots__ = (
        'INT', 'INS_SCA', 'NTL', 'TTL', 'NVL', 'TVL', 'NTR', 'TTR', 'NVR', 'TVR', 'NRL', 'TRL', 'NSL',
        'TSL', 'NRR', 'TRR', 'NSR', 'TSR')
    _formats = {
        'INT': '10s', 'INS_SCA': '9d', 'NTL': '15d', 'TTL': '15d', 'NVL': '15d', 'TVL': '15d',
        'NTR': '15d', 'TTR': '15d', 'NVR': '15d', 'TVR': '15d', 'NRL': '15d', 'TRL': '15d',
        'NSL': '15d', 'TSL': '15d', 'NRR': '15d', 'TRR': '15d', 'NSR': '15d', 'TSR': '15d'}


class INs(NITFLoop):
    _child_class = IN
    _count_size = 2


class SOUR(NITFElement):
    __slots__ = (
        '_BPs', 'PRT', 'URF', 'EDN', 'NAM', 'CDP', 'CDV', 'CDV27', 'SRN', 'SCA', 'UNISQU', 'SQU', 'UNIPCI',
        'PCI', 'WPC', 'NST', 'UNIHKE', 'HKE', 'LONHKE', 'LATHKE', 'QSS', 'QOD', 'CDV10', 'QLE', 'CPY', '_MIs',
        '_LIs', 'DAG', 'DCD', 'ELL', 'ELC', 'DVR', 'VDCDVR', 'SDA', 'VDCSDA', 'PRN', 'PCO', '_PRJs', 'XOR',
        'YOR', 'GRD', 'GRN', 'ZNA', '_INs')
    _formats = {
        'PRT': '10s', 'URF': '20s', 'EDN': '7s', 'NAM': '20s', 'CDP': '3d', 'CDV': '8s', 'CDV27': '8s',
        'SRN': '80s', 'SCA': '9s', 'UNISQU': '3s', 'SQU': '10d', 'UNIPCI': '3s', 'PCI': '4d', 'WPC': '3d',
        'NST': '3d', 'UNIHKE': '3s', 'HKE': '6d', 'LONHKE': '15d', 'LATHKE': '15d', 'QSS': '1s', 'QOD': '1s',
        'CDV10': '8s', 'QLE': '80s', 'CPY': '80s', 'DAG': '80s', 'DCD': '4s', 'ELL': '80s', 'ELC': '3s',
        'DVR': '80s', 'VDCDVR': '4s', 'SDA': '80s', 'VDCSDA': '4s', 'PRN': '80s', 'PCO': '2s', 'XOR': '15d',
        'YOR': '15d', 'GRD': '3s', 'GRN': '80s', 'ZNA': '4d'}
    _types = {'_BPs': BPs, '_MIs': MIs, '_LIs': LIs, '_PRJs': PRJs, '_INs': INs}
    _defaults = {'_BPs': {}, '_MIs': {}, '_LIs': {}, '_PRJs': {}, '_INs': {}}

    @property
    def BPs(self):  # type: () -> BPs
        return self._BPs

    @BPs.setter
    def BPs(self, value):
        # noinspection PyAttributeOutsideInit
        self._BPs = value

    @property
    def MIs(self):  # type: () -> MIs
        return self._MIs

    @MIs.setter
    def MIs(self, value):
        # noinspection PyAttributeOutsideInit
        self._MIs = value

    @property
    def LIs(self):  # type: () -> LIs
        return self._LIs

    @LIs.setter
    def LIs(self, value):
        # noinspection PyAttributeOutsideInit
        self._LIs = value

    @property
    def PRJs(self):  # type: () -> PRJs
        return self._PRJs

    @PRJs.setter
    def PRJs(self, value):
        # noinspection PyAttributeOutsideInit
        self._PRJs = value

    @property
    def INs(self):  # type: () -> INs
        return self._INs

    @INs.setter
    def INs(self, value):
        # noinspection PyAttributeOutsideInit
        self._INs = value


class SOURs(NITFLoop):
    _child_class = SOUR
    _count_size = 2


class SOURCB(TRE):
    __slots__ = ('TAG', 'IS_SCA', 'CPATCH', '_SOURs')
    _formats = {'TAG': '6s', 'IS_SCA': '9d', 'CPATCH': '10s'}
    _types = {'_SOURs': SOURs}
    _defaults = {'_SOURs': {}, 'TAG': 'SOURCB'}
    _enums = {'TAG': {'SOURCB', }}

    @property
    def SOURs(self):  # type: () -> SOURs
        return self._SOURs

    @SOURs.setter
    def SOURs(self, value):
        # noinspection PyAttributeOutsideInit
        self._SOURs = value
