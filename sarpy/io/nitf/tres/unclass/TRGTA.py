# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class TGT_QC(NITFElement):
    __slots__ = ('TGT_QCOMMENT',)
    _formats = {'TGT_QCOMMENT': '40s'}


class TGT_QCs(NITFLoop):
    _child_class = TGT_QC
    _count_size = 1


class TGT_CC(NITFElement):
    __slots__ = ('TGT_CCOMMENT',)
    _formats = {'TGT_CCOMMENT': '40s'}


class TGT_CCs(NITFLoop):
    _child_class = TGT_CC
    _count_size = 1


class REF_PT(NITFElement):
    __slots__ = (
        'TGT_REF', 'TGT_LL', 'TGT_ELEV', 'TGT_BAND', 'TGT_ROW', 'TGT_COL', 'TGT_PROW', 'TGT_PCOL')
    _formats = {
        'TGT_REF': '10s', 'TGT_LL': '21s', 'TGT_ELEV': '8s', 'TGT_BAND': '3s', 'TGT_ROW': '8d',
        'TGT_COL': '8d', 'TGT_PROW': '8d', 'TGT_PCOL': '8d'}


class REF_PTs(NITFLoop):
    _child_class = REF_PT
    _count_size = 1


class SCENE_TGTS(NITFElement):
    __slots__ = (
        'TGT_NAME', 'TGT_TYPE', 'TGT_VER', 'TGT_CAT', 'TGT_BE', 'TGT_SN', 'TGT_POSNUM', 'TGT_ATTITUDE_PITCH',
        'TGT_ATTITUDE_ROLL', 'TGT_ATTITUDE_YAW', 'TGT_DIM_LENGTH', 'TGT_DIM_WIDTH', 'TGT_DIM_HEIGHT',
        'TGT_AZIMUTH', 'TGT_CLTR_RATIO', 'TGT_STATE', 'TGT_COND', 'TGT_OBSCR', 'TGT_OBSCR_PCT', 'TGT_CAMO',
        'TGT_CAMO_PCT', 'TGT_UNDER', 'TGT_OVER', 'TGT_TTEXTURE', 'TGT_PAINT', 'TGT_SPEED', 'TGT_HEADING',
        '_TGT_QCs', '_TGT_CCs', '_REF_PTs')
    _formats = {
        'TGT_NAME': '25s', 'TGT_TYPE': '15s', 'TGT_VER': '6s', 'TGT_CAT': '5s', 'TGT_BE': '17s',
        'TGT_SN': '10s', 'TGT_POSNUM': '2s', 'TGT_ATTITUDE_PITCH': '6s', 'TGT_ATTITUDE_ROLL': '6s',
        'TGT_ATTITUDE_YAW': '6s', 'TGT_DIM_LENGTH': '5s', 'TGT_DIM_WIDTH': '5s', 'TGT_DIM_HEIGHT': '5s',
        'TGT_AZIMUTH': '6s', 'TGT_CLTR_RATIO': '8s', 'TGT_STATE': '10s', 'TGT_COND': '30s', 'TGT_OBSCR': '20s',
        'TGT_OBSCR_PCT': '3s', 'TGT_CAMO': '20s', 'TGT_CAMO_PCT': '3s', 'TGT_UNDER': '12s', 'TGT_OVER': '30s',
        'TGT_TTEXTURE': '45s', 'TGT_PAINT': '40s', 'TGT_SPEED': '3s', 'TGT_HEADING': '3s'}
    _types = {'_TGT_QCs': TGT_QCs, '_TGT_CCs': TGT_CCs, '_REF_PTs': REF_PTs}
    _defaults = {'_TGT_QCs': {}, '_TGT_CCs': {}, '_REF_PTs': {}}

    @property
    def TGT_QCs(self):  # type: () -> TGT_QCs
        return self._TGT_QCs

    @TGT_QCs.setter
    def TGT_QCs(self, value):
        # noinspection PyAttributeOutsideInit
        self._TGT_QCs = value

    @property
    def TGT_CCs(self):  # type: () -> TGT_CCs
        return self._TGT_CCs

    @TGT_CCs.setter
    def TGT_CCs(self, value):
        # noinspection PyAttributeOutsideInit
        self._TGT_CCs = value

    @property
    def REF_PTs(self):  # type: () -> REF_PTs
        return self._REF_PTs

    @REF_PTs.setter
    def REF_PTs(self, value):
        # noinspection PyAttributeOutsideInit
        self._REF_PTs = value


class SCENE_TGTSs(NITFLoop):
    _child_class = SCENE_TGTS
    _count_size = 3


class ATTRIBUTES(NITFElement):
    __slots__ = ('ATTR_TGT_NUM', 'ATTR_NAME', 'ATTR_CONDTN', 'ATTR_VALUE')
    _formats = {'ATTR_TGT_NUM': '3d', 'ATTR_NAME': '30s', 'ATTR_CONDTN': '35s', 'ATTR_VALUE': '10s'}


class ATTRIBUTESs(NITFLoop):
    _child_class = ATTRIBUTES
    _count_size = 3


class TRGTA(TRE):
    __slots__ = ('TAG', 'VERNUM', 'NO_VALID_TGTS', '_SCENE_TGTSs', '_ATTRIBUTESs')
    _formats = {'TAG': '6s', 'VERNUM': '4d', 'NO_VALID_TGTS': '3d'}
    _types = {'_SCENE_TGTSs': SCENE_TGTSs, '_ATTRIBUTESs': ATTRIBUTESs}
    _defaults = {'_SCENE_TGTSs': {}, '_ATTRIBUTESs': {}, 'TAG': 'TRGTA'}
    _enums = {'TAG': {'TRGTA', }}

    @property
    def SCENE_TGTSs(self):  # type: () -> SCENE_TGTSs
        return self._SCENE_TGTSs

    @SCENE_TGTSs.setter
    def SCENE_TGTSs(self, value):
        # noinspection PyAttributeOutsideInit
        self._SCENE_TGTSs = value

    @property
    def ATTRIBUTESs(self):  # type: () -> ATTRIBUTESs
        return self._ATTRIBUTESs

    @ATTRIBUTESs.setter
    def ATTRIBUTESs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ATTRIBUTESs = value
