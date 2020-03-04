# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class IPCOM(NITFElement):
    __slots__ = ('IPCOM',)
    _formats = {'IPCOM': '80s'}


class IPCOMs(NITFLoop):
    _child_class = IPCOM
    _count_size = 1


class EVENT(NITFElement):
    __slots__ = (
        'PDATE', 'PSITE', 'PAS', '_IPCOMs', 'IBPP', 'IPVTYPE', 'INBWC', 'DISP_FLAG', 'ROT_FLAG', 'ROT_ANGLE',
        'ASYM_FLAG', 'ZOOMROW', 'ZOOMCOL', 'PROJ_FLAG', 'SHARP_FLAG', 'SHARPFAM', 'SHARPMEM', 'MAG_FLAG',
        'MAG_LEVEL', 'DRA_FLAG', 'DRA_MULT', 'DRA_SUB', 'TTC_FLAG', 'TTCFAM', 'TTCMEM', 'DEVLUT_FLAG', 'OBPP',
        'OPVTYPE', 'OUTBWC')
    _formats = {
        'PDATE': '14s', 'PSITE': '10s', 'PAS': '10s', 'IBPP': '2d', 'IPVTYPE': '3s', 'INBWC': '10s',
        'DISP_FLAG': '1s', 'ROT_FLAG': '1d', 'ROT_ANGLE': '8s', 'ASYM_FLAG': '1s', 'ZOOMROW': '7s',
        'ZOOMCOL': '7s', 'PROJ_FLAG': '1s', 'SHARP_FLAG': '1d', 'SHARPFAM': '2d', 'SHARPMEM': '2d',
        'MAG_FLAG': '1d', 'MAG_LEVEL': '7s', 'DRA_FLAG': '1d', 'DRA_MULT': '7s', 'DRA_SUB': '5d',
        'TTC_FLAG': '1d', 'TTCFAM': '2d', 'TTCMEM': '2d', 'DEVLUT_FLAG': '1d', 'OBPP': '2d', 'OPVTYPE': '3s',
        'OUTBWC': '10s'}
    _types = {'_IPCOMs': IPCOMs}
    _defaults = {'_IPCOMs': {}}
    _if_skips = {
        'ROT_FLAG': {'condition': '!= 1', 'vars': ['ROT_ANGLE', ]},
        'ASYM_FLAG': {'condition': '!= 1', 'vars': ['ZOOMROW', 'ZOOMCOL']},
        'SHARP_FLAG': {'condition': '!= 1', 'vars': ['SHARPFAM', 'SHARPMEM']},
        'MAG_FLAG': {'condition': '!= 1', 'vars': ['MAG_LEVEL', ]},
        'DRA_FLAG': {'condition': '!= 1', 'vars': ['DRA_MULT', 'DRA_SUB']},
        'TTC_FLAG': {'condition': '!= 1', 'vars': ['TTCFAM', 'TTCMEM']},
    }

    @property
    def IPCOMs(self):  # type: () -> IPCOMs
        return self._IPCOMs

    @IPCOMs.setter
    def IPCOMs(self, value):
        # noinspection PyAttributeOutsideInit
        self._IPCOMs = value


class EVENTs(NITFLoop):
    _child_class = EVENT
    _count_size = 2


class HISTOA(TRE):
    __slots__ = ('TAG', 'SYSTYPE', 'PC', 'PE', 'REMAP_FLAG', 'LUTID', '_EVENTs')
    _formats = {'TAG': '6s', 'SYSTYPE': '20s', 'PC': '12s', 'PE': '4s', 'REMAP_FLAG': '1s', 'LUTID': '2d'}
    _types = {'_EVENTs': EVENTs}
    _defaults = {'_EVENTs': {}, 'TAG': 'HISTOA'}
    _enums = {'TAG': {'HISTOA', }}

    @property
    def EVENTs(self):  # type: () -> EVENTs
        return self._EVENTs

    @EVENTs.setter
    def EVENTs(self, value):
        # noinspection PyAttributeOutsideInit
        self._EVENTs = value
