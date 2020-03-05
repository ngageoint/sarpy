# -*- coding: utf-8 -*-
"""
Selected simple unclassified NITF file header TRE objects.
"""

from ...headers import TRE, UnknownTRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class AFTCA(TRE):
    def __init__(self, **kwargs):
        raise ValueError(
            'This is an abstract class, not meant to be implemented directly. Use one of'
            'AFTCA_132 (approved default) or AFTCA_154 for a direct implementation')

    @classmethod
    def from_bytes(cls, value, start):
        length = int(value[start+6:start+11])
        if length == 132:
            return AFTCA_132.from_bytes(value, start)
        elif length == 154:
            return AFTCA_154.from_bytes(value, start)
        else:
            return UnknownTRE.from_bytes(value, start)


class AFTCA_132(TRE):
    __slots__ = (
        'TAG', 'AC_MSN_ID', 'SCTYPE', 'SCNUM', 'SENSOR_ID', 'PATCH_TOT', 'MTI_TOT', 'PDATE', 'IMHOSTNO', 'IMREQID',
        'SCENE_SOURCE', 'MPLAN', 'ENTLOC', 'ENTALT', 'EXITLOC', 'EXITALT', 'TMAP', 'RCS', 'ROW_SPACING', 'COL_SPACING',
        'SENSERIAL', 'ABSWVER')
    _formats = {
        'TAG': '6s', 'AC_MSN_ID': '10s', 'SCTYPE': '1s', 'SCNUM': '4s', 'SENSOR_ID': '3s', 'PATCH_TOT': '4s',
        'MTI_TOT': '3s', 'PDATE': '7s', 'IMHOSTNO': '3s', 'IMREQID': '5s', 'SCENE_SOURCE': '1s', 'MPLAN': '2s',
        'ENTLOC': '21s', 'ENTALT': '6s', 'EXITLOC': '21s', 'EXITALT': '6s', 'TMAP': '7s', 'RCS': '3s',
        'ROW_SPACING': '7s', 'COL_SPACING': '7s', 'SENSERIAL': '4s', 'ABSWVER': '7s'}
    _defaults = {'TAG': 'AFTCA'}
    _enums = {'TAG': {'AFTCA', }}


class AFTCA_154(TRE):
    __slots__ = (
        'TAG', 'AC_MSN_ID', 'AC_TAIL_NO', 'SENSOR_ID', 'SCENE_SOURCE', 'SCNUM', 'PDATE', 'IMHOSTNO', 'IMREQID', 'MPLAN',
        'ENTLOC', 'ENTALT', 'EXITLOC', 'EXITALT', 'TMAP', 'ROW_SPACING', 'COL_SPACING', 'SENSERIAL', 'ABSWVER',
        'PATCH_TOT', 'MTI_TOT')
    _formats = {
        'TAG': '6s', 'AC_MSN_ID': '10s', 'AC_TAIL_NO': '10s', 'SENSOR_ID': '10s', 'SCENE_SOURCE': '1s', 'SCNUM': '6s',
        'PDATE': '8s', 'IMHOSTNO': '6s', 'IMREQID': '5s', 'MPLAN': '3s', 'ENTLOC': '21s', 'ENTALT': '6s',
        'EXITLOC': '21s', 'EXITALT': '6s', 'TMAP': '7s', 'ROW_SPACING': '7s', 'COL_SPACING': '7s', 'SENSERIAL': '6s',
        'ABSWVER': '7s', 'PATCH_TOT': '4s', 'MTI_TOT': '3s'}
    _defaults = {'TAG': 'AFTCA'}
    _enums = {'TAG': {'AFTCA', }}
