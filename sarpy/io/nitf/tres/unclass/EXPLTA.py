# -*- coding: utf-8 -*-
"""
Selected simple unclassified NITF file header TRE objects.
"""

from ...headers import TRE, UnknownTRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class EXPLTA(TRE):
    def __init__(self, **kwargs):
        raise ValueError(
            'This is an abstract class, not meant to be implemented directly. Use one of'
            'EXPLTA_87 (approved default) or EXPLTA_101 for a direct implementation')

    @classmethod
    def from_bytes(cls, value, start):
        length = int(value[start+6:start+11])
        if length == 87:
            return EXPLTA_87.from_bytes(value, start)
        elif length == 101:
            return EXPLTA_101.from_bytes(value, start)
        else:
            return UnknownTRE.from_bytes(value, start)


class EXPLTA_87(TRE):
    __slots__ = (
        'TAG', 'ANGLE_TO_NORTH', 'SQUINT_ANGLE', 'MODE', 'RESVD001', 'GRAZE_ANG', 'SLOPE_ANG', 'POLAR', 'NSAMP',
        'RESVD002', 'SEQ_NUM', 'PRIME_ID', 'PRIME_BE', 'RESVD003', 'N_SEC', 'IPR', 'RESVD004', 'RESVD005', 'RESVD006',
        'RESVD007')
    _formats = {
        'TAG': '6s', 'ANGLE_TO_NORTH': '3s', 'SQUINT_ANGLE': '3s', 'MODE': '3s', 'RESVD001': '16s', 'GRAZE_ANG': '2s',
        'SLOPE_ANG': '2s', 'POLAR': '2s', 'NSAMP': '5s', 'RESVD002': '1s', 'SEQ_NUM': '1s', 'PRIME_ID': '12s',
        'PRIME_BE': '15s', 'RESVD003': '1s', 'N_SEC': '2s', 'IPR': '2s', 'RESVD004': '2s', 'RESVD005': '2s',
        'RESVD006': '5s', 'RESVD007': '8s'}
    _defaults = {'TAG': 'EXPLTA'}
    _enums = {'TAG': {'EXPLTA', }}


class EXPLTA_101(TRE):
    __slots__ = (
        'TAG', 'ANGLE_TO_NORTH', 'SQUINT_ANGLE', 'MODE', 'RESVD001', 'GRAZE_ANG', 'SLOPE_ANG', 'POLAR', 'NSAMP',
        'RESVD002', 'SEQ_NUM', 'PRIME_ID', 'PRIME_BE', 'RESVD003', 'N_SEC', 'IPR')
    _formats = {
        'TAG': '6s', 'ANGLE_TO_NORTH': '7s', 'SQUINT_ANGLE': '7s', 'MODE': '3s', 'RESVD001': '16s', 'GRAZE_ANG': '5s',
        'SLOPE_ANG': '5s', 'POLAR': '2s', 'NSAMP': '5s', 'RESVD002': '1s', 'SEQ_NUM': '1s', 'PRIME_ID': '12s',
        'PRIME_BE': '15s', 'RESVD003': '1s', 'N_SEC': '2s', 'IPR': '2s'}
    _defaults = {'TAG': 'EXPLTA'}
    _enums = {'TAG': {'EXPLTA', }}
