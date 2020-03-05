# -*- coding: utf-8 -*-
"""
Selected simple unclassified NITF file header TRE objects.
"""

from ...headers import TRE, UnknownTRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class AIMIDA(TRE):
    def __init__(self, **kwargs):
        raise ValueError(
            'This is an abstract class, not meant to be implemented directly. Use one of'
            'AIMIDA_69 (approved default), AIMIDA_73, or AIMIDA_89 for a direct implementation')

    @classmethod
    def from_bytes(cls, value, start):
        length = int(value[start+6:start+11])
        if length == 69:
            return AIMIDA_69.from_bytes(value, start)
        elif length == 73:
            return AIMIDA_73.from_bytes(value, start)
        elif length == 89:
            return AIMIDA_89.from_bytes(value, start)
        else:
            return UnknownTRE.from_bytes(value, start)


class AIMIDA_69(TRE):
    __slots__ = (
        'TAG', 'MISSION_DATE', 'MISSION_NO', 'FLIGHT_NO', 'OP_NUM', 'START_SEGMENT', 'REPRO_NUM', 'REPLAY', 'RESVD001',
        'START_COLUMN', 'START_ROW', 'END_SEGMENT', 'END_COLUMN', 'END_ROW', 'COUNTRY', 'RESVD002', 'LOCATION', 'TIME',
        'CREATION_DATE')
    _formats = {
        'TAG': '6s', 'MISSION_DATE': '7s', 'MISSION_NO': '4s', 'FLIGHT_NO': '2s', 'OP_NUM': '3s', 'START_SEGMENT': '2s',
        'REPRO_NUM': '2s', 'REPLAY': '3s', 'RESVD001': '1s', 'START_COLUMN': '2s', 'START_ROW': '5s',
        'END_SEGMENT': '2s', 'END_COLUMN': '2s', 'END_ROW': '5s', 'COUNTRY': '2s', 'RESVD002': '4s', 'LOCATION': '11s',
        'TIME': '5s', 'CREATION_DATE': '7s'}
    _defaults = {'TAG': 'AIMIDA'}
    _enums = {'TAG': {'AIMIDA', }}


class AIMIDA_73(TRE):
    __slots__ = (
        'TAG', 'MISSION_DATE', 'MISSION_NO', 'FLIGHT_NO', 'OP_NUM', 'RESVD001', 'REPRO_NUM', 'REPLAY', 'RESVD002',
        'START_COLUMN', 'START_ROW', 'RESVD003', 'END_COLUMN', 'END_ROW', 'COUNTRY', 'RESVD004', 'LOCATION', 'TIME',
        'CREATION_DATE')
    _formats = {
        'TAG': '6s', 'MISSION_DATE': '8s', 'MISSION_NO': '4s', 'FLIGHT_NO': '2s', 'OP_NUM': '3s', 'RESVD001': '2s',
        'REPRO_NUM': '2s', 'REPLAY': '3s', 'RESVD002': '1s', 'START_COLUMN': '3s', 'START_ROW': '5s', 'RESVD003': '2s',
        'END_COLUMN': '3s', 'END_ROW': '5s', 'COUNTRY': '2s', 'RESVD004': '4s', 'LOCATION': '11s', 'TIME': '5s',
        'CREATION_DATE': '8s'}
    _defaults = {'TAG': 'AIMIDA'}
    _enums = {'TAG': {'AIMIDA', }}


class AIMIDA_89(TRE):
    __slots__ = (
        'TAG', 'MISSION_DATE_TIME', 'MISSION_NO', 'MISSION_ID', 'FLIGHT_NO', 'OP_NUM', 'CURRENT_SEGMENT', 'REPRO_NUM',
        'REPLAY', 'RESVD001', 'START_COLUMN', 'START_ROW', 'END_SEGMENT', 'END_COLUMN', 'END_ROW', 'COUNTRY',
        'RESVD002', 'LOCATION', 'RESVD003')
    _formats = {
        'TAG': '6s', 'MISSION_DATE_TIME': '14s', 'MISSION_NO': '4s', 'MISSION_ID': '10s', 'FLIGHT_NO': '2s',
        'OP_NUM': '3d', 'CURRENT_SEGMENT': '2s', 'REPRO_NUM': '2d', 'REPLAY': '3s', 'RESVD001': '1s',
        'START_COLUMN': '3d', 'START_ROW': '5d', 'END_SEGMENT': '2s', 'END_COLUMN': '3d', 'END_ROW': '5d',
        'COUNTRY': '2s', 'RESVD002': '4s', 'LOCATION': '11s', 'RESVD003': '13s'}
    _defaults = {'TAG': 'AIMIDA'}
    _enums = {'TAG': {'AIMIDA', }}
