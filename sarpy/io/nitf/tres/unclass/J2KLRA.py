# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class LAYER(NITFElement):
    __slots__ = ('LAYER_ID', 'BITRATE')
    _formats = {'LAYER_ID': '3d', 'BITRATE': '9s'}


class LAYERs(NITFLoop):
    _child_class = LAYER
    _count_size = 3


class J2KLRA(TRE):
    __slots__ = (
        'TAG', 'ORIG', 'NLEVELS_O', 'NBANDS_O', '_LAYERs', 'NLEVELS_I', 'NBANDS_I', 'NLAYERS_I')
    _formats = {
        'TAG': '6s', 'ORIG': '1d', 'NLEVELS_O': '2d', 'NBANDS_O': '5d', 'NLEVELS_I': '2d', 'NBANDS_I': '5d',
        'NLAYERS_I': '3d'}
    _types = {'_LAYERs': LAYERs}
    _defaults = {'_LAYERSs': {}, 'TAG': 'J2KLRA'}
    _enums = {'TAG': {'J2KLRA', }}
    _if_skips = {'ORIG': {'condition': 'not in [1, 3, 9]', 'vars': ['NLEVELS_I', 'NBANDS_I', 'NLAYERS_I']}}

    @property
    def LAYERs(self):  # type: () -> LAYERs
        return self._LAYERs

    @LAYERs.setter
    def LAYERs(self, value):
        # noinspection PyAttributeOutsideInit
        self._LAYERs = value
