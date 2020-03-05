# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class GRD(NITFElement):
    __slots__ = ('ZVL', 'BAD', 'LOD', 'LAD', 'LSO', 'PSO')
    _formats = {'ZVL': '10d', 'BAD': '10s', 'LOD': '12d', 'LAD': '12d', 'LSO': '11d', 'PSO': '11d'}


class GRDs(NITFLoop):
    _child_class = GRD
    _count_size = 2


class GRDPSB(TRE):
    __slots__ = ('TAG', '_GRDs')
    _formats = {'TAG': '6s'}
    _types = {'_GRDs': GRDs}
    _defaults = {'_GRDs': {}, 'TAG': 'GRDPSB'}
    _enums = {'TAG': {'GRDPSB', }}

    @property
    def GRDs(self):  # type: () -> GRDs
        return self._GRDs

    @GRDs.setter
    def GRDs(self, value):
        # noinspection PyAttributeOutsideInit
        self._GRDs = value
