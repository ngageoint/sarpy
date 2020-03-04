# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PRJ(NITFElement):
    __slots__ = ('PRJ',)
    _formats = {'PRJ': '15d'}


class PRJs(NITFLoop):
    _child_class = PRJ
    _count_size = 1


class PRJPSB(TRE):
    __slots__ = ('TAG', 'PRN', 'PCO', '_PRJs', 'XOR', 'YOR')
    _formats = {'TAG': '6s', 'PRN': '80s', 'PCO': '2s', 'XOR': '15d', 'YOR': '15d'}
    _types = {'_PRJs': PRJs}
    _defaults = {'_PRJs': {}, 'TAG': 'PRJPSB'}
    _enums = {'TAG': {'PRJPSB', }}

    @property
    def PRJs(self):  # type: () -> PRJs
        return self._PRJs

    @PRJs.setter
    def PRJs(self, value):
        # noinspection PyAttributeOutsideInit
        self._PRJs = value
