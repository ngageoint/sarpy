# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class VTGT(NITFElement):
    __slots__ = ('TGLOC', 'TGRDV', 'TGGSP', 'TGHEA', 'TGSIG', 'TGCAT')
    _formats = {'TGLOC': '21s', 'TGRDV': '4s', 'TGGSP': '3s', 'TGHEA': '3s', 'TGSIG': '2s', 'TGCAT': '1s'}


class VTGTs(NITFLoop):
    _child_class = VTGT
    _count_size = 3


class MTIRPA(TRE):
    __slots__ = (
        'TAG', 'DESTP', 'MTPID', 'PCHNO', 'WAMFN', 'WAMBN', 'UTC', 'SQNTA', 'COSGZ', '_VTGTs')
    _formats = {
        'TAG': '6s', 'DESTP': '2s', 'MTPID': '3s', 'PCHNO': '4s', 'WAMFN': '5s', 'WAMBN': '1s', 'UTC': '8s',
        'SQNTA': '5s', 'COSGZ': '7s'}
    _types = {'_VTGTs': VTGTs}
    _defaults = {'_VTGTs': {}, 'TAG': 'MTIRPA'}
    _enums = {'TAG': {'MTIRPA', }}

    @property
    def VTGTs(self):  # type: () -> VTGTs
        return self._VTGTs

    @VTGTs.setter
    def VTGTs(self, value):
        # noinspection PyAttributeOutsideInit
        self._VTGTs = value
