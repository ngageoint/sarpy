# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class EPHEM(NITFElement):
    __slots__ = ('EPHEM_X', 'EPHEM_Y', 'EPHEM_Z')
    _formats = {'EPHEM_X': '12d', 'EPHEM_Y': '12d', 'EPHEM_Z': '12d'}


class EPHEMs(NITFLoop):
    _child_class = EPHEM
    _count_size = 3


class CSEPHA(TRE):
    __slots__ = ('TAG', 'EPHEM_FLAG', 'DT_EPHEM', 'DATE_EPHEM', 'T0_EPHEM', '_EPHEMs')
    _formats = {'TAG': '6s', 'EPHEM_FLAG': '12s', 'DT_EPHEM': '5d', 'DATE_EPHEM': '8d', 'T0_EPHEM': '13d'}
    _types = {'_EPHEMs': EPHEMs}
    _defaults = {'_EPHEMs': {}, 'TAG': 'CSEPHA'}
    _enums = {'TAG': {'CSEPHA', }}

    @property
    def EPHEMs(self):  # type: () -> EPHEMs
        return self._EPHEMs

    @EPHEMs.setter
    def EPHEMs(self, value):
        # noinspection PyAttributeOutsideInit
        self._EPHEMs = value
