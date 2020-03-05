# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class RSMGIA(TRE):
    __slots__ = (
        'TAG', 'IID', 'EDITION', 'GR0', 'GRX', 'GRY', 'GRZ', 'GRXX', 'GRXY', 'GRXZ', 'GRYY', 'GRYZ', 'GRZZ', 'GC0',
        'GCX', 'GCY', 'GCZ', 'GCXX', 'GCXY', 'GCXZ', 'GCYY', 'GCYZ', 'GCZZ', 'GRNIS', 'GCNIS', 'GTNIS', 'GRSSIZ',
        'GCSSIZ')
    _formats = {
        'TAG': '6s', 'IID': '80s', 'EDITION': '40s', 'GR0': '21s', 'GRX': '21s', 'GRY': '21s', 'GRZ': '21s',
        'GRXX': '21s', 'GRXY': '21s', 'GRXZ': '21s', 'GRYY': '21s', 'GRYZ': '21s', 'GRZZ': '21s', 'GC0': '21s',
        'GCX': '21s', 'GCY': '21s', 'GCZ': '21s', 'GCXX': '21s', 'GCXY': '21s', 'GCXZ': '21s', 'GCYY': '21s',
        'GCYZ': '21s', 'GCZZ': '21s', 'GRNIS': '3d', 'GCNIS': '3d', 'GTNIS': '3d', 'GRSSIZ': '21s', 'GCSSIZ': '21s'}
    _defaults = {'TAG': 'RSMGIA'}
    _enums = {'TAG': {'RSMGIA', }}


class RSMIDA(TRE):
    __slots__ = (
        'TAG', 'IID', 'EDITION', 'ISID', 'SID', 'STID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'NRG',
        'NCG', 'TRG', 'TCG', 'GRNDD', 'XUOR', 'YUOR', 'ZUOR', 'XUXR', 'XUYR', 'XUZR', 'YUXR', 'YUYR', 'YUZR', 'ZUXR',
        'ZUYR', 'ZUZR', 'V1X', 'V1Y', 'V1Z', 'V2X', 'V2Y', 'V2Z', 'V3X', 'V3Y', 'V3Z', 'V4X', 'V4Y', 'V4Z', 'V5X',
        'V5Y', 'V5Z', 'V6X', 'V6Y', 'V6Z', 'V7X', 'V7Y', 'V7Z', 'V8X', 'V8Y', 'V8Z', 'GRPX', 'GRPY', 'GRPZ', 'FULLR',
        'FULLC', 'MINR', 'MAXR', 'MINC', 'MAXC', 'IE0', 'IER', 'IEC', 'IERR', 'IERC', 'IECC', 'IA0', 'IAR', 'IAC',
        'IARR', 'IARC', 'IACC', 'SPX', 'SVX', 'SAX', 'SPY', 'SVY', 'SAY', 'SPZ', 'SVZ', 'SAZ')
    _formats = {
        'TAG': '6s', 'IID': '80s', 'EDITION': '40s', 'ISID': '40s', 'SID': '40s', 'STID': '40s', 'YEAR': '4d',
        'MONTH': '2d', 'DAY': '2d', 'HOUR': '2d', 'MINUTE': '2d', 'SECOND': '9d', 'NRG': '8s', 'NCG': '8s',
        'TRG': '21s', 'TCG': '21s', 'GRNDD': '1s', 'XUOR': '21s', 'YUOR': '21s', 'ZUOR': '21s', 'XUXR': '21s',
        'XUYR': '21s', 'XUZR': '21s', 'YUXR': '21s', 'YUYR': '21s', 'YUZR': '21s', 'ZUXR': '21s', 'ZUYR': '21s',
        'ZUZR': '21s', 'V1X': '21s', 'V1Y': '21s', 'V1Z': '21s', 'V2X': '21s', 'V2Y': '21s', 'V2Z': '21s', 'V3X': '21s',
        'V3Y': '21s', 'V3Z': '21s', 'V4X': '21s', 'V4Y': '21s', 'V4Z': '21s', 'V5X': '21s', 'V5Y': '21s', 'V5Z': '21s',
        'V6X': '21s', 'V6Y': '21s', 'V6Z': '21s', 'V7X': '21s', 'V7Y': '21s', 'V7Z': '21s', 'V8X': '21s', 'V8Y': '21s',
        'V8Z': '21s', 'GRPX': '21s', 'GRPY': '21s', 'GRPZ': '21s', 'FULLR': '8s', 'FULLC': '8s', 'MINR': '8s',
        'MAXR': '8s', 'MINC': '8s', 'MAXC': '8s', 'IE0': '21s', 'IER': '21s', 'IEC': '21s', 'IERR': '21s',
        'IERC': '21s', 'IECC': '21s', 'IA0': '21s', 'IAR': '21s', 'IAC': '21s', 'IARR': '21s', 'IARC': '21s',
        'IACC': '21s', 'SPX': '21s', 'SVX': '21s', 'SAX': '21s', 'SPY': '21s', 'SVY': '21s', 'SAY': '21s', 'SPZ': '21s',
        'SVZ': '21s', 'SAZ': '21s'}
    _defaults = {'TAG': 'RSMIDA'}
    _enums = {'TAG': {'RSMIDA', }}


########
# RSMPCA

class RNTRM(NITFElement):
    __slots__ = ('RNPCF',)
    _formats = {'RNPCF': '21s'}


class RNTRMs(NITFLoop):
    _child_class = RNTRM
    _count_size = 3


class RDTRM(NITFElement):
    __slots__ = ('RDPCF',)
    _formats = {'RDPCF': '21s'}


class RDTRMs(NITFLoop):
    _child_class = RDTRM
    _count_size = 3


class CNTRM(NITFElement):
    __slots__ = ('CNPCF',)
    _formats = {'CNPCF': '21s'}


class CNTRMs(NITFLoop):
    _child_class = CNTRM
    _count_size = 3


class CDTRM(NITFElement):
    __slots__ = ('CDPCF',)
    _formats = {'CDPCF': '21s'}


class CDTRMs(NITFLoop):
    _child_class = CDTRM
    _count_size = 3


class RSMPCA(TRE):
    __slots__ = (
        'TAG', 'IID', 'EDITION', 'RSN', 'CSN', 'RFEP', 'CFEP', 'RNRMO', 'CNRMO', 'XNRMO', 'YNRMO', 'ZNRMO', 'RNRMSF',
        'CNRMSF', 'XNRMSF', 'YNRMSF', 'ZNRMSF', 'RNPWRX', 'RNPWRY', 'RNPWRZ', '_RNTRMs', 'RDPWRX', 'RDPWRY', 'RDPWRZ',
        '_RDTRMs', 'CNPWRX', 'CNPWRY', 'CNPWRZ', '_CNTRMs', 'CDPWRX', 'CDPWRY', 'CDPWRZ', '_CDTRMs')
    _formats = {
        'TAG': '6s', 'IID': '80s', 'EDITION': '40s', 'RSN': '3d', 'CSN': '3d', 'RFEP': '21s', 'CFEP': '21s',
        'RNRMO': '21s', 'CNRMO': '21s', 'XNRMO': '21s', 'YNRMO': '21s', 'ZNRMO': '21s', 'RNRMSF': '21s',
        'CNRMSF': '21s', 'XNRMSF': '21s', 'YNRMSF': '21s', 'ZNRMSF': '21s', 'RNPWRX': '1d', 'RNPWRY': '1d',
        'RNPWRZ': '1d', 'RDPWRX': '1d', 'RDPWRY': '1d', 'RDPWRZ': '1d', 'CNPWRX': '1d', 'CNPWRY': '1d', 'CNPWRZ': '1d',
        'CDPWRX': '1d', 'CDPWRY': '1d', 'CDPWRZ': '1d'}
    _types = {'_RNTRMs': RNTRMs, '_RDTRMs': RDTRMs, '_CNTRMs': CNTRMs, '_CDTRMs': CDTRMs}
    _defaults = {'_RNTRMs': {}, '_RDTRMs': {}, '_CNTRMs': {}, '_CDTRMs': {}, 'TAG': 'RSMPCA'}
    _enums = {'TAG': {'RSMPCA', }}

    @property
    def RNTRMs(self):  # type: () -> RNTRMs
        return self._RNTRMs

    @RNTRMs.setter
    def RNTRMs(self, value):
        # noinspection PyAttributeOutsideInit
        self._RNTRMs = value

    @property
    def RDTRMs(self):  # type: () -> RDTRMs
        return self._RDTRMs

    @RDTRMs.setter
    def RDTRMs(self, value):
        # noinspection PyAttributeOutsideInit
        self._RDTRMs = value

    @property
    def CNTRMs(self):  # type: () -> CNTRMs
        return self._CNTRMs

    @CNTRMs.setter
    def CNTRMs(self, value):
        # noinspection PyAttributeOutsideInit
        self._CNTRMs = value

    @property
    def CDTRMs(self):  # type: () -> CDTRMs
        return self._CDTRMs

    @CDTRMs.setter
    def CDTRMs(self, value):
        # noinspection PyAttributeOutsideInit
        self._CDTRMs = value


class RSMPIA(TRE):
    __slots__ = (
        'TAG', 'IID', 'EDITION', 'R0', 'RX', 'RY', 'RZ', 'RXX', 'RXY', 'RXZ', 'RYY', 'RYZ', 'RZZ', 'C0', 'CX', 'CY',
        'CZ', 'CXX', 'CXY', 'CXZ', 'CYY', 'CYZ', 'CZZ', 'RNIS', 'CNIS', 'TNIS', 'RSSIZ', 'CSSIZ')
    _formats = {
        'TAG': '6s', 'IID': '80s', 'EDITION': '40s', 'R0': '21s', 'RX': '21s', 'RY': '21s', 'RZ': '21s', 'RXX': '21s',
        'RXY': '21s', 'RXZ': '21s', 'RYY': '21s', 'RYZ': '21s', 'RZZ': '21s', 'C0': '21s', 'CX': '21s', 'CY': '21s',
        'CZ': '21s', 'CXX': '21s', 'CXY': '21s', 'CXZ': '21s', 'CYY': '21s', 'CYZ': '21s', 'CZZ': '21s', 'RNIS': '3d',
        'CNIS': '3d', 'TNIS': '3d', 'RSSIZ': '21s', 'CSSIZ': '21s'}
    _defaults = {'TAG': 'RSMPIA'}
    _enums = {'TAG': {'RSMPIA', }}
