# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PIAEQA(TRE):
    __slots__ = (
        'TAG', 'EQPCODE', 'EQPNOMEN', 'EQPMAN', 'OBTYPE', 'ORDBAT', 'CTRYPROD', 'CTRYDSN', 'OBJVIEW')
    _formats = {
        'TAG': '6s', 'EQPCODE': '7s', 'EQPNOMEN': '45s', 'EQPMAN': '64s', 'OBTYPE': '1s', 'ORDBAT': '3s',
        'CTRYPROD': '2s', 'CTRYDSN': '2s', 'OBJVIEW': '6s'}
    _defaults = {'TAG': 'PIAEQA'}
    _enums = {'TAG': {'PIAEQA', }}


class PIAEVA(TRE):
    __slots__ = ('TAG', 'EVENTNAME', 'EVENTTYPE')
    _formats = {'TAG': '6s', 'EVENTNAME': '38s', 'EVENTTYPE': '8s'}
    _defaults = {'TAG': 'PIAEVA'}
    _enums = {'TAG': {'PIAEVA', }}


class PIAIMB(TRE):
    __slots__ = (
        'TAG', 'CLOUD', 'STDRD', 'SMODE', 'SNAME', 'SRCE', 'CMGEN', 'SQUAL', 'MISNM', 'CSPEC', 'PJTID', 'GENER',
        'EXPLS', 'OTHRC')
    _formats = {
        'TAG': '6s', 'CLOUD': '3s', 'STDRD': '1s', 'SMODE': '12s', 'SNAME': '18s', 'SRCE': '255s', 'CMGEN': '2s',
        'SQUAL': '1s', 'MISNM': '7s', 'CSPEC': '32s', 'PJTID': '2s', 'GENER': '1s', 'EXPLS': '1s', 'OTHRC': '2s'}
    _defaults = {'TAG': 'PIAIMB'}
    _enums = {'TAG': {'PIAIMB', }}


class PIAIMC(TRE):
    __slots__ = (
        'TAG', 'CLOUDCVR', 'SRP', 'SENSMODE', 'SENSNAME', 'SOURCE', 'COMGEN', 'SUBQUAL', 'PIAMSNNUM', 'CAMSPECS',
        'PROJID', 'GENERATION', 'ESD', 'OTHERCOND', 'MEANGSD', 'IDATUM', 'IELLIP', 'PREPROC', 'IPROJ', 'SATTRACK')
    _formats = {
        'TAG': '6s', 'CLOUDCVR': '3d', 'SRP': '1s', 'SENSMODE': '12s', 'SENSNAME': '18s', 'SOURCE': '255s',
        'COMGEN': '2d', 'SUBQUAL': '1s', 'PIAMSNNUM': '7s', 'CAMSPECS': '32s', 'PROJID': '2s', 'GENERATION': '1d',
        'ESD': '1s', 'OTHERCOND': '2s', 'MEANGSD': '7d', 'IDATUM': '3s', 'IELLIP': '3s', 'PREPROC': '2s', 'IPROJ': '2s',
        'SATTRACK': '8d'}
    _defaults = {'TAG': 'PIAIMC'}
    _enums = {'TAG': {'PIAIMC', }}


class PIAPEA(TRE):
    __slots__ = ('TAG', 'LASTNME', 'FIRSTNME', 'MIDNME', 'DOB', 'ASSOCTRY')
    _formats = {'TAG': '6s', 'LASTNME': '28s', 'FIRSTNME': '28s', 'MIDNME': '28s', 'DOB': '6s', 'ASSOCTRY': '2s'}
    _defaults = {'TAG': 'PIAPEA'}
    _enums = {'TAG': {'PIAPEA', }}


class PIAPEB(TRE):
    __slots__ = ('TAG', 'LASTNME', 'FIRSTNME', 'MIDNME', 'DOB', 'ASSOCTRY')
    _formats = {'TAG': '6s', 'LASTNME': '28s', 'FIRSTNME': '28s', 'MIDNME': '28s', 'DOB': '8s', 'ASSOCTRY': '2s'}
    _defaults = {'TAG': 'PIAPEB'}
    _enums = {'TAG': {'PIAPEB', }}


##############
# PIAPRC

class ST(NITFElement):
    __slots__ = ('SECTITLE',)
    _formats = {'SECTITLE': '48s'}


class STs(NITFLoop):
    _child_class = ST
    _count_size = 2


class RO(NITFElement):
    __slots__ = ('REQORG',)
    _formats = {'REQORG': '64s'}


class ROs(NITFLoop):
    _child_class = RO
    _count_size = 2


class KW(NITFElement):
    __slots__ = ('KEYWORD',)
    _formats = {'KEYWORD': '255s'}


class KWs(NITFLoop):
    _child_class = KW
    _count_size = 2


class AR(NITFElement):
    __slots__ = ('ASSRPT',)
    _formats = {'ASSRPT': '20s'}


class ARs(NITFLoop):
    _child_class = AR
    _count_size = 2


class AT(NITFElement):
    __slots__ = ('ATEXT',)
    _formats = {'ATEXT': '255s'}


class ATs(NITFLoop):
    _child_class = AT
    _count_size = 2


class PIAPRC(TRE):
    __slots__ = (
        'TAG', 'ACCID', 'FMCTL', 'SDET', 'PCODE', 'PSUBE', 'PIDNM', 'PNAME', 'MAKER', 'CTIME', 'MAPID', '_STs', '_ROs',
        '_KWs', '_ARs', '_ATs')
    _formats = {
        'TAG': '6s', 'ACCID': '64s', 'FMCTL': '32s', 'SDET': '1s', 'PCODE': '2s', 'PSUBE': '6s', 'PIDNM': '20s',
        'PNAME': '10s', 'MAKER': '2s', 'CTIME': '14s', 'MAPID': '40s'}
    _types = {'_STs': STs, '_ROs': ROs, '_KWs': KWs, '_ARs': ARs, '_ATs': ATs}
    _defaults = {'_STs': {}, '_ROs': {}, '_KWs': {}, '_ARs': {}, '_ATs': {}, 'TAG': 'PIAPRC'}
    _enums = {'TAG': {'PIAPRC', }}

    @property
    def STs(self):  # type: () -> STs
        return self._STs

    @STs.setter
    def STs(self, value):
        # noinspection PyAttributeOutsideInit
        self._STs = value

    @property
    def ROs(self):  # type: () -> ROs
        return self._ROs

    @ROs.setter
    def ROs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ROs = value

    @property
    def KWs(self):  # type: () -> KWs
        return self._KWs

    @KWs.setter
    def KWs(self, value):
        # noinspection PyAttributeOutsideInit
        self._KWs = value

    @property
    def ARs(self):  # type: () -> ARs
        return self._ARs

    @ARs.setter
    def ARs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ARs = value

    @property
    def ATs(self):  # type: () -> ATs
        return self._ATs

    @ATs.setter
    def ATs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ATs = value


###############
# PIAPRD

class SECT(NITFElement):
    __slots__ = ('SECTITLE', 'PPNUM', 'TPP')
    _formats = {'SECTITLE': '40s', 'PPNUM': '5s', 'TPP': '3d'}


class SECTs(NITFLoop):
    _child_class = SECT
    _count_size = 2


class RQORG(NITFElement):
    __slots__ = ('REQORG',)
    _formats = {'REQORG': '64s'}


class RQORGs(NITFLoop):
    _child_class = RQORG
    _count_size = 2


class KEYWD(NITFElement):
    __slots__ = ('KEYWORD',)
    _formats = {'KEYWORD': '255s'}


class KEYWDs(NITFLoop):
    _child_class = KEYWD
    _count_size = 2


class ASRPT(NITFElement):
    __slots__ = ('ASSRPT',)
    _formats = {'ASSRPT': '20s'}


class ASRPTs(NITFLoop):
    _child_class = ASRPT
    _count_size = 2


class ATEXT(NITFElement):
    __slots__ = ('ATEXT',)
    _formats = {'ATEXT': '255s'}


class ATEXTs(NITFLoop):
    _child_class = ATEXT
    _count_size = 2


class PIAPRD(TRE):
    __slots__ = (
        'TAG', 'ACCESSID', 'FMCNTROL', 'SUBDET', 'PRODCODE', 'PRODCRSE', 'PRODIDNO', 'PRODSNME', 'PRODCRCD', 'PRODCRTM',
        'MAPID', '_SECTs', '_RQORGs', '_KEYWDs', '_ASRPTs', '_ATEXTs')
    _formats = {
        'TAG': '6s', 'ACCESSID': '64s', 'FMCNTROL': '32s', 'SUBDET': '1s', 'PRODCODE': '2s', 'PRODCRSE': '6s',
        'PRODIDNO': '20s', 'PRODSNME': '10s', 'PRODCRCD': '2s', 'PRODCRTM': '14s', 'MAPID': '40s'}
    _types = {'_SECTs': SECTs, '_RQORGs': RQORGs, '_KEYWDs': KEYWDs, '_ASRPTs': ASRPTs, '_ATEXTs': ATEXTs}
    _defaults = {'_SECTs': {}, '_RQORGs': {}, '_KEYWDs': {}, '_ASRPTs': {}, '_ATEXTs': {}, 'TAG': 'PIAPRD'}
    _enums = {'TAG': {'PIAPRD', }}

    @property
    def SECTs(self):  # type: () -> SECTs
        return self._SECTs

    @SECTs.setter
    def SECTs(self, value):
        # noinspection PyAttributeOutsideInit
        self._SECTs = value

    @property
    def RQORGs(self):  # type: () -> RQORGs
        return self._RQORGs

    @RQORGs.setter
    def RQORGs(self, value):
        # noinspection PyAttributeOutsideInit
        self._RQORGs = value

    @property
    def KEYWDs(self):  # type: () -> KEYWDs
        return self._KEYWDs

    @KEYWDs.setter
    def KEYWDs(self, value):
        # noinspection PyAttributeOutsideInit
        self._KEYWDs = value

    @property
    def ASRPTs(self):  # type: () -> ASRPTs
        return self._ASRPTs

    @ASRPTs.setter
    def ASRPTs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ASRPTs = value

    @property
    def ATEXTs(self):  # type: () -> ATEXTs
        return self._ATEXTs

    @ATEXTs.setter
    def ATEXTs(self, value):
        # noinspection PyAttributeOutsideInit
        self._ATEXTs = value


class PIATGA(TRE):
    __slots__ = (
        'TAG', 'TGTUTM', 'PIATGAID', 'PIACTRY', 'PIACAT', 'TGTGEO', 'DATUM', 'TGTNAME', 'PERCOVER')
    _formats = {
        'TAG': '6s', 'TGTUTM': '15s', 'PIATGAID': '15s', 'PIACTRY': '2s', 'PIACAT': '5s', 'TGTGEO': '15s',
        'DATUM': '3s', 'TGTNAME': '38s', 'PERCOVER': '3d'}
    _defaults = {'TAG': 'PIATGA'}
    _enums = {'TAG': {'PIATGA', }}


class PIATGB(TRE):
    __slots__ = (
        'TAG', 'TGTUTM', 'PIATGAID', 'PIACTRY', 'PIACAT', 'TGTGEO', 'DATUM', 'TGTNAME', 'PERCOVER', 'TGTLAT', 'TGTLON')
    _formats = {
        'TAG': '6s', 'TGTUTM': '15s', 'PIATGAID': '15s', 'PIACTRY': '2s', 'PIACAT': '5s', 'TGTGEO': '15s',
        'DATUM': '3s', 'TGTNAME': '38s', 'PERCOVER': '3d', 'TGTLAT': '10s', 'TGTLON': '11s'}
    _defaults = {'TAG': 'PIATGB'}
    _enums = {'TAG': {'PIATGB', }}
