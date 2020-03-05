# -*- coding: utf-8 -*-

from ...headers import NITFElement, NITFLoop, TRE


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CLCTNA(TRE):
    __slots__ = (
        'TAG', 'VERNUM', 'CLCTN_NAME', 'CLCTN_DESCRIPT', 'CLCTN_STDATE', 'CLCTN_SPDATE', 'CLCTN_LOC', 'COUNTRY',
        'SPONSOR', 'PERSONNEL', 'SCLCTN_NAME', 'SDESCRIPTION', 'SCLCTN_Z_OFF', 'SCLCTN_STDATE', 'SCLCTN_SPDATE',
        'SECURITY', 'SCG', 'SITE', 'SITE_NUM', 'SCN_NUM', 'FLIGHT_NUM', 'PASS_NUM', 'SCN_CNTR', 'ALTITUDE',
        'SCN_CONTENT', 'BGRND_TYPE', 'WX_STATION', 'WX_OVERVIEW', 'WX_FILE')
    _formats = {
        'TAG': '6s', 'VERNUM': '4d', 'CLCTN_NAME': '25s', 'CLCTN_DESCRIPT': '255s', 'CLCTN_STDATE': '8s',
        'CLCTN_SPDATE': '8s', 'CLCTN_LOC': '11s', 'COUNTRY': '2s', 'SPONSOR': '20s', 'PERSONNEL': '100s',
        'SCLCTN_NAME': '20s', 'SDESCRIPTION': '255s', 'SCLCTN_Z_OFF': '3s', 'SCLCTN_STDATE': '8s',
        'SCLCTN_SPDATE': '8s', 'SECURITY': '7s', 'SCG': '15s', 'SITE': '15s', 'SITE_NUM': '3s', 'SCN_NUM': '3s',
        'FLIGHT_NUM': '2s', 'PASS_NUM': '2s', 'SCN_CNTR': '11s', 'ALTITUDE': '5s', 'SCN_CONTENT': '50s',
        'BGRND_TYPE': '50s', 'WX_STATION': '20s', 'WX_OVERVIEW': '15s', 'WX_FILE': '30s'}
    _defaults = {'TAG': 'CLCTNA'}
    _enums = {'TAG': {'CLCTNA', }}


##############
# CLCTNB

class SITE(NITFElement):
    __slots__ = (
        'SCLCTN_NAME', 'SDESCRIPTION', 'SITE_NUM', 'SCN_NUM', 'SCLCTN_STDATE', 'SCLCTN_SPDATE', 'SCN_CNTR',
        'ALTITUDE', 'SCN_CONTENT', 'BGRND_TYPE', 'SITE_COV')
    _formats = {
        'SCLCTN_NAME': '20s', 'SDESCRIPTION': '255s', 'SITE_NUM': '3s', 'SCN_NUM': '3s', 'SCLCTN_STDATE': '8s',
        'SCLCTN_SPDATE': '8s', 'SCN_CNTR': '11s', 'ALTITUDE': '5s', 'SCN_CONTENT': '50s', 'BGRND_TYPE': '50s',
        'SITE_COV': '1s'}


class SITEs(NITFLoop):
    _child_class = SITE
    _count_size = 1


class CLCTNB(TRE):
    __slots__ = (
        'TAG', 'VERNUM', 'CLCTN_NAME', 'CLCTN_DESCRIPT', 'CLCTN_STDATE', 'CLCTN_SPDATE', 'CLCTN_LOC', 'SITE', 'COUNTRY',
        'SPONSOR', 'PERSONNEL', '_SITEs', 'SCLCTN_Z_OFF', 'SECURITY', 'SCG', 'FLIGHT_NUM', 'PASS_NUM', 'WX_STATION',
        'WX_OVERVIEW', 'WX_FILE')
    _formats = {
        'TAG': '6s', 'VERNUM': '4d', 'CLCTN_NAME': '25s', 'CLCTN_DESCRIPT': '255s', 'CLCTN_STDATE': '8s',
        'CLCTN_SPDATE': '8s', 'CLCTN_LOC': '11s', 'SITE': '15s', 'COUNTRY': '2s', 'SPONSOR': '20s', 'PERSONNEL': '100s',
        'SCLCTN_Z_OFF': '3s', 'SECURITY': '7s', 'SCG': '15s', 'FLIGHT_NUM': '2s', 'PASS_NUM': '2s', 'WX_STATION': '20s',
        'WX_OVERVIEW': '15s', 'WX_FILE': '30s'}
    _types = {'_SITEs': SITEs}
    _defaults = {'_SITEs': {}, 'TAG': 'CLCTNB'}
    _enums = {'TAG': {'CLCTNB', }}

    @property
    def SITEs(self):  # type: () -> SITEs
        return self._SITEs

    @SITEs.setter
    def SITEs(self, value):
        # noinspection PyAttributeOutsideInit
        self._SITEs = value
