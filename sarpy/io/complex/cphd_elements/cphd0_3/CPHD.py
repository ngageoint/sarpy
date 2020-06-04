# -*- coding: utf-8 -*-
"""
The Compensated Phase History Data 0.3 definition.
"""

from typing import Union

from ..base import DEFAULT_STRICT
# noinspection PyProtectedMember
from ...sicd_elements.base import Serializable, _SerializableDescriptor

from ...sicd_elements.CollectionInfo import CollectionInfoType
from .Data import DataType
from .Global import GlobalType
from .Channel import ChannelType
from .SRP import SRPTyp
from .Antenna import AntennaType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


#########
# Module variables
_CPHD_SPECIFICATION_VERSION = '0.3'
_CPHD_SPECIFICATION_DATE = '2011-04-15T00:00:00Z'
_CPHD_SPECIFICATION_NAMESPACE = 'urn:CPHD:0.3'


class CPHDType(Serializable):
    """
    """

    _fields = (
        'CollectionInfo', 'Data', 'Global', 'Channel', 'SRP', 'Antenna')
    _required = (
        'CollectionInfo', 'Data', 'Global', 'Channel', 'SRP')
    # descriptors
    CollectionInfo = _SerializableDescriptor(
        'CollectionInfo', CollectionInfoType, _required, strict=DEFAULT_STRICT,
        docstring='General information about the collection.')  # type: CollectionInfoType
    Data = _SerializableDescriptor(
        'Data', DataType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe binary data components contained in the '
                  'product.')  # type: DataType
    Global = _SerializableDescriptor(
        'Global', GlobalType, _required, strict=DEFAULT_STRICT,
        docstring='Global parameters that apply to metadata components and CPHD '
                  'signal arrays.')  # type: GlobalType
    Channel = _SerializableDescriptor(
        'Channel', ChannelType, _required, strict=DEFAULT_STRICT,
        docstring='Channel specific parameters for CPHD channels.')  # type: ChannelType
    SRP = _SerializableDescriptor(
        'SRP', SRPTyp, _required, strict=DEFAULT_STRICT,
        docstring='The Stabilization Refence Point (SRP) parameters.')  # type: SRPTyp
    Antenna = _SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna parameters that describe antenna orientation, mainlobe '
                  'steering and gain patterns vs. '
                  'time.')  # type: Union[None, AntennaType]

    def __init__(self, CollectionInfo=None, Data=None, Global=None, Channel=None,
                 SRP=None, Antenna=None, **kwargs):
        """

        Parameters
        ----------
        CollectionInfo : CollectionInfoType
        Data : DataType
        Global : GlobalType
        Channel : ChannelType
        SRP : SRPTyp
        Antenna : NOne|AntennaType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CollectionInfo = CollectionInfo
        self.Data = Data
        self.Global = Global
        self.Channel = Channel
        self.SRP = SRP
        self.Antenna = Antenna
        super(CPHDType, self).__init__(**kwargs)

    def to_xml_bytes(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        return super(CPHDType, self).to_xml_bytes(
            urn=_CPHD_SPECIFICATION_NAMESPACE, tag=tag, check_validity=check_validity, strict=strict)
