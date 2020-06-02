# -*- coding: utf-8 -*-
"""
The DataType definition.
"""

from typing import List

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from ..sicd_elements.base import Serializable, _StringDescriptor, _StringEnumDescriptor, \
    _IntegerDescriptor, _SerializableListDescriptor

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ChannelSizeType(Serializable):
    """
    The channel size definition.
    """

    _fields = (
        'Identifier', 'NumVectors', 'NumSamples', 'SignalArrayByteOffset', 'PVPArrayByteOffset',
        'CompressedSignalSize')
    _required = (
        'Identifier', 'NumVectors', 'NumSamples', 'SignalArrayByteOffset', 'PVPArrayByteOffset')
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    NumVectors = _IntegerDescriptor(
        'NumVectors', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='')  # type: int
    NumSamples = _IntegerDescriptor(
        'NumSamples', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='')  # type: int
    SignalArrayByteOffset = _IntegerDescriptor(
        'SignalArrayByteOffset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: int
    PVPArrayByteOffset = _IntegerDescriptor(
        'PVPArrayByteOffset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: int
    CompressedSignalSize = _IntegerDescriptor(
        'CompressedSignalSize', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='')  # type: int

    def __init__(self, Identifier=None, NumVectors=None, NumSamples=None, SignalArrayByteOffset=None,
                 PVPArrayByteOffset=None, CompressedSignalSize=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        NumVectors : int
        NumSamples : int
        SignalArrayByteOffset : int
        PVPArrayByteOffset : int
        CompressedSignalSize : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.NumVectors = NumVectors
        self.NumSamples = NumSamples
        self.SignalArrayByteOffset = SignalArrayByteOffset
        self.PVPArrayByteOffset = PVPArrayByteOffset
        self.CompressedSignalSize = CompressedSignalSize
        super(ChannelSizeType, self).__init__(**kwargs)


class SupportArraySizeType(Serializable):
    """
    The support array size.
    """

    _fields = ('Identifier', 'NumRows', 'NumCols', 'BytesPerElement', 'ArrayByteOffset')
    _required = _fields
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    NumRows = _IntegerDescriptor(
        'NumRows', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='')  # type: int
    NumCols = _IntegerDescriptor(
        'NumCols', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='')  # type: int
    BytesPerElement = _IntegerDescriptor(
        'BytesPerElement', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='')  # type: int
    ArrayByteOffset = _IntegerDescriptor(
        'ArrayByteOffset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: int

    def __init__(self, Identifier=None, NumRows=None, NumCols=None, BytesPerElement=None,
                 ArrayByteOffset=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        NumRows : int
        NumCols : int
        BytesPerElement : int
        ArrayByteOffset : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.NumRows = NumRows
        self.NumCols = NumCols
        self.BytesPerElement = BytesPerElement
        self.ArrayByteOffset = ArrayByteOffset
        super(SupportArraySizeType, self).__init__(**kwargs)


class DataType(Serializable):
    """
    The data type definition.
    """

    _fields = (
        'SignalArrayFormat', 'NumBytesPVP', 'NumCPHDChannels',
        'SignalCompressionID', 'Channels', 'NumSupportArrays', 'SupportArrays')
    _required = ('SignalArrayFormat', 'NumBytesPVP', 'NumCPHDChannels', 'Channels')
    _collections_tags = {
        'Channel': {'array': False, 'child_tag': 'Channel'},
        'SupportArrays': {'array': False, 'child_tag': 'SupportArray'}}
    # descriptors
    SignalArrayFormat = _StringEnumDescriptor(
        'SignalArrayFormat', ('CI2', 'CI4', 'CF8'), _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    NumBytesPVP = _IntegerDescriptor(
        'NumBytesPVP', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: int
    NumCPHDChannels = _IntegerDescriptor(
        'NumCPHDChannels', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='')  # type: int
    SignalCompressionID = _StringDescriptor(
        'SignalCompressionID', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    Channels = _SerializableListDescriptor(
        'Channels', ChannelSizeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: List[ChannelSizeType]
    SupportArrays = _SerializableListDescriptor(
        'SupportArrays', SupportArraySizeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: List[SupportArraySizeType]

    def __init__(self, SignalArrayFormat=None, NumBytesPVP=None, NumCPHDChannels=None,
                 SignalCompressionID=None, Channels=None, SupportArrays=None, **kwargs):
        """

        Parameters
        ----------
        SignalArrayFormat : str
        NumBytesPVP : int
        NumCPHDChannels : int
        SignalCompressionID : None|str
        Channels : List[ChannelSizeType]
        SupportArrays : None|List[SupportArraySizeType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.SignalArrayFormat = SignalArrayFormat
        self.NumBytesPVP = NumBytesPVP
        self.NumCPHDChannels = NumCPHDChannels
        self.SignalCompressionID = SignalCompressionID
        self.Channels = Channels
        self.SupportArrays = SupportArrays
        super(DataType, self).__init__(**kwargs)

    @property
    def NumSupportArrays(self):
        if self.SupportArrays is None:
            return 0
        else:
            return len(self.SupportArrays)
