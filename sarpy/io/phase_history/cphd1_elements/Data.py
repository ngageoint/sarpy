# -*- coding: utf-8 -*-
"""
The DataType definition.
"""

from typing import List

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _StringDescriptor, _StringEnumDescriptor, \
    _IntegerDescriptor, _SerializableListDescriptor

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ChannelSizeType(Serializable):
    """
    Parameters that define the Channel signal array and PVP array size and location.
    """

    _fields = (
        'Identifier', 'NumVectors', 'NumSamples', 'SignalArrayByteOffset', 'PVPArrayByteOffset',
        'CompressedSignalSize')
    _required = (
        'Identifier', 'NumVectors', 'NumSamples', 'SignalArrayByteOffset', 'PVPArrayByteOffset')
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies the CPHD channel for which the data '
                  'applies.')  # type: str
    NumVectors = _IntegerDescriptor(
        'NumVectors', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of vectors in the signal array.')  # type: int
    NumSamples = _IntegerDescriptor(
        'NumSamples', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of samples per vector in the signal array.')  # type: int
    SignalArrayByteOffset = _IntegerDescriptor(
        'SignalArrayByteOffset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Signal Array offset from the start of the Signal block (in bytes) to the '
                  'start of the Signal Array for the channel.')  # type: int
    PVPArrayByteOffset = _IntegerDescriptor(
        'PVPArrayByteOffset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='PVP Array offset from the start of the PVP block (in bytes) to the '
                  'start of the PVP Array for the channel.')  # type: int
    CompressedSignalSize = _IntegerDescriptor(
        'CompressedSignalSize', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Size (in bytes) of the compressed signal array byte sequence for the data channel. '
                  'Parameter included if and only if the signal arrays are stored in '
                  'compressed format.')  # type: int

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
    Support Array size parameters.
    """

    _fields = ('Identifier', 'NumRows', 'NumCols', 'BytesPerElement', 'ArrayByteOffset')
    _required = _fields
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='Unique string that identifies this support array.')  # type: str
    NumRows = _IntegerDescriptor(
        'NumRows', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of rows in the array.')  # type: int
    NumCols = _IntegerDescriptor(
        'NumCols', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Number of columns per row in the array.')  # type: int
    BytesPerElement = _IntegerDescriptor(
        'BytesPerElement', _required, strict=DEFAULT_STRICT, bounds=(1, None),
        docstring='Indicates the size in bytes of each data element in the support '
                  'array. Each element contains 1 or more binary-formatted '
                  'components.')  # type: int
    ArrayByteOffset = _IntegerDescriptor(
        'ArrayByteOffset', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Array offset from the start of the Support block (in bytes) to '
                  'the start of the support array.')  # type: int

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
    Parameters that describe binary data components contained in the product.
    """

    _fields = (
        'SignalArrayFormat', 'NumBytesPVP', 'NumCPHDChannels',
        'SignalCompressionID', 'Channels', 'NumSupportArrays', 'SupportArrays')
    _required = ('SignalArrayFormat', 'NumBytesPVP', 'Channels')
    _collections_tags = {
        'Channels': {'array': False, 'child_tag': 'Channel'},
        'SupportArrays': {'array': False, 'child_tag': 'SupportArray'}}
    # descriptors
    SignalArrayFormat = _StringEnumDescriptor(
        'SignalArrayFormat', ('CI2', 'CI4', 'CF8'), _required, strict=DEFAULT_STRICT,
        docstring='Signal Array sample binary format of the CPHD signal arrays in standard '
                  '(i.e. uncompressed) format, where `CI2` denotes a 1 byte signed integer '
                  "parameter, 2's complement format, and 2 Bytes Per Sample; `CI4` denotes "
                  "a 2 byte signed integer parameter, 2's complement format, and "
                  "4 Bytes Per Sample; `CF8` denotes a 4 byte floating point parameter, and "
                  "8 Bytes Per Sample.")  # type: str
    NumBytesPVP = _IntegerDescriptor(
        'NumBytesPVP', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Number of bytes per set of Per Vector Parameters, where there is '
                  'one set of PVPs for each CPHD signal vector')  # type: int
    SignalCompressionID = _StringDescriptor(
        'SignalCompressionID', _required, strict=DEFAULT_STRICT,
        docstring='Parameter that indicates the signal arrays are in compressed format. Value '
                  'identifies the method of decompression. Parameter included if and only if '
                  'the signal arrays are in compressed format.')  # type: str
    Channels = _SerializableListDescriptor(
        'Channels', ChannelSizeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that define the Channel signal array and PVP array size '
                  'and location.')  # type: List[ChannelSizeType]
    SupportArrays = _SerializableListDescriptor(
        'SupportArrays', SupportArraySizeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Support Array size parameters. Branch repeated for each binary support array. '
                  'Support Array referenced by its unique Support Array '
                  'identifier.')  # type: List[SupportArraySizeType]

    def __init__(self, SignalArrayFormat=None, NumBytesPVP=None,
                 SignalCompressionID=None, Channels=None, SupportArrays=None, **kwargs):
        """

        Parameters
        ----------
        SignalArrayFormat : str
        NumBytesPVP : int
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
        self.SignalCompressionID = SignalCompressionID
        self.Channels = Channels
        self.SupportArrays = SupportArrays
        super(DataType, self).__init__(**kwargs)

    @property
    def NumSupportArrays(self):
        """
        int: The number of support arrays.
        """

        if self.SupportArrays is None:
            return 0
        else:
            return len(self.SupportArrays)

    @property
    def NumCPHDChannels(self):
        """
        int: The number of CPHD channels.
        """

        if self.Channels is None:
            return 0
        else:
            return len(self.Channels)
