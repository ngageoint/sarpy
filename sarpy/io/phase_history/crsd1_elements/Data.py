"""
The DataType definition.
"""

from typing import List

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _StringDescriptor, _StringEnumDescriptor, \
    _IntegerDescriptor, _SerializableListDescriptor
from sarpy.io.phase_history.cphd1_elements.Data import SupportArraySizeType
from sarpy.io.phase_history.cphd1_elements.utils import binary_format_string_to_dtype

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Michael Stewart, Valkyrie")


class ChannelSizeType(Serializable):
    """
    Parameters that define the Channel signal array and PVP array size and location.
    """

    _fields = ('Identifier', 'NumVectors', 'NumSamples', 'SignalArrayByteOffset', 'PVPArrayByteOffset')
    _required = _fields
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies the CRSD channel (Ch_ID)'
                  ' for which the data applies.')  # type: str
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

    def __init__(self, Identifier=None, NumVectors=None, NumSamples=None, SignalArrayByteOffset=None,
                 PVPArrayByteOffset=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        NumVectors : int
        NumSamples : int
        SignalArrayByteOffset : int
        PVPArrayByteOffset : int
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
        super(ChannelSizeType, self).__init__(**kwargs)


class DataType(Serializable):
    """
    Parameters that describe binary data components contained in the product.
    """

    _fields = (
        'SignalArrayFormat', 'NumBytesPVP', 'NumCRSDChannels',
        'Channels', 'NumSupportArrays', 'SupportArrays')
    _required = ('SignalArrayFormat', 'NumBytesPVP', 'Channels')
    _collections_tags = {
        'Channels': {'array': False, 'child_tag': 'Channel'},
        'SupportArrays': {'array': False, 'child_tag': 'SupportArray'}}
    # descriptors
    SignalArrayFormat = _StringEnumDescriptor(
        'SignalArrayFormat', ('CI2', 'CI4', 'CF8'), _required, strict=DEFAULT_STRICT,
        docstring="Signal Array sample binary format of the CRSD signal arrays, where"
                  "`CI2` denotes a 1 byte signed integer parameter, 2's complement format, and 2 Bytes Per Sample;"
                  "`CI4` denotes a 2 byte signed integer parameter, 2's complement format, and 4 Bytes Per Sample;"
                  "`CF8` denotes a 4 byte floating point parameter, and 8 Bytes Per Sample.")  # type: str
    NumBytesPVP = _IntegerDescriptor(
        'NumBytesPVP', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Number of bytes per set of Per Vector Parameters, where there is '
                  'one set of PVPs for each CRSD signal vector')  # type: int
    Channels = _SerializableListDescriptor(
        'Channels', ChannelSizeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that define the Channel signal array and PVP array size '
                  'and location.')  # type: List[ChannelSizeType]
    SupportArrays = _SerializableListDescriptor(
        'SupportArrays', SupportArraySizeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Support Array size parameters. Branch repeated for each binary support array. '
                  'Support Array referenced by its unique Support Array '
                  'identifier.')  # type: List[SupportArraySizeType]

    def __init__(self, SignalArrayFormat=None, NumBytesPVP=None, Channels=None, SupportArrays=None, **kwargs):
        """

        Parameters
        ----------
        SignalArrayFormat : str
        NumBytesPVP : int
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
    def NumCRSDChannels(self):
        """
        int: The number of CRSD channels.
        """

        if self.Channels is None:
            return 0
        else:
            return len(self.Channels)

    def calculate_support_block_size(self):
        """
        Calculates the size of the support block in bytes as described by the SupportArray fields.
        """
        return sum([s.calculate_size() for s in self.SupportArrays])

    def calculate_pvp_block_size(self):
        """
        Calculates the size of the PVP block in bytes as described by the Data fields.
        """
        return self.NumBytesPVP * sum([c.NumVectors for c in self.Channels])

    def calculate_signal_block_size(self):
        """
        Calculates the size of the signal block in bytes as described by the Data fields.
        """
        num_bytes_per_sample = binary_format_string_to_dtype(self.SignalArrayFormat).itemsize
        return num_bytes_per_sample * sum([c.NumVectors * c.NumSamples for c in self.Channels])
