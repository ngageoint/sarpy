"""
The TxRcv type definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import FloatDescriptor, StringDescriptor, \
    StringEnumDescriptor, SerializableListDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import POLARIZATION_TYPE


class TxWFParametersType(Serializable):
    """
    Parameters that describe a Transmit Waveform.
    """

    _fields = (
        'Identifier', 'PulseLength', 'RFBandwidth', 'FreqCenter', 'LFMRate',
        'Polarization', 'Power')
    _required = (
        'Identifier', 'PulseLength', 'RFBandwidth', 'FreqCenter', 'Polarization')
    _numeric_format = {
        'PulseLength': FLOAT_FORMAT, 'RFBandwidth': FLOAT_FORMAT, 'FreqCenter': FLOAT_FORMAT,
        'LFMRate': FLOAT_FORMAT, 'Power': FLOAT_FORMAT}
    # descriptors
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this Transmit '
                  'Waveform.')  # type: str
    PulseLength = FloatDescriptor(
        'PulseLength', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Length of transmitted pulse, '
                  'in seconds.')  # type: float
    RFBandwidth = FloatDescriptor(
        'RFBandwidth', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Bandwidth of transmitted pulse, '
                  'in Hz.')  # type: float
    FreqCenter = FloatDescriptor(
        'FreqCenter', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Center frequency of the transmitted waveform, '
                  'in Hz.')  # type: float
    LFMRate = FloatDescriptor(
        'LFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Chirp rate of transmitted pulse if LFM, '
                  'in Hz/s.')  # type: Union[None, float]
    Polarization = StringEnumDescriptor(
        'Polarization', POLARIZATION_TYPE, _required, strict=DEFAULT_STRICT,
        docstring='The transmit polarization mode.')  # type: str
    Power = FloatDescriptor(
        'Power', _required, strict=DEFAULT_STRICT,
        docstring='Peak transmitted power at the interface to the antenna '
                  'in dBW.')  # type: Union[None, float]

    def __init__(self, Identifier=None, PulseLength=None, RFBandwidth=None,
                 FreqCenter=None, LFMRate=None, Polarization=None, Power=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        PulseLength : float
        RFBandwidth : float
        FreqCenter : float
        LFMRate : None|float
        Polarization : str
        Power : None|float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.PulseLength = PulseLength
        self.RFBandwidth = RFBandwidth
        self.FreqCenter = FreqCenter
        self.LFMRate = LFMRate
        self.Polarization = Polarization
        self.Power = Power
        super(TxWFParametersType, self).__init__(**kwargs)


class RcvParametersType(Serializable):
    """
    Parameters that describe a Receive configuration.
    """

    _fields = (
        'Identifier', 'WindowLength', 'SampleRate', 'IFFilterBW', 'FreqCenter',
        'LFMRate', 'Polarization', 'PathGain')
    _required = (
        'Identifier', 'WindowLength', 'SampleRate', 'IFFilterBW', 'FreqCenter',
        'Polarization')
    _numeric_format = {
        'WindowLength': FLOAT_FORMAT, 'SampleRate': FLOAT_FORMAT, 'IFFilterBW': FLOAT_FORMAT,
        'FreqCenter': FLOAT_FORMAT, 'LFMRate': FLOAT_FORMAT, 'PathGain': FLOAT_FORMAT}
    # descriptors
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this Receive '
                  'configuration.')  # type: str
    WindowLength = FloatDescriptor(
        'WindowLength', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Length of the receive window, in seconds.')  # type: float
    SampleRate = FloatDescriptor(
        'SampleRate', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Rate at which the signal in the receive window is sampled, '
                  'in Hz.')  # type: float
    IFFilterBW = FloatDescriptor(
        'IFFilterBW', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Bandwidth of the anti-aliasing filter prior to '
                  'sampling.')  # type: float
    FreqCenter = FloatDescriptor(
        'FreqCenter', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Center frequency of the demodulation signal, '
                  'in Hz.')  # type: float
    LFMRate = FloatDescriptor(
        'LFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Chirp rate of the demodulation signal if LFM, '
                  'in Hz/s.')  # type: Union[None, float]
    Polarization = StringEnumDescriptor(
        'Polarization', POLARIZATION_TYPE, _required, strict=DEFAULT_STRICT,
        docstring='The receive polarization mode.')  # type: str
    PathGain = FloatDescriptor(
        'PathGain', _required, strict=DEFAULT_STRICT,
        docstring='Receiver gain from the antenna interface to the ADC, '
                  'in dB.')  # type: Union[None, float]

    def __init__(self, Identifier=None, WindowLength=None, SampleRate=None, IFFilterBW=None,
                 FreqCenter=None, LFMRate=None, Polarization=None, PathGain=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        WindowLength : float
        SampleRate : float
        IFFilterBW : float
        FreqCenter : float
        LFMRate : None|float
        Polarization : str
        PathGain : None|float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.WindowLength = WindowLength
        self.SampleRate = SampleRate
        self.IFFilterBW = IFFilterBW
        self.FreqCenter = FreqCenter
        self.LFMRate = LFMRate
        self.Polarization = Polarization
        self.PathGain = PathGain
        super(RcvParametersType, self).__init__(**kwargs)


class TxRcvType(Serializable):
    """
    Parameters that describe the transmitted waveform(s) and receiver
    configurations used in the collection
    """

    _fields = ('NumTxWFs', 'TxWFParameters', 'NumRcvs', 'RcvParameters')
    _required = ('TxWFParameters', 'RcvParameters')
    _collections_tags = {
        'TxWFParameters': {'array': False, 'child_tag': 'TxWFParameters'},
        'RcvParameters': {'array': False, 'child_tag': 'RcvParameters'}}
    # descriptors
    TxWFParameters = SerializableListDescriptor(
        'TxWFParameters', TxWFParametersType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe a Transmit Waveform.')  # type: List[TxWFParametersType]
    RcvParameters = SerializableListDescriptor(
        'RcvParameters', RcvParametersType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe a Receive configuration.')  # type: List[RcvParametersType]

    def __init__(self, TxWFParameters=None, RcvParameters=None, **kwargs):
        """

        Parameters
        ----------
        TxWFParameters : List[TxWFParametersType]
        RcvParameters : List[RcvParametersType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxWFParameters = TxWFParameters
        self.RcvParameters = RcvParameters
        super(TxRcvType, self).__init__(**kwargs)

    @property
    def NumTxWFs(self):
        """
        int: The number of transmit waveforms used.
        """

        if self.TxWFParameters is None:
            return 0
        return len(self.TxWFParameters)

    @property
    def NumRcvs(self):
        """
        int: The number of receive configurations used.
        """

        if self.RcvParameters is None:
            return 0
        return len(self.RcvParameters)
