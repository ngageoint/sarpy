"""
The Channel definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Michael Stewart, Valkyrie")

from typing import Union, List

from sarpy.io.xml.base import Serializable, ParametersCollection
from sarpy.io.xml.descriptors import StringDescriptor, StringEnumDescriptor, \
    IntegerDescriptor, SerializableListDescriptor, ParametersDescriptor, \
    BooleanDescriptor, FloatDescriptor, SerializableDescriptor

from sarpy.io.phase_history.cphd1_elements.blocks import POLARIZATION_TYPE, AreaType
from sarpy.io.phase_history.cphd1_elements.Channel import DwellTimesType

from .base import DEFAULT_STRICT


class RcvAntennaType(Serializable):
    """
    The receive antenna information.
    """

    _fields = ('RcvAPCId', 'RcvAPATId')
    _required = _fields
    # descriptors
    RcvAPCId = StringDescriptor(
        'RcvAPCId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier for the Receive APC to be used to compute the receive antenna'
                  ' pattern as a function of time for the channel (APC_ID).')  # type: str
    RcvAPATId = StringDescriptor(
        'RcvAPATId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier for the Receive Antenna pattern to collect the signal data'
                  ' (APAT_ID).')  # type: str

    def __init__(self, RcvAPCId=None, RcvAPATId=None, **kwargs):
        """

        Parameters
        ----------
        RcvAPCId : str
        RcvAPATId : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RcvAPCId = RcvAPCId
        self.RcvAPATId = RcvAPATId
        super(RcvAntennaType, self).__init__(**kwargs)


class SignalRefLevelType(Serializable):
    """
    The signal power level information.
    """

    _fields = ('PSCRSD', 'PRcvDensity')
    _required = _fields
    _numeric_format = {fld: '0.16G' for fld in _fields}
    # descriptors
    PSCRSD = FloatDescriptor(
        'PSCRSD', _required, strict=DEFAULT_STRICT,
        docstring='Power level in the fast time signal vector for a CW tone at f = f_0_REF'
                  ' and for f_IC(v,t) = f_0_REF.')  # type: float
    PRcvDensity = FloatDescriptor(
        'PRcvDensity', _required, strict=DEFAULT_STRICT,
        docstring='Receive power density per unit area for a CW tone at f = f_0_REF that results in'
                  ' signal vector power PS_CRSD. Signal received from a far field source located'
                  ' along the receive antenna mainlobe boresight at t = trs(v_CH_REF).')  # type:float

    def __init__(self, PSCRSD=None, PRcvDensity=None, **kwargs):
        """

        Parameters
        ----------
        PSCRSD : float
        PRcvDensity : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PSCRSD = PSCRSD
        self.PRcvDensity = PRcvDensity
        super(SignalRefLevelType, self).__init__(**kwargs)


class NoiseLevelType(Serializable):
    """
    The thermal noise level information.
    """

    _fields = ('PNCRSD', 'BNCRSD')
    _required = _fields
    _numeric_format = {fld: '0.16G' for fld in _fields}
    # descriptors
    PNCRSD = FloatDescriptor(
        'PNCRSD', _required, strict=DEFAULT_STRICT,
        docstring='Noise power level in fast time signal vector for f_IC(v,t) = f_0(v_CH_REF).')  # type: float
    BNCRSD = FloatDescriptor(
        'BNCRSD', _required, strict=DEFAULT_STRICT,
        docstring='Noise Equivalent BW for the noise signal. Bandwidth BN_CRSD is expressed relative'
                  ' to the fast time sample rate for the channel (fs).')  # type:float

    def __init__(self, PNCRSD=None, BNCRSD=None, **kwargs):
        """

        Parameters
        ----------
        PNCRSD : float
        BNCRSD : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PNCRSD = PNCRSD
        self.BNCRSD = BNCRSD
        super(NoiseLevelType, self).__init__(**kwargs)


class TxAntennaType(Serializable):
    """
    The receive antenna information.
    """

    _fields = ('TxAPCId', 'TxAPATId')
    _required = _fields
    # descriptors
    TxAPCId = StringDescriptor(
        'TxAPCId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of Transmit APC to be used to compute the transmit antenna'
                  ' pattern as a function of time for the channel (APC_ID).')  # type: str
    TxAPATId = StringDescriptor(
        'TxAPATId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of the Transmit Antenna pattern used to form the channel'
                  ' signal array (APAT_ID).')  # type: str

    def __init__(self, TxAPCId=None, TxAPATId=None, **kwargs):
        """

        Parameters
        ----------
        TxAPCId : str
        TxAPATId : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxAPCId = TxAPCId
        self.TxAPATId = TxAPATId
        super(TxAntennaType, self).__init__(**kwargs)


class SARImagingType(Serializable):
    """
    The SAR Imaging parameters
    """

    _fields = ('TxLFMFixed', 'TxPol', 'DwellTimes', 'TxAntenna', 'ImageArea')
    _required = ('TxPol', 'DwellTimes')
    # descriptors
    TxLFMFixed = BooleanDescriptor(
        'TxLFMFixed', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate the same transmit LFM waveform is used for all pulses.'
                  ' vectors of the channel.')  # type: Union[None, bool]
    TxPol = StringEnumDescriptor(
        'TxPol', POLARIZATION_TYPE, _required, strict=DEFAULT_STRICT,
        docstring='Transmitted signal polarization for the channel.')  # type: str
    DwellTimes = SerializableDescriptor(
        'DwellTimes', DwellTimesType, _required, strict=DEFAULT_STRICT,
        docstring='COD Time and Dwell Time polynomials over the image area.')  # type: DwellTimesType
    TxAntenna = SerializableDescriptor(
        'TxAntenna', TxAntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna Phase Center and Antenna Pattern identifiers for the transmit antenna'
                  ' used to illuminate the imaged area.')  # type: Union[None, TxAntennaType]
    ImageArea = SerializableDescriptor(
        'ImageArea', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='SAR Image Area for the channel defined by a rectangle aligned with (IAX, IAY).'
                  ' May be reduced by the optional polygon.')  # type: Union[None, AreaType]

    def __init__(self, TxLFMFixed=None, TxPol=None, DwellTimes=None, TxAntenna=None,
                 ImageArea=None, **kwargs):
        """

        Parameters
        ----------
        TxLFMFixed : None|bool
        TxPol : PolarizationType
        DwellTimes : DwellTimesType
        TxAntenna : None|TxAntennaType
        ImageArea : None|AreaType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxLFMFixed = TxLFMFixed
        self.TxPol = TxPol
        self.DwellTimes = DwellTimes
        self.TxAntenna = TxAntenna
        self.ImageArea = ImageArea
        super(SARImagingType, self).__init__(**kwargs)


class ChannelParametersType(Serializable):
    """
    The CRSD data channel parameters
    """
    _fields = (
        'Identifier', 'RefVectorIndex', 'RefFreqFixed', 'FrcvFixed', 'DemodFixed',
        'F0Ref', 'Fs', 'BWInst', 'RcvPol', 'SignalNormal', 'RcvAntenna',
        'SignalRefLevel', 'NoiseLevel', 'AddedParameters', 'SARImaging')
    _required = (
        'Identifier', 'RefVectorIndex', 'RefFreqFixed', 'FrcvFixed', 'DemodFixed',
        'F0Ref', 'Fs', 'BWInst', 'RcvPol')
    _numeric_format = {
        'F0Ref': '0.16G', 'Fs': '0.16G', 'BWInst': '0.16G'}
    _collections_tags = {
        'AddedParameters': {'array': False, 'child_tag': 'AddedParameters'}}
    # descriptors
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this CRSD data channel.')  # type: str
    RefVectorIndex = IntegerDescriptor(
        'RefVectorIndex', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Index of the reference vector for the channel.')  # type: int
    RefFreqFixed = BooleanDescriptor(
        'RefFreqFixed', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate a constant reference frequency is used for'
                  ' the channel.')  # type: bool
    FrcvFixed = BooleanDescriptor(
        'FrcvFixed', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate a constant receive band is saved for the'
                  ' channel.')  # type: bool
    DemodFixed = BooleanDescriptor(
        'DemodFixed', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate a constant demodulation is used for the'
                  ' channel.')  # type: bool
    F0Ref = FloatDescriptor(
        'F0Ref', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Reference frequency for the reference signal vector.')  # type: float
    Fs = FloatDescriptor(
        'Fs', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Fast time sample rate for the signal array.')  # type: float
    BWInst = FloatDescriptor(
        'BWInst', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Nominal instantaneous bandwidth for the channel.')  # type: float
    RcvPol = StringEnumDescriptor(
        'RcvPol', POLARIZATION_TYPE, _required, strict=DEFAULT_STRICT,
        docstring='Receive polarization for the signal data processed to form the signal array.'
                  ' Parameter describes the E-Field orientation of the signal.')  # type: str
    SignalNormal = BooleanDescriptor(
        'SignalNormal', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate when all signal array vectors are normal.'
                  ' Included if and only if the SIGNAL PVP is also included.')  # type: Union[None, bool]
    RcvAntenna = SerializableDescriptor(
        'RcvAntenna', RcvAntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna Phase Center and Antenna Pattern identifiers for the receive antenna'
                  ' used to collect and form the signal array data.')  # type: Union[None, RcvAntennaType]
    SignalRefLevel = SerializableDescriptor(
        'SignalRefLevel', SignalRefLevelType, _required, strict=DEFAULT_STRICT,
        docstring='Signal power levels for a received CW signal with f = f_0_REF and polarization'
                  ' matched to RcvPol of the channel.')  # type: Union[None, SignalRefLevelType]
    NoiseLevel = SerializableDescriptor(
        'NoiseLevel', NoiseLevelType, _required, strict=DEFAULT_STRICT,
        docstring='Thermal noise level in the CRSD signal vector for f_IC(v,t) ='
                  ' f_0(v_CH_REF).')  # type: Union[None, NoiseLevelType]
    AddedParameters = ParametersDescriptor(
        'AddedParameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Additional free form parameters.')  # type: Union[None, ParametersCollection]
    SARImaging = SerializableDescriptor(
        'SARImaging', SARImagingType, _required, strict=DEFAULT_STRICT,
        docstring='Structure included for all SAR imaging collections.')  # type: Union[None, SARImagingType]

    def __init__(self, Identifier=None, RefVectorIndex=None, RefFreqFixed=None,
                 FrcvFixed=None, DemodFixed=None, F0Ref=None, Fs=None, BWInst=None,
                 RcvPol=None, SignalNormal=None, RcvAntenna=None, SignalRefLevel=None,
                 NoiseLevel=None, AddedParameters=None, SARImaging=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        RefVectorIndex : int
        RefFreqFixed : bool
        FrcvFixed : bool
        DemodFixed : bool
        F0Ref : float
        Fs : float
        BWInst : float
        RcvPol : str
        SignalNormal : None|bool
        RcvAntenna : None|RcvAntennaType
        SignalRefLevel : None|SignalRefLevelType
        NoiseLevel : None|NoiseLevelType
        AddedParameters : None|ParametersCollection
        SARImaging : None|SARImagingType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.RefVectorIndex = RefVectorIndex
        self.RefFreqFixed = RefFreqFixed
        self.FrcvFixed = FrcvFixed
        self.DemodFixed = DemodFixed
        self.F0Ref = F0Ref
        self.Fs = Fs
        self.BWInst = BWInst
        self.RcvPol = RcvPol
        self.SignalNormal = SignalNormal
        self.RcvAntenna = RcvAntenna
        self.SignalRefLevel = SignalRefLevel
        self.NoiseLevel = NoiseLevel
        self.AddedParameters = AddedParameters
        self.SARImaging = SARImaging
        super(ChannelParametersType, self).__init__(**kwargs)


class ChannelType(Serializable):
    """
    The channel definition.
    """

    _fields = ('RefChId', 'Parameters')
    _required = _fields
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameters'}}
    # descriptors
    RefChId = StringDescriptor(
        'RefChId', _required, strict=DEFAULT_STRICT,
        docstring='Channel ID for the Reference Channel in the '
                  'product.')  # type: str
    Parameters = SerializableListDescriptor(
        'Parameters', ChannelParametersType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameter Set that describes a CRSD data '
                  'channel.')  # type: List[ChannelParametersType]

    def __init__(self, RefChId=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        RefChId : str
        Parameters : List[ChannelParametersType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RefChId = RefChId
        self.Parameters = Parameters
        super(ChannelType, self).__init__(**kwargs)
