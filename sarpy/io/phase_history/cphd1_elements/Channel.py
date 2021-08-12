"""
The Channel definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List

from .base import DEFAULT_STRICT
from .blocks import POLARIZATION_TYPE, AreaType
from sarpy.io.xml.base import Serializable, SerializableArray, ParametersCollection
from sarpy.io.xml.descriptors import StringDescriptor, StringEnumDescriptor, StringListDescriptor, \
    IntegerDescriptor, FloatDescriptor, BooleanDescriptor, ParametersDescriptor, \
    SerializableDescriptor, SerializableListDescriptor, SerializableArrayDescriptor


class PolarizationType(Serializable):
    """
    Polarization(s) of the signals that formed the signal array.
    """

    _fields = ('TxPol', 'RcvPol')
    _required = _fields
    # descriptors
    TxPol = StringEnumDescriptor(
        'TxPol', POLARIZATION_TYPE, _required, strict=DEFAULT_STRICT,
        docstring='Transmitted signal polarization for the channel.')  # type: str
    RcvPol = StringEnumDescriptor(
        'RcvPol', POLARIZATION_TYPE, _required, strict=DEFAULT_STRICT,
        docstring='Receive polarization for the channel.')  # type: str

    def __init__(self, TxPol=None, RcvPol=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxPol = TxPol
        self.RcvPol = RcvPol
        super(PolarizationType, self).__init__(**kwargs)


class LFMEclipseType(Serializable):
    """
    The LFM Eclipse definition.
    """

    _fields = ('FxEarlyLow', 'FxEarlyHigh', 'FxLateLow', 'FxLateHigh')
    _required = _fields
    _numeric_format = {fld: '0.16G' for fld in _fields}
    # descriptors
    FxEarlyLow = FloatDescriptor(
        'FxEarlyLow', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring=r'FX domain minimum frequency value for an echo at '
                  r':math:`\Delta TOA = \Delta TOAE1 < \Delta TOA1`, in Hz.')  # type: float
    FxEarlyHigh = FloatDescriptor(
        'FxEarlyHigh', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='FX domain maximum frequency value for an echo at '
                  r':math:`\Delta TOA = \Delta TOAE1 < \Delta TOA1`, in Hz.')  # type: float
    FxLateLow = FloatDescriptor(
        'FxLateLow', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='FX domain minimum frequency value for an echo at '
                  r':math:`\Delta TOA = \Delta TOAE2 < \Delta TOA2`, in Hz.')  # type: float
    FxLateHigh = FloatDescriptor(
        'FxLateHigh', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='FX domain maximum frequency value for echo at '
                  r':math:`\Delta TOA = \Delta TOAE2 < \Delta TOA2`, in Hz.')  # type: float

    def __init__(self, FxEarlyLow=None, FxEarlyHigh=None, FxLateLow=None, FxLateHigh=None,
                 **kwargs):
        """

        Parameters
        ----------
        FxEarlyLow : float
        FxEarlyHigh : float
        FxLateLow : float
        FxLateHigh : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.FxEarlyLow = FxEarlyLow
        self.FxEarlyHigh = FxEarlyHigh
        self.FxLateLow = FxLateLow
        self.FxLateHigh = FxLateHigh
        super(LFMEclipseType, self).__init__(**kwargs)


class TOAExtendedType(Serializable):
    """
    The time-of-arrival (TOA) extended swath information.
    """

    _fields = ('TOAExtSaved', 'LFMEclipse')
    _required = ('TOAExtSaved', )
    _numeric_format = {'TOAExtSaved': '0.16G'}
    # descriptors
    TOAExtSaved = FloatDescriptor(
        'TOAExtSaved', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='TOA extended swath saved that includes both full and partially '
                  'eclipsed echoes.')  # type: float
    LFMEclipse = SerializableDescriptor(
        'LFMEclipse', LFMEclipseType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the FX domain signal content for partially '
                  'eclipsed echoes when the collection is performed with a Linear '
                  'FM waveform.')  # type: Union[None, LFMEclipseType]

    def __init__(self, TOAExtSaved=None, LFMEclipse=None, **kwargs):
        """

        Parameters
        ----------
        TOAExtSaved : float
        LFMEclipse : None|LFMEclipseType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TOAExtSaved = TOAExtSaved
        self.LFMEclipse = LFMEclipse
        super(TOAExtendedType, self).__init__(**kwargs)


class DwellTimesType(Serializable):
    """
    COD Time and Dwell Time polynomials over the image area.
    """

    _fields = ('CODId', 'DwellId')
    _required = _fields
    # descriptors
    CODId = StringDescriptor(
        'CODId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of the Center of Dwell Time polynomial that maps '
                  'reference surface position to COD time.')  # type: str
    DwellId = StringDescriptor(
        'DwellId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of the Dwell Time polynomial that maps reference '
                  'surface position to dwell time.')  # type: str

    def __init__(self, CODId=None, DwellId=None, **kwargs):
        """

        Parameters
        ----------
        CODId : str
        DwellId : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CODId = CODId
        self.DwellId = DwellId
        super(DwellTimesType, self).__init__(**kwargs)


class AntennaType(Serializable):
    """"
    Antenna Phase Center and Antenna Pattern identifiers for
    the antenna(s) used to collect and form the signal array data.
    """

    _fields = ('TxAPCId', 'TxAPATId', 'RcvAPCId', 'RcvAPATId')
    _required = _fields
    # descriptors
    TxAPCId = StringDescriptor(
        'TxAPCId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of Transmit APC to be used to compute the transmit '
                  'antenna pattern as a function of time for the channel.')  # type: str
    TxAPATId = StringDescriptor(
        'TxAPATId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of Transmit Antenna pattern used to form the channel '
                  'signal array.')  # type: str
    RcvAPCId = StringDescriptor(
        'RcvAPCId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of Receive APC to be used to compute the receive antenna '
                  'pattern as a function of time for the channel.')  # type: str
    RcvAPATId = StringDescriptor(
        'RcvAPATId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of Receive Antenna pattern used to form the '
                  'channel.')  # type: str

    def __init__(self, TxAPCId=None, TxAPATId=None, RcvAPCId=None, RcvAPATId=None, **kwargs):
        """

        Parameters
        ----------
        TxAPCId : str
        TxAPATId : str
        RcvAPCId : str
        RcvAPATId : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxAPCId = TxAPCId
        self.TxAPATId = TxAPATId
        self.RcvAPCId = RcvAPCId
        self.RcvAPATId = RcvAPATId
        super(AntennaType, self).__init__(**kwargs)


class TxRcvType(Serializable):
    """
    Parameters to identify the Transmit and Receive parameter sets used to collect the signal array.
    """

    _fields = ('TxWFId', 'RcvId')
    _required = _fields
    _collections_tags = {
        'TxWFId': {'array': False, 'child_tag': 'TxWFId'},
        'RcvId': {'array': False, 'child_tag': 'RcvId'}}
    # descriptors
    TxWFId = StringListDescriptor(
        'TxWFId', _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Identifier of the Transmit Waveform parameter set(s) that '
                  'were used.')  # type: List[str]
    RcvId = StringListDescriptor(
        'RcvId', _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Identifier of the Receive Parameter set(s) that were '
                  'used.')  # type: List[str]

    def __init__(self, TxWFId=None, RcvId=None, **kwargs):
        """

        Parameters
        ----------
        TxWFId : List[str]
        RcvId : List[str]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxWFId = TxWFId
        self.RcvId = RcvId
        super(TxRcvType, self).__init__(**kwargs)


class TgtRefLevelType(Serializable):
    """
    Signal level for an ideal point scatterer located at the SRP for reference
    signal vector.
    """

    _fields = ('PTRef', )
    _required = _fields
    _numeric_format = {'PTRef': '0.16G'}
    # descriptors
    PTRef = FloatDescriptor(
        'PTRef', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Target power level for a 1.0 square meter ideal point scatterer located '
                  'at the SRP. For FX Domain signal arrays, PTRef is the signal level at '
                  ':math:`fx = fx_C`. For TOA Domain, PTRef is the peak signal level at '
                  r':math:`\Delta TOA = 0`, and :math:`Power = |Signal|^2`.')  # type: float

    def __init__(self, PTRef=None, **kwargs):
        """

        Parameters
        ----------
        PTRef : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PTRef = PTRef
        super(TgtRefLevelType, self).__init__(**kwargs)


class FxPNPointType(Serializable):
    """
    Points that describe the noise profile.
    """

    _fields = ('Fx', 'PN')
    _required = _fields
    _numeric_format = {'FX': '0.16G', 'PN': '0.16G'}
    # descriptors
    Fx = FloatDescriptor(
        'Fx', _required, strict=DEFAULT_STRICT,
        docstring='Frequency value of this noise profile point, in Hz.')  # type: float
    PN = FloatDescriptor(
        'PN', _required, strict=DEFAULT_STRICT,
        docstring='Power level of this noise profile point.')  # type: float

    def __init__(self, Fx=None, PN=None, **kwargs):
        """

        Parameters
        ----------
        Fx : float
        PN : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Fx = Fx
        self.PN = PN
        super(FxPNPointType, self).__init__(**kwargs)


class FxNoiseProfileType(SerializableArray):
    _set_size = False
    _set_index = False


class NoiseLevelType(Serializable):
    """
    Thermal noise level for the reference signal vector.
    """

    _fields = ('PNRef', 'BNRef', 'FxNoiseProfile')
    _required = ('PNRef', 'BNRef')
    _collections_tags = {
        'FxNoiseProfile': {'array': True, 'child_tag': 'Point'}}
    _numeric_format = {'PNRef': '0.16G', 'BNRef': '0.16G'}
    # descriptors
    PNRef = FloatDescriptor(
        'PNRef', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Noise power level for thermal noise.')  # type: float
    BNRef = FloatDescriptor(
        'BNRef', _required, strict=DEFAULT_STRICT, bounds=(0, 1),
        docstring='Noise Equivalent BW for noise signal. Bandwidth BN is '
                  'expressed relative to the sample bandwidth.')  # type: float
    FxNoiseProfile = SerializableArrayDescriptor(
        'FxNoiseProfile', FxPNPointType, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=2, array_extension=FxNoiseProfileType,
        docstring='FX Domain Noise Level Profile. Power level for thermal noise (PN) vs. FX '
                  'frequency values.')  # type: Union[None, FxNoiseProfileType, List[FxPNPointType]]

    def __init__(self, PNRef=None, BNRef=None, FxNoiseProfile=None, **kwargs):
        """

        Parameters
        ----------
        PNRef : float
        BNRef : float
        FxNoiseProfile : FxNoiseProfileType|List[FxPNPointType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PNRef = PNRef
        self.BNRef = BNRef
        self.FxNoiseProfile = FxNoiseProfile
        super(NoiseLevelType, self).__init__(**kwargs)


class ChannelParametersType(Serializable):
    _fields = (
        'Identifier', 'RefVectorIndex', 'FXFixed', 'TOAFixed', 'SRPFixed',
        'SignalNormal', 'Polarization', 'FxC', 'FxBW', 'FxBWNoise', 'TOASaved',
        'TOAExtended', 'DwellTimes', 'ImageArea', 'Antenna', 'TxRcv',
        'TgtRefLevel', 'NoiseLevel')
    _required = (
        'Identifier', 'RefVectorIndex', 'FXFixed', 'TOAFixed', 'SRPFixed',
        'Polarization', 'FxC', 'FxBW', 'TOASaved', 'DwellTimes')
    _numeric_format = {
        'FxC': '0.16G', 'FxBW': '0.16G', 'FxBWNoise': '0.16G', 'TOASaved': '0.16G'}
    # descriptors
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this CPHD data channel.')  # type: str
    RefVectorIndex = IntegerDescriptor(
        'RefVectorIndex', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Index of the reference vector for the channel.')  # type: int
    FXFixed = BooleanDescriptor(
        'FXFixed', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate when a constant FX band is saved for all signal '
                  'vectors of the channel.')  # type: bool
    TOAFixed = BooleanDescriptor(
        'TOAFixed', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate when a constant TOA swath is saved for all '
                  'signal vectors of the channel.')  # type: bool
    SRPFixed = BooleanDescriptor(
        'SRPFixed', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate when a constant SRP position is used all '
                  'signal vectors of the channel.')  # type: bool
    SignalNormal = BooleanDescriptor(
        'SignalNormal', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate when all signal array vectors are normal. '
                  'Included if and only if the SIGNAL PVP is also included.')  # type: bool
    Polarization = SerializableDescriptor(
        'Polarization', PolarizationType, _required, strict=DEFAULT_STRICT,
        docstring='Polarization(s) of the signals that formed the signal '
                  'array.')  # type: PolarizationType
    FxC = FloatDescriptor(
        'FxC', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='FX center frequency value for saved bandwidth for the channel. '
                  'Computed from all vectors of the signal array.')  # type: float
    FxBW = FloatDescriptor(
        'FxBW', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='FX band spanned for the saved bandwidth for the channel. '
                  'Computed from all vectors of the signal array.')  # type: float
    FxBWNoise = FloatDescriptor(
        'FxBWNoise', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='FX signal bandwidth saved that includes noise signal below or '
                  'above the retained echo signal bandwidth.')  # type: float
    TOASaved = FloatDescriptor(
        'TOASaved', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='TOA swath saved for the full resolution echoes for the channel.')  # type: float
    TOAExtended = SerializableDescriptor(
        'TOAExtended', TOAExtendedType, _required, strict=DEFAULT_STRICT,
        docstring='TOA extended swath information.')  # type: Union[None, TOAExtendedType]
    DwellTimes = SerializableDescriptor(
        'DwellTimes', DwellTimesType, _required, strict=DEFAULT_STRICT,
        docstring='COD Time and Dwell Time polynomials over the image area.')  # type: DwellTimesType
    ImageArea = SerializableDescriptor(
        'ImageArea', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='Image Area for the CPHD channel defined by a rectangle aligned with '
                  '(IAX, IAY). May be reduced by the optional '
                  'polygon.')  # type: Union[None, AreaType]
    Antenna = SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna Phase Center and Antenna Pattern identifiers for the antenna(s) '
                  'used to collect and form the signal array data.')  # type: Union[None, AntennaType]
    TxRcv = SerializableDescriptor(
        'TxRcv', TxRcvType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters to identify the Transmit and Receive parameter sets '
                  'used to collect the signal array.')  # type: Union[None, TxRcvType]
    TgtRefLevel = SerializableDescriptor(
        'TgtRefLevel', TgtRefLevelType, _required, strict=DEFAULT_STRICT,
        docstring='Signal level for an ideal point scatterer located at the SRP for '
                  'reference signal vector.')  # type: Union[None, TgtRefLevelType]
    NoiseLevel = SerializableDescriptor(
        'NoiseLevel', NoiseLevelType, _required, strict=DEFAULT_STRICT,
        docstring='Thermal noise level for the reference signal '
                  'vector.')  # type: Union[None, NoiseLevelType]

    def __init__(self, Identifier=None, RefVectorIndex=None, FXFixed=None, TOAFixed=None,
                 SRPFixed=None, SignalNormal=None, Polarization=None, FxC=None, FxBW=None,
                 FxBWNoise=None, TOASaved=None, TOAExtended=None, DwellTimes=None,
                 ImageArea=None, Antenna=None, TxRcv=None, TgtRefLevel=None,
                 NoiseLevel=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        RefVectorIndex : int
        FXFixed : bool
        TOAFixed : bool
        SRPFixed : bool
        SignalNormal : None|bool
        Polarization : PolarizationType
        FxC : float
        FxBW : float
        FxBWNoise : None|float
        TOASaved : float
        TOAExtended : None|TOAExtendedType
        DwellTimes : DwellTimesType
        ImageArea : None|AreaType
        Antenna : None|AntennaType
        TxRcv : None|TxRcvType
        TgtRefLevel : None|TgtRefLevelType
        NoiseLevel : None|NoiseLevelType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.RefVectorIndex = RefVectorIndex
        self.FXFixed = FXFixed
        self.TOAFixed = TOAFixed
        self.SRPFixed = SRPFixed
        self.SignalNormal = SignalNormal
        self.Polarization = Polarization
        self.FxC = FxC
        self.FxBW = FxBW
        self.FxBWNoise = FxBWNoise
        self.TOASaved = TOASaved
        self.TOAExtended = TOAExtended
        self.DwellTimes = DwellTimes
        self.ImageArea = ImageArea
        self.Antenna = Antenna
        self.TxRcv = TxRcv
        self.TgtRefLevel = TgtRefLevel
        self.NoiseLevel = NoiseLevel
        super(ChannelParametersType, self).__init__(**kwargs)


class ChannelType(Serializable):
    """
    The channel definition.
    """

    _fields = (
        'RefChId', 'FXFixedCPHD', 'TOAFixedCPHD', 'SRPFixedCPHD',
        'Parameters', 'AddedParameters')
    _required = (
        'RefChId', 'FXFixedCPHD', 'TOAFixedCPHD', 'SRPFixedCPHD', 'Parameters')
    _collections_tags = {
        'Parameters': {'array': False, 'child_tag': 'Parameters'},
        'AddedParameters': {'array': False, 'child_tag': 'AddedParameters'}}
    # descriptors
    RefChId = StringDescriptor(
        'RefChId', _required, strict=DEFAULT_STRICT,
        docstring='Channel ID for the Reference Channel in the '
                  'product.')  # type: str
    FXFixedCPHD = BooleanDescriptor(
        'FXFixedCPHD', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate when a constant FX band is saved for all '
                  'signal vectors of all channels.')  # type: bool
    TOAFixedCPHD = BooleanDescriptor(
        'TOAFixedCPHD', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate when a constant TOA swath is saved for all '
                  'signal vectors of all channels.')  # type: bool
    SRPFixedCPHD = BooleanDescriptor(
        'SRPFixedCPHD', _required, strict=DEFAULT_STRICT,
        docstring='Flag to indicate when a constant SRP position is used all '
                  'signal vectors of all channels.')  # type: bool
    Parameters = SerializableListDescriptor(
        'Parameters', ChannelParametersType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameter Set that describes a CPHD data '
                  'channel.')  # type: List[ChannelParametersType]
    AddedParameters = ParametersDescriptor(
        'AddedParameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Additional free form parameters.')  # type: Union[None, ParametersCollection]

    def __init__(self, RefChId=None, FXFixedCPHD=None, TOAFixedCPHD=None,
                 SRPFixedCPHD=None, Parameters=None, AddedParameters=None, **kwargs):
        """

        Parameters
        ----------
        RefChId : str
        FXFixedCPHD : bool
        TOAFixedCPHD : bool
        SRPFixedCPHD : bool
        Parameters : List[ChannelParametersType]
        AddedParameters
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RefChId = RefChId
        self.FXFixedCPHD = FXFixedCPHD
        self.TOAFixedCPHD = TOAFixedCPHD
        self.SRPFixedCPHD = SRPFixedCPHD
        self.Parameters = Parameters
        self.AddedParameters = AddedParameters
        super(ChannelType, self).__init__(**kwargs)
