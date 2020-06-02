# -*- coding: utf-8 -*-
"""
The Channel definition.
"""

from typing import Union, List

from .base import DEFAULT_STRICT
from .blocks import POLARIZATION_TYPE, AreaType
# noinspection PyProtectedMember
from ..sicd_elements.base import Serializable, _StringDescriptor, _StringEnumDescriptor, \
    _IntegerDescriptor, _SerializableListDescriptor, _ParametersDescriptor, ParametersCollection, \
    _BooleanDescriptor, _FloatDescriptor, _SerializableDescriptor, _StringListDescriptor, \
    SerializableArray, _SerializableArrayDescriptor

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class PolarizationType(Serializable):
    """
    The polarization definition.
    """

    _fields = ('TxPol', 'RcvPol')
    _required = _fields
    # descriptors
    TxPol = _StringEnumDescriptor(
        'TxPol', POLARIZATION_TYPE, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    RcvPol = _StringEnumDescriptor(
        'RcvPol', POLARIZATION_TYPE, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

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
    # descriptors
    FxEarlyLow = _FloatDescriptor(
        'FxEarlyLow', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    FxEarlyHigh = _FloatDescriptor(
        'FxEarlyHigh', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    FxLateLow = _FloatDescriptor(
        'FxLateLow', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    FxLateHigh = _FloatDescriptor(
        'FxLateHigh', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float

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
    _fields = ('TOAExtSaved', 'LFMEclipse')
    _required = ('TOAExtSaved', )
    # descriptors
    TOAExtSaved = _FloatDescriptor(
        'TOAExtSaved', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    LFMEclipse = _SerializableDescriptor(
        'LFMEclipse', LFMEclipseType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, LFMEclipseType]

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
    _fields = ('CODId', 'DwellId')
    _required = _fields
    # descriptors
    CODId = _StringDescriptor(
        'CODId', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    DwellId = _StringDescriptor(
        'DwellId', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

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
    _fields = ('TxAPCId', 'TxAPATId', 'RcvAPCId', 'RcvAPATId')
    _required = _fields
    # descriptors
    TxAPCId = _StringDescriptor(
        'TxAPCId', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    TxAPATId = _StringDescriptor(
        'TxAPATId', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    RcvAPCId = _StringDescriptor(
        'RcvAPCId', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    RcvAPATId = _StringDescriptor(
        'RcvAPATId', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str

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
    _fields = ('TxWFId', 'RcvId')
    _required = _fields
    _collections_tags = {
        'TxWFId': {'array': False, 'child_tag': 'TxWFId'},
        'RcvId': {'array': False, 'child_tag': 'RcvId'}}
    # descriptors
    TxWFId = _StringListDescriptor(
        'TxWFId', _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='')  # type: List[str]
    RcvId = _StringListDescriptor(
        'RcvId', _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='')  # type: List[str]

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
    _fields = ('PTRef', )
    _required = _fields
    # descriptors
    PTRef = _FloatDescriptor(
        'PTRef', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float

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
    _fields = ('Fx', 'PN')
    _required = _fields
    # descriptors
    Fx = _FloatDescriptor(
        'Fx', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: float
    PN = _FloatDescriptor(
        'PN', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: float

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
    _fields = ('PNRef', 'BNRef', 'FxNoiseProfile')
    _required = ('PNRef', 'BNRef')
    _collections_tags = {
        'FxNoiseProfile': {'array': True, 'child_tag': 'Point'}}
    # descriptors
    PNRef = _FloatDescriptor(
        'PNRef', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    BNRef = _FloatDescriptor(
        'BNRef', _required, strict=DEFAULT_STRICT, bounds=(0, 1),
        docstring='')  # type: float
    FxNoiseProfile = _SerializableArrayDescriptor(
        'FxNoiseProfile', FxPNPointType, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=2, array_extension=FxNoiseProfileType,
        docstring='')  # type: Union[None, FxNoiseProfileType, List[FxPNPointType]]

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
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    RefVectorIndex = _IntegerDescriptor(
        'RefVectorIndex', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: int
    FXFixed = _BooleanDescriptor(
        'FXFixed', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: bool
    TOAFixed = _BooleanDescriptor(
        'TOAFixed', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: bool
    SRPFixed = _BooleanDescriptor(
        'SRPFixed', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: bool
    SignalNormal = _BooleanDescriptor(
        'SignalNormal', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: bool
    Polarization = _SerializableDescriptor(
        'Polarization', PolarizationType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: PolarizationType
    FxC = _FloatDescriptor(
        'FxC', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    FxBW = _FloatDescriptor(
        'FxBW', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    FxBWNoise = _FloatDescriptor(
        'FxBWNoise', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    TOASaved = _FloatDescriptor(
        'TOASaved', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='')  # type: float
    TOAExtended = _SerializableDescriptor(
        'TOAExtended', TOAExtendedType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, TOAExtendedType]
    DwellTimes = _SerializableDescriptor(
        'DwellTimes', DwellTimesType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: DwellTimesType
    ImageArea = _SerializableDescriptor(
        'ImageArea', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, AreaType]
    Antenna = _SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, AntennaType]
    TxRcv = _SerializableDescriptor(
        'TxRcv', TxRcvType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, TxRcvType]
    TgtRefLevel = _SerializableDescriptor(
        'TgtRefLevel', TgtRefLevelType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, TgtRefLevelType]
    NoiseLevel = _SerializableDescriptor(
        'NoiseLevel', NoiseLevelType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, NoiseLevelType]

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
        'AddedParameters': {'array': False, 'child_tag': 'Parameters'}}
    # descriptors
    RefChId = _StringDescriptor(
        'RefChId', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    FXFixedCPHD = _BooleanDescriptor(
        'FXFixedCPHD', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: bool
    TOAFixedCPHD = _BooleanDescriptor(
        'TOAFixedCPHD', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: bool
    SRPFixedCPHD = _BooleanDescriptor(
        'SRPFixedCPHD', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: bool
    Parameters = _SerializableListDescriptor(
        'Parameters', ChannelParametersType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: List[ChannelParametersType]
    AddedParameters = _ParametersDescriptor(
        'AddedParameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, ParametersCollection]

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
