# -*- coding: utf-8 -*-
"""
The ErrorStatisticsType definition.
"""

from .base import Serializable, DEFAULT_STRICT, _StringEnumDescriptor, _FloatDescriptor, \
    _SerializableDescriptor, _ParametersDescriptor, ParametersCollection
from .blocks import ErrorDecorrFuncType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CompositeSCPErrorType(Serializable):
    """
    Composite error statistics for the Scene Center Point. Slant plane range *(Rg)*
    and azimuth *(Az)* error statistics. Slant plane defined at *SCP COA*.
    """
    _fields = ('Rg', 'Az', 'RgAz')
    _required = _fields
    _numeric_format = {key: '0.16G' for key in _fields}
    # descriptors
    Rg = _FloatDescriptor(
        'Rg', _required, strict=DEFAULT_STRICT,
        docstring='Estimated range error standard deviation.')  # type: float
    Az = _FloatDescriptor(
        'Az', _required, strict=DEFAULT_STRICT,
        docstring='Estimated azimuth error standard deviation.')  # type: float
    RgAz = _FloatDescriptor(
        'RgAz', _required, strict=DEFAULT_STRICT,
        docstring='Estimated range and azimuth error correlation coefficient.')  # type: float

    def __init__(self, Rg=None, Az=None, RgAz=None, **kwargs):
        """

        Parameters
        ----------
        Rg : float
        Az : float
        RgAz : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Rg, self.Az, self.RgAz = Rg, Az, RgAz
        super(CompositeSCPErrorType, self).__init__(**kwargs)


class CorrCoefsType(Serializable):
    """Correlation Coefficient parameters."""
    _fields = (
        'P1P2', 'P1P3', 'P1V1', 'P1V2', 'P1V3', 'P2P3', 'P2V1', 'P2V2', 'P2V3',
        'P3V1', 'P3V2', 'P3V3', 'V1V2', 'V1V3', 'V2V3')
    _required = _fields
    _numeric_format = {key: '0.16G' for key in _fields}
    # descriptors
    P1P2 = _FloatDescriptor(
        'P1P2', _required, strict=DEFAULT_STRICT, docstring='`P1` and `P2` correlation coefficient.')  # type: float
    P1P3 = _FloatDescriptor(
        'P1P3', _required, strict=DEFAULT_STRICT, docstring='`P1` and `P3` correlation coefficient.')  # type: float
    P1V1 = _FloatDescriptor(
        'P1V1', _required, strict=DEFAULT_STRICT, docstring='`P1` and `V1` correlation coefficient.')  # type: float
    P1V2 = _FloatDescriptor(
        'P1V2', _required, strict=DEFAULT_STRICT, docstring='`P1` and `V2` correlation coefficient.')  # type: float
    P1V3 = _FloatDescriptor(
        'P1V3', _required, strict=DEFAULT_STRICT, docstring='`P1` and `V3` correlation coefficient.')  # type: float
    P2P3 = _FloatDescriptor(
        'P2P3', _required, strict=DEFAULT_STRICT, docstring='`P2` and `P3` correlation coefficient.')  # type: float
    P2V1 = _FloatDescriptor(
        'P2V1', _required, strict=DEFAULT_STRICT, docstring='`P2` and `V1` correlation coefficient.')  # type: float
    P2V2 = _FloatDescriptor(
        'P2V2', _required, strict=DEFAULT_STRICT, docstring='`P2` and `V2` correlation coefficient.')  # type: float
    P2V3 = _FloatDescriptor(
        'P2V3', _required, strict=DEFAULT_STRICT, docstring='`P2` and `V3` correlation coefficient.')  # type: float
    P3V1 = _FloatDescriptor(
        'P3V1', _required, strict=DEFAULT_STRICT, docstring='`P3` and `V1` correlation coefficient.')  # type: float
    P3V2 = _FloatDescriptor(
        'P3V2', _required, strict=DEFAULT_STRICT, docstring='`P3` and `V2` correlation coefficient.')  # type: float
    P3V3 = _FloatDescriptor(
        'P3V3', _required, strict=DEFAULT_STRICT, docstring='`P3` and `V3` correlation coefficient.')  # type: float
    V1V2 = _FloatDescriptor(
        'V1V2', _required, strict=DEFAULT_STRICT, docstring='`V1` and `V2` correlation coefficient.')  # type: float
    V1V3 = _FloatDescriptor(
        'V1V3', _required, strict=DEFAULT_STRICT, docstring='`V1` and `V3` correlation coefficient.')  # type: float
    V2V3 = _FloatDescriptor(
        'V2V3', _required, strict=DEFAULT_STRICT, docstring='`V2` and `V3` correlation coefficient.')  # type: float

    def __init__(self, P1P2=None, P1P3=None, P1V1=None, P1V2=None, P1V3=None,
                 P2P3=None, P2V1=None, P2V2=None, P2V3=None,
                 P3V1=None, P3V2=None, P3V3=None,
                 V1V2=None, V1V3=None,
                 V2V3=None, **kwargs):
        """

        Parameters
        ----------
        P1P2 : float
        P1P3 : float
        P1V1 : float
        P1V2 : float
        P1V3 : float
        P2P3 : float
        P2V1 : float
        P2V2 : float
        P2V3 : float
        P3V1 : float
        P3V2 : float
        P3V3 : float
        V1V2 : float
        V1V3 : float
        V2V3 : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.P1P2, self.P1P3, self.P1V1, self.P1V2, self.P1V3 = P1P2, P1P3, P1V1, P1V2, P1V3
        self.P2P3, self.P2V1, self.P2V2, self.P2V3 = P2P3, P2V1, P2V2, P2V3
        self.P3V1, self.P3V2, self.P3V3 = P3V1, P3V2, P3V3
        self.V1V2, self.V1V3 = V1V2, V1V3
        self.V2V3 = V2V3
        super(CorrCoefsType, self).__init__(**kwargs)


class PosVelErrType(Serializable):
    """
    Position and velocity error statistics for the radar platform.
    """

    _fields = ('Frame', 'P1', 'P2', 'P3', 'V1', 'V2', 'V3', 'CorrCoefs', 'PositionDecorr')
    _required = ('Frame', 'P1', 'P2', 'P3', 'V1', 'V2', 'V3')
    _numeric_format = {'P1': '0.16G', 'P2': '0.16G', 'P3': '0.16G', 'V1': '0.16G', 'V2': '0.16G', 'V3': '0.16G'}
    # class variables
    _FRAME_VALUES = ('ECF', 'RIC_ECF', 'RIC_ECI')
    # descriptors
    Frame = _StringEnumDescriptor(
        'Frame', _FRAME_VALUES, _required, strict=True,
        docstring='Coordinate frame used for expressing P,V errors statistics. Note: '
                  '*RIC = Radial, In-Track, Cross-Track*, where radial is defined to be from earth center through '
                  'the platform position.')  # type: str
    P1 = _FloatDescriptor(
        'P1', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 1 standard deviation.')  # type: float
    P2 = _FloatDescriptor(
        'P2', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 2 standard deviation.')  # type: float
    P3 = _FloatDescriptor(
        'P3', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 3 standard deviation.')  # type: float
    V1 = _FloatDescriptor(
        'V1', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 1 standard deviation.')  # type: float
    V2 = _FloatDescriptor(
        'V2', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 2 standard deviation.')  # type: float
    V3 = _FloatDescriptor(
        'V3', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 3 standard deviation.')  # type: float
    CorrCoefs = _SerializableDescriptor(
        'CorrCoefs', CorrCoefsType, _required, strict=DEFAULT_STRICT,
        docstring='Correlation Coefficient parameters.')  # type: CorrCoefsType
    PositionDecorr = _SerializableDescriptor(
        'PositionDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Platform position error decorrelation function.')  # type: ErrorDecorrFuncType

    def __init__(self, Frame=None, P1=None, P2=None, P3=None, V1=None, V2=None, V3=None,
                 CorrCoefs=None, PositionDecorr=None, **kwargs):
        """

        Parameters
        ----------
        Frame : str
        P1 : float
        P2 : float
        P3 : float
        V1 : float
        V2 : float
        V3 : float
        CorrCoefs : CorrCoefsType
        PositionDecorr : ErrorDecorrFuncType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Frame = Frame
        self.P1, self.P2, self.P3 = P1, P2, P3
        self.V1, self.V2, self.V3 = V1, V2, V3
        self.CorrCoefs, self.PositionDecorr = CorrCoefs, PositionDecorr
        super(PosVelErrType, self).__init__(**kwargs)


class RadarSensorErrorType(Serializable):
    """Radar sensor error statistics."""
    _fields = ('RangeBias', 'ClockFreqSF', 'TransmitFreqSF', 'RangeBiasDecorr')
    _required = ('RangeBias', )
    _numeric_format = {'RangeBias': '0.16G', 'ClockFreqSF': '0.16G', 'TransmitFreqSF': '0.16G'}
    # descriptors
    RangeBias = _FloatDescriptor(
        'RangeBias', _required, strict=DEFAULT_STRICT,
        docstring='Range bias error standard deviation.')  # type: float
    ClockFreqSF = _FloatDescriptor(
        'ClockFreqSF', _required, strict=DEFAULT_STRICT,
        docstring='Payload clock frequency scale factor standard deviation, '
                  r'where :math:`SF = (\Delta f)/f_0`.')  # type: float
    TransmitFreqSF = _FloatDescriptor(
        'TransmitFreqSF', _required, strict=DEFAULT_STRICT,
        docstring='Transmit frequency scale factor standard deviation, '
                  r'where :math:`SF = (\Delta f)/f_0`.')  # type: float
    RangeBiasDecorr = _SerializableDescriptor(
        'RangeBiasDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Range bias decorrelation rate.')  # type: ErrorDecorrFuncType

    def __init__(self, RangeBias=None, ClockFreqSF=None, TransmitFreqSF=None, RangeBiasDecorr=None, **kwargs):
        """

        Parameters
        ----------
        RangeBias : float
        ClockFreqSF : float
        TransmitFreqSF : float
        RangeBiasDecorr : ErrorDecorrFuncType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RangeBias, self.ClockFreqSF, self.TransmitFreqSF = RangeBias, ClockFreqSF, TransmitFreqSF
        self.RangeBiasDecorr = RangeBiasDecorr
        super(RadarSensorErrorType, self).__init__(**kwargs)


class TropoErrorType(Serializable):
    """Troposphere delay error statistics."""
    _fields = ('TropoRangeVertical', 'TropoRangeSlant', 'TropoRangeDecorr')
    _required = ()
    _numeric_format = {'TropoRangeVertical': '0.16G', 'TropoRangeSlant': '0.16G'}
    # descriptors
    TropoRangeVertical = _FloatDescriptor(
        'TropoRangeVertical', _required, strict=DEFAULT_STRICT,
        docstring='Troposphere two-way delay error for normal incidence standard deviation. '
                  r'Expressed as a range error. :math:`(\Delta R) = (\Delta T) \cdot (c/2)`.')  # type: float
    TropoRangeSlant = _FloatDescriptor(
        'TropoRangeSlant', _required, strict=DEFAULT_STRICT,
        docstring='Troposphere two-way delay error for the *SCP* line of sight at *COA* standard deviation. '
                  r'Expressed as a range error. :math:`(\Delta R) = (\Delta T) \cdot (c/2)`.')  # type: float
    TropoRangeDecorr = _SerializableDescriptor(
        'TropoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Troposphere range error decorrelation function.')  # type: ErrorDecorrFuncType

    def __init__(self, TropoRangeVertical=None, TropoRangeSlant=None, TropoRangeDecorr=None, **kwargs):
        """

        Parameters
        ----------
        TropoRangeVertical : float
        TropoRangeSlant : float
        TropoRangeDecorr : ErrorDecorrFuncType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TropoRangeVertical, self.TropoRangeSlant = TropoRangeVertical, TropoRangeSlant
        self.TropoRangeDecorr = TropoRangeDecorr
        super(TropoErrorType, self).__init__(**kwargs)


class IonoErrorType(Serializable):
    """Ionosphere delay error statistics."""
    _fields = ('IonoRangeVertical', 'IonoRangeSlant', 'IonoRgRgRateCC', 'IonoRangeDecorr')
    _required = ('IonoRgRgRateCC', )
    _numeric_format = {'IonoRangeVertical': '0.16G', 'IonoRangeSlant': '0.16G', 'IonoRgRgRateCC': '0.16G'}
    # descriptors
    IonoRangeVertical = _FloatDescriptor(
        'IonoRangeVertical', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere two-way delay error for normal incidence standard deviation. '
                  r'Expressed as a range error. :math:`(\Delta R) = (\Delta T) \cdot (c/2)`.')  # type: float
    IonoRangeSlant = _FloatDescriptor(
        'IonoRangeSlant', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere two-way delay rate of change error for normal incidence standard deviation. '
                  r'Expressed as a range rate error. :math:`(\Delta \dot{R}) = (\Delta \dot{T}) \cdot (c/2)`.')  # type: float
    IonoRgRgRateCC = _FloatDescriptor(
        'IonoRgRgRateCC', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error and range rate error correlation coefficient.')  # type: float
    IonoRangeDecorr = _SerializableDescriptor(
        'IonoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error decorrelation rate.')  # type: ErrorDecorrFuncType

    def __init__(self, IonoRangeVertical=None, IonoRangeSlant=None,
                 IonoRgRgRateCC=None, IonoRangeDecorr=None, **kwargs):
        """

        Parameters
        ----------
        IonoRangeVertical : float
        IonoRangeSlant : float
        IonoRgRgRateCC : float
        IonoRangeDecorr : ErrorDecorrFuncType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.IonoRangeVertical, self.IonoRangeSlant = IonoRangeVertical, IonoRangeSlant
        self.IonoRgRgRateCC = IonoRgRgRateCC
        self.IonoRangeDecorr = IonoRangeDecorr
        super(IonoErrorType, self).__init__(**kwargs)


class ErrorComponentsType(Serializable):
    """Error statistics by components."""
    _fields = ('PosVelErr', 'RadarSensor', 'TropoError', 'IonoError')
    _required = ('PosVelErr', 'RadarSensor')
    # descriptors
    PosVelErr = _SerializableDescriptor(
        'PosVelErr', PosVelErrType, _required, strict=DEFAULT_STRICT,
        docstring='Position and velocity error statistics for the radar platform.')  # type: PosVelErrType
    RadarSensor = _SerializableDescriptor(
        'RadarSensor', RadarSensorErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Radar sensor error statistics.')  # type: RadarSensorErrorType
    TropoError = _SerializableDescriptor(
        'TropoError', TropoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Troposphere delay error statistics.')  # type: TropoErrorType
    IonoError = _SerializableDescriptor(
        'IonoError', IonoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere delay error statistics.')  # type: IonoErrorType

    def __init__(self, PosVelErr=None, RadarSensor=None, TropoError=None, IonoError=None, **kwargs):
        """

        Parameters
        ----------
        PosVelErr : PosVelErrType
        RadarSensor : RadarSensorErrorType
        TropoError : TropoErrorType
        IonoError : IonoErrorType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PosVelErr, self.RadarSensor = PosVelErr, RadarSensor
        self.TropoError, self.IonoError = TropoError, IonoError
        super(ErrorComponentsType, self).__init__(**kwargs)


class ErrorStatisticsType(Serializable):
    """Parameters used to compute error statistics within the SICD sensor model."""
    _fields = ('CompositeSCP', 'Components', 'AdditionalParms')
    _required = ()
    _collections_tags = {'AdditionalParms': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    CompositeSCP = _SerializableDescriptor(
        'CompositeSCP', CompositeSCPErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Composite error statistics for the scene center point. *Slant plane range (Rg)* and *azimuth (Az)* '
                  'error statistics. Slant plane defined at *Scene Center Point, Center of Azimuth (SCP COA)*.'
    )  # type: CompositeSCPErrorType
    Components = _SerializableDescriptor(
        'Components', ErrorComponentsType, _required, strict=DEFAULT_STRICT,
        docstring='Error statistics by components.')  # type: ErrorComponentsType

    AdditionalParms = _ParametersDescriptor(
        'AdditionalParms', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Any additional parameters.')  # type: ParametersCollection

    def __init__(self, CompositeSCP=None, Components=None, AdditionalParms=None, **kwargs):
        """

        Parameters
        ----------
        CompositeSCP : CompositeSCPErrorType
        Components : ErrorComponentsType
        AdditionalParms : ParametersCollection|dict
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CompositeSCP = CompositeSCP
        self.Components = Components
        self.AdditionalParms = AdditionalParms
        super(ErrorStatisticsType, self).__init__(**kwargs)
