"""
The ErrorStatisticsType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Union, Optional, Dict
from sarpy.io.xml.base import Serializable, ParametersCollection
from sarpy.io.xml.descriptors import StringEnumDescriptor, FloatDescriptor, \
    SerializableDescriptor, ParametersDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import ErrorDecorrFuncType


class CompositeSCPErrorType(Serializable):
    """
    Composite error statistics for the Scene Center Point. Slant plane range *(Rg)*
    and azimuth *(Az)* error statistics. Slant plane defined at *SCP COA*.
    """
    _fields = ('Rg', 'Az', 'RgAz')
    _required = _fields
    _numeric_format = {key: FLOAT_FORMAT for key in _fields}
    # descriptors
    Rg = FloatDescriptor(
        'Rg', _required, strict=DEFAULT_STRICT,
        docstring='Estimated range error standard deviation.')  # type: float
    Az = FloatDescriptor(
        'Az', _required, strict=DEFAULT_STRICT,
        docstring='Estimated azimuth error standard deviation.')  # type: float
    RgAz = FloatDescriptor(
        'RgAz', _required, strict=DEFAULT_STRICT,
        docstring='Estimated range and azimuth error correlation coefficient.')  # type: float

    def __init__(
            self,
            Rg: float = None,
            Az: float = None,
            RgAz: float = None,
            **kwargs):
        """

        Parameters
        ----------
        Rg : float
        Az : float
        RgAz : float
        kwargs
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
    _numeric_format = {key: FLOAT_FORMAT for key in _fields}
    # descriptors
    P1P2 = FloatDescriptor(
        'P1P2', _required, strict=DEFAULT_STRICT, docstring='`P1` and `P2` correlation coefficient.')  # type: float
    P1P3 = FloatDescriptor(
        'P1P3', _required, strict=DEFAULT_STRICT, docstring='`P1` and `P3` correlation coefficient.')  # type: float
    P1V1 = FloatDescriptor(
        'P1V1', _required, strict=DEFAULT_STRICT, docstring='`P1` and `V1` correlation coefficient.')  # type: float
    P1V2 = FloatDescriptor(
        'P1V2', _required, strict=DEFAULT_STRICT, docstring='`P1` and `V2` correlation coefficient.')  # type: float
    P1V3 = FloatDescriptor(
        'P1V3', _required, strict=DEFAULT_STRICT, docstring='`P1` and `V3` correlation coefficient.')  # type: float
    P2P3 = FloatDescriptor(
        'P2P3', _required, strict=DEFAULT_STRICT, docstring='`P2` and `P3` correlation coefficient.')  # type: float
    P2V1 = FloatDescriptor(
        'P2V1', _required, strict=DEFAULT_STRICT, docstring='`P2` and `V1` correlation coefficient.')  # type: float
    P2V2 = FloatDescriptor(
        'P2V2', _required, strict=DEFAULT_STRICT, docstring='`P2` and `V2` correlation coefficient.')  # type: float
    P2V3 = FloatDescriptor(
        'P2V3', _required, strict=DEFAULT_STRICT, docstring='`P2` and `V3` correlation coefficient.')  # type: float
    P3V1 = FloatDescriptor(
        'P3V1', _required, strict=DEFAULT_STRICT, docstring='`P3` and `V1` correlation coefficient.')  # type: float
    P3V2 = FloatDescriptor(
        'P3V2', _required, strict=DEFAULT_STRICT, docstring='`P3` and `V2` correlation coefficient.')  # type: float
    P3V3 = FloatDescriptor(
        'P3V3', _required, strict=DEFAULT_STRICT, docstring='`P3` and `V3` correlation coefficient.')  # type: float
    V1V2 = FloatDescriptor(
        'V1V2', _required, strict=DEFAULT_STRICT, docstring='`V1` and `V2` correlation coefficient.')  # type: float
    V1V3 = FloatDescriptor(
        'V1V3', _required, strict=DEFAULT_STRICT, docstring='`V1` and `V3` correlation coefficient.')  # type: float
    V2V3 = FloatDescriptor(
        'V2V3', _required, strict=DEFAULT_STRICT, docstring='`V2` and `V3` correlation coefficient.')  # type: float

    def __init__(
            self,
            P1P2: float = None,
            P1P3: float = None,
            P1V1: float = None,
            P1V2: float = None,
            P1V3: float = None,
            P2P3: float = None,
            P2V1: float = None,
            P2V2: float = None,
            P2V3: float = None,
            P3V1: float = None,
            P3V2: float = None,
            P3V3: float = None,
            V1V2: float = None,
            V1V3: float = None,
            V2V3: float = None,
            **kwargs):
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
        kwargs
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
    _numeric_format = {
        'P1': FLOAT_FORMAT, 'P2': FLOAT_FORMAT, 'P3': FLOAT_FORMAT,
        'V1': FLOAT_FORMAT, 'V2': FLOAT_FORMAT, 'V3': FLOAT_FORMAT}
    # class variables
    _FRAME_VALUES = ('ECF', 'RIC_ECF', 'RIC_ECI')
    # descriptors
    Frame = StringEnumDescriptor(
        'Frame', _FRAME_VALUES, _required, strict=True,
        docstring='Coordinate frame used for expressing P,V errors statistics. Note: '
                  '*RIC = Radial, In-Track, Cross-Track*, where radial is defined to be from earth center through '
                  'the platform position.')  # type: str
    P1 = FloatDescriptor(
        'P1', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 1 standard deviation.')  # type: float
    P2 = FloatDescriptor(
        'P2', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 2 standard deviation.')  # type: float
    P3 = FloatDescriptor(
        'P3', _required, strict=DEFAULT_STRICT, docstring='Position coordinate 3 standard deviation.')  # type: float
    V1 = FloatDescriptor(
        'V1', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 1 standard deviation.')  # type: float
    V2 = FloatDescriptor(
        'V2', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 2 standard deviation.')  # type: float
    V3 = FloatDescriptor(
        'V3', _required, strict=DEFAULT_STRICT, docstring='Velocity coordinate 3 standard deviation.')  # type: float
    CorrCoefs = SerializableDescriptor(
        'CorrCoefs', CorrCoefsType, _required, strict=DEFAULT_STRICT,
        docstring='Correlation Coefficient parameters.')  # type: Optional[CorrCoefsType]
    PositionDecorr = SerializableDescriptor(
        'PositionDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Platform position error decorrelation function.')  # type: Optional[ErrorDecorrFuncType]

    def __init__(
            self,
            Frame: str = None,
            P1: float = None,
            P2: float = None,
            P3: float = None,
            V1: float = None,
            V2: float = None,
            V3: float = None,
            CorrCoefs: Optional[CorrCoefsType] = None,
            PositionDecorr: Optional[ErrorDecorrFuncType] = None,
            **kwargs):
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
        CorrCoefs : None|CorrCoefsType
        PositionDecorr : None|ErrorDecorrFuncType
        kwargs
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
    _numeric_format = {'RangeBias': FLOAT_FORMAT, 'ClockFreqSF': FLOAT_FORMAT, 'TransmitFreqSF': FLOAT_FORMAT}
    # descriptors
    RangeBias = FloatDescriptor(
        'RangeBias', _required, strict=DEFAULT_STRICT,
        docstring='Range bias error standard deviation.')  # type: float
    ClockFreqSF = FloatDescriptor(
        'ClockFreqSF', _required, strict=DEFAULT_STRICT,
        docstring='Payload clock frequency scale factor standard deviation, '
                  r'where :math:`SF = (\Delta f)/f_0`.')  # type: float
    TransmitFreqSF = FloatDescriptor(
        'TransmitFreqSF', _required, strict=DEFAULT_STRICT,
        docstring='Transmit frequency scale factor standard deviation, '
                  r'where :math:`SF = (\Delta f)/f_0`.')  # type: float
    RangeBiasDecorr = SerializableDescriptor(
        'RangeBiasDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Range bias decorrelation rate.')  # type: ErrorDecorrFuncType

    def __init__(
            self,
            RangeBias: float = None,
            ClockFreqSF: Optional[float] = None,
            TransmitFreqSF: Optional[float] = None,
            RangeBiasDecorr: Optional[ErrorDecorrFuncType] = None,
            **kwargs):
        """

        Parameters
        ----------
        RangeBias : float
        ClockFreqSF : None|float
        TransmitFreqSF : None|float
        RangeBiasDecorr : None|ErrorDecorrFuncType
        kwargs
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
    _numeric_format = {'TropoRangeVertical': FLOAT_FORMAT, 'TropoRangeSlant': FLOAT_FORMAT}
    # descriptors
    TropoRangeVertical = FloatDescriptor(
        'TropoRangeVertical', _required, strict=DEFAULT_STRICT,
        docstring='Troposphere two-way delay error for normal incidence standard deviation. '
                  r'Expressed as a range error. :math:`(\Delta R) = (\Delta T) \cdot (c/2)`.')  # type: float
    TropoRangeSlant = FloatDescriptor(
        'TropoRangeSlant', _required, strict=DEFAULT_STRICT,
        docstring='Troposphere two-way delay error for the *SCP* line of sight at *COA* standard deviation. '
                  r'Expressed as a range error. :math:`(\Delta R) = (\Delta T) \cdot (c/2)`.')  # type: float
    TropoRangeDecorr = SerializableDescriptor(
        'TropoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Troposphere range error decorrelation function.')  # type: ErrorDecorrFuncType

    def __init__(
            self,
            TropoRangeVertical: Optional[float] = None,
            TropoRangeSlant: Optional[float] = None,
            TropoRangeDecorr: Optional[ErrorDecorrFuncType] = None,
            **kwargs):
        """

        Parameters
        ----------
        TropoRangeVertical : None|float
        TropoRangeSlant : None|float
        TropoRangeDecorr : None|ErrorDecorrFuncType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TropoRangeVertical = TropoRangeVertical
        self.TropoRangeSlant = TropoRangeSlant
        self.TropoRangeDecorr = TropoRangeDecorr
        super(TropoErrorType, self).__init__(**kwargs)


class IonoErrorType(Serializable):
    """Ionosphere delay error statistics."""
    _fields = ('IonoRangeVertical', 'IonoRangeSlant', 'IonoRgRgRateCC', 'IonoRangeDecorr')
    _required = ('IonoRgRgRateCC', )
    _numeric_format = {
        'IonoRangeVertical': FLOAT_FORMAT, 'IonoRangeSlant': FLOAT_FORMAT, 'IonoRgRgRateCC': FLOAT_FORMAT}
    # descriptors
    IonoRangeVertical = FloatDescriptor(
        'IonoRangeVertical', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere two-way delay error for normal incidence standard deviation. '
                  r'Expressed as a range error. '
                  r':math:`(\Delta R) = (\Delta T) \cdot (c/2)`.')  # type: Optional[float]
    IonoRangeSlant = FloatDescriptor(
        'IonoRangeSlant', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere two-way delay rate of change error for normal '
                  'incidence standard deviation. Expressed as a range rate error. '
                  r':math:`(\Delta \dot{R}) = (\Delta \dot{T}) \cdot (c/2)`.')  # type: Optional[float]
    IonoRgRgRateCC = FloatDescriptor(
        'IonoRgRgRateCC', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error and range rate error correlation coefficient.')  # type: float
    IonoRangeDecorr = SerializableDescriptor(
        'IonoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error decorrelation rate.')  # type: Optional[ErrorDecorrFuncType]

    def __init__(
            self,
            IonoRangeVertical: Optional[float] = None,
            IonoRangeSlant: Optional[float] = None,
            IonoRgRgRateCC: float = None,
            IonoRangeDecorr: Optional[ErrorDecorrFuncType] = None,
            **kwargs):
        """

        Parameters
        ----------
        IonoRangeVertical : float
        IonoRangeSlant : float
        IonoRgRgRateCC : float
        IonoRangeDecorr : ErrorDecorrFuncType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.IonoRangeVertical = IonoRangeVertical
        self.IonoRangeSlant = IonoRangeSlant
        self.IonoRgRgRateCC = IonoRgRgRateCC
        self.IonoRangeDecorr = IonoRangeDecorr
        super(IonoErrorType, self).__init__(**kwargs)


class ErrorComponentsType(Serializable):
    """Error statistics by components."""
    _fields = ('PosVelErr', 'RadarSensor', 'TropoError', 'IonoError')
    _required = ('PosVelErr', 'RadarSensor')
    # descriptors
    PosVelErr = SerializableDescriptor(
        'PosVelErr', PosVelErrType, _required, strict=DEFAULT_STRICT,
        docstring='Position and velocity error statistics for the radar platform.')  # type: PosVelErrType
    RadarSensor = SerializableDescriptor(
        'RadarSensor', RadarSensorErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Radar sensor error statistics.')  # type: RadarSensorErrorType
    TropoError = SerializableDescriptor(
        'TropoError', TropoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Troposphere delay error statistics.')  # type: TropoErrorType
    IonoError = SerializableDescriptor(
        'IonoError', IonoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere delay error statistics.')  # type: IonoErrorType

    def __init__(
            self,
            PosVelErr: PosVelErrType = None,
            RadarSensor: RadarSensorErrorType = None,
            TropoError: Optional[TropoErrorType] = None,
            IonoError: Optional[IonoErrorType] = None,
            **kwargs):
        """

        Parameters
        ----------
        PosVelErr : PosVelErrType
        RadarSensor : RadarSensorErrorType
        TropoError : TropoErrorType
        IonoError : IonoErrorType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PosVelErr = PosVelErr
        self.RadarSensor = RadarSensor
        self.TropoError= TropoError
        self.IonoError = IonoError
        super(ErrorComponentsType, self).__init__(**kwargs)


class UnmodeledDecorrType(Serializable):
    """
    Unmodeled decorrelation function definition
    """

    _fields = ('Xrow', 'Ycol')
    _required = _fields
    # descriptors
    Xrow = SerializableDescriptor(
        'Xrow', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT)  # type: ErrorDecorrFuncType
    Ycol = SerializableDescriptor(
        'Ycol', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT)  # type: ErrorDecorrFuncType

    def __init__(
            self,
            Xrow: ErrorDecorrFuncType = None,
            Ycol: ErrorDecorrFuncType = None,
            **kwargs):
        """

        Parameters
        ----------
        Xrow : ErrorDecorrFuncType
        Ycol : ErrorDecorrFuncType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Xrow = Xrow
        self.Ycol = Ycol
        super(UnmodeledDecorrType, self).__init__(**kwargs)


class UnmodeledType(Serializable):
    _fields = ('Xrow', 'Ycol', 'XrowYcol', 'UnmodeledDecorr')
    _required = ('Xrow', 'Ycol', 'XrowYcol')
    _numeric_format = {fld: '0.17G' for fld in ('Xrow', 'Ycol', 'XrowYcol')}
    Xrow = FloatDescriptor(
        'Xrow', _required, strict=DEFAULT_STRICT)  # type: float
    Ycol = FloatDescriptor(
        'Ycol', _required, strict=DEFAULT_STRICT)  # type: float
    XrowYcol = FloatDescriptor(
        'XrowYcol', _required, strict=DEFAULT_STRICT)  # type: float
    UnmodeledDecorr = SerializableDescriptor(
        'UnmodeledDecorr', UnmodeledDecorrType, _required,
        strict=DEFAULT_STRICT)  # type: Optional[UnmodeledDecorrType]

    def __init__(
            self,
            Xrow: float = None,
            Ycol: float = None,
            XrowYcol: float = None,
            UnmodeledDecorr: Optional[UnmodeledDecorrType] = None,
            **kwargs):
        """

        Parameters
        ----------
        Xrow : float
        Ycol : float
        XrowYcol : float
        UnmodeledDecorr : None|UnmodeledDecorrType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Xrow = Xrow
        self.Ycol = Ycol
        self.XrowYcol = XrowYcol
        self.UnmodeledDecorr = UnmodeledDecorr
        super(UnmodeledType, self).__init__(**kwargs)


class ErrorStatisticsType(Serializable):
    """Parameters used to compute error statistics within the SICD sensor model."""
    _fields = ('CompositeSCP', 'Components', 'Unmodeled', 'AdditionalParms')
    _required = ()
    _collections_tags = {'AdditionalParms': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    CompositeSCP = SerializableDescriptor(
        'CompositeSCP', CompositeSCPErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Composite error statistics for the scene center point. '
                  '*Slant plane range (Rg)* and *azimuth (Az)* error statistics. '
                  'Slant plane defined at '
                  '*Scene Center Point, Center of Azimuth (SCP COA)*.')  # type: Optional[CompositeSCPErrorType]
    Components = SerializableDescriptor(
        'Components', ErrorComponentsType, _required, strict=DEFAULT_STRICT,
        docstring='Error statistics by components.')  # type: Optional[ErrorComponentsType]
    Unmodeled = SerializableDescriptor(
        'Unmodeled', UnmodeledType, _required, strict=DEFAULT_STRICT)  # type: Optional[UnmodeledType]

    AdditionalParms = ParametersDescriptor(
        'AdditionalParms', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Any additional parameters.')  # type: Optional[ParametersCollection]

    def __init__(
            self,
            CompositeSCP: Optional[CompositeSCPErrorType] = None,
            Components: Optional[ErrorComponentsType] = None,
            Unmodeled: Optional[UnmodeledType] = None,
            AdditionalParms: Union[None, ParametersCollection, Dict] = None,
            **kwargs):
        """

        Parameters
        ----------
        CompositeSCP : None|CompositeSCPErrorType
        Components : None|ErrorComponentsType
        Unmodeled : None|UnmodeledType
        AdditionalParms : None|ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CompositeSCP = CompositeSCP
        self.Components = Components
        self.Unmodeled = Unmodeled
        self.AdditionalParms = AdditionalParms
        super(ErrorStatisticsType, self).__init__(**kwargs)

    def version_required(self):
        """
        What SICD version is required?

        Returns
        -------
        Tuple[int, int, int]
        """

        return (1, 1, 0) if self.Unmodeled is None else (1, 3, 0)
