"""
The ErrorStatisticsType definition.
"""

from typing import List, Union

import numpy

from .base import Serializable, DEFAULT_STRICT, _StringEnumDescriptor, _FloatDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from .blocks import ParameterType, ErrorDecorrFuncType

__classification__ = "UNCLASSIFIED"


class CompositeSCPErrorType(Serializable):
    """
    Composite error statistics for the Scene Center Point. Slant plane range (Rg) and azimuth (Az) error
    statistics. Slant plane defined at SCP COA.
    """
    _fields = ('Rg', 'Az', 'RgAz')
    _required = _fields
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


class CorrCoefsType(Serializable):
    """Correlation Coefficient parameters."""
    _fields = (
        'P1P2', 'P1P3', 'P1V1', 'P1V2', 'P1V3', 'P2P3', 'P2V1', 'P2V2', 'P2V3',
        'P3V1', 'P3V2', 'P3V3', 'V1V2', 'V1V3', 'V2V3')
    _required = _fields
    # descriptors
    P1P2 = _FloatDescriptor(
        'P1P2', _required, strict=DEFAULT_STRICT, docstring='P1 and P2 correlation coefficient.')  # type: float
    P1P3 = _FloatDescriptor(
        'P1P3', _required, strict=DEFAULT_STRICT, docstring='P1 and P3 correlation coefficient.')  # type: float
    P1V1 = _FloatDescriptor(
        'P1V1', _required, strict=DEFAULT_STRICT, docstring='P1 and V1 correlation coefficient.')  # type: float
    P1V2 = _FloatDescriptor(
        'P1V2', _required, strict=DEFAULT_STRICT, docstring='P1 and V2 correlation coefficient.')  # type: float
    P1V3 = _FloatDescriptor(
        'P1V3', _required, strict=DEFAULT_STRICT, docstring='P1 and V3 correlation coefficient.')  # type: float
    P2P3 = _FloatDescriptor(
        'P2P3', _required, strict=DEFAULT_STRICT, docstring='P2 and P3 correlation coefficient.')  # type: float
    P2V1 = _FloatDescriptor(
        'P2V1', _required, strict=DEFAULT_STRICT, docstring='P2 and V1 correlation coefficient.')  # type: float
    P2V2 = _FloatDescriptor(
        'P2V2', _required, strict=DEFAULT_STRICT, docstring='P2 and V2 correlation coefficient.')  # type: float
    P2V3 = _FloatDescriptor(
        'P2V3', _required, strict=DEFAULT_STRICT, docstring='P2 and V3 correlation coefficient.')  # type: float
    P3V1 = _FloatDescriptor(
        'P3V1', _required, strict=DEFAULT_STRICT, docstring='P3 and V1 correlation coefficient.')  # type: float
    P3V2 = _FloatDescriptor(
        'P3V2', _required, strict=DEFAULT_STRICT, docstring='P3 and V2 correlation coefficient.')  # type: float
    P3V3 = _FloatDescriptor(
        'P3V3', _required, strict=DEFAULT_STRICT, docstring='P3 and V3 correlation coefficient.')  # type: float
    V1V2 = _FloatDescriptor(
        'V1V2', _required, strict=DEFAULT_STRICT, docstring='V1 and V2 correlation coefficient.')  # type: float
    V1V3 = _FloatDescriptor(
        'V1V3', _required, strict=DEFAULT_STRICT, docstring='V1 and V3 correlation coefficient.')  # type: float
    V2V3 = _FloatDescriptor(
        'V2V3', _required, strict=DEFAULT_STRICT, docstring='V2 and V3 correlation coefficient.')  # type: float


class PosVelErrType(Serializable):
    """Position and velocity error statistics for the radar platform."""
    _fields = ('Frame', 'P1', 'P2', 'P3', 'V1', 'V2', 'V3', 'CorrCoefs', 'PositionDecorr')
    _required = ('Frame', 'P1', 'P2', 'P3', 'V1', 'V2', 'V3')
    # class variables
    _FRAME_VALUES = ('ECF', 'RIC_ECF', 'RIC_ECI')
    # descriptors
    Frame = _StringEnumDescriptor(
        'Frame', _FRAME_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Coordinate frame used for expressing P,V errors statistics. Note: '
                  '*RIC = Radial, In-Track, Cross-Track*, where radial is defined to be from earth center through '
                  'the platform position. ')  # type: str
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


class RadarSensorErrorType(Serializable):
    """Radar sensor error statistics."""
    _fields = ('RangeBias', 'ClockFreqSF', 'TransmitFreqSF', 'RangeBiasDecorr')
    _required = ('RangeBias', )
    # descriptors
    RangeBias = _FloatDescriptor(
        'RangeBias', _required, strict=DEFAULT_STRICT,
        docstring='Range bias error standard deviation.')  # type: float
    ClockFreqSF = _FloatDescriptor(
        'ClockFreqSF', _required, strict=DEFAULT_STRICT,
        docstring='Payload clock frequency scale factor standard deviation, where SF = (Delta f)/f0.')  # type: float
    TransmitFreqSF = _FloatDescriptor(
        'TransmitFreqSF', _required, strict=DEFAULT_STRICT,
        docstring='Transmit frequency scale factor standard deviation, where SF = (Delta f)/f0.')  # type: float
    RangeBiasDecorr = _SerializableDescriptor(
        'RangeBiasDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Range bias decorrelation rate.')  # type: ErrorDecorrFuncType


class TropoErrorType(Serializable):
    """Troposphere delay error statistics."""
    _fields = ('TropoRangeVertical', 'TropoRangeSlant', 'TropoRangeDecorr')
    _required = ()
    # descriptors
    TropoRangeVertical = _FloatDescriptor(
        'TropoRangeVertical', _required, strict=DEFAULT_STRICT,
        docstring='Troposphere two-way delay error for normal incidence standard deviation. '
                  'Expressed as a range error. `(Delta R) = (Delta T) x c/2`.')  # type: float
    TropoRangeSlant = _FloatDescriptor(
        'TropoRangeSlant', _required, strict=DEFAULT_STRICT,
        docstring='Troposphere two-way delay error for the SCP line of sight at COA standard deviation. '
                  'Expressed as a range error. `(Delta R) = (Delta T) x c/2`.')  # type: float
    TropoRangeDecorr = _SerializableDescriptor(
        'TropoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Troposphere range error decorrelation function.')  # type: ErrorDecorrFuncType


class IonoErrorType(Serializable):
    """Ionosphere delay error statistics."""
    _fields = ('IonoRangeVertical', 'IonoRangeSlant', 'IonoRgRgRateCC', 'IonoRangeDecorr')
    _required = ('IonoRgRgRateCC', )
    # descriptors
    IonoRangeVertical = _FloatDescriptor(
        'IonoRangeVertical', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere two-way delay error for normal incidence standard deviation. '
                  'Expressed as a range error. `(Delta R) = (Delta T) x c/2`.')  # type: float
    IonoRangeSlant = _FloatDescriptor(
        'IonoRangeSlant', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere two-way delay rate of change error for normal incidence standard deviation. '
                  'Expressed as a range rate error. `(Delta Rdot) = (Delta Tdot) x c/2`.')  # type: float
    IonoRgRgRateCC = _FloatDescriptor(
        'IonoRgRgRateCC', _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error and range rate error correlation coefficient.')  # type: float
    IonoRangeDecorr = _SerializableDescriptor(
        'IonoRangeDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error decorrelation rate.')  # type: ErrorDecorrFuncType


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


class ErrorStatisticsType(Serializable):
    """Parameters used to compute error statistics within the SICD sensor model."""
    _fields = ('CompositeSCP', 'Components', 'AdditionalParms')
    _required = ()
    _collections_tags = {'AdditionalParms': {'array': True, 'child_tag': 'Parameter'}}
    # descriptors
    CompositeSCP = _SerializableDescriptor(
        'CompositeSCP', CompositeSCPErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Composite error statistics for the Scene Center Point. Slant plane range (Rg) and azimuth (Az) '
                  'error statistics. Slant plane defined at SCP COA.')  # type: CompositeSCPErrorType
    Components = _SerializableDescriptor(
        'Components', ErrorComponentsType, _required, strict=DEFAULT_STRICT,
        docstring='Error statistics by components.')  # type: ErrorComponentsType
    AdditionalParms = _SerializableArrayDescriptor(
        'AdditionalParms', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Any additional parameters.')  # type: Union[numpy.ndarray, List[ParameterType]]
