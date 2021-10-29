"""
The error parameters type definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union

from sarpy.io.xml.base import Serializable, ParametersCollection
from sarpy.io.xml.descriptors import FloatDescriptor, SerializableDescriptor, \
    ParametersDescriptor
from sarpy.io.complex.sicd_elements.blocks import ErrorDecorrFuncType
from sarpy.io.complex.sicd_elements.ErrorStatistics import PosVelErrType, TropoErrorType

from .base import DEFAULT_STRICT, FLOAT_FORMAT


class RadarSensorType(Serializable):
    """
    Radar sensor error statistics.
    """

    _fields = ('RangeBias', 'ClockFreqSF', 'CollectionStartTime', 'RangeBiasDecorr')
    _required = ('RangeBias', )
    _numeric_format = {'RangeBias': FLOAT_FORMAT, 'ClockFreqSF': FLOAT_FORMAT, 'CollectionStartTime': FLOAT_FORMAT}
    # descriptors
    RangeBias = FloatDescriptor(
        'RangeBias', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Range bias error standard deviation.')  # type: float
    ClockFreqSF = FloatDescriptor(
        'ClockFreqSF', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Payload clock frequency scale factor standard deviation, '
                  r'where :math:`SF = (\Delta f)/f_0`.')  # type: float
    CollectionStartTime = FloatDescriptor(
        'CollectionStartTime', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Collection Start time error standard deviation, '
                  'in seconds.')  # type: float
    RangeBiasDecorr = SerializableDescriptor(
        'RangeBiasDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Range Bias error decorrelation function.')  # type: ErrorDecorrFuncType

    def __init__(self, RangeBias=None, ClockFreqSF=None, CollectionStartTime=None,
                 RangeBiasDecorr=None, **kwargs):
        """

        Parameters
        ----------
        RangeBias : float
        ClockFreqSF : float
        CollectionStartTime : float
        RangeBiasDecorr : ErrorDecorrFuncType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RangeBias = RangeBias
        self.ClockFreqSF = ClockFreqSF
        self.CollectionStartTime = CollectionStartTime
        self.RangeBiasDecorr = RangeBiasDecorr
        super(RadarSensorType, self).__init__(**kwargs)


class IonoErrorType(Serializable):
    """
    Ionosphere delay error statistics.
    """

    _fields = ('IonoRangeVertical', 'IonoRangeRateVertical', 'IonoRgRgRateCC', 'IonoRangeVertDecorr')
    _required = ('IonoRgRgRateCC', )
    _numeric_format = {'IonoRangeVertical': FLOAT_FORMAT, 'IonoRangeRateVertical': FLOAT_FORMAT, 'IonoRgRgRateCC': FLOAT_FORMAT}
    # descriptors
    IonoRangeVertical = FloatDescriptor(
        'IonoRangeVertical', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Ionosphere two-way delay error for normal incidence standard deviation. '
                  r'Expressed as a range error. :math:`(\Delta R) = (\Delta T) \cdot (c/2)`.')  # type: float
    IonoRangeRateVertical = FloatDescriptor(
        'IonoRangeRateVertical', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Ionosphere two-way delay rate of change error for normal incidence standard deviation. '
                  r'Expressed as a range rate error. :math:`\dot{R} = \Delta \dot{TD_Iono} \times c/2`.')  # type: float
    IonoRgRgRateCC = FloatDescriptor(
        'IonoRgRgRateCC', _required, strict=DEFAULT_STRICT, bounds=(-1, 1),
        docstring='Ionosphere range error and range rate error correlation coefficient.')  # type: float
    IonoRangeVertDecorr = SerializableDescriptor(
        'IonoRangeVertDecorr', ErrorDecorrFuncType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere range error decorrelation fucntion.')  # type: ErrorDecorrFuncType

    def __init__(self, IonoRangeVertical=None, IonoRangeRateVertical=None,
                 IonoRgRgRateCC=None, IonoRangeVertDecorr=None, **kwargs):
        """

        Parameters
        ----------
        IonoRangeVertical : float
        IonoRangeRateVertical : float
        IonoRgRgRateCC : float
        IonoRangeVertDecorr : ErrorDecorrFuncType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.IonoRangeVertical = IonoRangeVertical
        self.IonoRangeRateVertical = IonoRangeRateVertical
        self.IonoRgRgRateCC = IonoRgRgRateCC
        self.IonoRangeVertDecorr = IonoRangeVertDecorr
        super(IonoErrorType, self).__init__(**kwargs)


class BistaticRadarSensorType(Serializable):
    """
    Error statistics for a single radar platform.
    """

    _fields = ('ClockFreqSF', 'CollectionStartTime')
    _required = ('CollectionStartTime', )
    _numeric_format = {'ClockFreqSF': FLOAT_FORMAT, 'CollectionStartTime': FLOAT_FORMAT}
    # descriptors
    ClockFreqSF = FloatDescriptor(
        'ClockFreqSF', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Payload clock frequency scale factor standard deviation, '
                  r'where :math:`SF = (\Delta f)/f_0`.')  # type: float
    CollectionStartTime = FloatDescriptor(
        'CollectionStartTime', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Collection Start time error standard deviation, '
                  'in seconds.')  # type: float

    def __init__(self, ClockFreqSF=None, CollectionStartTime=None, **kwargs):
        """

        Parameters
        ----------
        ClockFreqSF : float
        CollectionStartTime : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ClockFreqSF = ClockFreqSF
        self.CollectionStartTime = CollectionStartTime
        super(BistaticRadarSensorType, self).__init__(**kwargs)


class MonostaticType(Serializable):
    """
    Error parameters for monstatic collection.
    """

    _fields = ('PosVelErr', 'RadarSensor', 'TropoError', 'IonoError', 'AddedParameters')
    _required = ('PosVelErr', 'RadarSensor')
    _collections_tags = {'AddedParameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    PosVelErr = SerializableDescriptor(
        'PosVelErr', PosVelErrType, _required, strict=DEFAULT_STRICT,
        docstring='Position and velocity error statistics for the sensor '
                  'platform.')  # type: PosVelErrType
    RadarSensor = SerializableDescriptor(
        'RadarSensor', RadarSensorType, _required, strict=DEFAULT_STRICT,
        docstring='Radar sensor error statistics.')  # type: RadarSensorType
    TropoError = SerializableDescriptor(
        'TropoError', TropoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Troposphere delay error statistics.')  # type: TropoErrorType
    IonoError = SerializableDescriptor(
        'IonoError', IonoErrorType, _required, strict=DEFAULT_STRICT,
        docstring='Ionosphere delay error statistics.')  # type: IonoErrorType
    AddedParameters = ParametersDescriptor(
        'AddedParameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Additional error parameters.')  # type: ParametersCollection

    def __init__(self, PosVelErr=None, RadarSensor=None, TropoError=None, IonoError=None,
                 AddedParameters=None, **kwargs):
        """

        Parameters
        ----------
        PosVelErr : PosVelErrType
        RadarSensor : RadarSensorType
        TropoError : None|TropoErrorType
        IonoError : None|IonoErrorType
        AddedParameters : None|ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PosVelErr = PosVelErr
        self.RadarSensor = RadarSensor
        self.TropoError = TropoError
        self.IonoError = IonoError
        self.AddedParameters = AddedParameters
        super(MonostaticType, self).__init__(PosVelErr=PosVelErr, RadarSensor=RadarSensor, **kwargs)


class PlatformType(Serializable):
    """
    Basic bistatic platform error type definition.
    """
    _fields = ('PosVelErr', 'RadarSensor')
    _required = _fields
    # descriptors
    PosVelErr = SerializableDescriptor(
        'PosVelErr', PosVelErrType, _required, strict=DEFAULT_STRICT,
        docstring='Position and velocity error statistics for the sensor '
                  'platform.')  # type: PosVelErrType
    RadarSensor = SerializableDescriptor(
        'RadarSensor', BistaticRadarSensorType, _required, strict=DEFAULT_STRICT,
        docstring='Platform sensor error statistics.')  # type: BistaticRadarSensorType

    def __init__(self, PosVelErr=None, RadarSensor=None, **kwargs):
        """

        Parameters
        ----------
        PosVelErr : PosVelErrType
        RadarSensor : BistaticRadarSensorType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PosVelErr = PosVelErr
        self.RadarSensor = RadarSensor
        super(PlatformType, self).__init__(**kwargs)


class BistaticType(Serializable):
    """
    Error parameters for bistatic parameters.
    """

    _fields = ('TxPlatform', 'RcvPlatform', 'AddedParameters')
    _required = ('TxPlatform', )
    _collections_tags = {'AddedParameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    TxPlatform = SerializableDescriptor(
        'TxPlatform', PlatformType, _required, strict=DEFAULT_STRICT,
        docstring='Error statistics for the transmit platform.')  # type: PlatformType
    RcvPlatform = SerializableDescriptor(
        'RcvPlatform', PlatformType, _required, strict=DEFAULT_STRICT,
        docstring='Error statistics for the receive platform.')  # type: PlatformType
    AddedParameters = ParametersDescriptor(
        'AddedParameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Additional error parameters.')  # type: ParametersCollection

    def __init__(self, TxPlatform=None, RcvPlatform=None, AddedParameters=None, **kwargs):
        """

        Parameters
        ----------
        TxPlatform : PlatformType
        RcvPlatform : PlatformType
        AddedParameters : None|ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxPlatform = TxPlatform
        self.RcvPlatform = RcvPlatform
        self.AddedParameters = AddedParameters
        super(BistaticType, self).__init__(**kwargs)


class ErrorParametersType(Serializable):
    """
    Parameters that describe the statistics of errors in measured or estimated
    parameters that describe the collection.
    """

    _fields = ('Monostatic', 'Bistatic')
    _required = ()
    _choice = ({'required': True, 'collection': _fields}, )

    # descriptors
    Monostatic = SerializableDescriptor(
        'Monostatic', MonostaticType, _required, strict=DEFAULT_STRICT,
        docstring='The monstatic parameters.')  # type: Union[None, MonostaticType]
    Bistatic = SerializableDescriptor(
        'Bistatic', BistaticType, _required, strict=DEFAULT_STRICT,
        docstring='The bistatic parameters.')  # type: Union[None, BistaticType]

    def __init__(self, Monostatic=None, Bistatic=None, **kwargs):
        """

        Parameters
        ----------
        Monostatic : None|MonostaticType
        Bistatic : None|BistaticType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Monostatic = Monostatic
        self.Bistatic = Bistatic
        super(ErrorParametersType, self).__init__(**kwargs)
