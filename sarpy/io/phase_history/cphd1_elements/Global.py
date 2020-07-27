# -*- coding: utf-8 -*-
"""
The Global type definition.
"""

from typing import Union

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, Arrayable, _FloatDescriptor, \
    _DateTimeDescriptor, _StringEnumDescriptor, _IntegerEnumDescriptor, \
    _SerializableDescriptor

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class TimelineType(Serializable):
    """
    Parameters that describe the collection times for the data contained in the product.
    """

    _fields = ('CollectionStart', 'RcvCollectionStart', 'TxTime1', 'TxTime2')
    _required = ('CollectionStart', 'TxTime1', 'TxTime2')
    _numeric_format = {'TxTime1': '0.16G', 'TxTime2': '0.16G'}
    # descriptors
    CollectionStart = _DateTimeDescriptor(
        'CollectionStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Collection Start date and time (UTC). Time reference used for times '
                  'measured from collection start (i.e. slow time t = 0). For bistatic '
                  'collections, the time is the transmit platform collection '
                  'start time. The default display precision is microseconds, but this '
                  'does not that accuracy in value.')  # type: numpy.datetime64
    RcvCollectionStart = _DateTimeDescriptor(
        'RcvCollectionStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Receive only platform collection date and start time.')  # type: numpy.datetime64
    TxTime1 = _FloatDescriptor(
        'TxTime1', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Earliest TxTime value for any signal vector in the product. '
                  'Time relative to Collection Start in seconds.')  # type: float
    TxTime2 = _FloatDescriptor(
        'TxTime2', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Latest TxTime value for any signal vector in the product. '
                  'Time relative to Collection Start in seconds.')  # type: float

    def __init__(self, CollectionStart=None, RcvCollectionStart=None, TxTime1=None, TxTime2=None, **kwargs):
        """

        Parameters
        ----------
        CollectionStart : numpy.datetime64|datetime|date|str
        RcvCollectionStart : None|numpy.datetime64|datetime|date|str
        TxTime1 : float
        TxTime2 : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CollectionStart = CollectionStart
        self.RcvCollectionStart = RcvCollectionStart
        self.TxTime1 = TxTime1
        self.TxTime2 = TxTime2
        super(TimelineType, self).__init__(**kwargs)


class FxBandType(Serializable, Arrayable):
    """
    Parameters that describe the FX frequency limits for the signal array(s)
    contained in the product.
    """

    _fields = ('FxMin', 'FxMax')
    _required = _fields
    _numeric_format = {fld: '0.16G' for fld in _fields}
    # descriptors
    FxMin = _FloatDescriptor(
        'FxMin', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Minimum fx value for any signal vector in the product in '
                  'Hz.')  # type: float
    FxMax = _FloatDescriptor(
        'FxMax', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Maximum fx value for any signal vector in the product in '
                  'Hz.')  # type: float

    def __init__(self, FxMin=None, FxMax=None, **kwargs):
        """

        Parameters
        ----------
        FxMin : float
        FxMax : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.FxMin = FxMin
        self.FxMax = FxMax
        super(FxBandType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        return numpy.array([self.FxMin, self.FxMax], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        # type: (Union[numpy.ndarray, list, tuple]) -> FxBandType
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(FxMin=array[0], FxMax=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class TOASwathType(Serializable, Arrayable):
    """
    Parameters that describe the time-of-arrival (TOA) swath limits for the signal
    array(s) contained in the product.
    """

    _fields = ('TOAMin', 'TOAMax')
    _required = _fields
    _numeric_format = {fld: '0.16G' for fld in _fields}
    # descriptors
    TOAMin = _FloatDescriptor(
        'TOAMin', _required, strict=DEFAULT_STRICT,
        docstring=r'Minimum :math:`\Delta TOA` value for any signal vector in '
                  'the product, in seconds.')  # type: float
    TOAMax = _FloatDescriptor(
        'TOAMax', _required, strict=DEFAULT_STRICT,
        docstring=r'Maximum :math:`\Delta TOA` value for any signal vector in '
                  'the product, in seconds.')  # type: float

    def __init__(self, TOAMin=None, TOAMax=None, **kwargs):
        """

        Parameters
        ----------
        TOAMin : float
        TOAMax : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TOAMin = TOAMin
        self.TOAMax = TOAMax
        super(TOASwathType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        return numpy.array([self.TOAMin, self.TOAMax], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        # type: (Union[numpy.ndarray, list, tuple]) -> TOASwathType
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(TOAMin=array[0], TOAMax=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class TropoParametersType(Serializable):
    """
    Parameters used to compute the propagation delay due to the troposphere.
    """

    _fields = ('N0', 'RefHeight')
    _required = _fields
    _numeric_format = {'No': '0.16G'}
    # descriptors
    N0 = _FloatDescriptor(
        'N0', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Refractivity value of the troposphere for the imaged scene used '
                  'to form the product (dimensionless). Value at the IARP '
                  'location.')  # type: float
    RefHeight = _StringEnumDescriptor(
        'RefHeight', ('IARP', 'ZERO'), _required, strict=DEFAULT_STRICT,
        docstring='Reference Height for the `N0` value.')  # type: str

    def __init__(self, N0=None, RefHeight=None, **kwargs):
        """

        Parameters
        ----------
        N0 : float
        RefHeight : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.N0 = N0
        self.RefHeight = RefHeight
        super(TropoParametersType, self).__init__(**kwargs)


class IonoParametersType(Serializable):
    """
    Parameters used to compute propagation effects due to the ionosphere.
    """

    _fields = ('TECV', 'F2Height')
    _required = ('TECV', )
    _numeric_format = {fld: '0.16G' for fld in _fields}
    # descriptor
    TECV = _FloatDescriptor(
        'TECV', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Total Electron Content (TEC) integrated along TECU the Vertical (V), '
                  r'in units where :math:`1 TECU = 10^{16} e^{-}/m^{2}`')  # type: float
    F2Height = _FloatDescriptor(
        'F2Height', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The F2 height of the ionosphere, in '
                  'meters.')  # type: Union[None, float]

    def __init__(self, TECV=None, F2Height=None, **kwargs):
        """

        Parameters
        ----------
        TECV : float
        F2Height : None|float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TECV = TECV
        self.F2Height = F2Height
        super(IonoParametersType, self).__init__(**kwargs)


class GlobalType(Serializable):
    """
    The Global type definition.
    """

    _fields = (
        'DomainType', 'SGN', 'Timeline', 'FxBand', 'TOASwath', 'TropoParameters', 'IonoParameters')
    _required = ('DomainType', 'SGN', 'Timeline', 'FxBand', 'TOASwath')
    # descriptors
    DomainType = _StringEnumDescriptor(
        'DomainType', ('FX', 'TOA'), _required, strict=DEFAULT_STRICT,
        docstring='Indicates the domain represented by the sample dimension of the '
                  'CPHD signal array(s), where "FX" denotes Transmit Frequency, and '
                  '"TOA" denotes Difference in Time of Arrival')  # type: str
    SGN = _IntegerEnumDescriptor(
        'SGN', (-1, 1), _required, strict=DEFAULT_STRICT,
        docstring='Phase SGN applied to compute target signal phase as a function of '
                  r'target :math:`\Delta TOA^{TGT}`. Target phase in cycles. '
                  r'For simple phase model :math:`Phase(fx) = SGN \times fx \times \Delta TOA^{TGT}` '
                  r'In TOA domain, phase of the mainlobe peak '
                  r':math:`Phase(\Delta TOA^{TGT}) = SGN \times fx_C \times \Delta TOA^{TGT}`'
                  '.')  # type: int
    Timeline = _SerializableDescriptor(
        'Timeline', TimelineType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the collection times for the data contained '
                  'in the product')  # type: TimelineType
    FxBand = _SerializableDescriptor(
        'FxBand', FxBandType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the FX frequency limits for the signal array(s) '
                  'contained in the product.')  # type: FxBandType
    TOASwath = _SerializableDescriptor(
        'TOASwath', TOASwathType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the time-of-arrival (TOA) swath limits for the '
                  'signal array(s) contained in the product.')  # type: TOASwathType
    TropoParameters = _SerializableDescriptor(
        'TropoParameters', TropoParametersType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters used to compute the propagation delay due to the '
                  'troposphere.')  # type: Union[None, TropoParametersType]
    IonoParameters = _SerializableDescriptor(
        'IonoParameters', IonoParametersType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters used to compute propagation effects due to the '
                  'ionosphere.')  # type: Union[None, IonoParametersType]

    def __init__(self, DomainType=None, SGN=None, Timeline=None, FxBand=None, TOASwath=None,
                 TropoParameters=None, IonoParameters=None, **kwargs):
        """

        Parameters
        ----------
        DomainType : str
        SGN : int
        Timeline : TimelineType
        FxBand : FxBandType|numpy.ndarray|list|tuple
        TOASwath : TOASwathType|numpy.ndarray|list|tuple
        TropoParameters : None|TropoParametersType
        IonoParameters : None|IonoParametersType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DomainType = DomainType
        self.SGN = SGN
        self.Timeline = Timeline
        self.FxBand = FxBand
        self.TOASwath = TOASwath
        self.TropoParameters = TropoParameters
        self.IonoParameters = IonoParameters
        super(GlobalType, self).__init__(**kwargs)
