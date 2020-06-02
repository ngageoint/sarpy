# -*- coding: utf-8 -*-
"""
The Global type definition.
"""

from typing import Union

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from ..sicd_elements.base import Serializable, Arrayable, _FloatDescriptor, \
    _DateTimeDescriptor, _StringEnumDescriptor, _IntegerEnumDescriptor, \
    _SerializableDescriptor

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class TimelineType(Serializable):
    """
    The timeline element.
    """
    _fields = ('CollectionStart', 'RcvCollectionStart', 'TxTime1', 'TxTime2')
    _required = ('CollectionStart', 'TxTime1', 'TxTime2')
    # descriptors
    CollectionStart = _DateTimeDescriptor(
        'CollectionStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The collection start time. The default precision will be '
                  'microseconds.')  # type: numpy.datetime64
    RcvCollectionStart = _DateTimeDescriptor(
        'RcvCollectionStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The collection start time. The default precision will be '
                  'microseconds.')  # type: numpy.datetime64
    TxTime1 = _FloatDescriptor(
        'TxTime1', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The tx time 1?')  # type: float
    TxTime2 = _FloatDescriptor(
        'TxTime2', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The tx time 2?')  # type: float

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
    The FxBand information
    """
    _fields = ('FxMin', 'FxMax')
    _required = _fields
    # descriptors
    FxMin = _FloatDescriptor(
        'FxMin', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The Fx minimum.')  # type: float
    FxMax = _FloatDescriptor(
        'FxMax', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The Fx maximum.')  # type: float

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
    The TOASwath information
    """
    _fields = ('TOAMin', 'TOAMax')
    _required = _fields
    # descriptors
    TOAMin = _FloatDescriptor(
        'TOAMin', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The TOA minimum.')  # type: float
    TOAMax = _FloatDescriptor(
        'TOAMax', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The TOA maximum.')  # type: float

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
    The Troposphere parameters.
    """

    _fields = ('N0', 'RefHeight')
    _required = _fields
    # descriptors
    N0 = _FloatDescriptor(
        'N0', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The N0 parameter.')  # type: float
    RefHeight = _StringEnumDescriptor(
        'RefHeight', ('IARP', 'ZERO'), _required, strict=DEFAULT_STRICT,
        docstring='The reference for the height.')  # type: str

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
    The Ionosphere parameters.
    """

    _fields = ('TECV', 'F2Height')
    _required = ('TECV', )
    # descriptor
    TECV = _FloatDescriptor(
        'TECV', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The TECV parameter.')  # type: float
    F2Height = _FloatDescriptor(
        'F2Height', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='The F2 height.')  # type: Union[None, float]

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
        docstring='')  # type: str
    SGN = _IntegerEnumDescriptor(
        'SGN', (-1, 1), _required, strict=DEFAULT_STRICT,
        docstring='The sign.')  # type: int
    Timeline = _SerializableDescriptor(
        'Timeline', TimelineType, _required, strict=DEFAULT_STRICT,
        docstring='The timeline parameters.')  # type: TimelineType
    FxBand = _SerializableDescriptor(
        'FxBand', FxBandType, _required, strict=DEFAULT_STRICT,
        docstring='The FX band parameters.')  # type: FxBandType
    TOASwath = _SerializableDescriptor(
        'TOASwath', TOASwathType, _required, strict=DEFAULT_STRICT,
        docstring='The TOA swath.')  # type: TOASwathType
    TropoParameters = _SerializableDescriptor(
        'TropoParameters', TropoParametersType, _required, strict=DEFAULT_STRICT,
        docstring='The troposphere parameters.')  # type: Union[None, TropoParametersType]
    IonoParameters = _SerializableDescriptor(
        'IonoParameters', IonoParametersType, _required, strict=DEFAULT_STRICT,
        docstring='The ionosphere parameters.')  # type: Union[None, IonoParametersType]

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
