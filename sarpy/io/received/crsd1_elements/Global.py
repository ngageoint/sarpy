"""
The Global type definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Michael Stewart, Valkyrie")

from typing import Union

import numpy

from sarpy.io.xml.base import Serializable, Arrayable
from sarpy.io.xml.descriptors import FloatDescriptor, DateTimeDescriptor, \
    SerializableDescriptor
from sarpy.io.phase_history.cphd1_elements.Global import FxBandType, TropoParametersType, \
    IonoParametersType

from .base import DEFAULT_STRICT, FLOAT_FORMAT


class TimelineType(Serializable):
    """
    Parameters that describe the collection times for the data contained in the product.
    """

    _fields = ('CollectionRefTime', 'RcvTime1', 'RcvTime2')
    _required = ('CollectionRefTime', 'RcvTime1', 'RcvTime2')
    _numeric_format = {'RcvTime1': FLOAT_FORMAT, 'RcvTime2': FLOAT_FORMAT}
    # descriptors
    CollectionRefTime = DateTimeDescriptor(
        'CollectionRefTime', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Collection Reference Time (t_CRT). Time reference used for all receive times'
                  ' and all transmit times. All times are specified in seconds relative to t_CRT'
                  ' (i.e., t_CRT is slow time t = 0).')  # type: numpy.datetime64
    RcvTime1 = FloatDescriptor(
        'RcvTime1', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Earliest RcvTime value for any signal vector in the product.'
                  ' Time relative to Collection Reference Time.')  # type: float
    RcvTime2 = FloatDescriptor(
        'RcvTime2', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Latest RcvTime value for any signal vector in the product.'
                  ' Time relative to Collection Reference Time.')  # type: float

    def __init__(self, CollectionRefTime=None, RcvTime1=None, RcvTime2=None, **kwargs):
        """

        Parameters
        ----------
        CollectionRefTime : numpy.datetime64|datetime|date|str
        RcvTime1 : float
        RcvTime2 : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CollectionRefTime = CollectionRefTime
        self.RcvTime1 = RcvTime1
        self.RcvTime2 = RcvTime2
        super(TimelineType, self).__init__(**kwargs)


class FrcvBandType(Serializable, Arrayable):
    """
    Parameters that describe the received frequency limits for the signal array(s) contained in the product.
    """

    _fields = ('FrcvMin', 'FrcvMax')
    _required = _fields
    _numeric_format = {fld: FLOAT_FORMAT for fld in _fields}
    # descriptors
    FrcvMin = FloatDescriptor(
        'FrcvMin', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Minimum frcv_1 PVP value for any signal vector in the product.')  # type: float
    FrcvMax = FloatDescriptor(
        'FrcvMax', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Maximum frcv_2 PVP value for any signal vector in the product.')  # type: float

    def __init__(self, FrcvMin=None, FrcvMax=None, **kwargs):
        """

        Parameters
        ----------
        FrcvMin : float
        FrcvMax : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.FrcvMin = FrcvMin
        self.FrcvMax = FrcvMax
        super(FrcvBandType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        return numpy.array([self.FrcvMin, self.FrcvMax], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        # type: (Union[numpy.ndarray, list, tuple]) -> FrcvBandType
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(FrcvMin=array[0], FrcvMax=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class GlobalType(Serializable):
    """
    The Global type definition.
    """

    _fields = ('Timeline', 'FrcvBand', 'FxBand', 'TropoParameters', 'IonoParameters')
    _required = ('Timeline', 'FrcvBand')
    # descriptors
    Timeline = SerializableDescriptor(
        'Timeline', TimelineType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the collection times for the data contained '
                  'in the product')  # type: TimelineType
    FrcvBand = SerializableDescriptor(
        'FrcvBand', FrcvBandType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the Frcv frequency limits for the signal array(s) '
                  'contained in the product.')  # type: FrcvBandType
    FxBand = SerializableDescriptor(
        'FxBand', FxBandType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the FX frequency limits for the signal array(s) '
                  'contained in the product.')  # type: FxBandType
    TropoParameters = SerializableDescriptor(
        'TropoParameters', TropoParametersType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters used to compute the propagation delay due to the '
                  'troposphere.')  # type: Union[None, TropoParametersType]
    IonoParameters = SerializableDescriptor(
        'IonoParameters', IonoParametersType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters used to compute propagation effects due to the '
                  'ionosphere.')  # type: Union[None, IonoParametersType]

    def __init__(self, Timeline=None, FrcvBand=None, FxBand=None,
                 TropoParameters=None, IonoParameters=None, **kwargs):
        """

        Parameters
        ----------
        Timeline : TimelineType
        FrcvBand : FrcvBandType|numpy.ndarray|list|tuple
        FxBand : None|FxBandType|numpy.ndarray|list|tuple
        TropoParameters : None|TropoParametersType
        IonoParameters : None|IonoParametersType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Timeline = Timeline
        self.FrcvBand = FrcvBand
        self.FxBand = FxBand
        self.TropoParameters = TropoParameters
        self.IonoParameters = IonoParameters
        super(GlobalType, self).__init__(**kwargs)
