# -*- coding: utf-8 -*-
"""
The Global type definition for CPHD 0.3.
"""

from typing import Union, List

import numpy

from sarpy.io.phase_history.cphd1_elements.base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _FloatDescriptor, \
    _DateTimeDescriptor, _StringEnumDescriptor, _IntegerEnumDescriptor, \
    _SerializableDescriptor, _IntegerDescriptor, \
    _SerializableCPArrayDescriptor, SerializableCPArray
from sarpy.io.complex.sicd_elements.blocks import LatLonHAECornerRestrictionType, Poly2DType
from sarpy.io.complex.sicd_elements.RadarCollection import ReferencePointType, XDirectionType, \
    YDirectionType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class DwellTimeType(Serializable):
    """
    The dwell time object.
    """
    _fields = ('DwellTimePoly', 'CODTimePoly')
    _required = _fields
    # descriptors
    DwellTimePoly = _SerializableDescriptor(
        'DwellTimePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='The dwell time polynomial.')  # type: Poly2DType
    CODTimePoly = _SerializableDescriptor(
        'CODTimePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='The cod time polynomial.')  # type: Poly2DType

    def __init__(self, DwellTimePoly=None, CODTimePoly=None, **kwargs):
        """

        Parameters
        ----------
        DwellTimePoly : Poly2DType|numpy.ndarray|list|tuple
        CODTimePoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DwellTimePoly = DwellTimePoly
        self.CODTimePoly = CODTimePoly
        super(DwellTimeType, self).__init__(**kwargs)


class PlaneType(Serializable):
    """
    The reference plane.
    """

    _fields = ('RefPt', 'XDir', 'YDir', 'DwellTime')
    _required = ('RefPt', 'XDir', 'YDir')
    # other class variable
    # descriptors
    RefPt = _SerializableDescriptor(
        'RefPt', ReferencePointType, _required, strict=DEFAULT_STRICT,
        docstring='The reference point.')  # type: ReferencePointType
    XDir = _SerializableDescriptor(
        'XDir', XDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The X direction collection plane parameters.')  # type: XDirectionType
    YDir = _SerializableDescriptor(
        'YDir', YDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The Y direction collection plane parameters.')  # type: YDirectionType
    DwellTime = _SerializableDescriptor(
        'DwellTime', DwellTimeType, _required, strict=DEFAULT_STRICT,
        docstring='The dwell time parameters.')  # type: DwellTimeType

    def __init__(self, RefPt=None, XDir=None, YDir=None, DwellTime=None, **kwargs):
        """

        Parameters
        ----------
        RefPt : ReferencePointType
        XDir : XDirectionType
        YDir : YDirectionType
        DwellTime : DwellTimeType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RefPt = RefPt
        self.XDir = XDir
        self.YDir = YDir
        self.DwellTime = DwellTime
        super(PlaneType, self).__init__(**kwargs)


class ImageAreaType(Serializable):
    """
    The collection area.
    """

    _fields = ('Corner', 'Plane')
    _required = ('Corner', )
    _collections_tags = {
        'Corner': {'array': True, 'child_tag': 'ACP'}, }
    # descriptors
    Corner = _SerializableCPArrayDescriptor(
        'Corner', LatLonHAECornerRestrictionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The collection area corner point definition array.'
    )  # type: Union[SerializableCPArray, List[LatLonHAECornerRestrictionType]]
    Plane = _SerializableDescriptor(
        'Plane', PlaneType, _required, strict=DEFAULT_STRICT,
        docstring='A rectangular area in a geo-located display plane.')  # type: PlaneType

    def __init__(self, Corner=None, Plane=None, **kwargs):
        """

        Parameters
        ----------
        Corner : SerializableCPArray|List[LatLonHAECornerRestrictionType]|numpy.ndarray|list|tuple
        Plane : PlaneType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Corner = Corner
        self.Plane = Plane
        super(ImageAreaType, self).__init__(**kwargs)


class GlobalType(Serializable):
    """
    The Global type definition.
    """

    _fields = (
        'DomainType', 'PhaseSGN', 'RefFreqIndex', 'CollectStart',
        'CollectDuration', 'TxTime1', 'TxTime2', 'ImageArea')
    _required = (
        'DomainType', 'PhaseSGN', 'CollectStart', 'CollectDuration',
        'TxTime1', 'TxTime2', 'ImageArea')
    _numeric_format = {
        'CollectDuration': '0.16G', 'TxTime1': '0.16G', 'TxTime2': '0.16G'}
    # descriptors
    DomainType = _StringEnumDescriptor(
        'DomainType', ('FX', 'TOA'), _required, strict=DEFAULT_STRICT,
        docstring='Indicates the domain represented by the sample dimension of the '
                  'CPHD signal array(s), where "FX" denotes Transmit Frequency, and '
                  '"TOA" denotes Difference in Time of Arrival')  # type: str
    PhaseSGN = _IntegerEnumDescriptor(
        'PhaseSGN', (-1, 1), _required, strict=DEFAULT_STRICT,
        docstring='Phase SGN applied to compute target signal phase as a function of '
                  r'target :math:`\Delta TOA^{TGT}`. Target phase in cycles. '
                  r'For simple phase model :math:`Phase(fx) = SGN \times fx \times \Delta TOA^{TGT}` '
                  r'In TOA domain, phase of the mainlobe peak '
                  r':math:`Phase(\Delta TOA^{TGT}) = SGN \times fx_C \times \Delta TOA^{TGT}`'
                  '.')  # type: int
    RefFreqIndex = _IntegerDescriptor(
        'RefFreqIndex', _required, strict=DEFAULT_STRICT,
        docstring='Indicates if the RF frequency values are expressed as offsets from '
                  'a reference frequency (RefFreq).')  # type: Union[None, int]
    CollectStart = _DateTimeDescriptor(
        'CollectStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Collection Start date and time (UTC). Time reference used for times '
                  'measured from collection start (i.e. slow time t = 0). For bistatic '
                  'collections, the time is the transmit platform collection '
                  'start time. The default display precision is microseconds, but this '
                  'does not that accuracy in value.')  # type: numpy.datetime64
    CollectDuration = _FloatDescriptor(
        'CollectDuration', _required, strict=DEFAULT_STRICT,
        docstring='The duration of the collection, in seconds.')  # type: float
    TxTime1 = _FloatDescriptor(
        'TxTime1', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Earliest TxTime value for any signal vector in the product. '
                  'Time relative to Collection Start in seconds.')  # type: float
    TxTime2 = _FloatDescriptor(
        'TxTime2', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Latest TxTime value for any signal vector in the product. '
                  'Time relative to Collection Start in seconds.')  # type: float
    ImageArea = _SerializableDescriptor(
        'ImageArea', ImageAreaType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters describing the ground area covered by this '
                  'product.')  # type: ImageAreaType

    def __init__(self, DomainType=None, PhaseSGN=None, RefFreqIndex=None, CollectStart=None,
                 CollectDuration=None, TxTime1=None, TxTime2=None, ImageArea=None, **kwargs):
        """

        Parameters
        ----------
        DomainType : str
        PhaseSGN : int
        RefFreqIndex : None|int
        CollectStart : numpy.datetime64|datetime.datetime|str
        CollectDuration : float
        TxTime1 : float
        TxTime2 : float
        ImageArea : ImageAreaType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DomainType = DomainType
        self.PhaseSGN = PhaseSGN
        self.RefFreqIndex = RefFreqIndex
        self.CollectStart = CollectStart
        self.CollectDuration = CollectDuration
        self.TxTime1 = TxTime1
        self.TxTime2 = TxTime2
        self.ImageArea = ImageArea
        super(GlobalType, self).__init__(**kwargs)
