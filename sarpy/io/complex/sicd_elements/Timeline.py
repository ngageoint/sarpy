"""
The TimelineType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import List, Union, Optional
from datetime import datetime, date

import numpy

from sarpy.io.xml.base import Serializable, SerializableArray
from sarpy.io.xml.descriptors import FloatDescriptor, IntegerDescriptor, \
    DateTimeDescriptor, SerializableDescriptor, SerializableArrayDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import Poly1DType


class IPPSetType(Serializable):
    """
    The Inter-Pulse Parameter array element container.
    """

    # NOTE that this is simply defined as a child class ("Set") of the TimelineType in the SICD standard
    #   Defining it at root level clarifies the documentation, and giving it a more descriptive name is
    #   appropriate.
    _fields = ('TStart', 'TEnd', 'IPPStart', 'IPPEnd', 'IPPPoly', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    _numeric_format = {'TStart': FLOAT_FORMAT, 'TEnd': FLOAT_FORMAT, }
    # descriptors
    TStart = FloatDescriptor(
        'TStart', _required, strict=DEFAULT_STRICT,
        docstring='IPP start time relative to collection start time, i.e. offsets in seconds.')  # type: float
    TEnd = FloatDescriptor(
        'TEnd', _required, strict=DEFAULT_STRICT,
        docstring='IPP end time relative to collection start time, i.e. offsets in seconds.')  # type: float
    IPPStart = IntegerDescriptor(
        'IPPStart', _required, strict=True, docstring='Starting IPP index for the period described.')  # type: int
    IPPEnd = IntegerDescriptor(
        'IPPEnd', _required, strict=True, docstring='Ending IPP index for the period described.')  # type: int
    IPPPoly = SerializableDescriptor(
        'IPPPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='IPP index polynomial coefficients yield IPP index as a function of time.')  # type: Poly1DType
    index = IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The element array index.')  # type: int

    def __init__(
            self,
            TStart: float = None,
            TEnd: float = None,
            IPPStart: int = None,
            IPPEnd: int = None,
            IPPPoly: Union[Poly1DType, numpy.ndarray, list, tuple] = None,
            index: int = None,
            **kwargs):
        """

        Parameters
        ----------
        TStart : float
        TEnd : float
        IPPStart : int
        IPPEnd : int
        IPPPoly : Poly1DType|numpy.ndarray|list|tuple
        index : int
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TStart, self.TEnd = TStart, TEnd
        self.IPPStart, self.IPPEnd = IPPStart, IPPEnd
        self.IPPPoly = IPPPoly
        self.index = index
        super(IPPSetType, self).__init__(**kwargs)

    def _basic_validity_check(self) -> bool:
        condition = super(IPPSetType, self)._basic_validity_check()
        if self.TStart >= self.TEnd:
            self.log_validity_error(
                'TStart ({}) >= TEnd ({})'.format(self.TStart, self.TEnd))
            condition = False

        if self.IPPStart >= self.IPPEnd:
            self.log_validity_error(
                'IPPStart ({}) >= IPPEnd ({})'.format(self.IPPStart, self.IPPEnd))
            condition = False

        ipp_start_from_poly = round(self.IPPPoly(self.TStart))
        if self.IPPStart != ipp_start_from_poly:
            self.log_validity_error(
                f'IPPStart ({self.IPPStart}) inconsistent with IPPPoly(TStart) ({ipp_start_from_poly})'
            )
            condition = False

        ipp_end_from_poly = round(self.IPPPoly(self.TEnd) - 1)
        if self.IPPEnd != ipp_end_from_poly:
            self.log_validity_error(
                f'IPPEnd ({self.IPPEnd}) inconsistent with IPPPoly(TEnd) - 1 ({ipp_end_from_poly})'
            )
            condition = False

        prf = self.IPPPoly.derivative_eval((self.TStart + self.TEnd)/2)
        if prf < 0:
            self.log_validity_error(f'IPPSet has a negative PRF: {prf}')
            condition = False
        if prf > 100e3:
            self.log_validity_warning(f'IPPSet has an unreasonable PRF: {prf}')

        return condition


class TimelineType(Serializable):
    """
    The details for the imaging collection timeline.
    """

    _fields = ('CollectStart', 'CollectDuration', 'IPP')
    _required = ('CollectStart', 'CollectDuration', )
    _collections_tags = {'IPP': {'array': True, 'child_tag': 'Set'}}
    _numeric_format = {'CollectDuration': FLOAT_FORMAT, }
    # descriptors
    CollectStart = DateTimeDescriptor(
        'CollectStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The collection start time. The default precision will be microseconds.')  # type: numpy.datetime64
    CollectDuration = FloatDescriptor(
        'CollectDuration', _required, strict=DEFAULT_STRICT,
        docstring='The duration of the collection in seconds.')  # type: float
    IPP = SerializableArrayDescriptor(
        'IPP', IPPSetType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring="The Inter-Pulse Period (IPP) parameters array.")  # type: Union[SerializableArray, List[IPPSetType]]

    def __init__(
            self,
            CollectStart: Union[numpy.datetime64, datetime, date, str] = None,
            CollectDuration: float = None,
            IPP: Union[None, SerializableArray, List[IPPSetType]] = None,
            **kwargs):
        """

        Parameters
        ----------
        CollectStart : numpy.datetime64|datetime|date|str
        CollectDuration : float
        IPP : None|List[IPPSetType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CollectStart = CollectStart
        self.CollectDuration = CollectDuration
        self.IPP = IPP
        super(TimelineType, self).__init__(**kwargs)

    @property
    def CollectEnd(self) -> Optional[numpy.datetime64]:
        """
        None|numpy.datetime64: The collection end time, inferred from `CollectEnd` and `CollectDuration`,
        provided that both are populated.
        """

        if self.CollectStart is None or self.CollectDuration is None:
            return None

        return self.CollectStart + numpy.timedelta64(int(self.CollectDuration*1e6), 'us')

    def _check_ipp_consecutive(self) -> bool:
        if self.IPP is None or len(self.IPP) < 2:
            return True
        cond = True

        sorted_ipps = sorted(self.IPP, key=lambda x: x.index)
        tstarts = [x.TStart for x in sorted_ipps]
        tends = [x.TEnd for x in sorted_ipps]
        ippstarts = [x.IPPStart for x in sorted_ipps]
        ippends = [x.IPPEnd for x in sorted_ipps]
        if tstarts != sorted(tstarts):
            self.log_validity_error(f'The IPPSets are not in start time order. TStart: {tstarts}')
            cond = False

        if tends != sorted(tends):
            self.log_validity_error(f'The IPPSets are not in end time order. TEnd: {tends}')
            cond = False

        actual_indices = [x.index for x in sorted_ipps]
        if not numpy.array_equal(numpy.arange(1, len(self.IPP)+1), actual_indices):
            self.log_validity_error(f'IPPSets indices ({actual_indices}) are not 1..size; '
                                    'unable to perform adjacent IPPSet checks')
            return False

        tgaps = [ts - te for ts, te in zip(tstarts[1:], tends[:-1])]
        for ig, g in enumerate(tgaps):
            if g > 0:
                self.log_validity_error(f'There is a gap between IPPSet[index={ig+1}] and '
                                        f'IPPSet[index={ig+2}] of {g} seconds')
                cond = False
            if g < 0:
                self.log_validity_error(f'There is overlap between IPPSet[index={ig+1}] and '
                                        f'IPPSet[index={ig+2}] of {-g} seconds')
                cond = False

        igaps = [i_s - i_e for i_s, i_e in zip(ippstarts[1:], ippends[:-1])]
        for ig, g in enumerate(igaps):
            if g > 1:
                self.log_validity_error(f'There is a gap between IPPSet[index={ig+1}] and '
                                        f'IPPSet[index={ig+2}] of {g-1} IPPs')
                cond = False
            if g < 1:
                self.log_validity_error(f'There is overlap between IPPSet[index={ig+1}] and '
                                        f'IPPSet[index={ig+2}] of {1-g} IPPs')
                cond = False

        return cond

    def _check_ipp_times(self) -> bool:
        if self.IPP is None:
            return True

        cond = True
        min_time = min(x.TStart for x in self.IPP)
        max_time = max(x.TEnd for x in self.IPP)
        if min_time < 0:
            self.log_validity_error(f'Earliest TStart is negative: {min_time}')
            cond = False
        if not numpy.isclose(max_time - min_time, self.CollectDuration, atol=1e-2):
            self.log_validity_error(
                'time range in IPP entries ({}) not in keeping with populated '
                'CollectDuration ({})'.format(max_time-min_time, self.CollectDuration))
            cond = False
        return cond

    def _basic_validity_check(self) -> bool:
        condition = super(TimelineType, self)._basic_validity_check()
        condition &= self._check_ipp_consecutive()
        condition &= self._check_ipp_times()
        return condition
