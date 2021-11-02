"""
The TimelineType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import List, Union

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

    def __init__(self, TStart=None, TEnd=None, IPPStart=None, IPPEnd=None, IPPPoly=None, index=None, **kwargs):
        """

        Parameters
        ----------
        TStart : float
        TEnd : float
        IPPStart : int
        IPPEnd : int
        IPPPoly : Poly1DType|numpy.ndarray|list|tuple
        index : int
        kwargs : dict
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

    def _basic_validity_check(self):
        condition = super(IPPSetType, self)._basic_validity_check()
        if self.TStart >= self.TEnd:
            self.log_validity_error(
                'TStart ({}) >= TEnd ({})'.format(self.TStart, self.TEnd))
            condition = False

        if self.IPPStart >= self.IPPEnd:
            self.log_validity_error(
                'IPPStart ({}) >= IPPEnd ({})'.format(self.IPPStart, self.IPPEnd))
            condition = False

        exp_ipp_start = self.IPPPoly(self.TStart)
        exp_ipp_end = self.IPPPoly(self.TEnd)
        if abs(exp_ipp_start - self.IPPStart) > 1:
            self.log_validity_error(
                'IPPStart populated as {}, inconsistent with value ({}) '
                'derived from IPPPoly and TStart'.format(exp_ipp_start, self.IPPStart))
        if abs(exp_ipp_end - self.IPPEnd) > 1:
            self.log_validity_error(
                'IPPEnd populated as {}, inconsistent with value ({}) '
                'derived from IPPPoly and TEnd'.format(self.IPPEnd, exp_ipp_end))
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

    def __init__(self, CollectStart=None, CollectDuration=None, IPP=None, **kwargs):
        """

        Parameters
        ----------
        CollectStart : numpy.datetime64|datetime|date|str
        CollectDuration : float
        IPP : List[IPPSetType]
        kwargs : dict
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
    def CollectEnd(self):
        """
        None|numpy.datetime64: The collection end time, inferred from `CollectEnd` and `CollectDuration`,
        provided that both are populated.
        """

        if self.CollectStart is None or self.CollectDuration is None:
            return None

        return self.CollectStart + numpy.timedelta64(int(self.CollectDuration*1e6), 'us')

    def _check_ipp_consecutive(self):
        if self.IPP is None or len(self.IPP) < 2:
            return True
        cond = True
        for i in range(len(self.IPP)-1):
            el1 = self.IPP[i]
            el2 = self.IPP[i+1]
            if el1.IPPEnd + 1 != el2.IPPStart:
                self.log_validity_error(
                    'IPP entry {} IPPEnd ({}) is not consecutive with '
                    'entry {} IPPStart ({})'.format(i, el1.IPPEnd, i+1, el2.IPPStart))
                cond = False
            if el1.TEnd >= el2.TStart:
                self.log_validity_error(
                    'IPP entry {} TEnd ({}) is greater than entry {} TStart ({})'.format(i, el1.TEnd, i+1, el2.TStart))
        return cond

    def _check_ipp_times(self):
        if self.IPP is None:
            return True

        cond = True
        min_time = self.IPP[0].TStart
        max_time = self.IPP[0].TEnd
        for i in range(len(self.IPP)):
            entry = self.IPP[i]
            if entry.TStart < 0:
                self.log_validity_error('IPP entry {} has negative TStart ({})'.format(i, entry.TStart))
                cond = False
            if entry.TEnd > self.CollectDuration + 1e-2:
                self.log_validity_error(
                    'IPP entry {} has TEnd ({}) appreciably larger than '
                    'CollectDuration ({})'.format(i, entry.TEnd, self.CollectDuration))
                cond = False
            min_time = min(min_time, entry.TStart)
            max_time = max(max_time, entry.TEnd)
        if abs(max_time - min_time - self.CollectDuration) > 1e-2:
            self.log_validity_error(
                'time range in IPP entries ({}) not in keeping with populated '
                'CollectDuration ({})'.format(max_time-min_time, self.CollectDuration))
            cond = False
        return cond

    def _basic_validity_check(self):
        condition = super(TimelineType, self)._basic_validity_check()
        condition &= self._check_ipp_consecutive()
        condition &= self._check_ipp_times()
        return condition
