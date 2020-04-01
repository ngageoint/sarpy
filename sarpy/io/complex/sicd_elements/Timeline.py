# -*- coding: utf-8 -*-
"""
The TimelineType definition.
"""

from typing import List, Union

import numpy

from .base import Serializable, DEFAULT_STRICT, \
    _FloatDescriptor, _IntegerDescriptor, _DateTimeDescriptor, \
    _SerializableArrayDescriptor, _SerializableDescriptor
from .blocks import Poly1DType


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


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
    _numeric_format = {'TStart': '0.16G', 'TEnd': '0.16G', }
    # descriptors
    TStart = _FloatDescriptor(
        'TStart', _required, strict=DEFAULT_STRICT,
        docstring='IPP start time relative to collection start time, i.e. offsets in seconds.')  # type: float
    TEnd = _FloatDescriptor(
        'TEnd', _required, strict=DEFAULT_STRICT,
        docstring='IPP end time relative to collection start time, i.e. offsets in seconds.')  # type: float
    IPPStart = _IntegerDescriptor(
        'IPPStart', _required, strict=True, docstring='Starting IPP index for the period described.')  # type: int
    IPPEnd = _IntegerDescriptor(
        'IPPEnd', _required, strict=True, docstring='Ending IPP index for the period described.')  # type: int
    IPPPoly = _SerializableDescriptor(
        'IPPPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='IPP index polynomial coefficients yield IPP index as a function of time.')  # type: Poly1DType
    index = _IntegerDescriptor(
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


class TimelineType(Serializable):
    """
    The details for the imaging collection timeline.
    """

    _fields = ('CollectStart', 'CollectDuration', 'IPP')
    _required = ('CollectStart', 'CollectDuration', )
    _collections_tags = {'IPP': {'array': True, 'child_tag': 'Set'}}
    _numeric_format = {'CollectDuration': '0.16G', }
    # descriptors
    CollectStart = _DateTimeDescriptor(
        'CollectStart', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='The collection start time. The default precision will be microseconds.')  # type: numpy.datetime64
    CollectDuration = _FloatDescriptor(
        'CollectDuration', _required, strict=DEFAULT_STRICT,
        docstring='The duration of the collection in seconds.')  # type: float
    IPP = _SerializableArrayDescriptor(
        'IPP', IPPSetType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring="The Inter-Pulse Period (IPP) parameters array.")  # type: Union[numpy.ndarray, List[IPPSetType]]

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
