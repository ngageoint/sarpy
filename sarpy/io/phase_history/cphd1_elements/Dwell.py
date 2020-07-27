# -*- coding: utf-8 -*-
"""
The Dwell parameters definition.
"""

from typing import List

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _StringDescriptor, \
    _SerializableDescriptor, _SerializableListDescriptor
from sarpy.io.complex.sicd_elements.blocks import Poly2DType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CODTimeType(Serializable):
    """
    Center of Dwell (COD) time polynomial object.
    """

    _fields = ('Identifier', 'CODTimePoly')
    _required = _fields
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this COD Time '
                  'polynomial.')  # type: str
    CODTimePoly = _SerializableDescriptor(
        'CODTimePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='The polynomial.')  # type: Poly2DType

    def __init__(self, Identifier=None, CODTimePoly=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        CODTimePoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.CODTimePoly = CODTimePoly
        super(CODTimeType, self).__init__(**kwargs)


class DwellTimeType(Serializable):
    """
    The dwell time polynomial object.
    """
    _fields = ('Identifier', 'DwellTimePoly')
    _required = _fields
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this Dwell Time '
                  'polynomial.')  # type: str
    DwellTimePoly = _SerializableDescriptor(
        'DwellTimePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='The polynomial.')  # type: Poly2DType

    def __init__(self, Identifier=None, DwellTimePoly=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        DwellTimePoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.DwellTimePoly = DwellTimePoly
        super(DwellTimeType, self).__init__(**kwargs)


class DwellType(Serializable):
    """
    Parameters that specify the dwell time supported by the signal arrays
    contained in the CPHD product.
    """

    _fields = ('NumCODTimes', 'CODTimes', 'NumDwellTimes', 'DwellTimes')
    _required = ('CODTimes', 'DwellTimes')
    _collections_tags = {
        'CODTimes': {'array': False, 'child_tag': 'CODTime'},
        'DwellTimes': {'array': False, 'child_tag': 'DwellTime'}}
    # descriptors
    CODTimes = _SerializableListDescriptor(
        'CODTimes', CODTimeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The Center of Dwell (COD) time polynomials.')  # type: List[CODTimeType]
    DwellTimes = _SerializableListDescriptor(
        'DwellTimes', DwellTimeType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The dwell time polynomials.')  # type: List[DwellTimeType]

    def __init__(self, CODTimes=None, DwellTimes=None, **kwargs):
        """

        Parameters
        ----------
        CODTimes : List[CODTimeType]
        DwellTimes : List[DwellTimeType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CODTimes = CODTimes
        self.DwellTimes = DwellTimes
        super(DwellType, self).__init__(**kwargs)

    @property
    def NumCODTimes(self):
        """
        int: The number of cod time polynomial elements.
        """

        if self.CODTimes is None:
            return 0
        else:
            return len(self.CODTimes)

    @property
    def NumDwellTimes(self):
        """
        int: The number of dwell time polynomial elements.
        """

        if self.DwellTimes is None:
            return 0
        else:
            return len(self.DwellTimes)
