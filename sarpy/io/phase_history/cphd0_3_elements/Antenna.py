# -*- coding: utf-8 -*-
"""
The Antenna definition for CPHD 0.3.
"""

from typing import Union, List

from sarpy.io.phase_history.cphd1_elements.base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _SerializableListDescriptor, \
    _FloatDescriptor, _SerializableDescriptor
from sarpy.io.complex.sicd_elements.blocks import XYZPolyType
from sarpy.io.complex.sicd_elements.Antenna import AntParamType as AntParamTypeBase

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class HPBWType(Serializable):
    """
    Half power beamwidth parameters.
    """

    _fields = ('DCX', 'DCY')
    _required = _fields
    _numeric_format = {'DCX': '0.16G', 'DCY': '0.16G'}
    # descriptors
    DCX = _FloatDescriptor(
        'DCX', _required, strict=DEFAULT_STRICT,
        docstring='Half power beamwidth in the X-axis direction cosine '
                  '(DCX).')  # type: float
    DCY = _FloatDescriptor(
        'DCY', _required, strict=DEFAULT_STRICT,
        docstring='Half power beamwidth in the Y -axis direction cosine '
                  '(DCY).')  # type: float

    def __init__(self, DCX=None, DCY=None, **kwargs):
        """

        Parameters
        ----------
        DCX : float
        DCY : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DCX = DCX
        self.DCY = DCY
        super(HPBWType, self).__init__(**kwargs)


class AntParamType(AntParamTypeBase):
    """
    The antenna parameters container.
    """

    _fields = (
        'XAxisPoly', 'YAxisPoly', 'FreqZero', 'EB', 'HPBW', 'Array', 'Elem',
        'GainBSPoly', 'EBFreqShift', 'MLFreqDilation')
    _required = ('XAxisPoly', 'YAxisPoly', 'FreqZero', )
    _numeric_format = {'FreqZero': '0.16G'}
    # descriptors
    HPBW = _SerializableDescriptor(
        'HPBW', HPBWType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, HPBWType]

    def __init__(self, XAxisPoly=None, YAxisPoly=None, FreqZero=None, EB=None,
                 HPBW=None, Array=None, Elem=None, GainBSPoly=None, EBFreqShift=None,
                 MLFreqDilation=None, **kwargs):
        """
        Parameters
        ----------
        XAxisPoly : XYZPolyType
        YAxisPoly : XYZPolyType
        FreqZero : float
        EB : None|EBType
        HPBW : None|HPBWType
        Array : None|GainPhasePolyType
        Elem : None|GainPhasePolyType
        GainBSPoly : None|Poly1DType|numpy.ndarray|list|tuple
        EBFreqShift : None|bool
        MLFreqDilation : None|bool
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.HPBW = HPBW
        super(AntParamType, self).__init__(
            XAxisPoly=XAxisPoly, YAxisPoly=YAxisPoly, FreqZero=FreqZero, EB=EB,
            Array=Array, Elem=Elem, GainBSPoly=GainBSPoly, EBFreqShift=EBFreqShift,
            MLFreqDilation=MLFreqDilation, **kwargs)


class AntennaType(Serializable):
    """
    Antenna parameters that describe antenna orientation, mainlobe steering and
    gain patterns vs. time.
    """

    _fields = ('NumTxAnt', 'NumRcvAnt', 'NumTWAnt', 'Tx', 'Rcv', 'TwoWay')
    _required = ()
    _collections_tags = {
        'Tx': {'array': False, 'child_tag': 'Tx'},
        'Rcv': {'array': False, 'child_tag': 'Rcv'},
        'TwoWay': {'array': False, 'child_tag': 'TwoWay'}}
    # descriptors
    Tx = _SerializableListDescriptor(
        'Tx', AntParamType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Transmit antenna pattern parameters.'
    )  # type: Union[None, List[AntParamType]]
    Rcv = _SerializableListDescriptor(
        'Rcv', AntParamType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Receive antenna pattern parameters.'
    )  # type: Union[None, List[AntParamType]]
    TwoWay = _SerializableListDescriptor(
        'TwoWay', AntParamType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Two-way antenna pattern parameters.'
    )  # type: Union[None, List[AntParamType]]

    def __init__(self, Tx=None, Rcv=None, TwoWay=None, **kwargs):
        """

        Parameters
        ----------
        Tx : None|List[AntParamType]
        Rcv : None|List[AntParamType]
        TwoWay : None|List[AntParamType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Tx = Tx
        self.Rcv = Rcv
        self.TwoWay = TwoWay
        super(AntennaType, self).__init__(**kwargs)

    @property
    def NumTxAnt(self):
        """
        int: The number of transmit elements.
        """

        if self.Tx is None:
            return 0
        return len(self.Tx)

    @property
    def NumRcvAnt(self):
        """
        int: The number of receive elements.
        """

        if self.Rcv is None:
            return 0
        return len(self.Rcv)

    @property
    def NumTWAnt(self):
        """
        int: The number of two way elements.
        """

        if self.TwoWay is None:
            return 0
        return len(self.TwoWay)
