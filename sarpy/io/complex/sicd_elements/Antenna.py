# -*- coding: utf-8 -*-
"""
The AntennaType definition.
"""

import numpy

from .base import Serializable, DEFAULT_STRICT, _BooleanDescriptor, _FloatDescriptor, \
    _SerializableDescriptor
from .blocks import Poly1DType, XYZPolyType, GainPhasePolyType


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class EBType(Serializable):
    """
    Electrical boresight (EB) steering directions for an electronically steered array.
    """

    _fields = ('DCXPoly', 'DCYPoly')
    _required = _fields
    # descriptors
    DCXPoly = _SerializableDescriptor(
        'DCXPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight steering *X-axis direction cosine (DCX)* as a function of '
                  'slow time ``(variable 1)``.')  # type: Poly1DType
    DCYPoly = _SerializableDescriptor(
        'DCYPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight steering *Y-axis direction cosine (DCY)* as a function of '
                  'slow time ``(variable 1)``.')  # type: Poly1DType

    def __init__(self, DCXPoly=None, DCYPoly=None, **kwargs):
        """
        Parameters
        ----------
        DCXPoly : Poly1DType|numpy.ndarray|list|tuple
        DCYPoly : Poly1DType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DCXPoly = DCXPoly
        self.DCYPoly = DCYPoly
        super(EBType, self).__init__(**kwargs)

    def __call__(self, t):
        """
        Evaluate the polynomial at points `t`. This passes `t` straight through
        to :func:`polyval` of `numpy.polynomial.polynomial` for each of
        `DCXPoly,DCYPoly` components. If any of `DCXPoly,DCYPoly` is not populated,
        then `None` is returned.

        Parameters
        ----------
        t : float|int|numpy.ndarray
            The point(s) at which to evaluate.

        Returns
        -------
        None|numpy.ndarray
        """

        if self.DCXPoly is None or self.DCYPoly is None:
            return None
        return numpy.array([self.DCXPoly(t), self.DCYPoly(t)])


class AntParamType(Serializable):
    """
    The antenna parameters container.
    """

    _fields = (
        'XAxisPoly', 'YAxisPoly', 'FreqZero', 'EB', 'Array', 'Elem', 'GainBSPoly', 'EBFreqShift', 'MLFreqDilation')
    _required = ('XAxisPoly', 'YAxisPoly', 'FreqZero', 'Array')
    _numeric_format = {'FreqZero': '0.16G'}
    # descriptors
    XAxisPoly = _SerializableDescriptor(
        'XAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna X-Axis unit vector in ECF coordinates as a function of time ``(variable 1)``.'
    )  # type: XYZPolyType
    YAxisPoly = _SerializableDescriptor(
        'YAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna Y-Axis unit vector in ECF coordinates as a function of time ``(variable 1)``.'
    )  # type: XYZPolyType
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring='RF frequency *(f0)* used to specify the array pattern and electrical boresite *(EB)* '
                  'steering direction cosines.')  # type: float
    EB = _SerializableDescriptor(
        'EB', EBType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight *(EB)* steering directions for an electronically steered array.')  # type: EBType
    Array = _SerializableDescriptor(
        'Array', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='Array pattern polynomials that define the shape of the main-lobe.')  # type: GainPhasePolyType
    Elem = _SerializableDescriptor(
        'Elem', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='Element array pattern polynomials for electronically steered arrays.')  # type: GainPhasePolyType
    GainBSPoly = _SerializableDescriptor(
        'GainBSPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Gain polynomial *(in dB)* as a function of frequency for boresight *(BS)* at :math:`DCX=0, DCY=0`. '
                  'Frequency ratio :math:`(f-f0)/f0` is the input variable ``(variable 1)``, and the constant '
                  'coefficient is always `0.0`.')  # type: Poly1DType
    EBFreqShift = _BooleanDescriptor(
        'EBFreqShift', _required, strict=DEFAULT_STRICT,
        docstring="""
        Parameter indicating whether the electronic boresite shifts with frequency for an electronically steered array.
        
        * `False` - No shift with frequency.
        
        * `True` - Shift with frequency per ideal array theory.
        
        """)  # type: bool
    MLFreqDilation = _BooleanDescriptor(
        'MLFreqDilation', _required, strict=DEFAULT_STRICT,
        docstring="""
        Parameter indicating the mainlobe (ML) width changes with frequency.
        
        * `False` - No change with frequency.
        
        * `True` - Change with frequency per ideal array theory.
        
        """)  # type: bool

    def __init__(self, XAxisPoly=None, YAxisPoly=None, FreqZero=None, EB=None,
                 Array=None, Elem=None, GainBSPoly=None, EBFreqShift=None, MLFreqDilation=None, **kwargs):
        """
        Parameters
        ----------
        XAxisPoly : XYZPolyType
        YAxisPoly : XYZPolyType
        FreqZero : float
        EB : EBType
        Array : GainPhasePolyType
        Elem : GainPhasePolyType
        GainBSPoly : Poly1DType|numpy.ndarray|list|tuple
        EBFreqShift : bool
        MLFreqDilation : bool
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.XAxisPoly, self.YAxisPoly = XAxisPoly, YAxisPoly
        self.FreqZero = FreqZero
        self.EB = EB
        self.Array, self.Elem = Array, Elem
        self.GainBSPoly = GainBSPoly
        self.EBFreqShift, self.MLFreqDilation = EBFreqShift, MLFreqDilation
        super(AntParamType, self).__init__(**kwargs)

    def _apply_reference_frequency(self, reference_frequency):
        if self.FreqZero is not None:
            self.FreqZero += reference_frequency


class AntennaType(Serializable):
    """Parameters that describe the antenna illumination patterns during the collection."""
    _fields = ('Tx', 'Rcv', 'TwoWay')
    _required = ()
    # descriptors
    Tx = _SerializableDescriptor(
        'Tx', AntParamType, _required, strict=DEFAULT_STRICT,
        docstring='The transmit antenna parameters.')  # type: AntParamType
    Rcv = _SerializableDescriptor(
        'Rcv', AntParamType, _required, strict=DEFAULT_STRICT,
        docstring='The receive antenna parameters.')  # type: AntParamType
    TwoWay = _SerializableDescriptor(
        'TwoWay', AntParamType, _required, strict=DEFAULT_STRICT,
        docstring='The bidirectional transmit/receive antenna parameters.')  # type: AntParamType

    def __init__(self, Tx=None, Rcv=None, TwoWay=None, **kwargs):
        """

        Parameters
        ----------
        Tx : AntParamType
        Rcv : AntParamType
        TwoWay : AntParamType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Tx, self.Rcv, self.TwoWay = Tx, Rcv, TwoWay
        super(AntennaType, self).__init__(**kwargs)

    def _apply_reference_frequency(self, reference_frequency):
        """
        If the reference frequency is used, adjust the necessary fields accordingly.
        Expected to be called by SICD parent.

        Parameters
        ----------
        reference_frequency : float
            The reference frequency.

        Returns
        -------
        None
        """

        if self.Tx is not None:
            # noinspection PyProtectedMember
            self.Tx._apply_reference_frequency(reference_frequency)
        if self.Rcv is not None:
            # noinspection PyProtectedMember
            self.Rcv._apply_reference_frequency(reference_frequency)
        if self.TwoWay is not None:
            # noinspection PyProtectedMember
            self.TwoWay._apply_reference_frequency(reference_frequency)
