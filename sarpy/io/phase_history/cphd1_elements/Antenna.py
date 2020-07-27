# -*- coding: utf-8 -*-
"""
The Antenna type definition.
"""

from typing import Union, List

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _FloatDescriptor, _StringDescriptor, \
    _SerializableDescriptor, _BooleanDescriptor, _SerializableListDescriptor
from sarpy.io.complex.sicd_elements.blocks import Poly1DType, XYZType, XYZPolyType, GainPhasePolyType
from sarpy.io.complex.sicd_elements.Antenna import EBType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class AntCoordFrameType(Serializable):
    """
    Unit vectors that describe the orientation of an Antenna Coordinate Frame
    (ACF) as function of time.
    """

    _fields = ('Identifier', 'XAxisPoly', 'YAxisPoly')
    _required = _fields
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this ACF.')  # type: str
    XAxisPoly = _SerializableDescriptor(
        'XAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna X-Axis unit vector in ECF coordinates as a function '
                  'of time.')  # type: XYZPolyType
    YAxisPoly = _SerializableDescriptor(
        'YAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna Y-Axis unit vector in ECF coordinates as a function '
                  'of time.')  # type: XYZPolyType

    def __init__(self, Identifier=None, XAxisPoly=None, YAxisPoly=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        XAxisPoly : XYZPolyType
        YAxisPoly : XYZPolyType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.XAxisPoly = XAxisPoly
        self.YAxisPoly = YAxisPoly
        super(AntCoordFrameType, self).__init__(**kwargs)


class AntPhaseCenterType(Serializable):
    """
    Parameters that describe each Antenna Phase Center (APC).
    """

    _fields = ('Identifier', 'ACFId', 'APCXYZ')
    _required = _fields
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this APC.')  # type: str
    ACFId = _StringDescriptor(
        'ACFId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of Antenna Coordinate Frame used for computing the '
                  'antenna gain and phase patterns.')  # type: str
    APCXYZ = _SerializableDescriptor(
        'APCXYZ', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The APC location in the ACF XYZ coordinate '
                  'frame.')  # type: XYZType

    def __init__(self, Identifier=None, ACFId=None, APCXYZ=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        ACFId : str
        APCXYZ : XYZType|numpy.ndarray|list|tuple
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.ACFId = ACFId
        self.APCXYZ = APCXYZ
        super(AntPhaseCenterType, self).__init__(**kwargs)


class GainPhaseArrayType(Serializable):
    """
    Parameters that identify 2-D sampled Gain & Phase patterns at single
    frequency value.
    """

    _fields = ('Freq', 'ArrayId')
    _required = ('Freq', 'ArrayId')
    # descriptors
    Freq = _FloatDescriptor(
        'Freq', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Frequency value for which the sampled Array and Element '
                  'pattern(s) are provided, in Hz.')  # type: float
    ArrayId = _StringDescriptor(
        'ArrayId', _required, strict=DEFAULT_STRICT,
        docstring='Support array identifier of the sampled gain/phase of the array '
                  'at ref Frequency.')  # type: str
    ElementId = _StringDescriptor(
        'ElementId', _required, strict=DEFAULT_STRICT,
        docstring='Support array identifier of the sampled gain/phase of the element '
                  'at ref frequency.')  # type: str

    def __init__(self, Freq=None, ArrayId=None, ElementId=None, **kwargs):
        """

        Parameters
        ----------
        Freq : float
        ArrayId : str
        ElementId : None|str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Freq = Freq
        self.ArrayId = ArrayId
        self.ElementId = ElementId
        super(GainPhaseArrayType, self).__init__(**kwargs)


class AntPatternType(Serializable):
    """
    Parameter set that defines each Antenna Pattern as function time.
    """

    _fields = (
        'Identifier', 'FreqZero', 'GainZero', 'EBFreqShift', 'MLFreqDilation',
        'GainBSPoly', 'EB', 'Array', 'Element', 'GainPhaseArray')
    _required = (
        'Identifier', 'FreqZero', 'EB', 'Array', 'Element')
    _collections_tags = {
        'GainPhaseArray': {'array': False, 'child_tag': 'GainPhaseArray'}}
    _numeric_format = {'FreqZero': '0.16G', 'GainZero': '0.16G'}
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this ACF.')  # type: str
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring='The reference frequency value for which the Electrical Boresight '
                  'and array pattern polynomials are computed.')  # type: float
    GainZero = _FloatDescriptor(
        'GainZero', _required, strict=DEFAULT_STRICT,
        docstring='The reference antenna gain at zero steering angle at the '
                  'reference frequency, measured in dB.')  # type: float
    EBFreqShift = _BooleanDescriptor(
        'EBFreqShift', _required, strict=DEFAULT_STRICT,
        docstring="Parameter indicating whether the electronic boresite shifts with "
                  "frequency.")  # type: bool
    MLFreqDilation = _BooleanDescriptor(
        'MLFreqDilation', _required, strict=DEFAULT_STRICT,
        docstring="Parameter indicating the mainlobe (ML) width changes with "
                  "frequency.")  # type: bool
    GainBSPoly = _SerializableDescriptor(
        'GainBSPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Gain polynomial *(in dB)* as a function of frequency for boresight *(BS)* '
                  'at :math:`DCX=0, DCY=0`. '
                  'Frequency ratio :math:`(f-f0)/f0` is the input variable, and the constant '
                  'coefficient is always `0.0`.')  # type: Poly1DType
    EB = _SerializableDescriptor(
        'EB', EBType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight *(EB)* steering directions for an electronically '
                  'steered array.')  # type: EBType
    Array = _SerializableDescriptor(
        'Array', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='Array pattern polynomials that define the shape of the '
                  'main-lobe.')  # type: GainPhasePolyType
    Element = _SerializableDescriptor(
        'Element', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='Element array pattern polynomials for electronically steered '
                  'arrays.')  # type: GainPhasePolyType
    GainPhaseArray = _SerializableListDescriptor(
        'GainPhaseArray', GainPhaseArrayType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Array of parameters that identify 2-D sampled Gain and Phase patterns at '
                  'single frequency value.')  # type: Union[None, List[GainPhaseArrayType]]

    def __init__(self, Identifier=None, FreqZero=None, GainZero=None, EBFreqShift=None,
                 MLFreqDilation=None, GainBSPoly=None, EB=None, Array=None, Element=None,
                 GainPhaseArray=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        FreqZero : float
        GainZero : float
        EBFreqShift : bool
        MLFreqDilation : bool
        GainBSPoly : None|Poly1DType
        EB : EBType
        Array : GainPhasePolyType
        Element : GainPhasePolyType
        GainPhaseArray : None|List[GainPhaseArrayType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.FreqZero = FreqZero
        self.GainZero = GainZero
        self.EBFreqShift = EBFreqShift
        self.MLFreqDilation = MLFreqDilation
        self.GainBSPoly = GainBSPoly
        self.EB = EB
        self.Array = Array
        self.Element = Element
        self.GainPhaseArray = GainPhaseArray
        super(AntPatternType, self).__init__(**kwargs)


class AntennaType(Serializable):
    """
    Parameters that describe the transmit and receive antennas used to collect
    the signal array(s).
    """

    _fields = (
        'NumACFs', 'NumAPCs', 'NumAntPats', 'AntCoordFrame', 'AntPhaseCenter', 'AntPattern')
    _required = ('AntCoordFrame', 'AntPhaseCenter', 'AntPattern')
    _collections_tags = {
        'AntCoordFrame': {'array': False, 'child_tag': 'AntCoordFrame'},
        'AntPhaseCenter': {'array': False, 'child_tag': 'AntPhaseCenter'},
        'AntPattern': {'array': False, 'child_tag': 'AntPattern'}}
    # descriptors
    AntCoordFrame = _SerializableListDescriptor(
        'AntCoordFrame', AntCoordFrameType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Unit vectors that describe the orientation of an Antenna Coordinate Frame (ACF) '
                  'as function of time. Parameter set repeated for '
                  'each ACF.')  # type: List[AntCoordFrameType]
    AntPhaseCenter = _SerializableListDescriptor(
        'AntPhaseCenter', AntPhaseCenterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe each Antenna Phase Center (APC). Parameter '
                  'set repeated for each APC.')  # type: List[AntPhaseCenterType]
    AntPattern = _SerializableListDescriptor(
        'AntPattern', AntPatternType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameter set that defines each Antenna Pattern as function time. Parameters '
                  'set repeated for each Antenna Pattern.')  # type: List[AntPatternType]

    def __init__(self, AntCoordFrame=None, AntPhaseCenter=None, AntPattern=None, **kwargs):
        """

        Parameters
        ----------
        AntCoordFrame : List[AntCoordFrameType]
        AntPhaseCenter : List[AntPhaseCenterType]
        AntPattern : List[AntPatternType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.AntCoordFrame = AntCoordFrame
        self.AntPhaseCenter = AntPhaseCenter
        self.AntPattern = AntPattern
        super(AntennaType, self).__init__(**kwargs)

    @property
    def NumACFs(self):
        """
        int: The number of antenna coordinate frame elements.
        """

        if self.AntCoordFrame is None:
            return 0
        return len(self.AntCoordFrame)

    @property
    def NumAPCs(self):
        """
        int: The number of antenna phase center elements.
        """

        if self.AntPhaseCenter is None:
            return 0
        return len(self.AntPhaseCenter)

    @property
    def NumAntPats(self):
        """
        int: The number of antenna pattern elements.
        """

        if self.AntPattern is None:
            return 0
        return len(self.AntPattern)