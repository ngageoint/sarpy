"""
The Antenna type definition.
"""

from typing import List

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import Serializable, _FloatDescriptor, _StringDescriptor, \
    _SerializableDescriptor, _BooleanDescriptor, _SerializableListDescriptor
from sarpy.io.complex.sicd_elements.blocks import Poly1DType
from sarpy.io.phase_history.cphd1_elements.Antenna import AntPhaseCenterType

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Michael Stewart, Valkyrie")


class AntCoordFrameType(Serializable):
    """
    Antenna coordinate frame (ACF) in which one or more phase centers may lie.
    """

    _fields = ('Identifier', )
    _required = _fields
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this ACF.')  # type: str

    def __init__(self, Identifier=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        super(AntCoordFrameType, self).__init__(**kwargs)


class AntPatternType(Serializable):
    """
    Parameter set that defines each one-way Antenna Pattern.
    """

    _fields = (
        'Identifier', 'FreqZero', 'EBFreqShift', 'MLFreqDilation',
        'GainZero', 'GainBSPoly', 'ArrayGPId', 'ElementGPId')
    _required = (
        'Identifier', 'FreqZero', 'EBFreqShift', 'MLFreqDilation',
        'ArrayGPId', 'ElementGPId')
    _numeric_format = {'FreqZero': '0.16G', 'GainZero': '0.16G'}
    # descriptors
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this Antenna Pattern')  # type: str
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring='The reference frequency value for which the patterns are computed.')  # type: float
    EBFreqShift = _BooleanDescriptor(
        'EBFreqShift', _required, strict=DEFAULT_STRICT,
        docstring="Parameter indicating whether the electronic boresight shifts with "
                  "frequency.")  # type: bool
    MLFreqDilation = _BooleanDescriptor(
        'MLFreqDilation', _required, strict=DEFAULT_STRICT,
        docstring="Parameter indicating the mainlobe (ML) width changes with "
                  "frequency.")  # type: bool
    GainZero = _FloatDescriptor(
        'GainZero', _required, strict=DEFAULT_STRICT,
        docstring='The reference antenna gain at zero steering angle at the '
                  'reference frequency, measured in dB.')  # type: float
    GainBSPoly = _SerializableDescriptor(
        'GainBSPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Gain polynomial *(in dB)* as a function of frequency for boresight *(BS)* '
                  'at :math:`DCX=0, DCY=0`. '
                  'Frequency ratio :math:`(f-f0)/f0` is the input variable, and the constant '
                  'coefficient is always `0.0`.')  # type: Poly1DType
    ArrayGPId = _StringDescriptor(
        'ArrayGPId', _required, strict=DEFAULT_STRICT,
        docstring='Support array identifier of the sampled gain/phase of the array '
                  'at ref frequency.')  # type: str
    ElementGPId = _StringDescriptor(
        'ElementGPId', _required, strict=DEFAULT_STRICT,
        docstring='Support array identifier of the sampled gain/phase of the element '
                  'at ref frequency.')  # type: str

    def __init__(self, Identifier=None, FreqZero=None, EBFreqShift=None, MLFreqDilation=None,
                 GainZero=None, GainBSPoly=None, ArrayGPId=None, ElementGPId=None, **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        FreqZero : float
        EBFreqShift : bool
        MLFreqDilation : bool
        GainZero : None|float
        GainBSPoly : None|Poly1DType
        ArrayGPId : str
        ElementGPId : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.FreqZero = FreqZero
        self.EBFreqShift = EBFreqShift
        self.MLFreqDilation = MLFreqDilation
        self.GainZero = GainZero
        self.GainBSPoly = GainBSPoly
        self.ArrayGPId = ArrayGPId
        self.ElementGPId = ElementGPId
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
        docstring='Antenna coordinate frame (ACF) in which one or more phase centers'
                  ' may lie.')  # type: List[AntCoordFrameType]
    AntPhaseCenter = _SerializableListDescriptor(
        'AntPhaseCenter', AntPhaseCenterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe each Antenna Phase Center (APC). Parameter '
                  'set repeated for each APC.')  # type: List[AntPhaseCenterType]
    AntPattern = _SerializableListDescriptor(
        'AntPattern', AntPatternType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameter set that defines each one-way Antenna Pattern.')  # type: List[AntPatternType]

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
