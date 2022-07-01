"""
The Antenna type definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List, Tuple, Optional

import numpy

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from sarpy.io.complex.sicd_elements.blocks import Poly1DType, XYZType, \
    XYZPolyType, Poly2DType
from sarpy.io.xml.base import Serializable, Arrayable
from sarpy.io.xml.descriptors import FloatDescriptor, StringDescriptor, SerializableDescriptor, \
    BooleanDescriptor, SerializableListDescriptor


class AntCoordFrameType(Serializable):
    """
    Unit vectors that describe the orientation of an Antenna Coordinate Frame
    (ACF) as function of time.
    """

    _fields = ('Identifier', 'XAxisPoly', 'YAxisPoly', 'UseACFPVP')
    _required = ('Identifier', 'XAxisPoly', 'YAxisPoly')
    # descriptors
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this ACF.')  # type: str
    XAxisPoly = SerializableDescriptor(
        'XAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna X-Axis unit vector in ECF coordinates as a function '
                  'of time.')  # type: XYZPolyType
    YAxisPoly = SerializableDescriptor(
        'YAxisPoly', XYZPolyType, _required, strict=DEFAULT_STRICT,
        docstring='Antenna Y-Axis unit vector in ECF coordinates as a function '
                  'of time.')  # type: XYZPolyType
    UseACFPVP = BooleanDescriptor(
        'UseACFPVP', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: bool

    def __init__(
            self,
            Identifier: str = None,
            XAxisPoly: XYZPolyType = None,
            YAxisPoly: XYZPolyType = None,
            UseACFPVP: Optional[bool] = None,
            **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        XAxisPoly : XYZPolyType
        YAxisPoly : XYZPolyType
        UseACFPVP : None|bool
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifier = Identifier
        self.XAxisPoly = XAxisPoly
        self.YAxisPoly = YAxisPoly
        self.UseACFPVP = UseACFPVP
        super(AntCoordFrameType, self).__init__(**kwargs)

    def version_required(self) -> Tuple[int, int, int]:
        required = (1, 0, 1)
        if self.UseACFPVP is not None:
            required = max(required, (1, 1, 0))
        return required


class AntPhaseCenterType(Serializable):
    """
    Parameters that describe each Antenna Phase Center (APC).
    """

    _fields = ('Identifier', 'ACFId', 'APCXYZ')
    _required = _fields
    # descriptors
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this APC.')  # type: str
    ACFId = StringDescriptor(
        'ACFId', _required, strict=DEFAULT_STRICT,
        docstring='Identifier of Antenna Coordinate Frame used for computing the '
                  'antenna gain and phase patterns.')  # type: str
    APCXYZ = SerializableDescriptor(
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

    _fields = ('Freq', 'ArrayId', 'ElementId')
    _required = ('Freq', 'ArrayId')
    _numeric_format = {'Freq', FLOAT_FORMAT}
    # descriptors
    Freq = FloatDescriptor(
        'Freq', _required, strict=DEFAULT_STRICT, bounds=(0, None),
        docstring='Frequency value for which the sampled Array and Element '
                  'pattern(s) are provided, in Hz.')  # type: float
    ArrayId = StringDescriptor(
        'ArrayId', _required, strict=DEFAULT_STRICT,
        docstring='Support array identifier of the sampled gain/phase of the array '
                  'at ref Frequency.')  # type: str
    ElementId = StringDescriptor(
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


class FreqSFType(Serializable, Arrayable):
    _fields = ('DCXSF', 'DCYSF')
    _required = _fields
    _numeric_format = {key: '0.17E' for key in _fields}
    DCXSF = FloatDescriptor(
        'DCXSF', _required, strict=DEFAULT_STRICT, bounds=(0.0, 1.0),
        docstring='')  # type: float
    DCYSF = FloatDescriptor(
        'DCYSF', _required, strict=DEFAULT_STRICT, bounds=(0.0, 1.0),
        docstring='')  # type: float

    def __init__(self, DCXSF=None, DCYSF=None, **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DCXSF = DCXSF
        self.DCYSF = DCYSF
        super(FreqSFType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64) -> numpy.ndarray:
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [DCXSF, DCYSF]
        """

        return numpy.array([self.DCXSF, self.DCYSF], dtype=dtype)

    @classmethod
    def from_array(cls, array: numpy.ndarray):
        """
        Construct from a iterable.

        Parameters
        ----------
        array : numpy.ndarray|list|tuple

        Returns
        -------
        FreqSFType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError(
                    'Expected array to be of length 2,\n\t'
                    'and received `{}`'.format(array))
            return cls(DCXSF=array[0], DCYSF=array[1])
        raise ValueError(
            'Expected array to be numpy.ndarray, list, or tuple,\n\t'
            'got `{}`'.format(type(array)))


class AntPolRefType(Serializable, Arrayable):
    """
    Polarization reference type.
    """
    _fields = ('AmpX', 'AmpY', 'PhaseY')
    _required = _fields
    _numeric_format = {key: '0.17E' for key in _fields}
    AmpX = FloatDescriptor(
        'AmpX', _required, strict=DEFAULT_STRICT, bounds=(0.0, 1.0),
        docstring='E-field relative amplitude in ACF X direction')  # type: float
    AmpY = FloatDescriptor(
        'AmpY', _required, strict=DEFAULT_STRICT, bounds=(0.0, 1.0),
        docstring='E-field relative amplitude in ACF Y direction')  # type: float
    PhaseY = FloatDescriptor(
        'PhaseY', _required, strict=DEFAULT_STRICT, bounds=(-0.5, 0.5),
        docstring='Relative phase of the Y E-field '
                  'relative to the X E-field at f=f_0')  # type: float

    def __init__(
            self,
            AmpX: float = None,
            AmpY: float = None,
            PhaseY: float = None,
            **kwargs):
        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.AmpX = AmpX
        self.AmpY = AmpY
        self.PhaseY = PhaseY
        super(AntPolRefType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64) -> numpy.ndarray:
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [AmpX, AmpY, PhaseY]
        """

        return numpy.array([self.AmpX, self.AmpY, self.PhaseY], dtype=dtype)

    @classmethod
    def from_array(cls, array: numpy.ndarray):
        """
        Construct from a iterable.

        Parameters
        ----------
        array : numpy.ndarray|list|tuple

        Returns
        -------
        AntPolRefType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError(
                    'Expected array to be of length 3,\n\t'
                    'and received `{}`'.format(array))
            return cls(AmpX=array[0], AmpY=array[1], PhaseY=array[2])
        raise ValueError(
            'Expected array to be numpy.ndarray, list, or tuple,\n\t'
            'got `{}`'.format(type(array)))


class EBType(Serializable):
    """
    Electrical boresight (EB) steering directions for an electronically steered array.
    """

    _fields = ('DCXPoly', 'DCYPoly', 'UseEBPVP')
    _required = ('DCXPoly', 'DCYPoly')
    # descriptors
    DCXPoly = SerializableDescriptor(
        'DCXPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight steering *X-axis direction cosine (DCX)* as a function of '
                  'slow time ``(variable 1)``.')  # type: Poly1DType
    DCYPoly = SerializableDescriptor(
        'DCYPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight steering *Y-axis direction cosine (DCY)* as a function of '
                  'slow time ``(variable 1)``.')  # type: Poly1DType
    UseEBPVP = BooleanDescriptor(
        'UseEBPVP', _required, strict=DEFAULT_STRICT,
        docstring="")  # type: Optional[bool]

    def __init__(self, DCXPoly=None, DCYPoly=None, UseEBPVP=None, **kwargs):
        """
        Parameters
        ----------
        DCXPoly : Poly1DType|numpy.ndarray|list|tuple
        DCYPoly : Poly1DType|numpy.ndarray|list|tuple
        UseEBPVP : None|bool
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DCXPoly = DCXPoly
        self.DCYPoly = DCYPoly
        self.UseEBPVP = UseEBPVP
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

    def version_required(self) -> Tuple[int, int, int]:
        required = (1, 0, 1)
        if self.UseEBPVP is not None:
            required = max(required, (1, 1, 0))
        return required


class GainPhasePolyType(Serializable):
    """A container for the Gain and Phase Polygon definitions."""

    _fields = ('GainPoly', 'PhasePoly', 'AntGPid')
    _required = ('GainPoly', 'PhasePoly')
    # descriptors
    GainPoly = SerializableDescriptor(
        'GainPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='One-way signal gain (in dB) as a function of X-axis direction cosine (DCX) (variable 1) '
                  'and Y-axis direction cosine (DCY) (variable 2). Gain relative to gain at DCX = 0 '
                  'and DCY = 0, so constant coefficient is always 0.0.')  # type: Poly2DType
    PhasePoly = SerializableDescriptor(
        'PhasePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='One-way signal phase (in cycles) as a function of DCX (variable 1) and '
                  'DCY (variable 2). Phase relative to phase at DCX = 0 and DCY = 0, '
                  'so constant coefficient is always 0.0.')  # type: Poly2DType
    AntGPid = StringDescriptor(
        'AntGPid', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[str]

    def __init__(self, GainPoly=None, PhasePoly=None, AntGPid=None, **kwargs):
        """
        Parameters
        ----------
        GainPoly : Poly2DType|numpy.ndarray|list|tuple
        PhasePoly : Poly2DType|numpy.ndarray|list|tuple
        AntGPid : None|str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.GainPoly = GainPoly
        self.PhasePoly = PhasePoly
        self.AntGPid = AntGPid
        super(GainPhasePolyType, self).__init__(**kwargs)

    def __call__(self, x, y):
        """
        Evaluate a polynomial at points [`x`, `y`]. This passes `x`,`y` straight
        through to the call method for each component.

        Parameters
        ----------
        x : float|int|numpy.ndarray
            The first dependent variable of point(s) at which to evaluate.
        y : float|int|numpy.ndarray
            The second dependent variable of point(s) at which to evaluate.

        Returns
        -------
        numpy.ndarray
        """

        if self.GainPoly is None or self.PhasePoly is None:
            return None
        return numpy.array([self.GainPoly(x, y), self.PhasePoly(x, y)], dtype=numpy.float64)

    def minimize_order(self):
        """
        Trim the trailing zeros for each component coefficient array. This
        modifies the object in place.

        Returns
        -------
        None
        """

        self.GainPoly.minimize_order()
        self.PhasePoly.minimize_order()

    def version_required(self) -> Tuple[int, int, int]:
        required = (1, 0, 1)
        if self.AntGPid is not None:
            required = max(required, (1, 1, 0))
        return required


class AntPatternType(Serializable):
    """
    Parameter set that defines each Antenna Pattern as function time.
    """

    _fields = (
        'Identifier', 'FreqZero', 'GainZero', 'EBFreqShift', 'EBFreqShift',
        'MLFreqDilation', 'MLFreqDilationSF', 'GainBSPoly', 'AntPolRef',
        'EB', 'Array', 'Element', 'GainPhaseArray')
    _required = (
        'Identifier', 'FreqZero', 'EB', 'Array', 'Element')
    _collections_tags = {
        'GainPhaseArray': {'array': False, 'child_tag': 'GainPhaseArray'}}
    _numeric_format = {'FreqZero': FLOAT_FORMAT, 'GainZero': FLOAT_FORMAT}
    # descriptors
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='String that uniquely identifies this ACF.')  # type: str
    FreqZero = FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring='The reference frequency value for which the Electrical Boresight '
                  'and array pattern polynomials are computed.')  # type: float
    GainZero = FloatDescriptor(
        'GainZero', _required, strict=DEFAULT_STRICT,
        docstring='The reference antenna gain at zero steering angle at the '
                  'reference frequency, measured in dB.')  # type: float
    EBFreqShift = BooleanDescriptor(
        'EBFreqShift', _required, strict=DEFAULT_STRICT,
        docstring="Parameter indicating whether the electronic boresite shifts with "
                  "frequency.")  # type: bool
    EBFreqShiftSF = SerializableDescriptor(
        'EBFreqShiftSF', FreqSFType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[FreqSFType]
    MLFreqDilation = BooleanDescriptor(
        'MLFreqDilation', _required, strict=DEFAULT_STRICT,
        docstring="Parameter indicating the mainlobe (ML) width changes with "
                  "frequency.")  # type: bool
    MLFreqDilationSF = SerializableDescriptor(
        'MLFreqDilationSF', FreqSFType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Optional[FreqSFType]
    GainBSPoly = SerializableDescriptor(
        'GainBSPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Gain polynomial *(in dB)* as a function of frequency for boresight *(BS)* '
                  'at :math:`DCX=0, DCY=0`. '
                  'Frequency ratio :math:`(f-f0)/f0` is the input variable, and the constant '
                  'coefficient is always `0.0`.')  # type: Poly1DType
    AntPolRef = SerializableDescriptor(
        'AntPolRef', AntPolRefType, _required, strict=DEFAULT_STRICT,
        docstring='Polarization parameters for the EB steered to mechanical '
                  'boresight (EB_DCX = 0 and EB_DCY = 0).')  # type: AntPolRefType
    EB = SerializableDescriptor(
        'EB', EBType, _required, strict=DEFAULT_STRICT,
        docstring='Electrical boresight *(EB)* steering directions for an electronically '
                  'steered array.')  # type: EBType
    Array = SerializableDescriptor(
        'Array', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='Array pattern polynomials that define the shape of the '
                  'main-lobe.')  # type: GainPhasePolyType
    Element = SerializableDescriptor(
        'Element', GainPhasePolyType, _required, strict=DEFAULT_STRICT,
        docstring='Element array pattern polynomials for electronically steered '
                  'arrays.')  # type: GainPhasePolyType
    GainPhaseArray = SerializableListDescriptor(
        'GainPhaseArray', GainPhaseArrayType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Array of parameters that identify 2-D sampled Gain and Phase patterns at '
                  'single frequency value.')  # type: Union[None, List[GainPhaseArrayType]]

    def __init__(
            self,
            Identifier=None,
            FreqZero=None,
            GainZero=None,
            EBFreqShift=None,
            EBFreqShiftSF: Optional[FreqSFType] = None,
            MLFreqDilation=None,
            MLFreqDilationSF: Optional[FreqSFType] = None,
            GainBSPoly=None,
            AntPolRef: Union[None, AntPolRefType, numpy.ndarray, list, tuple] = None,
            EB=None,
            Array=None,
            Element=None,
            GainPhaseArray=None,
            **kwargs):
        """

        Parameters
        ----------
        Identifier : str
        FreqZero : float
        GainZero : float
        EBFreqShift : bool
        EBFreqShiftSF : None|FreqSFType|numpy.ndarray|list|tuple
        MLFreqDilation : bool
        MLFreqDilationSF : None|FreqSFType|numpy.ndarray|list|tuple
        GainBSPoly : None|Poly1DType|numpy.ndarray|list|tuple
        AntPolRef : None|AntPolRefType|numpy.ndarray|list|tuple
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
        self.EBFreqShiftSF = EBFreqShiftSF
        self.MLFreqDilation = MLFreqDilation
        self.MLFreqDilationSF = MLFreqDilationSF
        self.GainBSPoly = GainBSPoly
        self.AntPolRef = AntPolRef
        self.EB = EB
        self.Array = Array
        self.Element = Element
        self.GainPhaseArray = GainPhaseArray
        super(AntPatternType, self).__init__(**kwargs)

    def version_required(self) -> Tuple[int, int, int]:
        required = (1, 0, 1)
        for fld in ['EB', 'Array', 'Element']:
            val = getattr(self, fld)
            if val is not None:
                required = max(required, val.version_required())
        if self.EBFreqShiftSF is not None or \
                self.MLFreqDilationSF is not None or \
                self.AntPolRef is not None:
            required = (required, (1, 1, 0))
        return required


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
    AntCoordFrame = SerializableListDescriptor(
        'AntCoordFrame', AntCoordFrameType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Unit vectors that describe the orientation of an Antenna Coordinate Frame (ACF) '
                  'as function of time. Parameter set repeated for '
                  'each ACF.')  # type: List[AntCoordFrameType]
    AntPhaseCenter = SerializableListDescriptor(
        'AntPhaseCenter', AntPhaseCenterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe each Antenna Phase Center (APC). Parameter '
                  'set repeated for each APC.')  # type: List[AntPhaseCenterType]
    AntPattern = SerializableListDescriptor(
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

    def version_required(self) -> Tuple[int, int, int]:
        required = (1, 0, 1)
        if self.AntCoordFrame is not None:
            for entry in self.AntCoordFrame:
                required = max(required, entry.version_required())
        return required
