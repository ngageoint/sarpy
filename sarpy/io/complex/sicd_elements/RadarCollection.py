"""
The RadarCollectionType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import List, Union
import logging

import numpy

from sarpy.io.xml.base import Serializable, Arrayable, SerializableArray, \
    ParametersCollection, parse_float
from sarpy.io.xml.descriptors import StringDescriptor, StringEnumDescriptor, \
    FloatDescriptor, IntegerDescriptor, SerializableDescriptor, \
    SerializableArrayDescriptor, UnitVectorDescriptor, ParametersDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT, \
    SerializableCPArrayDescriptor, SerializableCPArray
from .blocks import XYZType, LatLonHAECornerRestrictionType, \
    POLARIZATION1_VALUES, POLARIZATION2_VALUES, DUAL_POLARIZATION_VALUES
from .utils import is_polstring_version1

import sarpy.geometry.geocoords as geocoords


logger = logging.getLogger(__name__)


def get_band_name(freq):
    """
    Gets the band names associated with the given frequency (in Hz).

    Parameters
    ----------
    freq : float

    Returns
    -------
    str
    """

    if freq is None:
        return 'UN'
    elif 3e6 <= freq < 3e7:
        return 'HF'
    elif 3e7 <= freq < 3e8:
        return 'VHF'
    elif 3e8 <= freq < 1e9:
        return 'UHF'
    elif 1e9 <= freq < 2e9:
        return 'L'
    elif 2e9 <= freq < 4e9:
        return 'S'
    elif 4e9 <= freq < 8e9:
        return 'C'
    elif 8e9 <= freq < 1.2e10:
        return 'X'
    elif 1.2e10 <= freq < 1.8e10:
        return 'KU'
    elif 1.8e10 <= freq < 2.7e10:
        return 'K'
    elif 2.7e10 <= freq < 4e10:
        return 'KA'
    elif 4e10 <= freq < 3e20:
        return 'MM'
    else:
        return 'UN'


class TxFrequencyType(Serializable, Arrayable):
    """
    The transmit frequency range.
    """

    _fields = ('Min', 'Max')
    _required = _fields
    _numeric_format = {'Min': FLOAT_FORMAT, 'Max': FLOAT_FORMAT}
    # descriptors
    Min = FloatDescriptor(
        'Min', _required, strict=DEFAULT_STRICT,
        docstring='The transmit minimum frequency in Hz.')  # type: float
    Max = FloatDescriptor(
        'Max', _required, strict=DEFAULT_STRICT,
        docstring='The transmit maximum frequency in Hz.')  # type: float

    def __init__(self, Min=None, Max=None, **kwargs):
        """

        Parameters
        ----------
        Min : float
        Max : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Min, self.Max = Min, Max
        super(TxFrequencyType, self).__init__(**kwargs)

    @property
    def center_frequency(self):
        """
        None|float: The center frequency
        """

        if self.Min is None or self.Max is None:
            return None
        return 0.5*(self.Min + self.Max)

    def _apply_reference_frequency(self, reference_frequency):
        if self.Min is not None:
            self.Min += reference_frequency
        if self.Max is not None:
            self.Max += reference_frequency

    def _basic_validity_check(self):
        condition = super(TxFrequencyType, self)._basic_validity_check()
        if self.Min is not None and self.Max is not None and self.Max < self.Min:
            self.log_validity_error(
                'Invalid frequency bounds Min ({}) > Max ({})'.format(self.Min, self.Max))
            condition = False
        return condition

    def get_band_abbreviation(self):
        """
        Gets the band abbreviation for the suggested name.

        Returns
        -------
        str
        """

        min_band = get_band_name(self.Min)
        max_band = get_band_name(self.Max)
        if min_band == max_band:
            return min_band + '_'*(3-len(min_band))
        elif min_band == 'UN' or max_band == 'UN':
            return 'UN_'
        else:
            return 'MB_'

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the data.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array with appropriate entry order
        """

        return numpy.array([self.Min, self.Max], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Min, Max]

        Returns
        -------
        LatLonType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Min=array[0], Max=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class WaveformParametersType(Serializable):
    """
    Transmit and receive demodulation waveform parameters.
    """

    _fields = (
        'TxPulseLength', 'TxRFBandwidth', 'TxFreqStart', 'TxFMRate', 'RcvDemodType', 'RcvWindowLength',
        'ADCSampleRate', 'RcvIFBandwidth', 'RcvFreqStart', 'RcvFMRate', 'index')
    _required = ()
    _set_as_attribute = ('index', )
    _numeric_format = {
        'TxPulseLength': FLOAT_FORMAT, 'TxRFBandwidth': '0.17E', 'TxFreqStart': '0.17E',
        'TxFMRate': '0.17E', 'RcvWindowLength': FLOAT_FORMAT, 'ADCSampleRate': '0.17E',
        'RcvIFBandwidth': '0.17E', 'RcvFreqStart': '0.17E', 'RcvFMRate': '0.17E'}

    # descriptors
    TxPulseLength = FloatDescriptor(
        'TxPulseLength', _required, strict=DEFAULT_STRICT,
        docstring='Transmit pulse length in seconds.')  # type: float
    TxRFBandwidth = FloatDescriptor(
        'TxRFBandwidth', _required, strict=DEFAULT_STRICT,
        docstring='Transmit RF bandwidth of the transmit pulse in Hz.')  # type: float
    TxFreqStart = FloatDescriptor(
        'TxFreqStart', _required, strict=DEFAULT_STRICT,
        docstring='Transmit Start frequency for Linear FM waveform in Hz, may be relative '
                  'to reference frequency.')  # type: float
    TxFMRate = FloatDescriptor(
        'TxFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Transmit FM rate for Linear FM waveform in Hz/second.')  # type: float
    RcvWindowLength = FloatDescriptor(
        'RcvWindowLength', _required, strict=DEFAULT_STRICT,
        docstring='Receive window duration in seconds.')  # type: float
    ADCSampleRate = FloatDescriptor(
        'ADCSampleRate', _required, strict=DEFAULT_STRICT,
        docstring='Analog-to-Digital Converter sampling rate in samples/second.')  # type: float
    RcvIFBandwidth = FloatDescriptor(
        'RcvIFBandwidth', _required, strict=DEFAULT_STRICT,
        docstring='Receive IF bandwidth in Hz.')  # type: float
    RcvFreqStart = FloatDescriptor(
        'RcvFreqStart', _required, strict=DEFAULT_STRICT,
        docstring='Receive demodulation start frequency in Hz, may be relative to reference frequency.')  # type: float
    index = IntegerDescriptor(
        'index', _required, strict=False, docstring="The array index.")  # type: int

    def __init__(self, TxPulseLength=None, TxRFBandwidth=None, TxFreqStart=None, TxFMRate=None,
                 RcvDemodType=None, RcvWindowLength=None, ADCSampleRate=None, RcvIFBandwidth=None,
                 RcvFreqStart=None, RcvFMRate=None, index=None, **kwargs):
        """

        Parameters
        ----------
        TxPulseLength : float
        TxRFBandwidth : float
        TxFreqStart : float
        TxFMRate : float
        RcvDemodType : str
        RcvWindowLength : float
        ADCSampleRate : float
        RcvIFBandwidth : float
        RcvFreqStart : float
        RcvFMRate : float
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self._RcvFMRate = None
        self.TxPulseLength, self.TxRFBandwidth = TxPulseLength, TxRFBandwidth
        self.TxFreqStart, self.TxFMRate = TxFreqStart, TxFMRate
        self.RcvWindowLength = RcvWindowLength
        self.ADCSampleRate = ADCSampleRate
        self.RcvIFBandwidth = RcvIFBandwidth
        self.RcvFreqStart = RcvFreqStart
        # NB: self.RcvDemodType is read only.
        if RcvDemodType == 'CHIRP' and RcvFMRate is None:
            self.RcvFMRate = 0.0
        else:
            self.RcvFMRate = RcvFMRate
        self.index = index
        super(WaveformParametersType, self).__init__(**kwargs)

    @property
    def RcvDemodType(self):  # type: () -> Union[None, str]
        """
        str: READ ONLY. Receive demodulation used when Linear FM waveform is
        used on transmit. This value is derived form the value of `RcvFMRate`.

        * `None` - `RcvFMRate` is `None`.

        * `'CHIRP'` - `RcvFMRate=0`.

        * `'STRETCH'` - `RcvFMRate` is non-zero.
        """

        if self._RcvFMRate is None:
            return None
        elif self._RcvFMRate == 0:
            return 'CHIRP'
        else:
            return 'STRETCH'

    @property
    def RcvFMRate(self):  # type: () -> Union[None, float]
        """
        float: Receive FM rate in Hz/sec. Also, determines the value of `RcvDemodType`. **Optional.**
        """
        return self._RcvFMRate

    @RcvFMRate.setter
    def RcvFMRate(self, value):
        if value is None:
            self._RcvFMRate = None
        else:
            try:
                self._RcvFMRate = parse_float(value, 'RcvFMRate', self)
            except Exception as e:
                logger.error(
                    'Failed parsing value {} for field RCVFMRate of type "float",\n\t'
                    'with error {} - {}.\n\t'
                    'The value has been set to None.'.format(value, type(e), e))
                self._RcvFMRate = None

    def _basic_validity_check(self):
        valid = super(WaveformParametersType, self)._basic_validity_check()
        return valid

    def derive(self):
        """
        Populate derived data in `WaveformParametersType`.

        Returns
        -------
        None
        """

        if self.TxPulseLength is not None and self.TxFMRate is not None and self.TxRFBandwidth is None:
            self.TxRFBandwidth = self.TxPulseLength*self.TxFMRate
        if self.TxPulseLength is not None and self.TxRFBandwidth is not None and self.TxFMRate is None:
            self.TxFMRate = self.TxRFBandwidth/self.TxPulseLength
        if self.TxFMRate is not None and self.TxRFBandwidth is not None and self.TxPulseLength is None:
            self.TxPulseLength = self.TxRFBandwidth/self.TxFMRate

    def _apply_reference_frequency(self, reference_frequency):
        if self.TxFreqStart is not None:
            self.TxFreqStart += reference_frequency
        if self.RcvFreqStart is not None:
            self.RcvFreqStart += reference_frequency


class TxStepType(Serializable):
    """
    Transmit sequence step details.
    """

    _fields = ('WFIndex', 'TxPolarization', 'index')
    _required = ('index', )
    _set_as_attribute = ('index', )
    # descriptors
    WFIndex = IntegerDescriptor(
        'WFIndex', _required, strict=DEFAULT_STRICT,
        docstring='The waveform number for this step.')  # type: int
    TxPolarization = StringEnumDescriptor(
        'TxPolarization', POLARIZATION2_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Transmit signal polarization for this step.')  # type: str
    index = IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT,
        docstring='The step index')  # type: int

    def __init__(self, WFIndex=None, TxPolarization=None, index=None, **kwargs):
        """

        Parameters
        ----------
        WFIndex : int
        TxPolarization : str
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.WFIndex = WFIndex
        self.TxPolarization = TxPolarization
        self.index = index
        super(TxStepType, self).__init__(**kwargs)


class ChanParametersType(Serializable):
    """
    Transmit receive sequence step details.
    """

    _fields = ('TxRcvPolarization', 'RcvAPCIndex', 'index')
    _required = ('TxRcvPolarization', 'index', )
    _set_as_attribute = ('index', )
    # descriptors
    TxRcvPolarization = StringEnumDescriptor(
        'TxRcvPolarization', DUAL_POLARIZATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Combined Transmit and Receive signal polarization for the channel.')  # type: str
    RcvAPCIndex = IntegerDescriptor(
        'RcvAPCIndex', _required, strict=DEFAULT_STRICT,
        docstring='Index of the Receive Aperture Phase Center (Rcv APC). Only include if Receive APC position '
                  'polynomial(s) are included.')  # type: int
    index = IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The parameter index')  # type: int

    def __init__(self, TxRcvPolarization=None, RcvAPCIndex=None, index=None, **kwargs):
        """

        Parameters
        ----------
        TxRcvPolarization : str
        RcvAPCIndex : int
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxRcvPolarization = TxRcvPolarization
        self.RcvAPCIndex = RcvAPCIndex
        self.index = index
        super(ChanParametersType, self).__init__(**kwargs)

    def get_transmit_polarization(self):
        if self.TxRcvPolarization is None:
            return None
        elif self.TxRcvPolarization in ['OTHER', 'UNKNOWN']:
            return 'OTHER'
        else:
            return self.TxRcvPolarization.split(':')[0]

    def permits_version_1_1(self):
        """
        Does this value permit storage in SICD version 1.1?

        Returns
        -------
        bool
        """

        return is_polstring_version1(self.TxRcvPolarization)


class ReferencePointType(Serializable):
    """The reference point definition"""
    _fields = ('ECF', 'Line', 'Sample', 'name')
    _required = ('ECF', 'Line', 'Sample')
    _set_as_attribute = ('name', )
    _numeric_format = {'Line': FLOAT_FORMAT, 'Sample': FLOAT_FORMAT}
    # descriptors
    ECF = SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The geographical coordinates for the reference point.')  # type: XYZType
    Line = FloatDescriptor(
        'Line', _required, strict=DEFAULT_STRICT,
        docstring='The reference point line index.')  # type: float
    Sample = FloatDescriptor(
        'Sample', _required, strict=DEFAULT_STRICT,
        docstring='The reference point sample index.')  # type: float
    name = StringDescriptor(
        'name', _required, strict=DEFAULT_STRICT,
        docstring='The reference point name.')  # type: str

    def __init__(self, ECF=None, Line=None, Sample=None, name=None, **kwargs):
        """

        Parameters
        ----------
        ECF : XYZType|numpy.ndarray|list|tuple
        Line : float
        Sample : float
        name : str
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ECF = ECF
        self.Line = Line
        self.Sample = Sample
        self.name = name
        super(ReferencePointType, self).__init__(**kwargs)


class XDirectionType(Serializable):
    """The X direction of the collect"""
    _fields = ('UVectECF', 'LineSpacing', 'NumLines', 'FirstLine')
    _required = _fields
    _numeric_format = {'LineSpacing': FLOAT_FORMAT, }
    # descriptors
    UVectECF = UnitVectorDescriptor(
        'UVectECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The unit vector in the X direction.')  # type: XYZType
    LineSpacing = FloatDescriptor(
        'LineSpacing', _required, strict=DEFAULT_STRICT,
        docstring='The collection line spacing in the X direction in meters.')  # type: float
    NumLines = IntegerDescriptor(
        'NumLines', _required, strict=DEFAULT_STRICT,
        docstring='The number of lines in the X direction.')  # type: int
    FirstLine = IntegerDescriptor(
        'FirstLine', _required, strict=DEFAULT_STRICT,
        docstring='The first line index.')  # type: int

    def __init__(self, UVectECF=None, LineSpacing=None, NumLines=None, FirstLine=None, **kwargs):
        """

        Parameters
        ----------
        UVectECF : XYZType|numpy.ndarray|list|tuple
        LineSpacing : float
        NumLines : int
        FirstLine : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.UVectECF = UVectECF
        self.LineSpacing = LineSpacing
        self.NumLines = NumLines
        self.FirstLine = FirstLine
        super(XDirectionType, self).__init__(**kwargs)


class YDirectionType(Serializable):
    """The Y direction of the collect"""
    _fields = ('UVectECF', 'SampleSpacing', 'NumSamples', 'FirstSample')
    _required = _fields
    _numeric_format = {'SampleSpacing': FLOAT_FORMAT, }
    # descriptors
    UVectECF = UnitVectorDescriptor(
        'UVectECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The unit vector in the Y direction.')  # type: XYZType
    SampleSpacing = FloatDescriptor(
        'SampleSpacing', _required, strict=DEFAULT_STRICT,
        docstring='The collection sample spacing in the Y direction in meters.')  # type: float
    NumSamples = IntegerDescriptor(
        'NumSamples', _required, strict=DEFAULT_STRICT,
        docstring='The number of samples in the Y direction.')  # type: int
    FirstSample = IntegerDescriptor(
        'FirstSample', _required, strict=DEFAULT_STRICT,
        docstring='The first sample index.')  # type: int

    def __init__(self, UVectECF=None, SampleSpacing=None, NumSamples=None, FirstSample=None, **kwargs):
        """

        Parameters
        ----------
        UVectECF : XYZType|numpy.ndarray|list|tuple
        SampleSpacing : float
        NumSamples : int
        FirstSample : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.UVectECF = UVectECF
        self.SampleSpacing = SampleSpacing
        self.NumSamples = NumSamples
        self.FirstSample = FirstSample
        super(YDirectionType, self).__init__(**kwargs)


class SegmentArrayElement(Serializable):
    """The reference point definition"""
    _fields = ('StartLine', 'StartSample', 'EndLine', 'EndSample', 'Identifier', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    StartLine = IntegerDescriptor(
        'StartLine', _required, strict=DEFAULT_STRICT,
        docstring='The starting line number of the segment.')  # type: int
    StartSample = IntegerDescriptor(
        'StartSample', _required, strict=DEFAULT_STRICT,
        docstring='The starting sample number of the segment.')  # type: int
    EndLine = IntegerDescriptor(
        'EndLine', _required, strict=DEFAULT_STRICT,
        docstring='The ending line number of the segment.')  # type: int
    EndSample = IntegerDescriptor(
        'EndSample', _required, strict=DEFAULT_STRICT,
        docstring='The ending sample number of the segment.')  # type: int
    Identifier = StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='Identifier for the segment data boundary.')
    index = IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT,
        docstring='The array index.')  # type: int

    def __init__(self, StartLine=None, StartSample=None, EndLine=None, EndSample=None,
                 Identifier=None, index=None, **kwargs):
        """

        Parameters
        ----------
        StartLine : int
        StartSample : int
        EndLine : int
        EndSample : int
        Identifier : str
        index : int
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.StartLine, self.EndLine = StartLine, EndLine
        self.StartSample, self.EndSample = StartSample, EndSample
        self.Identifier = Identifier
        self.index = index
        super(SegmentArrayElement, self).__init__(**kwargs)


class ReferencePlaneType(Serializable):
    """
    The reference plane.
    """

    _fields = ('RefPt', 'XDir', 'YDir', 'SegmentList', 'Orientation')
    _required = ('RefPt', 'XDir', 'YDir')
    _collections_tags = {'SegmentList': {'array': True, 'child_tag': 'Segment'}}
    # other class variable
    _ORIENTATION_VALUES = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'ARBITRARY')
    # descriptors
    RefPt = SerializableDescriptor(
        'RefPt', ReferencePointType, _required, strict=DEFAULT_STRICT,
        docstring='The reference point.')  # type: ReferencePointType
    XDir = SerializableDescriptor(
        'XDir', XDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The X direction collection plane parameters.')  # type: XDirectionType
    YDir = SerializableDescriptor(
        'YDir', YDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The Y direction collection plane parameters.')  # type: YDirectionType
    SegmentList = SerializableArrayDescriptor(
        'SegmentList', SegmentArrayElement, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The segment array.')  # type: Union[SerializableArray, List[SegmentArrayElement]]
    Orientation = StringEnumDescriptor(
        'Orientation', _ORIENTATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Describes the shadow intent of the display plane.')  # type: str

    def __init__(self, RefPt=None, XDir=None, YDir=None, SegmentList=None, Orientation=None, **kwargs):
        """

        Parameters
        ----------
        RefPt : ReferencePointType
        XDir : XDirectionType
        YDir : YDirectionType
        SegmentList : SerializableArray|List[SegmentArrayElement]
        Orientation : str
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RefPt = RefPt
        self.XDir, self.YDir = XDir, YDir
        self.SegmentList = SegmentList
        self.Orientation = Orientation
        super(ReferencePlaneType, self).__init__(**kwargs)

    def get_ecf_corner_array(self):
        """
        Use the XDir and YDir definitions to return the corner points in ECF coordinates as a `4x3` array.

        Returns
        -------
        numpy.ndarray
            The corner points of the collection area, with order following the AreaType order convention.
        """

        ecf_ref = self.RefPt.ECF.get_array()
        x_shift = self.XDir.UVectECF.get_array() * self.XDir.LineSpacing
        y_shift = self.YDir.UVectECF.get_array() * self.YDir.SampleSpacing
        # order convention
        x_offset = numpy.array(
            [self.XDir.FirstLine, self.XDir.FirstLine, self.XDir.NumLines, self.XDir.NumLines])
        y_offset = numpy.array(
            [self.YDir.FirstSample, self.YDir.NumSamples, self.YDir.NumSamples, self.YDir.FirstSample])
        corners = numpy.zeros((4, 3), dtype=numpy.float64)
        for i in range(4):
            corners[i, :] = \
                ecf_ref + x_shift*(x_offset[i] - self.RefPt.Line) + y_shift*(y_offset[i] - self.RefPt.Sample)
        return corners


class AreaType(Serializable):
    """
    The collection area.
    """

    _fields = ('Corner', 'Plane')
    _required = ('Corner', )
    _collections_tags = {
        'Corner': {'array': True, 'child_tag': 'ACP'}, }
    # descriptors
    Corner = SerializableCPArrayDescriptor(
        'Corner', LatLonHAECornerRestrictionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The collection area corner point definition array.'
    )  # type: Union[SerializableCPArray, List[LatLonHAECornerRestrictionType]]
    Plane = SerializableDescriptor(
        'Plane', ReferencePlaneType, _required, strict=DEFAULT_STRICT,
        docstring='A rectangular area in a geo-located display plane.')  # type: ReferencePlaneType

    def __init__(self, Corner=None, Plane=None, **kwargs):
        """

        Parameters
        ----------
        Corner : SerializableCPArray|List[LatLonHAECornerRestrictionType]|numpy.ndarray|list|tuple
        Plane : ReferencePlaneType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Corner = Corner
        self.Plane = Plane
        super(AreaType, self).__init__(**kwargs)
        self.derive()

    def _derive_corner_from_plane(self):
        # try to define the corner points - for SICD 0.5.
        if self.Corner is not None:
            return  # nothing to be done
        if self.Plane is None:
            return  # nothing to derive from
        corners = self.Plane.get_ecf_corner_array()
        self.Corner = [
            LatLonHAECornerRestrictionType(**{'Lat': entry[0], 'Lon': entry[1], 'HAE': entry[2], 'index': i+1})
            for i, entry in enumerate(geocoords.ecf_to_geodetic(corners))]

    def derive(self):
        """
        Derive the corner points from the plane, if necessary.

        Returns
        -------
        None
        """

        self._derive_corner_from_plane()


class RadarCollectionType(Serializable):
    """The Radar Collection Type"""
    _fields = (
        'TxFrequency', 'RefFreqIndex', 'Waveform', 'TxPolarization', 'TxSequence', 'RcvChannels', 'Area', 'Parameters')
    _required = ('TxFrequency', 'TxPolarization', 'RcvChannels')
    _collections_tags = {
        'Waveform': {'array': True, 'child_tag': 'WFParameters'},
        'TxSequence': {'array': True, 'child_tag': 'TxStep'},
        'RcvChannels': {'array': True, 'child_tag': 'ChanParameters'},
        'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    TxFrequency = SerializableDescriptor(
        'TxFrequency', TxFrequencyType, _required, strict=DEFAULT_STRICT,
        docstring='The transmit frequency range.')  # type: TxFrequencyType
    RefFreqIndex = IntegerDescriptor(
        'RefFreqIndex', _required, strict=DEFAULT_STRICT,
        docstring='The reference frequency index, if applicable. If present and non-zero, '
                  'all (most) RF frequency values are expressed as offsets from a reference '
                  'frequency.')  # type: int
    Waveform = SerializableArrayDescriptor(
        'Waveform', WaveformParametersType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Transmit and receive demodulation waveform parameters.'
    )  # type: Union[SerializableArray, List[WaveformParametersType]]
    TxPolarization = StringEnumDescriptor(
        'TxPolarization', POLARIZATION1_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The transmit polarization.')  # type: str
    TxSequence = SerializableArrayDescriptor(
        'TxSequence', TxStepType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The transmit sequence parameters array. If present, indicates the transmit signal steps through '
                  'a repeating sequence of waveforms and/or polarizations. '
                  'One step per Inter-Pulse Period.')  # type: Union[SerializableArray, List[TxStepType]]
    RcvChannels = SerializableArrayDescriptor(
        'RcvChannels', ChanParametersType, _collections_tags,
        _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Receive data channel parameters.')  # type: Union[SerializableArray, List[ChanParametersType]]
    Area = SerializableDescriptor(
        'Area', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='The imaged area covered by the collection.')  # type: AreaType
    Parameters = ParametersDescriptor(
        'Parameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='A parameters collections.')  # type: ParametersCollection

    def __init__(self, TxFrequency=None, RefFreqIndex=None, Waveform=None,
                 TxPolarization=None, TxSequence=None, RcvChannels=None,
                 Area=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        TxFrequency : TxFrequencyType|numpy.ndarray|list|tuple
        RefFreqIndex : int
        Waveform : SerializableArray|List[WaveformParametersType]
        TxPolarization : str
        TxSequence : SerializableArray|List[TxStepType]
        RcvChannels : SerializableArray|List[ChanParametersType]
        Area : AreaType
        Parameters : ParametersCollection|dict
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TxFrequency = TxFrequency
        self.RefFreqIndex = RefFreqIndex
        self.Waveform = Waveform
        self.TxPolarization = TxPolarization
        self.TxSequence = TxSequence
        self.RcvChannels = RcvChannels
        self.Area = Area
        self.Parameters = Parameters
        super(RadarCollectionType, self).__init__(**kwargs)

    def derive(self):
        """
        Populates derived data in RadarCollection. Expected to be called by `SICD` parent.

        Returns
        -------
        None
        """

        self._derive_tx_polarization()
        if self.Area is not None:
            self.Area.derive()
        if self.Waveform is not None:
            for entry in self.Waveform:
                entry.derive()
        self._derive_tx_frequency()  # call after waveform entry derive call
        self._derive_wf_params()

    def _derive_tx_polarization(self):
        def check_sequence():
            unique_entries = set(entry.TxPolarization for entry in self.TxSequence)
            if len(unique_entries) == 1:
                self.TxPolarization = self.TxSequence[0].TxPolarization
            else:
                self.TxPolarization = 'SEQUENCE'

        # TxPolarization was optional prior to SICD 1.0. It may need to be derived.
        if self.TxSequence is not None:
            check_sequence()
            return
        if self.TxPolarization is not None:
            return  # nothing to be done

        if self.RcvChannels is None:
            return  # nothing to derive from

        if len(self.RcvChannels) > 1:
            # TxSequence may need to be derived from RCvChannels, for SICD before 1.0 or poorly formed
            if self.TxSequence is not None:
                return
            elif self.RcvChannels is None:
                return  # nothing to derive from
            elif len(self.RcvChannels) < 2:
                return  # no need for step definition

            tx_pols = list(chan_param.get_transmit_polarization() for chan_param in self.RcvChannels)

            if len(tx_pols) > 1:
                self.TxSequence = [TxStepType(index=i+1, TxPolarization=tx_pol) for i, tx_pol in enumerate(tx_pols)]
                check_sequence()
            else:
                self.TxPolarization = tx_pols[0]
        else:
            self.TxPolarization = self.RcvChannels[0].get_transmit_polarization()

    def _derive_tx_frequency(self):
        if self.Waveform is None or self.Waveform.size == 0:
            return  # nothing to be done
        if not(self.TxFrequency is None or self.TxFrequency.Min is None or self.TxFrequency.Max is None):
            return  # no need to do anything

        if self.TxFrequency is None:
            self.TxFrequency = TxFrequencyType()
        if self.TxFrequency.Min is None:
            self.TxFrequency.Min = min(
                entry.TxFreqStart for entry in self.Waveform if entry.TxFreqStart is not None)
        if self.TxFrequency.Max is None:
            self.TxFrequency.Max = max(
                (entry.TxFreqStart + entry.TxRFBandwidth) for entry in self.Waveform if
                entry.TxFreqStart is not None and entry.TxRFBandwidth is not None)

    def _derive_wf_params(self):
        if self.TxFrequency is None or self.TxFrequency.Min is None or self.TxFrequency.Max is None:
            return  # nothing that we can do
        if self.Waveform is None or self.Waveform.size != 1:
            return  # nothing to be done

        entry = self.Waveform[0]  # only true for single waveform definition
        if entry.TxFreqStart is None:
            entry.TxFreqStart = self.TxFrequency.Min
        if entry.TxRFBandwidth is None:
            entry.TxRFBandwidth = self.TxFrequency.Max - self.TxFrequency.Min

    def _apply_reference_frequency(self, reference_frequency):
        """
        If the reference frequency is used, adjust the necessary fields accordingly.
        Expected to be called by `SICD` parent.

        Parameters
        ----------
        reference_frequency : float
            The reference frequency.

        Returns
        -------
        None
        """

        if self.TxFrequency is not None:
            # noinspection PyProtectedMember
            self.TxFrequency._apply_reference_frequency(reference_frequency)
        if self.Waveform is not None:
            for entry in self.Waveform:
                # noinspection PyProtectedMember
                entry._apply_reference_frequency(reference_frequency)
        self.RefFreqIndex = 0

    def get_polarization_abbreviation(self):
        """
        Gets the polarization collection abbreviation for the suggested name.

        Returns
        -------
        str
        """

        if self.RcvChannels is None:
            pol_count = 0
        else:
            pol_count = len(self.RcvChannels)
        if pol_count == 1:
            return 'S'
        elif pol_count == 2:
            return 'D'
        elif pol_count == 3:
            return 'T'
        elif pol_count > 3:
            return 'Q'
        else:
            return 'U'

    def _check_frequency(self):
        # type: () -> bool

        if self.RefFreqIndex is not None:
            return True

        if self.TxFrequency is not None and self.TxFrequency.Min is not None \
                and self.TxFrequency.Min <= 0:
            self.log_validity_error(
                'TxFrequency.Min is negative, but RefFreqIndex is not populated.')
            return False
        return True

    def _check_tx_sequence(self):
        # type: () -> bool

        cond = True
        if self.TxPolarization == 'SEQUENCE' and self.TxSequence is None:
            self.log_validity_error(
                'TxPolarization is populated as "SEQUENCE", but TxSequence is not populated.')
            cond = False
        if self.TxSequence is not None:
            if self.TxPolarization != 'SEQUENCE':
                self.log_validity_error(
                    'TxSequence is populated, but TxPolarization is populated as {}'.format(self.TxPolarization))
                cond = False
            tx_pols = list(set([entry.TxPolarization for entry in self.TxSequence]))
            if len(tx_pols) == 1:
                self.log_validity_error(
                    'TxSequence is populated, but the only unique TxPolarization '
                    'among the entries is {}'.format(tx_pols[0]))
                cond = False
        return cond

    def _check_waveform_parameters(self):
        """
        Validate the waveform parameters for consistency.

        Returns
        -------
        bool
        """

        def validate_entry(index, waveform):
            # type: (int, WaveformParametersType) -> bool
            this_cond = True
            try:
                if abs(waveform.TxRFBandwidth/(waveform.TxPulseLength*waveform.TxFMRate) - 1) > 1e-3:
                    self.log_validity_error(
                        'The TxRFBandwidth, TxPulseLength, and TxFMRate parameters of Waveform '
                        'entry {} are inconsistent'.format(index+1))
                    this_cond = False
            except (AttributeError, ValueError, TypeError):
                pass

            if waveform.RcvDemodType == 'CHIRP' and waveform.RcvFMRate != 0:
                self.log_validity_error(
                    'RcvDemodType == "CHIRP" and RcvFMRate != 0 in Waveform entry {}'.format(index+1))
                this_cond = False
            if waveform.RcvDemodType == 'STRETCH' and \
                    waveform.RcvFMRate is not None and waveform.TxFMRate is not None and \
                    abs(waveform.RcvFMRate/waveform.TxFMRate - 1) > 1e-3:
                self.log_validity_warning(
                    'RcvDemodType = "STRETCH", RcvFMRate = {}, and TxFMRate = {} in '
                    'Waveform entry {}. The RcvFMRate and TxFMRate should very likely '
                    'be the same.'.format(waveform.RcvFMRate, waveform.TxFMRate, index+1))

            if self.RefFreqIndex is None:
                if waveform.TxFreqStart <= 0:
                    self.log_validity_error(
                        'TxFreqStart is negative in Waveform entry {}, but RefFreqIndex '
                        'is not populated.'.format(index+1))
                    this_cond = False
                if waveform.RcvFreqStart is not None and waveform.RcvFreqStart <= 0:
                    self.log_validity_error(
                        'RcvFreqStart is negative in Waveform entry {}, but RefFreqIndex '
                        'is not populated.'.format(index + 1))
                    this_cond = False
            if waveform.TxPulseLength is not None and waveform.RcvWindowLength is not None and \
                    waveform.TxPulseLength > waveform.RcvWindowLength:
                self.log_validity_error(
                    'TxPulseLength ({}) is longer than RcvWindowLength ({}) in '
                    'Waveform entry {}'.format(waveform.TxPulseLength, waveform.RcvWindowLength, index+1))
                this_cond = False
            if waveform.RcvIFBandwidth is not None and waveform.ADCSampleRate is not None and \
                    waveform.RcvIFBandwidth > waveform.ADCSampleRate:
                self.log_validity_error(
                    'RcvIFBandwidth ({}) is longer than ADCSampleRate ({}) in '
                    'Waveform entry {}'.format(waveform.RcvIFBandwidth, waveform.ADCSampleRate, index+1))
                this_cond = False
            if waveform.RcvDemodType is not None and waveform.RcvDemodType == 'CHIRP' \
                    and waveform.TxRFBandwidth is not None and waveform.ADCSampleRate is not None \
                    and (waveform.TxRFBandwidth > waveform.ADCSampleRate):
                self.log_validity_error(
                    'RcvDemodType is "CHIRP" and TxRFBandwidth ({}) is larger than ADCSampleRate ({}) '
                    'in Waveform entry {}'.format(waveform.TxRFBandwidth, waveform.ADCSampleRate, index+1))
                this_cond = False
            if waveform.RcvWindowLength is not None and waveform.TxPulseLength is not None and \
                    waveform.TxFMRate is not None and waveform.RcvFreqStart is not None and \
                    waveform.TxFreqStart is not None and waveform.TxRFBandwidth is not None:
                freq_tol = (waveform.RcvWindowLength - waveform.TxPulseLength)*waveform.TxFMRate
                if waveform.RcvFreqStart >= waveform.TxFreqStart + waveform.TxRFBandwidth + freq_tol:
                    self.log_validity_error(
                        'RcvFreqStart ({}), TxFreqStart ({}), and TxRfBandwidth ({}) parameters are inconsistent '
                        'in Waveform entry {}'.format(
                            waveform.RcvFreqStart, waveform.TxFreqStart, waveform.TxRFBandwidth, index + 1))
                    this_cond = False
                if waveform.RcvFreqStart <= waveform.TxFreqStart - freq_tol:
                    self.log_validity_error(
                        'RcvFreqStart ({}) and TxFreqStart ({}) parameters are inconsistent '
                        'in Waveform entry {}'.format(waveform.RcvFreqStart, waveform.TxFreqStart, index + 1))
                    this_cond = False

            return this_cond

        if self.Waveform is None or len(self.Waveform) < 1:
            return True

        cond = True
        # fetch min/max TxFreq observed
        wf_min_freq = None
        wf_max_freq = None
        for entry in self.Waveform:
            freq_start = entry.TxFreqStart
            freq_bw = entry.TxRFBandwidth
            if freq_start is not None:
                wf_min_freq = freq_start if wf_min_freq is None else \
                    min(wf_min_freq, freq_start)
                if entry.TxRFBandwidth is not None:
                    wf_max_freq = freq_start + freq_bw if wf_max_freq is None else \
                        max(wf_max_freq, freq_start + freq_bw)

        if wf_min_freq is not None and self.TxFrequency is not None and self.TxFrequency.Min is not None:
            if abs(self.TxFrequency.Min/wf_min_freq - 1) > 1e-3:
                self.log_validity_error(
                    'The stated TxFrequency.Min is {}, but the minimum populated in a '
                    'Waveform entry is {}'.format(self.TxFrequency.Min, wf_min_freq))
                cond = False
        if wf_max_freq is not None and self.TxFrequency is not None and self.TxFrequency.Max is not None:
            if abs(self.TxFrequency.Max/wf_max_freq - 1) > 1e-3:
                self.log_validity_error(
                    'The stated TxFrequency.Max is {}, but the maximum populated in a '
                    'Waveform entry is {}'.format(self.TxFrequency.Max, wf_max_freq))
                cond = False
        for t_index, t_waveform in enumerate(self.Waveform):
            cond &= validate_entry(t_index, t_waveform)
        return cond

    def _basic_validity_check(self):
        valid = super(RadarCollectionType, self)._basic_validity_check()
        valid &= self._check_frequency()
        valid &= self._check_tx_sequence()
        valid &= self._check_waveform_parameters()
        return valid

    def permits_version_1_1(self):
        """
        Does this value permit storage in SICD version 1.1?

        Returns
        -------
        bool
        """

        if self.RcvChannels is None:
            return True

        cond = True
        for entry in self.RcvChannels:
            cond &= entry.permits_version_1_1()
        return cond
