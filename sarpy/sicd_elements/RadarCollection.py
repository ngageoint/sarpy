"""
The RadarCollectionType definition.
"""

import logging
from typing import List, Union

import numpy

from ._base import Serializable, DEFAULT_STRICT, \
    _StringDescriptor, _StringEnumDescriptor, _FloatDescriptor, _IntegerDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from ._blocks import ParameterType, XYZType, LatLonHAECornerRestrictionType

import sarpy.geometry.geocoords as geocoords


__classification__ = "UNCLASSIFIED"


class TxFrequencyType(Serializable):
    """The transmit frequency range"""
    _fields = ('Min', 'Max')
    _required = _fields
    # descriptors
    Min = _FloatDescriptor(
        'Min', required=_required, strict=DEFAULT_STRICT,
        docstring='The transmit minimum frequency in Hz.')  # type: float
    Max = _FloatDescriptor(
        'Max', required=_required, strict=DEFAULT_STRICT,
        docstring='The transmit maximum frequency in Hz.')  # type: float


class WaveformParametersType(Serializable):
    """Transmit and receive demodulation waveform parameters."""
    _fields = (
        'TxPulseLength', 'TxRFBandwidth', 'TxFreqStart', 'TxFMRate', 'RcvDemodType', 'RcvWindowLength',
        'ADCSampleRate', 'RcvIFBandwidth', 'RcvFreqStart', 'RcvFMRate')
    _required = ()
    # other class variables
    _DEMOD_TYPE_VALUES = ('STRETCH', 'CHIRP')
    # descriptors
    TxPulseLength = _FloatDescriptor(
        'TxPulseLength', _required, strict=DEFAULT_STRICT,
        docstring='Transmit pulse length in seconds.')  # type: float
    TxRFBandwidth = _FloatDescriptor(
        'TxRFBandwidth', _required, strict=DEFAULT_STRICT,
        docstring='Transmit RF bandwidth of the transmit pulse in Hz.')  # type: float
    TxFreqStart = _FloatDescriptor(
        'TxFreqStart', _required, strict=DEFAULT_STRICT,
        docstring='Transmit Start frequency for Linear FM waveform in Hz, may be relative '
                  'to reference frequency.')  # type: float
    TxFMRate = _FloatDescriptor(
        'TxFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Transmit FM rate for Linear FM waveform in Hz/second.')  # type: float
    RcvDemodType = _StringEnumDescriptor(
        'RcvDemodType', _DEMOD_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="Receive demodulation used when Linear FM waveform is used on transmit.")  # type: float
    RcvWindowLength = _FloatDescriptor(
        'RcvWindowLength', _required, strict=DEFAULT_STRICT,
        docstring='Receive window duration in seconds.')  # type: float
    ADCSampleRate = _FloatDescriptor(
        'ADCSampleRate', _required, strict=DEFAULT_STRICT,
        docstring='Analog-to-Digital Converter sampling rate in samples/second.')  # type: float
    RcvIFBandwidth = _FloatDescriptor(
        'RcvIFBandwidth', _required, strict=DEFAULT_STRICT,
        docstring='Receive IF bandwidth in Hz.')  # type: float
    RcvFreqStart = _FloatDescriptor(
        'RcvFreqStart', _required, strict=DEFAULT_STRICT,
        docstring='Receive demodulation start frequency in Hz, may be relative to reference frequency.')  # type: float
    RcvFMRate = _FloatDescriptor(
        'RcvFMRate', _required, strict=DEFAULT_STRICT,
        docstring='Receive FM rate. Should be 0 if RcvDemodType = "CHIRP".')  # type: float

    def _basic_validity_check(self):
        valid = super(WaveformParametersType, self)._basic_validity_check()
        if (self.RcvDemodType == 'CHIRP') and (self.RcvFMRate != 0):
            # TODO: VERIFY - should we simply reset RcvFMRate?
            logging.error(
                'In WaveformParameters, we have RcvDemodType == "CHIRP" and self.RcvFMRate non-zero.')
            valid = False
        return valid

    def derive(self):
        """
        Populate any derived data in WaveformParametersType.

        Returns
        -------
        None
        """

        if self.RcvDemodType == 'CHIRP' and self.RcvFMRate is None:
            self.RcvFMRate = 0.0  # this should be 0 anyways?
        if self.RcvFMRate == 0.0 and self.RcvDemodType is None:
            self.RcvDemodType = 'CHIRP'
        if self.TxPulseLength is not None and self.TxFMRate is not None and self.TxRFBandwidth is None:
            self.TxRFBandwidth = self.TxPulseLength*self.TxFMRate
        if self.TxPulseLength is not None and self.TxRFBandwidth is not None and self.TxFMRate is None:
            self.TxFMRate = self.TxRFBandwidth/self.TxPulseLength
        if self.TxFMRate is not None and self.TxRFBandwidth is not None and self.TxPulseLength is None:
            self.TxPulseLength = self.TxRFBandwidth/self.TxFMRate


class TxStepType(Serializable):
    """Transmit sequence step details"""
    _fields = ('WFIndex', 'TxPolarization', 'index')
    _required = ('index', )
    # other class variables
    _POLARIZATION2_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER')
    # descriptors
    WFIndex = _IntegerDescriptor(
        'WFIndex', _required, strict=DEFAULT_STRICT,
        docstring='The waveform number for this step.')  # type: int
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION2_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Transmit signal polarization for this step.')  # type: str
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT,
        docstring='The step index')  # type: int


class ChanParametersType(Serializable):
    """Transmit receive sequence step details"""
    _fields = ('TxRcvPolarization', 'RcvAPCIndex', 'index')
    _required = ('TxRcvPolarization', 'index', )
    # other class variables
    _DUAL_POLARIZATION_VALUES = (
        'V:V', 'V:H', 'H:V', 'H:H', 'RHC:RHC', 'RHC:LHC', 'LHC:RHC', 'LHC:LHC', 'OTHER', 'UNKNOWN')
    # descriptors
    TxRcvPolarization = _StringEnumDescriptor(
        'TxRcvPolarization', _DUAL_POLARIZATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Combined Transmit and Receive signal polarization for the channel.')  # type: str
    RcvAPCIndex = _IntegerDescriptor(
        'RcvAPCIndex', _required, strict=DEFAULT_STRICT,
        docstring='Index of the Receive Aperture Phase Center (Rcv APC). Only include if Receive APC position '
                  'polynomial(s) are included.')  # type: int
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT, docstring='The parameter index')  # type: int

    def get_transmit_polarization(self):
        if self.TxRcvPolarization is None:
            return None
        elif self.TxRcvPolarization in ['OTHER', 'UNKNOWN']:
            return 'OTHER'
        else:
            return self.TxRcvPolarization.split(':')[0]


class ReferencePointType(Serializable):
    """The reference point definition"""
    _fields = ('ECF', 'Line', 'Sample', 'name')
    _required = _fields
    _set_as_attribute = ('name', )
    # descriptors
    ECF = _SerializableDescriptor(
        'ECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The geographical coordinates for the reference point.')  # type: XYZType
    Line = _FloatDescriptor(
        'Line', _required, strict=DEFAULT_STRICT,
        docstring='The reference point line index.')  # type: float
    Sample = _FloatDescriptor(
        'Sample', _required, strict=DEFAULT_STRICT,
        docstring='The reference point sample index.')  # type: float
    name = _StringDescriptor(
        'name', _required, strict=DEFAULT_STRICT,
        docstring='The reference point name.')  # type: str


class XDirectionType(Serializable):
    """The X direction of the collect"""
    _fields = ('UVectECF', 'LineSpacing', 'NumLines', 'FirstLine')
    _required = _fields
    # descriptors
    UVectECF = _SerializableDescriptor(
        'UVectECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The unit vector in the X direction.')  # type: XYZType
    LineSpacing = _FloatDescriptor(
        'LineSpacing', _required, strict=DEFAULT_STRICT,
        docstring='The collection line spacing in the X direction in meters.')  # type: float
    NumLines = _IntegerDescriptor(
        'NumLines', _required, strict=DEFAULT_STRICT,
        docstring='The number of lines in the X direction.')  # type: int
    FirstLine = _IntegerDescriptor(
        'FirstLine', _required, strict=DEFAULT_STRICT,
        docstring='The first line index.')  # type: int


class YDirectionType(Serializable):
    """The Y direction of the collect"""
    _fields = ('UVectECF', 'SampleSpacing', 'NumSamples', 'FirstSample')
    _required = _fields
    # descriptors
    UVectECF = _SerializableDescriptor(
        'UVectECF', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='The unit vector in the Y direction.')  # type: XYZType
    SampleSpacing = _FloatDescriptor(
        'SampleSpacing', _required, strict=DEFAULT_STRICT,
        docstring='The collection sample spacing in the Y direction in meters.')  # type: float
    NumSamples = _IntegerDescriptor(
        'NumSamples', _required, strict=DEFAULT_STRICT,
        docstring='The number of samples in the Y direction.')  # type: int
    FirstSample = _IntegerDescriptor(
        'FirstSample', _required, strict=DEFAULT_STRICT,
        docstring='The first sample index.')  # type: int


class SegmentArrayElement(Serializable):
    """The reference point definition"""
    _fields = ('StartLine', 'StartSample', 'EndLine', 'EndSample', 'Identifier', 'index')
    _required = _fields
    _set_as_attribute = ('index', )
    # descriptors
    StartLine = _IntegerDescriptor(
        'StartLine', _required, strict=DEFAULT_STRICT,
        docstring='The starting line number of the segment.')  # type: int
    StartSample = _IntegerDescriptor(
        'StartSample', _required, strict=DEFAULT_STRICT,
        docstring='The starting sample number of the segment.')  # type: int
    EndLine = _IntegerDescriptor(
        'EndLine', _required, strict=DEFAULT_STRICT,
        docstring='The ending line number of the segment.')  # type: int
    EndSample = _IntegerDescriptor(
        'EndSample', _required, strict=DEFAULT_STRICT,
        docstring='The ending sample number of the segment.')  # type: int
    Identifier = _StringDescriptor(
        'Identifier', _required, strict=DEFAULT_STRICT,
        docstring='Identifier for the segment data boundary.')
    index = _IntegerDescriptor(
        'index', _required, strict=DEFAULT_STRICT,
        docstring='The array index.')  # type: int


class ReferencePlaneType(Serializable):
    """The reference plane"""
    _fields = ('RefPt', 'XDir', 'YDir', 'SegmentList', 'Orientation')
    _required = ('RefPt', 'XDir', 'YDir')
    _collections_tags = {'SegmentList': {'array': True, 'child_tag': 'SegmentList'}}
    # other class variable
    _ORIENTATION_VALUES = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'ARBITRARY')
    # descriptors
    RefPt = _SerializableDescriptor(
        'RefPt', ReferencePointType, _required, strict=DEFAULT_STRICT,
        docstring='The reference point.')  # type: ReferencePointType
    XDir = _SerializableDescriptor(
        'XDir', XDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The X direction collection plane parameters.')  # type: XDirectionType
    YDir = _SerializableDescriptor(
        'YDir', YDirectionType, _required, strict=DEFAULT_STRICT,
        docstring='The Y direction collection plane parameters.')  # type: YDirectionType
    SegmentList = _SerializableArrayDescriptor(
        'SegmentList', SegmentArrayElement, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The segment array.')  # type: Union[numpy.ndarray, List[SegmentArrayElement]]
    Orientation = _StringEnumDescriptor(
        'Orientation', _ORIENTATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Describes the shadow intent of the display plane.')  # type: str

    def get_ecf_corner_array(self):
        """
        Use the XDir and YDir definitions to return the corner points in ECF coordinates as a 4x3 array.

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
    """The collection area"""
    _fields = ('Corner', 'Plane')
    _required = ('Corner', )
    _collections_tags = {
        'Corner': {'array': False, 'child_tag': 'ACP'}, }
    # descriptors
    Corner = _SerializableArrayDescriptor(
        'Corner', LatLonHAECornerRestrictionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        minimum_length=4, maximum_length=4,
        docstring='The collection area corner point definition array.')  # type: List[LatLonHAECornerRestrictionType]
    Plane = _SerializableDescriptor(
        'Plane', ReferencePlaneType, _required, strict=DEFAULT_STRICT,
        docstring='A rectangular area in a geo-located display plane.')  # type: ReferencePlaneType

    def __init__(self, **kwargs):
        super(AreaType, self).__init__(**kwargs)

    def _derive_corner_from_plane(self):
        # try to define the corner points - for SICD 0.5.
        if self.Corner is not None:
            return  # nothing to be done
        if self.Plane is None:
            return  # nothing to derive from
        corners = self.Plane.get_ecf_corner_array()
        self.Corner = [
            LatLonHAECornerRestrictionType(**{'Lat': entry[0], 'Lon': entry[1], 'HAE': entry[2], 'index': i})
            for i, entry in enumerate(geocoords.ecf_to_geodetic(corners))]

    def derive(self):
        """
        Populate any internally derived data for AreaType.

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
        'RcvChannels': {'array': True, 'child_tag': 'RcvChannels'},
        'Parameters': {'array': False, 'child_tag': 'Parameters'}}
    # other class variables
    _POLARIZATION1_VALUES = ('V', 'H', 'RHC', 'LHC', 'OTHER', 'UNKNOWN', 'SEQUENCE')
    # descriptors
    TxFrequency = _SerializableDescriptor(
        'TxFrequency', TxFrequencyType, _required, strict=DEFAULT_STRICT,
        docstring='The transmit frequency range.')  # type: TxFrequencyType
    RefFreqIndex = _IntegerDescriptor(
        'RefFreqIndex', _required, strict=DEFAULT_STRICT,
        docstring='The reference frequency index, if applicable. if present, all RF frequency values are expressed '
                  'as offsets from a reference frequency.')  # type: int
    Waveform = _SerializableArrayDescriptor(
        'Waveform', WaveformParametersType, _collections_tags, _required,
        strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Transmit and receive demodulation waveform parameters.'
    )  # type: Union[numpy.ndarray, List[WaveformParametersType]]
    TxPolarization = _StringEnumDescriptor(
        'TxPolarization', _POLARIZATION1_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The transmit polarization.')  # type: str
    TxSequence = _SerializableArrayDescriptor(
        'TxSequence', TxStepType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The transmit sequence parameters array. If present, indicates the transmit signal steps through '
                  'a repeating sequence of waveforms and/or polarizations. '
                  'One step per Inter-Pulse Period.')  # type: Union[numpy.ndarray, List[TxStepType]]
    RcvChannels = _SerializableArrayDescriptor(
        'RcvChannels', ChanParametersType, _collections_tags,
        _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='Receive data channel parameters.')  # type: Union[numpy.ndarray, List[ChanParametersType]]
    Area = _SerializableDescriptor(
        'Area', AreaType, _required, strict=DEFAULT_STRICT,
        docstring='The imaged area covered by the collection.')  # type: AreaType
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='A parameters list.')  # type: List[ParameterType]

    def derive(self):
        """
        Populates derived data in RadarCollection. Expected to be called by SICD parent.

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
        # TxPolarization was optional prior to SICD 1.0. It may need to be derived.
        if self.TxSequence is not None:
            self.TxPolarization = 'SEQUENCE'
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

            steps = []
            for i, chanparam in enumerate(self.RcvChannels):
                # TODO: VERIFY - there's some effort to avoid repetition in sicd.py at line 1112. Is this necessary?
                #   What about WFIndex? Is it possible to derive that?
                steps.append(TxStepType(index=i, TxPolarization=chanparam.get_transmit_polarization()))
            self.TxSequence = steps
            self.TxPolarization = 'SEQUENCE'
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
