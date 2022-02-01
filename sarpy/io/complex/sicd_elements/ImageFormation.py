"""
The ImageFormationType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import List, Union

import numpy

from sarpy.io.xml.base import Serializable, Arrayable, ParametersCollection
from sarpy.io.xml.descriptors import StringDescriptor, StringEnumDescriptor, \
    FloatDescriptor, IntegerDescriptor, IntegerListDescriptor, BooleanDescriptor, \
    ComplexDescriptor, DateTimeDescriptor, SerializableDescriptor, \
    SerializableListDescriptor, ParametersDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
from .blocks import DUAL_POLARIZATION_VALUES
from .RadarCollection import get_band_name
from .utils import is_polstring_version1


class RcvChanProcType(Serializable):
    """The Received Processed Channels."""
    _fields = ('NumChanProc', 'PRFScaleFactor', 'ChanIndices')
    _required = ('NumChanProc', 'ChanIndices')
    _collections_tags = {
        'ChanIndices': {'array': False, 'child_tag': 'ChanIndex'}}
    _numeric_format = {'PRFScaleFactor': FLOAT_FORMAT}
    # descriptors
    NumChanProc = IntegerDescriptor(
        'NumChanProc', _required, strict=DEFAULT_STRICT,
        docstring='Number of receive data channels processed to form the image.')  # type: int
    PRFScaleFactor = FloatDescriptor(
        'PRFScaleFactor', _required, strict=DEFAULT_STRICT,
        docstring='Factor indicating the ratio of the effective PRF to the actual PRF.')  # type: float
    ChanIndices = IntegerListDescriptor(
        'ChanIndices', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Index of a data channel that was processed.')  # type: List[int]

    def __init__(self, NumChanProc=None, PRFScaleFactor=None, ChanIndices=None, **kwargs):
        """

        Parameters
        ----------
        NumChanProc : int
        PRFScaleFactor : float
        ChanIndices : List[int]
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NumChanProc = NumChanProc
        self.PRFScaleFactor = PRFScaleFactor
        self.ChanIndices = ChanIndices
        super(RcvChanProcType, self).__init__(**kwargs)


class TxFrequencyProcType(Serializable, Arrayable):
    """The transmit frequency range."""
    _fields = ('MinProc', 'MaxProc')
    _required = _fields
    _numeric_format = {'MinProc': '0.17E', 'MaxProc': '0.17E'}
    # descriptors
    MinProc = FloatDescriptor(
        'MinProc', _required, strict=DEFAULT_STRICT,
        docstring='The minimum transmit frequency processed to form the image, in Hz.')  # type: float
    MaxProc = FloatDescriptor(
        'MaxProc', _required, strict=DEFAULT_STRICT,
        docstring='The maximum transmit frequency processed to form the image, in Hz.')  # type: float

    def __init__(self, MinProc=None, MaxProc=None, **kwargs):
        """

        Parameters
        ----------
        MinProc : float
        MaxProc : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.MinProc, self.MaxProc = MinProc, MaxProc
        super(TxFrequencyProcType, self).__init__(**kwargs)

    @property
    def center_frequency(self):
        """
        None|float: The center frequency.
        """

        if self.MinProc is None or self.MaxProc is None:
            return None
        return 0.5*(self.MinProc + self.MaxProc)

    @property
    def bandwidth(self):
        """
        None|float: The bandwidth in Hz.
        """

        if self.MinProc is None or self.MaxProc is None:
            return None
        return self.MaxProc - self.MinProc

    def _apply_reference_frequency(self, reference_frequency):
        if self.MinProc is not None:
            self.MinProc += reference_frequency
        if self.MaxProc is not None:
            self.MaxProc += reference_frequency

    def _basic_validity_check(self):
        condition = super(TxFrequencyProcType, self)._basic_validity_check()
        if self.MinProc is not None and self.MaxProc is not None and self.MaxProc < self.MinProc:
            self.log_validity_error(
                'Invalid frequency bounds MinProc ({}) > MaxProc ({})'.format(self.MinProc, self.MaxProc))
            condition = False
        return condition

    def get_band_name(self):
        """
        Gets the band name.

        Returns
        -------
        str
        """

        min_band = get_band_name(self.MinProc)
        max_band = get_band_name(self.MaxProc)
        if min_band == max_band:
            return min_band
        else:
            return '{}_{}'.format(min_band, max_band)

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

        return numpy.array([self.MinProc, self.MaxProc], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [MinProc, MaxProc]

        Returns
        -------
        LatLonType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(MinProc=array[0], MaxProc=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class ProcessingType(Serializable):
    """The transmit frequency range"""
    _fields = ('Type', 'Applied', 'Parameters')
    _required = ('Type', 'Applied')
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    Type = StringDescriptor(
        'Type', _required, strict=DEFAULT_STRICT,
        docstring='The processing type identifier.')  # type: str
    Applied = BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Indicates whether the given processing type has been applied.')  # type: bool
    Parameters = ParametersDescriptor(
        'Parameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The parameters collection.')  # type: ParametersCollection

    def __init__(self, Type=None, Applied=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        Type : str
        Applied : bool
        Parameters : ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Type = Type
        self.Applied = Applied
        self.Parameters = Parameters
        super(ProcessingType, self).__init__(**kwargs)


class DistortionType(Serializable):
    """Distortion"""
    _fields = (
        'CalibrationDate', 'A', 'F1', 'F2', 'Q1', 'Q2', 'Q3', 'Q4',
        'GainErrorA', 'GainErrorF1', 'GainErrorF2', 'PhaseErrorF1', 'PhaseErrorF2')
    _required = ('A', 'F1', 'Q1', 'Q2', 'F2', 'Q3', 'Q4')
    _numeric_format = {key: FLOAT_FORMAT for key in _fields[1:]}
    # descriptors
    CalibrationDate = DateTimeDescriptor(
        'CalibrationDate', _required, strict=DEFAULT_STRICT,
        docstring='The calibration date.')
    A = FloatDescriptor(
        'A', _required, strict=DEFAULT_STRICT,
        docstring='Absolute amplitude scale factor.')  # type: float
    # receive distorion matrix
    F1 = ComplexDescriptor(
        'F1', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (2,2).')  # type: complex
    Q1 = ComplexDescriptor(
        'Q1', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (1,2).')  # type: complex
    Q2 = ComplexDescriptor(
        'Q2', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (2,1).')  # type: complex
    # transmit distortion matrix
    F2 = ComplexDescriptor(
        'F2', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (2,2).')  # type: complex
    Q3 = ComplexDescriptor(
        'Q3', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (2,1).')  # type: complex
    Q4 = ComplexDescriptor(
        'Q4', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (1,2).')  # type: complex
    # gain estimation error
    GainErrorA = FloatDescriptor(
        'GainErrorA', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter A.')  # type: float
    GainErrorF1 = FloatDescriptor(
        'GainErrorF1', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter F1.')  # type: float
    GainErrorF2 = FloatDescriptor(
        'GainErrorF2', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter F2.')  # type: float
    PhaseErrorF1 = FloatDescriptor(
        'PhaseErrorF1', _required, strict=DEFAULT_STRICT,
        docstring='Phase estimation error standard deviation (in dB) for parameter F1.')  # type: float
    PhaseErrorF2 = FloatDescriptor(
        'PhaseErrorF2', _required, strict=DEFAULT_STRICT,
        docstring='Phase estimation error standard deviation (in dB) for parameter F2.')  # type: float

    def __init__(self, CalibrationDate=None, A=None,
                 F1=None, Q1=None, Q2=None, F2=None, Q3=None, Q4=None,
                 GainErrorA=None, GainErrorF1=None, GainErrorF2=None,
                 PhaseErrorF1=None, PhaseErrorF2=None, **kwargs):
        """

        Parameters
        ----------
        CalibrationDate : numpy.datetime64|datetime|date|str
        A : float
        F1 : complex
        Q1 : complex
        Q2 : complex
        F2 : complex
        Q3 : complex
        Q4 : complex
        GainErrorA : float
        GainErrorF1 : float
        GainErrorF2 : float
        PhaseErrorF1 : float
        PhaseErrorF2 : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CalibrationDate = CalibrationDate
        self.A = A
        self.F1, self.Q1, self.Q2 = F1, Q1, Q2
        self.F2, self.Q3, self.Q4 = F2, Q3, Q4
        self.GainErrorA = GainErrorA
        self.GainErrorF1, self.GainErrorF2 = GainErrorF1, GainErrorF2
        self.PhaseErrorF1, self.PhaseErrorF2 = PhaseErrorF1, PhaseErrorF2
        super(DistortionType, self).__init__(**kwargs)


class PolarizationCalibrationType(Serializable):
    """The polarization calibration"""
    _fields = ('DistortCorrectApplied', 'Distortion')
    _required = _fields
    # descriptors
    DistortCorrectApplied = BooleanDescriptor(
        'DistortCorrectApplied', _required, strict=DEFAULT_STRICT,
        docstring='Indicates whether the polarization calibration has been applied.')  # type: bool
    Distortion = SerializableDescriptor(
        'Distortion', DistortionType, _required, strict=DEFAULT_STRICT,
        docstring='The distortion parameters.')  # type: DistortionType

    def __init__(self, DistortCorrectApplied=None, Distortion=None, **kwargs):
        """

        Parameters
        ----------
        DistortCorrectApplied : bool
        Distortion : DistortionType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DistortCorrectApplied = DistortCorrectApplied
        self.Distortion = Distortion
        super(PolarizationCalibrationType, self).__init__(**kwargs)


class ImageFormationType(Serializable):
    """The image formation process parameters."""
    _fields = (
        'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc', 'SegmentIdentifier',
        'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus', 'Processings',
        'PolarizationCalibration')
    _required = (
        'RcvChanProc', 'TxRcvPolarizationProc', 'TStartProc', 'TEndProc', 'TxFrequencyProc',
        'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus')
    _collections_tags = {'Processings': {'array': False, 'child_tag': 'Processing'}}
    _numeric_format = {'TStartProc': FLOAT_FORMAT, 'EndProc': FLOAT_FORMAT}
    # class variables
    _IMG_FORM_ALGO_VALUES = ('PFA', 'RMA', 'RGAZCOMP', 'OTHER')
    _ST_BEAM_COMP_VALUES = ('NO', 'GLOBAL', 'SV')
    _IMG_BEAM_COMP_VALUES = ('NO', 'SV')
    _AZ_AUTOFOCUS_VALUES = _ST_BEAM_COMP_VALUES
    _RG_AUTOFOCUS_VALUES = _ST_BEAM_COMP_VALUES
    # descriptors
    RcvChanProc = SerializableDescriptor(
        'RcvChanProc', RcvChanProcType, _required, strict=DEFAULT_STRICT,
        docstring='The received processed channels.')  # type: RcvChanProcType
    TxRcvPolarizationProc = StringEnumDescriptor(
        'TxRcvPolarizationProc', DUAL_POLARIZATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The combined transmit/receive polarization processed to form the image.')  # type: str
    TStartProc = FloatDescriptor(
        'TStartProc', _required, strict=DEFAULT_STRICT,
        docstring='Earliest slow time value for data processed to form the image '
                  'from `CollectionStart`.')  # type: float
    TEndProc = FloatDescriptor(
        'TEndProc', _required, strict=DEFAULT_STRICT,
        docstring='Latest slow time value for data processed to form the image from `CollectionStart`.')  # type: float
    TxFrequencyProc = SerializableDescriptor(
        'TxFrequencyProc', TxFrequencyProcType, _required, strict=DEFAULT_STRICT,
        docstring='The range of transmit frequency processed to form the image.')  # type: TxFrequencyProcType
    SegmentIdentifier = StringDescriptor(
        'SegmentIdentifier', _required, strict=DEFAULT_STRICT,
        docstring='Identifier that describes the image that was processed. '
                  'Must be included when `SICD.RadarCollection.Area.Plane.SegmentList` is included.')  # type: str
    ImageFormAlgo = StringEnumDescriptor(
        'ImageFormAlgo', _IMG_FORM_ALGO_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        The image formation algorithm used:

        * `PFA` - Polar Format Algorithm

        * `RMA` - Range Migration (Omega-K, Chirp Scaling, Range-Doppler)

        * `RGAZCOMP` - Simple range, Doppler compression.

        """)  # type: str
    STBeamComp = StringEnumDescriptor(
        'STBeamComp', _ST_BEAM_COMP_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Indicates if slow time beam shape compensation has been applied.

        * `NO` - No ST beam shape compensation.

        * `GLOBAL` - Global ST beam shape compensation applied.

        * `SV` - Spatially variant beam shape compensation applied.

        """)  # type: str
    ImageBeamComp = StringEnumDescriptor(
        'ImageBeamComp', _IMG_BEAM_COMP_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Indicates if image domain beam shape compensation has been applied.

        * `NO` - No image domain beam shape compensation.

        * `SV` - Spatially variant image domain beam shape compensation applied.

        """)  # type: str
    AzAutofocus = StringEnumDescriptor(
        'AzAutofocus', _AZ_AUTOFOCUS_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates if azimuth autofocus correction has been applied, with similar '
                  'interpretation as `STBeamComp`.')  # type: str
    RgAutofocus = StringEnumDescriptor(
        'RgAutofocus', _RG_AUTOFOCUS_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates if range autofocus correction has been applied, with similar '
                  'interpretation as `STBeamComp`.')  # type: str
    Processings = SerializableListDescriptor(
        'Processings', ProcessingType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters to describe types of specific processing that may have been applied '
                  'such as additional compensations.')  # type: Union[None, List[ProcessingType]]
    PolarizationCalibration = SerializableDescriptor(
        'PolarizationCalibration', PolarizationCalibrationType, _required, strict=DEFAULT_STRICT,
        docstring='The polarization calibration details.')  # type: PolarizationCalibrationType

    def __init__(self, RcvChanProc=None, TxRcvPolarizationProc=None,
                 TStartProc=None, TEndProc=None,
                 TxFrequencyProc=None, SegmentIdentifier=None, ImageFormAlgo=None,
                 STBeamComp=None, ImageBeamComp=None, AzAutofocus=None, RgAutofocus=None,
                 Processings=None, PolarizationCalibration=None, **kwargs):
        """

        Parameters
        ----------
        RcvChanProc : RcvChanProcType
        TxRcvPolarizationProc : str
        TStartProc : float
        TEndProc : float
        TxFrequencyProc : TxFrequencyProcType|numpy.ndarray|list|tuple
        SegmentIdentifier : str
        ImageFormAlgo : str
        STBeamComp : str
        ImageBeamComp :str
        AzAutofocus : str
        RgAutofocus : str
        Processings : None|List[ProcessingType]
        PolarizationCalibration : PolarizationCalibrationType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RcvChanProc = RcvChanProc
        self.TxRcvPolarizationProc = TxRcvPolarizationProc
        self.TStartProc, self.TEndProc = TStartProc, TEndProc
        if isinstance(TxFrequencyProc, (numpy.ndarray, list, tuple)) and len(TxFrequencyProc) >= 2:
            self.TxFrequencyProc = TxFrequencyProcType(MinProc=TxFrequencyProc[0], MaxProc=TxFrequencyProc[1])
        else:
            self.TxFrequencyProc = TxFrequencyProc
        self.SegmentIdentifier = SegmentIdentifier
        self.ImageFormAlgo = ImageFormAlgo
        self.STBeamComp, self.ImageBeamComp = STBeamComp, ImageBeamComp
        self.AzAutofocus, self.RgAutofocus = AzAutofocus, RgAutofocus
        self.Processings = Processings
        self.PolarizationCalibration = PolarizationCalibration
        super(ImageFormationType, self).__init__(**kwargs)

    def _basic_validity_check(self):
        condition = super(ImageFormationType, self)._basic_validity_check()
        if self.TStartProc is not None and self.TEndProc is not None and self.TEndProc < self.TStartProc:
            self.log_validity_error(
                'Invalid time processing bounds TStartProc ({}) > TEndProc ({})'.format(
                    self.TStartProc, self.TEndProc))
            condition = False
        return condition

    def _derive_tx_frequency_proc(self, RadarCollection):
        """
        Populate a default for processed frequency values, based on the assumption that the entire
        transmitted bandwidth was processed. This is expected to be called by SICD parent.

        Parameters
        ----------
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType

        Returns
        -------
        None
        """

        if RadarCollection is not None and RadarCollection.TxFrequency is not None and \
                RadarCollection.TxFrequency.Min is not None and RadarCollection.TxFrequency.Max is not None:
            # this is based on the assumption that the entire transmitted bandwidth was processed.
            if self.TxFrequencyProc is not None:
                self.TxFrequencyProc = TxFrequencyProcType(
                    MinProc=RadarCollection.TxFrequency.Min, MaxProc=RadarCollection.TxFrequency.Max)
                # how would it make sense to set only one end?
            elif self.TxFrequencyProc.MinProc is None:
                self.TxFrequencyProc.MinProc = RadarCollection.TxFrequency.Min
            elif self.TxFrequencyProc.MaxProc is None:
                self.TxFrequencyProc.MaxProc = RadarCollection.TxFrequency.Max

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

        if self.TxFrequencyProc is not None:
            # noinspection PyProtectedMember
            self.TxFrequencyProc._apply_reference_frequency(reference_frequency)

    def get_polarization(self):
        """
        Gets the transmit/receive polarization.

        Returns
        -------
        str
        """

        return self.TxRcvPolarizationProc if self.TxRcvPolarizationProc is not None else 'UNKNOWN'

    def get_polarization_abbreviation(self):
        """
        Gets the transmit/receive polarization abbreviation for the suggested name.

        Returns
        -------
        str
        """

        pol = self.TxRcvPolarizationProc
        if pol is None or pol in ('OTHER', 'UNKNOWN'):
            return 'UN'
        fp, sp = pol.split(':')
        return fp[0]+sp[0]

    def get_transmit_band_name(self):
        """
        Gets the transmit band name.

        Returns
        -------
        str
        """

        if self.TxFrequencyProc is not None:
            return self.TxFrequencyProc.get_band_name()
        else:
            return 'UN'

    def permits_version_1_1(self):
        """
        Does this value permit storage in SICD version 1.1?

        Returns
        -------
        bool
        """

        return is_polstring_version1(self.TxRcvPolarizationProc)
