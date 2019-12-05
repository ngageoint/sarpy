"""
The ImageFormationType definition.
"""

from typing import List

from .base import Serializable, DEFAULT_STRICT, \
    _StringDescriptor, _StringEnumDescriptor, _FloatDescriptor, _IntegerDescriptor, \
    _BooleanDescriptor, _ComplexDescriptor, _DateTimeDescriptor, _IntegerListDescriptor, \
    _SerializableDescriptor, _SerializableArrayDescriptor
from .blocks import ParameterType


__classification__ = "UNCLASSIFIED"


class RcvChanProcType(Serializable):
    """The Received Processed Channels."""
    _fields = ('NumChanProc', 'PRFScaleFactor', 'ChanIndices')
    _required = ('NumChanProc', 'ChanIndices')
    _collections_tags = {
        'ChanIndices': {'array': False, 'child_tag': 'ChanIndex'}}
    # descriptors
    NumChanProc = _IntegerDescriptor(
        'NumChanProc', _required, strict=DEFAULT_STRICT,
        docstring='Number of receive data channels processed to form the image.')  # type: int
    PRFScaleFactor = _FloatDescriptor(
        'PRFScaleFactor', _required, strict=DEFAULT_STRICT,
        docstring='Factor indicating the ratio of the effective PRF to the actual PRF.')  # type: float
    ChanIndices = _IntegerListDescriptor(  # TODO: CLARIFY - what is the intent of this one?
        'ChanIndices', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Index of a data channel that was processed.')  # type: List[int]


class TxFrequencyProcType(Serializable):
    """The transmit frequency range."""
    _fields = ('MinProc', 'MaxProc')
    _required = _fields
    # descriptors
    MinProc = _FloatDescriptor(
        'MinProc', _required, strict=DEFAULT_STRICT,
        docstring='The minimum transmit frequency processed to form the image, in Hz.')  # type: float
    MaxProc = _FloatDescriptor(
        'MaxProc', _required, strict=DEFAULT_STRICT,
        docstring='The maximum transmit frequency processed to form the image, in Hz.')  # type: float


class ProcessingType(Serializable):
    """The transmit frequency range"""
    _fields = ('Type', 'Applied', 'Parameters')
    _required = ('Type', 'Applied')
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    Type = _StringDescriptor(
        'Type', _required, strict=DEFAULT_STRICT,
        docstring='The processing type identifier.')  # type: str
    Applied = _BooleanDescriptor(
        'Applied', _required, strict=DEFAULT_STRICT,
        docstring='Indicates whether the given processing type has been applied.')  # type: bool
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The parameters list.')  # type: List[ParameterType]


class DistortionType(Serializable):
    """Distortion"""
    _fields = (
        'CalibrationDate', 'A', 'F1', 'F2', 'Q1', 'Q2', 'Q3', 'Q4',
        'GainErrorA', 'GainErrorF1', 'GainErrorF2', 'PhaseErrorF1', 'PhaseErrorF2')
    _required = ('A', 'F1', 'Q1', 'Q2', 'F2', 'Q3', 'Q4')
    # descriptors
    CalibrationDate = _DateTimeDescriptor(
        'CalibrationDate', _required, strict=DEFAULT_STRICT,
        docstring='The calibration date.')
    A = _FloatDescriptor(
        'A', _required, strict=DEFAULT_STRICT,
        docstring='Absolute amplitude scale factor.')  # type: float
    # receive distorion matrix
    F1 = _ComplexDescriptor(
        'F1', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (2,2).')  # type: complex
    Q1 = _ComplexDescriptor(
        'Q1', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (1,2).')  # type: complex
    Q2 = _ComplexDescriptor(
        'Q2', _required, strict=DEFAULT_STRICT,
        docstring='Receive distortion element (2,1).')  # type: complex
    # transmit distortion matrix
    F2 = _ComplexDescriptor(
        'F2', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (2,2).')  # type: complex
    Q3 = _ComplexDescriptor(
        'Q3', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (2, 1).')  # type: complex
    Q4 = _ComplexDescriptor(
        'Q4', _required, strict=DEFAULT_STRICT,
        docstring='Transmit distortion element (1, 2).')  # type: complex
    # gain estimation error
    GainErrorA = _FloatDescriptor(
        'GainErrorA', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter A.')  # type: float
    GainErrorF1 = _FloatDescriptor(
        'GainErrorF1', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter F1.')  # type: float
    GainErrorF2 = _FloatDescriptor(
        'GainErrorF2', _required, strict=DEFAULT_STRICT,
        docstring='Gain estimation error standard deviation (in dB) for parameter F2.')  # type: float
    PhaseErrorF1 = _FloatDescriptor(
        'PhaseErrorF1', _required, strict=DEFAULT_STRICT,
        docstring='Phase estimation error standard deviation (in dB) for parameter F1.')  # type: float
    PhaseErrorF2 = _FloatDescriptor(
        'PhaseErrorF2', _required, strict=DEFAULT_STRICT,
        docstring='Phase estimation error standard deviation (in dB) for parameter F2.')  # type: float


class PolarizationCalibrationType(Serializable):
    """The polarization calibration"""
    _fields = ('DistortCorrectApplied', 'Distortion')
    _required = _fields
    # descriptors
    DistortCorrectApplied = _BooleanDescriptor(
        'DistortCorrectApplied', _required, strict=DEFAULT_STRICT,
        docstring='Indicates whether the polarization calibration has been applied.')  # type: bool
    Distortion = _SerializableDescriptor(
        'Distortion', DistortionType, _required, strict=DEFAULT_STRICT,
        docstring='The distortion parameters.')  # type: DistortionType


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
    # class variables
    _DUAL_POLARIZATION_VALUES = (
        'V:V', 'V:H', 'H:V', 'H:H', 'RHC:RHC', 'RHC:LHC', 'LHC:RHC', 'LHC:LHC', 'OTHER', 'UNKNOWN')
    _IMG_FORM_ALGO_VALUES = ('PFA', 'RMA', 'RGAZCOMP', 'OTHER')
    _ST_BEAM_COMP_VALUES = ('NO', 'GLOBAL', 'SV')
    _IMG_BEAM_COMP_VALUES = ('NO', 'SV')
    _AZ_AUTOFOCUS_VALUES = _ST_BEAM_COMP_VALUES
    _RG_AUTOFOCUS_VALUES = _ST_BEAM_COMP_VALUES
    # descriptors
    RcvChanProc = _SerializableDescriptor(
        'RcvChanProc', RcvChanProcType, _required, strict=DEFAULT_STRICT,
        docstring='The received processed channels.')  # type: RcvChanProcType
    TxRcvPolarizationProc = _StringEnumDescriptor(
        'TxRcvPolarizationProc', _DUAL_POLARIZATION_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='The combined transmit/receive polarization processed to form the image.')  # type: str
    TStartProc = _FloatDescriptor(
        'TStartProc', _required, strict=DEFAULT_STRICT,
        docstring='Earliest slow time value for data processed to form the image from CollectionStart.')  # type: float
    TEndProc = _FloatDescriptor(
        'TEndProc', _required, strict=DEFAULT_STRICT,
        docstring='Latest slow time value for data processed to form the image from CollectionStart.')  # type: float
    TxFrequencyProc = _SerializableDescriptor(
        'TxFrequencyProc', TxFrequencyProcType, _required, strict=DEFAULT_STRICT,
        docstring='The range of transmit frequency processed to form the image.')  # type: TxFrequencyProcType
    SegmentIdentifier = _StringDescriptor(
        'SegmentIdentifier', _required, strict=DEFAULT_STRICT,
        docstring='Identifier that describes the image that was processed. '
                  'Must be included when SICD.RadarCollection.Area.Plane.SegmentList is included.')  # type: str
    ImageFormAlgo = _StringEnumDescriptor(
        'ImageFormAlgo', _IMG_FORM_ALGO_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        The image formation algorithm used:

        * `PFA` - Polar Format Algorithm

        * `RMA` - Range Migration (Omega-K, Chirp Scaling, Range-Doppler)

        * `RGAZCOMP` - Simple range, Doppler compression.

        """)  # type: str
    STBeamComp = _StringEnumDescriptor(
        'STBeamComp', _ST_BEAM_COMP_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Indicates if slow time beam shape compensation has been applied.

        * `"NO"` - No ST beam shape compensation.

        * `"GLOBAL"` - Global ST beam shape compensation applied.

        * `"SV"` - Spatially variant beam shape compensation applied.

        """)  # type: str
    ImageBeamComp = _StringEnumDescriptor(
        'ImageBeamComp', _IMG_BEAM_COMP_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Indicates if image domain beam shape compensation has been applied.

        * `"NO"` - No image domain beam shape compensation.

        * `"SV"` - Spatially variant image domain beam shape compensation applied.

        """)  # type: str
    AzAutofocus = _StringEnumDescriptor(
        'AzAutofocus', _AZ_AUTOFOCUS_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates if azimuth autofocus correction has been applied, with similar '
                  'interpretation as `STBeamComp`.')  # type: str
    RgAutofocus = _StringEnumDescriptor(
        'RgAutofocus', _RG_AUTOFOCUS_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates if range autofocus correction has been applied, with similar '
                  'interpretation as `STBeamComp`.')  # type: str
    Processings = _SerializableArrayDescriptor(
        'Processings', ProcessingType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters to describe types of specific processing that may have been applied '
                  'such as additional compensations.')  # type: List[ProcessingType]
    PolarizationCalibration = _SerializableDescriptor(
        'PolarizationCalibration', PolarizationCalibrationType, _required, strict=DEFAULT_STRICT,
        docstring='The polarization calibration details.')  # type: PolarizationCalibrationType

    def _derive_tx_frequency_proc(self, RadarCollection):
        """
        Populate a default for processed frequency values, based on the assumption that the entire
        transmitted bandwidth was processed. This is expected to be called by SICD parent.

        Parameters
        ----------
        RadarCollection : sarpy.sicd_elements.RadarCollectionType

        Returns
        -------
        None
        """

        if RadarCollection is not None and RadarCollection.TxFrequency is not None and \
                RadarCollection.TxFrequency.Min is not None and RadarCollection.TxFrequency.Max is not None:
            # this is based on the assumption that the entire transmitted bandwidth was processed.
            if self.TxFrequencyProc is not None:
                self.TxFrequencyProc = TxFrequencyProcType(
                    ProcMin=RadarCollection.TxFrequency.Min, ProcMax=RadarCollection.TxFrequency.Max)
                # TODO: does it make sense to set only one end or the other?
            elif self.TxFrequencyProc.MinProc is None:
                self.TxFrequencyProc.MinProc = RadarCollection.TxFrequency.Min
            elif self.TxFrequencyProc.MaxProc is None:
                self.TxFrequencyProc.MaxProc = RadarCollection.TxFrequency.Max

    def derive(self):
        """
        Populates derived data in ImageFormationType. Expected to be called by SICD parent.

        Returns
        -------
        None
        """

        pass
