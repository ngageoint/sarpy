"""
Multipurpose basic SIDD elements
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Union, Optional

import numpy

from sarpy.io.xml.base import Serializable, Arrayable
from sarpy.io.xml.descriptors import FloatDescriptor, StringEnumDescriptor

from .base import DEFAULT_STRICT, FLOAT_FORMAT
import sarpy.io.complex.sicd_elements.Radiometric as SicdRadiometric

# Reuse SIDD 2.0 types and make them available here
from sarpy.io.product.sidd2_elements.blocks import (
    ErrorStatisticsType,
    FilterType,
    FilterBankType,
    MatchInfoType,
    NewLookupTableType,
    PredefinedFilterType,
    PredefinedLookupType,
    Poly2DType,
    RadarModeType,
    RangeAzimuthType,
    ReferencePointType,
    RowColDoubleType,
    XYZPolyType,
    XYZType,
)

__REUSED__ = (  # to avoid unused imports
    ErrorStatisticsType,
    FilterType,
    FilterBankType,
    MatchInfoType,
    NewLookupTableType,
    PredefinedFilterType,
    PredefinedLookupType,
    Poly2DType,
    RadarModeType,
    RangeAzimuthType,
    ReferencePointType,
    RowColDoubleType,
    XYZPolyType,
    XYZType,
)

_len2_array_text = 'Expected array to be of length 2,\n\t' \
                   'and received `{}`'
_array_type_text = 'Expected array to be numpy.ndarray, list, or tuple,\n\t' \
                   'got `{}`'


############
# the SICommon namespace elements

class AngleZeroToExclusive360MagnitudeType(Serializable, Arrayable):
    """
    Represents a magnitude and angle.
    """

    _fields = ('Angle', 'Magnitude')
    _required = ('Angle', 'Magnitude')
    _numeric_format = {key: FLOAT_FORMAT for key in _fields}
    _child_xml_ns_key = {'Angle': 'sicommon', 'Magnitude': 'sicommon'}
    # Descriptor
    Angle = FloatDescriptor(
        'Angle', _required, strict=DEFAULT_STRICT, bounds=(0.0, 360),
        docstring='The angle.')  # type: float
    Magnitude = FloatDescriptor(
        'Magnitude', _required, strict=DEFAULT_STRICT, bounds=(0.0, None),
        docstring='The magnitude.')  # type: float

    def __init__(self, Angle=None, Magnitude=None, **kwargs):
        """

        Parameters
        ----------
        Angle : float
        Magnitude : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Angle = Angle
        self.Magnitude = Magnitude
        super(AngleZeroToExclusive360MagnitudeType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [Angle, Magnitude]
        """

        return numpy.array([self.Angle, self.Magnitude], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Angle, Magnitude]

        Returns
        -------
        AngleZeroToExclusive360MagnitudeType
        """

        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError(_len2_array_text.format(array))
            return cls(Angle=array[0], Magnitude=array[1])
        raise ValueError(_array_type_text.format(type(array)))


# SIDD 3.0 Radiometric is the same as SICD, but with an added field
class RadiometricType(SicdRadiometric.RadiometricType):
    _fields = ('NoiseLevel', 'RCSSFPoly', 'SigmaZeroSFPoly', 'BetaZeroSFPoly', 'SigmaZeroSFIncidenceMap',
               'GammaZeroSFPoly')
    _child_xml_ns_key = {
        'NoiseLevel': 'sicommon', 'RCSSFPoly': 'sicommon', 'SigmaZeroSFPoly': 'sicommon',
        'BetaZeroSFPoly': 'sicommon', 'SigmaZeroSFIncidenceMap': 'sicommon', 'GammaZeroSFPoly': 'sicommon'}

    _SIGMA_ZERO_SF_INCIDENCE_MAP_VALUES = ('APPLIED', 'NOT_APPLIED')
    # descriptors
    SigmaZeroSFIncidenceMap = StringEnumDescriptor(
        'SigmaZeroSFIncidenceMap', _SIGMA_ZERO_SF_INCIDENCE_MAP_VALUES,
        SicdRadiometric.RadiometricType._required, strict=DEFAULT_STRICT,
        docstring='Allowed Values: “APPLIED” or “NOT_APPLIED”')  # type: str

    def __init__(
            self,
            NoiseLevel: Optional[SicdRadiometric.NoiseLevelType_] = None,
            RCSSFPoly: Union[None, Poly2DType, numpy.ndarray, list, tuple] = None,
            SigmaZeroSFPoly: Union[None, Poly2DType, numpy.ndarray, list, tuple] = None,
            BetaZeroSFPoly: Union[None, Poly2DType, numpy.ndarray, list, tuple] = None,
            GammaZeroSFPoly: Union[None, Poly2DType, numpy.ndarray, list, tuple] = None,
            SigmaZeroSFIncidenceMap: str = None,
            **kwargs):
        """

        Parameters
        ----------
        NoiseLevel : NoiseLevelType_
        RCSSFPoly : Poly2DType|numpy.ndarray|list|tuple
        SigmaZeroSFPoly : Poly2DType|numpy.ndarray|list|tuple
        BetaZeroSFPoly : Poly2DType|numpy.ndarray|list|tuple
        GammaZeroSFPoly : Poly2DType|numpy.ndarray|list|tuple
        SigmaZeroSFIncidenceMap : str
        kwargs
        """
        self.SigmaZeroSFIncidenceMap = SigmaZeroSFIncidenceMap
        super().__init__(NoiseLevel=NoiseLevel,
                         RCSSFPoly=RCSSFPoly,
                         SigmaZeroSFPoly=SigmaZeroSFPoly,
                         BetaZeroSFPoly=BetaZeroSFPoly,
                         GammaZeroSFPoly=GammaZeroSFPoly,
                         **kwargs)


# The end of the SICommon namespace
#####################
