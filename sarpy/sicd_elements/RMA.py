"""
The RMAType definition.
"""

from typing import Union

from ._base import Serializable, DEFAULT_STRICT, \
    _StringEnumDescriptor, _FloatDescriptor, _BooleanDescriptor, \
    _SerializableDescriptor
from ._blocks import XYZType, Poly1DType, Poly2DType


__classification__ = "UNCLASSIFIED"


class RMRefType(Serializable):
    """Range migration reference element of RMA type."""
    _fields = ('PosRef', 'VelRef', 'DopConeAngRef')
    _required = _fields
    # descriptors
    PosRef = _SerializableDescriptor(
        'PosRef', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Platform reference position (ECF) used to establish the reference slant plane.')  # type: XYZType
    VelRef = _SerializableDescriptor(
        'VelRef', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Platform reference velocity vector (ECF) used to establish the reference '
                  'slant plane.')  # type: XYZType
    DopConeAngRef = _FloatDescriptor(
        'DopConeAngRef', _required, strict=DEFAULT_STRICT,
        docstring='Reference Doppler Cone Angle in degrees.')  # type: float


class INCAType(Serializable):
    """Parameters for Imaging Near Closest Approach (INCA) image description."""
    _fields = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly', 'DopCentroidPoly', 'DopCentroidCOA')
    _required = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly')
    # descriptors
    TimeCAPoly = _SerializableDescriptor(
        'TimeCAPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Time of Closest Approach as function of '
                  'image column (azimuth) coordinate (m). Time (in seconds) relative to '
                  'collection start.')  # type: Poly1DType
    R_CA_SCP = _FloatDescriptor(
        'R_CA_SCP', _required, strict=DEFAULT_STRICT,
        docstring='Range at Closest Approach (R_CA) for the scene center point (SCP) in meters.')  # type: float
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring='RF frequency (f0) in Hz used for computing Doppler Centroid values. Typical f0 set equal '
                  'to center transmit frequency.')  # type: float
    DRateSFPoly = _SerializableDescriptor(
        'DRateSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields *Doppler Rate scale factor (DRSF)* as a function of image '
                  'location. Yields `DRSF` as a function of image range coordinate (variable 1) and azimuth '
                  'coordinate (variable 2). Used to compute Doppler Rate at closest approach.')  # type: Poly2DType
    DopCentroidPoly = _SerializableDescriptor(
        'DopCentroidPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Doppler Centroid value as a function of image location (fdop_DC). '
                  'The fdop_DC is the Doppler frequency at the peak signal response. The polynomial is a function '
                  'of image range coordinate (variable 1) and azimuth coordinate (variable 2). '
                  '*Only used for Stripmap and Dynamic Stripmap collections.*')  # type: Poly2DType
    DopCentroidCOA = _BooleanDescriptor(
        'DopCentroidCOA', _required, strict=DEFAULT_STRICT,
        docstring="""Flag indicating that the COA is at the peak signal (fdop_COA = fdop_DC). `True` if Pixel COA at 
        peak signal for all pixels. `False` otherwise. *Only used for Stripmap and Dynamic Stripmap.*""")  # type: bool


class RMAType(Serializable):
    """Parameters included when the image is formed using the Range Migration Algorithm."""
    _fields = ('RMAlgoType', 'ImageType', 'RMAT', 'RMCR', 'INCA')
    _required = ('RMAlgoType', 'ImageType')
    _choice = ({'required': True, 'collection': ('RMAT', 'RMCR', 'INCA')}, )
    # class variables
    _RM_ALGO_TYPE_VALUES = ('OMEGA_K', 'CSA', 'RG_DOP')
    # descriptors
    RMAlgoType = _StringEnumDescriptor(
        'RMAlgoType', _RM_ALGO_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring="""
        Identifies the type of migration algorithm used:

        * `OMEGA_K` - Algorithms that employ Stolt interpolation of the Kxt dimension. `Kx = (Kf^2 – Ky^2)^0.5`

        * `CSA` - Wave number algorithm that process two-dimensional chirp signals.

        * `RG_DOP` - Range-Doppler algorithms that employ RCMC in the compressed range domain.

        """)  # type: str
    RMAT = _SerializableDescriptor(
        'RMAT', RMRefType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for RMA with Along Track (RMAT) motion compensation.')  # type: RMRefType
    RMCR = _SerializableDescriptor(
        'RMCR', RMRefType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for RMA with Cross Range (RMCR) motion compensation.')  # type: RMRefType
    INCA = _SerializableDescriptor(
        'INCA', INCAType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for Imaging Near Closest Approach (INCA) image description.')  # type: INCAType

    @property
    def ImageType(self):  # type: () -> Union[None, str]
        """
        str: READ ONLY attribute. Identifies the specific RM image type / metadata type supplied. This is determined by
        returning the (first) attribute among `RMAT`, `RMCR`, `INCA` which is populated. None will be returned if
        none of them are populated.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return None
