"""
The RgAzCompType definition.
"""

from ._base import Serializable, DEFAULT_STRICT, _FloatDescriptor, _SerializableDescriptor
from ._blocks import Poly1DType


__classification__ = "UNCLASSIFIED"


class RgAzCompType(Serializable):
    """Parameters included for a Range, Doppler image."""
    _fields = ('AzSF', 'KazPoly')
    _required = _fields
    # descriptors
    AzSF = _FloatDescriptor(
        'AzSF', _required, strict=DEFAULT_STRICT,
        docstring='Scale factor that scales image coordinate az = ycol (meters) to a delta cosine of the '
                  'Doppler Cone Angle at COA, (in 1/meter)')  # type: float
    KazPoly = _SerializableDescriptor(
        'KazPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields azimuth spatial frequency (Kaz = Kcol) as a function of '
                  'slow time (variable 1). Slow Time (sec) -> Azimuth spatial frequency (cycles/meter). '
                  'Time relative to collection start.')  # type: Poly1DType
