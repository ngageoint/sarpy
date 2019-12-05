"""
The RadiometricType definition.
"""

from .base import Serializable, DEFAULT_STRICT, _StringEnumDescriptor, _SerializableDescriptor
from .blocks import Poly2DType


__classification__ = "UNCLASSIFIED"


class NoiseLevelType(Serializable):
    """Noise level structure."""
    _fields = ('NoiseLevelType', 'NoisePoly')
    _required = _fields
    # class variables
    _NOISE_LEVEL_TYPE_VALUES = ('ABSOLUTE', 'RELATIVE')
    # descriptors
    NoiseLevelType = _StringEnumDescriptor(
        'NoiseLevelType', _NOISE_LEVEL_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates that the noise power polynomial yields either absolute power level or power '
                  'level relative to the SCP pixel location.')  # type: str
    NoisePoly = _SerializableDescriptor(
        'NoisePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial coefficients that yield thermal noise power (in dB) in a pixel as a function of '
                  'image row coordinate (variable 1) and column coordinate (variable 2).')  # type: Poly2DType

    def __init__(self, **kwargs):
        super(NoiseLevelType, self).__init__(**kwargs)
        self._derive_noise_level()

    def _derive_noise_level(self):
        if self.NoiseLevelType is not None:
            return
        if self.NoisePoly is None:
            return  # nothing to be done

        scp_val = self.NoisePoly.Coefs[0, 0]  # the value at SCP
        if scp_val == 1:
            # the relative noise levels should be 1 at SCP
            self.NoiseLevelType = 'RELATIVE'
        else:
            # it seems safe that it's not absolute, in this case?
            self.NoiseLevelType = 'ABSOLUTE'


class RadiometricType(Serializable):
    """The radiometric calibration parameters."""
    _fields = ('NoiseLevel', 'RCSSFPoly', 'SigmaZeroSFPoly', 'BetaZeroSFPoly', 'GammaZeroSFPoly')
    _required = ()
    # descriptors
    NoiseLevel = _SerializableDescriptor(
        'NoiseLevel', NoiseLevelType, _required, strict=DEFAULT_STRICT,
        docstring='Noise level structure.')  # type: NoiseLevelType
    RCSSFPoly = _SerializableDescriptor(
        'RCSSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to RCS (sqm) '
                  'as a function of image row coordinate (variable 1) and column coordinate (variable 2). '
                  'Scale factor computed for a target at `HAE = SCP_HAE`.')  # type: Poly2DType
    SigmaZeroSFPoly = _SerializableDescriptor(
        'SigmaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to clutter parameter '
                  'Sigma-Zero as a function of image row coordinate (variable 1) and column coordinate (variable 2). '
                  'Scale factor computed for a clutter cell at `HAE = SCP_HAE`.')  # type: Poly2DType
    BetaZeroSFPoly = _SerializableDescriptor(
        'BetaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to radar brightness '
                  'or Beta-Zero as a function of image row coordinate (variable 1) and column coordinate (variable 2). '
                  'Scale factor computed for a clutter cell at `HAE = SCP_HAE`.')  # type: Poly2DType
    GammaZeroSFPoly = _SerializableDescriptor(
        'GammaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to clutter parameter '
                  'Gamma-Zero as a function of image row coordinate (variable 1) and column coordinate (variable 2). '
                  'Scale factor computed for a clutter cell at `HAE = SCP_HAE`.')  # type: Poly2DType

    @classmethod
    def from_node(cls, node, kwargs=None):
        if kwargs is not None:
            kwargs = {}
        # NoiseLevelType and NoisePoly used to be at this level prior to SICD 1.0.
        if node.find('NoiseLevelType') is not None:
            kwargs['NoiseLevel'] = NoiseLevelType.from_node(node, kwargs=kwargs)
        return super(RadiometricType, cls).from_node(node, kwargs=kwargs)
