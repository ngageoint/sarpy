"""
The RadiometricType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import numpy

from sarpy.io.xml.base import Serializable, find_first_child
from sarpy.io.xml.descriptors import StringEnumDescriptor, SerializableDescriptor

from .base import DEFAULT_STRICT
from .blocks import Poly2DType


# noinspection PyPep8Naming
class NoiseLevelType_(Serializable):
    """
    Noise level structure.
    """

    _fields = ('NoiseLevelType', 'NoisePoly')
    _required = _fields
    # class variables
    _NOISE_LEVEL_TYPE_VALUES = ('ABSOLUTE', 'RELATIVE')
    # descriptors
    NoiseLevelType = StringEnumDescriptor(
        'NoiseLevelType', _NOISE_LEVEL_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring='Indicates that the noise power polynomial yields either absolute power level or power '
                  'level relative to the *SCP* pixel location.')  # type: str
    NoisePoly = SerializableDescriptor(
        'NoisePoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial coefficients that yield thermal noise power *(in dB)* in a pixel as a function of '
                  'image row coordinate *(variable 1)* and column coordinate *(variable 2)*.')  # type: Poly2DType

    def __init__(self, NoiseLevelType=None, NoisePoly=None, **kwargs):
        """

        Parameters
        ----------
        NoiseLevelType : str
        NoisePoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NoiseLevelType = NoiseLevelType
        self.NoisePoly = NoisePoly
        super(NoiseLevelType_, self).__init__(**kwargs)
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
    NoiseLevel = SerializableDescriptor(
        'NoiseLevel', NoiseLevelType_, _required, strict=DEFAULT_STRICT,
        docstring='Noise level structure.')  # type: NoiseLevelType_
    RCSSFPoly = SerializableDescriptor(
        'RCSSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to RCS *(m^2)* '
                  'as a function of image row coordinate *(variable 1)* and column coordinate *(variable 2)*. '
                  'Scale factor computed for a target at `HAE = SCP_HAE`.')  # type: Poly2DType
    SigmaZeroSFPoly = SerializableDescriptor(
        'SigmaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to clutter parameter '
                  'Sigma-Zero as a function of image row coordinate *(variable 1)* and column coordinate '
                  '*(variable 2)*. Scale factor computed for a clutter cell at `HAE = SCP_HAE`.')  # type: Poly2DType
    BetaZeroSFPoly = SerializableDescriptor(
        'BetaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to radar brightness '
                  'or Beta-Zero as a function of image row coordinate *(variable 1)* and column coordinate '
                  '*(variable 2)*. Scale factor computed for a clutter cell at `HAE = SCP_HAE`.')  # type: Poly2DType
    GammaZeroSFPoly = SerializableDescriptor(
        'GammaZeroSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial that yields a scale factor to convert pixel power to clutter parameter '
                  'Gamma-Zero as a function of image row coordinate *(variable 1)* and column coordinate '
                  '*(variable 2)*. Scale factor computed for a clutter cell at `HAE = SCP_HAE`.')  # type: Poly2DType

    def __init__(self, NoiseLevel=None, RCSSFPoly=None, SigmaZeroSFPoly=None,
                 BetaZeroSFPoly=None, GammaZeroSFPoly=None, **kwargs):
        """

        Parameters
        ----------
        NoiseLevel : NoiseLevelType_
        RCSSFPoly : Poly2DType|numpy.ndarray|list|tuple
        SigmaZeroSFPoly : Poly2DType|numpy.ndarray|list|tuple
        BetaZeroSFPoly : Poly2DType|numpy.ndarray|list|tuple
        GammaZeroSFPoly : Poly2DType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.NoiseLevel = NoiseLevel
        self.RCSSFPoly = RCSSFPoly
        self.SigmaZeroSFPoly = SigmaZeroSFPoly
        self.BetaZeroSFPoly = BetaZeroSFPoly
        self.GammaZeroSFPoly = GammaZeroSFPoly
        super(RadiometricType, self).__init__(**kwargs)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is not None:
            kwargs = {}
        # NoiseLevelType and NoisePoly used to be at this level prior to SICD 1.0.
        nkey = cls._child_xml_ns_key.get('NoiseLevelType', ns_key)
        nlevel = find_first_child(node, 'NoiseLevelType', xml_ns, nkey)
        if nlevel is not None:
            kwargs['NoiseLevel'] = NoiseLevelType_.from_node(nlevel, xml_ns, ns_key=ns_key, kwargs=kwargs)
        return super(RadiometricType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def _derive_parameters(self, Grid, SCPCOA):
        """
        Expected to be called by SICD parent.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType

        Returns
        -------
        None
        """

        if Grid is None or Grid.Row is None or Grid.Col is None:
            return

        area_sp = Grid.get_slant_plane_area()

        # We can define any SF polynomial from any other SF polynomial by just
        # scaling the coefficient array. If any are defined, use BetaZeroSFPolynomial
        # as the root, and derive them all
        if self.BetaZeroSFPoly is None:
            if self.RCSSFPoly is not None:
                self.BetaZeroSFPoly = Poly2DType(Coefs=self.RCSSFPoly.Coefs/area_sp)
            elif self.SigmaZeroSFPoly is not None:
                self.BetaZeroSFPoly = Poly2DType(
                    Coefs=self.SigmaZeroSFPoly.Coefs/numpy.cos(numpy.deg2rad(SCPCOA.SlopeAng)))
            elif self.GammaZeroSFPoly is not None:
                self.BetaZeroSFPoly = Poly2DType(
                    Coefs=self.GammaZeroSFPoly.Coefs*(numpy.sin(numpy.deg2rad(SCPCOA.GrazeAng)) /
                                                      numpy.cos(numpy.deg2rad(SCPCOA.SlopeAng))))

        if self.BetaZeroSFPoly is not None:
            # In other words, none of the SF polynomials are populated.
            if self.RCSSFPoly is None:
                self.RCSSFPoly = Poly2DType(Coefs=self.BetaZeroSFPoly.Coefs*area_sp)
            if self.SigmaZeroSFPoly is None:
                self.SigmaZeroSFPoly = Poly2DType(
                    Coefs=self.BetaZeroSFPoly.Coefs*numpy.cos(numpy.deg2rad(SCPCOA.SlopeAng)))
            if self.GammaZeroSFPoly is None:
                self.GammaZeroSFPoly = Poly2DType(
                    Coefs=self.BetaZeroSFPoly.Coefs*(numpy.cos(numpy.deg2rad(SCPCOA.SlopeAng)) /
                                                     numpy.sin(numpy.deg2rad(SCPCOA.GrazeAng))))
