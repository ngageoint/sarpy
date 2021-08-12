"""
The RMAType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from typing import Union

import numpy
from numpy.linalg import norm

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import StringEnumDescriptor, FloatDescriptor, \
    BooleanDescriptor, SerializableDescriptor

from .base import DEFAULT_STRICT
from .blocks import XYZType, Poly1DType, Poly2DType
from .utils import _get_center_frequency


class RMRefType(Serializable):
    """
    Range migration reference element of RMA type.
    """

    _fields = ('PosRef', 'VelRef', 'DopConeAngRef')
    _required = _fields
    _numeric_format = {'DopConeAngRef': '0.16G', }
    # descriptors
    PosRef = SerializableDescriptor(
        'PosRef', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Platform reference position in ECF coordinates used to establish '
                  'the reference slant plane.')  # type: XYZType
    VelRef = SerializableDescriptor(
        'VelRef', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Platform reference velocity vector in ECF coordinates used to establish '
                  'the reference slant plane.')  # type: XYZType
    DopConeAngRef = FloatDescriptor(
        'DopConeAngRef', _required, strict=DEFAULT_STRICT,
        docstring='Reference Doppler Cone Angle in degrees.')  # type: float

    def __init__(self, PosRef=None, VelRef=None, DopConeAngRef=None, **kwargs):
        """

        Parameters
        ----------
        PosRef : XYZType|numpy.ndarray|list|tuple
        VelRef : XYZType|numpy.ndarray|list|tuple
        DopConeAngRef : float
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.PosRef = PosRef
        self.VelRef = VelRef
        self.DopConeAngRef = DopConeAngRef
        super(RMRefType, self).__init__(**kwargs)


class INCAType(Serializable):
    """Parameters for Imaging Near Closest Approach (INCA) image description."""
    _fields = (
        'TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly', 'DopCentroidPoly', 'DopCentroidCOA')
    _required = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly')
    _numeric_format = {'R_CA_SCP': '0.16G', 'FreqZero': '0.16G'}
    # descriptors
    TimeCAPoly = SerializableDescriptor(
        'TimeCAPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields *Time of Closest Approach* as function of '
                  'image column *(azimuth)* coordinate in meters. Time relative to '
                  'collection start in seconds.')  # type: Poly1DType
    R_CA_SCP = FloatDescriptor(
        'R_CA_SCP', _required, strict=DEFAULT_STRICT,
        docstring='*Range at Closest Approach (R_CA)* for the *Scene Center Point (SCP)* in meters.')  # type: float
    FreqZero = FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring=r'*RF frequency* :\math:`(f_0)` in Hz used for computing '
                  r'Doppler Centroid values. Typical :math:`f_0` '
                  r'set equal o center transmit frequency.')  # type: float
    DRateSFPoly = SerializableDescriptor(
        'DRateSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields *Doppler Rate scale factor (DRSF)* '
                  'as a function of image location. Yields `DRSF` as a function of image '
                  'range coordinate ``(variable 1)`` and azimuth coordinate ``(variable 2)``. '
                  'Used to compute Doppler Rate at closest approach.')  # type: Poly2DType
    DopCentroidPoly = SerializableDescriptor(
        'DopCentroidPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Doppler Centroid value as a '
                  'function of image location *(fdop_DC)*. The *fdop_DC* is the '
                  'Doppler frequency at the peak signal response. The polynomial is a function '
                  'of image range coordinate ``(variable 1)`` and azimuth coordinate ``(variable 2)``. '
                  '*Note: Only used for Stripmap and Dynamic Stripmap collections.*')  # type: Poly2DType
    DopCentroidCOA = BooleanDescriptor(
        'DopCentroidCOA', _required, strict=DEFAULT_STRICT,
        docstring="""Flag indicating that the COA is at the peak signal :math`fdop_COA = fdop_DC`.
        
        * `True` - if Pixel COA at peak signal for all pixels.
        
        * `False` otherwise.
        
        *Note:* Only used for Stripmap and Dynamic Stripmap.""")  # type: bool

    def __init__(self, TimeCAPoly=None, R_CA_SCP=None, FreqZero=None, DRateSFPoly=None,
                 DopCentroidPoly=None, DopCentroidCOA=None, **kwargs):
        """

        Parameters
        ----------
        TimeCAPoly : Poly1DType|numpy.ndarray|list|tuple
        R_CA_SCP : float
        FreqZero : float
        DRateSFPoly : Poly2DType|numpy.ndarray|list|tuple
        DopCentroidPoly : Poly2DType|numpy.ndarray|list|tuple
        DopCentroidCOA : bool
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TimeCAPoly = TimeCAPoly
        self.R_CA_SCP = R_CA_SCP
        self.FreqZero = FreqZero
        self.DRateSFPoly = DRateSFPoly
        self.DopCentroidPoly = DopCentroidPoly
        self.DopCentroidCOA = DopCentroidCOA
        super(INCAType, self).__init__(**kwargs)

    def _apply_reference_frequency(self, reference_frequency):
        if self.FreqZero is not None:
            self.FreqZero += reference_frequency


class RMAType(Serializable):
    """Parameters included when the image is formed using the Range Migration Algorithm."""
    _fields = ('RMAlgoType', 'ImageType', 'RMAT', 'RMCR', 'INCA')
    _required = ('RMAlgoType', 'ImageType')
    _choice = ({'required': True, 'collection': ('RMAT', 'RMCR', 'INCA')}, )
    # class variables
    _RM_ALGO_TYPE_VALUES = ('OMEGA_K', 'CSA', 'RG_DOP')
    # descriptors
    RMAlgoType = StringEnumDescriptor(
        'RMAlgoType', _RM_ALGO_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring=r"""
        Identifies the type of migration algorithm used:

        * `OMEGA_K` - Algorithms that employ Stolt interpolation of the Kxt dimension. :math:`Kx = \sqrt{Kf^2 - Ky^2}`

        * `CSA` - Wave number algorithm that process two-dimensional chirp signals.

        * `RG_DOP` - Range-Doppler algorithms that employ *RCMC* in the compressed range domain.

        """)  # type: str
    RMAT = SerializableDescriptor(
        'RMAT', RMRefType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for *RMA with Along Track (RMAT)* motion compensation.')  # type: RMRefType
    RMCR = SerializableDescriptor(
        'RMCR', RMRefType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for *RMA with Cross Range (RMCR)* motion compensation.')  # type: RMRefType
    INCA = SerializableDescriptor(
        'INCA', INCAType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for *Imaging Near Closest Approach (INCA)* image description.')  # type: INCAType

    def __init__(self, RMAlgoType=None, RMAT=None, RMCR=None, INCA=None, **kwargs):
        """

        Parameters
        ----------
        RMAlgoType : str
        RMAT : RMRefType
        RMCR : RMRefType
        INCA : INCAType
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.RMAlgoType = RMAlgoType
        self.RMAT = RMAT
        self.RMCR = RMCR
        self.INCA = INCA
        super(RMAType, self).__init__(**kwargs)

    @property
    def ImageType(self):  # type: () -> Union[None, str]
        """
        str: READ ONLY attribute. Identifies the specific RM image type / metadata type supplied. This is determined by
        returning the (first) attribute among `'RMAT', 'RMCR', 'INCA'` which is populated. `None` will be returned if
        none of them are populated.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return None

    def _derive_parameters(self, SCPCOA, Position, RadarCollection, ImageFormation):
        """
        Expected to be called from SICD parent.

        Parameters
        ----------
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
        Position : sarpy.io.complex.sicd_elements.Position.PositionType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

        Returns
        -------
        None
        """

        if SCPCOA is None:
            return

        scp = None if SCPCOA.ARPPos is None else SCPCOA.ARPPos.get_array()

        im_type = self.ImageType
        if im_type in ['RMAT', 'RMCR']:
            rm_ref = getattr(self, im_type)  # type: RMRefType
            if rm_ref.PosRef is None and SCPCOA.ARPPos is not None:
                rm_ref.PosRef = SCPCOA.ARPPos.copy()
            if rm_ref.VelRef is None and SCPCOA.ARPVel is not None:
                rm_ref.VelRef = SCPCOA.ARPVel.copy()
            if scp is not None and rm_ref.PosRef is not None and rm_ref.VelRef is not None:
                pos_ref = rm_ref.PosRef.get_array()
                vel_ref = rm_ref.VelRef.get_array()
                uvel_ref = vel_ref/norm(vel_ref)
                ulos = (scp - pos_ref)  # it absolutely could be that scp = pos_ref
                ulos_norm = norm(ulos)
                if ulos_norm > 0:
                    ulos /= ulos_norm
                    if rm_ref.DopConeAngRef is None:
                        rm_ref.DopConeAngRef = numpy.rad2deg(numpy.arccos(numpy.dot(uvel_ref, ulos)))
        elif im_type == 'INCA':
            if scp is not None and self.INCA.TimeCAPoly is not None and \
                    Position is not None and Position.ARPPoly is not None:
                t_zero = self.INCA.TimeCAPoly.Coefs[0]
                ca_pos = Position.ARPPoly(t_zero)
                if self.INCA.R_CA_SCP is None:
                    self.INCA.R_CA_SCP = norm(ca_pos - scp)
            if self.INCA.FreqZero is None:
                self.INCA.FreqZero = _get_center_frequency(RadarCollection, ImageFormation)

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

        if self.INCA is not None:
            # noinspection PyProtectedMember
            self.INCA._apply_reference_frequency(reference_frequency)
