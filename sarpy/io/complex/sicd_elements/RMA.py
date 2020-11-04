# -*- coding: utf-8 -*-
"""
The RMAType definition.
"""

import logging
from typing import Union

import numpy
from numpy.linalg import norm
from scipy.constants import speed_of_light

from .base import Serializable, DEFAULT_STRICT, \
    _StringEnumDescriptor, _FloatDescriptor, _BooleanDescriptor, _SerializableDescriptor
from .blocks import XYZType, Poly1DType, Poly2DType
from .utils import _get_center_frequency

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class RMRefType(Serializable):
    """
    Range migration reference element of RMA type.
    """

    _fields = ('PosRef', 'VelRef', 'DopConeAngRef')
    _required = _fields
    _numeric_format = {'DopConeAngRef': '0.16G', }
    # descriptors
    PosRef = _SerializableDescriptor(
        'PosRef', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Platform reference position in ECF coordinates used to establish '
                  'the reference slant plane.')  # type: XYZType
    VelRef = _SerializableDescriptor(
        'VelRef', XYZType, _required, strict=DEFAULT_STRICT,
        docstring='Platform reference velocity vector in ECF coordinates used to establish '
                  'the reference slant plane.')  # type: XYZType
    DopConeAngRef = _FloatDescriptor(
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
    _fields = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly', 'DopCentroidPoly', 'DopCentroidCOA')
    _required = ('TimeCAPoly', 'R_CA_SCP', 'FreqZero', 'DRateSFPoly')
    _numeric_format = {'R_CA_SCP': '0.16G', 'FreqZero': '0.16G'}
    # descriptors
    TimeCAPoly = _SerializableDescriptor(
        'TimeCAPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields *Time of Closest Approach* as function of '
                  'image column *(azimuth)* coordinate in meters. Time relative to '
                  'collection start in seconds.')  # type: Poly1DType
    R_CA_SCP = _FloatDescriptor(
        'R_CA_SCP', _required, strict=DEFAULT_STRICT,
        docstring='*Range at Closest Approach (R_CA)* for the *Scene Center Point (SCP)* in meters.')  # type: float
    FreqZero = _FloatDescriptor(
        'FreqZero', _required, strict=DEFAULT_STRICT,
        docstring=r'*RF frequency* :\math:`(f_0)` in Hz used for computing Doppler Centroid values. Typical :math:`f_0` '
                  r'set equal o center transmit frequency.')  # type: float
    DRateSFPoly = _SerializableDescriptor(
        'DRateSFPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields *Doppler Rate scale factor (DRSF)* as a function of image '
                  'location. Yields `DRSF` as a function of image range coordinate ``(variable 1)`` and azimuth '
                  'coordinate ``(variable 2)``. Used to compute Doppler Rate at closest approach.')  # type: Poly2DType
    DopCentroidPoly = _SerializableDescriptor(
        'DopCentroidPoly', Poly2DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields Doppler Centroid value as a function of image location *(fdop_DC)*. '
                  'The *fdop_DC* is the Doppler frequency at the peak signal response. The polynomial is a function '
                  'of image range coordinate ``(variable 1)`` and azimuth coordinate ``(variable 2)``. '
                  '*Note: Only used for Stripmap and Dynamic Stripmap collections.*')  # type: Poly2DType
    DopCentroidCOA = _BooleanDescriptor(
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
    RMAlgoType = _StringEnumDescriptor(
        'RMAlgoType', _RM_ALGO_TYPE_VALUES, _required, strict=DEFAULT_STRICT,
        docstring=r"""
        Identifies the type of migration algorithm used:

        * `OMEGA_K` - Algorithms that employ Stolt interpolation of the Kxt dimension. :math:`Kx = \sqrt{Kf^2 - Ky^2}`

        * `CSA` - Wave number algorithm that process two-dimensional chirp signals.

        * `RG_DOP` - Range-Doppler algorithms that employ *RCMC* in the compressed range domain.

        """)  # type: str
    RMAT = _SerializableDescriptor(
        'RMAT', RMRefType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for *RMA with Along Track (RMAT)* motion compensation.')  # type: RMRefType
    RMCR = _SerializableDescriptor(
        'RMCR', RMRefType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters for *RMA with Cross Range (RMCR)* motion compensation.')  # type: RMRefType
    INCA = _SerializableDescriptor(
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

        SCP = None if SCPCOA.ARPPos is None else SCPCOA.ARPPos.get_array()

        im_type = self.ImageType
        if im_type in ['RMAT', 'RMCR']:
            RmRef = getattr(self, im_type)  # type: RMRefType
            if RmRef.PosRef is None and SCPCOA.ARPPos is not None:
                RmRef.PosRef = SCPCOA.ARPPos.copy()
            if RmRef.VelRef is None and SCPCOA.ARPVel is not None:
                RmRef.VelRef = SCPCOA.ARPVel.copy()
            if SCP is not None and RmRef.PosRef is not None and RmRef.VelRef is not None:
                pos_ref = RmRef.PosRef.get_array()
                vel_ref = RmRef.VelRef.get_array()
                uvel_ref = vel_ref/norm(vel_ref)
                uLOS = (SCP - pos_ref)  # it absolutely could be that SCP = pos_ref
                uLos_norm = norm(uLOS)
                if uLos_norm > 0:
                    uLOS /= uLos_norm
                    if RmRef.DopConeAngRef is None:
                        RmRef.DopConeAngRef = numpy.rad2deg(numpy.arccos(numpy.dot(uvel_ref, uLOS)))
        elif im_type == 'INCA':
            if SCP is not None and self.INCA.TimeCAPoly is not None and \
                    Position is not None and Position.ARPPoly is not None:
                t_zero = self.INCA.TimeCAPoly.Coefs[0]
                ca_pos = Position.ARPPoly(t_zero)
                if self.INCA.R_CA_SCP is None:
                    self.INCA.R_CA_SCP = norm(ca_pos - SCP)
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

    def _check_rmat(self, Grid, GeoData, RadarCollection, ImageFormation):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

        Returns
        -------
        bool
        """

        cond = True

        if Grid.Type != 'XCTYAT':
            logging.error(
                'The image formation algorithm is RMA/RMAT, which should yield '
                'Grid.Type == "XCTYAT", but Grid.Type is populated as "{}"'.format(Grid.Type))
            cond = False

        try:
            SCP = GeoData.SCP.ECF.get_array(dtype='float64')
            row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
            col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')
            position_ref = self.RMAT.PosRef.get_array(dtype='float64')
            velocity_ref = self.RMAT.VelRef.get_array(dtype='float64')
            LOS = (SCP - position_ref)
            uLOS = LOS/numpy.linalg.norm(LOS)
            left = numpy.cross(
                position_ref/numpy.linalg.norm(position_ref),
                velocity_ref/numpy.linalg.norm(velocity_ref))
            look = numpy.sign(left.dot(uLOS))
            uYAT = -look*velocity_ref/numpy.linalg.norm(velocity_ref)
            uSPN = numpy.cross(uLOS, uYAT)
            uSPN /= numpy.linalg.norm(uSPN)
            uXCT = numpy.cross(uYAT, uSPN)
        except (AttributeError, ValueError, TypeError):
            return cond

        if numpy.linalg.norm(row_uvect - uXCT) > 1e-3:
            logging.error(
                'The image formation algorithm is RMA/RMAT, and Row.UVectECF is '
                'populated as {}, but expected to be {}'.format(row_uvect, uXCT))
            cond = False
        if numpy.linalg.norm(col_uvect - uYAT) > 1e-3:
            logging.error(
                'The image formation algorithm is RMA/RMAT, and Col.UVectECF is '
                'populated as {}, but expected to be {}'.format(col_uvect, uYAT))
            cond = False
        exp_doppler_cone = numpy.rad2deg(numpy.arccos(uLOS.dot(velocity_ref/numpy.linalg.norm(velocity_ref))))
        if abs(exp_doppler_cone - self.RMAT.DopConeAngRef) > 1e-6:
            logging.error(
                'The image formation algorithm is RMA/RMAT, and RMAT.DopConeAngRef is '
                'populated as {}, but expected to be {}'.format(self.RMAT.DopConeAngRef, exp_doppler_cone))
            cond = False

        if RadarCollection.RefFreqIndex is None:
            center_freq = ImageFormation.TxFrequencyProc.center_frequency
            k_f_c = center_freq*2/speed_of_light
            exp_row_kctr = k_f_c*numpy.sin(numpy.deg2rad(self.RMAT.DopConeAngRef))
            exp_col_kctr = k_f_c*numpy.cos(numpy.deg2rad(self.RMAT.DopConeAngRef))
            try:
                if abs(exp_row_kctr/Grid.Row.KCtr - 1) > 1e-3:
                    logging.warning(
                        'The image formation algorithm is RMA/RMAT, the Row.KCtr is populated as {}, '
                        'and the expected value is {}'.format(Grid.Row.KCtr, exp_row_kctr))
                    cond = False
                if abs(exp_col_kctr/Grid.Col.KCtr - 1) > 1e-3:
                    logging.warning(
                        'The image formation algorithm is RMA/RMAT, the Col.KCtr is populated as {}, '
                        'and the expected value is {}'.format(Grid.Col.KCtr, exp_col_kctr))
                    cond = False
            except (AttributeError, ValueError, TypeError):
                pass
        return cond

    def _check_rmcr(self, Grid, GeoData, RadarCollection, ImageFormation):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

        Returns
        -------
        bool
        """

        cond = True

        if Grid.Type != 'XRGYCR':
            logging.error(
                'The image formation algorithm is RMA/RMCR, which should yield '
                'Grid.Type == "XRGYCR", but Grid.Type is populated as "{}"'.format(Grid.Type))
            cond = False

        try:
            SCP = GeoData.SCP.ECF.get_array(dtype='float64')
            row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
            col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')
            position_ref = self.RMCR.PosRef.get_array(dtype='float64')
            velocity_ref = self.RMCR.VelRef.get_array(dtype='float64')
            uXRG = SCP - position_ref
            uXRG /= numpy.linalg.norm(uXRG)
            left = numpy.cross(
                position_ref/numpy.linalg.norm(position_ref),
                velocity_ref/numpy.linalg.norm(velocity_ref))
            look = numpy.sign(left.dot(uXRG))
            uSPN = look*numpy.cross(velocity_ref/numpy.linalg.norm(velocity_ref), uXRG)
            uSPN /= numpy.linalg.norm(uSPN)
            uYCR = numpy.cross(uSPN, uXRG)
        except (AttributeError, ValueError, TypeError):
            return cond

        if numpy.linalg.norm(row_uvect - uXRG) > 1e-3:
            logging.error(
                'The image formation algorithm is RMA/RMCR, and Row.UVectECF is '
                'populated as {}, but expected to be {}'.format(row_uvect, uXRG))
            cond = False
        if numpy.linalg.norm(col_uvect - uYCR) > 1e-3:
            logging.error(
                'The image formation algorithm is RMA/RMCR, and Col.UVectECF is '
                'populated as {}, but expected to be {}'.format(col_uvect, uYCR))
            cond = False
        exp_doppler_cone = numpy.rad2deg(numpy.arccos(uXRG.dot(velocity_ref/numpy.linalg.norm(velocity_ref))))
        if abs(exp_doppler_cone - self.RMCR.DopConeAngRef) > 1e-6:
            logging.error(
                'The image formation algorithm is RMA/RMCR, and RMCR.DopConeAngRef is '
                'populated as {}, but expected to be {}'.format(self.RMCR.DopConeAngRef, exp_doppler_cone))
            cond = False
        if abs(Grid.Col.KCtr) > 1e-6:
            logging.error(
                'The image formation algorithm is RMA/RMCR, but Grid.Col.KCtr is non-zero ({}).'.format(Grid.Col.KCtr))
            cond = False

        if RadarCollection.RefFreqIndex is None:
            center_freq = ImageFormation.TxFrequencyProc.center_frequency
            k_f_c = center_freq*2/speed_of_light
            try:
                if abs(k_f_c/Grid.Row.KCtr - 1) > 1e-3:
                    logging.warning(
                        'The image formation algorithm is RMA/RMCR, the Row.KCtr is populated as {}, '
                        'and the expected value is {}'.format(Grid.Row.KCtr, k_f_c))
                    cond = False
            except (AttributeError, ValueError, TypeError):
                pass
        return cond

    def _check_inca(self, Grid, GeoData, RadarCollection, CollectionInfo, Position):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        CollectionInfo : sarpy.io.complex.sicd_elements.CollectionInfo.CollectionInfoType
        Position :  sarpy.io.complex.sicd_elements.Position.PositionType

        Returns
        -------
        bool
        """

        cond = True

        if Grid.Type != 'RGZERO':
            logging.error(
                'The image formation algorithm is RMA/INCA, which should yield '
                'Grid.Type == "RGZERO", but Grid.Type is populated as "{}"'.format(Grid.Type))
            cond = False

        if CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT':
            if self.INCA.DopCentroidPoly is not None:
                logging.error(
                    'The image formation algorithm is RMA/INCA, the '
                    'CollectionInfo.RadarMode.ModeType == "SPOTLIGHT", '
                    'and INCA.DopCentroidPoly is populated.')
                cond = False
            if self.INCA.DopCentroidCOA is True:
                logging.error(
                    'The image formation algorithm is RMA/INCA, the '
                    'CollectionInfo.RadarMode.ModeType == "SPOTLIGHT", '
                    'and INCA.DopCentroidCOA == True.')
                cond = False
        else:
            if self.INCA.DopCentroidPoly is None:
                logging.error(
                    'The image formation algorithm is RMA/INCA, the '
                    'CollectionInfo.RadarMode.ModeType == "{}", '
                    'and INCA.DopCentroidPoly is not populated.'.format(CollectionInfo.RadarMode.ModeType))
                cond = False
            if self.INCA.DopCentroidCOA is not True:
                logging.error(
                    'The image formation algorithm is RMA/INCA, the '
                    'CollectionInfo.RadarMode.ModeType == "{}", '
                    'and INCA.DopCentroidCOA is not True.'.format(CollectionInfo.RadarMode.ModeType))
                cond = False
            if Grid.Col.DeltaKCOAPoly is not None and self.INCA.DopCentroidPoly is not None:
                col_deltakcoa = Grid.Col.DeltaKCOAPoly.get_array(dtype='float64')
                dop_centroid = self.INCA.DopCentroidPoly.get_array(dtype='float64')
                rows = max(col_deltakcoa.shape[0], dop_centroid.shape[0])
                cols = max(col_deltakcoa.shape[1], dop_centroid.shape[1])
                exp_deltakcoa1 = numpy.zeros((rows, cols), dtype='float64')
                exp_deltakcoa2 = numpy.zeros((rows, cols), dtype='float64')
                exp_deltakcoa1[:col_deltakcoa.shape[0], :col_deltakcoa.shape[1]] = col_deltakcoa
                exp_deltakcoa2[:dop_centroid.shape[0], :dop_centroid.shape[1]] = dop_centroid*self.INCA.TimeCAPoly[1]
                if numpy.max(numpy.abs(exp_deltakcoa1 - exp_deltakcoa2)) > 1e-6:
                    logging.error(
                        'The image formation algorithm is RMA/INCA, but the Grid.Col.DeltaKCOAPoly ({}), '
                        'INCA.DopCentroidPoly ({}), and INCA.TimeCAPoly ({}) '
                        'are inconsistent.'.format(col_deltakcoa,
                                                   dop_centroid,
                                                   self.INCA.TimeCAPoly.get_array(dtype='float64')))
                    cond = False

        center_freq = RadarCollection.TxFrequency.center_frequency
        if abs(center_freq/self.INCA.FreqZero - 1) > 1e-5:
            logging.warning(
                'The image formation algorithm is RMA/INCA, and INCA.FreqZero ({}) '
                'should typically agree with center transmit frequency ({})'.format(self.INCA.FreqZero, center_freq))
            cond = False

        if abs(Grid.Col.KCtr) > 1e-8:
            logging.error(
                'The image formation algorithm is RMA/INCA, but Grid.Col.KCtr is '
                'non-zero ({})'.format(Grid.Col.KCtr))
            cond = False

        if RadarCollection.RefFreqIndex is None:
            exp_row_kctr = self.INCA.FreqZero*2/speed_of_light
            if abs(exp_row_kctr/Grid.Row.KCtr - 1) > 1e-8:
                logging.error(
                    'The image formation algorithm is RMA/INCA, the Grid.Row.KCtr is populated as ({}), '
                    'which is not consistent with INCA.FreqZero ({})'.format(Grid.Row.KCtr, self.INCA.FreqZero))
                cond = False

        try:
            SCP = GeoData.SCP.ECF.get_array(dtype='float64')
            row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
            col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')
            scp_time = self.INCA.TimeCAPoly[0]
            ca_pos = Position.ARPPoly(scp_time)
            ca_vel = Position.ARPPoly.derivative_eval(scp_time, der_order=1)
            RG = SCP - ca_pos
            uRG = RG/numpy.linalg.norm(RG)
            left = numpy.cross(ca_pos/numpy.linalg.norm(ca_pos), ca_vel/numpy.linalg.norm(ca_vel))
            look = numpy.sign(left.dot(uRG))
            uSPN = -look*numpy.cross(uRG, ca_vel)
            uSPN /= numpy.linalg.norm(uSPN)
            uAZ = numpy.cross(uSPN, uRG)
        except (AttributeError, ValueError, TypeError):
            return cond

        if numpy.linalg.norm(row_uvect - uRG) > 1e-3:
            logging.error(
                'The image formation algorithm is RMA/INCA, and Row.UVectECF is '
                'populated as {}, but expected to be {}'.format(row_uvect, uRG))
            cond = False
        if numpy.linalg.norm(col_uvect - uAZ) > 1e-3:
            logging.error(
                'The image formation algorithm is RMA/INCA, and Col.UVectECF is '
                'populated as {}, but expected to be {}'.format(col_uvect, uAZ))
            cond = False

        exp_R_CA_SCP = numpy.linalg.norm(RG)
        if abs(exp_R_CA_SCP - self.INCA.R_CA_SCP) > 1e-2:
            logging.error(
                'The image formation algorithm is RMA/INCA, and INCA.R_CA_SCP is '
                'populated as {}, but expected to be {}'.format(self.INCA.R_CA_SCP, exp_R_CA_SCP))
            cond = False

        drate_const = self.INCA.DRateSFPoly[0, 0]
        exp_drate_const = 1./abs(self.INCA.TimeCAPoly[1]*numpy.linalg.norm(ca_vel))
        if abs(exp_drate_const - drate_const) > 1e-3:
            logging.error(
                'The image formation algorithm is RMA/INCA, and the populated INCA.DRateSFPoly constant term ({}) '
                'and expected constant term ({}) are not consistent.'.format(drate_const, exp_drate_const))
            cond = False
        return cond

    def check_parameters(self, Grid, GeoData, RadarCollection, ImageFormation, CollectionInfo, Position):
        """
        Verify consistency of parameters.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType
        CollectionInfo : sarpy.io.complex.sicd_elements.CollectionInfo.CollectionInfoType
        Position : sarpy.io.complex.sicd_elements.Position.PositionType

        Returns
        -------
        bool
        """

        if self.ImageType == 'RMAT':
            return self._check_rmat(Grid, GeoData, RadarCollection, ImageFormation)
        elif self.ImageType == 'RMCR':
            return self._check_rmcr(Grid, GeoData, RadarCollection, ImageFormation)
        elif self.ImageType == 'INCA':
            return self._check_inca(Grid, GeoData, RadarCollection, CollectionInfo, Position)
        return True
