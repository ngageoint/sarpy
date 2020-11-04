# -*- coding: utf-8 -*-
"""
The RgAzCompType definition.
"""

import logging

import numpy
from numpy.linalg import norm
from scipy.constants import speed_of_light

from .base import Serializable, DEFAULT_STRICT, _FloatDescriptor, _SerializableDescriptor
from .blocks import Poly1DType


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class RgAzCompType(Serializable):
    """
    Parameters included for a Range, Doppler image.
    """

    _fields = ('AzSF', 'KazPoly')
    _required = _fields
    _numeric_format = {'AzSF': '0.16G'}
    # descriptors
    AzSF = _FloatDescriptor(
        'AzSF', _required, strict=DEFAULT_STRICT,
        docstring='Scale factor that scales image coordinate az = ycol (meters) to a delta cosine of the '
                  'Doppler Cone Angle at COA, *(in 1/m)*')  # type: float
    KazPoly = _SerializableDescriptor(
        'KazPoly', Poly1DType, _required, strict=DEFAULT_STRICT,
        docstring='Polynomial function that yields azimuth spatial frequency *(Kaz = Kcol)* as a function of '
                  'slow time ``(variable 1)``. That is '
                  r':math:`\text{Slow Time (sec)} \to \text{Azimuth spatial frequency (cycles/meter)}`. '
                  'Time relative to collection start.')  # type: Poly1DType

    def __init__(self, AzSF=None, KazPoly=None, **kwargs):
        """

        Parameters
        ----------
        AzSF : float
        KazPoly : Poly1DType|numpy.ndarray|list|tuple
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.AzSF = AzSF
        self.KazPoly = KazPoly
        super(RgAzCompType, self).__init__(**kwargs)

    def _derive_parameters(self, Grid, Timeline, SCPCOA):
        """
        Expected to be called by the SICD object.

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.GridType
        Timeline : sarpy.io.complex.sicd_elements.TimelineType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType

        Returns
        -------
        None
        """

        look = SCPCOA.look
        az_sf = -look*numpy.sin(numpy.deg2rad(SCPCOA.DopplerConeAng))/SCPCOA.SlantRange
        if self.AzSF is None:
            self.AzSF = az_sf
        elif abs(self.AzSF - az_sf) > 1e-3:
            logging.warning(
                'The derived value for RgAzComp.AzSF is {}, while the current '
                'setting is {}.'.format(az_sf, self.AzSF))

        if self.KazPoly is None:
            if Grid.Row.KCtr is not None and Timeline is not None and Timeline.IPP is not None and \
                    Timeline.IPP.size == 1 and Timeline.IPP[0].IPPPoly is not None and SCPCOA.SCPTime is not None:

                st_rate_coa = Timeline.IPP[0].IPPPoly.derivative_eval(SCPCOA.SCPTime, 1)

                krg_coa = Grid.Row.KCtr
                if Grid.Row is not None and Grid.Row.DeltaKCOAPoly is not None:
                    krg_coa += Grid.Row.DeltaKCOAPoly.Coefs[0, 0]

                # Scale factor described in SICD spec
                delta_kaz_per_delta_v = \
                    look*krg_coa*norm(SCPCOA.ARPVel.get_array()) * \
                    numpy.sin(numpy.deg2rad(SCPCOA.DopplerConeAng))/(SCPCOA.SlantRange*st_rate_coa)
                self.KazPoly = Poly1DType(Coefs=delta_kaz_per_delta_v*Timeline.IPP[0].IPPPoly.Coefs)

    def _check_kaz_poly(self, Timeline, Grid, SCPCOA, look, ARP_Vel):
        """

        Parameters
        ----------
        Timeline : sarpy.io.complex.sicd_elements.Timeline.TimelineType
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
        look : int
        ARP_Vel : numpy.ndarray

        Returns
        -------
        bool
        """

        cond = True
        if Timeline.IPP is not None and len(Timeline.IPP) == 1:
            try:
                st_rate_coa = Timeline.IPP[0].IPPPoly.derivative_eval(SCPCOA.SCPTime, der_order=1)
                if Grid.Row.DeltaKCOAPoly is not None:
                    krg_coa = Grid.Row.KCtr + Grid.Row.DeltaKCOAPoly.get_array(dtype='float64')
                else:
                    krg_coa = Grid.Row.KCtr
                delta_kaz_per_deltav = look*krg_coa*numpy.linalg.norm(ARP_Vel)*numpy.sin(numpy.deg2rad(SCPCOA.DopplerConeAng))/\
                                       (SCPCOA.SlantRange*st_rate_coa)
                if isinstance(delta_kaz_per_deltav, numpy.ndarray):
                    derived_kaz_poly = delta_kaz_per_deltav.dot(Timeline.IPP[0].IPPPoly.get_array(dtype='float64'))
                else:
                    derived_kaz_poly = delta_kaz_per_deltav*Timeline.IPP[0].IPPPoly.get_array(dtype='float64')

                kaz_populated = self.KazPoly.get_array(dtype='float64')
                if numpy.linalg.norm(kaz_populated - derived_kaz_poly) > 1e-3:
                    logging.error(
                        'The image formation algorithm is RGAZCOMP, Timeline.IPP has one element, '
                        'the RgAzComp.KazPoly populated as\n{}\n'
                        'but the expected value is\n{}'.format(kaz_populated, derived_kaz_poly))
                    cond = False
            except (AttributeError, ValueError, TypeError):
                pass
        return cond

    def _check_row_deltakcoa(self, Grid, RadarCollection, ImageFormation):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType

        Returns
        -------
        bool
        """

        if Grid.Row.DeltaKCOAPoly is None:
            return True

        cond = True
        row_deltakcoa = Grid.Row.DeltaKCOAPoly.get_array(dtype='float64')
        if numpy.any(row_deltakcoa != row_deltakcoa[0, 0]):
            logging.error(
                'The image formation algorithm is RGAZCOMP, the Grid.Row.DeltaKCOAPoly is defined, '
                'but all entries are not constant\n{}'.format(row_deltakcoa))
            cond = False

        if RadarCollection.RefFreqIndex is None:
            try:
                fc_proc = ImageFormation.TxFrequencyProc.center_frequency
                k_f_c = fc_proc*2/speed_of_light
                if row_deltakcoa.shape == (1, 1):
                    if abs(Grid.Row.KCtr - (k_f_c - row_deltakcoa[0, 0])) > 1e-6:
                        logging.error(
                            'The image formation algorithm is RGAZCOMP, the Grid.Row.DeltaCOAPoly is scalar, '
                            'and not in agreement with Grid.Row.KCtr and center frequency')

                        cond = False
                else:
                    if abs(Grid.Row.KCtr - k_f_c) > 1e-6:
                        logging.error(
                            'The image formation algorithm is RGAZCOMP, the Grid.Row.DeltaCOAPoly is not scalar, '
                            'and Grid.Row.KCtr not in agreement with center frequency')
                        cond = False
            except (AttributeError, ValueError, TypeError):
                pass
        return cond

    def _check_col_deltacoa(self, Grid):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType

        Returns
        -------
        bool
        """

        cond = True
        if Grid.Col.DeltaKCOAPoly is None:
            if Grid.Col.KCtr != 0:
                logging.error(
                    'The image formation algorithm is RGAZCOMP, the Grid.Col.DeltaKCOAPoly is not defined, '
                    'and Grid.Col.KCtr is non-zero.')
                cond = False
        else:
            col_deltakcoa = Grid.Col.DeltaKCOAPoly.get_array(dtype='float64')
            if numpy.any(col_deltakcoa != col_deltakcoa[0, 0]):
                logging.error(
                    'The image formation algorithm is RGAZCOMP, the Grid.Col.DeltaKCOAPoly is defined, '
                    'but all entries are not constant\n{}'.format(col_deltakcoa))
                cond = False

            if col_deltakcoa.shape == (1, 1) and abs(Grid.Col.KCtr + col_deltakcoa[0, 0]) > 1e-6:
                logging.error(
                    'The image formation algorithm is RGAZCOMP, the Grid.Col.DeltaCOAPoly is scalar, '
                    'and not in agreement with Grid.Col.KCtr')
                cond = False
        return cond

    def check_parameters(self, Grid, RadarCollection, SCPCOA, Timeline, ImageFormation, GeoData):
        """

        Parameters
        ----------
        Grid : sarpy.io.complex.sicd_elements.Grid.GridType
        RadarCollection : sarpy.io.complex.sicd_elements.RadarCollection.RadarCollectionType
        SCPCOA : sarpy.io.complex.sicd_elements.SCPCOA.SCPCOAType
        Timeline : sarpy.io.complex.sicd_elements.Timeline.TimelineType
        ImageFormation : sarpy.io.complex.sicd_elements.ImageFormation.ImageFormationType
        GeoData : sarpy.io.complex.sicd_elements.GeoData.GeoDataType

        Returns
        -------
        bool
        """

        cond = True
        if Grid.ImagePlane != 'SLANT':
            logging.error(
                'The image formation algorithm is RGAZCOMP, and Grid.ImagePlane is populated as "{}", '
                'but should be "SLANT"'.format(Grid.ImagePlane))
            cond = False
        if Grid.Type != 'RGAZIM':
            logging.error(
                'The image formation algorithm is RGAZCOMP, and Grid.Type is populated as "{}", '
                'but should be "RGAZIM"'.format(Grid.Type))
            cond = False

        try:
            SCP = GeoData.SCP.ECF.get_array(dtype='float64')
            row_uvect = Grid.Row.UVectECF.get_array(dtype='float64')
            col_uvect = Grid.Col.UVectECF.get_array(dtype='float64')
            ARP_Pos = SCPCOA.ARPPos.get_array(dtype='float64')
            ARP_Vel = SCPCOA.ARPVel.get_array(dtype='float64')
            uRG = SCP - ARP_Pos
            uRG /= numpy.linalg.norm(uRG)
            left = numpy.cross(ARP_Pos/numpy.linalg.norm(ARP_Pos), ARP_Vel/numpy.linalg.norm(ARP_Vel))
            look = numpy.sign(numpy.dot(left, uRG))
            Spn = -look*numpy.cross(uRG, ARP_Vel)
            uSpn = Spn/numpy.linalg.norm(Spn)
            derived_col_vec = numpy.cross(uSpn, uRG)
        except (AttributeError, ValueError, TypeError):
            return cond

        derived_AzSF = -look*numpy.sin(numpy.deg2rad(SCPCOA.DopplerConeAng))/SCPCOA.SlantRange
        if abs(self.AzSF - derived_AzSF) > 1e-6:
            logging.error(
                'The image formation algorithm is RGAZCOMP, RgAzComp.AzSF is populated as {}, '
                'but expected value is {}'.format(self.AzSF, derived_AzSF))
            cond = False

        if numpy.linalg.norm(uRG - row_uvect) > 1e-3:
            logging.error(
                'The image formation algorithm is RGAZCOMP, and Grid.Row.UVectECF is populated as \n{}\n'
                'which should agree with the unit range vector\n{}'.format(row_uvect, uRG))
            cond = False
        if numpy.linalg.norm(derived_col_vec - col_uvect) > 1e-3:
            logging.error(
                'The image formation algorithm is RGAZCOMP, and Grid.Col.UVectECF is populated as \n{}\n'
                'which should agree with derived vector\n{}'.format(col_uvect, derived_col_vec))
            cond = False

        cond &= self._check_kaz_poly(Timeline, Grid, SCPCOA, look, ARP_Vel)
        cond &= self._check_row_deltakcoa(Grid, RadarCollection, ImageFormation)
        cond &= self._check_col_deltacoa(Grid)
        return cond
