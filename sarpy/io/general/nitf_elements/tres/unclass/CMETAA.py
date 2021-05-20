
import numpy
from sarpy.geometry.geocoords import geodetic_to_ecf

from ..tre_elements import TREExtension, TREElement

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class CMETAAType(TREElement):
    def __init__(self, value):
        super(CMETAAType, self).__init__()
        self.add_field('RELATED_TRES', 's', 2, value)
        self.add_field('ADDITIONAL_TRES', 's', 120, value)
        self.add_field('RD_PRC_NO', 's', 12, value)
        self.add_field('IF_PROCESS', 's', 4, value)
        self.add_field('RD_CEN_FREQ', 's', 4, value)
        self.add_field('RD_MODE', 's', 5, value)
        self.add_field('RD_PATCH_NO', 's', 4, value)
        self.add_field('CMPLX_DOMAIN', 's', 5, value)
        self.add_field('CMPLX_MAG_REMAP_TYPE', 's', 4, value)
        self.add_field('CMPLX_LIN_SCALE', 's', 7, value)
        self.add_field('CMPLX_AVG_POWER', 's', 7, value)
        self.add_field('CMPLX_LINLOG_TP', 's', 5, value)
        self.add_field('CMPLX_PHASE_QUANT_FLAG', 's', 3, value)
        self.add_field('CMPLX_PHASE_QUANT_BIT_DEPTH', 's', 2, value)
        self.add_field('CMPLX_SIZE_1', 's', 2, value)
        self.add_field('CMPLX_IC_1', 's', 2, value)
        self.add_field('CMPLX_SIZE_2', 's', 2, value)
        self.add_field('CMPLX_IC_2', 's', 2, value)
        self.add_field('CMPLX_IC_BPP', 's', 5, value)
        self.add_field('CMPLX_WEIGHT', 's', 3, value)
        self.add_field('CMPLX_AZ_SLL', 's', 2, value)
        self.add_field('CMPLX_RNG_SLL', 's', 2, value)
        self.add_field('CMPLX_AZ_TAY_NBAR', 's', 2, value)
        self.add_field('CMPLX_RNG_TAY_NBAR', 's', 2, value)
        self.add_field('CMPLX_WEIGHT_NORM', 's', 3, value)
        self.add_field('CMPLX_SIGNAL_PLANE', 's', 1, value)

        self.add_field('IF_DC_SF_ROW', 's', 6, value)
        self.add_field('IF_DC_SF_COL', 's', 6, value)

        for part in range(1, 5):
            for comp in ['ROW', 'COL']:
                self.add_field('IF_PATCH_{}{}'.format(comp, part), 's', 6, value)

        self.add_field('IF_DC_IS_ROW', 's', 8, value)
        self.add_field('IF_DC_IS_COL', 's', 8, value)
        self.add_field('IF_IMG_ROW_DC', 's', 8, value)
        self.add_field('IF_IMG_COL_DC', 's', 8, value)

        for part in range(1, 5):
            for comp in ['ROW', 'COL']:
                self.add_field('IF_TILE_{}{}'.format(comp, part), 's', 6, value)

        self.add_field('IF_RD', 's', 1, value)
        self.add_field('IF_RGWLK', 's', 1, value)
        self.add_field('IF_KEYSTN', 's', 1, value)
        self.add_field('IF_LINSFT', 's', 1, value)
        self.add_field('IF_SUBPATCH', 's', 1, value)
        self.add_field('IF_GEODIST', 's', 1, value)
        self.add_field('IF_RGFO', 's', 1, value)
        self.add_field('IF_BEAM_COMP', 's', 1, value)
        self.add_field('IF_RGRES', 's', 8, value)
        self.add_field('IF_AZRES', 's', 8, value)
        self.add_field('IF_RSS', 's', 8, value)
        self.add_field('IF_AZSS', 's', 8, value)
        self.add_field('IF_RSR', 's', 8, value)
        self.add_field('IF_AZSR', 's', 8, value)

        self.add_field('IF_RFFT_SAMP', 's', 7, value)
        self.add_field('IF_AZFFT_SAMP', 's', 7, value)
        self.add_field('IF_RFFT_TOT', 's', 7, value)
        self.add_field('IF_AZFFT_TOT', 's', 7, value)

        self.add_field('IF_SUBP_ROW', 's', 6, value)
        self.add_field('IF_SUBP_COL', 's', 6, value)
        self.add_field('IF_SUB_RG', 's', 4, value)
        self.add_field('IF_SUB_AZ', 's', 4, value)

        self.add_field('IF_RFFTS', 's', 1, value)
        self.add_field('IF_AFFTS', 's', 1, value)
        self.add_field('IF_RANGE_DATA', 's', 7, value)
        self.add_field('IF_INCPH', 's', 1, value)
        for part in range(1, 4):
            for comp in ['NAME', 'AMOUNT']:
                self.add_field('IF_SR_{}{}'.format(comp, part), 's', 8, value)
        for part in range(1, 4):
            self.add_field('AF_TYPE{}'.format(part), 's', 5, value)

        self.add_field('POL_TR', 's', 1, value)
        self.add_field('POL_RE', 's', 1, value)
        self.add_field('POL_REFERENCE', 's', 40, value)
        self.add_field('POL', 's', 1, value)
        self.add_field('POL_REG', 's', 1, value)
        self.add_field('POL_ISO_1', 's', 5, value)
        self.add_field('POL_BAL', 's', 1, value)
        self.add_field('POL_BAL_MAG', 's', 8, value)
        self.add_field('POL_BAL_PHS', 's', 8, value)
        self.add_field('POL_HCOMP', 's', 1, value)
        self.add_field('POL_HCOMP_BASIS', 's', 10, value)
        for part in range(1, 4):
            self.add_field('POL_HCOMP_COEF{}'.format(part), 's', 9, value)
        self.add_field('POL_AFCOMP', 's', 1, value)
        self.add_field('POL_SPARE_A', 's', 15, value)
        self.add_field('POL_SPARE_N', 's', 9, value)

        self.add_field('T_UTC_YYYYMMMDD', 's', 9, value)
        self.add_field('T_HHMMSSUTC', 's', 6, value)
        self.add_field('T_HHMMSSLOCAL', 's', 6, value)

        self.add_field('CG_SRAC', 's', 11, value)
        self.add_field('CG_SLANT_CONFIDENCE', 's', 7, value)
        self.add_field('CG_CROSS', 's', 11, value)
        self.add_field('CG_CROSS_CONFIDENCE', 's', 7, value)
        self.add_field('CG_CAAC', 's', 9, value)
        self.add_field('CG_CONE_CONFIDENCE', 's', 6, value)
        self.add_field('CG_GPSAC', 's', 8, value)
        self.add_field('CG_GPSAC_CONFIDENCE', 's', 6, value)
        self.add_field('CG_SQUINT', 's', 8, value)
        self.add_field('CG_GAAC', 's', 7, value)
        self.add_field('CG_GAAC_CONFIDENCE', 's', 6, value)
        self.add_field('CG_INCIDENT', 's', 7, value)
        self.add_field('CG_SLOPE', 's', 7, value)
        self.add_field('CG_TILT', 's', 8, value)
        self.add_field('CG_LD', 's', 1, value)
        self.add_field('CG_NORTH', 's', 8, value)
        self.add_field('CG_NORTH_CONFIDENCE', 's', 6, value)
        self.add_field('CG_EAST', 's', 8, value)
        self.add_field('CG_RLOS', 's', 8, value)
        self.add_field('CG_LOS_CONFIDENCE', 's', 6, value)
        self.add_field('CG_LAYOVER', 's', 8, value)
        self.add_field('CG_SHADOW', 's', 8, value)
        self.add_field('CG_OPM', 's', 7, value)
        self.add_field('CG_MODEL', 's', 5, value)

        for comp in ['X', 'Y', 'Z']:
            self.add_field('CG_AMPT_{}'.format(comp), 's', 13, value)
        self.add_field('CG_AP_CONF_XY', 's', 6, value)
        self.add_field('CG_AP_CONF_Z', 's', 6, value)

        for comp in ['X', 'Y', 'Z']:
            self.add_field('CG_APCEN_{}'.format(comp), 's', 13, value)
        self.add_field('CG_APER_CONF_XY', 's', 6, value)
        self.add_field('CG_APER_CONF_Z', 's', 6, value)

        for comp in ['X', 'Y', 'Z']:
            self.add_field('CG_FPNUV_{}'.format(comp), 's', 9, value)
        for comp in ['X', 'Y', 'Z']:
            self.add_field('CG_IDPNUV{}'.format(comp), 's', 9, value)
        for comp in ['X', 'Y', 'Z']:
            self.add_field('CG_SCECN_{}'.format(comp), 's', 13, value)
        self.add_field('CG_SC_CONF_XY', 's', 6, value)
        self.add_field('CG_SC_CONF_Z', 's', 6, value)
        self.add_field('CG_SWWD', 's', 8, value)
        for comp in ['X', 'Y', 'Z']:
            self.add_field('CG_SNVEL_{}'.format(comp), 's', 10, value)
        for comp in ['X', 'Y', 'Z']:
            self.add_field('CG_SNACC_{}'.format(comp), 's', 10, value)
        for comp in ['ROLL', 'PITCH', 'YAW']:
            self.add_field('CG_SNATT_{}'.format(comp), 's', 8, value)
        for comp in ['X', 'Y', 'Z']:
            self.add_field('CG_GTP_{}'.format(comp), 's', 9, value)

        self.add_field('CG_MAP_TYPE', 's', 4, value)
        if self.CG_MAP_TYPE.strip() == 'GEOD':
            self.add_field('CG_PATCH_LATCEN', 's', 11, value)
            self.add_field('CG_PATCH_LNGCEN', 's', 12, value)
            self.add_field('CG_PATCH_LTCORUL', 's', 11, value)
            self.add_field('CG_PATCH_LGCORUL', 's', 12, value)
            self.add_field('CG_PATCH_LTCORUR', 's', 11, value)
            self.add_field('CG_PATCH_LGCORUR', 's', 12, value)
            self.add_field('CG_PATCH_LTCORLR', 's', 11, value)
            self.add_field('CG_PATCH_LGCORLR', 's', 12, value)
            self.add_field('CG_PATCH_LTCORLL', 's', 11, value)
            self.add_field('CG_PATH_LNGCOLL', 's', 12, value)
            self.add_field('CG_PATCH_LAT_CONFIDENCE', 's', 9, value)
            self.add_field('CG_PATCH_LONG_CONFIDENCE', 's', 9, value)
        elif self.CG_MAP_TYPE.strip() == 'MGRS':
            self.add_field('CG_MGRS_CENT', 's', 23, value)
            self.add_field('CG_MGRSCORUL', 's', 23, value)
            self.add_field('CG_MGRSCORUR', 's', 23, value)
            self.add_field('CG_MGRSCORLR', 's', 23, value)
            self.add_field('CG_MGRCORLL', 's', 23, value)
            self.add_field('CG_MGRS_CONFIDENCE', 's', 7, value)
            self.add_field('CG_MGRS_PAD', 's', 11, value)
        elif self.CG_MAP_TYPE.strip() == 'NA':
            self.add_field('CG_MAP_TYPE_BLANK', 's', 133, value)
        self.add_field('CG_SPARE_A', 's', 144, value)
        self.add_field('CA_CALPA', 's', 7, value)
        self.add_field('WF_SRTFR', 's', 14, value)
        self.add_field('WF_ENDFR', 's', 14, value)
        self.add_field('WF_CHRPRT', 's', 10, value)
        self.add_field('WF_WIDTH', 's', 9, value)
        self.add_field('WF_CENFRQ', 's', 13, value)
        self.add_field('WF_BW', 's', 13, value)
        self.add_field('WF_PRF', 's', 7, value)
        self.add_field('WF_PRI', 's', 9, value)
        self.add_field('WF_CDP', 's', 7, value)
        self.add_field('WF_NUMBER_OF_PULSES', 's', 9, value)
        self.add_field('VPH_COND', 's', 1, value)


class CMETAA(TREExtension):
    _tag_value = 'CMETAA'
    _data_type = CMETAAType

    def get_scp(self):
        """
        Gets the SCP location in ECF coordinates.

        Returns
        -------
        numpy.ndarray
        """

        cg_model = self.DATA.CG_MODEL.strip()
        scp_array = numpy.array(
            [float(self.DATA.CG_SCECN_X),
             float(self.DATA.CG_SCECN_Y),
             float(self.DATA.CG_SCECN_Z)], dtype='float64')
        if cg_model == 'ECEF':
            return scp_array
        elif cg_model == 'WGS84':
            return geodetic_to_ecf(scp_array)
        else:
            raise ValueError('Got unhandled CG_MODEL {}'.format(cg_model))

    def get_arp(self):
        """
        Gets the Aperture Position, at SCP Time, in ECF coordinates.

        Returns
        -------
        numpy.ndarray
        """

        cg_model = self.DATA.CG_MODEL.strip()
        arp_array = numpy.array(
            [float(self.DATA.CG_APCEN_X),
             float(self.DATA.CG_APCEN_Y),
             float(self.DATA.CG_APCEN_Z)], dtype='float64')
        if cg_model == 'ECEF':
            return arp_array
        elif cg_model == 'WGS84':
            return geodetic_to_ecf(arp_array)
        else:
            raise ValueError('Got unhandled CG_MODEL {}'.format(cg_model))

    def get_image_corners(self):
        """
        Gets the image corners in Lat/Lon, if feasible.

        Returns
        -------
        None|numpy.ndarray
        """

        if self.DATA.CG_MAP_TYPE.strip() != 'GEOD':
            return None

        return numpy.array([
                [float(self.DATA.CG_PATCH_LTCORUL), float(self.DATA.CG_PATCH_LGCORUL)],
                [float(self.DATA.CG_PATCH_LTCORUR), float(self.DATA.CG_PATCH_LGCORUR)],
                [float(self.DATA.CG_PATCH_LTCORLR), float(self.DATA.CG_PATCH_LGCORLR)],
                [float(self.DATA.CG_PATCH_LTCORLL), float(self.DATA.CG_PATCH_LNGCOLL)]], dtype='float64')

