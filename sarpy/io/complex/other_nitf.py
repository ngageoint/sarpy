"""
Work in progress for reading some other kind of complex NITF.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from typing import Union, Tuple, List, Optional, Callable, Sequence
import copy
from datetime import datetime

import numpy
from scipy.constants import foot

from sarpy.geometry.geocoords import geodetic_to_ecf, ned_to_ecf
from sarpy.geometry.latlon import num as lat_lon_parser

from sarpy.io.general.base import SarpyIOError
from sarpy.io.general.data_segment import DataSegment, SubsetSegment
from sarpy.io.general.format_function import FormatFunction, ComplexFormatFunction
from sarpy.io.general.nitf import extract_image_corners, NITFDetails, NITFReader
from sarpy.io.general.nitf_elements.security import NITFSecurityTags
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, ImageSegmentHeader0
from sarpy.io.general.nitf_elements.nitf_head import NITFHeader, NITFHeader0
from sarpy.io.general.nitf_elements.base import TREList
from sarpy.io.general.nitf_elements.tres.unclass.CMETAA import CMETAA
from sarpy.io.general.utils import is_file_like

from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    TxFrequencyType, WaveformParametersType, ChanParametersType
from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, TxFrequencyProcType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.PFA import PFAType


logger = logging.getLogger(__name__)

_iso_date_format = '{}-{}-{}T{}:{}:{}'

# NB: DO NOT implement is_a() here.
#   This will explicitly happen after other readers


########
# Define sicd structure from image sub-header information

def extract_sicd(
        img_header: Union[ImageSegmentHeader, ImageSegmentHeader0],
        transpose: True,
        nitf_header: Optional[Union[NITFHeader, NITFHeader0]] = None) -> SICDType:
    """
    Extract the best available SICD structure from relevant nitf header structures.

    Parameters
    ----------
    img_header : ImageSegmentHeader|ImageSegmentHeader0
    transpose : bool
    nitf_header : None|NITFHeader|NITFHeader0

    Returns
    -------
    SICDType
    """

    def get_collection_info() -> CollectionInfoType:
        isorce = img_header.ISORCE.strip()
        collector_name = None if len(isorce) < 1 else isorce

        iid2 = img_header.IID2.strip()
        core_name = img_header.IID1.strip() if len(iid2) < 1 else iid2

        class_str = img_header.Security.CLAS
        if class_str == 'T':
            classification = 'TOPSECRET'
        elif class_str == 'S':
            classification = 'SECRET'
        elif class_str == 'C':
            classification = 'CONFIDENTIAL'
        elif class_str == 'U':
            classification = 'UNCLASSIFIED'
        else:
            classification = ''
        ctlh = img_header.Security.CTLH.strip()
        if len(ctlh) < 1:
            classification += '//' + ctlh
        code = img_header.Security.CODE.strip()
        if len(code) < 1:
            classification += '//' + code

        return CollectionInfoType(
            CollectorName=collector_name,
            CoreName=core_name,
            Classification=classification)

    def get_image_data() -> ImageDataType:
        pvtype = img_header.PVTYPE
        if pvtype == 'C':
            if img_header.NBPP != 64:
                logger.warning(
                    'This NITF has complex bands that are not 64-bit.\n\t'
                    'This is not currently supported.')
            pixel_type = 'RE32F_IM32F'
        elif pvtype == 'R':
            if img_header.NBPP == 64:
                logger.warning(
                    'The real/imaginary data in the NITF are stored as 64-bit floating point.\n\t'
                    'The closest Pixel Type, RE32F_IM32F, will be used,\n\t'
                    'but there may be overflow issues if converting this file.')
            pixel_type = 'RE32F_IM32F'
        elif pvtype == 'SI':
            pixel_type = 'RE16I_IM16I'
        else:
            raise ValueError('Got unhandled PVTYPE {}'.format(pvtype))

        if transpose:
            rows = img_header.NCOLS
            cols = img_header.NROWS
        else:
            rows = img_header.NROWS
            cols = img_header.NCOLS
        return ImageDataType(
            PixelType=pixel_type,
            NumRows=rows,
            NumCols=cols,
            FirstRow=0,
            FirstCol=0,
            FullImage=(rows, cols),
            SCPPixel=(0.5 * rows, 0.5 * cols))

    def append_country_code(cc) -> None:
        if len(cc) > 0:
            if the_sicd.CollectionInfo is None:
                the_sicd.CollectionInfo = CollectionInfoType(CountryCodes=[cc, ])
            elif the_sicd.CollectionInfo.CountryCodes is None:
                the_sicd.CollectionInfo.CountryCodes = [cc, ]
            elif cc not in the_sicd.CollectionInfo.CountryCodes:
                the_sicd.CollectionInfo.CountryCodes.append(cc)

    def set_image_corners(icps: numpy.ndarray, override: bool = False) -> None:
        if the_sicd.GeoData is None:
            the_sicd.GeoData = GeoDataType(ImageCorners=icps)
        elif the_sicd.GeoData.ImageCorners is None or override:
            the_sicd.GeoData.ImageCorners = icps

    def set_arp_position(arp_ecf: numpy.ndarray, override: bool = False) -> None:
        if the_sicd.SCPCOA is None:
            the_sicd.SCPCOA = SCPCOAType(ARPPos=arp_ecf)
        elif override:
            # prioritize this information first - it should be more reliable than other sources
            the_sicd.SCPCOA.ARPPos = arp_ecf

    def set_scp(scp_ecf: numpy.ndarray, scp_pixel: Union[numpy.ndarray, list, tuple], override: bool = False) -> None:
        def set_scppixel():
            if the_sicd.ImageData is None:
                the_sicd.ImageData = ImageDataType(SCPPixel=scp_pixel)
            else:
                the_sicd.ImageData.SCPPixel = scp_pixel
        if the_sicd.GeoData is None:
            the_sicd.GeoData = GeoDataType(SCP=SCPType(ECF=scp_ecf))
            set_scppixel()
        elif the_sicd.GeoData.SCP is None or override:
            the_sicd.GeoData.SCP = SCPType(ECF=scp_ecf)
            set_scppixel()

    def set_collect_start(
            collect_start: Union[str, datetime, numpy.datetime64], override: bool = False) -> None:
        if the_sicd.Timeline is None:
            the_sicd.Timeline = TimelineType(CollectStart=collect_start)
        elif the_sicd.Timeline.CollectStart is None or override:
            the_sicd.Timeline.CollectStart = collect_start

    def set_uvects(row_unit: numpy.ndarray, col_unit: numpy.ndarray) -> None:
        if the_sicd.Grid is None:
            the_sicd.Grid = GridType(
                Row=DirParamType(UVectECF=row_unit),
                Col=DirParamType(UVectECF=col_unit))
            return

        if the_sicd.Grid.Row is None:
            the_sicd.Grid.Row = DirParamType(UVectECF=row_unit)
        elif the_sicd.Grid.Row.UVectECF is None:
            the_sicd.Grid.Row.UVectECF = row_unit

        if the_sicd.Grid.Col is None:
            the_sicd.Grid.Col = DirParamType(UVectECF=col_unit)
        elif the_sicd.Grid.Col.UVectECF is None:
            the_sicd.Grid.Col.UVectECF = col_unit

    def try_CMETAA() -> None:
        # noinspection PyTypeChecker
        tre = None if tres is None else tres['CMETAA']  # type: CMETAA
        if tre is None:
            return

        cmetaa = tre.DATA

        if the_sicd.GeoData is None:
            the_sicd.GeoData = GeoDataType()
        if the_sicd.SCPCOA is None:
            the_sicd.SCPCOA = SCPCOAType()
        if the_sicd.Grid is None:
            the_sicd.Grid = GridType()
        if the_sicd.Timeline is None:
            the_sicd.Timeline = TimelineType()
        if the_sicd.RadarCollection is None:
            the_sicd.RadarCollection = RadarCollectionType()
        if the_sicd.ImageFormation is None:
            the_sicd.ImageFormation = ImageFormationType()

        the_sicd.SCPCOA.SCPTime = 0.5*float(cmetaa.WF_CDP)
        the_sicd.GeoData.SCP = SCPType(ECF=tre.get_scp())
        the_sicd.SCPCOA.ARPPos = tre.get_arp()

        the_sicd.SCPCOA.SideOfTrack = cmetaa.CG_LD.strip().upper()
        the_sicd.SCPCOA.SlantRange = float(cmetaa.CG_SRAC)
        the_sicd.SCPCOA.DopplerConeAng = float(cmetaa.CG_CAAC)
        the_sicd.SCPCOA.GrazeAng = float(cmetaa.CG_GAAC)
        the_sicd.SCPCOA.IncidenceAng = 90 - float(cmetaa.CG_GAAC)
        if hasattr(cmetaa, 'CG_TILT'):
            the_sicd.SCPCOA.TwistAng = float(cmetaa.CG_TILT)
        if hasattr(cmetaa, 'CG_SLOPE'):
            the_sicd.SCPCOA.SlopeAng = float(cmetaa.CG_SLOPE)

        the_sicd.ImageData.SCPPixel = [int(cmetaa.IF_DC_IS_COL), int(cmetaa.IF_DC_IS_ROW)]
        img_corners = tre.get_image_corners()
        if img_corners is not None:
            the_sicd.GeoData.ImageCorners = img_corners

        if cmetaa.CMPLX_SIGNAL_PLANE.upper() == 'S':
            the_sicd.Grid.ImagePlane = 'SLANT'
        elif cmetaa.CMPLX_SIGNAL_PLANE.upper() == 'G':
            the_sicd.Grid.ImagePlane = 'GROUND'
        else:
            logger.warning(
                'Got unexpected CMPLX_SIGNAL_PLANE value {},\n\t'
                'setting ImagePlane to SLANT'.format(cmetaa.CMPLX_SIGNAL_PLANE))

        the_sicd.Grid.Row = DirParamType(
            SS=float(cmetaa.IF_RSS),
            ImpRespWid=float(cmetaa.IF_RGRES),
            Sgn=1 if cmetaa.IF_RFFTS.strip() == '-' else -1,  # opposite sign convention
            ImpRespBW=float(cmetaa.IF_RFFT_SAMP)/(float(cmetaa.IF_RSS)*float(cmetaa.IF_RFFT_TOT)))
        the_sicd.Grid.Col = DirParamType(
            SS=float(cmetaa.IF_AZSS),
            ImpRespWid=float(cmetaa.IF_AZRES),
            Sgn=1 if cmetaa.IF_AFFTS.strip() == '-' else -1,  # opposite sign convention
            ImpRespBW=float(cmetaa.IF_AZFFT_SAMP)/(float(cmetaa.IF_AZSS)*float(cmetaa.IF_AZFFT_TOT)))
        cmplx_weight = cmetaa.CMPLX_WEIGHT.strip().upper()
        if cmplx_weight == 'UWT':
            the_sicd.Grid.Row.WgtType = WgtTypeType(WindowName='UNIFORM')
            the_sicd.Grid.Col.WgtType = WgtTypeType(WindowName='UNIFORM')
        elif cmplx_weight == 'HMW':
            the_sicd.Grid.Row.WgtType = WgtTypeType(WindowName='HAMMING')
            the_sicd.Grid.Col.WgtType = WgtTypeType(WindowName='HAMMING')
        elif cmplx_weight == 'HNW':
            the_sicd.Grid.Row.WgtType = WgtTypeType(WindowName='HANNING')
            the_sicd.Grid.Col.WgtType = WgtTypeType(WindowName='HANNING')
        elif cmplx_weight == 'TAY':
            the_sicd.Grid.Row.WgtType = WgtTypeType(
                WindowName='TAYLOR',
                Parameters={
                    'SLL': '-{0:d}'.format(int(cmetaa.CMPLX_RNG_SLL)),
                    'NBAR': '{0:d}'.format(int(cmetaa.CMPLX_RNG_TAY_NBAR))})
            the_sicd.Grid.Col.WgtType = WgtTypeType(
                WindowName='TAYLOR',
                Parameters={
                    'SLL': '-{0:d}'.format(int(cmetaa.CMPLX_AZ_SLL)),
                    'NBAR': '{0:d}'.format(int(cmetaa.CMPLX_AZ_TAY_NBAR))})
        else:
            logger.warning(
                'Got unsupported CMPLX_WEIGHT value {}.\n\tThe resulting SICD will '
                'not have valid weight array populated'.format(cmplx_weight))
        the_sicd.Grid.Row.define_weight_function()
        the_sicd.Grid.Col.define_weight_function()

        # noinspection PyBroadException
        try:
            date_str = cmetaa.T_UTC_YYYYMMMDD
            time_str = cmetaa.T_HHMMSSUTC
            date_time = _iso_date_format.format(
                date_str[:4], date_str[4:6], date_str[6:8],
                time_str[:2], time_str[2:4], time_str[4:6])
            the_sicd.Timeline.CollectStart = numpy.datetime64(date_time, 'us')
        except Exception:
            logger.info('Failed extracting start time from CMETAA')
            pass
        the_sicd.Timeline.CollectDuration = float(cmetaa.WF_CDP)
        the_sicd.Timeline.IPP = [
            IPPSetType(TStart=0,
                       TEnd=float(cmetaa.WF_CDP),
                       IPPStart=0,
                       IPPEnd=numpy.floor(float(cmetaa.WF_CDP)*float(cmetaa.WF_PRF)),
                       IPPPoly=[0, float(cmetaa.WF_PRF)])]

        the_sicd.RadarCollection.TxFrequency = TxFrequencyType(
            Min=float(cmetaa.WF_SRTFR),
            Max=float(cmetaa.WF_ENDFR))
        the_sicd.RadarCollection.TxPolarization = cmetaa.POL_TR.upper()
        the_sicd.RadarCollection.Waveform = [WaveformParametersType(
            TxPulseLength=float(cmetaa.WF_WIDTH),
            TxRFBandwidth=float(cmetaa.WF_BW),
            TxFreqStart=float(cmetaa.WF_SRTFR),
            TxFMRate=float(cmetaa.WF_CHRPRT)*1e12)]
        tx_rcv_pol = '{}:{}'.format(cmetaa.POL_TR.upper(), cmetaa.POL_RE.upper())
        the_sicd.RadarCollection.RcvChannels = [
            ChanParametersType(TxRcvPolarization=tx_rcv_pol)]

        the_sicd.ImageFormation.TxRcvPolarizationProc = tx_rcv_pol
        if_process = cmetaa.IF_PROCESS.strip().upper()
        if if_process == 'PF':
            the_sicd.ImageFormation.ImageFormAlgo = 'PFA'
            scp_ecf = tre.get_scp()
            fpn_ned = numpy.array(
                [float(cmetaa.CG_FPNUV_X), float(cmetaa.CG_FPNUV_Y), float(cmetaa.CG_FPNUV_Z)], dtype='float64')
            ipn_ned = numpy.array(
                [float(cmetaa.CG_IDPNUVX), float(cmetaa.CG_IDPNUVY), float(cmetaa.CG_IDPNUVZ)], dtype='float64')
            fpn_ecf = ned_to_ecf(fpn_ned, scp_ecf, absolute_coords=False)
            ipn_ecf = ned_to_ecf(ipn_ned, scp_ecf, absolute_coords=False)
            the_sicd.PFA = PFAType(FPN=fpn_ecf, IPN=ipn_ecf)
        elif if_process in ['RM', 'CD']:
            the_sicd.ImageFormation.ImageFormAlgo = 'RMA'

        # the remainder of this is guesswork to define required fields
        the_sicd.ImageFormation.TStartProc = 0  # guess work
        the_sicd.ImageFormation.TEndProc = float(cmetaa.WF_CDP)
        the_sicd.ImageFormation.TxFrequencyProc = TxFrequencyProcType(
            MinProc=float(cmetaa.WF_SRTFR), MaxProc=float(cmetaa.WF_ENDFR))
        # all remaining guess work
        the_sicd.ImageFormation.STBeamComp = 'NO'
        the_sicd.ImageFormation.ImageBeamComp = 'SV' if cmetaa.IF_BEAM_COMP[0] == 'Y' else 'NO'
        the_sicd.ImageFormation.AzAutofocus = 'NO' if cmetaa.AF_TYPE[0] == 'N' else 'SV'
        the_sicd.ImageFormation.RgAutofocus = 'NO'

    def try_AIMIDA() -> None:
        tre = None if tres is None else tres['AIMIDA']
        if tre is None:
            return
        aimida = tre.DATA

        append_country_code(aimida.COUNTRY.strip())

        create_time = datetime.strptime(aimida.CREATION_DATE, '%d%b%y')
        if the_sicd.ImageCreation is None:
            the_sicd.ImageCreation = ImageCreationType(DateTime=create_time)
        elif the_sicd.ImageCreation.DateTime is None:
            the_sicd.ImageCreation.DateTime = create_time

        collect_start = datetime.strptime(aimida.MISSION_DATE+aimida.TIME, '%d%b%y%H%M')
        set_collect_start(collect_start, override=False)

    def try_AIMIDB() -> None:
        tre = None if tres is None else tres['AIMIDB']
        if tre is None:
            return
        aimidb = tre.DATA

        append_country_code(aimidb.COUNTRY.strip())

        if the_sicd.ImageFormation is not None and the_sicd.ImageFormation.SegmentIdentifier is None:
            the_sicd.ImageFormation.SegmentIdentifier = aimidb.CURRENT_SEGMENT.strip()

        date_str = aimidb.ACQUISITION_DATE
        collect_start = numpy.datetime64(_iso_date_format.format(
            date_str[:4], date_str[4:6], date_str[6:8],
            date_str[8:10], date_str[10:12], date_str[12:14]), 'us')
        set_collect_start(collect_start, override=False)

    def try_ACFT() -> None:
        if tres is None:
            return
        tre = tres['ACFTA']
        if tre is None:
            tre = tres['ACFTB']
        if tre is None:
            return
        acft = tre.DATA

        sensor_id = acft.SENSOR_ID.strip()
        if len(sensor_id) > 1:
            if the_sicd.CollectionInfo is None:
                the_sicd.CollectionInfo = CollectionInfoType(CollectorName=sensor_id)
            elif the_sicd.CollectionInfo.CollectorName is None:
                the_sicd.CollectionInfo.CollectorName = sensor_id

        row_ss = float(acft.ROW_SPACING)
        col_ss = float(acft.COL_SPACING)

        if hasattr(acft, 'ROW_SPACING_UNITS') and acft.ROW_SPACING_UNITS.strip().lower() == 'f':
            row_ss *= foot
        if hasattr(acft, 'COL_SPACING_UNITS') and acft.COL_SPACING_UNITS.strip().lower() == 'f':
            col_ss *= foot

        # NB: these values are actually ground plane values, and should be
        # corrected to slant plane if possible
        if the_sicd.SCPCOA is not None:
            if the_sicd.SCPCOA.GrazeAng is not None:
                col_ss *= numpy.cos(numpy.deg2rad(the_sicd.SCPCOA.GrazeAng))
            if the_sicd.SCPCOA.TwistAng is not None:
                row_ss *= numpy.cos(numpy.deg2rad(the_sicd.SCPCOA.TwistAng))

        if the_sicd.Grid is None:
            the_sicd.Grid = GridType(Row=DirParamType(SS=row_ss), Col=DirParamType(SS=col_ss))
            return

        if the_sicd.Grid.Row is None:
            the_sicd.Grid.Row = DirParamType(SS=row_ss)
        elif the_sicd.Grid.Row.SS is None:
            the_sicd.Grid.Row.SS = row_ss

        if the_sicd.Grid.Col is None:
            the_sicd.Grid.Col = DirParamType(SS=col_ss)
        elif the_sicd.Grid.Col.SS is None:
            the_sicd.Grid.Col.SS = col_ss

    def try_BLOCKA() -> None:
        tre = None if tres is None else tres['BLOCKA']
        if tre is None:
            return
        blocka = tre.DATA

        icps = []
        for fld_name in ['FRFC_LOC', 'FRLC_LOC', 'LRLC_LOC', 'LRFC_LOC']:
            value = getattr(blocka, fld_name)
            # noinspection PyBroadException
            try:
                lat_val = float(value[:10])
                lon_val = float(value[10:21])
            except ValueError:
                lat_val = lat_lon_parser(value[:10])
                lon_val = lat_lon_parser(value[10:21])

            icps.append([lat_val, lon_val])
        set_image_corners(numpy.array(icps, dtype='float64'), override=False)

    def try_MPDSRA() -> None:
        def valid_array(arr):
            return numpy.all(numpy.isfinite(arr)) and numpy.any(arr != 0)

        tre = None if tres is None else tres['MPDSRA']
        if tre is None:
            return
        mpdsra = tre.DATA

        scp_ecf = foot*numpy.array(
            [float(mpdsra.ORO_X), float(mpdsra.ORO_Y), float(mpdsra.ORO_Z)], dtype='float64')
        if valid_array(scp_ecf):
            set_scp(scp_ecf, (int(mpdsra.ORP_COLUMN) - 1, int(mpdsra.ORP_ROW) - 1), override=False)

        arp_pos_ned = foot*numpy.array(
            [float(mpdsra.ARP_POS_N), float(mpdsra.ARP_POS_E), float(mpdsra.ARP_POS_D)], dtype='float64')
        arp_vel_ned = foot*numpy.array(
            [float(mpdsra.ARP_VEL_N), float(mpdsra.ARP_VEL_E), float(mpdsra.ARP_VEL_D)], dtype='float64')
        arp_acc_ned = foot*numpy.array(
            [float(mpdsra.ARP_ACC_N), float(mpdsra.ARP_ACC_E), float(mpdsra.ARP_ACC_D)], dtype='float64')
        arp_pos = ned_to_ecf(arp_pos_ned, scp_ecf, absolute_coords=True) if valid_array(arp_pos_ned) else None
        set_arp_position(arp_pos, override=False)

        arp_vel = ned_to_ecf(arp_vel_ned, scp_ecf, absolute_coords=False) if valid_array(arp_vel_ned) else None
        if the_sicd.SCPCOA.ARPVel is None:
            the_sicd.SCPCOA.ARPVel = arp_vel
        arp_acc = ned_to_ecf(arp_acc_ned, scp_ecf, absolute_coords=False) if valid_array(arp_acc_ned) else None
        if the_sicd.SCPCOA.ARPAcc is None:
            the_sicd.SCPCOA.ARPAcc = arp_acc

        if the_sicd.PFA is not None and the_sicd.PFA.FPN is None:
            # TODO: is this already in meters?
            fpn_ecf = numpy.array(
                [float(mpdsra.FOC_X), float(mpdsra.FOC_Y), float(mpdsra.FOC_Z)], dtype='float64')  # *foot
            if valid_array(fpn_ecf):
                the_sicd.PFA.FPN = fpn_ecf

    def try_MENSRB() -> None:
        tre = None if tres is None else tres['MENSRB']
        if tre is None:
            return
        mensrb = tre.DATA

        arp_llh = numpy.array(
            [lat_lon_parser(mensrb.ACFT_LOC[:12]),
             lat_lon_parser(mensrb.ACFT_LOC[12:25]),
             foot*float(mensrb.ACFT_ALT)], dtype='float64')
        scp_llh = numpy.array(
            [lat_lon_parser(mensrb.RP_LOC[:12]),
             lat_lon_parser(mensrb.RP_LOC[12:25]),
             foot*float(mensrb.RP_ELV)], dtype='float64')
        # TODO: handle the conversion from msl to hae

        arp_ecf = geodetic_to_ecf(arp_llh)
        scp_ecf = geodetic_to_ecf(scp_llh)
        set_arp_position(arp_ecf, override=True)

        set_scp(scp_ecf, (int(mensrb.RP_COL)-1, int(mensrb.RP_ROW)-1), override=False)

        row_unit_ned = numpy.array(
            [float(mensrb.C_R_NC), float(mensrb.C_R_EC), float(mensrb.C_R_DC)], dtype='float64')
        col_unit_ned = numpy.array(
            [float(mensrb.C_AZ_NC), float(mensrb.C_AZ_EC), float(mensrb.C_AZ_DC)], dtype='float64')
        set_uvects(ned_to_ecf(row_unit_ned, scp_ecf, absolute_coords=False),
                   ned_to_ecf(col_unit_ned, scp_ecf, absolute_coords=False))

    def try_MENSRA() -> None:
        tre = None if tres is None else tres['MENSRA']
        if tre is None:
            return
        mensra = tre.DATA

        arp_llh = numpy.array(
            [lat_lon_parser(mensra.ACFT_LOC[:10]),
             lat_lon_parser(mensra.ACFT_LOC[10:21]),
             foot*float(mensra.ACFT_ALT)], dtype='float64')
        scp_llh = numpy.array(
            [lat_lon_parser(mensra.CP_LOC[:10]),
             lat_lon_parser(mensra.CP_LOC[10:21]),
             foot*float(mensra.CP_ALT)], dtype='float64')
        # TODO: handle the conversion from msl to hae

        arp_ecf = geodetic_to_ecf(arp_llh)
        scp_ecf = geodetic_to_ecf(scp_llh)
        set_arp_position(arp_ecf, override=True)

        # TODO: is this already zero based?
        set_scp(geodetic_to_ecf(scp_llh), (int(mensra.CCRP_COL), int(mensra.CCRP_ROW)), override=False)

        row_unit_ned = numpy.array(
            [float(mensra.C_R_NC), float(mensra.C_R_EC), float(mensra.C_R_DC)], dtype='float64')
        col_unit_ned = numpy.array(
            [float(mensra.C_AZ_NC), float(mensra.C_AZ_EC), float(mensra.C_AZ_DC)], dtype='float64')
        set_uvects(ned_to_ecf(row_unit_ned, scp_ecf, absolute_coords=False),
                   ned_to_ecf(col_unit_ned, scp_ecf, absolute_coords=False))

    def extract_corners() -> None:
        icps = extract_image_corners(img_header)
        if icps is None:
            return
        # TODO: include symmetry transform issue
        set_image_corners(icps, override=False)

    def extract_start() -> None:
        # noinspection PyBroadException
        try:
            date_str = img_header.IDATIM
            collect_start = numpy.datetime64(
                _iso_date_format.format(
                    date_str[:4], date_str[4:6], date_str[6:8],
                    date_str[8:10], date_str[10:12], date_str[12:14]), 'us')
        except Exception:
            logger.info('failed extracting start time from IDATIM tre')
            return

        set_collect_start(collect_start, override=False)

    # noinspection PyUnresolvedReferences
    tres = None if img_header.ExtendedHeader.data is None \
        else img_header.ExtendedHeader.data  # type: Union[None, TREList]

    collection_info = get_collection_info()
    image_data = get_image_data()
    the_sicd = SICDType(
        CollectionInfo=collection_info,
        ImageData=image_data)
    # apply the various tres and associated logic
    # NB: this should generally be in order of preference
    try_CMETAA()
    try_AIMIDB()
    try_AIMIDA()
    try_ACFT()
    try_BLOCKA()
    try_MPDSRA()
    try_MENSRA()
    try_MENSRB()
    extract_corners()
    extract_start()
    return the_sicd


# Helper methods for transforming data

def get_linear_magnitude_scaling(scale_factor: float):
    """
    Get a linear magnitude scaling function, to correct magnitude.

    Parameters
    ----------
    scale_factor : float
        The scale factor, according to the definition given in STDI-0002.

    Returns
    -------
    callable
    """

    def scaler(data):
        return data/scale_factor
    return scaler


def get_linear_power_scaling(scale_factor):
    """
    Get a linear power scaling function, to derive correct magnitude.

    Parameters
    ----------
    scale_factor : float
        The scale factor, according to the definition given in STDI-0002.

    Returns
    -------
    callable
    """

    def scaler(data):
        return numpy.sqrt(data/scale_factor)
    return scaler


def get_log_magnitude_scaling(scale_factor, db_per_step):
    """
    Gets the log magnitude scaling function, to derive correct magnitude.

    Parameters
    ----------
    scale_factor : float
        The scale factor, according to the definition given in STDI-0002.
    db_per_step : float
        The db_per_step factor, according to the definiton given in STDI-0002

    Returns
    -------
    callable
    """

    lin_scaler = get_linear_magnitude_scaling(scale_factor)

    def scaler(data):
        return lin_scaler(numpy.exp(0.05*numpy.log(10)*db_per_step*data))

    return scaler


def get_log_power_scaling(scale_factor, db_per_step):
    """
    Gets the log power scaling function, to derive correct magnitude.

    Parameters
    ----------
    scale_factor : float
        The scale factor, according to the definition given in STDI-0002.
    db_per_step : float
        The db_per_step factor, according to the definiton given in STDI-0002

    Returns
    -------
    callable
    """

    power_scaler = get_linear_power_scaling(scale_factor)

    def scaler(data):
        return power_scaler(numpy.exp(0.1*numpy.log(10)*db_per_step*data))

    return scaler


def get_linlog_magnitude_scaling(scale_factor, tipping_point):
    """
    Gets the magnitude scaling function for the model which
    is initially linear, and then switches to logarithmic beyond a fixed
    tipping point.

    Parameters
    ----------
    scale_factor : float
        The scale factor, according to the definition given in STDI-0002.
    tipping_point : float
        The tipping point between the two models.

    Returns
    -------
    callable
    """

    db_per_step = 20*numpy.log10(tipping_point)/tipping_point
    log_scaler = get_log_magnitude_scaling(scale_factor, db_per_step)

    def scaler(data):
        out = data/scale_factor
        above_tipping = (out > tipping_point)
        out[above_tipping] = log_scaler(data[above_tipping])
        return out
    return scaler


class ApplyAmplitudeScalingFunction(ComplexFormatFunction):
    __slots__ = ('_scaling_function', )
    _allowed_ordering = ('MP', 'PM')
    has_inverse = False

    def __init__(
            self,
            raw_dtype: Union[str, numpy.dtype],
            order: str,
            scaling_function: Optional[Callable] = None,
            raw_shape: Optional[Tuple[int, ...]] = None,
            formatted_shape: Optional[Tuple[int, ...]] = None,
            reverse_axes: Optional[Tuple[int, ...]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None,
            band_dimension: int = -1):
        """

        Parameters
        ----------
        raw_dtype : str|numpy.dtype
            The raw datatype. Valid options dependent on the value of order.
        order : str
            One of `('MP', 'PM')`, with allowable raw_dtype
            `('uint8', 'uint16', 'uint32', 'float32', 'float64')`.
        scaling_function : Optional[Callable]
        raw_shape : None|Tuple[int, ...]
        formatted_shape : None|Tuple[int, ...]
        reverse_axes : None|Tuple[int, ...]
        transpose_axes : None|Tuple[int, ...]
        band_dimension : int
            Which band is the complex dimension, **after** the transpose operation.
        """

        self._scaling_function = None
        ComplexFormatFunction.__init__(
            self, raw_dtype, order, raw_shape=raw_shape, formatted_shape=formatted_shape,
            reverse_axes=reverse_axes, transpose_axes=transpose_axes, band_dimension=band_dimension)
        self._set_scaling_function(scaling_function)

    @property
    def scaling_function(self) -> Optional[Callable]:
        """
        The magnitude scaling function.

        Returns
        -------
        None|Callable
        """

        return self._scaling_function

    def _set_scaling_function(self, value: Optional[Callable]):
        if value is None:
            self._scaling_function = None
            return
        if not isinstance(value, Callable):
            raise TypeError('scaling_function must be callable')
        self._scaling_function = value

    def _forward_magnitude_theta(
            self,
            data: numpy.ndarray,
            out: numpy.ndarray,
            magnitude: numpy.ndarray,
            theta: numpy.ndarray,
            subscript: Tuple[slice, ...]) -> None:
        if self._scaling_function is not None:
            magnitude = self._scaling_function(magnitude)
        ComplexFormatFunction._forward_magnitude_theta(
            self, data, out, magnitude, theta, subscript)


def _extract_transform_data(
        image_header: Union[ImageSegmentHeader, ImageSegmentHeader0],
        band_dimension: int):
    """
    Helper function for defining necessary transform_data definition for
    interpreting image segment data.

    Parameters
    ----------
    image_header : ImageSegmentHeader|ImageSegmentHeader0

    Returns
    -------
    None|str|callable
    """

    if len(image_header.Bands) != 2:
        raise ValueError('Got unhandled case of {} image bands'.format(len(image_header.Bands)))

    complex_order = image_header.Bands[0].ISUBCAT+image_header.Bands[1].ISUBCAT
    if complex_order not in ['IQ', 'QI', 'MP', 'PM']:
        raise ValueError('Got unhandled complex order `{}`'.format(complex_order))

    bpp = int(image_header.NBPP/8)
    pv_type = image_header.PVTYPE
    if pv_type == 'INT':
        raw_dtype = '>u{}'.format(bpp)
    elif pv_type == 'SI':
        raw_dtype = '>i{}'.format(bpp)
    elif pv_type == 'R':
        raw_dtype = '>f{}'.format(bpp)
    else:
        raise ValueError('Got unhandled PVTYPE {}'.format(pv_type))

    # noinspection PyUnresolvedReferences
    tre = None if img_header.ExtendedHeader.data is None else \
        img_header.ExtendedHeader.data['CMETAA']  # type: Optional[CMETAA]

    if tre is None:
        return ComplexFormatFunction(raw_dtype, complex_order, band_dimension=band_dimension)

    cmetaa = tre.DATA
    if cmetaa.CMPLX_PHASE_SCALING_TYPE.strip() != 'NS':
        raise ValueError(
            'Got unsupported CMPLX_PHASE_SCALING_TYPE {}'.format(
                cmetaa.CMPLX_PHASE_SCALING_TYPE))

    remap_type = cmetaa.CMPLX_MAG_REMAP_TYPE.strip()
    if remap_type == 'NS':
        if complex_order in ['IQ', 'QI']:
            return ComplexFormatFunction(raw_dtype, complex_order, band_dimension=band_dimension)
        else:
            raise ValueError(
                'Got unexpected state where cmetaa.CMPLX_MAG_REMAP_TYPE is "NS",\n\t '
                'but Band[0].ISUBCAT/Band[1].ISUBCAT = `{}`'.format(complex_order))
    elif remap_type not in ['LINM', 'LINP', 'LOGM', 'LOGP', 'LLM']:
        raise ValueError('Got unsupported CMETAA.CMPLX_MAG_REMAP_TYPE {}'.format(remap_type))

    if complex_order not in ['MP', 'PM']:
        raise ValueError(
            'Got unexpected state where cmetaa.CMPLX_MAG_REMAP_TYPE is `{}`,\n\t'
            'but Band[0].ISUBCAT/Band[1].ISUBCAT = `{}`'.format(
                remap_type, complex_order))

    scale_factor = float(cmetaa.CMPLX_LIN_SCALE)
    if remap_type == 'LINM':
        scaling_function = get_linear_magnitude_scaling(scale_factor)
    elif remap_type == 'LINP':
        scaling_function = get_linear_power_scaling(scale_factor)
    elif remap_type == 'LOGM':
        # NB: there is nowhere in the CMETAA structure to define
        #   the db_per_step value. Strangely, the use of this value is laid
        #   out in the STDI-0002 standards document, which defines CMETAA
        #   structure. We will generically use a value which maps the
        #   max uint8 value to the max int16 value.
        db_per_step = 300*numpy.log(2)/255.0
        scaling_function = get_log_magnitude_scaling(scale_factor, db_per_step)
    elif remap_type == 'LOGP':
        db_per_step = 300*numpy.log(2)/255.0
        scaling_function = get_log_power_scaling(scale_factor, db_per_step)
    elif remap_type == 'LLM':
        scaling_function = get_linlog_magnitude_scaling(
            scale_factor, int(cmetaa.CMPLX_LINLOG_TP))
    else:
        raise ValueError('Got unhandled CMETAA.CMPLX_MAG_REMAP_TYPE {}'.format(remap_type))
    return ApplyAmplitudeScalingFunction(raw_dtype, complex_order, scaling_function, band_dimension=band_dimension)


######
# The interpreter and reader objects

class ComplexNITFDetails(NITFDetails):
    """
    Details object for NITF file containing complex data.
    """

    __slots__ = (
        '_segment_status', '_segment_bands', '_sicd_meta', '_reverse_axes', '_transpose_axes')

    def __init__(
            self,
            file_name: str,
            reverse_axes: Union[None, int, Sequence[int]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF file containing a complex SICD
        reverse_axes : None|Sequence[int]
            Any entries should be restricted to `{0, 1}`. The presence of
            `0` means to reverse the rows (in the raw sense), and the presence
            of `1` means to reverse the columns (in the raw sense).
        transpose_axes : None|Tuple[int, ...]
            If presented this should be only `(1, 0)`.
        """

        self._reverse_axes = reverse_axes
        self._transpose_axes = transpose_axes
        self._segment_status = None
        self._sicd_meta = None
        self._segment_bands = None
        NITFDetails.__init__(self, file_name)
        self._find_complex_image_segments()
        if len(self.sicd_meta) == 0:
            raise SarpyIOError(
                'No complex valued image segments found in file {}'.format(file_name))

    @property
    def reverse_axes(self) -> Union[None, int, Sequence[int]]:
        return self._reverse_axes

    @property
    def transpose_axes(self) -> Optional[Tuple[int, ...]]:
        return self._transpose_axes

    @property
    def segment_status(self) -> Tuple[bool, ...]:
        """
        Tuple[bool, ...]: Where each image segment is viable for use.
        """

        return self._segment_status

    @property
    def sicd_meta(self) -> Tuple[SICDType, ...]:
        """
        Tuple[SICDType, ...]: The best inferred sicd structures.
        """

        return self._sicd_meta

    @property
    def segment_bands(self) -> Tuple[Tuple[int, Optional[int]], ...]:
        """
        This describes the structure for the output data segments from the NITF,
        with each entry of the form `(image_segment, output_band)`, where
        `output_band` will be `None` if the image segment has exactly one
        complex band.

        Returns
        -------
        Tuple[Tuple[int, Optional[int]], ...]
            The band details for use.
        """

        return self._segment_bands

    def _check_band_details(
            self,
            index: int,
            sicd_meta: List,
            segment_status: List,
            segment_bands: List):
        if len(segment_status) != index:
            raise ValueError('Inconsistent status checking state')
        image_header = self.img_headers[index]
        if image_header.ICAT.strip() not in ['SAR', 'SARIQ']:
            segment_status.append(False)
            return

        # construct a preliminary sicd
        sicd = extract_sicd(image_header, self._transpose_axes is not None)
        bands = image_header.Bands
        pvtype = image_header.PVTYPE

        # handle odd bands
        if (len(bands) % 2) == 1:
            if image_header.PVTYPE != 'C':
                # it's not complex, so we're done
                segment_status.append(False)
                return
            segment_status.append(True)
            sicd_meta.append(sicd)
            segment_bands.append((index, len(bands)))
            return

        # we have an even number of bands - ensure that the bands are marked
        #  IQ/QI/MP/PM
        order = bands[0].ISUBCAT + bands[1].ISUBCAT
        if order not in ['IQ', 'QI', 'MP', 'PM']:
            segment_status.append(False)
            return

        if len(bands) == 2:
            # this should be the most common by far

            segment_status.append(True)
            sicd_meta.append(sicd)
            segment_bands.append((index, 1))
            return

        for i in range(2, len(bands), 2):
            if order != bands[i].ISUBCAT + bands[i+1].ISUBCAT:
                logging.error(
                    'Image segment appears to multiband with switch complex ordering')
                segment_status.append(False)
                return

        if order in ['IQ', 'QI']:
            if pvtype not in ['SI', 'R']:
                logging.error(
                    'Image segment appears to be complex of order `{}`, \n\t'
                    'but PVTYPE is `{}`'.format(order, pvtype))
                segment_status.append(False)

        if order in ['MP', 'PM']:
            if pvtype not in ['INT', 'R']:
                logging.error(
                    'Image segment appears to be complex of order `{}`, \n\t'
                    'but PVTYPE is `{}`'.format(order, pvtype))
                segment_status.append(False)

        segment_status.append(True)
        sicd_meta.append(sicd)
        segment_bands.append((index, int(len(bands)/2)))

    def _find_complex_image_segments(self):
        """
        Find complex image segments.

        Returns
        -------
        None
        """

        sicd_meta = []
        segment_status = []
        segment_bands = []
        for index in range(len(self.img_headers)):
            self._check_band_details(index, sicd_meta, segment_status, segment_bands)
        self._segment_status = tuple(segment_status)
        use_sicd_meta = []
        use_segment_bands = []
        for (the_index, out_bands), sicd in zip(segment_bands, sicd_meta):
            if out_bands == 1:
                use_sicd_meta.append(sicd)
                use_segment_bands.append((the_index, None))
            else:
                for j in range(out_bands):
                    use_sicd_meta.append(sicd.copy())
                    use_segment_bands.append((the_index, j))
        self._sicd_meta = tuple(use_sicd_meta)
        self._segment_bands = tuple(use_segment_bands)


class ComplexNITFReader(NITFReader, SICDTypeReader):
    """
    A reader for complex valued NITF elements, this should be explicitly tried AFTER
    the SICDReader.
    """

    def __init__(
            self,
            nitf_details: Union[str, ComplexNITFDetails],
            reverse_axes: Union[None, int, Sequence[int]] = None,
            transpose_axes: Optional[Tuple[int, ...]] = None):
        """

        Parameters
        ----------
        nitf_details : str|ComplexNITFDetails
        reverse_axes : None|Sequence[int]
            Any entries should be restricted to `{0, 1}`. The presence of
            `0` means to reverse the rows (in the raw sense), and the presence
            of `1` means to reverse the columns (in the raw sense).
        transpose_axes : None|Tuple[int, ...]
            If presented this should be only `(1, 0)`.
        """

        if isinstance(nitf_details, str):
            nitf_details = ComplexNITFDetails(
                nitf_details, reverse_axes=reverse_axes, transpose_axes=transpose_axes)
        if not isinstance(nitf_details, ComplexNITFDetails):
            raise TypeError('The input argument for ComplexNITFReader must be a filename or '
                            'ComplexNITFDetails object.')

        SICDTypeReader.__init__(self, None, nitf_details.sicd_meta)
        NITFReader.__init__(
            self,
            nitf_details,
            reader_type="SICD",
            reverse_axes=nitf_details.reverse_axes,
            transpose_axes=nitf_details.transpose_axes)
        self._check_sizes()

    @property
    def nitf_details(self) -> ComplexNITFDetails:
        """
        ComplexNITFDetails: The NITF details object.
        """

        # noinspection PyTypeChecker
        return self._nitf_details

    def get_nitf_dict(self):
        """
        Populate a dictionary with the pertinent NITF header information. This
        is for use in more faithful preservation of NITF header information
        in copying or rewriting sicd files.

        Returns
        -------
        dict
        """

        out = {}
        security = {}
        security_obj = self.nitf_details.nitf_header.Security
        # noinspection PyProtectedMember
        for field in NITFSecurityTags._ordering:
            value = getattr(security_obj, field).strip()
            if value != '':
                security[field] = value
        if len(security) > 0:
            out['Security'] = security

        out['OSTAID'] = self.nitf_details.nitf_header.OSTAID
        out['FTITLE'] = self.nitf_details.nitf_header.FTITLE
        return out

    def populate_nitf_information_into_sicd(self):
        """
        Populate some pertinent NITF header information into the SICD structure.
        This provides more faithful copying or rewriting options.
        """

        nitf_dict = self.get_nitf_dict()
        for sicd_meta in self._sicd_meta:
            sicd_meta.NITF = copy.deepcopy(nitf_dict)

    def depopulate_nitf_information(self):
        """
        Eliminates the NITF information dict from the SICD structure.
        """

        for sicd_meta in self._sicd_meta:
            sicd_meta.NITF = {}

    def get_format_function(
            self,
            raw_dtype: numpy.dtype,
            complex_order: Optional[str],
            lut: Optional[numpy.ndarray],
            band_dimension: int,
            image_segment_index: Optional[int] = None,
            **kwargs) -> Optional[FormatFunction]:
        image_header = self.nitf_details.img_headers[image_segment_index]
        bands = len(image_header.Bands)
        if complex_order is not None and bands == 2:
            return _extract_transform_data(image_header, band_dimension)
        # TODO: strange nonstandard float16 handling?
        return NITFReader.get_format_function(
            self, raw_dtype, complex_order, lut, band_dimension, image_segment_index, **kwargs)

    def _check_image_segment_for_compliance(
            self,
            index: int,
            img_header: Union[ImageSegmentHeader, ImageSegmentHeader0]) -> bool:
        return self.nitf_details.segment_status[index]

    def find_image_segment_collections(self) -> Tuple[Tuple[int, ...]]:
        return tuple((entry[0], ) for entry in self.nitf_details.segment_bands)

    def create_data_segment_for_collection_element(self, collection_index: int) -> DataSegment:
        the_index, the_band = self.nitf_details.segment_bands[collection_index]
        if the_index not in self._image_segment_data_segments:
            data_segment = self.create_data_segment_for_image_segment(the_index, apply_format=True)
        else:
            data_segment = self._image_segment_data_segments[the_index]

        if the_band is None:
            return data_segment
        else:
            return SubsetSegment(
                data_segment, (slice(None, None, 1), slice(None, None, 1), slice(the_band, the_band+1, 1)),
                'formatted', close_parent=True)


def final_attempt(file_name: str) -> Optional[ComplexNITFReader]:
    """
    Contingency check to open for some other complex NITF type file.
    Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str|BinaryIO
        the file_name to check

    Returns
    -------
    ComplexNITFReader|None
    """

    if is_file_like(file_name):
        return None

    try:
        nitf_details = ComplexNITFDetails(file_name)
        logger.info('File {} is determined to be some other format complex NITF.')
        return ComplexNITFReader(nitf_details)
    except (SarpyIOError, ValueError):
        return None
