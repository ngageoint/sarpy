"""
Work in progress for reading some other kind of complex NITF.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from typing import Union

from datetime import datetime
import numpy
from scipy.constants import foot

from sarpy.geometry.geocoords import geodetic_to_ecf, ned_to_ecf
from sarpy.geometry.latlon import num as lat_lon_parser

from sarpy.io.general.base import SarpyIOError
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

# NB: DO NOT implement is_a() here. This will explicitly happen after other readers

def final_attempt(file_name):
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


########
# Define sicd structure from image sub-header information

def extract_sicd(img_header, symmetry, nitf_header=None):
    """
    Extract the best available SICD structure from relevant nitf header structures.

    Parameters
    ----------
    img_header : ImageSegmentHeader|ImageSegmentHeader0
    symmetry : tuple
    nitf_header : None|NITFHeader|NITFHeader0

    Returns
    -------
    SICDType
    """

    def get_collection_info():
        # type: () -> CollectionInfoType
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

    def get_image_data():
        # type: () -> ImageDataType
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

        if symmetry[2]:
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

    def append_country_code(cc):
        if len(cc) > 0:
            if the_sicd.CollectionInfo is None:
                the_sicd.CollectionInfo = CollectionInfoType(CountryCodes=[cc, ])
            elif the_sicd.CollectionInfo.CountryCodes is None:
                the_sicd.CollectionInfo.CountryCodes = [cc, ]
            elif cc not in the_sicd.CollectionInfo.CountryCodes:
                the_sicd.CollectionInfo.CountryCodes.append(cc)

    def set_image_corners(icps, override=False):
        if the_sicd.GeoData is None:
             the_sicd.GeoData = GeoDataType(ImageCorners=icps)
        elif the_sicd.GeoData.ImageCorners is None or override:
            the_sicd.GeoData.ImageCorners = icps

    def set_arp_position(arp_ecf, override=False):
        if the_sicd.SCPCOA is None:
            the_sicd.SCPCOA = SCPCOAType(ARPPos=arp_ecf)
        elif override:
            # prioritize this information first - it should be more reliable than other sources
            the_sicd.SCPCOA.ARPPos = arp_ecf

    def set_scp(scp_ecf, scp_pixel, override=False):
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

    def set_collect_start(collect_start, override=False):
        if the_sicd.Timeline is None:
            the_sicd.Timeline = TimelineType(CollectStart=collect_start)
        elif the_sicd.Timeline.CollectStart is None or override:
            the_sicd.Timeline.CollectStart = collect_start

    def set_uvects(row_unit, col_unit):
        if the_sicd.Grid is None:
            the_sicd.Grid = GridType(Row=DirParamType(UVectECF=row_unit),
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

    def try_CMETAA():
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
            date_time = '{}-{}-{}T{}:{}:{}'.format(
                date_str[:4], date_str[4:6], date_str[6:8],
                time_str[:2], time_str[2:4], time_str[4:6])
            the_sicd.Timeline.CollectStart = numpy.datetime64(date_time, 'us')
        except:
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

    def try_AIMIDA():
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

    def try_AIMIDB():
        tre = None if tres is None else tres['AIMIDB']
        if tre is None:
            return
        aimidb = tre.DATA

        append_country_code(aimidb.COUNTRY.strip())

        if the_sicd.ImageFormation is not None and the_sicd.ImageFormation.SegmentIdentifier is None:
            the_sicd.ImageFormation.SegmentIdentifier = aimidb.CURRENT_SEGMENT.strip()

        date_str = aimidb.ACQUISITION_DATE
        collect_start = numpy.datetime64('{}-{}-{}T{}:{}:{}'.format(
            date_str[:4], date_str[4:6], date_str[6:8],
            date_str[8:10], date_str[10:12], date_str[12:14]), 'us')
        set_collect_start(collect_start, override=False)

    def try_ACFT():
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

    def try_BLOCKA():
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
            except:
                lat_val = lat_lon_parser(value[:10])
                lon_val = lat_lon_parser(value[10:21])
            icps.append([lat_val, lon_val])
        set_image_corners(icps, override=False)

    def try_MPDSRA():
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

    def try_MENSRB():
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

    def try_MENSRA():
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

    def extract_corners():
        icps = extract_image_corners(img_header)
        if icps is None:
            return
        # TODO: include symmetry transform issue
        set_image_corners(icps, override=False)

    def extract_start():
        # noinspection PyBroadException
        try:
            date_str = img_header.IDATIM
            collect_start = numpy.datetime64('{}-{}-{}T{}:{}:{}'.format(
                date_str[:4], date_str[4:6], date_str[6:8],
                date_str[8:10], date_str[10:12], date_str[12:14]), 'us')
        except:
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

def get_linear_magnitude_scaling(scale_factor):
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


def get_qi_handler():
    """
    Get the QI band handler.

    Returns
    -------
    callable
    """

    # TODO: scaling function?

    def qi_handler(data):
        out = numpy.zeros((data.shape[0], data.shape[1], int(data.shape[2]/2)), dtype=numpy.complex64)
        out.real = data[:, :, 1::2]
        out.imag = data[:, :, 0::2]
        return out

    return qi_handler


def get_mp_handler(scaling_function=None):

    def mp_handler(data):
        out = numpy.zeros((data.shape[0], data.shape[1], int(data.shape[2]/2)), dtype=numpy.complex64)
        if data.dtype.name == 'uint8':
            theta = data[:, :, 1::2]*(2*numpy.pi/256)
        elif data.dtype.name == 'uint16':
            theta = data[:, :, 1::2] * (2 * numpy.pi / 65536)
        elif data.dtype.name == 'float32':
            theta = data[:, :, 1::2]
        else:
            raise ValueError('Got unsupported dtype {}'.format(data.dtype.name))

        amp = data[:, :, 0::2] if scaling_function is None else scaling_function(data[:, :, 0::2])
        out.real = amp*numpy.cos(theta)
        out.imag = amp*numpy.sin(theta)

        out.real = data[:, :, 1::2]
        out.imag = data[:, :, 0::2]
        return out
    return mp_handler


def get_pm_handler(scaling_function=None):

    def pm_handler(data):
        out = numpy.zeros((data.shape[0], data.shape[1], int(data.shape[2]/2)), dtype=numpy.complex64)
        if data.dtype.name == 'uint8':
            theta = data[:, :, 0::2]*(2*numpy.pi/256)
        elif data.dtype.name == 'uint16':
            theta = data[:, :, 0::2] * (2 * numpy.pi / 65536)
        elif data.dtype.name == 'float32':
            theta = data[:, :, 0::2]
        else:
            raise ValueError('Got unsupported dtype {}'.format(data.dtype.name))

        amp = data[:, :, 1::2] if scaling_function is None else scaling_function(data[:, :, 1::2])
        out.real = amp*numpy.cos(theta)
        out.imag = amp*numpy.sin(theta)

        out.real = data[:, :, 1::2]
        out.imag = data[:, :, 0::2]
        return out
    return pm_handler


def _extract_transform_data(img_header):
    """
    Helper function for defining necessary transform_data definition for
    interpreting image segment data.

    Parameters
    ----------
    img_header : ImageSegmentHeader|ImageSegmentHeader0

    Returns
    -------
    None|str|callable
    """

    if len(img_header.Bands) != 2:
        raise ValueError('Got unhandled case of {} image bands'.format(len(img_header.Bands)))

    isubcat_0 = img_header.Bands[0].ISUBCAT
    isubcat_1 = img_header.Bands[1].ISUBCAT
    # noinspection PyUnresolvedReferences
    tre = None if img_header.ExtendedHeader.data is None else \
        img_header.ExtendedHeader.data['CMETAA']  # type: Union[None, CMETAA]
    if tre is None:
        if isubcat_0 == 'I' and isubcat_1 == 'Q':
            return 'COMPLEX'
        elif isubcat_0 == 'Q' and isubcat_1 == 'I':
            return get_qi_handler()
        elif isubcat_0 == 'M' and isubcat_1 == 'P':
            return get_mp_handler()
        elif isubcat_0 == 'P' and isubcat_1 == 'M':
            return get_pm_handler()
        return None
    else:
        cmetaa = tre.DATA
        if cmetaa.CMPLX_PHASE_SCALING_TYPE.strip() != 'NS':
            raise ValueError(
                'Got unsupported CMPLX_PHASE_SCALING_TYPE {}'.format(
                    cmetaa.CMPLX_PHASE_SCALING_TYPE))

        remap_type = cmetaa.CMPLX_MAG_REMAP_TYPE.strip()
        if remap_type == 'NS':
            if isubcat_0 == 'I' and isubcat_1 == 'Q':
                return 'COMPLEX'
            elif isubcat_0 == 'Q' and isubcat_1 == 'I':
                return get_qi_handler()
            else:
                raise ValueError(
                    'Got unexpected state where cmetaa.CMPLX_MAG_REMAP_TYPE is "NS", '
                    'but image_header.Band[0].ISUBCAT = "{}" and '
                    'image_header.Band[0].ISUBCAT = "{}"'.format(isubcat_0, isubcat_1))
        elif remap_type not in ['LINM', 'LINP', 'LOGM', 'LOGP', 'LLM']:
            raise ValueError('Got unsupported CMETAA.CMPLX_MAG_REMAP_TYPE {}'.format(remap_type))

        combined_subcat = isubcat_0+isubcat_1
        if combined_subcat not in ['MP', 'PM']:
            raise ValueError(
                'Got unexpected state where cmetaa.CMPLX_MAG_REMAP_TYPE is "{}", '
                'but image_header.Band[0].ISUBCAT = "{}" and '
                'image_header.Band[0].ISUBCAT = "{}"'.format(remap_type, isubcat_0, isubcat_1))

        scale_factor = float(cmetaa.CMPLX_LIN_SCALE)
        if remap_type == 'LINM':
            scaling_function = get_linear_magnitude_scaling(scale_factor)
        elif remap_type == 'LINP':
            scaling_function = get_log_magnitude_scaling(scale_factor)
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

        return get_mp_handler(scaling_function) if combined_subcat == 'MP' \
            else get_pm_handler(scaling_function)


######
# The interpreter and reader objects

class ComplexNITFDetails(NITFDetails):
    """
    Details object for NITF file containing complex data.
    """

    __slots__ = ('_complex_segments', '_sicd_meta', '_symmetry', '_split_bands')

    def __init__(self, file_name, symmetry=(False, False, False), split_bands=True):
        """

        Parameters
        ----------
        file_name : str
            file name for a NITF file containing a complex SICD
        symmetry : tuple
        split_bands : bool
            Split multiple complex bands into single bands?
        """

        self._split_bands = split_bands
        self._symmetry = symmetry
        self._sicd_meta = None
        self._complex_segments = None
        super(ComplexNITFDetails, self).__init__(file_name)
        if self._nitf_header.ImageSegments.subhead_sizes.size == 0:
            raise SarpyIOError('There are no image segments defined.')
        self._find_complex_image_segments()
        if self._complex_segments is None:
            raise SarpyIOError('No complex valued (I/Q) image segments found in file {}'.format(file_name))

    @property
    def complex_segments(self):
        """
        List[dict]: The image details for each relevant image segment.
        """

        return self._complex_segments

    @property
    def sicd_meta(self):
        """
        Tuple[SICDType]: The best inferred sicd structures.
        """

        return self._sicd_meta

    def _find_complex_image_segments(self):
        """
        Find complex image segments.

        Returns
        -------
        None
        """

        def extract_band_details(index, image_header):
            # type: (int, Union[ImageSegmentHeader, ImageSegmentHeader0]) -> None

            # populate sicd_meta and complex_segments appropriately
            if image_header.ICAT.strip() not in ['SAR', 'SARIQ']:
                return

            if image_header.NBPP not in [8, 16, 32, 64]:
                return  # this should be redundant with general NITF checking

            sicd = extract_sicd(img_header, self._symmetry)

            bands = len(image_header.Bands)
            if bands == 1:
                if image_header.PVTYPE != 'C':
                    return  # this is not a complex dataset - maybe we should warn?
                if image_header.NBPP == 32:
                    logger.warning(
                        'File {} has image band at index {} of complex data type with 32 bits per pixel\n\t'
                        '(real/imaginary components) consisting of 16-bit floating points.\n\t'
                        'This is experimentally supported assuming the data follows the\n\t'
                        'ieee standard for half precision floating point, i.e. 1 bit sign, 5 bits exponent,\n\t'
                        'and 10 bits significand/mantissa'.format(self.file_name, index))
                    sicd_meta.append(sicd)
                    complex_segments.append({
                        'index': index,
                        'raw_dtype': numpy.dtype('>f2'),
                        'raw_bands': 2,
                        'transform_data': 'COMPLEX',
                        'output_bands': 1,
                        'output_dtype': numpy.dtype('>c8')})
                elif image_header.NBPP == 64:
                    sicd_meta.append(sicd)
                    complex_segments.append({
                        'index': index,
                        'raw_dtype': numpy.dtype('>c8'),
                        'raw_bands': 1,
                        'transform_data': None,
                        'output_bands': 1,
                        'output_dtype': numpy.dtype('>c8')})
                elif image_header.NBPP == 128:
                    sicd_meta.append(sicd)
                    complex_segments.append({
                        'index': index,
                        'raw_dtype': numpy.dtype('>c16'),
                        'raw_bands': 1,
                        'transform_data': None,
                        'output_bands': 1,
                        'output_dtype': numpy.dtype('>c16')})
                else:
                    logger.error(
                        'File {} has image band at index {} of complex type with bits per pixel value {}.\n\t'
                        'This is not currently supported and this band will be skipped.'.format(
                            self.file_name, index, image_header.NBPP))
                return

            # do some general assignment for input datatype
            bpp = int(image_header.NBPP/8)
            pv_type = image_header.PVTYPE
            if pv_type == 'INT':
                raw_dtype = '>u{}'.format(bpp)
            elif pv_type == 'SI':
                raw_dtype = '>i{}'.format(bpp)
            elif pv_type == 'R':
                raw_dtype ='>f{}'.format(bpp)
            else:
                logger.warning(
                    'Got unhandled PVTYPE {} for image band {}\n\t'
                    'in file {}. Skipping'.format(pv_type, index, self.file_name))
                return

            if bands == 2:
                # this should be the most common by far
                transform_data = _extract_transform_data(image_header)
                if transform_data is not None:
                    sicd_meta.append(sicd)
                    complex_segments.append({
                        'index': index,
                        'raw_dtype': raw_dtype,
                        'raw_bands': 2,
                        'transform_data': transform_data,
                        'output_bands': 1,
                        'output_dtype': numpy.complex64})
            elif (bands % 2) == 0:
                # this is explicitly to support the somewhat unusual RCM NITF format
                cont = True
                for j in range(0, bands, 2):
                    cont &= (image_header.Bands[j].ISUBCAT == 'I'
                             and image_header.Bands[j+1].ISUBCAT == 'Q')
                if self._split_bands and bands > 2:
                    for j in range(0, bands, 2):
                        complex_segments.append({
                            'index': index,
                            'raw_dtype': raw_dtype,
                            'raw_bands': bands,
                            'transform_data': 'COMPLEX',
                            'output_bands': 1,
                            'output_dtype': numpy.complex64,
                            'limit_to_raw_bands': numpy.array([j, j+1], dtype='int32')})
                        sicd_meta.append(sicd.copy())
                else:
                    sicd_meta.append(sicd)
                    complex_segments.append({
                        'index': index,
                        'raw_dtype': raw_dtype,
                        'raw_bands': bands,
                        'transform_data': 'COMPLEX',
                        'output_bands': int(bands/2),
                        'output_dtype': numpy.complex64})

            # ['raw_dtype', 'raw_bands', 'transform_data', 'output_bands', 'output_dtype', 'limit_to_raw_bands']
            if image_header.ICAT.strip() in ['SAR', 'SARIQ'] and ((bands % 2) == 0):
                # TODO: account for PVType == 'C' and ISUBCAT = 'M'/'P'
                cont = True
                for j in range(0, bands, 2):
                    cont &= (image_header.Bands[j].ISUBCAT == 'I'
                             and image_header.Bands[j+1].ISUBCAT == 'Q')
                return bands
            return 0

        sicd_meta = []
        complex_segments = []
        for i, img_header in enumerate(self.img_headers):
            extract_band_details(i, img_header)

        if len(sicd_meta) > 0:
            self._complex_segments = complex_segments
            self._sicd_meta = tuple(sicd_meta)


class ComplexNITFReader(NITFReader, SICDTypeReader):
    """
    A reader for complex valued NITF elements, this should be explicitly tried AFTER
    the SICDReader.
    """

    def __init__(self, nitf_details, symmetry=(False, False, False), split_bands=True):
        """

        Parameters
        ----------
        nitf_details : str|ComplexNITFDetails
        symmetry : tuple
            Passed through to ComplexNITFDetails() in the event that `nitf_details` is a file name.
        split_bands : bool
            Passed through to ComplexNITFDetails() in the event that `nitf_details` is a file name.
        """

        if isinstance(nitf_details, str):
            nitf_details = ComplexNITFDetails(nitf_details, symmetry=symmetry, split_bands=split_bands)
        if not isinstance(nitf_details, ComplexNITFDetails):
            raise TypeError('The input argument for ComplexNITFReader must be a filename or '
                            'ComplexNITFDetails object.')

        SICDTypeReader.__init__(self, nitf_details.sicd_meta)
        NITFReader.__init__(self, nitf_details, reader_type="SICD", symmetry=symmetry)
        self._check_sizes()

    @property
    def nitf_details(self):
        # type: () -> ComplexNITFDetails
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
            sicd_meta.NITF = nitf_dict

    def depopulate_nitf_information(self):
        """
        Eliminates the NITF information dict from the SICD structure.
        """

        for sicd_meta in self._sicd_meta:
            sicd_meta.NITF = {}

    def _find_segments(self):
        return [[entry['index'], ] for entry in self.nitf_details.complex_segments]

    def _construct_chipper(self, segment, index):
        entry = self.nitf_details.complex_segments[index]
        if entry['index'] != segment[0]:
            raise ValueError('Got incompatible entries.')

        kwargs = {}
        for key in ['raw_dtype', 'raw_bands', 'transform_data', 'output_bands', 'output_dtype', 'limit_to_raw_bands']:
            if key in entry:
                kwargs[key] = entry[key]
        return self._define_chipper(entry['index'], **kwargs)
