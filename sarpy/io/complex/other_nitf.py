# -*- coding: utf-8 -*-
"""
Work in progress for reading some other kind of complex NITF as a pseudo-sicd.
Note that this should happen as a fall-back from an actual SICD.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from typing import Union
from datetime import datetime

import numpy
from scipy.constants import foot

from sarpy.compliance import string_types
from sarpy.geometry.geocoords import geodetic_to_ecf, ned_to_ecf
from sarpy.geometry.latlon import num as lat_lon_parser

from sarpy.io.general.nitf import NITFDetails, extract_image_corners, NITFReader
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader
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
from sarpy.io.general.nitf_elements.base import TREList

from sarpy.io.general.nitf_elements.tres.unclass.CMETAA import CMETAA


# NB: The checking for open_complex() is included in the sicd.is_a().

def _extract_sicd(img_header, symmetry):
    """
    Extract the best available SICD structure from the given image header.

    Parameters
    ----------
    img_header : ImageSegmentHeader
    symmetry : tuple

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
            pixel_type = 'RE32F_IM32F'
        elif pvtype == 'R':
            if img_header.NBPP == 64:
                logging.warning(
                    'The real/complex data in the NITF are stored as 64-bit floating '
                    'point. The closest Pixel Type is RE32F_IM32F will be used, '
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
        if cmetaa.CMPLX_SIGNAL_PLANE[0].upper() == 'S':
            the_sicd.Grid.ImagePlane = 'SLANT'
        elif cmetaa.CMPLX_SIGNAL_PLANE[0].upper() == 'G':
            the_sicd.Grid.ImagePlane = 'GROUND'
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
        cmplx_weight = cmetaa.CMPLX_WEIGHT
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
                    'SLL': '{0:0.16G}'.format(-float(cmetaa.CMPLX_RNG_SLL)),
                    'NBAR': '{0:0.16G}'.format(float(cmetaa.CMPLX_RNG_TAY_NBAR))})
            the_sicd.Grid.Col.WgtType = WgtTypeType(
                WindowName='TAYLOR',
                Parameters={
                    'SLL': '{0:0.16G}'.format(-float(cmetaa.CMPLX_AZ_SLL)),
                    'NBAR': '{0:0.16G}'.format(float(cmetaa.CMPLX_AZ_TAY_NBAR))})
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
            file name for a NITF 2.1 file containing a SICD
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
            raise IOError('There are no image segments defined.')
        self._find_complex_image_segments()
        if self._complex_segments is None:
            raise IOError('No complex valued (I/Q) image segments found in file {}'.format(file_name))

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

        def extract_band_details(image_header):
            # type: (ImageSegmentHeader) -> int
            bands = len(image_header.Bands)
            if image_header.ICAT.strip() in ['SAR', 'SARIQ'] and ((bands % 2) == 0):
                cont = True
                for j in range(0, bands, 2):
                    cont &= (image_header.Bands[j].ISUBCAT == 'I'
                             and image_header.Bands[j+1].ISUBCAT == 'Q')
                return bands
            return 0

        sicd_meta = []
        complex_segments = []
        for i, img_header in enumerate(self.img_headers):
            complex_bands = extract_band_details(img_header)
            if complex_bands > 0:
                sicd = _extract_sicd(img_header, self._symmetry)
                if self._split_bands and complex_bands > 2:
                    for j in range(0, complex_bands, 2):
                        complex_segments.append(
                            {'index': i, 'output_bands': 1, 'limit_to_raw_bands': numpy.array([j, j+1], dtype='int32')})
                        sicd_meta.append(sicd.copy())
                else:
                    complex_segments.append({'index': i, })
                    sicd_meta.append(sicd)

        if len(sicd_meta) > 0:
            self._complex_segments = complex_segments
            self._sicd_meta = tuple(sicd_meta)


class ComplexNITFReader(NITFReader):
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

        if isinstance(nitf_details, string_types):
            nitf_details = ComplexNITFDetails(nitf_details, symmetry=symmetry, split_bands=split_bands)
        if not isinstance(nitf_details, ComplexNITFDetails):
            raise TypeError('The input argument for ComplexNITFReader must be a filename or '
                            'ComplexNITFDetails object.')
        super(ComplexNITFReader, self).__init__(nitf_details, is_sicd_type=True, symmetry=symmetry)

    @property
    def nitf_details(self):
        # type: () -> ComplexNITFDetails
        """
        ComplexNITFDetails: The NITF details object.
        """

        # noinspection PyTypeChecker
        return self._nitf_details

    def _find_segments(self):
        return [[entry['index'], ] for entry in self.nitf_details.complex_segments]

    def _construct_chipper(self, segment, index):
        entry = self.nitf_details.complex_segments[index]
        if entry['index'] != segment[0]:
            raise ValueError('Got incompatible entries.')
        return self._define_chipper(
            entry['index'], output_bands=entry.get('output_bands', None),
            limit_to_raw_bands=entry.get('limit_to_raw_bands', None))
