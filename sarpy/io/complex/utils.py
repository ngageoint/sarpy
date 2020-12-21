# -*- coding: utf-8 -*-
"""
Common functionality for converting metadata
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from typing import Union, Iterator

from datetime import datetime
import numpy
from numpy.polynomial import polynomial
from scipy.constants import foot

from sarpy.geometry.geocoords import geodetic_to_ecf, ned_to_ecf
from sarpy.geometry.latlon import num as lat_lon_parser

from sarpy.io.general.nitf import extract_image_corners
from sarpy.io.general.nitf_elements.image import ImageSegmentHeader, ImageSegmentHeader0
from sarpy.io.general.nitf_elements.nitf_head import NITFHeader, NITFHeader0
from sarpy.io.general.nitf_elements.base import TREList

from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.blocks import Poly2DType
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
from sarpy.io.general.nitf_elements.tres.unclass.CMETAA import CMETAA


def two_dim_poly_fit(x, y, z, x_order=2, y_order=2, x_scale=1, y_scale=1, rcond=None):
    """
    Perform fit of data to two dimensional polynomial.

    Parameters
    ----------
    x : numpy.ndarray
        the x data
    y : numpy.ndarray
        the y data
    z : numpy.ndarray
        the z data
    x_order : int
        the order for x
    y_order : int
        the order for y
    x_scale : float
        In order to help the fitting problem to become better conditioned, the independent
        variables can be scaled, the fit performed, and then the solution rescaled.
    y_scale : float
    rcond : None|float
        passed through to :func:`numpy.linalg.lstsq`.
    Returns
    -------
    numpy.ndarray
        the coefficient array
    """

    if not isinstance(x, numpy.ndarray) or not isinstance(y, numpy.ndarray) or not isinstance(z, numpy.ndarray):
        raise TypeError('x, y, z must be numpy arrays')
    if (x.size != z.size) or (y.size != z.size):
        raise ValueError('x, y, z must have the same cardinality size.')

    x = x.flatten()*x_scale
    y = y.flatten()*y_scale
    z = z.flatten()
    # first, we need to formulate this as A*t = z
    # where A has shape (x.size, (x_order+1)*(y_order+1))
    # and t has shape ((x_order+1)*(y_order+1), )
    A = numpy.empty((x.size, (x_order+1)*(y_order+1)), dtype=numpy.float64)
    for i, index in enumerate(numpy.ndindex((x_order+1, y_order+1))):
        A[:, i] = numpy.power(x, index[0])*numpy.power(y, index[1])
    # perform least squares fit
    sol, residuals, rank, sing_values = numpy.linalg.lstsq(A, z, rcond=rcond)
    if len(residuals) != 0:
        residuals /= float(x.size)
    sol = numpy.power(x_scale, numpy.arange(x_order+1))[:, numpy.newaxis] * \
          numpy.reshape(sol, (x_order+1, y_order+1)) * \
          numpy.power(y_scale, numpy.arange(y_order+1))
    return sol, residuals, rank, sing_values


def get_im_physical_coords(array, grid, image_data, direction):
    """
    Converts one dimension of "pixel" image (row or column) coordinates to
    "physical" image (range or azimuth in meters) coordinates, for use in the
    various two-variable sicd polynomials.

    Parameters
    ----------
    array : numpy.array|float|int
        either row or col coordinate component
    grid : sarpy.io.complex.sicd_elements.Grid.GridType
    image_data : sarpy.io.complex.sicd_elements.ImageData.ImageDataType
    direction : str
        one of 'Row' or 'Col' (case insensitive) to determine which
    Returns
    -------
    numpy.array|float
    """

    if direction.upper() == 'ROW':
        return (array - image_data.SCPPixel.Row + image_data.FirstRow)*grid.Row.SS
    elif direction.upper() == 'COL':
        return (array - image_data.SCPPixel.Col + image_data.FirstCol)*grid.Col.SS
    else:
        raise ValueError('Unrecognized direction {}'.format(direction))


def fit_time_coa_polynomial(inca, image_data, grid, dop_rate_scaled_coeffs, poly_order=2):
    """

    Parameters
    ----------
    inca : sarpy.io.complex.sicd_elements.RMA.INCAType
    image_data : sarpy.io.complex.sicd_elements.ImageData.ImageDataType
    grid : sarpy.io.complex.sicd_elements.Grid.GridType
    dop_rate_scaled_coeffs : numpy.ndarray
        the dop rate polynomial relative to physical coordinates - the is a
        common construct in converting metadata for csk/sentinel/radarsat
    poly_order : int
        the degree of the polynomial to fit.
    Returns
    -------
    Poly2DType
    """

    grid_samples = poly_order + 8
    coords_az = get_im_physical_coords(
        numpy.linspace(0, image_data.NumCols - 1, grid_samples, dtype='float64'), grid, image_data, 'col')
    coords_rg = get_im_physical_coords(
        numpy.linspace(0, image_data.NumRows - 1, grid_samples, dtype='float64'), grid, image_data, 'row')
    coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)
    time_ca_sampled = inca.TimeCAPoly(coords_az_2d)
    dop_centroid_sampled = inca.DopCentroidPoly(coords_rg_2d, coords_az_2d)
    doppler_rate_sampled = polynomial.polyval(coords_rg_2d, dop_rate_scaled_coeffs)
    time_coa_sampled = time_ca_sampled + dop_centroid_sampled / doppler_rate_sampled
    coefs, residuals, rank, sing_values = two_dim_poly_fit(
        coords_rg_2d, coords_az_2d, time_coa_sampled,
        x_order=poly_order, y_order=poly_order, x_scale=1e-3, y_scale=1e-3, rcond=1e-40)
    logging.info('The time_coa_fit details:\nroot mean square residuals = {}\nrank = {}\nsingular values = {}'.format(residuals, rank, sing_values))
    return Poly2DType(Coefs=coefs)


def snr_to_rniirs(bandwidth_area, signal, noise):
    """
    Calculate the information_density and RNIIRS estimate from bandwidth area and
    signal/noise estimates.

    It is assumed that geometric effects for signal and noise have been accounted for
    (i.e. use SigmaZeroSFPoly), and signal and noise have each been averaged to a
    single pixel value.

    This mapping has been empirically determined by fitting Shannon-Hartley channel
    capacity to RNIIRS for some sample images.

    Parameters
    ----------
    bandwidth_area : float
    signal : float
    noise : float

    Returns
    -------
    (float, float)
        The information_density and RNIIRS
    """

    information_density = bandwidth_area*numpy.log2(1 + signal/noise)

    a = numpy.array([3.7555, .3960], dtype=numpy.float64)
    # we have empirically fit so that
    #   rniirs = a_0 + a_1*log_2(information_density)

    # note that if information_density is sufficiently small, it will
    # result in negative values in the above functional form. This would be
    # invalid for RNIIRS by definition, so we must avoid this case.

    # We transition to a linear function of information_density
    # below a certain point. This point will be chosen to be the (unique) point
    # at which the line tangent to the curve intersects the origin, and the
    # linear approximation below that point will be defined by this tangent line.

    # via calculus, we can determine analytically where that happens
    # rniirs_transition = a[1]/numpy.log(2)
    iim_transition = numpy.exp(1 - numpy.log(2)*a[0]/a[1])
    slope = a[1]/(iim_transition*numpy.log(2))

    if information_density > iim_transition:
        return information_density, a[0] + a[1]*numpy.log2(information_density)
    else:
        return information_density, slope*information_density


def fit_position_xvalidation(time_array, position_array, velocity_array, max_degree=5):
    """
    Empirically fit the polynomials for the X, Y, Z ECF position array, using cross
    validation with the velocity array to determine the best fit degree up to a
    given maximum degree.

    Parameters
    ----------
    time_array : numpy.ndarray
    position_array : numpy.ndarray
    velocity_array : numpy.ndarray
    max_degree : int

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray,)
        The X, Y, Z polynomial coefficients.
    """

    if not isinstance(time_array, numpy.ndarray) or \
            not isinstance(position_array, numpy.ndarray) or \
            not isinstance(velocity_array, numpy.ndarray):
        raise TypeError('time_array, position_array, and velocity_array must be numpy.ndarray instances.')

    if time_array.ndim != 1 and time_array.size > 1:
        raise ValueError('time_array must be one-dimensional with at least 2 elements.')

    if position_array.shape != velocity_array.shape:
        raise ValueError('position_array and velocity_array must have the same shape.')

    if position_array.shape[0] != time_array.size:
        raise ValueError('The first dimension of position_array must be the same size as time_array.')

    if position_array.shape[1] != 3:
        raise ValueError('The second dimension of position array must have size 3, '
                         'representing X, Y, Z ECF coordinates.')

    max_degree = int(max_degree)
    if max_degree < 1:
        raise ValueError('max_degree must be at least 1.')
    if max_degree > 10:
        logging.warning('max_degree greater than 10 for polynomial fitting may lead '
                        'to poorly conditioned (i.e. badly behaved) fit.')

    deg = 1
    prev_vel_error = numpy.inf
    P_x, P_y, P_z = None, None, None
    while deg <= min(max_degree, position_array.shape[0]-1):
        # fit position
        P_x = polynomial.polyfit(time_array, position_array[:, 0], deg=deg)
        P_y = polynomial.polyfit(time_array, position_array[:, 1], deg=deg)
        P_z = polynomial.polyfit(time_array, position_array[:, 2], deg=deg)
        # extract estimated velocities
        vel_est = numpy.hstack(
            (polynomial.polyval(time_array, polynomial.polyder(P_x))[:, numpy.newaxis],
             polynomial.polyval(time_array, polynomial.polyder(P_y))[:, numpy.newaxis],
             polynomial.polyval(time_array, polynomial.polyder(P_z))[:, numpy.newaxis]))
        # check our velocity error
        vel_err = vel_est - velocity_array
        cur_vel_error = numpy.sum((vel_err * vel_err))
        deg += 1
        # stop if the error is not smaller than at the previous step
        if cur_vel_error >= prev_vel_error:
            break
    return P_x, P_y, P_z


def sicd_reader_iterator(reader, partitions=None, polarization=None, band=None):
    """
    Provides an iterator over a collection of partitions (tuple of tuple of integer
    indices for the reader) for a sicd type reader object.

    Parameters
    ----------
    reader : BaseReader
    partitions : None|tuple
        The partitions collection. If None, then partitioning from
        `reader.get_sicd_partitions()` will be used.
    polarization : None|str
        The polarization string to match.
    band : None|str
        The band to match.

    Returns
    -------
    Iterator[tuple]
        Yields the partition index, the sicd reader index, and the sicd structure.
    """

    def sicd_match():
        match = True
        if band is not None:
            match &= (this_sicd.get_transmit_band_name() == band)
        if polarization is not None:
            match &= (this_sicd.get_processed_polarization() == polarization)
        return match

    from sarpy.io.general.base import BaseReader  # avoid circular import issue

    if not isinstance(reader, BaseReader):
        raise TypeError('reader must be an instance of BaseReader. Got type {}'.format(type(reader)))
    if reader.reader_type != "SICD":
        raise ValueError('The provided reader must be of SICD type.')

    if partitions is None:
        partitions = reader.get_sicd_partitions()
    the_sicds = reader.get_sicds_as_tuple()
    for this_partition, entry in enumerate(partitions):
        for this_index in entry:
            this_sicd = the_sicds[this_index]
            if sicd_match():
                yield this_partition, this_index, this_sicd


#######
# Extract SICD structure from nitf header information
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
                logging.warning(
                    'This NITF has complex bands that are not 64-bit. This is not '
                    'currently supported.')
            pixel_type = 'RE32F_IM32F'
        elif pvtype == 'R':
            if img_header.NBPP == 64:
                logging.warning(
                    'The real/imaginary data in the NITF are stored as 64-bit floating '
                    'point. The closest Pixel Type, RE32F_IM32F, will be used, '
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
