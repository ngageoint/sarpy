# -*- coding: utf-8 -*-
"""
Functionality for reading Cosmo Skymed data into a SICD model.
"""

from collections import OrderedDict
from typing import Tuple, Dict
import warnings

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

try:
    import h5py
except ImportError:
    h5py = None
    warnings.warn('The h5py module is not successfully imported, '
                  'which precludes Cosmo Skymed reading capability!')

from .sicd_elements.blocks import Poly1DType, Poly2DType, RowColType
from .sicd_elements.SICD import SICDType
from .sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from .sicd_elements.ImageCreation import ImageCreationType
from .sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType, TxStepType
from .sicd_elements.ImageData import ImageDataType
from .sicd_elements.GeoData import GeoDataType, SCPType
from .sicd_elements.SCPCOA import SCPCOAType
from .sicd_elements.Position import PositionType, XYZPolyType
from .sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from .sicd_elements.Timeline import TimelineType, IPPSetType
from .sicd_elements.ImageFormation import ImageFormationType, TxFrequencyProcType
from .sicd_elements.RMA import RMAType, INCAType
from .sicd_elements.Radiometric import RadiometricType
from ...geometry import point_projection
from .radarsat import _2d_poly_fit
from .base import BaseChipper, BaseReader

__classification__ = "UNCLASSIFIED"


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a Cosmo Skymed file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    CSKReader|None
        `CSKReader` instance if Cosmo Skymed file, `None` otherwise
    """

    if h5py is None:
        return None

    try:
        csk_details = CSKDetails(file_name)
        print('File {} is determined to be a Cosmo Skymed file.'.format(file_name))
        return CSKReader(csk_details)
    except (IOError, KeyError, ValueError):
        # TODO: what all should we catch?
        return None


##########
# helper functions

def _extract_attrs(h5_element):
    out = OrderedDict()
    for key in h5_element.attrs:
        val = h5_element.attrs[key]
        out[key] = val.decode('utf-8') if isinstance(val, bytes) else val
    return out


def _get_seconds(dt1, dt2):  # type: (numpy.datetime64, numpy.datetime64) -> float
    tdt1 = dt1.astype('datetime64[ns]')
    tdt2 = dt2.astype('datetime64[ns]')  # convert both to nanosecond precision
    return (tdt1.astype('int64') - tdt2.astype('int64'))*1e-9


###########
# parser and interpreter for hdf5 attributes

class CSKDetails(object):
    __slots__ = ('_file_name', )

    def __init__(self, file_name):
        if h5py is None:
            raise ImportError("Can't read Cosmo Skymed files, because the h5py dependency is missing.")

        with h5py.File(file_name, 'r') as hf:
            sat_id = hf.attrs['Satellite ID'].decode('utf-8')
            prod_type = hf['Product Type'].decode('utf-8')

        if 'CSK' not in sat_id:
            raise ValueError('Expected hdf5 to be a CSK (attribute `Satellite ID` which contains "CSK"). '
                             'Got Satellite ID = {}.'.format(sat_id))
        if 'SCS' not in prod_type:
            raise ValueError('Expected hdf to contain complex products '
                             '(attribute `Product Type` which contains "SCS"). '
                             'Got Product Type = {}'.format(prod_type))

        self._file_name = file_name

    @property
    def file_name(self):
        return self._file_name

    def _get_hdf_dicts(self):
        with h5py.File(self._file_name, 'r') as hf:
            h5_dict = _extract_attrs(hf)
            band_dict = OrderedDict()
            shape_dict = OrderedDict()

            for gp_name in sorted(hf.keys()):
                gp = hf[gp_name]
                band_dict[gp_name] = _extract_attrs(gp)
                shape_dict[gp_name] = gp['SBI'].shape[:2]
        return h5_dict, band_dict, shape_dict

    @staticmethod
    def _parse_pol(str_in):
        return '{}:{}'.format(str_in[0], str_in[1])

    def _get_base_sicd(self, h5_dict, band_dict):
        # type: (dict, dict) -> SICDType

        def get_collection_info():  # type: () -> CollectionInfoType
            mode_type = 'STRIPMAP' if h5_dict['Acquisition Mode'] in \
                                      ['HIMAGE', 'PINGPONG', 'WIDEREGION', 'HUGEREGION'] else 'DYNAMIC STRIPMAP'
            return CollectionInfoType(Classification='UNCLASSIFIED',
                                      CollectorName=h5_dict['Satellite ID'],
                                      CoreName=str(h5_dict['Programmed Image ID']),
                                      CollectType='MONOSTATIC',
                                      RadarMode=RadarModeType(ModeId=h5_dict['Multi-Beam ID'],
                                                              ModeType=mode_type))

        def get_image_creation():  # type: () -> ImageCreationType
            return ImageCreationType(DateTime=numpy.datetime64(h5_dict['Product Generation UTC'], 'ns'))

        def get_grid():  # type: () -> GridType
            if h5_dict['Projection ID'] == 'SLANT RANGE/AZIMUTH':
                image_plane = 'SLANT'
                gr_type = 'RGZERO'
            else:
                image_plane = 'GROUND'
                gr_type = None
            # Row
            row_window_name = h5_dict['Range Focusing Weighting Function'].rstrip().upper()
            row_params = None
            if row_window_name == 'HAMMING':
                row_params = {'COEFFICIENT': '{0:15f}'.format(h5_dict['Range Focusing Weighting Coefficient'])}
            row = DirParamType(Sgn=-1,
                               KCtr=2*center_frequency/speed_of_light,
                               DeltaKCOAPoly=Poly2DType(Coefs=[[0, ], ]),
                               WgtType=WgtTypeType(WindowName=row_window_name, Parameters=row_params))
            # Col
            col_window_name = h5_dict['Azimuth Focusing Weighting Function'].rstrip().upper()
            col_params = None
            if col_window_name == 'HAMMING':
                col_params = {'COEFFICIENT': '{0:15f}'.format(h5_dict['Azimuth Focusing Weighting Coefficient'])}
            col = DirParamType(Sgn=-1,
                               KCtr=0,
                               WgtType=WgtTypeType(WindowName=col_window_name, Parameters=col_params))
            return GridType(ImagePlane=image_plane, Type=gr_type, Row=row, Col=col)

        def get_timeline():  # type: () -> TimelineType
            return TimelineType(CollectStart=collect_start,
                                CollectDuration=duration,
                                IPP=[IPPSetType(index=0, TStart=0, TEnd=0, IPPStart=0), ])

        def get_position():  # type: () -> PositionType
            T = h5_dict['State Vectors Times']  # in seconds relative to ref time
            T += ref_time_offset
            Pos = h5_dict['ECEF Satellite Position']
            Vel = h5_dict['ECEF Satellite Velocity']  # for cross validation
            # Let's perform polynomial fitting for the position with cross validation for overfitting checks
            deg = 1
            prev_vel_error = numpy.inf
            P_x, P_y, P_z = None, None, None
            while deg < Pos.shape[0]:
                # fit position
                P_x = polynomial.polyfit(T, Pos[:, 0], deg=deg)
                P_y = polynomial.polyfit(T, Pos[:, 1], deg=deg)
                P_z = polynomial.polyfit(T, Pos[:, 2], deg=deg)
                # extract estimated velocities
                Vel_est = numpy.array(
                    [polynomial.polyval(T, polynomial.polyder(P_x)),
                     polynomial.polyval(T, polynomial.polyder(P_y)),
                     polynomial.polyval(T, polynomial.polyder(P_z))], dtype=numpy.float64)
                # check our velocity error
                vel_err = Vel_est - Vel
                cur_vel_error = numpy.sum((vel_err*vel_err))
                # stop if the error is not smaller than at the previous step
                if cur_vel_error >= prev_vel_error:
                    break
            return PositionType(ARPPoly=XYZPolyType(X=P_x, Y=P_y, Z=P_z))

        def get_radar_collection():  # type: () -> RadarCollectionType
            tx_pols = []
            chan_params = []
            for i, bdname in enumerate(band_dict):
                pol = band_dict[bdname]['Polarisation']
                tx_pols.append(pol[0])
                chan_params.append(ChanParametersType(TxRcvPolarization=self._parse_pol(pol), index=i))
            if len(tx_pols) == 1:
                return RadarCollectionType(RcvChannels=chan_params, TxPolarization=tx_pols[0])
            else:
                return RadarCollectionType(RcvChannels=chan_params,
                                           TxPolarization='SEQUENCE',
                                           TxSequence=[TxStepType(TxPolarization=pol,
                                                                  index=i+1) for i, pol in enumerate(tx_pols)])

        def get_image_formation():  # type: () -> ImageFormationType
            return ImageFormationType(ImageFormAlgo='RMA',
                                      TStartProc=0,
                                      TEndProc=duration,
                                      STBeamComp='SV',
                                      ImageBeamComp='NO',
                                      AzAutofocus='NO',
                                      RgAutofocus='NO')

        def get_rma():  # type: () -> RMAType
            inca = INCAType(FreqZero=center_frequency)
            return RMAType(RMAlgoType='OMEGA_K',
                           INCA=inca)

        def get_scpcoa():  # type: () -> SCPCOAType
            return SCPCOAType(SideOfTrack=h5_dict['Look Side'][0:1].upper())

        # some common use parameters
        center_frequency = h5_dict['Radar Frequency']
        # relative times in csk are wrt some reference time - for sicd they should be relative to start time
        collect_start = numpy.datetime64(h5_dict['Scene Sensing Start UTC'], 'ns')
        collect_end = numpy.datetime64(h5_dict['Scene Sensing Stop UTC'], 'ns')
        duration = _get_seconds(collect_end, collect_start)
        ref_time = numpy.datetime64(h5_dict['Reference UTC'], 'ns')
        ref_time_offset = _get_seconds(ref_time, collect_start)

        # assemble our pieces
        collection_info = get_collection_info()
        image_creation = get_image_creation()
        grid = get_grid()
        timeline = get_timeline()
        position = get_position()
        radar_collection = get_radar_collection()
        image_formation = get_image_formation()
        rma = get_rma()
        scpcoa = get_scpcoa()

        return SICDType(CollectionInfo=collection_info,
                        ImageCreation=image_creation,
                        Grid=grid,
                        Timeline=timeline,
                        Position=position,
                        RadarCollection=radar_collection,
                        ImageFormation=image_formation,
                        RMA=rma,
                        SCPCOA=scpcoa)

    @staticmethod
    def _get_dop_poly_details(h5_dict):
        # type: (dict) -> Tuple[float, float, Poly1DType, Poly1DType, Poly1DType]
        def strip_poly(arr):
            # strip worthless (all zero) highest order terms
            # find last non-zero index
            last_ind = arr.size
            for i in range(arr.size-1, -1, -1):
                if arr[i] != 0:
                    break
                last_ind = i
            return Poly1DType(Coefs=arr[:last_ind])

        az_ref_time = h5_dict['Azimuth Polynomial Reference Time']  # seconds
        rg_ref_time = h5_dict['Range Polynomial Reference Time']
        dop_poly_az = strip_poly(h5_dict['Centroid vs Azimuth Time Polynomial'])
        dop_poly_rg = strip_poly(h5_dict['Centroid vs Range Time Polynomial'])
        dop_rate_poly_rg = strip_poly(h5_dict['Doppler Rate vs Range Time Polynomial'])
        return az_ref_time, rg_ref_time, dop_poly_az, dop_poly_rg, dop_rate_poly_rg

    def _get_band_specific_sicds(self, base_sicd, h5_dict, band_dict):
        # type: (SICDType, dict, dict) -> Dict[str, SICDType]

        az_ref_time, rg_ref_time, dop_poly_az, dop_poly_rg, dop_rate_poly_rg = self._get_dop_poly_details(h5_dict)
        center_frequency = h5_dict['Radar Frequency']
        # relative times in csk are wrt some reference time - for sicd they should be relative to start time
        collect_start = numpy.datetime64(h5_dict['Scene Sensing Start UTC'], 'ns')
        ref_time = numpy.datetime64(h5_dict['Reference UTC'], 'ns')
        ref_time_offset = _get_seconds(ref_time, collect_start)

        def update_scp_prelim(sicd, band_name):
            # type: (SICDType, str) -> None
            LLH = band_dict[band_name]['Centre Geodetic Coordinates']
            sicd.GeoData = GeoDataType(SCP=SCPType(LLH=LLH))  # EarthModel & ECF will be populated

        def update_image_data(sicd, band_name):
            # type: (SICDType, str) -> Tuple[float, float, float, float]

            # def scp_option2():  # type: () -> RowColType
            #     # Choose the SCP as the point closest to the middle of the image
            #     return RowColType(Row=int(rows/2), Col=int(cols/2))

            def scp_option1():  # type: () -> Tuple[RowColType, float, float, float, float]
                # Choose the SCP as the point closest to the reference zero-doppler and range
                # times in the CSK metadata.
                tt_az_first_time = band_dict[band_name]['Zero Doppler Azimuth First Time']
                # zero doppler time of first column
                tt_ss_az_s = band_dict[band_name]['Line Time Interval']
                scp_col_base = round((az_ref_time - tt_az_first_time) / tt_ss_az_s)
                if base_sicd.SCPCOA.SideOfTrack == 'L':
                    # we ned to reverse time order
                    scp_col = cols - 1 - scp_col_base
                    tt_ss_az_s *= -1
                    tt_az_first_time = band_dict[band_name]['Zero Doppler Azimuth Last Time']
                else:
                    scp_col = scp_col_base
                # zero doppler time of first row
                tt_rg_first_time = band_dict[band_name]['Zero Doppler Range First Time']
                # row spacing in range time (seconds)
                tt_ss_rg_s = band_dict[band_name]['Column Time Interval']
                scp_row = round((rg_ref_time - tt_rg_first_time) / tt_ss_rg_s)
                return RowColType(Row=scp_row, Col=scp_col), tt_rg_first_time, tt_ss_rg_s, tt_az_first_time, tt_ss_az_s

            cols, rows = band_dict[band_name]['shape']
            scp_pixel, t_rg_first_time, t_ss_rg_s, t_az_first_time, t_ss_az_s = scp_option1()
            sicd.ImageData = ImageDataType(NumRows=rows,
                                           NumCols=cols,
                                           FirstRow=0,
                                           FirstCol=0,
                                           PixelType='RE16I_IM16I',
                                           SCPPixel=scp_pixel)

            return t_rg_first_time, t_ss_rg_s, t_az_first_time, t_ss_az_s

        def update_timeline(sicd, band_name):  # type: (SICDType, str) -> None
            prf = band_dict[band_name]['PRF']
            duration = sicd.Timeline.CollectDuration
            ipp_el = sicd.Timeline.IPP[0]
            ipp_el.IPPEnd = duration
            ipp_el.TEnd = duration
            ipp_el.IPPPoly = Poly1DType(Coefs=(0, prf))

        def update_radar_collection(sicd, band_name, ind):  # type: (SICDType, str, int) -> None
            chirp_length = band_dict[band_name]['Range Chirp Length']
            chirp_rate = abs(band_dict[band_name]['Range Chirp Rate'])
            sample_rate = band_dict[band_name]['Sampling Rate']
            ref_dechirp_time = band_dict[band_name]['Reference Dechirping Time']
            win_length = band_dict[band_name]['Echo Sampling Window Length']
            rcv_fm_rate = 0 if numpy.isnan(ref_dechirp_time) else ref_dechirp_time  # TODO: is this the correct value?
            band_width = chirp_length*chirp_rate
            fr_min = center_frequency - 0.5*band_width
            fr_max = center_frequency + 0.5*band_width
            sicd.RadarCollection.TxFrequency = TxFrequencyType(Min=fr_min,
                                                               Max=fr_max)
            sicd.RadarCollection.Waveform = [WaveformParametersType(index=0,
                                                                    TxPulseLength=chirp_length,
                                                                    TxRFBandwidth=band_width,
                                                                    TxFreqStart=fr_min,
                                                                    TxFMRate=chirp_rate,
                                                                    ADCSampleRate=sample_rate,
                                                                    RcvFMRate=rcv_fm_rate,
                                                                    RcvWindowLength=win_length/sample_rate), ]
            sicd.ImageFormation.RcvChanProc.ChanIndices = [ind, ]
            sicd.ImageFormation.TxFrequencyProc = TxFrequencyProcType(MinProc=fr_min, MaxProc=fr_max)
            sicd.ImageFormation.TxRcvPolarizationProc = sicd.RadarCollection.RcvChannels[ind].TxRcvPolarization

        def update_rma_and_grid(sicd, band_name):  # type: (SICDType, str) -> None
            rg_scp_time = rg_first_time + (ss_rg_s*sicd.ImageData.SCPPixel.Row)
            az_scp_time = az_first_time + (ss_az_s*sicd.ImageData.SCPPixel.Col)
            r_ca_scp = rg_scp_time*speed_of_light/2
            sicd.RMA.INCA.R_CA_SCP = r_ca_scp
            # compute DRateSFPoly
            scp_ca_time = az_scp_time + ref_time_offset
            vel_poly = sicd.Position.ARPPoly.derivative(der_order=1, return_poly=True)
            vel_ca_vec = vel_poly(scp_ca_time)
            vel_ca_sq = numpy.sum(vel_ca_vec*vel_ca_vec)
            vel_ca = numpy.sqrt(vel_ca_sq)
            r_ca = numpy.array([r_ca_scp, 1.], dtype=numpy.float64)
            dop_rate_poly_rg_shifted = dop_rate_poly_rg.shift(
                rg_ref_time-rg_scp_time, alpha=ss_rg_s/row_ss, return_poly=False)
            drate_sf_poly = -(polynomial.polymul(dop_rate_poly_rg_shifted, r_ca) *
                              speed_of_light/(2*center_frequency*vel_ca_sq))
            # update grid
            sicd.Grid.Row.SS = row_ss
            sicd.Grid.Row.ImpRespBW = row_bw
            sicd.Grid.Row.DeltaK1 = -0.5 * row_bw
            sicd.Grid.Row.DeltaK2 = 0.5 * row_bw
            col_ss = vel_ca*ss_az_s*drate_sf_poly[0]
            sicd.Grid.Col.SS = col_ss
            col_bw = min(band_dict[band_name]['Azimuth Focusing Bandwidth'] * abs(ss_az_s), 1) / col_ss
            sicd.Grid.Col.ImpRespBW = col_bw
            # update inca
            sicd.RMA.INCA.DRateSFPoly = Poly2DType(Coefs=numpy.reshape(drate_sf_poly, (-1, 1)))
            sicd.RMA.INCA.TimeCAPoly = Poly1DType(Coefs=[scp_ca_time, ss_az_s/col_ss])
            # compute DopCentroidPoly & DeltaKCOAPoly
            dop_centroid_poly = numpy.zeros((dop_poly_rg.order1+1, dop_poly_az.order1+1), dtype=numpy.float64)
            dop_centroid_poly[0, 0] = dop_poly_rg(rg_scp_time-rg_ref_time) + \
                dop_poly_az(az_scp_time-az_ref_time) + \
                0.5*(dop_poly_rg[0] + dop_poly_az[0])
            dop_poly_rg_shifted = dop_poly_rg.shift(rg_ref_time-rg_scp_time, alpha=ss_rg_s/row_ss)
            dop_poly_az_shifted = dop_poly_az.shift(az_ref_time-az_scp_time, alpha=ss_az_s/col_ss)
            dop_centroid_poly[1:, 0] = dop_poly_rg_shifted[1:]
            dop_centroid_poly[0, 1:] = dop_poly_az_shifted[1:]
            sicd.RMA.INCA.DopCentroidPoly = Poly2DType(Coefs=dop_centroid_poly)
            sicd.RMA.INCA.DopCentroidCOA = True
            sicd.Grid.Col.DeltaKCOAPoly = Poly2DType(Coefs=dop_centroid_poly*ss_az_s/col_ss)

            image_data = sicd.ImageData
            grid = sicd.Grid
            inca = sicd.RMA.INCA
            poly_order = 2
            grid_samples = poly_order + 2
            coords_az = (numpy.linspace(0, image_data.NumCols - 1,
                                        grid_samples) - image_data.SCPPixel.Col) * grid.Col.SS
            coords_rg = (numpy.linspace(0, image_data.NumRows - 1,
                                        grid_samples) - image_data.SCPPixel.Row) * grid.Row.SS
            coords_az_2d, coords_rg_2d = numpy.meshgrid(coords_az, coords_rg)
            time_ca_sampled = inca.TimeCAPoly(coords_rg_2d)  # coords_az_2d) 1-D polynomial
            dop_centroid_sampled = inca.DopCentroidPoly(coords_rg_2d, coords_az_2d)
            doppler_rate_sampled = polynomial.polyval(coords_rg_2d, dop_rate_poly_rg_shifted)
            time_coa_sampled = time_ca_sampled + dop_centroid_sampled / doppler_rate_sampled
            grid.TimeCOAPoly = Poly2DType(
                Coefs=_2d_poly_fit(coords_rg_2d, coords_az_2d, time_coa_sampled, x_order=poly_order,
                                   y_order=poly_order))

        def update_radiometric(sicd, band_name):  # type: (SICDType, str) -> None
            if h5_dict['Range Spreading Loss Compensation Geometry'] != 'NONE':
                slant_range = h5_dict['Reference Slant Range']
                exp = h5_dict['Reference Slant Range Exponent']
                sf = slant_range**(2*exp)
                if h5_dict['Calibration Constant Compensation Flag'] == 0:
                    rsf = h5_dict['Rescaling Factor']
                    cal = band_dict[band_name]['Calibration Constant']
                    sf /= cal*(rsf**2)
                sicd.Radiometric = RadiometricType(BetaZeroSFPoly=Poly2DType(Coefs=[[sf, ], ]))

        def update_geodata(sicd):  # type: (SICDType) -> None
            ecf = point_projection.image_to_ground([sicd.ImageData.SCPPixel.Row, sicd.ImageData.SCPPixel.Col], sicd)
            sicd.GeoData.SCP = SCPType(ECF=ecf)  # LLH will be populated

        out = {}
        for i, bd_name in enumerate(band_dict):
            t_sicd = base_sicd.copy()
            update_scp_prelim(t_sicd, bd_name)  # set preliminary value for SCP (required for projection)
            row_bw = band_dict[bd_name]['Range Focusing Bandwidth']*2/speed_of_light
            row_ss = band_dict[bd_name]['Column Spacing']
            rg_first_time, ss_rg_s, az_first_time, ss_az_s = update_image_data(t_sicd, bd_name)
            update_timeline(t_sicd, bd_name)
            update_radar_collection(t_sicd, bd_name, i)
            update_rma_and_grid(t_sicd, bd_name)
            update_radiometric(t_sicd, bd_name)
            update_geodata(t_sicd)
            t_sicd.derive()
            out[bd_name] = t_sicd
        return out

    @staticmethod
    def _get_symmetry(base_sicd, h5_dict):
        line_order = h5_dict['Lines Order']
        column_order = h5_dict['Columns Order']
        symmetry = (
            (line_order == 'EARLY-LATE') != (base_sicd.SCPCOA.SideOfTrack == 'R'),
            column_order != 'NEAR-FAR',
            True)
        return symmetry

    def get_sicd_collection(self):
        """
        Get the sicd collection for the bands.

        Returns
        -------
        Tuple[Dict[str, SICDType], Dict[str, str], Tuple[bool, bool, bool]]
            the first entry is a dictionary of the form {band_name: sicd}
            the second entry is of the form {band_name: shape}
            the third entry is the symmetry tuple
        """

        h5_dict, band_dict, shape_dict = self._get_hdf_dicts()
        base_sicd = self._get_base_sicd(h5_dict, band_dict)
        return self._get_band_specific_sicds(base_sicd, h5_dict, band_dict), \
            shape_dict, self._get_symmetry(base_sicd, h5_dict)


################
# The CSK chipper and reader

class CSKBandChipper(BaseChipper):
    __slots__ = ('_file_name', '_band_name')

    def __init__(self, file_name, band_name, data_size, symmetry):
        self._file_name = file_name
        self._band_name = band_name
        super(CSKBandChipper, self).__init__(data_size, symmetry=symmetry, complex_type=True)

    def _read_raw_fun(self, range1, range2):
        r1, r2 = self._reorder_arguments(range1, range2)
        with h5py.File(self._file_name, 'r') as hf:
            gp = hf['{}/SBI'.format(self._band_name)]
            data = gp[r1[0]:r1[1]:r1[2], r2[0]:r2[1]:r2[2], :]
        return data.transpose((2, 0, 1))


class CSKReader(BaseReader):
    """
    Gets a reader type object for Cosmo Skymed files
    """

    __slots__ = ('_csk_details', )

    def __init__(self, csk_details):
        """

        Parameters
        ----------
        csk_details : str|CSKDetails
            file name or CSKDetails object
        """

        if isinstance(csk_details, str):
            csk_details = CSKDetails(csk_details)
        if not isinstance(csk_details, CSKDetails):
            raise TypeError('The input argument for RadarSatCSKReader must be a '
                            'filename or CSKDetails object')
        sicd_data, shape_dict, symmetry = csk_details.get_sicd_collection()
        chippers = []
        sicds = []
        for band_name in sicd_data:
            sicds.append(sicd_data[band_name])
            chippers.append(CSKBandChipper(csk_details.file_name, band_name, shape_dict[band_name], symmetry))
        super(CSKReader, self).__init__(sicds, chippers)
