"""
Functionality for reading Capella SAR data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Wade Schwartzkopf")


import logging
import json
from typing import Dict, Any, Tuple, Union
from collections import OrderedDict

from scipy.constants import speed_of_light
import numpy
from numpy.polynomial import polynomial

from sarpy.io.general.base import BaseReader, SarpyIOError
from sarpy.io.general.tiff import TiffDetails, NativeTiffChipper
from sarpy.io.general.utils import parse_timestring, get_seconds, is_file_like
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.utils import fit_position_xvalidation
from sarpy.io.complex.sicd_elements.blocks import XYZPolyType, Poly2DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Position import PositionType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, \
    WaveformParametersType, ChanParametersType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, \
    RcvChanProcType, ProcessingType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_

logger = logging.getLogger(__name__)


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a Capella SAR file.
    Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    CapellaReader|None
        `CapellaReader` instance if Capella file, `None` otherwise
    """

    if is_file_like(file_name):
        return None

    try:
        capella_details = CapellaDetails(file_name)
        logger.info('File {} is determined to be a Capella file.'.format(file_name))
        return CapellaReader(capella_details)
    except SarpyIOError:
        return None


#########
# helper functions

def avci_nacaroglu_window(M, alpha=1.25):
    """
    Avci-Nacaroglu Exponential window. See Doerry '17 paper window 4.40 p 154
    Parameters
    ----------
    M : int
    alpha : float
    """

    M2 = 0.5*M
    t = (numpy.arange(M) - M2)/M
    return numpy.exp(numpy.pi*alpha*(numpy.sqrt(1 - (2*t)**2) - 1))


###########
# parser and interpreter for tiff attributes

class CapellaDetails(object):
    """
    Parses and converts the Cosmo Skymed metadata
    """

    __slots__ = ('_tiff_details', '_img_desc_tags')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
        """

        # verify that the file is a tiff file
        self._tiff_details = TiffDetails(file_name)
        # verify that ImageDescription tiff tag exists
        if 'ImageDescription' not in self._tiff_details.tags:
            raise SarpyIOError('No "ImageDescription" tag in the tiff.')

        img_format = self._tiff_details.tags['ImageDescription']
        # verify that ImageDescription has a reasonable format
        try:
            self._img_desc_tags = json.loads(img_format)  # type: Dict[str, Any]
        except Exception as e:
            msg = 'Failed deserializing the ImageDescription tag as json with error {}'.format(e)
            logger.info(msg)
            raise SarpyIOError(msg)

        # verify the file is not compressed
        self._tiff_details.check_compression()
        # verify the file is not tiled
        self._tiff_details.check_tiled()

    @property
    def file_name(self):
        """
        str: the file name
        """

        return self._tiff_details.file_name

    @property
    def tiff_details(self):
        # type: () -> TiffDetails
        """
        TiffDetails: The tiff details object.
        """

        return self._tiff_details

    def get_symmetry(self):
        # type: () -> Tuple[bool, bool, bool]
        """
        Gets the symmetry definition.

        Returns
        -------
        Tuple[bool, bool, bool]
        """

        pointing = self._img_desc_tags['collect']['radar']['pointing'].lower()
        if pointing == 'left':
            return True, False, True
        elif pointing == 'right':
            return False, False, True
        else:
            raise ValueError('Got unhandled pointing value {}'.format(pointing))

    def get_sicd(self):
        """
        Get the SICD metadata for the image.

        Returns
        -------
        SICDType
        """

        def convert_string_dict(dict_in):
            # type: (dict) -> dict
            dict_out = OrderedDict()
            for key, val in dict_in.items():
                if isinstance(val, str):
                    dict_out[key] = val
                elif isinstance(val, int):
                    dict_out[key] = str(val)
                elif isinstance(val, float):
                    dict_out[key] = '{0:0.17G}'.format(val)
                else:
                    raise TypeError('Got unhandled type {}'.format(type(val)))
            return dict_out

        def extract_state_vector():
            # type: () -> (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            vecs = collect['state']['state_vectors']
            times = numpy.zeros((len(vecs), ), dtype=numpy.float64)
            positions = numpy.zeros((len(vecs), 3), dtype=numpy.float64)
            velocities = numpy.zeros((len(vecs), 3), dtype=numpy.float64)
            for i, entry in enumerate(vecs):
                times[i] = get_seconds(parse_timestring(entry['time'], precision='ns'), start_time, precision='ns')
                positions[i, :] = entry['position']
                velocities[i, :] = entry['velocity']
            return times, positions, velocities

        def get_radar_parameter(name):
            if name in radar:
                return radar[name]
            if len(radar_time_varying) > 0:
                element = radar_time_varying[0]
                if name in element:
                    return element[name]
            raise ValueError('Unable to determine radar parameter `{}`'.format(name))

        def get_collection_info():
            # type: () -> CollectionInfoType
            coll_name = collect['platform']
            mode = collect['mode'].strip().lower()
            if mode == 'stripmap':
                radar_mode = RadarModeType(ModeType='STRIPMAP', ModeID=mode)
            elif mode == 'spotlight':
                radar_mode = RadarModeType(ModeType='SPOTLIGHT', ModeID=mode)
            elif mode == 'sliding_spotlight':
                radar_mode = RadarModeType(ModeType='DYNAMIC STRIPMAP', ModeID=mode)
            else:
                raise ValueError('Got unhandled radar mode {}'.format(mode))

            return CollectionInfoType(
                CollectorName=coll_name,
                CoreName=collect['collect_id'],
                RadarMode=radar_mode,
                Classification='UNCLASSIFIED',
                CollectType='MONOSTATIC')

        def get_image_creation():
            # type: () -> ImageCreationType
            from sarpy.__about__ import __version__
            return ImageCreationType(
                Application=self._tiff_details.tags['Software'],
                DateTime=parse_timestring(self._img_desc_tags['processing_time'], precision='us'),
                Profile='sarpy {}'.format(__version__),
                Site='Unknown')

        def get_image_data():
            # type: () -> ImageDataType
            rows = int(img['columns'])  # capella uses flipped row/column definition?
            cols = int(img['rows'])
            if img['data_type'] == 'CInt16':
                pixel_type = 'RE16I_IM16I'
            else:
                raise ValueError('Got unhandled data_type {}'.format(img['data_type']))

            scp_pixel = (int(0.5 * rows), int(0.5 * cols))
            if radar['pointing'] == 'left':
                scp_pixel = (rows - scp_pixel[0] - 1, cols - scp_pixel[1] - 1)

            return ImageDataType(
                NumRows=rows,
                NumCols=cols,
                FirstRow=0,
                FirstCol=0,
                PixelType=pixel_type,
                FullImage=(rows, cols),
                SCPPixel=scp_pixel)

        def get_geo_data():
            # type: () -> GeoDataType
            return GeoDataType(SCP=SCPType(ECF=img['center_pixel']['target_position']))

        def get_position():
            # type: () -> PositionType
            px, py, pz = fit_position_xvalidation(state_time, state_position, state_velocity, max_degree=8)
            return PositionType(ARPPoly=XYZPolyType(X=px, Y=py, Z=pz))

        def get_grid():
            # type: () -> GridType

            def get_weight(window_dict):
                window_name = window_dict['name']
                if window_name.lower() == 'rectangular':
                    return WgtTypeType(WindowName='UNIFORM'), None
                elif window_name.lower() == 'avci-nacaroglu':
                    return WgtTypeType(
                        WindowName=window_name.upper(),
                        Parameters=convert_string_dict(window_dict['parameters'])), \
                           avci_nacaroglu_window(64, alpha=window_dict['parameters']['alpha'])
                else:
                    return WgtTypeType(
                        WindowName=window_name,
                        Parameters=convert_string_dict(window_dict['parameters'])), None

            image_plane = 'SLANT'
            grid_type = 'RGZERO'

            coa_time = parse_timestring(img['center_pixel']['center_time'], precision='ns')
            row_bw = img.get('processed_range_bandwidth', bw)
            row_imp_rsp_bw = 2*row_bw/speed_of_light
            row_wgt, row_wgt_funct = get_weight(img['range_window'])
            row = DirParamType(
                SS=img['image_geometry']['delta_range_sample'],
                Sgn=-1,
                ImpRespBW=row_imp_rsp_bw,
                ImpRespWid=img['range_resolution'],
                KCtr=2*fc/speed_of_light,
                DeltaK1=-0.5*row_imp_rsp_bw,
                DeltaK2=0.5*row_imp_rsp_bw,
                DeltaKCOAPoly=[[0.0, ], ],
                WgtFunct=row_wgt_funct,
                WgtType=row_wgt)

            # get timecoa value
            timecoa_value = get_seconds(coa_time, start_time)
            # find an approximation for zero doppler spacing - necessarily rough for backprojected images
            col_ss = img['pixel_spacing_row']
            dop_bw = img['processed_azimuth_bandwidth']

            col_wgt, col_wgt_funct = get_weight(img['azimuth_window'])
            col = DirParamType(
                SS=col_ss,
                Sgn=-1,
                ImpRespWid=img['azimuth_resolution'],
                ImpRespBW=dop_bw*abs(ss_zd_s)/col_ss,
                KCtr=0,
                WgtFunct=col_wgt_funct,
                WgtType=col_wgt)

            # TODO:
            #   column deltakcoa poly - it's in there at ["image"]["frequency_doppler_centroid_polynomial"]

            return GridType(
                ImagePlane=image_plane,
                Type=grid_type,
                TimeCOAPoly=[[timecoa_value, ], ],
                Row=row,
                Col=col)

        def get_radar_collection():
            # type: () -> RadarCollectionType

            freq_min = fc - 0.5*bw
            return RadarCollectionType(
                TxPolarization=radar['transmit_polarization'],
                TxFrequency=(freq_min, freq_min + bw),
                Waveform=[WaveformParametersType(
                    TxRFBandwidth=bw,
                    TxPulseLength=get_radar_parameter('pulse_duration'),
                    RcvDemodType='CHIRP',
                    ADCSampleRate=radar['sampling_frequency'],
                    TxFreqStart=freq_min)],
                RcvChannels=[ChanParametersType(
                    TxRcvPolarization='{}:{}'.format(radar['transmit_polarization'],
                                                     radar['receive_polarization']))])

        def get_timeline():
            # type: () -> TimelineType
            prf = radar['prf'][0]['prf']
            return TimelineType(
                CollectStart=start_time,
                CollectDuration=duration,
                IPP=[
                    IPPSetType(
                        TStart=0,
                        TEnd=duration,
                        IPPStart=0,
                        IPPEnd=duration*prf,
                        IPPPoly=(0, prf)), ])

        def get_image_formation():
            # type: () -> ImageFormationType

            algo = img['algorithm'].upper()
            processings = None
            if algo == 'BACKPROJECTION':
                processings = [ProcessingType(Type='Backprojected to DEM', Applied=True), ]
            else:
                logger.warning(
                    'Got unexpected algorithm, the results for the '
                    'sicd struture might be unexpected')

            if algo not in ('PFA', 'RMA', 'RGAZCOMP'):
                logger.warning(
                    'Image formation algorithm {} not one of the recognized SICD options, '
                    'being set to "OTHER".'.format(algo))
                algo = 'OTHER'

            return ImageFormationType(
                RcvChanProc=RcvChanProcType(NumChanProc=1, PRFScaleFactor=1),
                ImageFormAlgo=algo,
                TStartProc=0,
                TEndProc=duration,
                TxRcvPolarizationProc='{}:{}'.format(radar['transmit_polarization'], radar['receive_polarization']),
                TxFrequencyProc=(
                    radar_collection.TxFrequency.Min,
                    radar_collection.TxFrequency.Max),
                STBeamComp='NO',
                ImageBeamComp='NO',
                AzAutofocus='NO',
                RgAutofocus='NO',
                Processings=processings)

        def get_rma():
            # type: () -> RMAType
            img_geometry = img['image_geometry']
            near_range = img_geometry['range_to_first_sample']
            center_time = parse_timestring(img['center_pixel']['center_time'], precision='us')
            first_time = parse_timestring(img_geometry['first_line_time'], precision='us')
            zd_time_scp = get_seconds(center_time, first_time, 'us')
            r_ca_scp = near_range + image_data.SCPPixel.Row*grid.Row.SS
            time_ca_poly = numpy.array([zd_time_scp, -look*ss_zd_s/grid.Col.SS], dtype='float64')

            timecoa_value = get_seconds(center_time, start_time)
            arp_velocity = position.ARPPoly.derivative_eval(timecoa_value, der_order=1)
            vm_ca = numpy.linalg.norm(arp_velocity)
            inca = INCAType(
                R_CA_SCP=r_ca_scp,
                FreqZero=fc,
                TimeCAPoly=time_ca_poly,
                DRateSFPoly=[[1/(vm_ca*ss_zd_s/grid.Col.SS)], ]
            )

            return RMAType(
                RMAlgoType='RG_DOP',
                INCA=inca)

        def get_radiometric():
            # type: () -> Union[None, RadiometricType]
            if img['radiometry'].lower() != 'beta_nought':
                logger.warning(
                    'Got unrecognized Capella radiometry {},\n\t'
                    'skipping the radiometric metadata'.format(img['radiometry']))
                return None

            return RadiometricType(BetaZeroSFPoly=[[img['scale_factor']**2, ], ])

        def add_noise():
            if sicd.Radiometric is None:
                return

            nesz_raw = numpy.array(img['nesz_polynomial']['coefficients'], dtype='float64')
            test_value = polynomial.polyval(rma.INCA.R_CA_SCP, nesz_raw)
            if abs(test_value - img['nesz_peak']) > 100:
                # this polynomial reversed in early versions, so reverse if evaluated results are nonsense
                nesz_raw = nesz_raw[::-1]
            nesz_poly_raw = Poly2DType(Coefs=numpy.reshape(nesz_raw, (-1, 1)))
            noise_coeffs = nesz_poly_raw.shift(-rma.INCA.R_CA_SCP, 1, 0, 1, return_poly=False)
            # this is in nesz units, so shift to absolute units
            noise_coeffs[0] -= 10*numpy.log10(sicd.Radiometric.SigmaZeroSFPoly[0, 0])
            sicd.Radiometric.NoiseLevel = NoiseLevelType_(NoiseLevelType='ABSOLUTE', NoisePoly=noise_coeffs)

        # extract general use information
        collect = self._img_desc_tags['collect']
        img = collect['image']
        radar = collect['radar']
        radar_time_varying = radar.get('time_varying_parameters', [])

        start_time = parse_timestring(collect['start_timestamp'], precision='ns')
        end_time = parse_timestring(collect['stop_timestamp'], precision='ns')
        duration = get_seconds(end_time, start_time, precision='ns')
        state_time, state_position, state_velocity = extract_state_vector()
        bw = get_radar_parameter('pulse_bandwidth')
        fc = get_radar_parameter('center_frequency')
        ss_zd_s = img['image_geometry']['delta_line_time']
        look = -1 if radar['pointing'] == 'right' else 1

        # define the sicd elements
        collection_info = get_collection_info()
        image_creation = get_image_creation()
        image_data = get_image_data()
        geo_data = get_geo_data()
        position = get_position()
        grid = get_grid()
        radar_collection = get_radar_collection()
        timeline = get_timeline()
        image_formation = get_image_formation()
        rma = get_rma()
        radiometric = get_radiometric()

        sicd = SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            ImageData=image_data,
            GeoData=geo_data,
            Position=position,
            Grid=grid,
            RadarCollection=radar_collection,
            Timeline=timeline,
            ImageFormation=image_formation,
            RMA=rma,
            Radiometric=radiometric)
        sicd.derive()

        add_noise()
        sicd.populate_rniirs(override=False)
        return sicd


class CapellaReader(BaseReader, SICDTypeReader):
    """
    The Capella reader object.
    """

    __slots__ = ('_capella_details', )

    def __init__(self, capella_details):
        """

        Parameters
        ----------
        capella_details : str|CapellaDetails
        """

        if isinstance(capella_details, str):
            capella_details = CapellaDetails(capella_details)

        if not isinstance(capella_details, CapellaDetails):
            raise TypeError('The input argument for capella_details must be a '
                            'filename or CapellaDetails object')
        self._capella_details = capella_details
        sicd = self.capella_details.get_sicd()
        chipper = NativeTiffChipper(self.capella_details.tiff_details, symmetry=self.capella_details.get_symmetry())

        SICDTypeReader.__init__(self, sicd)
        BaseReader.__init__(self, chipper, reader_type="SICD")
        self._check_sizes()

    @property
    def capella_details(self):
        # type: () -> CapellaDetails
        """
        CapellaDetails: The capella details object.
        """

        return self._capella_details

    @property
    def file_name(self):
        return self.capella_details.file_name
