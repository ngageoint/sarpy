"""
Functionality for reading Capella SAR data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"
__author__ = ("Thomas McCullough", "Wade Schwartzkopf")


import logging
import json
from typing import Dict, Any, Tuple
from datetime import datetime
from collections import OrderedDict

from scipy.constants import speed_of_light
import numpy

from sarpy.compliance import string_types
from sarpy.io.general.base import BaseReader, SarpyIOError
from sarpy.io.general.tiff import TiffDetails, NativeTiffChipper
from sarpy.io.general.utils import parse_timestring, get_seconds, is_file_like
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.utils import fit_position_xvalidation
from sarpy.io.complex.sicd_elements.blocks import XYZPolyType
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
            logger.error('Failed deserializing the ImageDescription tag as json with error {}'.format(e))
            raise e
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
            return False, False, True
        elif pointing == 'right':
            return False, True, True
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
                if isinstance(val, string_types):
                    dict_out[key] = val
                elif isinstance(val, int):
                    dict_out[key] = str(val)
                elif isinstance(val, float):
                    dict_out[key] = '{0:0.16G}'.format(val)
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
            start_dt = start_time.astype('datetime64[us]').astype(datetime)
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
                CoreName='{}{}{}'.format(start_dt.strftime('%d%b%y').upper(),
                                         coll_name,
                                         start_dt.strftime('%H%M%S')),
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
            img = collect['image']
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
            return GeoDataType(SCP=SCPType(ECF=collect['image']['center_pixel']['target_position']))

        def get_position():
            # type: () -> PositionType
            px, py, pz = fit_position_xvalidation(state_time, state_position, state_velocity, max_degree=6)
            return PositionType(ARPPoly=XYZPolyType(X=px, Y=py, Z=pz))

        def get_grid():
            # type: () -> GridType

            def get_weight(window_dict):
                window_name = window_dict['name']
                if window_name.lower() == 'rectangular':
                    return WgtTypeType(WindowName='UNIFORM')
                else:
                    # TODO: what is the proper interpretation for the avci-nacaroglu window?
                    return WgtTypeType(
                        WindowName=window_name,
                        Parameters=convert_string_dict(window_dict['parameters']))

            img = collect['image']
            img_geometry = img['image_geometry']
            if img_geometry.get('type', None) == 'slant_plane':
                image_plane = 'SLANT'
            else:
                image_plane = 'OTHER'

            grid_type = 'PLANE'
            if self._img_desc_tags['product_type'] == 'SLC' and img['algorithm'] != 'backprojection':
                grid_type = 'RGZERO'

            coa_time = parse_timestring(img['center_pixel']['center_time'], precision='ns')
            row_imp_rsp_bw = 2*bw/speed_of_light
            row = DirParamType(
                SS=img['pixel_spacing_column'],
                Sgn=-1,
                ImpRespBW=row_imp_rsp_bw,
                ImpRespWid=img['range_resolution'],
                KCtr=2*fc/speed_of_light,
                DeltaK1=-0.5*row_imp_rsp_bw,
                DeltaK2=0.5*row_imp_rsp_bw,
                DeltaKCOAPoly=[[0.0, ], ],
                WgtType=get_weight(img['range_window']))

            # get timecoa value
            timecoa_value = get_seconds(coa_time, start_time)  # TODO: this is not generally correct
            # find an approximation for zero doppler spacing - necessarily rough for backprojected images
            # find velocity at coatime
            arp_velocity = position.ARPPoly.derivative_eval(timecoa_value, der_order=1)
            arp_speed = numpy.linalg.norm(arp_velocity)
            col_ss = img['pixel_spacing_row']
            dop_bw = img['processed_azimuth_bandwidth']
            # ss_zd_s = col_ss/arp_speed

            col = DirParamType(
                SS=col_ss,
                Sgn=-1,
                ImpRespWid=img['azimuth_resolution'],
                ImpRespBW=dop_bw/arp_speed,
                KCtr=0,
                WgtType=get_weight(img['azimuth_window']))

            # TODO:
            #   column deltakcoa poly - it's in there at ["image"]["frequency_doppler_centroid_polynomial"]
            #   weight functions?

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
            prf = collect['radar']['prf'][0]['prf']
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

            radar = collect['radar']
            algo = collect['image']['algorithm'].upper()
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

        # TODO: From Wade - Radiometric is not suitable?

        # extract general use information
        collect = self._img_desc_tags['collect']
        radar = collect['radar']
        radar_time_varying = radar.get('time_varying_parameters', [])

        start_time = parse_timestring(collect['start_timestamp'], precision='ns')
        end_time = parse_timestring(collect['stop_timestamp'], precision='ns')
        duration = get_seconds(end_time, start_time, precision='ns')
        state_time, state_position, state_velocity = extract_state_vector()
        bw = get_radar_parameter('pulse_bandwidth')
        fc = get_radar_parameter('center_frequency')

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

        sicd = SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            ImageData=image_data,
            GeoData=geo_data,
            Position=position,
            Grid=grid,
            RadarCollection=radar_collection,
            Timeline=timeline,
            ImageFormation=image_formation)
        sicd.derive()

        # this would be a rough estimate - waiting for radiometric data
        # sicd.populate_rniirs(override=False)
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

        if isinstance(capella_details, string_types):
            capella_details = CapellaDetails(capella_details)

        if not isinstance(capella_details, CapellaDetails):
            raise TypeError('The input argument for capella_details must be a '
                            'filename or CapellaDetails object')
        self._capella_details = capella_details
        sicd = self.capella_details.get_sicd()
        chipper = NativeTiffChipper(self.capella_details.tiff_details, symmetry=self.capella_details.get_symmetry())

        SICDTypeReader.__init__(self, sicd)
        BaseReader.__init__(self, chipper, reader_type="SICD")

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
