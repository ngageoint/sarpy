# -*- coding: utf-8 -*-
"""
Functionality for reading TerraSAR-X data into a SICD model.
"""

import os
import logging
from datetime import datetime
from xml.etree import ElementTree
from typing import List, Tuple, Union

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light
from scipy.interpolate import griddata

from sarpy.compliance import string_types, int_func
from sarpy.io.general.base import SubsetReader, BaseReader
from sarpy.io.general.tiff import TiffDetails, TiffReader
from sarpy.io.general.utils import get_seconds, parse_timestring

from sarpy.io.complex.sicd_elements.blocks import Poly1DType, Poly2DType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType
from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
from sarpy.io.complex.sicd_elements.Timeline import TimelineType, IPPSetType
from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType, TxFrequencyProcType
from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
from sarpy.geometry import point_projection
from sarpy.geometry.geocoords import geodetic_to_ecf
from sarpy.io.complex.utils import two_dim_poly_fit, get_im_physical_coords


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


########
# base expected functionality for a module with an implemented Reader

def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a TerraSAR-X file SSC package.
    Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    TSXReader|None
        `TSXReader` instance if TerraSAR-X file file, `None` otherwise
    """

    try:
        tsx_details = TSXDetails(file_name)
        print('Path {} is determined to be a TerraSAR-X file package.'.format(file_name))
        return TSXReader(tsx_details)
    except (IOError, AttributeError, SyntaxError, ElementTree.ParseError):
        return None

##########
# helper functions and basic interpreter

def _parse_xml(file_name, without_ns=False):
    root_node = ElementTree.parse(file_name).getroot()
    if without_ns:
        return root_node
    else:
        ns = dict([node for _, node in ElementTree.iterparse(file_name, events=('start-ns', ))])
        return ns, root_node


def _is_level1_product(prospective_file):
    with open(prospective_file, 'r') as fi:
        check = fi.read(30)
    return check.startswith('<level1Product')


class TSXDetails(object):
    """
    Parser and interpreter for the TerraSAR-X file package meta-data.
    """

    __slots__ = (
        '_main_file', '_georef_file', '_main_root', '_georef_root')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            The top-level directory, or the basic package xml file.
        """

        self._main_file = None
        self._georef_file = None
        self._main_root = None
        self._georef_root = None
        self._validate_file(file_name)

    def _validate_file(self, file_name):
        """
        Validate the input file location.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        None
        """

        if not isinstance(file_name, string_types):
            raise IOError('file_name must be of string type.')
        if not os.path.exists(file_name):
            raise IOError('file {} does not exist'.format(file_name))

        found_file = None
        if os.path.isdir(file_name):
            for entry in os.listdir(file_name):
                prop_file = os.path.join(entry)
                if os.path.isfile(prop_file) and os.path.splitext(prop_file)[1] == '.xml' \
                        and _is_level1_product(prop_file):
                    found_file = prop_file

            if found_file is None:
                raise IOError(
                    'The provided argument is a directory, but we found no level1Product xml file at the top level.')
        elif os.path.splitext(file_name)[1] == '.xml':
            if _is_level1_product(file_name):
                found_file = file_name
            else:
                raise IOError(
                    'The provided argument is an xml file, which is not a level1Product xml file.')
        else:
            raise IOError(
                'The provided argument is an file, but does not have .xml extension.')

        if file_name is None:
            raise ValueError('Unspecified error where main_file is not defined.')
        self._main_file = found_file
        self._main_root = _parse_xml(self._main_file, without_ns=True)

        georef_file = os.path.join(os.path.split(found_file)[0], 'ANNOTATION', 'GEOREF.xml')
        if not os.path.isfile(georef_file):
            logging.warning(
                'The input file was determined to be or contain a TerraSAR-X level 1 product file, '
                'but the ANNOTATION/GEOREF.xml is not in the expected relative location.')
        else:
            self._georef_file = georef_file
            self._georef_root = _parse_xml(self._georef_file, without_ns=True)

    @property
    def file_name(self):
        """
        str: the main file name
        """

        return self._main_file

    def _find_main(self, tag):
        """
        Pass through to ElementTree.Element.find(tag).

        Parameters
        ----------
        tag : str

        Returns
        -------
        ElementTree.Element
        """

        return self._main_root.find(tag)

    def _findall_main(self, tag):
        """
        Pass through to ElementTree.Element.findall(tag).

        Parameters
        ----------
        tag : str

        Returns
        -------
        List[ElementTree.Element
        """

        return self._main_root.findall(tag)

    def _find_georef(self, tag):
        """
        Pass through to ElementTree.Element.find(tag).

        Parameters
        ----------
        tag : str

        Returns
        -------
        ElementTree.Element
        """

        return None if self._georef_root is None else self._georef_root.find(tag)

    def _findall_georef(self, tag):
        """
        Pass through to ElementTree.Element.findall(tag).

        Parameters
        ----------
        tag : str

        Returns
        -------
        List[ElementTree.Element
        """

        return None if self._georef_root is None else self._georef_root.findall(tag)

    def _get_basic_sicd(self):
        """
        Define the common sicd elements.

        Returns
        -------
        SICDType
        """

        def get_collection_info():
            # type: () -> CollectionInfoType
            collector_name = self._find_main('./productInfo/missionInfo/mission').text
            core_name = self._find_main('./productInfo/sceneInfo/sceneID').text

            mode_id = self._find_main('./productInfo/acquisitionInfo/imagingMode').text
            if mode_id == 'ST':
                # TSX "staring" mode, corresponds to SICD spotlight
                mode_type = 'SPOTLIGHT'
            elif mode_id in ['SL', 'HS']:
                # confusing, but TSX mode "spolight" and "high-resolution spotlight",
                # which actually has a moving beam
                mode_type = 'DYNAMIC STRIPMAP'
            elif mode_id == 'SM':
                # TSX stripmap mode
                mode_type = 'STRIPMAP'
            elif mode_id == 'SC':
                # TSX scansar mode
                mode_type = 'STRIPMAP'
                raise ValueError('ScanSAR mode is currently unsupported')
            else:
                raise ValueError('Got unexpected mode id {}'.format(mode_id))

            return CollectionInfoType(
                CollectorName=collector_name,
                CoreName=core_name,
                CollectType='MONOSTATIC',
                RadarMode=RadarModeType(ModeID=mode_id, ModeType=mode_type),
                Classification='UNCLASSIFIED')

        def get_image_creation():
            # type: () -> ImageCreationType
            from sarpy.__about__ import __version__

            create_time = self._find_main('./generalHeader/generationTime').text
            site = self._find_main('./productInfo/level1ProcessingFacility').text
            app_node = self._find_main('./generalHeader/generationSystem')
            application = 'Unknown' if app_node is None else \
                '{} {}'.format(app_node.text, app_node.attrib.get('version', 'version_unknown'))
            return ImageCreationType(Application=application,
                                     DateTime=numpy.datetime64(create_time, 'us'),
                                     Site=site,
                                     Profile='sarpy {}'.format(__version__))

        def get_initial_grid():
            # type: () -> GridType
            proj_string = self._find_main('./setup/orderInfo/projection').text
            if proj_string == 'GROUNDRANGE':
                image_plane = 'GROUND'
            elif proj_string == 'SLANTRANGE':
                image_plane = 'SLANT'
            else:
                logging.warning('Got image projection {}'.format(proj_string))
                image_plane = 'OTHER'

            the_type = None
            if self._find_main('./productSpecific/complexImageInfo/imageCoordinateType').text == 'ZERODOPPLER':
                the_type = 'RGZERO'

            row_ss = 0.5*float(self._find_main('./productInfo/imageDataInfo/imageRaster/rowSpacing').text)*speed_of_light
            row_bw = 2*float(self._find_main('./processing/processingParameter/rangeLookBandwidth').text)/speed_of_light
            row_win_name = self._find_main('./processing/processingParameter/rangeWindowID').text
            row_wgt_type = WgtTypeType(WindowName=row_win_name)
            if row_win_name == 'HAMMING':
                row_wgt_type.Parameters = {
                    'COEFFICIENT': self._find_main('./processing/processingParameter/rangeWindowCoefficient').text}

            row = DirParamType(
                SS=row_ss,
                Sgn=-1,
                ImpRespBW=row_bw,
                KCtr=2*center_freq/speed_of_light,
                DeltaK1=-0.5*row_bw,
                DeltaK2=0.5*row_bw,
                DeltaKCOAPoly=[[0,],],
                WgtType=row_wgt_type)

            col_ss = float(self._find_main('./productSpecific/complexImageInfo/projectedSpacingAzimuth').text)
            col_win_name = self._find_main('./processing/processingParameter/azimuthWindowID').text
            col_wgt_type = WgtTypeType(WindowName=col_win_name)
            if col_win_name == 'HAMMING':
                col_wgt_type.Parameters = {
                    'COEFFICIENT': self._find_main('./processing/processingParameter/azimuthWindowCoefficient').text}
            col = DirParamType(
                SS=col_ss,
                Sgn=-1,
                ImpRespBW=dop_bw*ss_zd_s/col_ss,
                KCtr=0,
                WgtType=col_wgt_type)

            return GridType(
                ImagePlane=image_plane,
                Type=the_type,
                Row=row,
                Col=col)

        center_freq = float(self._find_main('./instrument/radarParameters/centerFrequency').text)
        dop_bw = float(self._find_main('./processing/processingParameter/azimuthLookBandwidth').text)
        ss_zd_s = float(self._find_main('./productInfo/imageDataInfo/imageRaster/columnSpacing').text)

        collection_info = get_collection_info()
        image_creation = get_image_creation()
        init_grid = get_initial_grid()

        return SICDType(
            CollectionInfo=collection_info,
            ImageCreation=image_creation,
            Grid=init_grid)

    def _populate_basic_image_data(self, sicd):
        """
        Populate the basic ImageData and GeoData. This assumes not ScanSAR mode.

        Parameters
        ----------
        sicd : SICDType

        Returns
        -------
        SICDType
        """

        def extract_location(the_node):
            # type: (ElementTree.Element) -> list
            return [float(the_node.find('./lat').text), float(the_node.find('./lon').text)]

        def define_corners(corn_nodes):
            # type: (list) -> Union[list, None]
            if len(corn_nodes) == 4:
                return [
                    extract_location(corn_nodes[0]),
                    extract_location(corn_nodes[2]),
                    extract_location(corn_nodes[3]),
                    extract_location(corn_nodes[1]),]
            else:
                logging.error('Found {} corner coordinates, so skipping'.format(len(corn_nodes)))
                return None

        # NB: the role of rows and columns is switched in TSX/SICD convention
        rows = int_func(self._find_main('./productInfo/imageDataInfo/imageRaster/numberOfColumns').text)
        cols = int_func(self._find_main('./productInfo/imageDataInfo/imageRaster/numberOfRows').text)

        if self._georef_root is not None:  # use this is better quality meta-data, if available
            # determine the middle grid location
            az_grid_pts = int_func(self._find_georef('./geolocationGrid/numberOfGridPoints/azimuth').text)
            rg_grid_pts = int_func(self._find_georef('./geolocationGrid/numberOfGridPoints/range').text)
            mid_az = int_func(round(az_grid_pts/2.0)) + 1
            mid_rg = int_func(round(rg_grid_pts/2.0)) + 1
            # find the appropriate grid location
            grid_node = self._find_georef('./geolocationGrid/gridPoint[@iaz="{}" @irg="{}"]'.format(mid_az, mid_rg))
            scp_row = int_func(grid_node.find('./col').text)
            scp_col = int_func(grid_node.find('./row').text)
            scp_llh = [
                float(grid_node.find('./lat').text),
                float(grid_node.find('./lon').text),
                float(grid_node.find('./height').text)]
        else:
            scp_row = int_func(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/refColumn').text) - 1
            scp_col = int_func(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/refRow').text) - 1
            scp_llh = [
                float(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/lat').text),
                float(self._find_main('./productInfo/sceneInfo/sceneCenterCoord/lon').text),
                float(self._find_main('./productInfo/sceneInfo/sceneAverageHeight').text)]

        corner_coords = define_corners(self._findall_main('./productInfo/sceneInfo/sceneCornerCoord'))

        sicd.ImageData = ImageDataType(
            NumRows=rows, NumCols=cols, FirstRow=0, FirstCol=0, FullImage=(rows, cols),
            PixelType='RE16I_IM16I', SCPPixel=(scp_row, scp_col))
        sicd.GeoData = GeoDataType(SCP=SCPType(LLH=scp_llh), ImageCorners=corner_coords)
        return sicd

#########
# the reader implementation

class TSXReader(BaseReader):
    pass
