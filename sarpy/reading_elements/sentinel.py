# -*- coding: utf-8 -*-
"""
Functionality for reading Sentinel-1 data into a SICD model.
"""

__classification__ = "UNCLASSIFIED"

import logging
import re
import os
from datetime import datetime
from xml.etree import ElementTree
from typing import List, Tuple

import numpy
from numpy.polynomial import polynomial
from scipy.constants import speed_of_light

from .tiff import TiffDetails, TiffReader

from ..sicd_elements.blocks import Poly1DType, Poly2DType
from ..sicd_elements.SICD import SICDType
from ..sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
from ..sicd_elements.ImageCreation import ImageCreationType
from ..sicd_elements.RadarCollection import RadarCollectionType, WaveformParametersType, \
    TxFrequencyType, ChanParametersType, TxStepType


########
# base expected functionality for a module with an implemented Reader


def is_a(file_name):
    """
    Tests whether a given file_name corresponds to a Sentinel file. Returns a reader instance, if so.

    Parameters
    ----------
    file_name : str
        the file_name to check

    Returns
    -------
    SentinelReader|None
        `SentinelReader` instance if Sentinel-1 file, `None` otherwise
    """

    try:
        sentinel_details = SentinelDetails(file_name)
        print('File {} is determined to be a Sentinel-1 product.xml file.'.format(file_name))
        return SentinelReader(sentinel_details)
    except IOError:
        # we don't want to catch parsing errors, for now
        return None


##########
# helper functions
def _parse_xml(file_name):
    ns = dict([node for _, node in ElementTree.iterparse(file_name, events=('start-ns'))])
    return ns, ElementTree.parse(file_name).getroot()


def _parse_product(file_name):
    def parse_pol(str_in):
        return '{}:{}'.format(str_in[0], str_in[1])

    ns, root_node = _parse_xml(file_name)
    # CollectionInfo
    platform = root_node.find('./metadataSection'
                              '/metadataObject[@ID="platform"]'
                              '/metadataWrap'
                              '/xmlData/safe:platform', ns)
    collector_name = platform.find('./safe:familyName', ns).text + \
                     platform.find('./safe:number', ns).text
    mode_id = platform.find('./safe:instrument'
                            '/safe:extension'
                            '/s1sarl1:instrumentMode'
                            '/s1sarl1:mode', ns).text
    if mode_id == 'SM':
        mode_type = 'STRIPMAP'
    else:
        # TOPSAR - closest SICD analog is Dynamic Stripmap
        mode_type = 'DYNAMIC STRIPMAP'
    collection_info = CollectionInfoType(CollectorName=collector_name,
                                         RadarMode=RadarModeType(ModeId=mode_id, ModeType=mode_type))
    # ImageCreation
    processing = root_node.find('./metadataSection'
                                '/metadataObject[@ID="processing"]'
                                '/metadataWrap'
                                '/xmlData'
                                '/safe:processing', ns)
    facility = processing.find('safe:facility', ns)
    software = facility.find('./safe:software', ns)
    image_creation = ImageCreationType(
        Application='{name} {version}'.format(**software.attrib),
        DateTime=processing.attrib['stop'],
        Site='{name}, {site}, {country}'.format(**facility.attrib),
        Profile='Prototype')
    # RadarCollection
    polarizations = root_node.findall('./metadataSection'
                                      '/metadataObject[@ID="generalProductInformation"]'
                                      '/metadataWrap'
                                      '/xmlData'
                                      '/s1sarl1:standAloneProductInformation'
                                      '/s1sarl1:transmitterReceiverPolarisation', ns)

    radar_collection = RadarCollectionType(RcvChannels=[
        ChanParametersType(TxRcvPolarization=parse_pol(pol.text), index=i) for i, pol in enumerate(polarizations)])
    return SICDType(CollectionInfo=collection_info, ImageCreation=image_creation, RadarCollection=radar_collection)


###########
# parser and interpreter for sentinel-1 manifest.safe file

class SentinelDetails(object):
    __slots__ = ('_file_name', '_root_node', '_ns', '_satellite', '_product_type')

    def __init__(self, file_name):
        self._file_name = file_name
        self._ns, self._root_node = _parse_xml(file_name)
        self._satellite = self.find('./metadataSection'
                                    '/metadataObject[@ID="platform"]'
                                    '/metadataWrap'
                                    '/xmlData'
                                    '/safe:platform'
                                    '/safe:familyName').text
        if self._satellite != 'SENTINEL-1':
            raise ValueError('The platform in the manifest.safe file is required '
                             'to be SENTINEL-1, got {}'.format(self._satellite))
        self._product_type = self.find('./metadataSection'
                                       '/metadataObject[@ID="generalProductInformation"]'
                                       '/metadataWrap'
                                       '/xmlData'
                                       '/s1sarl1:standAloneProductInformation'
                                       '/s1sarl1:productType').text
        if self._product_type != 'SLC':
            raise ValueError('The product type in the manifest.safe file is required '
                             'to be "SLC", got {}'.format(self._product_type))

    @property
    def file_name(self):
        return self._file_name

    @property
    def satellite(self):
        return self._satellite

    @property
    def product_type(self):
        return self._product_type

    def find(self, tag):
        """
        Pass through to ElementTree.Element.find(tag, ns).

        Parameters
        ----------
        tag : str

        Returns
        -------
        ElementTree.Element
        """

        return self._root_node.find(tag, self._ns)

    def findall(self, tag):
        """
        Pass through to ElementTree.Element.findall(tag, ns).

        Parameters
        ----------
        tag : str

        Returns
        -------
        List[ElementTree.Element
        """

        return self._root_node.findall(tag, self._ns)

    def _get_file_sets(self):
        """
        Extracts paths for measurement and metadata files from a Sentinel manifest.safe file.
        These files will be grouped according to "measurement data unit" implicit in the
        Sentinel structure.

        Returns
        -------
        List[dict]
        """

        def get_file_location(schema_type, ids):
            if isinstance(ids, str):
                ids = [ids, ]
            for id in ids:
                do = self.find('dataObjectSection/dataObject[@repID="{}"]/[@ID="{}"]'.format(schema_type, id))
                # TODO: missing a *, or is there an extra slash in the above?
                if do is None:
                    continue
                return os.path.join(base_path, do.find('./byteStream/fileLocation').attrib['href'])
            return None

        base_path = os.path.dirname(self._file_name)

        files = []
        for mdu in self.findall('./informationPackageMap'
                                '/xfdu:contentUnit'
                                '/xfdu:contentUnit/[@repID="s1Level1MeasurementSchema"]'):
            # TODO: missing a *, or is there an extra slash in the above?
            # get the data file for this measurement
            fnames = {'data': get_file_location('s1Level1MeasurementSchema',
                                                mdu.find('dataObjectPointer').attrib['dataObjectID'])}
            # get the ids for product, noise, and calibration associated with this measurement data unit
            ids = mdu.attrib['dmdID'].split()
            # translate these ids to data object ids=file ids for the data files
            fids = [self.find('./metadataSection'
                              '/metadataObject[@ID="{}"]'
                              '/dataObjectPointer'.format(did)).attrib['dataObjectID'] for did in ids]
            # NB: there is (at most) one of these per measurement data unit
            fnames['product'] = get_file_location('s1Level1ProductSchema', fids)
            fnames['noise'] = get_file_location('s1Level1NoiseSchema', fids)
            fnames['calibration'] = get_file_location('s1Level1CalibrationSchema', fids)
            files.append(fnames)
        return files

    def get_sicd_collection(self):
        pass


class SentinelReader(object):
    def __init__(self, sentinel_meta):
        """

        Parameters
        ----------
        sentinel_meta : str|SentinelDetails
        """

        if isinstance(sentinel_meta, str):
            sentinel_meta = SentinelDetails(sentinel_meta)
        if not isinstance(sentinel_meta, SentinelDetails):
            raise TypeError('Input argumnet for SentinelReader must be a file name or SentinelReader object.')

