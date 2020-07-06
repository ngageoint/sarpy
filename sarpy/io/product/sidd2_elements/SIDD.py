# -*- coding: utf-8 -*-
"""
The SIDDType 2.0 definition.
"""

import logging
from typing import Union
from collections import OrderedDict

# noinspection PyProtectedMember
from ...complex.sicd_elements.base import Serializable, _SerializableDescriptor, DEFAULT_STRICT
from .ProductCreation import ProductCreationType
from .Display import ProductDisplayType
from .GeoData import GeoDataType
from .Measurement import MeasurementType
from .ExploitationFeatures import ExploitationFeaturesType
from .DownstreamReprocessing import DownstreamReprocessingType
from .Compression import CompressionType
from .DigitalElevationData import DigitalElevationDataType
from .ProductProcessing import ProductProcessingType
from .Annotations import AnnotationsType
from ..sidd1_elements.SIDD import SIDDType as SIDDType1
from .blocks import ErrorStatisticsType, RadiometricType, MatchInfoType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


############
# namespace validate and definitIon of required entries in the namespace dictionary
_SIDD_SPECIFICATION_IDENTIFIER = 'SIDD Volume 1 Design & Implementation Description Document'
_SIDD_SPECIFICATION_VERSION = '2.0'
_SIDD_SPECIFICATION_DATE = '2019-05-31T00:00:00Z'
_SIDD_URN = 'urn:SIDD:2.0.0'
_ISM_URN = 'urn:us:gov:ic:ism:13'
_SFA_URN = 'urn:SFA:1.2.0'
_SICOMMON_URN = 'urn:SICommon:1.0'


def _validate_sidd_urn(xml_ns, ns_key):
    if xml_ns is None:
        raise ValueError('xml_ns must not be None for SIDD interpretation.')

    if ns_key is None or ns_key not in xml_ns:
        raise ValueError('ns_key must be a key in xml_ns.')

    sidd_urn = xml_ns[ns_key]
    if sidd_urn != _SIDD_URN:
        logging.warning('SIDD version 2 urn is expected to be "{}", '
                        'but we got "{}". Differences in standard may lead to deserialization '
                        'errors.'.format(_SIDD_URN, sidd_urn))


def _validate_ism_urn(xml_ns):
    if 'ism' not in xml_ns:
        the_val = None
        for key in xml_ns:
            val = xml_ns[key]
            if val.lower().startswith('urn:us:gov:ic:ism'):
                the_val = val
        if the_val is None:
            raise ValueError('Cannot find the required ism namespace.')
        xml_ns['ism'] = the_val

    ism_urn = xml_ns['ism']
    if ism_urn != _ISM_URN:
        logging.warning('SIDD version 2 "ism" namespace urn is expected to be "{}", '
                        'but we got "{}". Differences in standard may lead to deserialization '
                        'errors.'.format(_ISM_URN, ism_urn))


def _validate_sfa_urn(xml_ns):
    if 'sfa' not in xml_ns:
        the_val = None
        for key in xml_ns:
            val = xml_ns[key]
            if val.lower().startswith('urn:sfa:'):
                the_val = val
        if the_val is None:
            raise ValueError('Cannot find the required SFA namespace.')
        xml_ns['sfa'] = the_val

    sfa_urn = xml_ns['sfa']
    if sfa_urn != _SFA_URN:
        logging.warning('SIDD version 2 "SFA" namespace urn is expected to be "{}", '
                        'but we got "{}". Differences in standard may lead to deserialization '
                        'errors.'.format(_SFA_URN, sfa_urn))


def _validate_sicommon_urn(xml_ns):
    if 'sicommon' not in xml_ns:
        the_val = None
        for key in xml_ns:
            val = xml_ns[key]
            if val.lower().startswith('urn:sicommon:'):
                the_val = val
        if the_val is None:
            raise ValueError('Cannot find the required SICommon namespace.')
        xml_ns['sicommon'] = the_val

    sicommon_urn = xml_ns['sicommon']
    if sicommon_urn != _SICOMMON_URN:
        logging.warning('SIDD version 2 "SICommon" namespace urn is expected to be "{}", '
                        'but we got "{}". Differences in standard may lead to deserialization '
                        'errors.'.format(_SICOMMON_URN, sicommon_urn))


def _validate_xml_ns(xml_ns, ns_key):
    _validate_sidd_urn(xml_ns, ns_key)
    _validate_ism_urn(xml_ns)
    _validate_sfa_urn(xml_ns)
    _validate_sicommon_urn(xml_ns)


##########
# The SIDD object

class SIDDType(Serializable):
    """
    The root element of the SIDD 2.0 document.
    """

    _fields = (
        'ProductCreation', 'Display', 'GeoData', 'Measurement', 'ExploitationFeatures',
        'DownstreamReprocessing', 'ErrorStatistics', 'Radiometric', 'MatchInfo', 'Compression',
        'DigitalElevationData', 'ProductProcessing', 'Annotations')
    _required = (
        'ProductCreation', 'Display', 'GeoData', 'Measurement', 'ExploitationFeatures')
    # Descriptor
    ProductCreation = _SerializableDescriptor(
        'ProductCreation', ProductCreationType, _required, strict=DEFAULT_STRICT,
        docstring='Information related to processor, classification, and product type.')  # type: ProductCreationType
    Display = _SerializableDescriptor(
        'Display', ProductDisplayType, _required, strict=DEFAULT_STRICT,
        docstring='Contains information on the parameters needed to display the product in '
                  'an exploitation tool.')  # type: ProductDisplayType
    GeoData = _SerializableDescriptor(
        'GeoData', GeoDataType, _required, strict=DEFAULT_STRICT,
        docstring='Contains generic and extensible targeting and geographic region '
                  'information.')  # type: GeoDataType
    Measurement = _SerializableDescriptor(
        'Measurement', MeasurementType, _required, strict=DEFAULT_STRICT,
        docstring='Contains the metadata necessary for performing measurements.')  # type: MeasurementType
    ExploitationFeatures = _SerializableDescriptor(
        'ExploitationFeatures', ExploitationFeaturesType, _required, strict=DEFAULT_STRICT,
        docstring='Computed metadata regarding the input collections and '
                  'final product.')  # type: ExploitationFeaturesType
    DownstreamReprocessing = _SerializableDescriptor(
        'DownstreamReprocessing', DownstreamReprocessingType, _required, strict=DEFAULT_STRICT,
        docstring='Metadata describing any downstream processing of the '
                  'product.')  # type: Union[None, DownstreamReprocessingType]
    ErrorStatistics = _SerializableDescriptor(
        'ErrorStatistics', ErrorStatisticsType, _required, strict=DEFAULT_STRICT,
        docstring='Error statistics passed through from the SICD metadata.')  # type: Union[None, ErrorStatisticsType]
    Radiometric = _SerializableDescriptor(
        'Radiometric', RadiometricType, _required, strict=DEFAULT_STRICT,
        docstring='Radiometric information about the product.')  # type: Union[None, RadiometricType]
    MatchInfo = _SerializableDescriptor(
        'MatchInfo', MatchInfoType, _required, strict=DEFAULT_STRICT,
        docstring='Information about other collections that are matched to the current '
                  'collection. The current collection is the collection from which this '
                  'SIDD product was generated.')  # type: MatchInfoType
    Compression = _SerializableDescriptor(
        'Compression', CompressionType, _required, strict=DEFAULT_STRICT,
        docstring='Contains information regarding any compression that has occurred '
                  'to the image data.')  # type: CompressionType
    DigitalElevationData = _SerializableDescriptor(
        'DigitalElevationData', DigitalElevationDataType, _required, strict=DEFAULT_STRICT,
        docstring='This describes any Digital ElevatioNData included with '
                  'the SIDD product.')  # type: DigitalElevationDataType
    ProductProcessing = _SerializableDescriptor(
        'ProductProcessing', ProductProcessingType, _required, strict=DEFAULT_STRICT,
        docstring='Contains metadata related to algorithms used during '
                  'product generation.')  # type: ProductProcessingType
    Annotations = _SerializableDescriptor(
        'Annotations', AnnotationsType, _required, strict=DEFAULT_STRICT,
        docstring='List of annotations for the imagery.')  # type: AnnotationsType

    def __init__(self, ProductCreation=None, Display=None, GeoData=None,
                 Measurement=None, ExploitationFeatures=None, DownstreamReprocessing=None,
                 ErrorStatistics=None, Radiometric=None, MatchInfo=None, Compression=None,
                 DigitalElevationData=None, ProductProcessing=None, Annotations=None, **kwargs):
        """

        Parameters
        ----------
        ProductCreation : ProductCreationType
        Display : ProductDisplayType
        GeoData : GeoDataType
        Measurement : MeasurementType
        ExploitationFeatures : ExploitationFeaturesType
        DownstreamReprocessing : None|DownstreamReprocessingType
        ErrorStatistics : None|ErrorStatisticsType
        Radiometric : None|RadiometricType
        MatchInfo : None|MatchInfoType
        Compression : None|CompressionType
        DigitalElevationData : None|DigitalElevationDataType
        ProductProcessing : None|ProductProcessingType
        Annotations : None|AnnotationsType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ProductCreation = ProductCreation
        self.Display = Display
        self.GeoData = GeoData
        self.Measurement = Measurement
        self.ExploitationFeatures = ExploitationFeatures
        self.DownstreamReprocessing = DownstreamReprocessing
        self.ErrorStatistics = ErrorStatistics
        self.Radiometric = Radiometric
        self.MatchInfo = MatchInfo
        self.Compression = Compression
        self.DigitalElevationData = DigitalElevationData
        self.ProductProcessing = ProductProcessing
        self.Annotations = Annotations
        super(SIDDType, self).__init__(**kwargs)

    @staticmethod
    def get_xmlns_collection():
        """
        Gets the correct SIDD 2.0 dictionary of xml namespace details.

        Returns
        -------
        dict
        """

        return OrderedDict([
            ('xmlns', _SIDD_URN), ('xmlns:sicommon', _SICOMMON_URN),
            ('xmlns:sfa', _SFA_URN), ('xmlns:ism', _ISM_URN)])

    @staticmethod
    def get_des_details():
        """
        Gets the correct SIDD 2.0 DES subheader details.

        Returns
        -------
        dict
        """

        return OrderedDict([
            ('DESSHSI', _SIDD_SPECIFICATION_VERSION),
            ('DESSHSV', _SIDD_SPECIFICATION_VERSION),
            ('DESSHSD', _SIDD_SPECIFICATION_DATE),
            ('DESSHTN', _SIDD_URN)])

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if ns_key is None:
            raise ValueError('ns_key must be defined.')
        if xml_ns[ns_key].startswith('urn:SIDD:1.'):
            return SIDDType1.from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

        _validate_xml_ns(xml_ns, ns_key)
        return super(SIDDType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_xml_bytes(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        urn = self.get_xmlns_collection()
        return super(SIDDType, self).to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict)
