# -*- coding: utf-8 -*-
"""
The SIDDType 1.0 definition.
"""

import logging
from typing import Union

# noinspection PyProtectedMember
from ...sicd_elements.base import Serializable, _SerializableDescriptor, DEFAULT_STRICT
from ..ProductCreation import ProductCreationType
from .ProductDisplay import ProductDisplayType
from .GeographicAndTarget import GeographicAndTargetType
from .Measurement import MeasurementType
from .ExploitationFeatures import ExploitationFeaturesType
from ..DownstreamReprocessing import DownstreamReprocessingType
from ..ProductProcessing import ProductProcessingType
from ..Annotations import AnnotationsType
from ..blocks import ErrorStatisticsType, RadiometricType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

############
# namespace validate and definiton of required entries in the namespace dictionary
_sidd_urn = 'urn:SIDD:1.0.0'
_ism_urn = 'urn:us:gov:ic:ism'
_sfa_urn = 'urn:SFA:1.2.0'
_sicommon_urn = 'urn:SICommon:0.1'


def _validate_sidd_urn(xml_ns, ns_key):
    if xml_ns is None:
        raise ValueError('xml_ns must not be None for SIDD interpretation.')

    if ns_key is None or ns_key not in xml_ns:
        raise ValueError('ns_key must be a key in xml_ns.')

    sidd_urn = xml_ns[ns_key]
    if sidd_urn != _sidd_urn:
        logging.warning('SIDD version 1 urn is expected to be "{}", '
                        'but we got "{}". Differences in standard may lead to deserialization '
                        'errors.'.format(_sidd_urn, sidd_urn))


def _validate_ism_urn(xml_ns):
    if 'ism' not in xml_ns:
        for key in xml_ns:
            val = xml_ns[key]
            if val.lower().startswith('urn:us:gov:ic:ism'):
                xml_ns['ism'] = val

    ism_urn = xml_ns['ism']
    if ism_urn != _ism_urn:
        logging.warning('SIDD version 1 "ism" namespace urn is expected to be "{}", '
                        'but we got "{}". Differences in standard may lead to deserialization '
                        'errors.'.format(_ism_urn, ism_urn))


def _validate_sfa_urn(xml_ns):
    if 'sfa' not in xml_ns:
        for key in xml_ns:
            val = xml_ns[key]
            if val.lower().startswith('urn:sfa:'):
                xml_ns['sfa'] = val

    sfa_urn = xml_ns['sfa']
    if sfa_urn != _sfa_urn:
        logging.warning('SIDD version 1 "SFA" namespace urn is expected to be "{}", '
                        'but we got "{}". Differences in standard may lead to deserialization '
                        'errors.'.format(_sfa_urn, sfa_urn))


def _validate_sicommon_urn(xml_ns):
    if 'sicommon' not in xml_ns:
        for key in xml_ns:
            val = xml_ns[key]
            if val.lower().startswith('urn:sicommon:'):
                xml_ns['sicommon'] = val

    sicommon_urn = xml_ns['sicommon']
    if sicommon_urn != _sicommon_urn:
        logging.warning('SIDD version 1 "SICommon" namespace urn is expected to be "{}", '
                        'but we got "{}". Differences in standard may lead to deserialization '
                        'errors.'.format(_sicommon_urn, sicommon_urn))


def _validate_xml_ns(xml_ns, ns_key):
    _validate_sidd_urn(xml_ns, ns_key)
    _validate_ism_urn(xml_ns)
    _validate_sfa_urn(xml_ns)
    _validate_sicommon_urn(xml_ns)


##########
# The SIDD object

class SIDDType(Serializable):
    """
    The root element of the SIDD 1.0 document.
    """

    _fields = (
        'ProductCreation', 'Display', 'GeographicAndTarget', 'Measurement', 'ExploitationFeatures',
        'DownstreamReprocessing', 'ErrorStatistics', 'Radiometric', 'ProductProcessing', 'Annotations')
    _required = (
        'ProductCreation', 'Display', 'GeographicAndTarget', 'Measurement', 'ExploitationFeatures')
    # Descriptor
    ProductCreation = _SerializableDescriptor(
        'ProductCreation', ProductCreationType, _required, strict=DEFAULT_STRICT,
        docstring='Information related to processor, classification, and product type.')  # type: ProductCreationType
    Display = _SerializableDescriptor(
        'Display', ProductDisplayType, _required, strict=DEFAULT_STRICT,
        docstring='Contains information on the parameters needed to display the product in '
                  'an exploitation tool.')  # type: ProductDisplayType
    GeographicAndTarget = _SerializableDescriptor(
        'GeographicAndTarget', GeographicAndTargetType, _required, strict=DEFAULT_STRICT,
        docstring='Contains generic and extensible targeting and geographic region '
                  'information.')  # type: GeographicAndTargetType
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
    ProductProcessing = _SerializableDescriptor(
        'ProductProcessing', ProductProcessingType, _required, strict=DEFAULT_STRICT,
        docstring='Contains metadata related to algorithms used during '
                  'product generation.')  # type: ProductProcessingType
    Annotations = _SerializableDescriptor(
        'Annotations', AnnotationsType, _required, strict=DEFAULT_STRICT,
        docstring='List of annotations for the imagery.')  # type: AnnotationsType

    def __init__(self, ProductCreation=None, Display=None, GeographicAndTarget=None,
                 Measurement=None, ExploitationFeatures=None, DownstreamReprocessing=None,
                 ErrorStatistics=None, Radiometric=None, ProductProcessing=None, Annotations=None, **kwargs):
        """

        Parameters
        ----------
        ProductCreation : ProductCreationType
        Display : ProductDisplayType
        GeographicAndTarget : GeographicAndTargetType
        Measurement : MeasurementType
        ExploitationFeatures : ExploitationFeaturesType
        DownstreamReprocessing : None|DownstreamReprocessingType
        ErrorStatistics : None|ErrorStatisticsType
        Radiometric : None|RadiometricType
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
        self.GeographicAndTarget = GeographicAndTarget
        self.Measurement = Measurement
        self.ExploitationFeatures = ExploitationFeatures
        self.DownstreamReprocessing = DownstreamReprocessing
        self.ErrorStatistics = ErrorStatistics
        self.Radiometric = Radiometric
        self.ProductProcessing = ProductProcessing
        self.Annotations = Annotations
        super(SIDDType, self).__init__(**kwargs)

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        _validate_xml_ns(xml_ns, ns_key)
        return super(SIDDType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)
