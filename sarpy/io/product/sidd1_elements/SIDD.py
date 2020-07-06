# -*- coding: utf-8 -*-
"""
The SIDDType 1.0 definition.
"""

import logging
from typing import Union
from collections import OrderedDict

# noinspection PyProtectedMember
from ...complex.sicd_elements.base import Serializable, _SerializableDescriptor, DEFAULT_STRICT
from ..sidd2_elements.ProductCreation import ProductCreationType
from .Display import ProductDisplayType
from .GeographicAndTarget import GeographicAndTargetType
from .Measurement import MeasurementType
from .ExploitationFeatures import ExploitationFeaturesType
from ..sidd2_elements.DownstreamReprocessing import DownstreamReprocessingType
from ..sidd2_elements.ProductProcessing import ProductProcessingType
from ..sidd2_elements.Annotations import AnnotationsType
from ..sidd2_elements.blocks import ErrorStatisticsType, RadiometricType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

############
# namespace validate and definitIon of required entries in the namespace dictionary
_SIDD_SPECIFICATION_IDENTIFIER = 'SIDD Volume 1 Design & Implementation Description Document'
_SIDD_SPECIFICATION_VERSION = '1.0'
_SIDD_SPECIFICATION_DATE = '2011-08-31T00:00:00Z'
_SIDD_URN = 'urn:SIDD:1.0.0'
_ISM_URN = 'urn:us:gov:ic:ism'
_SFA_URN = 'urn:SFA:1.2.0'
_SICOMMON_URN = 'urn:SICommon:0.1'


def _validate_sidd_urn(xml_ns, ns_key):
    if xml_ns is None:
        raise ValueError('xml_ns must not be None for SIDD interpretation.')

    if ns_key is None or ns_key not in xml_ns:
        raise ValueError('ns_key must be a key in xml_ns.')

    sidd_urn = xml_ns[ns_key]
    if sidd_urn != _SIDD_URN:
        logging.warning('SIDD version 1 urn is expected to be "{}", '
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
        logging.warning('SIDD version 1 "ism" namespace urn is expected to be "{}", '
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
        logging.warning('SIDD version 1 "SFA" namespace urn is expected to be "{}", '
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
        logging.warning('SIDD version 1 "SICommon" namespace urn is expected to be "{}", '
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
        docstring='Information related to processor, classification, '
                  'and product type.')  # type: ProductCreationType
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
        docstring='Contains the metadata necessary for performing '
                  'measurements.')  # type: MeasurementType
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
        docstring='Error statistics passed through from the SICD '
                  'metadata.')  # type: Union[None, ErrorStatisticsType]
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
                 ErrorStatistics=None, Radiometric=None, ProductProcessing=None,
                 Annotations=None, **kwargs):
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

    @staticmethod
    def get_xmlns_collection():
        """
        Gets the correct SIDD 1.0 dictionary of xml namespace details.

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
        Gets the correct SIDD 1.0 DES subheader details.

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
        _validate_xml_ns(xml_ns, ns_key)
        return super(SIDDType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_xml_bytes(self, urn=None, tag=None, check_validity=False, strict=DEFAULT_STRICT):
        urn = self.get_xmlns_collection()
        return super(SIDDType, self).to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict)
