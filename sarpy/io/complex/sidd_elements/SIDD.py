# -*- coding: utf-8 -*-
"""
The SIDDType definition.
"""

from typing import Union

# noinspection PyProtectedMember
from ..sicd_elements.base import Serializable, _SerializableDescriptor, DEFAULT_STRICT
from .ProductCreation import ProductCreationType
from .ProductDisplay import ProductDisplayType
from .GeographicAndTarget import GeographicAndTargetType
from .Measurement import MeasurementType
from .ExploitationFeatures import ExploitationFeaturesType
from .DownstreamReprocessing import DownstreamReprocessingType
from ..sicd_elements.ErrorStatistics import ErrorStatisticsType
from ..sicd_elements.Radiometric import RadiometricType
from ..sicd_elements.MatchInfo import MatchInfoType
from .Compression import CompressionType
from .DigitalElevationData import DigitalElevationDataType
from .ProductProcessing import ProductProcessingType
from .Annotations import AnnotationsType

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class SIDDType(Serializable):
    """
    The root element of the SIDD document.
    """

    _fields = (
        'ProductCreation', 'Display', 'GeographicAndTarget', 'Measurement', 'ExploitationFeatures',
        'DownstreamReprocessing', 'ErrorStatistics', 'Radiometric', 'MatchInfo', 'Compression',
        'DigitalElevationData', 'ProductProcessing', 'Annotations')
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

    def __init__(self, ProductCreation=None, Display=None, GeographicAndTarget=None,
                 Measurement=None, ExploitationFeatures=None, DownstreamReprocessing=None,
                 ErrorStatistics=None, Radiometric=None, MatchInfo=None, Compression=None,
                 DigitalElevationData=None, ProductProcessing=None, Annotations=None, **kwargs):
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
        MatchInfo : None|MatchInfoType
        Compression : None|CompressionType
        DigitalElevationData : None|DigitalElevationDataType
        ProductProcessing : None|ProductProcessingType
        Annotations : None|AnnotationsType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.ProductCreation = ProductCreation
        self.Display = Display
        self.GeographicAndTarget = GeographicAndTarget
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
