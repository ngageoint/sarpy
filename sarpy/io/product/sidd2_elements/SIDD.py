"""
The SIDDType 2.0 definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from typing import Union, Tuple
from collections import OrderedDict
from copy import deepcopy

import numpy

from sarpy.io.xml.base import Serializable, parse_xml_from_string, parse_xml_from_file
from sarpy.io.xml.descriptors import SerializableDescriptor
from sarpy.geometry import point_projection
from sarpy.io.product.sidd_schema import get_specification_identifier, \
    get_urn_details, validate_xml_ns

from .base import DEFAULT_STRICT
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


logger = logging.getLogger(__name__)

############
# namespace validate and definitIon of required entries in the namespace dictionary
_SIDD_SPECIFICATION_IDENTIFIER = get_specification_identifier()

_SIDD_URN = 'urn:SIDD:2.0.0'
_sidd_details = get_urn_details(_SIDD_URN)
_SIDD_SPECIFICATION_VERSION = _sidd_details['version']
_SIDD_SPECIFICATION_DATE = _sidd_details['date']
_ISM_URN = _sidd_details['ism_urn']
_SFA_URN = _sidd_details['sfa_urn']
_SICOMMON_URN = _sidd_details['sicommon_urn']


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
    ProductCreation = SerializableDescriptor(
        'ProductCreation', ProductCreationType, _required, strict=DEFAULT_STRICT,
        docstring='Information related to processor, classification, and product type.')  # type: ProductCreationType
    Display = SerializableDescriptor(
        'Display', ProductDisplayType, _required, strict=DEFAULT_STRICT,
        docstring='Contains information on the parameters needed to display the product in '
                  'an exploitation tool.')  # type: ProductDisplayType
    GeoData = SerializableDescriptor(
        'GeoData', GeoDataType, _required, strict=DEFAULT_STRICT,
        docstring='Contains generic and extensible targeting and geographic region '
                  'information.')  # type: GeoDataType
    Measurement = SerializableDescriptor(
        'Measurement', MeasurementType, _required, strict=DEFAULT_STRICT,
        docstring='Contains the metadata necessary for performing measurements.')  # type: MeasurementType
    ExploitationFeatures = SerializableDescriptor(
        'ExploitationFeatures', ExploitationFeaturesType, _required, strict=DEFAULT_STRICT,
        docstring='Computed metadata regarding the input collections and '
                  'final product.')  # type: ExploitationFeaturesType
    DownstreamReprocessing = SerializableDescriptor(
        'DownstreamReprocessing', DownstreamReprocessingType, _required, strict=DEFAULT_STRICT,
        docstring='Metadata describing any downstream processing of the '
                  'product.')  # type: Union[None, DownstreamReprocessingType]
    ErrorStatistics = SerializableDescriptor(
        'ErrorStatistics', ErrorStatisticsType, _required, strict=DEFAULT_STRICT,
        docstring='Error statistics passed through from the SICD metadata.')  # type: Union[None, ErrorStatisticsType]
    Radiometric = SerializableDescriptor(
        'Radiometric', RadiometricType, _required, strict=DEFAULT_STRICT,
        docstring='Radiometric information about the product.')  # type: Union[None, RadiometricType]
    MatchInfo = SerializableDescriptor(
        'MatchInfo', MatchInfoType, _required, strict=DEFAULT_STRICT,
        docstring='Information about other collections that are matched to the current '
                  'collection. The current collection is the collection from which this '
                  'SIDD product was generated.')  # type: MatchInfoType
    Compression = SerializableDescriptor(
        'Compression', CompressionType, _required, strict=DEFAULT_STRICT,
        docstring='Contains information regarding any compression that has occurred '
                  'to the image data.')  # type: CompressionType
    DigitalElevationData = SerializableDescriptor(
        'DigitalElevationData', DigitalElevationDataType, _required, strict=DEFAULT_STRICT,
        docstring='This describes any Digital ElevatioNData included with '
                  'the SIDD product.')  # type: DigitalElevationDataType
    ProductProcessing = SerializableDescriptor(
        'ProductProcessing', ProductProcessingType, _required, strict=DEFAULT_STRICT,
        docstring='Contains metadata related to algorithms used during '
                  'product generation.')  # type: ProductProcessingType
    Annotations = SerializableDescriptor(
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

        nitf = kwargs.get('_NITF', {})
        if not isinstance(nitf, dict):
            raise TypeError('Provided NITF options are required to be in dictionary form.')
        self._NITF = nitf

        self._coa_projection = None
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

    @property
    def coa_projection(self):
        """
        The COA Projection object, if previously defined through using :func:`define_coa_projection`.

        Returns
        -------
        None|sarpy.geometry.point_projection.COAProjection
        """

        return self._coa_projection

    @property
    def NITF(self):
        """
        Optional dictionary of NITF header information, pertains only to subsequent
        SIDD file writing.

        Returns
        -------
        Dict
        """

        return self._NITF

    def can_project_coordinates(self):
        """
        Determines whether the necessary elements are populated to permit projection
        between image and physical coordinates. If False, then the (first discovered)
        reason why not will be logged at error level.

        Returns
        -------
        bool
        """

        if self._coa_projection is not None:
            return True

        if self.Measurement.ProjectionType != 'PlaneProjection':
            logger.error(
                'Formulating a projection is only supported for PlaneProjection, '
                'got {}.'.format(self.Measurement.ProjectionType))
            return False
        return True

    def define_coa_projection(self, delta_arp=None, delta_varp=None, range_bias=None,
                              adj_params_frame='ECF', overide=True):
        """
        Define the COAProjection object.

        Parameters
        ----------
        delta_arp : None|numpy.ndarray|list|tuple
            ARP position adjustable parameter (ECF, m).  Defaults to 0 in each coordinate.
        delta_varp : None|numpy.ndarray|list|tuple
            VARP position adjustable parameter (ECF, m/s).  Defaults to 0 in each coordinate.
        range_bias : float|int
            Range bias adjustable parameter (m), defaults to 0.
        adj_params_frame : str
            One of ['ECF', 'RIC_ECF', 'RIC_ECI'], specifying the coordinate frame used for
            expressing `delta_arp` and `delta_varp` parameters.
        overide : bool
            should we redefine, if it is previously defined?

        Returns
        -------
        None
        """

        if not self.can_project_coordinates():
            logger.error('The COAProjection object cannot be defined.')
            return

        if self._coa_projection is not None and not overide:
            return

        self._coa_projection = point_projection.COAProjection.from_sidd(
            self, delta_arp=delta_arp, delta_varp=delta_varp, range_bias=range_bias,
            adj_params_frame=adj_params_frame)

    def project_ground_to_image(self, coords, **kwargs):
        """
        Transforms a 3D ECF point to pixel (row/column) coordinates. This is
        implemented in accordance with the SICD Image Projections Description Document.
        **Really Scene-To-Image projection.**"

        Parameters
        ----------
        coords : numpy.ndarray|tuple|list
            ECF coordinate to map to scene coordinates, of size `N x 3`.
        kwargs
            The keyword arguments for the :func:`sarpy.geometry.point_projection.ground_to_image` method.

        Returns
        -------
        Tuple[numpy.ndarray, float, int]
            * `image_points` - the determined image point array, of size `N x 2`. Following
              the SICD convention, he upper-left pixel is [0, 0].
            * `delta_gpn` - residual ground plane displacement (m).
            * `iterations` - the number of iterations performed.

        See Also
        --------
        sarpy.geometry.point_projection.ground_to_image
        """

        if 'use_structure_coa' not in kwargs:
            kwargs['use_structure_coa'] = True
        return point_projection.ground_to_image(coords, self, **kwargs)

    def project_ground_to_image_geo(self, coords, ordering='latlong', **kwargs):
        """
        Transforms a 3D Lat/Lon/HAE point to pixel (row/column) coordinates. This is
        implemented in accordance with the SICD Image Projections Description Document.
        **Really Scene-To-Image projection.**"

        Parameters
        ----------
        coords : numpy.ndarray|tuple|list
            ECF coordinate to map to scene coordinates, of size `N x 3`.
        ordering : str
            If 'longlat', then the input is `[longitude, latitude, hae]`.
            Otherwise, the input is `[latitude, longitude, hae]`. Passed through
            to :func:`sarpy.geometry.geocoords.geodetic_to_ecf`.
        kwargs
            The keyword arguments for the :func:`sarpy.geometry.point_projection.ground_to_image_geo` method.

        Returns
        -------
        Tuple[numpy.ndarray, float, int]
            * `image_points` - the determined image point array, of size `N x 2`. Following
              the SICD convention, he upper-left pixel is [0, 0].
            * `delta_gpn` - residual ground plane displacement (m).
            * `iterations` - the number of iterations performed.

        See Also
        --------
        sarpy.geometry.point_projection.ground_to_image_geo
        """

        if 'use_structure_coa' not in kwargs:
            kwargs['use_structure_coa'] = True
        return point_projection.ground_to_image_geo(coords, self, ordering=ordering, **kwargs)

    def project_image_to_ground(self, im_points, projection_type='HAE', **kwargs):
        """
        Transforms image coordinates to ground plane ECF coordinate via the algorithm(s)
        described in SICD Image Projections document.

        Parameters
        ----------
        im_points : numpy.ndarray|list|tuple
            the image coordinate array
        projection_type : str
            One of `['PLANE', 'HAE', 'DEM']`. Type `DEM` is a work in progress.
        kwargs
            The keyword arguments for the :func:`sarpy.geometry.point_projection.image_to_ground` method.

        Returns
        -------
        numpy.ndarray
            Ground Plane Point (in ECF coordinates) corresponding to the input image coordinates.

        See Also
        --------
        sarpy.geometry.point_projection.image_to_ground
        """

        if 'use_structure_coa' not in kwargs:
            kwargs['use_structure_coa'] = True
        return point_projection.image_to_ground(
            im_points, self, projection_type=projection_type, **kwargs)

    def project_image_to_ground_geo(self, im_points, ordering='latlong', projection_type='HAE', **kwargs):
        """
        Transforms image coordinates to ground plane WGS-84 coordinate via the algorithm(s)
        described in SICD Image Projections document.

        Parameters
        ----------
        im_points : numpy.ndarray|list|tuple
            the image coordinate array
        projection_type : str
            One of `['PLANE', 'HAE', 'DEM']`. Type `DEM` is a work in progress.
        ordering : str
            Determines whether return is ordered as `[lat, long, hae]` or `[long, lat, hae]`.
            Passed through to :func:`sarpy.geometry.geocoords.ecf_to_geodetic`.
        kwargs
            The keyword arguments for the :func:`sarpy.geometry.point_projection.image_to_ground_geo` method.

        Returns
        -------
        numpy.ndarray
            Ground Plane Point (in ECF coordinates) corresponding to the input image coordinates.
        See Also
        --------
        sarpy.geometry.point_projection.image_to_ground_geo
        """

        if 'use_structure_coa' not in kwargs:
            kwargs['use_structure_coa'] = True
        return point_projection.image_to_ground_geo(
            im_points, self, ordering=ordering, projection_type=projection_type, **kwargs)

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
            ('DESSHSI', _SIDD_SPECIFICATION_IDENTIFIER),
            ('DESSHSV', _SIDD_SPECIFICATION_VERSION),
            ('DESSHSD', _SIDD_SPECIFICATION_DATE),
            ('DESSHTN', _SIDD_URN)])

    @classmethod
    def from_node(cls, node, xml_ns, ns_key='default', kwargs=None):
        if ns_key is None:
            raise ValueError('ns_key must be defined.')
        if ns_key not in xml_ns:
            raise ValueError('ns_key {} is not in the xml namespace'.format(ns_key))
        if xml_ns[ns_key].startswith('urn:SIDD:1.'):
            return SIDDType1.from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

        valid_ns = validate_xml_ns(xml_ns, ns_key)
        if not xml_ns[ns_key].startswith('urn:SIDD:2.'):
            raise ValueError('Cannot use urn {} for SIDD version 2.0'.format(xml_ns[ns_key]))
        if not valid_ns:
            logger.warning(
                'SIDD namespace validation failed,\n\t'
                'which may lead to subsequent deserialization failures')
        return super(SIDDType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_xml_bytes(self, urn=None, tag='SIDD', check_validity=False, strict=DEFAULT_STRICT):
        if urn is None:
            urn = self.get_xmlns_collection()
        return super(SIDDType, self).to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict)

    def to_xml_string(self, urn=None, tag='SIDD', check_validity=False, strict=DEFAULT_STRICT):
        return self.to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict).decode('utf-8')

    def copy(self):
        """
        Provides a deep copy.

        Returns
        -------
        SIDDType
        """

        out = super(SIDDType, self).copy()
        out._NITF = deepcopy(self._NITF)
        return out

    @classmethod
    def from_xml_file(cls, file_path):
        """
        Construct the sidd object from a stand-alone xml file path.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        SIDDType
        """

        root_node, xml_ns = parse_xml_from_file(file_path)
        ns_key = 'default' if 'default' in xml_ns else None
        return cls.from_node(root_node, xml_ns=xml_ns, ns_key=ns_key)

    @classmethod
    def from_xml_string(cls, xml_string):
        """
        Construct the sidd object from an xml string.

        Parameters
        ----------
        xml_string : str|bytes

        Returns
        -------
        SIDDType
        """

        root_node, xml_ns = parse_xml_from_string(xml_string)
        ns_key = 'default' if 'default' in xml_ns else None
        return cls.from_node(root_node, xml_ns=xml_ns, ns_key=ns_key)
