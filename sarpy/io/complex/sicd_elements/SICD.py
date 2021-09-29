"""
The SICDType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import logging
from copy import deepcopy
from collections import OrderedDict

import numpy

from sarpy.geometry import point_projection
from sarpy.io.complex.naming.utils import get_sicd_name
from sarpy.io.complex.sicd_schema import get_urn_details, get_specification_identifier

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import SerializableDescriptor

from .base import DEFAULT_STRICT
from .CollectionInfo import CollectionInfoType
from .ImageCreation import ImageCreationType
from .ImageData import ImageDataType
from .GeoData import GeoDataType
from .Grid import GridType
from .Timeline import TimelineType
from .Position import PositionType
from .RadarCollection import RadarCollectionType
from .ImageFormation import ImageFormationType
from .SCPCOA import SCPCOAType
from .Radiometric import RadiometricType
from .Antenna import AntennaType
from .ErrorStatistics import ErrorStatisticsType
from .MatchInfo import MatchInfoType
from .RgAzComp import RgAzCompType
from .PFA import PFAType
from .RMA import RMAType
from .validation_checks import detailed_validation_checks

logger = logging.getLogger(__name__)

#########
# Module variables
_SICD_SPECIFICATION_IDENTIFIER = get_specification_identifier()


_SICD_SPECIFICATION_NAMESPACE_1_2 = 'urn:SICD:1.2.1'
_details_1_2 = get_urn_details(_SICD_SPECIFICATION_NAMESPACE_1_2)
_SICD_SPECIFICATION_VERSION_1_2 = _details_1_2['version']
_SICD_SPECIFICATION_DATE_1_2 = _details_1_2['date']


_SICD_SPECIFICATION_NAMESPACE_1_1 = 'urn:SICD:1.1.0'
_details_1_1 = get_urn_details(_SICD_SPECIFICATION_NAMESPACE_1_1)
_SICD_SPECIFICATION_VERSION_1_1 = _details_1_1['version']
_SICD_SPECIFICATION_DATE_1_1 = _details_1_1['date']


class SICDType(Serializable):
    """
    Sensor Independent Complex Data object, containing all the relevant data to formulate products.
    """

    _fields = (
        'CollectionInfo', 'ImageCreation', 'ImageData', 'GeoData', 'Grid', 'Timeline', 'Position',
        'RadarCollection', 'ImageFormation', 'SCPCOA', 'Radiometric', 'Antenna', 'ErrorStatistics',
        'MatchInfo', 'RgAzComp', 'PFA', 'RMA')
    _required = (
        'CollectionInfo', 'ImageData', 'GeoData', 'Grid', 'Timeline', 'Position',
        'RadarCollection', 'ImageFormation', 'SCPCOA')
    _choice = ({'required': False, 'collection': ('RgAzComp', 'PFA', 'RMA')}, )
    # descriptors
    CollectionInfo = SerializableDescriptor(
        'CollectionInfo', CollectionInfoType, _required, strict=False,
        docstring='General information about the collection.')  # type: CollectionInfoType
    ImageCreation = SerializableDescriptor(
        'ImageCreation', ImageCreationType, _required, strict=False,
        docstring='General information about the image creation.')  # type: ImageCreationType
    ImageData = SerializableDescriptor(
        'ImageData', ImageDataType, _required, strict=False,  # it is senseless to not have this element
        docstring='The image pixel data.')  # type: ImageDataType
    GeoData = SerializableDescriptor(
        'GeoData', GeoDataType, _required, strict=False,
        docstring='The geographic coordinates of the image coverage area.')  # type: GeoDataType
    Grid = SerializableDescriptor(
        'Grid', GridType, _required, strict=False,
        docstring='The image sample grid.')  # type: GridType
    Timeline = SerializableDescriptor(
        'Timeline', TimelineType, _required, strict=False,
        docstring='The imaging collection time line.')  # type: TimelineType
    Position = SerializableDescriptor(
        'Position', PositionType, _required, strict=False,
        docstring='The platform and ground reference point coordinates as a function of time.')  # type: PositionType
    RadarCollection = SerializableDescriptor(
        'RadarCollection', RadarCollectionType, _required, strict=False,
        docstring='The radar collection information.')  # type: RadarCollectionType
    ImageFormation = SerializableDescriptor(
        'ImageFormation', ImageFormationType, _required, strict=False,
        docstring='The image formation process.')  # type: ImageFormationType
    SCPCOA = SerializableDescriptor(
        'SCPCOA', SCPCOAType, _required, strict=False,
        docstring='*Center of Aperture (COA)* for the *Scene Center Point (SCP)*.')  # type: SCPCOAType
    Radiometric = SerializableDescriptor(
        'Radiometric', RadiometricType, _required, strict=False,
        docstring='The radiometric calibration parameters.')  # type: RadiometricType
    Antenna = SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=False,
        docstring='Parameters that describe the antenna illumination patterns during the collection.'
    )  # type: AntennaType
    ErrorStatistics = SerializableDescriptor(
        'ErrorStatistics', ErrorStatisticsType, _required, strict=False,
        docstring='Parameters used to compute error statistics within the *SICD* sensor model.'
    )  # type: ErrorStatisticsType
    MatchInfo = SerializableDescriptor(
        'MatchInfo', MatchInfoType, _required, strict=False,
        docstring='Information about other collections that are matched to the '
                  'current collection. The current collection is the collection '
                  'from which this *SICD* product was generated.')  # type: MatchInfoType
    RgAzComp = SerializableDescriptor(
        'RgAzComp', RgAzCompType, _required, strict=False,
        docstring='Parameters included for a *Range, Doppler* image.')  # type: RgAzCompType
    PFA = SerializableDescriptor(
        'PFA', PFAType, _required, strict=False,
        docstring='Parameters included when the image is formed using the '
                  '*Polar Formation Algorithm (PFA)*.')  # type: PFAType
    RMA = SerializableDescriptor(
        'RMA', RMAType, _required, strict=False,
        docstring='Parameters included when the image is formed using the '
                  '*Range Migration Algorithm (RMA)*.')  # type: RMAType

    def __init__(self, CollectionInfo=None, ImageCreation=None, ImageData=None,
                 GeoData=None, Grid=None, Timeline=None, Position=None, RadarCollection=None,
                 ImageFormation=None, SCPCOA=None, Radiometric=None, Antenna=None,
                 ErrorStatistics=None, MatchInfo=None,
                 RgAzComp=None, PFA=None, RMA=None, **kwargs):
        """

        Parameters
        ----------
        CollectionInfo : CollectionInfoType
        ImageCreation : ImageCreationType
        ImageData : ImageDataType
        GeoData : GeoDataType
        Grid : GridType
        Timeline : TimelineType
        Position : PositionType
        RadarCollection : RadarCollectionType
        ImageFormation : ImageFormationType
        SCPCOA : SCPCOAType
        Radiometric : RadiometricType
        Antenna : AntennaType
        ErrorStatistics : ErrorStatisticsType
        MatchInfo : MatchInfoType
        RgAzComp : RgAzCompType
        PFA : PFAType
        RMA : RMAType
        kwargs : dict
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
        self.CollectionInfo = CollectionInfo
        self.ImageCreation = ImageCreation
        self.ImageData = ImageData
        self.GeoData = GeoData
        self.Grid = Grid
        self.Timeline = Timeline
        self.Position = Position
        self.RadarCollection = RadarCollection
        self.ImageFormation = ImageFormation
        self.SCPCOA = SCPCOA
        self.Radiometric = Radiometric
        self.Antenna = Antenna
        self.ErrorStatistics = ErrorStatistics
        self.MatchInfo = MatchInfo
        self.RgAzComp = RgAzComp
        self.PFA = PFA
        self.RMA = RMA
        super(SICDType, self).__init__(**kwargs)

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
        SICD file writing.

        Returns
        -------
        Dict
        """

        return self._NITF

    @property
    def ImageFormType(self):  # type: () -> str
        """
        str: *READ ONLY* Identifies the specific image formation type supplied. This is determined by
        returning the (first) attribute among `RgAzComp`, `PFA`, `RMA` which is populated. `OTHER` will be returned if
        none of them are populated.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return 'OTHER'

    def update_scp(self, point, coord_system='ECF'):
        """
        Modify the SCP point, and modify the associated SCPCOA fields.

        Parameters
        ----------
        point : numpy.ndarray|tuple|list
        coord_system : str
            Either 'ECF' or 'LLH', and 'ECF' will take precedence.

        Returns
        -------
        None
        """

        if isinstance(point, (list, tuple)):
            point = numpy.array(point, dtype='float64')
        if not isinstance(point, numpy.ndarray):
            raise TypeError('point must be an numpy.ndarray')
        if point.shape != (3, ):
            raise ValueError('point must be a one-dimensional, 3 element array')
        if coord_system == 'LLH':
            self.GeoData.SCP.LLH = point
        else:
            self.GeoData.SCP.ECF = point

        if self.SCPCOA is not None:
            self.SCPCOA.rederive(self.Grid, self.Position, self.GeoData)

    def _basic_validity_check(self):
        condition = super(SICDType, self)._basic_validity_check()
        condition &= detailed_validation_checks(self)
        return condition

    def is_valid(self, recursive=False, stack=False):
        all_required = self._basic_validity_check()
        if not recursive:
            return all_required

        valid_children = self._recursive_validity_check(stack=stack)
        return all_required & valid_children

    def define_geo_image_corners(self, override=False):
        """
        Defines the GeoData image corner points (if possible), if they are not already defined.

        Returns
        -------
        None
        """

        if self.GeoData is None:
            self.GeoData = GeoDataType()

        if self.GeoData.ImageCorners is not None and not override:
            return  # nothing to be done

        try:
            vertex_data = self.ImageData.get_full_vertex_data(dtype=numpy.float64)
            corner_coords = self.project_image_to_ground_geo(vertex_data)
        except (ValueError, AttributeError):
            return

        self.GeoData.ImageCorners = corner_coords

    def define_geo_valid_data(self):
        """
        Defines the GeoData valid data corner points (if possible), if they are not already defined.

        Returns
        -------
        None
        """

        if self.GeoData is None or self.GeoData.ValidData is not None:
            return  # nothing to be done

        try:
            valid_vertices = self.ImageData.get_valid_vertex_data(dtype=numpy.float64)
            if valid_vertices is not None:
                self.GeoData.ValidData = self.project_image_to_ground_geo(valid_vertices)
        except AttributeError:
            pass

    def derive(self):
        """
        Populates any potential derived data in the SICD structure. This should get called after reading an XML,
        or as a user desires.

        Returns
        -------
        None
        """

        # Note that there is dependency in calling order between steps - don't naively rearrange the following.
        if self.SCPCOA is None:
            self.SCPCOA = SCPCOAType()

        # noinspection PyProtectedMember
        self.SCPCOA._derive_scp_time(self.Grid)

        if self.Grid is not None:
            # noinspection PyProtectedMember
            self.Grid._derive_time_coa_poly(self.CollectionInfo, self.SCPCOA)

        # noinspection PyProtectedMember
        self.SCPCOA._derive_position(self.Position)

        if self.Position is None and self.SCPCOA.ARPPos is not None and \
                self.SCPCOA.ARPVel is not None and self.SCPCOA.SCPTime is not None:
            self.Position = PositionType()  # important parameter derived in the next step
        if self.Position is not None:
            # noinspection PyProtectedMember
            self.Position._derive_arp_poly(self.SCPCOA)

        if self.GeoData is not None:
            self.GeoData.derive()  # ensures both coordinate systems are defined for SCP

        if self.Grid is not None:
            # noinspection PyProtectedMember
            self.Grid.derive_direction_params(self.ImageData)

        if self.RadarCollection is not None:
            self.RadarCollection.derive()

        if self.ImageFormation is not None:
            # call after RadarCollection.derive(), and only if the entire transmitted bandwidth was used to process.
            # noinspection PyProtectedMember
            self.ImageFormation._derive_tx_frequency_proc(self.RadarCollection)

        # noinspection PyProtectedMember
        self.SCPCOA._derive_geometry_parameters(self.GeoData)

        # verify ImageFormation things make sense
        im_form_algo = None
        if self.ImageFormation is not None and self.ImageFormation.ImageFormAlgo is not None:
            im_form_algo = self.ImageFormation.ImageFormAlgo.upper()
        if im_form_algo == 'RGAZCOMP':
            # Check Grid settings
            if self.Grid is None:
                self.Grid = GridType()
            # noinspection PyProtectedMember
            self.Grid._derive_rg_az_comp(self.GeoData, self.SCPCOA, self.RadarCollection, self.ImageFormation)

            # Check RgAzComp settings
            if self.RgAzComp is None:
                self.RgAzComp = RgAzCompType()
            # noinspection PyProtectedMember
            self.RgAzComp._derive_parameters(self.Grid, self.Timeline, self.SCPCOA)
        elif im_form_algo == 'PFA':
            if self.PFA is None:
                self.PFA = PFAType()
            # noinspection PyProtectedMember
            self.PFA._derive_parameters(self.Grid, self.SCPCOA, self.GeoData, self.Position, self.Timeline)

            if self.Grid is not None:
                # noinspection PyProtectedMember
                self.Grid._derive_pfa(
                    self.GeoData, self.RadarCollection, self.ImageFormation, self.Position, self.PFA)
        elif im_form_algo == 'RMA':
            if self.RMA is not None:
                # noinspection PyProtectedMember
                self.RMA._derive_parameters(self.SCPCOA, self.Position, self.RadarCollection, self.ImageFormation)
            if self.Grid is not None:
                # noinspection PyProtectedMember
                self.Grid._derive_rma(self.RMA, self.GeoData, self.RadarCollection, self.ImageFormation, self.Position)

        self.define_geo_image_corners()
        self.define_geo_valid_data()
        if self.Radiometric is not None:
            # noinspection PyProtectedMember
            self.Radiometric._derive_parameters(self.Grid, self.SCPCOA)

    def get_transmit_band_name(self):
        """
        Gets the processed transmit band name.

        Returns
        -------
        str
        """

        if self.ImageFormation is None:
            return 'UN'
        return self.ImageFormation.get_transmit_band_name()

    def get_processed_polarization_abbreviation(self):
        """
        Gets the processed polarization abbreviation (two letters).

        Returns
        -------
        str
        """

        if self.ImageFormation is None:
            return 'UN'
        return self.ImageFormation.get_polarization_abbreviation()

    def get_processed_polarization(self):
        """
        Gets the processed polarization.

        Returns
        -------
        str
        """

        if self.ImageFormation is None:
            return 'UN'
        return self.ImageFormation.get_polarization()

    def apply_reference_frequency(self, reference_frequency):
        """
        If the reference frequency is used, adjust the necessary fields accordingly.

        Parameters
        ----------
        reference_frequency : float
            The reference frequency.

        Returns
        -------
        None
        """

        if self.RadarCollection is None:
            raise ValueError('RadarCollection is not defined. The reference frequency cannot be applied.')
        elif not self.RadarCollection.RefFreqIndex:  # it's None or 0
            raise ValueError(
                'RadarCollection.RefFreqIndex is not defined. The reference frequency should not be applied.')

        # noinspection PyProtectedMember
        self.RadarCollection._apply_reference_frequency(reference_frequency)
        if self.ImageFormation is not None:
            # noinspection PyProtectedMember
            self.ImageFormation._apply_reference_frequency(reference_frequency)
        if self.Antenna is not None:
            # noinspection PyProtectedMember
            self.Antenna._apply_reference_frequency(reference_frequency)
        if self.RMA is not None:
            # noinspection PyProtectedMember
            self.RMA._apply_reference_frequency(reference_frequency)

    def get_ground_resolution(self):
        """
        Gets the ground resolution for the sicd.

        Returns
        -------
        (float, float)
        """

        graze = numpy.deg2rad(self.SCPCOA.GrazeAng)
        twist = numpy.deg2rad(self.SCPCOA.TwistAng)
        row_ss = self.Grid.Row.SS
        col_ss = self.Grid.Col.SS

        row_ground = abs(float(row_ss/numpy.cos(graze)))
        col_ground = float(numpy.sqrt((numpy.tan(graze)*numpy.tan(twist)*row_ss)**2
                                      + (col_ss/numpy.cos(twist))**2))
        return row_ground, col_ground

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

        # GeoData elements?
        if self.GeoData is None:
            logger.error('Formulating a projection is not feasible because GeoData is not populated.')
            return False
        if self.GeoData.SCP is None:
            logger.error('Formulating a projection is not feasible because GeoData.SCP is not populated.')
            return False
        if self.GeoData.SCP.ECF is None:
            logger.error('Formulating a projection is not feasible because GeoData.SCP.ECF is not populated.')
            return False

        # ImageData elements?
        if self.ImageData is None:
            logger.error('Formulating a projection is not feasible because ImageData is not populated.')
            return False
        if self.ImageData.FirstRow is None:
            logger.error('Formulating a projection is not feasible because ImageData.FirstRow is not populated.')
            return False
        if self.ImageData.FirstCol is None:
            logger.error('Formulating a projection is not feasible because ImageData.FirstCol is not populated.')
            return False
        if self.ImageData.SCPPixel is None:
            logger.error('Formulating a projection is not feasible because ImageData.SCPPixel is not populated.')
            return False
        if self.ImageData.SCPPixel.Row is None:
            logger.error('Formulating a projection is not feasible because ImageData.SCPPixel.Row is not populated.')
            return False
        if self.ImageData.SCPPixel.Col is None:
            logger.error('Formulating a projection is not feasible because ImageData.SCPPixel.Col is not populated.')
            return False

        # Position elements?
        if self.Position is None:
            logger.error('Formulating a projection is not feasible because Position is not populated.')
            return False
        if self.Position.ARPPoly is None:
            logger.error('Formulating a projection is not feasible because Position.ARPPoly is not populated.')
            return False

        # Grid elements?
        if self.Grid is None:
            logger.error('Formulating a projection is not feasible because Grid is not populated.')
            return False
        if self.Grid.TimeCOAPoly is None:
            logger.warning(
                'Formulating a projection may be inaccurate, because Grid.TimeCOAPoly is not populated and '
                'a constant approximation will be used.')
        if self.Grid.Row is None:
            logger.error('Formulating a projection is not feasible because Grid.Row is not populated.')
            return False
        if self.Grid.Row.SS is None:
            logger.error('Formulating a projection is not feasible because Grid.Row.SS is not populated.')
            return False
        if self.Grid.Col is None:
            logger.error('Formulating a projection is not feasible because Grid.Col is not populated.')
            return False
        if self.Grid.Col.SS is None:
            logger.error('Formulating a projection is not feasible because Grid.Col.SS is not populated.')
            return False
        if self.Grid.Type is None:
            logger.error('Formulating a projection is not feasible because Grid.Type is not populated.')
            return False

        # specifics for Grid.Type value
        if self.Grid.Type == 'RGAZIM':
            if self.ImageFormation is None:
                logger.error(
                    'Formulating a projection is not feasible because Grid.Type = "RGAZIM",\n\t'
                    'but ImageFormation is not populated.')
                return False
            if self.ImageFormation.ImageFormAlgo is None:
                logger.error(
                    'Formulating a projection is not feasible because Grid.Type = "RGAZIM",\n\t'
                    'but ImageFormation.ImageFormAlgo is not populated.')
                return False

            if self.ImageFormation.ImageFormAlgo == 'PFA':
                if self.PFA is None:
                    logger.error(
                        'ImageFormation.ImageFormAlgo is "PFA",\n\t'
                        'but the PFA parameter is not populated.\n\t'
                        'No projection can be done.')
                    return False
                if self.PFA.PolarAngPoly is None:
                    logger.error(
                        'ImageFormation.ImageFormAlgo is "PFA",\n\t'
                        'but the PFA.PolarAngPoly parameter is not populated.\n\t'
                        'No projection can be done.')
                    return False
                if self.PFA.SpatialFreqSFPoly is None:
                    logger.error(
                        'ImageFormation.ImageFormAlgo is "PFA",\n\t'
                        'but the PFA.SpatialFreqSFPoly parameter is not populated.\n\t'
                        'No projection can be done.')
                    return False
            elif self.ImageFormation.ImageFormAlgo == 'RGAZCOMP':
                if self.RgAzComp is None:
                    logger.error(
                        'ImageFormation.ImageFormAlgo is "RGAZCOMP",\n\t'
                        'but the RgAzComp parameter is not populated.\n\t'
                        'No projection can be done.')
                    return False
                if self.RgAzComp.AzSF is None:
                    logger.error(
                        'ImageFormation.ImageFormAlgo is "RGAZCOMP",\n\t'
                        'but the RgAzComp.AzSF parameter is not populated.\n\t'
                        'No projection can be done.')
                    return False
            else:
                logger.error(
                    'Grid.Type = "RGAZIM", and got unhandled ImageFormation.ImageFormAlgo {}.\n\t'
                    'No projection can be done.'.format(self.ImageFormation.ImageFormAlgo))
                return False
        elif self.Grid.Type == 'RGZERO':
            if self.RMA is None or self.RMA.INCA is None:
                logger.error(
                    'Grid.Type is "RGZERO", but the RMA.INCA parameter is not populated.\n\t'
                    'No projection can be done.')
                return False
            if self.RMA.INCA.R_CA_SCP is None or self.RMA.INCA.TimeCAPoly is None \
                    or self.RMA.INCA.DRateSFPoly is None:
                logger.error(
                    'Grid.Type is "RGZERO", but the parameters\n\t'
                    'R_CA_SCP, TimeCAPoly, or DRateSFPoly of RMA.INCA parameter are not populated.\n\t'
                    'No projection can be done.')
                return False
        elif self.Grid.Type in ['XRGYCR', 'XCTYAT', 'PLANE']:
            if self.Grid.Row.UVectECF is None or self.Grid.Col.UVectECF is None:
                logger.error(
                    'Grid.Type is one of ["XRGYCR", "XCTYAT", "PLANE"], but the UVectECF parameter of '
                    'Grid.Row or Grid.Col is not populated.\n\t'
                    'No projection can be formulated.')
                return False
        else:
            logger.error(
                'Unhandled Grid.Type {},\n\t'
                'unclear how to formulate a projection.'.format(self.Grid.Type))
            return False

        # logger.info('Consider calling sicd.define_coa_projection if the sicd structure is defined.')
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

        self._coa_projection = point_projection.COAProjection.from_sicd(
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

    def populate_rniirs(self, signal=None, noise=None, override=False):
        """
        Given the signal and noise values (in sigma zero power units),
        calculate and populate an estimated RNIIRS value.

        Parameters
        ----------
        signal : None|float
        noise : None|float
        override : bool
            Override the value, if present.

        Returns
        -------
        None
        """

        from sarpy.processing.rgiqe import populate_rniirs_for_sicd
        populate_rniirs_for_sicd(self, signal=signal, noise=noise, override=override)

    def get_suggested_name(self, product_number=1):
        """
        Get the suggested name stem for the sicd and derived data.

        Returns
        -------
        str
        """

        sugg_name = get_sicd_name(self, product_number)
        if sugg_name is not None:
            return sugg_name
        elif self.CollectionInfo.CoreName is not None:
            return self.CollectionInfo.CoreName
        return 'Unknown_Sicd{}'.format(product_number)

    def get_des_details(self, check_version1_compliance=False):
        """
        Gets the correct current SICD DES subheader details.

        Parameters
        ----------
        check_version1_compliance : bool
            If true and structure is compatible, the version 1.1 information will
            be returned. Otherwise, the most recent supported version will be
            returned .

        Returns
        -------
        dict
        """

        if check_version1_compliance and (
                (self.ImageFormation is None or self.ImageFormation.permits_version_1_1()) and
                (self.RadarCollection is None or self.RadarCollection.permits_version_1_1())):
            spec_version = _SICD_SPECIFICATION_VERSION_1_1
            spec_date = _SICD_SPECIFICATION_DATE_1_1
            spec_ns = _SICD_SPECIFICATION_NAMESPACE_1_1
        else:
            spec_version = _SICD_SPECIFICATION_VERSION_1_2
            spec_date = _SICD_SPECIFICATION_DATE_1_2
            spec_ns = _SICD_SPECIFICATION_NAMESPACE_1_2

        return OrderedDict([
            ('DESSHSI', _SICD_SPECIFICATION_IDENTIFIER),
            ('DESSHSV', spec_version),
            ('DESSHSD', spec_date),
            ('DESSHTN', spec_ns)])

    def copy(self):
        """
        Provides a deep copy.

        Returns
        -------
        SICDType
        """

        out = super(SICDType, self).copy()
        out._NITF = deepcopy(self._NITF)
        return out

    def to_xml_bytes(self, urn=None, tag='SICD', check_validity=False, strict=DEFAULT_STRICT):
        if urn is None:
            urn = _SICD_SPECIFICATION_NAMESPACE_1_2
        return super(SICDType, self).to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict)

    def to_xml_string(self, urn=None, tag='SICD', check_validity=False, strict=DEFAULT_STRICT):
        return self.to_xml_bytes(urn=urn, tag=tag, check_validity=check_validity, strict=strict).decode('utf-8')
