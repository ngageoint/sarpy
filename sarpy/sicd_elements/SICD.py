"""
The SICDType definition.
"""

import logging

import numpy
from numpy.polynomial import polynomial as numpy_poly

from ._base import Serializable, DEFAULT_STRICT, _SerializableDescriptor
from ._blocks import Poly1DType, Poly2DType, XYZType, XYZPolyType

from .CollectionInfo import CollectionInfoType
from .ImageCreation import ImageCreationType
from .ImageData import ImageDataType
from .GeoData import GeoDataType
from .Grid import GridType
from .Timeline import TimelineType
from .Position import PositionType
from .RadarCollection import RadarCollectionType
from .ImageFormation import ImageFormationType, TxFrequencyProcType
from .SCPCOA import SCPCOAType
from .Radiometric import RadiometricType
from .Antenna import AntennaType
from .ErrorStatistics import ErrorStatisticsType
from .MatchInfo import MatchInfoType
from .RgAzComp import RgAzCompType
from .PFA import PFAType
from .RMA import RMAType


# TODO:
#   1.) implement the necessary sicd version 0.4 & 0.5 compatibility manipulations - noted in the body.
#   2.) determine necessary and appropriate formatting issues for serialization/deserialization
#       i.) proper precision for numeric serialization
#       ii.) is there any ridiculous formatting for latitude or longitude?
#   3.) determine and implement appropriate class methods for proper functionality
#       how are things used, and what helper functions do we need?


__classification__ = "UNCLASSIFIED"


class SICDType(Serializable):
    """Sensor Independent Complex Data object, containing all the relevant data to formulate products."""
    _fields = (
        'CollectionInfo', 'ImageCreation', 'ImageData', 'GeoData', 'Grid', 'Timeline', 'Position',
        'RadarCollection', 'ImageFormation', 'SCPCOA', 'Radiometric', 'Antenna', 'ErrorStatistics',
        'MatchInfo', 'RgAzComp', 'PFA', 'RMA')
    _required = (
        'CollectionInfo', 'ImageData', 'GeoData', 'Grid', 'Timeline', 'Position',
        'RadarCollection', 'ImageFormation', 'SCPCOA')
    _choice = ({'required': False, 'collection': ('RgAzComp', 'PFA', 'RMA')}, )
    # descriptors
    CollectionInfo = _SerializableDescriptor(
        'CollectionInfo', CollectionInfoType, _required, strict=DEFAULT_STRICT,
        docstring='General information about the collection.')  # type: CollectionInfoType
    ImageCreation = _SerializableDescriptor(
        'ImageCreation', ImageCreationType, _required, strict=DEFAULT_STRICT,
        docstring='General information about the image creation.')  # type: ImageCreationType
    ImageData = _SerializableDescriptor(
        'ImageData', ImageDataType, _required, strict=DEFAULT_STRICT,
        docstring='The image pixel data.')  # type: ImageDataType
    GeoData = _SerializableDescriptor(
        'GeoData', GeoDataType, _required, strict=DEFAULT_STRICT,
        docstring='The geographic coordinates of the image coverage area.')  # type: GeoDataType
    Grid = _SerializableDescriptor(
        'Grid', GridType, _required, strict=DEFAULT_STRICT,
        docstring='The image sample grid.')  # type: GridType
    Timeline = _SerializableDescriptor(
        'Timeline', TimelineType, _required, strict=DEFAULT_STRICT,
        docstring='The imaging collection time line.')  # type: TimelineType
    Position = _SerializableDescriptor(
        'Position', PositionType, _required, strict=DEFAULT_STRICT,
        docstring='The platform and ground reference point coordinates as a function of time.')  # type: PositionType
    RadarCollection = _SerializableDescriptor(
        'RadarCollection', RadarCollectionType, _required, strict=DEFAULT_STRICT,
        docstring='The radar collection information.')  # type: RadarCollectionType
    ImageFormation = _SerializableDescriptor(
        'ImageFormation', ImageFormationType, _required, strict=DEFAULT_STRICT,
        docstring='The image formation process.')  # type: ImageFormationType
    SCPCOA = _SerializableDescriptor(
        'SCPCOA', SCPCOAType, _required, strict=DEFAULT_STRICT,
        docstring='Center of Aperture (COA) for the Scene Center Point (SCP).')  # type: SCPCOAType
    Radiometric = _SerializableDescriptor(
        'Radiometric', RadiometricType, _required, strict=DEFAULT_STRICT,
        docstring='The radiometric calibration parameters.')  # type: RadiometricType
    Antenna = _SerializableDescriptor(
        'Antenna', AntennaType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that describe the antenna illumination patterns during the collection.'
    )  # type: AntennaType
    ErrorStatistics = _SerializableDescriptor(
        'ErrorStatistics', ErrorStatisticsType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters used to compute error statistics within the SICD sensor model.'
    )  # type: ErrorStatisticsType
    MatchInfo = _SerializableDescriptor(
        'MatchInfo', MatchInfoType, _required, strict=DEFAULT_STRICT,
        docstring='Information about other collections that are matched to the current collection. The current '
                  'collection is the collection from which this SICD product was generated.')  # type: MatchInfoType
    RgAzComp = _SerializableDescriptor(
        'RgAzComp', RgAzCompType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters included for a Range, Doppler image.')  # type: RgAzCompType
    PFA = _SerializableDescriptor(
        'PFA', PFAType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters included when the image is formed using the Polar Formation Algorithm.')  # type: PFAType
    RMA = _SerializableDescriptor(
        'RMA', RMAType, _required, strict=DEFAULT_STRICT,
        docstring='Parameters included when the image is formed using the Range Migration Algorithm.')  # type: RMAType

    @property
    def ImageFormType(self):  # type: () -> str
        """
        str: READ ONLY attribute. Identifies the specific image formation type supplied. This is determined by
        returning the (first) attribute among `RgAzComp`, `PFA`, `RMA` which is populated. `OTHER` will be returned if
        none of them are populated.
        """

        for attribute in self._choice[0]['collection']:
            if getattr(self, attribute) is not None:
                return attribute
        return 'OTHER'

    def _validate_image_segment_id(self):  # type: () -> bool
        if self.ImageFormation is None or self.RadarCollection is None:
            return False

        # get the segment identifier
        seg_id = self.ImageFormation.SegmentIdentifier
        # get the segment list
        try:
            seg_list = self.RadarCollection.Area.Plane.SegmentList
        except AttributeError:
            seg_list = None

        if seg_id is None:
            if seg_list is None:
                return True
            else:
                logging.error(
                    'ImageFormation.SegmentIdentifier is not populated, but RadarCollection.Area.Plane.SegmentList '
                    'is populated. ImageFormation.SegmentIdentifier should be set to identify the appropriate segment.')
                return False
        else:
            if seg_list is None:
                logging.error(
                    'ImageFormation.SegmentIdentifier is populated as {}, but RadarCollection.Area.Plane.SegmentList '
                    'is not populated.'.format(seg_id))
                return False
            else:
                # let's double check that seg_id is sensibly populated
                the_ids = [entry.Identifier for entry in seg_list]
                if seg_id in the_ids:
                    return True
                else:
                    logging.error(
                        'ImageFormation.SegmentIdentifier is populated as {}, but this is not one of the possible '
                        'identifiers in the RadarCollection.Area.Plane.SegmentList definition {}. '
                        'ImageFormation.SegmentIdentifier should be set to identify the '
                        'appropriate segment.'.format(seg_id, the_ids))
                    return False

    def _validate_image_form(self):  # type: () -> bool
        if self.ImageFormation is None:
            logging.error(
                'ImageFormation attribute is not populated, and ImagFormType is {}. This '
                'cannot be valid.'.format(self.ImageFormType))
            return False  # nothing more to be done.

        alg_types = []
        for alg in ['RgAzComp', 'PFA', 'RMA']:
            if getattr(self, alg) is not None:
                alg_types.append(alg)

        if len(alg_types) > 1:
            logging.error(
                'ImageFormation.ImageFormAlgo is set as {}, and multiple SICD image formation parameters {} are set. '
                'Only one image formation algorithm should be set, and ImageFormation.ImageFormAlgo '
                'should match.'.format(self.ImageFormation.ImageFormAlgo, alg_types))
            return False
        elif len(alg_types) == 0:
            if self.ImageFormation.ImageFormAlgo is None:
                # TODO: is this correct?
                logging.warning(
                    'ImageFormation.ImageFormAlgo is not set, and there is no corresponding RgAzComp, PFA, or RMA '
                    'SICD parameters. Setting it to "OTHER".'.format(self.ImageFormation.ImageFormAlgo))
                self.ImageFormation.ImageFormAlgo = 'OTHER'
                return True
            elif self.ImageFormation.ImageFormAlgo != 'OTHER':
                logging.error(
                    'No RgAzComp, PFA, or RMA SICD parameters exist, but ImageFormation.ImageFormAlgo '
                    'is set as {}.'.format(self.ImageFormation.ImageFormAlgo))
                return False
            return True
        else:
            if self.ImageFormation.ImageFormAlgo == alg_types[0].upper():
                return True
            elif self.ImageFormation.ImageFormAlgo is None:
                logging.warning(
                    'Image formation algorithm(s) {} is populated, but ImageFormation.ImageFormAlgo was not set. '
                    'ImageFormation.ImageFormAlgo has been set.'.format(alg_types[0]))
                self.ImageFormation.ImageFormAlgo = alg_types[0].upper()
                return True
            else:  # they are different values
                # TODO: is resetting it the correct decision?
                logging.warning(
                    'Only the image formation algorithm {} is populated, but ImageFormation.ImageFormAlgo '
                    'was set as {}. ImageFormation.ImageFormAlgo has been '
                    'changed.'.format(alg_types[0], self.ImageFormation.ImageFormAlgo))
                self.ImageFormation.ImageFormAlgo = alg_types[0].upper()
                return True

    def _basic_validity_check(self):
        condition = super(SICDType, self)._basic_validity_check()
        # do our image formation parameters match, as appropriate?
        condition &= self._validate_image_form()
        # does the image formation segment identifier and radar collection make sense?
        condition &= self._validate_image_segment_id()
        return condition

    def derive(self):
        """
        Populates any potential derived data in the SICD structure. This should get called after reading an XML,
        or as a user desires.

        Returns
        -------
        None
        """

        def derive_scp_time():
            if self.SCPCOA is not None and self.SCPCOA.SCPTime is None:
                if self.Grid is None or self.Grid.TimeCOAPoly is None:
                    return
                self.SCPCOA.SCPTime = self.Grid.TimeCOAPoly.Coefs[0, 0]

        def derive_time_coa_poly():
            # this only works for spotlight mode collect
            if self.SCPCOA is not None and self.SCPCOA.SCPTime is not None and \
                    self.Grid is not None and self.Grid.TimeCOAPoly is None:
                try:
                    if self.CollectionInfo.RadarMode.ModeType == 'SPOTLIGHT':
                        self.Grid.TimeCOAPoly = Poly2DType(Coefs=[[self.SCPCOA.SCPTime, ], ])
                except (AttributeError, ValueError):
                    pass

        def derive_aperture_position():
            if self.Position is not None and self.Position.ARPPoly is not None and \
                    self.SCPCOA is not None and self.SCPCOA.SCPTime is not None:
                # set aperture position, velocity, and acceleration at scptime from position polynomial, if necessary
                poly = self.Position.ARPPoly
                scptime = self.SCPCOA.SCPTime

                if self.SCPCOA.ARPPos is None:
                    self.SCPCOA.ARPPos = XYZType(X=poly.X(scptime),
                                                 Y=poly.Y(scptime),
                                                 Z=poly.Z(scptime))
                if self.SCPCOA.ARPVel is None:
                    self.SCPCOA.ARPVel = XYZType(X=numpy_poly.polyval(scptime, numpy_poly.polyder(poly.X.Coefs, 1)),
                                                 Y=numpy_poly.polyval(scptime, numpy_poly.polyder(poly.Y.Coefs, 1)),
                                                 Z=numpy_poly.polyval(scptime, numpy_poly.polyder(poly.Z.Coefs, 1)))
                if self.SCPCOA.ARPAcc is None:
                    self.SCPCOA.ARPAcc = XYZType(X=numpy_poly.polyval(scptime, numpy_poly.polyder(poly.X.Coefs, 2)),
                                                 Y=numpy_poly.polyval(scptime, numpy_poly.polyder(poly.Y.Coefs, 2)),
                                                 Z=numpy_poly.polyval(scptime, numpy_poly.polyder(poly.Z.Coefs, 2)))
            elif self.SCPCOA is not None and self.SCPCOA.ARPPos is not None and \
                    self.SCPCOA.ARPVel is not None and self.SCPCOA.SCPTime is not None:
                # set the aperture position polynomial from position, time, acceleration at scptime, if necessary
                # NB: this assumes constant velocity and acceleration. Maybe that's not terrible?
                if self.Position is None:
                    self.Position = PositionType()

                if self.Position.ARPPoly is None:
                    if self.SCPCOA.ARPAcc is None:
                        self.SCPCOA.ARPAcc = XYZType(X=0, Y=0, Z=0)
                    # define the polynomial
                    coefs = numpy.zeros((3, 3), dtype=numpy.float64)
                    scptime = self.SCPCOA.SCPTime
                    pos = self.SCPCOA.ARPPos.get_array()
                    vel = self.SCPCOA.ARPVel.get_array()
                    acc = self.SCPCOA.ARPAcc.get_array()
                    coefs[0, :] = pos - vel*scptime + 0.5*acc*scptime*scptime
                    coefs[1, :] = vel - acc*scptime
                    coefs[2, :] = acc
                    self.Position.ARPPoly = XYZPolyType(X=Poly1DType(Coefs=coefs[:, 0]),
                                                        Y=Poly1DType(Coefs=coefs[:, 1]),
                                                        Z=Poly1DType(Coefs=coefs[:, 2]))

        def derive_image_formation():
            # TODO: introduce some boolean option for default settings?
            if self.ImageFormation is None:
                return  # nothing to be done
            if self.RadarCollection is not None and self.RadarCollection.TxFrequency.Min is not None and \
                    self.RadarCollection.TxFrequency.Max is not None:
                # this is based on the assumption that the entire transmitted bandwidth was processed.
                if self.ImageFormation.TxFrequencyProc is None:
                    self.ImageFormation.TxFrequencyProc = TxFrequencyProcType(
                        ProcMin=self.RadarCollection.TxFrequency.Min, ProcMax=self.RadarCollection.TxFrequency.Max)
                elif self.ImageFormation.TxFrequencyProc.MinProc is None:  # does this really make sense?
                    self.ImageFormation.TxFrequencyProc.MinProc = self.RadarCollection.TxFrequency.Min
                elif self.ImageFormation.TxFrequencyProc.MaxProc is None:
                    self.ImageFormation.TxFrequencyProc.MaxProc = self.RadarCollection.TxFrequency.Max

        derive_scp_time()
        derive_time_coa_poly()
        derive_aperture_position()

        if self.GeoData is not None:
            self.GeoData.derive()  # ensures both coordinate systems are defined for SCP

        if self.Grid is not None:
            if self.ImageData is None:
                self.Grid.derive(None)  # derives Row/Col parameters internal to Grid
            else:
                self.Grid.derive(self.ImageData.get_valid_vertex_data())

        if self.RadarCollection is not None:
            self.RadarCollection.derive()

        derive_image_formation()  # call after RadarCollection.derive()

        if self.SCPCOA is not None and self.GeoData is not None and self.GeoData.SCP is not None and \
                self.GeoData.SCP.ECF is not None:
            self.SCPCOA.derive(self.GeoData.SCP.ECF.get_array())

        # TODO: continue here from sicd.py 1610-2013

# TODO: properly incorporate derived fields kludgery. See sicd.py line 1261.
#  This is quite long and unmodular. This should be implemented at the proper level,
#  and then recursively called, but not until we are sure that we are done with construction.
