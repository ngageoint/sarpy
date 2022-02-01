"""
Definition for the SensorInfo NGA modified RDE/AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = "Thomas McCullough"


from typing import Optional
import numpy

from sarpy.io.xml.base import Serializable, Arrayable
from sarpy.io.xml.descriptors import SerializableDescriptor, StringDescriptor, \
    StringEnumDescriptor, FloatDescriptor
from sarpy.io.complex.sicd_elements.blocks import XYZType
from sarpy.io.complex.sicd_elements.SICD import SICDType
from sarpy.geometry.geocoords import ecf_to_geodetic, ecf_to_ned

from .blocks import LatLonEleType


class BeamWidthType(Serializable, Arrayable):
    _fields = ('Azimuth', 'Elevation')
    _required = _fields
    _numeric_format = {key: '0.17G' for key in _fields}
    # Descriptors
    Azimuth = FloatDescriptor(
        'Azimuth', _required, strict=True, docstring='The Azimuth attribute.')  # type: float
    Elevation = FloatDescriptor(
        'Elevation', _required, strict=True, docstring='The Elevation attribute.')  # type: float

    def __init__(self, Azimuth=None, Elevation=None, **kwargs):
        """
        Parameters
        ----------
        Azimuth : float
        Elevation : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Azimuth, self.Elevation = Azimuth, Elevation
        super(BeamWidthType, self).__init__(**kwargs)

    def get_array(self, dtype='float64'):
        """
        Gets an array representation of the class instance.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            numpy data type of the return

        Returns
        -------
        numpy.ndarray
            array of the form [Azimuth, Elevation]
        """

        return numpy.array([self.Azimuth, self.Elevation], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Azimuth, Elevation]

        Returns
        -------
        BeamWidthType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 2:
                raise ValueError('Expected array to be of length 2, and received {}'.format(array))
            return cls(Azimuth=array[0], Elevation=array[1])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class SquintAngleType(Serializable):
    _fields = ('GroundPlane', 'SlantPlane')
    _required = _fields
    _numeric_format = {el: '0.17G' for el in _fields}
    # descriptor
    GroundPlane = FloatDescriptor(
        'GroundPlane', _required,
        docstring='Measured angle between the sensor line-of-sight and the '
                  'lateral axis of the aircraft as projected into the'
                  'ground plane')  # type: float
    SlantPlane = FloatDescriptor(
        'SlantPlane', _required,
        docstring='Measured angle between the sensor line-of-sight and the '
                  'lateral axis of the aircraft as projected into the'
                  'slant plane')  # type: float

    def __init__(self, GroundPlane=None, SlantPlane=None, **kwargs):
        """
        Parameters
        ----------
        GroundPlane : float
        SlantPlane : float
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.GroundPlane = GroundPlane
        self.SlantPlane = SlantPlane
        super(SquintAngleType, self).__init__(**kwargs)


class AircraftLocationType(Serializable, Arrayable):
    """A three-dimensional geographic point in WGS-84 coordinates."""
    _fields = ('Lat', 'Lon', 'Altitude')
    _required = _fields
    _numeric_format = {'Lat': '0.17G', 'Lon': '0.17G', 'Altitude': '0.17G'}
    # descriptors
    Lat = FloatDescriptor(
        'Lat', _required, strict=True,
        docstring='The latitude attribute. Assumed to be WGS-84 coordinates.'
    )  # type: float
    Lon = FloatDescriptor(
        'Lon', _required, strict=True,
        docstring='The longitude attribute. Assumed to be WGS-84 coordinates.'
    )  # type: float
    Altitude = FloatDescriptor(
        'Altitude', _required, strict=True,
        docstring='The Height Above Ellipsoid (in meters) attribute. '
                  'Assumed to be WGS-84 coordinates.')  # type: float

    def __init__(self, Lat=None, Lon=None, Altitude=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        Altitude : float
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Lat = Lat
        self.Lon = Lon
        self.Altitude = Altitude
        super(AircraftLocationType, self).__init__(**kwargs)

    def get_array(self, dtype=numpy.float64):
        """
        Gets an array representation of the data.

        Parameters
        ----------
        dtype : str|numpy.dtype|numpy.number
            data type of the return

        Returns
        -------
        numpy.ndarray
            data array with appropriate entry order
        """

        return numpy.array([self.Lat, self.Lon, self.Altitude], dtype=dtype)

    @classmethod
    def from_array(cls, array):
        """
        Create from an array type entry.

        Parameters
        ----------
        array: numpy.ndarray|list|tuple
            assumed [Lat, Lon, Altitude]

        Returns
        -------
        AircraftLocationType
        """
        if array is None:
            return None
        if isinstance(array, (numpy.ndarray, list, tuple)):
            if len(array) < 3:
                raise ValueError('Expected array to be of length 3, and received {}'.format(array))
            return cls(Lat=array[0], Lon=array[1], Altitude=array[2])
        raise ValueError('Expected array to be numpy.ndarray, list, or tuple, got {}'.format(type(array)))


class SensorInfoType(Serializable):
    _fields = (
        'Name', 'SensorMfg', 'OperatingAgency', 'Type', 'Mode', 'Band',
        'Bandwidth', 'CenterFrequency', 'NearRange', 'SlantRangeSwathWidth',
        'Polarization', 'Range', 'DepressionAngle', 'LinearDynamicRange',
        'BeamWidth', 'Aimpoint', 'AircraftHeading', 'AircraftTrackAngle', 'Look', 'SquintAngle',
        'AircraftLocation', 'AircraftVelocity', 'FlightNumber', 'PassNumber')
    _required = (
        'Name', 'Type', 'Band', 'Bandwidth', 'CenterFrequency', 'Polarization', 'Range',
        'DepressionAngle', 'Aimpoint', 'AircraftHeading', 'AircraftTrackAngle',
        'Look', 'SquintAngle', 'AircraftLocation', 'AircraftVelocity')
    _numeric_format = {
        'Bandwidth': '0.17G', 'CenterFrequency': '0.17G', 'NearRange': '0.17G',
        'SlantRangeSwathWidth': '0.17G', 'Range': '0.17G', 'DepressionAngle': '0.17G',
        'LinearDynamicRange': '0.17G', 'AircraftHeading': '0.17G', 'AircraftTrackAngle': '0.17G',}
    # descriptors
    Name = StringDescriptor(
        'Name', _required,
        docstring='The name of the sensor')  # type: str
    SensorMfg = StringDescriptor(
        'SensorMfg', _required,
        docstring='The manufacturer of the sensor')  # type: Optional[str]
    OperatingAgency = StringDescriptor(
        'OperatingAgency', _required,
        docstring='The agency or company that operates the sensor')  # type: Optional[str]
    Type = StringDescriptor(
        'Type', _required,
        docstring='The type of sensor (i.e SAR or EO)')  # type: str
    Mode = StringDescriptor(
        'Mode', _required,
        docstring='Sensor operating mode')  # type: Optional[str]
    Band = StringDescriptor(
        'Band', _required,
        docstring='designation of the sensor frequency band')  # type: str
    Bandwidth = FloatDescriptor(
        'Bandwidth', _required,
        docstring='Radio Frequency bandwidth of the sensor system in GHz')  # type: float
    CenterFrequency = FloatDescriptor(
        'CenterFrequency', _required,
        docstring='Center operating frequency of the sensor system in GHz')  # type: float
    NearRange = FloatDescriptor(
        'NearRange', _required,
        docstring='The slant range distance measured from the sensor to the '
                  'near range of the image')  # type: Optional[float]
    SlantRangeSwathWidth = FloatDescriptor(
        'SlantRangeSwathWidth', _required,
        docstring='The width of the image as measured in the slant range'
    )  # type: Optional[float]
    Polarization = StringDescriptor(
        'Polarization', _required,
        docstring='The polarization of the transmitted/received signals')  # type: str
    Range = FloatDescriptor(
        'Range', _required,
        docstring='Measured slant range between the sensor aperture '
                  'and the scene center')  # type: float
    DepressionAngle = FloatDescriptor(
        'DepressionAngle', _required,
        docstring='Measured depression angle between the sensor line-of-sight '
                  'and the local horizontal reference plane')  # type: float
    LinearDynamicRange = FloatDescriptor(
        'LinearDynamicRange', _required,
        docstring="The span of the signal amplitudes (or power levels) over "
                  "which the system's response is linear. Typically the ratio "
                  "of the largest input signal that causes a 1 db compression "
                  "in receiver dynamic gain and the minimum signal defined by "
                  "receiver noise.")  # type: Optional[float]
    BeamWidth = SerializableDescriptor(
        'BeamWidth', BeamWidthType, _required,
        docstring='The width of the radar beam at its half power'
    )  # type: Optional[BeamWidthType]
    Aimpoint = SerializableDescriptor(
        'Aimpoint', LatLonEleType, _required,
        docstring='The sensor aim point')  # type: LatLonEleType
    AircraftHeading = FloatDescriptor(
        'AircraftHeading', _required,
        docstring='Aircraft heading relative to True North, in degrees'
    )  # type: float
    AircraftTrackAngle = FloatDescriptor(
        'AircraftTrackAngle', _required,
        docstring='The bearing from the aircraft position at the first pulse '
                  'to the aircraft position at the last')  # type: float
    Look = StringEnumDescriptor(
        'Look', {'Left', 'Right', 'Nadir'}, _required,
        docstring='Direction of the sensor look angle relative to aircraft '
                  'motion')  # type: str
    SquintAngle = SerializableDescriptor(
        'SquintAngle', SquintAngleType, _required,
        docstring='Measured angle between the sensor line-of-sight and the '
                  'lateral axis of the aircraft')  # type: SquintAngleType
    AircraftLocation = SerializableDescriptor(
        'AircraftLocation', AircraftLocationType, _required,
        docstring='The aircraft location (at scene center COA time?)')  # type: AircraftLocationType
    AircraftVelocity = SerializableDescriptor(
        'AircraftVelocity', XYZType, _required,
        docstring='Aircraft velocity in ECEF coordinates (at scene center COA time?)')  # type: XYZType
    FlightNumber = StringDescriptor(
        'FlightNumber', _required,
        docstring='The aircraft flight number')  # type: Optional[str]
    PassNumber = StringDescriptor(
        'PassNumber', _required,
        docstring='The aircraft pass number')  # type: Optional[str]

    def __init__(self, Name=None, SensorMfg=None, OperatingAgency=None,
                 Type=None, Mode=None, Band=None, Bandwidth=None,
                 CenterFrequency=None, NearRange=None, SlantRangeSwathWidth=None,
                 Polarization=None, Range=None, DepressionAngle=None,
                 LinearDynamicRange=None, BeamWidth=None, Aimpoint=None,
                 AircraftHeading=None, AircraftTrackAngle=None,
                 Look=None, SquintAngle=None,
                 AircraftLocation=None, AircraftVelocity=None,
                 FlightNumber=None, PassNumber=None, **kwargs):
        """
        Parameters
        ----------
        Name : None|str
        SensorMfg : None|str
        OperatingAgency : None|str
        Type : str
        Mode : None|str
        Band : None|str
        Bandwidth : None|float
        CenterFrequency : None|float
        NearRange : None|float
        SlantRangeSwathWidth : None|float
        Polarization : None|str
        Range : float
        DepressionAngle : float
        LinearDynamicRange : None|float
        BeamWidth : BeamWidthType
        Aimpoint : LatLonEleType|numpy.ndarray|list|tuple
        AircraftHeading : None|float
        AircraftTrackAngle : None|float
        Look : str
        SquintAngle : SquintAngleType
        AircraftLocation : AircraftLocationType|numpy.ndarray|list|tuple
        AircraftVelocity : XYZType|numpy.ndarray|list|tuple
        FlightNumber : None|int
        PassNumber : None|int

        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.SensorMfg = SensorMfg
        self.OperatingAgency = OperatingAgency
        self.Type = Type
        self.Mode = Mode
        self.Band = Band
        self.Bandwidth = Bandwidth
        self.CenterFrequency = CenterFrequency
        self.NearRange = NearRange
        self.SlantRangeSwathWidth = SlantRangeSwathWidth
        self.Polarization = Polarization
        self.Range = Range
        self.DepressionAngle = DepressionAngle
        self.LinearDynamicRange = LinearDynamicRange
        self.BeamWidth = BeamWidth
        self.Aimpoint = Aimpoint
        self.AircraftHeading = AircraftHeading
        self.AircraftTrackAngle = AircraftTrackAngle
        self.Look = Look
        self.SquintAngle = SquintAngle
        self.AircraftLocation = AircraftLocation
        self.AircraftVelocity = AircraftVelocity
        self.FlightNumber = FlightNumber
        self.PassNumber = PassNumber
        super(SensorInfoType, self).__init__(**kwargs)

    @classmethod
    def from_sicd(cls, sicd):
        """
        Construct the sensor info from a sicd structure

        Parameters
        ----------
        sicd : SICDType

        Returns
        -------
        SensorInfoType
        """

        transmit_freq_proc = sicd.ImageFormation.TxFrequencyProc
        center_freq = transmit_freq_proc.center_frequency*1e-9
        bandwidth = transmit_freq_proc.bandwidth*1e-9
        polarization = sicd.ImageFormation.get_polarization().replace(':', '')
        look = 'Left' if sicd.SCPCOA.SideOfTrack == 'L' else 'Right'
        arp_pos_llh = ecf_to_geodetic(sicd.SCPCOA.ARPPos.get_array())

        # calculate heading
        heading_ned = ecf_to_ned(sicd.SCPCOA.ARPVel.get_array(), sicd.SCPCOA.ARPPos.get_array(), absolute_coords=False)
        heading = numpy.rad2deg(numpy.arctan2(heading_ned[1], heading_ned[0]))
        # calculate track angle
        first_pos_ecf = sicd.Position.ARPPoly(0)
        last_pos_ecf = sicd.Position.ARPPoly(sicd.Timeline.CollectDuration)
        diff_ned = ecf_to_ned(last_pos_ecf - first_pos_ecf, sicd.SCPCOA.ARPPos.get_array(), absolute_coords=False)
        track_angle = numpy.rad2deg(numpy.arctan2(diff_ned[1], diff_ned[0]))

        return SensorInfoType(
            Name=sicd.CollectionInfo.CollectorName,
            Type='SAR',
            Mode=sicd.CollectionInfo.RadarMode.ModeType,
            Band=sicd.ImageFormation.get_transmit_band_name(),
            Bandwidth=bandwidth,
            CenterFrequency=center_freq,
            Polarization=polarization,
            Range=sicd.SCPCOA.SlantRange,
            DepressionAngle=sicd.SCPCOA.GrazeAng,
            Aimpoint=sicd.GeoData.SCP.LLH.get_array(),
            AircraftHeading=heading,
            AircraftTrackAngle=track_angle,
            Look=look,
            SquintAngle=SquintAngleType(
                SlantPlane=sicd.SCPCOA.DopplerConeAng,
                GroundPlane=sicd.SCPCOA.Squint),
            AircraftLocation=arp_pos_llh,
            AircraftVelocity=sicd.SCPCOA.ARPVel.get_array())
