"""
Definition for the DetailSubCollectionInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")


# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import DEFAULT_STRICT, \
    _FloatDescriptor, _SerializableDescriptor, _StringDescriptor, Serializable
from sarpy.io.complex.sicd_elements.blocks import LatLonHAEType
from .blocks import DateRangeType


# TODO: to be reviewed

# TODO: Does a type like this already exist?
class LatLonEleType(Serializable):
    _fields = ('Lat', 'Lon', 'Ele')
    _required = _fields
    # descriptors
    Lat = _FloatDescriptor(
        'Lat', _required, strict=DEFAULT_STRICT, default_value=0.0,
        docstring="Latitude of the center of the collection site.")
    Lon = _FloatDescriptor(
        'Lon', _required, strict=DEFAULT_STRICT, default_value=0.0,
        docstring="Longitude of the center of the collection site.")
    Ele = _FloatDescriptor(
        'Ele', _required, strict=DEFAULT_STRICT, default_value=0.0,
        docstring="Elevation of the center of the collection site in HAE.")

    def __init__(self, Lat=None, Lon=None, Ele=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        Ele : float
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Lat = Lat
        self.Lon = Lon
        self.Ele = Ele
        super(LatLonEleType, self).__init__(**kwargs)


class DetailSubCollectionInfoType(Serializable):
    _fields = ('Name', 'SiteName', 'SiteNumber', 'SceneNumber', 'Description',
               'Duration', 'SiteCenterLocation', 'SceneContentDescription',
               'SiteBackgroundType')
    _required = ('Description', 'SiteCenterLocation', 'SceneContentDescription')
    # descriptors
    Name = _StringDescriptor(
        'Name', _required,
        docstring="Name of the subcollection.")
    SiteName = _StringDescriptor(
        'SiteName', _required,
        docstring="Name of the subcollection site location.")
    SiteNumber = _StringDescriptor(
        'SiteNumber', _required,
        docstring="Site number of the subcollection.")
    SceneNumber = _StringDescriptor(
        'SceneNumber', _required,
        docstring="Scene number of the subcollection.")
    Description = _StringDescriptor(
        'Description', _required, default_value="",
        docstring="Description of the subcollection (e.g., Main array).")
    Duration = _SerializableDescriptor(
        'Duration', DateRangeType, _required, strict=DEFAULT_STRICT,
        docstring="Begin and end dates of the subcollection.")
    SiteCenterLocation = _SerializableDescriptor(
        'SiteCenterLocation', LatLonHAEType, _required, strict=DEFAULT_STRICT,
        docstring="Location of the center of the collection site.")
    SceneContentDescription = _StringDescriptor(
        'SceneContentDescription', _required, default_value="",
        docstring="Description of the general scene contents.")
    SiteBackgroundType = _StringDescriptor(
        'SiteBackgroundType', _required,
        docstring="Description of the background.")

    def __init__(self, Name=None, SiteName=None, SiteNumber=None,
                 SceneNumber=None, Description=None, Duration=None,
                 SiteCenterLocation=None, SceneContentDescription=None,
                 SiteBackgroundType=None, **kwargs):
        """
        Parameters
        ----------
        Name : None|str
        SiteName : None|str
        SiteNumber : None|str
        SceneNumber : None|str
        Description : str
        Duration : None|DateRangeType
        SiteCenterLocation : LatLonHAEType
        SceneContentDescription : None|str
        SiteBackgroundType : None|str
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.SiteName = SiteName
        self.SiteNumber = SiteNumber
        self.SceneNumber = SceneNumber
        self.Description = Description
        self.Duration = Duration
        self.SiteCenterLocation = SiteCenterLocation
        self.SceneContentDesription = SceneContentDescription
        self.SiteBackgroundType = SiteBackgroundType
        super(DetailSubCollectionInfoType, self).__init__(**kwargs)
