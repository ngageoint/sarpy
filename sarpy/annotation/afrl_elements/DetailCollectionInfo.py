"""
Definition for the DetailCollectionInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")


from typing import Optional
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import DEFAULT_STRICT, \
    _IntegerDescriptor, _SerializableDescriptor, _StringDescriptor, \
    Serializable, _FloatDescriptor
from .blocks import DateRangeType


class LocationType(Serializable):
    _fields = ('Lat', 'Lon', 'Name')
    _required = ('Lat', 'Lon')
    # descriptors
    Lat = _FloatDescriptor(
        'Lat', _required, strict=DEFAULT_STRICT,
        docstring="General latitude of the data collection.")  # type: float
    Lon = _FloatDescriptor(
        'Lon', _required, strict=DEFAULT_STRICT,
        docstring="General longitude of the data collection.")  # type: float
    Name = _StringDescriptor(
        'Name', _required,
        docstring="Common name of the collection location.")  # type: Optional[str]

    def __init__(self, Lat=None, Lon=None, Name=None, **kwargs):
        """
        Parameters
        ----------
        Lat : float
        Lon : float
        Name : None|str
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Lat = Lat
        self.Lon = Lon
        self.Name = Name
        super(LocationType, self).__init__(**kwargs)


class DetailCollectionInfoType(Serializable):
    _fields = ('Name', 'ProgramName', 'Sponsor', 'Date', 'Location',
               'NumberOfSites')
    _required = ()
    # descriptors
    Name = _StringDescriptor(
        'Name', _required,
        docstring="Name of the collection.")  # type: Optional[str]
    ProgramName = _StringDescriptor(
        'StringDescriptor', _required,
        docstring="Name of the program that collected the data.")  # type: Optional[str]
    Sponsor = _StringDescriptor(
        'Sponsor', _required,
        docstring="Sponsoring agency/organization of the data collection.")  # type: Optional[str]
    Date = _SerializableDescriptor(
        'Date', DateRangeType, _required, strict=DEFAULT_STRICT,
        docstring="Begin and end dates of the data collection.")  # type: Optional[DateRangeType]
    Location = _SerializableDescriptor(
        'Location', LocationType, _required, strict=DEFAULT_STRICT,
        docstring="General location of the data collection.")  # type: Optional[LocationType]
    NumberOfSites = _IntegerDescriptor(
        'NumberOfSites', _required, strict=DEFAULT_STRICT,
        docstring="Number of different sites contained in the data collection.")  # type: Optional[int]

    def __init__(self, Name=None, ProgramName=None, Sponsor=None, Date=None,
                 Location=None, NumberOfSites=None, **kwargs):
        """
        Parameters
        ----------
        Name : None|str
        ProgramName : None|str
        Sponsor : None|str
        Date : None|DateRangeType
        Location : None|LocationType
        NumberOfSites : None|int
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Name = Name
        self.ProgramName = ProgramName
        self.Sponsor = Sponsor
        self.Date = Date
        self.Location = Location
        self.NumberOfSites = NumberOfSites
        super(DetailCollectionInfoType, self).__init__(**kwargs)
