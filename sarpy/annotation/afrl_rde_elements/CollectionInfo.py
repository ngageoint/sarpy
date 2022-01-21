"""
Definition for the CollectionInfo NGA modified RDE/AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = "Thomas McCullough"


from typing import Optional

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import IntegerDescriptor, SerializableDescriptor, \
    StringDescriptor, FloatDescriptor

from .base import DEFAULT_STRICT
from .blocks import DateTimeRangeType


class LocationType(Serializable):
    _fields = ('Lat', 'Lon', 'Name')
    _required = ('Lat', 'Lon')
    # descriptors
    Lat = FloatDescriptor(
        'Lat', _required, strict=DEFAULT_STRICT,
        docstring="General latitude of the data collection.")  # type: float
    Lon = FloatDescriptor(
        'Lon', _required, strict=DEFAULT_STRICT,
        docstring="General longitude of the data collection.")  # type: float
    Name = StringDescriptor(
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


class CollectionInfoType(Serializable):
    _fields = (
        'Name', 'ProgramName', 'Sponsor', 'Date', 'Location', 'NumberOfSites')
    _required = ('Date', )
    # descriptors
    Name = StringDescriptor(
        'Name', _required,
        docstring="Name of the collection.")  # type: Optional[str]
    ProgramName = StringDescriptor(
        'StringDescriptor', _required,
        docstring="Name of the program that collected the data.")  # type: Optional[str]
    Sponsor = StringDescriptor(
        'Sponsor', _required,
        docstring="Sponsoring agency/organization of the data collection.")  # type: Optional[str]
    Date = SerializableDescriptor(
        'Date', DateTimeRangeType, _required, strict=DEFAULT_STRICT,
        docstring="Begin and end dates of the data collection.")  # type: Optional[DateTimeRangeType]
    Location = SerializableDescriptor(
        'Location', LocationType, _required, strict=DEFAULT_STRICT,
        docstring="General location of the data collection.")  # type: Optional[LocationType]
    NumberOfSites = IntegerDescriptor(
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
        Date : None|DateTimeRangeType|list|tuple
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
        super(CollectionInfoType, self).__init__(**kwargs)
