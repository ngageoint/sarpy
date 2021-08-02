"""
Definition for the DetailCollectionInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")


# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import DEFAULT_STRICT, \
    _DateTimeDescriptor, _FloatDescriptor, _IntegerDescriptor, \
    _SerializableDescriptor, _StringDescriptor, Serializable


class DateType(Serializable):
    _fields = ('Begin', 'End')
    _required = ()
    # descriptors
    Begin = _DateTimeDescriptor(
        'Begin', _required, strict=DEFAULT_STRICT, numpy_datetime_units='D',
        docstring="Begin date of the data collection.")
    End = _DateTimeDescriptor(
        'End', _required, strict=DEFAULT_STRICT, numpy_datetime_units='D',
        docstring="End date of the data collection.")

    def __init__(self, Begin='', End='', **kwargs):
        """
        Parameters
        ----------
        Begin : None|str
        End : None|str
        kwargs
            Other keyword arguments
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Begin = Begin
        self.End = End
        super(DateType, self).__init__(**kwargs)


class LocationType(Serializable):
    _fields = ('Lat', 'Lon', 'Name')
    _required = ()
    # descriptors
    Lat = _FloatDescriptor(
        'Lat', _required, strict=DEFAULT_STRICT,
        docstring="General latitude of the data collection.")
    Lon = _FloatDescriptor(
        'Lon', _required, strict=DEFAULT_STRICT,
        docstring="General longitude of the data collection.")
    Name = _StringDescriptor(
        'Name', _required,
        docstring="Common name of the collection location.")

    def __init__(self, Lat=None, Lon=None, Name=None, **kwargs):
        """
        Parameters
        ----------
        Lat : None|float
        Lon : None|float
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
        docstring="Name of the collection.")
    ProgramName = _StringDescriptor(
        'StringDescriptor', _required,
        docstring="Name of the program that collected the data.")
    Sponsor = _StringDescriptor(
        'Sponsor', _required,
        docstring="Sponsoring agency/organization of the data collection.")
    Date = _SerializableDescriptor(
        'Date', DateType, _required, strict=DEFAULT_STRICT,
        docstring="Begin and end dates of the data collection.")
    Location = _SerializableDescriptor(
        'Location', LocationType, _required, strict=DEFAULT_STRICT,
        docstring="General location of the data collection.")
    NumberOfSites = _IntegerDescriptor(
        'NumberOfSites', _required, strict=DEFAULT_STRICT,
        docstring="Number of different sites contained in the data collection.")

    def __init__(self, Name=None, ProgramName=None, Sponsor=None, Date=None,
                 Location=None, NumberOfSites=None, **kwargs):
        """
        Parameters
        ----------
        Name : None|str
        ProgramName : None|str
        Sponsor : None|str
        Date : None|DateType
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
