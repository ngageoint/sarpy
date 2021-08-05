"""
Definition for the DetailCollectionInfo AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = ("Thomas McCullough", "Thomas Rackers")


from typing import Optional
# noinspection PyProtectedMember
from sarpy.io.complex.sicd_elements.base import DEFAULT_STRICT, \
    _IntegerDescriptor, _SerializableDescriptor, _StringDescriptor, \
    Serializable
from .blocks import DateRangeType, LatLonWithNameType


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
        'Location', LatLonWithNameType, _required, strict=DEFAULT_STRICT,
        docstring="General location of the data collection.")  # type: Optional[LatLonWithNameType]
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
        Location : None|LatLonWithNameType
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
