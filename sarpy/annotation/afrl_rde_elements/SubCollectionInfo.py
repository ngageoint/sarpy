"""
Definition for the SubCollectionInfo NGA modified RDE/AFRL labeling object
"""

__classification__ = "UNCLASSIFIED"
__authors__ = "Thomas McCullough"


from typing import Optional

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import SerializableDescriptor, StringDescriptor

from .base import DEFAULT_STRICT
from .blocks import DateTimeRangeType, LatLonEleType


class SubCollectionInfoType(Serializable):
    _fields = ('Name', 'SiteName', 'SiteNumber', 'SceneNumber', 'Description',
               'Duration', 'SiteCenterLocation', 'SceneContentDescription',
               'SiteBackgroundType')
    _required = ('Name', 'SiteCenterLocation', 'SceneContentDescription')
    # descriptors
    Name = StringDescriptor(
        'Name', _required,
        docstring="Name of the subcollection.")  # type: str
    SiteName = StringDescriptor(
        'SiteName', _required,
        docstring="Name of the subcollection site location.")  # type: Optional[str]
    SiteNumber = StringDescriptor(
        'SiteNumber', _required,
        docstring="Site number of the subcollection.")  # type: Optional[str]
    SceneNumber = StringDescriptor(
        'SceneNumber', _required,
        docstring="Scene number of the subcollection.")  # type: Optional[str]
    Description = StringDescriptor(
        'Description', _required,
        docstring="Description of the subcollection (e.g., Main array).")  # type: Optional[str]
    Duration = SerializableDescriptor(
        'Duration', DateTimeRangeType, _required, strict=DEFAULT_STRICT,
        docstring="Begin and end dates of the subcollection.")  # type: Optional[DateTimeRangeType]
    SiteCenterLocation = SerializableDescriptor(
        'SiteCenterLocation', LatLonEleType, _required, strict=DEFAULT_STRICT,
        docstring="Location of the center of the collection site.")  # type: LatLonEleType
    SceneContentDescription = StringDescriptor(
        'SceneContentDescription', _required, default_value="",
        docstring="Description of the general scene contents.")  # type: str
    SiteBackgroundType = StringDescriptor(
        'SiteBackgroundType', _required,
        docstring="Description of the background.")  # type: Optional[str]

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
        Description : NOne|str
        Duration : None|DateTimeRangeType|list|tuple
        SiteCenterLocation : LatLonEleType|numpy.ndarray|list|tuple
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
        self.SceneContentDescription = SceneContentDescription
        self.SiteBackgroundType = SiteBackgroundType
        super(SubCollectionInfoType, self).__init__(**kwargs)
