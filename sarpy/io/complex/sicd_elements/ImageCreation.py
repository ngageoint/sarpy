"""
The ImageCreation elements.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


import numpy

from sarpy.io.xml.base import Serializable
from sarpy.io.xml.descriptors import StringDescriptor, DateTimeDescriptor

from .base import DEFAULT_STRICT


class ImageCreationType(Serializable):
    """
    General information about the image creation.
    """

    _fields = ('Application', 'DateTime', 'Site', 'Profile')
    _required = ()
    # descriptors
    Application = StringDescriptor(
        'Application', _required, strict=DEFAULT_STRICT,
        docstring='Name and version of the application used to create the image.')  # type: str
    DateTime = DateTimeDescriptor(
        'DateTime', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Date and time the image creation application processed the image (UTC).')  # type: numpy.datetime64
    Site = StringDescriptor(
        'Site', _required, strict=DEFAULT_STRICT,
        docstring='The creation site of this SICD product.')  # type: str
    Profile = StringDescriptor(
        'Profile', _required, strict=DEFAULT_STRICT,
        docstring='Identifies what profile was used to create this SICD product.')  # type: str

    def __init__(self, Application=None, DateTime=None, Site=None, Profile=None, **kwargs):
        """

        Parameters
        ----------
        Application : str
        DateTime : numpy.datetime64|datetime|date|str
        Site : str
        Profile : str
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Application = Application
        self.DateTime = DateTime
        self.Site = Site
        self.Profile = Profile
        super(ImageCreationType, self).__init__(**kwargs)
