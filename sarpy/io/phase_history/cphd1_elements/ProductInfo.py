"""
The ProductInfo elements.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List

import numpy

from sarpy.io.xml.base import Serializable, ParametersCollection
from sarpy.io.xml.descriptors import StringDescriptor, DateTimeDescriptor, \
    ParametersDescriptor, SerializableListDescriptor
from .base import DEFAULT_STRICT


class CreationInfoType(Serializable):
    """
    Parameters that provide general information about the CPHD product generation.
    """

    _fields = ('Application', 'DateTime', 'Site', 'Parameters')
    _required = ('DateTime', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    Application = StringDescriptor(
        'Application', _required, strict=DEFAULT_STRICT,
        docstring='Name and version of the application used to create the CPHD.')  # type: str
    DateTime = DateTimeDescriptor(
        'DateTime', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Date and time the image creation application processed the image (UTC).')  # type: numpy.datetime64
    Site = StringDescriptor(
        'Site', _required, strict=DEFAULT_STRICT,
        docstring='The creation site of this CPHD product.')  # type: str
    Parameters = ParametersDescriptor(
        'Parameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Additional parameters.')  # type: Union[None, ParametersCollection]

    def __init__(self, Application=None, DateTime=None, Site=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        Application : str
        DateTime : numpy.datetime64|datetime|date|str
        Site : str
        Profile : str
        Parameters : None|ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Application = Application
        self.DateTime = DateTime
        self.Site = Site
        self.Parameters = Parameters
        super(CreationInfoType, self).__init__(**kwargs)


class ProductInfoType(Serializable):
    """
    Parameters that provide general information about the CPHD product and/or the
    derived products that may be created from it.
    """

    _fields = ('Profile', 'CreationInfos', 'Parameters')
    _required = ()
    _collections_tags = {
        'CreationInfos': {'array': False, 'child_tag': 'CreationInfo'},
        'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    Profile = StringDescriptor(
        'Profile', _required, strict=DEFAULT_STRICT,
        docstring='Identifies what profile was used to create this CPHD product.')  # type: str
    CreationInfos = SerializableListDescriptor(
        'CreationInfos', CreationInfoType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Parameters that provide general information about the CPHD '
                  'product generation.')  # type: Union[None, List[CreationInfoType]]
    Parameters = ParametersDescriptor(
        'Parameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Additional parameters.')  # type: Union[None, ParametersCollection]

    def __init__(self, Profile=None, CreationInfos=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        Profile : str
        CreationInfos : None|List[CreationInfoType]
        Parameters : None|ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Profile = Profile
        self.CreationInfos = CreationInfos
        self.Parameters = Parameters
        super(ProductInfoType, self).__init__(**kwargs)
