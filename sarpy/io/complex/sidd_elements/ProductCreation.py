# -*- coding: utf-8 -*-
"""
The ProductCreationType definition.
"""

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from ..sicd_elements.base import Serializable, _SerializableDescriptor, _StringDescriptor, \
    _DateTimeDescriptor, _ParametersDescriptor, ParametersCollection

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


class ProcessorInformationType(Serializable):
    """
    Details regarding the processor.
    """
    _fields = ('Application', 'ProcessingDateTime', 'Site', 'Profile')
    _required = ()
    # descriptors
    Application = _StringDescriptor(
        'Application', _required, strict=DEFAULT_STRICT,
        docstring='Name and version of the application used to create the image.')  # type: str
    ProcessingDateTime = _DateTimeDescriptor(
        'ProcessingDateTime', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Date and time the image creation application processed the image (UTC).')  # type: numpy.datetime64
    Site = _StringDescriptor(
        'Site', _required, strict=DEFAULT_STRICT,
        docstring='The creation site of this SICD product.')  # type: str
    Profile = _StringDescriptor(
        'Profile', _required, strict=DEFAULT_STRICT,
        docstring='Identifies what profile was used to create this SICD product.')  # type: str

    def __init__(self, Application=None, ProcessingDateTime=None, Site=None, Profile=None, **kwargs):
        """

        Parameters
        ----------
        Application : str
        ProcessingDateTime : numpy.datetime64|datetime|date|str
        Site : str
        Profile : str
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.Application = Application
        self.ProcessingDateTime = ProcessingDateTime
        self.Site = Site
        self.Profile = Profile
        super(ProcessorInformationType, self).__init__(**kwargs)


class ProductClassificationType(Serializable):
    """
    The overall classification of the product.
    """
    _fields = ('SecurityExtensions', )
    _required = ()
    _collections_tags = {'SecurityExtensions': {'array': False, 'child_tag': 'SecurityExtension'}}
    # Descriptor
    SecurityExtensions = _ParametersDescriptor(
        'SecurityExtensions', _collections_tags, required=_required, strict=DEFAULT_STRICT,
        docstring='Extensible parameters used to support profile-specific needs related to '
                  'product security.')  # type: ParametersCollection
    # TODO: what are these attribute groups?

    def __init__(self, SecurityExtensions=None, **kwargs):
        """

        Parameters
        ----------
        SecurityExtensions : ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.SecurityExtensions = SecurityExtensions
        super(ProductClassificationType, self).__init__(**kwargs)


class ProductCreationType(Serializable):
    """
    Contains general information about product creation.
    """

    _fields =(
        'ProcessorInformation', 'ProductClassification', 'ProductName', 'ProductClass',
        'ProductType', 'ProductCreationExtensions')
    _required = (
        'ProcessorInformation', 'ProductClassification', 'ProductName', 'ProductClass')
    _collections_tags = {'ProductCreationExtensions': {'array': False, 'child_tag': 'ProductCreationExtension'}}
    # Descriptors
    ProcessorInformation = _SerializableDescriptor(
        'ProcessorInformation', ProcessorInformationType, _required, strict=DEFAULT_STRICT,
        docstring='Details regarding processor.')  # type: ProcessorInformationType
    ProductClassification = _SerializableDescriptor(
        'ProductClassification', ProductClassificationType, _required, strict=DEFAULT_STRICT,
        docstring='The overall classification of the product.')  # type: ProductClassificationType
    ProductName = _StringDescriptor(
        'ProductName', _required, strict=DEFAULT_STRICT,
        docstring='The output product name defined by the processor.')  # type: str
    ProductClass = _StringDescriptor(
        'ProductClass', _required, strict=DEFAULT_STRICT,
        docstring='Class of product. Examples - :code:`Dynamic Image, Amplitude Change Detection, '
                  'Coherent Change Detection`')  # type: str
    ProductType = _StringDescriptor(
        'ProductType', _required, strict=DEFAULT_STRICT,
        docstring='Type of sub-product. Examples - :code:`Frame #, Reference, Match`. '
                  'This field is only needed if there is a suite of associated products.')  # type: str
    ProductCreationExtensions = _ParametersDescriptor(
        'ProductCreationExtensions', _collections_tags, required=_required, strict=DEFAULT_STRICT,
        docstring='Extensible parameters used to support profile-specific needs related to '
                  'product creation.')  # type: ParametersCollection

    def __init__(self, ProcessorInformation, ProductClassification, ProductName, ProductClass,
                 ProductType, ProductCreationExtensions, **kwargs):
        """

        Parameters
        ----------
        ProcessorInformation : ProcessorInformationType
        ProductClassification : ProductClassificationType
        ProductName : str
        ProductClass : str
        ProductType : str
        ProductCreationExtensions : ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        self.ProcessorInformation = ProcessorInformation
        self.ProductClassification = ProductClassification
        self.ProductName = ProductName
        self.ProductClass = ProductClass
        self.ProductType = ProductType
        self.ProductCreationExtensions = ProductCreationExtensions
        super(ProductCreationType, self).__init__(**kwargs)
