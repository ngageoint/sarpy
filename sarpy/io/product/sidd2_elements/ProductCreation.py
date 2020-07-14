# -*- coding: utf-8 -*-
"""
The ProductCreationType definition.
"""

from typing import Union

import numpy

from .base import DEFAULT_STRICT
# noinspection PyProtectedMember
from ...complex.sicd_elements.base import Serializable, _SerializableDescriptor, \
    _StringDescriptor, _StringEnumDescriptor, _IntegerDescriptor, _DateTimeDescriptor, \
    _ParametersDescriptor, ParametersCollection

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
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Application = Application
        self.ProcessingDateTime = ProcessingDateTime
        self.Site = Site
        self.Profile = Profile
        super(ProcessorInformationType, self).__init__(**kwargs)


class ProductClassificationType(Serializable):
    """
    The overall classification of the product.
    """
    _fields = (
        'DESVersion', 'resourceElement', 'createDate', 'compliesWith',
        'classification', 'ownerProducer', 'SCIcontrols', 'SARIdentifier',
        'disseminationControls', 'FGIsourceOpen', 'FGIsourceProtected', 'releasableTo',
        'nonICmarkings', 'classifiedBy', 'compilationReason', 'derivativelyClassifiedBy',
        'classificationReason', 'nonUSControls', 'derivedFrom', 'declassDate',
        'declassEvent', 'declassException', 'typeOfExemptedSource', 'dateOfExemptedSource',
        'SecurityExtensions')
    _required = ('DESVersion', 'createDate', 'classification', 'ownerProducer')
    _collections_tags = {'SecurityExtensions': {'array': False, 'child_tag': 'SecurityExtension'}}
    _set_as_attribute = (
        'DESVersion', 'resourceElement', 'createDate', 'compliesWith',
        'classification', 'ownerProducer', 'SCIcontrols', 'SARIdentifier',
        'disseminationControls', 'FGIsourceOpen', 'FGIsourceProtected', 'releasableTo',
        'nonICmarkings', 'classifiedBy', 'compilationReason', 'derivativelyClassifiedBy',
        'classificationReason', 'nonUSControls', 'derivedFrom', 'declassDate',
        'declassEvent', 'declassException', 'typeOfExemptedSource', 'dateOfExemptedSource')
    _child_xml_ns_key = {key: 'ism' for key in _fields if key != 'SecurityExtensions'}
    # Descriptor
    DESVersion = _IntegerDescriptor(
        'DESVersion', _required, strict=DEFAULT_STRICT, default_value=13,
        docstring='The version number of the DES. Should there be multiple specified in an instance document '
                  'the one at the root node is the one that will apply to the entire document.')  # type: int
    createDate = _StringDescriptor(
        'createDate', _required, strict=DEFAULT_STRICT,
        docstring='This should be a date of format :code:`YYYY-MM-DD`, but this is not checked.')  # type: str
    compliesWith = _StringDescriptor(
        'compliesWith', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    classification = _StringEnumDescriptor(
        'classification', ('U', 'C', 'R', 'S', 'TS'), _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    ownerProducer = _StringDescriptor(
        'ownerProducer', _required, strict=DEFAULT_STRICT,  # default_value='USA',
        docstring='')  # type: str
    SCIcontrols = _StringDescriptor(
        'SCIcontrols', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    SARIdentifier = _StringDescriptor(
        'SARIdentifier', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    disseminationControls = _StringDescriptor(
        'disseminationControls', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    FGIsourceOpen = _StringDescriptor(
        'FGIsourceOpen', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    FGIsourceProtected = _StringDescriptor(
        'FGIsourceProtected', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    releasableTo = _StringDescriptor(
        'releasableTo', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    nonICmarkings = _StringDescriptor(
        'nonICmarkings', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    classifiedBy = _StringDescriptor(
        'classifiedBy', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    compilationReason = _StringDescriptor(
        'compilationReason', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    derivativelyClassifiedBy = _StringDescriptor(
        'derivativelyClassifiedBy', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    classificationReason = _StringDescriptor(
        'classificationReason', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    nonUSControls = _StringDescriptor(
        'nonUSControls', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    derivedFrom = _StringDescriptor(
        'derivedFrom', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    declassDate = _StringDescriptor(
        'declassDate', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    declassEvent = _StringDescriptor(
        'declassEvent', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    declassException = _StringDescriptor(
        'declassException', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    typeOfExemptedSource = _StringDescriptor(
        'typeOfExemptedSource', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    dateOfExemptedSource = _StringDescriptor(
        'dateOfExemptedSource', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    SecurityExtensions = _ParametersDescriptor(
        'SecurityExtensions', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Extensible parameters used to support profile-specific needs related to '
                  'product security.')  # type: ParametersCollection

    def __init__(self, DESVersion=13, createDate=None, compliesWith=None,
                 classification='U', ownerProducer='USA', SCIcontrols=None, SARIdentifier=None,
                 disseminationControls=None, FGIsourceOpen=None, FGIsourceProtected=None, releasableTo=None,
                 nonICmarkings=None, classifiedBy=None, compilationReason=None, derivativelyClassifiedBy=None,
                 classificationReason=None, nonUSControls=None, derivedFrom=None, declassDate=None,
                 declassEvent=None, declassException=None, typeOfExemptedSource=None, dateOfExemptedSource=None,
                 SecurityExtensions=None, **kwargs):
        """

        Parameters
        ----------
        DESVersion : int
        createDate : str
        compliesWith : None|str
        classification : str
        ownerProducer : str
        SCIcontrols : None|str
        SARIdentifier : None|str
        disseminationControls : None|str
        FGIsourceOpen : None|str
        FGIsourceProtected : None|str
        releasableTo : None|str
        nonICmarkings : None|str
        classifiedBy : None|str
        compilationReason : None|str
        derivativelyClassifiedBy : None|str
        classificationReason : None|str
        nonUSControls : None|str
        derivedFrom : None|str
        declassDate : None|str
        declassEvent : None|str
        declassException : None|str
        typeOfExemptedSource : None|str
        dateOfExemptedSource : None|str
        SecurityExtensions : None|ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.DESVersion = DESVersion
        self.createDate = createDate
        self.compliesWith = compliesWith
        self.classification = classification
        self.ownerProducer = ownerProducer
        self.SCIcontrols = SCIcontrols
        self.SARIdentifier = SARIdentifier
        self.disseminationControls = disseminationControls
        self.FGIsourceOpen = FGIsourceOpen
        self.FGIsourceProtected = FGIsourceProtected
        self.releasableTo = releasableTo
        self.nonICmarkings = nonICmarkings
        self.classifiedBy = classifiedBy
        self.compilationReason = compilationReason
        self.derivativelyClassifiedBy = derivativelyClassifiedBy
        self.classificationReason = classificationReason
        self.nonUSControls = nonUSControls
        self.derivedFrom = derivedFrom
        self.declassDate = declassDate
        self.declassEvent = declassEvent
        self.declassException = declassException
        self.typeOfExemptedSource = typeOfExemptedSource
        self.dateOfExemptedSource = dateOfExemptedSource
        self.SecurityExtensions = SecurityExtensions
        super(ProductClassificationType, self).__init__(**kwargs)

    @property
    def resourceElement(self):
        return 'true'


class ProductCreationType(Serializable):
    """
    Contains general information about product creation.
    """

    _fields = (
        'ProcessorInformation', 'Classification', 'ProductName', 'ProductClass',
        'ProductType', 'ProductCreationExtensions')
    _required = (
        'ProcessorInformation', 'Classification', 'ProductName', 'ProductClass')
    _collections_tags = {'ProductCreationExtensions': {'array': False, 'child_tag': 'ProductCreationExtension'}}
    # Descriptors
    ProcessorInformation = _SerializableDescriptor(
        'ProcessorInformation', ProcessorInformationType, _required, strict=DEFAULT_STRICT,
        docstring='Details regarding processor.')  # type: ProcessorInformationType
    Classification = _SerializableDescriptor(
        'Classification', ProductClassificationType, _required, strict=DEFAULT_STRICT,
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
                  'This field is only needed if there is a suite of associated '
                  'products.')  # type: Union[None, str]
    ProductCreationExtensions = _ParametersDescriptor(
        'ProductCreationExtensions', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Extensible parameters used to support profile-specific needs related to '
                  'product creation.')  # type: ParametersCollection

    def __init__(self, ProcessorInformation=None, Classification=None, ProductName=None,
                 ProductClass=None, ProductType=None, ProductCreationExtensions=None, **kwargs):
        """

        Parameters
        ----------
        ProcessorInformation : ProcessorInformationType
        Classification : ProductClassificationType
        ProductName : str
        ProductClass : str
        ProductType : str
        ProductCreationExtensions : ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.ProcessorInformation = ProcessorInformation
        self.Classification = Classification
        self.ProductName = ProductName
        self.ProductClass = ProductClass
        self.ProductType = ProductType
        self.ProductCreationExtensions = ProductCreationExtensions
        super(ProductCreationType, self).__init__(**kwargs)
