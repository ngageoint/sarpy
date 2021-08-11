"""
The ProductCreationType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

import logging
from typing import Union
from datetime import datetime

import numpy

from sarpy.io.xml.base import Serializable, ParametersCollection
from sarpy.io.xml.descriptors import SerializableDescriptor, IntegerDescriptor, \
    StringDescriptor, StringEnumDescriptor, DateTimeDescriptor, ParametersDescriptor
from sarpy.io.complex.sicd_elements.SICD import SICDType

from .base import DEFAULT_STRICT


logger = logging.getLogger(__name__)


def extract_classification_from_sicd(sicd):
    """
    Extract a SIDD style classification from a SICD classification string.

    Parameters
    ----------
    sicd : SICDType

    Returns
    -------
    str
    """

    if not isinstance(sicd, SICDType):
        raise TypeError('Requires SICDType instance, got type {}'.format(type(sicd)))

    c_str = sicd.CollectionInfo.Classification.upper().split('//')[0].strip()

    clas = None
    if c_str.startswith('UNCLASS') or c_str == 'U':
        clas = 'U'
    elif c_str.startswith('CONF') or c_str == 'C':
        clas = 'C'
    elif c_str.startswith('TOP ') or c_str == 'TS':
        clas = 'TS'
    elif c_str.startswith('SEC') or c_str == 'S':
        clas = 'S'
    elif c_str == 'FOUO' or c_str.startswith('REST') or c_str == 'R':
        clas = 'R'
    else:
        logger.critical(
            'Unclear how to extract classification code for classification string {}.\n\t'
            'It will default to unclassified, and should be set appropriately.'.format(c_str))
    return clas


class ProcessorInformationType(Serializable):
    """
    Details regarding the processor.
    """
    _fields = ('Application', 'ProcessingDateTime', 'Site', 'Profile')
    _required = ('Application', 'ProcessingDateTime', 'Site')
    # descriptors
    Application = StringDescriptor(
        'Application', _required, strict=DEFAULT_STRICT,
        docstring='Name and version of the application used to create the image.')  # type: str
    ProcessingDateTime = DateTimeDescriptor(
        'ProcessingDateTime', _required, strict=DEFAULT_STRICT, numpy_datetime_units='us',
        docstring='Date and time the image creation application processed the image (UTC).')  # type: numpy.datetime64
    Site = StringDescriptor(
        'Site', _required, strict=DEFAULT_STRICT,
        docstring='The creation site of this SICD product.')  # type: str
    Profile = StringDescriptor(
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
        'DESVersion', 'resourceElement', 'createDate', 'compliesWith', 'ISMCATCESVersion',
        'classification', 'ownerProducer', 'SCIcontrols', 'SARIdentifier',
        'disseminationControls', 'FGIsourceOpen', 'FGIsourceProtected', 'releasableTo',
        'nonICmarkings', 'classifiedBy', 'compilationReason', 'derivativelyClassifiedBy',
        'classificationReason', 'nonUSControls', 'derivedFrom', 'declassDate',
        'declassEvent', 'declassException', 'typeOfExemptedSource', 'dateOfExemptedSource',
        'SecurityExtensions')
    _required = (
        'DESVersion', 'createDate', 'classification', 'ownerProducer', 'compliesWith',
        'ISMCATCESVersion')
    _collections_tags = {'SecurityExtensions': {'array': False, 'child_tag': 'SecurityExtension'}}
    _set_as_attribute = (
        'DESVersion', 'resourceElement', 'createDate', 'compliesWith', 'ISMCATCESVersion',
        'classification', 'ownerProducer', 'SCIcontrols', 'SARIdentifier',
        'disseminationControls', 'FGIsourceOpen', 'FGIsourceProtected', 'releasableTo',
        'nonICmarkings', 'classifiedBy', 'compilationReason', 'derivativelyClassifiedBy',
        'classificationReason', 'nonUSControls', 'derivedFrom', 'declassDate',
        'declassEvent', 'declassException', 'typeOfExemptedSource', 'dateOfExemptedSource')
    _child_xml_ns_key = {the_field: 'ism' for the_field in _fields if the_field != 'SecurityExtensions'}
    # Descriptor
    DESVersion = IntegerDescriptor(
        'DESVersion', _required, strict=DEFAULT_STRICT, default_value=13,
        docstring='The version number of the DES. Should there be multiple specified in an instance document '
                  'the one at the root node is the one that will apply to the entire document.')  # type: int
    createDate = StringDescriptor(
        'createDate', _required, strict=DEFAULT_STRICT,
        docstring='This should be a date of format :code:`YYYY-MM-DD`, but this is not checked.')  # type: str
    compliesWith = StringEnumDescriptor(
        'compliesWith', ('USGov', 'USIC', 'USDOD', 'OtherAuthority'), _required,
        strict=DEFAULT_STRICT, default_value='USGov',
        docstring='The ISM rule sets with which the document may complies.')  # type: Union[None, str]
    ISMCATCESVersion = StringDescriptor(
        'ISMCATCESVersion', _required, strict=DEFAULT_STRICT, default_value='201903',
        docstring='')  # type: Union[None, str]
    classification = StringEnumDescriptor(
        'classification', ('U', 'C', 'R', 'S', 'TS'), _required, strict=DEFAULT_STRICT,
        docstring='')  # type: str
    ownerProducer = StringDescriptor(
        'ownerProducer', _required, strict=DEFAULT_STRICT,  # default_value='USA',
        docstring='')  # type: str
    SCIcontrols = StringDescriptor(
        'SCIcontrols', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    SARIdentifier = StringDescriptor(
        'SARIdentifier', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    disseminationControls = StringDescriptor(
        'disseminationControls', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    FGIsourceOpen = StringDescriptor(
        'FGIsourceOpen', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    FGIsourceProtected = StringDescriptor(
        'FGIsourceProtected', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    releasableTo = StringDescriptor(
        'releasableTo', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    nonICmarkings = StringDescriptor(
        'nonICmarkings', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    classifiedBy = StringDescriptor(
        'classifiedBy', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    compilationReason = StringDescriptor(
        'compilationReason', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    derivativelyClassifiedBy = StringDescriptor(
        'derivativelyClassifiedBy', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    classificationReason = StringDescriptor(
        'classificationReason', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    nonUSControls = StringDescriptor(
        'nonUSControls', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    derivedFrom = StringDescriptor(
        'derivedFrom', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    declassDate = StringDescriptor(
        'declassDate', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    declassEvent = StringDescriptor(
        'declassEvent', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    declassException = StringDescriptor(
        'declassException', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    typeOfExemptedSource = StringDescriptor(
        'typeOfExemptedSource', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    dateOfExemptedSource = StringDescriptor(
        'dateOfExemptedSource', _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, str]
    SecurityExtensions = ParametersDescriptor(
        'SecurityExtensions', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Extensible parameters used to support profile-specific needs related to '
                  'product security.')  # type: ParametersCollection

    def __init__(self, DESVersion=13, createDate=None, compliesWith='USGov', ISMCATCESVersion='201903',
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
        ISMCATCESVersion : None|str
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
        self.ISMCATCESVersion = ISMCATCESVersion
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

    @classmethod
    def from_sicd(cls, sicd, create_date=None):
        """
        Extract best guess from SICD.

        Parameters
        ----------
        sicd : SICDType
        create_date : str

        Returns
        -------
        ProductClassificationType
        """

        clas = extract_classification_from_sicd(sicd)

        if create_date is None:
            create_date = datetime.now().strftime('%Y-%m-%d')

        return cls(classification=clas, createDate=create_date)


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
    ProcessorInformation = SerializableDescriptor(
        'ProcessorInformation', ProcessorInformationType, _required, strict=DEFAULT_STRICT,
        docstring='Details regarding processor.')  # type: ProcessorInformationType
    Classification = SerializableDescriptor(
        'Classification', ProductClassificationType, _required, strict=DEFAULT_STRICT,
        docstring='The overall classification of the product.')  # type: ProductClassificationType
    ProductName = StringDescriptor(
        'ProductName', _required, strict=DEFAULT_STRICT,
        docstring='The output product name defined by the processor.')  # type: str
    ProductClass = StringDescriptor(
        'ProductClass', _required, strict=DEFAULT_STRICT,
        docstring='Class of product. Examples - :code:`Dynamic Image, Amplitude Change Detection, '
                  'Coherent Change Detection`')  # type: str
    ProductType = StringDescriptor(
        'ProductType', _required, strict=DEFAULT_STRICT,
        docstring='Type of sub-product. Examples - :code:`Frame #, Reference, Match`. '
                  'This field is only needed if there is a suite of associated '
                  'products.')  # type: Union[None, str]
    ProductCreationExtensions = ParametersDescriptor(
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

    @classmethod
    def from_sicd(cls, sicd, product_class):
        """
        Generate from a SICD for the given product class.

        Parameters
        ----------
        sicd : SICDType
        product_class : str

        Returns
        -------
        ProductCreationType
        """

        if not isinstance(sicd, SICDType):
            raise TypeError('Requires SICDType instance, got type {}'.format(type(sicd)))

        from sarpy.__about__ import __title__, __version__

        proc_info = ProcessorInformationType(
            Application='{} {}'.format(__title__, __version__),
            ProcessingDateTime=numpy.datetime64(datetime.now()),
            Site='Unknown')
        classification = ProductClassificationType.from_sicd(sicd)
        return cls(ProcessorInformation=proc_info,
                   Classification=classification,
                   ProductName=product_class,
                   ProductClass=product_class)
