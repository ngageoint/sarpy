"""
The ProductDisplayType definition for SIDD 1.0.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"

from typing import Union, List
from xml.etree import ElementTree
from collections import OrderedDict

import numpy

from sarpy.io.product.sidd2_elements.base import DEFAULT_STRICT
from sarpy.io.product.sidd2_elements.blocks import LatLonArrayElementType
from sarpy.io.xml.base import Serializable, SerializableArray, ParametersCollection, \
    find_children
from sarpy.io.xml.descriptors import SerializableDescriptor, SerializableArrayDescriptor, \
    SerializableListDescriptor, StringDescriptor, StringListDescriptor, ParametersDescriptor


class GeographicInformationType(Serializable):
    """
    Geographic information.
    """

    _fields = ('CountryCodes', 'SecurityInfo', 'GeographicInfoExtensions')
    _required = ()
    _collections_tags = {
        'CountryCodes': {'array': False, 'child_tag': 'CountryCode'},
        'GeographicInfoExtensions': {'array': False, 'child_tag': 'GeographicInfoExtension'}}
    # descriptors
    CountryCodes = StringListDescriptor(
        'CountryCodes', _required, strict=DEFAULT_STRICT,
        docstring="List of country codes for region covered by the image.")  # type: List[str]
    SecurityInfo = StringDescriptor(
        'SecurityInfo', _required, strict=DEFAULT_STRICT,
        docstring='Specifies classification level or special handling designators '
                  'for this geographic region.')  # type: Union[None, str]
    GeographicInfoExtensions = ParametersDescriptor(
        'GeographicInfoExtensions', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Implementation specific geographic information.')  # type: ParametersCollection

    def __init__(self, CountryCodes=None, SecurityInfo=None, GeographicInfoExtensions=None, **kwargs):
        """

        Parameters
        ----------
        CountryCodes : None|List[str]
        SecurityInfo : None|str
        GeographicInfoExtensions : None|ParametersCollection|dict
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CountryCodes = CountryCodes
        self.SecurityInfo = SecurityInfo
        self.GeographicInfoExtensions = GeographicInfoExtensions
        super(GeographicInformationType, self).__init__(**kwargs)


class TargetInformationType(Serializable):
    """
    Information about the target.
    """
    _fields = ('Identifiers', 'Footprint', 'TargetInformationExtensions')
    _required = ()
    _collections_tags = {
        'Identifiers': {'array': False, 'child_tag': 'Identifier'},
        'Footprint': {'array': True, 'child_tag': 'Vertex'},
        'TargetInformationExtensions': {'array': False, 'child_tag': 'TargetInformationExtension'}}
    # Descriptors
    Identifiers = ParametersDescriptor(
        'Identifiers', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Target may have one or more identifiers.  Examples: names, BE numbers, etc. Use '
                  'the "name" attribute to describe what this is.')  # type: ParametersCollection
    Footprint = SerializableArrayDescriptor(
        'Footprint', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Target footprint as defined by polygonal '
                  'shape.')  # type: Union[SerializableArray, List[LatLonArrayElementType]]
    TargetInformationExtensions = ParametersDescriptor(
        'TargetInformationExtensions', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Generic extension. Could be used to indicate type of target, '
                  'terrain, etc.')  # type: ParametersCollection

    def __init__(self, Identifiers=None, Footprint=None, TargetInformationExtensions=None, **kwargs):
        """

        Parameters
        ----------
        Identifiers : None|ParametersCollection|dict
        Footprint : None|List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        TargetInformationExtensions : None|ParametersCollection|dict
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.Identifiers = Identifiers
        self.Footprint = Footprint
        self.TargetInformationExtensions = TargetInformationExtensions
        super(TargetInformationType, self).__init__(**kwargs)


class GeographicCoverageType(Serializable):
    """
    The geographic coverage area for the product.
    """

    _fields = ('GeoregionIdentifiers', 'Footprint', 'GeographicInfo')
    _required = ('Footprint', )
    _collections_tags = {
        'GeoregionIdentifiers': {'array': False, 'child_tag': 'GeoregionIdentifier'},
        'Footprint': {'array': True, 'child_tag': 'Vertex'},
        'SubRegions': {'array': False, 'child_tag': 'SubRegion'}}
    # Descriptors
    GeoregionIdentifiers = ParametersDescriptor(
        'GeoregionIdentifiers', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Target may have one or more identifiers.  Examples: names, BE numbers, etc. Use '
                  'the "name" attribute to describe what this is.')  # type: ParametersCollection
    Footprint = SerializableArrayDescriptor(
        'Footprint', LatLonArrayElementType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Estimated ground footprint of the '
                  'product.')  # type: Union[None, SerializableArray, List[LatLonArrayElementType]]
    GeographicInfo = SerializableDescriptor(
        'GeographicInfo', GeographicInformationType, _required, strict=DEFAULT_STRICT,
        docstring='')  # type: Union[None, GeographicInformationType]

    def __init__(self, GeoregionIdentifiers=None, Footprint=None, SubRegions=None, GeographicInfo=None, **kwargs):
        """

        Parameters
        ----------
        GeoregionIdentifiers : None|ParametersCollection|dict
        Footprint : None|List[LatLonArrayElementType]|numpy.ndarray|list|tuple
        SubRegions : None|List[GeographicCoverageType]
        GeographicInfo : None|GeographicInformationType
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.GeoregionIdentifiers = GeoregionIdentifiers
        self.Footprint = Footprint
        self.GeographicInfo = GeographicInfo

        self._SubRegions = []
        if SubRegions is None:
            pass
        elif isinstance(SubRegions, GeographicCoverageType):
            self.addSubRegion(SubRegions)
        elif isinstance(SubRegions, (list, tuple)):
            for el in SubRegions:
                self.addSubRegion(el)
        else:
            raise ('SubRegions got unexpected type {}'.format(type(SubRegions)))
        super(GeographicCoverageType, self).__init__(**kwargs)

    @property
    def SubRegions(self):
        """
        List[GeographicCoverageType]: list of sub-regions.
        """

        return self._SubRegions

    def addSubRegion(self, value):
        """
        Add the given SubRegion to the SubRegions list.

        Parameters
        ----------
        value : GeographicCoverageType

        Returns
        -------
        None
        """

        if isinstance(value, ElementTree.Element):
            value = GeographicCoverageType.from_node(value, self._xml_ns, ns_key=self._xml_ns_key)
        elif isinstance(value, dict):
            value = GeographicCoverageType.from_dict(value)

        if isinstance(value, GeographicCoverageType):
            self._SubRegions.append(value)
        else:
            raise TypeError('Trying to set SubRegion element with unexpected type {}'.format(type(value)))

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        if kwargs is None:
            kwargs = OrderedDict()
        kwargs['SubRegions'] = find_children(node, 'SubRegion', xml_ns, ns_key)
        return super(GeographicCoverageType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)

    def to_node(self, doc, tag, ns_key=None, parent=None, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        node = super(GeographicCoverageType, self).to_node(
            doc, tag, ns_key=ns_key, parent=parent, check_validity=check_validity, strict=strict, exclude=exclude)
        # slap on the SubRegion children
        sub_key = self._child_xml_ns_key.get('SubRegions', ns_key)
        for entry in self._SubRegions:
            entry.to_node(doc, 'SubRegion', ns_key=sub_key, parent=node, strict=strict)
        return node

    def to_dict(self, check_validity=False, strict=DEFAULT_STRICT, exclude=()):
        out = super(GeographicCoverageType, self).to_dict(check_validity=check_validity, strict=strict, exclude=exclude)
        # slap on the SubRegion children
        if len(self.SubRegions) > 0:
            out['SubRegions'] = [
                entry.to_dict(check_validity=check_validity, strict=strict) for entry in self._SubRegions]
        return out


class GeographicAndTargetType(Serializable):
    """
    Container specifying the image coverage area in geographic coordinates, as
    well as optional data about the target of the collection/product.
    """

    _fields = ('GeographicCoverage', 'TargetInformations')
    _required = ('GeographicCoverage', )
    _collections_tags = {'TargetInformations': {'array': False, 'child_tag': 'TargetInformation'}}
    # Descriptors
    GeographicCoverage = SerializableDescriptor(
        'GeographicCoverage', GeographicCoverageType, _required, strict=DEFAULT_STRICT,
        docstring='Provides geographic coverage information.')  # type: GeographicCoverageType
    TargetInformations = SerializableListDescriptor(
        'TargetInformations', TargetInformationType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='Provides target specific geographic '
                  'information.')  # type: Union[None, List[TargetInformationType]]

    def __init__(self, GeographicCoverage=None, TargetInformations=None, **kwargs):
        """

        Parameters
        ----------
        GeographicCoverage : GeographicCoverageType
        TargetInformations : None|List[TargetInformationType]
        kwargs
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.GeographicCoverage = GeographicCoverage
        self.TargetInformations = TargetInformations
        super(GeographicAndTargetType, self).__init__(**kwargs)
