"""
The MatchInfoType definition.
"""

from xml.etree import ElementTree
from typing import List

from ._base import Serializable, _get_node_value, DEFAULT_STRICT, \
    _StringDescriptor, _IntegerDescriptor, _SerializableArrayDescriptor
from ._blocks import ParameterType


__classification__ = "UNCLASSIFIED"


class MatchCollectionType(Serializable):
    """The match collection type."""
    _fields = ('CoreName', 'MatchIndex', 'Parameters')
    _required = ('CoreName', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    CoreName = _StringDescriptor(
        'CoreName', _required, strict=DEFAULT_STRICT,
        docstring='Unique identifier for the match type.')  # type: str
    MatchIndex = _IntegerDescriptor(
        'MatchIndex', _required, strict=DEFAULT_STRICT,
        docstring='Collection sequence index for the match collection, assuming '
                  'that this makes sense.')  # type: int
    Parameters = _SerializableArrayDescriptor(
        'Parameters', ParameterType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The match parameters.')  # type: List[ParameterType]


class MatchType(Serializable):
    """The is an array element for match information."""
    _fields = ('TypeId', 'CurrentIndex', 'NumMatchCollections', 'MatchCollections')
    _required = ('TypeId',)
    _collections_tags = {'MatchCollections': {'array': False, 'child_tag': 'MatchCollection'}}
    # descriptors
    TypeId = _StringDescriptor(
        'TypeId', _required, strict=DEFAULT_STRICT,
        docstring='The match type identifier. *Examples - "MULTI-IMAGE", "COHERENT" or "STEREO"*')  # type: str
    CurrentIndex = _IntegerDescriptor(
        'CurrentIndex', _required, strict=DEFAULT_STRICT,
        docstring='Collection sequence index for the current collection. That is, which collection in the '
                  'collection series (defined in MatchCollections) is this collection? '
                  '(1-based enumeration).')  # type: int
    MatchCollections = _SerializableArrayDescriptor(
        'MatchCollections', MatchCollectionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The match collections.')  # type: List[MatchCollectionType]

    @property
    def NumMatchCollections(self):
        """int: The number of match collections for this match type."""
        if self.MatchCollections is None:
            return 0
        else:
            return len(self.MatchCollections)


class MatchInfoType(Serializable):
    """The match information container. This contains data for multiple collection taskings."""
    # TODO: Provide a robust example, so this element isn't worthless.

    _fields = ('NumMatchTypes', 'MatchTypes')
    _required = ('MatchTypes', )
    _collections_tags = {'MatchTypes': {'array': False, 'child_tag': 'MatchType'}}
    # descriptors
    MatchTypes = _SerializableArrayDescriptor(
        'MatchTypes', MatchType, _collections_tags, _required, strict=DEFAULT_STRICT, minimum_length=1,
        docstring='The match types list.')  # type: List[MatchType]

    @property
    def NumMatchTypes(self):
        """int: The number of types of matched collections."""
        if self.MatchTypes is None:
            return 0
        else:
            return len(self.MatchTypes)

    @classmethod
    def _from_node_0_5(cls, node):
        """Helper method, not really for public usage. For XML deserialization from SICD version prior to 1.0.

        Parameters
        ----------
        node : ElementTree.Element
            dom element for serialized class instance

        Returns
        -------
        Serializable
            corresponding class instance
        """

        def get_element(tid, cid, cname, params):
            return {
                'TypeId': tid,
                'CurrentIndex': cid,
                'MatchCollections': [{'CoreName': cname, 'Parameters': params}, ]}

        # Note that this is NOT converting the MatchType.MatchCollection in spirit.
        # There isn't enough structure to guarantee that you actually can. This will
        # always yield MatchType.MatchCollection length 1, because the collection details are stuffed
        # into the parameters free form, while CurrentIndex is extracted and actually yields the
        # collection index number (likely larger than 1). This is at least confusing, but more likely
        # completely misleading.
        match_types = []
        for cnode in node.findall('Collect'):  # assumed non-empty
            # this describes one series of collects, possibly with more than one MatchType = TypeId
            # It is not clear how it would be possible to deconflict a repeat of MatchType between
            # Collect tags, so I will not.
            core_name = _get_node_value(cnode.find('CoreName'))
            current_index = None
            parameters = []
            for pnode in cnode.findall('Parameter'):
                name = pnode.attrib['name']
                value = _get_node_value(pnode)
                if name == 'CURRENT_INSTANCE':
                    current_index = int(value)  # extract the current index (and exclude)
                else:
                    parameters.append({'name': name, 'value': value})  # copy the parameter
            if current_index is None:
                continue  # I don't know what we would do?
            for tnode in cnode.findall('MatchType'):
                type_id = _get_node_value(tnode)
                match_types.append(get_element(type_id, current_index, core_name, parameters))
        if len(match_types) > 0:
            return cls(MatchTypes=match_types)
        else:
            return None

    @classmethod
    def from_node(cls, node, kwargs=None):
        if node.find('Collect') is not None:
            # This is from SICD version prior to 1.0, so handle manually.
            return cls._from_node_0_5(node)
        else:
            return super(MatchInfoType, cls).from_node(node, kwargs=kwargs)
