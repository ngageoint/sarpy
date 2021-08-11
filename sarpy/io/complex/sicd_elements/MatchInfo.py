"""
The MatchInfoType definition.
"""

__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


from xml.etree import ElementTree
from typing import List

from sarpy.io.xml.base import Serializable, ParametersCollection, \
    get_node_value, find_first_child, find_children
from sarpy.io.xml.descriptors import StringDescriptor, IntegerDescriptor, \
    SerializableListDescriptor, ParametersDescriptor

from .base import DEFAULT_STRICT


class MatchCollectionType(Serializable):
    """The match collection type."""
    _fields = ('CoreName', 'MatchIndex', 'Parameters')
    _required = ('CoreName', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    CoreName = StringDescriptor(
        'CoreName', _required, strict=DEFAULT_STRICT,
        docstring='Unique identifier for the match type.')  # type: str
    MatchIndex = IntegerDescriptor(
        'MatchIndex', _required, strict=DEFAULT_STRICT,
        docstring='Collection sequence index for the match collection, assuming '
                  'that this makes sense.')  # type: int
    Parameters = ParametersDescriptor(
        'Parameters', _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The match parameters.')  # type: ParametersCollection

    def __init__(self, CoreName=None, MatchIndex=None, Parameters=None, **kwargs):
        """

        Parameters
        ----------
        CoreName : str
        MatchIndex : int
        Parameters : ParametersCollection|dict
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.CoreName = CoreName
        self.MatchIndex = MatchIndex
        self.Parameters = Parameters
        super(MatchCollectionType, self).__init__(**kwargs)


class MatchType(Serializable):
    """The is an array element for match information."""
    _fields = ('TypeID', 'CurrentIndex', 'NumMatchCollections', 'MatchCollections')
    _required = ('TypeID',)
    _collections_tags = {'MatchCollections': {'array': False, 'child_tag': 'MatchCollection'}}
    # descriptors
    TypeID = StringDescriptor(
        'TypeID', _required, strict=DEFAULT_STRICT,
        docstring='The match type identifier. *Examples - "MULTI-IMAGE", "COHERENT" or "STEREO"*')  # type: str
    CurrentIndex = IntegerDescriptor(
        'CurrentIndex', _required, strict=DEFAULT_STRICT,
        docstring='Collection sequence index for the current collection. That is, which collection in the '
                  'collection series (defined in MatchCollections) is this collection? '
                  '(1-based enumeration).')  # type: int
    MatchCollections = SerializableListDescriptor(
        'MatchCollections', MatchCollectionType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The match collections.')  # type: List[MatchCollectionType]

    def __init__(self, TypeID=None, CurrentIndex=None, MatchCollections=None, **kwargs):
        """

        Parameters
        ----------
        TypeID : str
        CurrentIndex : int
        MatchCollections : List[MatchCollectionType]
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.TypeID = TypeID
        self.CurrentIndex = CurrentIndex
        self.MatchCollections = MatchCollections
        super(MatchType, self).__init__(**kwargs)

    @property
    def NumMatchCollections(self):
        """int: The number of match collections for this match type."""
        if self.MatchCollections is None:
            return 0
        else:
            return len(self.MatchCollections)


class MatchInfoType(Serializable):
    """
    The match information container. This contains data for multiple collection taskings.
    """

    _fields = ('NumMatchTypes', 'MatchTypes')
    _required = ('MatchTypes', )
    _collections_tags = {'MatchTypes': {'array': False, 'child_tag': 'MatchType'}}
    # descriptors
    MatchTypes = SerializableListDescriptor(
        'MatchTypes', MatchType, _collections_tags, _required, strict=DEFAULT_STRICT,
        docstring='The match types list.')  # type: List[MatchType]

    def __init__(self, MatchTypes=None, **kwargs):
        """

        Parameters
        ----------
        MatchTypes : List[MatchType]
        kwargs : dict
        """

        if '_xml_ns' in kwargs:
            self._xml_ns = kwargs['_xml_ns']
        if '_xml_ns_key' in kwargs:
            self._xml_ns_key = kwargs['_xml_ns_key']
        self.MatchTypes = MatchTypes
        super(MatchInfoType, self).__init__(**kwargs)

    @property
    def NumMatchTypes(self):
        """int: The number of types of matched collections."""
        if self.MatchTypes is None:
            return 0
        else:
            return len(self.MatchTypes)

    @classmethod
    def _from_node_0_5(cls, node, xml_ns, ns_key):
        """
        Helper method, not really for public usage. For XML deserialization from SICD version prior to 1.0.

        Parameters
        ----------
        node : ElementTree.Element
            dom element for serialized class instance
        xml_ns : dict
            The xml namespace dictionary
        ns_key : str
            The namespace key in the dictionary
        Returns
        -------
        Serializable
            corresponding class instance
        """

        def get_element(tid, cid, cname, params):
            return {
                'TypeID': tid,
                'CurrentIndex': cid,
                'MatchCollections': [{'CoreName': cname, 'Parameters': params}, ]}

        # Note that this is NOT converting the MatchType.MatchCollection in spirit.
        # There isn't enough structure to guarantee that you actually can. This will
        # always yield MatchType.MatchCollection length 1, because the collection details are stuffed
        # into the parameters free form, while CurrentIndex is extracted and actually yields the
        # collection index number (likely larger than 1). This is at least confusing, but more likely
        # completely misleading.
        match_types = []

        coll_key = cls._child_xml_ns_key.get('Collect', ns_key)
        cnodes = find_children(node, 'Collect', xml_ns, coll_key)
        for cnode in cnodes:  # assumed non-empty
            # this describes one series of collects, possibly with more than one MatchType = TypeID
            # It is not clear how it would be possible to deconflict a repeat of MatchType between
            # Collect tags, so I will not.
            core_key = cls._child_xml_ns_key.get('CoreName', ns_key)
            core_name = get_node_value(find_first_child(cnode, 'CoreName', xml_ns, core_key))
            current_index = None
            parameters = []

            pkey = cls._child_xml_ns_key.get('Parameters', ns_key)
            pnodes = find_children(cnode, 'Parameter', xml_ns, pkey)
            for pnode in pnodes:
                name = pnode.attrib['name']
                value = get_node_value(pnode)
                if name == 'CURRENT_INSTANCE':
                    current_index = int(value)  # extract the current index (and exclude)
                else:
                    parameters.append({'name': name, 'value': value})  # copy the parameter
            if current_index is None:
                continue  # I don't know what we would do?
            mt_key = cls._child_xml_ns_key.get('MatchType', ns_key)
            mtypes = find_children(cnode, 'MatchType', xml_ns, mt_key)
            for tnode in mtypes:
                type_id = get_node_value(tnode)
                match_types.append(get_element(type_id, current_index, core_name, parameters))
        if len(match_types) > 0:
            # noinspection PyTypeChecker
            return cls(MatchTypes=match_types)
        else:
            return None

    @classmethod
    def from_node(cls, node, xml_ns, ns_key=None, kwargs=None):
        coll_key = cls._child_xml_ns_key.get('Collect', ns_key)
        coll = find_first_child(node, 'Collect', xml_ns, coll_key)
        if coll is not None:
            # This is from SICD version prior to 1.0, so handle manually.
            return cls._from_node_0_5(node, xml_ns, ns_key)
        else:
            return super(MatchInfoType, cls).from_node(node, xml_ns, ns_key=ns_key, kwargs=kwargs)
