"""
The MatchInfoType definition.
"""

from typing import List

from ._base import Serializable, DEFAULT_STRICT, \
    _StringDescriptor, _IntegerDescriptor, _SerializableArrayDescriptor
from ._blocks import ParameterType


__classification__ = "UNCLASSIFIED"


class MatchCollectionType(Serializable):
    """The match collection type."""
    _fields = ('CoreName', 'MatchIndex', 'Parameters')
    _required = ('CoreName', )
    _collections_tags = {'Parameters': {'array': False, 'child_tag': 'Parameter'}}
    # descriptors
    CoreName = _StringDescriptor(  # TODO: VERIFY - is there no validator for this? Is it unstructured? Examples?
        'CoreName', _required, strict=DEFAULT_STRICT,
        docstring='Unique identifier for the match type.')  # type: str
    MatchIndex = _IntegerDescriptor(  # TODO: CLARIFY - what is the purpose of this? What's it an index into?
        'MatchIndex', _required, strict=DEFAULT_STRICT,
        docstring='Collection sequence index for the match collection.')  # type: int
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
        docstring='The match type identifier. *Examples - "COHERENT" or "STEREO"*')  # type: str
    CurrentIndex = _IntegerDescriptor(  # TODO: CLARIFY - what is the purpose of this? What's it an index into?
        'CurrentIndex', _required, strict=DEFAULT_STRICT,
        docstring='Collection sequence index for the current collection.')  # type: int
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

    # TODO: allow for sicd 0.5 version, see sicd.py line 1196. Why all this concern with removing duplicates?
    #   * Prior to 1.0, there was no MatchType.TypeId field. Instead, there was an optional
    #       MatchType.MatchType field (really?) that played the same role.
    #   * In this case, Matchtype.CurrentIndex also does not exist. Instead, one of the elements of MatchCollections
    #       will have a parameter "CURRENT_INSTANCE"=<CurrentIndex>. That appears to be the only purpose for that
    #       element of of MatchCollections, and it does not appear to be copied at sicd.py line


class MatchInfoType(Serializable):
    """The match information container."""
    _fields = ('NumMatchTypes', 'MatchTypes')
    _required = ('MatchTypes', )
    _collections_tags = {'MatchTypes': {'array': False, 'child_tag': 'MatchType'}}
    # descriptors
    # TODO: VERIFY - in sicd.py, it looks like the choice was to call this Collect.
    #   That's not used anywhere in this code base (besides sicd.py lines ~ 1200).
    #   Does it matter?
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
