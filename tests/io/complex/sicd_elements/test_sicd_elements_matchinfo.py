#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.io.complex.sicd_elements import MatchInfo
from sarpy.io.xml.base import parse_xml_from_string


def test_matchinfo(kwargs):
    match_coll_type = MatchInfo.MatchCollectionType(CoreName='TEST', **kwargs)
    assert match_coll_type._xml_ns == kwargs['_xml_ns']
    assert match_coll_type._xml_ns_key == kwargs['_xml_ns_key']

    match_type = MatchInfo.MatchType(TypeID='TEST', **kwargs)
    assert match_type._xml_ns == kwargs['_xml_ns']
    assert match_type._xml_ns_key == kwargs['_xml_ns_key']
    assert match_type.NumMatchCollections == 0

    match_type = MatchInfo.MatchType(TypeID='TEST', MatchCollections=[match_coll_type])
    assert match_type.NumMatchCollections == 1

    match_info_type = MatchInfo.MatchInfoType(MatchTypes=None, **kwargs)
    assert match_info_type._xml_ns == kwargs['_xml_ns']
    assert match_info_type._xml_ns_key == kwargs['_xml_ns_key']
    assert match_info_type.NumMatchTypes == 0

    match_info_type = MatchInfo.MatchInfoType(MatchTypes=[match_type], **kwargs)
    assert match_info_type.NumMatchTypes == 1

    match_node_str = '''
        <MatchInfo>
            <NumMatchTypes>1</NumMatchTypes>
            <MatchType>
                <TypeID>COHERENT</TypeID>
                <NumMatchCollections>0</NumMatchCollections>
            </MatchType>
        </MatchInfo>
    '''
    node, ns = parse_xml_from_string(match_node_str)
    match_info_type1 = match_info_type.from_node(node, ns)
    assert match_info_type1.NumMatchTypes == 1
    assert match_info_type1.MatchTypes[0].TypeID == 'COHERENT'
    assert match_info_type1.MatchTypes[0].NumMatchCollections == 0
