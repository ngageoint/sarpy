# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#

import copy
import os
import re
import shutil
import tempfile

from lxml import etree
import numpy as np
import pytest

from sarpy.consistency import cphd_consistency
from sarpy.io.phase_history.cphd_schema import get_schema_path

TEST_FILE_NAMES = {
    'simple': 'spotlight_example.cphd',
    'has_antenna': 'has_antenna.cphd',
    'bistatic': 'bistatic.cphd',
}
TEST_FILE_PATHS = {k: os.path.join(os.environ['SARPY_TEST_PATH'], 'cphd', v) for k,v in TEST_FILE_NAMES.items()}

GOOD_CPHD = TEST_FILE_PATHS['simple']
DEFAULT_SCHEMA = get_schema_path(version='1.0.1')

def make_elem(tag, text=None, children=None, namespace=None, attributes=None, **attrib):
    """
    Creates described element.

    Creates the Element with tag name, text, and attributes given. Attributes
    can be specified as either a dictionary or keyword arguments.

    Parameters
    ----------
    tag : str
        A string that will become the tag name.
    text : str
        A string that will become the text in the element. (Default: ``None``)
    parent : lxml.etree.ElementTree.Element
        The parent element. (Default: ``None``)
    children : lxml.etree.ElementTree
        The children elements. (Default: ``None``)
    namespace : str
        The string containing the namespace. (Default: ``None``)
    attributes : dict
        A dictionary mapping attribute names to values. (Default: ``None``)
    **attrib : list
        Keyword arguments that map to attributes. (Default: ``None``)

    Returns
    -------
    lxml.etree.ElementTree.Element
    """

    if attributes is None:
        attributes = {}
    if text is not None:
        if isinstance(text, bool):
            text = str(text).lower()
        if not isinstance(text, str):
            text = repr(text)
    attrib = copy.copy(attrib)
    attrib.update(attributes)
    attrib = {key: str(value) for key, value in attrib.items()}
    if namespace is not None:
        tag = '{{{namespace}}}{tag}'.format(namespace=namespace, tag=tag)
    retval = etree.Element(tag, attrib)
    if text is not None:
        retval.text = str(text)
    if children is not None:
        retval.extend([child for child in children if child is not None])
    return retval


@pytest.fixture
def tmpdir():
    dirname = tempfile.mkdtemp()
    yield dirname
    shutil.rmtree(dirname)


def _read_xml_str(cphd_path):
    with open(cphd_path, 'rb') as fid:
        header = cphd_consistency.read_header(fid)
        fid.seek(header['XML_BLOCK_BYTE_OFFSET'], 0)
        xml_block_size = header['XML_BLOCK_SIZE']
        return fid.read(xml_block_size).decode()


@pytest.fixture(scope='module')
def good_xml_str():
    return _read_xml_str(GOOD_CPHD)


@pytest.fixture(scope='module')
def good_xml_with_antenna():
    xml_str = _read_xml_str(TEST_FILE_PATHS['has_antenna'])
    xml_root = etree.fromstring(xml_str)
    return cphd_consistency.strip_namespace(xml_root)


@pytest.fixture
def good_xml(good_xml_str):
    good_xml_root = etree.fromstring(good_xml_str)
    good_xml_root_no_ns = cphd_consistency.strip_namespace(etree.fromstring(good_xml_str))
    yield {'with_ns': good_xml_root, 'without_ns': good_xml_root_no_ns,
           'nsmap': {'ns': re.match(r'\{(.*)\}', good_xml_root.tag).group(1)}}


@pytest.fixture
def good_header():
    with open(GOOD_CPHD, 'rb') as fid:
        return cphd_consistency.read_header(fid)


def remove_nodes(*nodes):
    for node in nodes:
        node.getparent().remove(node)


def copy_xml(elem):
    return etree.fromstring(etree.tostring(elem))


def test_from_file_cphd():
    cphdcon = cphd_consistency.CphdConsistency.from_file(str(GOOD_CPHD), DEFAULT_SCHEMA, True)
    assert isinstance(cphdcon, cphd_consistency.CphdConsistency)
    cphdcon.check()
    assert not cphdcon.failures()


def test_from_file_xml(good_xml_str, tmpdir):
    xml_file = os.path.join(tmpdir, 'cphd.xml')
    with open(xml_file, 'w') as fid:
        fid.write(good_xml_str)

    cphdcon = cphd_consistency.CphdConsistency.from_file(str(xml_file), DEFAULT_SCHEMA, False)
    assert isinstance(cphdcon, cphd_consistency.CphdConsistency)
    cphdcon.check()
    assert not cphdcon.failures()


def test_main(good_xml_str, tmpdir):
    assert not cphd_consistency.main([str(GOOD_CPHD), '--schema', DEFAULT_SCHEMA, '--signal-data'])
    assert not cphd_consistency.main([str(GOOD_CPHD), '--noschema'])
    assert not cphd_consistency.main([str(GOOD_CPHD)])

    xml_file = os.path.join(tmpdir, 'cphd.xml')
    with open(xml_file, 'w') as fid:
        fid.write(good_xml_str)
    assert not cphd_consistency.main([str(xml_file), '-v'])


@pytest.mark.parametrize('cphd_file', TEST_FILE_PATHS.values())
def test_main_each_file(cphd_file):
    assert not cphd_consistency.main([cphd_file])


def test_xml_schema_error(good_xml):
    bad_xml = copy_xml(good_xml['with_ns'])

    remove_nodes(*bad_xml.xpath('./ns:Global/ns:DomainType', namespaces=good_xml['nsmap']))
    cphd_con = cphd_consistency.CphdConsistency(bad_xml, pvps={}, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)
    cphd_con.check('check_against_schema')
    assert cphd_con.failures()


def test_check_unconnected_ids_severed_node(good_xml):
    bad_xml = copy_xml(good_xml['without_ns'])

    bad_xml.find('./Dwell/CODTime/Identifier').text += '-make-bad'
    cphd_con = cphd_consistency.CphdConsistency(bad_xml, pvps={}, header=good_header, filename=None,
                                                schema=DEFAULT_SCHEMA,
                                                check_signal_data=False)
    cphd_con.check('check_unconnected_ids')
    assert cphd_con.failures()


def test_check_unconnected_ids_extra_node(good_xml):
    bad_xml = copy_xml(good_xml['without_ns'])

    first_acf = bad_xml.find('./Antenna/AntCoordFrame')
    extra_acf = copy.deepcopy(first_acf)
    extra_acf.find('./Identifier').text += '_superfluous'
    first_acf.getparent().append(extra_acf)

    cphd_con = cphd_consistency.CphdConsistency(bad_xml, pvps={}, header=good_header, filename=None,
                                                schema=DEFAULT_SCHEMA,
                                                check_signal_data=False)
    cphd_con.check('check_unconnected_ids')
    assert cphd_con.failures()


def test_check_classification_and_release_info_error(good_xml, good_header):
    bad_xml = copy_xml(good_xml['without_ns'])

    bad_xml.find('./CollectionID/ReleaseInfo').text += '-make-bad'
    cphd_con = cphd_consistency.CphdConsistency(bad_xml, pvps={}, header=good_header, filename=None,
                                                             schema=DEFAULT_SCHEMA,
                                                             check_signal_data=False)
    cphd_con.check('check_classification_and_release_info')
    assert cphd_con.failures()


def test_error_in_check(good_xml):
    bad_xml = copy_xml(good_xml['with_ns'])
    remove_nodes(*bad_xml.xpath('./ns:Channel/ns:Parameters/ns:DwellTimes/ns:CODId', namespaces=good_xml['nsmap']))

    cphd_con = cphd_consistency.CphdConsistency(bad_xml, pvps={}, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)
    tocheck = []
    for chan_id in bad_xml.findall('./ns:Data/ns:Channel/ns:Identifier', namespaces=good_xml['nsmap']):
        tocheck.append('check_channel_dwell_exist_{}'.format(chan_id.text))
    cphd_con.check(tocheck)
    assert cphd_con.failures()


def test_polygon_size_error(good_xml):
    bad_xml = copy_xml(good_xml['with_ns'])
    ia_polygon_node = bad_xml.find('./ns:SceneCoordinates/ns:ImageArea/ns:Polygon', namespaces=good_xml['nsmap'])
    ia_polygon_node.attrib['size'] = "12345678890"

    cphd_con = cphd_consistency.CphdConsistency(bad_xml, pvps={}, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)
    cphd_con.check('check_global_imagearea_polygon')
    assert cphd_con.failures()


def test_polygon_winding_error(good_xml):
    bad_xml = copy_xml(good_xml['with_ns'])
    ia_polygon_node = bad_xml.find('./ns:SceneCoordinates/ns:ImageArea/ns:Polygon', namespaces=good_xml['nsmap'])
    size = int(ia_polygon_node.attrib['size'])
    # Reverse the order of the vertices
    for vertex in ia_polygon_node:
        vertex.attrib['index'] = str(size - int(vertex.attrib['index']) + 1)

    cphd_con = cphd_consistency.CphdConsistency(bad_xml, pvps={}, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)
    cphd_con.check('check_global_imagearea_polygon')
    assert cphd_con.failures()


@pytest.fixture
def xml_with_signal_normal(good_xml):
    root = copy_xml(good_xml['with_ns'])
    pvps = {}
    for channel_node in root.findall('./ns:Data/ns:Channel', namespaces=good_xml['nsmap']):
        chan_id = channel_node.findtext('./ns:Identifier', namespaces=good_xml['nsmap'])
        num_vect = int(channel_node.findtext('./ns:NumVectors', namespaces=good_xml['nsmap']))
        pvps[chan_id] = np.ones(num_vect, dtype=[('SIGNAL', 'i8')])
        chan_param_node = root.xpath('./ns:Channel/ns:Parameters/ns:Identifier[text()="{}"]/..'.format(chan_id),
                                     namespaces=good_xml['nsmap'])[0]
        chan_param_node.append(make_elem('SignalNormal', 'true', namespace=good_xml['nsmap']['ns']))


    return pvps, root, good_xml['nsmap']


def test_signalnormal(xml_with_signal_normal):
    pvps, root, nsmap = xml_with_signal_normal

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_signalnormal_{}'.format(key) for key in pvps.keys()]
    cphd_con.check(tocheck)
    assert not cphd_con.failures()


def test_signalnormal_bad_pvp(xml_with_signal_normal):
    pvps, root, nsmap = xml_with_signal_normal

    for idx, pvp in enumerate(pvps.values()):
        pvp['SIGNAL'][idx] = 0
    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)
    tocheck = ['check_channel_signalnormal_{}'.format(key) for key in pvps.keys()]
    cphd_con.check(tocheck)
    assert len(cphd_con.failures()) == len(pvps)

    for norm_node in root.findall('./ns:Channel/ns:Parameters/ns:SignalNormal', namespaces=nsmap):
        norm_node.text = 'false'
    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)
    cphd_con.check(tocheck)
    assert not cphd_con.failures()

    no_sig_pvp = {name: np.zeros(pvp.shape, dtype=[('notsignal', 'i8')]) for name, pvp in pvps.items()}
    cphd_con = cphd_consistency.CphdConsistency(root, pvps=no_sig_pvp, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)
    cphd_con.check(tocheck)
    assert cphd_con.failures()


@pytest.fixture
def xml_without_fxfixed(good_xml):
    root = copy_xml(good_xml['with_ns'])
    pvps = {}
    for channel_node in root.findall('./ns:Data/ns:Channel', namespaces=good_xml['nsmap']):
        chan_id = channel_node.findtext('./ns:Identifier', namespaces=good_xml['nsmap'])
        num_vect = int(channel_node.findtext('./ns:NumVectors', namespaces=good_xml['nsmap']))
        pvps[chan_id] = np.zeros(num_vect, dtype=[('FX1', 'f8'), ('FX2', 'f8')])
        pvps[chan_id]['FX1'] = np.linspace(1.0, 1.1, num_vect)
        pvps[chan_id]['FX2'] = np.linspace(2.0, 2.2, num_vect)
        chan_param_node = root.xpath('./ns:Channel/ns:Parameters/ns:Identifier[text()="{}"]/..'.format(chan_id),
                                     namespaces=good_xml['nsmap'])[0]
        chan_param_node.find('./ns:FXFixed', namespaces=good_xml['nsmap']).text = 'false'

    root.find('./ns:Channel/ns:FXFixedCPHD', namespaces=good_xml['nsmap']).text = 'false'
    return pvps, root, good_xml['nsmap']


def test_fxfixed(xml_without_fxfixed):
    pvps, root, nsmap = xml_without_fxfixed

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_fxfixed_{}'.format(key) for key in pvps.keys()]
    tocheck.append('check_file_fxfixed')
    cphd_con.check(tocheck)
    assert not cphd_con.failures()


@pytest.fixture
def xml_without_toafixed(good_xml):
    root = copy_xml(good_xml['with_ns'])
    pvps = {}
    for channel_node in root.findall('./ns:Data/ns:Channel', namespaces=good_xml['nsmap']):
        chan_id = channel_node.findtext('./ns:Identifier', namespaces=good_xml['nsmap'])
        num_vect = int(channel_node.findtext('./ns:NumVectors', namespaces=good_xml['nsmap']))
        pvps[chan_id] = np.zeros(num_vect, dtype=[('TOA1', 'f8'), ('TOA2', 'f8')])
        pvps[chan_id]['TOA1'] = np.linspace(1.0, 1.1, num_vect)
        pvps[chan_id]['TOA2'] = np.linspace(2.0, 2.2, num_vect)
        chan_param_node = root.xpath('./ns:Channel/ns:Parameters/ns:Identifier[text()="{}"]/..'.format(chan_id),
                                     namespaces=good_xml['nsmap'])[0]
        chan_param_node.find('./ns:TOAFixed', namespaces=good_xml['nsmap']).text = 'false'

    root.find('./ns:Channel/ns:TOAFixedCPHD', namespaces=good_xml['nsmap']).text = 'false'
    return pvps, root, good_xml['nsmap']


def test_channel_toafixed(xml_without_toafixed):
    pvps, root, nsmap = xml_without_toafixed

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_toafixed_{}'.format(key) for key in pvps.keys()]
    tocheck.append('check_file_toafixed')
    cphd_con.check(tocheck)
    assert not cphd_con.failures()


@pytest.fixture
def xml_without_srpfixed(good_xml):
    root = copy_xml(good_xml['with_ns'])
    pvps = {}
    for channel_node in root.findall('./ns:Data/ns:Channel', namespaces=good_xml['nsmap']):
        chan_id = channel_node.findtext('./ns:Identifier', namespaces=good_xml['nsmap'])
        num_vect = int(channel_node.findtext('./ns:NumVectors', namespaces=good_xml['nsmap']))
        pvps[chan_id] = np.zeros(num_vect, dtype=[('SRPPos', 'f8', 3)])
        pvps[chan_id]['SRPPos'][:, 0] = np.linspace(1.0, 10, num_vect)
        pvps[chan_id]['SRPPos'][:, 1] = np.linspace(2.0, 20, num_vect)
        pvps[chan_id]['SRPPos'][:, 2] = np.linspace(3.0, 30, num_vect)
        chan_param_node = root.xpath('./ns:Channel/ns:Parameters/ns:Identifier[text()="{}"]/..'.format(chan_id),
                                     namespaces=good_xml['nsmap'])[0]
        chan_param_node.find('./ns:SRPFixed', namespaces=good_xml['nsmap']).text = 'false'

    root.find('./ns:Channel/ns:SRPFixedCPHD', namespaces=good_xml['nsmap']).text = 'false'
    return pvps, root, good_xml['nsmap']


def test_channel_srpfixed(xml_without_srpfixed):
    pvps, root, nsmap = xml_without_srpfixed

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_srpfixed_{}'.format(key) for key in pvps.keys()]
    tocheck.append('check_file_srpfixed')
    cphd_con.check(tocheck)
    assert not cphd_con.failures()


@pytest.fixture
def xml_with_txrcv(good_xml):
    root = copy_xml(good_xml['with_ns'])
    root.append(make_elem('TxRcv', namespace=good_xml['nsmap']['ns'], children=[
        make_elem('NumTxWFs', 2, namespace=good_xml['nsmap']['ns']),
        make_elem('TxWFParameters', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('Identifier', 'wf_unit_test_1', namespace=good_xml['nsmap']['ns']),
        ]),
        make_elem('TxWFParameters', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('Identifier', 'wf_unit_test_2', namespace=good_xml['nsmap']['ns']),
        ]),
        make_elem('NumRcvs', 2, namespace=good_xml['nsmap']['ns']),
        make_elem('RcvParameters', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('Identifier', 'rcv_unit_test_1', namespace=good_xml['nsmap']['ns']),
        ]),
        make_elem('RcvParameters', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('Identifier', 'rcv_unit_test_2', namespace=good_xml['nsmap']['ns']),
        ])
    ]))
    chan_param_node = root.xpath('./ns:Channel/ns:Parameters',
                                 namespaces=good_xml['nsmap'])[0]
    chan_param_node.append(make_elem('TxRcv', namespace=good_xml['nsmap']['ns'], children=[
        make_elem('TxWFId', 'wf_unit_test_1', namespace=good_xml['nsmap']['ns']),
        make_elem('TxWFId', 'wf_unit_test_2', namespace=good_xml['nsmap']['ns']),
        make_elem('RcvId', 'rcv_unit_test_1', namespace=good_xml['nsmap']['ns']),
        make_elem('RcvId', 'rcv_unit_test_2', namespace=good_xml['nsmap']['ns']),
    ]))
    chan_ids = [chan_param_node.findtext('./ns:Identifier', namespaces=good_xml['nsmap'])]

    return chan_ids, root, good_xml['nsmap']


def test_txrcv(xml_with_txrcv):
    chan_ids, root, nsmap = xml_with_txrcv

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_txrcv_exist_{}'.format(key) for key in chan_ids]
    cphd_con.check(tocheck)
    assert not cphd_con.failures()


def test_txrcv_bad_txwfid(xml_with_txrcv):
    chan_ids, root, nsmap = xml_with_txrcv

    chan_param_node = root.xpath('./ns:Channel/ns:Parameters/ns:Identifier[text()="{}"]/..'.format(chan_ids[0]),
                                 namespaces=nsmap)[0]
    chan_param_node.xpath('./ns:TxRcv/ns:TxWFId', namespaces=nsmap)[-1].text = 'missing'

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_txrcv_exist_{}'.format(key) for key in chan_ids]
    cphd_con.check(tocheck)
    assert cphd_con.failures()


def test_antenna_bad_acf_count(good_xml):
    root = copy_xml(good_xml['with_ns'])
    antenna_node = root.find('./ns:Antenna', namespaces=good_xml['nsmap'])
    antenna_node.xpath('./ns:NumACFs', namespaces=good_xml['nsmap'])[-1].text += '2'
    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                schema=DEFAULT_SCHEMA,
                                                check_signal_data=False)
    cphd_con.check('check_antenna')
    assert cphd_con.failures()


def test_antenna_bad_apc_count(good_xml):
    root = copy_xml(good_xml['with_ns'])
    antenna_node = root.find('./ns:Antenna', namespaces=good_xml['nsmap'])
    antenna_node.xpath('./ns:NumAPCs', namespaces=good_xml['nsmap'])[-1].text += '2'
    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                schema=DEFAULT_SCHEMA,
                                                check_signal_data=False)
    cphd_con.check('check_antenna')
    assert cphd_con.failures()


def test_antenna_bad_antpats_count(good_xml):
    root = copy_xml(good_xml['with_ns'])
    antenna_node = root.find('./ns:Antenna', namespaces=good_xml['nsmap'])
    antenna_node.xpath('./ns:NumAntPats', namespaces=good_xml['nsmap'])[-1].text += '2'
    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                schema=DEFAULT_SCHEMA,
                                                check_signal_data=False)
    cphd_con.check('check_antenna')
    assert cphd_con.failures()


def test_antenna_non_matching_acfids(good_xml):
    root = copy_xml(good_xml['with_ns'])
    antenna_node = root.find('./ns:Antenna', namespaces=good_xml['nsmap'])
    antenna_node.xpath('./ns:AntPhaseCenter/ns:ACFId', namespaces=good_xml['nsmap'])[-1].text += '_wrong'
    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                schema=DEFAULT_SCHEMA,
                                                check_signal_data=False)
    cphd_con.check('check_antenna')
    assert cphd_con.failures()


def test_txrcv_bad_rcvid(xml_with_txrcv):
    chan_ids, root, nsmap = xml_with_txrcv

    chan_param_node = root.xpath('./ns:Channel/ns:Parameters/ns:Identifier[text()="{}"]/..'.format(chan_ids[0]),
                                 namespaces=nsmap)[0]
    chan_param_node.xpath('./ns:TxRcv/ns:RcvId', namespaces=nsmap)[-1].text = 'missing'

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_txrcv_exist_{}'.format(key) for key in chan_ids]
    cphd_con.check(tocheck)
    assert cphd_con.failures()


def test_txrcv_missing_channel_node(xml_with_txrcv):
    chan_ids, root, nsmap = xml_with_txrcv

    chan_param_node = root.xpath('./ns:Channel/ns:Parameters/ns:Identifier[text()="{}"]/..'.format(chan_ids[0]),
                                 namespaces=nsmap)[0]
    remove_nodes(*chan_param_node.findall('./ns:TxRcv', nsmap))

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                schema=DEFAULT_SCHEMA,
                                                check_signal_data=False)

    cphd_con.check('check_txrcv_ids_in_channel')
    assert cphd_con.failures()


@pytest.fixture
def xml_with_fxbwnoise(good_xml):
    root = copy_xml(good_xml['with_ns'])
    pvps = {}
    for channel_node in root.findall('./ns:Data/ns:Channel', namespaces=good_xml['nsmap']):
        chan_id = channel_node.findtext('./ns:Identifier', namespaces=good_xml['nsmap'])
        num_vect = int(channel_node.findtext('./ns:NumVectors', namespaces=good_xml['nsmap']))
        pvps[chan_id] = np.zeros(num_vect, dtype=[('FXN1', 'f8'), ('FXN2', 'f8')])
        pvps[chan_id]['FXN1'] = np.linspace(1, 2, num_vect)
        pvps[chan_id]['FXN2'] = pvps[chan_id]['FXN1'] * 1.1
        pvps[chan_id]['FXN1'][10] = np.nan
        pvps[chan_id]['FXN2'][10] = np.nan
        chan_param_node = root.xpath('./ns:Channel/ns:Parameters/ns:Identifier[text()="{}"]/..'.format(chan_id),
                                     namespaces=good_xml['nsmap'])[0]
        chan_param_node.append(make_elem('FxBWNoise', 1.2, namespace=good_xml['nsmap']['ns']))

    return pvps, root, good_xml['nsmap']


def test_fxbwnoise(xml_with_fxbwnoise):
    pvps, root, nsmap = xml_with_fxbwnoise

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_fxbwnoise_{}'.format(key) for key in pvps.keys()]
    cphd_con.check(tocheck)
    assert not cphd_con.failures()


def test_fxbwnoise_bad_domain(xml_with_fxbwnoise):
    pvps, root, nsmap = xml_with_fxbwnoise

    root.find('./ns:Global/ns:DomainType', namespaces=nsmap).text = 'TOA'

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_fxbwnoise_{}'.format(key) for key in pvps.keys()]
    cphd_con.check(tocheck)
    assert cphd_con.failures()


def test_fxbwnoise_bad_value(xml_with_fxbwnoise):
    pvps, root, nsmap = xml_with_fxbwnoise

    chan_id = list(pvps.keys())[-1]
    pvps[chan_id]['FXN1'][0] = 0.5

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=pvps, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = ['check_channel_fxbwnoise_{}'.format(key) for key in pvps.keys()]
    cphd_con.check(tocheck)
    assert cphd_con.failures()


def test_geoinfo_polygons(good_xml):
    root = copy_xml(good_xml['with_ns'])
    root.append(make_elem('GeoInfo', namespace=good_xml['nsmap']['ns'], children=[
        make_elem('Polygon', size='3', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('Vertex', index='1', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('Lat', 0.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Lon', 0.0, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('Vertex', index='2', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('Lat', 1.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Lon', 0.0, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('Vertex', index='3', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('Lat', 1.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Lon', 1.0, namespace=good_xml['nsmap']['ns']),
            ]),
        ])
    ]))

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    cphd_con.check('check_geoinfo_polygons')
    assert not cphd_con.failures()


def test_geoinfo_polygons_bad_order(good_xml):
    root = copy_xml(good_xml['with_ns'])
    root.append(make_elem('GeoInfo', namespace=good_xml['nsmap']['ns'], children=[
        make_elem('Polygon', size='3', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('Vertex', index='1', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('Lat', 0.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Lon', 0.0, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('Vertex', index='2', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('Lat', 0.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Lon', 1.0, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('Vertex', index='3', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('Lat', 1.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Lon', 1.0, namespace=good_xml['nsmap']['ns']),
            ]),
        ])
    ]))

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    cphd_con.check('check_geoinfo_polygons')
    assert cphd_con.failures()

@pytest.fixture
def xml_with_channel_imagearea(good_xml):
    root = copy_xml(good_xml['with_ns'])
    for chan_param_node in root.xpath('./ns:Channel/ns:Parameters', namespaces=good_xml['nsmap']):
        chan_param_node.append(make_elem('ImageArea', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('X1Y1', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('X', -50, namespace=good_xml['nsmap']['ns']),
                make_elem('Y', -50, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('X2Y2', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('X', 50, namespace=good_xml['nsmap']['ns']),
                make_elem('Y', 50, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('Polygon', size='4', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('Vertex', index='1', namespace=good_xml['nsmap']['ns'], children=[
                    make_elem('X', -50.0, namespace=good_xml['nsmap']['ns']),
                    make_elem('Y', 0.0, namespace=good_xml['nsmap']['ns']),
                ]),
                make_elem('Vertex', index='2', namespace=good_xml['nsmap']['ns'], children=[
                    make_elem('X', 0.0, namespace=good_xml['nsmap']['ns']),
                    make_elem('Y', 50.0, namespace=good_xml['nsmap']['ns']),
                ]),
                make_elem('Vertex', index='3', namespace=good_xml['nsmap']['ns'], children=[
                    make_elem('X', 50.0, namespace=good_xml['nsmap']['ns']),
                    make_elem('Y', 0.0, namespace=good_xml['nsmap']['ns']),
                ]),
                make_elem('Vertex', index='4', namespace=good_xml['nsmap']['ns'], children=[
                    make_elem('X', 0.0, namespace=good_xml['nsmap']['ns']),
                    make_elem('Y', -50.0, namespace=good_xml['nsmap']['ns']),
                ]),
            ])
        ]))

    return root, good_xml['nsmap']


def test_channel_image_area(xml_with_channel_imagearea):
    root, nsmap = xml_with_channel_imagearea

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    tocheck = []
    for chan_id in root.findall('./ns:Data/ns:Channel/ns:Identifier', namespaces=nsmap):
        tocheck.append('check_channel_imagearea_x1y1_{}'.format(chan_id.text))
        tocheck.append('check_channel_imagearea_polygon_{}'.format(chan_id.text))
    cphd_con.check(tocheck)
    assert not cphd_con.failures()


@pytest.fixture
def xml_with_extendedarea(good_xml):
    root = copy_xml(good_xml['with_ns'])
    scene = root.find('./ns:SceneCoordinates', namespaces=good_xml['nsmap'])
    scene.append(make_elem('ExtendedArea', namespace=good_xml['nsmap']['ns'], children=[
        make_elem('X1Y1', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('X', -1000, namespace=good_xml['nsmap']['ns']),
            make_elem('Y', -1000, namespace=good_xml['nsmap']['ns']),
        ]),
        make_elem('X2Y2', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('X', 1000, namespace=good_xml['nsmap']['ns']),
            make_elem('Y', 1000, namespace=good_xml['nsmap']['ns']),
        ]),
        make_elem('Polygon', size='4', namespace=good_xml['nsmap']['ns'], children=[
            make_elem('Vertex', index='1', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('X', -1000.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Y', 0.0, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('Vertex', index='2', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('X', 0.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Y', 1000.0, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('Vertex', index='3', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('X', 1000.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Y', 0.0, namespace=good_xml['nsmap']['ns']),
            ]),
            make_elem('Vertex', index='4', namespace=good_xml['nsmap']['ns'], children=[
                make_elem('X', 0.0, namespace=good_xml['nsmap']['ns']),
                make_elem('Y', -1000.0, namespace=good_xml['nsmap']['ns']),
            ]),
        ])
    ]))
    return root, good_xml['nsmap']


def test_extended_imagearea(xml_with_extendedarea):
    root, nsmap = xml_with_extendedarea

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    cphd_con.check(['check_extended_imagearea_polygon', 'check_extended_imagearea_x1y1_x2y2'])
    assert not cphd_con.failures()


def test_extended_imagearea_polygon_bad_extent(xml_with_extendedarea):
    root, nsmap = xml_with_extendedarea
    root.find('./ns:SceneCoordinates/ns:ExtendedArea/ns:X2Y2/ns:X', namespaces=nsmap).text = '2000'

    cphd_con = cphd_consistency.CphdConsistency(root, pvps=None, header=None, filename=None,
                                                                  schema=DEFAULT_SCHEMA,
                                                                  check_signal_data=False)

    cphd_con.check('check_extended_imagearea_polygon')
    assert cphd_con.failures()


def test_antenna_missing_channel_node(good_xml_with_antenna):
    bad_xml = copy_xml(good_xml_with_antenna)
    remove_nodes(*bad_xml.findall('./Channel/Parameters/Antenna'))

    cphd_con = cphd_consistency.CphdConsistency(bad_xml, pvps=None, header=None, filename=None,
                                                schema=DEFAULT_SCHEMA,
                                                check_signal_data=False)

    cphd_con.check('check_antenna_ids_in_channel')
    assert cphd_con.failures()


def test_refgeom_bad_root():
    cphd_con = cphd_consistency.CphdConsistency.from_file(GOOD_CPHD,
                                                          schema=DEFAULT_SCHEMA,
                                                          check_signal_data=False)
    bad_node = cphd_con.xml.find('./ReferenceGeometry/SRPCODTime')
    bad_node.text = '24' + bad_node.text

    cphd_con.check('check_refgeom_root')
    assert cphd_con.failures()


def test_refgeom_bad_monostatic():
    cphd_con = cphd_consistency.CphdConsistency.from_file(GOOD_CPHD,
                                                          schema=DEFAULT_SCHEMA,
                                                          check_signal_data=False)
    bad_node = cphd_con.xml.find('./ReferenceGeometry/Monostatic/AzimuthAngle')
    bad_node.text = str((float(bad_node.text) + 3) % 360)

    cphd_con.check('check_refgeom_monostatic')
    assert cphd_con.failures()


def test_refgeom_bad_bistatic():
    cphd_con = cphd_consistency.CphdConsistency.from_file(TEST_FILE_PATHS['bistatic'],
                                                          schema=DEFAULT_SCHEMA,
                                                          check_signal_data=False)
    bad_node = cphd_con.xml.find('./ReferenceGeometry/Bistatic/RcvPlatform/SlantRange')
    bad_node.text = '2' + bad_node.text

    cphd_con.check('check_refgeom_bistatic')
    assert cphd_con.failures()
