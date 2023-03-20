#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import re

import lxml.etree
import pytest

import sarpy.io.product.sidd_schema as sarpy_sidd


@pytest.mark.parametrize('sidd_version', sarpy_sidd.get_versions())
def test_validate_xml_ns(sidd_version):
    xml_ns, ns_key = get_sidd_nsmap(sidd_version)
    assert sarpy_sidd.validate_xml_ns(xml_ns, ns_key)


def get_sidd_nsmap(sidd_version):
    schema_root = lxml.etree.parse(sarpy_sidd.get_schema_path(sidd_version)).getroot()
    reverse_nsmap = {v: k for k, v in schema_root.nsmap.items()}
    return schema_root.nsmap, reverse_nsmap[schema_root.get('targetNamespace')]


@pytest.fixture
def sidd_nsmap():
    return get_sidd_nsmap(sarpy_sidd.get_versions()[-1])


def test_validate_xml_ns_no_ns_key(sidd_nsmap):
    xml_ns, ns_key = sidd_nsmap
    del xml_ns[ns_key]
    with pytest.raises(ValueError):
        sarpy_sidd.validate_xml_ns(xml_ns, ns_key)


def test_validate_xml_ns_unmapped(sidd_nsmap):
    xml_ns, _ = sidd_nsmap
    xml_ns['bad'] = 'urn:SIDD:9.9.9'
    assert not sarpy_sidd.validate_xml_ns(xml_ns, 'bad')


def test_validate_xml_change_keys(sidd_nsmap):
    xml_ns, ns_key = sidd_nsmap
    xml_ns_changed_keys = {f'{k}_changed': v for k, v in xml_ns.items()}
    assert sarpy_sidd.validate_xml_ns(xml_ns_changed_keys, f'{ns_key}_changed')


def test_validate_xml_mismatched_ns(sidd_nsmap, caplog):
    xml_ns, ns_key = sidd_nsmap
    xml_ns['ism'] += '_make_bad'
    assert not sarpy_sidd.validate_xml_ns(xml_ns, ns_key)
    assert re.search(r'namespace urn is expected to be.*but we got', caplog.text)


def test_validate_xml_missing_required_ns(sidd_nsmap, caplog):
    xml_ns, ns_key = sidd_nsmap
    del xml_ns['ism']
    assert not sarpy_sidd.validate_xml_ns(xml_ns, ns_key)
    assert re.search(r'No.* namespace defined.', caplog.text)


def test_validate_xml_missing_optional_ns(sidd_nsmap):
    xml_ns, ns_key = sidd_nsmap
    del xml_ns['sfa']
    assert sarpy_sidd.validate_xml_ns(xml_ns, ns_key)
