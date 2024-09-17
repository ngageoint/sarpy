#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import copy

import lxml.etree
import pytest

from sarpy.consistency import cphd_consistency
import sarpy.io.phase_history.cphd1_elements.CPHD as sarpy_cphd1
from sarpy.io.phase_history import cphd_schema


NAMESPACE_MAPPING = {  # get_schema_path doesn't use the actual namespace
    "http://api.nsgreg.nga.mil/schema/cphd/1.0.1": "urn:CPHD:1.0.1",
    "http://api.nsgreg.nga.mil/schema/cphd/1.1.0": "urn:CPHD:1.1.0",
}


@pytest.fixture(
    params=[
        "data/syntax-only-cphd-1.0.1-monostatic.xml",
        "data/syntax-only-cphd-1.0.1-bistatic.xml",
        "data/syntax-only-cphd-1.1.0-monostatic-minimal.xml",
        "data/syntax-only-cphd-1.1.0-monostatic.xml",
        "data/syntax-only-cphd-1.1.0-bistatic.xml",
    ]
)
def cphd_xml(request, tests_path):
    cphd_path = tests_path / request.param
    cphd_con = cphd_consistency.CphdConsistency.from_file(cphd_path)
    cphd_con.check("check_against_schema")
    assert not cphd_con.failures()
    assert cphd_con.passes()
    return cphd_path


def test_cphdtype_to_from_xml(cphd_xml, tmp_path):
    """Ensure that XML nodes are preserved when going from XML, through SarPy, then back to XML."""
    original_tree = lxml.etree.parse(str(cphd_xml))
    original_read = sarpy_cphd1.CPHDType.from_xml_file(cphd_xml)

    out_file = tmp_path / "read_then_write.xml"
    xmlns_urn = lxml.etree.QName(original_tree.getroot()).namespace
    out_file.write_text(original_read.to_xml_string(urn=xmlns_urn, check_validity=True))
    reread_tree = lxml.etree.parse(str(out_file))

    original_nodes = {original_tree.getelementpath(x) for x in original_tree.iter()}
    new_nodes = {reread_tree.getelementpath(x) for x in reread_tree.iter()}
    assert original_nodes == new_nodes


@pytest.mark.parametrize(
    "cphd_xml",
    [
        "data/syntax-only-cphd-1.1.0-monostatic-minimal.xml",
    ],
)
def test_cphdtype_required_fields(cphd_xml, tests_path):
    xml_file = tests_path / cphd_xml
    tree = lxml.etree.parse(xml_file)
    xmlns_urn = lxml.etree.QName(tree.getroot()).namespace
    schema = lxml.etree.XMLSchema(
        lxml.etree.parse(cphd_schema.get_schema_path(NAMESPACE_MAPPING[xmlns_urn]))
    )

    # Make sure we're starting with a truly minimal example
    paths = [tree.getelementpath(el) for el in tree.getroot().iterdescendants()]
    for path in paths:
        ctree = copy.deepcopy(tree)
        node = ctree.find(path)
        parent = node.getparent()
        parent.remove(node)
        assert not schema.validate(ctree)

    cphdtype = sarpy_cphd1.CPHDType.from_xml_file(xml_file)
    assert cphdtype.is_valid(recursive=True)
