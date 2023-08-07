#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import lxml.etree
import pytest

from sarpy.consistency import cphd_consistency
import sarpy.io.phase_history.cphd1_elements.CPHD as sarpy_cphd1


@pytest.fixture(
    params=[
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
    out_file.write_text(original_read.to_xml_string(check_validity=True))
    reread_tree = lxml.etree.parse(str(out_file))

    original_nodes = {original_tree.getelementpath(x) for x in original_tree.iter()}
    new_nodes = {reread_tree.getelementpath(x) for x in reread_tree.iter()}
    assert original_nodes == new_nodes
