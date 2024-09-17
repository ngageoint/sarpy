#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import pytest

from sarpy.io.phase_history.cphd1_elements import CPHD


@pytest.fixture()
def cphd(tests_path):
    xml_file = tests_path / 'data/syntax-only-cphd-1.1.0-monostatic.xml'
    structure = CPHD.CPHDType().from_xml_file(xml_file)

    return structure


@pytest.fixture()
def bistatic_cphd(tests_path):
    xml_file = tests_path / 'data/syntax-only-cphd-1.1.0-bistatic.xml'
    structure = CPHD.CPHDType().from_xml_file(xml_file)

    return structure


@pytest.fixture()
def kwargs():
    return {'_xml_ns': 'ns', '_xml_ns_key': 'key'}


@pytest.fixture()
def tol():
    return 1e-8

