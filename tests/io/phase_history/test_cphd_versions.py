#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import lxml.etree
import pytest

import sarpy.io.phase_history.cphd1_elements.CPHD as sarpy_cphd1


@pytest.mark.parametrize(
    "cphd_xml_path",
    [
        "data/syntax-only-cphd-1.0.1-monostatic.xml",
        "data/syntax-only-cphd-1.0.1-bistatic.xml",
        "data/syntax-only-cphd-1.1.0-monostatic.xml",
        "data/syntax-only-cphd-1.1.0-bistatic.xml",
    ],
)
def test_cphd_version_required(cphd_xml_path, tests_path):
    """Ensure the CPHD code can detect the correct version based on the XML."""
    xml_path = str(tests_path / cphd_xml_path)
    if "1.0.1" in xml_path:
        cphd = sarpy_cphd1.CPHDType.from_xml_file(xml_path)
        assert cphd.version_required() == (1, 0, 1)

    if "1.1.0" in xml_path:
        cphd = sarpy_cphd1.CPHDType.from_xml_file(xml_path)
        assert cphd.version_required() == (1, 1, 0)


def test_cphd_version_required_with_delay_bias(tests_path, tmp_path):
    # Take a 1.0.1 xml and add a 1.1.0 specific node
    xml_101 = tests_path / "data/syntax-only-cphd-1.0.1-bistatic.xml"
    tree = lxml.etree.parse(str(xml_101))

    radar_sensor = tree.find(
        "{*}ErrorParameters/{*}Bistatic/{*}RcvPlatform/{*}RadarSensor"
    )
    delay_bias = lxml.etree.Element("DelayBias")
    delay_bias.text = "0.001"
    radar_sensor.append(delay_bias)

    modified_xml_101 = tmp_path / "modified_xml_101.xml"
    tree.write(str(modified_xml_101))
    modified_cphd = sarpy_cphd1.CPHDType.from_xml_file(modified_xml_101)
    assert modified_cphd.version_required() == (1, 1, 0)
