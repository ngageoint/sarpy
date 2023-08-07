#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import GeoData


@pytest.fixture
def geo_info_doc():
    root = ET.Element("GeoInfoType")
    root.attrib["name"] = "targets"

    point_node = ET.SubElement(root, "Point")
    lat_node = ET.SubElement(point_node, "Lat")
    lat_node.text = "1.0"
    lon_node = ET.SubElement(point_node, "Lon")
    lon_node.text = "2.0"

    doc = ET.ElementTree(root)
    return doc


def test_sicd_elements_geoinfotype(sicd, geo_info_doc, kwargs):
    # Init with kwargs
    geo_info_type = GeoData.GeoInfoType(name="test", **kwargs)
    assert geo_info_type._xml_ns == kwargs["_xml_ns"]
    assert geo_info_type._xml_ns_key == kwargs["_xml_ns_key"]

    # Instantiate with various GeoInfos styles
    geo_info_type = GeoData.GeoInfoType(
        name="target0",
        Point=[1, 2, 3],
        Line=[[1, 2], [3, 4]],
        Polygon=[[1, 2], [3, 4], [5, 6], [7, 8]],
        GeoInfos=sicd.GeoData.GeoInfos,
    )
    assert geo_info_type.FeatureType == "Point"

    geo_info_type1 = GeoData.GeoInfoType(name="target0", GeoInfos=geo_info_type)
    assert geo_info_type1.GeoInfos[0] == geo_info_type

    geo_info_type = GeoData.GeoInfoType(
        name="targets",
        GeoInfos=[
            sicd.GeoData.GeoInfos[0].GeoInfos[0],
            sicd.GeoData.GeoInfos[0].GeoInfos[1],
        ],
    )
    assert geo_info_type.FeatureType is None
    assert geo_info_type.getGeoInfo("target0") == [sicd.GeoData.GeoInfos[0].GeoInfos[0]]

    with pytest.raises(ValueError, match="GeoInfos got unexpected type"):
        GeoData.GeoInfoType(name="target0", GeoInfos=3.0)

    geo_info_dict = geo_info_type.to_dict()
    geo_info_type.addGeoInfo(geo_info_dict)

    # Invalid type
    with pytest.raises(
        TypeError, match="Trying to set GeoInfo element with unexpected type"
    ):
        geo_info_type = geo_info_type.addGeoInfo(3.0)

    # Basic validity checks
    assert geo_info_type._validate_features()
    assert geo_info_type._basic_validity_check()

    # to/from_node round trip
    this_node = geo_info_type.to_node(doc=geo_info_doc, tag="GeoInfoType")
    assert this_node.tag == "GeoInfoType"
    assert this_node.attrib["name"] == "targets"
    geo_info_type1 = geo_info_type.from_node(this_node, None)
    assert isinstance(geo_info_type1, GeoData.GeoInfoType)
    assert geo_info_type1.name == "targets"


def test_sicd_elements_scptype(kwargs):
    # Init with kwargs
    scp_type = GeoData.SCPType(LLH=[0.0, 0.0, 0.0], **kwargs)
    assert scp_type._xml_ns == kwargs["_xml_ns"]
    assert scp_type._xml_ns_key == kwargs["_xml_ns_key"]
    assert np.all(scp_type.LLH.get_array() == [0.0, 0.0, 0.0])
    assert np.all(scp_type.ECF.get_array() == [6378137.0, 0.0, 0.0])

    assert scp_type.get_image_center_abbreviation() == "00N000E"


def test_sicd_elements_geodatatype(sicd, geo_info_doc, kwargs):
    # Init with kwargs
    scp_type = GeoData.GeoDataType(**kwargs)
    assert scp_type._xml_ns == kwargs["_xml_ns"]
    assert scp_type._xml_ns_key == kwargs["_xml_ns_key"]

    # Instantiate with GeoInfoType and lists
    geo_data_type = GeoData.GeoDataType(
        EarthModel="WGS_84",
        SCP=sicd.GeoData.SCP,
        ImageCorners=sicd.GeoData.ImageCorners,
        GeoInfos=sicd.GeoData.GeoInfos[0],
    )
    assert geo_data_type.GeoInfos == sicd.GeoData.GeoInfos

    geo_data_type1 = GeoData.GeoDataType(
        GeoInfos=[
            sicd.GeoData.GeoInfos[0].GeoInfos[0],
            sicd.GeoData.GeoInfos[0].GeoInfos[1],
        ]
    )
    assert geo_data_type1.GeoInfos[0] == geo_data_type.GeoInfos[0].GeoInfos[0]
    assert geo_data_type1.getGeoInfo("target0") == [
        sicd.GeoData.GeoInfos[0].GeoInfos[0]
    ]

    # Try with invalid GeoInfo type
    with pytest.raises(ValueError, match="GeoInfos got unexpected type"):
        GeoData.GeoDataType(GeoInfos=3.0)

    # derive is a placeholder, just invoke it here
    geo_data_type.derive()

    # Verify adding GeoInfos with different types
    assert (len(geo_data_type.GeoInfos)) == 1
    geo_data_type.setGeoInfo(sicd.GeoData.GeoInfos[0].GeoInfos[0])
    assert (len(geo_data_type.GeoInfos)) == 2
    geo_data_type.setGeoInfo(geo_data_type.GeoInfos[0].GeoInfos[0].to_dict())
    assert (len(geo_data_type.GeoInfos)) == 3

    # Try with invalid value
    with pytest.raises(
        TypeError, match="Trying to set GeoInfo element with unexpected type"
    ):
        geo_data_type = geo_data_type.setGeoInfo(3.0)

    # Verify to/from_node round trip
    this_node = geo_data_type.to_node(doc=geo_info_doc, tag="GeoInfoType")
    assert this_node.tag == "GeoInfoType"
    geo_data_type1 = geo_data_type.from_node(this_node, None)
    assert geo_data_type1.to_dict() == geo_data_type.to_dict()

    # Assert type is valid
    assert geo_data_type._basic_validity_check()
