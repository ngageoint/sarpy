#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from collections import OrderedDict
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements.blocks import LatLonRestrictionType, LatLonArrayElementType
from sarpy.io.phase_history.cphd1_elements import GeoInfo
from sarpy.io.xml.base import parse_xml_from_string


@pytest.fixture
def line_doc():
    root = ET.Element("LineType")

    doc = ET.ElementTree(root)
    return doc


@pytest.fixture
def poly_doc():
    root = ET.Element("PolygonType")

    doc = ET.ElementTree(root)
    return doc


@pytest.fixture
def geo_info_doc():
    root = ET.Element("GeoInfoType")
    root.attrib["name"] = "target0"

    doc = ET.ElementTree(root)
    return doc


def test_cphd1_elements_linetype(cphd, line_doc, kwargs):
    # Init with kwargs
    line_type = GeoInfo.LineType(Endpoint=[[1, 2], [3, 4]], **kwargs)
    assert line_type.size == 2
    assert isinstance(line_type.Endpoint, np.ndarray)
    assert line_type._xml_ns == kwargs["_xml_ns"]
    assert line_type._xml_ns_key == kwargs["_xml_ns_key"]

    # Test getitem/setitem
    first_endpoint = line_type[0]
    assert first_endpoint.Lat == 1.0
    assert first_endpoint.Lon == 2.0
    assert first_endpoint.index == 1
    point = LatLonArrayElementType(Lat=11.0, Lon=31.0, index=1)
    line_type[0] = point
    assert line_type.Endpoint[0] == point

    line_type.Endpoint = None
    assert line_type.size == 0

    line_type.Endpoint = np.array(
        [cphd.GeoInfo[0].Line[0][0], cphd.GeoInfo[0].Line[0][1]]
    )
    assert line_type.size == 2

    # dict endpints
    line_type.Endpoint = [{"Lat": "1.0", "Lon": "3.0"}, {"Lat": "5.0", "Lon": "7.0"}]
    assert line_type.size == 2

    # tuple endpoints
    line_type.Endpoint = [(1.0, 3.0), (2.0, 4.0)]
    assert line_type.size == 2

    # Endpoint with length 1
    with pytest.raises(
        ValueError, match="LineType must have at least 2 endpoints, got 1"
    ):
        line_type.Endpoint = [(1.0, 3.0)]

    # Invalid types
    with pytest.raises(
        TypeError, match="Got unexpected type for element of Endpoint array"
    ):
        line_type.Endpoint = [(1.0, 3.0), 3.0]

    with pytest.raises(TypeError, match="Got unexpected type for Endpoint array"):
        line_type.Endpoint = {"Lat": "1.0", "Lon": "3.0"}

    # to/from_node round trip
    line_type.Endpoint = [cphd.GeoInfo[0].Line[0][0], cphd.GeoInfo[0].Line[0][1]]
    this_node = line_type.to_node(doc=line_doc, tag="LineType")
    assert this_node.tag == "LineType"
    assert len(this_node.findall("Endpoint")) == 2
    line_type1 = line_type.from_node(this_node, None)
    assert isinstance(line_type1, GeoInfo.LineType)
    assert np.all(
        line_type1.Endpoint[0].get_array() == cphd.GeoInfo[0].Line[0][0].get_array()
    )

    this_node = line_type.to_node(doc=line_doc, tag="LineType", ns_key="test")
    assert this_node.tag == "test:LineType"

    line_dict = line_type.to_dict()
    assert isinstance(line_dict, OrderedDict)


def test_cphd1_elements_polygontype(cphd, poly_doc, kwargs):
    # Init with kwargs
    poly_type = GeoInfo.PolygonType(Vertex=[[1, 2], [3, 4], [4, 1]], **kwargs)
    assert poly_type.size == 3
    assert isinstance(poly_type.Vertex, np.ndarray)
    assert poly_type._xml_ns == kwargs["_xml_ns"]
    assert poly_type._xml_ns_key == kwargs["_xml_ns_key"]

    # Test getitem/setitem
    first_vertex = poly_type[0]
    assert first_vertex.Lat == 1.0
    assert first_vertex.Lon == 2.0
    assert first_vertex.index == 1
    point = LatLonArrayElementType(Lat=11.0, Lon=31.0, index=1)
    poly_type[0] = point
    assert poly_type.Vertex[0] == point

    poly_type._array = None
    assert poly_type.size == 0
    assert poly_type.Vertex == np.array([0])

    poly_type.Vertex = None
    assert poly_type.size == 0

    poly_type.Vertex = np.array(
        [
            cphd.GeoInfo[0].Polygon[0].Vertex[0],
            cphd.GeoInfo[0].Polygon[0].Vertex[1],
            cphd.GeoInfo[0].Polygon[0].Vertex[2],
        ]
    )
    assert poly_type.size == 3

    # dict vertices
    poly_type.Vertex = [
        {"Lat": "1.0", "Lon": "3.0"},
        {"Lat": "5.0", "Lon": "7.0"},
        {"Lat": "9.0", "Lon": "11.0"},
    ]
    assert poly_type.size == 3

    # tuple vertices
    poly_type.Vertex = [(1.0, 3.0), (5.0, 7.0), (9.0, 11.0)]
    assert poly_type.size == 3

    # Vertex with length 2
    with pytest.raises(
        ValueError, match="PolygonType must have at least 3 vertices, got 2"
    ):
        poly_type.Vertex = [(1.0, 3.0), (5.0, 7.0)]

    # Invalid types
    with pytest.raises(
        TypeError, match="Got unexpected type for element of Vertex array"
    ):
        poly_type.Vertex = [(1.0, 3.0), (5.0, 7.0), 3.0]

    with pytest.raises(TypeError, match="Got unexpected type for Vertex array"):
        poly_type.Vertex = {
            "vertex1": {"Lat": "1.0", "Lon": "3.0"},
            "vertex2": {"Lat": "5.0", "Lon": "7.0"},
            "vertex3": {"Lat": "9.0", "Lon": "11.0"},
        }

    # to/from_node round trip
    poly_type.Vertex = [
        cphd.GeoInfo[0].Polygon[0].Vertex[0],
        cphd.GeoInfo[0].Polygon[0].Vertex[1],
        cphd.GeoInfo[0].Polygon[0].Vertex[2],
    ]
    this_node = poly_type.to_node(doc=poly_doc, tag="PolygonType")
    assert this_node.tag == "PolygonType"
    assert len(this_node.findall("Vertex")) == 3
    poly_type1 = poly_type.from_node(this_node, None)
    assert isinstance(poly_type1, GeoInfo.PolygonType)
    assert np.all(
        poly_type1.Vertex[0].get_array()
        == cphd.GeoInfo[0].Polygon[0].Vertex[0].get_array()
    )

    this_node = poly_type.to_node(doc=poly_doc, tag="PolygonType", ns_key="test")
    assert this_node.tag == "test:PolygonType"

    line_dict = poly_type.to_dict()
    assert isinstance(line_dict, OrderedDict)


def test_cphd1_elements_geoinfotype(cphd, geo_info_doc, kwargs, caplog):
    # Init with kwargs
    geo_info_type = GeoInfo.GeoInfoType(name="test", **kwargs)
    assert geo_info_type._xml_ns == kwargs["_xml_ns"]
    assert geo_info_type._xml_ns_key == kwargs["_xml_ns_key"]

    # Instantiate with various GeoInfos styles
    point = LatLonRestrictionType(Lat=1.0, Lon=3.0)
    line_type = GeoInfo.LineType(Endpoint=[[1, 2], [3, 4]])
    poly_type = GeoInfo.PolygonType(Vertex=[[1, 2], [3, 4], [4, 1]])
    geo_info_type = GeoInfo.GeoInfoType(
        name="targets",
        Point=point,
        Line=line_type,
        Polygon=poly_type,
        GeoInfo=cphd.GeoInfo[0].GeoInfo,
    )
    assert isinstance(geo_info_type.Point[0], LatLonRestrictionType)
    assert geo_info_type.Point[0] == point
    assert isinstance(geo_info_type.Line[0], GeoInfo.LineType)
    assert geo_info_type.Line[0] == line_type
    assert isinstance(geo_info_type.Polygon[0], GeoInfo.PolygonType)
    assert geo_info_type.Polygon[0] == poly_type
    assert geo_info_type.GeoInfo == cphd.GeoInfo[0].GeoInfo

    geo_info_type1 = GeoInfo.GeoInfoType(name="targets", GeoInfo=geo_info_type)
    assert geo_info_type1.GeoInfo[0] == geo_info_type

    # List and Tuple of GeoInfo
    assert len(geo_info_type.GeoInfo) == 1
    geo_info_type = GeoInfo.GeoInfoType(
        name="targets", GeoInfo=[cphd.GeoInfo[0].GeoInfo[0], cphd.GeoInfo[0].GeoInfo[0]]
    )
    assert len(geo_info_type.GeoInfo) == 2

    assert len(geo_info_type1.GeoInfo) == 1
    geo_info_type1 = GeoInfo.GeoInfoType(
        name="targets", GeoInfo=(cphd.GeoInfo[0].GeoInfo[0], cphd.GeoInfo[0].GeoInfo[0])
    )
    assert len(geo_info_type1.GeoInfo) == 2

    with pytest.raises(ValueError, match="GeoInfo got unexpected type"):
        geo_info_type = GeoInfo.GeoInfoType(
            name="targets", GeoInfo=(cphd.GeoInfo[0].GeoInfo[0].to_dict)
        )

    assert len(geo_info_type.getGeoInfo("target0")) == 2

    # Add GeoInfo to an empty instance
    geo_info_type1 = GeoInfo.GeoInfoType(name="test")
    geo_dict = geo_info_type.GeoInfo[0].to_dict()
    geo_info_type1.addGeoInfo(geo_dict)

    with pytest.raises(
        TypeError, match="Trying to set GeoInfo element with unexpected type"
    ):
        geo_info_type1.addGeoInfo(3.0)

    # Taken from cphd.GeoInfo[0].GeoInfo[0]
    node_str = """
        <GeoInfo name="target0">
            <Desc name="ecef">[6378137.0, -202.78989383631944, 289.6139826697846]</Desc>
            <Point>
                <Lat>1.23</Lat>
                <Lon>2.34</Lon>
            </Point>
        </GeoInfo>
    """
    geo_info_type = GeoInfo.GeoInfoType(name="test", GeoInfo=cphd.GeoInfo[0].GeoInfo[0])
    node, ns = parse_xml_from_string(node_str)
    geo_info_type1 = geo_info_type.from_node(node, ns)
    assert (
        geo_info_type1.Descriptions["ecef"]
        == geo_info_type.GeoInfo[0].Descriptions["ecef"]
    )
    assert np.all(
        geo_info_type1.Point[0].get_array()
        == geo_info_type.GeoInfo[0].Point[0].get_array()
    )

    this_node = geo_info_type.to_node(doc=geo_info_doc, tag="GeoInfoType")
    assert this_node.tag == "GeoInfoType"

    geo_info_dict = geo_info_type.to_dict()
    assert np.all(
        geo_info_dict["GeoInfo"][0]["Descriptions"]["ecef"]
        == cphd.GeoInfo[0].GeoInfo[0].Descriptions["ecef"]
    )
