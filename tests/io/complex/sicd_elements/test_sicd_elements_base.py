#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import base
from sarpy.io.complex.sicd_elements import blocks


@pytest.fixture
def this_doc():
    root = ET.Element("ValidData")
    root.attrib["size"] = "4"

    doc = ET.ElementTree(root)
    return doc


@pytest.fixture
def cp_array_descriptor():
    return base.SerializableCPArrayDescriptor(
        name="ImageCorners",
        child_type=blocks.LatLonCornerStringType,
        tag_dict={
            "ValidData": {"array": True, "child_tag": "Vertex"},
            "ImageCorners": {"array": True, "child_tag": "ICP"},
        },
        required=("EarthModel", "SCP", "ImageCorners"),
        strict=True,
        docstring="The geographic image corner points array.",
    )


@pytest.fixture
def cp_array():
    return base.SerializableCPArray(
        coords=[[1, 2], [3, 4], [5, 6], [7, 8]],
        name="ValidData",
        child_tag="Vertex",
        child_type=blocks.LatLonCornerType,
        _xml_ns={
            "ValidData": {"array": True, "child_tag": "Vertex"},
            "ImageCorners": {"array": True, "child_tag": "ICP"},
        },
        _xml_ns_key="Vertex",
    )


@pytest.fixture
def cp_array_string():
    return base.SerializableCPArray(
        coords=[[11, 21], [31, 41], [51, 61], [71, 81]],
        name="ImageCorners",
        child_tag="ICP",
        child_type=blocks.LatLonCornerStringType,
        _xml_ns={
            "ValidData": {"array": True, "child_tag": "Vertex"},
            "ImageCorners": {"array": True, "child_tag": "ICP"},
        },
        _xml_ns_key="ICP",
    )


def test_base_serializable_cp_array_descriptor(cp_array_descriptor):
    # Smoke test (all valid)
    assert cp_array_descriptor.name == "ImageCorners"
    assert cp_array_descriptor.child_type == blocks.LatLonCornerStringType
    assert cp_array_descriptor.array is True
    assert cp_array_descriptor.child_tag == "ICP"
    assert cp_array_descriptor.required is True
    assert cp_array_descriptor.strict is True

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Attribute ImageCorners is populated in the `_collection_tags` dictionary without `array`=True. This is inconsistent with using SerializableCPArrayDescriptor"
        ),
    ):
        base.SerializableCPArrayDescriptor(
            name="ImageCorners",
            child_type=blocks.LatLonCornerStringType,
            tag_dict={
                "ValidData": {"array": True, "child_tag": "Vertex"},
                "ImageCorners": {"array": False, "child_tag": "ICP"},
            },
            required=("EarthModel", "SCP", "ImageCorners"),
            strict=True,
            docstring="The geographic image corner points array.",
        )


def test_base_serializable_cp_array_descriptor_setter(cp_array):
    class TestClass:
        # Define the property using the descriptor
        coords = base.SerializableCPArrayDescriptor(
            name="ValidData",
            child_type=blocks.LatLonCornerType,
            tag_dict={"ValidData": {"array": True, "child_tag": "Vertex"}},
            required=(),
            strict=False,
        )

    test_instance = TestClass()
    test_instance.coords = [[10, 20], [30, 40], [50, 60], [70, 80]]
    assert len(test_instance.coords) == 4
    assert np.all(test_instance.coords[1].get_array() == [30, 40])
    test_instance.coords = cp_array
    assert len(test_instance.coords) == 4
    assert np.all(test_instance.coords[1].get_array() == [3, 4])
    test_instance.coords = None
    assert test_instance.coords is None


def test_serializable_cp_array(cp_array, cp_array_string, this_doc):
    # Smoke test
    assert cp_array._name == "ValidData"
    assert cp_array._child_type == blocks.LatLonCornerType
    assert cp_array._child_tag == "Vertex"
    assert len(cp_array._array) == 4

    # Check class properties
    assert np.all(cp_array.FRFC == cp_array._array[0].get_array())
    assert np.all(cp_array.FRLC == cp_array._array[1].get_array())
    assert np.all(cp_array.LRLC == cp_array._array[2].get_array())
    assert np.all(cp_array.LRFC == cp_array._array[3].get_array())

    empty_cp_array = base.SerializableCPArray(
        name="EmptyArray", child_tag="FakeTag", child_type=blocks.LatLonCornerStringType
    )
    assert empty_cp_array.FRFC is None
    assert empty_cp_array.LRFC is None
    assert empty_cp_array.FRLC is None
    assert empty_cp_array.LRLC is None

    # verify _check_indices code
    cp_array.set_array([[10, 20], [30, 40], [50, 60], [70, 80]])
    cp_array_string.set_array([[10, 20], [30, 40], [50, 60], [70, 80]])

    # to_node checks
    this_node = cp_array.to_node(doc=this_doc, tag="ValidData", ns_key="test")
    assert len(this_node.findall("test:Vertex")) == 4

    this_node = cp_array.to_node(doc=this_doc, tag="ValidData")
    assert len(this_node.findall("Vertex")) == 4
    for i, node in enumerate(this_node.findall("Vertex")):
        assert int(node.attrib["index"]) == i + 1
        assert float(node.find("Lat").text) == cp_array._array[i][0]
        assert float(node.find("Lon").text) == cp_array._array[i][1]

    assert empty_cp_array.to_node(doc=this_doc, tag="ValidData") is None

