import pytest
from sarpy.io.product.sidd2_elements.Compression import J2KSubtype, J2KType, CompressionType
import xml.etree.ElementTree as ET
import numpy
import re

def test_j2ksubtype_init():
    layer_info = [0.0, 1.0, 2.0]
    j2ksubtype = J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info)


def test_j2ksubtype_init_kwargs():
    layer_info = [0.0, 1.0, 2.0]
    ns_value = "test_namespace_value"
    ns_key = "test_namespace_key"

    j2ksubtype = J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info, _xml_ns=ns_value, _xml_ns_key=ns_key)
    
    # checks if the xml ns value and keys are assigned correctly
    assert j2ksubtype._xml_ns == ns_value
    assert j2ksubtype._xml_ns_key == ns_key

# this function is used as a helper to create an XML that mimic's the structure of LayerInfo to be used for testing
def create_layerinfo_xml_helper(num_layers, bitrates):
    root = ET.Element("LayerInfo", numLayers=str(num_layers))
    for rate in bitrates:
        bitrate_element = ET.SubElement(root, "Bitrate")
        value_element = ET.SubElement(bitrate_element, "value")
        value_element.text = str(rate)
    return root

def test_j2ksubtype_set_layer_info_type_element():
    bitrates = [1.0, 2.0, 3.0]
    xml_element = create_layerinfo_xml_helper(3, bitrates)

    layer_info = [0.0, 1.0, 2.0]
    j2ksubtype = J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info)

    j2ksubtype.setLayerInfoType(xml_element)

    assert numpy.array_equal(j2ksubtype.LayerInfo, bitrates)

def test_j2ksubtype_set_layer_info_no_layer_info_passed():
    bitrates = [1.0, 2.0, 3.0]
    xml_element = create_layerinfo_xml_helper(3, bitrates)

    j2ksubtype = J2KSubtype(NumWaveletLevels=3, NumBands=5)

    j2ksubtype.setLayerInfoType(xml_element)

    assert numpy.array_equal(j2ksubtype.LayerInfo, bitrates)

def test_j2ksubtype_init_failure_layer_info():
    # makes sure that an error is thrown if LayerInfo is neither an ElementTree, list, tuple, ndarray, or None
    layer_info = "abc"
    with pytest.raises(TypeError,
                       match=re.escape("Invalid input type for LayerInfo: <class 'str'>. Must be an ElementTree, list, tuple, ndarray, or None.")):
        J2KSubtype(NumWaveletLevels=3, NumBands=7, LayerInfo=layer_info)

@pytest.fixture()
def setup_j2ksubtype():
    layer_info = [0.0, 1.0, 2.0]
    yield J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info)

def test_j2ktype_init(setup_j2ksubtype):
    layer_info = [0.0, 1.0, 2.0]
    J2KType(setup_j2ksubtype)

@pytest.fixture()
def setup_j2ktype(setup_j2ksubtype):
    yield J2KType(setup_j2ksubtype)

def test_j2ktype_init_w_parsed(setup_j2ksubtype):
    layer_info = [0.0, 1.0, 2.0]
    original = setup_j2ksubtype
    parsed = J2KSubtype(NumWaveletLevels=2, NumBands=4, LayerInfo=layer_info)
    J2KType(original, parsed)

def test_j2ktype_init_kwargs(setup_j2ksubtype):
    layer_info = [0.0, 1.0, 2.0]
    ns_value = "test_namespace_value"
    ns_key = "test_namespace_key"

    j2ksubtype = setup_j2ksubtype
    j2ktype = J2KType(j2ksubtype,_xml_ns=ns_value, _xml_ns_key=ns_key)

    # checks if the xml ns value and keys are assigned correctly
    assert j2ktype._xml_ns == ns_value
    assert j2ktype._xml_ns_key == ns_key

def test_compressiontype_init(setup_j2ktype):
    j2ktype = setup_j2ktype
    CompressionType(j2ktype)

def test_compressiontype_init_kwargs(setup_j2ktype):
    j2ktype = setup_j2ktype
    ns_value = "test_namespace_value"
    ns_key = "test_namespace_key"

    compressiontype = CompressionType(j2ktype, _xml_ns=ns_value, _xml_ns_key=ns_key)
    
    # checks if the xml ns value and keys are assigned correctly
    assert compressiontype._xml_ns == ns_value
    assert compressiontype._xml_ns_key == ns_key