import pytest
from sarpy.io.product.sidd2_elements.Compression import J2KSubtype, J2KType, CompressionType
import xml.etree.ElementTree as ET
import numpy
import re

# started with 56% coverage

def test_j2ksubtype_init():
    layer_info = [0.0, 1.0, 2.0]
    j2ksubtype = J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info)

def test_j2ksubtype_init_no_parameters():
    with pytest.raises(ValueError, 
                        match = re.escape("Attribute NumWaveletLevels of class J2KSubtype cannot be assigned None.")):
        J2KSubtype()

# DEFAULT_STRICT value being used in the J2KSubtype and Type class definitions was set to False, so errors don't get raised for incorrect typing
# I replaced DEFAULT_STRICT with True in the class definitions because otherwise the test_j2ksubtype_init_no_parameters test would not throw any errors
# despite not being passed any arguments

def test_j2ksubtype_init_failure_num_wavelet_level():
    layer_info = [0.0, 1.0, 2.0]
    j2k = J2KSubtype(NumWaveletLevels="abc", NumBands=5, LayerInfo=layer_info)
    # we want some kind of error to be thrown her because NumWaveletLevels should not accept a string

def test_j2ksubtype_init_kwargs():
    layer_info = [0.0, 1.0, 2.0]
    ns_value = "test_namespace_value"
    ns_key = "test_namespace_key"

    j2ksubtype = J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info, _xml_ns=ns_value, _xml_ns_key=ns_key)
    
    # checks if the xml ns value and keys are assigned correctly
    assert j2ksubtype._xml_ns == ns_value
    assert j2ksubtype._xml_ns_key == ns_key

def create_layerinfo_xml_helper(num_layers, bitrates):
    root = ET.Element("LayerInfo", numLayers=str(num_layers))
    for rate in bitrates:
        bitrate_element = ET.SubElement(root, "Bitrate")
        value_element = ET.SubElement(bitrate_element, "value")
        value_element.text = str(rate)
    return root

def test_j2ksubtype_set_layer_info_type():
    bitrates = [1.0, 2.0, 3.0]
    xml_element = create_layerinfo_xml_helper(3, bitrates)

    layer_info = [0.0, 1.0, 2.0]
    j2ksubtype = J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info)

    j2ksubtype.setLayerInfoType(xml_element)

    assert numpy.array_equal(j2ksubtype.LayerInfo, bitrates)

@pytest.fixture()
def setup_j2ksubtype():
    layer_info = [0.0, 1.0, 2.0]
    yield J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info)

def test_j2ktype_init():
    layer_info = [0.0, 1.0, 2.0]
    original = J2KSubtype(NumWaveletLevels=3, NumBands=5, LayerInfo=layer_info)

    j2ktype = J2KType(original)

@pytest.fixture()
def setup_j2ktype(setup_j2ksubtype):
    yield J2KType(setup_j2ksubtype)

def test_j2ktype_init_w_parsed(setup_j2ksubtype):
    layer_info = [0.0, 1.0, 2.0]
    original = setup_j2ksubtype
    parsed = J2KSubtype(NumWaveletLevels=2, NumBands=4, LayerInfo=layer_info)
    j2ktype = J2KType(original, parsed)

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
    compressiontype = CompressionType(j2ktype)

def test_compressiontype_init_kwargs(setup_j2ktype):
    j2ktype = setup_j2ktype
    ns_value = "test_namespace_value"
    ns_key = "test_namespace_key"

    compressiontype = CompressionType(j2ktype, _xml_ns=ns_value, _xml_ns_key=ns_key)
    # checks if the xml ns value and keys are assigned correctly
    assert compressiontype._xml_ns == ns_value
    assert compressiontype._xml_ns_key == ns_key