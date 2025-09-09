#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from collections import OrderedDict
import copy
import re
from typing import Tuple
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import blocks
from sarpy.io.xml.base import parse_xml_from_string


LATLONHAE = [33.483888, -112.073706, 100.0]
ROWCOL = [1, 2]


@pytest.fixture
def poly1d_doc(sicd):
    root = Element("Poly1DType")
    root.attrib["order1"] = str(sicd.Position.ARPPoly.X.order1)

    coefs_node = SubElement(root, "Coefs")

    coef_values = sicd.Position.ARPPoly.X.Coefs
    for i, coef_value in enumerate(coef_values):
        coef_node = SubElement(coefs_node, "Coef")
        coef_node.attrib["exponent1"] = str(i)
        coef_node.text = str(coef_value)

    doc = ElementTree(root)
    return doc


@pytest.fixture
def poly2d_doc(sicd):
    root = Element("Poly2DType")
    root.attrib["order1"] = str(sicd.Radiometric.RCSSFPoly.order1)
    root.attrib["order2"] = str(sicd.Radiometric.RCSSFPoly.order2)

    coefs_node = SubElement(root, "Coefs")

    coef_values = sicd.Radiometric.RCSSFPoly.Coefs
    for i in np.arange(len(coef_values)):
        for j, coef_value in enumerate(coef_values[i]):
            coef_node = SubElement(coefs_node, "Coef")
            coef_node.attrib["exponent1"] = str(i)
            coef_node.attrib["exponent2"] = str(j)
            coef_node.text = str(coef_value)

    doc = ElementTree(root)
    return doc


def test_blocks_xyztype(kwargs):
    # Smoke test
    xyz_type = blocks.XYZType(X=1.0, Y=2.0, Z=3.0)
    assert xyz_type.X == 1.0
    assert xyz_type.Y == 2.0
    assert xyz_type.Z == 3.0
    assert not hasattr(xyz_type, "_xml_ns")
    assert not hasattr(xyz_type, "_xml_ns_key")

    # Init with kwargs
    xyz_type = blocks.XYZType(X=1.0, Y=2.0, Z=3.0, **kwargs)
    assert xyz_type._xml_ns == kwargs["_xml_ns"]
    assert xyz_type._xml_ns_key == kwargs["_xml_ns_key"]

    # get_array
    expected_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert np.array_equal(xyz_type.get_array(), expected_array)

    # from_array
    xyz_type = blocks.XYZType.from_array(None)
    assert xyz_type is None

    array = [1.0, 2.0, 3.0]
    xyz_type = blocks.XYZType.from_array(array)
    assert xyz_type.X == 1.0
    assert xyz_type.Y == 2.0
    assert xyz_type.Z == 3.0

    # from_array errors
    array = [1.0, 2.0]
    with pytest.raises(
        ValueError,
        match=re.escape(f"Expected array to be of length 3, and received `{array}`"),
    ):
        blocks.XYZType.from_array(array)

    array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        blocks.XYZType.from_array(array)


@pytest.mark.parametrize(
    "array, class_to_test",
    [
        (LATLONHAE[0:2], blocks.LatLonType),
        (LATLONHAE[0:2], blocks.LatLonRestrictionType),
    ],
)
def test_blocks_latlon_classes(array, class_to_test, kwargs, tol):
    # Smoke test
    class_instance = class_to_test(Lat=array[0], Lon=array[1])
    assert class_instance.Lat == pytest.approx(array[0], abs=tol)
    assert class_instance.Lon == pytest.approx(array[1], abs=tol)
    assert not hasattr(class_instance, "_xml_ns")
    assert not hasattr(class_instance, "_xml_ns_key")

    # Init with kwargs
    class_instance = class_to_test(Lat=array[0], Lon=array[1], **kwargs)
    assert class_instance._xml_ns == kwargs["_xml_ns"]
    assert class_instance._xml_ns_key == kwargs["_xml_ns_key"]

    # from_array
    class_instance = class_to_test.from_array(None)
    assert class_instance is None

    class_instance = class_to_test.from_array(array)
    assert isinstance(class_instance, class_to_test)
    assert class_instance.Lat == pytest.approx(array[0], abs=tol)
    assert class_instance.Lon == pytest.approx(array[1], abs=tol)

    # from_array errors
    bad_array = array[0:1]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected array to be of length 2, and received `{bad_array}`"
        ),
    ):
        class_to_test.from_array(bad_array)

    bad_array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        class_to_test.from_array(bad_array)


def test_blocks_latlontype_getarray(tol):
    latlon_type = blocks.LatLonType(Lat=LATLONHAE[0], Lon=LATLONHAE[1])
    result = latlon_type.get_array(dtype=np.float64, order="LAT")
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.tolist() == pytest.approx(LATLONHAE[0:2], abs=tol)

    result = latlon_type.get_array(dtype=np.float64, order="LON")
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.tolist() == pytest.approx([LATLONHAE[1], LATLONHAE[0]], abs=tol)


def test_blocks_latlontype_dmsformat():
    latlon_type = blocks.LatLonType(Lat=LATLONHAE[0], Lon=LATLONHAE[1])
    result = latlon_type.dms_format(frac_secs=True)
    assert isinstance(result, Tuple)
    assert isinstance(result[0], Tuple)
    assert isinstance(result[1], Tuple)
    assert len(result[0]) == 4
    assert len(result[1]) == 4

    result = latlon_type.dms_format(frac_secs=False)
    assert isinstance(result, Tuple)
    assert isinstance(result[0], Tuple)
    assert isinstance(result[1], Tuple)
    assert len(result[0]) == 4
    assert len(result[1]) == 4


def test_lltype_restriction(tol):
    # Lat/Lon outside restricted range
    latlon_restricted_type = blocks.LatLonRestrictionType(Lat=100, Lon=190)
    assert latlon_restricted_type.Lat == pytest.approx(-80.0, abs=tol)
    assert latlon_restricted_type.Lon == pytest.approx(-170.0, abs=tol)


@pytest.mark.parametrize(
    "array, index, class_to_test",
    [
        (LATLONHAE[0:2], 1, blocks.LatLonArrayElementType),
        (LATLONHAE[0:2], 1, blocks.LatLonCornerType),
        (LATLONHAE[0:2], "3:LRLC", blocks.LatLonCornerStringType),
    ],
)
def test_blocks_latlon_classes_with_index(array, index, class_to_test, kwargs, tol):
    # Smoke test
    class_instance = class_to_test(Lat=array[0], Lon=array[1], index=index)
    assert class_instance.Lat == pytest.approx(array[0], abs=tol)
    assert class_instance.Lon == pytest.approx(array[1], abs=tol)
    assert class_instance.index == index
    assert not hasattr(class_instance, "_xml_ns")
    assert not hasattr(class_instance, "_xml_ns_key")

    # Init with kwargs
    class_instance = class_to_test(Lat=array[0], Lon=array[1], index=index, **kwargs)
    assert class_instance._xml_ns == kwargs["_xml_ns"]
    assert class_instance._xml_ns_key == kwargs["_xml_ns_key"]

    # from_array
    class_instance = class_to_test.from_array(None)
    assert class_instance is None

    class_instance = class_to_test.from_array(array)
    assert isinstance(class_instance, class_to_test)
    assert class_instance.Lat == pytest.approx(array[0], abs=tol)
    assert class_instance.Lon == pytest.approx(array[1], abs=tol)

    # from_array errors
    bad_array = array[0:1]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected array to be of length 2, and received `{bad_array}`"
        ),
    ):
        class_to_test.from_array(bad_array)

    bad_array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        class_to_test.from_array(bad_array)


@pytest.mark.parametrize(
    "array, class_to_test",
    [
        (LATLONHAE, blocks.LatLonHAEType),
        (LATLONHAE, blocks.LatLonHAERestrictionType),
    ],
)
def test_blocks_latlonhae_classes(array, class_to_test, kwargs, tol):
    # Smoke test
    class_instance = class_to_test(Lat=array[0], Lon=array[1], HAE=array[2])
    assert class_instance.Lat == pytest.approx(array[0], abs=tol)
    assert class_instance.Lon == pytest.approx(array[1], abs=tol)
    assert class_instance.HAE == pytest.approx(array[2], abs=tol)
    assert not hasattr(class_instance, "_xml_ns")
    assert not hasattr(class_instance, "_xml_ns_key")

    # Init with kwargs
    class_instance = class_to_test(Lat=array[0], Lon=array[1], HAE=array[2], **kwargs)
    assert class_instance._xml_ns == kwargs["_xml_ns"]
    assert class_instance._xml_ns_key == kwargs["_xml_ns_key"]

    # get_array
    result = class_instance.get_array(dtype=np.float64, order="LAT")
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.tolist() == pytest.approx(array, abs=tol)

    result = class_instance.get_array(dtype=np.float64, order="LON")
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.tolist() == pytest.approx([array[1], array[0], array[2]], abs=tol)

    # from_array
    class_instance = class_to_test.from_array(None)
    assert class_instance is None

    class_instance = class_to_test.from_array(array)
    assert isinstance(class_instance, class_to_test)
    assert class_instance.Lat == pytest.approx(array[0], abs=tol)
    assert class_instance.Lon == pytest.approx(array[1], abs=tol)
    assert class_instance.HAE == pytest.approx(array[2], abs=tol)

    # from_array errors
    bad_array = array[0:2]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected array to be of length 3, and received `{bad_array}`"
        ),
    ):
        class_to_test.from_array(bad_array)

    bad_array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        class_to_test.from_array(bad_array)


@pytest.mark.parametrize(
    "array, index, class_to_test",
    [
        (LATLONHAE, 1, blocks.LatLonHAECornerRestrictionType),
        (LATLONHAE, "3:LRLC", blocks.LatLonHAECornerStringType),
    ],
)
def test_blocks_latlonhae_classes_with_index(array, index, class_to_test, kwargs, tol):
    # Smoke test
    class_instance = class_to_test(
        Lat=array[0], Lon=array[1], HAE=array[2], index=index
    )
    assert class_instance.Lat == pytest.approx(array[0], abs=tol)
    assert class_instance.Lon == pytest.approx(array[1], abs=tol)
    assert class_instance.HAE == pytest.approx(array[2], abs=tol)
    assert class_instance.index == index
    assert not hasattr(class_instance, "_xml_ns")
    assert not hasattr(class_instance, "_xml_ns_key")

    # Init with kwargs
    class_instance = class_to_test(
        Lat=array[0], Lon=array[1], HAE=array[2], index=index, **kwargs
    )
    assert class_instance._xml_ns == kwargs["_xml_ns"]
    assert class_instance._xml_ns_key == kwargs["_xml_ns_key"]

    # from_array
    class_instance = class_to_test.from_array(None)
    assert class_instance is None

    class_instance = class_to_test.from_array(array)
    assert isinstance(class_instance, class_to_test)
    assert class_instance.Lat == pytest.approx(array[0], abs=tol)
    assert class_instance.Lon == pytest.approx(array[1], abs=tol)
    assert class_instance.HAE == pytest.approx(array[2], abs=tol)

    # from_array errors
    bad_array = array[0:2]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected array to be of length 3, and received `{bad_array}`"
        ),
    ):
        class_to_test.from_array(bad_array)

    bad_array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        class_to_test.from_array(bad_array)


def test_blocks_rowcoltype(kwargs, tol):
    # Smoke test
    row_col = blocks.RowColType(Row=ROWCOL[0], Col=ROWCOL[1])
    assert row_col.Row == pytest.approx(ROWCOL[0], abs=tol)
    assert row_col.Col == pytest.approx(ROWCOL[1], abs=tol)
    assert not hasattr(row_col, "_xml_ns")
    assert not hasattr(row_col, "_xml_ns_key")

    # Init with kwargs
    row_col = blocks.RowColType(Row=ROWCOL[0], Col=ROWCOL[1], **kwargs)
    assert row_col._xml_ns == kwargs["_xml_ns"]
    assert row_col._xml_ns_key == kwargs["_xml_ns_key"]

    # get_array
    result = row_col.get_array(dtype=np.float64)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.tolist() == pytest.approx(ROWCOL, abs=tol)

    # from_array
    row_col = blocks.RowColType.from_array(None)
    assert row_col is None

    row_col = blocks.RowColType.from_array(ROWCOL)
    assert isinstance(row_col, blocks.RowColType)
    assert row_col.Row == pytest.approx(ROWCOL[0], abs=tol)
    assert row_col.Col == pytest.approx(ROWCOL[1], abs=tol)

    # from_array errors
    bad_array = ROWCOL[0:1]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected array to be of length 2, and received `{bad_array}`"
        ),
    ):
        blocks.RowColType.from_array(bad_array)

    bad_array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        blocks.RowColType.from_array(bad_array)


def test_blocks_rowcolarrayelement(kwargs, tol):
    # Smoke test
    row_col_arr = blocks.RowColArrayElement(Row=ROWCOL[0], Col=ROWCOL[1], index=1)
    assert row_col_arr.Row == pytest.approx(ROWCOL[0], abs=tol)
    assert row_col_arr.Col == pytest.approx(ROWCOL[1], abs=tol)
    assert row_col_arr.index == 1
    assert not hasattr(row_col_arr, "_xml_ns")
    assert not hasattr(row_col_arr, "_xml_ns_key")

    # Init with kwargs
    row_col_arr = blocks.RowColArrayElement(
        Row=ROWCOL[0], Col=ROWCOL[1], index=1, **kwargs
    )
    assert row_col_arr._xml_ns == kwargs["_xml_ns"]
    assert row_col_arr._xml_ns_key == kwargs["_xml_ns_key"]

    # from_array
    row_col_arr = blocks.RowColArrayElement.from_array(None)
    assert row_col_arr is None

    row_col_arr = blocks.RowColArrayElement.from_array(ROWCOL)
    assert isinstance(row_col_arr, blocks.RowColArrayElement)
    assert row_col_arr.Row == pytest.approx(ROWCOL[0], abs=tol)
    assert row_col_arr.Col == pytest.approx(ROWCOL[1], abs=tol)

    # from_array errors
    bad_array = ROWCOL[0:1]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected array to be of length 2, and received `{bad_array}`"
        ),
    ):
        blocks.RowColArrayElement.from_array(bad_array)

    bad_array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        blocks.RowColArrayElement.from_array(bad_array)


def test_blocks_poly1dtype(sicd, poly1d_doc, kwargs):
    # Smoke test
    poly = blocks.Poly1DType(Coefs=sicd.Position.ARPPoly.X.Coefs)
    assert poly.order1 == 5
    assert not hasattr(poly, "_xml_ns")
    assert not hasattr(poly, "_xml_ns_key")

    # Init with kwargs
    poly = blocks.Poly1DType(Coefs=sicd.Position.ARPPoly.X.Coefs, **kwargs)
    assert poly._xml_ns == kwargs["_xml_ns"]
    assert poly._xml_ns_key == kwargs["_xml_ns_key"]

    assert np.all(poly.Coefs == sicd.Position.ARPPoly.X.Coefs)

    # Setter
    poly.Coefs = np.asarray(sicd.Position.ARPPoly.Y.Coefs, dtype=np.float32)
    assert poly.Coefs.dtype.name == "float64"

    poly.Coefs = sicd.Position.ARPPoly.Y.Coefs

    poly.Coefs = sicd.Position.ARPPoly.Y.Coefs.tolist()

    # Setter errors
    with pytest.raises(
        ValueError,
        match="The coefficient array for a Poly1DType instance must be defined",
    ):
        poly.Coefs = None

    with pytest.raises(
        ValueError, match="Coefs for class Poly1D must be a list or numpy.ndarray"
    ):
        poly.Coefs = {"Coefs": 1}

    with pytest.raises(
        ValueError, match="Coefs for class Poly1D must be one-dimensional"
    ):
        poly.Coefs = [sicd.Position.ARPPoly.X.Coefs, sicd.Position.ARPPoly.Y.Coefs]

    assert poly(0) == poly.Coefs[0]
    assert poly[0] == poly.Coefs[0]

    poly1 = copy.copy(poly)
    poly1[5] = 2e-11
    assert poly1.Coefs[5] == 2e-11

    # Poly derivative
    coefs = poly.derivative(der_order=1)
    assert isinstance(coefs, np.ndarray)
    assert coefs[0] == poly.Coefs[1]

    poly1 = poly.derivative(der_order=1, return_poly=True)
    assert isinstance(poly1, blocks.Poly1DType)
    assert poly1.Coefs[0] == poly.Coefs[1]

    # Poly derivative eval
    assert poly.derivative_eval(0) == poly.Coefs[1]

    shifted_coefs = poly.shift(1.1, 2.0)
    assert isinstance(shifted_coefs, np.ndarray)
    shifted_poly = poly.shift(1.1, 2.0, return_poly=True)
    assert isinstance(shifted_poly, blocks.Poly1DType)

    # from_array
    poly = blocks.Poly1DType.from_array(None)
    assert poly is None

    poly = blocks.Poly1DType.from_array(sicd.Position.ARPPoly.X.Coefs)
    assert isinstance(poly, blocks.Poly1DType)
    assert np.all(poly.Coefs == sicd.Position.ARPPoly.X.Coefs)

    # get_array
    coefs = poly.get_array()
    assert np.all(coefs == sicd.Position.ARPPoly.X.Coefs)

    # from_node and to_node
    node_str = """
        <Poly order1="5">
            <Coef exponent1="0">7228127.9124448663</Coef>
            <Coef exponent1="1">352.53242998756502</Coef>
            <Coef exponent1="2">-3.5891719134975157</Coef>
            <Coef exponent1="3">-5.7694198643316104e-05</Coef>
            <Coef exponent1="4">2.7699968593303768e-07</Coef>
            <Coef exponent1="5">2.1592636134572539e-09</Coef>
        </Poly>
    """

    poly = blocks.Poly1DType.from_array(sicd.Position.ARPPoly.X.Coefs)
    node, ns = parse_xml_from_string(node_str)
    poly1 = poly.from_node(node, ns)
    assert isinstance(poly1, blocks.Poly1DType)
    assert np.all(poly.Coefs == poly1.Coefs)

    this_node = poly.to_node(doc=poly1d_doc, tag="Poly1DType")
    assert this_node.tag == "Poly1DType"
    assert this_node.attrib["order1"] == str(len(poly.Coefs) - 1)
    assert len(this_node.findall("Coef")) == len(poly.Coefs)
    for i, node in enumerate(this_node.findall("Coef")):
        assert int(node.attrib["exponent1"]) == i
        assert float(node.text) == poly.Coefs[i]

    coef_dict = poly.to_dict()
    assert isinstance(coef_dict, OrderedDict)

    # minimize_order
    poly.minimize_order()
    assert len(poly.Coefs) == 6

    poly[5] = 0.0
    poly.minimize_order()
    assert len(poly.Coefs) == 5

    poly[3] = 0.0
    poly.minimize_order()
    assert len(poly.Coefs) == 5

    poly[1:6] = 0.0
    poly.minimize_order()
    assert len(poly.Coefs) == 1

    poly[0] = 0.0
    poly.minimize_order()
    assert len(poly.Coefs) == 1

def test_blocks_poly1dtype__eq__(sicd, poly1d_doc, kwargs):
    # Smoke test
    poly01 = blocks.Poly1DType(Coefs=sicd.Position.ARPPoly.X.Coefs)
    poly02 = blocks.Poly1DType(Coefs=sicd.Position.ARPPoly.X.Coefs)
    assert(poly01 == poly02)


def test_blocks_poly2dtype(sicd, poly2d_doc, kwargs):
    # Smoke test
    poly = blocks.Poly2DType(Coefs=sicd.Radiometric.RCSSFPoly.Coefs)
    assert poly.order1 == 5
    assert poly.order2 == 6
    assert not hasattr(poly, "_xml_ns")
    assert not hasattr(poly, "_xml_ns_key")

    # Init with kwargs
    poly = blocks.Poly2DType(Coefs=sicd.Radiometric.RCSSFPoly.Coefs, **kwargs)
    assert poly._xml_ns == kwargs["_xml_ns"]
    assert poly._xml_ns_key == kwargs["_xml_ns_key"]

    assert np.all(poly.Coefs == sicd.Radiometric.RCSSFPoly.Coefs)

    # Setter
    poly.Coefs = sicd.Radiometric.RCSSFPoly.Coefs.tolist()

    poly.Coefs = np.asarray(sicd.Radiometric.RCSSFPoly.Coefs, np.float32)
    assert poly.Coefs.dtype.name == "float64"

    poly.Coefs = sicd.Radiometric.RCSSFPoly.Coefs
    assert poly(0, 0) == poly.Coefs[0][0]
    assert poly[0, 0] == poly.Coefs[0][0]

    # Setter errors
    with pytest.raises(
        ValueError,
        match="The coefficient array for a Poly2DType instance must be defined",
    ):
        poly.Coefs = None

    with pytest.raises(
        ValueError, match="Coefs for class Poly2D must be a list or numpy.ndarray"
    ):
        poly.Coefs = {"Coefs": 1}

    with pytest.raises(
        ValueError, match="Coefs for class Poly2D must be two-dimensional"
    ):
        poly.Coefs = sicd.Radiometric.RCSSFPoly.Coefs[0]

    poly1 = copy.copy(poly)
    poly1[0, 5] = 2e-11
    assert poly1.Coefs[0][5] == 2e-11

    # shift smoke test
    shifted_coefs = poly.shift(1.1, 2.0, 2.1, 3.0)
    assert isinstance(shifted_coefs, np.ndarray)
    shifted_poly = poly.shift(1.1, 2.0, 2.1, 3.0, return_poly=True)
    assert isinstance(shifted_poly, blocks.Poly2DType)

    # from_array
    poly = blocks.Poly2DType.from_array(None)
    assert poly is None

    poly = blocks.Poly2DType.from_array(sicd.Radiometric.RCSSFPoly.Coefs)
    assert isinstance(poly, blocks.Poly2DType)
    assert np.all(poly.Coefs == sicd.Radiometric.RCSSFPoly.Coefs)

    # get_array
    coefs = poly.get_array()
    assert np.all(coefs == sicd.Radiometric.RCSSFPoly.Coefs)

    # from_node and to_node
    node_str = """
        <Poly order1="5" order2="6">
            <Coef exponent1="0" exponent2="0">234.567891</Coef>
            <Coef exponent1="0" exponent2="1">0.0123456789</Coef>
            <Coef exponent1="0" exponent2="2">3.45678912e-05</Coef>
            <Coef exponent1="0" exponent2="3">1.23456789e-09</Coef>
            <Coef exponent1="0" exponent2="4">2.34567891e-12</Coef>
            <Coef exponent1="0" exponent2="5">1.23456789e-16</Coef>
            <Coef exponent1="0" exponent2="6">1.23456789e-19</Coef>
            <Coef exponent1="1" exponent2="0">-0.023456789</Coef>
            <Coef exponent1="1" exponent2="1">-5.67891234e-06</Coef>
            <Coef exponent1="1" exponent2="2">-4.56789123e-09</Coef>
            <Coef exponent1="1" exponent2="3">-8.91234567e-13</Coef>
            <Coef exponent1="1" exponent2="4">-4.56789123e-16</Coef>
            <Coef exponent1="1" exponent2="5">-9.12345678e-20</Coef>
            <Coef exponent1="1" exponent2="6">-3.45678912e-23</Coef>
            <Coef exponent1="2" exponent2="0">5.67891234e-05</Coef>
            <Coef exponent1="2" exponent2="1">3.45678912e-09</Coef>
            <Coef exponent1="2" exponent2="2">7.89123456e-12</Coef>
            <Coef exponent1="2" exponent2="3">4.56789123e-16</Coef>
            <Coef exponent1="2" exponent2="4">5.67891234e-19</Coef>
            <Coef exponent1="2" exponent2="5">6.78912345e-23</Coef>
            <Coef exponent1="2" exponent2="6">5.67891234e-26</Coef>
            <Coef exponent1="3" exponent2="0">-6.78912345e-09</Coef>
            <Coef exponent1="3" exponent2="1">-1.23456789e-12</Coef>
            <Coef exponent1="3" exponent2="2">-1.23456789e-15</Coef>
            <Coef exponent1="3" exponent2="3">-7.89123456e-20</Coef>
            <Coef exponent1="3" exponent2="4">-6.78912345e-23</Coef>
            <Coef exponent1="3" exponent2="5">-5.67891234e-26</Coef>
            <Coef exponent1="3" exponent2="6">-1.23456789e-29</Coef>
            <Coef exponent1="4" exponent2="0">7.89123456e-12</Coef>
            <Coef exponent1="4" exponent2="1">5.67891234e-16</Coef>
            <Coef exponent1="4" exponent2="2">1.23456789e-18</Coef>
            <Coef exponent1="4" exponent2="3">1.23456789e-22</Coef>
            <Coef exponent1="4" exponent2="4">1.23456789e-25</Coef>
            <Coef exponent1="4" exponent2="5">1.23456789e-29</Coef>
            <Coef exponent1="4" exponent2="6">6.78912345e-33</Coef>
            <Coef exponent1="5" exponent2="0">-1.23456789e-15</Coef>
            <Coef exponent1="5" exponent2="1">-1.23456789e-19</Coef>
            <Coef exponent1="5" exponent2="2">-1.23456789e-22</Coef>
            <Coef exponent1="5" exponent2="3">-8.91234567e-26</Coef>
            <Coef exponent1="5" exponent2="4">-4.56789123e-29</Coef>
            <Coef exponent1="5" exponent2="5">9.12345678e-33</Coef>
            <Coef exponent1="5" exponent2="6">2.34567891e-36</Coef>
        </Poly>
    """

    poly = blocks.Poly2DType.from_array(sicd.Radiometric.RCSSFPoly.Coefs)
    node, ns = parse_xml_from_string(node_str)
    poly1 = poly.from_node(node, ns)
    assert isinstance(poly1, blocks.Poly2DType)
    assert np.all(poly.Coefs == poly1.Coefs)

    this_node = poly.to_node(doc=poly2d_doc, tag="Poly2DType")
    assert this_node.tag == "Poly2DType"
    assert this_node.attrib["order1"] == str(len(poly.Coefs) - 1)
    assert this_node.attrib["order2"] == str(len(poly.Coefs[0]) - 1)
    assert len(this_node.findall("Coef")) == len(poly.Coefs.flatten())
    for i, node in enumerate(this_node.findall("Coef")):
        assert float(node.text) == poly.Coefs.flatten()[i]

    coef_dict = poly.to_dict()
    assert isinstance(coef_dict, OrderedDict)

    # nothing to do
    poly.minimize_order()
    assert np.shape(poly.Coefs) == (6, 7)

    # last non-zero row index and last non-zero column index not zero
    poly[:, 6] = 0.0
    poly.minimize_order()
    assert np.shape(poly.Coefs) == (6, 6)

    # last non-zero column index is zero
    poly1 = copy.copy(poly)
    poly1[:, 1:6] = 0.0
    poly1.minimize_order()
    assert np.shape(poly1.Coefs) == (6, 1)

    # last non-zero row index is zero
    poly[1:6, :] = 0.0
    poly.minimize_order()
    assert np.shape(poly.Coefs) == (1, 6)

    # both last non-zero row and column index is zero
    poly[0, 1:6] = 0.0
    poly.minimize_order()
    assert np.shape(poly.Coefs) == (1, 1)

    poly[0] = 0.0
    poly.minimize_order()
    assert poly.Coefs[0][0] == 0.0


def test_blocks_xyzpolytype(sicd, kwargs):
    # Smoke test
    poly = blocks.XYZPolyType(
        X=sicd.Position.ARPPoly.X,
        Y=sicd.Position.ARPPoly.Y,
        Z=sicd.Position.ARPPoly.Z
    )
    assert poly.X.order1 == 5
    assert poly.Y.order1 == 5
    assert poly.Z.order1 == 5
    assert not hasattr(poly, "_xml_ns")
    assert not hasattr(poly, "_xml_ns_key")

    # Init with kwargs
    poly = blocks.XYZPolyType(
        X=sicd.Position.ARPPoly.X,
        Y=sicd.Position.ARPPoly.Y,
        Z=sicd.Position.ARPPoly.Z,
        **kwargs,
    )
    assert poly._xml_ns == kwargs["_xml_ns"]
    assert poly._xml_ns_key == kwargs["_xml_ns_key"]

    assert np.array_equal(
        poly(0),
        np.array(
            [
                sicd.Position.ARPPoly.X[0],
                sicd.Position.ARPPoly.Y[0],
                sicd.Position.ARPPoly.Z[0],
            ]
        ),
    )

    # Caller
    poly_eval = poly([0, 1, 2])
    assert len(poly_eval) == 3

    # get_array
    coeff_arr = poly.get_array()
    assert coeff_arr[0] == sicd.Position.ARPPoly.X
    assert coeff_arr[1] == sicd.Position.ARPPoly.Y
    assert coeff_arr[2] == sicd.Position.ARPPoly.Z

    coeff_arr = poly.get_array(dtype=np.float64)
    assert np.array_equal(
        coeff_arr,
        np.array(
            [
                sicd.Position.ARPPoly.X.Coefs,
                sicd.Position.ARPPoly.Y.Coefs,
                sicd.Position.ARPPoly.Z.Coefs,
            ]
        ),
    )

    # from_array
    poly = blocks.XYZPolyType.from_array(None)
    assert poly is None

    array = np.array(
        [
            sicd.Position.ARPPoly.X.Coefs,
            sicd.Position.ARPPoly.Y.Coefs,
            sicd.Position.ARPPoly.Z.Coefs,
        ]
    )
    poly = blocks.XYZPolyType.from_array(array)
    assert isinstance(poly, blocks.XYZPolyType)
    assert np.all(poly.X.Coefs == sicd.Position.ARPPoly.X.Coefs)
    assert np.all(poly.Y.Coefs == sicd.Position.ARPPoly.Y.Coefs)
    assert np.all(poly.Z.Coefs == sicd.Position.ARPPoly.Z.Coefs)

    # from_array errors
    bad_array = array[0:2]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected array to be of length 3, and received `{bad_array}`"
        ),
    ):
        blocks.XYZPolyType.from_array(bad_array)

    bad_array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        blocks.XYZPolyType.from_array(bad_array)

    # poly derivative
    coefs = poly.derivative(der_order=1)
    assert coefs[0][0] == poly.X.Coefs[1]
    assert coefs[1][0] == poly.Y.Coefs[1]
    assert coefs[2][0] == poly.Z.Coefs[1]

    poly1 = poly.derivative(der_order=1, return_poly=True)
    assert isinstance(poly1, blocks.XYZPolyType)
    assert poly1.X.Coefs[0] == poly.X.Coefs[1]
    assert poly1.Y.Coefs[0] == poly.Y.Coefs[1]
    assert poly1.Z.Coefs[0] == poly.Z.Coefs[1]

    # poly derivative_eval
    coefs = poly.derivative_eval(0)
    assert np.array_equal(
        coefs, np.array([poly.X.Coefs[1], poly.Y.Coefs[1], poly.Z.Coefs[1]])
    )

    # poly shift
    shifted_coefs = poly.shift(1.1, 2.0)
    assert isinstance(shifted_coefs[0], np.ndarray)
    assert isinstance(shifted_coefs[1], np.ndarray)
    assert isinstance(shifted_coefs[2], np.ndarray)
    shifted_poly = poly.shift(1.1, 2.0, return_poly=True)
    assert isinstance(shifted_poly, blocks.XYZPolyType)

    # poly minimize_order
    poly.minimize_order()
    assert len(poly.X.Coefs) == 6
    assert len(poly.Y.Coefs) == 6
    assert len(poly.Z.Coefs) == 6

    poly.X[5] = 0.0
    poly.minimize_order()
    assert len(poly.X.Coefs) == 5

    poly.X[3] = 0.0
    poly.minimize_order()
    assert len(poly.X.Coefs) == 5


def test_blocks_xyzpolyattrtype(sicd, kwargs):
    # Smoke test
    poly = blocks.XYZPolyAttributeType(
        X=sicd.Position.ARPPoly.X,
        Y=sicd.Position.ARPPoly.Y,
        Z=sicd.Position.ARPPoly.Z,
        index=1,
    )
    assert poly.X.order1 == 5
    assert poly.Y.order1 == 5
    assert poly.Z.order1 == 5
    assert not hasattr(poly, "_xml_ns")
    assert not hasattr(poly, "_xml_ns_key")

    # Init with kwargs
    poly = blocks.XYZPolyAttributeType(
        X=sicd.Position.ARPPoly.X,
        Y=sicd.Position.ARPPoly.Y,
        Z=sicd.Position.ARPPoly.Z,
        index=1,
        **kwargs,
    )
    assert poly._xml_ns == kwargs["_xml_ns"]
    assert poly._xml_ns_key == kwargs["_xml_ns_key"]

    # from_array
    poly = blocks.XYZPolyAttributeType.from_array(None)
    assert poly is None

    array = np.array(
        [
            sicd.Position.ARPPoly.X.Coefs,
            sicd.Position.ARPPoly.Y.Coefs,
            sicd.Position.ARPPoly.Z.Coefs,
        ]
    )
    poly = blocks.XYZPolyAttributeType.from_array(array)
    assert isinstance(poly, blocks.XYZPolyAttributeType)
    assert np.all(poly.X.Coefs == sicd.Position.ARPPoly.X.Coefs)
    assert np.all(poly.Y.Coefs == sicd.Position.ARPPoly.Y.Coefs)
    assert np.all(poly.Z.Coefs == sicd.Position.ARPPoly.Z.Coefs)

    # from_array errors
    bad_array = array[0:2]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected array to be of length 3, and received `{bad_array}`"
        ),
    ):
        blocks.XYZPolyAttributeType.from_array(bad_array)

    bad_array = "invalid"
    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        blocks.XYZPolyAttributeType.from_array(bad_array)


def test_blocks_gainphasepolytype(sicd, kwargs):
    # Smoke test
    poly = blocks.GainPhasePolyType(
        GainPoly=sicd.Antenna.Tx.Array.GainPoly,
        PhasePoly=sicd.Antenna.Tx.Array.PhasePoly,
    )
    assert poly.GainPoly.order1 == 8
    assert poly.GainPoly.order2 == 8
    assert poly.PhasePoly.order1 == 0
    assert poly.PhasePoly.order2 == 0
    assert not hasattr(poly, "_xml_ns")
    assert not hasattr(poly, "_xml_ns_key")

    # Init with kwargs
    poly = blocks.GainPhasePolyType(
        GainPoly=sicd.Antenna.Tx.Array.GainPoly,
        PhasePoly=sicd.Antenna.Tx.Array.PhasePoly,
        **kwargs,
    )
    assert poly._xml_ns == kwargs["_xml_ns"]
    assert poly._xml_ns_key == kwargs["_xml_ns_key"]

    # Poly minimize_order
    poly.minimize_order()
    assert np.shape(poly.GainPoly.Coefs) == (9, 9)
    assert np.shape(poly.PhasePoly.Coefs) == (1, 1)

    poly.GainPoly.Coefs[8][:] = 0.0
    poly.minimize_order()
    assert np.shape(poly.GainPoly.Coefs) == (8, 9)

    poly.GainPoly.Coefs[3][:] = 0.0
    poly.minimize_order()
    assert np.shape(poly.GainPoly.Coefs) == (8, 9)

    poly_eval = poly(0, 0)
    assert np.array_equal(poly_eval, np.array([0.0, 0.0]))
    poly = blocks.GainPhasePolyType(
        GainPoly=None, PhasePoly=sicd.Antenna.Tx.Array.PhasePoly
    )
    assert poly(0, 0) is None


def test_blocks_errordecorrfunctype(kwargs):
    # Smoke test
    error_decorr = blocks.ErrorDecorrFuncType(CorrCoefZero=0.0, DecorrRate=0.5)
    assert not hasattr(error_decorr, "_xml_ns")
    assert not hasattr(error_decorr, "_xml_ns_key")

    error_decorr = blocks.ErrorDecorrFuncType(
        CorrCoefZero=0.0, DecorrRate=0.5, **kwargs
    )
    assert error_decorr._xml_ns == kwargs["_xml_ns"]
    assert error_decorr._xml_ns_key == kwargs["_xml_ns_key"]
