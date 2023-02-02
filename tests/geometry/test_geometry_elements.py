#
# Copyright 2022 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.geometry import geometry_elements


def test_line_string():
    line_string = geometry_elements.LineString()

    # Check intersection
    line_string.coordinates = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    assert not line_string.self_intersection()
    line_string.coordinates = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    assert line_string.self_intersection()

def test_multi_line_string():
    poly_coords = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
    extended_poly_coords = [[0, 3], [3, 3], [3, 4], [0, 4], [0, 3]]

    multi_line_string = geometry_elements.MultiLineString(coordinates=[poly_coords, extended_poly_coords])

    # Check bounding box
    truth_box = [0, 0, 3, 4]
    box = multi_line_string.get_bbox()
    assert box == truth_box

def test_polygon():
    outer_ring_coords = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
    intersecting_ring_coords = [[0, 1], [1, 1], [1, 2], [0, 2], [0, 1]]

    test_poly = geometry_elements.Polygon([outer_ring_coords])

    # Check intersection by adding inner ring
    assert not test_poly.self_intersection()
    test_poly.add_inner_ring(intersecting_ring_coords)
    assert test_poly.self_intersection()

def test_multi_polygon():
    poly_coords1 = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
    poly_coords2 = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
    test_poly1 = geometry_elements.Polygon([poly_coords1])
    test_poly2 = geometry_elements.Polygon([poly_coords2])

    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])

    # Check bounding box
    truth_box = [0, 0, 3, 3]
    box = multi_poly.get_bbox()
    assert box == truth_box
