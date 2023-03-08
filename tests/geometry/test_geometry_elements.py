#
# Copyright 2022 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.geometry import geometry_elements

@pytest.fixture(scope='module')
def test_elements():
    point = [33.447899, -112.097254]
    point_json = {
        "type": "Point",
        "coordinates": point
    }

    line_string = [
        [33.447899, -112.097254],
        [33.448364, -112.072789]
    ]
    line_string_json = {
        "type": "LineString",
        "coordinates": line_string
    }

    poly = [
        [33.846868, -112.269723],
        [33.831174, -111.637440],
        [33.244145, -111.669740],
        [33.237457, -112.261898],
        [33.846868, -112.269723]
    ]
    poly_json = {
        "type": "Polygon",
        "coordinates": [poly]
    }

    return {"point": point,
            "point_json": point_json,
            "line_string": line_string,
            "line_string_json": line_string_json,
            "poly": poly,
            "poly_json": poly_json}


def test_feature(test_elements):
    # test Feature class
    feature = geometry_elements.Feature()
    assert feature.uid is not None
    assert feature.geometry is None

    feature.geometry = geometry_elements.Geometry().from_dict(test_elements['point_json'])
    assert np.all(feature.geometry.coordinates == test_elements['point'])

    feature.properties = 'my properties'
    assert feature.properties == 'my properties'

    feature_dict = feature.to_dict()
    feature1 = geometry_elements.Feature().from_dict(feature_dict)
    assert np.all(feature1.geometry.coordinates == test_elements['point'])
    assert feature1.properties == 'my properties'

    feature2 = feature.replicate()
    assert np.all(feature2.geometry.coordinates == test_elements['point'])
    assert feature2.properties == 'my properties'

    # test FeatureCollection class
    feature_collection = geometry_elements.FeatureCollection([feature, feature1, feature2])
    assert len(feature_collection.features) == 3

    feature_collection.features = [feature2]
    assert len(feature_collection.features) == 4

    idx = feature_collection.get_integer_index(feature2.uid)
    assert idx == 3

    feature_collection_dict = feature_collection.to_dict()
    feature_collection1 = geometry_elements.FeatureCollection().from_dict(feature_collection_dict)
    assert len(feature_collection1) == 4

    feature_collection.add_feature(feature1)
    assert len(feature_collection.features) == 5

    feature_collection2 = feature_collection.replicate()
    assert len(feature_collection2.features) == 5
    feature_collection3 = feature_collection1.replicate()
    assert len(feature_collection3.features) == 4


def test_geometry_collection(test_elements):
    geom = geometry_elements.GeometryCollection([test_elements['point_json'],
                                                 test_elements['line_string_json'],
                                                 test_elements['poly_json']])

    # check properties
    assert len(geom.collection) == 3
    assert len(geom.geometries) == 3

    # check setter
    geom.geometries = []
    assert len(geom.collection) == 0

    geom.geometries = [test_elements['point_json']]
    assert len(geom.collection) == 1

    # check setter error conditions
    with pytest.raises(TypeError, match='geometries must be None or a list of Geometry objects'):
        geom.geometries = test_elements['point_json']

    with pytest.raises(TypeError, match='geometries must be a list of Geometry objects'):
        geom.geometries = [test_elements['point_json'], 10]

    # check bounding box
    geom.geometries = [test_elements['poly_json']]
    coord_arr = np.asarray(test_elements['poly'])
    truth_box = [min(coord_arr[:, 0]),
                 min(coord_arr[:, 1]),
                 max(coord_arr[:, 0]),
                 max(coord_arr[:, 1])]
    box = geom.get_bbox()
    assert box == truth_box

    # check to/from dict
    geom.geometries = [test_elements['point_json'], test_elements['line_string_json'], test_elements['poly_json']]
    geom_dict = geom.to_dict()
    geom1 = geometry_elements.GeometryCollection.from_dict(geom_dict)
    assert dir(geom) == dir(geom1)
    assert np.all(geom.geometries[0].coordinates == geom1.geometries[0].coordinates)
    assert np.all(geom.geometries[1].get_coordinate_list() ==
                  geom1.geometries[1].get_coordinate_list())
    assert np.all(geom.geometries[2].get_coordinate_list() ==
                  geom1.geometries[2].get_coordinate_list())

    # check assemble from collection
    point = geometry_elements.Point(coordinates=test_elements['point'])
    line_string = geometry_elements.LineString(coordinates=test_elements['line_string'])
    geom = geometry_elements.GeometryCollection([test_elements['point_json'], test_elements['line_string_json'], test_elements['poly_json']])

    geom1 = geometry_elements.GeometryCollection()
    geom1 = geom1.assemble_from_collection(point, line_string, geom)
    list_of_geometries = geom1.geometries
    assert len(list_of_geometries) == 5
    assert list_of_geometries[0].get_coordinate_list() == test_elements['point']
    assert list_of_geometries[1].get_coordinate_list() == test_elements['line_string']
    assert list_of_geometries[2].get_coordinate_list() == test_elements['point']
    assert list_of_geometries[3].get_coordinate_list() == test_elements['line_string']
    assert list_of_geometries[4].get_coordinate_list()[0] == test_elements['poly']

    # No args
    geom2 = geom1.assemble_from_collection()
    assert isinstance(geom2, geometry_elements.GeometryCollection)


def test_polygons():
    outer_ring_coords = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
    inner_ring_coords = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
    intersecting_ring_coords = [[0, 1], [1, 1], [1, 2], [0, 2], [0, 1]]

    # check inner/outer ring
    ring1 = geometry_elements.LinearRing(outer_ring_coords)
    ring2 = geometry_elements.LinearRing(inner_ring_coords)
    ring3 = geometry_elements.LinearRing(intersecting_ring_coords)
    assert np.all(ring1.get_coordinate_list() == outer_ring_coords)
    assert np.all(ring2.get_coordinate_list() == inner_ring_coords)
    assert np.all(ring3.get_coordinate_list() == intersecting_ring_coords)

    test_poly = geometry_elements.Polygon([ring1, ring2])
    # check inner/outer ring
    outer_ring = test_poly.outer_ring
    assert np.all(outer_ring.get_coordinate_list() == outer_ring_coords)

    inner_rings = test_poly.inner_rings
    assert np.all(inner_rings[0].get_coordinate_list() == inner_ring_coords)

    # check intersection by adding inner ring
    assert not test_poly.self_intersection()
    test_poly.add_inner_ring(intersecting_ring_coords)
    assert test_poly.self_intersection()

    # check to/from dict
    polygon_dict = test_poly.to_dict()
    test_polygon1 = geometry_elements.Polygon().from_dict(polygon_dict)
    assert test_polygon1.outer_ring.get_coordinate_list() == outer_ring_coords
    inner_rings = test_polygon1.inner_rings
    assert inner_rings[0].get_coordinate_list() == inner_ring_coords[::-1]
    assert inner_rings[1].get_coordinate_list() == intersecting_ring_coords[::-1]

    # check get perimeter
    ring1_perimeter = ring1.get_perimeter()
    ring2_perimeter = ring2.get_perimeter()
    ring3_perimeter = ring3.get_perimeter()
    poly_perimeter = test_poly.get_perimeter()
    assert poly_perimeter == ring1_perimeter + ring2_perimeter + ring3_perimeter

    # check set outer ring
    test_polygon1 = geometry_elements.Polygon([ring1])
    assert test_polygon1.get_area() == 9

    test_polygon1.set_outer_ring([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
    assert test_polygon1.get_area() == 100

    # check centroid
    test_polygon1 = geometry_elements.Polygon()
    assert test_polygon1.get_centroid() is None

    test_polygon1 = geometry_elements.Polygon([outer_ring_coords])
    assert np.all(test_polygon1.get_centroid() == [1.5, 1.5])

    # check contain coordinates
    test_polygon1 = geometry_elements.Polygon([outer_ring_coords])
    assert np.all(test_polygon1.contain_coordinates([1], [1]))
    assert np.all(test_polygon1.contain_coordinates([1, 2, 3], [0, 1, 2]))
    assert not np.all(test_polygon1.contain_coordinates([4], [4]))

    # check grid contained
    test_polygon1 = geometry_elements.Polygon([outer_ring_coords])

    # grid_y must be monotonically increasing
    with pytest.raises(ValueError, match='grid_y must be monotonically increasing'):
        test_polygon1.grid_contained(np.array([1, 2, 3, 4, 5]), np.array([1, 1, 1]))
    # grid_x must be monotonically increasing
    with pytest.raises(ValueError, match='grid_x must be monotonically increasing'):
        test_polygon1.grid_contained(np.array([1, 1, 1, 1]), np.array([1, 2, 3, 4, 5]))
    # grid_x and grid_y must be one dimensional
    with pytest.raises(ValueError, match='grid_x and grid_y must be one dimensional'):
        test_polygon1.grid_contained(np.array([[1, 2, 3], [3, 4, 5]]), np.array([1, 2, 3]))

    assert np.all(test_polygon1.grid_contained(np.array([1, 2, 3]), np.array([0, 1, 2])))

    contained = test_polygon1.grid_contained(np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))
    for i, val in enumerate(contained):
        for j in range(len(val)):
            if i <= 3 and j <= 3:
                assert val[j]
            else:
                assert not val[j]


def test_multi_polygon(test_elements):
    poly_coords1 = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
    poly_coords2 = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
    poly_coords3 = [[1, 2], [2, 2], [2, 3], [1, 3], [1, 2]]
    test_poly1 = geometry_elements.Polygon([poly_coords1])
    test_poly2 = geometry_elements.Polygon([poly_coords2])
    test_poly3 = geometry_elements.Polygon([poly_coords3])

    # check instantiation and properties
    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2, test_poly3])

    assert len(multi_poly.collection) == 3
    assert len(multi_poly.polygons) == 3

    # check polygon setter
    multi_poly = geometry_elements.MultiPolygon()

    assert multi_poly.polygons is None
    multi_poly.polygons = [test_poly1, test_poly2, test_poly3]
    assert len(multi_poly.polygons) == 3

    # check bounding box
    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])

    truth_box = [0, 0, 3, 3]
    box = multi_poly.get_bbox()
    assert box == truth_box

    # check to/from dict
    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])
    multi_poly_dict = multi_poly.to_dict()
    multi_poly1 = geometry_elements.MultiPolygon().from_dict(multi_poly_dict)
    assert multi_poly1.get_coordinate_list()[0] == test_poly1.get_coordinate_list()
    assert multi_poly1.get_coordinate_list()[1] == test_poly2.get_coordinate_list()

    # check get perimeter
    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])
    poly1_perimeter = test_poly1.get_perimeter()
    poly2_perimeter = test_poly2.get_perimeter()
    assert multi_poly.get_perimeter() == poly1_perimeter + poly2_perimeter

    # check contain coordinates
    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])
    assert np.all(multi_poly.contain_coordinates([1], [1]))
    assert np.all(multi_poly.contain_coordinates([1, 2, 3], [0, 1, 2]))
    assert not np.all(multi_poly.contain_coordinates([4], [4]))

    # check grid contained
    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])
    # grid_y must be monotonically increasing
    with pytest.raises(ValueError, match='grid_y must be monotonically increasing'):
        multi_poly.grid_contained(np.array([1, 2, 3, 4, 5]), np.array([1, 1, 1]))
    # grid_x must be monotonically increasing
    with pytest.raises(ValueError, match='grid_x must be monotonically increasing'):
        multi_poly.grid_contained(np.array([1, 1, 1, 1]), np.array([1, 2, 3, 4, 5]))
    # grid_x and grid_y must be one dimensional
    with pytest.raises(ValueError, match='grid_x and grid_y must be one dimensional'):
        multi_poly.grid_contained(np.array([[1,2,3], [3, 4, 5]]), np.array([1, 2, 3]))

    assert np.all(multi_poly.grid_contained(np.array([1, 2, 3]), np.array([0, 1, 2])))

    contained = multi_poly.grid_contained(np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))
    for i, val in enumerate(contained):
        for j in range(len(val)):
            if i <= 3 and j <= 3:
                assert val[j]
            else:
                assert not val[j]

    # check get_minimum_distance
    multi_poly = geometry_elements.MultiPolygon()
    assert multi_poly.get_minimum_distance([0, 0]) == float('inf')

    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])
    assert multi_poly.get_minimum_distance([0, 0]) == 0.0
    assert multi_poly.get_minimum_distance([0, 4]) == 1.0
    assert multi_poly.get_minimum_distance([4, 0]) == 1.0
    assert multi_poly.get_minimum_distance([4, 4]) == pytest.approx(np.sqrt(2), abs=1e-8)

    # check assemble from collection
    ring = geometry_elements.LinearRing(poly_coords1)
    poly = geometry_elements.Polygon([poly_coords2])
    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])
    geom_coll = geometry_elements.GeometryCollection([test_elements['poly_json']])

    multi_poly1 = geometry_elements.MultiPolygon()
    multi_poly1 = multi_poly1.assemble_from_collection(ring, poly, multi_poly, geom_coll)
    assert multi_poly1.polygons[0].get_coordinate_list()[0] == poly_coords1
    assert multi_poly1.polygons[1].get_coordinate_list()[0] == poly_coords2
    assert multi_poly1.polygons[2].get_coordinate_list()[0] == poly_coords1
    assert multi_poly1.polygons[3].get_coordinate_list()[0] == poly_coords2
    assert multi_poly1.polygons[4].get_coordinate_list()[0] == test_elements['poly']

    # No args
    multi_poly2 = multi_poly1.assemble_from_collection()
    assert isinstance(multi_poly2, geometry_elements.MultiPolygon)


def test_point(test_elements):
    # test Point class
    point = geometry_elements.Point(coordinates=test_elements['point'])
    assert np.all(point.coordinates == test_elements['point'])

    point.coordinates = [0.0, 0.0]
    assert np.all(point.coordinates == [0.0, 0.0])

    # Coordinates must be a one-dimensional array
    with pytest.raises(ValueError, match='coordinates must be a one-dimensional array'):
        point.coordinates = np.zeros((2, 3))
    # Coordinates must have between 2 and 4 entries
    with pytest.raises(ValueError, match='coordinates must have between 2 and 4 entries'):
        point.coordinates = [0.0]
    # Coordinates must have between 2 and 4 entries
    with pytest.raises(ValueError, match='coordinates must have between 2 and 4 entries'):
        point.coordinates = [0.0, 0.0, 0.0, 0.0, 0.0]

    point.coordinates = test_elements['point']
    assert np.all(point.get_bbox() == test_elements['point'] + test_elements['point'])

    point.coordinates = test_elements['point']
    assert point.get_coordinate_list() == test_elements['point']

    point_dict = point.to_dict()
    point1 = geometry_elements.Point().from_dict(point_dict)
    assert point.get_coordinate_list() == point1.get_coordinate_list()

    point = geometry_elements.Point(coordinates=[1, 1])
    point1 = [0, 0]
    assert point.get_minimum_distance(point1) == pytest.approx(np.sqrt(2), abs=1e-8)


def test_multi_point(test_elements):
    # test MultiPoint class
    multi_point = geometry_elements.MultiPoint()
    assert multi_point.get_coordinate_list() is None

    multi_point = geometry_elements.MultiPoint(coordinates=test_elements['poly'])
    assert np.all(multi_point.get_coordinate_list() == test_elements['poly'])

    multi_point1 = geometry_elements.MultiPoint(coordinates=multi_point)
    assert np.all(multi_point.get_coordinate_list() == multi_point1.get_coordinate_list())
    assert len(multi_point1.collection) == len(test_elements['poly'])

    assert len(multi_point1.points) == len(test_elements['poly'])

    multi_point1.points = test_elements['line_string']
    assert np.all(multi_point1.get_coordinate_list() == test_elements['line_string'])
    assert len(multi_point1.collection) == len(test_elements['line_string'])

    multi_point_dict = multi_point.to_dict()
    multi_point1 = geometry_elements.MultiPoint().from_dict(multi_point_dict)
    assert multi_point.get_coordinate_list() == multi_point1.get_coordinate_list()

    # test assemble from collection
    point = geometry_elements.Point(coordinates=test_elements['point'])
    multi_point1 = geometry_elements.MultiPoint()
    multi_point1 = multi_point1.assemble_from_collection(multi_point, point)
    truth_coords = test_elements['poly']
    truth_coords.append(test_elements['point'])
    assert multi_point1.get_coordinate_list() == truth_coords

    # No args
    multi_point2 = multi_point1.assemble_from_collection()
    assert isinstance(multi_point2, geometry_elements.MultiPoint)

    # test bounding box
    coord_arr = np.asarray(test_elements['poly'])
    truth_box = [min(coord_arr[:, 0]),
                 min(coord_arr[:, 1]),
                 max(coord_arr[:, 0]),
                 max(coord_arr[:, 1])]
    box = multi_point1.get_bbox()
    assert box == truth_box

    multi_point2 = geometry_elements.MultiPoint()
    assert multi_point2.get_bbox() is None


def test_line_string(test_elements):
    # test LineString class
    line_string = geometry_elements.LineString(coordinates=test_elements['line_string'])
    assert np.all(line_string.coordinates == test_elements['line_string'])

    line_string1 = geometry_elements.LineString(coordinates=line_string)
    assert np.all(line_string1.coordinates == line_string.coordinates)

    line_string.coordinates = [[0.0, 0.0], [1.0, 1.0]]
    assert np.all(line_string.coordinates == [[0.0, 0.0], [1.0, 1.0]])

    # Coordinates must be a two-dimensional array
    with pytest.raises(ValueError, match='coordinates must be a two-dimensional array'):
        line_string.coordinates = np.zeros((2, 3, 2))
    # The second dimension of coordinates must have between 2 and 4 entries
    with pytest.raises(ValueError, match='The second dimension of coordinates must have between 2 and 4 entries'):
        line_string.coordinates = [[0.0]]
    # The second dimension of coordinates must have between 2 and 4 entries
    with pytest.raises(ValueError, match='The second dimension of coordinates must have between 2 and 4 entries'):
        line_string.coordinates = [[0.0, 0.0, 0.0, 0.0, 0.0]]

    # check intersection
    line_string = geometry_elements.LineString()

    line_string.coordinates = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    assert not line_string.self_intersection()
    line_string.coordinates = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    assert line_string.self_intersection()

    # check bounding box
    line_string = geometry_elements.LineString(coordinates=test_elements['line_string'])

    coord_arr = np.asarray(test_elements['line_string'])
    truth_box = [min(coord_arr[:, 0]),
                 min(coord_arr[:, 1]),
                 max(coord_arr[:, 0]),
                 max(coord_arr[:, 1])]
    box = line_string.get_bbox()
    assert box == truth_box

    # check coordinate list getter
    coord_list = line_string.get_coordinate_list()
    assert coord_list == test_elements['line_string']

    # check to/from dict
    line_string_dict = line_string.to_dict()
    line_string1 = geometry_elements.LineString().from_dict(line_string_dict)
    assert line_string.get_coordinate_list() == line_string1.get_coordinate_list()

    # check get length
    line_string1.coordinates = [[0.0, 0.0], [1.0, 1.0]]
    assert line_string1.get_length() == pytest.approx(np.sqrt(2), abs=1e-8)

    # check get minimum distance
    line_string1.coordinates = [[0.0, 0.0], [1.0, 1.0]]
    test_point = [0.0, 0.0]
    assert line_string1.get_minimum_distance(test_point) == 0.0

    test_point = [1.0, 0.0]
    assert line_string1.get_minimum_distance(test_point) == pytest.approx(1.0 / np.sqrt(2), abs=1e-8)


def test_multi_line_string(test_elements):
    # test instantiate with a list of lists
    multi_line_string = geometry_elements.MultiLineString(coordinates=[test_elements['poly'], test_elements['poly']])
    assert len(multi_line_string.lines) == 2
    assert len(multi_line_string.lines[0].coordinates) == len(test_elements['poly'])
    assert len(multi_line_string.lines[1].coordinates) == len(test_elements['poly'])

    # test instantiate with a MultiLineString
    multi_line_string1 = geometry_elements.MultiLineString(coordinates=multi_line_string)
    assert len(multi_line_string1.lines) == 2
    assert len(multi_line_string1.lines[0].coordinates) == len(test_elements['poly'])
    assert len(multi_line_string1.lines[1].coordinates) == len(test_elements['poly'])

    # check line setter
    multi_line_string1.lines = None
    assert multi_line_string1.lines is None

    multi_line_string1.lines = [test_elements['poly'], test_elements['poly']]
    assert len(multi_line_string1.lines) == 2
    assert len(multi_line_string1.lines[0].coordinates) == len(test_elements['poly'])
    assert len(multi_line_string1.lines[1].coordinates) == len(test_elements['poly'])

    # Coordinates must be a two-dimensional array
    with pytest.raises(ValueError, match='coordinates must be a two-dimensional array'):
        multi_line_string1.lines = test_elements['poly']

    # check bounding box
    poly_coords1 = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
    extended_poly_coords = [[0, 3], [3, 3], [3, 4], [0, 4], [0, 3]]

    bb_line_string = geometry_elements.MultiLineString(coordinates=[poly_coords1, extended_poly_coords])

    truth_box = [0, 0, 3, 4]
    box = bb_line_string.get_bbox()
    assert box == truth_box

    # check coordinate list getter
    coord_list = multi_line_string.get_coordinate_list()
    assert coord_list[0] == test_elements['poly']
    assert coord_list[1] == test_elements['poly']

    # check to/from dict
    multi_line_string_dict = multi_line_string.to_dict()
    multi_line_string1 = geometry_elements.MultiLineString().from_dict(multi_line_string_dict)
    assert multi_line_string.get_coordinate_list() == multi_line_string1.get_coordinate_list()

    # check get minimum distance
    multi_line_string1.lines = [[[0.0, 0.0], [2.0, 0.0]],
                                [[0.0, 2.0], [2.0, 2.0]]]
    test_point = [0.0, 1.0]
    assert multi_line_string1.get_minimum_distance(test_point) == 1.0

    test_point = [1.0, 0.0]
    assert multi_line_string1.get_minimum_distance(test_point) == 0

    # check assemble from collection
    multi_line_string1 = multi_line_string1.assemble_from_collection()
    assert isinstance(multi_line_string1, geometry_elements.MultiLineString)

    line_string = geometry_elements.LineString(coordinates=test_elements['line_string'])
    multi_line_string1 = geometry_elements.MultiLineString()
    multi_line_string1 = multi_line_string1.assemble_from_collection(multi_line_string, line_string)
    assert multi_line_string1.lines[0].get_coordinate_list() == test_elements['poly']
    assert multi_line_string1.lines[1].get_coordinate_list() == test_elements['poly']
    assert multi_line_string1.lines[2].get_coordinate_list() == test_elements['line_string']


def test_linear_ring():
    outer_ring_coords = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]

    # check init and get_coordinate_list
    ring1 = geometry_elements.LinearRing()
    assert isinstance(ring1, geometry_elements.LinearRing)
    assert ring1.get_coordinate_list() is None

    ring2 = geometry_elements.LinearRing(outer_ring_coords)
    ring3 = geometry_elements.LinearRing(ring2)
    assert ring2.get_coordinate_list() == ring3.get_coordinate_list()

    # check reverse_orientation (no coordinates)
    ring1.reverse_orientation()
    assert ring1.get_coordinate_list() is None

    # check reverse_orientation (with coordinates)
    ring2.reverse_orientation()
    assert np.all(np.array(ring2.get_coordinate_list()) == np.array(outer_ring_coords)[::-1, :])

    # check bounding_box
    truth_box = np.array([[0., 3.], [0., 3.]])
    box = ring2.bounding_box
    assert np.all(box == truth_box)

    ring2.coordinates = None
    assert ring2.get_coordinate_list() is None

    with pytest.raises(ValueError, match='coordinates must be two-dimensional'):
        ring2.coordinates = outer_ring_coords[0]
    with pytest.raises(ValueError, match='coordinates must have between 2 and 4 entries'):
        ring2.coordinates = [[0], [1]]
    with pytest.raises(ValueError, match='coordinates must have between 2 and 4 entries'):
        ring2.coordinates = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]


def test_basic_assemble(test_elements):
    poly_coords1 = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
    poly_coords2 = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
    test_poly1 = geometry_elements.Polygon([poly_coords1])
    test_poly2 = geometry_elements.Polygon([poly_coords2])

    multi_point = geometry_elements.MultiPoint(coordinates=test_elements['poly'])
    multi_line_string = geometry_elements.MultiLineString(coordinates=[test_elements['poly'], test_elements['poly']])
    multi_poly = geometry_elements.MultiPolygon([test_poly1, test_poly2])

    collective_type = geometry_elements.basic_assemble_from_collection(multi_point)
    assert collective_type.type == 'MultiPoint'
    collective_type = geometry_elements.basic_assemble_from_collection(multi_line_string)
    assert collective_type.type == 'MultiLineString'
    collective_type = geometry_elements.basic_assemble_from_collection(multi_poly)
    assert collective_type.type == 'MultiPolygon'
    collective_type = geometry_elements.basic_assemble_from_collection(multi_point, multi_line_string, multi_poly)
    assert collective_type.type == 'GeometryCollection'
