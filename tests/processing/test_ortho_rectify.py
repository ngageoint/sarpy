import pathlib

import numpy as np
import pytest

import sarpy.geometry.geometry_elements
import sarpy.io.complex.sicd
import sarpy.processing.ortho_rectify


@pytest.fixture
def temp_sicd(tmp_path):
    sicd_xml = pathlib.Path(__file__).parents[1] / "data/example.sicd.xml"
    sicd_meta = sarpy.io.complex.sicd.SICDType.from_xml_file(str(sicd_xml))
    sicd_file = tmp_path / "data-example.sicd"
    with sarpy.io.complex.sicd.SICDWriter(str(sicd_file), sicd_meta):
        pass  # don't care about pixels
    yield sicd_file


def test_ortho_iterator_bounds(temp_sicd):
    reader = sarpy.io.complex.sicd.is_a(str(temp_sicd))
    # contrived test case to exaggerate range/range-rate curvature on "closing" edge
    reader.sicd_meta.Grid.Col.SS *= 100
    rowcol_vertices_open = np.array([
        [1494, 1723],
        [0, 1723],
        [0, 0],
        [1494, 0],
    ])
    rowcol_vertices_closed = np.concatenate((rowcol_vertices_open, rowcol_vertices_open[0, np.newaxis]), axis=0)
    rc_multipoint = sarpy.geometry.geometry_elements.MultiPoint(rowcol_vertices_open)
    ortho_helper = sarpy.processing.ortho_rectify.OrthorectificationHelper(reader)
    bounds_open = ortho_helper.get_orthorectification_bounds_from_pixel_object(rowcol_vertices_open)
    bounds_closed = ortho_helper.get_orthorectification_bounds_from_pixel_object(rowcol_vertices_closed)
    bounds_geo = ortho_helper.get_orthorectification_bounds_from_pixel_object(rc_multipoint)
    assert np.allclose(bounds_open, bounds_closed)
    assert np.allclose(bounds_open, bounds_geo)
