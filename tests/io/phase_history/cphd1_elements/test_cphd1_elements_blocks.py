#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.io.phase_history.cphd1_elements import blocks


@pytest.mark.parametrize(
    "array, class_to_test",
    [
        ([100.0, 200.0], blocks.LSType),
        ([100.0, 200.0], blocks.XYType),
    ],
)
def test_blocks_xytype(array, class_to_test):
    """Test LSType and XYType classes"""
    this_type = class_to_test(array[0], array[1])
    assert np.all(this_type.get_array() == np.array(array))

    assert this_type.from_array(None) is None

    this_type_1 = this_type.from_array([200.0, 300.0])
    assert np.all(this_type_1.get_array() == np.array([200.0, 300.0]))

    with pytest.raises(ValueError, match="Expected array to be of length 2"):
        bad_array = [1]
        this_type.from_array(bad_array)

    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        bad_array = {1}
        this_type.from_array(bad_array)


@pytest.mark.parametrize(
    "array, members, class_to_test",
    [
        ([100.0, 200.0], ['Line', 'Sample'], blocks.LSVertexType),
        ([100.0, 200.0], ['X', 'Y'], blocks.XYVertexType),
    ],
)
def test_blocks_xyvertextype(array, members, class_to_test):
    """Test LSVertexType and XYVertexType classes"""
    vertex_type = class_to_test([0, 0], 1)

    assert vertex_type.from_array(None, 1) is None

    vertex_type_1 = vertex_type.from_array(array, 1)
    assert getattr(vertex_type_1, members[0]) == array[0]
    assert getattr(vertex_type_1, members[1]) == array[1]
    assert vertex_type_1.index == 1

    with pytest.raises(ValueError, match="Expected array to be of length 2"):
        bad_vertex = [1]
        vertex_type.from_array(bad_vertex, 1)

    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        bad_vertex = {1}
        vertex_type.from_array(bad_vertex, 1)

