#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.io.phase_history.cphd1_elements import Global


@pytest.mark.parametrize(
    "array, class_to_test",
    [
        ([100.0, 200.0], Global.FxBandType),
        ([100.0, 200.0], Global.TOASwathType),
    ],
)
def test_global_fxband_and_toaswath_types(array, class_to_test):
    """Test FxBandType and TOASwathType classes"""
    class_type = class_to_test(array[0], array[1])
    assert np.all(class_type.get_array() == np.array(array))

    assert class_type.from_array(None) is None

    class_type_1 = class_type.from_array([200.0, 300.0])
    assert np.all(class_type_1.get_array() == np.array([200.0, 300.0]))

    with pytest.raises(ValueError, match="Expected array to be of length 2"):
        bad_array = [1]
        class_type.from_array(bad_array)

    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        bad_array = {1}
        class_type.from_array(bad_array)

