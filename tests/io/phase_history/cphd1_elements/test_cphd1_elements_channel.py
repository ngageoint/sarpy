#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.io.phase_history.cphd1_elements import Channel


def test_channel_polreftype():
    """Test PolarizationRefType class"""
    pol_ref_arr = [0.5, 0.5, 0.0]
    tpol_ref_type = Channel.PolarizationRefType(pol_ref_arr[0], pol_ref_arr[1], pol_ref_arr[2])
    assert np.all(tpol_ref_type.get_array() == np.array(pol_ref_arr))

    assert tpol_ref_type.from_array(None) is None

    tpol_ref_type_1 = tpol_ref_type.from_array([0.6, 0.6, -0.1])
    assert np.all(tpol_ref_type_1.get_array() == np.array([0.6, 0.6, -0.1]))

    with pytest.raises(ValueError, match="Expected array to be of length 3"):
        bad_array = [0.6, 0.6]
        tpol_ref_type.from_array(bad_array)

    with pytest.raises(
        ValueError, match="Expected array to be numpy.ndarray, list, or tuple"
    ):
        bad_array = {0.6}
        tpol_ref_type.from_array(bad_array)

