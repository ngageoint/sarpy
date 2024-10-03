#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np

from sarpy.io.phase_history.cphd1_elements import Dwell


def test_dwell_dwelltype():
    """Test DwellType class"""
    dwell_type = Dwell.DwellType()
    assert dwell_type.NumCODTimes == 0
    assert dwell_type.NumDwellTimes == 0

    expected_num_cods = 2
    dwell_type.CODTimes = [
        Dwell.CODTimeType(Identifier=str(x), CODTimePoly=np.zeros((3, 4)))
        for x in range(expected_num_cods)
    ]
    assert dwell_type.NumCODTimes == expected_num_cods

    expected_num_dwells = 8
    dwell_type.DwellTimes = [
        Dwell.DwellTimeType(Identifier=str(x), DwellTimePoly=np.zeros((3, 4)))
        for x in range(expected_num_dwells)
    ]
    assert dwell_type.NumDwellTimes == expected_num_dwells

    dwell_type.DwellTimes.clear()
    assert dwell_type.NumDwellTimes == 0

