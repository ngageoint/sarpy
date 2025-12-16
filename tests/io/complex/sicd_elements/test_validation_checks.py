# Generated using Copilot
# sarpy/io/complex/sicd_elements/test_validation_checks.py
# Pseudocode plan:

# 1. Import _pfa_check_stdeskew from the parent package using a relative import.
# 2. Create a test class using unittest.TestCase.
# 3. Test when PFA.STDeskew is None or Applied is False (should return True).
# 4. Test when Grid.TimeCOAPoly is constant (shape (1, 1) or all elements < 1e-6 except the first), and PFA.STDeskew.Applied is True (should return False).
# 5. Test when Grid.Row.DeltaKCOAPoly and PFA.STDeskew.STDSPhasePoly are present and differ by more than 1e-6 (should return False).
# 6. Test when Grid.Row.DeltaKCOAPoly and PFA.STDeskew.STDSPhasePoly are present and agree (should return True).
# 7. Use mocks for the required attributes and methods.

import unittest
import numpy as np
from unittest.mock import MagicMock

from sarpy.io.complex.sicd_elements.validation_checks import _pfa_check_stdeskew

class DummyPoly:
    def __init__(self, arr):
        self._arr = np.array(arr)
    def get_array(self, dtype=None):
        return self._arr

class DummySTDeskew:
    def __init__(self, applied=True, stds_phase_poly=None):
        self.Applied       = applied
        self.STDSPhasePoly = stds_phase_poly
    def log_validity_error(self, msg): pass
    def log_validity_warning(self, msg): pass

class DummyPFA:
    def __init__(self, stdeskew=None):
        self.STDeskew = stdeskew
    def log_validity_error(self, msg): self._last_error = msg
    def log_validity_warning(self, msg): self._last_warning = msg

class DummyGridRow:
    def __init__(self, delta_kcoa_poly=None):
        self.DeltaKCOAPoly = delta_kcoa_poly

class DummyGrid:
    def __init__(self, timecoa_poly=None, row=None):
        self.TimeCOAPoly = timecoa_poly
        self.Row = row

class TestPfaCheckStdeskew(unittest.TestCase):
    def test_stdeskew_none(self):
        pfa = DummyPFA(stdeskew=None)
        grid = DummyGrid()
        self.assertTrue(_pfa_check_stdeskew(pfa, grid))

    def test_stdeskew_applied_false(self):
        stdeskew = DummySTDeskew(applied=False)
        pfa = DummyPFA(stdeskew=stdeskew)
        grid = DummyGrid()
        self.assertTrue(_pfa_check_stdeskew(pfa, grid))

    def test_timecoa_poly_constant(self):
        stdeskew = DummySTDeskew(applied=True)
        pfa = DummyPFA(stdeskew=stdeskew)
        # shape (1,1)
        timecoa_poly = DummyPoly([[1.0]])
        grid = DummyGrid(timecoa_poly=timecoa_poly)
        self.assertFalse(_pfa_check_stdeskew(pfa, grid))
        # all elements < 1e-6 except first
        arr = np.zeros((2,2))
        arr[0,0] = 1.0
        timecoa_poly = DummyPoly(arr)
        grid = DummyGrid(timecoa_poly=timecoa_poly)
        self.assertFalse(_pfa_check_stdeskew(pfa, grid))

    def test_row_deltakcoa_and_stdsphasepoly_agree(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        stdeskew = DummySTDeskew(applied=True, stds_phase_poly=DummyPoly(arr))
        pfa = DummyPFA(stdeskew=stdeskew)
        grid = DummyGrid(row=DummyGridRow(delta_kcoa_poly=DummyPoly(arr)))
        self.assertTrue(_pfa_check_stdeskew(pfa, grid))

    def test_row_deltakcoa_and_stdsphasepoly_disagree(self):
        arr1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr2 = np.array([[1.0, 2.0], [3.0, 5.0]])  # last element differs
        stdeskew = DummySTDeskew(applied=True, stds_phase_poly=DummyPoly(arr1))
        pfa = DummyPFA(stdeskew=stdeskew)
        grid = DummyGrid(row=DummyGridRow(delta_kcoa_poly=DummyPoly(arr2)))
        self.assertFalse(_pfa_check_stdeskew(pfa, grid))

    def test_row_none(self):
        stdeskew = DummySTDeskew(applied=True)
        pfa = DummyPFA(stdeskew=stdeskew)
        grid = DummyGrid(row=None)
        self.assertTrue(_pfa_check_stdeskew(pfa, grid))

    def test_row_deltakcoa_none(self):
        stdeskew = DummySTDeskew(applied=True)
        pfa = DummyPFA(stdeskew=stdeskew)
        grid = DummyGrid(row=DummyGridRow(delta_kcoa_poly=None))
        self.assertTrue(_pfa_check_stdeskew(pfa, grid))

    def test_stdsphasepoly_none(self):
        stdeskew = DummySTDeskew(applied=True, stds_phase_poly=None)
        pfa = DummyPFA(stdeskew=stdeskew)
        grid = DummyGrid(row=DummyGridRow(delta_kcoa_poly=DummyPoly([[1.0]])))
        self.assertTrue(_pfa_check_stdeskew(pfa, grid))

if __name__ == "__main__":
    unittest.main()