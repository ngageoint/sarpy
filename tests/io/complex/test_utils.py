
import numpy
from numpy.polynomial import polynomial
from sarpy.io.complex.utils import two_dim_poly_fit

from tests import unittest


class TestRadarSatUtils(unittest.TestCase):
    def test_two_dim_poly_fit(self):
        coeffs = numpy.arange(9).reshape((3, 3))
        y, x = numpy.meshgrid(numpy.arange(2, 6), numpy.arange(-2, 2))
        z = polynomial.polyval2d(x, y, coeffs)
        t_coeffs, residuals, rank, sing_vals = two_dim_poly_fit(x, y, z, x_order=2, y_order=2)
        diff = (numpy.abs(coeffs - t_coeffs) < 1e-10)
        self.assertTrue(numpy.all(diff))
