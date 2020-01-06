import numpy
from numpy.polynomial import polynomial

from .. import unittest

from sarpy.reading_elements.radarsat import _2d_poly_fit


class TestRadarSatUtils(unittest.TestCase):
    def test_2d_poly_fit(self):
        coeffs = numpy.arange(9).reshape((3, 3))
        y, x = numpy.meshgrid(numpy.arange(4), numpy.arange(4))
        z = polynomial.polyval2d(x, y, coeffs)
        t_coeffs = _2d_poly_fit(x.flatten(), y.flatten(), z.flatten(), x_order=2, y_order=2)
        diff = (numpy.abs(coeffs - t_coeffs) < 1e-10)
        self.assertTrue(numpy.all(diff))
