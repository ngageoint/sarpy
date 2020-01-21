# -*- coding: utf-8 -*-

import numpy
from numpy.polynomial import polynomial
from sarpy.io.complex.utils import two_dim_poly_fit

from . import unittest


class TestRadarSatUtils(unittest.TestCase):
    def test_two_dim_poly_fit(self):
        coeffs = numpy.arange(9).reshape((3, 3))
        y, x = numpy.meshgrid(numpy.arange(4), numpy.arange(4))
        z = polynomial.polyval2d(x, y, coeffs)
        t_coeffs = two_dim_poly_fit(x, y, z, x_order=2, y_order=2)
        diff = (numpy.abs(coeffs - t_coeffs) < 1e-10)
        self.assertTrue(numpy.all(diff))

