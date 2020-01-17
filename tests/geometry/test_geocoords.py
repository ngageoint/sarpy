# -*- coding: utf-8 -*-

import numpy
from sarpy.geometry import geocoords

from . import unittest


llh = numpy.array([[0, 0, 0], [0, 180, 0]], dtype=numpy.float64)
ecf = numpy.array([[6378137, 0, 0], [-6378137, 0, 0]], dtype=numpy.float64)
tolerance = 1e-8


class TestGeocoords(unittest.TestCase):
    def test_ecf_to_geodetic(self):
        out = geocoords.ecf_to_geodetic(ecf[0, :])
        with self.subTest(msg="basic shape check"):
            self.assertEqual(out.shape, (3, ))
        with self.subTest(msg="basic value check"):
            self.assertTrue(numpy.all(numpy.abs(out - llh[0, :]) < tolerance))

        out2 = geocoords.ecf_to_geodetic(ecf)
        with self.subTest(msg="2d shape check"):
            self.assertEqual(out2.shape, (2, 3))
        with self.subTest(msg="2d value check"):
            self.assertTrue(numpy.all(numpy.abs(out2 - llh) < tolerance))

        with self.subTest(msg="error check"):
            self.assertRaises(ValueError, geocoords.ecf_to_geodetic, numpy.arange(4))

    def test_geodetic_to_ecf(self):
        out = geocoords.geodetic_to_ecf(llh[0, :])
        with self.subTest(msg="basic shape check"):
            self.assertEqual(out.shape, (3,))
        with self.subTest(msg="basic value check"):
            self.assertTrue(numpy.all(numpy.abs(out - ecf[0, :]) < tolerance))

        out2 = geocoords.geodetic_to_ecf(llh)
        with self.subTest(msg="2d shape check"):
            self.assertEqual(out2.shape, (2, 3))
        with self.subTest(msg="2d value check"):
            self.assertTrue(numpy.all(numpy.abs(out2 - ecf) < tolerance))

        with self.subTest(msg="error check"):
            self.assertRaises(ValueError, geocoords.geodetic_to_ecf, numpy.arange(4))
