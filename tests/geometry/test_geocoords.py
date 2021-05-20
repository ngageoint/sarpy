
import numpy
from sarpy.geometry import geocoords

from tests import unittest


llh = numpy.array([[0, 0, 0], [0, 180, 0], [90, 0, 0], [-90, 0, 0]], dtype='float64')
ecf = numpy.array([[6378137, 0, 0], [-6378137, 0, 0], [0, 0, 6356752.314245179], [0, 0, -6356752.314245179]], dtype='float64')
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
            self.assertEqual(out2.shape, ecf.shape)
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
            self.assertEqual(out2.shape, llh.shape)
        with self.subTest(msg="2d value check"):
            self.assertTrue(numpy.all(numpy.abs(out2 - ecf) < tolerance))

        with self.subTest(msg="error check"):
            self.assertRaises(ValueError, geocoords.geodetic_to_ecf, numpy.arange(4))

    def test_values_both_ways(self):
        shp = (8, 5)
        rand_llh = numpy.empty(shp + (3, ), dtype=numpy.float64)
        rand_llh[:, :, 0] = 180*(numpy.random.rand(*shp) - 0.5)
        rand_llh[:, :, 1] = 360*(numpy.random.rand(*shp) - 0.5)
        rand_llh[:, :, 2] = 1e5*numpy.random.rand(*shp)

        rand_ecf = geocoords.geodetic_to_ecf(rand_llh)
        rand_llh2 = geocoords.ecf_to_geodetic(rand_ecf)
        rand_ecf2 = geocoords.geodetic_to_ecf(rand_llh2)

        llh_diff = numpy.abs(rand_llh - rand_llh2)
        ecf_diff = numpy.abs(rand_ecf - rand_ecf2)

        with self.subTest(msg="llh match"):
            self.assertTrue(numpy.all(llh_diff < tolerance))

        with self.subTest(msg="ecf match"):
            self.assertTrue(numpy.all(ecf_diff < tolerance))
