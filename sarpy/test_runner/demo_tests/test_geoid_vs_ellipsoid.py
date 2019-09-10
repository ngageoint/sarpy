from __future__ import division

from sarpy.io.DEM.geoid import GeoidHeight
from sarpy.test_runner.demo_tests import geoid_2008_1_fname
import numpy as np
import unittest


class TestGeoidVsEllipsoid(unittest.TestCase):

    def test_egm2008_1(self):
        geoid_height = GeoidHeight(geoid_2008_1_fname)

        # comparisons taken from https://geographiclib.sourceforge.io/cgi-bin/GeoidEval

        # additional lat / lons can be added, just putting in a few here to get an initial unit test going.
        latlon_diff_array = [
            [(0, 0), 17.226],
            [(20, 50), -30.7048]
        ]

        tolerance = 0.0001

        for entry in latlon_diff_array:
            latlon = entry[0]
            diff = entry[1]
            computed_height_diff = geoid_height.get(latlon[0], latlon[1])
            assert(np.isclose(diff, computed_height_diff, rtol=tolerance))
        print("all height difference results between "
              "https://geographiclib.sourceforge.io/cgi-bin/GeoidEval "
              "and the computed results match to within a tolerance of "
              + str(tolerance) + " meters")


if __name__ == '__main__':
    unittest.main()


