from __future__ import division

from sarpy.io.DEM.geoid import GeoidHeight
from sarpy import sarpy_support_dir
import numpy as np
from sarpy.utils import file_utils
import unittest

geoid_2008_1_fname = file_utils.get_path_from_subdirs(sarpy_support_dir, ['geoid_models',
                                                                          'egm2008-1',
                                                                          'geoids',
                                                                          'egm2008-1.pgm'])

geoid_test_data_fname = file_utils.get_path_from_subdirs(sarpy_support_dir, ['geoid_models',
                                                                             'geoid_test_data',
                                                                             'numpy_geoid_heights.npy'])


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

    def test_egm2008_1_w_geographiclib_test_data(self):
        geoid_height = GeoidHeight(geoid_2008_1_fname)
        test_heights = np.load(geoid_test_data_fname)
        dims = test_heights.shape
        height_diffs_bilinear = np.zeros(dims[0])
        height_diffs_cubic = np.zeros(dims[0])
        # max errors listed in interpolation and quantization errors table at:
        # https://geographiclib.sourceforge.io/html/geoid.html#testgeoid
        bilinear_max_error = 0.025
        cubic_max_error = 0.0022
        # decimate by a factor of 500 for speed
        for i in np.arange(0, dims[0], 500):
            test_line = test_heights[i]
            lat = test_line[0]
            lon = test_line[1]
            egm2008_height = test_line[4]
            height_diffs_bilinear[i] = geoid_height.get(lat, lon, cubic=False) - egm2008_height
            height_diffs_cubic[i] = geoid_height.get(lat, lon, cubic=True) - egm2008_height
        assert np.abs(height_diffs_bilinear.min()) < bilinear_max_error
        assert np.abs(height_diffs_bilinear.max()) < bilinear_max_error

        assert np.abs(height_diffs_cubic.min()) < cubic_max_error
        assert np.abs(height_diffs_cubic.max()) < cubic_max_error

        print("computed geoid heights match test data provided by geographiclib within " + str(bilinear_max_error) + " meters.")


if __name__ == '__main__':
    unittest.main()
