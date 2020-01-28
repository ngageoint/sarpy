import time
import os
import logging

import numpy

from . import unittest

import sarpy.io.DEM.geoid as geoid


def generic_geoid_test(instance, test_file, geoid_file):
    assert isinstance(instance, unittest.TestCase)

    zcol = 4  # TODO: determine this appropriately
    with open(test_file, 'r') as fi:
        lins = fi.read().splitlines()
    lats = numpy.zeros((len(lins),), dtype=numpy.float64)
    lons = numpy.zeros((len(lins),), dtype=numpy.float64)
    zs = numpy.zeros((len(lins),), dtype=numpy.float64)
    for i, lin in enumerate(lins):
        slin = lin.strip().split()
        lats[i] = float(slin[0])
        lons[i] = float(slin[1])
        zs[i] = float(slin[zcol])
    logging.info('number of test points {}'.format(lats.size))

    with instance.subTest(msg="Load geoid file"):
        start = time.time()
        gh = geoid.GeoidHeight(file_name=geoid_file)
        logging.info('time loading geoid file {}'.format(time.time() - start))

    recs = 10
    # small linear test
    with instance.subTest(msg='Small linear interpolation test'):
        start = time.time()
        zs1 = gh.get(lats[:recs], lons[:recs], cubic=False)
        logging.info('linear - time {}, diff {}'.format(time.time() - start, zs1 - zs[:recs]))

    # small cubic test
    with instance.subTest(msg='Small cubic interpolation test'):
        start = time.time()
        zs1 = gh.get(lats[:recs], lons[:recs], cubic=True)
        logging.info('cubic - time {}, diff {}'.format(time.time() - start, zs1 - zs[:recs]))

    # full linear test
    with instance.subTest(msg="Full linear interpolation test"):
        start = time.time()
        zs1 = gh.get(lats, lons, cubic=False)
        diff = numpy.abs(zs1 - zs)
        max_diff = numpy.max(diff)
        mean_diff = numpy.mean(diff)
        logging.info('linear - time {}, max diff - {}, mean diff - {}'.format(time.time() - start, max_diff, mean_diff))
        instance.assertLessEqual(max_diff, 2, msg="Max difference should be less than 2 meters")
        instance.assertLessEqual(mean_diff, 0.5, msg="Mean difference should be (much) less than 0.5 meters.")

    # full cubic test
    with instance.subTest(msg="Full cubic interpolation test"):
        start = time.time()
        zs1 = gh.get(lats, lons, cubic=True)
        diff = numpy.abs(zs1 - zs)
        max_diff = numpy.max(diff)
        mean_diff = numpy.mean(diff)
        logging.info('cubic - time {}, max diff - {}, mean diff - {}'.format(time.time() - start, max_diff, mean_diff))
        instance.assertLessEqual(max_diff, 2, msg="Max difference should be less than 2 meters")
        instance.assertLessEqual(mean_diff, 0.5, msg="Mean difference should be (much) less than 0.5 meters.")


class TestGeoidHeight(unittest.TestCase):

    @classmethod
    def setUp(cls):
        # todo: fix this up
        cls.test_root = os.path.expanduser(os.path.join('~', 'Desktop', 'sarpy_testing', 'geoid'))

    def test_geoid_height(self):
        # establish file location
        test_file = os.path.join(self.test_root, 'GeoidHeights.dat')

        tested = 0
        for fil in ['egm2008-5.pgm', 'egm2008-2_5.pgm', 'egm2008-1.pgm']:
            geoid_file = os.path.join(self.test_root, fil)
            if os.path.exists(test_file) and os.path.exists(geoid_file):
                tested += 1
                generic_geoid_test(self, test_file, geoid_file)
            else:
                logging.info('No file {} or {} found'.format(test_file, geoid_file))

        self.assertTrue(tested > 0, msg="No files for testing found")
