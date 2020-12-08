import time
import os
import logging
import numpy
import json

import sarpy.io.DEM.geoid as geoid
from tests import unittest, parse_file_entry


test_file = None
geoid_files = []
this_loc = os.path.abspath(__file__)
file_reference = os.path.join(os.path.split(this_loc)[0], 'geoid.json')  # specifies file locations
if os.path.isfile(file_reference):
    with open(file_reference, 'r') as fi:
        the_files = json.load(fi)
        test_file = parse_file_entry(the_files.get('test_file', None))
        for entry in the_files.get('geoid_files', []):
            the_file = parse_file_entry(entry)
            if the_file is not None:
                geoid_files.append(the_file)


def generic_geoid_test(instance, test_file, geoid_file):
    assert isinstance(instance, unittest.TestCase)

    _, gname = os.path.split(geoid_file)
    if gname.lower().startswith('egm84'):
        zcol = 2
    elif gname.lower().startswith('egm96'):
        zcol = 3
    else:
        zcol = 4
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
    @unittest.skipIf(test_file is None or len(geoid_files) == 0, 'No test file or geoid files found')
    def test_geoid_height(self):
        for fil in geoid_files:
            generic_geoid_test(self, test_file, fil)
