import time
import os

import numpy

import sarpy.io.DEM.geoid as geoid

# TODO: refactor this into some unit tests

# establish file location
stem = 'C:/Users/jr80407/Desktop/sarpy_testing/geoid'

# parse test set
test_file = os.path.join(stem, 'GeoidHeights.dat')
with open(test_file, 'r') as fi:
    lins = fi.read().splitlines()
lats = numpy.zeros((len(lins),), dtype=numpy.float64)
lons = numpy.zeros((len(lins),), dtype=numpy.float64)
zs = numpy.zeros((len(lins),), dtype=numpy.float64)
for i, lin in enumerate(lins):
    slin = lin.strip().split()
    lats[i] = float(slin[0])
    lons[i] = float(slin[1])
    zs[i] = float(slin[4])
print('number of test points {}'.format(lats.size))

start = time.time()
gh = geoid.GeoidHeight(file_name=os.path.join(stem, 'egm2008-5.pgm'))
print('time initializing {}'.format(time.time() - start))


recs = 10
# test one
start = time.time()
zs1 = gh.get(lats[:recs], lons[:recs], cubic=False)
print('linear - time {}, diff {}'.format(time.time() - start, zs1 - zs[:recs]))
# test two
start = time.time()
zs1 = gh.get(lats[:recs], lons[:recs], cubic=True)
print('cubic - time {}, diff {}'.format(time.time() - start, zs1 - zs[:recs]))


# test one
start = time.time()
zs1 = gh.get(lats, lons, cubic=False)
diff = numpy.abs(zs1 - zs)
print('linear - time {}, max diff - {}, mean diff - {}'.format(time.time() - start, numpy.max(diff), numpy.mean(diff)))
# test two
start = time.time()
zs1 = gh.get(lats, lons, cubic=True)
diff = numpy.abs(zs1 - zs)
print('cubic - time {}, max diff - {}, mean diff - {}'.format(time.time() - start, numpy.max(diff), numpy.mean(diff)))
