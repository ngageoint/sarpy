import numpy
from matplotlib import pyplot
import time

import sys
sys.path.append('../..')

from sarpy.geometry.geometry_elements import Polygon


def random_points():
    # create our polygon object
    segment_count = 12
    # starts at (0, 0) and goes up to (1, 0) with negative y coordinates by <segment_count> random steps,
    # then back to (0, 0) with positive y coordinates by <segment_count> random steps.
    coords = numpy.zeros((2 * segment_count + 1, 2), dtype=numpy.float64)
    # fill in the randomly generated x coordinates
    x1 = numpy.random.rand(segment_count + 1)
    x1[0] = 0
    l1 = numpy.cumsum(x1)
    coords[:segment_count+1, 0] = l1 / l1[-1]
    x2 = numpy.cumsum(numpy.random.rand(segment_count))
    coords[segment_count + 1:, 0] = 1 - x2 / x2[-1]
    # fill in the randomly generated y coordinates
    coords[1:segment_count, 1] = -numpy.random.rand(segment_count - 1)
    coords[segment_count+1:2*segment_count, 1] = numpy.random.rand(segment_count - 1)

    poly = Polygon(coordinates=[coords, ])

    # craft some sample points
    samples = 10000

    pts = numpy.random.rand(samples, 2)
    pts[:, 0] *= 1.2
    pts[:, 0] -= 0.1
    pts[:, 1] *= 2
    pts[:, 1] -= 1


    start = time.time()
    in_poly_condition = poly.contain_coordinates(pts[:, 0], pts[:, 1])
    lapsed = time.time() - start
    print('lapsed = {}, lapsed/point = {}'.format(lapsed, lapsed/samples))

    # now, do grid check
    grid_samples = 1001
    x_grid = numpy.linspace(-0.1, 1.1, grid_samples)
    y_grid = numpy.linspace(-1.1, 1.1, grid_samples)

    start = time.time()
    in_poly_condition2 = poly.grid_contained(x_grid[:-1], y_grid[:-1])
    lapsed = time.time() - start
    print('lapsed = {}, lapsed/point = {}'.format(lapsed, lapsed/((grid_samples - 1)**2)))

    fig, axs = pyplot.subplots(nrows=2, ncols=1, sharex='col', sharey='col')
    axs[0].scatter(pts[in_poly_condition, 0], pts[in_poly_condition, 1], color='r', marker='.', s=16)
    axs[0].scatter(pts[~in_poly_condition, 0], pts[~in_poly_condition, 1], color='b', marker='.', s=16)
    axs[0].plot(coords[:, 0], coords[:, 1], 'k-')
    y2d, x2d = numpy.meshgrid(y_grid, x_grid, indexing='xy')
    axs[1].pcolormesh(x2d, y2d, in_poly_condition2, cmap='jet')
    axs[1].plot(coords[:, 0], coords[:, 1], 'k-', lw=2, zorder=99)
    pyplot.show()
