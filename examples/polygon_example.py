"""
Basic polygon example - this needs to be reworked
"""

import numpy
from matplotlib import pyplot
import time

from sarpy.geometry.geometry_elements import Polygon


def generate_random_polygon(segment_count=12):
    """
    Generate the coordinates for a random polygon going from (-1, 0) to (1, 0) and back.
    It will be contained in square [-1, 1] x [-1, 1].

    Parameters
    ----------
    segment_count : int

    Returns
    -------
    numpy.ndarray
    """

    # starts at (-1, 0) and goes up to (1, 0) with negative y coordinates by <segment_count> random steps,
    # then back to (-1, 0) with positive y coordinates by <segment_count> random steps.
    coords = numpy.zeros((2 * segment_count + 1, 2), dtype=numpy.float64)
    # fill in the randomly generated x coordinates
    x1 = numpy.random.rand(segment_count + 1)
    x1[0] = 0
    l1 = numpy.cumsum(x1)
    coords[:segment_count+1, 0] = -1 + 2*l1/l1[-1]
    x2 = numpy.cumsum(numpy.random.rand(segment_count))
    coords[segment_count + 1:, 0] = 1 - 2*x2/x2[-1]
    # fill in the randomly generated y coordinates
    coords[1:segment_count, 1] = -numpy.random.rand(segment_count - 1)
    coords[segment_count+1:2*segment_count, 1] = numpy.random.rand(segment_count - 1)
    return coords


def basic_check():
    """
    Example for checks on a basic polygon.

    Returns
    -------
    None
    """

    # create our polygon object with coordinates bounded by square [0, 1]x[-1, 1]
    coords = generate_random_polygon()
    poly = Polygon(coordinates=[coords, ])

    #############################
    # perform random samples check
    samples = 10000

    pts = 2.2*numpy.random.rand(samples, 2) - 1.1

    start = time.time()
    in_poly_condition = poly.contain_coordinates(pts[:, 0], pts[:, 1])
    lapsed = time.time() - start
    print('basic poly: lapsed = {}, lapsed/point = {}'.format(lapsed, lapsed/samples))

    ###########################
    # perform grid check
    grid_samples = 1001
    x_grid = numpy.linspace(-1.1, 1.1, grid_samples)
    y_grid = numpy.linspace(-1.1, 1.1, grid_samples)

    start = time.time()
    in_poly_condition2 = poly.grid_contained(x_grid[:-1], y_grid[:-1])
    lapsed = time.time() - start
    print('basic poly: lapsed = {}, lapsed/point = {}'.format(lapsed, lapsed/((grid_samples - 1)**2)))

    #############################
    # visualize results
    fig, axs = pyplot.subplots(nrows=2, ncols=1, sharex='col', sharey='col')
    fig.suptitle('Basic polygon example')
    axs[0].scatter(pts[in_poly_condition, 0], pts[in_poly_condition, 1], color='r', marker='.', s=16)
    axs[0].scatter(pts[~in_poly_condition, 0], pts[~in_poly_condition, 1], color='b', marker='.', s=16)
    axs[0].plot(coords[:, 0], coords[:, 1], 'k-')
    y2d, x2d = numpy.meshgrid(y_grid, x_grid, indexing='xy')
    axs[1].pcolormesh(x2d, y2d, in_poly_condition2, cmap='jet')
    axs[1].plot(coords[:, 0], coords[:, 1], 'k-', lw=2, zorder=99)
    pyplot.show()


def compound_poly_check():
    """
    Example for compound polygon with a hole in it.

    Returns
    -------
    None
    """

    # create our polygon object with coordinates bounded by square [0, 1]x[-1, 1]
    outer_coords = numpy.array([
        [-1, 0], [-0.5, -1], [0.5, -1], [1, 0], [0.5, 1], [-0.5, 1], [-1, 0],], dtype='float64')
    inner_coords = 0.5*generate_random_polygon()
    poly = Polygon(coordinates=[outer_coords, inner_coords])

    #############################
    # perform random samples check
    samples = 10000
    pts = 2.2*numpy.random.rand(samples, 2) - 1.1

    start = time.time()
    in_poly_condition = poly.contain_coordinates(pts[:, 0], pts[:, 1])
    lapsed = time.time() - start
    print('compound poly: lapsed = {}, lapsed/point = {}'.format(lapsed, lapsed/samples))

    ###########################
    # perform grid check
    grid_samples = 1001
    x_grid = numpy.linspace(-1.1, 1.1, grid_samples)
    y_grid = numpy.linspace(-1.1, 1.1, grid_samples)

    start = time.time()
    in_poly_condition2 = poly.grid_contained(x_grid[:-1], y_grid[:-1])
    lapsed = time.time() - start
    print('compound poly: lapsed = {}, lapsed/point = {}'.format(lapsed, lapsed/((grid_samples - 1)**2)))

    #############################
    # visualize results
    fig, axs = pyplot.subplots(nrows=2, ncols=1, sharex='col', sharey='col')
    fig.suptitle('Compound polygon example')
    axs[0].scatter(pts[in_poly_condition, 0], pts[in_poly_condition, 1], color='r', marker='.', s=16)
    axs[0].scatter(pts[~in_poly_condition, 0], pts[~in_poly_condition, 1], color='b', marker='.', s=16)
    axs[0].plot(outer_coords[:, 0], outer_coords[:, 1], 'k-')
    axs[0].plot(inner_coords[:, 0], inner_coords[:, 1], 'k-')
    y2d, x2d = numpy.meshgrid(y_grid, x_grid, indexing='xy')
    axs[1].pcolormesh(x2d, y2d, in_poly_condition2, cmap='jet')
    axs[1].plot(outer_coords[:, 0], outer_coords[:, 1], 'k-')
    axs[1].plot(inner_coords[:, 0], inner_coords[:, 1], 'k-')
    pyplot.show()


if __name__ == '__main__':
    basic_check()
    compound_poly_check()
