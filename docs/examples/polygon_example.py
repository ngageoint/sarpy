import numpy
from matplotlib import pyplot
import time

from sarpy.geometry.polygon import Polygon


# craft our sample polygon
segment_count = 12
xs = numpy.zeros((2 * segment_count + 1,))
ys = numpy.zeros((2 * segment_count + 1,))

x1 = numpy.random.rand(segment_count + 1)
x1[0] = 0
l1 = numpy.cumsum(x1)
xs[:segment_count + 1] = l1 / l1[-1]
x2 = numpy.cumsum(numpy.random.rand(segment_count))
xs[segment_count + 1:] = 1 - x2 / x2[-1]
ys[1:segment_count] = numpy.random.rand(segment_count - 1)
ys[segment_count + 1:2 * segment_count] = -numpy.random.rand(segment_count - 1)

# craft some sample points
samples = 100000
pts = numpy.random.rand(samples, 2)
pts[:, 0] *= 1.2
pts[:, 0] -= 0.1
pts[:, 1] *= 2
pts[:, 1] -= 1

poly = Polygon(xs, ys)

start = time.time()
in_poly_condition = poly.contained(pts[:, 0], pts[:, 1])
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
axs[0].plot(xs, ys, 'k-')
y2d, x2d = numpy.meshgrid(y_grid, x_grid, indexing='xy')
axs[1].pcolormesh(x2d, y2d, in_poly_condition2, cmap='jet')
axs[1].plot(xs, ys, 'k-', lw=2, zorder=99)
pyplot.show()
