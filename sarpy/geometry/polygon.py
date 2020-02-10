# -*- coding: utf-8 -*-

import numpy

def _compress_identical(x_coord, y_coord):
    include = numpy.zeros(x_coord.shape, dtype=numpy.bool)
    include[-1] = True
    for i, (xbv, xev, ybv, yev) in enumerate(zip(x_coord[:-1], x_coord[1:], y_coord[:-1], y_coord[1:])):
        if (xbv != xev) or (ybv != yev):
            # keep if this point if different that the one that follows
            include[i] = True
    return x_coord[include], y_coord[include]


class Polygon(object):
    __slots__ = (
        '_x_coords', '_y_coords', '_bounding_box', '_x_segments', '_y_segments',
        '_x_diff', '_y_diff')

    """
    Class for representing a polygon with associated basic functionality. 
    It is assumed that the coordinate space is for geographical or pixel coordinates.
    """

    def __init__(self, x_coords, y_coords):
        """

        Parameters
        ----------
        x_coords : numpy.ndarray|list|tuple
        y_coords : numpy.ndarray|list|tuple
        """

        if not isinstance(x_coords, numpy.ndarray):
            x_coords = numpy.array(x_coords, dtype=numpy.float64)
        if not isinstance(y_coords, numpy.ndarray):
            y_coords = numpy.array(y_coords, dtype=numpy.float64)

        if x_coords.dtype.name != y_coords.dtype.name:
            raise ValueError(
                'x_coords and y_coords must have the same data type. Got {} and {}'.format(x_coords.dtype.name, y_coords.dtype.name))

        if x_coords.shape != y_coords.shape:
            raise ValueError(
                'x_coords and y_coords must be the same shape. Got {} and {}'.format(x_coords.shape, y_coords.shape))
        if len(x_coords.shape) != 1:
            raise ValueError(
                'x_coords and y_coords must be one-dimensional. Got shape {}'.format(x_coords.shape))

        x_coords, y_coords = _compress_identical(x_coords, y_coords)

        if (x_coords[0] != x_coords[-1]) or (y_coords[0] != y_coords[-1]):
            x_coords = numpy.hstack((x_coords, x_coords[0]))
            y_coords = numpy.hstack((y_coords, y_coords[0]))

        if x_coords.size < 4:
            raise ValueError(
                'After compressing repeated (in sequence) points and ensuring first and last point are the same, '
                'x_coords and y_coords must have length at least 4 (i.e. a triangle). '
                'Got x_coords={}, y_coords={}.'.format(x_coords, y_coords))

        self._x_coords = x_coords
        self._y_coords = y_coords

        # construct bounding box
        self._bounding_box = numpy.empty((2, 2), dtype=x_coords.dtype)
        self._bounding_box[0, :] = (numpy.min(self._x_coords), numpy.max(self._x_coords))
        self._bounding_box[1, :] = (numpy.min(self._y_coords), numpy.max(self._y_coords))
        # construct a monotonic listing in x of intervals and indices of edges which apply
        self._x_segments = self._construct_segmentation(self._x_coords, self._y_coords)
        # construct a monotonic listing in y of intervals and indices of edges which apply
        self._y_segments = self._construct_segmentation(self._y_coords, self._x_coords)
        # construct diffs
        self._x_diff = self._x_coords[1:] - self._x_coords[:-1]
        self._y_diff = self._y_coords[1:] - self._y_coords[:-1]

    @staticmethod
    def _construct_segmentation(coords, o_coords):
        def overlap(fst, lst, segment):
            if fst == lst and fst == segment['min']:
                return True
            if segment['max'] <= fst or lst <= segment['min']:
                return False
            return True

        def do_min_val_value(segment, val1, val2):
            segment['min_value'] = min(val1, val2, segment['min_value'])
            segment['max_value'] = max(val1, val2, segment['max_value'])

        inds = numpy.argsort(coords[:-1])
        segments = []
        beg_val = coords[inds[0]]
        val = None
        for ind in inds[1:]:
            val = coords[ind]
            if val > beg_val:  # make a new segment
                segments.append({'min': beg_val, 'max': val, 'inds': [], 'min_value': numpy.inf, 'max_value': -numpy.inf})
                beg_val = val
        else:
            # it may have ended without appending the segment
            if val > beg_val:
                segments.append({'min': beg_val, 'max': val, 'inds': [], 'min_value': numpy.inf, 'max_value': -numpy.inf})
        del beg_val, val

        # now, let's populate the inds lists and min/max_values elements
        for i, (beg_value, end_value, ocoord1, ocoord2) in enumerate(zip(coords[:-1], coords[1:], o_coords[:-1], o_coords[1:])):
            first, last = (beg_value, end_value) if beg_value <= end_value else (end_value, beg_value)
            # check all the segments for overlap
            for j, seg in enumerate(segments):
                if overlap(first, last, seg):
                    seg['inds'].append(i)
                    do_min_val_value(seg, ocoord1, ocoord2)

        return tuple(segments)

    @property
    def bounding_box(self):  # type: () -> numpy.ndarray
        """
        numpy.ndarray: The bounding box of the form [[x_min, x_max], [y_min, y_max]].
        *Note that could be extremely misleading for a lat/lon polygon crossing
        the boundary of discontinuity and/or surrounding a pole.*
        """

        return self._bounding_box

    def get_area(self):
        """
        Gets the area of the polygon.

        Returns
        -------
        float
        """

        return abs(float(0.5*numpy.sum(self._x_coords[:-1]*self._y_coords[1:] - self._x_coords[1:]*self._y_coords[:-1])))

    def get_centroid(self):
        """
        Gets the centroid of the polygon - note that this may not actually lie in
        the polygon interior for non-convex polygon.

        Returns
        -------
        numpy.ndarray
        """

        arr = self._x_coords[:-1]*self._y_coords[1:] - self._x_coords[1:]*self._y_coords[:-1]
        area = 0.5*numpy.sum(arr)  # signed area
        x = numpy.sum(0.5*(self._x_coords[:-1] + self._x_coords[1:])*arr)
        y = numpy.sum(0.5*(self._y_coords[:-1] + self._y_coords[1:])*arr)
        return numpy.array([x, y], dtype=numpy.float64)/(3*area)

    def _contained_segment_data(self, x, y):
        """
        This is a helper function for the polygon containment effort.
        This determines whether the x or y segmentation should be utilized, and
        the details for doing so.

        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray

        Returns
        -------
        (int|None, int|None, str)
            the segment index start (inclusive), the segment index end (exclusive),
            and "x" or "y" for which segmentation is better.
        """

        def segment(coord, segments):
            tmin = coord.min()
            tmax = coord.max()

            if tmax < segments[0]['min'] or tmin > segments[-1]['max']:
                return None, None

            if len(segments) == 1:
                return 0, 0

            t_first_ind = None if tmin > segments[0]['max'] else 0
            t_last_ind = None if tmax < segments[-1]['min'] else len(segments)

            for i, seg in enumerate(segments):
                if seg['min'] <= tmin < seg['max']:
                    t_first_ind = i
                if seg['min'] <= tmax <= seg['max']:
                    t_last_ind = i+1
                if t_first_ind is not None and t_last_ind is not None:
                    break
            return t_first_ind, t_last_ind

        # let's determine first/last x & y segments and which is better (fewer)
        x_first_ind, x_last_ind = segment(x, self._x_segments)
        if x_first_ind is None:
            return None, None, 'x'

        y_first_ind, y_last_ind = segment(y, self._y_segments)
        if y_first_ind is None:
            return None, None, 'y'

        if (y_last_ind - y_first_ind) < (x_last_ind - x_first_ind):
            return y_first_ind, y_last_ind, 'y'
        return x_first_ind, x_last_ind, 'x'

    def _contained_do_segment(self, x, y, segment, direction):
        """
        Helper function for polygon containment effort.

        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray
        segment : dict
        direction : str

        Returns
        -------
        numpy.ndarray
        """

        # we require that all these points are relevant to this slice
        in_poly = numpy.zeros(x.shape, dtype=numpy.bool)
        crossing_counts = numpy.zeros(x.shape, dtype=numpy.int32)
        indices= segment['inds']

        for i in indices:
            nx, ny = self._y_diff[i],  -self._x_diff[i]
            crossing = (x - self._x_coords[i])*nx + (y - self._y_coords[i])*ny
            # dot product of vector connecting (x, y) to segment vertex with normal vector of segment
            crossing_counts[crossing > 0] += 1 # positive crossing number
            crossing_counts[crossing < 0] -= 1 # negative crossing number
            if numpy.any(crossing_counts == 0):
                # include points on the edge
                if direction == 'x':
                    in_poly[(crossing == 0) & (segment['min_value'] <= y) & (y <= segment['max_value'])] = True
                else:
                    in_poly[(crossing == 0) & (segment['min_value'] <= x) & (x <= segment['max_value'])] = True
        in_poly |= (crossing_counts != 0) # the interior points are defined by the sum of crossing numbers being non-zero
        return in_poly

    def _contained(self, x, y):
        """
        Helper method for polygon inclusion.
        Parameters
        ----------
        x : numpy.ndarray
        y : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        out = numpy.zeros(x.shape, dtype=numpy.bool)

        ind_beg, ind_end, direction = self._contained_segment_data(x, y)
        if ind_beg is None:
            return out  # it missed the whole bounding box

        for index in range(ind_beg, ind_end):
            if direction == 'x':
                seg = self._x_segments[index]
                mask = ((x >= seg['min']) & (x <= seg['max']) & (y >= seg['min_value']) & (y <= seg['max_value']))
            else:
                seg = self._y_segments[index]
                mask = ((y >= seg['min']) & (y <= seg['max']) & (x >= seg['min_value']) & (x <= seg['max_value']))
            if numpy.any(mask):
                out[mask] = self._contained_do_segment(x[mask], y[mask], seg, direction)
        return out

    def contained(self, pts_x, pts_y, block_size=None):
        """
        Determines inclusion of the given points in the interior of the polygon.
        The methodology here is based on the Jordan curve theorem approach.

        ** Warning: This method may provide erroneous results for a lat/lon polygon
        crossing the bound of discontinuity and/or surrounding a pole.**

        * Note: If the points constitute an x/y grid, then the grid contained method will
        be much more performant.*

        Parameters
        ----------
        pts_x : numpy.ndarray|list|tuple|float|int
        pts_y : numpy.ndarray|list|tuple|float|int
        block_size : None|int
            If provided, processing block size. The minimum value used will be
            50,000.

        Returns
        -------
        numpy.ndarray|bool
            boolean array indicating inclusion.
        """

        if not isinstance(pts_x, numpy.ndarray):
            pts_x = numpy.array(pts_x, dtype=numpy.float64)
        if not isinstance(pts_y, numpy.ndarray):
            pts_y = numpy.array(pts_y, dtype=numpy.float64)

        if pts_x.shape != pts_y.shape:
            raise ValueError(
                'pts_x and pts_y must be the same shape. Got {} and {}'.format(pts_x.shape, pts_y.shape))

        o_shape = pts_x.shape

        if len(o_shape) == 0:
            pts_x = numpy.reshape(pts_x, (1, ))
            pts_y = numpy.reshape(pts_y, (1, ))
        else:
            pts_x = numpy.reshape(pts_x, (-1, ))
            pts_y = numpy.reshape(pts_y, (-1, ))

        if block_size is not None:
            block_size = int(block_size)
            block_size = max(50000, block_size)

        if block_size is None or pts_x.size <= block_size:
            in_poly = self._contained(pts_x, pts_y)
        else:
            in_poly = numpy.zeros(pts_x.shape, dtype=numpy.bool)
            start_block = 0
            while start_block < pts_x.size:
                end_block = min(start_block+block_size, pts_x.size)
                in_poly[start_block:end_block] = self._contained(
                    pts_x[start_block:end_block], pts_y[start_block:end_block])
                start_block = end_block

        if len(o_shape) == 0:
            return in_poly[0]
        else:
            return numpy.reshape(in_poly, o_shape)

    def grid_contained(self, grid_x, grid_y):
        """
        Determines inclusion of a coordinate grid inside the polygon. The coordinate
        grid is defined by the two one-dimensional coordinate arrays `grid_x` and `grid_y`.

        Parameters
        ----------
        grid_x : nuumpy.ndarray
        grid_y : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            boolean mask for point inclusion of the grid. Output is of shape
            `(grid_x.size, grid_y.size)`.
        """

        if not isinstance(grid_x, numpy.ndarray):
            grid_x = numpy.array(grid_x, dtype=numpy.float64)
        if not isinstance(grid_y, numpy.ndarray):
            grid_y = numpy.array(grid_y, dtype=numpy.float64)
        if len(grid_x.shape) != 1 or len(grid_y.shape) != 1:
            raise ValueError('grid_x and grid_y must be one dimensional.')
        if numpy.any((grid_x[1:] - grid_x[:-1]) <= 0):
            raise ValueError('grid_x must be monotonically increasing')
        if numpy.any((grid_y[1:] - grid_y[:-1]) <= 0):
            raise ValueError('grid_y must be monotonically increasing')

        out = numpy.zeros((grid_x.size, grid_y.size), dtype=numpy.bool)
        first_ind, last_ind, direction = self._contained_segment_data(grid_x, grid_y)
        if first_ind is None:
            return out  # it missed the whole bounding box

        first_ind, last_ind, direction = self._contained_segment_data(grid_x, grid_y)
        x_inds = numpy.arange(grid_x.size)
        y_inds = numpy.arange(grid_y.size)
        for index in range(first_ind, last_ind):
            if direction == 'x':
                seg = self._x_segments[index]
                start_x = x_inds[grid_x >= seg['min']].min()
                end_x = x_inds[grid_x <= seg['max']].max() + 1
                start_y = y_inds[grid_y >= seg['min_value']].min() if grid_y[-1] >= seg['min_value'] else None
                end_y = y_inds[grid_y <= seg['max_value']].max() + 1 if start_y is not None else None
            else:
                seg = self._y_segments[index]
                start_x = x_inds[grid_x >= seg['min_value']].min() if grid_x[-1] >= seg['min_value'] else None
                end_x = x_inds[grid_x <= seg['max_value']].max() + 1 if start_x is not None else None
                start_y = y_inds[grid_y >= seg['min']].min()
                end_y = y_inds[grid_y <= seg['max']].max() + 1

            if start_x is not None and end_x is not None and start_y is not None and end_y is not None:
                y_temp, x_temp = numpy.meshgrid(grid_y[start_y:end_y], grid_x[start_x:end_x], indexing='xy')
                out[start_x:end_x, start_y:end_y] = self._contained_do_segment(x_temp, y_temp, seg, direction)
        return out
