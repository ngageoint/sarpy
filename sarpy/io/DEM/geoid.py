# -*- coding: utf-8 -*-

import numpy


class GeoidBadDataFile(Exception):
    pass


class GeoidHeight(object):
    """
    Calculate the height of the WGS84 geoid above the ellipsoid at any given latitude and longitude.
    The appropriate file is available for download at http://geographiclib.sourceforge.net/1.18/geoid.html
    """
    __slots__ = ('_offset', '_scale', '_width', '_height', '_header_length', '_memory_map', '_lon_res', '_lat_res')

    c0 = 240
    c3 = numpy.array((
        (9,   -18, -88,    0,  96,   90,   0,   0, -60, -20),
        (-9,   18,   8,    0, -96,   30,   0,   0,  60, -20),
        (9,   -88, -18,   90,  96,    0, -20, -60,   0,   0),
        (186, -42, -42, -150, -96, -150,  60,  60,  60,  60),
        (54,  162, -78,   30, -24,  -90, -60,  60, -60,  60),
        (-9,  -32,  18,   30,  24,    0,  20, -60,   0,   0),
        (-9,    8,  18,   30, -96,    0, -20,  60,   0,   0),
        (54,  -78, 162,  -90, -24,   30,  60, -60,  60, -60),
        (-54,  78,  78,   90, 144,   90, -60, -60, -60, -60),
        (9,    -8, -18,  -30, -24,    0,  20,  60,   0,   0),
        (-9,   18, -32,    0,  24,   30,   0,   0, -60,  20),
        (9,   -18,  -8,    0, -24,  -30,   0,   0,  60,  20)), dtype=numpy.float64)

    c0n = 372
    c3n = numpy.array((
        (0,   0, -131, 0,  138,  144, 0,   0, -102, -31),
        (0,   0,    7, 0, -138,   42, 0,   0,  102, -31),
        (62,  0,  -31, 0,    0,  -62, 0,   0,    0,  31),
        (124, 0,  -62, 0,    0, -124, 0,   0,    0,  62),
        (124, 0,  -62, 0,    0, -124, 0,   0,    0,  62),
        (62,  0,  -31, 0,    0,  -62, 0,   0,    0,  31),
        (0,   0,   45, 0, -183,   -9, 0,  93,   18,   0),
        (0,   0,  216, 0,   33,   87, 0, -93,   12, -93),
        (0,   0,  156, 0,  153,   99, 0, -93,  -12, -93),
        (0,   0,  -45, 0,   -3,    9, 0,  93,  -18,   0),
        (0,   0,  -55, 0,   48,   42, 0,   0,  -84,  31),
        (0,   0,   -7, 0,  -48,  -42, 0,   0,   84,  31)), dtype=numpy.float64)

    c0s = 372
    c3s = numpy.array((
        (18,   -36, -122,   0,  120,  135, 0,   0,  -84, -31),
        (-18,   36,   -2,   0, -120,   51, 0,   0,   84, -31),
        (36,  -165,  -27,  93,  147,   -9, 0, -93,   18,   0),
        (210,   45, -111, -93,  -57, -192, 0,  93,   12,  93),
        (162,  141,  -75, -93, -129, -180, 0,  93,  -12,  93),
        (-36,  -21,   27,  93,   39,    9, 0, -93,  -18,   0),
        (0,      0,   62,   0,    0,   31, 0,   0,    0, -31),
        (0,      0,  124,   0,    0,   62, 0,   0,    0, -62),
        (0,      0,  124,   0,    0,   62, 0,   0,    0, -62),
        (0,      0,   62,   0,    0,   31, 0,   0,    0, -31),
        (-18,   36,  -64,   0,   66,   51, 0,   0, -102,  31),
        (18,   -36,    2,   0,  -66,  -51, 0,   0,  102,  31)), dtype=numpy.float64)

    def __init__(self, file_name="egm2008-1.pgm"):
        self._offset = None
        self._scale = None

        with open(file_name, "rb") as f:
            line = f.readline()
            if line != b"P5\012" and line != b"P5\015\012":
                raise GeoidBadDataFile("No PGM header")
            headerlen = len(line)
            while True:
                line = f.readline().decode('utf-8')
                if len(line) == 0:
                    raise GeoidBadDataFile("EOF before end of file header")
                headerlen += len(line)
                if line.startswith('# Offset '):
                    try:
                        self._offset = int(line[9:])
                    except ValueError as e:
                        raise GeoidBadDataFile("Error reading offset", e)
                elif line.startswith('# Scale '):
                    try:
                        self._scale = float(line[8:])
                    except ValueError as e:
                        raise GeoidBadDataFile("Error reading scale", e)
                elif not line.startswith('#'):
                    try:
                        slin = line.split()
                        self._width, self._height = int(slin[0]), int(slin[1])
                    except ValueError as e:
                        raise GeoidBadDataFile("Bad PGM width&height line", e)
                    break
            line = f.readline().decode('utf-8')
            headerlen += len(line)
            levels = int(line)
            if levels != 65535:
                raise GeoidBadDataFile("PGM file must have 65535 gray levels")
            if self._offset is None:
                raise GeoidBadDataFile("PGM file does not contain offset")
            if self._scale is None:
                raise GeoidBadDataFile("PGM file does not contain scale")

            if self._width < 2 or self._height < 2:
                raise GeoidBadDataFile("Raster size too small")
            self._header_length = headerlen

        self._memory_map = numpy.memmap(file_name,
                                        dtype=numpy.dtype('>u2'),
                                        mode='r',
                                        offset=self._header_length,
                                        shape=(self._height, self._width))
        self._lon_res = self._width/360.0
        self._lat_res = (self._height - 1)/180.0

    def _get_raw(self, ix, iy):
        boolc = (iy < 0)
        iy[boolc] *= -1
        ix[boolc] += int(self._width/2)

        boolc = (iy >= self._height)
        iy[boolc] = 2*(self._height - 1) - iy[boolc]
        ix[boolc] += int(self._width/2)

        boolc = (ix < 0)
        ix[boolc] += self._width

        boolc = (ix >= self._width)
        ix[boolc] -= self._width

        return self._memory_map[iy, ix]

    def _linear(self, ix, dx, iy, dy):
        a = (1 - dx) * self._get_raw(ix, iy) + dx * self._get_raw(ix +1, iy)
        b = (1 - dx) * self._get_raw(ix, iy+1) + dx * self._get_raw(ix+1, iy+1)
        return (1 - dy) * a + dy * b

    def _cubic(self, ix, dx, iy, dy):
        v = numpy.vstack((
            self._get_raw(ix    , iy - 1),
            self._get_raw(ix + 1, iy - 1),
            self._get_raw(ix - 1, iy    ),
            self._get_raw(ix    , iy    ),
            self._get_raw(ix + 1, iy    ),
            self._get_raw(ix + 2, iy    ),
            self._get_raw(ix - 1, iy + 1),
            self._get_raw(ix    , iy + 1),
            self._get_raw(ix + 1, iy + 1),
            self._get_raw(ix + 2, iy + 1),
            self._get_raw(ix    , iy + 2),
            self._get_raw(ix + 1, iy + 2)))

        t = numpy.zeros((10, ix.size), dtype=numpy.float64)
        b1 = (iy == 0)
        b2 = (iy == self._height -2)
        b3 = ~(b1 | b2)
        if numpy.any(b1):
            t[:,b1] = (self.c3n.T/self.c0n).dot(v[:, b1])
        if numpy.any(b2):
            t[:, b2] = (self.c3s.T/self.c0s).dot(v[:, b2])
        if numpy.any(b3):
            t[:, b3] = (self.c3.T/self.c0).dot(v[:, b3])

        return t[0] + \
               dx*(t[1] + dx*(t[3] + dx*t[6])) + \
               dy*(t[2] + dx*(t[4] + dx*t[7]) + dy*(t[5] + dx*t[8] + dy*t[9]))

    def _do_block(self, lat, lon, cubic):
        lon[lon < 0] += 360
        fx = lon*self._lon_res
        fy = (90 - lat)* self._lat_res

        ix = numpy.cast[numpy.int32](numpy.floor(fx))
        iy = numpy.cast[numpy.int32](numpy.floor(fy))

        dx = fx - ix
        dy = fy - iy

        iy[iy == self._height -1] -= 1  # edge effects?

        if cubic:
            return self._offset + self._scale*self._cubic(ix, dx, iy, dy)
        else:
            return self._offset + self._scale*self._linear(ix, dx, iy, dy)

    def get(self, lat, lon, cubic=True, blocksize=50000):
        if not isinstance(lat, numpy.ndarray):
            lat = numpy.array(lat)
        if not isinstance(lon, numpy.ndarray):
            lon = numpy.array(lon)
        if lat.shape != lon.shape:
            raise ValueError(
                'lat and lon must have the same shape, got '
                'lat.shape = {}, lon.shape = {}'.format(lat.shape, lon.shape))
        o_shape = lat.shape
        lat = numpy.reshape(lat, (-1, ))
        lon = numpy.reshape(lon, (-1, ))

        out = numpy.empty(lat.shape, dtype=numpy.float64)

        blocksize = max(10000, int(blocksize))
        start_block = 0
        while start_block < lat.size:
            end_block = min(start_block+blocksize, lat.size)
            out[start_block:end_block] = self._do_block(lat[start_block:end_block], lon[start_block:end_block], cubic)
            start_block = end_block

        if o_shape == ():
            return float(out[0])
        else:
            return numpy.reshape(out, o_shape)


if __name__ == '__main__':
    import time
    # parse test set
    test_file = '/Users/tom/Downloads/GeoidHeights.dat'
    with open(test_file, 'r') as fi:
        lins = fi.read().splitlines()
    lats = numpy.zeros((len(lins), ), dtype=numpy.float64)
    lons = numpy.zeros((len(lins), ), dtype=numpy.float64)
    zs = numpy.zeros((len(lins), ), dtype=numpy.float64)
    for i, lin in enumerate(lins):
        slin = lin.strip().split()
        lats[i] = float(slin[0])
        lons[i] = float(slin[1])
        zs[i] = float(slin[4])
    print('number of test points {}'.format(lats.size))

    gh = GeoidHeight(file_name='/Users/tom/Downloads/geoids/egm2008-5.pgm')
    from sarpy.deprecated.io.DEM.geoid import GeoidHeight as GH2
    gh2 = GH2(name='/Users/tom/Downloads/geoids/egm2008-5.pgm')
    recs = 10
    # test one
    start = time.time()
    zs1 = gh.get(lats[:recs], lons[:recs], cubic=False)
    print('linear - {}, {}, {}'.format(zs[:recs], zs1, time.time() - start))
    # test two
    start = time.time()
    zs1 = gh.get(lats[:recs], lons[:recs], cubic=True)
    print('cubic - {}, {}, {}'.format(zs[:recs], zs1, time.time() - start))
    # test three
    zs1 = numpy.zeros((recs, ), dtype=numpy.float64)
    start = time.time()
    for i in range(recs):
        zs1[i] = gh2.get(lats[i], lons[i], cubic=False)
    print('old - {}, {}, {}'.format(zs[:recs], zs1, time.time() - start))

    recs = 100000
    # test one
    start = time.time()
    zs1 = gh.get(lats[:recs], lons[:recs], cubic=False)
    print('linear - {}'.format(time.time() - start))
    # test two
    start = time.time()
    zs1 = gh.get(lats[:recs], lons[:recs], cubic=True)
    print('cubic - {}'.format(time.time() - start))
    # test three
    zs1 = numpy.zeros((recs, ), dtype=numpy.float64)
    start = time.time()
    for i in range(recs):
        zs1[i] = gh2.get(lats[i], lons[i], cubic=False)
    print('old - {}'.format(time.time() - start))
