# -*- coding: utf-8 -*-
"""
Provides a class for calculating the height of the WGS84 geoid above the ellipsoid
at any given latitude and longitude.

**Some Accuracy Details:**
Using the 5 minute pgm and linear interpolation, the average error `|calculated - real|`
is around 5 millimeters, and the worst case error is around 30 centimeters. Using cubic
interpolation, the average error drops to about 3 mm, and the worst case is about 17 cm.

Using the 1 minute pgm and linear interpolation, the average error is around 0.5 mm and
worst case error around 1 cm. Using cubic interpolation, the average error is still around
0.5 mm, and worst case error around 2 mm.

The accuracy obtained using the 5, 2.5, or 1 minute pgm are likely all more than suitable
for any SAR application. The the accuracy clearly increases with finer grid.

**Some Processing Speed and Resource Details:**
A memory map into the pgm file is established, which requires relatively little "real" RAM,
but an amount of virtual memory on par with the file size. The speed seems to generally scale
close to inverse linearly with pixel size, so using the 5 minute pgm is generally ~4-5 times
faster than using the 1 minute pgm.

Using the 5 minute pgm, processing rate of around 4-8 million points per second for linear
interpolation and 1-2 million points per second for cubic interpolation. Using the 1 minute pgm,
this processing rate drops to 1-2 million points per second for linear interpolation, and
~million points per second using the cubic interpolation. These rates depend of a variety
of factors including processor speed, hard drive speed, and how your operating system handles
memory maps.

The 5 minute pgm is about 25 times smaller at around 18 MB, while the 1 minute pgm file is
around 450 MB.

**File Locations:**
As of January 2020, the egm2008 pgm files are available for download at
https://geographiclib.sourceforge.io/html/geoid.html

Specifically 1 minute data is available at
https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-1.tar.bz2
or
https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-1.zip

Specifically the 5 minute data is available at
https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-5.tar.bz2
or
https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-5.zip
"""

import os
import numpy

from sarpy.compliance import string_types
from sarpy.io.DEM.utils import argument_validation


__classification__ = "UNCLASSIFIED"
__author__ = "Thomas McCullough"


############
# module variables
_C0 = 240
_C3 = numpy.array((
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

_C0N = 372
_C3N = numpy.array((
    (0, 0, -131, 0, 138, 144, 0, 0, -102, -31),
    (0, 0, 7, 0, -138, 42, 0, 0, 102, -31),
    (62, 0, -31, 0, 0, -62, 0, 0, 0, 31),
    (124, 0, -62, 0, 0, -124, 0, 0, 0, 62),
    (124, 0, -62, 0, 0, -124, 0, 0, 0, 62),
    (62, 0, -31, 0, 0, -62, 0, 0, 0, 31),
    (0, 0, 45, 0, -183, -9, 0, 93, 18, 0),
    (0, 0, 216, 0, 33, 87, 0, -93, 12, -93),
    (0, 0, 156, 0, 153, 99, 0, -93, -12, -93),
    (0, 0, -45, 0, -3, 9, 0, 93, -18, 0),
    (0, 0, -55, 0, 48, 42, 0, 0, -84, 31),
    (0, 0, -7, 0, -48, -42, 0, 0, 84, 31)), dtype=numpy.float64)

_C0S = 372
_C3S = numpy.array((
    (18, -36, -122, 0, 120, 135, 0, 0, -84, -31),
    (-18, 36, -2, 0, -120, 51, 0, 0, 84, -31),
    (36, -165, -27, 93, 147, -9, 0, -93, 18, 0),
    (210, 45, -111, -93, -57, -192, 0, 93, 12, 93),
    (162, 141, -75, -93, -129, -180, 0, 93, -12, 93),
    (-36, -21, 27, 93, 39, 9, 0, -93, -18, 0),
    (0, 0, 62, 0, 0, 31, 0, 0, 0, -31),
    (0, 0, 124, 0, 0, 62, 0, 0, 0, -62),
    (0, 0, 124, 0, 0, 62, 0, 0, 0, -62),
    (0, 0, 62, 0, 0, 31, 0, 0, 0, -31),
    (-18, 36, -64, 0, 66, 51, 0, 0, -102, 31),
    (18, -36, 2, 0, -66, -51, 0, 0, 102, 31)), dtype=numpy.float64)

_SEARCH_FILES = ('egm2008-5.pgm', 'egm2008-2_5.pgm', 'egm2008-1.pgm', 'egm96-5.pgm', 'egm96-15.pgm')


def find_geoid_file_from_dir(dir_name, search_files=None):
    """
    Find the geoid file.

    Parameters
    ----------
    dir_name : str
    search_files : str|List[str]

    Returns
    -------
    str
    """

    geoid_dir = os.path.join(dir_name, 'geoid')
    if not os.path.exists(geoid_dir):
        raise IOError(
            'Input is a directory, and beneath it we expect to find '
            'files in directory "geoid"')
    if search_files is None:
        search_files = []
    elif isinstance(search_files, string_types):
        search_files = [search_files, ]
    else:
        search_files = list(search_files)

    for entry in _SEARCH_FILES:
        if entry not in search_files:
            search_files.append(entry)

    our_file = None
    for fil in search_files:
        file_name = os.path.join(geoid_dir, fil)
        if os.path.exists(file_name):
            our_file = file_name
            break

    if our_file is None:
        raise IOError(
            'input is a directory and we expect to find one of the files {} '
            'in the directory "geoid" beneath it'.format(search_files))

    return our_file


class GeoidHeight(object):
    """
    Calculator for the height of the WGS84 geoid above the ellipsoid at any
    given latitude and longitude, based on one of the egm .pgm files.

    We are set up to use a dem/geoid parent directory. In this case, we expect
    our egm .pgm to be in the `<root_dir>/geoid` directory, and we will search
    in order of preference
    ('egm2008-1.pgm', 'egm2008-2_5.pgm', 'egm2008-5.pgm', 'egm96-5.pgm', 'egm96-15.pgm')
    """

    __slots__ = (
        '_offset', '_scale', '_width', '_height', '_header_length', '_memory_map',
        '_lon_res', '_lat_res')

    def __init__(self, file_name):
        """

        Parameters
        ----------
        file_name : str
            path to a egm2008 pgm file
        """

        self._offset = None
        self._scale = None

        if os.path.isdir(file_name):
            file_name = find_geoid_file_from_dir(file_name)

        with open(file_name, "rb") as f:
            line = f.readline()
            if line != b"P5\012" and line != b"P5\015\012":
                raise IOError("No PGM header")
            headerlen = len(line)
            while True:
                line = f.readline().decode('utf-8')
                if len(line) == 0:
                    raise IOError("EOF before end of file header")
                headerlen += len(line)
                if line.startswith('# Offset '):
                    try:
                        self._offset = int(line[9:])
                    except ValueError as e:
                        raise IOError("Error reading offset", e)
                elif line.startswith('# Scale '):
                    try:
                        self._scale = float(line[8:])
                    except ValueError as e:
                        raise IOError("Error reading scale", e)
                elif not line.startswith('#'):
                    try:
                        slin = line.split()
                        self._width, self._height = int(slin[0]), int(slin[1])
                    except ValueError as e:
                        raise IOError("Bad PGM width&height line", e)
                    break
            line = f.readline().decode('utf-8')
            headerlen += len(line)
            levels = int(line)
            if levels != 65535:
                raise IOError("PGM file must have 65535 gray levels")
            if self._offset is None:
                raise IOError("PGM file does not contain offset")
            if self._scale is None:
                raise IOError("PGM file does not contain scale")

            if self._width < 2 or self._height < 2:
                raise IOError("Raster size too small")
            self._header_length = headerlen

        self._memory_map = numpy.memmap(file_name,
                                        dtype=numpy.dtype('>u2'),
                                        mode='r',
                                        offset=self._header_length,
                                        shape=(self._height, self._width))
        self._lon_res = self._width/360.0
        self._lat_res = (self._height - 1)/180.0

    def _get_raw(self, ix, iy):
        # these manipulations are required for edge effects
        boolc = (iy < 0)
        iy[boolc] *= -1
        # ix[boolc] += int(self._width/2)  # why is this here?

        boolc = (iy >= self._height)
        iy[boolc] = 2*(self._height - 1) - iy[boolc]
        # ix[boolc] += int(self._width/2)  # why is this here?

        boolc = (ix < 0)
        ix[boolc] += self._width

        boolc = (ix >= self._width)
        ix[boolc] -= self._width

        return self._memory_map[iy, ix]

    def _linear(self, ix, dx, iy, dy):
        a = (1 - dx) * self._get_raw(ix, iy) + dx * self._get_raw(ix + 1, iy)
        b = (1 - dx) * self._get_raw(ix, iy+1) + dx * self._get_raw(ix+1, iy+1)
        return (1 - dy) * a + dy * b

    def _cubic(self, ix, dx, iy, dy):
        v = numpy.vstack((
            self._get_raw(ix, iy - 1),
            self._get_raw(ix + 1, iy - 1),
            self._get_raw(ix - 1, iy),
            self._get_raw(ix, iy),
            self._get_raw(ix + 1, iy),
            self._get_raw(ix + 2, iy),
            self._get_raw(ix - 1, iy + 1),
            self._get_raw(ix, iy + 1),
            self._get_raw(ix + 1, iy + 1),
            self._get_raw(ix + 2, iy + 1),
            self._get_raw(ix, iy + 2),
            self._get_raw(ix + 1, iy + 2)))

        t = numpy.zeros((10, ix.size), dtype=numpy.float64)
        b1 = (iy == 0)
        b2 = (iy == self._height - 2)
        b3 = ~(b1 | b2)
        if numpy.any(b1):
            t[:, b1] = (_C3N.T/_C0N).dot(v[:, b1])
        if numpy.any(b2):
            t[:, b2] = (_C3S.T/_C0S).dot(v[:, b2])
        if numpy.any(b3):
            t[:, b3] = (_C3.T/_C0).dot(v[:, b3])

        return t[0] + \
            dx*(t[1] + dx*(t[3] + dx*t[6])) + \
            dy*(t[2] + dx*(t[4] + dx*t[7]) + dy*(t[5] + dx*t[8] + dy*t[9]))

    def _do_block(self, lat, lon, cubic):
        fx = lon*self._lon_res
        fx[fx < 0] += 360*self._lon_res
        fy = (90 - lat)*self._lat_res

        ix = numpy.cast[numpy.int32](numpy.floor(fx))
        iy = numpy.cast[numpy.int32](numpy.floor(fy))

        dx = fx - ix
        dy = fy - iy

        iy[iy == self._height - 1] -= 1  # edge effects?

        if cubic:
            return self._offset + self._scale*self._cubic(ix, dx, iy, dy)
        else:
            return self._offset + self._scale*self._linear(ix, dx, iy, dy)

    def get(self, lat, lon, cubic=True, block_size=50000):
        """
        Calculate the height of the geoid above the ellipsoid in meters at the given points.

        Parameters
        ----------
        lat : numpy.ndarray
        lon : numpy.ndarray
        cubic : bool
            Use a simple cubic spline interpolation, otherwise us simple linear.
            Default is `True`.
        block_size : None|int
            If `None`, then the entire calculation will proceed as a single block.
            Otherwise, block processing using blocks of the given size will be used.
            The minimum value used for this is 50,000, and any smaller value will be
            replaced with 50,000. Default is 50,000.

        Returns
        -------
        numpy.ndarray
        """

        o_shape, lat, lon = argument_validation(lat, lon)

        if block_size is None:
            out = self._do_block(lat, lon, cubic)
        else:
            block_size = max(50000, int(block_size))
            out = numpy.empty(lat.shape, dtype=numpy.float64)
            start_block = 0
            while start_block < lat.size:
                end_block = min(start_block+block_size, lat.size)
                out[start_block:end_block] = self._do_block(
                    lat[start_block:end_block], lon[start_block:end_block], cubic)
                start_block = end_block

        if o_shape == ():
            return float(out[0])
        else:
            return numpy.reshape(out, o_shape)

    def __call__(self, lat, lon):
        return self.get(lat, lon)

    @classmethod
    def from_directory(cls, dir_name, search_files=None):
        """
        Create the GeoidHeight object from a search directory.

        Parameters
        ----------
        dir_name : str
        search_files : str|List[str]

        Returns
        -------
        GeoidHeight
        """

        our_file = find_geoid_file_from_dir(dir_name, search_files=search_files)
        return cls(our_file)
